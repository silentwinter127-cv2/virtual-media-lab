"""
Signal representation, frequency-domain primitives, and C11/OpenMP/AVX2 native kernel bridge.

The engine deliberately keeps *luma* and *chroma* as separate planes through
most of the chain.  That is the only way to model what actually happens in a
composite tape format:

    the two components are filtered by different bandwidths, then they are
    physically added into ONE signal, and the player must separate them
    again -- imperfectly.

Cross-color (fine luma detail turning into colour) and cross-luma (colour
turning into moving dot patterns) are *consequences* of that round trip, not
overlays.  They appear here because we really do add and really do separate.
"""
from __future__ import annotations

import ctypes
from ctypes import c_float, c_int, c_void_p
from dataclasses import dataclass, field
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
from typing import Any

import numpy as np
from scipy.fft import irfft as _sp_irfft, next_fast_len as _next_fast_len, rfft as _sp_rfft
from scipy.ndimage import convolve1d
from scipy.signal import lfilter as _lfilter

from .standards import VideoStandard

try:
    import cv2 as _cv2
    _HAVE_CV2 = True
except Exception:
    _cv2 = None  # type: ignore[assignment]
    _HAVE_CV2 = False

TWO_PI = 2.0 * np.pi
_ONE_KERNEL = np.array([1.0], dtype=np.float32)

# NTSC / SMPTE 170M weights (also the Rec.601 luma weights)
RGB_TO_Y = np.array([0.299, 0.587, 0.114], dtype=np.float64)

_GAUSS_CACHE: dict[int, np.ndarray] = {}
_GAUSS_HALF_CACHE: dict[int, np.ndarray] = {}
_HP_HALF_CACHE: dict[int, np.ndarray] = {}
_SUBCARRIER_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_FFT_BP_CACHE: dict[tuple, np.ndarray] = {}

# ---------------------------------------------------------------------------
# Cross-platform C11 + OpenMP + AVX2/SIMD native kernel loader
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_C_SRC = _HERE / "vml_kernels.c"
_LIB: ctypes.CDLL | None = None
_LIB_FAILED = False
_LIB_LOCK = threading.Lock()
_TLS = threading.local()


def _try_compile_native() -> Path | None:
    if not _C_SRC.exists():
        return None
    sys_name = sys.platform
    mach = platform.machine().lower()
    if sys_name == "win32":
        out_dll = _HERE / f"vml_kernels_win_{mach}_omp.dll"
        if out_dll.exists() and out_dll.stat().st_mtime >= _C_SRC.stat().st_mtime:
            return out_dll
        gcc = shutil.which("gcc") or shutil.which("clang")
        if gcc:
            try:
                cmd = [
                    gcc, "-O3", "-ffast-math", "-fopenmp", "-march=native",
                    "-shared", str(_C_SRC), "-o", str(out_dll),
                ]
                res = subprocess.run(cmd, capture_output=True, timeout=15)
                if res.returncode == 0 and out_dll.exists():
                    return out_dll
            except Exception:
                pass
        cl = shutil.which("cl")
        if cl:
            try:
                cmd = [
                    cl, "/O2", "/fp:fast", "/openmp", "/arch:AVX2", "/LD",
                    str(_C_SRC), f"/Fe:{out_dll}",
                ]
                res = subprocess.run(cmd, capture_output=True, cwd=str(_HERE), timeout=15)
                if res.returncode == 0 and out_dll.exists():
                    return out_dll
            except Exception:
                pass
        return None

    ext = ".dylib" if sys_name == "darwin" else ".so"
    prebuilt = _HERE / f"vml_kernels_linux_{mach}{ext}"
    if prebuilt.exists() and (not _C_SRC.exists() or prebuilt.stat().st_mtime >= _C_SRC.stat().st_mtime):
        return prebuilt

    out_so = _HERE / f"vml_kernels_{sys_name}_{mach}{ext}"
    if out_so.exists() and out_so.stat().st_mtime >= _C_SRC.stat().st_mtime:
        return out_so

    cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
    if cc:
        for flags in (
            ["-O3", "-ffast-math", "-fopenmp", "-march=native", "-shared", "-fPIC"],
            ["-O3", "-ffast-math", "-march=native", "-shared", "-fPIC"],
            ["-O3", "-ffast-math", "-shared", "-fPIC"],
        ):
            try:
                cmd = [cc, *flags, str(_C_SRC), "-o", str(out_so), "-lm"]
                res = subprocess.run(cmd, capture_output=True, timeout=15)
                if res.returncode == 0 and out_so.exists():
                    return out_so
            except Exception:
                continue
    return prebuilt if prebuilt.exists() else None


def _bind_signatures(lib: ctypes.CDLL) -> None:
    lib.vml_has_openmp.argtypes = []
    lib.vml_has_openmp.restype = c_int

    lib.vml_value_noise_sample_f32.argtypes = [
        c_void_p, c_void_p, c_float,
        c_void_p, c_void_p, c_void_p, c_void_p,
        c_void_p, c_int, c_int, c_int,
    ]
    lib.vml_value_noise_sample_f32.restype = None

    lib.vml_rgb_to_yiq_f32.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int]
    lib.vml_rgb_to_yiq_f32.restype = None

    lib.vml_yiq_to_rgb_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p,
        c_int, c_int, c_int, c_int, c_void_p,
    ]
    lib.vml_yiq_to_rgb_f32.restype = None

    lib.vml_composite_encode_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_float, c_float, c_float, c_int, c_void_p, c_int,
    ]
    lib.vml_composite_encode_f32.restype = None

    lib.vml_remod_chroma_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_void_p, c_int,
    ]
    lib.vml_remod_chroma_f32.restype = None

    lib.vml_demod_scale_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_float, c_void_p, c_void_p, c_int,
    ]
    lib.vml_demod_scale_f32.restype = None

    lib.vml_comb_split_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_float,
    ]
    lib.vml_comb_split_f32.restype = None

    lib.vml_sym_fir_horiz_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int,
    ]
    lib.vml_sym_fir_horiz_f32.restype = None

    lib.vml_sym_fir_vert_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_float, c_float,
    ]
    lib.vml_sym_fir_vert_f32.restype = None

    lib.vml_sym_fir_2d_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int, c_float,
    ]
    lib.vml_sym_fir_2d_f32.restype = None

    lib.vml_demod_chroma_fused_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_float,
        c_void_p, c_int, c_void_p, c_int,
        c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int,
    ]
    lib.vml_demod_chroma_fused_f32.restype = None

    lib.vml_ca_vert_merge_f32.argtypes = [
        c_void_p, c_void_p, c_int,
        c_void_p, c_void_p, c_int,
        c_void_p, c_void_p, c_int,
        c_void_p, c_int, c_int,
    ]
    lib.vml_ca_vert_merge_f32.restype = None

    lib.vml_one_pole_rows_f32.argtypes = [
        c_void_p, c_void_p, c_int, c_int, c_float, c_int,
    ]
    lib.vml_one_pole_rows_f32.restype = None

    lib.vml_preemphasis_rows_f32.argtypes = [
        c_void_p, c_void_p, c_int, c_int, c_float, c_float,
    ]
    lib.vml_preemphasis_rows_f32.restype = None

    lib.vml_biquad_rows_f32.argtypes = [
        c_void_p, c_void_p, c_int, c_int,
        c_float, c_float, c_float, c_float, c_float, c_float, c_float,
    ]
    lib.vml_biquad_rows_f32.restype = None

    lib.vml_freq_response_lpf_rows_f32.argtypes = [
        c_void_p, c_void_p, c_int, c_int, c_float, c_float,
    ]
    lib.vml_freq_response_lpf_rows_f32.restype = None

    lib.vml_horizontal_shift_2d_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_int, c_int, c_int,
    ]
    lib.vml_horizontal_shift_2d_f32.restype = None

    lib.vml_horizontal_shift_3planes_f32.argtypes = [
        c_void_p, c_void_p, c_void_p,
        c_void_p, c_void_p, c_void_p,
        c_void_p, c_int, c_int, c_int,
    ]
    lib.vml_horizontal_shift_3planes_f32.restype = None

    lib.vml_vertical_shift_2d_f32.argtypes = [
        c_void_p, c_void_p, c_float, c_int, c_int,
    ]
    lib.vml_vertical_shift_2d_f32.restype = None

    lib.vml_roll_off_f32.argtypes = [
        c_void_p, c_void_p, c_int, c_float, c_float, c_float, c_float,
    ]
    lib.vml_roll_off_f32.restype = None

    lib.vml_signal_noise_envelope_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p,
        c_void_p, c_void_p, c_int, c_int,
    ]
    lib.vml_signal_noise_envelope_f32.restype = None

    lib.vml_camera_apply_sensor_fpn_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_float, c_float, c_float, c_float, c_float, c_float, c_int,
    ]
    lib.vml_camera_apply_sensor_fpn_f32.restype = None

    lib.vml_color_science_fast_f32.argtypes = [
        c_void_p, c_void_p, c_int,
        c_float, c_float, c_float, c_float, c_int, c_int,
    ]
    lib.vml_color_science_fast_f32.restype = None

    lib.vml_camera_wb_expose_blur_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_int,
        c_float, c_float, c_float, c_float,
    ]
    lib.vml_camera_wb_expose_blur_f32.restype = None

    lib.vml_ca_warp_split_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p,
        c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_int, c_int,
    ]
    lib.vml_ca_warp_split_f32.restype = None

    lib.vml_smear_upsample_blend_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int,
    ]
    lib.vml_smear_upsample_blend_f32.restype = None

    lib.vml_upsample_add_rgb_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int,
    ]
    lib.vml_upsample_add_rgb_f32.restype = None

    lib.vml_chroma_rotate_2d_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_float, c_int,
    ]
    lib.vml_chroma_rotate_2d_f32.restype = None

    lib.vml_format_color_noise_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_float, c_int, c_int,
    ]
    lib.vml_format_color_noise_f32.restype = None

    lib.vml_tape_noise_floor_f32.argtypes = [
        c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
        c_float, c_float, c_void_p, c_int, c_int,
    ]
    lib.vml_tape_noise_floor_f32.restype = None


def get_native_lib() -> ctypes.CDLL | None:
    global _LIB, _LIB_FAILED
    if _LIB is not None:
        return _LIB
    if _LIB_FAILED or os.environ.get("VML_DISABLE_NATIVE") == "1":
        return None
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        if _LIB_FAILED:
            return None
        candidates: list[Path] = []
        compiled = _try_compile_native()
        if compiled is not None:
            candidates.append(compiled)
        if sys.platform == "win32":
            candidates.append(_HERE / "vml_kernels_win_amd64.dll")
        elif sys.platform.startswith("linux"):
            candidates.append(_HERE / "vml_kernels_linux_x86_64.so")

        for cand in candidates:
            if cand is not None and cand.exists():
                try:
                    lib = ctypes.CDLL(str(cand))
                    _bind_signatures(lib)
                    _LIB = lib
                    return _LIB
                except Exception:
                    continue
        _LIB_FAILED = True
        return None


def get_scratch(name: str, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    cache = getattr(_TLS, "scratch", None)
    if cache is None:
        cache = {}
        _TLS.scratch = cache
    arr = cache.get(name)
    if arr is None or arr.shape != shape or arr.dtype != dtype:
        arr = np.empty(shape, dtype=dtype)
        cache[name] = arr
    return arr


# ---------------------------------------------------------------------------
# kernel design
# ---------------------------------------------------------------------------
def gaussian_kernel(sigma: float) -> np.ndarray:
    """
    Finite Gaussian kernel, normalised to unity DC gain.
    """
    sigma = float(max(1e-6, sigma))
    if sigma < 0.30:
        return _ONE_KERNEL
    key = int(round(sigma * 100000.0))
    hit = _GAUSS_CACHE.get(key)
    if hit is not None:
        return hit
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    out = (k / k.sum()).astype(np.float32)
    if len(_GAUSS_CACHE) > 256:
        _GAUSS_CACHE.pop(next(iter(_GAUSS_CACHE)), None)
    _GAUSS_CACHE[key] = out
    return out


def gaussian_half_kernel(sigma: float) -> tuple[np.ndarray, int]:
    """Return (k[R:], R) for symmetric C FIR kernels."""
    sigma = float(max(1e-6, sigma))
    if sigma < 0.30:
        return _ONE_KERNEL, 0
    key = int(round(sigma * 100000.0))
    hit = _GAUSS_HALF_CACHE.get(key)
    if hit is not None:
        return hit, len(hit) - 1
    k = gaussian_kernel(sigma)
    r = len(k) // 2
    kh = np.ascontiguousarray(k[r:], dtype=np.float32)
    if len(_GAUSS_HALF_CACHE) > 256:
        _GAUSS_HALF_CACHE.pop(next(iter(_GAUSS_HALF_CACHE)), None)
    _GAUSS_HALF_CACHE[key] = kh
    return kh, r


def highpass_kernel(sigma: float) -> np.ndarray:
    """Complementary high-pass: delta - lowpass (DC removed exactly)."""
    k = gaussian_kernel(sigma)
    hp = -k.copy()
    c = len(k) // 2
    hp[c] += 1.0
    return hp


def highpass_half_kernel(sigma: float) -> tuple[np.ndarray, int]:
    """Return (hp[R:], R) for symmetric C highpass FIR kernels."""
    sigma = float(max(1e-6, sigma))
    if sigma < 0.30:
        return np.zeros(1, dtype=np.float32), 0
    key = int(round(sigma * 100000.0))
    hit = _HP_HALF_CACHE.get(key)
    if hit is not None:
        return hit, len(hit) - 1
    hp = highpass_kernel(sigma)
    r = len(hp) // 2
    hph = np.ascontiguousarray(hp[r:], dtype=np.float32)
    if len(_HP_HALF_CACHE) > 256:
        _HP_HALF_CACHE.pop(next(iter(_HP_HALF_CACHE)), None)
    _HP_HALF_CACHE[key] = hph
    return hph, r


def bandpass_kernel(lo_sigma: float, hi_sigma: float) -> np.ndarray:
    """Band-pass as lowpass(hi) - lowpass(lo), both centred."""
    khi = gaussian_kernel(hi_sigma)
    klo = gaussian_kernel(lo_sigma)
    n = max(len(khi), len(klo))
    if n % 2 == 0:
        n += 1
    out = np.zeros(n, dtype=np.float32)
    for k in (khi, -klo):
        c = len(k) // 2
        off = n // 2 - c
        out[off:off + len(k)] += k
    return out


def _fft_fir_1d(plane: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Exact reflect-padded 1D FIR filter via float32 pocketfft for wide kernels."""
    klen = len(kernel)
    rad = klen // 2
    pad_width = [(0, 0)] * plane.ndim
    pad_width[axis] = (rad, rad)
    padded = np.pad(plane, pad_width, mode="symmetric")
    n = padded.shape[axis]
    nfft = _next_fast_len(n + klen - 1, real=True)
    fk = _sp_rfft(kernel[::-1], n=nfft)
    shape = [1] * plane.ndim
    shape[axis] = -1
    fk = fk.reshape(shape)
    fp = _sp_rfft(padded, n=nfft, axis=axis)
    out = _sp_irfft(fp * fk, n=nfft, axis=axis)
    sl = [slice(None)] * plane.ndim
    sl[axis] = slice(2 * rad, 2 * rad + plane.shape[axis])
    return out[tuple(sl)].astype(plane.dtype, copy=False)


def apply_fir(plane: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Separable FIR along one axis, reflect-padded at the edges."""
    klen = len(kernel)
    if klen == 1:
        return plane
    if (
        _HAVE_CV2
        and plane.ndim in (2, 3)
        and plane.dtype == np.float32
        and (plane.ndim == 2 or plane.shape[2] <= 4)
        and klen <= 255
        and plane.shape[0] > klen
        and plane.shape[1] > klen
    ):
        k_rev = kernel[::-1]
        if axis == 1:
            return _cv2.sepFilter2D(plane, _cv2.CV_32F, k_rev, _ONE_KERNEL, borderType=_cv2.BORDER_REFLECT)
        if axis == 0:
            return _cv2.sepFilter2D(plane, _cv2.CV_32F, _ONE_KERNEL, k_rev, borderType=_cv2.BORDER_REFLECT)
    if klen >= 39 and plane.dtype == np.float32 and plane.shape[axis] >= klen:
        return _fft_fir_1d(plane, kernel, axis)
    return convolve1d(plane, kernel, axis=axis, mode="reflect")


def lowpass(plane: np.ndarray, sigma: float, axis: int | None = None) -> np.ndarray:
    kh, r = gaussian_half_kernel(sigma)
    if r == 0:
        return plane
    lib = get_native_lib()
    if (
        lib is not None
        and plane.ndim in (2, 3)
        and plane.dtype == np.float32
        and r <= 32
    ):
        src = np.ascontiguousarray(plane, dtype=np.float32)
        h, w = src.shape[0], src.shape[1]
        c = 1 if src.ndim == 2 else int(src.shape[2])
        dst = np.empty_like(src)
        if axis is None:
            tmp = get_scratch("fir2d_tmp", src.shape, np.float32)
            lib.vml_sym_fir_2d_f32(
                src.ctypes.data, tmp.ctypes.data, dst.ctypes.data,
                kh.ctypes.data, r, h, w, c, 0.0,
            )
            return dst
        if axis == 1:
            lib.vml_sym_fir_horiz_f32(
                src.ctypes.data, dst.ctypes.data, kh.ctypes.data, r, h, w, c,
            )
            return dst
        if axis == 0:
            lib.vml_sym_fir_vert_f32(
                src.ctypes.data, dst.ctypes.data, kh.ctypes.data, r, h, w, c, 0.0, 1.0,
            )
            return dst

    k = gaussian_kernel(sigma)
    klen = len(k)
    if klen == 1:
        return plane
    if axis is None:
        if (
            _HAVE_CV2
            and plane.ndim in (2, 3)
            and plane.dtype == np.float32
            and (plane.ndim == 2 or plane.shape[2] <= 4)
            and klen <= 255
            and plane.shape[0] > klen
            and plane.shape[1] > klen
        ):
            return _cv2.sepFilter2D(plane, _cv2.CV_32F, k, k, borderType=_cv2.BORDER_REFLECT)
        return apply_fir(apply_fir(plane, k, 1), k, 0)
    return apply_fir(plane, k, axis)


def unsharp_mask_2d(rgb: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    """Fused 2D Gaussian blur + unsharp mask: rgb + amount * (rgb - lowpass(rgb, sigma))."""
    if abs(amount) <= 1e-6:
        return rgb
    kh, r = gaussian_half_kernel(sigma)
    if r == 0:
        return rgb
    lib = get_native_lib()
    if lib is not None and rgb.ndim in (2, 3) and rgb.dtype == np.float32 and r <= 32:
        src = np.ascontiguousarray(rgb, dtype=np.float32)
        h, w = src.shape[0], src.shape[1]
        c = 1 if src.ndim == 2 else int(src.shape[2])
        tmp = get_scratch("fir2d_tmp", src.shape, np.float32)
        dst = np.empty_like(src)
        lib.vml_sym_fir_2d_f32(
            src.ctypes.data, tmp.ctypes.data, dst.ctypes.data,
            kh.ctypes.data, r, h, w, c, float(amount),
        )
        return dst
    blurred = lowpass(rgb, sigma)
    if _HAVE_CV2 and rgb.dtype == np.float32:
        return _cv2.addWeighted(rgb, 1.0 + float(amount), blurred, -float(amount), 0.0)
    return (rgb + np.float32(amount) * (rgb - blurred)).astype(np.float32, copy=False)


def highpass(plane: np.ndarray, sigma: float, axis: int | None = None) -> np.ndarray:
    if axis == 1:
        hph, r = highpass_half_kernel(sigma)
        if r == 0:
            return np.zeros_like(plane)
        lib = get_native_lib()
        if lib is not None and plane.ndim in (2, 3) and plane.dtype == np.float32 and r <= 32:
            src = np.ascontiguousarray(plane, dtype=np.float32)
            h, w = src.shape[0], src.shape[1]
            c = 1 if src.ndim == 2 else int(src.shape[2])
            dst = np.empty_like(src)
            lib.vml_sym_fir_horiz_f32(
                src.ctypes.data, dst.ctypes.data, hph.ctypes.data, r, h, w, c,
            )
            return dst
    k = highpass_kernel(sigma)
    if axis is None:
        return apply_fir(apply_fir(plane, k, 1), k, 0)
    return apply_fir(plane, k, axis)


def bandpass(plane: np.ndarray, lo_sigma: float, hi_sigma: float,
             axis: int | None = None) -> np.ndarray:
    k = bandpass_kernel(lo_sigma, hi_sigma)
    if axis is None:
        return apply_fir(apply_fir(plane, k, 1), k, 0)
    return apply_fir(plane, k, axis)


def horizontal_shift(plane: np.ndarray, shifts: np.ndarray,
                     mode: str = "reflect") -> np.ndarray:
    """
    Shift every row of `plane` horizontally by its own (fractional) amount.
    """
    h, w = plane.shape[0], plane.shape[1]
    sh = np.ascontiguousarray(np.asarray(shifts, dtype=np.float32).reshape(-1))
    if sh.size < h:
        sh = np.ascontiguousarray(np.pad(sh, (0, h - sh.size), mode="edge"))
    elif sh.size > h:
        sh = np.ascontiguousarray(sh[:h])
    max_s = float(np.max(np.abs(sh)))
    if max_s < 1e-6:
        return plane

    lib = get_native_lib()
    if lib is not None and plane.ndim == 2 and plane.dtype == np.float32:
        src = np.ascontiguousarray(plane, dtype=np.float32)
        dst = np.empty_like(src)
        mode_id = 2 if mode == "wrap" else (1 if mode == "edge" else 0)
        lib.vml_horizontal_shift_2d_f32(
            src.ctypes.data, dst.ctypes.data, sh.ctypes.data, h, w, mode_id,
        )
        return dst

    left = np.floor(sh).astype(np.int32)
    step = np.where(sh > 1e-6, -1, np.where(sh < -1e-6, 1, 0)).astype(np.int32)
    frac = (sh - left).astype(plane.dtype)

    if mode != "wrap" and plane.ndim == 2:
        pad = int(max(abs(int(left.min())), abs(int(left.max()) + 1)) + 1)
        pad = int(min(pad, 4096))
        if pad > 0:
            pad_mode = "reflect" if mode == "reflect" else "edge"
            padded = np.pad(plane, ((0, 0), (pad, pad)), mode=pad_mode)
        else:
            padded = plane
        out = np.empty_like(plane)
        c0_arr = pad - left
        c1_arr = c0_arr + step
        if h <= 32:
            for r in range(h):
                s_r = int(step[r])
                c0 = int(c0_arr[r])
                if s_r == 0 or frac[r] == 0.0:
                    out[r] = padded[r, c0 : c0 + w]
                else:
                    c1 = int(c1_arr[r])
                    a = padded[r, c0 : c0 + w]
                    b = padded[r, c1 : c1 + w]
                    out[r] = a + (b - a) * frac[r]
            return out
        pair_key = (c0_arr.astype(np.int64) << 16) | (c1_arr.astype(np.int64) & 0xFFFF)
        for uk in np.unique(pair_key):
            c0 = int(uk >> 16)
            c1 = int(uk & 0xFFFF)
            rows = np.flatnonzero(pair_key == uk)
            if c0 == c1:
                out[rows] = padded[rows, c0 : c0 + w]
            else:
                f = frac[rows, None]
                a = padded[rows, c0 : c0 + w]
                out[rows] = a + (padded[rows, c1 : c1 + w] - a) * f
        return out

    cols = np.arange(w, dtype=np.int32)[None, :]
    step_2d = step[:, None]
    rows = np.arange(h, dtype=np.int32)[:, None]
    if mode == "wrap":
        idx0 = np.mod(cols - left[:, None], w)
        idx1 = np.mod(idx0 + step_2d, w)
        a = plane[rows, idx0]
        b = plane[rows, idx1]
    else:
        pad = int(max(abs(int(left.min())), abs(int(left.max()) + 1)) + 1)
        pad = int(min(pad, 4096))
        if pad > 0:
            pad_mode = "reflect" if mode == "reflect" else "edge"
            padded = np.pad(plane, ((0, 0), (pad, pad)) + ((0, 0),) * (plane.ndim - 2),
                            mode=pad_mode)
        else:
            padded = plane
        idx0 = cols - left[:, None] + pad
        idx1 = idx0 + step_2d
        a = padded[rows, idx0]
        b = padded[rows, idx1]

    frac_2d = frac[:, None] if plane.ndim == 2 else frac[:, None, None]
    b = b - a
    b *= frac_2d
    b += a
    return b.astype(plane.dtype, copy=False)


def horizontal_shift_3planes(
    y: np.ndarray, i: np.ndarray, q: np.ndarray,
    shifts: np.ndarray, mode: str = "reflect"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift (y, i, q) simultaneously by the same row shift vector."""
    h, w = y.shape[0], y.shape[1]
    sh = np.ascontiguousarray(np.asarray(shifts, dtype=np.float32).reshape(-1))
    if sh.size < h:
        sh = np.ascontiguousarray(np.pad(sh, (0, h - sh.size), mode="edge"))
    elif sh.size > h:
        sh = np.ascontiguousarray(sh[:h])
    if float(np.max(np.abs(sh))) < 1e-6:
        return y, i, q
    lib = get_native_lib()
    if lib is not None and y.ndim == 2 and y.dtype == np.float32:
        yc = np.ascontiguousarray(y, dtype=np.float32)
        ic = np.ascontiguousarray(i, dtype=np.float32)
        qc = np.ascontiguousarray(q, dtype=np.float32)
        yo = np.empty_like(yc)
        io = np.empty_like(ic)
        qo = np.empty_like(qc)
        mode_id = 2 if mode == "wrap" else (1 if mode == "edge" else 0)
        lib.vml_horizontal_shift_3planes_f32(
            yc.ctypes.data, ic.ctypes.data, qc.ctypes.data,
            yo.ctypes.data, io.ctypes.data, qo.ctypes.data,
            sh.ctypes.data, h, w, mode_id,
        )
        return yo, io, qo
    return horizontal_shift(y, sh, mode), horizontal_shift(i, sh, mode), horizontal_shift(q, sh, mode)


def vertical_shift(plane: np.ndarray, shift: float, mode: str = "edge") -> np.ndarray:
    """Whole-plane (or whole-picture) vertical shift in pixels."""
    h = plane.shape[0]
    if abs(shift) < 1e-6:
        return plane
    lib = get_native_lib()
    if lib is not None and plane.ndim == 2 and plane.dtype == np.float32:
        src = np.ascontiguousarray(plane, dtype=np.float32)
        dst = np.empty_like(src)
        lib.vml_vertical_shift_2d_f32(
            src.ctypes.data, dst.ctypes.data, float(shift), h, int(plane.shape[1]),
        )
        return dst
    i0 = int(np.floor(shift))
    frac = float(shift - i0)
    rows0 = np.clip(np.arange(h) + i0, 0, h - 1)
    rows1 = np.clip(np.arange(h) + i0 + (1 if shift > 0 else -1), 0, h - 1)
    a = plane[rows0]
    b = plane[rows1]
    return (a + (b - a) * np.float32(frac)).astype(plane.dtype, copy=False)


def shift_line_band(plane: np.ndarray, row0: int, row1: int, amount: float) -> np.ndarray:
    """Shift only rows [row0, row1) -- used for head-switching / creases."""
    row0 = int(max(0, row0))
    row1 = int(min(plane.shape[0], row1))
    if row1 <= row0 or abs(amount) < 1e-6:
        return plane
    band = plane[row0:row1]
    shifts = np.full(band.shape[0], amount, dtype=np.float32)
    plane = plane.copy()
    plane[row0:row1] = horizontal_shift(band, shifts)
    return plane


# ---------------------------------------------------------------------------
# colour transforms
# ---------------------------------------------------------------------------
def rgb_to_yiq(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lib = get_native_lib()
    if lib is not None and rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.float32:
        src = np.ascontiguousarray(rgb, dtype=np.float32)
        h, w = src.shape[0], src.shape[1]
        y = np.empty((h, w), dtype=np.float32)
        i = np.empty((h, w), dtype=np.float32)
        q = np.empty((h, w), dtype=np.float32)
        lib.vml_rgb_to_yiq_f32(src.ctypes.data, y.ctypes.data, i.ctypes.data, q.ctypes.data, h * w)
        return y, i, q
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = r * np.float32(0.299) + g * np.float32(0.587) + b * np.float32(0.114)
    i = r * np.float32(0.5959) - g * np.float32(0.2746) - b * np.float32(0.3213)
    q = r * np.float32(0.2115) - g * np.float32(0.5227) + b * np.float32(0.3112)
    return y, i, q


def yiq_to_rgb(y: np.ndarray, i: np.ndarray, q: np.ndarray) -> np.ndarray:
    lib = get_native_lib()
    if lib is not None and y.ndim == 2 and y.dtype == np.float32:
        yc = np.ascontiguousarray(y, dtype=np.float32)
        ic = np.ascontiguousarray(i, dtype=np.float32)
        qc = np.ascontiguousarray(q, dtype=np.float32)
        h, w = yc.shape[0], yc.shape[1]
        rgb = np.empty((h, w, 3), dtype=np.float32)
        lib.vml_yiq_to_rgb_f32(
            yc.ctypes.data, ic.ctypes.data, qc.ctypes.data, None,
            0, h, w, 0, rgb.ctypes.data,
        )
        return rgb
    r = y + np.float32(0.9563) * i + np.float32(0.6210) * q
    g = y - np.float32(0.2721) * i - np.float32(0.6474) * q
    b = y - np.float32(1.1070) * i + np.float32(1.7046) * q
    if _HAVE_CV2 and y.ndim == 2 and y.dtype == np.float32:
        return _cv2.merge((r, g, b))
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# the frame passed between pipeline stages
# ---------------------------------------------------------------------------
@dataclass
class SignalFrame:
    """
    A frame *as a video signal*, not as a picture.

    y : luma plane,  float32 (H, W), nominal 0..1
    i, q : chroma planes, float32 (H, W)
    """
    y: np.ndarray
    i: np.ndarray
    q: np.ndarray

    @property
    def height(self) -> int:
        return int(self.y.shape[0])

    @property
    def width(self) -> int:
        return int(self.y.shape[1])

    def copy(self) -> "SignalFrame":
        sf = SignalFrame(self.y.copy(), self.i.copy(), self.q.copy())
        if hasattr(self, "color_system"):
            sf.color_system = getattr(self, "color_system")  # type: ignore[attr-defined]
        if getattr(self, "_rf_direct", False):
            sf._rf_direct = True  # type: ignore[attr-defined]
        return sf

    @classmethod
    def from_rgb(cls, rgb: np.ndarray, std: VideoStandard) -> "SignalFrame":
        y, i, q = rgb_to_yiq(rgb.astype(np.float32, copy=False))
        return cls(y.astype(np.float32, copy=False),
                   i.astype(np.float32, copy=False),
                   q.astype(np.float32, copy=False))

    def to_rgb(self, cross_luma: np.ndarray | None = None) -> np.ndarray:
        is_pal = 1 if str(getattr(self, "color_system", "NTSC")).upper() == "PAL" else 0
        lib = get_native_lib()
        if lib is not None and self.y.ndim == 2 and self.y.dtype == np.float32:
            yc = np.ascontiguousarray(self.y, dtype=np.float32)
            ic = np.ascontiguousarray(self.i, dtype=np.float32)
            qc = np.ascontiguousarray(self.q, dtype=np.float32)
            clc = np.ascontiguousarray(cross_luma, dtype=np.float32) if cross_luma is not None else None
            h, w = yc.shape[0], yc.shape[1]
            rgb = np.empty((h, w, 3), dtype=np.float32)
            lib.vml_yiq_to_rgb_f32(
                yc.ctypes.data, ic.ctypes.data, qc.ctypes.data,
                clc.ctypes.data if clc is not None else None,
                is_pal, h, w, 1, rgb.ctypes.data,
            )
            return rgb

        y = self.y if cross_luma is None else self.y + cross_luma
        if is_pal:
            alt = np.where((np.arange(self.height) % 2 == 0)[:, None], 1.0, -1.0).astype(np.float32)
            q = self.q * alt
        else:
            q = self.q
        rgb = yiq_to_rgb(y, self.i, q)
        np.clip(rgb, 0.0, 1.0, out=rgb)
        return rgb.astype(np.float32, copy=False)

    def blend_to(self, other: "SignalFrame", t: float) -> "SignalFrame":
        t = float(np.clip(t, 0.0, 1.0))
        return SignalFrame(
            self.y * (1 - t) + other.y * t,
            self.i * (1 - t) + other.i * t,
            self.q * (1 - t) + other.q * t,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "y_mean": float(self.y.mean()),
            "y_std": float(self.y.std()),
            "i_std": float(self.i.std()),
            "q_std": float(self.q.std()),
        }


# ---------------------------------------------------------------------------
# composite encoding: the step that creates crosstalk by construction
# ---------------------------------------------------------------------------
def _subcarrier_tables(width: int, height: int, std: VideoStandard) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (int(width), int(height), std.name,
           round(float(std.subcarrier_rad_per_sample), 7),
           round(float(std.samples_per_line), 4))
    hit = _SUBCARRIER_CACHE.get(key)
    if hit is not None:
        return hit
    inc = std.subcarrier_rad_per_sample
    cols = np.arange(width, dtype=np.float64)[None, :]
    rows = np.arange(height, dtype=np.float64)[:, None]
    ph = np.ascontiguousarray((inc * cols + std.line_hz * 0.0 * rows + inc * std.samples_per_line * rows), dtype=np.float32)
    cos_ph = np.ascontiguousarray(np.cos(ph), dtype=np.float32)
    sin_ph = np.ascontiguousarray(np.sin(ph), dtype=np.float32)
    res = (ph, cos_ph, sin_ph)
    if len(_SUBCARRIER_CACHE) > 24:
        _SUBCARRIER_CACHE.pop(next(iter(_SUBCARRIER_CACHE)), None)
    _SUBCARRIER_CACHE[key] = res
    return res


def subcarrier_phase(width: int, height: int, std: VideoStandard) -> np.ndarray:
    """
    (H, W) subcarrier phase at sample centres.
    """
    return _subcarrier_tables(width, height, std)[0]


def composite_encode(y: np.ndarray, i: np.ndarray, q: np.ndarray,
                     std: VideoStandard, chroma_gain: float = 0.63,
                     burst_phase: float = 0.0) -> np.ndarray:
    """Y + modulated chroma -> one composite signal (0..1 nominal)."""
    _, cos_ph, sin_ph = _subcarrier_tables(y.shape[1], y.shape[0], std)
    lib = get_native_lib()
    if lib is not None and y.ndim == 2 and y.dtype == np.float32:
        yc = np.ascontiguousarray(y, dtype=np.float32)
        ic = np.ascontiguousarray(i, dtype=np.float32)
        qc = np.ascontiguousarray(q, dtype=np.float32)
        out = np.empty_like(yc)
        has_b = 1 if abs(burst_phase) >= 1e-8 else 0
        cb = float(np.cos(burst_phase)) if has_b else 1.0
        sb = float(np.sin(burst_phase)) if has_b else 0.0
        lib.vml_composite_encode_f32(
            yc.ctypes.data, ic.ctypes.data, qc.ctypes.data,
            cos_ph.ctypes.data, sin_ph.ctypes.data,
            float(chroma_gain), cb, sb, has_b,
            out.ctypes.data, yc.size,
        )
        return out
    if abs(burst_phase) < 1e-8:
        c = np.float32(chroma_gain) * (i * cos_ph + q * sin_ph)
    else:
        cb = np.float32(np.cos(burst_phase))
        sb = np.float32(np.sin(burst_phase))
        cos_tot = cos_ph * cb - sin_ph * sb
        sin_tot = sin_ph * cb + cos_ph * sb
        c = np.float32(chroma_gain) * (i * cos_tot + q * sin_tot)
    return (y + c).astype(np.float32, copy=False)


def _reflect_pad(plane: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return plane
    pad = int(min(pad, max(1, plane.shape[1] // 2 - 1)))
    axes = ((0, 0), (pad, pad)) + ((0, 0),) * (plane.ndim - 2)
    return np.pad(plane, axes, mode="reflect")


def fft_bandpass(plane: np.ndarray, lo_hz: float, hi_hz: float,
                 std: VideoStandard, smooth_hz: float | None = None,
                 unity_at_hz: float | None = None) -> np.ndarray:
    """
    Zero-phase horizontal band-pass with an exact, flat pass-band.
    """
    pad = int(min(64, max(32, plane.shape[1] // 16)))
    src = _reflect_pad(plane, pad)
    n = src.shape[1]
    bp_key = (
        n,
        round(float(lo_hz), 1),
        round(float(hi_hz), 1),
        round(float(std.pixel_rate_hz), 1),
        None if smooth_hz is None else round(float(smooth_hz), 1),
        None if unity_at_hz is None else round(float(unity_at_hz), 1),
    )
    response = _FFT_BP_CACHE.get(bp_key)
    if response is None:
        freqs = np.fft.rfftfreq(n, d=1.0) * std.pixel_rate_hz
        order = 4
        hi_c = 0.5 * (lo_hz + hi_hz) + 0.5 * (hi_hz - lo_hz) * 1.6
        lo_c = 0.5 * (lo_hz + hi_hz) - 0.5 * (hi_hz - lo_hz) * 1.6
        hi_c = max(hi_c, lo_hz * 1.02)
        lo_c = max(1e4, min(lo_c, hi_c * 0.97))
        hi = 1.0 / (1.0 + (freqs / hi_c) ** (2 * order))
        lo = (freqs / lo_c) ** (2 * order) / (1.0 + (freqs / lo_c) ** (2 * order))
        resp = hi * lo
        if unity_at_hz:
            g = float(np.interp(unity_at_hz, freqs, resp))
            if g > 1e-6:
                resp = resp / g
        response = resp.astype(np.float32)
        if len(_FFT_BP_CACHE) > 64:
            _FFT_BP_CACHE.pop(next(iter(_FFT_BP_CACHE)), None)
        _FFT_BP_CACHE[bp_key] = response
    spec = _sp_rfft(src, axis=1, workers=2, overwrite_x=True)
    resp_view = response[None, :] if plane.ndim == 2 else response[None, :, None]
    spec *= resp_view
    out = _sp_irfft(spec, n=n, axis=1, workers=2, overwrite_x=True).astype(plane.dtype, copy=False)
    return out[:, pad:pad + plane.shape[1]] if plane.ndim == 2 else \
        out[:, pad:pad + plane.shape[1], :]


def split_composite(comp: np.ndarray, std: VideoStandard,
                    comb_depth: float = 0.0, chroma_bandwidth_hz: float = 0.8e6,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Band-split a composite signal into (luma, chroma).
    """
    fsc = std.subcarrier_hz
    half = max(float(chroma_bandwidth_hz), 0.30e6)
    chroma_raw = fft_bandpass(comp, max(0.10e6, fsc - 1.6 * half), fsc + 1.6 * half,
                              std, unity_at_hz=fsc)

    lib = get_native_lib()
    if lib is not None and comp.ndim == 2 and comp.dtype == np.float32:
        cc = np.ascontiguousarray(comp, dtype=np.float32)
        ch_in = np.ascontiguousarray(chroma_raw, dtype=np.float32)
        luma_out = np.empty_like(cc)
        chroma_out = np.empty_like(cc)
        lib.vml_comb_split_f32(
            cc.ctypes.data, ch_in.ctypes.data,
            luma_out.ctypes.data, chroma_out.ctypes.data,
            cc.shape[0], cc.shape[1], float(comb_depth),
        )
        return luma_out, chroma_out

    chroma = chroma_raw
    if comb_depth > 0.001:
        delayed = np.empty_like(chroma)
        delayed[1:] = chroma[:-1]
        delayed[0] = chroma[0]
        combed = 0.5 * (chroma + delayed)
        mixed = (1.0 - comb_depth) * chroma + comb_depth * combed
        gain = 1.0 - comb_depth
        if abs(gain) > 1e-3:
            chroma = mixed / gain

    luma = comp - chroma
    return luma.astype(np.float32, copy=False), chroma.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# receiver Y/C separation topologies
# ---------------------------------------------------------------------------
YC_SEPARATION = ("band", "box", "line1", "line2", "fir2d", "iir")


def separation_kind_name(value) -> str:
    try:
        k = int(round(float(value)))
    except (TypeError, ValueError):
        return "band"
    if 0 <= k < len(YC_SEPARATION):
        return YC_SEPARATION[k]
    return "band"


FIR2D_DIAG_W = 0.0625


def _phase_alternates(std: VideoStandard) -> bool:
    cycles = std.subcarrier_hz / std.line_hz
    return abs(cycles - round(cycles)) > 0.25


def chroma_vert_blend(chroma: np.ndarray, feedback: float = 0.5) -> np.ndarray:
    """
    A one-pole vertical feedback filter on the chroma, as ntsc-rs's `chroma_vert_blend` is.

        out[i] = k*out[i-1] + (1-k)*in[i]
    """
    k = float(np.clip(feedback, 0.0, 0.995))
    if k <= 1e-6 or chroma.shape[0] <= 1:
        return chroma
    out = np.empty_like(chroma)
    out[0] = chroma[0]
    inv = np.float32(1.0 - k)
    b_arr = np.array([inv], dtype=chroma.dtype)
    a_arr = np.array([1.0, -k], dtype=chroma.dtype)
    zi_arr = (np.float32(k) * chroma[0:1]).astype(chroma.dtype, copy=False)
    out[1:], _ = _lfilter(b_arr, a_arr, chroma[1:], axis=0, zi=zi_arr)
    return out


def chroma_vert_coherence(feedback: float) -> float:
    """The 1/e coherence length in lines, for reporting and for a check to compare against."""
    k = float(np.clip(feedback, 0.0, 0.995))
    return 0.0 if k <= 1e-6 else float(-1.0 / np.log(k))


def yc_separate(comp: np.ndarray, std: VideoStandard, kind: str = "band",
                comb_depth: float = 0.0, aperture: int = 2,
                chroma_bandwidth_hz: float = 0.8e6,
                vert_blend: float = 0.5,
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a composite signal into (luma, chroma) using a named receiver topology.
    """
    if kind not in YC_SEPARATION:
        raise ValueError("unknown Y/C separation %r; expected one of %s"
                         % (kind, ", ".join(YC_SEPARATION)))
    if kind == "band":
        return split_composite(comp, std, comb_depth=comb_depth,
                               chroma_bandwidth_hz=chroma_bandwidth_hz)

    c = np.asarray(comp, np.float32)
    h, w = c.shape
    vert_ok = _phase_alternates(std) and h >= 3

    if kind == "box":
        ap = max(1, int(aperture))
        acc = np.zeros_like(c)
        n = 0
        for k in range(-ap, ap + 1):
            acc += np.roll(c, k, axis=1)
            n += 1
        luma = acc / float(n)
    elif kind == "line1" and vert_ok:
        prev = np.empty_like(c)
        prev[0] = c[0]
        prev[1:] = c[:-1]
        luma = 0.5 * (c + prev)
    elif kind == "line2" and vert_ok:
        prev = np.empty_like(c)
        nxt = np.empty_like(c)
        prev[0] = c[0]
        prev[1:] = c[:-1]
        nxt[-1] = c[-1]
        nxt[:-1] = c[1:]
        luma = 0.5 * c + 0.25 * prev + 0.25 * nxt
    elif kind == "iir" and vert_ok:
        luma = chroma_vert_blend(c, vert_blend)
    elif kind == "fir2d":
        ap = max(1, int(aperture))
        acc = np.zeros_like(c)
        total = 0.0

        def _tap(plane: np.ndarray, dy: int, dx: int, weight: float) -> None:
            nonlocal acc, total
            t = plane if dy == 0 else np.roll(plane, -dy, axis=0)
            acc = acc + weight * np.roll(t, dx, axis=1)
            total += weight

        _tap(c, 0, 0, 0.5)
        for k in range(1, ap + 1):
            w = 0.125 if k == 1 else 0.0625
            _tap(c, 0, k, w)
            _tap(c, 0, -k, w)
        if vert_ok:
            _tap(c, -1, 0, 0.125)
            _tap(c, 1, 0, 0.125)
            for dx in (1, -1):
                _tap(c, -1, dx, FIR2D_DIAG_W)
                _tap(c, 1, dx, FIR2D_DIAG_W)
        luma = acc / max(1e-6, total)
    else:
        return yc_separate(comp, std, "box", comb_depth, aperture, chroma_bandwidth_hz,
                           vert_blend)

    chroma = c - luma
    fsc = std.subcarrier_hz
    half = max(float(chroma_bandwidth_hz), 0.30e6)
    chroma = fft_bandpass(chroma, max(0.10e6, fsc - 1.6 * half), fsc + 1.6 * half,
                          std, unity_at_hz=fsc)
    return luma.astype(np.float32, copy=False), chroma.astype(np.float32, copy=False)


def demod_chroma(chroma: np.ndarray, std: VideoStandard,
                 i_bw_hz: float = 1.3e6, q_bw_hz: float = 0.5e6,
                 phase_offset: np.ndarray | None = None,
                 pre_scale: float = 1.0,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Synchronous detection of I and Q with their (asymmetric) bandwidths.
    """
    _, cos_ph, sin_ph = _subcarrier_tables(chroma.shape[1], chroma.shape[0], std)
    w = chroma.shape[1]
    eff_w = min(w, int(std.samples_per_line))
    i_lp = std.pixels_for_bandwidth(max(1e5, i_bw_hz), eff_w)
    q_lp = std.pixels_for_bandwidth(max(1e5, q_bw_hz), eff_w)

    lib = get_native_lib()
    if phase_offset is None and lib is not None and chroma.ndim == 2 and chroma.dtype == np.float32:
        kh_i, r_i = gaussian_half_kernel(i_lp)
        kh_q, r_q = gaussian_half_kernel(q_lp)
        if r_i > 0 and r_q > 0 and r_i <= 32 and r_q <= 32:
            ch_c = np.ascontiguousarray(chroma, dtype=np.float32)
            h = ch_c.shape[0]
            tmp_i = get_scratch("demod_ti", ch_c.shape, np.float32)
            tmp_q = get_scratch("demod_tq", ch_c.shape, np.float32)
            i_out = np.empty_like(ch_c)
            q_out = np.empty_like(ch_c)
            lib.vml_demod_chroma_fused_f32(
                ch_c.ctypes.data, cos_ph.ctypes.data, sin_ph.ctypes.data,
                float(2.0 * pre_scale),
                kh_i.ctypes.data, r_i,
                kh_q.ctypes.data, r_q,
                tmp_i.ctypes.data, tmp_q.ctypes.data,
                i_out.ctypes.data, q_out.ctypes.data,
                h, w,
            )
            return i_out, q_out

    if phase_offset is not None:
        po = np.asarray(phase_offset, dtype=np.float32).reshape(-1, 1)
        co = np.cos(po)
        so = np.sin(po)
        cos_tot = cos_ph * co - sin_ph * so
        sin_tot = sin_ph * co + cos_ph * so
    else:
        cos_tot = cos_ph
        sin_tot = sin_ph
    sc = np.float32(2.0 * pre_scale)
    i = lowpass(chroma * (sc * cos_tot), i_lp, axis=1)
    q = lowpass(chroma * (sc * sin_tot), q_lp, axis=1)
    return i.astype(np.float32, copy=False), q.astype(np.float32, copy=False)


def remod_chroma(i: np.ndarray, q: np.ndarray, std: VideoStandard,
                 chroma_gain: float = 0.63) -> np.ndarray:
    _, cos_ph, sin_ph = _subcarrier_tables(i.shape[1], i.shape[0], std)
    lib = get_native_lib()
    if lib is not None and i.ndim == 2 and i.dtype == np.float32:
        ic = np.ascontiguousarray(i, dtype=np.float32)
        qc = np.ascontiguousarray(q, dtype=np.float32)
        out = np.empty_like(ic)
        lib.vml_remod_chroma_f32(
            ic.ctypes.data, qc.ctypes.data,
            cos_ph.ctypes.data, sin_ph.ctypes.data,
            float(chroma_gain), out.ctypes.data, ic.size,
        )
        return out
    return (np.float32(chroma_gain) * (i * cos_ph + q * sin_ph)).astype(np.float32, copy=False)


def chroma_under_luma(chroma: np.ndarray, std: VideoStandard,
                      chroma_bandwidth_hz: float) -> np.ndarray:
    """
    VHS / Video8 record chroma *below* luma (colour-under), heterodyned down to
    ~629 kHz.
    """
    sigma_x = std.pixels_for_bandwidth(max(1e4, chroma_bandwidth_hz), chroma.shape[1])
    out = lowpass(chroma, sigma_x, axis=1)
    out = lowpass(out, 1.35, axis=0)   # 2-line vertical averaging in the receiver
    return out.astype(np.float32, copy=False)
