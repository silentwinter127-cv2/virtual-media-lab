"""
Analog-style recursive (IIR) filters, accelerated via C11 + OpenMP row-parallel kernels.

Why this exists, and why it is not just a different way to blur:

Every filter in this project was originally a zero-phase Gaussian FIR.  That is
convenient -- no group delay to compensate, no ringing -- and it is also *wrong*
in one specific way that matters visually.  A real video amplifier is a
capacitive network: a cascade of one-pole sections.  Its amplitude response is
roughly Gaussian-ish, but its **phase response is not zero**, and the resulting
group delay is precisely what makes chroma sit slightly to the right of luma on
a composite signal.  Colour fringing on hard edges is partly a *delay* artefact,
so a zero-phase filter cannot produce it however the bandwidth is set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
def one_pole_coeffs(cutoff_hz: float, rate_hz: float, sections: int = 1) -> float:
    """
    Single-pole coefficient for `y[n] = a*x[n] + (1-a)*y[n-1]`.
    """
    if cutoff_hz <= 0.0 or rate_hz <= 0.0:
        return 1.0
    n = max(1, int(sections))
    w = 2.0 * np.pi * min(float(cutoff_hz), 0.5 * float(rate_hz)) / float(rate_hz)
    k = 2.0 ** (1.0 / n) - 1.0
    return float(np.clip(w * w * k / (1.0 + w * w * k), 1e-6, 0.9999))


def apply_one_pole(plane: np.ndarray, a: float, sections: int = 1,
                   axis: int = 1) -> np.ndarray:
    """
    Cascaded one-pole low-pass along an axis, as a real recursive filter.
    """
    if sections <= 0 or a >= 0.9999 or a <= 1e-6:
        return plane
    from ..core.signal import get_native_lib
    lib = get_native_lib()
    if lib is not None and axis == 1 and plane.ndim == 2:
        src = np.ascontiguousarray(plane, dtype=np.float32)
        dst = np.empty_like(src)
        lib.vml_one_pole_rows_f32(
            src.ctypes.data, dst.ctypes.data,
            src.shape[0], src.shape[1], float(a), int(sections),
        )
        return dst

    from scipy.signal import lfilter
    out = plane.astype(np.float32, copy=True)
    zi_value = out.take([0], axis=axis) if out.shape[axis] else out
    zi = (np.float32(1.0 - a) * np.repeat(zi_value, 1, axis=axis)).astype(np.float32, copy=False)
    b_arr = np.array([a], dtype=np.float32)
    a_arr = np.array([1.0, -(1.0 - a)], dtype=np.float32)
    for _ in range(int(sections)):
        out, _ = lfilter(b_arr, a_arr, out, axis=axis, zi=zi)
    return out.astype(np.float32, copy=False)


def apply_butterworth(plane: np.ndarray, cutoff_hz: float, rate_hz: float,
                      axis: int = 1) -> np.ndarray:
    """
    2nd-order Butterworth biquad along an axis.
    """
    if cutoff_hz <= 0.0 or rate_hz <= 0.0:
        return plane
    f = min(2.0 * cutoff_hz, rate_hz) / rate_hz
    omega = np.pi * f
    sn, cs = np.sin(omega), np.cos(omega)
    alpha = sn / (2.0 * 0.7071067811865476)
    gain = 1.0 / (1.0 + alpha)
    b0 = float((1.0 - cs) * 0.5 * gain)
    b1 = float((1.0 - cs) * gain)
    b2 = b0
    a1 = float(-2.0 * cs * gain)
    a2 = float((1.0 - alpha) * gain)

    from ..core.signal import get_native_lib
    lib = get_native_lib()
    if lib is not None and axis == 1 and plane.ndim == 2:
        src = np.ascontiguousarray(plane, dtype=np.float32)
        dst = np.empty_like(src)
        lib.vml_biquad_rows_f32(
            src.ctypes.data, dst.ctypes.data,
            src.shape[0], src.shape[1],
            b0, b1, b2, a1, a2, 0.0, 1.0,
        )
        return dst

    from scipy.signal import lfilter
    b = np.array([b0, b1, b2], dtype=np.float32)
    a = np.array([1.0, a1, a2], dtype=np.float32)
    src_f32 = plane.astype(np.float32, copy=False)
    return lfilter(b, a, src_f32, axis=axis).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
@dataclass
class FilterChoice:
    """Which filter family a stage should use."""
    kind: str = "gaussian"      # "gaussian" | "constant_k" | "butterworth"
    sections: int = 3

    @property
    def recursive(self) -> bool:
        return self.kind in ("constant_k", "butterworth")

    def label(self) -> str:
        return {"gaussian": "高斯（零相位，理想）",
                "constant_k": "恒 K 型（真实电路，有群延迟）",
                "butterworth": "巴特沃斯（较陡）"}.get(self.kind, self.kind)


def lowpass_hz(plane: np.ndarray, cutoff_hz: float, rate_hz: float,
               choice: FilterChoice) -> np.ndarray:
    """
    Apply a horizontal low-pass with the chosen filter family.
    """
    if cutoff_hz <= 0.0 or cutoff_hz >= rate_hz:
        return plane
    if choice.kind == "constant_k":
        a = one_pole_coeffs(cutoff_hz, rate_hz, sections=choice.sections)
        return apply_one_pole(plane, a, sections=choice.sections)
    if choice.kind == "butterworth":
        return apply_butterworth(plane, cutoff_hz, rate_hz)
    from ..core.signal import lowpass
    sigma = 0.1325 / (cutoff_hz / rate_hz)
    out = lowpass(plane, sigma, axis=1)
    k = int(max(1.0, round(3.0 * sigma)))
    taps = np.exp(-0.5 * (np.arange(-k, k + 1, dtype=np.float64) / sigma) ** 2)
    total = float(taps.sum())
    if total > 1e-9:
        out = out / total
    return out.astype(plane.dtype)


def group_delay_samples(cutoff_hz: float, rate_hz: float,
                        sections: int = 3) -> float:
    """
    Group delay of the cascade, in samples.
    """
    a = one_pole_coeffs(cutoff_hz, rate_hz, sections=sections)
    return float(sections * (1.0 - a) / max(1e-9, a))


# ---------------------------------------------------------------------------
def notch(plane: np.ndarray, freq_norm: float, quality: float,
          amount: float) -> np.ndarray:
    """
    IIR notch filter, for the ringing that a sharp luminance transition excites.
    """
    f = float(np.clip(freq_norm, 0.0, 1.0))
    if f <= 1e-6 or amount <= 1e-6:
        return plane
    bandwidth = (f / max(1e-3, quality)) * np.pi
    w = f * np.pi
    beta = np.tan(bandwidth * 0.5)
    gain = 1.0 / (1.0 + beta)
    b0 = float(gain)
    b1 = float(-2.0 * np.cos(w) * gain)
    b2 = b0
    a1 = b1
    a2 = float(2.0 * gain - 1.0)

    from ..core.signal import get_native_lib
    lib = get_native_lib()
    if lib is not None and plane.ndim == 2:
        src = np.ascontiguousarray(plane, dtype=np.float32)
        dst = np.empty_like(src)
        lib.vml_biquad_rows_f32(
            src.ctypes.data, dst.ctypes.data,
            src.shape[0], src.shape[1],
            b0, b1, b2, a1, a2, 1.0, float(amount),
        )
        return dst

    from scipy.signal import lfilter
    b = np.array([b0, b1, b2], dtype=np.float32)
    a = np.array([1.0, a1, a2], dtype=np.float32)
    src_f32 = plane.astype(np.float32, copy=False)
    ring = lfilter(b, a, src_f32, axis=1)
    return (src_f32 + np.float32(amount) * ring).astype(np.float32, copy=False)


def preemphasis(plane: np.ndarray, cutoff_hz: float, rate_hz: float,
                amount: float) -> np.ndarray:
    """
    High-frequency pre-emphasis / aperture correction.
    """
    if amount <= 1e-6:
        return plane
    a = one_pole_coeffs(cutoff_hz, rate_hz, sections=1)
    from ..core.signal import get_native_lib
    lib = get_native_lib()
    if lib is not None and plane.ndim == 2:
        src = np.ascontiguousarray(plane, dtype=np.float32)
        dst = np.empty_like(src)
        lib.vml_preemphasis_rows_f32(
            src.ctypes.data, dst.ctypes.data,
            src.shape[0], src.shape[1], float(a), float(amount),
        )
        return dst
    lp = apply_one_pole(plane, a, sections=1)
    return (plane + amount * (plane - lp)).astype(np.float32)
