"""
Deterministic stochastic processes with the *right temporal structure*.

The single most important engineering rule in the reference plan:

    "Do not let all random numbers be regenerated every frame."

A tape being played back is one physical object being read continuously.  Its
noise has

  * a fixed spatial texture (the same grain pattern on the same tape area),
  * a slowly varying temporal component (head/tape contact, transport drift),
  * intermittent localised defects (dropouts, creases, stains).

This module provides the building blocks: seeded streams, mean-reverting
(Ornstein-Uhlenbeck-like) processes for transport drift, and value noise with
separable spatial and temporal correlation lengths.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# deterministic seed derivation
# ---------------------------------------------------------------------------
def derive_seed(base: int, *tags: object) -> int:
    """
    Derive a stable 32-bit sub-seed from a project seed plus tags.

    Using a hash (instead of `base + counter`) means adding a new noise source
    later cannot shift the random draws of every existing source, which would
    break render reproducibility of previously saved projects.
    """
    h = hashlib.blake2s(digest_size=8)
    h.update(str(int(base)).encode())
    for t in tags:
        h.update(b"\x00")
        h.update(str(t).encode())
    return int.from_bytes(h.digest()[:4], "little")


def rng_for(base_seed: int, *tags: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(base_seed, *tags))


# ---------------------------------------------------------------------------
# mean-reverting process (transport / time-base drift)
# ---------------------------------------------------------------------------
@dataclass
class OUProcess:
    """
    Discrete Ornstein-Uhlenbeck process: x[n] = x[n-1] + theta*(mu-x) + sigma*w

    Used for quantities that are physically anchored (transport speed, head
    drum phase, tape tension) and cannot wander off to infinity, unlike a pure
    random walk.  `corr_len` is in samples (lines or fields) and is converted
    into a per-step reversion coefficient.

    A VCR's transport drift is the *same continuous process* no matter how much
    of the clip is rendered, so the process is stateful: `advance` continues
    where the previous call stopped.  That is what lets the tape plan grow
    lazily instead of being allocated for the whole clip up front -- sizing the
    plan from a "frames are unbounded" render context is exactly the bug that
    made an early build allocate 19 GB.
    """
    rng: np.random.Generator
    sigma: float
    corr_len: float = 8.0
    theta_scale: float = 1.0
    _state: float = 0.0

    @property
    def theta(self) -> float:
        corr = max(1e-6, float(self.corr_len))
        return float(np.clip(self.theta_scale / corr, 1e-6, 1.0))

    def reset(self) -> None:
        self._state = 0.0

    def advance(self, n: int) -> np.ndarray:
        """
        Return the next `n` samples, continuing from the internal state.

        Implemented as an exponential moving average over white noise with
        `lfilter`, i.e. O(n) inside compiled code.  Output is scaled to the
        requested `sigma`, which is the process's stationary standard
        deviation.
        """
        from scipy.signal import lfilter

        n = int(max(0, n))
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        theta = self.theta
        w = self.rng.standard_normal(n)
        out = lfilter([theta], [1.0, -(1.0 - theta)], w).astype(np.float64)
        self._state = float(out[-1])
        if self.sigma != 1.0:
            out *= float(self.sigma)
        return out

    def sample(self, n: int, initial: float | None = None) -> np.ndarray:
        """One-shot sample of `n` fresh samples."""
        if initial is not None:
            self._state = float(initial)
        return self.advance(n)

    def sample_fast(self, n: int) -> np.ndarray:
        """Alias used at call sites that want the cheap vectorised path."""
        return self.advance(n)

    def prefix(self, n: int) -> np.ndarray:
        """
        The first `n` samples from a fresh start (does not disturb state).
        Cached incrementally so sequential 36,000-frame exports are O(1) per frame
        instead of O(N^2) while preserving exact determinism.
        """
        n = int(max(0, n))
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        buf = getattr(self, "_prefix_buf", None)
        if buf is None:
            buf = np.zeros(0, dtype=np.float64)
            self._prefix_buf = buf
            self._prefix_state = 0.0
        if len(buf) < n:
            from scipy.signal import lfilter
            need = max(n - len(buf), 256)
            theta = self.theta
            w = self.rng.standard_normal(need)
            zi = np.array([(1.0 - theta) * self._prefix_state], dtype=np.float64)
            chunk, zf = lfilter([theta], [1.0, -(1.0 - theta)], w, zi=zi)
            self._prefix_state = float(zf[0]) / max(1e-12, 1.0 - theta)
            if self.sigma != 1.0:
                chunk = chunk * float(self.sigma)
            buf = np.concatenate([buf, chunk.astype(np.float64)])
            self._prefix_buf = buf
        return buf[:n]


# ---------------------------------------------------------------------------
# coherent value noise (spatial texture + slow temporal change)
# ---------------------------------------------------------------------------
_LATTICE_CACHE: dict[tuple[int, int, int], np.ndarray] = {}
_STATIC_FIELD_CACHE: dict[tuple[int, int, int, int], np.ndarray] = {}


class ValueNoise3D:
    """
    Seeded, tileable-ish value noise over (frame, y, x) with independent
    spatial and temporal correlation lengths.

    Implementation: random lattice at a coarse grid, then separable trilinear
    interpolation (first blending the two temporal lattice planes on the coarse
    (gy, gx) grid, then interpolating Y and X). Cost is O(N) per frame with a
    single 2D pass instead of 8 full-frame fancy-indexing gathers, while
    producing bit-identical trilinear interpolation.
    """

    def __init__(self, seed: int, height: int, width: int,
                 spatial_scale: float = 24.0, time_scale: float = 12.0,
                 temporal_drift: float = 1.0):
        self.seed = int(seed)
        self.height = int(height)
        self.width = int(width)
        self.spatial_scale = max(2.0, float(spatial_scale))
        self.time_scale = max(1.0, float(time_scale))
        self.temporal_drift = float(temporal_drift)
        gy = int(np.ceil(self.height / self.spatial_scale)) + 4
        gx = int(np.ceil(self.width / self.spatial_scale)) + 4
        gt = 8  # temporal lattice period; wraps slowly
        lat_key = (self.seed, gy, gx)
        lat = _LATTICE_CACHE.get(lat_key)
        if lat is None:
            rng = np.random.default_rng(self.seed)
            lat = rng.standard_normal((gt, gy, gx)).astype(np.float32)
            if len(_LATTICE_CACHE) > 64:
                _LATTICE_CACHE.pop(next(iter(_LATTICE_CACHE)), None)
            _LATTICE_CACHE[lat_key] = lat
        self._lattice = lat
        self._gt, self._gy, self._gx = gt, gy, gx
        # pre-computed fractional coordinates and integer indices in x/y
        self._fy = (np.arange(self.height, dtype=np.float32) / self.spatial_scale)
        self._fx = (np.arange(self.width, dtype=np.float32) / self.spatial_scale)
        self._y0 = np.floor(self._fy).astype(np.int32)
        self._y1 = self._y0 + 1
        self._x0 = np.floor(self._fx).astype(np.int32)
        self._x1 = self._x0 + 1
        self._wy = (self._fy - self._y0)[:, None].astype(np.float32)
        self._wx = (self._fx - self._x0)[None, :].astype(np.float32)
        self._wy_1d = np.ascontiguousarray(self._wy.reshape(-1), dtype=np.float32)
        self._wx_1d = np.ascontiguousarray(self._wx.reshape(-1), dtype=np.float32)

    def sample_origin(self, frame_index: int) -> float:
        """Return the scalar value at (0, 0) without allocating a full (H, W) array."""
        t = frame_index / self.time_scale * self.temporal_drift
        t0 = int(np.floor(t)) % self._gt
        t1 = (t0 + 1) % self._gt
        ft = float(t - np.floor(t))
        v0 = float(self._lattice[t0, 0, 0])
        v1 = float(self._lattice[t1, 0, 0])
        return v0 + (v1 - v0) * ft

    def sample(self, frame_index: int) -> np.ndarray:
        """Return a (H, W) float32 field with zero mean, unit-ish variance."""
        t = frame_index / self.time_scale * self.temporal_drift
        t0 = int(np.floor(t)) % self._gt
        t1 = (t0 + 1) % self._gt
        ft = float(t - np.floor(t))

        from .signal import get_native_lib
        lib = get_native_lib()
        if lib is not None:
            lat0 = np.ascontiguousarray(self._lattice[t0], dtype=np.float32)
            lat1 = np.ascontiguousarray(self._lattice[t1], dtype=np.float32)
            out = np.empty((self.height, self.width), dtype=np.float32)
            lib.vml_value_noise_sample_f32(
                lat0.ctypes.data, lat1.ctypes.data, ft,
                self._y0.ctypes.data, self._wy_1d.ctypes.data,
                self._x0.ctypes.data, self._wx_1d.ctypes.data,
                out.ctypes.data, self.height, self.width, self._gx,
            )
            return out

        ft_f32 = np.float32(ft)
        lat0 = self._lattice[t0]
        lat1 = self._lattice[t1]
        lat = lat0 + (lat1 - lat0) * ft_f32

        y0, y1 = self._y0, self._y1
        x0, x1 = self._x0, self._x1
        col_y = lat[y0] + (lat[y1] - lat[y0]) * self._wy
        c0 = col_y[:, x0]
        c1 = col_y[:, x1]
        c1 -= c0
        c1 *= self._wx
        c1 += c0
        return c1.astype(np.float32, copy=False)

    def static_field(self, smooth: int = 1) -> np.ndarray:
        """A time-independent spatial texture (for 'fixed grain')."""
        sm = max(0, int(smooth))
        key = (self.seed, self.height, self.width, sm)
        hit = _STATIC_FIELD_CACHE.get(key)
        if hit is not None:
            return hit
        rng = np.random.default_rng(self.seed ^ 0x5EED)
        f = rng.standard_normal((self.height, self.width)).astype(np.float32)
        for _ in range(sm):
            f = 0.5 * (f + np.roll(f, 1, axis=0))
            f = 0.5 * (f + np.roll(f, 1, axis=1))
        s = f.std()
        out = (f / s if s > 1e-9 else f).astype(np.float32, copy=False)
        if len(_STATIC_FIELD_CACHE) > 48:
            _STATIC_FIELD_CACHE.pop(next(iter(_STATIC_FIELD_CACHE)), None)
        _STATIC_FIELD_CACHE[key] = out
        return out


def bandlimited_noise(rng: np.random.Generator, shape: Sequence[int],
                      corr_y: float = 1.0, corr_x: float = 1.0) -> np.ndarray:
    """
    Gaussian white noise low-passed with a separable exponential kernel.

    corr_* = 1.0 gives white noise; larger values give a coarser grain, which
    is how spatial noise *scale* is controlled separately from noise *level*.
    Implemented with a single-pole IIR (lfilter) so cost stays O(H*W).
    """
    from scipy.signal import lfilter

    shape = tuple(int(s) for s in shape)
    n = rng.standard_normal(shape).astype(np.float32)
    if corr_x > 1.001:
        a = 1.0 - 1.0 / float(corr_x)
        b_x = np.array([1.0 - a], dtype=np.float32)
        a_x = np.array([1.0, -a], dtype=np.float32)
        n = lfilter(b_x, a_x, n, axis=1).astype(np.float32, copy=False)
    if corr_y > 1.001:
        a = 1.0 - 1.0 / float(corr_y)
        b_y = np.array([1.0 - a], dtype=np.float32)
        a_y = np.array([1.0, -a], dtype=np.float32)
        n = lfilter(b_y, a_y, n, axis=0).astype(np.float32, copy=False)
    s = float(n.std())
    if s > 1e-9:
        n /= np.float32(s)
    return n
