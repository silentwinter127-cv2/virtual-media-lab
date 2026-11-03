"""
Stage 02 -- signal encoding / the recording format.

This is where a picture *becomes a signal*, and it is the stage that most
cleanly separates tape formats that look similar in still frames:

  composite analog (VHS / Video8 / Hi8)
      1. luma is bandwidth limited (VHS ~3 MHz, S-VHS ~5 MHz)
      2. chroma is heterodyned *below* luma and limited to <1 MHz
      3. the two are ADDED into one composite signal
      4. vertical chroma resolution collapses (chroma is ~40 lines tall)

  digital component (DV / MiniDV)
      keeps Y and chroma separate but subsamples chroma and quantises it
      (see atsim.codecs.dv)

The composite *sum* in step 3 is deliberately kept as a real signal plane, so
the artefacts created by trying to separate it again appear later in the player
instead of being painted on here.
"""
from __future__ import annotations

import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..codecs.analog_iir import FilterChoice, lowpass_hz
from ..core.rng import OUProcess, ValueNoise3D, rng_for
from ..core.signal import (SignalFrame, composite_encode, highpass, lowpass,
                           subcarrier_phase)
from ..core.standards import VideoStandard
from ..codecs.test_signal_gen import TestSignalSpec, generate_test_pattern


class FormatStage(Stage):
    name = "format"
    layers = ("format",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self.std: VideoStandard = ctx.std
        self._chroma_hist: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.lag_cold_frames = 0
        self.lag_applied_frames = 0
        self._read_params()
        # line-to-line chroma phase error (fast) -- "velocity error"
        self._line_phase = OUProcess(
            rng_for(ctx.seed, "format", "line_phase"), sigma=0.35, corr_len=1.6)
        # field-to-field phase error (slow) -- burst / APC drift
        self._field_phase = OUProcess(
            rng_for(ctx.seed, "format", "field_phase"), sigma=0.6, corr_len=12.0)
        sc = max(1e-6, ctx.scale)
        self._noise_field = ValueNoise3D(
            seed=ctx.seed * 3 + 1, height=ctx.height, width=ctx.width,
            spatial_scale=2.6 / sc, time_scale=max(1.5, ctx.fps / 3.0))
        self._noise_env = ValueNoise3D(
            seed=ctx.seed * 3 + 2, height=ctx.height, width=ctx.width,
            spatial_scale=40.0 / sc, time_scale=max(2.0, ctx.fps / 6.0))
        self._noise_shape = (ctx.height, ctx.width)

    def _read_params(self) -> None:
        cfg = self.cfg_of("format")
        self.luma_bw = float(cfg.get("luma_bandwidth", 3.0)) * 1e6
        self.chroma_bw = float(cfg.get("chroma_bandwidth", 0.75)) * 1e6
        self._digital_component = False
        try:
            all_params = self.ctx.params or {}
            codec_cfg = all_params["codec"] if "codec" in all_params else {}
            self._digital_component = float(codec_cfg.get("dv_enabled", 0.0)) > 0.5
        except Exception:                                     # noqa: BLE001
            self._digital_component = False
        self.chroma_vavg = float(cfg.get("chroma_vertical_averaging", 0.85))
        self.chroma_lag = float(np.clip(cfg.get("chroma_lag", 0.0), 0.0, 0.95))
        self.chroma_lag_frames = int(round(np.clip(
            cfg.get("chroma_lag_frames", 4.0), 1.0, 8.0)))
        self.chroma_gain = float(cfg.get("chroma_gain", 1.0))
        self.edge_boost = float(cfg.get("edge_boost", 0.25))
        self.color_noise = float(cfg.get("color_noise", 0.25))
        self.test_spec = TestSignalSpec.from_params(cfg)

    def on_params_changed(self) -> None:
        self._read_params()

    # ------------------------------------------------------------------
    def prepare(self) -> None:
        self._chroma_hist = {}
        self.lag_cold_frames = 0
        self.lag_applied_frames = 0

    def timeline(self) -> dict:
        if self.chroma_lag <= 1e-4:
            return {}
        return {"chroma_lag": {
            "weight": self.chroma_lag,
            "memory_frames": self.chroma_lag_frames,
            "lagged_frames": self.lag_applied_frames,
            "cold_frames": self.lag_cold_frames,
        }}

    # ------------------------------------------------------------------
    def _resize_noise(self, shape: tuple[int, int]) -> None:
        h, w = int(shape[0]), int(shape[1])
        if getattr(self, "_noise_shape", None) == (h, w):
            return
        sc = max(1e-6, self.ctx.scale)
        seed = self.ctx.seed
        self._noise_field = ValueNoise3D(
            seed=seed * 3 + 1, height=h, width=w,
            spatial_scale=2.6 / sc, time_scale=max(1.5, self.ctx.fps / 3.0))
        self._noise_env = ValueNoise3D(
            seed=seed * 3 + 2, height=h, width=w,
            spatial_scale=40.0 / sc, time_scale=max(2.0, self.ctx.fps / 6.0))
        self._noise_shape = (h, w)

    def _noise_for(self, shape: tuple[int, int], index: int) -> np.ndarray:
        self._resize_noise(shape)
        return self._noise_field.sample(index).astype(np.float32)

    def _noise_env_for(self, shape: tuple[int, int], index: int) -> np.ndarray:
        self._resize_noise(shape)
        return self._noise_env.sample(index).astype(np.float32)

    # ------------------------------------------------------------------
    def _choice(self) -> FilterChoice:
        cfg = self.cfg_of("format")
        kind = int(round(float(cfg.get("filter_kind", 1.0))))
        sections = int(round(float(cfg.get("filter_sections", 3.0))))
        return FilterChoice(
            {0: "gaussian", 1: "constant_k", 2: "butterworth"}.get(kind, "constant_k"),
            sections=max(1, sections))

    def _limit_luma(self, y: np.ndarray) -> np.ndarray:
        w = y.shape[1]
        rate = self.std.pixel_rate_hz * (w / float(self.std.samples_per_line)) if w > 760 else self.std.pixel_rate_hz
        return lowpass_hz(y, self.luma_bw, rate, self._choice())

    def _boost_edges(self, y: np.ndarray) -> np.ndarray:
        if self.edge_boost < 1e-4:
            return y
        w = y.shape[1]
        scale_w = (w / float(self.std.samples_per_line)) if w > 760 else 1.0
        px_per_cycle = (2.0 * np.pi / self.std.subcarrier_rad_per_sample) * scale_w
        hp = highpass(y, max(1.0, px_per_cycle / 3.2), axis=1)
        return y + (0.45 * self.edge_boost) * hp

    def _limit_chroma(self, i: np.ndarray, q: np.ndarray,
                      index: int | None = None):
        choice = self._choice()
        h, w = i.shape[:2]
        rate = self.std.pixel_rate_hz * (w / float(self.std.samples_per_line)) if w > 760 else self.std.pixel_rate_hz
        if not self._digital_component:
            i = lowpass_hz(i, self.chroma_bw, rate, choice)
            q = lowpass_hz(q, self.chroma_bw, rate, choice)
            if self.chroma_vavg > 1e-4:
                k = float(np.clip(self.chroma_vavg, 0.0, 1.0))
                v_sig = 1.35 * (h / 540.0) if h > 580 else 1.35
                from ..core.signal import get_native_lib, gaussian_half_kernel
                lib = get_native_lib()
                kh, r_k = gaussian_half_kernel(v_sig)
                if lib is not None and r_k > 0 and r_k <= 32 and i.ndim == 2 and i.dtype == np.float32:
                    ic = np.ascontiguousarray(i, dtype=np.float32)
                    qc = np.ascontiguousarray(q, dtype=np.float32)
                    io = np.empty_like(ic)
                    qo = np.empty_like(qc)
                    lib.vml_sym_fir_vert_f32(ic.ctypes.data, io.ctypes.data, kh.ctypes.data, r_k, h, w, 1, float(1.0 - k), float(k))
                    lib.vml_sym_fir_vert_f32(qc.ctypes.data, qo.ctypes.data, kh.ctypes.data, r_k, h, w, 1, float(1.0 - k), float(k))
                    i, q = io, qo
                else:
                    i = (1.0 - k) * i + k * lowpass(i, v_sig, axis=0)
                    q = (1.0 - k) * q + k * lowpass(q, v_sig, axis=0)
        i, q = self._lag_chroma(i, q, index)
        if abs(self.chroma_gain - 1.0) > 1e-4:
            i = i * np.float32(self.chroma_gain)
            q = q * np.float32(self.chroma_gain)
        return i, q

    def _lag_chroma(self, i: np.ndarray, q: np.ndarray,
                    index: int | None):
        if self.chroma_lag <= 1e-4 or index is None:
            return i, q
        prev = self._chroma_hist.get(int(index) - 1)
        if prev is None:
            self.lag_cold_frames += 1
            return i, q
        pi, pq = prev
        if pi.shape != i.shape or pq.shape != q.shape:
            self.lag_cold_frames += 1
            return i, q
        k = self.chroma_lag
        i = (1.0 - k) * i + k * pi
        q = (1.0 - k) * q + k * pq
        self.lag_applied_frames += 1
        return i, q

    def _remember_chroma(self, index: int | None, i: np.ndarray, q: np.ndarray) -> None:
        if index is None:
            return
        n = int(index)
        self._chroma_hist[n] = (np.array(i, dtype=np.float32, copy=True),
                                np.array(q, dtype=np.float32, copy=True))
        cutoff = n - self.chroma_lag_frames + 1
        for old in [k for k in self._chroma_hist if k < cutoff]:
            del self._chroma_hist[old]

    # ------------------------------------------------------------------
    def on_params_changed(self) -> None:
        cfg = self.cfg_of("format")
        self.luma_bw = float(cfg.get("luma_bandwidth", 3.0)) * 1e6
        self.chroma_bw = float(cfg.get("chroma_bandwidth", 0.75)) * 1e6
        self.chroma_vavg = float(cfg.get("chroma_vertical_averaging", 0.85))
        self.chroma_lag = float(np.clip(cfg.get("chroma_lag", 0.0), 0.0, 0.95))
        self.chroma_gain = float(cfg.get("chroma_gain", 1.0))
        self.edge_boost = float(cfg.get("edge_boost", 0.25))
        self.color_noise = float(cfg.get("color_noise", 0.25))
        self.test_spec = TestSignalSpec.from_params(cfg)

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        rgb = buf.rgb
        if self.test_spec.is_active():
            rgb = generate_test_pattern(
                width=rgb.shape[1], height=rgb.shape[0],
                mode=self.test_spec.pattern_mode,
                frame_index=int(buf.index), fps=self.ctx.fps,
                std=self.std, amplitude_ire=self.test_spec.amplitude_ire,
            )
        sig = SignalFrame.from_rgb(rgb, self.std)
        y, i, q = sig.y, sig.i, sig.q

        y = self._limit_luma(y)
        y = self._boost_edges(y)
        i, q = self._limit_chroma(i, q, buf.index)

        out_phase = 0.0
        if self.color_noise > 1e-4:
            field = int(buf.meta.get("field", 0))
            ph = self._line_phase.advance(i.shape[0]) * 0.9
            fp = self._field_phase.prefix(buf.index + field + 1)[-1] * 0.5
            ph = ph + fp
            amp = float(self.color_noise * 0.018)
            n = self._noise_for(i.shape, buf.index)
            n_env = self._noise_env_for(i.shape, buf.index)
            from ..core.signal import get_native_lib
            lib = get_native_lib()
            if lib is not None and i.ndim == 2 and i.dtype == np.float32:
                ic = np.ascontiguousarray(i, dtype=np.float32)
                qc = np.ascontiguousarray(q, dtype=np.float32)
                cos_1d = np.ascontiguousarray(np.cos(ph), dtype=np.float32)
                sin_1d = np.ascontiguousarray(np.sin(ph), dtype=np.float32)
                lib.vml_format_color_noise_f32(
                    ic.ctypes.data, qc.ctypes.data,
                    n.ctypes.data, n_env.ctypes.data,
                    cos_1d.ctypes.data, sin_1d.ctypes.data,
                    amp, ic.shape[0], ic.shape[1],
                )
                i, q = ic, qc
            else:
                cos_p = np.cos(ph)[:, None].astype(np.float32)
                sin_p = np.sin(ph)[:, None].astype(np.float32)
                env = 0.55 + 0.75 * np.abs(n_env)
                i = i + amp * (n * env) * cos_p
                q = q + amp * (n * env) * sin_p
            out_phase = float(ph.mean())

        comp = composite_encode(y, i, q, self.std, chroma_gain=0.63)
        self._remember_chroma(buf.index, i, q)

        frame = SignalFrame(y.astype(np.float32), i.astype(np.float32), q.astype(np.float32))
        out = FrameBuffer(
            rgb=rgb, composite=comp, index=buf.index, time=buf.time, std=self.std,
            meta=dict(buf.meta),
        )
        out.meta["signal"] = frame
        out.meta["format"] = "composite"
        out.meta["chroma_phase"] = out_phase
        out.meta["tap_format_signal"] = frame
        if self.test_spec.is_active():
            out.meta["tap_camera_rgb"] = rgb
        return out
