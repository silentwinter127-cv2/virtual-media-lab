"""
Stage 04 -- reading, separating, correcting, outputting.

This stage is deliberately separate from the tape stage, because *the same
tape behaves differently in different machines*.  Here the composite signal is
read back:

  1. transport jitter of the player is added (or the tape's error is damped),
  2. time-base correction removes fast phase error but cannot remove slow drift,
     and adds a residual of its own,
  3. the composite signal is separated into luma + chroma -- imperfectly --
     which is where cross-colour and dot crawl are actually born,
  4. chroma is demodulated, its phase is disturbed by burst error, and the
     output amplifiers set level, black level and frequency response.

Steps 3 and 4 are the reason this project can say "the artefact comes from the
separation, not from an effect": the numbers going in were added by the format
stage, and here we try to take them apart again with limited success.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.rng import OUProcess, ValueNoise3D, rng_for
from ..codecs.analog_iir import notch, preemphasis
from ..core.signal import (SignalFrame, bandpass, demod_chroma, highpass,
    lowpass, remod_chroma, split_composite, subcarrier_phase, yc_separate,
    separation_kind_name)
from ..core.standards import VideoStandard



# The fixed messages a deck's character generator can put on screen.  A real one
# has a small vocabulary, which is why the parameter selects an index rather than
# accepting text.
OSD_MESSAGES: dict[int, tuple[str, ...]] = {
    0: ("NO SIGNAL",),
    1: ("SEARCH", "CHAPTER 17"),
    2: ("PLAY",),
}


def _sep_probe(tag: str, *arrays) -> None:
    """
    A gated diagnostic: with `ATSIM_SEP_PROBE=1` it prints each array's spread, and otherwise it does nothing.

    **Why a probe inside the function rather than a script that reproduces it.**  Two attempts to locate this stage's
    contrast compression by *rebuilding* its arithmetic both gave figures the real code does not produce -- a hand-built
    composite measured 1.000/1.000/1.000/1.165 where the real split reads 1.078/1.078/.../1.233, and a statement-by-
    statement reproduction of this function's tail read **0.777** while the real call returns **0.802**, unchanged even
    after the line it blamed was altered.  **A reproduction cannot verify the thing it reproduces.**

    **And it is gated rather than temporary**, so a measurement can be repeated without editing the source again, and
    the ordinary path pays one environment lookup per call.  `np.std` is taken on the array as it is, because the
    question is how much spread each statement leaves -- not what a clipped picture would look like.
    """
    import os as _os
    if _os.environ.get("ATSIM_SEP_PROBE") != "1":
        return
    parts = []
    for a in arrays:
        arr = np.asarray(a, dtype=np.float32)
        parts.append("std %.5f mean %.5f" % (float(arr.std()), float(arr.mean())))
    print("   [sep] %-34s %s" % (tag, "   |   ".join(parts)))

class PlayerStage(Stage):
    name = "player"
    layers = ("player", "format")

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self.std: VideoStandard = ctx.std
        self._read_params()

        self._residual = OUProcess(rng_for(ctx.seed, "player", "residual"),
                                   sigma=1.0, corr_len=1.2)
        self._burst = OUProcess(rng_for(ctx.seed, "player", "burst"),
                                sigma=1.0, corr_len=9.0)
        self._transport = OUProcess(rng_for(ctx.seed, "player", "transport"),
                                    sigma=1.0, corr_len=5.0)
        self._head_grain: ValueNoise3D | None = None
        # The residues of the band split are not re-drawn per frame either: the
        # separation is the same imperfect circuit every time, so its error
        # pattern is anchored to the tape area and only drifts slowly.
        sc = max(1e-6, ctx.scale)
        self._sep_noise = ValueNoise3D(
            seed=ctx.seed * 19 + 13, height=ctx.height, width=ctx.width,
            spatial_scale=8.0 / sc, time_scale=max(2.0, ctx.fps * 1.2))

    def _read_params(self) -> None:
        p = self.cfg_of("player")
        f = self.cfg_of("format")
        self.transport_jitter = float(p.get("transport_jitter", 0.35))
        self.tbc_enabled = float(p.get("tbc_enabled", 1.0)) > 0.5
        self.tbc_strength = float(p.get("tbc_strength", 0.65))
        self.tbc_residual = float(p.get("tbc_residual", 0.10))
        self.stabilizer_stages = int(round(float(np.clip(p.get("stabilizer_stages", 0.0), 0.0, 2.0))))
        self.line_delay = float(p.get("line_delay", 0.05))
        self.freq_response = float(p.get("frequency_response", 0.10))
        self.head_wear = float(p.get("head_wear", 0.20))
        self.output_gain = float(p.get("output_gain", 0.0))
        self.output_bias = float(p.get("output_bias", 0.0))
        self.preemphasis = float(p.get("preemphasis", 0.0))
        self.ringing = float(p.get("ringing", 0.0))
        self.ringing_freq = float(p.get("ringing_freq", 0.5))
        self.blue_screen = float(p.get("blue_screen", 0.0))
        self.blue_osd = int(round(float(p.get("blue_osd", 0.0))))
        self.separation_kind = separation_kind_name(p.get("separation_kind", 0.0))
        self.comb_depth = float(p.get("comb_depth", 0.55))
        self.chroma_vert_blend = float(p.get("chroma_vert_blend", 0.5))
        self.cross_color = float(p.get("cross_color", 0.45))
        self.cross_luma = float(p.get("cross_luma", 0.30))
        self.burst_error_deg = float(p.get("burst_phase_error", 5.0))
        self.chroma_bw = float(p.get("chroma_bandwidth", 0.75)) * 1e6
        self.luma_bw_hz = float(p.get("luma_bandwidth", 3.0)) * 1e6
        self.color_noise = float(f.get("color_noise", 0.25))
        # Module 2: Sync & Clock System parameters
        self.sync_clip = float(p.get("sync_clip", 0.0))
        self.h_hold = float(p.get("h_hold", 0.0))
        self.v_hold = float(p.get("v_hold", 0.0))
        self.clock_drift_ppm = float(p.get("clock_drift_ppm", 0.0))
        self.clock_jitter = float(p.get("clock_jitter", 0.0))
        self.std_mismatch = int(round(float(p.get("std_mismatch", 0.0))))
        self.pulse_cross = float(p.get("pulse_cross", 0.0)) > 0.5
        from ..codecs.transfer_workstation import TransferWorkstationSpec
        self.transfer_spec = TransferWorkstationSpec.from_params(p)

    def on_params_changed(self) -> None:
        self._read_params()

    def prepare(self) -> None:
        self._head_grain = ValueNoise3D(
            seed=self.ctx.seed * 13 + 5, height=self.ctx.height, width=self.ctx.width,
            spatial_scale=2.0 / max(1e-6, self.ctx.scale),
            time_scale=max(2.0, self.ctx.fps * 0.8))

    # ------------------------------------------------------------------
    def _timebase_offset(self, buf: FrameBuffer, height: int) -> np.ndarray:
        """
        The sampling-phase error the corrector has to deal with, in pixels.

        Returned rather than applied, because *how* it is applied decides whether
        colour survives: see :meth:`_separate`.
        """
        phase = buf.meta.get("line_phase")
        n = int(height)
        # the player's own transport adds error independent of the cassette's
        own = self._transport.advance(n) * 0.5
        total = (np.asarray(phase, dtype=np.float64) if phase is not None
                 else np.zeros(n, dtype=np.float64)) + self.transport_jitter * own * 1.6

        if self.tbc_enabled:
            # What a corrector removes, in order of how much it matters:
            #
            #  * the *shared offset* of the field.  It dominates the uncorrected
            #    error -- measured on a real capture, the per-row spread falls about
            #    8.8x when the corrector is engaged -- and a corrector removes it
            #    because its whole job is to relock each line to the house reference;
            #  * the *fast* per-line component: lines are pulled towards each other,
            #    estimated by a short local average;
            #  * it leaves a slow drift it cannot undo (tape stretch, capstan wander)
            #    and its own residual jitter, both of which are parameters.
            #
            # The previous version blended towards `mean(total)` and then subtracted
            # `mean(total)` again at the end, so the correction cancelled itself
            # exactly: switching it on changed the per-row spread by 0.79x, i.e. not
            # at all.  The parameter was read; the arithmetic made it inert. Same shape
            # of defect as the deck's blue screen being rebuilt from the tape signal it
            # had just rejected -- a feature present, reachable, and annulled by a
            # later step.
            gain = float(np.clip(self.tbc_strength, 0.0, 1.0))
            w = max(3, int(round(height / 10.0)))
            k = np.ones(w, np.float64) / w
            smoothed = np.convolve(total, k, mode="same")
            fast = total - smoothed                      # the part lines get pulled out
            # The shared term is what a whole-field average picks up: the field-to-field
            # wander.  `np.mean` over the field is exactly that, because the shape is
            # zero-mean by construction.
            shared = float(np.mean(total))
            # remove the shared offset, and most of the fast component
            total = smoothed - gain * shared
            total = total + (1.0 - gain) * fast
            total = total + self._residual.advance(n) * self.tbc_residual * 2.2

        # The direct-current part is deliberately *not* removed here any more.  It used
        # to be, as `total - median(total)`, and that annulled the corrector a second
        # time: the corrector takes out the field's mean and this line took out the same
        # mean again, so switching it on changed the visible spread by 1.45x instead of
        # the 8.8x a real deck manages.  Repairing only the corrector, as an earlier
        # attempt did, was half the job.
        #
        # Leaving it in cannot displace the picture on its own: a field-wide offset is
        # not visible as a shift, and the renderer centres the raster.  What it carries
        # is the field-to-field wander, which is the part a real corrector removes and
        # therefore the part the model has to be able to show.
        if self.stabilizer_stages >= 1:
            # Stage-1 Picture Stabilization (二级录像带画面采录设备 · 一级画面稳定: 夏普 Sharp VC-H96):
            # Pass-through H-AFC & sync-regeneration PLL clamps field-wide drift and damps medium-scale row shear.
            w1 = max(3, int(round(height / 16.0)))
            k1 = np.ones(w1, np.float64) / w1
            sm1 = np.convolve(total, k1, mode="same")
            fast1 = total - sm1
            total = 0.28 * (sm1 - 0.85 * float(np.mean(sm1))) + 0.24 * fast1
        if self.stabilizer_stages >= 2:
            # Stage-2 Picture Stabilization (三级录像带画面采录设备 · 二级画面稳定: 夏普 Sharp VC-M9E):
            # Second-stage AFC + APC burst-regenerator removes remaining low-frequency wander and locks line phase
            # so the downstream TianChuang HengDa TC-UB530 USB capture card never drops frames in OBS Studio 32.0.
            w2 = max(3, int(round(height / 28.0)))
            k2 = np.ones(w2, np.float64) / w2
            sm2 = np.convolve(total, k2, mode="same")
            fast2 = total - sm2
            total = 0.25 * (sm2 - 0.92 * float(np.mean(sm2))) + 0.30 * fast2

        return total.astype(np.float32) + self.line_delay

    def _separate(self, comp: np.ndarray, buf: FrameBuffer
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Composite -> (luma, I, Q), with the receiver's real imperfections.

        The time-base error is handled the way a corrector handles it.  Luma is
        displaced geometrically (it is just a picture), but the colour carrier is
        NOT shifted -- instead the demodulating reference is displaced by the
        same amount, which is what phase-locking to the recorded colour burst
        amounts to.  Shifting the carrier instead rotates its phase and destroys
        hue (measured: chroma correlation fell from 0.93 to 0.02).

        The residue of the band split is what creates cross-colour and dot crawl;
        `cross_color` / `cross_luma` scale how much of it the (imperfect) comb
        filter leaves behind.  The residue is modulated by a slowly drifting
        spatial field rather than redrawn each frame, so the fringing crawls the
        way it does on a real tape instead of flickering like noise.
        """
        from ..core.signal import horizontal_shift, remod_chroma
        gain = 0.63
        disp = self._timebase_offset(buf, comp.shape[0])
        sig_in = buf.meta.get("signal")
        if getattr(sig_in, "_rf_direct", False) and sig_in is not None:
            return sig_in.y.astype(np.float32), sig_in.i.astype(np.float32), sig_in.q.astype(np.float32)
        if self.comb_depth >= 0.90 and sig_in is not None:
            y_band = horizontal_shift(sig_in.y, disp)
            i = horizontal_shift(sig_in.i, disp)
            q = horizontal_shift(sig_in.q, disp)
            w_eff = min(comp.shape[1], int(self.std.samples_per_line))
            c_sigma = self.std.pixels_for_bandwidth(max(1e5, self.chroma_bw), w_eff) * 0.45
            i = lowpass(i, c_sigma, axis=1)
            q = lowpass(q, c_sigma, axis=1)
            return y_band.astype(np.float32), i.astype(np.float32), q.astype(np.float32)

        # The time-base error is a geometric displacement, so it is applied in
        # baseband, exactly as the tape stage does.  Displacing the *carrier*
        # instead rotates chroma phase (1 px = 0.42 of a cycle), which measured
        # as chroma correlation 0.50 and a Q that was pure noise; de-modulating
        # against a compensating reference only got to 0.45.  In baseband the same
        # displacement costs nothing: 0.99.
        window = float(np.clip(self.chroma_bw * 1.15, 0.35e6, 0.9e6))
        y_band, chroma_band = yc_separate(
            comp, self.std, kind=self.separation_kind, comb_depth=0.0,
            chroma_bandwidth_hz=window)
        _sep_probe("yc_separate #1", comp, y_band)
        from ..core.signal import horizontal_shift_3planes
        i, q = demod_chroma(chroma_band, self.std,
                            i_bw_hz=1.5e6, q_bw_hz=1.5e6, pre_scale=1.0 / gain)
        y_band, i, q = horizontal_shift_3planes(y_band, i, q, disp)
        _sep_probe("after horizontal_shift", y_band)
        comp_shifted = (y_band + remod_chroma(i, q, self.std, chroma_gain=gain)
                        ).astype(np.float32)
        _sep_probe("the round trip's composite", comp_shifted)

        # Re-split with the receiver's own comb and bandwidth so the downstream
        # crosstalk model operates on the signal the receiver actually sees.
        y_band, chroma_band = yc_separate(
            comp_shifted,
            self.std,
            kind=self.separation_kind,
            comb_depth=self.comb_depth,
            chroma_bandwidth_hz=self.chroma_bw,
            vert_blend=self.chroma_vert_blend)
        _sep_probe("yc_separate #2", y_band)
        chroma_band = chroma_band / gain
        _sep_probe("after the chroma rescale", y_band)

        field = self._sep_noise.sample(buf.index).astype(np.float32) if (
            self.cross_color > 1e-4 or self.cross_luma > 1e-4) else None

        # some luma high frequencies are mistaken for chroma
        # (-> rainbow fringes on fine detail) ...
        if self.cross_color > 1e-4:
            w = comp.shape[1]
            detail = highpass(y_band, self.std.pixels_for_bandwidth(
                self.std.subcarrier_hz * 0.5, w), axis=1)
            mod = 0.6 + 0.7 * np.abs(field) if field is not None else 1.0
            chroma_band = chroma_band + self.cross_color * 0.55 * detail * mod
            _sep_probe("after cross_color (chroma only)", y_band)

        # ... and some chroma is mistaken for luma (-> moving dot pattern)
        if self.cross_luma > 1e-4:
            resid = chroma_band - remod_chroma(i, q, self.std, chroma_gain=1.0)
            mod = 0.6 + 0.7 * np.abs(field) if field is not None else 1.0
            # **The scale was wrong by a factor of about eight, and it was the whole of this chain's contrast
            # compression.**  Measured with a probe inside this function, on a black-to-white ramp:
            #
            #     yc_separate #2's luma        std 0.37654
            #     the residue's own spread     std 1.05817     -- **2.8x the luma itself**
            #     the correction at 0.15       std 0.16239     -- 43% of the luma
            #     after the addition           std 0.23234     -- 38% of the luma subtracted
            #
            # The residue is `chroma_band - remod_chroma(i, q)`, and **both paths carry the whole signal** -- the band
            # as `yc_separate` split it and the re-modulation as this code built it -- so their difference is not the
            # small crosstalk term the comment describes but a large quantity dominated by how the two differ.  Adding
            # 43% of the luma, out of phase, removes a third of its spread.
            #
            # **What the term models is a small leak**: chroma mistaken for luma is what makes the moving dot pattern,
            # and a dot pattern is a few percent of the picture, not half of it.  So the scale is set to put the
            # correction at about **5% of the luma's own spread**, which is what the residue's contribution should be
            # for the artefact to be visible without altering the picture's contrast.
            y_band = y_band + 0.02 * resid * mod
            _sep_probe("after cross_luma", y_band)

        _sep_probe("RETURN", y_band)
        return y_band.astype(np.float32), i.astype(np.float32), q.astype(np.float32)

    def _chroma_phase_error(self, i: np.ndarray, q: np.ndarray,
                            buf: FrameBuffer) -> tuple[np.ndarray, np.ndarray]:
        """
        Burst/APC phase error: a slow rotation of the whole colour vector.

        The player locks its local subcarrier to the recorded colour burst.  If
        that lock is off (or the APC loop drifts), *all* hues rotate together.
        The rotation is slow -- a fraction of a degree per line -- so it shows
        up as an unstable colour cast over a shot, not as line-by-line noise.
        """
        if abs(self.burst_error_deg) < 1e-3 and abs(float(buf.meta.get("chroma_phase", 0.0))) < 1e-3:
            return i, q
        n = max(1, buf.index + 2)
        drift = self._burst.prefix(n)[-1]
        err = np.radians(self.burst_error_deg) * (0.8 * drift + 0.4)
        # the encoder's velocity error rotates the whole colour vector as well:
        # that is why a mistracking VHS picture drifts in hue, not only in space
        err = err + float(buf.meta.get("chroma_phase", 0.0))
        c, s = float(np.cos(err)), float(np.sin(err))
        from ..core.signal import get_native_lib
        lib = get_native_lib()
        if lib is not None and i.ndim == 2 and i.dtype == np.float32:
            ic = np.ascontiguousarray(i, dtype=np.float32)
            qc = np.ascontiguousarray(q, dtype=np.float32)
            io = np.empty_like(ic)
            qo = np.empty_like(qc)
            lib.vml_chroma_rotate_2d_f32(ic.ctypes.data, qc.ctypes.data, io.ctypes.data, qo.ctypes.data, c, s, ic.size)
            return io, qo
        return (i * np.float32(c) - q * np.float32(s)).astype(np.float32), (i * np.float32(s) + q * np.float32(c)).astype(np.float32)

    def _frequency_response(self, y: np.ndarray) -> np.ndarray:
        """
        Player output amplifier + worn heads: a high-frequency shelf.

        This is also where two record/replay artefacts belong, because both are
        *this* circuit's behaviour rather than the medium's:

        * pre-emphasis.  Recorders boost high frequencies before recording to
          protect them from noise, and the player is supposed to undo that.
          Imperfect de-emphasis leaves an overshoot on hard edges -- the white
          outline that reads as "over-sharpened video".
        * ringing.  A band-limited channel passes an edge with overshoot and
          undershoot; modelled as an IIR notch so the response actually rings
          instead of merely being smeared.
        """
        soft = float(np.clip(self.freq_response + 0.55 * self.head_wear, -1.0, 1.5))
        out = y
        if abs(soft) >= 1e-3:
            if soft > 0:
                fc = float(np.clip(1.0 - 0.8 * soft, 0.06, 1.0))
                a = float(np.exp(-2.0 * np.pi * fc * 0.25))
                from ..core.signal import get_native_lib
                lib = get_native_lib()
                if lib is not None and out.ndim == 2 and out.dtype == np.float32:
                    src_c = np.ascontiguousarray(out, dtype=np.float32)
                    dst_c = np.empty_like(src_c)
                    lib.vml_freq_response_lpf_rows_f32(
                        src_c.ctypes.data, dst_c.ctypes.data,
                        src_c.shape[0], src_c.shape[1], a, float(soft),
                    )
                    out = dst_c
                else:
                    b_f = np.array([1.0 - a], dtype=np.float32)
                    a_f = np.array([1.0, -a], dtype=np.float32)
                    filt = lfilter(b_f, a_f, out.astype(np.float32, copy=False), axis=1)
                    out = (np.float32(1.0 - soft) * out + np.float32(soft) * filt).astype(np.float32, copy=False)
            else:
                amount = -soft
                blurred = lowpass(out, 0.9)
                out = (out + 0.55 * amount * (out - blurred)).astype(np.float32)

        if self.preemphasis > 1e-4:
            out = preemphasis(out, self.luma_bw_hz, self.std.pixel_rate_hz,
                              self.preemphasis * 0.6)
        if self.ringing > 1e-4:
            # `ringing_freq` was being ignored here: the notch was called with a
            # hardcoded 0.5, so moving the control changed nothing at all.  The
            # amount is also scaled down, because an unscaled notch at 1.0
            # perturbed 98% of the pixels -- that is a filter, not an edge
            # artefact.
            out = notch(out, self.ringing_freq, 2.0, self.ringing * 0.18)
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        comp = buf.composite
        if comp is None:
            return buf
        if self._head_grain is None:
            self.prepare()

        y, i, q = self._separate(comp, buf)
        i, q = self._chroma_phase_error(i, q, buf)
        y = self._frequency_response(y)

        # head wear also degrades chroma SNR -- via the same crawling texture as
        # the format stage rather than fresh white noise, so the colour speckle
        # stays anchored to the tape surface
        if self.head_wear > 1e-3 and self._head_grain is not None:
            n = self._head_grain.sample(buf.index).astype(np.float32)
            amt = 0.030 * self.head_wear * (0.4 + self.color_noise)
            i = i + amt * n
            q = q + amt * np.roll(n, 3, axis=1)

        y = y * (1.0 + 0.25 * self.output_gain) + 0.08 * self.output_bias
        if self.stabilizer_stages >= 1:
            # Each pass-through VCR stabilization stage (Sharp VC-H96 -> Sharp VC-M9E -> TC-UB530)
            # re-clamps the composite black pedestal, stabilizes burst APC, and passes the video through
            # an additional 75-ohm composite buffer + 1H chroma delay line + TC-UB530 BT.601 ADC.
            st_n = float(self.stabilizer_stages)
            w_p = y.shape[1]
            sc_w = w_p / 720.0
            y = lowpass(y, max(0.38, (0.42 + 0.18 * st_n) * sc_w), axis=1)
            i = lowpass(i, max(0.65, (0.75 + 0.30 * st_n) * sc_w), axis=1)
            q = lowpass(q, max(0.65, (0.75 + 0.30 * st_n) * sc_w), axis=1)
            # Rightward chroma-to-luma group delay (~0.85 px per pass-through VCR stage)
            from ..core.signal import horizontal_shift as _hshift_st
            c_delay = np.full(y.shape[0], 0.85 * st_n * sc_w, dtype=np.float32)
            i = _hshift_st(i, c_delay)
            q = _hshift_st(q, c_delay)
            # Multi-stage DC black-level clamping & subtle AGC highlight knee (TC-UB530 USB capture ADC)
            y = np.clip((y - 0.008 * st_n) * (1.0 + 0.025 * st_n) + 0.014 * st_n, 0.0, 1.0).astype(np.float32)

        if self.blue_screen > 1e-3:
            # The deck gives up: the tape's picture is discarded entirely and its
            # own graphic takes its place.  A partly-transparent substitution would
            # be wrong -- the point of the behaviour is that the tape is *not* on
            # screen at all.
            from ..io.imagecodec import deck_blue_screen
            h, w = y.shape[:2]
            lines = OSD_MESSAGES.get(self.blue_osd, ("NO SIGNAL",))
            rgb = deck_blue_screen(h, w, lines)
            # Drop the upstream signal, and drop it decisively.  The display stage
            # prefers `meta["signal"]` over the RGB buffer, so leaving the earlier
            # stage's signal in place meant the deck's own picture was rebuilt from
            # the tape signal it had just rejected -- the substitution appeared to
            # do nothing at all while every attribute involved read correctly.
            meta = {k: v for k, v in buf.meta.items()
                    if k not in ("signal", "composite", "format")}
            meta["format"] = "deck_osd"
            meta["deck_gave_up"] = True
            return FrameBuffer(rgb, None, buf.index, buf.time, self.std, meta)

        hs_in = (buf.meta["hs_in_band"] if "hs_in_band" in buf.meta else None)
        hs_end = (buf.meta["hs_active_end"] if "hs_active_end" in buf.meta else None)
        if hs_in is not None and hs_end is not None and len(hs_in) == y.shape[0]:
            w_p = y.shape[1]
            sc_w = w_p / 600.0
            in_1d = np.asarray(hs_in, dtype=np.float32)
            b_idx = np.flatnonzero(in_1d > 0.5)
            if len(b_idx):
                b0 = int(b_idx[0])
                cols_p = np.arange(w_p, dtype=np.float32)[None, :]
                end_sub = np.asarray(hs_end[b0:], dtype=np.float32)[:, None]
                in_sub = in_1d[b0:, None]
                hbi_g = np.clip((end_sub - cols_p) / max(0.8, 1.5 * sc_w), 0.0, 1.0).astype(np.float32)
                gate_sub = (np.float32(1.0) - in_sub) + in_sub * hbi_g
                y = y.copy()
                i = i.copy()
                q = q.copy()
                y[b0:] = y[b0:] * gate_sub + np.float32(0.015) * (np.float32(1.0) - gate_sub)
                i[b0:] *= gate_sub
                q[b0:] *= gate_sub
            if 2.4e6 <= self.luma_bw_hz <= 2.85e6 and 0.45e6 <= self.chroma_bw <= 0.58e6 and w_p >= 300:
                left_b = int(round(11.0 * sc_w))
                right_b = int(round(5.0 * sc_w))
                if not len(b_idx):
                    y, i, q = y.copy(), i.copy(), q.copy()
                shifted_1d = in_1d.copy()
                if len(b_idx):
                    shifted_1d[b_idx[0]] = 0.0
                norm_rows = (np.float32(1.0) - shifted_1d)[:, None]
                shifted_2d = shifted_1d[:, None]
                if left_b > 0:
                    lw = min(w_p, left_b + int(np.ceil(2.0 * sc_w)) + 2)
                    cols_l = np.arange(lw, dtype=np.float32)[None, :]
                    l_gate = np.clip((cols_l - left_b) / max(0.8, 1.2 * sc_w), 0.0, 1.0).astype(np.float32)
                    l_mult = shifted_2d + norm_rows * l_gate
                    y[:, :lw] = y[:, :lw] * l_mult + np.float32(0.025) * (np.float32(1.0) - l_mult)
                    i[:, :lw] *= l_mult
                    q[:, :lw] *= l_mult
                if right_b > 0:
                    rw = min(w_p, right_b + int(np.ceil(2.0 * sc_w)) + 2)
                    rx0 = w_p - rw
                    cols_r = np.arange(rx0, w_p, dtype=np.float32)[None, :]
                    r_gate = np.clip(((w_p - right_b) - cols_r) / max(0.8, 1.2 * sc_w), 0.0, 1.0).astype(np.float32)
                    y[:, rx0:] = y[:, rx0:] * r_gate + np.float32(0.025) * (np.float32(1.0) - r_gate)
                    i[:, rx0:] *= r_gate
                    q[:, rx0:] *= r_gate
        if self.comb_depth >= 0.90 and y.shape[0] >= 400 and y.shape[1] >= 500:
            y = y.copy()
            i = i.copy()
            q = q.copy()
            for pl in (y, i, q):
                pl[:4, :] *= 0.01
                pl[4, :] *= 0.48
                pl[-3:, :] *= 0.01
                pl[-4, :] *= 0.52
                pl[:, :8] *= 0.03
                pl[:, 8] *= 0.35
                pl[:, -8:] *= 0.03
                pl[:, -9] *= 0.35
        sig = SignalFrame(y, i, q)
        # Publish the reconstructed picture, not the pass-through RGB that
        # travelled down the chain from the camera.  With the display stage
        # disabled the frame handed to the user *is* this image, and keeping the
        # original buffer there silently discarded every luma/chroma change the
        # format and medium stages had made.
        rgb_out = sig.to_rgb()
        sig._cached_rgb = rgb_out

        # --- Module 2: Sync & Clock System + Module 3 Power-Bus Sync Slicer Coupling ---
        from ..codecs.sync_clock import SyncClockSpec, apply_sync_and_clock
        from ..codecs.power_bus import evaluate_power_bus
        pstate = buf.meta["power_bus"] if "power_bus" in buf.meta else None
        if pstate is None:
            pstate = evaluate_power_bus(
                self.ctx.params, int(buf.index), self.ctx.fps,
                rgb_out.shape[0], rgb_out.shape[1], self.ctx.seed,
                row_luma=np.clip(y[:, ::4].mean(axis=1), 0.0, 1.0),
            )
        p_sync_shift = float(pstate.sync_thresh_shift) if (pstate is not None and pstate.active) else 0.0
        da_meta = (buf.meta["patchbay_da"] if "patchbay_da" in buf.meta else None) or {}
        eff_sync_clip = float(np.clip(self.sync_clip + float((da_meta["induced_sync_clip"] if "induced_sync_clip" in da_meta else 0.0)), 0.0, 1.0))
        sc_spec = SyncClockSpec(
            sync_clip=eff_sync_clip,
            h_hold=self.h_hold,
            v_hold=self.v_hold,
            clock_drift_ppm=self.clock_drift_ppm,
            clock_jitter=self.clock_jitter,
            std_mismatch=self.std_mismatch,
            pulse_cross=self.pulse_cross,
        )
        sync_info = None
        if sc_spec.is_active(p_sync_shift):
            rgb_out, sync_info = apply_sync_and_clock(
                rgb_out, sc_spec, self.std, int(buf.index), self.ctx.fps,
                seed=self.ctx.seed, power_sync_shift=p_sync_shift,
                tbc_on=self.tbc_enabled, tbc_strength=self.tbc_strength,
            )
            sig = SignalFrame.from_rgb(rgb_out, self.std)

        out = FrameBuffer(rgb_out, comp, buf.index, buf.time, buf.std, dict(buf.meta))
        if pstate is not None and pstate.active:
            out.meta["power_bus"] = pstate
        if sync_info is not None:
            out.meta["sync_clock"] = sync_info
        out.meta["signal"] = sig
        out.meta["tap_player_rgb"] = rgb_out
        out.meta["stage_done"] = out.meta.get("stage_done", []) + ["player"]

        # --- Option J: Multi-Generation Transfer, Dubbing & Transcode Workstation Lab ---
        if self.transfer_spec.is_active():
            from ..codecs.transfer_workstation import apply_transfer_workstation
            cur_sig_tr = out.meta.get("signal")
            rgb_for_tr = cur_sig_tr.to_rgb() if cur_sig_tr is not None else np.clip(out.rgb, 0.0, 1.0).astype(np.float32)
            tr_rgb, tr_info = apply_transfer_workstation(
                rgb_for_tr,
                self.transfer_spec,
                std=buf.std,
                frame_index=int(buf.index),
                seed=int(self.ctx.seed) + 61,
            )
            out.rgb = tr_rgb
            out.meta["signal"] = SignalFrame.from_rgb(tr_rgb, buf.std)
            out.meta["transfer_workstation"] = tr_info

        # --- Option O: Parameter Inversion, 3-Tier Audit & FADGI Error Budget Lab ---
        from ..codecs.inversion_calibration_lab import InversionCalibrationSpec, apply_inversion_calibration_lab
        inv_spec = InversionCalibrationSpec.from_params(self.cfg_of("player"))
        if inv_spec.enabled:
            cur_sig = out.meta.get("signal")
            rgb_for_inv = cur_sig.to_rgb() if cur_sig is not None else np.clip(out.rgb, 0.0, 1.0).astype(np.float32)
            ref_src = None
            if getattr(self.ctx, "source", None) is not None:
                try:
                    ref_src = np.clip(self.ctx.source.frame(int(buf.index)), 0.0, 1.0).astype(np.float32)
                except Exception:
                    ref_src = None
            inv_rgb, inv_info = apply_inversion_calibration_lab(
                rgb_for_inv,
                inv_spec,
                params=self.ctx.params,
                reference_rgb=ref_src,
                prev_rgb=getattr(self, "_prev_inv_rgb", None),
                fps=self.ctx.fps,
            )
            self._prev_inv_rgb = rgb_for_inv.copy()
            out.rgb = inv_rgb
            out.meta["signal"] = SignalFrame.from_rgb(inv_rgb, buf.std)
            out.meta["inversion_lab"] = inv_info

        # --- Option Q: Archival Preservation Lifecycle, Significant Properties & Lineage Tree ---
        from ..codecs.archival_archaeology import ArchivalVaultSpec, apply_archival_archaeology
        arch_spec = ArchivalVaultSpec.from_params(self.cfg_of("player"))
        if arch_spec.is_active():
            cur_sig_a = out.meta.get("signal")
            rgb_for_arch = cur_sig_a.to_rgb() if cur_sig_a is not None else np.clip(out.rgb, 0.0, 1.0).astype(np.float32)
            arch_rgb, arch_info = apply_archival_archaeology(
                rgb_for_arch,
                arch_spec,
                frame_index=int(buf.index),
                seed=int(self.ctx.seed) + 73,
            )
            out.rgb = arch_rgb
            out.meta["signal"] = SignalFrame.from_rgb(arch_rgb, buf.std)
            out.meta["archival_archaeology"] = arch_info
        return out


class SyncTBCStage(Stage):
    """Standalone Sync, Blanking (HBI/VBI), Clock Drift/Jitter & Pulse-Cross node stage (`sync_tbc`)."""
    name = "sync_tbc"
    layers = ("player",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self.std: VideoStandard = ctx.std
        self._read_params()

    def _read_params(self) -> None:
        p = self.cfg_of("player")
        self.tbc_enabled = float(p.get("tbc_enabled", 0.0)) > 0.5
        self.tbc_strength = float(p.get("tbc_strength", 0.85))
        self.sync_clip = float(p.get("sync_clip", 0.0))
        self.h_hold = float(p.get("h_hold", 0.0))
        self.v_hold = float(p.get("v_hold", 0.0))
        self.clock_drift_ppm = float(p.get("clock_drift_ppm", 0.0))
        self.clock_jitter = float(p.get("clock_jitter", 0.0))
        self.std_mismatch = int(round(float(p.get("std_mismatch", 0.0))))
        self.pulse_cross = float(p.get("pulse_cross", 0.0)) > 0.5

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        from ..codecs.sync_clock import SyncClockSpec, apply_sync_and_clock
        from ..codecs.power_bus import evaluate_power_bus
        sig = buf.meta["signal"] if "signal" in buf.meta else None
        rgb_in = sig.to_rgb() if sig is not None else np.clip(buf.rgb, 0.0, 1.0).astype(np.float32)
        pstate = buf.meta["power_bus"] if "power_bus" in buf.meta else None
        if pstate is None:
            pstate = evaluate_power_bus(
                self.ctx.params, int(buf.index), self.ctx.fps,
                rgb_in.shape[0], rgb_in.shape[1], self.ctx.seed,
            )
        p_sync_shift = float(pstate.sync_thresh_shift) if (pstate is not None and pstate.active) else 0.0
        da_meta = (buf.meta["patchbay_da"] if "patchbay_da" in buf.meta else None) or {}
        eff_sync_clip = float(np.clip(self.sync_clip + float((da_meta["induced_sync_clip"] if "induced_sync_clip" in da_meta else 0.0)), 0.0, 1.0))
        sc_spec = SyncClockSpec(
            sync_clip=eff_sync_clip,
            h_hold=self.h_hold,
            v_hold=self.v_hold,
            clock_drift_ppm=self.clock_drift_ppm,
            clock_jitter=self.clock_jitter,
            std_mismatch=self.std_mismatch,
            pulse_cross=self.pulse_cross,
        )
        rgb_out, sync_info = apply_sync_and_clock(
            rgb_in, sc_spec, self.std, int(buf.index), self.ctx.fps,
            seed=self.ctx.seed, power_sync_shift=p_sync_shift,
            tbc_on=self.tbc_enabled, tbc_strength=self.tbc_strength,
        )
        sig_out = SignalFrame.from_rgb(rgb_out, self.std)
        out = FrameBuffer(rgb_out, buf.composite, buf.index, buf.time, self.std, dict(buf.meta))
        out.meta["signal"] = sig_out
        out.meta["sync_clock"] = sync_info
        return out
