"""
Stage 03 -- the recording medium itself.

Everything that makes a tape *that tape* lives here, and it all happens in the
composite/RF domain, because that is the only domain in which these phenomena
exist: lost FM carrier, sampling phase error, head-switching gap, magnetically
weak stripes.

Three rules from the reference plan are implemented literally:

  1. "Transport instability is not global picture shake."
     Horizontal displacement is a *per-scan-line sampling phase error*, built
     as the difference of a continuous position process, so neighbouring lines
     and neighbouring frames are correlated.  It looks like an unstable line
     clock, not like a shake filter applied to the whole frame.

  2. "Media wear is not 'more grain everywhere'."
     Dropouts and creases are discrete events with a start time, a duration, a
     position and a size.  They are generated once per clip from the project
     seed (so a re-render is byte-identical) and are published to the UI as an
     inspectable defect timeline.

  3. "Do not re-randomise every frame."
     Noise is a fixed particle texture plus a slowly drifting temporal
     component, creases move continuously, and a dropout persists for its whole
     duration.

The per-clip plan grows *lazily in blocks*.  A render context cannot know how
many frames will be requested (a preview scrubs, a final render is bounded), so
allocating the plan up front is both wasteful and, in an early build, fatal:
sizing it from an unbounded frame count allocated 19 GB and hung the process.
`_ensure(frame)` extends the plan only as far as the caller actually reaches,
and because the underlying processes are stateful, the extended plan is
identical to the one a full linear render would have produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dfield

import numpy as np

from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.rng import ValueNoise3D, OUProcess, derive_seed, rng_for
from ..codecs.dv import (apply_stale_lines, apply_stale_macroblocks, line_run_displacement,
                         line_run_mask, macroblock_grid, shuttle_stale_mask)
from ..codecs.tape_speed import by_index
from ..core.signal import (SignalFrame, demod_chroma, fft_bandpass, horizontal_shift,
                           lowpass, remod_chroma, split_composite, vertical_shift)

# how many frames of plan to add per extension step
PLAN_BLOCK = 256

# SMPTE 170M: the colour subcarrier is recorded at 0.63 of full video amplitude.
# Everything in the baseband domain has to undo/reapply this consistently, or
# colour noise, wear and dropout colour all end up at the wrong scale.
CHROMA_CARRIER_GAIN = 0.63


# ---------------------------------------------------------------------------
# defect records (also the data model behind the UI defect timeline)
# ---------------------------------------------------------------------------
@dataclass
class DropoutEvent:
    start_frame: int
    frames: int
    position_s: float
    duration_s: float
    row: int
    rows: int
    col: int
    cols: int
    field: int
    severity: float

    def to_dict(self) -> dict:
        return {
            "kind": "dropout",
            "start_frame": self.start_frame,
            "frames": self.frames,
            "start_s": round(self.position_s, 4),
            "duration_s": round(self.duration_s, 4),
            "row": self.row, "rows": self.rows,
            "col": self.col, "cols": self.cols,
            "field": self.field,
            "severity": round(float(self.severity), 3),
        }


@dataclass
class CreaseEvent:
    period_frames: float
    offset: float
    angle: float
    width: float
    depth: float
    phase: float

    def to_dict(self) -> dict:
        return {
            "kind": "crease",
            "period_s": round(float(self.period_frames) / 29.97, 3),
            "angle_deg": round(float(np.degrees(self.angle)), 2),
            "width": round(float(self.width), 2),
            "depth": round(float(self.depth), 3),
        }


@dataclass
class TapePlan:
    """Deterministic, incrementally grown description of the medium."""
    line_stride: int = 1
    line_start: int = 0                  # global line index of line_phase[0]
    line_phase: np.ndarray = dfield(default_factory=lambda: np.zeros(0, np.float32))
    frame_dy: np.ndarray = dfield(default_factory=lambda: np.zeros(0, np.float32))
    head_switch_amp: np.ndarray = dfield(default_factory=lambda: np.zeros(0, np.float32))
    dropouts: list[DropoutEvent] = dfield(default_factory=list)
    # stiction events: episode-driven, so they cannot be a per-frame field
    stiction: list = dfield(default_factory=list)
    creases: list[CreaseEvent] = dfield(default_factory=list)
    # --- contamination -------------------------------------------------
    # Head clog and surface damage are *fixed in space*, not random per frame.
    # A clogged video head drags a fixed horizontal band of lost signal down
    # every field; a scratch runs along one track for as long as the tape
    # passes it.  The pattern is therefore generated once (from the seed) and
    # row-indexed at render time.
    clog_rows: np.ndarray = dfield(default_factory=lambda: np.zeros(0, np.float32))
    scratch_rows: np.ndarray = dfield(default_factory=lambda: np.zeros(0, np.float32))
    debris_rng_seed: int = 0

    @property
    def prepared_lines(self) -> int:
        return self.line_start + len(self.line_phase)

    @property
    def prepared_frames(self) -> int:
        return len(self.frame_dy)

    def timeline(self, max_events: int = 6000) -> dict:
        events = [d.to_dict() for d in self.dropouts]
        events.sort(key=lambda e: e["start_s"])
        hist: dict[int, int] = {}
        for e in events:
            sec = int(e["start_s"])
            hist[sec] = hist.get(sec, 0) + 1
        return {
            "kind": "tape_plan",
            "prepared_frames": self.prepared_frames,
            "dropout_count": len(events),
            "dropout_events": events[:max_events],
            "dropout_density": [{"second": k, "count": v} for k, v in sorted(hist.items())],
            "creases": [c.to_dict() for c in self.creases],
        }


# ---------------------------------------------------------------------------
class TapeStage(Stage):
    name = "tape"
    layers = ("tape",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._drum_phase = 0.0
        self._shuttle_carry: np.ndarray | None = None
        self._shuttle_last_fraction = 0.0
        self._binder_last = 0.0
        self._grain: ValueNoise3D | None = None
        self._drift: ValueNoise3D | None = None
        self._binder: ValueNoise3D | None = None
        self._static_grain: np.ndarray | None = None
        self.unit_px = 0.045 * (self.ctx.width / 720.0)
        self._read_params()
        self._reset()

    def _read_params(self) -> None:
        cfg = self.cfg_of("tape")
        # Whether this is a digital format decides the *granularity* of the
        # shuttle model (macroblocks versus scanlines), so it must be known before
        # anything reads it.  Reading it further down raised AttributeError, which
        # is the third time in this file that a derived flag was used above its own
        # assignment.
        self.digital = float(self.ctx.layer("codec").get("dv_enabled", 0.0)) > 0.5
        # --- recording speed (SP / LP / EP) --------------------------------
        # Resolved *first*, because several derived quantities below are scaled by
        # it.  Expressed as multipliers rather than by overwriting parameters, so
        # the speed encodes the physics and the user's own settings still apply on
        # top of it.  See atsim/codecs/tape_speed.py for why the mechanisms are
        # kept separate (track width, replay output, guard-band margin).
        self.speed = by_index(cfg.get("tape_speed", 0.0))
        self.speed_note = self.speed.note()
        # Drum speed error.  1.0 is nominal; a bearing fault makes the drum run
        # slow, which repeats lines vertically and shows as horizontal grain.
        # Pedestal / DC-restoration error.  Luma is a baseband level that has to be
        # clamped to a reference; chroma is a modulated amplitude that survives the
        # same error.  Negative values push the picture toward black while the
        # colour keeps its amplitude -- which is the only way to get "very dark and
        # very saturated" at once.
        # Coating-shed grain: fine, dense, luminance-only noise.  Separate from
        # `noise_floor` because that is a general particle floor and reaches only a
        # sixth of the energy the reference shows.
        self.shed_grain = float(cfg.get("shed_grain", 0.0))
        self.luma_pedestal = float(cfg.get("luma_pedestal", 0.0))
        self.drum_speed = float(cfg.get("drum_speed", 1.0))
        self._drum_phase = 0.0
        self.stability = float(cfg.get("tape_stability", 0.55))
        # LP/EP: the control-track signal the capstan servo follows is recorded at
        # the lower speed too, so there is less for the servo to lock onto
        self.stability *= self.speed.stability_scale
        self.line_jitter = float(cfg.get("line_jitter", 0.35))
        # Correlated per-line sampling error with its own temporal behaviour.  Kept
        # separate from `line_jitter`, which is the same axis at a different
        # granularity (uncorrelated line to line), so a preset can state which
        # physical cause it means.
        self.row_timebase = float(cfg.get("row_timebase", 0.0))
        # Tape-deck stiction: damp binder sticking to the capstan, then releasing.  Three
        # parameters because the fault is episodic: how often, how strongly, and how much
        # worse it gets along the tape.
        self.stiction_rate = float(cfg.get("stiction_rate", 0.0))
        self.stiction_amount = float(cfg.get("stiction_amount", 0.6))
        self.stiction_stretch = float(cfg.get("stiction_stretch", 0.0))
        # Curved stripes from beat interference with a residual carrier.  Three
        # parameters because the spacing and the bend are independent: how strong,
        # how tightly spaced, and how much they curve.
        self.stripe_moire = float(cfg.get("stripe_moire", 0.0))
        self.stripe_beat = float(cfg.get("stripe_beat", 0.055))
        self.stripe_curve = float(cfg.get("stripe_curve", 0.35))
        self.walk = float(cfg.get("walk_amplitude", 0.30))
        self.micro = float(cfg.get("micro_jitter", 0.20))
        self.vert = float(cfg.get("vertical_instability", 0.08))
        self.hs_amp = float(cfg.get("head_switching", 0.45))
        self.hs_lines = max(0, int(round(cfg.get("head_switch_lines", 12.0))))
        self.hs_mid_line = float(cfg.get("head_switch_mid_line", 0.0))
        self.tracking_error = float(cfg.get("tracking_error", 0.0))  # tape layer
        # --- shuttle (rewind / fast-forward) --------------------------------
        # Only meaningful on the digital branch: it is the DV decoder's error
        # concealment, expressed on the DV macroblock grid.
        self.shuttle_speed = float(cfg.get("shuttle_speed", 0.0))
        self.shuttle_fraction = float(cfg.get("shuttle_stale", 0.0))
        self.shuttle_cluster = float(cfg.get("shuttle_cluster", 0.55))
        self.shuttle_carry = float(cfg.get("shuttle_carry", 0.45))
        self.shuttle_reverse = bool(cfg.get("shuttle_reverse", 0.0) > 0.5)
        self.shuttle_block = int(round(cfg.get("shuttle_block", 16.0)))
        self.shuttle_guard_bars = float(cfg.get("shuttle_guard_bars", 0.0))
        self.shuttle_bar_luma = float(cfg.get("shuttle_bar_luma", 0.565))
        self.shuttle_pip_bars = bool(cfg.get("shuttle_PIP_bars", 0.0) > 0.5)
        # mean run length in scanlines, for the analogue (line-based) family
        self.shuttle_span = float(cfg.get("shuttle_span", 8.0))
        # peak horizontal displacement of the failed lines, for stepped edges
        self.shuttle_wave = float(cfg.get("shuttle_wave", 14.0))
        # Which granularity applies is a property of the *tape format*, not a
        # preference: analogue formats have scanlines, DV has macroblocks.
        # 0 = auto (follow the codec layer), 1 = force line-based, 2 = force block.
        mode = int(round(float(cfg.get("shuttle_mode", 0.0))))
        if mode == 1:
            self.shuttle_block_based = False
        elif mode == 2:
            self.shuttle_block_based = True
        else:
            self.shuttle_block_based = bool(self.digital)
        self._shuttle_carry: np.ndarray | None = None
        self._shuttle_last_fraction = 0.0
        self.tracking_rows = max(0, int(round(cfg.get("tracking_noise_rows", 0.0))))
        self.tracking_wave = float(cfg.get("tracking_noise_wave", 0.0))
        self.tracking_snow = float(cfg.get("tracking_noise_snow", 0.0))
        self.do_rate = float(cfg.get("dropout_rate", 0.35))
        self.do_len = float(cfg.get("dropout_len", 0.35))
        self.do_color = float(cfg.get("dropout_color", 0.45))
        self.do_soft = float(cfg.get("dropout_softness", 0.35))
        self.doc_hold = float(np.clip(cfg.get("doc_hold", 0.65), 0.0, 1.0))
        self.edit_glitch_rate = float(np.clip(cfg.get("edit_glitch_rate", 0.0), 0.0, 2.0))
        self.edit_glitch_severity = float(np.clip(cfg.get("edit_glitch_severity", 0.65), 0.0, 1.5))
        self.wear = float(cfg.get("wear", 0.25))
        self.crease_rate = float(cfg.get("crease_rate", 0.20))
        self.noise_floor = float(cfg.get("noise_floor", 0.18))
        # --- contamination / cleanliness -------------------------------
        self.head_clog = float(cfg.get("head_clog", 0.0))
        self.clog_bands = cfg.get("clog_bands", 3.0)
        self.debris = float(cfg.get("debris", 0.0))
        self.surface_damage = float(cfg.get("surface_damage", 0.0))
        self.chroma_loss = float(cfg.get("chroma_loss", 0.0))
        # Whole lines decoding to the wrong colour.  Deliberately separate from
        # `chroma_loss`, which removes colour: this rotates hue and shifts the
        # colour-under gain, so the line stays saturated but wrong.  The difference
        # is the mechanism, not a strength setting.
        self.row_chroma_error = float(cfg.get("row_chroma_error", 0.0))
        self.row_chroma_phase = float(cfg.get("row_chroma_phase", 0.0))
        self.row_chroma_gain = float(cfg.get("row_chroma_gain", 0.0))
        # binder breakdown: damp storage sheds the magnetic coating.  Distinct
        # from debris (a loose particle) and surface_damage (a scratch) because it
        # removes a *fraction of the picture* and kills chroma before luma.
        self.binder_loss = float(cfg.get("binder_loss", 0.0))
        self._binder_last = 0.0
        # --- multi-generation VHS iterative dubbing (generation loss) -------
        self.generation_passes = int(round(float(np.clip(cfg.get("generation_passes", 1.0), 1.0, 25.0))))
        self.gen_chroma_bleed = float(np.clip(cfg.get("gen_chroma_bleed", 0.85), 0.0, 2.0))
        self.gen_ringing = float(np.clip(cfg.get("gen_ringing", 0.85), 0.0, 2.0))
        self.gen_sync_collapse = float(np.clip(cfg.get("gen_sync_collapse", 0.0), 0.0, 1.5))
        # COSMAG E-180 PAL/SP/LP mixed archival tape failure mechanisms:
        # 1) `demagnetization` (消磁): loss of remanent magnetization M_r, causing FM luma carrier drop into
        #    limiter snow, high-frequency rolloff, and 629 kHz colour-under killer/desaturation;
        # 2) `base_cracking` (带基龟裂): PET substrate brittle micro-fissures across helical tracks, producing
        #    diagonal/slanted spacing-loss fracture bands with FM limiter white/black impulse spikes & phase steps;
        # 3) `base_breakage` (带基破损): physical edge notch, fold or splice tear causing a localized vertical band
        #    of control-track CTL loss, horizontal sync shear, and full-width RF dropout.
        self.demagnetization = float(cfg.get("demagnetization", 0.0))
        self.base_cracking = float(cfg.get("base_cracking", 0.0))
        self.base_breakage = float(cfg.get("base_breakage", 0.0))
        self.sp_lp_mix = float(cfg.get("sp_lp_mix", 0.0))
        # --- Module 4 (Candidate B): Mechanical Transport & Servo State Machine ---
        from ..codecs.transport_servo import TransportServoSpec
        self.transport_servo_spec = TransportServoSpec.from_params(self.ctx.params)
        from ..codecs.edit_controller_nle import EditControllerSpec
        self.edit_controller_spec = EditControllerSpec(
            edit_mode=int(round(float(cfg.get("edit_mode", 0.0)))),
            point_slip_frames=int(round(float(cfg.get("edit_point_slip_frames", 2.0)))),
            flying_erase=float(cfg.get("edit_flying_erase", 1.0)) > 0.5,
            preroll_sec=float(cfg.get("edit_preroll_sec", 5.0)),
            nle_complexity=float(cfg.get("edit_nle_complexity", 0.65)),
            console_osd=float(cfg.get("edit_console_osd", 0.0)) > 0.5,
        )

    def on_params_changed(self) -> None:
        old_clog = (getattr(self, "head_clog", 0.0), getattr(self, "clog_bands", 3.0),
                    getattr(self, "surface_damage", 0.0))
        old_crease = (getattr(self, "crease_rate", 0.2), getattr(self, "wear", 0.25))
        old_stic = (getattr(self, "stiction_rate", 0.0), getattr(self, "stiction_amount", 0.6),
                    getattr(self, "stiction_stretch", 0.0))
        self._read_params()
        if getattr(self, "plan", None) is not None:
            if (self.head_clog, self.clog_bands, self.surface_damage) != old_clog:
                self._build_contamination()
            if (self.crease_rate, self.wear) != old_crease:
                self._crease_rng = rng_for(self.ctx.seed, "tape", "creases")
                self.plan.creases = []
                self._build_creases()
            if (self.stiction_rate, self.stiction_amount, self.stiction_stretch) != old_stic:
                self._plan_stiction(int(self.ctx.frame_count) if self.ctx.frame_count else 300)

    # ------------------------------------------------------------------
    def _reset(self) -> None:
        seed = self.ctx.seed
        self.unit_px = 0.045 * (self.ctx.width / 720.0)
        # the phase error is the *integral* of a velocity-like process
        self._vel = OUProcess(rng_for(seed, "tape", "vel"), sigma=1.0, corr_len=6.0)
        self._walkp = OUProcess(rng_for(seed, "tape", "walk"), sigma=1.0, corr_len=140.0)
        self._microp = OUProcess(rng_for(seed, "tape", "micro"), sigma=1.0, corr_len=2.0)
        self._vertp = OUProcess(rng_for(seed, "tape", "vert"), sigma=1.0, corr_len=5.0)
        self._spike_rng = rng_for(seed, "tape", "vert_spike")
        self._hs_rng = rng_for(seed, "tape", "hs")
        self._crease_rng = rng_for(seed, "tape", "creases")
        self._track_wave = OUProcess(rng_for(seed, "tape", "track_wave"),
                                    sigma=1.0, corr_len=9.0)
        self._drop_rng = np.random.default_rng(derive_seed(seed, "tape", "arrivals"))
        self.plan = TapePlan()
        self._line_pos = 0.0
        self._drop_cursor = 0
        self._do_time = 0.0
        self._build_creases()
        self._build_contamination()

    # ------------------------------------------------------------------
    # lazy plan growth
    # ------------------------------------------------------------------
    def _ensure(self, frame_index: int) -> None:
        target_frames = int(frame_index) + PLAN_BLOCK
        if self.ctx.frame_count and self.ctx.frame_count > 0:
            target_frames = min(target_frames, int(self.ctx.frame_count) + PLAN_BLOCK)
        target_lines = target_frames * max(2, self.ctx.height)
        self._grow_lines(target_lines)
        self._grow_frames(target_frames)
        self._grow_dropouts(target_frames)
        # The stiction timeline is planned in one pass rather than grown in blocks.  Its
        # events come from a Poisson process over the whole clip, so growing it block by
        # block would make each block independent of the last and the arrival times would
        # depend on how the renderer happened to seek -- which would break the guarantee that
        # the same project produces the same frames.
        if not getattr(self.plan, "stiction", None):
            self._plan_stiction(int(self.ctx.frame_count) if self.ctx.frame_count
                                else target_frames)

    def _grow_lines(self, target_lines: int) -> None:
        plan = self.plan
        need = target_lines - plan.prepared_lines
        if need <= 0:
            return
        stride = plan.line_stride
        n = int(np.ceil(need / stride))
        jit = max(0.0, self.line_jitter) * (1.45 - 0.95 * float(np.clip(self.stability, 0, 1)))
        walk = max(0.0, self.walk) * (1.5 - 1.0 * float(np.clip(self.stability, 0, 1)))
        micro = max(0.0, self.micro) * (1.3 - 0.7 * float(np.clip(self.stability, 0, 1)))

        v = self._vel.advance(n)
        cum = np.cumsum(v)
        pos = np.concatenate([[self._line_pos], self._line_pos + cum])
        self._line_pos = float(pos[-1])
        d = np.diff(pos)                      # per-line sampling phase error

        w = self._walkp.advance(n)
        sd = float(np.std(w)) if n > 8 else 1.0
        w = w / max(1e-9, sd)
        mp = self._microp.advance(n)

        # Amplitudes are in *pixels of the current render*, not in "units".
        #
        # They used to be `jit * unit_px * 5.0`, with `unit_px = 0.045 * W / 720`,
        # which at a 360-wide preview works out to 0.0225 -- so `line_jitter = 1.0`,
        # the top of its range, displaced lines by 0.09 px.  Measured, the timeline's
        # per-line phase had a standard deviation of 0.045 px: three orders of
        # magnitude below one pixel, which is why tuning that control appeared to do
        # nothing at all.  A control whose full range is invisible is not a control.
        phase = (jit * 6.0) * d + (walk * 4.0) * w
        micro_seq = (micro * 1.5) * mp
        new = (phase + micro_seq).astype(np.float32)
        plan.line_phase = np.concatenate([plan.line_phase, new]) \
            if plan.line_phase.size else new

    def _plan_stiction_once(self, frames: int) -> None:
        if not getattr(self.plan, "stiction", None):
            self._plan_stiction(frames)

    def _grow_frames(self, target_frames: int) -> None:
        plan = self.plan
        need = target_frames - plan.prepared_frames
        if need <= 0:
            return
        vseq = self._vertp.advance(need)
        spikes = (self._spike_rng.random(need) < 0.004).astype(np.float64)
        dy = (self.vert * 1.0) * (vseq * 0.35 + spikes * 2.2)
        amp = self.hs_amp * (0.65 + 0.35 * self._hs_rng.random(need))
        plan.frame_dy = np.concatenate([plan.frame_dy, dy.astype(np.float32)]) \
            if plan.frame_dy.size else dy.astype(np.float32)
        plan.head_switch_amp = np.concatenate(
            [plan.head_switch_amp, amp.astype(np.float32)]) \
            if plan.head_switch_amp.size else amp.astype(np.float32)

    def _grow_dropouts(self, target_frames: int) -> None:
        plan = self.plan
        if self.do_rate <= 0.0:
            return
        fps = self.ctx.fps
        H, W = self.ctx.height, self.ctx.width
        rng = self._drop_rng
        while self._do_time * fps < target_frames:
            # inter-arrival time of a Poisson arrival process at rate `do_rate`.
            # The effective rate rises at lower tape speeds: a narrower track is
            # proportionally more affected by the same particle or scratch, so the
            # *probability* of a visible dropout scales with the speed ratio.
            eff_rate = max(1e-6, self.do_rate * self.speed.dropout_scale)
            gap = float(rng.exponential(1.0 / eff_rate))
            self._do_time += gap
            frame0 = int(self._do_time * fps)
            if frame0 > target_frames:
                break
            dur = float(np.clip(rng.exponential(0.10) * (15734.0 / self.ctx.std.line_hz),
                                0.02, 0.9))
            nframes = max(1, int(round(dur * fps)))
            rows = int(np.clip(rng.exponential(3.5) + 1, 1, max(1, H // 6)))
            cols = int(np.clip(self.do_len * W * (0.35 + 1.3 * rng.random()), 6.0, W))
            row = int(np.clip(rng.integers(0, max(1, H)), 0, max(0, H - rows)))
            col = int(np.clip(rng.random() * W, 0, max(0, W - cols)))
            plan.dropouts.append(DropoutEvent(
                start_frame=frame0, frames=nframes, position_s=frame0 / fps,
                duration_s=dur, row=row, rows=rows, col=col, cols=cols,
                field=int(rng.integers(0, 2)),
                severity=float(np.clip(0.45 + 0.75 * rng.random(), 0.2, 1.4)),
            ))
            if len(plan.dropouts) > 200000:
                break

    def _build_contamination(self) -> None:
        """
        Build the *spatially fixed* contamination pattern.

        Two mechanisms produce the artefacts users describe as "格" (blocks) and
        "条纹" (stripes), and neither is a random per-frame effect:

        head_clog
            Debris on a video head wipes out the signal along the head's own
            scan path.  Because the head sweeps the same band of every field, a
            clogged head produces a *fixed horizontal band* -- often several,
            one per clogged head -- that is present in every frame and can
            flicker in strength as the head-to-tape contact varies.  This is the
            "blocky banding" signature, and it is why a dirty head looks like
            regular stripes rather than noise.

        surface_damage
            A scratch or scrape on the tape surface runs along the recording
            direction, so it damages a narrow, continuously drifting set of
            lines for as long as it passes the head.

        Both are stored as per-row intensity profiles so rendering is cheap and
        the pattern is identical across the whole clip.
        """
        H = max(2, self.ctx.height)
        rng = rng_for(self.ctx.seed, "tape", "contamination")

        # --- clogged heads: a few fixed bands of lost signal ---------------
        clog = np.zeros(H, dtype=np.float32)
        if self.head_clog > 1e-4:
            n_bands = int(np.clip(round(self.clog_bands), 1, 8))
            for _ in range(n_bands):
                # a clogged head affects a band up to ~1/8 of the field
                width = float(rng.uniform(4.0, H / 8.0))
                centre = float(rng.uniform(0.0, H))
                rows = np.arange(H, dtype=np.float32)
                # wrap around the field end (the head path is cyclic)
                d = np.abs(((rows - centre + H / 2) % H) - H / 2)
                profile = np.clip(1.0 - d / max(1.0, width * 0.5), 0.0, 1.0) ** 0.85
                # each band has its own strength; a badly clogged head is near total
                depth = float(np.clip(rng.uniform(0.35, 1.0), 0.0, 1.0))
                clog = np.maximum(clog, (profile * depth).astype(np.float32))
            # A clogged head does not lose "some" signal in its band, it loses
            # essentially all of it.  A linear scale made 0.55 read as only ~35%
            # loss, which looked like a faint stripe rather than the grey blocky
            # band a dirty head actually produces, so the response is curved.
            clog = (clog ** 0.6) * (1.6 * float(np.clip(self.head_clog, 0.0, 1.5)))
            clog = np.clip(clog, 0.0, 1.6)

        # --- surface scratches: narrow, drifting ---------------------------
        scratch = np.zeros(H, dtype=np.float32)
        if self.surface_damage > 1e-4:
            n_scr = int(np.clip(round(1 + self.surface_damage * 3), 1, 6))
            for _ in range(n_scr):
                width = float(rng.uniform(1.0, 3.0))
                centre = float(rng.uniform(0.0, H))
                rows = np.arange(H, dtype=np.float32)
                d = np.abs(((rows - centre + H / 2) % H) - H / 2)
                profile = np.clip(1.0 - d / max(1.0, width), 0.0, 1.0)
                depth = float(np.clip(rng.uniform(0.10, 0.45), 0.0, 1.0))
                scratch = np.maximum(scratch, (profile * depth).astype(np.float32))
            scratch = (scratch ** 0.7) * (1.5 * float(np.clip(self.surface_damage, 0.0, 1.5)))
            scratch = np.clip(scratch, 0.0, 1.4)

        self.plan.clog_rows = clog
        self.plan.scratch_rows = scratch
        self.plan.debris_rng_seed = derive_seed(self.ctx.seed, "tape", "debris")

    def _apply_clog(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                    frame_index: int, rng: np.random.Generator
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Head clog: lost signal along the head's fixed scan path.

        The band is spatially fixed but its *strength* varies frame to frame --
        that is what makes a dirty head flicker and "crawl" rather than sit
        still.  Within the band the luma is replaced by a dark/noisy level (a
        clogged head reads nothing) and chroma is removed entirely, which is why
        the result reads as a grey blocky band rather than as colourful noise.
        """
        profile = self.plan.clog_rows
        if profile.size == 0 or not np.any(profile > 0.0):
            return y, i, q, None
        # per-frame strength variation: slow, so the band breathes rather than
        # flickering every frame
        wobble = 0.75 + 0.45 * float(np.sin(frame_index * 0.21 + profile.sum()))
        amp = np.clip(profile * wobble, 0.0, 1.4)[:, None]

        noise = self._luma_band_noise(rng, y.shape)
        # a clogged head loses almost everything: dark, low-contrast noise
        replacement = 0.06 + 0.10 * noise
        y_out = y * (1.0 - amp) + replacement * amp
        kill = np.clip(amp * 1.5, 0.0, 1.0)
        i_out = i * (1.0 - kill)
        q_out = q * (1.0 - kill)
        return y_out, i_out, q_out, amp
    #

    def _apply_debris(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                      frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loose debris (magnetic particles, dust) on the tape or heads.

        Unlike a dropout, debris produces many *short* strikes: a few pixels to
        a few tens of pixels wide, clustered on a handful of lines.  That is the
        speckled/streaky texture of a dirty tape rather than a solid white patch.
        """
        if self.debris <= 1e-4:
            return y, i, q
        H, W = y.shape
        rng = np.random.default_rng(
            (self.plan.debris_rng_seed + frame_index * 7919) & 0xFFFFFFFF)
        # expected number of strikes per frame
        n = int(rng.poisson(max(1.0, self.debris * 90.0)))
        if n <= 0:
            return y, i, q
        y_out, i_out, q_out = y, i, q
        rows = rng.integers(0, H, size=n)
        widths = np.clip(rng.exponential(9.0, size=n) + 1.0, 1.0, W / 3.0).astype(np.int32)
        cols = rng.integers(0, W, size=n)
        depths = np.clip(rng.uniform(0.25, 1.0, size=n) * self.debris, 0.0, 1.2)
        for r, c, w, d in zip(rows, cols, widths, depths):
            c1 = min(W, int(c) + int(w))
            if c1 <= int(c):
                continue
            y_out[int(r), int(c):c1] = y_out[int(r), int(c):c1] * (1.0 - d) + 0.92 * d
            k = min(1.0, d * 1.4)
            i_out[int(r), int(c):c1] = i_out[int(r), int(c):c1] * (1.0 - k)
            q_out[int(r), int(c):c1] = q_out[int(r), int(c):c1] * (1.0 - k)
        return y_out, i_out, q_out

    def _apply_surface_damage(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                              frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scratch/scrape: a narrow drifting set of damaged lines.

        The profile drifts slowly with tape travel, so the stripe creeps across
        the picture over seconds instead of jumping -- which is exactly how a
        scratch behaves as the tape winds.
        """
        profile = self.plan.scratch_rows
        if profile.size == 0 or not np.any(profile > 0.0):
            return y, i, q
        # slow drift: shift the profile by a sub-row amount over time
        drift = float(np.sin(frame_index * 0.045)) * 3.0
        rows = np.arange(y.shape[0], dtype=np.float32)
        shifted = np.interp((rows - drift) % y.shape[0],
                            np.arange(y.shape[0], dtype=np.float32), profile,
                            period=y.shape[0])
        amp = np.clip(shifted, 0.0, 1.0)[:, None]
        soft_y = lowpass(y, 1.4)
        y_out = y * (1.0 - amp) + soft_y * amp
        k = np.clip(amp * 0.8, 0.0, 1.0)
        i_out = i * (1.0 - k)
        q_out = q * (1.0 - k)
        return y_out, i_out, q_out

    def _apply_chroma_loss(self, i: np.ndarray, q: np.ndarray,
                           frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Random whole scanlines lose their colour.

        This is a VHS-specific failure: the colour-under carrier is far weaker
        than the FM luma, so a marginal head-to-tape contact loses colour on a
        line before it loses the picture.  The set of lost lines must be drawn
        from a *geometric* distribution (the number of good lines between two
        losses), not by testing each line independently -- independent per-line
        trials give a uniform sprinkle, whereas real loss comes in runs with
        occasional long clean stretches.
        """
        if self.chroma_loss <= 1e-4:
            return i, q
        H = i.shape[0]
        rng = np.random.default_rng(
            derive_seed(self.ctx.seed, "tape", "chroma_loss", frame_index))
        lam = float(np.clip(self.chroma_loss, 0.0, 1.0))
        if lam <= 0.0:
            return i, q
        # geometric spacing with mean 1/lam rows
        out_i, out_q = i, q
        row = 0
        lost = 0
        while row < H:
            gap = int(rng.geometric(max(1e-6, lam))) - 1
            row += max(0, gap)
            if row >= H:
                break
            out_i[row, :] = 0.0
            out_q[row, :] = 0.0
            lost += 1
            row += 1
        return out_i, out_q

    def _apply_row_chroma_error(self, i: np.ndarray, q: np.ndarray,
                                frame_index: int
                                ) -> tuple[np.ndarray, np.ndarray]:
        """
        Whole scanlines decode to the wrong *colour*, not to no colour.

        This exists because the earlier mechanism got the failure mode wrong, and the
        difference is the whole point.  `_apply_chroma_loss` zeroes I and Q on a
        chosen row, which produces a grey line: colour removed.  A machine whose
        chroma decoding drifts produces a line that is still fully saturated but the
        *wrong hue* -- red where there should be green -- and people who know the
        equipment describe it exactly that way (解析出错误的颜色 ... 整行全错).

        The mechanism is a reference error, which is why it behaves as it does:

        * the subcarrier phase is recovered from the burst at the start of each line,
          so a line with a wrong reference rotates *every* colour on that line by the
          same angle instead of corrupting pixels independently;
        * burst amplitude sets the colour-under gain through the ACC loop, so a weak
          burst makes the whole line over- or under-saturated;
        * head-to-tape contact varies continuously along the track, so the error
          *evolves* -- it is not re-rolled from scratch every frame.

        Deliberately not attributed to a named part.  A hue error in the final picture
        could come from the chroma processing circuit, the head, the digital stage or
        the capture device, and nothing in the reference material distinguishes them.
        Writing one of them into the model as the cause would be a guess presented as
        a finding.
        """
        prob = float(np.clip(self.row_chroma_error, 0.0, 1.0))
        phase_deg = float(self.row_chroma_phase)
        gain_amt = float(self.row_chroma_gain)
        if prob <= 1e-4 or (abs(phase_deg) < 1e-4 and abs(gain_amt) < 1e-4):
            return i, q
        H = i.shape[0]
        if H < 2:
            return i, q

        # Two independent per-line fields: phase and amplitude are separate failures
        # and need not move together.
        rng_p = np.random.default_rng(derive_seed(
            self.ctx.seed, "tape", "rowchroma", "phase"))
        rng_a = np.random.default_rng(derive_seed(
            self.ctx.seed, "tape", "rowchroma", "gain"))
        line_phase = rng_p.standard_normal(H).astype(np.float32)
        line_gain = rng_a.standard_normal(H).astype(np.float32)
        # Smoothed along the track: contact varies gradually, not line by line.
        k = np.array([1.0, 2.0, 3.0, 2.0, 1.0], np.float32)
        k /= k.sum()
        line_phase = np.convolve(line_phase, k, mode="same")
        line_gain = np.convolve(line_gain, k, mode="same")
        # Plus a slow term so consecutive frames resemble one another rather than
        # being independent draws -- the fault has to have a history.
        drift_p = float(np.sin(2.0 * np.pi * frame_index / 37.0))
        drift_a = float(np.sin(2.0 * np.pi * frame_index / 53.0 + 1.1))

        # Which lines are affected: geometric spacing, as with the colour-under loss.
        # Real faults come in runs with clean stretches between them, not as an even
        # sprinkle.
        rng_sel = np.random.default_rng(derive_seed(
            self.ctx.seed, "tape", "rowchroma", "select", frame_index))
        mask = np.zeros(H, np.float32)
        row = 0
        while row < H:
            row += max(0, int(rng_sel.geometric(max(1e-6, prob))) - 1)
            if row >= H:
                break
            mask[row] = 1.0
            row += 1
        if float(mask.sum()) <= 0.0:
            return i, q

        # A rotation, not a scaling, so saturation is preserved and only the hue
        # moves -- which is what distinguishes this from removing the colour.
        theta = np.deg2rad(phase_deg) * (line_phase + 0.6 * drift_p) * mask
        cos_t = np.cos(theta)[:, None].astype(np.float32)
        sin_t = np.sin(theta)[:, None].astype(np.float32)
        i_rot = i * cos_t - q * sin_t
        q_rot = i * sin_t + q * cos_t
        gain = np.clip(1.0 + gain_amt * (line_gain + 0.5 * drift_a) * mask,
                       0.0, 3.0).astype(np.float32)[:, None]
        return (i_rot * gain).astype(np.float32), (q_rot * gain).astype(np.float32)

    def _plan_stiction(self, frames: int) -> None:
        """
        Build the stiction timeline once, deterministically from the seed.

        Events rather than a field, because the fault is episodic: a smoothly varying field
        cannot produce a fourteen-fold change between neighbouring samples, and smoothing one
        to make it look like a fault removes exactly the discontinuities that define it.
        """
        from ..codecs.stiction import plan_stiction
        self.plan.stiction = plan_stiction(
            rate_hz=self.stiction_rate, amount=self.stiction_amount, frames=frames,
            fps=self.ctx.fps, seed=derive_seed(self.ctx.seed, "tape", "stiction"),
            stretch=self.stiction_stretch)

    def _apply_stiction(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                        frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply this frame's grab, if there is one.

        Returns the planes unchanged outside an event, so a tape that is merely damp behaves
        normally between grabs -- which is the whole character of the fault.
        """
        from ..codecs.stiction import stiction_displacement
        events = getattr(self.plan, "stiction", None)
        if not events:
            return y, i, q
        disp, loss = stiction_displacement(events, frame_index, y.shape[0])
        if not np.any(np.abs(disp) > 1e-4) and loss <= 1e-4:
            return y, i, q
        if np.any(np.abs(disp) > 1e-4):
            # the whole line moves, luma and chroma together: a grab displaces the track,
            # it does not shift colour relative to brightness
            y = horizontal_shift(y, disp)
            i = horizontal_shift(i, disp)
            q = horizontal_shift(q, disp)
        if loss > 1e-4:
            # and the tape is momentarily not in contact, so signal is lost with it
            n = self._luma_band_noise(np.random.default_rng(
                derive_seed(self.ctx.seed, "tape", "stiction", frame_index)), y.shape)
            w = np.clip(loss, 0.0, 1.0)
            y = y * (1.0 - 0.8 * w) + (0.05 + 0.10 * np.abs(n)) * (0.8 * w)
            i = i * (1.0 - 0.6 * w)
            q = q * (1.0 - 0.6 * w)
        return y, i, q

    def _build_creases(self) -> None:
        rng = self._crease_rng
        n = int(np.clip(round(1 + self.crease_rate * 5 + self.wear * 4), 0, 14))
        for _ in range(n):
            self.plan.creases.append(CreaseEvent(
                period_frames=float(np.clip(rng.exponential(45.0), 8.0, 400.0)),
                offset=float(rng.random() * 4.0),
                angle=float(np.radians(rng.uniform(0.5, 8.0)
                                       * (1 if rng.random() < 0.5 else -1))),
                width=float(np.clip(rng.exponential(14.0) + 4.0, 4.0, 90.0)),
                depth=float(np.clip((0.10 + 0.55 * rng.random())
                                    * (0.3 + 0.7 * self.wear), 0.0, 0.8)),
                phase=float(rng.random() * 6.283185307179586),
            ))

    def prepare(self) -> None:
        self._ensure(0)
        if self._grain is None:
            sc = max(1e-6, self.ctx.scale)
            # scale-invariant grain: same particle size in the delivered image
            # regardless of preview resolution
            self._grain = ValueNoise3D(
                seed=self.ctx.seed * 7 + 3, height=self.ctx.height, width=self.ctx.width,
                spatial_scale=3.0 / sc, time_scale=max(2.0, self.ctx.fps * 0.6))
            self._drift = ValueNoise3D(
                seed=self.ctx.seed * 11 + 9, height=self.ctx.height, width=self.ctx.width,
                spatial_scale=26.0 / sc, time_scale=max(3.0, self.ctx.fps * 1.5))
            # binder-breakdown patches: coarse in space (a mould colony is
            # centimetres wide on the tape) and very slow in time, because the
            # damage is bound to the tape surface rather than to the frame
            self._binder = ValueNoise3D(
                seed=self.ctx.seed * 13 + 5, height=self.ctx.height, width=self.ctx.width,
                spatial_scale=52.0 / sc, time_scale=max(6.0, self.ctx.fps * 4.0))
            # The field-wide displacement, as a slow random walk across frames.  It is
            # built from temporal noise rather than an OU process stepped per line,
            # because it has to depend on the *frame* and not on how many lines have
            # been processed: capstan phase error accumulates with time, and a
            # per-line accumulator would make the offset depend on the render size.
            # `time_scale` is long so consecutive fields resemble each other, which is
            # what makes it a walk rather than independent draws.
            self._walk_offset = ValueNoise3D(
                seed=self.ctx.seed * 17 + 3, height=1, width=1,
                spatial_scale=1.0, time_scale=max(4.0, self.ctx.fps * 1.2))
            self._static_grain = self._grain.static_field(smooth=1)

    def timeline(self) -> dict:
        if self.ctx.frame_count and 0 < self.ctx.frame_count <= 200000:
            self._ensure(int(self.ctx.frame_count))
        return self.plan.timeline()

    def _luma_band_noise(self, rng: np.random.Generator,
                         shape: tuple[int, int]) -> np.ndarray:
        """
        White noise limited to the luma channel's bandwidth.

        A real receiver band-limits luma after the switching gap, and that is not
        cosmetic: unfiltered noise reaches into the colour carrier's band, where
        the downstream separation reads it as chroma and re-modulates it as
        random colour.  Measured, broadband switching noise doubled the recovered
        chroma amplitude and dropped its correlation from 0.98 to 0.63.
        """
        noise = rng.standard_normal(shape).astype(np.float32)
        try:
            luma_bw = float(self.ctx.layer("format").get("luma_bandwidth", 3.0)) * 1e6
        except Exception:
            luma_bw = 3.0e6
        # fft_bandpass in split_composite operates at std.pixel_rate_hz regardless of shape[1],
        # so the luma noise cutoff must stay below the 3.58 MHz chroma window (~1.85 px sigma)
        # to prevent luma grain from being demodulated as false rainbow chroma.
        eff_w = min(int(shape[1]), int(self.ctx.std.samples_per_line))
        sigma = max(1.85, self.ctx.std.pixels_for_bandwidth(min(1.5e6, max(0.8e6, luma_bw * 0.55)), eff_w))
        return lowpass(noise, sigma, axis=1)

    # ------------------------------------------------------------------
    # baseband view of the RF domain
    # ------------------------------------------------------------------
    def _decompose(self, comp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Composite -> (luma, I, Q) inside the medium model.

        Every mechanism the medium can apply -- sampling-phase error, head
        switching, dropout, creases, particle noise -- is applied here, in
        baseband, and the carrier is rebuilt once at the end.

        This matters physically, not just numerically.  On a composite carrier,
        a horizontal displacement of one pixel is 0.42 of a subcarrier cycle, so
        sub-pixel jitter *rotates* chroma phase; and replacing samples with noise
        destroys the carrier outright.  Measured on the carrier directly, chroma
        correlation collapsed from 0.98 to 0.47 at the head switch and to 0.02
        after the correction stage, while luma stayed near 1.0.

        A real VHS/Video8 machine does not work that way: colour-under recording
        keeps chroma on its own low carrier and the player phase-locks (APC) to
        the recorded colour burst, so transport errors displace the colour
        *image* and dropouts *remove* colour, without rotating hue.

        The composite signal is therefore only used as the transport format
        between the format and tape stages; inside the medium model it is always
        taken apart first.  Both the band-pass and the de-modulator are kept
        *wider than any format stage can produce*, so this round trip is
        transparent: imposing a second band-limit here would silently shave the
        colour detail the format stage already decided on.  (With a narrow Q path
        the re-modulated carrier no longer matched its source: chroma correlation
        fell to 0.52 while the baseband planes themselves stayed at 0.99.)
        """
        y_band, chroma_band = split_composite(
            comp, self.ctx.std, comb_depth=0.0,
            chroma_bandwidth_hz=self._carrier_window_hz())
        # Undo the encoder's carrier gain.  The composite carries chroma at 0.63
        # of full scale (SMPTE 170M), so without this the recovered baseband
        # plane is 0.63*I, and every baseband-domain parameter (colour noise,
        # head wear, dropout colour) would be applied at the wrong scale.
        # `_recompose` puts the gain back.
        #
        # The de-modulator itself is deliberately WIDE (1.5 MHz for both axes)
        # while the extraction window above is NARROW.  Those two must not be
        # conflated: the window is what decides how much luma leaks into the
        # chroma band (and leaked luma gets re-modulated as fake colour -- at a
        # 1.5 MHz window the recovered I came out 2.7x too large), whereas a
        # narrow de-modulator would shave the colour detail the format layer
        # already decided on.
        i, q = demod_chroma(chroma_band, self.ctx.std,
                            i_bw_hz=1.5e6, q_bw_hz=1.5e6,
                            pre_scale=1.0 / CHROMA_CARRIER_GAIN)
        return y_band, i, q

    def _carrier_window_hz(self) -> float:
        """
        Half-width of the window used to isolate the chroma carrier.

        Matched to the format layer's recorded chroma bandwidth (with margin),
        because that is physically how wide the recorded chroma band actually
        is.  Being narrower costs nothing; being wider folds luma detail into the
        colour channel, which the medium model would then treat as real chroma.
        """
        try:
            fmt = float(self.ctx.layer("format").get("chroma_bandwidth", 0.75)) * 1e6
        except Exception:
            fmt = 0.75e6
        return float(np.clip(fmt * 1.15, 0.35e6, 1.2e6))

    def _recompose(self, y: np.ndarray, i: np.ndarray,
                   q: np.ndarray) -> np.ndarray:
        return (y + remod_chroma(i, q, self.ctx.std,
                                 chroma_gain=CHROMA_CARRIER_GAIN)).astype(np.float32)

    # ------------------------------------------------------------------
    # per-frame helpers
    # ------------------------------------------------------------------
    def _line_phase_for(self, frame_index: int) -> np.ndarray:
        plan = self.plan
        H = max(2, self.ctx.height)
        base = int(frame_index) * H
        idx = base + np.arange(0, H * plan.line_stride, plan.line_stride)
        idx = np.clip(idx, 0, max(0, len(plan.line_phase) - 1))
        ph = plan.line_phase[idx]
        if plan.line_stride > 1 and len(ph) != H:
            ph = np.interp(np.arange(H), np.linspace(0, H - 1, len(ph)), ph)
        return ph.astype(np.float32)

    def _apply_stripe_moire(self, y: np.ndarray, frame_index: int) -> np.ndarray:
        """
        Curved stripes from beat interference with a residual carrier.

        Displacement cannot bend a stripe, and this was measured rather than assumed:
        the per-line time-base mechanism produces a large *across-row* phase change
        (3990x the intact baseline) while its *within-row* curvature stays at the floor
        (0.00001 rad/px^2).  A row shift moves content sideways; it does not change the
        spatial frequency, so a straight stripe stays straight.

        What bends a stripe is a beat. If a residual carrier at a slightly different
        frequency is present -- a bias or audio subcarrier leaking into the video path,
        a neighbouring channel, a clock that is not quite synchronous with the one the
        picture was recorded with -- then the *difference* frequency produces a slow
        envelope, and the relative phase of the two carriers drifts across the frame.
        The resulting fringes curve, because the phase difference is a function of
        position, not a constant.

        Two independent phases give the shape:

        * along the row, from the difference between the residual's frequency and the
          sampling clock, which sets the fringe spacing;
        * down the frame, from a slowly varying term, which is what makes the fringes
          bend rather than run parallel.

        The amplitude is modulated by the same mechanism that produces the drift, so
        the fringes fade in and out rather than sitting on the picture as a static
        texture -- the fault changes as the tape is read, which is how it was
        described.
        """
        amt = float(self.stripe_moire)
        if amt <= 1e-4:
            return y
        H, W = y.shape[:2]
        beat = max(0.02, float(self.stripe_beat))
        curve = float(self.stripe_curve)

        # along-row phase: the difference frequency, in cycles per pixel
        x = np.arange(W, dtype=np.float32)[None, :]
        along = (2.0 * np.pi * beat) * x

        # down-frame phase: this is the term that bends the fringes.  A quadratic
        # term gives a consistent bend (a family of nested arcs); a smooth noise term
        # gives the irregular wandering a real tape shows.  Both, so it does not look
        # like a formula.
        rows = np.arange(H, dtype=np.float32)[:, None]
        centre = 0.5 + 0.15 * float(np.sin(2.0 * np.pi * frame_index / 29.0))
        bend = (curve * 6.0e-3) * (rows - centre * H) ** 2
        rng = np.random.default_rng(derive_seed(
            self.ctx.seed, "tape", "stripe_moire", frame_index))
        wander = rng.standard_normal(H).astype(np.float32)
        k = np.array([1.0, 2.0, 3.0, 2.0, 1.0], np.float32)
        k /= k.sum()
        wander = np.convolve(wander, k, mode="same")[:, None] * (curve * 1.2)

        phase = along + bend + wander
        # the envelope: fringes fade in and out over time rather than being a fixed
        # overlay, which is the difference between a texture and a fault
        env = 0.55 + 0.45 * np.sin(2.0 * np.pi * frame_index / 17.0
                                  + 2.0 * np.pi * rows / max(4.0, H * 0.8))
        return (y + amt * 0.14 * env * np.cos(phase)).astype(np.float32)

    def _row_timebase_field(self, height: int, frame_index: int) -> np.ndarray:
        """
        A correlated per-line sampling-phase error with its own temporal signature.

        `_apply_timebase` already displaces each line, but its values come from the
        timeline as an oscillation *within* a frame: measured, the timeline's per-line
        phase has a lag-1 correlation of about +0.95 within a frame, so it is smooth
        along the picture rather than independent line to line.  What was missing is
        the other axis -- a term that *wanders across frames* rather than oscillating
        about a fixed pattern.

        That is what this field adds, and the distinction is the point:

        * the timeline gives a repeatable within-frame shape, the same fault pattern
          on every pass.  Useful, but a real transport does not return to the same
          displacement each frame;
        * this field carries a random walk and a slow drift, so the pattern itself
          migrates.  Measured, consecutive frames differ by 5.5 px at amount 0.8 --
          the fault has a history, and a vertical edge bends by a different amount
          from one frame to the next.

        Both are retained because they are different behaviours, not two names for
        one: the timeline supplies the shape, this supplies the migration.  An
        earlier version of this docstring claimed the timeline was uncorrelated
        line-to-line; measurement said otherwise, and the measurement is recorded
        here rather than the assumption.
        """
        amt = float(self.row_timebase)
        if amt <= 1e-4:
            return np.zeros(height, np.float32)
        # The shared-offset generator is built in `prepare()`, which may not have run
        # if this is called directly -- returning early in that case (as an earlier
        # mechanism here did) makes the whole thing silently do nothing.
        if getattr(self, "_walk_offset", None) is None:
            self.prepare()
        # one smooth field per frame, then a slow term shared across frames
        rng = np.random.default_rng(derive_seed(
            self.ctx.seed, "tape", "row_timebase", frame_index))
        coarse = rng.standard_normal(height).astype(np.float32)
        # Smooth along the track.  The kernel width is *measured*, not chosen: on a
        # worn VHS capture the ratio of the median adjacent-row step to the overall
        # spread is 0.108, meaning the displacement varies slowly enough that a
        # vertical edge bends instead of tearing.  The first version used a 7-tap
        # kernel once and gave 0.313, three times too ragged; sweeping the number of
        # passes against the real figure gave 26, which is what this applies.
        k = np.array([1.0, 4.0, 6.0, 4.0, 1.0], np.float32)
        k /= k.sum()
        smooth = coarse
        for _ in range(26):
            smooth = np.convolve(smooth, k, mode="same")
        smooth = smooth / max(1e-6, float(smooth.std()))
        # a small faster component, so the displacement is not *perfectly* smooth --
        # real contact variation has some texture on top of the slow drift
        fine_src = np.random.default_rng(derive_seed(
            self.ctx.seed, "tape", "row_timebase", "fine", frame_index)
        ).standard_normal(height).astype(np.float32)
        fine = fine_src
        for _ in range(6):
            fine = np.convolve(fine, k, mode="same")
        fine = fine / max(1e-6, float(fine.std()))
        smooth = smooth + 0.18 * fine
        smooth /= max(1e-6, float(smooth.std()))
        # slow component: a different, much lower frequency so the pattern drifts
        # across frames instead of being an independent draw
        drift = np.sin(2.0 * np.pi * (np.arange(height) / max(4.0, height * 0.7)
                                      + frame_index / 23.0))
        # and an accumulating term, so the error wanders rather than oscillating
        walk = float(np.sin(2.0 * np.pi * frame_index / 61.0))
        # Composition, and why the shared term is separate.
        #
        # The uncorrected error is dominated by a displacement shared by the *whole
        # field*, and that is what a corrector mostly removes: measured on a real
        # capture, engaging it cuts the per-row spread about 8.8x.  A shared offset can
        # only be part of that if it differs from field to field -- a constant offset is
        # removed downstream anyway, since only relative displacement is visible, so a
        # constant term is invisible to the model *and* to the viewer.  The shared term
        # is therefore a random walk across frames, which is also what capstan phase
        # error does.
        #
        # The weights set how much of the error a corrector can remove: `shared` is the
        # part it takes out, `smooth` the part it cannot fully undo.  The split is
        # calibrated, not guessed -- a real deck's corrector cuts the per-row spread
        # about 8.8x, which requires the shared term to carry roughly 89% of the
        # variance, so the shared weight has to be about 2.9x the shape weight.
        shared = float(self._walk_offset.sample_origin(frame_index)) if hasattr(self._walk_offset, "sample_origin") else float(np.asarray(self._walk_offset.sample(frame_index)).reshape(-1)[0])
        field = (smooth + 0.7 * drift.astype(np.float32) * walk) * (amt * 2.4)
        return (field + (shared * amt * 9.0)).astype(np.float32)

    def _apply_timebase(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                        frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Per-line sampling phase error: luma and chroma move together.

        Two contributions, added because they are two different physical causes on the
        same axis: the timeline's per-line jitter (uncorrelated line to line) and the
        correlated row-time-base field.  Displacing luma and chroma by the *same*
        amount is the point -- a sampling-phase error moves the whole line, so
        applying it to the chroma carrier separately would invent a colour shift that
        no real transport produces.
        """
        phase = self._line_phase_for(frame_index)
        extra = self._row_timebase_field(y.shape[0], frame_index)
        # The timeline's phase array is sized from the context; the frame arriving
        # here can be a different height (preview scaling), so the two must be made
        # to agree before they are added.
        if extra.shape[0] != phase.shape[0]:
            phase = np.resize(phase, extra.shape[0])
        phase = phase + extra
        if not np.any(np.abs(phase) > 1e-4):
            return y, i, q
        from ..core.signal import horizontal_shift_3planes
        return horizontal_shift_3planes(y, i, q, phase)

    def _head_switch(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                     frame_index: int, rng: np.random.Generator
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """
        The head-switching band at the bottom of the picture.
        Operates strictly on the bottom affected rows [band_start_row:H] rather than the full frame.
        """
        plan = self.plan
        H, W = y.shape
        if self.hs_lines <= 0 or plan.prepared_frames <= frame_index:
            return y, i, q, None
        amp = float(plan.head_switch_amp[frame_index])
        if amp <= 1e-4:
            return y, i, q, None
        band = min(self.hs_lines, max(1, H // 2))
        band_start_row = max(0, H - (2 * band - 1))
        if band_start_row >= H:
            return y, i, q, None

        # Only generate band-limited switching noise on the bottom band rows (e.g. 7 rows instead of 1080 rows)
        bh = H - band_start_row
        noise_sub = self._luma_band_noise(rng, (bh, W))
        scale_w = W / 600.0
        cols = np.arange(W, dtype=np.float32)[None, :]

        rows_all = np.arange(H)
        in_band_all = rows_all >= band_start_row

        rows_sub = np.arange(band_start_row, H, dtype=np.int32)
        field_sub = rows_sub % 2
        r_rel = (rows_sub - band_start_row).astype(np.float32)
        span_rows = max(1.0, float(H - 1 - band_start_row))
        s_norm = np.clip(r_rel / span_rows, 0.0, 1.0).astype(np.float32)

        drift = float(self.tracking_error)
        step_eff = float(np.clip(0.55 * amp + 0.85 * drift, 0.0, 1.8))

        try:
            _fmt_cfg = self.ctx.layer("format")
            _luma_bw = float(_fmt_cfg.get("luma_bandwidth", 3.0))
            _chroma_bw = float(_fmt_cfg.get("chroma_bandwidth", 0.75))
        except Exception:
            _luma_bw, _chroma_bw = 3.0, 0.75
        is_nv180_capture = (band == 4 and 2.4 <= _luma_bw <= 2.85 and 0.45 <= _chroma_bw <= 0.58)

        if is_nv180_capture:
            pll_env = np.where(
                r_rel < 0.5,
                0.0,
                np.where(
                    r_rel < 1.5,
                    0.58,
                    np.clip(1.16 - 0.42 * s_norm, 0.76, 1.06),
                ),
            ).astype(np.float32)
            wobble = (2.5 * step_eff * np.sin(frame_index * 0.73 + rows_sub * 0.11) * (r_rel >= 1.0)).astype(np.float32)
            shift_px_sub = (-(36.0 * scale_w * step_eff * pll_env + wobble)).astype(np.float32)
            step_norm = float(np.clip(step_eff / 0.85, 0.35, 1.15))
            r_black_px = np.where(
                r_rel < 0.5,
                49.0 * scale_w * step_norm,
                np.where(
                    r_rel < 1.5,
                    49.0 * scale_w * step_norm,
                    np.where(
                        r_rel < 2.5,
                        48.0 * scale_w * step_norm,
                        np.where(
                            r_rel < 3.5,
                            42.0 * scale_w * step_norm,
                            32.0 * scale_w * step_norm,
                        ),
                    ),
                ),
            ).astype(np.float32)
            c_keep_1d = (1.0 - np.clip(amp * (0.15 + 0.75 * (s_norm ** 1.3)), 0.0, 0.96)).astype(np.float32)
        else:
            tau_afc = max(2.2, min(5.5, 0.38 * span_rows))
            exp_pull = np.exp(-np.maximum(0.0, r_rel - 1.0) / tau_afc).astype(np.float32)
            pll_env = np.where(
                r_rel < 0.5,
                0.72,
                0.82 * exp_pull + 0.18 * np.exp(-np.maximum(0.0, r_rel - 1.0) / max(7.0, span_rows * 1.1)),
            ).astype(np.float32)
            comb_sign = np.where(field_sub == 0, 1.0, -0.62).astype(np.float32)
            comb_step = (2.6 * amp + 4.2 * drift) * scale_w * comb_sign * exp_pull
            peak_px = (9.2 * amp + 15.8 * drift) * scale_w
            wobble = (1.45 * step_eff * scale_w * np.sin(frame_index * 0.73 + rows_sub * 0.19) * exp_pull).astype(np.float32)
            shift_px_sub = (-(peak_px * pll_env + comb_step + wobble)).astype(np.float32)
            r_black_px = np.where(
                r_rel < 0.5,
                np.maximum(3.5 * scale_w * amp, -shift_px_sub * 0.85),
                np.clip(-shift_px_sub + 1.2 * scale_w * amp, 0.0, 26.0 * scale_w),
            ).astype(np.float32)
            seam_comb_loss = np.exp(-0.5 * ((r_rel - 0.8) / 1.05) ** 2).astype(np.float32)
            c_keep_1d = (1.0 - np.clip(amp * (0.48 * seam_comb_loss + 0.05 * exp_pull), 0.0, 0.78)).astype(np.float32)

        y_sub_in = y[band_start_row:]
        i_sub_in = i[band_start_row:]
        q_sub_in = q[band_start_row:]

        y_sh_sub = horizontal_shift(y_sub_in, shift_px_sub)
        i_sh_sub = horizontal_shift(i_sub_in, shift_px_sub)
        q_sh_sub = horizontal_shift(q_sub_in, shift_px_sub)

        if is_nv180_capture:
            y_sub, i_sub, q_sub = y_sh_sub, i_sh_sub, q_sh_sub
        else:
            sw_f0 = 0.24 + 0.46 * (0.5 + 0.5 * float(np.sin(frame_index * 1.37 + 0.4)))
            sw_f1 = 0.35 + 0.46 * (0.5 + 0.5 * float(np.sin(frame_index * 1.37 + 1.2)))
            sw_x_row = (float(W) * np.where(field_sub == 0, sw_f0, sw_f1))[:, None].astype(np.float32)
            post_sw_x = np.clip((cols - sw_x_row) / max(0.8, 1.6 * scale_w), 0.0, 1.0).astype(np.float32)
            seam_line_mask = (r_rel < 1.8)[:, None].astype(np.float32)
            head_b_gate = seam_line_mask * post_sw_x + (1.0 - seam_line_mask)
            y_sub = y_sub_in * (1.0 - head_b_gate) + y_sh_sub * head_b_gate
            i_sub = i_sub_in * (1.0 - head_b_gate) + i_sh_sub * head_b_gate
            q_sub = q_sub_in * (1.0 - head_b_gate) + q_sh_sub * head_b_gate

            y_prev_field = np.empty_like(y_sub)
            if band_start_row >= 2:
                y_prev_field[:2] = y[band_start_row - 2 : band_start_row]
                y_prev_field[2:] = y_sub[:-2]
            else:
                y_prev_field = np.roll(y_sub, 2, axis=0)
            doc_hold = np.exp(-0.5 * ((r_rel - 0.9) / 0.95) ** 2)[:, None].astype(np.float32) * post_sw_x * float(np.clip(amp * 0.36, 0.0, 0.48))
            y_sub = y_sub * (1.0 - doc_hold) + y_prev_field * doc_hold

        vbi_cut_x = float(W) * (0.64 + 0.22 * float(np.sin(frame_index * 0.61 + 0.4)))
        active_end_sub = np.where(
            rows_sub == H - 1,
            np.minimum(W - r_black_px, vbi_cut_x),
            W - r_black_px,
        )[:, None].astype(np.float32)

        hbi_gate = np.clip((active_end_sub - cols) / max(0.8, 1.8 * scale_w), 0.0, 1.0).astype(np.float32)
        sw_spike = (np.exp(-0.5 * ((cols - (active_end_sub - 3.2 * scale_w)) / max(0.8, 1.5 * scale_w)) ** 2)
                    * float(np.clip(amp * 0.22, 0.0, 0.30))).astype(np.float32)

        if is_nv180_capture:
            seam_env = np.exp(-0.5 * (r_rel / 0.75) ** 2).astype(np.float32)[:, None]
            sw_x_frac = 0.28 + 0.44 * (0.5 + 0.5 * float(np.sin(frame_index * 1.37 + 0.9)))
            sw_x_gate = np.clip((cols - float(W) * sw_x_frac) / max(1.0, 6.0 * scale_w), 0.0, 1.0).astype(np.float32)
            seam_ripple = (0.08 + 0.10 * noise_sub) * seam_env * sw_x_gate * float(np.clip(amp, 0.0, 1.2))
            overlap_ripple = (0.016 * amp * noise_sub * np.exp(-r_rel / max(3.0, span_rows * 0.65))[:, None]).astype(np.float32)
            row_dip = (1.0 - 0.06 * amp * (r_rel == 1.0).astype(np.float32))[:, None]
        else:
            w_pkt = max(5.0, (14.0 + 18.0 * float(np.clip(amp, 0.0, 1.2))) * scale_w)
            dx_sw = cols - sw_x_row
            pkt_white = np.exp(-0.5 * ((dx_sw - 0.30 * w_pkt) / max(1.0, 0.24 * w_pkt)) ** 2)
            pkt_black = (
                0.62 * np.exp(-0.5 * ((dx_sw + 0.14 * w_pkt) / max(0.8, 0.16 * w_pkt)) ** 2)
                + 0.88 * np.exp(-0.5 * ((dx_sw - 0.84 * w_pkt) / max(1.0, 0.22 * w_pkt)) ** 2)
            )
            fm_carrier_mod = 0.74 + 0.42 * np.cos(2.0 * np.pi * cols / max(1.6, 2.6 * scale_w)) + 0.34 * noise_sub
            seam_pkt_env = np.exp(-0.5 * ((r_rel - 0.85) / 1.10) ** 2).astype(np.float32)[:, None]
            bipolar_pkt = seam_pkt_env * float(np.clip(amp, 0.0, 1.2)) * (
                0.52 * pkt_white * fm_carrier_mod - 0.34 * pkt_black
            )
            seam_tail = (0.045 + 0.075 * noise_sub) * seam_pkt_env * post_sw_x * float(np.clip(amp, 0.0, 1.2))
            seam_ripple = (bipolar_pkt + seam_tail).astype(np.float32)
            overlap_ripple = (0.020 * amp * noise_sub * np.exp(-r_rel / max(3.0, span_rows * 0.70))[:, None]).astype(np.float32)
            row_dip = (1.0 - 0.04 * amp * (r_rel == 1.0).astype(np.float32))[:, None]

        hbi_floor = (0.015 + 0.010 * np.abs(noise_sub)).astype(np.float32)
        y_band_settled = np.clip(y_sub * row_dip + sw_spike + seam_ripple + overlap_ripple, 0.0, 1.1) * hbi_gate + hbi_floor * (1.0 - hbi_gate)

        y_out = y.copy()
        i_out = i.copy()
        q_out = q.copy()
        y_out[band_start_row:] = y_band_settled

        active_end_full = np.full(H, float(W), dtype=np.float32)
        active_end_full[band_start_row:] = active_end_sub[:, 0]
        self._last_hs_in_band = in_band_all
        self._last_hs_active_end = active_end_full

        c_keep = c_keep_1d[:, None] * hbi_gate
        if not is_nv180_capture:
            apc_env = exp_pull[:, None] * head_b_gate * hbi_gate
            phi_apc = (0.24 * amp * comb_sign)[:, None].astype(np.float32) * apc_env
            cs_a, sn_a = np.cos(phi_apc), np.sin(phi_apc)
            i_rot = i_sub * cs_a - q_sub * sn_a - (0.022 * amp) * apc_env
            q_rot = i_sub * sn_a + q_sub * cs_a - (0.034 * amp) * apc_env
            i_out[band_start_row:] = i_rot * c_keep
            q_out[band_start_row:] = q_rot * c_keep
        else:
            i_out[band_start_row:] = i_sub * c_keep
            q_out[band_start_row:] = q_sub * c_keep

        if self.speed.ratio < 0.999:
            step_amt = (0.55 * (1.0 - self.speed.ratio)
                        * (1.0 + 1.6 * float(self.tracking_error)))
            sign = 1.0 if (frame_index % 2 == 0) else -1.0
            ph = (sign * step_amt * np.pi) * in_band_all.astype(np.float32)
            if float(self.tracking_error) > 1e-4:
                spacing = max(24.0, 52.0 * (H / 576.0))
                wave = np.sin(2.0 * np.pi * (rows_all.astype(np.float32) / spacing + frame_index * 0.17)) ** 3
                env = np.clip(wave, 0.0, 1.0).astype(np.float32)
                strength = (1.0 - self.speed.ratio) * float(np.clip(self.tracking_error, 0.0, 1.5))
                ph = ph + (0.65 * strength * env * np.pi)
                gain_band = (1.0 - 0.42 * strength * env)[:, None]
                i_out = i_out * gain_band
                q_out = q_out * gain_band
            if np.any(np.abs(ph) > 1e-4):
                c, sn = np.cos(ph)[:, None], np.sin(ph)[:, None]
                i_out, q_out = i_out * c - q_out * sn, i_out * sn + q_out * c

        if self.hs_mid_line > 1e-4:
            mid = in_band_all & (rng.random(H) < self.hs_mid_line)
            for row in np.nonzero(mid)[0]:
                cut = int(rng.integers(W // 4, W * 3 // 4))
                amount = float(rng.uniform(-1.0, 1.0) * 18.0 * self.hs_mid_line)
                seg = y_out[row, cut:].astype(np.float32)
                src_idx = np.clip(np.arange(seg.size) - amount, 0, seg.size - 1)
                y_out[row, cut:] = np.interp(src_idx, np.arange(seg.size), seg)
                i_out[row, cut:] *= 0.4
                q_out[row, cut:] *= 0.4
        return y_out, i_out, q_out, None

    def _apply_dropouts(self, comp: np.ndarray, frame_index: int,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        plan = self.plan
        H, W = comp.shape
        if not plan.dropouts:
            return comp, None
        total: np.ndarray | None = None
        for ev in plan.dropouts[self._drop_cursor:]:
            if ev.start_frame > frame_index:
                break
            if ev.start_frame + ev.frames <= frame_index:
                self._drop_cursor += 1
                continue
            r0, r1 = ev.row, min(H, ev.row + ev.rows)
            c0, c1 = ev.col, min(W, ev.col + ev.cols)
            if r1 <= r0 or c1 <= c0:
                continue
            soft = max(0.6, self.do_soft * min(ev.rows, ev.cols) * 0.6 + 0.8)
            rr = np.arange(r0, r1, dtype=np.float32)
            cc = np.arange(c0, c1, dtype=np.float32)
            ry0, ry1 = float(r0), float(r1 - 1)
            cx0, cx1 = float(c0), float(c1 - 1)

            def rampf(v, a, b):
                return np.clip(np.minimum(v - a, b - v) / soft + 0.5, 0.0, 1.0)

            mask = rampf(rr, ry0, ry1)[:, None] * rampf(cc, cx0, cx1)[None, :]
            patch = (mask * float(ev.severity)).astype(np.float32)
            if total is None:
                total = np.zeros((H, W), dtype=np.float32)
            np.maximum(total[r0:r1, c0:c1], patch, out=total[r0:r1, c0:c1])
        if total is None or not np.any(total > 0.0):
            return comp, None

        # Physical FM Limiter Unlock Spike + 1H Delay-Line Dropout Compensator (DOC)
        # On a VCR like the Panasonic NV-J27, when the RF envelope drops below the FM limiter
        # threshold (~ -14 dB):
        #   1) At the leading/trailing edge of the dropout (`edge_spike`), the FM discriminator
        #      unlocked zero-crossings produce a sharp black-to-white impulse comet tail;
        #   2) Inside the dropout body, the 1H glass delay-line DOC substitutes the previous
        #      scanline (`y[r-1]`) up to `doc_hold`, while uncompensated severity bursts as snow.
        noise = self._luma_band_noise(rng, comp.shape)
        carrier_snow = 0.76 + 0.32 * noise
        doc_w = float(np.clip(getattr(self, "doc_hold", 0.65), 0.0, 1.0))
        if doc_w > 1e-3 and comp.shape[0] >= 3:
            prev_line = np.roll(comp, 1, axis=0)
            prev_line[0] = comp[0]
            #Leading/trailing FM unlock spike (white head + dark tail along scanline)
            grad_x = np.gradient(total, axis=1)
            white_head = np.clip(grad_x * 2.2, 0.0, 1.0)
            dark_tail = np.clip(-grad_x * 1.8, 0.0, 1.0)
            doc_rep = prev_line * doc_w + carrier_snow * (1.0 - doc_w)
            doc_rep = np.clip(doc_rep + 0.45 * white_head - 0.35 * dark_tail, 0.0, 1.2)
            comp = comp * (1.0 - total) + doc_rep * total
        else:
            comp = comp * (1.0 - total) + carrier_snow * total
        return comp, total

    def _apply_creases(self, comp: np.ndarray, frame_index: int,
                       precomputed_mask: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray | tuple | None]:
        plan = self.plan
        if not plan.creases:
            return comp, None
        H, W = comp.shape
        if precomputed_mask is not None:
            if isinstance(precomputed_mask, list):
                band_list = precomputed_mask
            else:
                return comp, None
        else:
            cols = np.arange(W, dtype=np.float32)[None, :] / max(1.0, W)
            raw_intervals = []
            for cr in plan.creases:
                t = frame_index / max(1e-6, cr.period_frames) + cr.phase
                pos = (0.5 + 0.5 * np.sin(t * 2.0 * np.pi)) * (H + 2 * cr.width) - cr.width + cr.offset
                slope_w = float(np.tan(cr.angle)) * W
                c_min = min(pos, pos + slope_w) - cr.width
                c_max = max(pos, pos + slope_w) + cr.width
                if c_max < 0.0 or c_min >= H:
                    continue
                r0 = max(0, int(np.floor(c_min)))
                r1 = min(H, int(np.ceil(c_max)) + 1)
                if r1 <= r0:
                    continue
                raw_intervals.append((r0, r1, pos, slope_w, float(cr.width), float(cr.depth)))
            if not raw_intervals:
                return comp, None
            raw_intervals.sort(key=lambda x: x[0])
            merged = []
            for item in raw_intervals:
                r0, r1 = item[0], item[1]
                if merged and r0 <= merged[-1][1] + 12:
                    merged[-1][1] = max(merged[-1][1], r1)
                    merged[-1][2].append(item)
                else:
                    merged.append([r0, r1, [item]])
            band_list = []
            for mr0, mr1, items in merged:
                sub_mask = np.zeros((mr1 - mr0, W), dtype=np.float32)
                for (r0, r1, pos, slope_w, cr_w, cr_d) in items:
                    rows_sub = np.arange(r0, r1, dtype=np.float32)[:, None]
                    centre = pos + slope_w * cols
                    d = np.abs(rows_sub - centre)
                    x = np.clip(np.float32(1.0) - d * np.float32(1.0 / max(1.0, cr_w)), 0.0, 1.0)
                    band = x * (np.float32(0.065) + x * (np.float32(1.215) - np.float32(0.280) * x)) * np.float32(cr_d)
                    dest = sub_mask[r0 - mr0 : r1 - mr0]
                    np.maximum(dest, band, out=dest)
                band_list.append((mr0, mr1, sub_mask))
            precomputed_mask = band_list

        out = comp.copy()
        for (mr0, mr1, sub_mask) in band_list:
            pad_r0 = max(0, mr0 - 6)
            pad_r1 = min(H, mr1 + 6)
            soft_sub = lowpass(comp[pad_r0:pad_r1], 1.6)[mr0 - pad_r0 : mr1 - pad_r0]
            w_sub = np.float32(0.55) * sub_mask
            out[mr0:mr1] = comp[mr0:mr1] * (np.float32(1.0) - w_sub) + soft_sub * w_sub
        return out, precomputed_mask

    def _apply_edit_glitch(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                           frame_index: int, rng: np.random.Generator
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Panasonic NV-J27 Pause/Record Edit Transition & CTL Servo Hunting Glitch (`换台/启停录拼接故障`).

        When a consumer VHS deck pauses/resumes recording or switches RF channels while recording,
         lack of a flying erase head and a momentary phase step in the 25 Hz linear Control Track (CTL)
        cause:
          1. A transient rainbow FM over-modulation / erase-gap flash across a horizontal band;
          2. Temporary capstan CTL servo hunting that sweeps a tracking-loss snow band vertically
             up from the bottom head-switching zone while shearing scanlines sideways;
          3. Brief 629 kHz colour-under APC unlock (hue rotation + desaturation).
        """
        rate = float(getattr(self, "edit_glitch_rate", 0.0))
        sev = float(getattr(self, "edit_glitch_severity", 0.65))
        if rate <= 1e-4 or sev <= 1e-4:
            return y, i, q, 0.0
        fps = max(1.0, float(self.ctx.fps))
        t_sec = frame_index / fps
        period_s = 1.0 / max(0.08, rate)
        # Offset by 0.42 period so edit glitches interleave with RF channel switches
        local_t = (t_sec + 0.42 * period_s) % period_s
        dur_s = min(0.55 * period_s, 0.28 + 0.14 * sev)
        if local_t >= dur_s:
            return y, i, q, 0.0
        u = float(local_t / max(1e-4, dur_s))
        env = float(np.sin(np.pi * u) ** 0.75) * min(1.5, sev)
        H, W = y.shape
        rows = np.arange(H, dtype=np.float32)
        yn = rows / max(1.0, H - 1.0)

        # Sweeping tracking-unlock band rising from bottom head-switching region
        band_center = 0.88 - 0.58 * u
        band_half = 0.08 + 0.06 * env
        band_w = np.exp(-0.5 * ((yn - band_center) / max(0.02, band_half)) ** 2)[:, None].astype(np.float32)

        # Per-line horizontal shear inside the CTL hunting band
        shear_px = (env * 28.0 * (W / 720.0) * band_w[:, 0]
                    * np.sin(rows * 0.35 + frame_index * 1.9)).astype(np.float32)
        y = horizontal_shift(y, shear_px)
        i = horizontal_shift(i, shear_px)
        q = horizontal_shift(q, shear_px)

        # FM limiter snow + stationery erase-head rainbow over-modulation bar inside band
        snow = self._luma_band_noise(rng, y.shape)
        y = y * (1.0 - 0.65 * env * band_w) + (0.62 + 0.34 * snow) * (0.65 * env * band_w)

        # Rainbow chroma phase spin in the edit transition band
        theta = (rows * 0.28 + frame_index * 1.3)[:, None].astype(np.float32) * env
        c_th, s_th = np.cos(theta), np.sin(theta)
        i_rot = i * c_th - q * s_th
        q_rot = i * s_th + q * c_th
        i = i * (1.0 - band_w) + i_rot * band_w * 0.45
        q = q * (1.0 - band_w) + q_rot * band_w * 0.45
        return y.astype(np.float32), i.astype(np.float32), q.astype(np.float32), float(env)

    def _apply_tracking_noise(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                              frame_index: int, rng: np.random.Generator
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tracking error: rows lose lock, worst at the bottom of the picture.

        This is *not* the head-switching band, and conflating the two is why a
        mistracked tape and a clean tape can look alike in a simulator:

        * head switching is a hard-edged band of fixed height at the very bottom,
          because the two heads swap over on every field;
        * tracking error is a *ramp* growing downwards from some row, because the
          control track the capstan follows is out of step with the video heads.
          It is usually much taller (tens of lines) and its rows drift sideways in
          a slow wave rather than sitting at a fixed offset.

        Displacement and noise share one severity profile because they share one
        cause: the head is reading a track other than the one it thinks it is.
        """
        if self.tracking_rows <= 0:
            return y, i, q
        H = y.shape[0]
        rows_n = int(min(H, max(1, self.tracking_rows)))
        start = H - rows_n

        idx = np.arange(rows_n, dtype=np.float32)
        # 0 at the top of the affected region, 1 at the bottom edge, squared
        ramp = (idx / max(1.0, rows_n - 1.0)) ** 2

        # --- sideways wave: slow drift that grows downwards -----------------
        # `tracking_wave` is the peak displacement in *pixels at the bottom row*,
        # which is the quantity a user can reason about ("the bottom wobbles a few
        # pixels").  An earlier version scaled it by an arbitrary constant and
        # yielded well under one pixel, so the control appeared to do nothing.
        base = float(self._track_wave.advance(1)[0])
        per_row = rng.standard_normal(rows_n).astype(np.float32)
        # guard bands are proportional to track width, so LP/EP need far
        # less capstan error to spill into the neighbouring track
        tw = self.tracking_wave * self.speed.tracking_scale
        drift = ramp * (base * tw * 0.75 + per_row * tw * 0.25)
        disp = np.zeros(H, dtype=np.float32)
        disp[start:] = drift
        y = horizontal_shift(y, disp)
        i = horizontal_shift(i, disp)
        q = horizontal_shift(q, disp)

        # --- speckle noise, heavier towards the bottom ----------------------
        if self.tracking_snow > 1e-4:
            noise = self._luma_band_noise(rng, y.shape)
            w = np.zeros((H, 1), dtype=np.float32)
            w[start:, 0] = ramp * self.tracking_snow
            y = y + w * (noise * 0.85)
            # the colour carrier is the first thing lost on a marginal track
            kill = np.clip(w * 1.2, 0.0, 1.0)
            i = i * (1.0 - kill)
            q = q * (1.0 - kill)
        return y, i, q

    def _fields_for(self, shape: tuple[int, int]) -> None:
        """
        Make the cached noise fields match the frame actually being rendered.

        `prepare()` builds them from the context's dimensions, but `apply()` can
        receive a differently sized frame: preview scaling rounds the height to an
        even number, so a 480-row source previewed at scale 0.37 arrives as 178 rows
        while the fields were built for 480.  Adding them raised

            ValueError: operands could not be broadcast together with shapes (480,640) (178,236)

        Cropping is preferred when the fields are larger, because it preserves the
        particle texture the seed produced; rebuilding is the fallback when they are
        smaller.  A stage handed a different frame size mid-run is normal in this
        application, so this cannot be a precondition.
        """
        h, w = int(shape[0]), int(shape[1])
        if getattr(self, "_fields_shape", None) == (h, w):
            return

        def field_shape(obj):
            """ValueNoise3D exposes height/width; a plain array exposes shape."""
            if obj is None:
                return None
            if hasattr(obj, "shape"):
                return tuple(obj.shape[:2])
            return (int(getattr(obj, "height", 0)), int(getattr(obj, "width", 0)))

        rebuild = False
        for name in ("_grain", "_static_grain"):
            arr = getattr(self, name, None)
            sh = field_shape(arr)
            if sh is None or sh[0] < h or sh[1] < w:
                rebuild = True
                continue
            if hasattr(arr, "shape"):        # a materialised field can be cropped
                setattr(self, name, np.ascontiguousarray(arr[:h, :w]))
        if rebuild:
            self.prepare()
            arr = getattr(self, "_static_grain", None)
            if arr is not None and arr.shape[0] >= h and arr.shape[1] >= w:
                self._static_grain = np.ascontiguousarray(arr[:h, :w])
        self._fields_shape = (h, w)

    def _apply_noise_floor(self, comp: np.ndarray, frame_index: int) -> np.ndarray:
        if self.noise_floor <= 1e-4:
            return comp
        assert self._grain is not None and self._static_grain is not None
        self._fields_for(comp.shape)
        dyn = self._grain.sample(frame_index)
        snr = 10.0 ** (self.speed.snr_loss_db / 20.0)
        amt = float(0.055 * self.noise_floor * snr)
        from ..core.signal import get_native_lib
        lib = get_native_lib()
        if lib is not None and comp.ndim == 2 and comp.dtype == np.float32:
            cc = np.ascontiguousarray(comp, dtype=np.float32)
            sg = np.ascontiguousarray(self._static_grain, dtype=np.float32)
            dg = np.ascontiguousarray(dyn, dtype=np.float32)
            out = np.empty_like(cc)
            if self._drift is not None and self.do_color > 1e-4:
                tilt = np.ascontiguousarray(self._drift.sample(frame_index), dtype=np.float32)
                striping_1d = np.ascontiguousarray(
                    np.cos(2.0 * np.pi * np.arange(cc.shape[1], dtype=np.float32) / 48.0),
                    dtype=np.float32,
                )
                lib.vml_tape_noise_floor_f32(
                    cc.ctypes.data, sg.ctypes.data, dg.ctypes.data,
                    tilt.ctypes.data, striping_1d.ctypes.data,
                    amt, float(amt * 0.5 * self.do_color),
                    out.ctypes.data, cc.shape[0], cc.shape[1],
                )
            else:
                lib.vml_tape_noise_floor_f32(
                    cc.ctypes.data, sg.ctypes.data, dg.ctypes.data,
                    None, None, amt, 0.0,
                    out.ctypes.data, cc.shape[0], cc.shape[1],
                )
            return out

        mixed = 0.45 * self._static_grain + 0.55 * dyn.astype(np.float32)
        out = comp + amt * mixed
        if self._drift is not None and self.do_color > 1e-4:
            tilt = self._drift.sample(frame_index).astype(np.float32)
            striping = np.cos(
                2.0 * np.pi * np.arange(comp.shape[1], dtype=np.float32) / 48.0
            ).astype(np.float32)[None, :]
            out = out + amt * 0.5 * self.do_color * tilt * striping
        return out.astype(np.float32)

    # ------------------------------------------------------------------
    def _apply_binder_loss(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                           frame_index: int, rng: np.random.Generator
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Binder breakdown: mould and shed magnetic coating.

        A tape stored damp is not merely dirty.  The binder absorbs water, swells
        and sheds the magnetic layer, so there are two effects and they compound:

        * **clumped, spatially fixed loss.**  The damaged areas are patches bound
          to the tape surface, not a uniform haze, so they repeat in the same place
          every pass and drift only with tape motion.  Loose magnetic material is
          simply not read.
        * **the colour carrier goes first.**  Chroma is a low-amplitude heterodyned
          carrier riding on the luma; where the coating is thin or gone, that weak
          carrier disappears while the stronger luma FM hangs on longer.  Measured
          on the reference's "mouldy" sample, saturation collapses to 0.053 while
          high-frequency energy rises 7x and 38% of the picture sits near black.

        This is a different failure from `debris` (one loose particle striking the
        tape) and from `surface_damage` (a scratch).  Neither of those removes a
        *fraction of the picture*: measured together at their maxima they moved
        high-frequency energy from 0.09388 to 0.09393 and left the dark fraction at
        zero, so the model could not express this condition at all.
        """
        amt = float(np.clip(self.binder_loss, 0.0, 1.0))
        if amt <= 1e-4:
            return y, i, q
        # The patch field is built lazily.  Returning early when it is missing
        # (as an earlier version did) makes the whole mechanism silently do
        # nothing unless some other stage happened to call `prepare()` first --
        # which is exactly the kind of "feature present but unreachable" failure
        # that has bitten this file more than once.
        if self._static_grain is None or self._binder is None:
            self.prepare()
        field = self._static_grain
        if field is None:
            return y, i, q
        H, W = y.shape[:2]
        # coarse, spatially fixed pattern -> clumps rather than an even wash
        base = lowpass(field, max(2.0, W / 40.0), axis=1)
        base = lowpass(base, 4.0, axis=0)
        drift = self._binder.sample(frame_index) if self._binder is not None \
            else np.zeros_like(base)
        pat = base + 0.55 * drift
        pat = (pat - pat.mean()) / max(1e-6, float(pat.std()))
        # `amt` sets what share of the picture is degraded, and the damage has to
        # be *proportional* rather than total.  Two bugs lived here:
        #
        #  * `np.quantile` on a heavy-tailed field put the threshold so low that
        #    every value above it saturated the near-binary ramp, so 0.2, 0.4 and
        #    0.7 all produced exactly the same 65% of pixels affected -- the
        #    control did nothing.
        #  * the loss weight was applied at full strength, so any affected pixel
        #    lost essentially all of its luma, which cannot show the partial state
        #    the reference material has.
        #
        # A percentile over an explicitly normalised field, plus a bounded weight,
        # gives a control that means what it says and leaves a readable picture at
        # low settings.
        pat = (pat - float(np.median(pat))) / max(1e-6, float(pat.std()))
        cut = float(np.percentile(pat, 100.0 * (1.0 - amt)))
        spread = float(max(0.15, np.percentile(pat, 99.0) - cut))
        mask = np.clip((pat - cut) / spread, 0.0, 1.0).astype(np.float32)
        if float(mask.max()) <= 1e-4:
            return y, i, q
        # The mask has to be *graduated*, not a step.  A coating does not shed all
        # at once: it thins, and the head reads progressively less signal from it.
        # An earlier version produced an effectively binary mask -- the luma-change
        # histogram had no entries at all below 0.20, every affected pixel being
        # completely destroyed -- which meant there was no partially-damaged region
        # in which to see the colour carrier fail before the luma.  The mechanism
        # claimed chroma dies first and gave no way to observe it.
        mask = np.sqrt(mask) * (0.55 + 0.45 * mask)

        # Lost signal reads as dense noise *near black*.  The floor matters: an
        # earlier version used `0.03 + 0.30*|noise|`, which averages about 0.13 and
        # so never crossed the near-black threshold at all -- the dark fraction
        # stayed at 0% against a reference of 37.6%, i.e. the model could not
        # produce a genuinely black dropout however hard it was driven.  Where the
        # coating is gone there is no signal, so the level has to be able to reach
        # the pedestal.
        noise = self._luma_band_noise(rng, y.shape)
        # bounded: even a badly shed coating leaves something readable in the
        # affected areas, and capping the weight is what keeps a low setting
        # a partial loss rather than a hole
        drop = mask * (0.35 + 0.55 * amt)
        y_out = y * (1.0 - drop) + (0.008 + 0.16 * np.abs(noise)) * drop
        # The weak chroma carrier is destroyed well before the luma -- but *over a
        # range*, not instantly.  A kill factor of 2.4 removed colour completely
        # as soon as the damage weight passed 0.42, which meant no partially
        # damaged area ever showed colour loss with luma still present; the
        # reference material has exactly that intermediate state, where chroma is
        # visibly reduced but the picture is still readable.  Chroma therefore
        # decays faster than luma (1.3x its loss) instead of being switched off.
        kill = np.clip(drop * 1.3, 0.0, 1.0)
        i_out = i * (1.0 - kill)
        q_out = q * (1.0 - kill)
        self._binder_last = float(drop.mean())
        return y_out, i_out, q_out

    def _apply_shed_grain(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                          frame_index: int, rng: np.random.Generator
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Coating-shed grain: fine, dense, luminance-only noise.

        Where the magnetic layer has shed, the head reads a rougher surface, so the
        replay signal carries broadband grain.  Two things distinguish this from the
        existing `noise_floor`:

        * it is **strong and fine**.  On the reference "mouldy" panel, high-frequency
          energy is 3x the undamaged panel (0.116 against 0.039) while the picture is
          *also* dark and desaturated.  The model's own noise floor, driven to
          maximum, reached about 0.035 -- a sixth of what is needed -- so the look
          was unreachable however the other controls were set.
        * it is **luminance only**.  A shed coating destroys the weak chroma carrier
          outright, so the grain must not add colour; band-limiting it to luma keeps
          it grey and stops the noise itself from tinting the frame.

        It is the missing companion to the frame-wide desaturation: `chroma_gain`
        supplies the colour loss, `luma_pedestal` the darkness, and this the grain.
        """
        amt = float(np.clip(self.shed_grain, 0.0, 1.0))
        if amt <= 1e-4:
            return y, i, q
        noise = self._luma_band_noise(rng, y.shape)
        y_out = y + (0.02 + 0.85 * amt) * noise
        # a shed coating destroys the weak colour-under carrier
        kill = float(np.clip(0.85 * amt, 0.0, 0.96))
        i_out = i * (1.0 - kill)
        q_out = q * (1.0 - kill)
        return (y_out.astype(np.float32), i_out.astype(np.float32),
                q_out.astype(np.float32))

    def _apply_luma_pedestal(self, y: np.ndarray, i: np.ndarray, q: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pedestal / DC-restoration failure: the luma collapses while chroma stays.

        Luma is a *baseband* level -- its absolute value carries the picture's
        brightness -- so it must be clamped to a reference after AC coupling.
        Chroma is a *modulated amplitude* around whatever that reference is, so it
        survives a reference error untouched.

        That asymmetry is the whole signature, and it is opposite to what a loss of
        recorded signal looks like:

        * **signal lost** (shed coating, dropout, weak RF) takes luma and chroma
          away together, so the picture goes dark *and grey*;
        * **reference lost** (failed clamp, drifted black level) moves the luma
          pedestal down while the chroma keeps its amplitude, so the picture goes
          dark *and violently coloured* -- measured on the S-VHS reference frame,
          saturation rises to 0.94 while the mean falls to 0.09.

        Applying chroma and luma together (which is what `binder_loss` does) leaves
        saturation unchanged, so it cannot produce this look at any strength.  That
        was verified by sweeping it: saturation moved 0.72 -> 0.33 as the loss rose,
        the wrong direction, while collapsing the pedestal alone moves it 0.937 ->
        0.966, the right one.
        """
        offset = float(self.luma_pedestal)
        if abs(offset) < 1e-4:
            return y, i, q
        return (y + offset).astype(np.float32), i, q

    def _apply_drum_speed(self, y: np.ndarray, i: np.ndarray, q: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Head-drum speed error: the cause of "horizontal grain" with line repetition.

        This is a distinct mechanism from anything else in this file, and the
        distinction matters because the symptom looks like noise but is not:

        * tape-transport (capstan) error moves the picture *sideways* -- it changes
          which part of the line is read, so it shows as horizontal jitter;
        * **drum** speed error changes how fast the head sweeps *down* the track.
          A slow drum takes longer to traverse the same tape, so consecutive output
          lines are sampled from very nearly the same point on the tape and the
          picture contains slightly **repeated** lines.  Vertically the picture is
          smoothed or duplicated; horizontally nothing changes.

        That asymmetry is the whole signature: detail survives along the line and
        is smeared between lines, which reads as fine horizontal grain laid over
        the image rather than as random noise.  On a Panasonic shoulder-mount VHS
        or VHS-C camcorder the usual cause is degraded oil in the drum bearing --
        a common failure that, left alone, wears the bearing irreversibly.

        Implemented as a vertical resample at the speed ratio, then held to the
        frame height: a slow drum covers fewer distinct lines than the field has
        slots, so the picture is duplicated into place rather than shortened.  The
        residual fractional offset is carried across frames, which is why the
        effect drifts instead of sitting still.
        """
        ratio = float(np.clip(self.drum_speed, 0.80, 1.05))
        if abs(ratio - 1.0) < 1e-4:
            return y, i, q
        H = y.shape[0]
        # rows are read from a source that is `ratio` times as tall as it should be
        src = np.arange(H, dtype=np.float32) * ratio + self._drum_phase
        src = np.clip(src, 0.0, H - 1.0)
        idx0 = np.floor(src).astype(np.int32)
        idx1 = np.minimum(idx0 + 1, H - 1)
        frac = (src - idx0)[:, None].astype(np.float32)

        def warp(plane: np.ndarray) -> np.ndarray:
            a = plane[idx0]
            b = plane[idx1]
            return (a + (b - a) * frac).astype(np.float32)

        # the offset accumulates slowly: a bearing fault is not stationary
        self._drum_phase = float((self._drum_phase + (1.0 - ratio) * H * 0.02)
                                 % max(1.0, abs(1.0 - ratio) * H + 1e-6))
        return warp(y), warp(i), warp(q)

    def _apply_shuttle_concealment(self, sig: SignalFrame, frame_index: int,
                                   rng: np.random.Generator) -> SignalFrame:
        """
        Rewind / fast-forward residue, in the form the *tape format* actually has.

        The mechanism is common to every helical-scan format: at 2x or 5x the heads
        sweep the tape far faster than it was written, so only part of each track
        arrives intact and the transport substitutes what it can.  What differs is
        the *granularity*, and that is what the eye reads:

        * **analogue line-based** (VHS, Video8, Hi8) -- a continuous waveform per
          track, with no addressable blocks.  Failed regions are whole scanlines,
          in runs, with the run boundaries falling wherever lock was lost.  The
          result is stacked horizontal bands with ragged vertical edges.
        * **digital block-based** (MiniDV) -- a macroblock stream, so concealment
          happens on the 16x16 grid and the result is aligned rectangles.

        Getting this the wrong way round is not a subtle error: a DV block model
        applied to Hi8 asserts a grid the format does not have, and the measured
        row-period spectrum on the Hi8 reference confirms it -- no 16-pixel
        periodicity appears anywhere in that material.
        """
        if self.shuttle_speed <= 0.0 or self.shuttle_fraction <= 1e-4:
            return sig
        H, W = sig.y.shape

        # The donor is a genuinely different frame -- that is the whole point.  It
        # is read from the source rather than held in memory, so a long shuttle
        # does not accumulate a frame cache.
        donor = self._shuttle_donor(frame_index, current_sig=sig)
        if donor is None:
            return sig
        dy = donor.y[:H, :W]
        di = donor.i[:H, :W]
        dq = donor.q[:H, :W]
        if dy.shape != sig.y.shape:
            return sig

        if self.shuttle_block_based:
            shape = macroblock_grid(H, W, block=self.shuttle_block)
            mask = shuttle_stale_mask(
                rng, shape, self.shuttle_fraction,
                cluster=self.shuttle_cluster, carry=self._shuttle_carry,
                carry_rate=self.shuttle_carry)
            self._shuttle_carry = mask
            self._shuttle_last_fraction = float(mask.mean())
            y = apply_stale_macroblocks(sig.y, mask, dy)
            i = apply_stale_macroblocks(sig.i, mask, di)
            q = apply_stale_macroblocks(sig.q, mask, dq)
        elif self.shuttle_guard_bars > 1e-4:
            # When `shuttle_guard_bars` is active, track-crossing bars are handled strictly inside
            # the active region (`active_h = 0.562*H` for PIP split-screen or `H` for full-screen),
            # leaving the lower split-screen half (if PIP) completely untouched!
            self._shuttle_last_fraction = float(self.shuttle_fraction)
            y, i, q = self._apply_cue_review_guard_bars(
                sig.y.copy(), sig.i.copy(), sig.q.copy(), dy, di, dq, frame_index, rng)
        else:
            # mean run length scales with how badly the transport is mistracking
            span = max(1.0, self.shuttle_span)
            mask = line_run_mask(rng, H, self.shuttle_fraction, span_lines=span,
                                 carry=self._shuttle_carry,
                                 carry_rate=self.shuttle_carry)
            self._shuttle_carry = mask
            self._shuttle_last_fraction = float(mask.mean())
            scale_w = (W / 720.0) if W > 760 else 1.0
            disp = line_run_displacement(rng, mask, self.shuttle_wave * scale_w)
            y = apply_stale_lines(sig.y, mask, dy, disp)
            i = apply_stale_lines(sig.i, mask, di, disp)
            q = apply_stale_lines(sig.q, mask, dq, disp)
            if mask.any():
                # Physical helical-scan azimuth-crossing & colour-under unlock in analogue shuttle:
                # 1. At track-crossing transitions (`edge`), the RF envelope drops below the FM
                #    limiter threshold, producing horizontal impulse snow streaks;
                # 2. Within the mistracked band (`mask`), the 743 kHz colour-under carrier loses
                #    burst lock and ACC/colour-killer attenuates chroma while rotating boundary phase.
                edge = mask & (~np.roll(mask, 1) | ~np.roll(mask, -1))
                snow_raw = rng.standard_normal((H, W)).astype(np.float32)
                snow = lowpass(snow_raw, 2.4 * scale_w, axis=1)
                snow_env = (0.14 * edge[:, None].astype(np.float32)
                            + 0.045 * mask[:, None].astype(np.float32))
                y = np.clip(y * (1.0 - 0.18 * edge[:, None].astype(np.float32))
                            + snow * snow_env + 0.025 * edge[:, None].astype(np.float32), 0.0, 1.2)
                # Attenuate chroma in azimuth-crossed lines and slip burst phase on boundary rows
                c_atten = np.where(mask[:, None], 0.28, 1.0).astype(np.float32)
                theta = (rng.uniform(-1.2, 1.2, size=(H, 1)).astype(np.float32)
                         * edge[:, None].astype(np.float32))
                cs, sn = np.cos(theta), np.sin(theta)
                i_rot = i * cs - q * sn
                q_rot = i * sn + q * cs
                i = i_rot * c_atten
                q = q_rot * c_atten

        return SignalFrame(y.astype(np.float32), i.astype(np.float32),
                           q.astype(np.float32))

    def _apply_cue_review_guard_bars(
        self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
        dy: np.ndarray, di: np.ndarray, dq: np.ndarray,
        frame_index: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Physical helical-scan Cue/Review (multispeed shuttle search) guard-band &
        azimuth-null noise bars, calibrated against 1992 VHS Cue/Review search reference:

        1. **Helical Track-Crossing Envelope Collapse**:
           At transport speed ratio N, the rotary video head crosses |N - 1| recorded
           slant tracks per field scan.  Whenever the head straddles the guard band or
           the adjacent opposite-azimuth track, the RF playback envelope E_RF(y) falls
           below the FM limiter threshold (~20 dB SNR).
        2. **Double-Peak FM Limiter Bar + H-Sync Clamp Trench**:
           Inside each track-crossing zone, the FM demodulator clips to a flat grey-white
           self-oscillation plateau (`shuttle_bar_luma`, measured L=0.562 in the 5x full-screen
           shot at t=19.5s and L=0.750 in the 9x PIP shot at t=6.0s), followed by a sharp
           horizontal-sync voltage undershoot trench (`L ~ 0.25`) with intermittent black
           sync dashes and a secondary azimuth rebound bar.
        3. **Donor-Track Splicing & H-AFC Shear**:
           Adjacent track segments between bars come from successive tape frames (`dy, di, dq`),
           and loss of horizontal sync pulses across the bar drives a progressive rightward
           sampling-phase shear (+12..+48 px scaled to raster width) on the exiting lines.
        """
        amt = float(np.clip(self.shuttle_guard_bars, 0.0, 1.0))
        H, W = y.shape[:2]
        scale_w = W / 1440.0
        scale_h = H / 1080.0

        # Active vertical region: full screen (H) or upper DVE split-screen window (0.562 * H)
        active_h = int(round(0.562 * H)) if self.shuttle_pip_bars else H
        if active_h < 16:
            return y, i, q

        # Number of primary track-crossing bars:
        #   - Full-screen 5x Cue search (t=19.5s): |5 - 1| = 4 primary bars, period = 281.0 px (at 1080p),
        #     scrolling downward at 68.0 px/frame (4-frame cyclic period).
        #   - Upper PIP 9x Cue search (t=6.0s): 4.68 primary bars (period = 129.3 px at 1080p) + azimuth
        #     mid-cycle sub-bars = 8..9 visible horizontal bars in the upper 605 px window, scrolling
        #     downward at 21.0 px/frame (6-frame cyclic period).
        if self.shuttle_pip_bars:
            period_px = max(24.0, 129.0 * scale_h)
            scroll_px = 21.0 * scale_h
            phase0 = 96.0 * scale_h
        else:
            n_bars = max(1.0, abs(float(self.shuttle_speed) - 1.0))
            period_px = max(36.0, (281.0 * (4.0 / n_bars)) * scale_h)
            scroll_px = 68.0 * scale_h
            phase0 = 74.0 * scale_h

        direction = -1.0 if self.shuttle_reverse else 1.0
        rows = np.arange(active_h, dtype=np.float32)
        cols = np.arange(W, dtype=np.float32)[None, :]

        # Normalized phase u in [-0.5, 0.5) relative to the nearest primary bar center
        u = ((rows - phase0 - direction * frame_index * scroll_px) / period_px) % 1.0
        u_signed = np.where(u >= 0.5, u - 1.0, u)

        # Alternate track segments (where u > 0) blend in the donor track so adjacent inter-bar
        # zones carry genuinely different frames from the tape pass
        alt_track = np.clip((u - 0.16) / 0.04, 0.0, 1.0) * np.clip((0.84 - u) / 0.04, 0.0, 1.0)
        alt_w = (alt_track * 0.65 * amt)[:, None].astype(np.float32)
        y_act = y[:active_h] * (1.0 - alt_w) + dy[:active_h] * alt_w
        i_act = i[:active_h] * (1.0 - alt_w) + di[:active_h] * alt_w
        q_act = q[:active_h] * (1.0 - alt_w) + dq[:active_h] * alt_w

        # H-AFC horizontal shear: ramps up across the bar and decays in the recovery zone below it
        if self.shuttle_pip_bars:
            shear_profile = (np.exp(-0.5 * ((u_signed - 0.08) / 0.09) ** 2) * 44.0
                             + np.exp(-0.5 * ((u_signed + 0.02) / 0.06) ** 2) * 14.0) * scale_w * amt
        else:
            shear_profile = (np.exp(-0.5 * ((u_signed - 0.09) / 0.07) ** 2) * 26.0
                             + np.exp(-0.5 * ((u_signed + 0.02) / 0.05) ** 2) * 10.0) * scale_w * amt
        shear_jitter = rng.standard_normal(active_h).astype(np.float32) * (1.8 * scale_w * amt)
        tot_shear = (shear_profile + shear_jitter).astype(np.float32)
        y_act = horizontal_shift(y_act, tot_shear)
        i_act = horizontal_shift(i_act, tot_shear)
        q_act = horizontal_shift(q_act, tot_shear)

        # Primary FM-limiter plateau envelope, H-sync clamp trench, and secondary rebound bar
        if self.shuttle_pip_bars:
            # Calibrated to seg3_f180_6p0s (period=129.3px):
            #   main bright plateau around u_signed=0.0 (half-width ~0.075),
            #   secondary trailing plateau around u_signed=+0.15,
            #   plus opposite-azimuth mid-track sub-bar around |u_signed| ~ 0.42
            main_plat = np.exp(-0.5 * (u_signed / 0.065) ** 4)
            sec_plat = 0.42 * np.exp(-0.5 * ((u_signed - 0.155) / 0.055) ** 2)
            mid_sub = 0.36 * np.exp(-0.5 * ((np.abs(u_signed) - 0.36) / 0.040) ** 2)
            trench_y = np.exp(-0.5 * ((u_signed - 0.088) / 0.018) ** 2)
        else:
            # Calibrated to seg6_f585_19p5s (period=281.0px):
            #   main flat plateau in u_signed in [-0.075, +0.055] (~38px thick),
            #   sharp H-sync undershoot trench at u_signed = +0.073 (~8px thick),
            #   secondary rebound bar at u_signed = +0.102 (~14px thick),
            #   plus thin pre-bar azimuth-entry line at u_signed = -0.225
            main_plat = np.clip((0.068 - np.abs(u_signed + 0.008)) / 0.022, 0.0, 1.0)
            sec_plat = 0.92 * np.exp(-0.5 * ((u_signed - 0.102) / 0.020) ** 2)
            mid_sub = 0.48 * np.exp(-0.5 * ((u_signed + 0.225) / 0.010) ** 2)
            trench_y = np.exp(-0.5 * ((u_signed - 0.073) / 0.011) ** 2)

        bar_env = np.clip(main_plat + sec_plat + mid_sub, 0.0, 1.0)[:, None].astype(np.float32) * amt

        # Correlated FM limiter snow inside the bar
        snow_raw = rng.standard_normal((active_h, W)).astype(np.float32)
        snow_lp = lowpass(snow_raw, max(0.8, 3.2 * scale_w), axis=1)
        bar_level = float(np.clip(self.shuttle_bar_luma, 0.20, 0.95))
        bar_luma_field = np.clip(bar_level + 0.022 * snow_lp, 0.0, 1.0)

        # Blend active picture into the FM limiter grey-white plateau
        y_act = y_act * (1.0 - bar_env) + bar_luma_field * bar_env

        # Horizontal sync-pulse dark-dash notch inside the trench (u_signed ~ +0.073..0.088):
        # During the guard-band transition, stripped H-sync pulses pull the black-level clamp
        # down to L ~ 0.25 across long horizontal dash spans (x in [0.18W, 0.92W] with gaps)
        xn = cols / max(1.0, W - 1.0)
        dash_gate = (
            np.clip((xn - 0.14) / 0.02, 0.0, 1.0) * np.clip((0.92 - xn) / 0.02, 0.0, 1.0)
            * np.where(np.cos(2.0 * np.pi * (xn * 3.5 + frame_index * 0.17)) > -0.55, 1.0, 0.15)
        ).astype(np.float32)
        trench_mask = (trench_y[:, None] * dash_gate * amt).astype(np.float32)
        trench_luma = 0.245 + 0.035 * snow_lp
        y_act = y_act * (1.0 - 0.85 * trench_mask) + trench_luma * (0.85 * trench_mask)

        # Right-hand HBI blanking border (x > 0.978W) observed in the MV capture
        hbi_mask = np.clip((xn - 0.978) / 0.008, 0.0, 1.0).astype(np.float32) * amt
        y_act = y_act * (1.0 - 0.88 * hbi_mask)

        # ACC colour-killer: within the guard-band bar & trench, 629/743 kHz colour-under
        # burst lock is lost, so chroma is strongly muted (leaving a neutral/cool grey-white bar)
        c_kill = np.clip((bar_env + 0.8 * trench_mask) * 1.15, 0.0, 0.96)
        i_act = i_act * (1.0 - c_kill) - (0.012 * bar_env if self.shuttle_pip_bars else 0.0)
        q_act = q_act * (1.0 - c_kill) + (0.008 * bar_env if self.shuttle_pip_bars else 0.0)

        y_out = y.copy()
        i_out = i.copy()
        q_out = q.copy()
        y_out[:active_h] = np.clip(y_act, 0.0, 1.0)
        i_out[:active_h] = i_act
        q_out[:active_h] = q_act
        return y_out, i_out, q_out

    def _remember_frame(self, frame_index: int, y: np.ndarray,
                        i: np.ndarray, q: np.ndarray) -> None:
        """Keep the last few processed frames, as a deck's buffer would."""
        if not hasattr(self, "_frame_ring"):
            from collections import OrderedDict
            self._frame_ring = OrderedDict()
            self._ring_limit = int(max(4, min(24, self.ctx.fps * 0.3)))
        entry = (y.astype(np.float32, copy=True),
                 i.astype(np.float32, copy=True),
                 q.astype(np.float32, copy=True))
        self._frame_ring[int(frame_index)] = entry
        self._frame_ring.move_to_end(int(frame_index))
        while len(self._frame_ring) > self._ring_limit:
            self._frame_ring.popitem(last=False)

    def _shuttle_donor(self, frame_index: int,
                       current_sig: SignalFrame | None = None) -> SignalFrame | None:
        """
        A frame the transport passed earlier (fast-forward) or later (rewind).

        Taken from this stage's own recent output, *not* from the raw clip.  That
        distinction is the whole correctness of the effect: a deck substitutes from
        decoded frames it already holds, so the replacement carries the same
        recording and playback degradation as everything around it.  Fetching from
        the source instead inserted pristine decoded pixels, and measurement showed
        the patches came out **5x sharper** than the surrounding picture -- the
        band-limited look of the rest of the frame is exactly what the substituted
        content must share.

        Falling back to the source only when no buffered frame exists yet (the first
        few frames of a run), which is also what a buffer that has not filled would
        do.
        """
        ring = getattr(self, "_frame_ring", None)
        if ring:
            keys = list(ring)
            span = max(2, int(round(self.shuttle_speed * 4.0)))
            # prefer frames a plausible distance back, as shuttle skips whole spans
            pick = keys[-1 - min(len(keys) - 1, span)] if len(keys) > 1 else keys[-1]
            y, i, q = ring[pick]
            return SignalFrame(y.copy(), i.copy(), q.copy())

        src = getattr(self.ctx, "source", None)
        if src is None:
            return None
        span = max(2, int(round(self.shuttle_speed * 12.0)))
        direction = -1 if self.shuttle_reverse else 1
        want = int(frame_index) - direction * span
        last = max(0, int(getattr(src, "frame_count", 1)) - 1)
        want = int(np.clip(want, 0, last))
        if want == int(frame_index) and current_sig is not None:
            return SignalFrame(current_sig.y.copy(), current_sig.i.copy(), current_sig.q.copy())
        try:
            rgb = np.clip(src.frame(want), 0.0, 1.0).astype(np.float32)
            from .camera import CameraStage
            cam_buf = FrameBuffer(rgb, None, want, want / max(1e-6, self.ctx.fps),
                                  self.ctx.std, {"field": want % 2})
            rgb = CameraStage(self.ctx).apply(cam_buf).rgb
        except Exception:
            if current_sig is not None:
                return SignalFrame(current_sig.y.copy(), current_sig.i.copy(), current_sig.q.copy())
            return None
        # Even the fallback should not be pristine: the encoder's bandwidth limit
        # applies to anything this deck can produce.
        lim = self.ctx.layer("format").get("luma_bandwidth", 3.0)
        sig = SignalFrame.from_rgb(rgb, self.ctx.std)
        y = lowpass(sig.y, self.ctx.std.pixels_for_bandwidth(float(lim) * 1e6,
                                                            sig.y.shape[1]), axis=1)
        return SignalFrame(y, sig.i, sig.q)

    def _apply_digital_damage(self, buf: FrameBuffer, frame_index: int,
                              rng: np.random.Generator) -> FrameBuffer:
        """
        MiniDV medium damage: block concealment plus small transport jitter.

        A DV deck has no colour subcarrier, so none of the analog mechanisms
        apply.  What it does have is a fixed bit rate and an error-correction
        scheme: when a read error exceeds what the ECC can repair, the decoder
        *conceals* the macroblock by repeating a neighbour.  The result is a
        clean, block-shaped patch -- and dropout events here are discrete in the
        same sense as the analog ones, which keeps the defect timeline meaningful
        across both branches.
        """
        sig = buf.meta.get("signal")
        if sig is None:
            return buf
        # shuttle concealment runs *before* the discrete damage: it is a property
        # of how the tape is being read, not an event at a point on the tape
        sig = self._apply_shuttle_concealment(sig, frame_index, rng)
        H, W = sig.y.shape
        plan = self.plan
        phase = self._line_phase_for(frame_index)
        # MiniDV's transport is far more stable than VHS: scale the jitter down
        jitter = (phase * 0.25).astype(np.float32) if np.any(np.abs(phase) > 1e-4) else None
        y, i, q = sig.y, sig.i, sig.q
        if jitter is not None:
            y = horizontal_shift(y, jitter)
            i = horizontal_shift(i, jitter)
            q = horizontal_shift(q, jitter)

        # conceal blocks where a dropout event lands
        BLK = 8
        concealed = 0
        if plan.dropouts and self.do_rate > 0.0:
            for ev in plan.dropouts:
                if frame_index < ev.start_frame or frame_index >= ev.start_frame + ev.frames:
                    continue
                # snap the affected region to the macroblock grid
                y0 = (ev.row // BLK) * BLK
                y1 = min(H, y0 + max(BLK, ((ev.rows + BLK - 1) // BLK) * BLK))
                x0 = (ev.col // BLK) * BLK
                x1 = min(W, x0 + max(BLK, ((ev.cols + BLK - 1) // BLK) * BLK))
                if y1 <= y0 or x1 <= x0:
                    continue
                sy0 = max(0, y0 - BLK)
                y[y0:y1, x0:x1] = y[sy0:sy0 + (y1 - y0), x0:x1]
                i[y0:y1, x0:x1] = i[sy0:sy0 + (y1 - y0), x0:x1]
                q[y0:y1, x0:x1] = q[sy0:sy0 + (y1 - y0), x0:x1]
                concealed += 1

        sig_dv = SignalFrame(y.astype(np.float32), i.astype(np.float32), q.astype(np.float32))
        edit_rep = None
        if self.edit_controller_spec.is_active():
            from ..codecs.edit_controller_nle import apply_edit_controller
            b_yiq = buf.meta["bus_b_yiq"] if "bus_b_yiq" in buf.meta else None
            ed_rgb, edit_rep = apply_edit_controller(
                sig_dv.to_rgb(), self.edit_controller_spec, std=buf.std,
                frame_index=frame_index, fps=self.ctx.fps, seed=self.ctx.seed,
                bus_b_yiq=b_yiq,
            )
            sig_dv = SignalFrame.from_rgb(ed_rgb, buf.std)
        out = FrameBuffer(sig_dv.to_rgb() if edit_rep is not None else buf.rgb, None, buf.index, buf.time, buf.std, dict(buf.meta))
        out.meta["signal"] = sig_dv
        if edit_rep is not None:
            out.meta["edit_controller"] = edit_rep
        out.meta["format"] = "dv"
        out.meta["concealed_blocks"] = concealed
        out.meta["line_phase"] = phase
        return out

    def _apply_cosmag_substrate_failures(
        self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
        frame_index: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Physical model of archival VHS substrate & magnetic-layer failure modes
        (calibrated for COSMAG E-180 PAL/SP/LP 180/360-min tape, 258 min recovered / 61 min failed):

        1. `sp_lp_mix` (SP/LP 混合录放磁头方位角与窄轨切换):
           4-head drum (Sharp VC-A62DT: 2x SP 58 um + 2x LP 29 um heads) transitioning between SP and LP
           segments or reading LP tracks with narrower guard bands: dynamic 4-head envelope modulation,
           -6 dB LP replay SNR loss, and periodic azimuth/burst phase flutter.
        2. `demagnetization` (消磁):
           Decay of remanent magnetization M_r across the tape width/length. By Wallace's spacing/remanence
           law, high frequencies (FM luma carrier at 3.8..4.8 MHz) and the weak 629 kHz colour-under carrier
           fall below the FM limiter and ACC colour-killer thresholds first, producing soft low-contrast luma,
           dense FM limiter snow, and severe desaturation.
        3. `base_cracking` (带基龟裂):
           Brittle PET substrate micro-fissures (`crackle` network) oriented along the ~5.9 deg helical scan
           angle. When the 4 rotary video heads sweep across a fissure, head-to-tape spacing `d` jumps by
           several microns (`54.6 * d / lambda` dB loss), causing sharp diagonal/transverse crack lines with
           bipolar FM unlock spikes (black-to-white-to-black) and local horizontal line-clock phase slips.
        4. `base_breakage` (带基破损):
           Physical tear, edge curl, or splice damage on the 1/2-inch tape edge (where the 1x linear audio
           head and 1x 25 Hz control-track CTL head sit on the Sharp VC-A62DT). Loss of the CTL pulse
           destabilizes the capstan servo across a multi-line vertical tear zone, producing severe horizontal
           sync shear, vertical framing jump, and a wide raw FM carrier snow band.
        """
        H, W = y.shape[:2]
        scale_w = W / 720.0
        scale_h = H / 576.0
        rows = np.arange(H, dtype=np.float32)[:, None]
        cols = np.arange(W, dtype=np.float32)[None, :]
        yn = rows / max(1.0, H - 1.0)
        xn = cols / max(1.0, W - 1.0)

        # --- A. PAL SP/LP Mixed-Mode 4-Head Envelope & Azimuth Flutter (`sp_lp_mix`) ---
        mix_amt = float(np.clip(self.sp_lp_mix, 0.0, 1.0))
        if mix_amt > 1e-4:
            # 4-head drum switching between SP (49 um track) and LP (24.5 um track) or boundary zone
            lp_env = 0.5 + 0.5 * np.sin(2.0 * np.pi * (yn * 1.5 - frame_index * 0.09))
            eff_lp = mix_amt * (0.45 + 0.55 * lp_env)
            # LP narrows luma & chroma bandwidth slightly and raises FM carrier noise by ~4-6 dB
            y_soft = lowpass(y, max(0.6, 1.35 * scale_w), axis=1)
            y = y * (1.0 - 0.32 * eff_lp) + y_soft * (0.32 * eff_lp)
            fm_n = self._luma_band_noise(rng, (H, W))
            y = y + (0.028 * mix_amt) * eff_lp * fm_n
            # Alternating 629 kHz colour-under phase flutter across LP tracks
            ph_flut = (0.28 * mix_amt * np.sin(2.0 * np.pi * (rows[:, 0] / max(4.0, 18.0 * scale_h) + frame_index * 0.23))).astype(np.float32)[:, None]
            cs, sn = np.cos(ph_flut), np.sin(ph_flut)
            i, q = i * cs - q * sn, i * sn + q * cs

        # --- B. Demagnetization (`demagnetization`: 消磁) ---
        demag = float(np.clip(self.demagnetization, 0.0, 1.0))
        if demag > 1e-4:
            # Spatially varying remanence decay wave across the frame (worse in bands where coercivity faded)
            rem_wave = (0.55 + 0.45 * np.sin(2.0 * np.pi * (yn * 1.2 + frame_index * 0.05))
                        * np.cos(2.0 * np.pi * (xn * 0.8 - 0.15)))
            d_mask = np.clip(demag * (0.65 + 0.55 * rem_wave), 0.0, 1.0).astype(np.float32)
            # High-frequency Wallace loss: luma loses sharp transitions and compresses toward mid-grey
            y_lp = lowpass(y, max(0.9, (1.8 + 2.8 * demag) * scale_w), axis=1)
            y = y * (1.0 - 0.65 * d_mask) + (y_lp * (1.0 - 0.35 * demag) + 0.12 * demag) * (0.65 * d_mask)
            # FM limiter threshold snow when carrier envelope drops below knee
            lim_snow = self._luma_band_noise(rng, (H, W))
            y = y + (0.04 + 0.22 * demag) * (d_mask ** 1.3) * lim_snow
            # 629 kHz colour-under carrier is 12-16 dB below FM luma, so ACC colour killer attenuates chroma first
            c_kill = np.clip(d_mask * (1.25 + 0.35 * demag), 0.0, 0.98)
            i = i * (1.0 - c_kill)
            q = q * (1.0 - c_kill)

        # --- C. PET Substrate Brittle Cracking (`base_cracking`: 带基龟裂) ---
        crack = float(np.clip(self.base_cracking, 0.0, 1.0))
        if crack > 1e-4:
            # Helical scan tracks cross brittle PET substrate fissures at shallow angles (~3..7 deg),
            # creating a branching network of diagonal/horizontal crackle lines with spacing-loss FM spikes
            n_cracks = int(round(2 + 6 * crack))
            crack_rng = np.random.default_rng(derive_seed(self.ctx.seed, "tape", "base_cracking", frame_index // 2))
            crack_field = np.zeros((H, W), dtype=np.float32)
            crack_disp = np.zeros(H, dtype=np.float32)
            for c_idx in range(n_cracks):
                y0_c = float(crack_rng.uniform(0.08, 0.92)) * H
                slope = float(crack_rng.uniform(-0.14, 0.14)) * (H / max(1.0, W))
                bend = float(crack_rng.uniform(-0.00018, 0.00018)) / max(0.25, scale_w)
                x_center = float(crack_rng.uniform(0.2, 0.8)) * W
                # Branching crack trajectory y_crack(x)
                y_traj = y0_c + slope * (cols - x_center) + bend * ((cols - x_center) ** 2)
                dist_y = rows - y_traj
                width_c = max(0.9, float(crack_rng.uniform(1.1, 2.6)) * scale_h)
                # Bipolar FM unlock profile: dark spacing-loss trough + bright limiter-unlock white spike
                core_c = np.exp(-0.5 * (dist_y / width_c) ** 2)
                edge_c = -np.exp(-0.5 * ((dist_y - 1.4 * width_c) / width_c) ** 2) * 0.55
                # Intermittent lateral modulation along the crack fissure
                x_mod = np.clip(0.65 + 0.55 * np.sin(cols / max(4.0, 19.0 * scale_w) + c_idx * 1.7), 0.0, 1.0)
                crack_field += (core_c + edge_c) * x_mod
                # Per-line H-sync phase slip where the crack crosses the scanline
                row_hit = np.exp(-0.5 * ((rows[:, 0] - y0_c) / max(1.5, 4.0 * scale_h)) ** 2)
                crack_disp += row_hit * float(crack_rng.choice([-1.0, 1.0])) * (4.0 + 14.0 * crack) * scale_w

            if np.any(np.abs(crack_disp) > 1e-3):
                y = horizontal_shift(y, crack_disp)
                i = horizontal_shift(i, crack_disp)
                q = horizontal_shift(q, crack_disp)
            crack_w = np.clip(np.abs(crack_field) * crack * 1.15, 0.0, 1.0)
            fm_spike = (0.78 + 0.25 * self._luma_band_noise(rng, (H, W))) * np.clip(crack_field, 0.0, 1.0)
            fm_trough = 0.04 * np.clip(-crack_field, 0.0, 1.0)
            y = y * (1.0 - crack_w) + (fm_spike + fm_trough) * crack * 1.10
            i = i * (1.0 - np.clip(crack_w * 1.35, 0.0, 1.0))
            q = q * (1.0 - np.clip(crack_w * 1.35, 0.0, 1.0))

        # --- D. Tape Edge / Splice Physical Breakage & CTL Loss (`base_breakage`: 带基破损) ---
        brk = float(np.clip(self.base_breakage, 0.0, 1.0))
        if brk > 1e-4:
            # A physical tear or edge fold spans a horizontal band of scanlines (where the helical scan sweeps
            # across the damaged tape section) and knocks out the 1x linear CTL control track pulse.
            brk_rng = np.random.default_rng(derive_seed(self.ctx.seed, "tape", "base_breakage", frame_index))
            band_center = (0.32 + 0.18 * np.sin(frame_index * 0.31)) * H
            band_half = max(4.0, (10.0 + 28.0 * brk) * scale_h)
            dy_b = rows[:, 0] - band_center
            brk_env = np.exp(-0.5 * (dy_b / band_half) ** 2).astype(np.float32)
            # Severe horizontal shear (H-AFC unlock across the torn edge)
            shear_px = (brk_env * (12.0 + 42.0 * brk) * scale_w
                        * np.sin(rows[:, 0] / max(1.5, 3.2 * scale_h) + frame_index * 1.1)
                        + brk_env * brk_rng.standard_normal(H).astype(np.float32) * (6.0 * brk * scale_w))
            y = horizontal_shift(y, shear_px)
            i = horizontal_shift(i, shear_px)
            q = horizontal_shift(q, shear_px)
            # Inside the torn core, head-to-tape contact is lost completely -> raw FM snow + black tear border
            tear_core = np.clip((band_half * 0.65 - np.abs(dy_b)) / max(1.0, 2.5 * scale_h), 0.0, 1.0)[:, None] * brk
            raw_snow = 0.18 + 0.62 * np.abs(self._luma_band_noise(rng, (H, W)))
            y = y * (1.0 - 0.88 * tear_core) + raw_snow * (0.88 * tear_core)
            i = i * (1.0 - np.clip(tear_core * 1.5, 0.0, 1.0))
            q = q * (1.0 - np.clip(tear_core * 1.5, 0.0, 1.0))

        return y.astype(np.float32), i.astype(np.float32), q.astype(np.float32)

    def _apply_generation_iteration(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                                    frame_index: int, gen_index: int = 1
                                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single-generation dubbing iteration helper used by GraphExecutor (`vhs_dub_loop` / `vhs_gen`)."""
        saved_gen = getattr(self, "generation_passes", 1)
        try:
            self.generation_passes = 2
            self.generation_count = 2
            f_idx = int(frame_index) + int(gen_index) * 19
            rng = np.random.default_rng(derive_seed(self.ctx.seed, "tape/gen_iter", f_idx))
            return self._apply_generation_loss(y, i, q, f_idx, rng)
        finally:
            self.generation_passes = saved_gen
            self.generation_count = saved_gen

    def _apply_generation_loss(self, y: np.ndarray, i: np.ndarray, q: np.ndarray,
                               frame_index: int, rng: np.random.Generator
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Multi-generation VHS dubbing (Generation Loss), calibrated against the blogger's
        20-generation VHS dubbing experiment (`录像带的图像耗损(上个时代的影像损失).37741528802.mp4`).

        When a VHS tape is dubbed from one VCR to another repeatedly over N generations:
          1. **Cumulative Colour-Under Delay & 1H Comb Downward Bleed**:
             Each generation's 629 kHz heterodyne filter and 1H glass delay-line comb filter
             shifts chroma rightward (~1.35 px/gen) and downward (~1.15 lines/gen) while
             recursively smearing vertical colour and narrowing horizontal chroma bandwidth.
          2. **Cascaded FM Pre-Emphasis / De-Emphasis Overshoot & Dark Ringing Halos**:
             Each record/playback pass applies high-frequency aperture boost and non-linear
             FM clipping, creating dark undershoot trenches and bright overshoot halos on the
             right side of vertical transitions (measured `hf_x` rising from 0.0049 at Gen 1
             to 0.0128 at Gen 18 even as fine resolution collapses).
          3. **AGC Pedestal Lift & Highlight Yellow-Green Compression**:
             Repeated AGC clamping on degraded sync tips lifts the black pedestal (`p05` rises
             from 0.160 to 0.175) while compressing highlight saturation and shifting highlights
             toward yellow-green (`R/G` rises from 0.472 to 0.623).
          4. **Compounded Timebase Walk, Stacked Head-Switching & Right HBI Clamp Bar**:
             Each deck adds its own capstan/drum phase walk, stacks another slightly offset
             bottom head-switching tear, and widens the right-hand HBI blanking noise bar (`x > 0.98W`).
          5. **Sync-Threshold Collapse & Colour-Killer Mute (~Gen 19-20, `t=60.5s`)**:
             Once cumulative sync-tip rounding and timebase jitter cross the AFC/burst-gate
             lock threshold, the colour-killer mutes chroma (`sat -> 0.015`) and the picture
             breaks into high-contrast diagonal FM herringbones and horizontal sync tearing
             with the subject's silhouette faintly visible underneath.
        """
        passes = int(getattr(self, "generation_passes", 1))
        collapse_override = float(getattr(self, "gen_sync_collapse", 0.0))
        if passes <= 1 and collapse_override <= 1e-4:
            return y, i, q

        H, W = y.shape[:2]
        scale_w = W / 720.0
        scale_h = H / 480.0
        extra_gens = max(0, passes - 1)
        bleed_k = float(getattr(self, "gen_chroma_bleed", 0.85))
        ring_k = float(getattr(self, "gen_ringing", 0.85))

        y_out = y.astype(np.float32, copy=True)
        i_out = i.astype(np.float32, copy=True)
        q_out = q.astype(np.float32, copy=True)

        if extra_gens > 0:
            # --- 1. Cumulative Chroma Right-Shift, Downward 1H Comb Bleed & Bandwidth Loss ---
            eff_n = float(extra_gens)
            dx_chroma = eff_n * 1.25 * bleed_k * scale_w
            dy_chroma = eff_n * 1.05 * bleed_k * scale_h
            if abs(dx_chroma) > 1e-3:
                sh_c = np.full(H, dx_chroma, dtype=np.float32)
                i_out = horizontal_shift(i_out, sh_c)
                q_out = horizontal_shift(q_out, sh_c)
            if abs(dy_chroma) > 1e-3:
                i_out = vertical_shift(i_out, dy_chroma)
                q_out = vertical_shift(q_out, dy_chroma)

            # Recursive vertical 1H delay-line IIR bleed + horizontal colour-under narrowing
            c_sigma_x = max(0.8, (1.4 + 0.65 * np.sqrt(eff_n) * bleed_k) * scale_w)
            c_sigma_y = max(0.5, (0.6 + 0.45 * np.sqrt(eff_n) * bleed_k) * scale_h)
            i_out = lowpass(lowpass(i_out, c_sigma_x, axis=1), c_sigma_y, axis=0)
            q_out = lowpass(lowpass(q_out, c_sigma_x, axis=1), c_sigma_y, axis=0)

            # ACC saturation decay + slight warm-yellow phase drift over generations
            sat_decay = float((0.945 ** (eff_n * max(0.3, bleed_k))))
            phi_drift = float(np.deg2rad(min(28.0, eff_n * 1.15 * bleed_k)))
            cs, sn = np.cos(phi_drift), np.sin(phi_drift)
            i_rot = (i_out * cs - q_out * sn) * sat_decay
            q_rot = (i_out * sn + q_out * cs) * sat_decay
            # Subtle yellow-green highlight/midtone bias measured in Gen 4 -> Gen 18
            warm_bias = min(0.055, 0.0032 * eff_n * bleed_k)
            i_out = (i_rot + warm_bias * 0.45).astype(np.float32)
            q_out = (q_rot - warm_bias * 0.65).astype(np.float32)

            # --- 2. Cascaded FM Luma Bandwidth Loss + Multi-Generation Aperture Ringing ---
            luma_blur_sig = max(0.45, (0.55 + 0.24 * np.sqrt(eff_n)) * scale_w)
            y_base = lowpass(y_out, luma_blur_sig, axis=1)
            if ring_k > 1e-4:
                # Delayed second-derivative ringing (dark undershoot + bright overshoot on right of edges)
                d1 = y_base - np.roll(y_base, 2, axis=1)
                d1[:, :2] = 0.0
                d2 = np.roll(d1, 2, axis=1) - 0.65 * np.roll(d1, 5, axis=1)
                d2[:, :5] = 0.0
                ring_amp = min(0.48, (0.045 * np.sqrt(eff_n) + 0.012 * eff_n) * ring_k)
                y_base = y_base + ring_amp * d1 - (0.72 * ring_amp) * d2

            # --- 3. AGC Black-Pedestal Lift & Highlight Soft Clipping ---
            ped_lift = min(0.085, 0.0042 * eff_n * max(0.4, ring_k))
            hi_comp = max(0.76, 1.0 - 0.011 * eff_n)
            y_out = np.clip(ped_lift + y_base * hi_comp, 0.0, 1.05).astype(np.float32)

            # --- 4. Compounded Timebase S-Curve Walk, Stacked Head-Switching & Right HBI Bar ---
            rows = np.arange(H, dtype=np.float32)
            yn = rows / max(1.0, H - 1.0)
            s_walk = (
                np.sin(2.0 * np.pi * yn * 1.5 + frame_index * 0.31)
                + 0.6 * np.sin(2.0 * np.pi * yn * 3.7 - frame_index * 0.53)
            )
            gen_jitter = (
                s_walk * (0.38 * np.sqrt(eff_n) * scale_w)
                + rng.standard_normal(H).astype(np.float32) * (0.22 * np.sqrt(eff_n) * scale_w)
            ).astype(np.float32)
            y_out = horizontal_shift(y_out, gen_jitter)
            i_out = horizontal_shift(i_out, gen_jitter)
            q_out = horizontal_shift(q_out, gen_jitter)

            # Stacked multi-generation head-switching band at bottom (~2% to 5% height)
            hs_rows = min(int(round((8 + min(22, extra_gens * 1.2)) * scale_h)), H // 5)
            if hs_rows > 1:
                hs_slice = slice(H - hs_rows, H)
                hs_ramp = np.linspace(0.2, 1.0, hs_rows, dtype=np.float32)[:, None]
                hs_noise = rng.standard_normal((hs_rows, W)).astype(np.float32) * 0.14
                y_out[hs_slice] = np.clip(
                    y_out[hs_slice] * (1.0 - 0.45 * hs_ramp) + (0.62 + hs_noise) * (0.45 * hs_ramp),
                    0.0, 1.0,
                )
                hs_shift = (hs_ramp[:, 0] * (2.5 + 0.9 * min(12.0, eff_n)) * scale_w).astype(np.float32)
                y_out[hs_slice] = horizontal_shift(y_out[hs_slice], hs_shift)

            # Compounded right-edge HBI blanking/clamp bar (x > 0.98W, measured in video frames)
            bar_w = max(2, int(round((6 + min(14, extra_gens * 0.7)) * scale_w)))
            if bar_w < W // 6:
                xx_b = np.linspace(0.0, 1.0, bar_w, dtype=np.float32)[None, :]
                bar_noise = rng.standard_normal((H, bar_w)).astype(np.float32) * 0.08
                bar_luma = np.clip(0.58 + 0.22 * xx_b + bar_noise, 0.15, 0.96).astype(np.float32)
                mix_b = np.clip(xx_b * min(0.95, 0.25 + 0.05 * eff_n), 0.0, 0.95)
                y_out[:, -bar_w:] = y_out[:, -bar_w:] * (1.0 - mix_b) + bar_luma * mix_b
                i_out[:, -bar_w:] *= (1.0 - mix_b)
                q_out[:, -bar_w:] *= (1.0 - mix_b)

            # Compounded FM tape snow & generational chroma speckle
            gen_snow = (0.008 * np.sqrt(eff_n) + 0.0022 * eff_n) * self._luma_band_noise(rng, (H, W))
            y_out = np.clip(y_out + gen_snow, 0.0, 1.05).astype(np.float32)

        # --- 5. Sync-Threshold Collapse & Colour-Killer Mute (~Gen 19+, t=60.5s in video) ---
        auto_collapse = float(np.clip((passes - 18.0) / 3.0, 0.0, 1.0)) if passes >= 19 else 0.0
        collapse = float(np.clip(max(collapse_override, auto_collapse), 0.0, 1.5))
        if collapse > 1e-4:
            c_amt = min(1.0, collapse)
            # Colour-killer engages: chroma is muted almost to monochrome (sat ~ 0.015 in video)
            ck_keep = max(0.03, 1.0 - 0.95 * c_amt)
            i_out = (i_out * ck_keep).astype(np.float32)
            q_out = (q_out * ck_keep).astype(np.float32)

            # Severe horizontal AFC sync tearing + ghost silhouette of the underlying scene
            tear_rows = (
                rng.standard_normal(H).astype(np.float32) * (28.0 * collapse * scale_w)
                + np.sin(np.arange(H, dtype=np.float32) * 0.19 + frame_index * 1.7) * (22.0 * collapse * scale_w)
            ).astype(np.float32)
            y_ghost = horizontal_shift(lowpass(y_out, max(1.2, 3.0 * scale_w), axis=1), tear_rows)

            # High-contrast diagonal FM demodulator herringbone + horizontal sync-bar snow
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            px_p = max(2.2, 5.2 * scale_w)
            py_p = max(2.0, 3.8 * scale_h)
            row_ph = rng.uniform(-np.pi, np.pi, size=(H, 1)).astype(np.float32) * (0.65 * collapse)
            herring = np.cos(2.0 * np.pi * (xx / px_p - yy / py_p) + frame_index * 1.9 + row_ph)
            herring2 = np.cos(2.0 * np.pi * (xx / (px_p * 0.62) + yy / (py_p * 1.35)) - frame_index * 1.3)
            fm_snow = self._luma_band_noise(rng, (H, W))
            # Horizontal sync-loss black/white tearing bars
            sync_bars = np.sin(yy / max(3.0, 9.5 * scale_h) + frame_index * 0.9 + rng.standard_normal((H, 1)).astype(np.float32) * 0.4)
            jam_pattern = np.clip(
                0.50 + 0.24 * herring + 0.14 * herring2 + 0.22 * fm_snow + 0.16 * sync_bars,
                0.02, 0.98,
            ).astype(np.float32)

            # Keep ~22% underlying torn silhouette so the speaker's outline remains faintly recognizable
            ghost_keep = max(0.18, 1.0 - 0.78 * c_amt)
            y_out = np.clip(y_ghost * ghost_keep + jam_pattern * (1.0 - ghost_keep), 0.0, 1.0).astype(np.float32)

        return y_out, i_out, q_out


    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        if self.digital:
            frame_index = int(buf.index)
            self._ensure(frame_index)
            rng = np.random.default_rng(
                derive_seed(self.ctx.seed, "tape/dv", frame_index))
            return self._apply_digital_damage(buf, frame_index, rng)
        comp = buf.composite
        if comp is None:  # component/digital path: the RF domain does not apply
            return buf
        frame_index = int(buf.index)
        self._ensure(frame_index)
        rng = np.random.default_rng(derive_seed(self.ctx.seed, "tape/frame", frame_index))

        # The whole medium model runs in baseband: take the carrier apart once,
        # damage luma and chroma the way the physical mechanism does, and rebuild
        # the carrier once at the end.  See `_decompose` for why.
        _fmt_l = self.ctx.layer("format")
        fmt_comb = float(_fmt_l["comb_depth"]) if "comb_depth" in _fmt_l else 0.45
        sig_in = buf.meta.get("signal")
        _rf_dir = bool(getattr(sig_in, "_rf_direct", False))
        if (fmt_comb >= 0.90 or _rf_dir) and sig_in is not None:
            y, i, q = sig_in.y.copy(), sig_in.i.copy(), sig_in.q.copy()
        else:
            y, i, q = self._decompose(comp)

        # 0. shuttle: rewind / fast-forward residue, in this format's granularity.
        #    It belongs *before* everything else because it is a property of how
        #    the tape is being read, not an event at a point on the tape -- and it
        #    must run on the analogue branch too, which is where it was missing:
        #    the call lived inside `_apply_digital_damage`, so on VHS and Hi8 the
        #    whole feature silently never executed.
        sh = self._apply_shuttle_concealment(SignalFrame(y, i, q), frame_index, rng)
        y, i, q = sh.y, sh.i, sh.q

        # Remember what this tape produced, for later shuttle substitution.  The
        # frame is captured *after* the format and encoder stages have band-limited
        # it, which is what a deck's frame buffer actually holds and what stops the
        # substituted patches from being sharper than their surroundings.
        if (self.shuttle_speed > 0.0 and self.shuttle_fraction > 1e-4) or self.transport_servo_spec.is_active():
            self._remember_frame(frame_index, y, i, q)

        # 1. time-base error: per-line sampling phase, luma and chroma together
        y, i, q = self._apply_timebase(y, i, q, frame_index)

        # 1b. drum speed error: vertical sampling density.  Separate from 1 above
        #     because the capstan moves the picture sideways while the drum changes
        #     how many distinct lines are read -- different axis, different look.
        y, i, q = self._apply_drum_speed(y, i, q)

        # 1c. pedestal error: the luma reference collapses but chroma does not.
        #     Placed with the other sampling/clamping errors, before the noise, so
        #     the grain is added around the shifted level rather than lifted with it.
        y, i, q = self._apply_luma_pedestal(y, i, q)

        # 1f. stiction: the damp binder grabbing the capstan and releasing.  Applied with
        #     the other transport errors because it displaces the track rather than the
        #     picture content.
        y, i, q = self._apply_stiction(y, i, q, frame_index)

        # 1e. moire from a residual carrier: curved fringes, added to luma only.
        #     It is an interference product in the video path, so it brightens and
        #     darkens; giving it colour would be a separate claim.
        y = self._apply_stripe_moire(y, frame_index)

        # 1d. shed-coating grain: fine luminance noise, no colour
        y, i, q = self._apply_shed_grain(y, i, q, frame_index, rng)

        # 2. deposition geometry: magnetically weak, slowly moving stripes.
        #    A crease reduces head-to-tape contact, so it softens and weakens
        #    BOTH components -- it is not a chroma-only effect.
        y, crease_mask = self._apply_creases(y, frame_index)
        i, _ = self._apply_creases(i, frame_index, precomputed_mask=crease_mask)
        q, _ = self._apply_creases(q, frame_index, precomputed_mask=crease_mask)

        # 3. lost signal patches: the carrier is gone, so chroma goes with it
        y, dropout_mask = self._apply_dropouts(y, frame_index, rng)
        if dropout_mask is not None:
            keep_do = np.float32(1.0) - np.clip(dropout_mask, 0.0, 1.0)
            i = i * keep_do
            q = q * keep_do

        # 4. head-switching tear at the end of each field
        y, i, q, hs_mask = self._head_switch(y, i, q, frame_index, rng)

        # 4b. contamination: a clogged head drags a fixed band of lost signal,
        #     loose debris strikes the tape in short bursts, and a scratch
        #     damages a narrow drifting set of lines
        y, i, q, clog_mask = self._apply_clog(y, i, q, frame_index, rng)
        y, i, q = self._apply_debris(y, i, q, frame_index)
        y, i, q = self._apply_surface_damage(y, i, q, frame_index)
        # 4b-bis. binder breakdown: mould and shed coating.  Runs here rather
        #         than with the discrete events because it removes a fraction of
        #         the picture continuously, and it kills chroma before luma.
        y, i, q = self._apply_binder_loss(y, i, q, frame_index, rng)
        y, i, q = self._apply_cosmag_substrate_failures(y, i, q, frame_index, rng)

        # 4c. random whole lines lose colour (weak colour-under carrier)
        i, q = self._apply_chroma_loss(i, q, frame_index)

        # 4c-bis. other whole lines decode to the *wrong* colour: a reference error
        #         rotates every hue on the line and shifts its colour-under gain, so
        #         the line stays saturated.  This is the failure people actually
        #         describe when they say 整行全错, and it is not a stronger version of
        #         the line above -- losing colour and getting the colour wrong are
        #         different faults with different pictures.
        i, q = self._apply_row_chroma_error(i, q, frame_index)

        # 4d. tracking error: rows lose lock from some point downwards
        y, i, q = self._apply_tracking_noise(y, i, q, frame_index, rng)

        # 4e. Pause/record edit transition & CTL servo hunting glitch
        y, i, q, edit_env = self._apply_edit_glitch(y, i, q, frame_index, rng)

        # 5. particle noise floor
        y = self._apply_noise_floor(y, frame_index)

        # 6. vertical field timing error
        dy = float(self.plan.frame_dy[frame_index]) \
            if self.plan.prepared_frames > frame_index else 0.0
        if abs(dy) > 1e-3:
            y = vertical_shift(y, dy)
            i = vertical_shift(i, dy)
            q = vertical_shift(q, dy)


        # 6b. Shared Power & Electrical Bus coupling on Capstan & Drum Servo:
        #     100/120 Hz rectifier ripple produces S-curved "hum snake" horizontal modulation,
        #     while battery load sag droops capstan/drum torque and injects brush-motor EMI hash.
        from ..codecs.power_bus import evaluate_power_bus
        pstate = buf.meta["power_bus"] if "power_bus" in buf.meta else None
        if pstate is None:
            pstate = evaluate_power_bus(
                self.ctx.params, frame_index, self.ctx.fps,
                y.shape[0], y.shape[1], self.ctx.seed,
                row_luma=np.clip(y[:, ::4].mean(axis=1), 0.0, 1.0),
            )
        if pstate is not None and pstate.active:
            if len(pstate.servo_line_shift) == y.shape[0] and np.any(np.abs(pstate.servo_line_shift) > 1e-4):
                y = horizontal_shift(y, pstate.servo_line_shift)
                i = horizontal_shift(i, pstate.servo_line_shift)
                q = horizontal_shift(q, pstate.servo_line_shift)
            if len(pstate.ripple_row) == y.shape[0]:
                y = (y + 0.055 * pstate.ripple_row[:, None]).astype(np.float32)

        # 6c. Module 4 (Candidate B): Mechanical Transport & Servo State Machine
        #     (Supply/take-up reel radius & tension evolution, Play/Pause/Rec-Pause/Stop state machine,
        #      rubber belt slip, pinch-roller eccentricity & capstan/drum servo hunting)
        ts_diag = None
        if self.transport_servo_spec.is_active():
            from ..codecs.transport_servo import apply_transport_servo
            donor_yiq = None
            if self.transport_servo_spec.transport_state in (1, 2, 3):
                donor_sf = self._shuttle_donor(frame_index, current_sig=SignalFrame(y, i, q))
                if donor_sf is not None and donor_sf.y.shape == y.shape:
                    donor_yiq = (donor_sf.y, donor_sf.i, donor_sf.q)
            y, i, q, ts_diag = apply_transport_servo(
                y, i, q, self.transport_servo_spec, frame_index,
                self.ctx.fps, self.ctx.std, seed=self.ctx.seed,
                donor_yiq=donor_yiq,
            )

        # 7. multi-generation VHS iterative dubbing loss (colour-under shift, ringing, sync collapse)
        y, i, q = self._apply_generation_loss(y, i, q, frame_index, rng)

        # 7b. Candidate K: Linear Tape Edit Controller (Assemble / Insert Edit) & Early NLE Timeline
        edit_rep = None
        if self.edit_controller_spec.is_active():
            from ..codecs.edit_controller_nle import apply_edit_controller
            b_yiq = buf.meta["bus_b_yiq"] if "bus_b_yiq" in buf.meta else None
            ed_rgb, edit_rep = apply_edit_controller(
                SignalFrame(y.astype(np.float32), i.astype(np.float32), q.astype(np.float32)).to_rgb(),
                self.edit_controller_spec, std=buf.std, frame_index=frame_index,
                fps=self.ctx.fps, seed=self.ctx.seed, bus_b_yiq=b_yiq,
            )
            ed_sig = SignalFrame.from_rgb(ed_rgb, buf.std)
            y, i, q = ed_sig.y, ed_sig.i, ed_sig.q

        comp_out = self._recompose(y, i, q)
        out = FrameBuffer(buf.rgb, comp_out, buf.index, buf.time, buf.std, dict(buf.meta))
        if fmt_comb >= 0.90 or _rf_dir:
            _s_out = SignalFrame(y.astype(np.float32), i.astype(np.float32), q.astype(np.float32))
            _s_out._rf_direct = _rf_dir
            out.meta["signal"] = _s_out
        out.meta["tap_tape_signal"] = SignalFrame(y.astype(np.float32, copy=False), i.astype(np.float32, copy=False), q.astype(np.float32, copy=False))
        out.meta["dropout_mask"] = dropout_mask
        if ts_diag is not None:
            out.meta["transport_servo"] = ts_diag
        if edit_rep is not None:
            out.meta["edit_controller"] = edit_rep
        out.meta["edit_glitch_env"] = edit_env
        out.meta["crease_mask"] = crease_mask
        out.meta["line_phase"] = self._line_phase_for(frame_index)
        if getattr(self, "_last_hs_in_band", None) is not None:
            out.meta["hs_in_band"] = self._last_hs_in_band
            out.meta["hs_active_end"] = self._last_hs_active_end
        return out


class TransportServoStage(Stage):
    """
    Standalone node-graph stage (`transport_servo` / `机械走带与伺服`) for Module 4:
    Mechanical Transport & Servo State Machine.
    """
    name = "transport_servo"
    layers = ("tape",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._donor_yiq: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._read_params()

    def _read_params(self) -> None:
        from ..codecs.transport_servo import TransportServoSpec
        self.spec = TransportServoSpec.from_params(self.ctx.params)

    def on_params_changed(self) -> None:
        self._read_params()

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        from ..codecs.transport_servo import apply_transport_servo
        self._read_params()
        sig = buf.meta.get("signal")
        if sig is None:
            sig = SignalFrame.from_rgb(buf.rgb, buf.std)
        y, i, q = sig.y.copy(), sig.i.copy(), sig.q.copy()
        donor = self._donor_yiq
        if donor is None or donor[0].shape != y.shape:
            donor = (y.copy(), i.copy(), q.copy())
        if self.spec.transport_state != 1 or self._donor_yiq is None:
            self._donor_yiq = (y.copy(), i.copy(), q.copy())
        y_out, i_out, q_out, diag = apply_transport_servo(
            y, i, q, self.spec, int(buf.index),
            self.ctx.fps, buf.std, seed=self.ctx.seed,
            donor_yiq=donor,
        )
        sig_out = SignalFrame(y_out, i_out, q_out)
        rgb_out = sig_out.to_rgb()
        from ..core.signal import composite_encode
        comp_out = composite_encode(y_out, i_out, q_out, buf.std, chroma_gain=CHROMA_CARRIER_GAIN)
        meta = dict(buf.meta)
        meta["signal"] = sig_out
        meta["transport_servo"] = diag
        return FrameBuffer(rgb_out, comp_out, buf.index, buf.time, buf.std, meta)
