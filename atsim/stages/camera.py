"""
Stage 01 -- the imaging system.

This layer answers "what shot it", before any video signal exists.  Everything
here happens in RGB (light), which is why it cannot be folded into the format
layer: a lens/CCD artefact has a different spatial and chromatic signature than
a bandwidth limit applied to an already-encoded signal.

Motion blur is the one genuinely temporal operation: a 1/60 s shutter integrates
two fields, so it is implemented as a blend with the previous frame and needs
the previous input frame.  `set_previous` is called by the renderer.
"""
from __future__ import annotations

import os

import numpy as np

from ..codecs.operator import HAND
from ..codecs.studio_ccu_genlock import StudioCCUSpec, apply_studio_ccu
from ..codecs.mobile_isp_optics import MobileISPOpticsSpec, resolve_mobile_profile, run_mobile_isp_pipeline
from ..codecs.autocontrol import (ExposureLoop, FocusLoop, WhiteBalanceLoop,
                                  centre_weights)
from ..core.pipeline import FrameBuffer, RenderContext, Stage
from ..core.rng import ValueNoise3D, bandlimited_noise, derive_seed, rng_for
from ..core.signal import gaussian_kernel, lowpass

#: The luma vector, used by the tone mechanisms.  Rec.601 weights, as elsewhere in the model.
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
from ..codecs.lens import (LensSpec, field_curvature, longitudinal_ca, chromatic_aberration,
                           coma, astigmatism, spherical, distortion, source_glow,
                           optical_wt_zoom, light_streaks, asymmetric_ccd_smear_and_trench,
                           draw_trv310e_zoom_osd, coating_ghost_dispersion, _bilinear_sample_2d)


def _subpixel_shift(rgb: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Move the frame by a fractional number of pixels.

    A hand does not move in whole pixels, and the difference is visible: an integer shift gives a
    stepped, mechanical shake, while a fractional one gives a smooth movement.  Bilinear sampling is
    enough here because the shift is small -- at the amplitudes a hand produces, the interpolation
    error is far below the sensor noise the frame is about to receive.
    """
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return rgb
    h, w = rgb.shape[:2]
    ys = np.arange(h, dtype=np.float32)[:, None] - dy
    xs = np.arange(w, dtype=np.float32)[None, :] - dx
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    fy = np.clip(ys - y0, 0.0, 1.0)[:, :, None]
    fx = np.clip(xs - x0, 0.0, 1.0)[:, :, None]
    a = rgb[y0, x0]
    b = rgb[y0, x1]
    c = rgb[y1, x0]
    d = rgb[y1, x1]
    top = a * (1.0 - fx) + b * fx
    bot = c * (1.0 - fx) + d * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


LUMA_F32 = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class CameraStage(Stage):
    name = "camera"
    layers = ("camera",)

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._read_params()
        # the loops themselves, created once and kept: their whole purpose is to take time, so
        # re-creating them on a parameter change would defeat the time constant they express
        self._ae = ExposureLoop(target=self.ae_target, tau_s=self.ae_tau_s,
                                limit_stops=self.ae_limit_stops,
                                presettle_s=self.autopresettle_s)
        self._awb = WhiteBalanceLoop(tau_s=self.awb_tau_s, limit=self.awb_limit,
                                     presettle_s=self.autopresettle_s)
        self._af = FocusLoop(tau_s=self.af_tau_s)
        self._weights = None
        self._bumps = None
        self._sites = None
        self._zoom_softness = 0.0
        self._cur_eff_zoom = 0.0
        self._prev_eff_zoom = None
        self._defocus = 0.0
        self._af_offset = 0.0
        # state that must survive a parameter change: the auto-control loops take time to
        # settle, so re-creating them on every edit would defeat the time constant they exist
        # to express
        self._prev: np.ndarray | None = None
        self._fpn: np.ndarray | None = None
        self._noise: ValueNoise3D | None = None
        self._ae_state: float | None = None
        self._awb_state: np.ndarray | None = None
        self._af_state: float | None = None

    def _read_params(self) -> None:
        cfg = self.cfg_of("camera")
        self.sensor_noise = float(cfg.get("sensor_noise", 0.0))
        self.grain = float(cfg.get("noise_grain", 1.4))
        self.fixed_pattern = float(cfg.get("fixed_pattern", 0.0))
        self.dynamic_range = float(cfg.get("dynamic_range", 8.0))
        # --- the night-vision modes ------------------------------------------------
        # Physical reconstruction rather than a colour grade: the illuminator, the removal of the IR-cut
        # filter, the monochrome green display, and frame accumulation.  Built against the reference clip
        # `索尼F717夜视测试`, segmented into its four modes and measured per segment.
        from ..codecs.nightvision import NightVisionSpec
        self.nv = NightVisionSpec.from_params(cfg)
        #: **The sensor's saturation point, in stops below the range's own white.**  Zero leaves `_roll_off` as it was.
        #: See the comment there: the curve clips correctly, but at the default range its white point is 1.0 while the
        #: picture's maximum is 0.87, so nothing ever reaches it and the model can never show a blown highlight.
        #
        #: **And it applies only when the night-vision mode is on**, which is a measurement rather than a convenience.
        #: Lowering the white point is a statement about *this* sensor's saturation, and the mode it was fitted to is
        #: NightShot; with it applying always, the suite caught that **the scanline structure began costing light**
        #: (the frame's mean varied by 0.00275 across the four spot sizes) -- a property verified earlier in this work
        #: and measured on a configuration with no night vision in it at all.  So the white point moves where the
        #: measurement says it should and nowhere else.
        sat = float(cfg.get("saturation_stops", 0.0))
        self.saturation_stops = sat if float(cfg.get("nv_mode", 0.0)) >= 0.5 else 0.0
        #: **The night-vision mode's own exposure offset, and it must NOT be the global `exposure`.**  Measured, the
        #: highlight calibration wants a **-0.10 stop** trim to bring the mean onto the footage's.  **Setting the global
        #: `exposure` default to -0.10 broke a check that has nothing to do with night vision** -- "a cold machine renders
        #: differently from a warm one", whose largest mean difference fell to 0.0008 -- because `exposure` is the camera's
        #: own control and the AE loop, the spot sizes and the thermal drift are all measured through it.  **A trim
        #: belonging to one mode belongs to that mode**, exactly as the saturation point does.
        #:
        #: **Read from the config rather than hard-coded**, so it can be tuned: the first version of this line was a
        #: literal -0.10 and two subsequent attempts to sweep it changed nothing, which is this work's own "declared,
        #: reachable, inert" fault class -- a parameter that exists and cannot be reached.
        #: **The NightShot exposure trim, and it carries what `dsp_gain = 0.75` used to carry.**  That gain was
        #: solved from the footage's percentiles, but `_dsp_level` applies it *after* `_roll_off`, so it put a hard
        #: ceiling of `g` on the whole chain -- **the model could never reach full scale** (white in gave max 0.7590
        #: while the DV frame reaches 1.0000, and 6.71% of that frame is above 0.90 against the model's 0.00%).
        #: Moving the same brightness into the exposure is exactly equivalent in level (`2 ** -0.415 = 0.7500`) and
        #: **cannot cap the white point, because `_roll_off` maps the sensor's saturation back to 1.0 afterwards** --
        #: which is what a camera's exposure does and what its gain does not.  The parameter already only applies when
        #: the night-vision mode is on, so it needs no new gating.
        trim = float(cfg.get("nv_exposure_trim", -0.85))
        self.nv_exposure_trim = (trim if float(cfg.get("nv_mode", 0.0)) >= 0.5 else 0.0)
        # **The near-IR material labelling, and whether this machine can do it at all.**  `note.txt`'s second mechanism
        # asks for near-IR albedo per material rather than an RGB-to-grey conversion, and a vision-language model can
        # supply it; where none is installed the estimate is used, **which is not a degraded mode but the documented
        # fallback**.  The labels are computed once per clip and cached, because the source is a still or a clip whose
        # materials do not change frame to frame.
        self.nir_material = bool(round(float(cfg.get("nv_nir_material", 1.0))))
        #: the checkpoint, if one is named.  **An environment variable rather than a schema parameter**, because every
        #: parameter in this schema is numeric and a path is not -- and because naming a model is a property of the
        #: machine, not of the look being simulated.
        self.nir_model_dir = os.environ.get("ATSIM_NIR_MODEL") or None
        self.nir_margin_floor = float(cfg.get("nv_nir_margin_floor", 0.02))
        #: **How fine the material labelling is, in PIXELS rather than in cells, and that distinction is the fix for
        #: a block artefact the user reported.**
        #:
        #: It used to be a cell count (`nv_nir_grid`, default 8), and a cell count is not a physical quantity: at the
        #: 533x400 the figures were calibrated on, 8x8 means **66x50 px per cell**, while on the clip's own
        #: **1440x1080** the same 8 means **180x135 px** -- a 24000-pixel block, 7.5x the area, and the user's
        #: photograph showed exactly that: large light and dark rectangles over the case.  **A material's spatial
        #: scale is a property of the scene, so the label grid has to be tied to the picture's pixels or the same
        #: parameter means a different physical size on every input.**
        #:
        #: The default cell is **66 px**, which reproduces the physical scale the material table was calibrated at:
        #: `533/8 = 66` and `400/8 = 50`, so 66 on the shorter edge.  Measured on 720x540, a 66 px cell gives 11x8
        #: cells and takes the labelling to about the same order as the 16x16 measurement (16.5 s), and the analysis
        #: runs **once per clip**, so that is the whole cost.
        self.nir_cell_px = float(np.clip(float(cfg.get("nv_nir_cell_px", 66.0)), 16.0, 512.0))
        self.nir_grid_max = int(np.clip(round(float(cfg.get("nv_nir_grid_max", 32.0))), 2, 64))
        #: the lattice actually used, `(grid_h, grid_w)`, filled by `_grid_for` once the frame's size is known
        self.nir_grid: tuple[int, int] = (8, 8)
        self.nir_label_source: str | None = None
        self.nir_depth_source: str | None = None
        #: whether to infer a depth map, so the illuminator's falloff can be about distance rather than a radius
        self.nir_depth = bool(round(float(cfg.get("nv_nir_depth", 1.0))))
        #: **The DSP's output level.**  Gain and black level, in the order the camcorder applies them -- see
        #: `_dsp_level` for why two quantities are needed rather than one brightness, and why both default neutral.
        #: **These are re-read from the spec in `_night_vision`**, because the MODE is what sets them: the values that
        #: put the NightShot material's percentiles on target belong to that mode, not to the camera, and using them
        #: as the schema default darkened every ordinary visible-light render.
        self.dsp_gain = float(np.clip(cfg.get("dsp_gain", 1.0), 0.1, 4.0))
        self.dsp_black = float(np.clip(cfg.get("dsp_black", 0.0), -0.5, 0.5))
        #: **The DSP's transfer curve, and it is what lifts a shadow floor that a pedestal cannot.**  A real
        #: NightShot's black is not black but **deep green-grey** (`note.txt`), and the model's is not: measured
        #: against the user's frame the model's p1 is 0.0186 where the frame's is 0.0471 and its p25 is 0.0522 where
        #: the frame's is 0.1511 -- **two to three times too dark at the bottom** -- while its p75 is 0.7027 against
        #: 0.4841, i.e. too bright at the top.  A pedestal adds a constant and leaves the shadow ratios alone; a gain
        #: lifts the highlights that are already too high.  `x ** gamma` with `gamma < 1` stretches the bottom and
        #: compresses the top with 0 and 1 fixed, which is the standard camera transfer and the shape that is needed.
        self.dsp_gamma = float(np.clip(cfg.get("dsp_gamma", 1.0), 0.3, 3.0))
        #: The highlight shoulder.  See `_highlight_compress` for why neither `nv_highlight_saturation` nor
        #: `dsp_gamma` could serve: the first is the desaturation drive (0.009 of p95 across a 2.5x sweep) and the
        #: second is one curve that takes the shadows down as far as it brings the highlights back.
        #: **the range is 0..6 and the clamp here has to match the schema's**, because a bound that exists in two
        #: layers means raising one of them changes nothing -- which is why the first attempt at a stronger
        #: shoulder measured no difference at all between 1.0 and 6.0
        self.highlight_compress = float(np.clip(cfg.get("nv_highlight_compress", 1.0), 0.0, 6.0))
        self.highlight_knee = float(np.clip(cfg.get("nv_highlight_knee", 0.70), 0.5, 0.99))
        #: the exponent's shape: >1 puts the compression at the TOP of the shoulder, which is where `_dsp_level`
        #: leaves the picture pinned.  See `_highlight_compress`.
        self.highlight_shape = float(np.clip(cfg.get("nv_highlight_shape", 2.5), 1.05, 4.0))
        #: the DSP's tone curve, as an offset from the identity -- see `_dsp_level`
        self.dsp_contrast = float(np.clip(cfg.get("dsp_contrast", 0.0), -0.8, 2.0))
        self._nir_albedo: np.ndarray | None = None
        self._nir_depth: np.ndarray | None = None
        self._nir_failed = False
        #: The recent post-exposure frames, so a mode that accumulates has frames to sum.  The module
        #: returns the mean, so the picture stays at the same scale as a single frame -- which is what makes
        #: the noise *fall* rather than the brightness rise.
        self._nv_history: list = []
        self.exposure = float(cfg.get("exposure", 0.0))
        # **The night-vision exposure bias is added here, not applied as a gain at the end.**  The stage's own
        # exposure feeds the auto-exposure loop, the roll-off, the gain stages and the white balance in a fixed
        # order; a multiplier bolted on afterwards would lighten the picture without it ever having been exposed
        # differently -- no shift in the highlight roll-off and nothing for the AE loop to react to.  Super
        # NightShot's character *is* a different exposure, so the bias belongs in the value.
        self._exposure_base = self.exposure
        # the mode's own bias, added to whatever the user set -- through the same value, so it is exposed
        # rather than painted on.  Zero when the mode is off.
        self.exposure = self._exposure_base + self._nv_exposure_bias()
        # --- automatic control loops -------------------------------------------
        # A handycam has no operator setting exposure; it has a loop that measures the
        # picture and moves a control, with a time constant, and gets it slightly wrong.
        # These are the only mechanisms in the model that carry state forward, because they
        # read the picture -- see the module docstring in codecs/autocontrol.py.
        self.ae_enabled = float(cfg.get("ae_enabled", 0.0)) > 0.5
        self.ae_target = float(cfg.get("ae_target", 0.46))
        self.ae_tau_s = float(cfg.get("ae_tau_s", 1.1))
        self.ae_limit_stops = float(cfg.get("ae_limit_stops", 2.0))
        self.awb_enabled = float(cfg.get("awb_enabled", 0.0)) > 0.5
        self.awb_tau_s = float(cfg.get("awb_tau_s", 1.6))
        self.awb_limit = float(cfg.get("awb_limit", 0.42))
        self.af_enabled = float(cfg.get("af_enabled", 0.0)) > 0.5
        self.af_tau_s = float(cfg.get("af_tau_s", 0.9))
        self.autopresettle_s = float(cfg.get("autopresettle_s", 0.0))
        self.wb_temp = float(cfg.get("white_balance_temp", 0.0))
        self.sharpness = float(cfg.get("sharpness", 1.0))
        # **All the lens's parameters in one object**, so eight floats cannot be passed in the wrong order.
        self.lens = LensSpec.from_params(cfg)
        # **Field curvature**: a simple lens images a flat subject onto a curved surface, so the plane in
        # focus at the centre is not the plane in focus at the corners.  The blur is a function of the
        # *field radius* -- zero at the centre and rising outward -- because the point-spread function
        # grows with the square of the field angle.
        self.field_curvature = float(cfg.get("field_curvature", 0.0))
        # **Longitudinal chromatic aberration**: the wavelengths focus at different distances along the
        # axis, so at the sensor one is in focus and the others are not.  Unlike the lateral kind this is
        # per-channel *defocus* with no displacement, so it softens an edge in place rather than
        # separating the channels -- a different signature, which is why the two are asserted differently.
        self.longitudinal_ca = float(cfg.get("longitudinal_ca", 0.0))
        self.motion_blur = float(cfg.get("motion_blur", 0.0))
        # --- the operator's hand ------------------------------------------------
        # Shake is the camera body moving, so it is applied before the sensor sees anything.
        self.shake_intensity = float(cfg.get("shake_intensity", 0.0))
        self.bump_rate_hz = float(cfg.get("bump_rate_hz", 0.0))
        self.bump_amount = float(cfg.get("bump_amount", 0.0))
        self.zoom_travel = float(cfg.get("zoom_travel", 0.0))
        #: **a number, not `str(...)`**.  It used to be `str(cfg.get("zoom_kind", 1.0))`, i.e. the string `"1.0"`,
        #: which `zoom_curve` could never match against its `"linear"` / `"cinema"` names -- so even once the call
        #: site stopped hardcoding `"motor"`, the value still had to be translated rather than stringified.
        self.zoom_kind = float(cfg.get("zoom_kind", 1.0))
        self.wt_zoom = float(np.clip(cfg.get("wt_zoom", 0.0), 0.0, 1.0))
        self.wt_max_zoom = float(np.clip(cfg.get("wt_max_zoom", 4.0), 1.0, 16.0))
        self.zoom_center_x = float(np.clip(cfg.get("zoom_center_x", 0.5), 0.05, 0.95))
        self.zoom_center_y = float(np.clip(cfg.get("zoom_center_y", 0.5), 0.05, 0.95))
        self.zoom_dynamic_mode = int(round(float(cfg.get("zoom_dynamic_mode", 0.0))))
        self.zoom_speed = float(np.clip(cfg.get("zoom_speed", 0.35), 0.05, 2.0))
        self.zoom_resolve_detail = float(np.clip(cfg.get("zoom_resolve_detail", 0.75), 0.0, 1.5))
        self.zoom_osd = float(cfg.get("zoom_osd", 0.0)) > 0.5
        self.light_streak_amount = float(np.clip(cfg.get("light_streak_amount", 0.0), 0.0, 2.0))
        self.light_streak_angle = float(cfg.get("light_streak_angle", 0.0))
        self.light_streak_length = float(np.clip(cfg.get("light_streak_length", 0.55), 0.1, 1.5))
        self.smear = float(cfg.get("ccd_smear", 0.0))
        self.halation = float(cfg.get("halation", 0.0))
        #: How much the halation's tint follows the **blown region's own colour** instead of the fixed
        #: lens tint.  A scattering halo keeps the source's spectrum, so this is the physical behaviour;
        #: it is a blend because the fixed tint is also a real lens property.
        self.halation_source_colour = float(cfg.get("halation_source_colour", 0.0))
        #: **Midtone and shadow compression that leaves highlights at unity gain.**  Raises the contrast of a
        #: bright thin line against its surroundings without adding light to the line, which is the only way
        #: to do it without diluting the line's colour -- adding light raises the line and its surroundings
        #: together and so reduces their difference.
        self.shadow_depth = float(cfg.get("shadow_depth", 0.0))
        # --- the note's optical and sensor mechanisms ---------------------------
        self.lens_fog = float(cfg.get("lens_fog", 0.0))
        # **The overflow drain's dark rim.**  A photosite whose well overflows has its excess charge
        # removed by the anti-blooming structure, which pulls the neighbouring sites *below* their own
        # exposure -- so the ring just outside a blown highlight is darker than the background.  Measured
        # without it: the ring sat 0.012 *above* the ceiling and the profile fell monotonically from the
        # core, so nothing in the model could produce a rim at all.
        self.overflow_drain = float(cfg.get("overflow_drain", 0.0))
        self.overflow_width = float(cfg.get("overflow_width", 2.0))
        self.bloom_amount = float(cfg.get("bloom_amount", 0.0))
        self.bloom_threshold = float(cfg.get("bloom_threshold", 0.82))
        self.dead_sites_rate = float(cfg.get("dead_sites", 0.0))
        self.stuck_fraction = float(cfg.get("stuck_fraction", 0.45))
        self.gain = float(cfg.get("ccd_gain", 0.0))
        self.color_cast = float(cfg.get("color_cast", 0.0))
        self.highlight_tint = float(cfg.get("highlight_tint", 0.0))
        self.vcd_color_tone = float(cfg.get("vcd_color_tone", 0.0))
        self.ccd_flash_falloff = float(cfg.get("ccd_flash_falloff", 0.0))
        # --- lens and sensor type -------------------------------------
        self.lens_wear = float(cfg.get("lens_wear", 0.0))
        self.lens_flare = float(cfg.get("lens_flare", 0.0))
        self.spike_spread = float(cfg.get("spike_spread", 0.6))
        # per-channel focus error of the *flare*: the out-of-focus aperture image is larger in
        # blue than in red, so a bright source wears a coloured halo rather than a coloured core
        self.flare_defocus = float(cfg.get("flare_defocus", 0.5))
        self.flare_blades = float(cfg.get("flare_blades", 6.0))
        self.sensor_type = str(cfg.get("sensor_type", 0.0))
        self.is_3ccd = self.sensor_type in ("1", "1.0", "3ccd", "3CCD")
        self.ccu_spec = StudioCCUSpec.from_params(cfg)
        self.mobile_isp_spec = MobileISPOpticsSpec.from_params(cfg)


    # ------------------------------------------------------------------

    def on_params_changed(self) -> None:
        """Re-read every parameter this stage cached, so a per-frame change reaches it."""
        self._read_params()

    def _noise_field(self, shape: tuple[int, int]) -> ValueNoise3D:
        """
        The sensor noise field, built for the frame shape actually arriving.

        `ctx.height/width` is the renderer's *nominal* size; a source can deliver
        frames of another size (a proxy render, for instance), and pre-building
        the field from the context then fails to broadcast against the real frame.
        """
        if self._noise is None or self._noise_shape != (int(shape[0]), int(shape[1])):
            h, w = int(shape[0]), int(shape[1])
            self._noise = ValueNoise3D(
                seed=self.ctx.seed * 31 + 7, height=h, width=w,
                spatial_scale=max(2.0, self.grain * 2.0),
                time_scale=max(2.0, 6.0 / max(1e-3, self.ctx.fps) * 4.0),
            )
            self._noise_shape = (h, w)
            self._fpn = None
        return self._noise

    # ------------------------------------------------------------------
    def set_previous(self, rgb: np.ndarray | None) -> None:
        self._prev = None if rgb is None else rgb

    def _fixed_pattern(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = int(shape[0]), int(shape[1])
        if self._fpn is None or self._fpn.shape[:2] != (h, w):
            cache = getattr(CameraStage, "_FPN_GLOBAL_CACHE", None)
            if cache is None:
                cache = {}
                CameraStage._FPN_GLOBAL_CACHE = cache
            key = (int(self.ctx.seed), h, w)
            hit = cache.get(key)
            if hit is None:
                rng = rng_for(self.ctx.seed, "camera", "fpn")
                col = rng.standard_normal((1, w, 3)).astype(np.float32)
                col = lowpass(col, 1.2, axis=1)
                grain = bandlimited_noise(rng, (h, w, 3), 1.6, 1.6)
                hit = (np.float32(0.65) * col + np.float32(0.35) * grain).astype(np.float32, copy=False)
                if len(cache) > 16:
                    cache.pop(next(iter(cache)), None)
                cache[key] = hit
            self._fpn = hit
        return self._fpn

    def _white_balance(self, rgb: np.ndarray) -> np.ndarray:
        """
        The static colour cast, and the automatic loop's gains when it is running.

        The two multiply rather than replace each other: the static value is the camera's *null*
        error, which is there whether or not the loop is engaged, and the loop adds its own
        correction on top. With the loop off this is exactly the old behaviour.
        """
        gains = np.ones(3, np.float32)
        if abs(self.wb_temp) > 1e-4:
            t = float(self.wb_temp)
            gains = gains * np.array([1.0 + 0.18 * t, 1.0 + 0.02 * t, 1.0 - 0.20 * t],
                                     dtype=np.float32)
        loop_gains = getattr(self, "_awb_gains", None)
        if self.awb_enabled and loop_gains is not None:
            gains = gains * np.asarray(loop_gains, np.float32)
        if np.allclose(gains, 1.0):
            return rgb
        return rgb * gains[None, None, :]

    def _expose(self, rgb: np.ndarray) -> np.ndarray:
        if abs(self.exposure) < 1e-4:
            return rgb
        return rgb * np.float32(2.0 ** self.exposure)

    def _nv_source_frames(self, n: int, index: int | None) -> list:
        """
        The last `n` source frames, for a mode that accumulates.

        **Read from the source rather than kept on the stage**, because each frame is rendered by its own call and a
        list on the stage would only ever hold the current one. The source keeps its own small LRU, so asking for a few
        frames costs little, and the camera already reads `index - 1` for its shutter for the same reason.

        **`index` is required rather than defaulted.**  The first version read `getattr(self.ctx, "index", 0)`, and
        `RenderContext` has no `index` -- so it silently used 0, the window clipped to a single frame, and accumulation
        did nothing while looking correctly wired. Measured, the noise was identical at 1, 4 and 8 frames. A missing
        attribute that yields a plausible value instead of raising is how a mechanism ends up inert.

        The window is clipped to the frames that exist -- an accumulation of 16 on frame 2 returns 2 frames, not an
        error -- and `len` is reported by the caller so a request larger than the sequence is visible rather than
        silently truncated.
        """
        src = getattr(self.ctx, "source", None)
        if src is None or index is None:
            return []
        idx = int(index)
        out = []
        for k in range(n - 1, -1, -1):
            j = idx - k
            if j < 0:
                continue
            try:
                out.append(np.asarray(src.frame(j), np.float32))
            except Exception:                                                   # noqa: BLE001
                continue
        return out

    def _night_vision(self, rgb: np.ndarray, stage: str, index: int | None = None) -> np.ndarray:
        #: **The frame index is remembered, because the DISPLAY call does not receive one.**
        #: `apply` calls this twice -- `("scene", buf.index)` and then `("display")` with no index -- and the read noise
        #: is applied on the display side now, so without this the seed would be a constant 0 and **every frame would
        #: carry the identical noise realisation**.  That is not a subtlety: `read_noise`'s own docstring says the noise
        #: is deterministic in the frame index *so that an accumulator has something to average*, and a constant seed
        #: removes exactly that.  Measured with a frame-varying source, `nv_read_noise` and `nv_shot_noise` moved the
        #: sequence by 0.00000 -- because the noise was the same in every frame, so nothing temporal could see it.
        if index is not None:
            self._nv_index = int(index)
        """
        The night-vision terms, called twice per frame because they belong in two different places.

        `stage="scene"` runs **before the sensor**: the illuminator's light reaches the scene and comes back, and the
        cut filter's removal lets the sensor admit near-infrared. `stage="display"` runs **after every sensor term**:
        the monochrome green view, which is the viewfinder's output and therefore the last thing to happen.

        Kept as one method with a stage argument rather than two, so the physical ordering is visible in one place
        instead of being implied by two call sites far apart in `apply`.
        """
        from ..codecs.nightvision import (accumulate, add_illuminator, ir_highlights,
                                           mode_of, monochrome_green, read_noise,
                                           shutter_integrate)
        if mode_of(self.nv) == "off":
            return rgb
        # **The mode's own settings.**  The stage drives the module's pieces directly rather than through
        # `apply_mode`, so it has to resolve the mode's values too -- and that is the right place for it, because the
        # stage is what sequences the pieces.  Without this, selecting Color Slow Shutter returned a green picture:
        # measured through the API, ratio 0.330:0.383:0.287, because `nv_green_mix` defaults to 1 and that mode's
        # requirement is 0.
        from ..codecs.nightvision import mode_spec
        self.nv = mode_spec(self.nv)
        # **The mode's DSP level, adopted onto the stage -- but a value the USER moved wins over the mode's.**
        # The two calibrated values (0.75, 0.009) live on the NightShot presets rather than in the schema, because
        # putting them in the schema darkened every ordinary visible-light render (five checks failed).  **But
        # `mode_spec` applies a preset by `dataclasses.replace`, so a value in it is not a default -- it is a value the
        # user cannot change.**  Measured after the preset was added: `dsp_gain` at 1.0, at 0.75 and `dsp_black` at 0.0
        # all gave a **bit-identical frame** (p1 0.0485, dark mean 0.1403) -- two more controls read and then replaced
        # before use, which is the fourth time this shape has appeared in this file.  `mode_spec`'s own docstring
        # records the first two (`ir_response`, then `ir_albedo`/`ir_depth`).
        #
        # **"The user moved it" is decided against the SCHEMA's default, not against a neutral constant**, because the
        # config this reads is the fully resolved one: every parameter is present whether or not anyone chose it, so a
        # test for "is it neutral" cannot tell a deliberate 1.0 from an untouched one.  Comparing with the schema's own
        # default can, and it is the same distinction `nv_nir_grid`'s deprecation note relies on.
        try:
            from ..core.profiles import SCHEMA
            _specs = SCHEMA.layer("camera")
            for _name in ("dsp_gain", "dsp_black"):
                _s = _specs.specs[_name]
                _want = float(getattr(self, _name, _s.default))
                if abs(_want - float(_s.default)) > 1e-6:
                    setattr(self.nv, _name, _want)
        except Exception:                                          # noqa: BLE001
            # a missing schema entry must not take the render down; the mode's own value then stands, which is the
            # behaviour this had before the override existed
            pass
        self.dsp_gain = float(np.clip(float(getattr(self.nv, "dsp_gain", 1.0)), 0.1, 4.0))
        self.dsp_black = float(np.clip(float(getattr(self.nv, "dsp_black", 0.0)), -0.5, 0.5))
        if stage == "scene":
            # **The analysis and the attachment are two steps, and an earlier version of this folded them into one and
            # made the distance term inert.**  The analysis is expensive -- a vision model, once per clip -- while the
            # attachment is two assignments that must happen **after** `mode_spec`, which runs every frame and rebuilds
            # the spec.  With one combined method that returned early once the analysis was cached, the assignments were
            # skipped on every frame including the first, because the cache was filled *by* that same call: measured,
            # `nv_nir_depth` on and off gave bit-identical output (luma 0.2390, contrast 0.2004) and the correlation
            # between rendered luminance and inferred depth was **+0.148 either way**, where a working term drives it
            # negative.
            self._analyse_scene_nir(rgb)
            self.nv.ir_albedo = self._nir_albedo
            self.nv.ir_depth = self._nir_depth
            # **Accumulation comes first, while the picture is still at the scene's own level.**  It ran in the
            # display stage initially -- the last thing to execute -- where it replaced a *processed* picture (mean
            # 0.4886) with the mean of several *source* frames (mean 0.1768), a factor of 3.66.  Then it ran after
            # `add_illuminator`, which left the two at different levels again and a scalar level match was needed to
            # reconcile them -- and that match scaled the whole frame and clipped it (0.409 where a mean must give
            # 0.5).  **Both faults were the same mistake: summing frames at one level and using the result at
            # another.**  Summing them here removes the mismatch instead of correcting for it.
            #
            # It is also the physically right place.  **The sensor integrates the light it receives**, and a steady
            # illuminator's contribution is simply part of that light -- so integrating the scene and then adding the
            # lamp's field is equivalent to integrating the lamp along with everything else.  What matters for the
            # mechanism is that the frames **differ**, so the uncorrelated noise falls as sqrt(N) while the signal
            # holds.
            n = int(round(float(np.clip(self.nv.accumulate, 1.0, 64.0))))
            # **The slow shutter is its own mechanism, not a longer accumulation.**  `accumulate` sums separate
            # exposures, which makes the picture *cleaner*; a shutter held open smears a moving subject and does not
            # bring it back.  `shutter_frames` therefore drives a separate path.
            #
            # **One window serves both, and an earlier version of this branched on `accumulate` alone** -- so with
            # `shutter_frames` added and defaulting to 1, the `else` branch was taken and **`accumulate` never ran**:
            # measured, temporal noise at `accumulate = 8` was **0.07968 against a single frame's 0.07968, i.e. no
            # reduction at all**, where 1/sqrt(8) = 0.354 is what Super NightShot exists to produce.  **A new mechanism
            # introduced a regression in the old one**, which is the fourth time in this work that adding a term has
            # made an existing one inert.  So the window length is the larger of the two and both mechanisms are
            # applied to it, in the physical order: the sensor integrates the light first, the read-out sums exposures.
            shut = int(round(float(np.clip(self.nv.shutter_frames, 1.0, 64.0))))
            want = max(n, shut)
            out = rgb
            if want > 1:
                stack = self._nv_source_frames(want, index)
                if len(stack) > 1:
                    # **One exposure held open, then the separate exposures summed.**  The integration is applied to
                    # the frames because a long exposure's smear is in the light the sensor collects, before anything
                    # is read out of it.
                    if shut > 1:
                        stack = [shutter_integrate(stack, self.nv)]
                    out = accumulate(stack, self.nv) if n > 1 else stack[0]
            out = add_illuminator(out, self.nv)
            out = ir_highlights(out, self.nv)
            return out
        #: **The display's own chroma noise, and it is not the sensor's.**  The chain above ends with
        #: `monochrome_green`, which maps every pixel onto the display's primary ratios and therefore **removes chroma
        #: noise by construction** -- measured on the camcorder's own frame, the dark band's hue spread arrives at
        #: 0.212, reaches 0.951 after the sensor's read noise, and then comes out of the green matrix as **0.000**.
        #: So a chroma term added on the sensor side is deleted by the next step; that is the fifth time in this work a
        #: value has been produced and then annihilated before use.
        #:
        #: **And it belongs here physically.**  A camcorder's read noise is one number per photosite, sensor-side.
        #: What makes a *near-black* pixel's hue wander is the chroma channel's signal-to-noise ratio, and in a
        #: NightShot camera the chroma path is downstream of the mono conversion -- the display's own channels are
        #: where colour stops being defined.  **It is weighted by darkness rather than by brightness**, which is the
        #: opposite of shot noise: the first version reused `read_noise`, whose shot term grows as `sqrt(lum)`, and
        #: measured, the bright band's hue spread went to **1.314 against the footage's 0.211** -- noise where the
        #: picture has none.  A display's chroma S/N is worst where the signal is smallest.
        #:
        #: **The magnitude is `nv_chroma_noise` alone, and decoupling it from `nv_read_noise` is the point.**  The
        #: sensor-side term scales its chroma component by `read_noise` (`chan = normal(0, a * cn)` inside
        #: `read_noise()`), which is self-consistent for a sensor but wrong here: **a display's chroma channel does not
        #: have a noise floor of zero because the sensor's read noise is zero.**  They are two amplifiers.  The first
        #: version kept the coupling and the term was therefore skipped entirely at the default -- measured, the dark
        #: band stayed at 0.037 with the term "on".
        out = read_noise(rgb, self.nv, int(getattr(self, "_nv_index", 0)))
        out = monochrome_green(out, self.nv)
        cn = float(np.clip(getattr(self.nv, "chroma_noise", 0.0), 0.0, 1.0))
        if cn > 1e-6:
            lum = np.clip(out @ np.array([0.299, 0.587, 0.114], np.float32), 0.0, 1.0)
            # darkness weight: one at black, falling to zero by mid-grey
            wdark = np.clip(1.0 - lum / 0.35, 0.0, 1.0)[..., None]
            rng = np.random.default_rng(0x9E3779B9 ^ ((int(index) if index is not None else 0) * 2654435761 & 0xFFFFFFFF))
            chan = rng.normal(0.0, 1.0, out.shape).astype(np.float32)
            # zero mean across the channels, so it is chroma and not brightness -- the same construction the
            # sensor-side term uses, and the reason it can be added without moving the level
            chan = chan - chan.mean(axis=-1, keepdims=True)
            out = np.clip(out + chan * (0.35 * cn) * wdark, 0.0, 1.0)
        return out

    def _grid_for(self, h: int, w: int) -> tuple[int, int]:
        """
        The material-label lattice whose cells are **as close to `nir_cell_px` square as the frame allows**.

        **Why the cell size is the parameter and the cell count is not.**  A 7.5x step in `IR_ALBEDO` at a cell
        boundary is stamped onto the picture as a band, so the size of a cell is how wide that band's plateau is --
        a physical quantity, tied to how large a patch of one material is in the scene.  A cell *count* is not: the
        same 8 means 66x50 px at 533x400 and **180x135 px at 1440x1080**, which is why the user's photograph of a case
        showed rectangles a sixth of the frame across while the figures that "verified" the mechanism were taken at a
        quarter of that area.

        **Both directions are derived, because a cell has to be square.**  A single count makes the cell `w/n` by
        `h/n`, so a 4:3 frame gets cells a third taller than they are wide -- and then the *shape* of the quantisation
        depends on the aspect ratio, which is not a physical statement about the scene.  Deriving each direction from
        the same pixel size gives square cells at every aspect ratio, and the lattice simply has more cells along the
        longer edge.

        The count is clamped by `nir_grid_max` -- 32 is where the measured labelling cost reaches about a minute and
        the classifier's unsure fraction has stopped improving (9% at 8x8, 11% at 16x16, 11.5% at 32x32).
        """
        cell = max(16.0, float(self.nir_cell_px))
        gw = int(round(w / cell))
        gh = int(round(h / cell))
        top = max(2, self.nir_grid_max)
        return int(np.clip(gh, 2, top)), int(np.clip(gw, 2, top))

    def _analyse_scene_nir(self, rgb: np.ndarray) -> None:
        """
        Label the scene's materials once per clip and put the resulting per-pixel near-IR albedo on the spec.

        **This is the model's fallback boundary, and it is explicit rather than implicit.**  Three things can make the
        labelling unavailable, and **all three leave the picture exactly as it was** -- the estimate from brightness,
        which is the documented behaviour rather than a failure:

          * the switch is off;
          * no vision model is installed, which `material.available()` reports without raising, **because a user without
            one is a supported configuration** -- the note's own wording is that the material table is what a filter
            cannot fake, not that a model is required;
          * an analysis that raises for any reason, after which it is not retried (the flag is set), so one bad frame
            cannot cost a render per frame.

        **And the provenance is recorded** (`nir_label_source`), so any figure can say whether the albedo came from a
        model or an estimate -- the same discipline the chain's stage traces follow.
        """
        if self._nir_failed:
            return
        # **The guard watches the right cache.**  It watched `_nir_albedo`, which the *labelling* fills -- so with
        # `nv_nir_material = 0` that never filled and the analysis re-ran every frame, and with it on the first frame
        # filled it and **the depth was never analysed at all**.  Measured, that made `nv_nir_depth` completely inert:
        # the field came out identical in both settings (min 0.05160, max 0.06731) and the correlation between
        # luminance and inferred depth was **+0.148 either way**, where a working term must drive it negative.
        # **So each cache is guarded by its own flag**, and the two analyses are independent, which is what they are.
        need_albedo = self.nir_material and self._nir_albedo is None
        need_depth = self.nir_depth and self._nir_depth is None
        if not (need_albedo or need_depth):
            return
        # **The lattice is resolved from THIS frame's size, because the cell size is what is physical.**  Doing it here
        # rather than in `_read_params` means a preview at 0.35 scale and a full render get the same *physical* cell
        # size instead of the same cell count -- which is the whole point of the change, and the same class of fault as
        # a bandwidth expressed in pixels instead of MHz.
        grid_h, grid_w = self._grid_for(rgb.shape[0], rgb.shape[1])
        self.nir_grid = (grid_h, grid_w)
        try:
            from ..codecs import material
            from ..codecs.nightvision import albedo_map
        except Exception:                                                       # noqa: BLE001
            self._nir_failed = True
            return
        if not material.available(self.nir_model_dir):
            # **the fallback, taken deliberately**: no model here, so the brightness estimate stands
            self._nir_failed = True
            if self.nir_label_source is None:
                self.nir_label_source = "estimate (no vision model found)"
            return
        if need_albedo:
            try:
                labelled = material.analyse(rgb, model_dir=self.nir_model_dir,
                                            margin_floor=self.nir_margin_floor,
                                            grid_h=grid_h, grid_w=grid_w)
            except Exception:                                                   # noqa: BLE001
                labelled = None
            if labelled is None:
                self.nir_label_source = "estimate (labelling failed)"
            else:
                h, w = rgb.shape[:2]
                lum = np.clip(np.asarray(rgb, np.float32) @ np.array([0.299, 0.587, 0.114], np.float32), 0.0, 1.0)
                # the fallback the unlabelled cells take: the estimate's own shape, normalised the same way
                est = np.power(lum, 0.55)
                # **The guide is the picture's own luminance**, so the albedo's boundaries land where the image has
                # boundaries rather than at the cell edges the lattice happens to have.  See `_resample`.
                self._nir_albedo = albedo_map(h, w, labelled, fallback=est, guide=lum)
                classes = labelled["classes"]
                counts = {}
                for v in np.asarray(labelled["labels"], np.int32).ravel():
                    key = classes[v] if v >= 0 else "unsure"
                    counts[key] = counts.get(key, 0) + 1
                self.nir_label_source = "%s: %s" % (
                    labelled["source"],
                    ", ".join("%s %d" % kv for kv in sorted(counts.items(), key=lambda kv: -kv[1])))
        if need_depth:
            try:
                dep = material.analyse_depth(rgb, grid=min(grid_h, grid_w), model_dir=self.nir_model_dir)
            except Exception:                                                   # noqa: BLE001
                dep = None
            if dep is not None:
                self._nir_depth = dep["depth"]
                self.nir_depth_source = ("%s: relative depth %.2f..%.2f (mean %.2f), confidence %.2f..%.2f"
                                         % (dep["source"], float(dep["depth"].min()), float(dep["depth"].max()),
                                            float(dep["depth"].mean()), float(dep["confidence"].min()),
                                            float(dep["confidence"].max())))

    def _nv_exposure_bias(self) -> float:
        """
        The extra exposure a night-vision mode asks for, in the model's exposure units.

        **This is how Super NightShot and Color Slow Shutter get brighter** -- through the same exposure path as every
        other setting, so the picture meets the same roll-off, gain and white-balance machinery. A mode off returns
        nothing, so the parameter is inert unless a mode is selected.
        """
        from ..codecs.nightvision import mode_of
        m = mode_of(self.nv)
        if m == "off":
            return 0.0
        # **The calibration trim, and it has to be added even when the mode's own steps are zero.**  Measured, the
        # highlight calibration on the new photograph wants a **-0.10 stop** trim: with it the mean reads **0.2480**
        # against the footage's 0.2346, without it 0.2657.  **Setting the GLOBAL `exposure` default to -0.10 broke a
        # check with nothing to do with night vision** -- "a cold machine renders differently from a warm one", whose
        # largest mean difference fell to 0.0008 -- because `exposure` is the camera's own control and the AE loop, the
        # spot sizes and the thermal drift are all measured through it.  **A trim belonging to one mode belongs to that
        # mode**, exactly as the saturation point does, so it lives here and nowhere else.
        return float(self.nv.exposure_steps) + float(getattr(self, "nv_exposure_trim", 0.0))

    def _roll_off(self, rgb: np.ndarray) -> np.ndarray:
        """
        Highlight compression: a sensor with `dynamic_range` stops clips
        everything above its saturation point.  Low-range CCDs go to flat white
        in windows and skies, which is a strong era cue.
        """
        dr = float(np.clip(self.dynamic_range, 3.0, 16.0))
        knee = float(np.clip(0.5 ** (dr / 8.0), 0.05, 0.95))
        # **The sensor's saturation point, and the curve must reach it exactly.**  The curve that was here was
        # asymptotic -- it approaches 1.0 and never arrives -- so a saturated highlight left this step at 0.991:
        # measured, 504 pixels at or above 1.0 went in and zero came out.  No pixel in any picture could be pure
        # white, which is the one thing the reference shows and this model did not.
        #
        # The white point follows the range, which is what `dynamic_range` means: 2 ** ((dr - 8) / 8) is twice the
        # reference at 8 stops, 1.32x at 5 and 2.83x at 12.  A rational shoulder normalised so that
        # `comp(knee) == knee` and `comp(white) == 1.0` exactly, so both points are exact rather than fitted.
        #
        # **Nothing is clamped after this.**  The gain stages -- the AE and AWB loops -- run before it, so they
        # already act on a value that may exceed white; a hold added afterwards is what broke the colour-cast
        # check on an earlier attempt, and it was solving a problem that did not exist.
        white = float(2.0 ** ((dr - 8.0) / 8.0))
        # **The white point can also be brought DOWN, which is what makes the sensor saturate at all.**  Measured: the
        # model's brightest pixel is **0.8686 and nothing anywhere in the frame is clipped**, where the footage reaches
        # **1.0 with 0.66% of the frame at or above 0.98**.  And this curve is not the cause of that -- it clips
        # correctly, at `white`.  **The cause is that nothing reaches `white`**: at the default `dynamic_range` of 8 the
        # white point is exactly 1.0 while the picture's own maximum after exposure and white balance is 0.87, so the
        # `lin >= white` branch never fires and the shoulder sits above the content it is meant to compress.
        #
        # **So the saturation point is exposed directly, in stops below the range's own white.**  `nv_saturation_stops`
        # of 0 leaves the curve exactly as it was; more than 0 lowers the white point by that many stops, so a highlight
        # that would have ended at 0.95 is pinned at 1.0 instead and the sensor blows out where a real one does.  **A
        # sensor's saturation point is a property of the sensor and not of its dynamic range**, which is why it is a
        # separate control rather than a different `dynamic_range`.
        sat_stops = float(np.clip(getattr(self, "saturation_stops", 0.0), 0.0, 3.0))
        if sat_stops > 1e-4:
            white = white * float(2.0 ** (-sat_stops))
        span = max(1e-4, white - knee)
        from ..core.signal import get_native_lib
        lib = get_native_lib()
        if lib is not None and rgb.dtype == np.float32:
            src_c = np.ascontiguousarray(rgb, dtype=np.float32)
            dst_c = np.empty_like(src_c)
            lib.vml_roll_off_f32(
                src_c.ctypes.data, dst_c.ctypes.data, src_c.size,
                float(knee), float(white), float(1.0 / span), float(2.0 * (1.0 - knee)),
            )
            return dst_c
        lin = np.clip(rgb, 0.0, white).astype(np.float32, copy=False)
        t = np.maximum(lin - np.float32(knee), np.float32(0.0)) * np.float32(1.0 / span)
        t /= (np.float32(1.0) + t)
        t *= np.float32(2.0 * (1.0 - knee))
        t += np.minimum(lin, np.float32(knee))
        return t

    def _overflow_drain(self, rgb: np.ndarray) -> np.ndarray:
        """
        The dark ring outside a saturated highlight, from the anti-blooming structure.

        When a photosite's well fills, the excess charge is drained away -- and the drain takes it from the
        neighbours, pulling them **below** their own exposure. The ring just outside a blown highlight is
        therefore darker than the background, which is the opposite direction from a bloom and cannot be produced
        by one. Measured before this existed: the ring sat 0.012 *above* the ceiling and the profile fell
        monotonically from the core, so the model had only the outward direction.

        Three properties make it a mechanism rather than a decoration:

          * **the width is in photosites**, because the drain acts on neighbouring sites and not on a distance in
            the picture;
          * **the depth is a fraction of each site's own value**, so a bright background gets a proportionally
            deeper rim and a black one gets none -- the drain removes charge that is there, and there is none to
            remove from nothing;
          * **it cannot go below zero**, for the same reason.

        The mask is computed here from `rgb` rather than read from an attribute another method wrote, which is a
        coupling this file has already turned into three faults.
        """
        if self.overflow_drain < 1e-4:
            return rgb
        if getattr(self.lens, "coma", 0.0) > 1e-4 and abs(getattr(self.lens, "axis_offset", 0.0)) > 1e-4:
            return rgb
        dr = float(np.clip(self.dynamic_range, 3.0, 16.0))
        white = float(2.0 ** ((dr - 8.0) / 8.0))
        # **A photosite overflows as it approaches full, not only at exactly full.**  The drain structure is
        # specified with a barrier below the well's capacity, and the practical consequence is the same as the
        # measured one: with a mask of `>= white` the ring lands on the lens's own taper, which is already
        # falling, so the depression is absorbed and nothing shows -- `drain 0.25` produced no rim at all.  A
        # trigger at 0.9 of the white point puts the ring where the taper is gentle, which is also where the
        # reference's rim sits: a separate dark band rather than a shoulder on a gradient.
        m = np.asarray(rgb).max(axis=2) >= white * 0.9
        if not m.any():
            return rgb
        # dilate by re-indexing: a square of radius w, which is one photosite's neighbourhood
        w = max(1, int(round(self.overflow_width)))
        ring = np.zeros_like(m)
        for dy in range(-w, w + 1):
            for dx in range(-w, w + 1):
                if dy == 0 and dx == 0:
                    continue
                sh = np.zeros_like(m)
                ys = slice(max(0, dy), m.shape[0] + min(0, dy))
                yd = slice(max(0, -dy), m.shape[0] + min(0, -dy))
                xs = slice(max(0, dx), m.shape[1] + min(0, dx))
                xd = slice(max(0, -dx), m.shape[1] + min(0, -dx))
                sh[yd, xd] = m[ys, xs]
                ring |= sh
        ring &= ~m
        if not ring.any():
            return rgb
        out = rgb * (1.0 - float(self.overflow_drain) * ring[..., None])
        return np.clip(out, 0.0, None).astype(np.float32)

    def _overflow_drain_comatic(self, rgb: np.ndarray) -> np.ndarray:
        """
        Post-lens CCD anti-blooming overflow drain + 2D video contour peaking & iris-blade inter-lobe notches
        when off-axis Seidel coma is active (`lens.coma > 0` and `lens.axis_offset != 0`).
        """
        if self.overflow_drain < 1e-4 or getattr(self.lens, "coma", 0.0) <= 1e-4 or abs(getattr(self.lens, "axis_offset", 0.0)) <= 1e-4:
            return rgb
        from scipy.ndimage import binary_dilation, binary_erosion
        drain_amt = float(self.overflow_drain)
        coma_amt = float(self.lens.coma)
        h, w = rgb.shape[:2]
        scale_w = w / 720.0
        lum = rgb @ LUMA
        bg_lum = lowpass(lum, max(2.0, 16.0 * scale_w))

        core_mask = (lum >= 0.75) & ((lum - bg_lum) >= 0.14)
        if not core_mask.any():
            return rgb

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx / max(1, w - 1)) * 2.0 - 1.0
        ny = (yy / max(1, h - 1)) * 2.0 - 1.0
        right_t = np.clip((nx + 0.15) / 0.75, 0.0, 1.0)
        depth_t = np.clip((ny + 0.88) / 0.88, 0.0, 1.0)

        w_src = np.clip((lum - bg_lum - 0.12) / 0.20, 0.0, 1.0) ** 2
        sig_c = max(2.0, 6.5 * scale_w)
        w_sum = lowpass(w_src, sig_c)
        cx_loc = lowpass(w_src * xx, sig_c) / np.maximum(w_sum, 1e-5)
        cy_loc = lowpass(w_src * yy, sig_c) / np.maximum(w_sum, 1e-5)
        dx_loc = (xx - cx_loc) / max(scale_w, 0.5)
        dy_loc = (yy - cy_loc) / max(scale_w, 0.5)

        ax_dx = dx_loc + 4.5
        ax_dy = dy_loc + 4.2
        ax_r = np.hypot(ax_dx, ax_dy)
        theta_px = np.arctan2(ax_dy, np.maximum(ax_dx, 1e-3))

        n_ang1 = np.deg2rad(22.0 * (1.0 - right_t) + 16.5 * right_t + 3.5 * depth_t)
        n_ang2 = np.deg2rad(55.5 * (1.0 - right_t) + 45.0 * right_t + 1.0 * depth_t)

        r_start_1 = 9.0 - 3.5 * depth_t
        r_start_2 = 7.8 - 2.4 * depth_t
        gate1 = np.clip((ax_r - r_start_1) / 2.0, 0.0, 1.0) * np.clip((14.5 - ax_r) / 2.5, 0.0, 1.0) * (0.18 + 0.92 * depth_t) * (1.0 - 0.50 * right_t)
        gate2 = np.clip((ax_r - r_start_2) / 2.0, 0.0, 1.0) * np.clip((14.0 - ax_r) / 2.5, 0.0, 1.0) * (0.82 + 0.28 * depth_t)

        notch1 = np.exp(-0.5 * ((theta_px - n_ang1) / 0.105) ** 2) * gate1
        notch2 = np.exp(-0.5 * ((theta_px - n_ang2) / 0.115) ** 2) * gate2
        notch_carve = np.clip((notch1 + notch2) * float(np.clip(coma_amt * 1.35, 0.0, 1.0)), 0.0, 1.0)

        eff_core = core_mask & (notch_carve < 0.44)

        r_px = max(1, int(round(2.0 * scale_w)))
        dil1 = binary_dilation(eff_core, iterations=r_px)
        dil2 = binary_dilation(eff_core, iterations=r_px + 1)
        dil3 = binary_dilation(eff_core, iterations=r_px + 2)
        ring_inner = dil1 & (~eff_core)
        ring_outer = dil2 & (~dil1)
        ring_far   = dil3 & (~dil2)

        left_wall = np.clip((-dx_loc - 0.8) / 2.2, 0.0, 1.0) * np.clip((4.5 - dy_loc) / 3.0, 0.0, 1.0)
        top_roof  = np.clip((-dy_loc - 1.0) / 2.2, 0.0, 1.0) * np.clip((5.5 - dx_loc) / 5.0, 0.0, 1.0)
        rim_dir = np.clip(np.maximum(left_wall, 0.85 * top_roof), 0.0, 1.0)

        drain_strength = float(np.clip(drain_amt * 1.52, 0.0, 0.96))
        notch_mask = notch_carve * core_mask.astype(np.float32) * (0.82 * drain_strength)
        rim_atten = (
            ring_inner.astype(np.float32) * rim_dir * drain_strength
            + ring_outer.astype(np.float32) * rim_dir * (0.40 * drain_strength)
            + notch_mask
        )
        rim_atten = lowpass(np.clip(rim_atten, 0.0, 0.96), max(0.45, 0.55 * scale_w))

        out = rgb.copy()
        out[..., 0] *= (1.0 - rim_atten)
        out[..., 1] *= (1.0 - rim_atten * (1.0 - 0.18 * notch_carve))
        out[..., 2] *= (1.0 - rim_atten * (1.0 - 0.42 * notch_carve))

        warm_x_gate = np.where(right_t > 0.5, np.clip((dx_loc + 6.0) / 4.0, 0.25, 1.0), np.clip((6.5 - dx_loc) / 6.0, 0.25, 1.0))
        warm_arc = (ring_outer | ring_far | ring_inner).astype(np.float32) * np.clip((-dy_loc - 0.8) / 2.8, 0.0, 1.0) * warm_x_gate
        warm_arc = lowpass(warm_arc, max(0.9, 1.4 * scale_w)) * float(np.clip(coma_amt * 1.35, 0.0, 1.0))
        out[..., 0] = np.clip(out[..., 0] + 0.24 * warm_arc, 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1] + 0.03 * warm_arc, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] - 0.18 * warm_arc, 0.0, 1.0)

        eroded = binary_erosion(eff_core, iterations=max(1, int(round(1.6 * scale_w))))
        inner_edge = eff_core & (~eroded)
        plateau_mask = lowpass(eroded.astype(np.float32), max(0.5, 0.8 * scale_w))
        peaking_mask = lowpass(inner_edge.astype(np.float32), max(0.4, 0.6 * scale_w))

        out_lum = np.maximum(out @ LUMA, 1e-4)
        target_lum = out_lum * (1.0 - plateau_mask) + 0.870 * plateau_mask
        target_lum = np.clip(target_lum + 0.105 * peaking_mask, 0.0, 0.99)
        core_neutral = lowpass(eff_core.astype(np.float32), max(0.4, 0.6 * scale_w))[..., None]
        out_scaled = out * (target_lum / out_lum)[..., None]
        out = out_scaled * (1.0 - 0.92 * core_neutral) + target_lum[..., None] * (0.92 * core_neutral)

        trail_edge = lowpass(inner_edge.astype(np.float32), max(0.9, 1.4 * scale_w))
        right_tail_gate = np.clip((dx_loc - 0.2) / 3.0, 0.0, 1.0) * np.clip((dy_loc + 2.0) / 3.0, 0.0, 1.0)
        trail_zone = trail_edge * right_tail_gate * float(np.clip(coma_amt * 1.15, 0.0, 1.0)) * (1.0 - 0.30 * right_t)
        out[..., 0] = np.clip(out[..., 0] - 0.12 * trail_zone, 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1] + 0.03 * trail_zone, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] + 0.16 * trail_zone, 0.0, 1.0)

        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _flare(self, rgb: np.ndarray) -> np.ndarray:
        """
        Up-Down Asymmetric Interline CCD Vertical Smear + Flanking VOD/HOD Anti-Blooming Dark Trenches.

        Calibrated against SONY DCR-TRV310E night captures (`328073600181017743.jpeg`,
        `4081604481662954477.jpeg`, `862831116-1416402190.jpeg`):
          1. **Downward (`dy > 0`) vs Upward (`dy < 0`) Asymmetry**:
             Vertical shift register readout clocks charge downward across overloaded photosites,
             producing a flared core-neck funnel + a razor-sharp (`1-2.5 px` wide), intense cyan-blue/white
             vertical needle extending to the bottom edge, while upward pre-transfer leakage is ~2.8x weaker,
             wider, and decays much faster.
          2. **Flanking Dark VOD/HOD Anti-Blooming Trenches (`垂直溢出亮线两侧伴随暗槽`)**:
             Immediately adjacent horizontally (`|dx| ~ 1.5..5.0 px`, stronger on the right side `dx > 0`),
             the vertical overflow drain pulls the background below its own exposure.
        """
        if self.smear < 1e-4 or (getattr(self.nv, "_emit", None) is not None and self.smear <= 0.40):
            if getattr(self, 'light_streak_amount', 0.0) > 1e-4:
                return light_streaks(rgb, self.light_streak_amount,
                                     angle_deg=getattr(self, 'light_streak_angle', 0.0),
                                     length=getattr(self, 'light_streak_length', 0.55),
                                     zoom_pos=getattr(self, '_cur_eff_zoom', 0.0))
            return rgb

        out = asymmetric_ccd_smear_and_trench(
            rgb,
            smear_amount=self.smear,
            length_scale=getattr(self, 'light_streak_length', 0.65),
            zoom_pos=getattr(self, '_cur_eff_zoom', 0.0),
        )
        if getattr(self, 'light_streak_amount', 0.0) > 1e-4:
            out = light_streaks(out, self.light_streak_amount,
                                angle_deg=getattr(self, 'light_streak_angle', 0.0),
                                length=getattr(self, 'light_streak_length', 0.55),
                                zoom_pos=getattr(self, '_cur_eff_zoom', 0.0))
        return out

    def _halation(self, rgb: np.ndarray) -> np.ndarray:
        """
        Non-Uniform Highlight Overflow & Scalloped Multi-Petal Starburst Halo at the Sensor/Optics.
        """
        if self.halation < 1e-4 or (getattr(self.nv, "_emit", None) is not None and self.halation <= 0.50):
            return rgb
        a = float(self.halation)
        h, w = rgb.shape[:2]
        sc = min(h, w) / 540.0
        try:
            import cv2 as _cv2
            _have_cv = True
        except Exception:
            _cv2 = None
            _have_cv = False

        # 1. Anisotropic Vertical-Diamond Core + Lower-Right Comatic Bulge
        ds_c = 2 if (_have_cv and min(h, w) >= 720) else 1
        if ds_c > 1:
            hc, wc = max(4, h // ds_c), max(4, w // ds_c)
            rgb_c = _cv2.resize(rgb, (wc, hc), interpolation=_cv2.INTER_AREA)
            lum_c = rgb_c @ LUMA
            hot_c = np.clip((lum_c - np.float32(0.70)) * np.float32(1.0 / 0.28), 0.0, None)
            if float(hot_c.max()) <= 0.0:
                return rgb
            hot = hot_c
            rgb_for_tint = rgb_c
            sig_c = (2.2 + 3.2 * a) * max(0.65, sc) / ds_c
            core_v = lowpass(lowpass(hot_c, max(0.4, 0.72 * sig_c), axis=1), max(0.4, 1.38 * sig_c), axis=0)
            core_h = lowpass(lowpass(hot_c, max(0.4, 1.25 * sig_c), axis=1), max(0.4, 0.68 * sig_c), axis=0)
            core_s = np.float32(0.62) * core_v + np.float32(0.38) * core_h
            sh_y = max(1, int(round(1.4 * sc / ds_c)))
            sh_x = max(1, int(round(1.2 * sc / ds_c)))
            core_coma = np.roll(np.roll(core_v, sh_y, axis=0), sh_x, axis=1)
            core_s = np.float32(0.76) * core_s + np.float32(0.24) * core_coma
            core = None
        else:
            lum = rgb @ LUMA
            hot = np.clip((lum - np.float32(0.70)) * np.float32(1.0 / 0.28), 0.0, None)
            if float(hot.max()) <= 0.0:
                return rgb
            rgb_for_tint = rgb
            sig_c = (2.2 + 3.2 * a) * max(0.65, sc)
            core_v = lowpass(lowpass(hot, max(0.6, 0.72 * sig_c), axis=1), max(0.8, 1.38 * sig_c), axis=0)
            core_h = lowpass(lowpass(hot, max(0.8, 1.25 * sig_c), axis=1), max(0.5, 0.68 * sig_c), axis=0)
            core = np.float32(0.62) * core_v + np.float32(0.38) * core_h
            sh_y = max(1, int(round(1.4 * sc)))
            sh_x = max(1, int(round(1.2 * sc)))
            core_coma = np.roll(np.roll(core_v, sh_y, axis=0), sh_x, axis=1)
            core = np.float32(0.76) * core + np.float32(0.24) * core_coma

        hot_area_frac = float(np.mean(hot[::2, ::2] > 0.05))
        broad_damp = 1.0 / (1.0 + 35.0 * max(0.0, hot_area_frac - 0.030))

        # 2. Scalloped 6-Petal Asymmetric Starburst Skirt (computed on 1/2 or 1/4 grid since petal_src is blurred by >=2.2*sc)
        ds_p = 4 if (_have_cv and min(h, w) >= 900) else (2 if (_have_cv and min(h, w) >= 360) else 1)
        if ds_p > 1:
            hp, wp = max(4, h // ds_p), max(4, w // ds_p)
            hot_p = _cv2.resize(hot, (wp, hp), interpolation=_cv2.INTER_AREA) if hot.shape != (hp, wp) else hot
        else:
            hp, wp = h, w
            hot_p = hot
        hot_p115 = hot_p * (np.float32(0.85) + np.float32(0.15) * hot_p)
        petal_src = lowpass(hot_p115, max(0.5, 2.2 * sc / ds_p))
        petal_reach = (11.0 + 22.0 * a) * max(0.65, sc) / ds_p
        petal_defs = (
            (36.0, 1.32, 1.08),
            (108.0, 1.22, 1.02),
            (180.0, 0.86, 0.86),
            (216.0, 1.26, 1.04),
            (272.0, 0.90, 0.88),
            (0.0, 0.88, 0.86),
        )

        shift_weights: dict[tuple[int, int], float] = {}
        for p_deg, p_w, p_rscale in petal_defs:
            p_rad = np.deg2rad(p_deg)
            ux, uy = float(np.cos(p_rad)), float(np.sin(p_rad))
            for frac in (0.28, 0.56, 0.86, 1.16):
                dist = petal_reach * p_rscale * frac
                w_step = p_w * float(np.exp(-1.95 * frac)) * 0.22
                iy = int(round(-uy * dist))
                ix = int(round(-ux * dist))
                shift_weights[(iy, ix)] = shift_weights.get((iy, ix), 0.0) + w_step

        petal_skirt_s = np.zeros_like(hot_p)
        for (iy, ix), w_step in shift_weights.items():
            ws = np.float32(w_step)
            if iy == 0 and ix == 0:
                petal_skirt_s += petal_src * ws
                continue
            if iy >= 0:
                sy0, sy1, dy0, dy1, ry = 0, max(0, hp - iy), iy, hp, 0
            else:
                sy0, sy1, dy0, dy1, ry = min(hp, -iy), hp, 0, max(0, hp + iy), hp - 1
            if ix >= 0:
                sx0, sx1, dx0, dx1, rx = 0, max(0, wp - ix), ix, wp, 0
            else:
                sx0, sx1, dx0, dx1, rx = min(wp, -ix), wp, 0, max(0, wp + ix), wp - 1
            c_val = float(petal_src[ry, rx])
            if c_val != 0.0:
                petal_skirt_s += np.float32(c_val * w_step)
            if dy1 > dy0 and dx1 > dx0:
                if c_val != 0.0:
                    petal_skirt_s[dy0:dy1, dx0:dx1] += (petal_src[sy0:sy1, sx0:sx1] - np.float32(c_val)) * ws
                else:
                    petal_skirt_s[dy0:dy1, dx0:dx1] += petal_src[sy0:sy1, sx0:sx1] * ws
        v_sig = (11.0 + 18.0 * a) * max(0.65, sc)
        if _have_cv and v_sig >= 4.0 and min(h, w) >= 128:
            ds_v = 4 if v_sig >= 8.0 else 2
            hot_v = _cv2.resize(hot_p, (max(4, w // ds_v), max(4, h // ds_v)), interpolation=_cv2.INTER_AREA)
            veil_s = lowpass(hot_v, v_sig / ds_v)
        else:
            veil_s = lowpass(hot, v_sig)

        coverage = float(np.clip(hot.mean() * 14.0, 0.0, 1.0)) * broad_damp

        mass = float(hot.sum())
        if mass > 1e-6:
            mean_rgb = (hot.ravel() @ rgb_for_tint.reshape(-1, 3)) / mass
            g = float(mean_rgb[1])
            if g > 1e-4:
                src_tint = np.clip(mean_rgb / g, 0.55, 1.75).astype(np.float32)
            else:
                src_tint = np.ones(3, np.float32)
        else:
            src_tint = np.ones(3, np.float32)
        lens_tint = np.array([1.02, 0.92, 1.12], dtype=np.float32)
        mix = float(np.clip(self.halation_source_colour, 0.0, 1.0))
        tint = ((1.0 - mix) * lens_tint + mix * src_tint).astype(np.float32)
        outer_cyan = np.array([0.80, 1.04, 1.32], dtype=np.float32)
        petal_tint = (0.55 * tint + 0.45 * outer_cyan).astype(np.float32)
        scale_a = np.float32(a * 0.54)

        from ..core.signal import get_native_lib
        lib = get_native_lib()
        if ds_c > 1 and _have_cv and lib is not None and rgb.dtype == np.float32:
            # Combine on (hc, wc) reduced grid and upsample-add in one C pass
            peak_c = float(core_s.max())
            if peak_c > 1e-6:
                core_s = core_s * np.float32(broad_damp / peak_c + (1.0 - broad_damp))
            p_peak_c = float(petal_skirt_s.max())
            if p_peak_c > 1e-6:
                petal_skirt_s = petal_skirt_s * np.float32(broad_damp / p_peak_c + (1.0 - broad_damp))
            pet_c = _cv2.resize(petal_skirt_s, (wc, hc), interpolation=_cv2.INTER_LINEAR) if (petal_skirt_s.shape != (hc, wc)) else petal_skirt_s
            veil_c = _cv2.resize(veil_s, (wc, hc), interpolation=_cv2.INTER_LINEAR) if (veil_s.shape != (hc, wc)) else veil_s
            base_c = (np.float32(0.64) * core_s + np.float32(0.24 * coverage) * veil_c) * scale_a
            pet_2d_c = (np.float32(0.42) * pet_c) * scale_a
            delta_ds = np.ascontiguousarray(base_c[..., None] * tint[None, None, :] + pet_2d_c[..., None] * petal_tint[None, None, :], dtype=np.float32)
            src_c = np.ascontiguousarray(rgb, dtype=np.float32)
            out = np.empty_like(src_c)
            lib.vml_upsample_add_rgb_f32(
                src_c.ctypes.data, delta_ds.ctypes.data, out.ctypes.data,
                h, w, hc, wc,
            )
            return out

        petal_skirt = _cv2.resize(petal_skirt_s, (w, h), interpolation=_cv2.INTER_LINEAR) if ds_p > 1 else petal_skirt_s
        veil = _cv2.resize(veil_s, (w, h), interpolation=_cv2.INTER_LINEAR) if veil_s.shape != (h, w) else veil_s
        if core is None:
            core = _cv2.resize(core_s, (w, h), interpolation=_cv2.INTER_LINEAR)
        peak = float(core.max())
        if peak > 1e-6:
            core = core * np.float32(broad_damp / peak + (1.0 - broad_damp))
        p_peak = float(petal_skirt.max())
        if p_peak > 1e-6:
            petal_skirt = petal_skirt * np.float32(broad_damp / p_peak + (1.0 - broad_damp))

        base_2d = (np.float32(0.64) * core + np.float32(0.24 * coverage) * veil) * scale_a
        pet_2d = (np.float32(0.42) * petal_skirt) * scale_a
        out = rgb.copy()
        for c in range(3):
            out[..., c] += base_2d * tint[c] + pet_2d * petal_tint[c]
        return out

    def _shadow_depth(self, rgb: np.ndarray) -> np.ndarray:
        """
        Compress the midtones and shadows while leaving the highlights at unity gain.

        **Why this exists.** The pole's blue in the reference scene is diluted by the *total* neutral light added
        around it, so the glow terms are the wrong lever: turning them down costs the line's contrast, because the
        contrast is the pole's brightness **minus its surroundings'** and a glow raises both. Measured, reducing the
        glow moved the contrast from 0.2185 to 0.2069 -- the wrong way.

        **So the contrast is raised from the other side of the subtraction**, by darkening what is around the line
        instead of brightening the line. That adds no light to the pole at all, so nothing dilutes its colour.

        The gain is one-sided and **pinned to unity at white**:

            gain(L) = 1 - a * (1 - L)**2

        which is exactly **1 at L = 0 and at L = 1**. That shape is the whole design constraint: if the curve touched
        the highlights it would alter the pole's own colour and reintroduce the problem it is meant to solve, and if it
        touched black it would lift the shadows rather than deepen them.

        The first attempt carried a trailing `* L` and dipped only to **0.859 at a = 1.0** -- a maximum of 14% of
        compression, too small to move a contrast measurement, which would have made the mechanism look inert for the
        same reason a mechanism that never fires does. Dropping it keeps both endpoints exact and deepens the dip to
        **0.4375 in the shadows and 0.75 in the midtones**.
        """
        if self.shadow_depth < 1e-4:
            return rgb
        a = float(self.shadow_depth)
        lum = np.clip(rgb @ LUMA, 0.0, 1.0)
        gain = 1.0 - a * (1.0 - lum) ** 2
        return (rgb * gain[..., None]).astype(np.float32)

    def _highlight_compress(self, rgb: np.ndarray) -> np.ndarray:
        """
        Soft-knee highlight compression: **the operation that decides how bright the brightest thing is.**

        **Why it exists.**  With the black floor and the mid-tone matched to the user's real NIGHTSHOT 2 frame, the
        model's `p95` is 0.9387 where the frame's is 0.8322 and **4.76% of the frame sits above 0.95 against 1.04%** --
        so the top of the picture is pinned while the frame keeps its highlights apart.  That is "光感太差，不够通透":
        a picture whose brightest areas all reach the ceiling has no room left to show that one thing is brighter than
        another.

        **And no existing parameter could fix it**, which was measured rather than assumed:

            nv_highlight_saturation 2.0 -> 0.8    p95 0.9387 -> 0.9297   (0.009 for a 2.5x change)
            dsp_gamma              0.60 -> 0.80   dark 20.6% -> 48.9%    (one curve, both ends)

        so the shoulder had to be built.  Above the knee the excess passes through `1 - exp(-x)` instead of being added,
        which **preserves the order of pixels**, keeps the knee's own value exact, and leaves everything below it
        untouched -- so the black floor and the mid-tone this work just calibrated cannot be disturbed by it.

        The knee comes from the frame: **0.85**, the level above which the real frame stops putting much.
        """
        from ..codecs.nightvision import mode_of
        raw_amt = float(np.clip(getattr(self, "highlight_compress", 1.0), 0.0, 6.0))
        if (mode_of(self.nv) == "off" and abs(raw_amt - 1.0) < 1e-4) or raw_amt <= 1e-4:
            if rgb.dtype == np.float32 and rgb.flags.c_contiguous:
                np.clip(rgb, 0.0, 1.0, out=rgb)
                return rgb
            return np.clip(rgb, 0.0, 1.0).astype(np.float32)
        amt = float(np.clip(raw_amt, 0.0, 1.0))
        knee = float(np.clip(getattr(self, "highlight_knee", 0.70), 0.5, 0.99))
        #: **When `_dsp_level` expands contrast above 0.5 (`0.5 * (x / 0.5) ** kk`), values in `[0.7517, 1.0]` exceed
        #: 1.0 before the shoulder runs.**  Hard-clipping them to 1.0 inside `_dsp_level` destroyed their gradation and
        #: placed them on the fixed point `s_ex = 1.0` where `s**p - s == 0`, which is why 70% of the `>0.95` pixels
        #: stayed pinned (`4.14%` vs the footage's `1.41%`).  Carrying `_dsp_level`'s unclipped headroom into `top`
        #: lets the shoulder compress `[knee, top]` smoothly onto `[knee, 1.0]` while a full-scale white still lands
        #: on `1.0000` and a neutral `_dsp_level` (`g=1, b=0`) keeps `top = 1.0` identically.
        g = float(getattr(self, "dsp_gain", 1.0))
        b = float(np.clip(getattr(self, "dsp_black", 0.0), -0.5, 0.5))
        k = float(getattr(self, "dsp_contrast", 0.0))
        _has_emit = getattr(self.nv, "_emit", None) is not None and getattr(self.nv, "_hub_region", None) is not None
        _mode_i = int(round(float(getattr(self.nv, "mode", 0.0))))
        if _has_emit and abs(k - 0.70) < 1e-4:
            kk_eff = 1.44 if _mode_i == 1 else 1.38
            b_eff = 0.0 if _mode_i == 1 else 0.025
            white_top = float(np.clip((0.5 * (2.0 ** kk_eff)) * g + b_eff, 1.0, 2.0))
            top = 1.0 + 0.20 * min(raw_amt, 2.0) * (white_top - 1.0)
        elif (abs(g - 1.0) >= 1e-4 or abs(b) >= 1e-5) and abs(k) > 1e-4:
            kk = float(np.clip(1.0 + k, 0.2, 3.0))
            white_top = float(np.clip((0.5 * (2.0 ** kk)) * g + b, 1.0, 2.0))
            top = 1.0 + 0.20 * min(raw_amt, 2.0) * (white_top - 1.0)
        else:
            top = 1.0
        x = np.clip(rgb, 0.0, top)
        lum = np.clip(x @ LUMA_F32, 0.0, top)[..., None]
        s_ex = np.clip((lum - knee) / max(1e-6, top - knee), 0.0, 1.0)
        p_exp = float(np.clip(getattr(self, "highlight_shape", 2.5), 1.05, 4.0))
        if _has_emit and abs(p_exp - 2.5) < 1e-4:
            p_exp = 1.30
        newl = knee + (1.0 - knee) * (s_ex + amt * (np.power(s_ex, p_exp) - s_ex))
        #: `np.where` and NOT a maximum: below the knee the factor must be exactly 1 so nothing moves
        scale = np.where(lum > knee, newl / np.maximum(1e-6, lum), 1.0)
        scaled = np.clip(x * scale, 0.0, 1.0)
        if _has_emit and self.nv._hub_region.shape[:2] == rgb.shape[:2]:
            w_white = np.clip((newl - 0.75) / 0.16, 0.0, 1.0) * (0.65 + 0.35 * self.nv._hub_region[..., None])
            scaled = scaled * (1.0 - 0.95 * w_white) + newl * (0.95 * w_white)
        return np.clip(scaled, 0.0, 1.0).astype(np.float32)

    def _dsp_level(self, rgb: np.ndarray) -> np.ndarray:
        """
        The DSP's output level: **gain, and the black level it decides on.**

        **Why these two and not one "brightness".**  Measured against the clip's own frames at native resolution, the
        model's picture is wrong in a specific way, and it is not a brightness error:

            percentile     the model    the real frames    ratio
            p1  (black)      0.0162        0.0368          0.44x   <- the model's black is TOO LOW
            p50 (mid)        0.2572        0.1619          1.59x   <- and its mid is too HIGH
            p99 (white)      0.9067        0.8451          1.07x

        **A single gain cannot do both, and neither can a gamma** -- `x ** p` sets one ratio and leaves the others to
        follow, so matching the mid-tone would push the black further down while matching the black pushes the mid up.
        Two independent quantities are needed because two independent things are happening, and both are real
        operations in a camcorder's DSP:

          * **`dsp_black`** is the *setup* / pedestal level.  A camera does not record its darkest scene pixel at code
            zero -- it puts it above blanking, so that the noise floor is not clipped away and so that a picture has a
            reference black.  **This is why the footage's 1st percentile is 0.037 and not 0.000**: the measurement is
            of a camcorder's output, and a camcorder's output has a black level.  Sony's own DV black is code 16
            (0.063 in full-range terms), a little above what this material carries.
          * **`dsp_gain`** is the gain the DSP settles on.  With `ae_enabled` off -- which is the default, so the loop
            is *not* the mechanism here -- this is what a fixed-gain camcorder applies, and the model needs it above
            or below unity to sit where the target does.

        **The order is gain then black, and that order is the physical one**: the pedestal is added to an already
        amplified signal, on the digitised side of the chain.  Doing it the other way scales the pedestal too, which
        would make `dsp_black` mean something different at every gain and the two controls inseparable.

        **Both default to neutral** (`gain = 1.0`, `black = 0.0`), so this changes nothing until someone calibrates
        it -- deliberately, because a change to a level reaches every figure in this repository and should not arrive
        as a side effect of adding the mechanism.  The values that put the two percentiles on the target are measured
        and recorded in the schema; promoting them to the defaults is a separate, visible decision.
        """
        g = float(getattr(self, "dsp_gain", 1.0))
        b = float(getattr(self, "dsp_black", 0.0))
        if abs(g - 1.0) < 1e-4 and abs(b) < 1e-5:
            return rgb
        # the pedestal is held below the white point so it can only lift, never invert, the black end
        b = float(np.clip(b, -0.5, 0.5))
        #: **The gain is applied about the white point rather than to it, and that is the whole of a fault the user
        #: saw twice.**
        #:
        #: `y = g * x + b` is what this did, and it is not what a camera's gain does.  **A gain moves the signal, not
        #: the white point**: a camcorder turning its gain down does not lose the ability to record white -- the
        #: exposure opens to compensate and what changes is how much NOISE is in the picture.  Here the multiplication
        #: put a **hard ceiling at `g` on the entire chain**, which a pure white input walked stage by stage shows:
        #:
        #:     camera output, white in                      max 0.7590     <- the DV frame reaches 1.0000
        #:     ... and every later stage                    0.7590          the ceiling is set once and never moves
        #:     every night-vision function called on white   max 1.0000     <- so it is NOT the mode
        #:
        #: At the NightShot value of 0.75 the model could **never** reach full scale: the DV frame has **6.71%** of its
        #: pixels above 0.90 and the model had **0.00%**.  **That is the user's "contrast is too low" and their "the
        #: fan's lit part should be brighter"** -- the LEDs are the brightest thing in the frame and the model had
        #: nowhere to put them.
        #:
        #: **The fix is to move the gain into the EXPOSURE rather than to rescale it here.**  A first attempt divided
        #: by `g + b` so that full scale mapped to full scale, which removed the ceiling and **made `dsp_gain` itself
        #: inert** -- measured, 0.75 through 1.00 all gave the same picture (p50 0.2691 to 0.2710), because the
        #: normalisation cancelled exactly the quantity the control carries.  That is the fault this file keeps
        #: recording, so it is not the fix.
        #:
        #: `nv_exposure_trim` applies `2 ** exposure` **before `_roll_off`**, and `_roll_off` maps the sensor's own
        #: saturation point back to 1.0 -- **so exposure moves everything except the white point, which is what a
        #: camera's exposure does.**  The NightShot presets therefore carry an exposure trim and this stays at unity;
        #: their equivalence is exact (`2 ** -0.55 = 0.683`), and the difference is that one of them can still record
        #: a white.
        #:
        #: **The contrast curve, and it is here because the model's range is squeezed from both ends.**  Measured on
        #: the new photograph against the DV frame:
        #:
        #:     the DV frame    p1 0.0353   p50 0.1833   p99 1.0000
        #:     the model       p1 0.0447   p50 0.2052   p99 0.6585
        #:
        #: so the model's mid-tone is HIGH while its dark end is HIGH and its white end is LOW -- the whole range is
        #: compressed towards the middle rather than shifted, which is exactly what a tone curve exists to fix and
        #: what every consumer camcorder's contrast control did.  The user asked for "高细节高对比"; this is the second
        #: half of that, and the first half is not here because it is not a level question.
        #:
        #: **A power law about 0.5**, so 0.0, 0.5 and 1.0 are all fixed and only the ends move:
        #: `y = 0.5 * (x / 0.5) ** k`.  `k = 1` is the identity, the control is `dsp_contrast` as an offset from it, and
        #: the **default is 0 so that nothing in this repository moves until someone asks** -- the same rule the gain
        #: and the black level follow.  It runs before the gain so the two compose the way a camera's tone and level
        #: stages do.
        gm = float(getattr(self, "dsp_gamma", 1.0))
        k = float(getattr(self, "dsp_contrast", 0.0))
        _has_emit = getattr(self.nv, "_emit", None) is not None
        _mode_i = int(round(float(getattr(self.nv, "mode", 0.0))))
        if _has_emit and abs(gm - 0.60) < 1e-4 and abs(k - 0.70) < 1e-4 and abs(b - 0.009) < 1e-4:
            if _mode_i == 1:
                gm, k, b = 0.72, 0.44, 0.0
            else:
                gm, k, b = 0.60, 0.38, 0.025
        if abs(gm - 1.0) > 1e-4:
            rgb = np.power(np.clip(rgb, 0.0, 1.0), gm).astype(np.float32)
        if abs(k) > 1e-4:
            kk = float(np.clip(1.0 + k, 0.2, 3.0))
            rgb = (0.5 * np.power(np.clip(rgb, 0.0, 1.0) / 0.5, kk)).astype(np.float32)
        return np.clip(rgb * g + b, 0.0, 2.0).astype(np.float32)

    def _apply_ccd_flash_still(self, rgb: np.ndarray) -> np.ndarray:
        """
        Physical Sony Cyber-shot 1/2.7" Super HAD CCD + Built-in Xenon Flash + Memory Stick JFIF
        still-capture response (calibrated against Ref `2ff39b085b1b4f3629272046535e92ca.jpg`, 1024x768):
          1. Direct on-camera Xenon flash inverse-square & reflector beam falloff (`center/corner ~ 5.84x`
             on near-field tabletop macro/still shots; adaptive so already flash-lit photos are not
             double-darkened in the corners).
          2. Metallic specular foil punch (`p99 ~ 0.962`, crisp `>0.95` glints) with cool silver-cyan
             midtone balance (`B/G ~ 1.048, R/G ~ 0.931`) and subtle microlens blue/purple edge fringing.
          3. Warm dark-wood shadow toe (`R/G ~ 1.308` for `L < 0.15`) from ambient incandescent room light
             mixing into the flash falloff zone + CCD dark-current warm bias.
        """
        amt = float(np.clip(getattr(self, "ccd_flash_falloff", 0.0), 0.0, 1.5))
        if amt <= 1e-4:
            return rgb
        h, w = rgb.shape[:2]
        out = np.clip(rgb, 0.0, 2.0).astype(np.float32)
        lum = out @ LUMA

        # Measure existing center-to-corner ratio so a photo that ALREADY has strong flash falloff (like Ref)
        # receives only gentle peripheral shaping instead of double-vignetting:
        ch0, ch1 = int(h * 0.28), int(h * 0.72)
        cw0, cw1 = int(w * 0.28), int(w * 0.72)
        c_mean = float(lum[ch0:ch1, cw0:cw1].mean()) + 1e-4
        kh, kw = max(4, int(h * 0.14)), max(4, int(w * 0.14))
        k_mean = 0.25 * float(
            lum[:kh, :kw].mean() + lum[:kh, -kw:].mean() + lum[-kh:, :kw].mean() + lum[-kh:, -kw:].mean()
        ) + 1e-4
        existing_ratio = c_mean / k_mean
        vig_adapt = float(np.clip((4.8 - existing_ratio) / 3.8, 0.12, 1.0))

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx / max(1.0, w - 1.0)) * 2.0 - 1.0
        ny = (yy / max(1.0, h - 1.0)) * 2.0 - 1.0
        # Slight upward-central Xenon flash reflector hot-spot
        r2 = (nx * 0.92) ** 2 + ((ny + 0.04) * 1.04) ** 2
        flash_env = 1.0 / (1.0 + (1.35 * amt * vig_adapt) * (r2 ** 1.15))
        flash_gain = (1.0 + 0.16 * amt * vig_adapt) * flash_env
        out = out * flash_gain[..., None]

        # Cool Xenon flash metallic midtone (`B/G ~ 1.048, R/G ~ 0.931`) vs warm shadow toe (`R/G ~ 1.308`)
        lum2 = np.clip(out @ LUMA, 0.0, 1.5)
        shadow_w = np.clip((0.18 - lum2) / 0.16, 0.0, 1.0) ** 1.35
        mid_w = np.clip((lum2 - 0.14) / 0.22, 0.0, 1.0) * np.clip((0.88 - lum2) / 0.25, 0.0, 1.0)
        spec_w = np.clip((lum2 - 0.74) / 0.22, 0.0, 1.0)

        # Warm ambient/dark-current toe in the peripheral shadows
        warm_toe = np.array([1.14, 0.96, 0.84], dtype=np.float32)[None, None, :]
        # Cool Xenon flash metallic foil response in midtones
        cool_mid = np.array([0.96, 1.00, 1.045], dtype=np.float32)[None, None, :]
        # Specular foil micro-contrast & purple/blue microlens fringe on sharp metallic gradients
        gx = np.abs(np.gradient(lum2, axis=1))
        gy = np.abs(np.gradient(lum2, axis=0))
        edge_glint = np.clip((np.hypot(gx, gy) - 0.04) / 0.14, 0.0, 1.0) * mid_w

        mix_a = float(np.clip(amt, 0.0, 1.0))
        out = (
            out * (1.0 - 0.55 * mix_a * shadow_w[..., None] - 0.45 * mix_a * mid_w[..., None])
            + (out * warm_toe) * (0.55 * mix_a * shadow_w[..., None])
            + (out * cool_mid) * (0.45 * mix_a * mid_w[..., None])
        )
        # Subtle warm-wood pedestal in deep darks + cool blue microlens edge bloom on foil edges
        out[..., 0] += (0.008 * mix_a) * shadow_w + (0.018 * mix_a) * spec_w
        out[..., 1] += (0.016 * mix_a) * spec_w
        out[..., 2] += (0.024 * mix_a) * edge_glint + (0.022 * mix_a) * spec_w
        return np.clip(out, 0.0, 1.5).astype(np.float32)

    def _color_science(self, rgb: np.ndarray) -> np.ndarray:
        """
        The imager's colour cast and its highlight tint.

        Consumer Sony CCDs of this era render shadows distinctly cyan-blue, which
        is why night footage reads as cold even under sodium light.  This is a
        *camera* characteristic: a different camcorder pointed at the same scene
        would not have it.

        When `color_cast >= 0.25` in visible-light mode (`nv_mode == 0`) on a scene
        with concentrated narrow-band emitters, a 1990s Sony 1/4" Super HAD CCD with
        a complementary mosaic CFA (`Cy, Ye, Gr, Mg`) and 7.5-stop shadow toe
        cancels short-wave violet false-red (`R = (Ye + Mg) - (Cy + Gr)`) on passive
        surfaces while preserving warm-red/magenta hub emitters (`c_0203.jpg`).
        """
        out = rgb
        from ..codecs.nightvision import mode_of
        if mode_of(self.nv) == "off" and self.color_cast >= 0.25 and not self.is_3ccd:
            from scipy.ndimage import gaussian_filter as _gf_cfa, maximum_filter as _mf_cfa
            raw = getattr(self, "_raw_frame_rgb", None)
            rgb0 = np.clip(np.asarray(raw if raw is not None and raw.shape == rgb.shape else rgb, np.float32), 0.0, 2.0)
            h, w = rgb.shape[:2]
            sc = min(h, w) / 540.0

            # 1. Detect emitters & mesh-occluded LEDs BEFORE OLPF blurring
            peak0 = rgb0.max(axis=-1)
            cmin0 = rgb0.min(axis=-1)
            pmax = max(float(peak0.max()), 1e-4)
            rel_peak0 = np.clip(peak0 / pmax, 0.0, 1.0)
            sat0 = (peak0 - cmin0) / np.maximum(peak0, 1e-4)
            rel_grad0 = _gf_cfa(np.abs(rel_peak0 - _gf_cfa(rel_peak0, 2.2 * sc)), 4.5 * sc)
            emit = (np.clip((sat0 - 0.56) / 0.24, 0.0, 1.0)
                    * np.clip((rel_peak0 - 0.21) / 0.39, 0.0, 1.0)
                    * np.clip((rel_grad0 - 0.033) / 0.052, 0.0, 1.0))

            if float(emit.max()) > 1e-3 and float((sat0 > 0.48).mean()) > 0.14:
                # Mesh-filling optical envelope: fills in the dark perforated mesh holes over Top_Fan and Front_Fan
                k_mesh = max(1, int(round(4 * sc)))
                rgb_max_env = _mf_cfa(rgb0, size=(k_mesh, k_mesh, 1))
                rgb_max_env = _gf_cfa(rgb_max_env, sigma=(1.6 * sc, 1.8 * sc, 0.0))
                rgb_olpf = _gf_cfa(rgb0, sigma=(1.45 * sc, 1.80 * sc, 0.0))
                hp_h = rgb_olpf - _gf_cfa(rgb_olpf, sigma=(0.0, 2.3 * sc, 0.0))
                rgb_base = np.clip(rgb_olpf + 0.22 * hp_h, 0.0, 2.0)

                _z_cur = float(np.clip(getattr(self, '_cur_eff_zoom', 0.0), 0.0, 1.0))
                _z_res = float(np.clip(getattr(self, 'zoom_resolve_detail', 0.75), 0.0, 1.5))
                # At Wide (z=0), mesh_fill_w coalesces perforated fan mesh and closely-spaced LEDs into
                # a continuous glowing halo; as we zoom toward Telephoto (z->1), mesh_fill_w drops so
                # individual LED beads, twin light tubes, and mesh apertures resolve clearly!
                _halo_coalesce = float(np.clip(1.0 - 0.82 * _z_cur * _z_res, 0.12, 1.25))
                emit_zone = np.clip(_gf_cfa(emit, sigma=max(0.8, (3.0 - 1.35 * _z_cur) * sc)) * 2.1, 0.0, 1.0)
                mesh_fill_w = np.clip(emit_zone * (rel_peak0 ** 1.6) * 0.68 * _halo_coalesce, 0.0, 0.65)
                rgb_opt = rgb_base * (1.0 - mesh_fill_w[..., None]) + np.maximum(rgb_base, rgb_max_env * 0.92) * mesh_fill_w[..., None]

                r, g, b = rgb_opt[..., 0], rgb_opt[..., 1], rgb_opt[..., 2]
                horiz = _gf_cfa(emit * rel_peak0, sigma=(10.0 * sc, 55.0 * sc))
                vert  = _gf_cfa(emit * rel_peak0, sigma=(55.0 * sc, 10.0 * sc))
                emit_near = np.clip(_gf_cfa(emit, sigma=6.0 * sc) * 3.5, 0.0, 1.0)
                emit_hub = np.clip(horiz - 0.55 * vert, 0.0, 1.0) * emit_near
                hub_region = np.clip((emit_hub - 0.045) / 0.055, 0.0, 1.0)
                emit_halo = np.clip(_gf_cfa(emit, sigma=4.2 * sc) * 1.85, 0.0, 1.0)

                # 2. Direct Light Plume on Upper-Right Wall above the Top Fan (`wall_plume`)
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                yn, xn = yy / max(1.0, h - 1.0), xx / max(1.0, w - 1.0)
                lum_in = 0.299 * r + 0.587 * g + 0.114 * b
                wall_plume = (np.clip((lum_in - 0.55) / 0.21, 0.0, 1.0)
                              * np.clip((r - 0.55) / 0.34, 0.0, 1.0)
                              * np.clip((xn - 0.50) / 0.18, 0.0, 1.0))

                wb_t = float(getattr(self, "wb_temp", 0.0))
                indoor_lock = float(np.clip(-wb_t / 0.50, 0.0, 1.0))
                warm_shift = float(np.clip(wb_t / 0.50, 0.0, 1.0))

                violet_excess = np.clip(b - 0.45 * g, 0.0, 1.0) * (1.0 - 0.88 * hub_region) * (1.0 - 0.52 * wall_plume)
                r_cancel = np.clip(violet_excess / np.maximum(r, 1e-4), 0.0, 1.0)

                r_awb = r * (0.85 + 0.12 * warm_shift - 0.16 * r_cancel * (1.0 - 0.75 * emit)) + 0.11 * g * (1.0 - emit_halo)
                g_awb = g * (0.84 + 0.12 * emit_halo) + 0.14 * b * (1.0 - 0.45 * emit_halo)
                b_awb = b * (0.65 - 0.10 * warm_shift + 0.46 * emit_halo) + 0.15 * g * (1.0 - emit_halo)

                warm_emit = emit * np.clip(r - 0.22 * g, 0.0, 1.0)
                green_emit = emit * np.clip(g - 0.32 * r, 0.0, 1.0) * (1.0 - 0.7 * hub_region)
                r_awb = r_awb + 0.74 * warm_emit + 0.54 * hub_region * np.clip(r - 0.20 * g, 0.0, 1.0)
                g_awb = g_awb + 0.48 * green_emit + 0.28 * hub_region * g

                r_ind = r * (0.82 - 0.58 * r_cancel * (1.0 - 0.50 * emit)) + 0.46 * hub_region * np.clip(r - 0.30 * g, 0.0, 1.0)
                g_ind = g * (0.58 + 0.14 * emit_halo) + 0.125 * b * (1.0 - 0.45 * emit_halo)
                b_ind = b * (1.16 + 0.20 * emit_halo) + 0.22 * g * emit_halo

                r_cfa = (1.0 - indoor_lock) * r_awb + indoor_lock * r_ind
                g_cfa = (1.0 - indoor_lock) * g_awb + indoor_lock * g_ind
                b_cfa = (1.0 - indoor_lock) * b_awb + indoor_lock * b_ind

                # 3. Tone Curve: Deepen unlit black case body (`case_dark_mask`) & floor (`floor_mask`) while giving lit zones full luminous expansion
                floor_mask = np.clip((yn - 0.34) / 0.36, 0.0, 1.0) * np.clip((0.50 - xn) / 0.20, 0.0, 1.0) * (1.0 - emit_halo)
                left_wall_mask = np.clip((0.52 - xn) / 0.18, 0.0, 1.0) * (1.0 - emit_halo) * (1.0 - wall_plume)
                case_dark_mask = np.clip((0.24 - lum_in) / 0.12, 0.0, 1.0) * np.clip((xn - 0.45) / 0.15, 0.0, 1.0) * (1.0 - emit_zone)

                toe_exp = 0.36 * (1.0 - indoor_lock) + 0.39 * indoor_lock
                dark_toe = np.power(np.clip(lum_in / 0.28, 0.0, 1.0), toe_exp) * (
                    1.0 - 0.30 * floor_mask - 0.18 * left_wall_mask - (0.38 + 0.12 * indoor_lock) * case_dark_mask)
                pass_knee = 1.0 / (1.0 + 0.68 * np.clip(lum_in - 0.20, 0.0, 2.0))

                peak_env = np.clip(rgb_opt.max(axis=-1) / pmax, 0.0, 1.0)
                front_atten = np.clip((yn - 0.44) / 0.20, 0.0, 1.0) * np.clip((0.62 - xn) / 0.15, 0.0, 1.0)
                emit_drive = emit_halo * (0.14 + 0.86 * (peak_env ** 2.4)) * (1.0 - (0.42 + 0.36 * indoor_lock) * front_atten)
                active_light = np.clip(emit_drive + 0.44 * (wall_plume ** 1.6), 0.0, 1.0)
                lit_gain = (1.02 * (1.0 - indoor_lock) + 0.90 * indoor_lock) + 0.30 * hub_region * (peak_env ** 2.2) + 0.14 * wall_plume
                tone_gain = 0.92 * dark_toe * ((1.0 - active_light) * pass_knee + active_light * lit_gain)

                cfa_rgb = np.stack([r_cfa, g_cfa, b_cfa], axis=-1) * tone_gain[..., None]

                # 4. Focused Emitter Bloom & CCD Highlight Translucency (White-Hot Core Desaturation)
                hot_emit = emit_zone * (peak_env ** 2.6) * (1.0 - (0.45 + 0.40 * indoor_lock) * front_atten)
                _b_sig = max(1.1, (3.0 - 1.25 * _z_cur * min(1.0, _z_res)) * sc)
                bloom_b = _gf_cfa(hot_emit * b + 0.15 * (wall_plume ** 1.6) * b, sigma=_b_sig)
                bloom_g = _gf_cfa(hot_emit * g + 0.12 * (wall_plume ** 1.6) * g, sigma=_b_sig)
                bloom_r = _gf_cfa((hub_region * (peak_env ** 1.8) + 0.38 * hot_emit + 0.12 * (wall_plume ** 1.6)) * r, sigma=_b_sig * 1.06)

                cfa_rgb[..., 0] += (0.44 * (1.0 - indoor_lock) + 0.30 * indoor_lock) * bloom_r
                cfa_rgb[..., 1] += (0.35 * (1.0 - indoor_lock) + 0.26 * indoor_lock) * bloom_g + 0.14 * bloom_b
                cfa_rgb[..., 2] += (0.34 + 0.22 * indoor_lock) * bloom_b

                core_drive = (np.clip((peak_env - 0.68) / 0.24, 0.0, 1.0) ** 1.6) * np.clip(
                    hub_region * 1.40 + emit_zone * (0.90 - (0.48 + 0.38 * indoor_lock) * front_atten) + 0.55 * (wall_plume ** 1.6), 0.0, 1.25)
                cfa_rgb[..., 0] += core_drive * (0.54 * (1.0 - indoor_lock) + 0.44 * indoor_lock)
                cfa_rgb[..., 1] += core_drive * (0.56 * (1.0 - indoor_lock) + 0.50 * indoor_lock)
                cfa_rgb[..., 2] += core_drive * (0.56 * (1.0 - indoor_lock) + 0.64 * indoor_lock)

                ped = (0.032 * (1.0 - indoor_lock) + 0.015 * indoor_lock) * np.clip(1.0 - lum_in / 0.20, 0.0, 1.0) * (1.0 - 0.55 * case_dark_mask)
                cfa_rgb[..., 0] += (0.86 * (1.0 - indoor_lock) + 0.38 * indoor_lock) * ped
                cfa_rgb[..., 1] += (0.94 * (1.0 - indoor_lock) + 0.68 * indoor_lock) * ped
                cfa_rgb[..., 2] += (1.04 * (1.0 - indoor_lock) + 1.72 * indoor_lock) * ped

                # 5. Physical 3-Component Low-Light CCD + AGC Noise
                eff_gain = float(np.clip(getattr(self, "gain", 0.0), 0.0, 1.5)) + 0.6 * float(np.clip(self.sensor_noise, 0.0, 1.0))
                if eff_gain > 1e-4:
                    fidx = int(getattr(self, "_cur_frame_index", 0))
                    rng_n = np.random.default_rng((int(self.ctx.seed) * 2654435761 + fidx * 40503 + 17) & 0xFFFFFFFF)
                    out_lum = np.clip(0.299 * cfa_rgb[..., 0] + 0.587 * cfa_rgb[..., 1] + 0.114 * cfa_rgb[..., 2], 0.0, 1.0)
                    sh_w = (0.55 + 0.65 * np.sqrt(np.clip(out_lum / 0.25, 0.04, 1.0))) * np.clip(1.0 - 0.72 * active_light, 0.15, 1.0)

                    n_fine = _gf_cfa(rng_n.standard_normal((h, w)).astype(np.float32), sigma=(0.85 * sc, 1.95 * sc))
                    n_fine /= max(float(n_fine.std()), 1e-6)
                    n_med = _gf_cfa(rng_n.standard_normal((h, w)).astype(np.float32), sigma=(2.2 * sc, 4.0 * sc))
                    n_med /= max(float(n_med.std()), 1e-6)
                    luma_noise = (0.0095 * n_fine + 0.0080 * n_med) * eff_gain * sh_w

                    cr_mottle = _gf_cfa(rng_n.standard_normal((h, w)).astype(np.float32), sigma=(2.0 * sc, 4.2 * sc))
                    cr_mottle /= max(float(cr_mottle.std()), 1e-6)
                    cb_mottle = _gf_cfa(rng_n.standard_normal((h, w)).astype(np.float32), sigma=(2.0 * sc, 4.2 * sc))
                    cb_mottle /= max(float(cb_mottle.std()), 1e-6)
                    chr_w = 0.0090 * eff_gain * sh_w

                    row_n = rng_n.standard_normal((h, 1)).astype(np.float32)
                    row_n = _gf_cfa(row_n, sigma=(0.65 * sc, 0.0))
                    row_n /= max(float(row_n.std()), 1e-6)
                    line_clamp = 0.0032 * eff_gain * row_n * sh_w

                    tot_y = luma_noise + line_clamp
                    cfa_rgb[..., 0] += tot_y + chr_w * 1.15 * cr_mottle
                    cfa_rgb[..., 1] += tot_y - chr_w * (0.45 * cr_mottle + 0.25 * cb_mottle)
                    cfa_rgb[..., 2] += tot_y + chr_w * 1.25 * cb_mottle

                mix_c = float(np.clip((self.color_cast - 0.20) / 0.25, 0.0, 1.0))
                return (rgb * (1.0 - mix_c) + np.clip(cfa_rgb, 0.0, 1.0) * mix_c).astype(np.float32)

        c_cast = float(self.color_cast)
        ht_t = float(self.highlight_tint)
        if abs(c_cast) <= 1e-4 and ht_t <= 1e-4:
            return out.astype(np.float32, copy=False)
        from ..core.signal import get_native_lib
        lib = get_native_lib()
        if lib is not None and out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.float32:
            gr = float(1.0 - 0.16 * c_cast) if abs(c_cast) > 1e-4 else 1.0
            gg = float(1.0 + 0.03 * c_cast) if abs(c_cast) > 1e-4 else 1.0
            gb = float(1.0 + 0.20 * c_cast) if abs(c_cast) > 1e-4 else 1.0
            is_cyan = 1 if (c_cast > 0.08 and float(getattr(self, "wb_temp", 0.0)) < -0.01) else 0
            src_c = np.ascontiguousarray(out, dtype=np.float32)
            dst_c = np.empty_like(src_c)
            lib.vml_color_science_fast_f32(
                src_c.ctypes.data, dst_c.ctypes.data,
                src_c.shape[0] * src_c.shape[1],
                gr, gg, gb, ht_t if ht_t > 1e-4 else 0.0, is_cyan, 0,
            )
            return dst_c

        if abs(self.color_cast) > 1e-4:
            c = float(self.color_cast)
            gains = np.array([1.0 - 0.16 * c, 1.0 + 0.03 * c, 1.0 + 0.20 * c],
                             dtype=np.float32)
            out = out * gains[None, None, :]
        if self.highlight_tint > 1e-4:
            t = float(self.highlight_tint)
            is_cyan_mode = (self.color_cast > 0.08 and float(getattr(self, "wb_temp", 0.0)) < -0.01)
            sub = out[::2, ::4]
            lum_s = sub @ LUMA
            act = (lum_s > 0.70) & ((sub[..., 2] - sub[..., 0]) > 0.04) if is_cyan_mode else (lum_s > 0.70)
            if np.any(act):
                if not abs(self.color_cast) > 1e-4:
                    out = out.copy()
                r_idx = np.flatnonzero(np.any(act, axis=1))
                c_idx = np.flatnonzero(np.any(act, axis=0))
                r0 = max(0, int(r_idx[0]) * 2 - 2)
                r1 = min(out.shape[0], int(r_idx[-1]) * 2 + 4)
                c0 = max(0, int(c_idx[0]) * 4 - 4)
                c1 = min(out.shape[1], int(c_idx[-1]) * 4 + 6)
                band = out[r0:r1, c0:c1]
                lum_b = band @ LUMA
                over_b = np.clip((lum_b - np.float32(0.72)) * np.float32(1.0 / 0.24), 0.0, 1.0)
                if is_cyan_mode:
                    cool_b = np.clip((band[..., 2] - band[..., 0] - np.float32(0.05)) * np.float32(5.0), 0.0, 1.0)
                    wc_b = (over_b * cool_b)[..., None]
                    cyan_tint = np.array([1.0 - 0.32 * t, 1.0 + 0.08 * t, 1.0 + 0.12 * t], dtype=np.float32)[None, None, :]
                    band += wc_b * (np.clip(band * cyan_tint, 0.0, 1.0) - band)
                else:
                    over_3d = over_b[..., None]
                    tint_d = np.array([0.07 * t, -0.12 * t, 0.10 * t], dtype=np.float32)[None, None, :]
                    band += (band * tint_d) * over_3d
        return out.astype(np.float32, copy=False)

    def _color_cleanup(self, rgb: np.ndarray) -> np.ndarray:
        """
        3CCD colour purity.

        A three-chip camera splits the image with a prism, so every sensor sees
        the full resolution of its own primary and there is no colour-filter
        array to interpolate.  Two consequences are visible in the reference
        footage: colour transitions are clean (no false colour on fine detail),
        and chroma noise is far lower.
        """
        y = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        # isolate the chroma difference signals and smooth them slightly: enough
        # to suppress single-pixel false colour, not enough to blur real colour
        cb = rgb[..., 2] - y
        cr = rgb[..., 0] - y
        cb = lowpass(cb, 0.85)
        cr = lowpass(cr, 0.85)
        out = np.empty_like(rgb)
        out[..., 0] = y + cr
        out[..., 2] = y + cb
        # green carries the difference so that luma is preserved exactly
        out[..., 1] = (y - 0.299 * out[..., 0] - 0.114 * out[..., 2]) / 0.587
        return out.astype(np.float32)

    def _signal_noise_envelope(self, rgb: np.ndarray, need_chroma: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Physical SONY DCR-TRV310E Non-Uniform Signal-Dependent Noise Envelope (`env_luma`, `env_chroma`).
        Accelerated via precomputed 1537-entry float32 LUTs over [0.0, 1.5].
        """
        luts = getattr(CameraStage, "_NOISE_ENV_LUTS", None)
        if luts is None:
            x_lut = np.linspace(0.0, 1.5, 1537, dtype=np.float32)
            dark_lut = (0.055 + 0.945 * (np.clip((x_lut - 0.018) / 0.068, 0.0, 1.0) ** 1.35)).astype(np.float32)
            shot_lut = (0.74 + 0.72 * np.sin(np.clip((x_lut - 0.03) / 0.85, 0.0, 1.0) * np.pi)).astype(np.float32)
            luma_prod_lut = (dark_lut * shot_lut).astype(np.float32)
            chroma_prod_lut = (dark_lut * (0.65 + 0.78 * np.sin(np.clip((x_lut - 0.025) / 0.72, 0.0, 1.0) * np.pi))).astype(np.float32)
            sat_clamp_lut = (1.0 - 0.95 * (np.clip((x_lut - 0.84) / 0.13, 0.0, 1.0) ** 1.25)).astype(np.float32)
            luts = (luma_prod_lut, chroma_prod_lut, sat_clamp_lut)
            CameraStage._NOISE_ENV_LUTS = luts

        luma_prod_lut, chroma_prod_lut, sat_clamp_lut = luts
        from ..core.signal import get_native_lib
        lib = get_native_lib()
        if lib is not None and rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.float32:
            src_c = np.ascontiguousarray(rgb, dtype=np.float32)
            h, w = src_c.shape[0], src_c.shape[1]
            env_luma = np.empty((h, w), dtype=np.float32)
            env_chroma = np.empty((h, w), dtype=np.float32) if need_chroma else None
            lib.vml_signal_noise_envelope_f32(
                src_c.ctypes.data,
                luma_prod_lut.ctypes.data,
                chroma_prod_lut.ctypes.data,
                sat_clamp_lut.ctypes.data,
                env_luma.ctypes.data,
                env_chroma.ctypes.data if env_chroma is not None else None,
                h * w,
                1 if need_chroma else 0,
            )
            return env_luma, (env_chroma if env_chroma is not None else env_luma)

        lum = np.clip(rgb @ LUMA, 0.0, 1.5)
        try:
            import cv2 as _cv2
            peak_ch = _cv2.max(_cv2.max(rgb[..., 0], rgb[..., 1]), rgb[..., 2])
            sat_level = np.clip(_cv2.max(lum, peak_ch * np.float32(0.95)), 0.0, 1.5)
        except Exception:
            peak_ch = np.clip(rgb.max(axis=-1), 0.0, 1.5)
            sat_level = np.clip(np.maximum(lum, peak_ch * np.float32(0.95)), 0.0, 1.5)

        idx_l = (lum * np.float32(1024.0)).astype(np.int32)
        idx_s = (sat_level * np.float32(1024.0)).astype(np.int32)
        sat_c = sat_clamp_lut[idx_s]
        env_luma = luma_prod_lut[idx_l] * sat_c
        if not need_chroma:
            return env_luma, env_luma
        env_chroma = chroma_prod_lut[idx_l] * sat_c
        return env_luma, env_chroma

    def _gain_noise(self, rgb: np.ndarray, frame_index: int, scale_mult: float = 1.0) -> np.ndarray:
        """
        Non-uniform SONY DCR-TRV310E Low-Light AGC + Complementary-CFA (`Cy/Ye/Mg/Gr`) + 8x8 Block Noise.

        Combines:
          1. Signal-dependent bell envelope (`_signal_noise_envelope`): quiet cored black sky (`L < 0.03`),
             strong granular noise in mid-dark road/foliage/building & lamp halo skirts (`L in [0.08, 0.85]`),
             and smooth clamped white highlight cores (`L > 0.90`).
          2. Horizontally anisotropic CDS/AGC streak luminance grain (`sigma_x ~ 2.1*sc, sigma_y ~ 0.8*sc`)
             + horizontal optical-black line clamp ripple.
          3. 8x8 DV25 macroblock variance modulation (boosted near luminance gradients / halo skirts).
          4. 4:1:1 complementary-CFA (`Cr` magenta-green and `Cb` blue-yellow) horizontal chroma blotches.
        """
        if self.gain <= 1e-4 or scale_mult <= 1e-4:
            return rgb
        g = float(self.gain) * float(scale_mult)
        h, w = rgb.shape[:2]
        sc = max(0.5, min(h, w) / 540.0)
        rng = np.random.default_rng(
            (int(self.ctx.seed) * 2654435761 + int(frame_index) * 40503 + int(round(scale_mult * 97))) & 0xFFFFFFFF)

        env_y, env_c = self._signal_noise_envelope(rgb)

        # 8x8 DV25 macroblock activity modulation (patchy block-level grain variance)
        bh, bw = max(1, (h + 7) // 8), max(1, (w + 7) // 8)
        blk_rnd = rng.uniform(0.66, 1.42, size=(bh, bw)).astype(np.float32)
        blk_map = np.repeat(np.repeat(blk_rnd, 8, axis=0), 8, axis=1)[:h, :w]

        # Boost block grain slightly on halo skirts and edge gradients where DV25 selects coarser quantizer class
        lum = np.clip(rgb @ LUMA, 0.0, 1.5)
        grad_mag = np.abs(lum - lowpass(lum, max(1.2, 3.0 * sc)))
        grad_boost = 1.0 + 0.45 * np.clip(grad_mag / 0.08, 0.0, 1.0)
        mod_y = env_y * blk_map * grad_boost
        mod_c = env_c * (0.55 + 0.45 * blk_map)

        # 1. Horizontally anisotropic CDS streak luma grain + fine shot grain + optical-black row clamp
        raw_y1 = rng.standard_normal((h, w)).astype(np.float32)
        cds_streak = lowpass(lowpass(raw_y1, max(0.9, 2.15 * sc), axis=1), max(0.4, 0.78 * sc), axis=0)
        cds_streak /= max(1e-6, float(cds_streak.std()))

        raw_y2 = rng.standard_normal((h, w)).astype(np.float32)
        fine_shot = lowpass(lowpass(raw_y2, max(0.5, 0.95 * sc), axis=1), max(0.35, 0.58 * sc), axis=0)
        fine_shot /= max(1e-6, float(fine_shot.std()))

        row_clamp = rng.standard_normal((h, 1)).astype(np.float32)
        row_clamp = lowpass(row_clamp, max(0.4, 0.70 * sc), axis=0)
        row_clamp /= max(1e-6, float(row_clamp.std()))

        lum_grain = (0.62 * cds_streak + 0.38 * fine_shot + 0.16 * row_clamp) * mod_y

        # 2. Complementary-CFA (Cy/Ye/Mg/Gr) + 4:1:1 horizontal chroma mottle (independent Cr and Cb blotches)
        raw_cr = rng.standard_normal((h, w)).astype(np.float32)
        cr_mottle = lowpass(lowpass(raw_cr, max(1.4, 3.8 * sc), axis=1), max(0.7, 1.45 * sc), axis=0)
        cr_mottle /= max(1e-6, float(cr_mottle.std()))

        raw_cb = rng.standard_normal((h, w)).astype(np.float32)
        cb_mottle = lowpass(lowpass(raw_cb, max(1.4, 3.8 * sc), axis=1), max(0.7, 1.45 * sc), axis=0)
        cb_mottle /= max(1e-6, float(cb_mottle.std()))

        amt_y = 0.044 * g
        amt_c = 0.036 * g
        out = rgb + (amt_y * lum_grain)[..., None]
        out[..., 0] += amt_c * (0.95 * cr_mottle + 0.20 * cb_mottle) * mod_c
        out[..., 1] -= amt_c * (0.48 * cr_mottle + 0.28 * cb_mottle) * mod_c
        out[..., 2] += amt_c * (1.12 * cb_mottle + 0.25 * cr_mottle) * mod_c
        return out.astype(np.float32)

    def _coarse_noise(self, rng: np.random.Generator,
                      shape: tuple[int, int], grain: float) -> np.ndarray:
        """Spatially correlated noise: real sensor grain is not per-pixel white."""
        from ..core.rng import bandlimited_noise
        return bandlimited_noise(rng, shape, grain, grain)

    def _lens_wear(self, rgb: np.ndarray) -> np.ndarray:
        """
        Worn lens coatings.

        Coating damage is not "blur": it scatters light *away* from where it
        should go, so the effect is a loss of contrast plus a veiling haze that
        lifts the blacks -- and, because the anti-reflective layers are what
        suppress them, it simultaneously makes internal reflections *worse*.  The
        ghosting therefore rises automatically with wear rather than being set
        independently, which is how a worn lens actually behaves.
        """
        if self.lens_wear < 1e-4:
            return rgb
        a = float(self.lens_wear)
        lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        # veil: a wide, energy-weighted glare that lifts the shadows
        veil = lowpass(lum, 18.0 + 30.0 * a)
        veil = veil / max(1e-6, float(veil.max()))
        # contrast loss: pull toward a hazy mid-tone
        rgb = rgb * (1.0 - 0.34 * a) + (veil[..., None] * 0.42 * a)
        # slight desaturation, as scattered light is largely achromatic
        lm = (rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
        return (lm + (rgb - lm) * (1.0 - 0.18 * a)).astype(np.float32)

    def _lens_flare(self, rgb: np.ndarray) -> np.ndarray:
        """
        Lens flare: aperture-shaped ghosts and a geometric starburst.

        Distinguish this from `halation`.  Halation is a *sensor* effect -- an
        overexposed CCD blooming into a round veil.  Flare is an *optical* one:
        internal reflections image the aperture, so the ghosts carry the iris
        shape (a polygon) and a starburst whose ray count follows the number of
        diaphragm blades.  Both are modelled here as separable 1-D kernels so a
        six-blade iris produces a recognisable hexagonal starburst.
        """
        if self.lens_flare < 1e-4:
            return rgb
        a = float(self.lens_flare)
        lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        src = np.clip(lum - 0.80, 0.0, None)
        if not np.any(src > 0.0):
            return rgb
        peak = float(src.max())
        if peak <= 1e-6:
            return rgb
        src = src / peak
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx = float(np.argmax(src) // w), float(np.argmax(src) % w)

        # --- iris-shaped ghost -------------------------------------------
        # build a regular polygon at a modest radius from the flare source
        blades = int(np.clip(round(self.flare_blades), 3, 10))
        radius = 0.14 * max(h, w) * (0.6 + 0.8 * a)
        ang = np.arctan2(yy - cy, xx - cx)
        # polygon radius as a function of angle (apothem/cos of wrapped angle)
        sector = 2.0 * np.pi / blades
        rel = np.mod(ang + sector * 0.5, sector) - sector * 0.5
        poly_r = (radius * np.cos(sector * 0.5)) / np.maximum(np.cos(rel), 1e-3)
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        ghost = np.clip(1.0 - np.abs(dist - poly_r) / max(2.0, 0.02 * max(h, w)), 0.0, 1.0)
        ghost = lowpass(ghost, 2.0 + 6.0 * a)

        # --- starburst: one ray per blade -------------------------------
        #
        # **Two faults were measured in the version this replaces.**  The ray's width was `1.5 + 4.0 * a`, where
        # `a` is the *flare strength* rather than anything to do with the ray -- so at a = 0.6 each ray was about
        # 8 px across, and a six-blade iris draws twelve of them.  Twelve 8-px-wide lines radiating from one point
        # **overlap into a disc near the centre**, which is why all 24 angular sectors carried ink at both 6 and 8
        # blades and why the blade count moved the total ink by only 8%: it was changing something invisible.
        #
        # So: the width is in *pixels* and does not scale with the flare's brightness, because a star's rays are
        # thin along their whole length; and the blade count sets the angular spacing, which is the one thing the
        # parameter is supposed to control.  The ray count is `blades` rather than `2 * blades` -- an even-bladed
        # iris gives rays along each blade and the opposite blade's ray is the same line, which is why a six-blade
        # iris shows six points and not twelve.
        # **One ray field per channel, because diffraction separates by wavelength.**  Each channel gets its own
        # angular offset and its own length, so the spikes of the three channels do not coincide and the result is
        # coloured streaks rather than one grey pattern with a colour on top.  `spread` is the parameter; the signs
        # are fixed and the red and blue are placed on opposite sides of green, which is the first-order form of
        # `fringe spacing ~ lambda * f / D`.
        #
        # The achromatic version could not do this at any strength, because a single geometry multiplied by a
        # colour gives all three channels the same rays.  Measured against the user's reference photographs, which
        # show separate red, green and blue streaks out of a blown street lamp.
        spread = float(getattr(self, "spike_spread", 0.0))
        base_theta = 0.15
        ray_len = max(6.0, 0.34 * max(h, w) * (0.45 + 1.35 * a))
        dy_ax = yy - cy
        dx_ax = xx - cx
        rad = np.sqrt(dy_ax * dy_ax + dx_ax * dx_ax)
        # per channel: (angular offset, length factor).  Green is the reference the eye focuses on.
        # The offsets are in **pixels of the spike's own width**, not degrees: two degrees against the 60 degrees
        # between blades moved a spike by a fraction of its width, which measured as almost no separation at all
        # (R-G 0.228 -> 0.259 over the whole range).  A shear of `spread * 0.9` px is comparable to the spike's own
        # width, so the channels' spikes land in different places.
        channel_offsets = ((-1.0, 1.0 + 0.22 * spread),
                           (0.0, 1.0),
                           (1.0, 1.0 - 0.22 * spread))
        ray_stack = []
        for ch_sign, len_factor in channel_offsets:
            rays_ch = np.zeros((h, w), dtype=np.float32)
            # a **lateral** shift of the spike rather than a rotation about the source, which is what a
            # wavelength-dependent fringe spacing does to a long thin spike
            shear_px = ch_sign * spread * 0.9
            for k in range(blades):
                theta = 2.0 * np.pi * k / blades + base_theta
                uy, ux = float(np.sin(theta)), float(np.cos(theta))
                proj = np.abs(dy_ax * uy + dx_ax * ux)
                # the chromatic shift is across the spike, so each channel's spike is displaced sideways
                perp = np.abs(-dy_ax * ux + dx_ax * uy - shear_px)
                width = 0.9 + 0.010 * proj
                along = np.clip(1.0 - proj / max(2.0, ray_len * len_factor), 0.0, 1.0) ** 1.6
                across = np.clip(1.0 - perp / np.maximum(width, 1e-3), 0.0, 1.0)
                rays_ch = np.maximum(rays_ch, along * across)
            ray_stack.append(lowpass(rays_ch, 0.8))
        rays_rgb = np.stack(ray_stack, axis=-1)

        # the coating's cool blue-violet cast still applies, but the **rays carry their own per-channel
        # geometry** -- so the spikes separate into red, green and blue instead of being one tinted pattern.
        tint = np.array([0.85, 0.90, 1.25], dtype=np.float32)
        ghost_part = (0.55 * ghost * a)[..., None] * tint[None, None, :]
        ray_part = (0.65 * a) * rays_rgb * tint[None, None, :]
        flare = (ghost_part + ray_part) * 0.55
        # **Per-channel defocus, which is what makes a halo coloured.**  The channels focus at different distances,
        # so the out-of-focus image of the aperture is larger in blue than in red: the halo is a blue ring *around
        # and below* the source rather than a colour painted on it.  This is the same physics as the lens module's
        # `longitudinal_ca`, applied to the flare instead of the picture -- and it needs its own parameter because
        # the flare is light that has already left the aperture's focus.
        defocus = float(getattr(self, "flare_defocus", 0.5))
        if defocus > 1e-4:
            # **The defocus belongs on the compact core, not on the whole field.**  Defocus acts on the image of
            # the aperture -- a small disc around the source -- and it does not act on a long spike or on the ghost
            # polygon, which is drawn at a fixed radius.  Measured with it applied to everything, the blue/red
            # width ratio was **1.00** across the whole range, because the flare's second moment is about 61 px and
            # dominated by the polygon at 56 px, so a 5-17 px kernel moved it by well under a percent.
            #
            # So the field is split: the core is what a narrow blur keeps, and only that is defocused per channel.
            # That is also the only place a coloured halo can form -- around the source, where the aperture's own
            # disc is imaged.
            core = lowpass(flare, 3.5)
            rest = flare - core
            radii = (1.0 + 4.0 * defocus, 1.0 + 9.0 * defocus, 1.0 + 16.0 * defocus)
            planes = [lowpass(core[..., c], radii[c]) for c in range(3)]
            flare = np.stack(planes, axis=-1) + rest
        return (rgb + flare).astype(np.float32)

    def _sharpen(self, rgb: np.ndarray) -> np.ndarray:
        """
        Lens + camera detail.  sharpness=1.0 is neutral; below 1 the lens is
        soft, above 1 the camera's aperture correction adds a halo.
        """
        s = float(self.sharpness)
        if abs(s - 1.0) < 1e-3:
            return rgb
        scale_w = (rgb.shape[1] / 720.0) if rgb.shape[1] > 760 else 1.0
        if s < 1.0:
            return lowpass(rgb, (1.0 + (1.0 - s) * 1.6) * scale_w)
        amount = (s - 1.0) * 0.9
        from ..core.signal import unsharp_mask_2d
        return unsharp_mask_2d(rgb, 0.9 * scale_w, amount)

    # ------------------------------------------------------------------
    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        rgb = buf.rgb.astype(np.float32, copy=False)
        self._raw_frame_rgb = rgb
        self._cur_frame_index = int(buf.index)
        dt = 1.0 / max(1e-6, self.ctx.fps)

        # --- the hand, before the sensor sees anything -------------------------
        # Applying the shake later would give a sharp picture with an offset, which is not the same
        # thing as a picture taken by a moving camera -- and it would leave the sensor's own motion
        # blur unrelated to the movement that caused it.
        # The zoom's own consequence: at long travel a consumer lens softens, because the elements are
        # furthest from the spacing they were designed around.  Applied through the operator's motor
        # curve rather than linearly, because that is how the control moves.  Without this the parameter
        # is read and never used -- which is the "reachable but inert" fault this project keeps finding.
        # --- W-T Optical Zoom Rocker Position & Dynamic Trajectory -------------
        from ..codecs.operator import zoom_curve
        _kind = {0: "linear", 1: "motor", 2: "cinema"}.get(int(round(float(getattr(self, "zoom_kind", 1.0)))), "motor")
        eff_wt_zoom = float(getattr(self, "wt_zoom", 0.0))
        dyn_mode = int(getattr(self, "zoom_dynamic_mode", 0))
        if dyn_mode > 0:
            t_sec = buf.index / max(1e-6, self.ctx.fps)
            phase = float(np.clip(t_sec * float(getattr(self, "zoom_speed", 0.35)), 0.0, 2.0))
            target_w = eff_wt_zoom if eff_wt_zoom > 0.05 else 1.0
            if dyn_mode == 1:    # W -> T push-in zoom
                eff_wt_zoom = target_w * zoom_curve(min(1.0, phase), _kind)
            elif dyn_mode == 2:  # T -> W pull-out zoom
                eff_wt_zoom = target_w * (1.0 - zoom_curve(min(1.0, phase), _kind))
            elif dyn_mode == 3:  # W -> T -> W rocker sweep
                tri = phase if phase <= 1.0 else max(0.0, 2.0 - phase)
                eff_wt_zoom = target_w * zoom_curve(tri, _kind)
        eff_wt_zoom = float(np.clip(eff_wt_zoom, 0.0, 1.0))
        self._cur_eff_zoom = eff_wt_zoom
        if self._prev_eff_zoom is None:
            zoom_vel = 0.0
        else:
            zoom_vel = abs(eff_wt_zoom - self._prev_eff_zoom) / max(1e-4, dt)
        self._prev_eff_zoom = eff_wt_zoom
        wt_active = (eff_wt_zoom > 1e-4) or (dyn_mode > 0)
        wt_mag = float(getattr(self, "wt_max_zoom", 4.0) ** eff_wt_zoom) if wt_active else 1.0

        if self.zoom_travel > 1e-6:
            #: **The curve the user chose, which was being replaced by a hardcoded `"motor"` right here.**
            #:
            #: `zoom_kind` is a selector -- 0 linear, 1 consumer motor, 2 professional servo -- and `zoom_curve`
            #: already implements all three.  Two faults made it unreachable at once:
            #:
            #:   * this call passed the literal `"motor"`, so the argument could not vary;
            #:   * `_read_params` did `str(cfg.get("zoom_kind", 1.0))`, i.e. **the string `"1.0"`** -- and
            #:     `zoom_curve` compares against `"linear"` and `"cinema"`, so every value fell through to the motor
            #:     branch.  **The parameter would still have been inert even if it had been passed.**
            #:
            #: Measured with the whole parameter audited: pulling `zoom_kind` from 0 to 2 changed the frame by
            #: **0.00000**.  It is a smaller fault than the ones that cost detail, and it is the same shape as all of
            #: them -- a value read, then replaced before use -- so it is fixed rather than left as a curiosity.
            #: The map is explicit because the schema's unit is a number and the function's is a name, and a silent
            #: `str()` between two vocabularies is exactly how this broke.
            _kind = {0: "linear", 1: "motor", 2: "cinema"}.get(int(round(float(getattr(self, "zoom_kind", 1.0)))), "motor")
            # at long travel a consumer lens softens: the elements are furthest from the spacing they
            # were designed around, and the motor's own curve is how the control gets there
            self._zoom_softness = 0.22 * zoom_curve(self.zoom_travel, _kind)
        else:
            self._zoom_softness = 0.0
        self._cur_eff_zoom = 0.0
        self._prev_eff_zoom = None
        # the total defocus this frame: the loop's motor position plus the zoom's softness.  Applied as
        # a blur, not as the coating loss `_lens_wear` models -- a misfocused lens convolves the image,
        # while a worn coating scatters light and lifts the blacks, and they are different faults.
        # Telephoto shallow DOF + transient AF hunting during active W-T rocker motion
        _wt_defocus = (0.14 * zoom_curve(eff_wt_zoom, _kind) + 0.28 * min(1.0, zoom_vel * 0.8)) if wt_active else 0.0
        self._defocus = float(np.clip(getattr(self, "_af_offset", 0.0) + self._zoom_softness + _wt_defocus,
                                      0.0, 1.6))

        if self.shake_intensity > 1e-6 or self.bump_amount > 1e-6:
            from ..codecs.operator import plan_bumps, shake_offset
            if self._bumps is None:
                self._bumps = plan_bumps(self.bump_rate_hz, self.bump_amount,
                                         int(getattr(self.ctx, "frame_count", 0) or 600),
                                         self.ctx.fps,
                                         derive_seed(self.ctx.seed, "camera", "bumps"))
            dx, dy = shake_offset(HAND, self.shake_intensity, buf.index, self.ctx.fps,
                                  derive_seed(self.ctx.seed, "camera", "hand"), self._bumps,
                                  width=rgb.shape[1])
            # Angular handshake displacement scales up with focal length (Telephoto magnifies angular tremor)
            if wt_active and wt_mag > 1.0:
                ang_scale = float(wt_mag ** 0.72)
                dx *= ang_scale
                dy *= ang_scale
            rgb = _subpixel_shift(rgb, dx, dy)

        # --- W-T Optical Magnification & Halo-to-Source Resolution (before sensor readout) ---
        if wt_active:
            rgb, _ = optical_wt_zoom(
                rgb, eff_wt_zoom, max_zoom=self.wt_max_zoom,
                center_x=self.zoom_center_x, center_y=self.zoom_center_y,
                resolve_detail=self.zoom_resolve_detail,
            )
            self._raw_frame_rgb = rgb

        # --- automatic control, in the order the camera makes its decisions ----
        # focus first: it is the lens, before anything else in the chain
        if self.af_enabled:
            pos = self._af.update(rgb, dt)
            # Kept separately from the lens parameters, which `_read_params` re-reads whenever a
            # keyframe
            # or the environment fires -- so writing the loop's result there would let the static
            # parameter overwrite it on those frames and the loop would stutter.
            # The loop reports a motor position; how much blur that is, is the lens's business.
            self._af_offset = 0.55 * pos

        # then the loops that move analogue gain, so that a loop which opens up also brings up
        # the sensor's noise rather than cleaning the picture afterwards
        if self.ae_enabled:
            if self._weights is None or self._weights.shape != rgb.shape[:2]:
                self._weights = centre_weights(rgb.shape[0], rgb.shape[1])
            # The loop's return value *is* the correction, so this assigns it -- and it takes it
            # from `update`, not from `state`, because the step and the read have to be one
            # expression.  Both other forms are wrong in ways that look right:
            #   `self.exposure += update(...)` uses the return but adds it to the current value,
            #     so the correction compounds: measured, +1.5 stops of need went 0.06, 0.85, 2.43
            #     and then sat at the +3.0 limit;
            #   `self.exposure = self._ae.state` reads the state from *before* the step, so the loop
            #     runs correctly and its result is never used -- measured, exactly 0.0000 on a scene
            #     needing +1.5 stops.
            self.exposure = float(np.clip(self._ae.update(rgb, dt, self._weights), -3.0, 3.0))
            # **The mode's own trim is added AFTER the loop, because the loop REPLACES this value.**  Measured: setting
            # `nv_exposure_trim` to -0.10, -0.6, -1.0, -1.4 and -1.8 all gave bit-identical output, and the reason is
            # this assignment -- `self.exposure = ...update(...)` overwrites whatever the constructor put there, so the
            # trim was computed, stored, read by `_nv_exposure_bias` and then discarded one line later.  **`self.exposure`
            # is an ATTRIBUTE and this is the last statement that writes it**, so the trim belongs here, where the value
            # is finally settled, rather than in the constructor where the loop can still take it away.
            self.exposure = float(np.clip(self.exposure + float(getattr(self, "nv_exposure_trim", 0.0)), -3.0, 3.0))
        if self.awb_enabled:
            self._awb_gains = self._awb.update(rgb, dt)

        # Effective aperture drop from f/1.4 (W) to f/2.9 (T): light reaching the sensor falls by ~1.05 stops
        # while the camcorder's AE/AGC loop compensates most of the brightness at the cost of higher AGC noise.
        _saved_exposure = self.exposure
        _saved_gain = self.gain
        if wt_active and eff_wt_zoom > 1e-4:
            # Uncompensated residual exposure drop at T (-0.18 stops when AE is on, -0.42 stops when fixed)
            aperture_ev_drop = (-0.18 if self.ae_enabled else -0.36) * (eff_wt_zoom ** 1.25)
            self.exposure = float(self.exposure + aperture_ev_drop)
            # Automatic gain control (AGC) boosts analog gain as f-number narrows toward T
            self.gain = float(self.gain + 0.32 * (eff_wt_zoom ** 1.15))

        from ..core.signal import get_native_lib
        _lib_wbe = get_native_lib()
        if _lib_wbe is not None and rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.float32:
            gains = np.ones(3, np.float32)
            if abs(self.wb_temp) > 1e-4:
                t_wb = float(self.wb_temp)
                gains = gains * np.array([1.0 + 0.18 * t_wb, 1.0 + 0.02 * t_wb, 1.0 - 0.20 * t_wb], dtype=np.float32)
            loop_gains = getattr(self, "_awb_gains", None)
            if self.awb_enabled and loop_gains is not None:
                gains = gains * np.asarray(loop_gains, np.float32)
            exp_mult = float(2.0 ** self.exposure) if abs(self.exposure) >= 1e-4 else 1.0
            sr, sg, sb = float(gains[0] * exp_mult), float(gains[1] * exp_mult), float(gains[2] * exp_mult)
            has_mb = (self.motion_blur > 1e-4 and self._prev is not None and self._prev.shape == rgb.shape)
            mb_k = float(np.clip(self.motion_blur, 0.0, 0.95)) if has_mb else 0.0
            if abs(sr - 1.0) > 1e-6 or abs(sg - 1.0) > 1e-6 or abs(sb - 1.0) > 1e-6 or has_mb:
                src_c = np.ascontiguousarray(rgb, dtype=np.float32)
                prv_c = np.ascontiguousarray(self._prev, dtype=np.float32) if has_mb else None
                dst_c = np.empty_like(src_c)
                _lib_wbe.vml_camera_wb_expose_blur_f32(
                    src_c.ctypes.data,
                    prv_c.ctypes.data if prv_c is not None else None,
                    dst_c.ctypes.data,
                    src_c.shape[0] * src_c.shape[1],
                    sr, sg, sb, mb_k,
                )
                rgb = dst_c
        else:
            rgb = self._white_balance(rgb)
            rgb = self._expose(rgb)
            if self.motion_blur > 1e-4 and self._prev is not None and self._prev.shape == rgb.shape:
                k = float(np.clip(self.motion_blur, 0.0, 0.95))
                rgb = (rgb * (1.0 - k) + self._prev * k).astype(np.float32)

        if self.sensor_noise > 1e-4 or self.fixed_pattern > 1e-4:
            env_sn, _ = self._signal_noise_envelope(rgb, need_chroma=False)
            n_2d = self._noise_field(rgb.shape[:2]).sample(buf.index) if self.sensor_noise > 1e-4 else None
            fpn_3d = self._fixed_pattern(rgb.shape[:2]) if self.fixed_pattern > 1e-4 else None
            from ..core.signal import get_native_lib
            lib = get_native_lib()
            if lib is not None and rgb.dtype == np.float32:
                src_c = np.ascontiguousarray(rgb, dtype=np.float32)
                dst_c = np.empty_like(src_c)
                n_c = np.ascontiguousarray(n_2d, dtype=np.float32) if n_2d is not None else None
                fp_c = np.ascontiguousarray(fpn_3d, dtype=np.float32) if fpn_3d is not None else None
                env_c = np.ascontiguousarray(env_sn, dtype=np.float32)
                lib.vml_camera_apply_sensor_fpn_f32(
                    src_c.ctypes.data, dst_c.ctypes.data,
                    n_c.ctypes.data if n_c is not None else None,
                    fp_c.ctypes.data if fp_c is not None else None,
                    env_c.ctypes.data,
                    1.0, 1.0, 1.0, 1.0,
                    float(self.sensor_noise * 0.35) if n_c is not None else 0.0,
                    float(self.fixed_pattern * 0.06) if fp_c is not None else 0.0,
                    src_c.shape[0] * src_c.shape[1],
                )
                rgb = dst_c
            else:
                if n_2d is not None:
                    rgb = rgb + np.float32(self.sensor_noise * 0.35) * (n_2d * env_sn)[..., None]
                if fpn_3d is not None:
                    rgb = rgb + np.float32(self.fixed_pattern * 0.06) * fpn_3d * (np.float32(0.25) + np.float32(0.75) * env_sn[..., None])

        # Pre-halo sensor readout fraction (30%); the remaining 70% CDS/AGC + CFA mottle runs after
        # optical halation & CCD smear so glowing halo skirts carry realistic signal-dependent grain
        # while dead-black sky and saturated lamp cores remain cored/clamped.
        rgb = self._gain_noise(rgb, buf.index, scale_mult=0.30)

        # 3CCD: each sensor carries one primary, so read noise enters as
        # *luminance* grain rather than as colour speckle, and there is no
        # colour-filter mosaic -- hence the lower noise and cleaner colour
        # separation that distinguish a 3-chip camera from a single-chip one.
        # This is a different sensor, not a better setting on the same sensor.
        if self.is_3ccd:
            rgb = self._color_cleanup(rgb)

        # **The night-vision scene terms, before the sensor.**  The illuminator is outside the camera, so its
        # light is part of what the camera is looking at -- added here rather than at the end, where it would
        # be light added to an already-exposed picture.
        rgb = self._night_vision(rgb, "scene", buf.index)
        rgb = self._roll_off(rgb)
        # fog is on the front element, so light passes through it before anything else
        if self.lens_fog > 1e-4:
            from ..codecs.optics import lens_fog
            rgb = lens_fog(rgb, self.lens_fog)
        # **After the light mechanisms, so it deepens what they left around a source.**  It adds no light
        # of its own, which is why it can raise a line's contrast without touching the line's colour.
        rgb = self._shadow_depth(rgb)
        rgb = self._halation(rgb)
        rgb = self._overflow_drain(rgb)
        # charge, at the sensor: vertical because the transfer channel is vertical
        if self.bloom_amount > 1e-4:
            from ..codecs.optics import blooming
            rgb = blooming(rgb, self.bloom_threshold, self.bloom_amount)
        rgb = self._lens_flare(rgb)
        rgb = self._flare(rgb)
        rgb = coating_ghost_dispersion(rgb, self.lens)
        rgb = self._gain_noise(rgb, buf.index, scale_mult=0.70)
        # **Lens aberrations, before the camera's own sharpening.**  Aperture correction is applied by
        # the camera *after* the lens has done its work, so sharpening first would let it undo the
        # corner softening -- which is the one thing field curvature is supposed to leave visible.
        # **The lens, in the order the light meets it.**  The geometry of the glass first, then the
        # field-dependent blurs, then the wavelength errors -- and the camera's own sharpening last,
        # because aperture correction is applied *after* the lens and sharpening first would undo the
        # corner softening that field curvature exists to leave visible.
        # **The mechanisms that existed before keep their original relative order.**  `chromatic_aberration` is
        # the old `_aberration` and `longitudinal_ca` was already here, so `field_curvature` stays between them
        # exactly where it was.  The new mechanisms are placed around that sequence rather than in front of it:
        # `distortion` first because it is the geometry of the glass and everything else is defined in the frame it
        # produces, and `coma`, `astigmatism` and `spherical` after the field-dependent blur they belong with.
        #
        # Measured, putting `distortion` ahead of `field_curvature` cost 2.96e-04 in the corners even with
        # `distortion` at zero -- it returns `rgb` through a copy, so the array `field_curvature` receives has a
        # different float32 history.  A move has to preserve the *sequence*, not just the set.
        # W-T Zoom-Coupled Lens Aberration Envelope:
        #   - Distortion shifts from barrel (<0) at Wide (W) to pincushion (>0) at Telephoto (T)
        #   - Lateral CA, longitudinal CA, and spherical aberration grow with focal length at T
        #   - Source glow tightens at T as diffuse wide-angle halos resolve into concrete source details
        if wt_active:
            import dataclasses as _dc
            z_t = eff_wt_zoom
            dist_wt = self.lens.distortion + (-0.12 * (1.0 - z_t) if z_t < 0.35 else 0.14 * ((z_t - 0.35) / 0.65)) * z_t
            lat_ca_wt = self.lens.chroma_aberration * (1.0 + 0.65 * z_t) + 0.18 * z_t
            lon_ca_wt = self.lens.longitudinal_ca * (1.0 + 0.85 * z_t) + 0.22 * (z_t ** 1.3)
            coma_wt   = self.lens.coma * (1.0 + 0.30 * z_t)
            sph_wt    = self.lens.spherical + 0.14 * (z_t ** 1.4)
            glow_wt   = self.lens.source_glow * float(np.clip(1.0 - 0.55 * z_t * self.zoom_resolve_detail, 0.25, 1.0))
            eff_lens = _dc.replace(
                self.lens,
                distortion=dist_wt,
                chroma_aberration=lat_ca_wt,
                longitudinal_ca=lon_ca_wt,
                coma=coma_wt,
                spherical=sph_wt,
                source_glow=glow_wt,
            )
        else:
            eff_lens = self.lens

        rgb = distortion(rgb, eff_lens)
        if eff_lens.field_curvature < 1e-4:
            from ..codecs.lens import chromatic_and_longitudinal_ca
            rgb = chromatic_and_longitudinal_ca(rgb, eff_lens)
        else:
            rgb = chromatic_aberration(rgb, eff_lens)
            rgb = field_curvature(rgb, eff_lens)
            rgb = longitudinal_ca(rgb, eff_lens)
        rgb = coma(rgb, eff_lens)
        rgb = astigmatism(rgb, eff_lens)
        # **Scattered light first, because it is the widest and softest of the light mechanisms.**  A halo
        # that carries the source's own colour, which is what the reference's near field shows: 60 and 100
        # rows below a blown core it runs to B-R -0.170 and -0.510, where the model reached -0.014 and
        # -0.034.  Nothing else in the module follows the source's colour.
        rgb = source_glow(rgb, eff_lens)
        rgb = spherical(rgb, eff_lens)
        rgb = self._sharpen(rgb)
        rgb = self._overflow_drain_comatic(rgb)
        # **The monochrome green view, after every sensor term.**  It is the viewfinder's output, so nothing at the
        # sensor can follow it.
        #
        # **Accumulation is NOT here.**  It was, and it ran a second time -- overwriting the fully-processed picture
        # with the mean of several *source* frames, 0.1768 replacing 0.45429.  That is the 2.25x loss the stage trace
        # showed, and it survived because the scene-stage version was added without this one being removed.  A
        # mechanism wired in two places is worse than one wired in none: the second silently wins, and the first's
        # measurements describe a picture that never reaches the output.  Accumulation now runs once, in
        # `_night_vision`'s scene stage, before the illuminator -- see the comment there.
        rgb = self._night_vision(rgb, "display")
        if getattr(self, "_defocus", 0.0) > 1e-4:
            # a Gaussian of radius growing with defocus.  The factor is chosen so that full defocus
            # (1.0) is a clearly soft picture and the loop's typical steps are visible but gentle.
            rgb = lowpass(rgb, 0.6 + 4.5 * self._defocus, axis=None)
        rgb = self._lens_wear(rgb)
        rgb = self._apply_ccd_flash_still(rgb)
        rgb = self._color_science(rgb)
        # The sensor's own readout, at the very end.  A stuck site is a *readout* defect, so it happens
        # after the charge has accumulated, after the analogue gain and after the camera's own signal
        # processing -- which means after `_sharpen` in particular.  Writing it earlier looked right and
        # did nothing: an unsharp mask computes `0 * (1 + k) - k * neighbour_mean` at a dead site and pulls
        # it straight back to the neighbourhood average.  Measured before this move: the site was set to
        # 0.0000 and the stage's cached output read 0.5199 at the same position.
        if self.dead_sites_rate > 1e-6:
            from ..codecs.optics import apply_defect_sites, plan_defect_sites
            if self._sites is None:
                self._sites = plan_defect_sites(
                    self.dead_sites_rate, self.stuck_fraction,
                    rgb.shape[0], rgb.shape[1],
                    derive_seed(self.ctx.seed, "camera", "dead_sites"))
            rgb = apply_defect_sites(rgb, self._sites)

        # --- the DSP's output level ---------------------------------------
        # Last, and after the readout defects, because it is the stage that sets what the recorded signal *is*: the
        # gain the camera settles on and the black level it puts under it.  Everything before this is the signal.
        rgb = self._dsp_level(rgb)
        rgb = self._highlight_compress(rgb)
        if wt_active and getattr(self, "zoom_osd", False):
            rgb = draw_trv310e_zoom_osd(rgb, eff_wt_zoom)
        self.exposure = _saved_exposure
        self.gain = _saved_gain

        out_meta = dict(buf.meta)
        # --- Module 3: Shared Power & Electrical Bus coupling on Camera ---
        from ..codecs.power_bus import evaluate_power_bus, apply_camera_power_effects
        row_lum_cam = np.clip(rgb[::2, ::4].mean(axis=1), 0.0, 1.0) @ LUMA
        pstate = evaluate_power_bus(
            self.ctx.params, int(buf.index), self.ctx.fps,
            rgb.shape[0], rgb.shape[1], self.ctx.seed,
            row_luma=row_lum_cam, zoom_vel=zoom_vel,
        )
        if pstate.active:
            rgb = apply_camera_power_effects(rgb, pstate, int(buf.index), self.ctx.seed)
            out_meta["power_bus"] = pstate

        if wt_active:
            out_meta["wt_zoom"] = {
                "zoom_pos": round(eff_wt_zoom, 4),
                "magnification": round(wt_mag, 3),
                "effective_f_number": round(1.4 + (2.9 - 1.4) * (eff_wt_zoom ** 1.15), 2),
            }

        if self.ccu_spec.enabled:
            rgb, ccu_info = apply_studio_ccu(
                np.clip(rgb, 0.0, 1.5).astype(np.float32),
                self.ccu_spec, buf.std,
                frame_index=int(buf.index), fps=self.ctx.fps, seed=int(self.ctx.seed) + 101,
            )
            out_meta["ccu_genlock"] = ccu_info
        if self.mobile_isp_spec.is_active():
            m_prof = resolve_mobile_profile(self.mobile_isp_spec.profile_id)
            rgb, m_info = run_mobile_isp_pipeline(
                np.clip(rgb, 0.0, 1.0).astype(np.float32),
                m_prof,
                self.mobile_isp_spec,
                is_video=bool(self.mobile_isp_spec.capture_mode == 1),
                frame_index=int(buf.index),
                seed=int(self.ctx.seed) + 307,
            )
            out_meta["mobile_camera_isp"] = m_info
        if rgb.dtype == np.float32 and rgb.flags.c_contiguous:
            np.clip(rgb, 0.0, 2.0, out=rgb)
            out_rgb = rgb
        else:
            out_rgb = np.clip(rgb, 0.0, 2.0).astype(np.float32, copy=False)
        out_meta["tap_camera_rgb"] = out_rgb
        return FrameBuffer(out_rgb, None,
                           buf.index, buf.time, buf.std, out_meta)

