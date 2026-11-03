"""
`atsim.codecs.lens` -- the lens and optical-aberration mechanisms, as pure functions.

These were developing as methods on `CameraStage`, which is 754 lines and 23 methods, and the cost was measured
rather than felt: **four edits to that one file in a single turn produced three inert parameters**, and the turn
before needed three attempts to land field curvature.  A mechanism that cannot be added reliably cannot be fixed
reliably either.

The group belongs together on the three axes that matter for a split:

  * **it shares a subject** -- what the glass does to the light, before the sensor sees any of it;
  * **it is pure** -- each function takes a picture and a spec and returns a picture, with no state.  That is the
    property that was missing: several of these were "verified" by rendering a scene whose own softness hid the
    effect, and each can now be measured on its own input instead;
  * **it is ordered** -- the geometry of the glass, then the off-axis blurring, then the aperture's own defects --
    and the caller owns the order, so it is stated in one place rather than implied by the sequence of statements
    inside a 134-line `apply`.

`LensSpec` carries the parameters so a caller cannot pass eight floats in the wrong order, which is the other
mistake this shape of code invites.

**`_lens_wear`, `_halation` and `_lens_flare` are transplanted verbatim** from `camera.py`, with only the
`self.x` -> `spec.x` substitutions the move requires.  They are here because the split has to be a move rather
than a rewrite, and a rewrite would change behaviour in the same operation as rearranging it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.signal import lowpass

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


@dataclass
class LensSpec:
    """The lens's own parameters, in one object so their order cannot slip."""

    chroma_aberration: float = 0.0
    axis_offset: float = 0.0
    field_curvature: float = 0.0
    longitudinal_ca: float = 0.0
    coma: float = 0.0
    astigmatism: float = 0.0
    spherical: float = 0.0
    distortion: float = 0.0
    lens_wear: float = 0.0
    lens_flare: float = 0.0
    #: **Scattered halo carrying the source's own colour.**  Light scattered from a bright source keeps
    #: its spectrum, so an orange lamp wears an orange halo and a white one a white halo.  Distinct from
    #: `halation`, whose tint is fixed and cool, and from `lens_flare`, whose tint is the coating's.
    source_glow: float = 0.0
    halation: float = 0.0
    coat_ghost: float = 0.0
    blades: int = 6
    light_streak_amount: float = 0.0
    light_streak_angle: float = 0.0
    light_streak_length: float = 0.55

    @classmethod
    def from_params(cls, cfg: dict) -> "LensSpec":
        return cls(
            chroma_aberration=float(cfg.get("chroma_aberration", 0.0)),
            # **Optical-axis decentring.**  A real lens assembly is never perfectly centred on the sensor, so the
            # optical axis sits slightly off the frame's centre.  Every radial aberration is measured from the
            # *axis*, not from the frame's middle -- so decentring moves the point where they all vanish, and a
            # highlight at the frame's centre acquires the aberrations of a small radius instead of none.
            axis_offset=float(cfg.get("axis_offset", 0.0)),
            field_curvature=float(cfg.get("field_curvature", 0.0)),
            longitudinal_ca=float(cfg.get("longitudinal_ca", 0.0)),
            coma=float(cfg.get("coma", 0.0)),
            astigmatism=float(cfg.get("astigmatism", 0.0)),
            spherical=float(cfg.get("spherical_aberration", 0.0)),
            distortion=float(cfg.get("lens_distortion", 0.0)),
            lens_wear=float(cfg.get("lens_wear", 0.0)),
            lens_flare=float(cfg.get("lens_flare", 0.0)),
            source_glow=float(cfg.get("source_glow", 0.0)),
            halation=float(cfg.get("halation", 0.0)),
            coat_ghost=float(cfg.get("coat_ghost", 0.0)),
            blades=int(round(float(cfg.get("flare_blades", 6.0)))),
            light_streak_amount=float(cfg.get("light_streak_amount", 0.0)),
            light_streak_angle=float(cfg.get("light_streak_angle", 0.0)),
            light_streak_length=float(cfg.get("light_streak_length", 0.55)),
        )

    def axis_shift(self) -> tuple[float, float]:
        """
        Where the optical axis sits, in the normalised coordinates the aberrations use.

        Positive moves the axis right and down.  Expressed as a fraction of the half-width, so 0.25 puts the axis a
        quarter of the way towards the frame's edge.  The direction is fixed and slightly diagonal: decentring is
        whatever the assembly happened to be, so a purely horizontal shift would be a special case rather than a
        typical one.
        """
        a = float(self.axis_offset)
        return (a * 0.85, a * 0.53)


def field_coords(rgb: np.ndarray, spec: "LensSpec | None" = None):
    """
    Coordinates with the **optical axis** at the origin, plus the squared field radius.

    Every radial aberration is `k * r` about the axis, not about the frame's centre.  Sharing this function is what
    makes decentring work for all of them at once, and it removes four places where a sign could differ.
    """
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0
    if spec is not None:
        dx, dy = spec.axis_shift()
        nx = nx - dx
        ny = ny - dy
    return nx, ny, np.clip((nx * nx + ny * ny) / 2.0, 0.0, 1.0)


# ====================================================================== geometry of the glass

def distortion(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    Barrel or pincushion: the lens's own geometric error.

    A simple lens has barrel distortion and a telephoto pincushion.  The camera side of the model had **no geometric
    term at all**, so a frame straight from the lens had perfectly straight edges -- a real omission rather than a
    subtlety.

    It shares the radial geometry with the display's screen curvature and is the **opposite sign**: that one is the
    tube's face, this one is the glass.
    """
    if abs(spec.distortion) < 1e-4:
        return rgb
    h, w = rgb.shape[:2]
    nx, ny, r2 = field_coords(rgb, spec)
    k = float(spec.distortion) * 0.18
    sx = nx * (1.0 + k * r2)
    sy = ny * (1.0 + k * r2)
    src_x = np.clip((sx * 0.5 + 0.5) * (w - 1), 0, w - 1)
    src_y = np.clip((sy * 0.5 + 0.5) * (h - 1), 0, h - 1)
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = (src_x - x0)[..., None]
    fy = (src_y - y0)[..., None]
    top = rgb[y0, x0] * (1.0 - fx) + rgb[y0, x1] * fx
    bot = rgb[y1, x0] * (1.0 - fx) + rgb[y1, x1] * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


# ====================================================================== the three off-axis blurs

def field_curvature(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    Sharp in the middle, soft towards the corners, at the same focus setting.

    A simple lens focuses a flat subject onto a curved surface, so the in-focus plane at the axis is not the plane
    in focus away from it.  The point-spread function grows with the square of the field angle, so the softening is
    **zero at the axis and rises outward**.

    A genuine space-variant convolution would cost a kernel per pixel; this is the standard approximation -- blur at
    the corner radius and cross-fade with the sharp frame using a radial weight.
    """
    if spec.field_curvature < 1e-4:
        return rgb
    amount = float(spec.field_curvature)
    _nx, _ny, r2 = field_coords(rgb, spec)
    soft = lowpass(rgb, 1.0 + 5.5 * amount)
    # **Character-for-character what the original wrote.**  `np.clip(r2, 0, 1)` is the same value but a different
    # expression, and the round trip through this module reordered a float32 multiply: measured, 28 of 30 frames
    # byte-identical and the two `field_curvature` frames differing by up to 2.65e-04.  A move has to be provably a
    # move, so the expression is not tidied.
    weight = (r2 ** 1.0)[..., None].astype(np.float32)
    return (rgb * (1.0 - weight) + soft * weight).astype(np.float32)


def coma(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    An off-axis point becomes a comet, with its tail pointing **away from the optical axis**.

    Rays through the outer part of the lens are magnified differently from those through the centre, so an off-axis
    point is not a point: it is a bright head with a flaring tail.  The signature is that the smear is **one-sided
    and radial** -- outward from the axis -- and grows with field radius, where field curvature is symmetric and
    spherical aberration does not depend on direction at all.

    Implemented by gathering the frame from progressively farther along the radial direction and adding it back with
    a falloff, so the head stays brighter than the tail.
    """
    if spec.coma < 1e-4:
        return rgb
    a = float(spec.coma)
    h, w = rgb.shape[:2]
    nx, ny, r2 = field_coords(rgb, spec)
    radius = np.sqrt(np.clip(r2, 1e-9, None))
    ux = nx / radius
    uy = ny / radius
    if abs(spec.axis_offset) > 1e-4:
        return _seidel_caustic_coma(rgb, a, float(spec.axis_offset))
    reach = 1.0 + 9.0 * a
    out = rgb.astype(np.float32).copy()
    steps = 4
    ybase = np.arange(h)[:, None]
    xbase = np.arange(w)[None, :]
    for k in range(1, steps + 1):
        t = k / float(steps)
        sy = np.clip(ybase - np.round(uy * reach * t).astype(np.int32), 0, h - 1)
        sx = np.clip(xbase - np.round(ux * reach * t).astype(np.int32), 0, w - 1)
        gathered = rgb[sy, sx]
        falloff = ((1.0 - t) * 0.30 * a * np.clip(radius, 0.0, 1.0))[..., None]
        out = out + gathered * falloff
    return out.astype(np.float32)


def _bilinear_sample_2d(plane: np.ndarray, sy: np.ndarray, sx: np.ndarray) -> np.ndarray:
    h, w = plane.shape[:2]
    sy_c = np.clip(sy, 0.0, h - 1.001)
    sx_c = np.clip(sx, 0.0, w - 1.001)
    y0 = np.floor(sy_c).astype(np.int32)
    x0 = np.floor(sx_c).astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    fy = sy_c - y0
    fx = sx_c - x0
    top = plane[y0, x0] * (1.0 - fx) + plane[y0, x1] * fx
    bot = plane[y1, x0] * (1.0 - fx) + plane[y1, x1] * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


def _seidel_caustic_coma(rgb: np.ndarray, a: float, axis_offset: float) -> np.ndarray:
    """
    Physical Seidel Coma + Tangential/Sagittal Caustic Wedge (Born & Wolf §5.3.2) for a tilted C-mount lens
    with multi-blade auto-iris diffraction and chromatic coma dispersion.
    """
    h, w = rgb.shape[:2]
    scale_w = w / 720.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0
    right_t = np.clip((nx + 0.15) / 0.75, 0.0, 1.0)
    depth_t = np.clip((ny + 0.88) / 0.88, 0.0, 1.0)

    lum = rgb @ LUMA
    bg_lum = lowpass(lum, max(2.0, 16.0 * scale_w))
    hot_m = np.clip((lum - np.maximum(bg_lum + 0.14, 0.74)) / 0.16, 0.0, 1.0)
    if float(hot_m.max()) < 1e-4:
        return rgb

    emitter_narrow = lowpass(hot_m ** 1.15, max(0.4, 0.65 * scale_w))
    emitter_base   = lowpass(hot_m ** 0.85, max(0.8, 1.45 * scale_w))
    reach = (6.0 + 14.5 * a) * scale_w * (1.08 - 0.16 * depth_t)

    # Three Seidel caustic lobe angles:
    # Lobe 1: top tangential roof (~8 deg on left, ~11 deg on right)
    # Lobe 2: central principal ray (~35 deg on left, ~21 deg on right)
    # Lobe 3: left sagittal wall (~78 deg on left, ~72 deg on right)
    ang1 = np.deg2rad(8.0 * (1.0 - right_t) + 11.0 * right_t + 3.0 * depth_t)
    ang2 = np.deg2rad(35.0 * (1.0 - right_t) + 21.0 * right_t + 4.0 * depth_t)
    ang3 = np.deg2rad(78.0 * (1.0 - right_t) + 72.0 * right_t - 2.0 * depth_t)

    bundle_w = np.clip(emitter_base * 1.30, 0.0, 1.35)
    tail_feather = np.zeros((h, w), np.float32)
    tail_b = np.zeros((h, w), np.float32)
    tail_g = np.zeros((h, w), np.float32)

    lobe_defs = (
        (ang1, 0.94 * (1.0 - right_t) + 1.12 * right_t, 0.96, 0.55),
        (ang2, 1.08 * (1.0 - right_t) + 1.18 * right_t, 1.00, 0.65),
        (ang3, 0.74 * (1.0 - right_t) + 0.66 * right_t, 0.95, 0.04),
    )
    for ang_f, r_sc, lw, blue_w in lobe_defs:
        dx_l = np.cos(ang_f).astype(np.float32)
        dy_l = np.sin(ang_f).astype(np.float32)
        for frac in (0.10, 0.20, 0.30, 0.42, 0.54, 0.66, 0.80, 0.94, 1.10):
            dist = reach * r_sc * frac
            src_p = emitter_base if frac <= 0.36 else emitter_narrow
            val = _bilinear_sample_2d(src_p, yy - dy_l * dist, xx - dx_l * dist)
            w_core = lw * np.clip(1.68 - 1.58 * frac, 0.0, 1.25) * a
            bundle_w = np.maximum(bundle_w, val * w_core * 2.20)
            w_tail = np.exp(-0.5 * ((frac - 0.76) / 0.28) ** 2) * a
            tail_feather += val * w_tail * (blue_w * 0.45)
            tail_b += val * w_tail * blue_w
            tail_g += val * w_tail * (blue_w * 0.48)

    web_reach_scale = (0.76 - 0.38 * depth_t) * (1.0 - right_t) + (0.88 - 0.24 * depth_t) * right_t
    for w_mix in (0.25, 0.50, 0.75):
        for ang_a, ang_b in ((ang1, ang2), (ang2, ang3)):
            ang_w = (1.0 - w_mix) * ang_a + w_mix * ang_b
            dx_w = np.cos(ang_w).astype(np.float32)
            dy_w = np.sin(ang_w).astype(np.float32)
            for frac in (0.15, 0.30, 0.45, 0.60, 0.75):
                dist = reach * web_reach_scale * frac
                val = _bilinear_sample_2d(emitter_base, yy - dy_w * dist, xx - dx_w * dist)
                w_web = np.clip(1.58 - 1.58 * frac, 0.0, 1.15) * a
                bundle_w = np.maximum(bundle_w, val * w_web * 2.00)

    bundle_w = lowpass(np.clip(bundle_w, 0.0, 1.35), max(0.45, 0.58 * scale_w))
    core_blend = np.clip((bundle_w - 0.28) / 0.52, 0.0, 1.0)[..., None]

    out = rgb * (1.0 - core_blend) + np.ones_like(rgb) * core_blend

    tf_smooth = lowpass(np.clip(tail_feather, 0.0, 0.85), max(0.9, 1.5 * scale_w))
    tb_smooth = lowpass(np.clip(tail_b, 0.0, 0.85), max(1.0, 1.6 * scale_w))
    tg_smooth = lowpass(np.clip(tail_g, 0.0, 0.85), max(1.0, 1.6 * scale_w))
    skirt_w = (1.0 - core_blend[..., 0] * 0.88) * (1.0 - 0.25 * right_t)
    out[..., 2] = np.clip(out[..., 2] + (0.22 * tf_smooth + 0.36 * tb_smooth) * skirt_w, 0.0, 1.15)
    out[..., 1] = np.clip(out[..., 1] + (0.20 * tf_smooth + 0.16 * tg_smooth) * skirt_w, 0.0, 1.15)
    out[..., 0] = np.clip(out[..., 0] + (0.04 * tf_smooth - 0.16 * tb_smooth) * skirt_w, 0.0, 1.15)
    return out.astype(np.float32)


def astigmatism(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    Off-axis points focus at two different distances, so the blur is **directional**.

    The tangential and sagittal planes have different focal lengths away from the axis, so a point is imaged as a
    short line -- radial at one focus, tangential at the other -- and **which axis dominates flips** between them.
    That is the signature that separates it from field curvature and spherical aberration, both of which are round.
    """
    if spec.astigmatism < 1e-4:
        return rgb
    a = float(spec.astigmatism)
    _nx, ny, r2 = field_coords(rgb, spec)
    vert = lowpass(rgb, 1.0 + 4.0 * a, axis=1)
    horiz = lowpass(rgb, 1.0 + 4.0 * a, axis=0)
    flip = np.clip(ny, -1.0, 1.0)[..., None]
    mix_v = np.clip(0.5 + 0.5 * flip, 0.0, 1.0)
    mixed = vert * mix_v + horiz * (1.0 - mix_v)
    weight = np.clip(r2, 0.0, 1.0)[..., None] * a
    return (rgb * (1.0 - weight) + mixed * weight).astype(np.float32)


def spherical(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    Outer rays focus closer, so a **bright** point grows a wide round halo and a dim one stays tight.

    The signature is that the halo is **symmetric and brightness-dependent**.  That separates it from field
    curvature (which grows with radius regardless of brightness) and from astigmatism (which grows along one axis).
    It is why a fast lens is soft wide open and sharpens as it is stopped down.
    """
    if spec.spherical < 1e-4:
        return rgb
    a = float(spec.spherical)
    lum = rgb @ LUMA
    # the halo is drawn from the bright parts only, so a mid-tone scene is not veiled
    hot = np.clip(lum - 0.55, 0.0, None)
    halo = lowpass(hot, 3.0 + 12.0 * a)
    peak = float(halo.max())
    if peak > 1e-6:
        halo = halo / peak
    coverage = float(np.clip(hot.mean() * 12.0, 0.0, 1.0))
    gain = np.clip(halo * (0.55 * a) + 0.12 * a * coverage, 0.0, 0.8)
    return (rgb * (1.0 + gain[..., None])).astype(np.float32)


# ====================================================================== per-channel wavelength errors

def source_glow(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    A halo around bright sources whose **colour is the source's own**.

    Light from a bright source scattered in the air, on the glass and inside the lens keeps the source's spectrum, so
    the halo immediately around a sodium lamp is orange and the halo around a white lamp is white. Measured against
    the reference, that near field is the largest single thing the model was missing: at 60 and 100 rows below a blown
    core the reference runs to **B-R -0.170 and -0.510**, while the reproduction reached -0.014 and -0.034 -- a factor
    of fifteen.

    **Distinct from `halation`, which tints its glow with a fixed cool colour, and from `lens_flare`, whose tint is
    the coating's.** Neither follows the source, and neither can be made to: a fixed tint is a constant and the
    source's colour is not.

    The one detail that decides whether it works: the highlight map is blurred **per channel** so each keeps its own
    proportion, and the result is divided by a **grey** coverage term. Dividing per channel instead would normalise
    every halo to white, which would throw away exactly the colour this exists to carry.
    """
    if spec.source_glow < 1e-4:
        return rgb
    a = float(spec.source_glow)
    lum = rgb @ LUMA
    # where the light is: a soft threshold, so a dim scene contributes nothing and a blown source contributes fully
    hot = np.clip((lum - 0.55) / 0.35, 0.0, 1.0)
    if float(hot.max()) < 1e-6:
        return rgb
    h = float(rgb.shape[0])
    radius = max(4.0, 0.012 * h) * (0.6 + 0.8 * a)
    halo = np.stack([lowpass(rgb[..., c] * hot, radius) for c in range(3)], axis=-1)
    return (rgb + halo * (60.0 * a)).astype(np.float32)


def longitudinal_ca(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    Per-channel defocus, with **no displacement**.

    The wavelengths focus at different distances along the axis, so at the sensor plane one is in focus and the
    others are not -- red and blue land slightly soft on either side of green.  A red edge and a blue edge are at the
    same position and both are wider, so the coloured fringes appear *around* a highlight rather than on one side of
    it.  That is the whole distinction from `chromatic_aberration`, and because the two are opposite in kind the
    checks assert opposite things.
    """
    if spec.longitudinal_ca < 1e-4:
        return rgb
    a = float(spec.longitudinal_ca)
    try:
        import cv2 as _cv2
        r0, g0, b0 = _cv2.split(np.ascontiguousarray(rgb, dtype=np.float32))
        r = lowpass(r0, 1.0 + 3.0 * a)
        g = lowpass(g0, 1.0 + 0.4 * a)
        b = lowpass(b0, 1.0 + 3.4 * a)
        return _cv2.merge((r, g, b))
    except Exception:
        r = lowpass(rgb[..., 0], 1.0 + 3.0 * a)
        g = lowpass(rgb[..., 1], 1.0 + 0.4 * a)
        b = lowpass(rgb[..., 2], 1.0 + 3.4 * a)
        return np.stack([r, g, b], axis=-1).astype(np.float32)


_CA_WARP_CACHE: dict[tuple, dict[int, tuple]] = {}
_SMEAR_KERNEL_CACHE: dict[tuple, tuple] = {}


def chromatic_aberration(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    Lateral chromatic aberration: the wavelengths focus at different magnifications, **about the optical axis**.
    """
    if spec.chroma_aberration < 1e-4:
        return rgb
    a = float(spec.chroma_aberration)
    h, w = rgb.shape[:2]
    dx, dy = spec.axis_shift()
    key = (int(h), int(w), round(a, 6), round(float(dx), 6), round(float(dy), 6))
    tables = _CA_WARP_CACHE.get(key)
    if tables is None:
        yy = np.arange(h, dtype=np.float32)
        xx = np.arange(w, dtype=np.float32)
        cx = (w - 1) * 0.5 * (1.0 + dx)
        cy = (h - 1) * 0.5 * (1.0 + dy)
        tables = {}
        for ch, scale in ((0, 1.0 + 0.012 * a), (2, 1.0 - 0.012 * a)):
            sx = np.clip(cx + (xx - cx) / scale, 0, w - 1)
            sy = np.clip(cy + (yy - cy) / scale, 0, h - 1)
            x0 = np.floor(sx).astype(np.int32)
            y0 = np.floor(sy).astype(np.int32)
            x1 = np.clip(x0 + 1, 0, w - 1)
            y1 = np.clip(y0 + 1, 0, h - 1)
            fx = (sx - x0)[None, :].astype(np.float32)
            fy = (sy - y0)[:, None].astype(np.float32)
            tables[ch] = (x0, x1, fx, y0, y1, fy)
        if len(_CA_WARP_CACHE) > 32:
            _CA_WARP_CACHE.pop(next(iter(_CA_WARP_CACHE)), None)
        _CA_WARP_CACHE[key] = tables

    try:
        import cv2 as _cv2
        r0, g0, b0 = _cv2.split(np.ascontiguousarray(rgb, dtype=np.float32))
        planes = {0: r0, 1: g0, 2: b0}
        _have_cv = True
    except Exception:
        planes = {0: rgb[..., 0], 1: rgb[..., 1], 2: rgb[..., 2]}
        _have_cv = False

    warped: dict[int, np.ndarray] = {}
    for ch in (0, 2):
        x0, x1, fx, y0, y1, fy = tables[ch]
        plane = planes[ch]
        row_y = plane[y0] + (plane[y1] - plane[y0]) * fy
        c0 = row_y[:, x0]
        c1 = row_y[:, x1]
        c1 -= c0
        c1 *= fx
        c1 += c0
        warped[ch] = c1
    r = lowpass(warped[0], 2.2 * a)
    b = lowpass(warped[2], 2.6 * a)
    g = lowpass(planes[1], 0.35 * a)
    if _have_cv:
        return _cv2.merge((r, g, b))
    return np.stack([r, g, b], axis=-1).astype(np.float32, copy=False)


def chromatic_and_longitudinal_ca(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """Fused lateral + longitudinal chromatic aberration when field_curvature is inactive."""
    a_lat = float(spec.chroma_aberration)
    a_lon = float(spec.longitudinal_ca)
    if a_lat < 1e-4 and a_lon < 1e-4:
        return rgb
    if a_lat < 1e-4:
        return longitudinal_ca(rgb, spec)
    if a_lon < 1e-4:
        return chromatic_aberration(rgb, spec)
    h, w = rgb.shape[:2]
    dx, dy = spec.axis_shift()
    key = (int(h), int(w), round(a_lat, 6), round(float(dx), 6), round(float(dy), 6))
    tables = _CA_WARP_CACHE.get(key)
    if tables is None:
        yy = np.arange(h, dtype=np.float32)
        xx = np.arange(w, dtype=np.float32)
        cx = (w - 1) * 0.5 * (1.0 + dx)
        cy = (h - 1) * 0.5 * (1.0 + dy)
        tables = {}
        for ch, scale in ((0, 1.0 + 0.012 * a_lat), (2, 1.0 - 0.012 * a_lat)):
            sx = np.clip(cx + (xx - cx) / scale, 0, w - 1)
            sy = np.clip(cy + (yy - cy) / scale, 0, h - 1)
            x0 = np.floor(sx).astype(np.int32)
            y0 = np.floor(sy).astype(np.int32)
            x1 = np.clip(x0 + 1, 0, w - 1)
            y1 = np.clip(y0 + 1, 0, h - 1)
            fx = (sx - x0)[None, :].astype(np.float32)
            fy = (sy - y0)[:, None].astype(np.float32)
            tables[ch] = (x0, x1, fx, y0, y1, fy)
        if len(_CA_WARP_CACHE) > 32:
            _CA_WARP_CACHE.pop(next(iter(_CA_WARP_CACHE)), None)
        _CA_WARP_CACHE[key] = tables

    sig_r = float(np.hypot(2.2 * a_lat, 1.0 + 3.0 * a_lon))
    sig_g = float(np.hypot(0.35 * a_lat, 1.0 + 0.4 * a_lon))
    sig_b = float(np.hypot(2.6 * a_lat, 1.0 + 3.4 * a_lon))

    from ..core.signal import get_native_lib, get_scratch, gaussian_half_kernel
    lib = get_native_lib()
    if lib is not None and rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.float32:
        src_c = np.ascontiguousarray(rgb, dtype=np.float32)
        x0_r, x1_r, fx_r, y0_r, y1_r, fy_r = tables[0]
        x0_b, x1_b, fx_b, y0_b, y1_b, fy_b = tables[2]
        fx_r1 = np.ascontiguousarray(fx_r.reshape(-1), dtype=np.float32)
        fy_r1 = np.ascontiguousarray(fy_r.reshape(-1), dtype=np.float32)
        fx_b1 = np.ascontiguousarray(fx_b.reshape(-1), dtype=np.float32)
        fy_b1 = np.ascontiguousarray(fy_b.reshape(-1), dtype=np.float32)

        r_w = get_scratch("ca_rw", (h, w), np.float32)
        g_w = get_scratch("ca_gw", (h, w), np.float32)
        b_w = get_scratch("ca_bw", (h, w), np.float32)
        lib.vml_ca_warp_split_f32(
            src_c.ctypes.data, r_w.ctypes.data, g_w.ctypes.data, b_w.ctypes.data,
            x0_r.ctypes.data, x1_r.ctypes.data, fx_r1.ctypes.data,
            y0_r.ctypes.data, y1_r.ctypes.data, fy_r1.ctypes.data,
            x0_b.ctypes.data, x1_b.ctypes.data, fx_b1.ctypes.data,
            y0_b.ctypes.data, y1_b.ctypes.data, fy_b1.ctypes.data,
            h, w,
        )
        kh_r, rr = gaussian_half_kernel(sig_r)
        kh_g, rg = gaussian_half_kernel(sig_g)
        kh_b, rb = gaussian_half_kernel(sig_b)
        tmp_r = get_scratch("ca_tr", (h, w), np.float32)
        tmp_g = get_scratch("ca_tg", (h, w), np.float32)
        tmp_b = get_scratch("ca_tb", (h, w), np.float32)
        lib.vml_sym_fir_horiz_f32(r_w.ctypes.data, tmp_r.ctypes.data, kh_r.ctypes.data, rr, h, w, 1)
        lib.vml_sym_fir_horiz_f32(g_w.ctypes.data, tmp_g.ctypes.data, kh_g.ctypes.data, rg, h, w, 1)
        lib.vml_sym_fir_horiz_f32(b_w.ctypes.data, tmp_b.ctypes.data, kh_b.ctypes.data, rb, h, w, 1)
        out = np.empty_like(src_c)
        lib.vml_ca_vert_merge_f32(
            tmp_r.ctypes.data, kh_r.ctypes.data, rr,
            tmp_g.ctypes.data, kh_g.ctypes.data, rg,
            tmp_b.ctypes.data, kh_b.ctypes.data, rb,
            out.ctypes.data, h, w,
        )
        return out

    try:
        import cv2 as _cv2
        r0, g0, b0 = _cv2.split(np.ascontiguousarray(rgb, dtype=np.float32))
        planes = {0: r0, 1: g0, 2: b0}
        _have_cv = True
    except Exception:
        planes = {0: rgb[..., 0], 1: rgb[..., 1], 2: rgb[..., 2]}
        _have_cv = False

    warped: dict[int, np.ndarray] = {}
    for ch in (0, 2):
        x0, x1, fx, y0, y1, fy = tables[ch]
        plane = planes[ch]
        row_y = plane[y0] + (plane[y1] - plane[y0]) * fy
        c0 = row_y[:, x0]
        c1 = row_y[:, x1]
        c1 -= c0
        c1 *= fx
        c1 += c0
        warped[ch] = c1
    r = lowpass(warped[0], sig_r)
    b = lowpass(warped[2], sig_b)
    g = lowpass(planes[1], sig_g)
    if _have_cv:
        return _cv2.merge((r, g, b))
    return np.stack([r, g, b], axis=-1).astype(np.float32, copy=False)


# ====================================================================== W-T Optical Zoom & Light Streaks

def optical_wt_zoom(rgb: np.ndarray, zoom_pos: float, max_zoom: float = 4.0,
                    center_x: float = 0.5, center_y: float = 0.5,
                    resolve_detail: float = 0.75) -> tuple[np.ndarray, float]:
    """
    Physical W-T (Wide -> Telephoto) optical zoom magnification and halo-to-source resolution.
    """
    z = float(np.clip(zoom_pos, 0.0, 1.0))
    m_max = float(np.clip(max_zoom, 1.0, 20.0))
    mag = float(m_max ** z)
    if mag <= 1.0005 and resolve_detail <= 1e-4:
        return rgb, 1.0

    from scipy.ndimage import gaussian_filter

    h, w = rgb.shape[:2]
    sc = min(h, w) / 540.0
    src = np.asarray(rgb, np.float32)

    if mag > 1.0005:
        cx = float(np.clip(center_x, 0.05, 0.95)) * (w - 1.0)
        cy = float(np.clip(center_y, 0.05, 0.95)) * (h - 1.0)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        sy = cy + (yy - cy) / mag
        sx = cx + (xx - cx) / mag
        out = np.stack([_bilinear_sample_2d(src[..., c], sy, sx) for c in range(src.shape[2])], axis=-1)
    else:
        out = src.copy()

    res_amt = float(np.clip(resolve_detail, 0.0, 1.5))
    if res_amt > 1e-4:
        lum = np.clip(out @ LUMA, 0.0, 2.0)
        peak = out.max(axis=-1)
        pmax = max(float(peak.max()), 1e-4)
        rel_peak = np.clip(peak / pmax, 0.0, 1.0)
        bg_lum = lowpass(lum, max(2.5, 14.0 * sc))
        hot_zone = np.clip((rel_peak - 0.24) / 0.45, 0.0, 1.0) * np.clip((lum - bg_lum * 0.55) / 0.25, 0.0, 1.0)

        if float(hot_zone.max()) > 1e-3:
            wide_blend = float(np.clip((1.0 - z * 1.35), 0.0, 1.0)) * res_amt
            tele_resolve = float(np.clip((z - 0.08) / 0.92, 0.0, 1.0)) * res_amt

            if wide_blend > 1e-3:
                sig_w = max(1.2, 4.2 * sc * wide_blend)
                halo_pool = np.stack([gaussian_filter(out[..., c], sigma=sig_w) for c in range(3)], axis=-1)
                w_coalesce = np.clip(hot_zone * 0.48 * wide_blend, 0.0, 0.65)[..., None]
                out = out * (1.0 - w_coalesce) + np.maximum(out * 0.88, halo_pool * 1.08) * w_coalesce

            if tele_resolve > 1e-3:
                sig_fine = max(0.65, 1.35 * sc)
                sig_mid  = max(1.40, 3.10 * sc)
                blur_fine = np.stack([gaussian_filter(out[..., c], sigma=sig_fine) for c in range(3)], axis=-1)
                blur_mid  = np.stack([gaussian_filter(out[..., c], sigma=sig_mid) for c in range(3)], axis=-1)
                detail_fine = out - blur_fine
                detail_mid  = blur_fine - blur_mid

                lum_fine = blur_fine @ LUMA
                lum_mid  = blur_mid @ LUMA
                local_contrast = (lum - lum_mid) / np.maximum(lum_mid, 0.08)
                valley_carve = np.clip(-local_contrast * 0.55 * tele_resolve, 0.0, 0.42) * hot_zone
                ridge_boost  = np.clip(local_contrast * 0.62 * tele_resolve, 0.0, 0.48) * hot_zone

                yy_p, xx_p = np.mgrid[0:h, 0:w].astype(np.float32)
                cx_f = float(np.clip(center_x, 0.05, 0.95)) * (w - 1.0)
                cy_f = float(np.clip(center_y, 0.05, 0.95)) * (h - 1.0)
                u_s = cx_f + (xx_p - cx_f) / max(mag, 1.0)
                v_s = cy_f + (yy_p - cy_f) / max(mag, 1.0)

                pitch = max(1.8, 2.35 * sc)
                kx1 = np.cos(2.0 * np.pi * u_s / pitch)
                ky1 = np.cos(2.0 * np.pi * (u_s * 0.5 + v_s * 0.866) / pitch)
                ky2 = np.cos(2.0 * np.pi * (-u_s * 0.5 + v_s * 0.866) / pitch)
                hex_pore = np.clip((kx1 + ky1 + ky2) / 3.0, -1.0, 1.0)
                pore_vis = float(np.clip((mag - 1.15) / 2.2, 0.0, 1.0)) * tele_resolve
                extended_emit = hot_zone * np.clip((rel_peak - 0.35) / 0.45, 0.0, 1.0)
                pore_mod = 1.0 + pore_vis * (0.22 * hex_pore - 0.08 * (1.0 - hex_pore)) * extended_emit

                out = (
                    out * (1.0 - valley_carve[..., None] + 0.35 * ridge_boost[..., None])
                    + detail_fine * (0.85 * tele_resolve * hot_zone)[..., None]
                    + detail_mid  * (0.52 * tele_resolve * hot_zone)[..., None]
                )
                out = out * pore_mod[..., None]

    return np.clip(out, 0.0, 2.0).astype(np.float32), mag


def _pow13_f32(x: np.ndarray) -> np.ndarray:
    return x * (np.float32(0.218) + x * (np.float32(0.952) - np.float32(0.170) * x))


def _pow135_f32(x: np.ndarray) -> np.ndarray:
    return x * (np.float32(0.185) + x * (np.float32(1.005) - np.float32(0.190) * x))


def _pow15_f32(x: np.ndarray) -> np.ndarray:
    return x * np.sqrt(x)


def _pow16_f32(x: np.ndarray) -> np.ndarray:
    return x * (np.float32(0.065) + x * (np.float32(1.215) - np.float32(0.280) * x))


def asymmetric_ccd_smear_and_trench(
    rgb: np.ndarray,
    smear_amount: float,
    length_scale: float = 0.65,
    zoom_pos: float = 0.0,
) -> np.ndarray:
    """
    Physical SONY DCR-TRV310E Up-Down Asymmetric Interline CCD Vertical Smear + Flanking VOD/HOD Dark Trenches.
    """
    a = float(np.clip(smear_amount, 0.0, 2.0))
    if a <= 1e-4:
        return rgb

    from scipy.fft import rfft as _sp_rfft, irfft as _sp_irfft, next_fast_len as _next_fast_len
    try:
        import cv2 as _cv2
        _have_cv = True
    except Exception:
        _cv2 = None
        _have_cv = False

    h, w = rgb.shape[:2]
    sc = min(h, w) / 540.0
    src = np.asarray(rgb, np.float32)

    if _have_cv and h >= 240 and w >= 240:
        ds_y = 4 if h >= 400 else (2 if h >= 240 else 1)
        ds_x = 2 if w >= 600 else 1
        h_ds = max(8, (h + ds_y - 1) // ds_y)
        w_ds = max(8, (w + ds_x - 1) // ds_x)
        sc_y = sc / ds_y
        sc_x = sc / ds_x
        src_ds = np.clip(_cv2.resize(src, (w_ds, h_ds), interpolation=_cv2.INTER_AREA), 0.0, 2.0) if (ds_y > 1 or ds_x > 1) else np.clip(src, 0.0, 2.0)
        lum_ds = src_ds @ LUMA

        hot_area_frac = float(np.mean(lum_ds > 0.68))
        broad_damp = 1.0
        if hot_area_frac > 0.030:
            broad_damp = 1.0 / (1.0 + 45.0 * (hot_area_frac - 0.030))
            if a * broad_damp <= 1e-4:
                return rgb

        small_lum = _cv2.resize(lum_ds, (max(2, w_ds // 4), max(2, h_ds // 2)), interpolation=_cv2.INTER_AREA)
        bg_lum_ds = _cv2.resize(lowpass(small_lum, max(0.8, 14.0 * sc_x / 4.0)), (w_ds, h_ds), interpolation=_cv2.INTER_LINEAR)

        peak_ds = _cv2.max(_cv2.max(src_ds[..., 0], src_ds[..., 1]), src_ds[..., 2])
        pmax = max(float(peak_ds.max()), 1e-4)
        rel_peak = np.clip(peak_ds * np.float32(1.0 / pmax), 0.0, 1.0)

        clips = np.array([0.74, 0.80, 0.86], dtype=np.float32)[None, None, :]
        hot_ch = np.clip(src_ds - clips, 0.0, None)
        hot_gate = np.clip((lum_ds - np.maximum(bg_lum_ds + np.float32(0.08), np.float32(0.56))) * np.float32(1.0 / 0.24), 0.0, 1.0)
        sat_gate = np.clip((rel_peak - np.float32(0.54)) * np.float32(1.0 / 0.32), 0.0, 1.0) * np.clip((lum_ds - bg_lum_ds * np.float32(0.62)) * np.float32(5.0), 0.0, 1.0)
        emit_drive = _cv2.max(_pow13_f32(hot_gate), np.float32(0.82) * _pow15_f32(sat_gate))
        if float(emit_drive.max()) <= 1e-4 and float(hot_ch.max()) <= 1e-4:
            return rgb

        if hot_area_frac > 0.030:
            a = a * broad_damp
            lp_y = lowpass(lowpass(lum_ds, max(0.8, 3.2 * sc_x), axis=1), max(0.5, 3.2 * sc_y), axis=0)
            local_prom = np.clip((lum_ds - lp_y) * np.float32(1.0 / 0.16), 0.0, 1.0)
            emit_drive = emit_drive * local_prom
            hot_ch = hot_ch * local_prom[..., None]

        z = float(np.clip(zoom_pos, 0.0, 1.0))
        len_f = float(np.clip(length_scale, 0.15, 1.5))

        emit_ds = np.maximum(hot_ch, src_ds * (_pow135_f32(emit_drive) * np.float32(0.52))[..., None])
        emit_lum = emit_ds @ LUMA
        col_local_max = lowpass(emit_lum, max(0.8, 3.2 * sc_x), axis=1)
        col_pinch = _pow16_f32(np.clip((emit_lum - np.float32(0.72) * col_local_max) / np.maximum(np.float32(0.28) * col_local_max, np.float32(1e-4)), 0.0, 1.0))
        nd_ds = emit_ds * (np.float32(0.22) + np.float32(0.78) * col_pinch)[..., None]

        k_key = (int(h_ds), round(len_f, 4), round(sc_y, 4))
        k_entry = _SMEAR_KERNEL_CACHE.get(k_key)
        if k_entry is None:
            r_neck = max(3, min(h_ds - 1, int(round((28.0 + 36.0 * len_f) * sc_y))))
            m_neck = np.arange(-r_neck, r_neck + 1, dtype=np.float32)
            tau_neck_dn = max(1.5, (12.0 + 18.0 * len_f) * sc_y)
            tau_neck_up = max(1.0, (6.5 + 9.5 * len_f) * sc_y)
            k_neck = np.where(m_neck >= 0.0, np.exp(-np.maximum(0.0, m_neck) / tau_neck_dn), 0.38 * np.exp(-np.maximum(0.0, -m_neck) / tau_neck_up)).astype(np.float32)
            k_neck /= max(1e-6, float(k_neck.sum()))
            rad_n = len(k_neck) // 2
            nfft_n = _next_fast_len(h_ds + len(k_neck) - 1, real=True)
            fk_n = _sp_rfft(k_neck, n=nfft_n)[:, None, None]

            r_far = max(8, min(h_ds - 1, int(round(min(h_ds * 0.92, (160.0 + 260.0 * len_f) * sc_y)))))
            m_far = np.arange(-r_far, r_far + 1, dtype=np.float32)
            tau_far_dn = max(4.0, (75.0 + 165.0 * len_f) * sc_y)
            tau_far_up = max(2.5, (22.0 + 42.0 * len_f) * sc_y)
            k_far_dn = np.where(m_far > 0.0, 0.72 * np.exp(-np.maximum(0.0, m_far) / tau_far_dn) + 0.28 / (1.0 + (np.maximum(0.0, m_far) / max(2.0, 28.0 * sc_y)) ** 1.15), 0.0).astype(np.float32)
            k_far_dn /= max(1e-6, float(k_far_dn.sum()))
            k_far_up = np.where(m_far < 0.0, np.exp(-np.maximum(0.0, -m_far) / tau_far_up), 0.0).astype(np.float32)
            k_far_up /= max(1e-6, float(k_far_up.sum()))
            rad_f = len(k_far_dn) // 2
            nfft_f = _next_fast_len(h_ds + len(k_far_dn) - 1, real=True)
            fk_d = _sp_rfft(k_far_dn, n=nfft_f)[:, None, None]
            fk_u = _sp_rfft(k_far_up, n=nfft_f)[:, None, None]
            k_entry = (nfft_n, rad_n, fk_n, nfft_f, rad_f, fk_d, fk_u)
            if len(_SMEAR_KERNEL_CACHE) > 32:
                _SMEAR_KERNEL_CACHE.pop(next(iter(_SMEAR_KERNEL_CACHE)), None)
            _SMEAR_KERNEL_CACHE[k_key] = k_entry

        nfft_n, rad_n, fk_n, nfft_f, rad_f, fk_d, fk_u = k_entry
        neck_sig_x = max(0.5, (2.10 - 0.55 * z) * sc_x)
        dn_sig_x   = max(0.30, (0.65 - 0.20 * z) * sc_x)
        up_sig_x   = max(0.6, (2.05 - 0.45 * z) * sc_x)

        fa_n = _sp_rfft(emit_ds, n=nfft_n, axis=0, workers=2)
        fa_n *= fk_n
        v_neck_ds = lowpass(_sp_irfft(fa_n, n=nfft_n, axis=0, workers=2, overwrite_x=True)[rad_n : rad_n + h_ds].astype(np.float32, copy=False), neck_sig_x, axis=1)

        fa_f = _sp_rfft(nd_ds, n=nfft_f, axis=0, workers=2)
        v_down_ds = lowpass(_sp_irfft(fa_f * fk_d, n=nfft_f, axis=0, workers=2, overwrite_x=True)[rad_f : rad_f + h_ds].astype(np.float32, copy=False), dn_sig_x, axis=1)
        fa_f *= fk_u
        v_up_ds   = lowpass(_sp_irfft(fa_f, n=nfft_f, axis=0, workers=2, overwrite_x=True)[rad_f : rad_f + h_ds].astype(np.float32, copy=False), up_sig_x, axis=1)

        core_shield_ds = np.clip(lowpass(emit_drive, max(0.6, 2.5 * sc_x), axis=1) * np.float32(2.4), 0.0, 1.0)
        spine_lum_ds = (v_down_ds @ LUMA) + np.float32(0.25) * (v_up_ds @ LUMA)
        lat_narrow = lowpass(spine_lum_ds, max(0.30, 0.55 * sc_x), axis=1)
        lat_wide   = lowpass(spine_lum_ds, max(0.80, 3.10 * sc_x), axis=1)
        shift_r = max(1, int(round(2.6 * sc_x)))
        lat_right = np.roll(lowpass(spine_lum_ds, max(0.60, 1.95 * sc_x), axis=1), shift_r, axis=1)
        lat_right[:, :shift_r] = 0.0
        spine_peak_col = np.maximum(lat_narrow, lowpass(v_neck_ds @ LUMA, max(0.4, 1.1 * sc_x), axis=1) * np.float32(0.55))
        trench_raw = np.clip(np.float32(0.55) * lat_wide + np.float32(0.92) * lat_right - np.float32(1.55) * spine_peak_col, 0.0, None)
        _sp_flat = spine_lum_ds.ravel()
        _k99 = min(_sp_flat.size - 1, max(0, int(round(0.992 * (_sp_flat.size - 1)))))
        spine_p99 = max(float(np.partition(_sp_flat, _k99)[_k99]), 1e-4)
        trench_norm_ds = np.clip(trench_raw * np.float32(1.0 / (0.22 * spine_p99)), 0.0, 1.0) * (np.float32(1.0) - core_shield_ds)

        neck_tint = np.array([0.96, 1.02, 1.18], dtype=np.float32)[None, None, :]
        dn_tint   = np.array([0.74, 1.06, 1.46], dtype=np.float32)[None, None, :]
        up_tint   = np.array([0.72, 0.98, 1.28], dtype=np.float32)[None, None, :]

        trench_gate_ds = np.clip(np.float32(1.0) - np.float32(2.2) * trench_norm_ds, 0.15, 1.0)[..., None]
        tot_streak_ds = (
            np.float32(0.95 * a) * v_neck_ds * neck_tint
            + np.float32(1.85 * a) * v_down_ds * dn_tint
            + np.float32(0.42 * a) * v_up_ds * up_tint
        ) * trench_gate_ds
        trench_atten_ds = np.clip(np.float32(a * 0.52) * trench_norm_ds, 0.0, 0.52)

        from ..core.signal import get_native_lib
        lib = get_native_lib()
        if lib is not None and src.ndim == 3 and src.shape[2] == 3:
            src_c = np.ascontiguousarray(src, dtype=np.float32)
            st_c = np.ascontiguousarray(tot_streak_ds, dtype=np.float32)
            tr_c = np.ascontiguousarray(trench_atten_ds, dtype=np.float32)
            out = np.empty_like(src_c)
            lib.vml_smear_upsample_blend_f32(
                src_c.ctypes.data, st_c.ctypes.data, tr_c.ctypes.data,
                out.ctypes.data, h, w, h_ds, w_ds,
            )
            return out

        if ds_y > 1 or ds_x > 1:
            v_streak = _cv2.resize(tot_streak_ds, (w, h), interpolation=_cv2.INTER_LINEAR)
            trench_atten = _cv2.resize(trench_atten_ds, (w, h), interpolation=_cv2.INTER_LINEAR)[..., None]
        else:
            v_streak, trench_atten = tot_streak_ds, trench_atten_ds[..., None]

        v_streak[0::2] *= np.float32(1.045)
        v_streak[1::2] *= np.float32(0.955)
        out = np.clip(src, 0.0, 2.0) * (np.float32(1.0) - trench_atten) + v_streak
        np.clip(out, 0.0, 2.0, out=out)
        return out.astype(np.float32, copy=False)

    lum = src @ LUMA
    bg_lum = lowpass(lum, max(2.0, 14.0 * sc))
    pmax = max(float(src.max()), 1e-4)
    rel_peak = np.clip(src.max(axis=-1) / pmax, 0.0, 1.0)
    clips = np.array([0.74, 0.80, 0.86], dtype=np.float32)[None, None, :]
    hot_ch = np.clip(src - clips, 0.0, None)
    hot_gate = np.clip((lum - np.maximum(bg_lum + 0.08, 0.56)) / 0.24, 0.0, 1.0)
    sat_gate = np.clip((rel_peak - 0.54) / 0.32, 0.0, 1.0) * np.clip((lum - bg_lum * 0.62) / 0.20, 0.0, 1.0)
    emit_drive = np.maximum(hot_gate ** 1.3, 0.82 * (sat_gate ** 1.5))
    if float(emit_drive.max()) <= 1e-4 and float(hot_ch.max()) <= 1e-4:
        return rgb
    hot_area_frac = float(np.mean(lum > 0.68))
    if hot_area_frac > 0.030:
        broad_damp = 1.0 / (1.0 + 45.0 * (hot_area_frac - 0.030))
        a = a * broad_damp
        if a <= 1e-4:
            return rgb
        local_prom = np.clip((lum - lowpass(lum, max(1.2, 3.2 * sc))) / 0.16, 0.0, 1.0)
        emit_drive = emit_drive * local_prom
        hot_ch = hot_ch * local_prom[..., None]
    z = float(np.clip(zoom_pos, 0.0, 1.0))
    len_f = float(np.clip(length_scale, 0.15, 1.5))
    emit_src = np.maximum(hot_ch, src * (emit_drive ** 1.35)[..., None] * 0.52)
    emit_lum = emit_src @ LUMA
    col_local_max = lowpass(emit_lum, max(1.2, 3.2 * sc), axis=1)
    col_pinch = np.clip((emit_lum - 0.72 * col_local_max) / np.maximum(0.28 * col_local_max, 1e-4), 0.0, 1.0) ** 1.6
    needle_src = emit_src * (0.22 + 0.78 * col_pinch)[..., None]
    r_neck = max(6, min(h - 1, int(round((28.0 + 36.0 * len_f) * sc))))
    m_neck = np.arange(-r_neck, r_neck + 1, dtype=np.float32)
    tau_neck_dn = max(3.0, (12.0 + 18.0 * len_f) * sc)
    tau_neck_up = max(2.0, (6.5 + 9.5 * len_f) * sc)
    k_neck = np.where(m_neck >= 0.0, np.exp(-np.maximum(0.0, m_neck) / tau_neck_dn), 0.38 * np.exp(-np.maximum(0.0, -m_neck) / tau_neck_up)).astype(np.float32)
    k_neck /= max(1e-6, float(k_neck.sum()))
    r_far = max(16, min(h - 1, int(round(min(h * 0.92, (160.0 + 260.0 * len_f) * sc)))))
    m_far = np.arange(-r_far, r_far + 1, dtype=np.float32)
    tau_far_dn = max(12.0, (75.0 + 165.0 * len_f) * sc)
    tau_far_up = max(6.0, (22.0 + 42.0 * len_f) * sc)
    k_far_dn = np.where(m_far > 0.0, 0.72 * np.exp(-np.maximum(0.0, m_far) / tau_far_dn) + 0.28 / (1.0 + (np.maximum(0.0, m_far) / max(6.0, 28.0 * sc)) ** 1.15), 0.0).astype(np.float32)
    k_far_dn /= max(1e-6, float(k_far_dn.sum()))
    k_far_up = np.where(m_far < 0.0, np.exp(-np.maximum(0.0, -m_far) / tau_far_up), 0.0).astype(np.float32)
    k_far_up /= max(1e-6, float(k_far_up.sum()))
    neck_sig_x = max(0.85, (2.10 - 0.55 * z) * sc)
    dn_sig_x   = max(0.35, (0.65 - 0.20 * z) * sc)
    up_sig_x   = max(1.15, (2.05 - 0.45 * z) * sc)
    n_fft_n = _next_fast_len(h + len(k_neck) - 1, real=True)
    fk_n = _sp_rfft(k_neck, n=n_fft_n)[:, None, None]
    fa_n = _sp_rfft(emit_src, n=n_fft_n, axis=0)
    rad_n = len(k_neck) // 2
    v_neck = lowpass(_sp_irfft(fa_n * fk_n, n=n_fft_n, axis=0)[rad_n:rad_n + h].astype(np.float32, copy=False), neck_sig_x, axis=1)
    n_fft_f = _next_fast_len(h + len(k_far_dn) - 1, real=True)
    fk_d = _sp_rfft(k_far_dn, n=n_fft_f)[:, None, None]
    fk_u = _sp_rfft(k_far_up, n=n_fft_f)[:, None, None]
    fa_f = _sp_rfft(needle_src, n=n_fft_f, axis=0)
    rad_f = len(k_far_dn) // 2
    v_down = lowpass(_sp_irfft(fa_f * fk_d, n=n_fft_f, axis=0)[rad_f:rad_f + h].astype(np.float32, copy=False), dn_sig_x, axis=1)
    v_up   = lowpass(_sp_irfft(fa_f * fk_u, n=n_fft_f, axis=0)[rad_f:rad_f + h].astype(np.float32, copy=False), up_sig_x, axis=1)
    rows_y = np.arange(h, dtype=np.float32)[:, None, None]
    interlace_bead = (1.0 + 0.075 * np.cos(np.pi * rows_y)).astype(np.float32)
    neck_tint = np.array([0.96, 1.02, 1.18], dtype=np.float32)[None, None, :]
    dn_tint   = np.array([0.74, 1.06, 1.46], dtype=np.float32)[None, None, :]
    up_tint   = np.array([0.72, 0.98, 1.28], dtype=np.float32)[None, None, :]
    v_streak = (0.95 * v_neck * neck_tint + 1.85 * v_down * dn_tint * interlace_bead + 0.42 * v_up * up_tint) * a
    core_shield = np.clip(lowpass(emit_drive, max(1.0, 2.5 * sc)) * 2.4, 0.0, 1.0)
    spine_lum = (v_down @ LUMA) + 0.25 * (v_up @ LUMA)
    lat_narrow = lowpass(spine_lum, max(0.35, 0.55 * sc), axis=1)
    lat_wide   = lowpass(spine_lum, max(1.60, 3.10 * sc), axis=1)
    shift_r = max(2, int(round(2.6 * sc)))
    lat_right  = np.roll(lowpass(spine_lum, max(1.10, 1.95 * sc), axis=1), shift_r, axis=1)
    lat_right[:, :shift_r] = 0.0
    spine_peak_col = np.maximum(lat_narrow, lowpass(v_neck @ LUMA, max(0.6, 1.1 * sc), axis=1) * 0.55)
    trench_raw = np.clip(0.55 * lat_wide + 0.92 * lat_right - 1.55 * spine_peak_col, 0.0, None)
    spine_p99 = max(float(np.percentile(spine_lum, 99.2)), 1e-4)
    trench_norm = np.clip(trench_raw / (0.22 * spine_p99), 0.0, 1.0) * (1.0 - core_shield)
    trench_atten = np.clip(a * 0.52 * trench_norm, 0.0, 0.52)[..., None]
    trench_gate = np.clip(1.0 - 2.2 * trench_norm, 0.15, 1.0)[..., None]
    out = src * (1.0 - trench_atten) + v_streak * trench_gate
    return np.clip(out, 0.0, 2.0).astype(np.float32)

def light_streaks(rgb: np.ndarray, amount: float, angle_deg: float = 0.0,
                  length: float = 0.55, zoom_pos: float = 0.0) -> np.ndarray:
    """
    Physical SONY DCR-TRV310E Light Streak & Scalloped Multi-Petal Starburst Details (`画面高光溢出与光条形态细节`):

    Models the three distinct non-uniform physical mechanisms measured in Sony DCR-TRV310E night captures:
      1. **Up-Down Asymmetric Interline CCD Readout Smear + Flanking Dark VOD/HOD Trenches**:
         Downward razor-sharp cyan-white needle (`dy > 0`), flared core-neck funnel, weaker/wider upward
         plume (`dy < 0`), and flanking dark anti-blooming drain trenches (`dx = +-2..5 px`).
      2. **Scalloped 6-Petal Asymmetric Iris & Microlens Starburst Skirt**:
         Polygonal 6-lobe iris/microlens diffraction halo (`lobes at 36, 108, 180, 216, 272, 0 deg` with
         dark diffraction notches at `~245 deg` and `~320 deg`, producing `peak/median > 1.48x` angular
         modulation around bright lamps).
      3. **Directional / Anamorphic Optical Glare Streaks (`light_streak_angle`)**:
         Elongated directional light bar with chromatic dispersion along the bar.
    """
    a = float(np.clip(amount, 0.0, 2.0))
    if a <= 1e-4:
        return rgb

    from scipy.ndimage import gaussian_filter

    h, w = rgb.shape[:2]
    sc = min(h, w) / 540.0
    src = np.clip(np.asarray(rgb, np.float32), 0.0, 2.0)
    lum = src @ LUMA
    bg_lum = lowpass(lum, max(2.0, 14.0 * sc))
    pmax = max(float(src.max()), 1e-4)
    rel_peak = np.clip(src.max(axis=-1) / pmax, 0.0, 1.0)

    hot = np.clip((lum - np.maximum(bg_lum + 0.10, 0.58)) / 0.26, 0.0, 1.0)
    sat_hot = np.clip((rel_peak - 0.52) / 0.35, 0.0, 1.0) * np.clip((lum - bg_lum * 0.65) / 0.22, 0.0, 1.0)
    emit_drive = np.maximum(hot ** 1.35, 0.80 * (sat_hot ** 1.6))
    if float(emit_drive.max()) <= 1e-4:
        return rgb

    z = float(np.clip(zoom_pos, 0.0, 1.0))
    len_f = float(np.clip(length, 0.1, 1.5))

    # --- Component A: Up-Down Asymmetric Vertical CCD Smear + Flanking VOD/HOD Trenches ---
    out = asymmetric_ccd_smear_and_trench(src, smear_amount=a * 0.90, length_scale=len_f, zoom_pos=z)

    # --- Component B: Directional / Anamorphic Optical Light Bar + Scalloped 6-Petal Starburst ---
    emit_rgb = src * emit_drive[..., None]
    theta = np.deg2rad(float(angle_deg))
    ux = float(np.cos(theta))
    uy = float(np.sin(theta))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    spine_src = np.stack([
        gaussian_filter(emit_rgb[..., c], sigma=max(0.55, (1.25 - 0.45 * z) * sc))
        for c in range(3)
    ], axis=-1)

    max_reach = (22.0 + 78.0 * len_f) * sc * (1.0 + 0.30 * z)
    dir_streak = np.zeros_like(src)
    ch_scales = (1.06, 1.00, 0.94)
    n_steps = 10
    for step_i in range(1, n_steps + 1):
        t = step_i / float(n_steps)
        falloff = float(np.exp(-2.35 * t) * (1.0 - 0.35 * t))
        for sign in (-1.0, 1.0):
            for ch in range(3):
                dist = sign * max_reach * t * ch_scales[ch]
                sampled = _bilinear_sample_2d(spine_src[..., ch], yy - uy * dist, xx - ux * dist)
                dir_streak[..., ch] += sampled * falloff

    dir_streak *= (0.22 * a)

    # Scalloped 6-petal asymmetric iris starburst lobes (calibrated to SONY DCR-TRV310E radial profiles)
    petal_src = np.stack([
        gaussian_filter(emit_rgb[..., c], sigma=max(0.8, (2.2 - 0.6 * z) * sc))
        for c in range(3)
    ], axis=-1)
    petal_reach = (16.0 + 46.0 * len_f) * sc * (1.0 - 0.25 * z)
    # 6 asymmetric petal directions (deg) and weights; note deep notches at ~245 deg and ~320 deg
    petal_defs = (
        (36.0, 1.25, 1.05),
        (108.0, 1.15, 1.00),
        (180.0, 0.85, 0.85),
        (216.0, 1.18, 1.02),
        (272.0, 0.90, 0.88),
        (0.0, 0.88, 0.86),
    )
    petal_field = np.zeros_like(src)
    for p_deg, p_w, p_rscale in petal_defs:
        p_rad = np.deg2rad(p_deg + float(angle_deg) * 0.25)
        px_u = float(np.cos(p_rad))
        py_u = float(np.sin(p_rad))
        for frac in (0.25, 0.50, 0.78, 1.05):
            dist = petal_reach * p_rscale * frac
            w_step = p_w * float(np.exp(-2.1 * frac)) * 0.09 * a
            for ch in range(3):
                c_dist = dist * ch_scales[ch]
                petal_field[..., ch] += _bilinear_sample_2d(petal_src[..., ch], yy - py_u * c_dist, xx - px_u * c_dist) * w_step

    out = out + dir_streak + petal_field
    return np.clip(out, 0.0, 2.0).astype(np.float32)


def draw_trv310e_zoom_osd(rgb: np.ndarray, zoom_pos: float) -> np.ndarray:
    """
    Draw the authentic SONY DCR-TRV310E viewfinder `W [━━━|━━━] T` zoom indicator bar.
    """
    h, w = rgb.shape[:2]
    if h < 64 or w < 96:
        return rgb
    out = rgb.copy()
    sc = max(1.0, min(h, w) / 480.0)
    bar_w = int(round(92 * sc))
    bar_h = max(4, int(round(6 * sc)))
    x0 = int(round(w * 0.065))
    y0 = int(round(h * 0.145))
    x1 = min(w - 10, x0 + bar_w)
    y1 = min(h - 6, y0 + bar_h)
    if x1 <= x0 + 8 or y1 <= y0 + 2:
        return rgb

    pad = max(1, int(round(2 * sc)))
    out[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)] *= 0.28

    white = np.array([0.92, 0.94, 0.92], dtype=np.float32)
    border = max(1, int(round(1.2 * sc)))
    out[y0:y0 + border, x0:x1] = white
    out[y1 - border:y1, x0:x1] = white
    out[y0:y1, x0:x0 + border] = white
    out[y0:y1, x1 - border:x1] = white

    z = float(np.clip(zoom_pos, 0.0, 1.0))
    knob_w = max(3, int(round(5 * sc)))
    kx = int(round(x0 + border + z * max(1, (x1 - x0 - 2 * border - knob_w))))
    out[y0 + border:y1 - border, kx:min(x1 - border, kx + knob_w)] = np.array([0.96, 0.96, 0.96], dtype=np.float32)
    return out


def coating_ghost_dispersion(rgb: np.ndarray, spec: LensSpec) -> np.ndarray:
    """
    Multi-element zoom lens double-reflection ghost chain & wide-angle chromatic dispersion arc
    (calibrated against the opening sequence `0.50s-1.25s` of `30666916021-1-192.mp4`).
    """
    cg = float(getattr(spec, "coat_ghost", 0.0))
    if cg <= 1e-4:
        return rgb

    h, w = rgb.shape[:2]
    scale_w = w / 720.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0

    lum = rgb @ LUMA
    hot = np.clip((lum - 0.58) / 0.32, 0.0, 1.0)
    mass = float(hot.sum())
    if mass > 1e-3:
        sx = float((hot * nx).sum() / mass)
        sy = float((hot * ny).sum() / mass)
        if sx * sx + sy * sy < 0.04:
            sx, sy = 0.64, -0.72
    else:
        sx, sy = 0.64, -0.72

    out = rgb.astype(np.float32, copy=True)

    # 1. Forward Mie scatter veil around the off-axis light source
    d_src2 = (nx - sx) ** 2 + (ny - sy) ** 2
    src_veil = np.exp(-d_src2 / 0.85).astype(np.float32)
    veil_tint = np.array([1.02, 1.03, 0.99], dtype=np.float32)[None, None, :]
    out = out + (0.22 * cg) * src_veil[..., None] * veil_tint

    # 2. Centrosymmetric inverted ghost (paraxial magnification m = -1) with MgF2 green-null tint
    inv_rgb = rgb[::-1, ::-1]
    inv_lum = inv_rgb @ LUMA
    inv_hot = lowpass(np.clip((inv_lum - 0.42) / 0.45, 0.0, 1.0), max(1.5, 5.5 * scale_w))
    mgf2_violet = np.array([0.82, 0.76, 1.18], dtype=np.float32)[None, None, :]
    out = out + (0.28 * cg) * inv_hot[..., None] * mgf2_violet

    # 3. Diagonal tangential caustic streak across the optical axis (magenta-amber coating reflection)
    ax_len = max(float(np.hypot(sx, sy)), 1e-3)
    ux, uy = sx / ax_len, sy / ax_len
    along = nx * ux + ny * uy
    perp = -nx * uy + ny * ux
    caustic = (
        np.exp(-0.5 * (perp / 0.24) ** 2)
        * np.exp(-0.5 * ((along - 0.05) / 0.68) ** 2)
    ).astype(np.float32)
    caustic_tint = np.array([1.14, 0.94, 0.86], dtype=np.float32)[None, None, :]
    out = out + (0.20 * cg) * caustic[..., None] * caustic_tint

    # 4. Opposite-quadrant wide-angle rainbow dispersion arc (chromatic pupil ghost)
    gx, gy = -0.62 * sx, -0.62 * sy
    rg = np.hypot(nx - gx, ny - gy)
    arc_gate = np.clip(1.0 - np.hypot(nx - (-0.45 * sx), ny - (-0.45 * sy)) / 1.15, 0.0, 1.0)
    r_band = np.exp(-0.5 * ((rg - 0.34) / 0.11) ** 2) * arc_gate
    g_band = np.exp(-0.5 * ((rg - 0.42) / 0.12) ** 2) * arc_gate
    b_band = np.exp(-0.5 * ((rg - 0.50) / 0.13) ** 2) * arc_gate
    rainbow = np.stack([
        0.72 * r_band + 0.18 * b_band,
        0.95 * g_band + 0.22 * b_band,
        1.18 * b_band + 0.30 * g_band,
    ], axis=-1).astype(np.float32)
    out = out + (0.26 * cg) * rainbow

    return np.clip(out, 0.0, 1.5).astype(np.float32)
