"""
Physical CRT display optics, electron-beam dynamics, RGB phosphor geometry, and video-amplifier faults.

A real cathode-ray tube is an electro-optical system rather than a static 1-D cosine overlay:

1. **Dynamic Electron Beam Spot Blooming (`spot_bloom`)**:
   As cathode current $I_k$ rises in bright regions, space-charge repulsion inside the electron gun
   widens the beam spot $\\sigma_{\\text{eff}}(y, x) = \\sigma_0 \\cdot (1 + k_{\\text{bloom}} \\cdot Y(y, x)^p)$.
   Dark and mid-tone lines stay narrow with crisp black inter-line gaps, while intense highlights expand
   vertically and horizontally, bridging the scanline gaps just as a real CRT does.

2. **True RGB Phosphor Subpixel Triads, Black Matrix & Trinitron Damper Wires (`phosphor_rgb`, `mask_type`)**:
   Instead of multiplying all three channels by an identical scalar shadow-mask greyscale pattern,
   the three electron beams excite spatially distinct R, G, and B phosphor stripes (Aperture Grille,
   `mask_type=0`), staggered brick slots (Slot Mask, `mask_type=1`), or hexagonal delta triads
   (Dot Mask, `mask_type=2`), separated by a carbon Black Matrix guard band. Aperture Grille tubes
   also cast the signature faint horizontal shadow of one or two tungsten tension damper wires.

3. **Leaded Faceplate Glass Internal Halation (`glass_halation`) & P22 Phosphor Saturation (`phosphor_saturation`)**:
   Light emitted by the phosphor layer undergoes total internal reflection at the front glass-air
   interface of the thick (10-14 mm) leaded faceplate, forming a wide secondary halation ring/veil
   around bright features on dark backgrounds, while P22 phosphors saturate non-linearly at high drive.

4. **Deflection Yoke Corner Astigmatism (`corner_defocus`), Line-Load EHT Sag (`ehv_sag`), & Bezel (`bezel`)**:
   Oblique beam landing at the screen corners introduces radial astigmatism; bright horizontal scanlines
   load the flyback high-voltage supply and sag outward; and a curved physical bezel frames the raster.

5. **Moisture & Corrosion in the Video Amplifier / Deflection Supply (`moisture_effects`, `apply_deflection`)**:
   Preserves the diagnostic amplifier bias drift, per-cathode aging cast, and sagging deflection raster scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.signal import horizontal_shift, lowpass

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass
class CRTMoisture:
    """What the damp amplifier does to the picture, reported so the caller can log it."""

    bias_drift: float
    cast: tuple[float, float, float]
    deflection: float


def moisture_effects(moisture: float, corrosion: float, age: float = 0.3) -> CRTMoisture:
    """
    Translate the user's levels into the three physical quantities the stage applies.

    Kept pure so the relationship can be checked directly without rendering a frame.
    """
    m = float(np.clip(moisture, 0.0, 1.0))
    c = float(np.clip(corrosion, 0.0, 1.0))
    a = float(np.clip(age, 0.0, 1.0))

    # the bias shifts with both moisture and corrosion: a leaky path to ground moves the
    # black level up, which is why a damp set washes the picture out rather than darkening it
    bias = 0.18 * m + 0.12 * c

    # the three cathode amplifiers do not age alike: green's tube/transistor typically runs
    # hottest and drifts furthest, red next, blue least.  Age scales the *spread* between
    # them, so a new set has almost no cast even when damp and an old one has a strong one.
    severity = 0.65 * m + 0.55 * c
    spread = severity * (0.08 + 0.52 * a)
    cast = (
        float(1.0 - 0.55 * spread),   # red falls back
        float(1.0 + 0.45 * spread),   # green comes up -> the familiar greenish-yellow cast
        float(1.0 - 0.80 * spread),   # blue falls furthest
    )

    # the deflection supply sags with the same leakage
    defl = float(np.clip(0.55 * m + 0.65 * c, 0.0, 1.0))
    return CRTMoisture(bias_drift=bias, cast=cast, deflection=defl)


def apply_deflection(rgb: np.ndarray, deflection: float, frame_index: int,
                     seed: int = 0) -> tuple[np.ndarray, dict]:
    """
    Shrink and wobble the raster, because the deflection supply is sagging.

    This is the diagnostic part of the fault: nothing upstream of the CRT can move the
    *edges of the raster*, because a tape cannot bend the picture. When the supply sags the
    scan does not reach the bezel, so a narrow dark border appears and its width breathes
    with the supply ripple.
    """
    d = float(np.clip(deflection, 0.0, 1.0))
    if d <= 1e-4:
        return rgb, {}

    h, w = rgb.shape[:2]
    # the raster shrinks by a few percent, and breathes slowly with the supply's ripple
    breath = 0.018 * d * float(np.sin(frame_index * 0.37 + seed * 0.1))
    scale = float(np.clip(1.0 - 0.065 * d + breath, 0.86, 1.0))

    yy = (np.arange(h, dtype=np.float32) - h * 0.5) / scale + h * 0.5
    xx = (np.arange(w, dtype=np.float32) - w * 0.5) / scale + w * 0.5
    # and a slight horizontal bow, because the horizontal output stage is loaded asymmetrically
    bow = (d * 3.5) * np.sin(np.linspace(0.0, np.pi, h, dtype=np.float32))

    y_idx = np.clip(np.round(yy).astype(np.int32), 0, h - 1)[:, None]
    x_2d = xx[None, :] + bow[:, None]
    x_idx = np.clip(np.round(x_2d).astype(np.int32), 0, w - 1)
    inside = ((yy >= 0.0) & (yy < h))[:, None] & (x_2d >= 0.0) & (x_2d < w)
    out = np.where(inside[..., None], rgb[y_idx, x_idx], np.float32(0.0))

    return out, {"raster_scale": round(scale, 4)}


# ===========================================================================
# Physical CRT Beam, Phosphor Triad, Halation & Deflection Optics
# ===========================================================================

def build_phosphor_mask(h: int, w: int, mask_pitch: float, mask_depth: float,
                        mask_type: int = 0, phosphor_rgb: float = 0.65
                        ) -> np.ndarray | None:
    """
    Construct a 3-channel `(H, W, 3)` RGB phosphor subpixel & Black Matrix transmittance map.

    * `mask_type == 0` (**Aperture Grille / Trinitron**):
      Continuous vertical R, G, B phosphor stripes separated by dark Black Matrix guard slots,
      plus faint horizontal tungsten tension damper wire shadows at ~30% and ~70% vertical height.
    * `mask_type == 1` (**Slot Mask / Staggered In-Line TV**):
      Vertical R, G, B phosphor slots broken by horizontal shadow-mask bridges every `slot_h` rows,
      with alternate triad columns staggered vertically by half a slot period (brick lattice).
    * `mask_type == 2` (**Dot Triad / Delta Gun Shadow Mask**):
      Hexagonal staggered circular phosphor dot triads where R, G, B dots sit at 120-degree phase
      offsets and alternate row bands shift by half a triad pitch.

    Each channel plane is normalized to unit mean (`mask[..., c].mean() == 1.0`) so engaging the
    mask or changing `phosphor_rgb` never alters overall brightness or neutral white balance.
    """
    if mask_pitch <= 1e-3 or mask_depth <= 1e-4:
        return None

    pitch = max(1.5, float(mask_pitch))
    depth = float(np.clip(mask_depth, 0.0, 1.0))
    rgb_sep = float(np.clip(phosphor_rgb, 0.0, 1.0))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Scalar luminance shadow-mask envelope (preserves exact compatibility when phosphor_rgb == 0)
    phase = (xx / pitch) % 1.0
    if mask_type == 1:
        band = np.floor(yy / max(1.0, pitch)) % 2.0
        shift = band * 0.5
        m_scalar = 0.5 + 0.5 * np.cos(2.0 * np.pi * (((xx / pitch) + shift) % 1.0))
        # Horizontal slot bridge bars across each triad cell
        slot_h = max(2.0, pitch * 1.15)
        bridge = 1.0 - 0.38 * depth * (np.sin(np.pi * (yy / slot_h)) ** 8)
        m_scalar = m_scalar * bridge
    elif mask_type == 2:
        cy = np.floor(yy / max(1.0, pitch))
        shift = (cy % 2.0) * 0.5
        dx = ((xx / pitch) + shift) % 1.0 - 0.5
        dy = (yy / pitch) % 1.0 - 0.5
        r2 = dx * dx + dy * dy
        m_scalar = np.exp(-r2 / 0.08)
    else:
        vert = np.cos(2.0 * np.pi * phase)
        m_scalar = 0.5 + 0.5 * vert

    scalar_mask = 1.0 - depth * (1.0 - m_scalar)
    scalar_mask = (scalar_mask / max(1e-6, float(scalar_mask.mean()))).astype(np.float32)

    if rgb_sep <= 1e-4:
        return np.repeat(scalar_mask[..., None], 3, axis=2)

    # Subpixel R, G, B phosphor triad profiles at phases (0, 1/3, 2/3) of the triad pitch.
    # A triad spans `triad_w = max(2.4, pitch * 1.5)` pixels so that even at pitch=2.0..3.0 px
    # the three R/G/B subpixels are well-resolved without severe Moire aliasing.
    triad_w = max(2.2, pitch * 1.35)
    if mask_type == 1:
        slot_h = max(2.0, pitch * 1.25)
        col_idx = np.floor(xx / triad_w)
        y_stag = yy + (col_idx % 2.0) * (slot_h * 0.5)
        slot_env = 1.0 - (0.45 * depth) * (np.cos(np.pi * (y_stag / slot_h)) ** 6)
        x_phase = (xx / triad_w) % 1.0
    elif mask_type == 2:
        dot_h = max(1.8, pitch * 1.05)
        row_band = np.floor(yy / dot_h)
        x_phase = ((xx / triad_w) + (row_band % 2.0) * 0.5) % 1.0
        dy_dot = ((yy / dot_h) % 1.0) - 0.5
        slot_env = np.exp(-(dy_dot * dy_dot) / max(0.06, 0.22 - 0.10 * depth))
    else:
        x_phase = (xx / triad_w) % 1.0
        slot_env = np.ones_like(xx, dtype=np.float32)
        # Trinitron aperture grille horizontal tungsten damper wire shadow (1 or 2 wires)
        if h >= 64:
            for wy_frac in (0.31, 0.71):
                wy = wy_frac * h
                wire_prof = np.exp(-0.5 * ((yy - wy) / 0.65) ** 2)
                slot_env = slot_env * (1.0 - 0.14 * depth * wire_prof)

    # Black matrix guard bands between phosphor stripes + subpixel selectivity
    channels = []
    centers = (0.167, 0.500, 0.833)
    for c_phase in centers:
        # Wrapped distance on [0, 1) circle
        d_phi = np.abs(((x_phase - c_phase + 0.5) % 1.0) - 0.5)
        # Gaussian phosphor stripe + plateau with cross-excitation floor so single-channel colors stay rich
        sub_profile = np.exp(-0.5 * (d_phi / 0.19) ** 2)
        ch_mask = (1.0 - depth * 0.88) + (depth * 0.88) * sub_profile * slot_env
        ch_mask = ch_mask / max(1e-6, float(ch_mask.mean()))
        channels.append(ch_mask.astype(np.float32))

    rgb_mask = np.stack(channels, axis=-1)
    # Blend smoothly between scalar mask and full RGB subpixel triad mask
    out_mask = scalar_mask[..., None] * (1.0 - rgb_sep) + rgb_mask * rgb_sep
    for ch in range(3):
        out_mask[..., ch] /= max(1e-6, float(out_mask[..., ch].mean()))
    return out_mask.astype(np.float32)


def apply_scanline_beam(rgb: np.ndarray, base_row_mask: np.ndarray,
                        scanlines: float, spot_sigma: float, scan_pitch: float,
                        spot_bloom: float = 0.0, field_parity: int = 0,
                        interlace: float = 0.0) -> np.ndarray:
    """
    Apply the electron-beam scanline profile with dynamic cathode-current spot blooming (`spot_bloom`).

    * When `spot_bloom <= 0` (or on a uniform flat field), this reduces to multiplying by `base_row_mask`,
      guaranteeing zero light loss (`spread < 0.002`) and monotone focus control (`spot_sigma`).
    * When `spot_bloom > 0`, bright highlights drive high beam current:
      1. The vertical Gaussian spot widens in proportion to local luminance $Y(y, x)$, filling in the
         dark inter-line gaps inside bright objects while keeping dark/midtone scanlines crisp.
      2. Bright scanline cores spill vertically into adjacent scanline troughs and horizontally along
         the active line, reproducing authentic CRT highlight line-bridging.
    """
    if scanlines <= 1e-4:
        return rgb * base_row_mask

    h, w = rgb.shape[:2]
    bloom_k = float(np.clip(spot_bloom, 0.0, 2.0))
    if bloom_k <= 1e-4 and interlace <= 1e-4:
        return (rgb * base_row_mask).astype(np.float32)

    scale_h = (h / 540.0) if h > 580 else 1.0
    pitch = max(1.05, float(scan_pitch) * scale_h)
    sigma0 = max(1e-3, float(spot_sigma) * scale_h)

    # Interlace field phase shift: odd/even fields shift the beam centres slightly
    field_shift = (0.18 * float(np.clip(interlace, 0.0, 1.0)) * pitch
                   * (0.5 if (field_parity % 2 == 1) else -0.5))
    rows = np.arange(h, dtype=np.float32)[:, None] - field_shift
    centres = np.round(rows / pitch) * pitch
    dy = rows - centres  # (H, 1) distance to nearest scanline center

    # Base 1D beam profile (normalized to unit mean, identical to DisplayStage._masks)
    beam0 = np.exp(-0.5 * (dy / sigma0) ** 2)
    mask0 = 1.0 - scanlines * (1.0 - beam0)
    mask0_norm = mask0 / max(1e-6, float(mask0.mean()))

    if bloom_k <= 1e-4:
        # Divide the *base* beam profile out of `base_row_mask` and put the shifted one back in, so the
        # mask is applied exactly once.  It used to be
        #
        #     vig_ratio = base_row_mask / base_row_mask.mean(axis=0, keepdims=True)
        #     return rgb * (mask0_norm[..., None] * vig_ratio)
        #
        # and normalising per *column* leaves the beam structure in: a row mask is column-constant, so
        # `base_row_mask` divided by its own column mean is that same mask again, and the frame received
        # mask squared.  Measured on a flat field (the only input on which the mask's own gain is visible
        # at all): out/input 1.38316 at a spot of 0.20, 1.10204 at 0.42, 1.00814 at 0.90, 1.00202 at
        # 1.30 -- a **focus control that brightens the picture**, which is what the check
        # "the scanline structure costs no light" was measuring as a 0.05210 mean drift on real content.
        rows0 = np.arange(h, dtype=np.float32)[:, None]
        centres0 = np.round(rows0 / pitch) * pitch
        beam_base = np.exp(-0.5 * ((rows0 - centres0) / sigma0) ** 2)
        mask_base = 1.0 - scanlines * (1.0 - beam_base)
        mask_base = mask_base / max(1e-6, float(mask_base.mean()))
        vig_ratio = base_row_mask / np.maximum(1e-6, mask_base[..., None])
        return (rgb * (mask0_norm[..., None] * vig_ratio)).astype(np.float32)

    # Local luminance (smoothed slightly along the scanline and across 1 row so the beam responds
    # to the local cathode drive across the scanline gap)
    lum = np.clip(rgb @ LUMA, 0.0, 1.5)
    lum_env = lowpass(lowpass(lum, 1.2, axis=1), 0.85, axis=0)

    # Space-charge spot widening: only significant above mid-grey so dark/midtone scanlines stay sharp
    drive = np.clip((lum_env - 0.35) / 0.65, 0.0, 1.5) ** 1.35
    sigma_eff = sigma0 * (1.0 + 1.35 * bloom_k * drive)
    beam_dyn = np.exp(-0.5 * (dy / np.maximum(1e-3, sigma_eff)) ** 2)
    mask_dyn = 1.0 - scanlines * (1.0 - beam_dyn)

    # Normalize mask_dyn per-intensity-bin against mask0's mean so flat fields keep exact mean=1.0
    # while bright features bridge the dark gaps between scanlines
    bridge_weight = np.clip(drive * bloom_k * 0.85, 0.0, 0.92)
    eff_scan_mask = mask0_norm * (1.0 - bridge_weight) + (mask_dyn / max(1e-6, float(mask0.mean()))) * bridge_weight

    # Extract vignette component from base_row_mask (by dividing out the 1D row mean of base_row_mask)
    row_mean_base = np.maximum(1e-6, base_row_mask.mean(axis=1, keepdims=True))
    vig_2d = base_row_mask / row_mean_base

    out = rgb * (eff_scan_mask[..., None] * vig_2d)

    # Vertical/horizontal highlight spot spill into adjacent scanline gaps on intense highlights
    hot_spill = np.clip(rgb - 0.68, 0.0, None)
    if float(hot_spill.max()) > 1e-4:
        spill_v = lowpass(lowpass(hot_spill, 0.9 * scale_h, axis=0), 1.4 * scale_h, axis=1)
        gap_openness = np.clip(1.0 - beam0, 0.0, 1.0)[..., None]
        out = out + (0.38 * bloom_k * scanlines) * spill_v * gap_openness

    target_mean = float((rgb * base_row_mask).mean())
    cur_mean = float(out.mean())
    if cur_mean > 1e-6:
        out = out * (target_mean / cur_mean)
    return out.astype(np.float32)


def apply_faceplate_halation(rgb: np.ndarray, glass_halation: float,
                             bloom: float, scale_w: float = 1.0) -> np.ndarray:
    """
    Two-stage CRT light scatter:
      1. **Electron-beam & phosphor tight bloom (`bloom`)**: Energy-proportional halo around bright
         highlights (never normalized by `1 / peak`, which previously turned a single dim pixel at
         0.73 into a full-strength glow).
      2. **Double-layer leaded glass faceplate internal halation (`glass_halation`)**:
         Light emitted by the phosphor reflects off the front glass-air interface (beyond the critical
         angle of leaded glass) and scatters back onto the phosphor screen at a wide radius, creating
         the characteristic soft CRT halation ring/veil with a subtle warm-amber/cyan glass tint.
    """
    out = rgb.astype(np.float32, copy=True)

    if bloom > 1e-3:
        # Energy-proportional highlight extraction (smooth knee above 0.65)
        hot_rgb = np.clip(out - 0.62, 0.0, None)
        if float(hot_rgb.max()) > 1e-5:
            tight = lowpass(hot_rgb, max(0.8, (2.2 + 4.5 * bloom) * scale_w))
            mid = lowpass(hot_rgb, max(1.8, (6.0 + 11.0 * bloom) * scale_w))
            # Soft peak adaptation so genuinely bright highlights glow strongly while barely-above-threshold
            # regions glow proportionally
            _flat_h = hot_rgb.ravel(); _k995 = int(0.995 * (_flat_h.size - 1)); ref_scale = 1.0 / max(0.28, float(np.partition(_flat_h, _k995)[_k995]) + 0.12)
            ref_scale = float(np.clip(ref_scale, 0.8, 3.2))
            out = out + (bloom * 0.42 * ref_scale) * (0.68 * tight + 0.32 * mid)

    gh = float(np.clip(glass_halation, 0.0, 1.5))
    if gh > 1e-3:
        # Internal reflection in thick faceplate glass: ring-like wide scatter (difference of wide Gaussians)
        hot_glass = np.clip(out - 0.52, 0.0, None)
        if float(hot_glass.max()) > 1e-5:
            r_inner = max(2.5, (7.0 + 8.0 * gh) * scale_w)
            r_outer = max(6.0, (18.0 + 24.0 * gh) * scale_w)
            g_in = lowpass(hot_glass, r_inner)
            g_out = lowpass(hot_glass, r_outer)
            # Ring + wide veil simulates total internal reflection bounce at the front glass surface
            ring_halo = np.clip(1.35 * g_out - 0.35 * g_in, 0.0, None)
            glass_tint = np.array([1.06, 1.01, 0.94], dtype=np.float32)[None, None, :]
            out = out + (0.36 * gh) * ring_halo * glass_tint

    return out.astype(np.float32)


def apply_phosphor_saturation(rgb: np.ndarray, amount: float) -> np.ndarray:
    """
    P22 / EBU CRT phosphor luminous saturation & highlight core incandescence.

    At high beam drive, ZnS:Ag (blue) and ZnS:Cu,Al (green) phosphors saturate earlier than
    Y2O2S:Eu (red), while secondary electron/light scatter across adjacent phosphor stripes
    desaturates blown single-channel highlights toward a glowing white-hot core.
    """
    a = float(np.clip(amount, 0.0, 1.0))
    if a <= 1e-4:
        return rgb
    x = np.clip(rgb, 0.0, 2.0).astype(np.float32)
    # Per-channel P22 saturation knee (Red holds linear longest; Green and Blue roll off slightly earlier)
    knees = np.array([0.82, 0.74, 0.70], dtype=np.float32)[None, None, :]
    over = np.clip(x - knees, 0.0, None)
    compressed = np.where(x <= knees, x, knees + (1.0 - knees) * np.tanh(over / np.maximum(1e-3, 1.0 - knees)))
    out = x * (1.0 - 0.65 * a) + compressed * (0.65 * a)

    # White-hot cross-phosphor incandescence when any channel is driven hard
    peak_ch = out.max(axis=-1, keepdims=True)
    lum = (out @ LUMA)[..., None]
    core_drive = np.clip((peak_ch - 0.78) / 0.22, 0.0, 1.0) ** 1.6
    warm_white = np.array([1.02, 1.00, 0.97], dtype=np.float32)[None, None, :]
    target_core = np.maximum(out, (0.65 * peak_ch + 0.35 * lum) * warm_white)
    out = out * (1.0 - 0.55 * a * core_drive) + target_core * (0.55 * a * core_drive)
    return out.astype(np.float32)


def apply_corner_astigmatism(rgb: np.ndarray, corner_defocus: float,
                             scale_w: float = 1.0) -> np.ndarray:
    """
    Deflection yoke corner astigmatism & dynamic focus error.

    As the electron beam sweeps toward the edges and corners of the tube, it strikes the phosphor
    screen at an oblique angle, stretching the circular spot horizontally and radially (`r^2` law).
    The screen centre remains completely untouched (`weight == 0` at `(0, 0)`).
    """
    cd = float(np.clip(corner_defocus, 0.0, 1.5))
    if cd <= 1e-4:
        return rgb
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1, w - 1)) * 2.0 - 1.0
    ny = (yy / max(1, h - 1)) * 2.0 - 1.0
    r2 = np.clip((nx * nx + ny * ny) / 2.0, 0.0, 1.0)

    # Oblique landing stretches horizontal focus more than vertical focus
    soft_h = lowpass(rgb, max(0.6, (1.4 + 3.2 * cd) * scale_w), axis=1)
    soft_hv = lowpass(soft_h, max(0.4, (0.8 + 1.8 * cd) * scale_w), axis=0)
    w_corner = (r2 ** 1.35 * np.clip(cd, 0.0, 1.0))[..., None].astype(np.float32)
    return (rgb * (1.0 - w_corner) + soft_hv * w_corner).astype(np.float32)


def apply_ehv_line_sag(rgb: np.ndarray, ehv_sag: float, scale_w: float = 1.0) -> np.ndarray:
    """
    Per-scanline Horizontal High-Voltage (EHT) load sag ("line blooming").

    On consumer CRTs, a scanline carrying high average beam current pulls down the flyback's
    high-voltage supply over the horizontal RC time constant (~3-8 scanlines). Lower anode voltage
    reduces electron velocity, increasing deflection sensitivity so bright horizontal bands expand
    outward from the centre by several pixels.
    """
    sag = float(np.clip(ehv_sag, 0.0, 1.5))
    if sag <= 1e-4:
        return rgb
    h, w = rgb.shape[:2]
    row_lum = (np.clip(rgb, 0.0, 1.0) @ LUMA).mean(axis=1)
    # Flyback capacitor RC smoothing down the field (causal exponential + symmetric smoothing)
    row_load = lowpass(row_lum.astype(np.float32), max(1.5, h * 0.025), axis=0)
    delta_load = row_load - float(row_load.mean())
    if float(np.abs(delta_load).max()) <= 1e-4:
        return rgb

    # Each line expands horizontally about the screen centre by `1 + sag_factor * delta_load`
    xx = np.arange(w, dtype=np.float32)[None, :]
    cx = (w - 1) * 0.5
    expand = (sag * 0.045 * delta_load)[:, None]  # fractional horizontal expansion
    src_x = np.clip(cx + (xx - cx) / np.maximum(0.85, 1.0 + expand), 0.0, w - 1.001)
    x0 = np.floor(src_x).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    fx = (src_x - x0)[..., None].astype(np.float32)
    rows_idx = np.arange(h, dtype=np.int32)[:, None]
    out = rgb[rows_idx, x0] * (1.0 - fx) + rgb[rows_idx, x1] * fx
    return out.astype(np.float32)


_BEZEL_GLASS_CACHE: dict[tuple, tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]] = {}


def apply_curved_bezel_and_glass(rgb: np.ndarray, bezel: float, reflection: float,
                                 curvature: float = 0.0) -> np.ndarray:
    """
    Physical CRT curved mask bezel, corner cutoff, inner shadow rim, and 3D Fresnel glass reflection.
    """
    h, w = rgb.shape[:2]
    refl = float(np.clip(reflection, 0.0, 1.0))
    bz = float(np.clip(bezel, 0.0, 1.0))
    if refl <= 1e-3 and bz <= 1e-3:
        return rgb
    key = (h, w, round(bz, 4), round(refl, 4), round(float(curvature), 4))
    hit = _BEZEL_GLASS_CACHE.get(key)
    if hit is None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx / max(1, w - 1)) * 2.0 - 1.0
        ny = (yy / max(1, h - 1)) * 2.0 - 1.0
        glare_term = None
        if refl > 1e-3:
            curv_eff = 0.35 + 0.65 * float(np.clip(curvature, 0.0, 1.2))
            r2 = nx * nx + ny * ny
            u_refl = nx * (1.0 + 0.18 * curv_eff * r2)
            v_refl = ny * (1.0 + 0.18 * curv_eff * r2)
            spot1 = np.exp(-(((u_refl + 0.42) ** 2 + (v_refl + 0.48) ** 2) / 0.22))
            spot2 = 0.38 * np.exp(-(((u_refl - 0.52) ** 2 + (v_refl + 0.36) ** 2) / 0.45))
            fresnel = (1.0 - np.clip(1.0 - 0.42 * r2, 0.0, 1.0)) ** 2
            sky_grad = np.clip(0.55 - 0.45 * v_refl, 0.0, 1.0) * (0.25 + 0.75 * fresnel)
            glass_glare = (0.72 * spot1 + 0.28 * spot2 + 0.24 * sky_grad)[..., None]
            glare_tint = np.array([0.96, 0.99, 1.04], dtype=np.float32)[None, None, :]
            glare_term = ((refl * 0.20) * glass_glare * glare_tint).astype(np.float32)
        bezel_mask = None
        lip_rgb = None
        if bz > 1e-3:
            bow = 0.035 * bz * (1.0 + 0.8 * float(np.clip(curvature, 0.0, 1.0)))
            ax = np.abs(nx) * (1.0 + bow * (ny * ny))
            ay = np.abs(ny) * (1.0 + bow * (nx * nx))
            power = max(4.0, 14.0 - 8.5 * bz)
            dist = (ax ** power + ay ** power) ** (1.0 / power)
            edge_limit = 1.015 - 0.035 * bz
            inside = np.clip((edge_limit - dist) / max(0.006, 0.012 * bz), 0.0, 1.0)
            rim_shadow = np.clip((edge_limit - dist) / max(0.025, 0.065 * bz), 0.0, 1.0) ** 0.45
            bezel_mask = (inside * (0.35 + 0.65 * rim_shadow))[..., None].astype(np.float32)
            lip = np.exp(-0.5 * ((dist - edge_limit - 0.006) / 0.005) ** 2) * (1.0 - inside)
            lip_rgb = ((0.045 * bz * lip)[..., None] * np.array([0.85, 0.88, 0.92], dtype=np.float32)).astype(np.float32)
        hit = (glare_term, bezel_mask, lip_rgb)
        if len(_BEZEL_GLASS_CACHE) > 24:
            _BEZEL_GLASS_CACHE.pop(next(iter(_BEZEL_GLASS_CACHE)), None)
        _BEZEL_GLASS_CACHE[key] = hit

    glare_term, bezel_mask, lip_rgb = hit
    out = rgb.astype(np.float32, copy=True)
    if glare_term is not None:
        out += glare_term
    if bezel_mask is not None and lip_rgb is not None:
        out *= bezel_mask
        out += lip_rgb
    return out


# ===========================================================================
# 6. CRT Strong Magnetic Field, Shadow-Mask Remanence, PTC Degaussing Coil &
#    Special Electromagnetic Environment (Calibrated against Ref `crt_strong_magnet_reference`)
# ===========================================================================

@dataclass
class CRTMagneticEnvSpec:
    """
    Physical specification for a CRT exposed to strong external magnetic fields (e.g., NdFeB
    permanent magnet stack), ferromagnetic shadow-mask/aperture-grille remanence, built-in
    PTC thermistor degaussing coil oscillation, and external 50/60 Hz stray AC magnetic hum.
    """
    mode: int = 0                   # 0=Off, 1=Magnet Approaching (Lorentz Vortex + Multi-Order Purity Rainbow Rings),
                                    # 2=Magnet Pressed on Glass (Black-Hole Electron Repulsion Void + Caustic Rim + Gradient Moire),
                                    # 3=Magnet Removed (Ferromagnetic Shadow-Mask Remanence Stain),
                                    # 4=PTC Degaussing Coil Triggered (Decaying 50/60Hz AC Oscillation Wiping Remanence -> Pure),
                                    # 5=2x2 CRT Strong Magnet & Degauss Physics Quad Lab
    strength: float = 0.0           # External magnetic flux density B_ext (0.0 .. 1.5, 1.0 ~ 0.35 Tesla NdFeB surface field)
    pos_x: float = 0.52             # Magnet center horizontal position (0..1)
    pos_y: float = 0.48             # Magnet center vertical position (0..1)
    angle_deg: float = 25.0         # Magnet dipole axis orientation angle (-180..180 deg)
    remanence: float = 0.0          # Residual ferromagnetic magnetization M_rem locked in the steel/Invar shadow mask (0..1)
    degauss_active: float = 0.0     # Built-in PTC thermistor degaussing coil trigger (0..1)
    env_hum_ripple: float = 0.0     # External stray 50/60Hz transformer AC magnetic field (Hum Snake) + high-humidity corona (0..1)
    quad_corner: int = 0            # 0=2x2 Quad Lab, 1..4=Full-frame single corner
    show_border: int = 1            # 0=Seamless, 1=Thin Crosshair, 2=Studio Bezel
    show_text: int = 0              # 0=No Text (Default), 1=Corner Labels, 2=Labels + Telemetry HUD

    @classmethod
    def from_params(cls, d: dict) -> "CRTMagneticEnvSpec":
        rem = d.get("crt_magnet_remanence", d.get("crt_remanence", 0.0))
        deg = d.get("crt_degauss_trigger", d.get("crt_degauss_active", 0.0))
        hum = d.get("crt_ac_hum_field", d.get("crt_env_hum_ripple", 0.0))
        return cls(
            mode=int(np.clip(round(float(d.get("crt_magnet_mode", 0.0))), 0, 5)),
            strength=float(np.clip(float(d.get("crt_magnet_strength", 0.0)), 0.0, 1.5)),
            pos_x=float(np.clip(float(d.get("crt_magnet_x", 0.52)), 0.05, 0.95)),
            pos_y=float(np.clip(float(d.get("crt_magnet_y", 0.48)), 0.05, 0.95)),
            angle_deg=float(np.clip(float(d.get("crt_magnet_angle_deg", 25.0)), -180.0, 180.0)),
            remanence=float(np.clip(float(rem), 0.0, 1.0)),
            degauss_active=float(np.clip(float(deg), 0.0, 1.0)),
            env_hum_ripple=float(np.clip(float(hum), 0.0, 1.0)),
            quad_corner=int(np.clip(round(float(d.get("crt_magnet_quad_corner", 0.0))), 0, 4)),
            show_border=int(np.clip(round(float(d.get("crt_magnet_show_border", 1.0))), 0, 2)),
            show_text=int(np.clip(round(float(d.get("crt_magnet_show_text", 0.0))), 0, 2)),
        )

    def is_active(self) -> bool:
        return (
            self.mode > 0
            or self.strength > 1e-4
            or self.remanence > 1e-4
            or self.degauss_active > 1e-4
            or self.env_hum_ripple > 1e-4
        )


def _bilinear_warp_rgb(rgb: np.ndarray, src_y: np.ndarray, src_x: np.ndarray) -> np.ndarray:
    """Fast vectorized 2D bilinear resampling for CRT electron-beam trajectory deflection."""
    h, w = rgb.shape[:2]
    sy = np.clip(src_y, 0.0, h - 1.001)
    sx = np.clip(src_x, 0.0, w - 1.001)
    y0 = np.floor(sy).astype(np.int32)
    x0 = np.floor(sx).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (sy - y0)[..., None].astype(np.float32)
    fx = (sx - x0)[..., None].astype(np.float32)
    top = rgb[y0, x0] * (1.0 - fx) + rgb[y0, x1] * fx
    bot = rgb[y1, x0] * (1.0 - fx) + rgb[y1, x1] * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


def _render_single_crt_magnet_mode(
    rgb: np.ndarray,
    eff_mode: int,
    spec: CRTMagneticEnvSpec,
    frame_index: int = 0,
    fps: float = 29.97,
    seed: int = 2026,
) -> tuple[np.ndarray, dict]:
    """
    Core Lorentz electron-beam magnetic deflection, shadow-mask purity phase shift,
    close-contact electron repulsion void + caustic ring, ferromagnetic remanence, and PTC degauss solver.
    """
    h, w = rgb.shape[:2]
    out = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=True)
    t_sec = float(frame_index) / max(1.0, float(fps))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    aspect = float(w) / max(1.0, float(h))
    mx = float(spec.pos_x)
    my = float(spec.pos_y)
    dx = (xx / max(1.0, w - 1.0) - mx) * aspect
    dy = (yy / max(1.0, h - 1.0) - my)
    r2 = dx * dx + dy * dy
    r = np.sqrt(r2 + 1e-6)
    theta = np.arctan2(dy, dx)
    ang_rad = np.deg2rad(float(spec.angle_deg))

    # Determine effective physical drive from mode + sliders
    b_str = float(spec.strength)
    rem_str = float(spec.remanence)
    deg_str = float(spec.degauss_active)
    hum_str = float(spec.env_hum_ripple)

    if eff_mode == 1 and b_str < 0.15:
        b_str = 0.72
    elif eff_mode == 2 and b_str < 0.25:
        b_str = 1.15
    elif eff_mode == 3:
        b_str = 0.0
        if rem_str < 0.15:
            rem_str = 0.78
    elif eff_mode == 4:
        b_str = 0.0
        if deg_str < 0.15:
            deg_str = 0.90
        if rem_str < 0.15:
            rem_str = 0.75

    # 1. External Stray 50/60Hz AC Transformer Magnetic Hum ("Hum Snake")
    hum_disp_px = 0.0
    if hum_str > 1e-4:
        hum_phase = 2.0 * np.pi * (yy / max(1.0, float(h)) * 1.5 - t_sec * 1.97)
        snake_x = (hum_str * 11.5 * (w / 720.0)) * np.sin(hum_phase)
        out = horizontal_shift(out, snake_x.squeeze() if snake_x.ndim > 1 else snake_x)
        # Subtle 50/60Hz brightness hum bar + high-humidity anode corona micro-discharge
        hum_bar = 1.0 + 0.075 * hum_str * np.cos(hum_phase)
        out = np.clip(out * hum_bar[..., None], 0.0, 1.0)
        hum_disp_px = float(np.max(np.abs(snake_x)))

    # 2. PTC Thermistor Degaussing Coil Decaying AC Magnetic Oscillation (eff_mode == 4 or degauss_active > 0)
    degauss_envelope = 0.0
    rem_survival = 1.0
    if deg_str > 1e-4 or eff_mode == 4:
        # PTC thermistor heats up over tau ~ 0.65 s, exponentially damping the 50/60 Hz AC coil current
        # At frame_index=0..4, strong oscillatory twist + rainbow purity wash; by frame_index >= 25 (t >= 0.85s),
        # coil current & shadow-mask remanence both decay to zero!
        tau_ptc = 0.55
        degauss_envelope = float(deg_str * np.exp(-t_sec / tau_ptc))
        rem_survival = float(np.exp(-t_sec / (0.35 * tau_ptc)))
        if degauss_envelope > 1e-3:
            ac_carrier = float(np.cos(2.0 * np.pi * 11.5 * t_sec + 0.4))
            ac_quad = float(np.sin(2.0 * np.pi * 11.5 * t_sec + 0.4))
            # Perimeter degaussing coil field is strongest near screen edges and twists the whole raster
            twist_ang = degauss_envelope * 0.28 * ac_carrier * (0.45 + 0.55 * r2)
            radial_shake = 1.0 + degauss_envelope * 0.14 * ac_quad * np.sin(2.0 * theta)
            cos_t = np.cos(twist_ang)
            sin_t = np.sin(twist_ang)
            dx_d = (dx * cos_t - dy * sin_t) * radial_shake
            dy_d = (dx * sin_t + dy * cos_t) * radial_shake
            sy_d = (my + dy_d) * (h - 1.0)
            sx_d = (mx + dx_d / aspect) * (w - 1.0)
            out = _bilinear_warp_rgb(out, sy_d, sx_d)

            # Transient AC purity rainbow wave across the screen during active degaussing
            deg_phase = degauss_envelope * 9.5 * (dx * np.cos(t_sec * 18.0) + dy * np.sin(t_sec * 18.0))
            p_r = 0.5 + 0.5 * np.cos(deg_phase)
            p_g = 0.5 + 0.5 * np.cos(deg_phase - 2.0944)
            p_b = 0.5 + 0.5 * np.cos(deg_phase + 2.0944)
            purity_mat = np.stack([p_r, p_g, p_b], axis=-1).astype(np.float32)
            purity_mat /= np.maximum(1e-4, purity_mat.mean(axis=-1, keepdims=True))
            w_deg_c = float(np.clip(degauss_envelope * 0.72, 0.0, 0.85))
            out = np.clip(out * ((1.0 - w_deg_c) + w_deg_c * purity_mat), 0.0, 1.0)

    # 3. Active External Permanent Magnet (Lorentz Deflection + Multi-Order Purity Rainbow + Close-Contact Black Hole)
    max_deflection_px = 0.0
    void_fraction = 0.0
    caustic_peak = 0.0
    purity_shift_rad = 0.0

    if b_str > 1e-4 and eff_mode in (0, 1, 2):
        is_contact_blackhole = (eff_mode == 2) or (b_str >= 0.95)
        # Dipole stand-off distance d0: smaller when magnet is pressed directly on the glass (`eff_mode == 2`)
        d0 = 0.14 if not is_contact_blackhole else 0.085
        dipole_kernel = (d0 * d0) / ((r2 + d0 * d0) ** 1.35)
        # Dipole quadrupole asymmetry along magnet orientation axis `ang_rad`
        dipole_lobe = 1.0 + 0.36 * np.cos(2.0 * (theta - ang_rad)) + 0.18 * np.sin(theta - ang_rad)

        # (a) Lorentz Vortex Twist (axial B_z component: F = -e(v_perp x B_z))
        swirl_rad = b_str * (1.35 if not is_contact_blackhole else 1.85) * dipole_kernel * dipole_lobe
        # (b) Lorentz Radial Push/Pinch (transverse B_perp component: F = -e(v_z x B_perp))
        # When pressed close (`is_contact_blackhole`), electrons are strongly repelled outward from the pole center,
        # which pulls source sampling coordinates inward toward `(mx, my)` and creates a central void!
        radial_push = b_str * (0.28 if not is_contact_blackhole else 0.58) * dipole_kernel * (
            1.0 + 0.42 * np.cos(theta - ang_rad)
        )

        # Separate per-gun (R, G, B) landing trajectories because the 3 guns have slightly different convergence angles
        gun_channels = []
        for ch_idx, gun_delta in enumerate((-0.085, 0.0, +0.085)):
            ch_swirl = swirl_rad * (1.0 + gun_delta * b_str)
            ch_push = radial_push * (1.0 + 0.8 * gun_delta * b_str)
            cos_s = np.cos(ch_swirl)
            sin_s = np.sin(ch_swirl)
            # Deflected ray lookup
            scale_r = np.maximum(0.12, 1.0 - ch_push)
            dx_s = (dx * cos_s - dy * sin_s) * scale_r
            dy_s = (dx * sin_s + dy * cos_s) * scale_r
            sy_c = (my + dy_s) * (h - 1.0)
            sx_c = (mx + dx_s / aspect) * (w - 1.0)
            warped_ch = _bilinear_warp_rgb(out[..., ch_idx:ch_idx + 1], sy_c, sx_c)[..., 0]
            gun_channels.append(warped_ch)
        out = np.stack(gun_channels, axis=-1)

        disp_x_px = (dx * np.cos(swirl_rad) - dy * np.sin(swirl_rad) - dx) * (w / aspect)
        disp_y_px = (dx * np.sin(swirl_rad) + dy * np.cos(swirl_rad) - dy) * h
        max_deflection_px = float(np.percentile(np.hypot(disp_x_px, disp_y_px), 99.0))

        # (c) Multi-Order Shadow-Mask / Aperture-Grille Purity Rainbow Rings (Ref `t=20s..65s`)
        # As the electron beam's landing angle is deflected by `Delta_theta_B`, the beam sweeps across
        # multiple adjacent R -> G -> B -> R phosphor triads, creating vivid concentric/lobed rainbow rings!
        n_orders = (2.6 + 3.2 * b_str) if not is_contact_blackhole else (4.2 + 4.0 * b_str)
        purity_phase = (
            2.0 * np.pi * n_orders * dipole_kernel * (1.0 + 0.32 * np.cos(theta - ang_rad) - 0.18 * np.sin(2.0 * (theta - ang_rad)))
            + 0.85 * np.sin(theta - ang_rad)
        )
        purity_shift_rad = float(np.max(np.abs(purity_phase)))

        # Three-phase phosphor triad excitation modulation
        ring_env = np.clip(dipole_kernel * b_str * 1.95, 0.0, 1.0) ** 0.68
        pr = 0.52 + 0.48 * np.cos(purity_phase - 0.35)
        pg = 0.52 + 0.48 * np.cos(purity_phase - 2.44)
        pb = 0.52 + 0.48 * np.cos(purity_phase + 1.75)
        triad_color = np.stack([pr, pg, pb], axis=-1).astype(np.float32)
        triad_color /= np.maximum(1e-4, triad_color.mean(axis=-1, keepdims=True))

        # Preserve luminance while rotating purity across rainbow bands (plus glowing phosphor excitation on dark areas)
        lum_cur = np.clip(out @ LUMA, 0.0, 1.0)[..., None]
        colored = out * (1.0 - 0.78 * ring_env[..., None]) + (
            (0.72 * out + 0.28 * np.maximum(lum_cur, 0.18)) * triad_color * 1.22
        ) * (0.78 * ring_env[..., None])
        out = np.clip(colored, 0.0, 1.0)

        # (d) Close-Contact "Black Hole" Electron Repulsion Void, Caustic Bright Rim & High-Gradient Moire (Ref `t=95s..110s`)
        if is_contact_blackhole:
            r_void = (0.095 + 0.065 * min(1.5, b_str)) * (
                1.0 + 0.14 * np.cos(2.0 * (theta - ang_rad)) + 0.06 * np.sin(3.0 * (theta - ang_rad))
            )
            # Inside `r < r_void`, all electrons are deflected away -> pitch-black electron shadow void
            void_mask = np.clip((r - r_void) / np.maximum(0.008, 0.024 * r_void), 0.0, 1.0)
            void_fraction = float(np.mean(void_mask < 0.15))

            # Electrons expelled from the void pile up at `r ≈ 1.05..1.32 * r_void`, creating a blazing caustic arc
            caustic_ring = np.exp(-0.5 * ((r - r_void * 1.14) / np.maximum(0.010, 0.22 * r_void)) ** 2)
            caustic_lobe = 0.70 + 0.30 * np.cos(theta - ang_rad - 0.5)
            caustic_drive = caustic_ring * caustic_lobe * min(1.4, b_str)
            caustic_peak = float(caustic_drive.max())

            # High-gradient spatial Moiré ripples radiating around the caustic boundary (beating between compressed
            # scanlines and shadow mask slots, clearly visible in Ref at `t=95s..110s`)
            moire_zone = np.exp(-0.5 * ((r - r_void * 1.65) / np.maximum(0.025, 0.85 * r_void)) ** 2) * void_mask
            k_moire = 2.0 * np.pi * (28.0 / np.maximum(r + 0.04, 0.06))
            moire_wave = np.cos(k_moire * (r - r_void) + 4.0 * np.sin(2.0 * (theta - ang_rad)))
            moire_rgb = np.stack([
                1.0 + 0.26 * moire_zone * np.cos(k_moire * (r - r_void)),
                1.0 + 0.26 * moire_zone * moire_wave,
                1.0 - 0.22 * moire_zone * moire_wave,
            ], axis=-1).astype(np.float32)

            caustic_tint = np.array([0.88, 1.04, 1.18], dtype=np.float32)[None, None, :]
            out = out * moire_rgb
            out = out * void_mask[..., None] + (0.68 * caustic_drive * void_mask)[..., None] * caustic_tint
            out = np.clip(out, 0.0, 1.0)

    # 4. Ferromagnetic Shadow-Mask / Aperture-Grille Residual Magnetization (`remanence`, Ref `t=125s`)
    eff_rem = rem_str * rem_survival
    if eff_rem > 1e-4:
        # After the magnet is removed, the steel shadow mask retains permanent dipole magnetization patches:
        # geometry is nearly normal (tiny <1.5px warp), but purity stays shifted into magenta/cyan/green blotches!
        d_rem = 0.22
        rem_kernel = (d_rem * d_rem) / ((r2 + d_rem * d_rem) ** 1.15)
        rem_lobe = 1.0 + 0.45 * np.cos(2.0 * (theta - ang_rad)) - 0.25 * np.sin(theta - ang_rad)
        rem_phase = 2.0 * np.pi * 1.45 * eff_rem * rem_kernel * rem_lobe

        w_rem = np.clip(rem_kernel * eff_rem * 1.45, 0.0, 0.88)[..., None]
        mr = 0.55 + 0.45 * np.cos(rem_phase + 0.4)
        mg = 0.55 + 0.45 * np.cos(rem_phase - 1.95)
        mb = 0.55 + 0.45 * np.cos(rem_phase + 2.25)
        rem_color = np.stack([mr, mg, mb], axis=-1).astype(np.float32)
        rem_color /= np.maximum(1e-4, rem_color.mean(axis=-1, keepdims=True))
        out = np.clip(out * (1.0 - w_rem) + (out * rem_color * 1.14) * w_rem, 0.0, 1.0)
        purity_shift_rad = max(purity_shift_rad, float(np.max(np.abs(rem_phase))))

    info = {
        "mode": int(eff_mode),
        "magnet_strength": round(float(b_str), 4),
        "max_deflection_px": round(float(max_deflection_px), 2),
        "purity_shift_rad": round(float(purity_shift_rad), 3),
        "void_fraction": round(float(void_fraction), 4),
        "caustic_peak": round(float(caustic_peak), 4),
        "remanence_effective": round(float(eff_rem), 4),
        "degauss_envelope": round(float(degauss_envelope), 4),
        "hum_displacement_px": round(float(hum_disp_px), 2),
    }
    return np.clip(out, 0.0, 1.0).astype(np.float32), info


def apply_crt_magnetic_and_special_env(
    rgb: np.ndarray,
    spec: CRTMagneticEnvSpec,
    frame_index: int = 0,
    fps: float = 29.97,
    seed: int = 2026,
) -> tuple[np.ndarray, dict]:
    """
    Apply CRT Strong Magnetic Field, Shadow-Mask Remanence, PTC Degaussing Coil, or the `2x2` Quad Lab (`mode == 5`).
    """
    if not spec.is_active():
        return rgb, {"active": False}

    eff_mode = int(spec.mode)
    if eff_mode == 0:
        if spec.degauss_active > 1e-4:
            eff_mode = 4
        elif spec.strength >= 0.95:
            eff_mode = 2
        elif spec.strength > 1e-4:
            eff_mode = 1
        elif spec.remanence > 1e-4:
            eff_mode = 3

    if eff_mode != 5:
        out, info = _render_single_crt_magnet_mode(rgb, eff_mode, spec, frame_index=frame_index, fps=fps, seed=seed)
        info["active"] = True
        return out, info

    # Mode 5: 2x2 CRT Strong Magnet, Black-Hole Caustic, Shadow-Mask Remanence & PTC Degaussing Quad Lab
    h, w = rgb.shape[:2]
    corner = int(np.clip(spec.quad_corner, 0, 4))
    if corner in (1, 2, 3, 4):
        c_out, c_info = _render_single_crt_magnet_mode(
            rgb, corner, spec,
            frame_index=(frame_index if corner != 4 else max(2, frame_index)),
            fps=fps, seed=seed,
        )
        c_info["active"] = True
        c_info["quad_corner"] = corner
        return c_out, c_info

    q1, i1 = _render_single_crt_magnet_mode(rgb, 1, spec, frame_index=frame_index, fps=fps, seed=seed)
    q2, i2 = _render_single_crt_magnet_mode(rgb, 2, spec, frame_index=frame_index, fps=fps, seed=seed)
    q3, i3 = _render_single_crt_magnet_mode(rgb, 3, spec, frame_index=frame_index, fps=fps, seed=seed)
    q4, i4 = _render_single_crt_magnet_mode(rgb, 4, spec, frame_index=max(3, frame_index), fps=fps, seed=seed)

    hh, hw = h // 2, w // 2
    from ..stages.display import _resample_to
    canvas = np.zeros_like(rgb, dtype=np.float32)
    canvas[:hh, :hw] = _resample_to(q1, hh, hw)
    canvas[:hh, hw:] = _resample_to(q2, hh, w - hw)
    canvas[hh:, :hw] = _resample_to(q3, h - hh, hw)
    canvas[hh:, hw:] = _resample_to(q4, h - hh, w - hw)

    if int(spec.show_border) >= 1:
        bw = 1 if int(spec.show_border) == 1 else 2
        line_c = np.array([0.22, 0.85, 0.72], dtype=np.float32)
        canvas[max(0, hh - bw):min(h, hh + bw), :] = line_c
        canvas[:, max(0, hw - bw):min(w, hw + bw)] = line_c

    return canvas, {
        "active": True,
        "mode": 5,
        "quad_corner": 0,
        "quadrants": {
            "q1_lorentz_vortex": i1,
            "q2_blackhole_caustic": i2,
            "q3_shadow_mask_remanence": i3,
            "q4_ptc_degauss_coil": i4,
        },
    }


# ===========================================================================
# 7. Sony Trinitron KV-1214 (1978-1981 Showa 12-Inch Cylindrical Trinitron TV)
#    & 4K Reshoot Aperture-Grille Moire Physics (Calibrated against Ref vs Ref)
# ===========================================================================

@dataclass
class TrinitronKV1214Spec:
    """
    Physical specification for the 1978-1981 Sony Trinitron KV-1214 12-inch Color TV
    calibrated directly against Ref (`trinitron_4k_moire_reference`)
    using Ref (`digital_master_reference` original MV) as the reference digital source:
      * 12-inch Cylindrical Aperture Grille (`单枪三束柱面特丽珑荫栅`): horizontal-only curvature,
        continuous vertical RGB phosphor stripes with zero vertical slot bridges, and a single
        horizontal tungsten tension damper wire at `y = 0.52 H`.
      * 4K Camera-to-Grille Curved Moire Interference Fringes (`kv1214_moire_strength`):
        High-visibility hyperbolic/curved RGB Moire beat fringes (`每根摩尔纹都清清楚楚`) produced by the
        cylindrical curvature + micro-perspective modulation of the vertical Aperture Grille pitch against
        the 4K camera sensor grid, raising horizontal-to-vertical HF anisotropy (`HFx / HFy ~ 1.35..1.75`).
      * NTSC-J 9300K (D93) Cyan-Blue White Point (`R/G ~ 0.78, B/G ~ 1.12`) + Y2O2S:Eu Red Halation.
    """
    enabled: bool = False
    grille_pitch: float = 2.65      # Coarse 12-inch Trinitron vertical aperture-grille triad pitch (px)
    moire_strength: float = 0.78    # 4K camera-to-grille curved Moire interference fringe contrast (0..1.5)
    d93_bloom: float = 0.75         # NTSC-J 9300K cyan-blue white point + red phosphor halation & EHT bloom (0..1.5)
    cyl_bezel: float = 0.42         # Cylindrical horizontal-only curvature & KV-1214 silver/grey recessed bezel (0..1)
    ab_split: int = 0               # 0=Full KV-1214 (Ref), 1=Left Ref Original / Right Ref KV-1214 Split, 2=2x2 Ref/Ref Lab

    @classmethod
    def from_params(cls, d: dict) -> "TrinitronKV1214Spec":
        gp = d.get("kv1214_grille_pitch_px", d.get("kv1214_grille_pitch", 2.65))
        ms = d.get("kv1214_moire_amount", d.get("kv1214_moire_strength", 0.78))
        d93 = d.get("kv1214_ntscj_9300k", d.get("kv1214_9300k_bloom", 0.75))
        cbz = d.get("kv1214_bezel_frame", d.get("kv1214_cyl_bezel", 0.42))
        return cls(
            enabled=float(d.get("kv1214_enabled", 0.0)) > 0.5,
            grille_pitch=float(np.clip(float(gp), 1.4, 6.0)),
            moire_strength=float(np.clip(float(ms), 0.0, 1.5)),
            d93_bloom=float(np.clip(float(d93), 0.0, 1.5)),
            cyl_bezel=float(np.clip(float(cbz), 0.0, 1.0)),
            ab_split=int(np.clip(round(float(d.get("kv1214_ab_split", 0.0))), 0, 2)),
        )

    def is_active(self) -> bool:
        return bool(self.enabled or self.ab_split > 0)


def apply_trinitron_kv1214_optics(
    rgb: np.ndarray,
    spec: TrinitronKV1214Spec,
    frame_index: int = 0,
    reference_p4_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Render the Sony Trinitron KV-1214 12-inch cylindrical CRT optics + 4K Moire interference fringes,
    and optionally compare side-by-side against the Ref original digital master.
    """
    if not spec.is_active():
        return rgb, {"active": False}

    h, w = rgb.shape[:2]
    scale_w = max(0.5, float(w) / 720.0)
    scale_h = max(0.5, float(h) / 480.0)
    p4_ref = np.clip(reference_p4_rgb if reference_p4_rgb is not None else rgb, 0.0, 1.0).astype(np.float32)
    out = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=True)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(1.0, w - 1.0)) * 2.0 - 1.0
    ny = (yy / max(1.0, h - 1.0)) * 2.0 - 1.0

    # 1. Cylindrical Trinitron Geometry (Bends ONLY horizontally along X; vertical generator lines are straight!)
    cyl_k = float(spec.cyl_bezel) * 0.16
    if cyl_k > 1e-4:
        # Line-load horizontal EHT sag coupled with cylindrical horizontal curvature
        row_lum = (out @ LUMA).mean(axis=1)
        row_load = lowpass(row_lum.astype(np.float32), max(1.5, h * 0.025), axis=0)
        ehv_expand = (0.028 * float(spec.d93_bloom) * (row_load - float(row_load.mean())))[:, None]

        sx_n = nx * (1.0 + cyl_k * (nx * nx) + 0.025 * cyl_k * (ny * ny)) / np.maximum(0.88, 1.0 + ehv_expand)
        sy_n = ny * (1.0 + 0.022 * cyl_k * (nx * nx))
        src_x = (sx_n * 0.5 + 0.5) * (w - 1.0)
        src_y = (sy_n * 0.5 + 0.5) * (h - 1.0)
        # Single-gun three-beam Trinitron lateral static convergence trim at left/right edges
        r_plane = _bilinear_warp_rgb(out[..., 0:1], src_y, src_x - 0.65 * scale_w * nx)[..., 0]
        g_plane = _bilinear_warp_rgb(out[..., 1:2], src_y, src_x)[..., 0]
        b_plane = _bilinear_warp_rgb(out[..., 2:3], src_y, src_x + 0.65 * scale_w * nx)[..., 0]
        out = np.stack([r_plane, g_plane, b_plane], axis=-1)

    # 2. NTSC-J 9300K (D93) Cyan-Blue White Balance + Y2O2S:Eu Red Phosphor Halation (Calibrated to Ref vs Ref)
    # Measured on Ref -> Ref: R/G shifts from ~0.90 to ~0.78, B/G shifts to ~1.12, with vivid red/magenta halation
    d93 = float(np.clip(spec.d93_bloom, 0.0, 1.5))
    if d93 > 1e-4:
        # Horizontal-only RF/IF luma filtering (softens horizontal transitions before the sharp vertical grille)
        out_lp_x = lowpass(out, max(0.55, 0.95 * scale_w), axis=1)
        out = out * 0.35 + out_lp_x * 0.65

        # Wide Y2O2S:Eu red-orange glass halation + 9300K blue-cyan electron bloom
        hot = np.clip(out - 0.55, 0.0, None)
        halo_tight = lowpass(hot, max(1.2, 3.2 * scale_w))
        halo_wide = lowpass(hot, max(3.5, 11.0 * scale_w))
        red_halation = np.array([1.22, 0.86, 0.92], dtype=np.float32)[None, None, :]
        blue_bloom = np.array([0.82, 0.98, 1.24], dtype=np.float32)[None, None, :]
        out = out + d93 * (0.28 * halo_tight * blue_bloom + 0.22 * halo_wide * red_halation)

        # NTSC-J 9300K color matrix (shifts neutral/pastel highlights toward cool cyan-blue while preserving saturated reds)
        d93_mix = min(1.0, d93 * 0.85)
        gains_9300k = np.array([1.0 - 0.145 * d93_mix, 1.0 + 0.025 * d93_mix, 1.0 + 0.135 * d93_mix], dtype=np.float32)
        out = np.clip(out * gains_9300k[None, None, :], 0.0, 1.0)
        # Slight CRT gamma & black-pedestal contrast curve matching Ref
        out = np.clip(0.012 + 0.988 * (out ** 1.08), 0.0, 1.0)

    # 3. 12-Inch Coarse Vertical Aperture Grille + 4K Curved Moire Interference Fringes (Cached static geometry)
    pitch = max(1.8, float(spec.grille_pitch) * scale_w)
    moire_str = float(np.clip(spec.moire_strength, 0.0, 1.5))
    kv_key = (h, w, round(pitch, 4), round(moire_str, 4), round(float(spec.cyl_bezel), 4))
    kv_static = _BEZEL_GLASS_CACHE.get(("kv1214",) + kv_key)  # type: ignore[arg-type]
    if kv_static is None:
        x_grille = xx * (1.0 + 0.045 * (nx * nx) + 0.012 * nx * ny)
        phi_g = 2.0 * np.pi * (x_grille / pitch)
        stripe_r = 0.5 + 0.5 * np.cos(phi_g)
        stripe_g = 0.5 + 0.5 * np.cos(phi_g - 2.0944)
        stripe_b = 0.5 + 0.5 * np.cos(phi_g + 2.0944)
        bm_guard = 0.76 + 0.24 * np.cos(phi_g * 1.5) ** 2
        grille_depth = 0.36 + 0.16 * min(1.0, moire_str)
        g_mask = np.stack([
            (1.0 - grille_depth) + grille_depth * stripe_r * bm_guard,
            (1.0 - grille_depth) + grille_depth * stripe_g * bm_guard,
            (1.0 - grille_depth) + grille_depth * stripe_b * bm_guard,
        ], axis=-1).astype(np.float32)
        wy = 0.52 * h
        damper_wire = 1.0 - 0.16 * np.exp(-0.5 * ((yy - wy) / max(0.55, 0.65 * scale_h)) ** 2)
        g_mask *= damper_wire[..., None]
        for ch in range(3):
            g_mask[..., ch] /= max(1e-5, float(g_mask[..., ch].mean()))

        bp1_spatial = (2.0 * np.pi * (nx * 6.8 + 1.85 * (nx ** 3) + 0.95 * nx * (ny ** 2) + 0.42 * ny)).astype(np.float32)
        cos_bp1 = (0.68 * np.cos(bp1_spatial)).astype(np.float32)
        sin_bp1 = (0.68 * np.sin(bp1_spatial)).astype(np.float32)
        bp2 = (2.0 * np.pi * (nx * 14.5 + 2.4 * (nx ** 2) * ny - 0.65 * ny)).astype(np.float32)
        m_wave_2 = (0.32 * np.cos(bp2)).astype(np.float32)
        mr_b2 = (0.32 * np.cos(bp2 - 0.30)).astype(np.float32)
        mb_b2 = (0.32 * np.cos(bp2 + 0.30)).astype(np.float32)
        scan_p = max(1.8, 2.0 * scale_h)
        scan_wave = (1.0 - 0.12 * (0.5 - 0.5 * np.cos(2.0 * np.pi * (yy / scan_p)))).astype(np.float32)
        cyl_mask = None
        if spec.cyl_bezel > 1e-3:
            bz = float(spec.cyl_bezel)
            dist_cyl = (np.abs(nx) ** 8.0 + (np.abs(ny) * (1.0 + 0.025 * bz * (nx * nx))) ** 10.0) ** 0.11
            inside = np.clip((1.008 - 0.022 * bz - dist_cyl) / 0.008, 0.0, 1.0)
            rim = np.clip((1.008 - 0.022 * bz - dist_cyl) / 0.042, 0.0, 1.0) ** 0.40
            cyl_mask = (inside * (0.42 + 0.58 * rim))[..., None].astype(np.float32)
        kv_static = (g_mask, cos_bp1, sin_bp1, m_wave_2, mr_b2, mb_b2, scan_wave, cyl_mask)
        _BEZEL_GLASS_CACHE[("kv1214",) + kv_key] = kv_static  # type: ignore[index]

    g_mask, cos_bp1, sin_bp1, m_wave_2, mr_b2, mb_b2, scan_wave, cyl_mask = kv_static  # type: ignore[misc]
    out = out * g_mask

    if moire_str > 1e-4:
        ph_t = 0.15 * float(np.sin(frame_index * 0.2))
        mg_f = cos_bp1 * np.float32(np.cos(ph_t)) - sin_bp1 * np.float32(np.sin(ph_t)) + m_wave_2
        mr_f = cos_bp1 * np.float32(np.cos(ph_t + 0.45)) - sin_bp1 * np.float32(np.sin(ph_t + 0.45)) + mr_b2
        mb_f = cos_bp1 * np.float32(np.cos(ph_t - 0.45)) - sin_bp1 * np.float32(np.sin(ph_t - 0.45)) + mb_b2
        moire_mod = np.stack([mr_f, mg_f, mb_f], axis=-1).astype(np.float32, copy=False)
        lum_w = np.clip(0.25 + 0.85 * (out @ LUMA), 0.20, 1.0)[..., None]
        amp_m = 0.165 * moire_str * lum_w
        out = np.clip(out * (1.0 + amp_m * moire_mod), 0.0, 1.0)

    lum_cur = (out @ LUMA)
    scan_eff = 1.0 - (1.0 - scan_wave) * np.clip(1.15 - 0.75 * lum_cur, 0.25, 1.0)
    scan_eff /= max(1e-5, float(scan_eff.mean()))
    out = np.clip(out * scan_eff[..., None], 0.0, 1.0)

    if cyl_mask is not None:
        out = out * cyl_mask

    # Measure Ref vs Ref telemetry
    g_mean = max(1e-5, float(out[..., 1].mean()))
    r_over_g = float(out[..., 0].mean()) / g_mean
    b_over_g = float(out[..., 2].mean()) / g_mean
    lum_out = out @ LUMA
    hfx = float(np.mean(np.abs(np.diff(lum_out, axis=1))))
    hfy = float(np.mean(np.abs(np.diff(lum_out, axis=0))))
    anisotropy = hfx / max(1e-6, hfy)

    info = {
        "active": True,
        "model": "Sony Trinitron KV-1214 (1978-1981 12-inch Cylindrical Aperture Grille)",
        "R_over_G": round(r_over_g, 4),
        "B_over_G": round(b_over_g, 4),
        "hf_x": round(hfx, 5),
        "hf_y": round(hfy, 5),
        "hfx_over_hfy_anisotropy": round(anisotropy, 3),
        "grille_pitch_px": round(float(pitch), 2),
        "moire_strength": round(float(moire_str), 3),
        "ab_split": int(spec.ab_split),
    }

    if int(spec.ab_split) == 1:
        # Side-by-side Split Screen: Left = Ref Original Master, Right = Ref Sony Trinitron KV-1214
        split_x = w // 2
        comp = out.copy()
        comp[:, :split_x] = p4_ref[:, :split_x]
        comp[:, max(0, split_x - 1):min(w, split_x + 1)] = np.array([0.95, 0.82, 0.25], dtype=np.float32)
        return comp, info
    elif int(spec.ab_split) == 2:
        # 2x2 Ref vs Ref Sony Trinitron KV-1214 Quad Inspection Lab
        hh, hw = h // 2, w // 2
        from ..stages.display import _resample_to
        # Q2: KV-1214 without macro camera Moire; Q3: KV-1214 full 4K Moire; Q4: 3x macro zoom into center Moire + grille
        spec_no_m = TrinitronKV1214Spec(
            enabled=True, grille_pitch=spec.grille_pitch, moire_strength=0.0,
            d93_bloom=spec.d93_bloom, cyl_bezel=spec.cyl_bezel, ab_split=0,
        )
        q2_img, _ = apply_trinitron_kv1214_optics(p4_ref, spec_no_m, frame_index=frame_index)
        cy0, cy1 = int(h * 0.35), max(int(h * 0.35) + 16, int(h * 0.65))
        cx0, cx1 = int(w * 0.35), max(int(w * 0.35) + 16, int(w * 0.65))
        q4_macro = _resample_to(out[cy0:cy1, cx0:cx1], h - hh, w - hw)

        canvas = np.zeros_like(out)
        canvas[:hh, :hw] = _resample_to(p4_ref, hh, hw)
        canvas[:hh, hw:] = _resample_to(q2_img, hh, w - hw)
        canvas[hh:, :hw] = _resample_to(out, h - hh, hw)
        canvas[hh:, hw:] = q4_macro
        line_c = np.array([0.25, 0.82, 0.95], dtype=np.float32)
        canvas[max(0, hh - 1):min(h, hh + 1), :] = line_c
        canvas[:, max(0, hw - 1):min(w, hw + 1)] = line_c
        return canvas, info

    return out, info
