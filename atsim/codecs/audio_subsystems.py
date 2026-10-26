"""
`atsim.codecs.audio_subsystems` -- Complete 10-Subsystem Physical Audio & A/V Synchronization Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, lfilter, sosfilt


def _lp(sig: np.ndarray, cutoff_hz: float, sr: int, order: int = 4) -> np.ndarray:
    nyq = 0.5 * float(sr)
    fc = float(np.clip(cutoff_hz, 40.0, nyq * 0.96))
    if fc >= nyq * 0.95:
        return sig.astype(np.float32, copy=False)
    sos = butter(order, fc / nyq, btype="low", output="sos")
    return sosfilt(sos, sig, axis=0).astype(np.float32)


def _hp(sig: np.ndarray, cutoff_hz: float, sr: int, order: int = 2) -> np.ndarray:
    nyq = 0.5 * float(sr)
    fc = float(np.clip(cutoff_hz, 12.0, nyq * 0.85))
    sos = butter(order, fc / nyq, btype="high", output="sos")
    return sosfilt(sos, sig, axis=0).astype(np.float32)


def _bp(sig: np.ndarray, lo_hz: float, hi_hz: float, sr: int, order: int = 2) -> np.ndarray:
    nyq = 0.5 * float(sr)
    lo = float(np.clip(lo_hz, 18.0, nyq * 0.82))
    hi = float(np.clip(hi_hz, lo + 60.0, nyq * 0.96))
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfilt(sos, sig, axis=0).astype(np.float32)


def _peaking_eq(sig: np.ndarray, center_hz: float, q: float, gain_db: float, sr: int) -> np.ndarray:
    if abs(gain_db) < 0.05 or center_hz <= 20.0 or center_hz >= 0.47 * sr:
        return sig.astype(np.float32, copy=False)
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * center_hz / float(sr)
    alpha = np.sin(w0) / (2.0 * max(0.15, q))
    cos_w0 = np.cos(w0)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return lfilter(b, a, sig, axis=0).astype(np.float32)


# ===========================================================================
# SUBSYSTEM 01 · 麦克风与录音前端 (Microphone & Recording Front-End)
# ===========================================================================
@dataclass
class MicPreampFrontendSpec:
    mic_type: int = 0
    polar_pattern: int = 1
    off_axis_deg: float = 0.0
    sensitivity_db: float = 0.0
    self_noise_level: float = 0.0
    preamp_gain_db: float = 0.0
    agc_amount: float = 0.0
    agc_speed: float = 0.5
    clip_mode: int = 0
    overdrive: float = 0.0
    proximity_effect: float = 0.0
    plosive_pop: float = 0.0
    chassis_vibration: float = 0.0

    @classmethod
    def from_params(cls, params: dict) -> "MicPreampFrontendSpec":
        p = params or {}
        aud = p.get("audio") if isinstance(p.get("audio"), dict) else p
        tape = p.get("tape") if isinstance(p.get("tape"), dict) else {}
        m_type = int(round(float(aud.get("mic_type", 0.0))))
        vib_base = float(np.clip(aud.get("mic_chassis_vibration", 0.0), 0.0, 1.0))
        if m_type == 6 and vib_base < 0.15:
            vib_base = max(vib_base, 0.25 * float(np.clip(tape.get("walk_amplitude", 0.0) + tape.get("shuttle_speed", 0.0) * 0.1, 0.0, 1.0)))
        return cls(
            mic_type=int(np.clip(m_type, 0, 7)),
            polar_pattern=int(np.clip(round(float(aud.get("mic_polar_pattern", 1.0))), 0, 4)),
            off_axis_deg=float(np.clip(aud.get("mic_off_axis_deg", 0.0), 0.0, 180.0)),
            sensitivity_db=float(np.clip(aud.get("mic_sensitivity_db", 0.0), -24.0, 24.0)),
            self_noise_level=float(np.clip(aud.get("mic_self_noise", 0.0), 0.0, 1.0)),
            preamp_gain_db=float(np.clip(aud.get("mic_preamp_gain_db", 0.0), -18.0, 30.0)),
            agc_amount=float(np.clip(aud.get("mic_agc_amount", 0.0), 0.0, 1.0)),
            agc_speed=float(np.clip(aud.get("mic_agc_speed", 0.5), 0.0, 1.0)),
            clip_mode=int(np.clip(round(float(aud.get("mic_clip_mode", 0.0))), 0, 2)),
            overdrive=float(np.clip(aud.get("mic_overdrive", 0.0), 0.0, 1.0)),
            proximity_effect=float(np.clip(aud.get("proximity_effect", 0.0), 0.0, 1.0)),
            plosive_pop=float(np.clip(aud.get("plosive_pop", 0.0), 0.0, 1.0)),
            chassis_vibration=vib_base,
        )

    def is_active(self) -> bool:
        return (
            self.mic_type > 0
            or self.off_axis_deg > 1.0
            or abs(self.sensitivity_db) > 0.1
            or self.self_noise_level > 1e-4
            or abs(self.preamp_gain_db) > 0.1
            or self.agc_amount > 1e-4
            or self.overdrive > 1e-4
            or self.proximity_effect > 1e-4
            or self.plosive_pop > 1e-4
            or self.chassis_vibration > 1e-4
        )


def apply_mic_preamp_frontend(
    sig: np.ndarray,
    spec: MicPreampFrontendSpec,
    sr: int = 44100,
    seed: int = 20240101,
    start_s: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not spec.is_active():
        return sig.astype(np.float32, copy=False), {"active": False, "mic_type": 0}

    out = np.asarray(sig, dtype=np.float32).copy()
    n_samples = len(out)
    rng = np.random.default_rng((int(seed) * 1664525 + int(round(start_s * 1000)) + 101) & 0xFFFFFFFF)
    t_arr = start_s + np.arange(n_samples, dtype=np.float64) / float(sr)

    m_type = int(spec.mic_type)
    capsule_names = {
        0: "direct_line", 1: "dynamic_sm58", 2: "condenser_u87", 3: "ribbon_rca44",
        4: "electret_webcam_pc", 5: "carbon_button_tel", 6: "camcorder_builtin_stereo", 7: "mobile_pinhole_mems",
    }

    intrinsic_sens_db = 0.0
    intrinsic_ein = 0.0
    if m_type == 1:
        out = _hp(out, 55.0, sr, order=2)
        out = _peaking_eq(out, 4200.0, q=1.2, gain_db=3.5, sr=sr)
        out = _peaking_eq(out, 220.0, q=1.0, gain_db=-1.5, sr=sr)
        out = _lp(out, 13500.0, sr, order=3)
        intrinsic_sens_db = -2.0
        intrinsic_ein = 0.0008
    elif m_type == 2:
        out = _hp(out, 25.0, sr, order=1)
        out = _peaking_eq(out, 10500.0, q=0.8, gain_db=2.8, sr=sr)
        out = _lp(out, 19500.0, sr, order=3)
        intrinsic_sens_db = +3.5
        intrinsic_ein = 0.0003
    elif m_type == 3:
        out = _hp(out, 35.0, sr, order=1)
        out = _peaking_eq(out, 280.0, q=0.9, gain_db=3.8, sr=sr)
        out = _lp(out, 9800.0, sr, order=3)
        intrinsic_sens_db = -4.5
        intrinsic_ein = 0.0014
    elif m_type == 4:
        out = _hp(out, 215.0, sr, order=3)
        out = _peaking_eq(out, 3800.0, q=2.4, gain_db=7.5, sr=sr)
        out = _lp(out, 9000.0, sr, order=4)
        intrinsic_sens_db = +1.5
        intrinsic_ein = 0.0065
    elif m_type == 5:
        out = _bp(out, 340.0, 3200.0, sr, order=4)
        out = _peaking_eq(out, 1150.0, q=1.8, gain_db=5.5, sr=sr)
        out = (out + 0.34 * (out * out) - 0.08).astype(np.float32)
        out = _hp(out, 300.0, sr, order=2)
        intrinsic_sens_db = +1.0
        intrinsic_ein = 0.0110
    elif m_type == 6:
        out = _hp(out, 135.0, sr, order=2)
        out = _peaking_eq(out, 2650.0, q=1.8, gain_db=4.2, sr=sr)
        out = _lp(out, 12500.0, sr, order=3)
        mid = 0.5 * (out[:, 0] + out[:, 1])
        side = 0.5 * (out[:, 0] - out[:, 1]) * 0.55
        out = np.stack([mid + side, mid - side], axis=-1)
        intrinsic_sens_db = 0.0
        intrinsic_ein = 0.0038
    elif m_type == 7:
        mono = 0.5 * (out[:, 0] + out[:, 1])
        mono_2d = np.stack([mono, mono], axis=-1)
        out = _hp(mono_2d, 270.0, sr, order=3)
        out = _peaking_eq(out, 3150.0, q=2.8, gain_db=6.8, sr=sr)
        out = _lp(out, 7200.0, sr, order=4)
        intrinsic_sens_db = +1.0
        intrinsic_ein = 0.0080

    polar_alphas = {0: 1.0, 1: 0.5, 2: 0.37, 3: 0.0, 4: 0.22}
    alpha = polar_alphas.get(int(spec.polar_pattern), 0.5)
    theta_rad = np.deg2rad(float(spec.off_axis_deg))
    polar_gain = float(alpha + (1.0 - alpha) * np.cos(theta_rad))
    if int(spec.polar_pattern) == 4:
        sin_half = float(np.sin(0.5 * theta_rad))
        polar_gain *= float(max(0.08, 1.0 - 0.78 * (sin_half ** 1.2)))
    out = out * polar_gain

    if spec.off_axis_deg > 5.0 and int(spec.polar_pattern) > 0:
        sin_half_sq = float(np.sin(0.5 * theta_rad) ** 2)
        fc_offaxis = float(np.clip(18000.0 * (1.0 - 0.78 * sin_half_sq * (1.0 - 0.5 * alpha)), 1200.0, 18000.0))
        out = _lp(out, fc_offaxis, sr, order=2)

    prox = float(spec.proximity_effect)
    if prox > 1e-4 and int(spec.polar_pattern) > 0:
        dir_factor = float((1.0 - alpha) * max(0.2, abs(np.cos(theta_rad))))
        bass = _lp(out, 185.0, sr, order=2)
        out = out + (1.25 * prox * dir_factor) * bass

    pop_amt = float(spec.plosive_pop)
    if pop_amt > 1e-4:
        env_fast = gaussian_filter1d(np.mean(np.abs(out), axis=1), sigma=max(2, int(0.004 * sr)))
        d_env = np.diff(env_fast, prepend=env_fast[0]) * float(sr)
        pop_triggers = np.where((d_env > 14.0) & (env_fast > 0.06))[0]
        if len(pop_triggers) > 0:
            pop_wave = np.zeros(n_samples, dtype=np.float32)
            t_idx = 0
            while t_idx < len(pop_triggers):
                pos = int(pop_triggers[t_idx])
                dur = int(rng.uniform(0.060, 0.125) * sr)
                end_p = min(n_samples, pos + dur)
                t_pop = np.linspace(0.0, float(end_p - pos) / sr, end_p - pos, dtype=np.float32)
                freq_pop = float(rng.uniform(42.0, 66.0))
                decay = np.exp(-t_pop * float(rng.uniform(18.0, 30.0)))
                thump = pop_amt * float(rng.uniform(0.32, 0.68)) * np.sin(2.0 * np.pi * freq_pop * t_pop) * decay
                pop_wave[pos:end_p] += thump
                t_idx += max(1, int(0.11 * sr))
            out += pop_wave[:, None]

    vib = float(spec.chassis_vibration)
    if vib > 1e-4:
        raw_rumble = rng.standard_normal((n_samples, 2)).astype(np.float32)
        rumble_lp = _bp(raw_rumble, 24.0, 82.0, sr, order=2)
        grip_env = (0.35 + 0.65 * np.maximum(0.0, np.sin(2.0 * np.pi * 1.8 * t_arr + 0.4) * np.cos(2.0 * np.pi * 0.7 * t_arr)))[:, None].astype(np.float32)
        motor_whine = (
            0.012 * np.sin(2.0 * np.pi * 420.0 * t_arr + 0.2 * np.sin(2.0 * np.pi * 1.5 * t_arr))
            + 0.007 * np.sin(2.0 * np.pi * 840.0 * t_arr)
            + 0.004 * np.sin(2.0 * np.pi * 1260.0 * t_arr)
        )[:, None].astype(np.float32)
        out = out + vib * (0.095 * rumble_lp * grip_env + motor_whine)

    total_gain_db = intrinsic_sens_db + float(spec.sensitivity_db) + float(spec.preamp_gain_db)
    linear_gain = float(10.0 ** (total_gain_db / 20.0))
    out = out * linear_gain

    noise_amp = intrinsic_ein + 0.025 * float(spec.self_noise_level)
    if noise_amp > 1e-6:
        white_n = rng.standard_normal((n_samples, 2)).astype(np.float32)
        pink_lp = _lp(white_n, 450.0, sr, order=1)
        preamp_noise = (0.65 * white_n + 0.85 * pink_lp) * noise_amp * max(0.5, min(4.0, linear_gain * 0.7))
        out = out + preamp_noise

    agc_amt = float(spec.agc_amount)
    agc_gain_trace_db = 0.0
    if agc_amt > 1e-4:
        env_inst = np.max(np.abs(out), axis=1)
        sigma_atk = max(2, int((0.008 - 0.006 * spec.agc_speed) * sr))
        sigma_rel = max(20, int((0.38 - 0.22 * spec.agc_speed) * sr))
        env_atk = gaussian_filter1d(env_inst, sigma=sigma_atk)
        env_rel = gaussian_filter1d(env_inst, sigma=sigma_rel)
        env_det = np.maximum(env_atk * 0.85, env_rel)
        target_level = 0.34
        desired_gain = np.clip(target_level / np.maximum(0.035, env_det), 0.22, 4.2)
        applied_gain = (1.0 - agc_amt) + agc_amt * desired_gain
        out = out * applied_gain[:, None].astype(np.float32)
        agc_gain_trace_db = float(20.0 * np.log10(max(1e-4, float(np.max(applied_gain)) / max(1e-4, float(np.min(applied_gain))))))

    od = float(spec.overdrive)
    clipped_fraction = 0.0
    if od > 1e-4 or int(spec.clip_mode) > 0:
        drive = 1.0 + 4.2 * od
        driven = out * drive
        clipped_fraction = float(np.mean(np.abs(driven) > 0.88))
        if int(spec.clip_mode) == 0:
            asym = 0.16 * od
            out = (np.tanh(driven + asym * (driven * driven)) - np.tanh(asym)).astype(np.float32)
        elif int(spec.clip_mode) == 1:
            rail = max(0.28, 0.92 - 0.48 * od)
            out = (np.clip(driven, -rail, rail) / max(0.5, rail)).astype(np.float32) * 0.88
        else:
            rail = max(0.35, 0.95 - 0.45 * od)
            scaled = driven / rail
            over_mask = np.abs(scaled) > 1.0
            wrapped = np.where(over_mask, np.mod(scaled + 1.0, 2.0) - 1.0, scaled)
            out = (0.65 * np.clip(scaled, -1.0, 1.0) + 0.35 * wrapped).astype(np.float32) * 0.88

    meta = {
        "active": True,
        "mic_type": m_type,
        "capsule": capsule_names.get(m_type, "custom"),
        "polar_pattern": int(spec.polar_pattern),
        "off_axis_deg": round(float(spec.off_axis_deg), 1),
        "polar_gain": round(float(polar_gain), 4),
        "agc_swing_db": round(float(agc_gain_trace_db), 2),
        "clipped_fraction": round(float(clipped_fraction), 4),
    }
    return out.astype(np.float32), meta


# ===========================================================================
# SUBSYSTEM 02 · 旧手机扬声器与听筒 (Legacy Mobile Speaker & Earpiece)
# ===========================================================================
@dataclass
class MobileSpeakerSimSpec:
    mode: int = 0
    f0_hz: float = 680.0
    drive_spl: float = 0.35
    cavity_resonance: float = 0.45
    volume_limiter: float = 0.40

    @classmethod
    def from_params(cls, params: dict) -> "MobileSpeakerSimSpec":
        p = params or {}
        aud = p.get("audio") if isinstance(p.get("audio"), dict) else p
        mode = int(np.clip(round(float(aud.get("mobile_spk_mode", 0.0))), 0, 4))
        default_f0 = {0: 680.0, 1: 360.0, 2: 780.0, 3: 520.0, 4: 440.0}.get(mode, 680.0)
        raw_f0 = float(aud.get("mobile_spk_f0_hz", default_f0))
        if abs(raw_f0 - 680.0) < 1.0 and mode in (1, 2, 3, 4):
            f0 = default_f0
        else:
            f0 = raw_f0
        return cls(
            mode=mode,
            f0_hz=float(np.clip(f0, 200.0, 1600.0)),
            drive_spl=float(np.clip(aud.get("mobile_spk_drive", 0.35), 0.0, 1.0)),
            cavity_resonance=float(np.clip(aud.get("mobile_spk_cavity_res", 0.45), 0.0, 1.0)),
            volume_limiter=float(np.clip(aud.get("mobile_spk_limiter", 0.40), 0.0, 1.0)),
        )

    def is_active(self) -> bool:
        return self.mode > 0


def apply_mobile_speaker_sim(
    sig: np.ndarray,
    spec: MobileSpeakerSimSpec,
    sr: int = 44100,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not spec.is_active():
        return sig.astype(np.float32, copy=False), {"active": False, "mode": 0}

    out = np.asarray(sig, dtype=np.float32)
    mono = 0.5 * (out[:, 0] + out[:, 1])
    x = np.stack([mono, mono], axis=-1)

    mode = int(spec.mode)
    f0 = float(spec.f0_hz)
    cav = float(spec.cavity_resonance)
    drv = float(spec.drive_spl)
    lim = float(spec.volume_limiter)

    if mode == 1:
        x = _hp(x, max(240.0, f0 * 0.85), sr, order=3)
        x = _lp(x, 3550.0, sr, order=5)
        x = _peaking_eq(x, 1150.0, q=2.0, gain_db=4.5 * cav, sr=sr)
        x = _peaking_eq(x, 2450.0, q=2.5, gain_db=3.0 * cav, sr=sr)
    elif mode == 2:
        x = _hp(x, f0, sr, order=4)
        x = _peaking_eq(x, f0 * 1.08, q=3.2, gain_db=5.5, sr=sr)
        x = _peaking_eq(x, 2450.0, q=3.0, gain_db=8.0 * cav, sr=sr)
        x = _peaking_eq(x, 3850.0, q=3.4, gain_db=6.5 * cav, sr=sr)
        x = _lp(x, 6400.0, sr, order=4)
    elif mode == 3:
        x = _hp(x, f0, sr, order=3)
        x = _peaking_eq(x, f0 * 1.05, q=2.4, gain_db=4.2, sr=sr)
        x = _peaking_eq(x, 1850.0, q=2.2, gain_db=5.5 * cav, sr=sr)
        x = _peaking_eq(x, 4200.0, q=2.6, gain_db=5.0 * cav, sr=sr)
        x = _lp(x, 9200.0, sr, order=4)
    else:
        x = _hp(x, f0, sr, order=3)
        x = _peaking_eq(x, 980.0, q=2.5, gain_db=6.0 * cav, sr=sr)
        x = _peaking_eq(x, 2750.0, q=2.8, gain_db=5.5 * cav, sr=sr)
        x = _lp(x, 4400.0, sr, order=5)

    if cav > 0.05:
        d1 = max(1, int(round(0.00045 * sr)))
        d2 = max(2, int(round(0.00115 * sr)))
        if d2 < len(x):
            shell = np.zeros_like(x)
            shell[d1:] += x[:-d1] * (0.24 * cav)
            shell[d2:] -= x[:-d2] * (0.15 * cav)
            x = x + shell

    if drv > 1e-4:
        gain_spl = 1.0 + 3.6 * drv
        u = x * gain_spl
        asym = 0.22 * drv
        nl = u + asym * (u * u) - (0.18 * drv) * (u * u * u)
        x = (np.tanh(nl) / max(0.75, np.tanh(gain_spl))).astype(np.float32)
        x = _hp(x, max(200.0, f0 * 0.72), sr, order=2)

    if lim > 1e-4:
        env = gaussian_filter1d(np.max(np.abs(x), axis=1), sigma=max(2, int(0.004 * sr)))
        ceiling = max(0.25, 0.85 - 0.42 * lim)
        gr = np.where(env > ceiling, ceiling / np.maximum(1e-4, env), 1.0).astype(np.float32)
        makeup = 1.0 + 0.35 * lim
        x = np.clip(x * gr[:, None] * makeup, -0.95, 0.95).astype(np.float32)

    mode_names = {1: "earpiece_receiver", 2: "feature_phone_handsfree", 3: "early_smartphone_speaker", 4: "voicemail_speakerphone"}
    return x.astype(np.float32), {
        "active": True,
        "mode": mode,
        "preset_name": mode_names.get(mode, "speaker"),
        "f0_hz": round(f0, 1),
        "drive_spl": round(drv, 3),
    }


# ===========================================================================
# SUBSYSTEM 03 · 数字采样与音频格式转换 (Digital Sampling & Format Conversion)
# ===========================================================================
@dataclass
class DigitalSamplingSpec:
    enabled: bool = False
    sample_rate_hz: float = 44100.0
    bit_depth: float = 16.0
    resample_filter: int = 0
    dither_mode: int = 0
    channel_mode: int = 0
    adc_clip_level: float = 1.0
    clock_jitter: float = 0.0
    clock_drift_ppm: float = 0.0

    @classmethod
    def from_params(cls, params: dict) -> "DigitalSamplingSpec":
        p = params or {}
        aud = p.get("audio") if isinstance(p.get("audio"), dict) else p
        sr_hz = float(np.clip(aud.get("adc_sample_rate_hz", 44100.0), 4000.0, 48000.0))
        bits = float(np.clip(aud.get("adc_bit_depth", 16.0), 4.0, 24.0))
        r_filt = int(np.clip(round(float(aud.get("adc_resample_filter", 0.0))), 0, 2))
        d_mode = int(np.clip(round(float(aud.get("adc_dither_mode", 0.0))), 0, 3))
        ch_mode = int(np.clip(round(float(aud.get("adc_channel_mode", 0.0))), 0, 3))
        clip_lv = float(np.clip(aud.get("adc_clip_level", 1.0), 0.20, 1.0))
        jit = float(np.clip(aud.get("adc_clock_jitter", 0.0), 0.0, 1.0))
        drift = float(np.clip(aud.get("adc_clock_drift_ppm", 0.0), -2000.0, 2000.0))
        en = (
            float(aud.get("adc_enabled", 0.0)) > 0.5
            or sr_hz < 43000.0
            or bits < 15.5
            or ch_mode > 0
            or clip_lv < 0.98
            or jit > 1e-4
            or abs(drift) > 1.0
        )
        return cls(
            enabled=en,
            sample_rate_hz=sr_hz,
            bit_depth=bits,
            resample_filter=r_filt,
            dither_mode=d_mode,
            channel_mode=ch_mode,
            adc_clip_level=clip_lv,
            clock_jitter=jit,
            clock_drift_ppm=drift,
        )

    def is_active(self) -> bool:
        return bool(self.enabled)


def apply_digital_sampling_converter(
    sig: np.ndarray,
    spec: DigitalSamplingSpec,
    sr: int = 44100,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not spec.is_active():
        return sig.astype(np.float32, copy=False), {"active": False}

    out = np.asarray(sig, dtype=np.float32).copy()
    n_samples = len(out)
    rng = np.random.default_rng((int(seed) ^ 0x3C6EF372) & 0xFFFFFFFF)

    ch_mode = int(spec.channel_mode)
    if ch_mode == 1:
        mono = 0.5 * (out[:, 0] + out[:, 1])
        out = np.stack([mono, mono], axis=-1)
    elif ch_mode == 2:
        delayed_r = np.roll(out[:, 1], max(1, int(round(0.00035 * sr))))
        comb_mono = 0.55 * (out[:, 0] - 0.85 * delayed_r)
        out = np.stack([comb_mono, comb_mono], axis=-1)
    elif ch_mode == 3:
        out = np.stack([out[:, 0], out[:, 0]], axis=-1)

    drift_ppm = float(spec.clock_drift_ppm)
    jit = float(spec.clock_jitter)
    if abs(drift_ppm) > 0.5 or jit > 1e-4:
        base_idx = np.arange(n_samples, dtype=np.float64) * (1.0 + drift_ppm * 1e-6)
        if jit > 1e-4:
            white_j = rng.standard_normal(n_samples)
            wander_j = gaussian_filter1d(rng.standard_normal(n_samples), sigma=max(2, int(0.002 * sr)))
            dt_samples = jit * (0.85 * white_j + 1.45 * wander_j)
            base_idx = base_idx + dt_samples
        idx_c = np.clip(base_idx, 0.0, float(n_samples - 1.0001))
        i0 = np.floor(idx_c).astype(np.int32)
        i1 = np.minimum(i0 + 1, n_samples - 1)
        frac = (idx_c - i0)[:, None].astype(np.float32)
        out = out[i0] * (1.0 - frac) + out[i1] * frac

    target_sr = float(np.clip(spec.sample_rate_hz, 4000.0, float(sr)))
    r_filt = int(spec.resample_filter)
    alias_ratio = float(sr) / target_sr
    if alias_ratio > 1.01:
        nyq_target = 0.5 * target_sr
        if r_filt == 0:
            out = _lp(out, nyq_target * 0.92, sr, order=8)
        elif r_filt == 1:
            n_sub = max(2, int(round(n_samples / alias_ratio)))
            sub_pos = np.linspace(0.0, float(n_samples - 1), n_sub, dtype=np.float64)
            s0 = np.floor(sub_pos).astype(np.int32)
            s1 = np.minimum(s0 + 1, n_samples - 1)
            sf = (sub_pos - s0)[:, None].astype(np.float32)
            sub_sig = out[s0] * (1.0 - sf) + out[s1] * sf
            rec_pos = np.linspace(0.0, float(n_sub - 1), n_samples, dtype=np.float64)
            r0 = np.floor(rec_pos).astype(np.int32)
            r1 = np.minimum(r0 + 1, n_sub - 1)
            rf = (rec_pos - r0)[:, None].astype(np.float32)
            out = sub_sig[r0] * (1.0 - rf) + sub_sig[r1] * rf
        else:
            step_idx = np.floor(np.arange(n_samples, dtype=np.float64) / alias_ratio).astype(np.int32)
            sample_indices = np.clip(np.round(step_idx * alias_ratio).astype(np.int32), 0, n_samples - 1)
            out = out[sample_indices]

    clip_lv = float(spec.adc_clip_level)
    if clip_lv < 0.999:
        out = np.clip(out / clip_lv, -1.0, 1.0).astype(np.float32) * clip_lv

    bits = float(spec.bit_depth)
    d_mode = int(spec.dither_mode)
    q_err_rms = 0.0
    if bits < 23.5 or d_mode > 0:
        half_steps = float(max(4.0, 2.0 ** (bits - 1.0)))
        lsb = 1.0 / half_steps
        pre_q = out.copy()
        if d_mode == 0:
            out = (np.round(out * half_steps) / half_steps).astype(np.float32)
        elif d_mode == 1:
            dither = rng.uniform(-0.5 * lsb, 0.5 * lsb, size=out.shape).astype(np.float32)
            out = (np.round((out + dither) * half_steps) / half_steps).astype(np.float32)
        elif d_mode == 2:
            d1 = rng.uniform(-0.5 * lsb, 0.5 * lsb, size=out.shape).astype(np.float32)
            d2 = rng.uniform(-0.5 * lsb, 0.5 * lsb, size=out.shape).astype(np.float32)
            out = (np.round((out + d1 + d2) * half_steps) / half_steps).astype(np.float32)
        else:
            d_raw = (rng.uniform(-0.5 * lsb, 0.5 * lsb, size=out.shape)
                     + rng.uniform(-0.5 * lsb, 0.5 * lsb, size=out.shape)).astype(np.float32)
            d_hp = np.diff(d_raw, axis=0, prepend=d_raw[:1])
            out = (np.round((out + d_hp) * half_steps) / half_steps).astype(np.float32)
        q_err_rms = float(np.sqrt(np.mean((out - pre_q) ** 2)))

    return out.astype(np.float32), {
        "active": True,
        "sample_rate_hz": round(target_sr, 1),
        "bit_depth": round(bits, 1),
        "resample_filter": r_filt,
        "dither_mode": d_mode,
        "channel_mode": ch_mode,
        "quant_error_rms": round(q_err_rms, 6),
    }


# ===========================================================================
# SUBSYSTEM 04 · 感知音频编码 (Perceptual Audio Coding: MP3 / AAC / AMR / Speech)
# ===========================================================================
@dataclass
class PerceptualCodecSpec:
    codec_type: int = 0
    bitrate_kbps: float = 64.0
    pre_echo: float = 0.35
    birdies_amount: float = 0.40
    generations: int = 1

    @classmethod
    def from_params(cls, params: dict) -> "PerceptualCodecSpec":
        p = params or {}
        aud = p.get("audio") if isinstance(p.get("audio"), dict) else p
        c_type = int(np.clip(round(float(aud.get("pcodec_type", 0.0))), 0, 4))
        br = float(np.clip(aud.get("pcodec_bitrate_kbps", 64.0), 4.75, 320.0))
        pe = float(np.clip(aud.get("pcodec_pre_echo", 0.35), 0.0, 1.0))
        bd = float(np.clip(aud.get("pcodec_birdies", 0.40), 0.0, 1.0))
        gens = int(np.clip(round(float(aud.get("pcodec_generations", 1.0))), 1, 8))
        return cls(
            codec_type=c_type,
            bitrate_kbps=br,
            pre_echo=pe,
            birdies_amount=bd,
            generations=gens,
        )

    def is_active(self) -> bool:
        return self.codec_type > 0


def apply_perceptual_audio_codec(
    sig: np.ndarray,
    spec: PerceptualCodecSpec,
    sr: int = 44100,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not spec.is_active():
        return sig.astype(np.float32, copy=False), {"active": False, "codec_type": 0}

    out = np.asarray(sig, dtype=np.float32).copy()
    n_samples = len(out)
    c_type = int(spec.codec_type)
    br = float(spec.bitrate_kbps)
    gens = int(max(1, spec.generations))

    codec_names = {1: "mp3", 2: "aac_lc_sbr", 3: "amr_nb_3gpp", 4: "lowbitrate_speech"}

    if c_type == 3:
        eff_br = float(np.clip(br, 4.75, 12.2))
        fc_cut = 3450.0 + 40.0 * (eff_br - 4.75)
        mono = 0.5 * (out[:, 0] + out[:, 1])
        out = np.stack([mono, mono], axis=-1)
        out = _bp(out, 220.0, fc_cut, sr, order=4)
        win_len = 512
    elif c_type == 4:
        eff_br = float(np.clip(br, 8.0, 32.0))
        fc_cut = float(np.clip(2800.0 + 140.0 * eff_br, 3200.0, 7200.0))
        mono = 0.5 * (out[:, 0] + out[:, 1])
        out = np.stack([mono, mono], axis=-1)
        out = _bp(out, 160.0, fc_cut, sr, order=4)
        win_len = 512
    elif c_type == 1:
        eff_br = float(np.clip(br, 16.0, 320.0))
        fc_cut = float(np.clip(3600.0 + 102.0 * eff_br - 0.14 * (eff_br ** 2), 4500.0, 19500.0))
        win_len = 1024
        if eff_br < 96.0:
            js_mix = float(np.clip((96.0 - eff_br) / 80.0, 0.0, 0.85))
            mid = 0.5 * (out[:, 0] + out[:, 1])
            side = 0.5 * (out[:, 0] - out[:, 1]) * (1.0 - js_mix)
            out = np.stack([mid + side, mid - side], axis=-1)
    else:
        eff_br = float(np.clip(br, 16.0, 320.0))
        fc_cut = float(np.clip(4800.0 + 115.0 * eff_br - 0.15 * (eff_br ** 2), 5800.0, 20000.0))
        win_len = 512

    comp_stress = float(np.clip((128.0 / max(8.0, eff_br)) ** 0.72, 0.08, 3.2)) * (0.45 + 0.85 * float(spec.birdies_amount))

    spectral_hole_ratio_accum = 0.0
    total_frames_counted = 0

    for g_idx in range(gens):
        out = _lp(out, min(fc_cut * (0.95 ** g_idx), 0.47 * sr), sr, order=6)
        shift = (g_idx * 137) % (win_len // 2)
        if shift > 0 and shift < n_samples:
            padded = np.pad(out, ((shift, win_len), (0, 0)), mode="edge")
        else:
            padded = np.pad(out, ((0, win_len), (0, 0)), mode="edge")

        hop = win_len // 2
        win = np.hanning(win_len).astype(np.float32)[:, None]
        rec_buf = np.zeros_like(padded)
        w_norm = np.zeros((len(padded), 1), dtype=np.float32)

        freqs = np.fft.rfftfreq(win_len, 1.0 / float(sr))
        bark_edges = np.array([0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 24000], dtype=np.float64)
        bin_band = np.clip(np.digitize(freqs, bark_edges) - 1, 0, len(bark_edges) - 2)

        for p0 in range(0, len(padded) - win_len, hop):
            seg = padded[p0 : p0 + win_len] * win
            spec_f = np.fft.rfft(seg, axis=0)
            mag = np.abs(spec_f)

            global_peak = np.maximum(np.max(mag, axis=0, keepdims=True), 1e-6)
            hf_weight = (1.0 + 1.8 * np.clip((freqs - 1500.0) / 5500.0, 0.0, 1.5))[:, None]
            mask_thresh = global_peak * (0.014 * comp_stress) * hf_weight

            keep_mask = np.ones_like(mag)
            holes_this_frame = 0
            active_bands = 0
            for b_id in range(int(bin_band.max()) + 1):
                b_sel = (bin_band == b_id)
                if not np.any(b_sel):
                    continue
                active_bands += 1
                b_energy = np.mean(mag[b_sel], axis=0, keepdims=True)
                b_thr = np.mean(mask_thresh[b_sel], axis=0, keepdims=True)
                gate = np.clip((b_energy - 0.72 * b_thr) / np.maximum(0.35 * b_thr, 1e-6), 0.0, 1.0)
                keep_mask[b_sel] = gate
                if float(np.mean(gate)) < 0.25:
                    holes_this_frame += 1

            if active_bands > 0:
                spectral_hole_ratio_accum += holes_this_frame / float(active_bands)
                total_frames_counted += 1

            q_step = max(1e-4, 0.015 * comp_stress * float(np.mean(global_peak)))
            spec_masked = spec_f * keep_mask
            mag_m = np.abs(spec_masked)
            phase_m = np.angle(spec_masked)
            mag_q = np.round(mag_m / q_step) * q_step
            spec_q = mag_q * np.exp(1j * phase_m)

            rec_seg = np.fft.irfft(spec_q, n=win_len, axis=0).astype(np.float32)

            pe_amt = float(spec.pre_echo)
            if pe_amt > 1e-4:
                e_first = float(np.mean(seg[: win_len // 2] ** 2))
                e_second = float(np.mean(seg[win_len // 2 :] ** 2))
                if e_second > max(1e-5, e_first * 5.0):
                    pre_burst = rec_seg[win_len // 2 :][::-1] * (0.28 * pe_amt * min(1.5, comp_stress * 0.6))
                    rec_seg[: win_len // 2] += pre_burst

            rec_buf[p0 : p0 + win_len] += rec_seg * win
            w_norm[p0 : p0 + win_len] += win * win

        valid = w_norm[:, 0] > 1e-4
        rec_buf[valid] = rec_buf[valid] / w_norm[valid]
        out = rec_buf[shift : shift + n_samples]

        if c_type == 2 and eff_br <= 64.0:
            mid_band = _bp(out, 2400.0, 4800.0, sr, order=2)
            sbr_hf = _hp(np.abs(mid_band) * np.sign(np.roll(mid_band, 1, axis=0)), 5000.0, sr, order=2)
            out = out + 0.16 * sbr_hf
        elif c_type in (3, 4):
            celp_step = max(2, int(round(sr / 8000.0)))
            held = out[(np.arange(n_samples) // celp_step) * celp_step]
            out = 0.65 * out + 0.35 * held
            out = _peaking_eq(out, 1050.0, q=2.6, gain_db=3.2, sr=sr)
            out = _peaking_eq(out, 2350.0, q=3.0, gain_db=3.8, sr=sr)
            if c_type == 3:
                out = _lp(out, 3600.0, sr, order=5)

    avg_hole_ratio = spectral_hole_ratio_accum / max(1, total_frames_counted)
    return out.astype(np.float32), {
        "active": True,
        "codec_type": c_type,
        "codec_name": codec_names.get(c_type, "perceptual"),
        "bitrate_kbps": round(eff_br, 2),
        "cutoff_hz": round(fc_cut, 1),
        "generations": gens,
        "spectral_hole_ratio": round(float(avg_hole_ratio), 4),
    }


# ===========================================================================
# SUBSYSTEM 05 · 移动电话与网络语音 (Mobile Telephony & VoIP Real-Time Speech)
# ===========================================================================
@dataclass
class TelecomVoiceSpec:
    mode: int = 0
    wideband: int = 0
    vad_threshold: float = 0.25
    cng_level: float = 0.20
    packet_loss_pct: float = 0.0
    plc_mode: int = 1
    aec_residue: float = 0.0
    ns_strength: float = 0.0
    jitter_ms: float = 0.0
    latency_ms: float = 0.0

    @classmethod
    def from_params(cls, params: dict) -> "TelecomVoiceSpec":
        p = params or {}
        aud = p.get("audio") if isinstance(p.get("audio"), dict) else p
        mode = int(np.clip(round(float(aud.get("telecom_mode", 0.0))), 0, 3))
        wb = int(np.clip(round(float(aud.get("telecom_wideband", 0.0))), 0, 1))
        vad = float(np.clip(aud.get("telecom_vad_threshold", 0.25 if mode > 0 else 0.0), 0.0, 1.0))
        cng = float(np.clip(aud.get("telecom_cng_level", 0.20 if mode > 0 else 0.0), 0.0, 1.0))
        ploss = float(np.clip(aud.get("telecom_packet_loss_pct", 14.0 if mode == 3 else (3.5 if mode > 0 else 0.0)), 0.0, 60.0))
        plc = int(np.clip(round(float(aud.get("telecom_plc_mode", 1.0))), 0, 1))
        aec = float(np.clip(aud.get("telecom_aec_residue", 0.25 if mode in (2, 3) else 0.0), 0.0, 1.0))
        ns = float(np.clip(aud.get("telecom_ns_strength", 0.35 if mode > 0 else 0.0), 0.0, 1.0))
        jit = float(np.clip(aud.get("telecom_jitter_ms", 35.0 if mode == 3 else (12.0 if mode == 2 else 0.0)), 0.0, 150.0))
        lat = float(np.clip(aud.get("telecom_latency_ms", 120.0 if mode == 3 else (45.0 if mode > 0 else 0.0)), 0.0, 600.0))
        return cls(
            mode=mode,
            wideband=wb,
            vad_threshold=vad,
            cng_level=cng,
            packet_loss_pct=ploss,
            plc_mode=plc,
            aec_residue=aec,
            ns_strength=ns,
            jitter_ms=jit,
            latency_ms=lat,
        )

    def is_active(self) -> bool:
        return (
            self.mode > 0
            or self.packet_loss_pct > 0.1
            or self.aec_residue > 1e-4
            or self.ns_strength > 1e-4
            or self.jitter_ms > 0.5
        )


def apply_telecom_voice_channel(
    sig: np.ndarray,
    spec: TelecomVoiceSpec,
    sr: int = 44100,
    seed: int = 20240101,
    start_s: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not spec.is_active():
        return sig.astype(np.float32, copy=False), {"active": False, "mode": 0}

    out = np.asarray(sig, dtype=np.float32).copy()
    n_samples = len(out)
    rng = np.random.default_rng((int(seed) * 1103515245 + int(round(start_s * 1000)) + 505) & 0xFFFFFFFF)
    t_arr = start_s + np.arange(n_samples, dtype=np.float64) / float(sr)

    mono = 0.5 * (out[:, 0] + out[:, 1])

    if int(spec.wideband) == 0:
        mono_col = _bp(mono[:, None], 300.0, 3400.0, sr, order=4)
    else:
        mono_col = _bp(mono[:, None], 55.0, 7000.0, sr, order=4)
    mono = mono_col[:, 0]

    ns_amt = float(spec.ns_strength)
    if ns_amt > 1e-4:
        win_len = 256
        hop = 128
        win = np.hanning(win_len).astype(np.float32)
        ns_out = np.zeros(n_samples + win_len, dtype=np.float32)
        w_norm = np.zeros(n_samples + win_len, dtype=np.float32)
        padded = np.pad(mono, (0, win_len), mode="edge")
        for p0 in range(0, n_samples, hop):
            seg = padded[p0 : p0 + win_len] * win
            sp = np.fft.rfft(seg)
            mag = np.abs(sp)
            floor_est = np.percentile(mag, 35.0) * (1.0 + 2.2 * ns_amt)
            gain_ns = np.maximum(0.06 * (1.0 - ns_amt), (mag - floor_est) / np.maximum(mag, 1e-6))
            rec = np.fft.irfft(sp * gain_ns, n=win_len).astype(np.float32)
            ns_out[p0 : p0 + win_len] += rec * win
            w_norm[p0 : p0 + win_len] += win * win
        valid = w_norm[:n_samples] > 1e-4
        mono[valid] = ns_out[:n_samples][valid] / w_norm[:n_samples][valid]

    aec_amt = float(spec.aec_residue)
    if aec_amt > 1e-4:
        echo_delay = int(round(0.145 * sr))
        if echo_delay < n_samples:
            far_echo = np.zeros_like(mono)
            far_echo[echo_delay:] = mono[:-echo_delay]
            far_echo = _bp(far_echo[:, None], 400.0, 2600.0, sr, order=2)[:, 0]
            leak_mask = (np.abs(far_echo) > 0.08).astype(np.float32)
            duck = 1.0 - 0.45 * aec_amt * leak_mask
            mono = mono * duck + far_echo * (0.28 * aec_amt * leak_mask)

    vad_thr = float(spec.vad_threshold)
    cng_lv = float(spec.cng_level)
    dtx_frames = 0
    packet_samples = max(32, int(round(0.020 * sr)))
    total_packets = max(1, (n_samples + packet_samples - 1) // packet_samples)

    if vad_thr > 1e-4 or cng_lv > 1e-4:
        cng_raw = rng.standard_normal(n_samples).astype(np.float32)
        cng_shaped = _bp(cng_raw[:, None], 320.0, 2400.0, sr, order=2)[:, 0] * (0.018 * cng_lv)
        rms_global = float(np.sqrt(np.mean(mono ** 2)) + 1e-5)
        abs_thr = rms_global * (0.15 + 0.95 * vad_thr)
        for pk in range(total_packets):
            i0 = pk * packet_samples
            i1 = min(n_samples, i0 + packet_samples)
            pkt_rms = float(np.sqrt(np.mean(mono[i0:i1] ** 2)))
            if pkt_rms < abs_thr:
                mono[i0:i1] = cng_shaped[i0:i1]
                dtx_frames += 1

    ploss_pct = float(spec.packet_loss_pct)
    lost_packets = 0
    if ploss_pct > 0.05:
        p_loss = ploss_pct / 100.0
        in_burst = False
        for pk in range(total_packets):
            i0 = pk * packet_samples
            i1 = min(n_samples, i0 + packet_samples)
            if in_burst:
                in_burst = bool(rng.random() < min(0.75, 0.35 + p_loss))
            else:
                in_burst = bool(rng.random() < p_loss * 0.75)
            if in_burst:
                lost_packets += 1
                if int(spec.plc_mode) == 0 or i0 < packet_samples:
                    mono[i0:i1] = 0.0
                else:
                    pitch_len = max(24, int(round(0.0085 * sr)))
                    template = mono[max(0, i0 - pitch_len) : i0]
                    if len(template) > 0:
                        reps = int(np.ceil((i1 - i0) / float(len(template))))
                        concealed = np.tile(template, reps)[: i1 - i0] * 0.86
                        mono[i0:i1] = concealed

    if int(spec.mode) == 1:
        tdma_buzz = (
            0.006 * np.sin(2.0 * np.pi * 216.7 * t_arr)
            + 0.004 * np.sin(2.0 * np.pi * 433.3 * t_arr)
            + 0.003 * np.sin(2.0 * np.pi * 866.7 * t_arr)
        ) * (0.4 + 0.6 * (np.sin(2.0 * np.pi * 2.2 * t_arr) > 0.2))
        mono = mono + tdma_buzz.astype(np.float32)

    jit_ms = float(spec.jitter_ms)
    if jit_ms > 0.5:
        warp_depth = min(0.075, (jit_ms / 1000.0) * 1.4)
        speed_mod = 1.0 + warp_depth * np.sin(2.0 * np.pi * 1.6 * t_arr + 0.5 * np.cos(2.0 * np.pi * 0.7 * t_arr))
        warped_idx = np.clip(np.cumsum(speed_mod) - speed_mod[0], 0.0, float(n_samples - 1.001))
        w0 = np.floor(warped_idx).astype(np.int32)
        w1 = np.minimum(w0 + 1, n_samples - 1)
        wf = (warped_idx - w0).astype(np.float32)
        mono = mono[w0] * (1.0 - wf) + mono[w1] * wf

    out = np.stack([mono, mono], axis=-1).astype(np.float32)
    return out, {
        "active": True,
        "mode": int(spec.mode),
        "wideband": bool(spec.wideband),
        "dtx_suppressed_packets": int(dtx_frames),
        "lost_packets": int(lost_packets),
        "total_packets": int(total_packets),
        "plc_mode": int(spec.plc_mode),
        "latency_ms": round(float(spec.latency_ms), 1),
    }


# ===========================================================================
# SUBSYSTEM 06 · 声学空间与麦克风位置 (Acoustic Space & Microphone Placement)
# ===========================================================================
@dataclass
class AcousticSpaceSpec:
    preset: int = 0
    room_length_m: float = 4.2
    room_width_m: float = 3.4
    room_height_m: float = 2.7
    rt60_s: float = 0.35
    wall_absorption: float = 0.45
    standing_wave_modes: float = 0.40
    mic_distance_m: float = 0.45
    occlusion: float = 0.0
    ambient_noise: float = 0.15

    @classmethod
    def from_params(cls, params: dict) -> "AcousticSpaceSpec":
        p = params or {}
        aud = p.get("audio") if isinstance(p.get("audio"), dict) else p
        preset = int(np.clip(round(float(aud.get("space_preset", 0.0))), 0, 4))
        legacy_rev = float(aud.get("studio_room_reverb", 0.0))
        if preset == 0 and legacy_rev > 1e-4:
            preset = 1

        geom_defaults = {
            0: (4.2, 3.4, 2.7, 0.30, 0.50, 0.30, 0.35),
            1: (3.4, 2.8, 2.6, 0.28, 0.55, 0.55, 0.30),
            2: (12.5, 1.7, 3.0, 1.35, 0.12, 0.65, 1.80),
            3: (9.6, 7.2, 3.4, 0.82, 0.32, 0.40, 2.60),
            4: (25.0, 25.0, 15.0, 0.18, 0.88, 0.0, 1.60),
        }
        d_lx, d_ly, d_lz, d_rt60, d_abs, d_modes, d_dist = geom_defaults.get(preset, geom_defaults[0])

        raw_rt60 = float(aud.get("space_rt60_s", -1.0))
        if raw_rt60 < 0.0:
            rt60 = d_rt60 if preset > 0 else max(0.25, legacy_rev * 0.8)
        else:
            rt60 = raw_rt60

        raw_dist = float(aud.get("space_mic_distance_m", -1.0))
        dist = d_dist if raw_dist < 0.0 else raw_dist

        raw_abs = float(aud.get("space_wall_absorption", -1.0))
        w_abs = d_abs if raw_abs < 0.0 else raw_abs

        raw_modes = float(aud.get("space_standing_waves", -1.0))
        sw_modes = d_modes if raw_modes < 0.0 else raw_modes

        return cls(
            preset=preset,
            room_length_m=float(np.clip(aud.get("space_room_length_m", d_lx), 1.2, 40.0)),
            room_width_m=float(np.clip(aud.get("space_room_width_m", d_ly), 1.0, 40.0)),
            room_height_m=float(np.clip(aud.get("space_room_height_m", d_lz), 1.8, 20.0)),
            rt60_s=float(np.clip(rt60, 0.05, 4.0)),
            wall_absorption=float(np.clip(w_abs, 0.02, 0.98)),
            standing_wave_modes=float(np.clip(sw_modes, 0.0, 1.0)),
            mic_distance_m=float(np.clip(dist, 0.05, 12.0)),
            occlusion=float(np.clip(aud.get("space_occlusion", 0.0), 0.0, 1.0)),
            ambient_noise=float(np.clip(aud.get("space_ambient_noise", 0.18 if preset > 0 else 0.0), 0.0, 1.0)),
        )

    def is_active(self) -> bool:
        return (
            self.preset > 0
            or self.occlusion > 1e-4
            or self.ambient_noise > 1e-4
            or self.mic_distance_m > 0.60
        )


def apply_acoustic_space_simulator(
    sig: np.ndarray,
    spec: AcousticSpaceSpec,
    sr: int = 44100,
    seed: int = 20240101,
    start_s: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not spec.is_active():
        return sig.astype(np.float32, copy=False), {"active": False, "preset": 0}

    out = np.asarray(sig, dtype=np.float32).copy()
    n_samples = len(out)
    rng = np.random.default_rng((int(seed) * 22695477 + int(round(start_s * 1000)) + 606) & 0xFFFFFFFF)
    t_arr = start_s + np.arange(n_samples, dtype=np.float64) / float(sr)

    c_sound = 343.0
    lx, ly, lz = float(spec.room_length_m), float(spec.room_width_m), float(spec.room_height_m)
    dist = float(spec.mic_distance_m)
    w_abs = float(spec.wall_absorption)
    rt60 = float(spec.rt60_s)
    preset = int(spec.preset)

    vol_m3 = max(4.0, lx * ly * lz)
    d_crit = max(0.35, 0.057 * np.sqrt(vol_m3 / max(0.1, rt60)))
    direct_gain = float(np.clip(1.0 / (1.0 + (dist / max(0.35, d_crit)) ** 0.85), 0.18, 1.0))
    reverb_send = float(np.clip(1.0 - direct_gain + 0.22 * min(1.5, rt60), 0.05, 0.92))

    if dist > 0.5:
        fc_air = float(np.clip(19000.0 * np.exp(-0.085 * (dist - 0.4)), 2800.0, 19500.0))
        direct_sig = _lp(out, fc_air, sr, order=2) * direct_gain
    else:
        direct_sig = out * direct_gain

    modal_freqs_hz: list[float] = []
    if preset != 4 and spec.standing_wave_modes > 1e-4:
        f_x1 = c_sound / (2.0 * lx)
        f_y1 = c_sound / (2.0 * ly)
        f_z1 = c_sound / (2.0 * lz)
        modes = [f_x1, f_y1, f_z1, 2.0 * f_x1, 2.0 * f_y1, np.sqrt(f_x1 ** 2 + f_y1 ** 2)]
        modal_freqs_hz = [round(float(f), 1) for f in sorted(modes) if 32.0 <= f <= 380.0]
        sw_gain_db = (6.5 * float(spec.standing_wave_modes)) * (1.15 - 0.55 * w_abs)
        sw_q = 4.5 + 6.0 * (1.0 - w_abs)
        for fm in modal_freqs_hz[:4]:
            direct_sig = _peaking_eq(direct_sig, fm, q=sw_q, gain_db=sw_gain_db, sr=sr)

    er_accum = np.zeros_like(out)
    if preset == 4:
        taps_ms = [4.2, 52.0, 88.5]
        tap_gains = [0.22, 0.18, 0.11]
    elif preset == 2:
        tau_flutter_ms = (2.0 * ly / c_sound) * 1000.0
        taps_ms = [tau_flutter_ms * k for k in range(1, 9)]
        refl_coeff = float(np.clip(1.0 - 0.55 * w_abs, 0.45, 0.90))
        tap_gains = [0.42 * (refl_coeff ** k) for k in range(1, 9)]
    else:
        path_deltas_m = [
            max(0.3, 2.0 * (0.35 * ly)),
            max(0.4, 2.0 * (0.65 * ly)),
            max(0.5, 2.0 * (0.40 * lz)),
            max(0.6, 2.0 * (0.60 * lz)),
            max(0.8, 2.0 * (0.35 * lx)),
            max(1.1, 2.0 * (0.65 * lx)),
        ]
        taps_ms = [(dm / c_sound) * 1000.0 for dm in path_deltas_m]
        refl_coeff = float(np.clip(1.0 - 0.70 * w_abs, 0.20, 0.88))
        tap_gains = [refl_coeff / (1.0 + 0.35 * dm) for dm in path_deltas_m]

    for idx_t, (d_ms, g_t) in enumerate(zip(taps_ms, tap_gains)):
        d_samp = int(round((d_ms / 1000.0) * sr))
        if 1 <= d_samp < n_samples:
            if idx_t % 2 == 0:
                er_accum[d_samp:, 0] += out[:-d_samp, 0] * g_t
                er_accum[d_samp:, 1] += out[:-d_samp, 1] * (g_t * 0.72)
            else:
                er_accum[d_samp:, 0] += out[:-d_samp, 0] * (g_t * 0.72)
                er_accum[d_samp:, 1] += out[:-d_samp, 1] * g_t

    fc_wall = float(np.clip(14000.0 * (1.0 - 0.78 * w_abs), 1400.0, 14000.0))
    er_accum = _lp(er_accum, fc_wall, sr, order=2)

    tail_accum = np.zeros_like(out)
    if rt60 > 0.08 and preset > 0:
        max_tail_s = min(0.55, rt60 * 0.75)
        n_velvet = 18
        v_delays_s = np.linspace(0.018, max_tail_s, n_velvet)
        for v_idx, vd_s in enumerate(v_delays_s):
            d_s = int(round(vd_s * sr))
            if 1 <= d_s < n_samples:
                env_decay = float(10.0 ** (-3.0 * vd_s / max(0.08, rt60))) / np.sqrt(n_velvet)
                pol = 1.0 if (v_idx % 2 == 0) else -1.0
                swap = 1 if (v_idx % 3 == 0) else 0
                tail_accum[d_s:, 0] += pol * out[:-d_s, swap] * env_decay
                tail_accum[d_s:, 1] -= pol * out[:-d_s, 1 - swap] * env_decay
        tail_accum = _lp(tail_accum, max(1100.0, fc_wall * 0.65), sr, order=3)
        tail_accum = _hp(tail_accum, 95.0, sr, order=2)

    wet_sig = er_accum * (0.45 * reverb_send) + tail_accum * (0.55 * reverb_send)
    out = direct_sig + wet_sig

    occ = float(spec.occlusion)
    if occ > 1e-4:
        fc_occ = float(np.clip(16000.0 * ((1.0 - 0.88 * occ) ** 2.2), 320.0, 16000.0))
        out = _lp(out, fc_occ, sr, order=4)
        out = _peaking_eq(out, 280.0, q=1.6, gain_db=4.5 * occ, sr=sr) * (1.0 - 0.35 * occ)

    amb = float(spec.ambient_noise)
    if amb > 1e-4:
        raw_amb = rng.standard_normal((n_samples, 2)).astype(np.float32)
        if preset == 1:
            fan_hum = (0.008 * np.sin(2.0 * np.pi * 120.0 * t_arr) + 0.005 * np.sin(2.0 * np.pi * 238.0 * t_arr))[:, None].astype(np.float32)
            air_bed = _bp(raw_amb, 180.0, 2200.0, sr, order=2) * 0.012
            out = out + amb * (fan_hum + air_bed)
        elif preset == 2:
            ballast = (0.010 * np.sin(2.0 * np.pi * 100.0 * t_arr) + 0.006 * np.sin(2.0 * np.pi * 300.0 * t_arr))[:, None].astype(np.float32)
            hall_bed = _bp(raw_amb, 220.0, 1600.0, sr, order=2) * 0.016
            out = out + amb * (ballast + hall_bed)
        elif preset == 3:
            hvac = _lp(raw_amb, 220.0, sr, order=2) * 0.022
            murmur = _bp(raw_amb, 380.0, 1750.0, sr, order=2) * (0.015 * (0.6 + 0.4 * np.sin(2.0 * np.pi * 0.9 * t_arr))[:, None].astype(np.float32))
            out = out + amb * (hvac + murmur)
        elif preset == 4:
            wind_mod = np.maximum(0.0, np.sin(2.0 * np.pi * 0.85 * t_arr + 0.3) * np.cos(2.0 * np.pi * 0.37 * t_arr))[:, None].astype(np.float32)
            wind_rumble = _lp(raw_amb, 130.0, sr, order=3) * (0.045 * wind_mod)
            traffic = _bp(raw_amb, 150.0, 1200.0, sr, order=2) * 0.016
            out = out + amb * (wind_rumble + traffic)
        else:
            out = out + amb * (_bp(raw_amb, 120.0, 3000.0, sr, order=2) * 0.014)

    preset_names = {0: "direct", 1: "bedroom_desk", 2: "tiled_corridor", 3: "classroom_hall", 4: "outdoor_street"}
    return out.astype(np.float32), {
        "active": True,
        "preset": preset,
        "preset_name": preset_names.get(preset, "custom"),
        "room_dims_m": [round(lx, 2), round(ly, 2), round(lz, 2)],
        "rt60_s": round(rt60, 3),
        "mic_distance_m": round(dist, 2),
        "direct_to_reverb_ratio": round(float(direct_gain / max(1e-4, reverb_send)), 3),
        "axial_modes_hz": modal_freqs_hz[:4],
    }


# ===========================================================================
# SUBSYSTEM 07 · 播放器与系统音频链 (Media Player & OS System Audio Chain)
# ===========================================================================
@dataclass
class SystemAudioChainSpec:
    preset: int = 0
    mixer_gain: float = 1.0
    src_mode: int = 0
    channel_map: int = 0
    eq_preset: int = 0
    spatial_3d: float = 0.0
    buffer_underrun: float = 0.0
    device_switch_pop: float = 0.0
    mixer_clip: float = 0.0

    @classmethod
    def from_params(cls, params: dict) -> "SystemAudioChainSpec":
        p = params or {}
        aud = p.get("audio") if isinstance(p.get("audio"), dict) else p
        pr = int(np.clip(round(float(aud.get("sys_audio_preset", 0.0))), 0, 4))
        src_m = int(np.clip(round(float(aud.get("sys_src_mode", 1.0 if pr == 1 else (2.0 if pr == 2 else 0.0)))), 0, 2))
        ch_m = int(np.clip(round(float(aud.get("sys_channel_map", 2.0 if pr == 4 else 0.0))), 0, 3))
        eq_p = int(np.clip(round(float(aud.get("sys_eq_preset", 3.0 if pr == 3 else 0.0))), 0, 3))
        sp_3d = float(np.clip(aud.get("sys_spatial_3d", 0.45 if pr == 3 else 0.0), 0.0, 1.0))
        und = float(np.clip(aud.get("sys_buffer_underrun", 0.25 if pr == 2 else 0.0), 0.0, 1.0))
        sw_pop = float(np.clip(aud.get("sys_device_switch_pop", 0.30 if pr == 4 else 0.0), 0.0, 1.0))
        m_clip = float(np.clip(aud.get("sys_mixer_clip", 0.25 if pr in (1, 2, 3) else 0.0), 0.0, 1.0))
        m_gain = float(np.clip(aud.get("sys_mixer_gain", 1.0), 0.0, 2.5))
        return cls(
            preset=pr,
            mixer_gain=m_gain,
            src_mode=src_m,
            channel_map=ch_m,
            eq_preset=eq_p,
            spatial_3d=sp_3d,
            buffer_underrun=und,
            device_switch_pop=sw_pop,
            mixer_clip=m_clip,
        )

    def is_active(self) -> bool:
        return (
            self.preset > 0
            or abs(self.mixer_gain - 1.0) > 0.01
            or self.src_mode > 0
            or self.channel_map > 0
            or self.eq_preset > 0
            or self.spatial_3d > 1e-4
            or self.buffer_underrun > 1e-4
            or self.device_switch_pop > 1e-4
            or self.mixer_clip > 1e-4
        )


def apply_system_audio_chain(
    sig: np.ndarray,
    spec: SystemAudioChainSpec,
    sr: int = 44100,
    seed: int = 20240101,
    start_s: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not spec.is_active():
        return sig.astype(np.float32, copy=False), {"active": False, "preset": 0}

    out = np.asarray(sig, dtype=np.float32).copy()
    n_samples = len(out)
    rng = np.random.default_rng((int(seed) * 134775813 + int(round(start_s * 1000)) + 707) & 0xFFFFFFFF)
    t_arr = start_s + np.arange(n_samples, dtype=np.float64) / float(sr)

    if int(spec.src_mode) == 1:
        ratio = 48000.0 / 44100.0
        n_mid = max(4, int(round(n_samples * ratio)))
        p_up = np.linspace(0.0, float(n_samples - 1), n_mid, dtype=np.float64)
        i0 = np.floor(p_up).astype(np.int32)
        i1 = np.minimum(i0 + 1, n_samples - 1)
        f_up = (p_up - i0)[:, None].astype(np.float32)
        up_sig = out[i0] * (1.0 - f_up) + out[i1] * f_up
        p_dn = np.linspace(0.0, float(n_mid - 1), n_samples, dtype=np.float64)
        j0 = np.floor(p_dn).astype(np.int32)
        j1 = np.minimum(j0 + 1, n_mid - 1)
        f_dn = (p_dn - j0)[:, None].astype(np.float32)
        out = up_sig[j0] * (1.0 - f_dn) + up_sig[j1] * f_dn
        out += (0.0018 * np.sin(2.0 * np.pi * 1000.0 * t_arr) + 0.0012 * np.sin(2.0 * np.pi * 8000.0 * t_arr))[:, None].astype(np.float32)
    elif int(spec.src_mode) == 2:
        step = max(2, int(round(sr / 22050.0)))
        idx_hold = (np.arange(n_samples) // step) * step
        out = 0.65 * out[idx_hold] + 0.35 * _lp(out, 9800.0, sr, order=3)

    eq_p = int(spec.eq_preset)
    if eq_p == 1:
        out = _peaking_eq(out, 110.0, q=1.1, gain_db=5.5, sr=sr)
        out = _peaking_eq(out, 900.0, q=1.2, gain_db=-3.5, sr=sr)
        out = _peaking_eq(out, 7500.0, q=1.2, gain_db=5.0, sr=sr)
    elif eq_p == 2:
        out = _peaking_eq(out, 100.0, q=1.2, gain_db=-2.5, sr=sr)
        out = _peaking_eq(out, 2200.0, q=1.1, gain_db=5.0, sr=sr)
    elif eq_p == 3:
        out = _hp(out, 95.0, sr, order=2)
        out = _peaking_eq(out, 145.0, q=2.2, gain_db=7.0, sr=sr)
        out = _peaking_eq(out, 3600.0, q=1.6, gain_db=3.2, sr=sr)
        out = _lp(out, 12800.0, sr, order=3)

    sp_3d = float(spec.spatial_3d)
    if sp_3d > 1e-4:
        mid = 0.5 * (out[:, 0] + out[:, 1])
        side = 0.5 * (out[:, 0] - out[:, 1])
        side_delayed = np.roll(side, max(1, int(round(0.0006 * sr))))
        side_wide = side * (1.0 + 1.15 * sp_3d) + 0.45 * sp_3d * side_delayed
        out = np.stack([mid + side_wide, mid - side_wide], axis=-1).astype(np.float32)

    ch_m = int(spec.channel_map)
    if ch_m == 1:
        out = out[:, ::-1].copy()
    elif ch_m == 2:
        diff_lr = 0.65 * (out[:, 0] - 0.88 * out[:, 1])
        out = np.stack([diff_lr, -diff_lr * 0.92], axis=-1).astype(np.float32)
    elif ch_m == 3:
        out = np.stack([out[:, 0], out[:, 0]], axis=-1)

    und = float(spec.buffer_underrun)
    underrun_events = 0
    if und > 1e-4:
        buf_len = max(128, int(round(0.032 * sr)))
        for b0 in range(buf_len, n_samples - buf_len * 2, buf_len * 4):
            if rng.random() < und * 0.55:
                underrun_events += 1
                out[b0 : b0 + buf_len] = out[b0 - buf_len : b0]
                out[b0 + buf_len : b0 + 2 * buf_len] = out[b0 - buf_len : b0]

    sw_pop = float(spec.device_switch_pop)
    if sw_pop > 1e-4:
        sw_pos = int(0.35 * n_samples)
        mute_len = min(n_samples - sw_pos, int(0.045 * sr))
        if mute_len > 8:
            out[sw_pos : sw_pos + mute_len] *= 0.04
            t_pop = np.arange(mute_len, dtype=np.float32) / float(sr)
            pop_pulse = sw_pop * 0.65 * np.exp(-t_pop / 0.0045) * np.cos(2.0 * np.pi * 180.0 * t_pop)
            out[sw_pos : sw_pos + mute_len] += pop_pulse[:, None]

    out = out * float(spec.mixer_gain)
    m_clip = float(spec.mixer_clip)
    if m_clip > 1e-4:
        ceil = max(0.30, 0.95 - 0.55 * m_clip)
        out = np.clip(out, -ceil, ceil) / ceil * 0.92

    return out.astype(np.float32), {
        "active": True,
        "preset": int(spec.preset),
        "src_mode": int(spec.src_mode),
        "channel_map": ch_m,
        "eq_preset": eq_p,
        "underrun_events": int(underrun_events),
    }


# ===========================================================================
# SUBSYSTEM 08 · 铃声、MIDI 与移动音源 (Ringtones, MIDI & Mobile Sound Engines)
# ===========================================================================
@dataclass
class MobileRingtoneSynthSpec:
    mode: int = 0
    polyphony: int = 4
    tick_quant_ms: float = 25.0
    env_steps: int = 16
    wavetable_bits: int = 8
    mix: float = 1.0

    @classmethod
    def from_params(cls, params: dict) -> "MobileRingtoneSynthSpec":
        p = params or {}
        aud = p.get("audio") if isinstance(p.get("audio"), dict) else p
        mode = int(np.clip(round(float(aud.get("ringtone_mode", 0.0))), 0, 4))
        default_poly = {0: 16, 1: 1, 2: 4, 3: 16, 4: 4}.get(mode, 4)
        raw_poly = int(round(float(aud.get("ringtone_polyphony", default_poly))))
        if mode == 1:
            poly = 1
        else:
            poly = int(np.clip(raw_poly, 1, 64))
        return cls(
            mode=mode,
            polyphony=poly,
            tick_quant_ms=float(np.clip(aud.get("ringtone_tick_quant_ms", 25.0), 0.0, 150.0)),
            env_steps=int(np.clip(round(float(aud.get("ringtone_env_steps", 16.0))), 2, 128)),
            wavetable_bits=int(np.clip(round(float(aud.get("ringtone_wavetable_bits", 8.0))), 4, 16)),
            mix=float(np.clip(aud.get("ringtone_mix", 1.0), 0.0, 1.0)),
        )

    def is_active(self) -> bool:
        return self.mode > 0 and self.mix > 1e-4


def apply_mobile_ringtone_synth(
    sig: np.ndarray,
    spec: MobileRingtoneSynthSpec,
    sr: int = 44100,
    seed: int = 20240101,
    start_s: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not spec.is_active():
        return sig.astype(np.float32, copy=False), {"active": False, "mode": 0}

    n_samples = len(sig)
    t_arr = start_s + np.arange(n_samples, dtype=np.float64) / float(sr)
    mode = int(spec.mode)
    poly = int(max(1, spec.polyphony))
    env_steps = int(max(2, spec.env_steps))
    wt_bits = int(np.clip(spec.wavetable_bits, 4, 16))

    tick_s = max(0.004, float(spec.tick_quant_ms) / 1000.0) if spec.tick_quant_ms > 1.0 else 0.004

    melody_hz = np.array([
        1318.51, 1174.66, 739.99, 739.99, 830.61, 830.61,
        1108.73, 987.77, 587.33, 587.33, 659.25, 659.25,
        987.77, 880.00, 554.37, 554.37, 659.25, 659.25,
        880.00, 880.00, 880.00, 880.00, 0.0, 0.0,
    ], dtype=np.float64)
    harm1_hz = melody_hz * 0.5
    harm2_hz = np.where(melody_hz > 0, melody_hz * (2.0 / 3.0), 0.0)
    bass_hz = np.array([
        220.0, 220.0, 293.66, 293.66, 329.63, 329.63,
        220.0, 220.0, 293.66, 293.66, 329.63, 329.63,
        220.0, 220.0, 277.18, 277.18, 329.63, 329.63,
        220.0, 277.18, 329.63, 440.00, 220.0, 0.0,
    ], dtype=np.float64)

    step_dur_s = 0.16
    t_quant = np.floor(t_arr / tick_s) * tick_s
    step_idx = (np.floor(t_quant / step_dur_s).astype(np.int32)) % len(melody_hz)
    note_local_t = np.mod(t_arr, step_dur_s)

    raw_env = np.where(note_local_t < 0.010, note_local_t / 0.010, np.exp(-(note_local_t - 0.010) / 0.12))
    raw_env = np.where(note_local_t > step_dur_s * 0.88, 0.0, raw_env)
    stepped_env = np.round(raw_env * env_steps) / float(env_steps)

    voice_stolen_count = 0

    if mode == 4:
        win_len = max(256, int(round(max(0.025, tick_s) * sr)))
        mono_in = 0.5 * (sig[:, 0] + sig[:, 1])
        synth_mono = np.zeros(n_samples, dtype=np.float32)
        freqs_fft = np.fft.rfftfreq(win_len, 1.0 / float(sr))
        valid_bins = (freqs_fft >= 140.0) & (freqs_fft <= 3800.0)
        bin_indices = np.where(valid_bins)[0]

        for p0 in range(0, n_samples, win_len):
            p1 = min(n_samples, p0 + win_len)
            seg = mono_in[p0:p1]
            if len(seg) < 16:
                continue
            sp = np.abs(np.fft.rfft(seg, n=win_len))
            mag_sub = sp[bin_indices]
            candidate_peaks = int(np.sum(mag_sub > np.max(mag_sub) * 0.25))
            if candidate_peaks > poly:
                voice_stolen_count += (candidate_peaks - poly)
            top_k = min(poly, len(bin_indices))
            top_idx = bin_indices[np.argsort(mag_sub)[-top_k:]]
            t_seg = np.arange(p1 - p0, dtype=np.float64) / float(sr)
            env_seg = float(np.round(np.sqrt(np.mean(seg ** 2)) * env_steps) / env_steps)
            frame_wave = np.zeros(p1 - p0, dtype=np.float32)
            for b_i in top_idx:
                f_hz = float(freqs_fft[b_i])
                midi_note = np.round(69.0 + 12.0 * np.log2(max(20.0, f_hz) / 440.0))
                f_q = 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))
                if poly == 1:
                    w = np.sign(np.sin(2.0 * np.pi * f_q * t_seg)).astype(np.float32)
                else:
                    mod = 1.4 * np.sin(2.0 * np.pi * (2.0 * f_q) * t_seg) * np.exp(-t_seg * 8.0)
                    w = np.sin(2.0 * np.pi * f_q * t_seg + mod).astype(np.float32)
                frame_wave += w / max(1.0, np.sqrt(top_k))
            synth_mono[p0:p1] = frame_wave * min(0.85, env_seg * 2.2)
        synth_out = np.stack([synth_mono, synth_mono], axis=-1)

    elif mode == 1 or poly == 1:
        voice_stolen_count = int(len(melody_hz) * 3)
        f_lead = melody_hz[step_idx]
        phase = 2.0 * np.pi * np.cumsum(f_lead) / float(sr)
        on_gate = ((f_lead > 10.0) & (note_local_t < step_dur_s * 0.84)).astype(np.float32)
        sq = np.where(np.sin(phase) >= 0.0, 0.55, -0.55).astype(np.float32) * on_gate
        synth_out = np.stack([sq, sq], axis=-1)
        synth_out = _hp(synth_out, 550.0, sr, order=3)
        synth_out = _peaking_eq(synth_out, 2750.0, q=3.2, gain_db=6.5, sr=sr)
        synth_out = _lp(synth_out, 6500.0, sr, order=3)

    elif mode == 2:
        voices_hz = [melody_hz[step_idx], harm1_hz[step_idx], harm2_hz[step_idx], bass_hz[step_idx]]
        active_voices = voices_hz[: min(len(voices_hz), poly)]
        if poly < len(voices_hz):
            voice_stolen_count = int((len(voices_hz) - poly) * (n_samples / max(1.0, step_dur_s * sr)))
        fm_l = np.zeros(n_samples, dtype=np.float32)
        fm_r = np.zeros(n_samples, dtype=np.float32)
        for v_i, f_arr in enumerate(active_voices):
            ph_c = 2.0 * np.pi * np.cumsum(f_arr) / float(sr)
            ph_m = 2.0 * np.pi * np.cumsum(f_arr * (2.0 if v_i < 2 else 3.0)) / float(sr)
            mod_idx = (1.85 if v_i == 0 else 1.15) * stepped_env
            op_mod = np.sin(ph_m) * mod_idx
            lut_steps = float(2 ** (wt_bits - 1))
            op_car = np.round(np.sin(ph_c + op_mod) * lut_steps) / lut_steps
            v_sig = (op_car * stepped_env * (0.42 if v_i == 0 else 0.24)).astype(np.float32) * (f_arr > 10.0)
            pan = 0.5 + 0.25 * ((-1) ** v_i)
            fm_l += v_sig * pan
            fm_r += v_sig * (1.0 - pan)
        synth_out = np.stack([fm_l, fm_r], axis=-1)

    else:
        voices_hz = [melody_hz[step_idx], harm1_hz[step_idx], harm2_hz[step_idx], bass_hz[step_idx]]
        active_voices = voices_hz[: min(len(voices_hz), poly)]
        if poly < len(voices_hz):
            voice_stolen_count = int((len(voices_hz) - poly) * (n_samples / max(1.0, step_dur_s * sr)))
        wt_len = 128
        theta_wt = np.linspace(0.0, 2.0 * np.pi, wt_len, endpoint=False)
        raw_wt = (0.55 * np.sin(theta_wt) + 0.28 * np.sin(2.0 * theta_wt + 0.4)
                  + 0.16 * np.sin(3.0 * theta_wt + 1.1) + 0.09 * np.sin(5.0 * theta_wt))
        wt_levels = float(2 ** (wt_bits - 1))
        quant_wt = (np.round(raw_wt * wt_levels) / wt_levels).astype(np.float32)

        wt_l = np.zeros(n_samples, dtype=np.float32)
        wt_r = np.zeros(n_samples, dtype=np.float32)
        for v_i, f_arr in enumerate(active_voices):
            ph_idx = np.mod(np.cumsum(f_arr / float(sr)) * wt_len, float(wt_len))
            idx_int = np.floor(ph_idx).astype(np.int32) % wt_len
            v_sig = quant_wt[idx_int] * stepped_env.astype(np.float32) * (0.45 if v_i == 0 else 0.25) * (f_arr > 10.0)
            wt_l += v_sig * (0.6 if v_i % 2 == 0 else 0.4)
            wt_r += v_sig * (0.4 if v_i % 2 == 0 else 0.6)
        synth_out = np.stack([wt_l, wt_r], axis=-1)
        synth_out = _lp(synth_out, 11025.0, sr, order=4)

    mix = float(spec.mix)
    out = sig * (1.0 - mix) + synth_out * mix
    return out.astype(np.float32), {
        "active": True,
        "mode": mode,
        "polyphony": poly,
        "voice_stolen_events": int(voice_stolen_count),
        "tick_quant_ms": round(float(spec.tick_quant_ms), 1),
        "wavetable_bits": wt_bits,
    }


# ===========================================================================
# SUBSYSTEM 09 · 广播与网络平台的响度处理 (Broadcast & Platform Loudness / EBU R 128)
# ===========================================================================
@dataclass
class LoudnessProcessorSpec:
    mode: int = 0
    target_lufs: float = -14.0
    compressor_drive: float = 0.0
    true_peak_ceiling_dbtp: float = -1.0
    retranscode_passes: int = 1

    @classmethod
    def from_params(cls, params: dict) -> "LoudnessProcessorSpec":
        p = params or {}
        aud = p.get("audio") if isinstance(p.get("audio"), dict) else p
        mode = int(np.clip(round(float(aud.get("loudness_mode", 0.0))), 0, 4))
        default_lufs = {0: -14.0, 1: -23.0, 2: -9.5, 3: -14.0, 4: -12.0}.get(mode, -14.0)
        raw_lufs = float(aud.get("loudness_target_lufs", default_lufs))
        if abs(raw_lufs - (-14.0)) < 0.1 and mode in (1, 2, 4):
            t_lufs = default_lufs
        else:
            t_lufs = raw_lufs
        comp = float(np.clip(aud.get("loudness_compressor", 0.78 if mode == 2 else (0.35 if mode in (1, 4) else 0.0)), 0.0, 1.0))
        tp_ceil = float(np.clip(aud.get("loudness_true_peak_dbtp", -1.0 if mode == 1 else (-0.3 if mode == 2 else -1.0)), -9.0, 0.0))
        passes = int(np.clip(round(float(aud.get("loudness_retranscode_passes", 3.0 if mode == 4 else 1.0))), 1, 6))
        return cls(
            mode=mode,
            target_lufs=float(np.clip(t_lufs, -32.0, -5.0)),
            compressor_drive=comp,
            true_peak_ceiling_dbtp=tp_ceil,
            retranscode_passes=passes,
        )

    def is_active(self) -> bool:
        return self.mode > 0 or self.compressor_drive > 1e-4


def measure_ebu_r128_loudness(sig: np.ndarray, sr: int = 44100) -> dict[str, float]:
    arr = np.asarray(sig, dtype=np.float32)
    if arr.size == 0:
        return {"integrated_lufs": -70.0, "true_peak_dbtp": -70.0, "lra_lu": 0.0, "crest_db": 0.0}
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=-1)

    k_sig = _peaking_eq(arr, min(4500.0, 0.42 * sr), q=0.5, gain_db=4.0, sr=sr)
    k_sig = _hp(k_sig, 38.0, sr, order=2)

    mean_sq = float(np.mean(np.sum(k_sig ** 2, axis=1)))
    integrated_lufs = float(-0.691 + 10.0 * np.log10(max(1e-12, mean_sq)))

    blk = max(64, int(round(0.100 * sr)))
    n_blks = max(1, len(k_sig) // blk)
    if n_blks >= 3:
        trimmed = k_sig[: n_blks * blk].reshape(n_blks, blk, 2)
        blk_ms = np.mean(np.sum(trimmed ** 2, axis=2), axis=1)
        blk_lufs = -0.691 + 10.0 * np.log10(np.maximum(1e-10, blk_ms))
        active_blks = blk_lufs[blk_lufs > (integrated_lufs - 20.0)]
        if len(active_blks) >= 2:
            lra_lu = float(np.percentile(active_blks, 95.0) - np.percentile(active_blks, 10.0))
        else:
            lra_lu = float(np.ptp(blk_lufs))
    else:
        lra_lu = 1.5

    peak_sample = float(np.max(np.abs(arr)))
    mid_interp = 0.5 * (arr[:-1] + arr[1:]) if len(arr) > 1 else arr
    true_peak_lin = max(peak_sample, float(np.max(np.abs(mid_interp))) * 1.015)
    true_peak_dbtp = float(20.0 * np.log10(max(1e-7, true_peak_lin)))

    rms_lin = float(np.sqrt(np.mean(arr ** 2)))
    crest_db = float(20.0 * np.log10(max(1e-7, true_peak_lin) / max(1e-7, rms_lin)))

    return {
        "integrated_lufs": round(integrated_lufs, 2),
        "true_peak_dbtp": round(true_peak_dbtp, 2),
        "lra_lu": round(max(0.1, lra_lu), 2),
        "crest_db": round(max(0.5, crest_db), 2),
    }


def apply_broadcast_platform_loudness(
    sig: np.ndarray,
    spec: LoudnessProcessorSpec,
    sr: int = 44100,
) -> tuple[np.ndarray, dict[str, Any]]:
    in_metrics = measure_ebu_r128_loudness(sig, sr=sr)
    if not spec.is_active():
        return sig.astype(np.float32, copy=False), {"active": False, "mode": 0, "input": in_metrics, "output": in_metrics}

    out = np.asarray(sig, dtype=np.float32).copy()
    mode = int(spec.mode)
    comp = float(spec.compressor_drive)
    passes = int(max(1, spec.retranscode_passes))

    for p_idx in range(passes):
        if comp > 1e-4:
            b_low = _lp(out, 220.0, sr, order=2)
            b_high = _hp(out, 3800.0, sr, order=2)
            b_mid = out - b_low - b_high

            def _compress_band(band: np.ndarray, thresh: float, tau_ms: float, makeup: float) -> np.ndarray:
                env = gaussian_filter1d(np.max(np.abs(band), axis=1), sigma=max(2, int((tau_ms / 1000.0) * sr)))
                ratio = 1.0 + 7.5 * comp
                exponent = 1.0 - 1.0 / ratio
                gr = np.where(env > thresh, (thresh / np.maximum(1e-5, env)) ** exponent, 1.0).astype(np.float32)
                return band * gr[:, None] * (1.0 + makeup * comp)

            b_low_c = _compress_band(b_low, 0.16, 18.0, 0.65)
            b_mid_c = _compress_band(b_mid, 0.14, 8.0, 0.85)
            b_high_c = _compress_band(b_high, 0.10, 4.0, 0.95)
            out = (b_low_c + b_mid_c + b_high_c).astype(np.float32)

        eff_target = float(spec.target_lufs) + (1.2 * ((-1) ** p_idx) if mode == 4 and p_idx > 0 else 0.0)
        cur_m = measure_ebu_r128_loudness(out, sr=sr)
        cur_lufs = cur_m["integrated_lufs"]
        if cur_lufs > -65.0:
            delta_db = float(np.clip(eff_target - cur_lufs, -24.0, 24.0))
            gain_norm = float(10.0 ** (delta_db / 20.0))
            out = out * gain_norm

        ceil_lin = float(10.0 ** (float(spec.true_peak_ceiling_dbtp) / 20.0))
        env_pk = gaussian_filter1d(np.max(np.abs(out), axis=1), sigma=max(1, int(0.0012 * sr)))
        lim_gain = np.where(env_pk > ceil_lin, ceil_lin / np.maximum(1e-6, env_pk), 1.0).astype(np.float32)
        out = out * lim_gain[:, None]
        out = np.clip(out, -ceil_lin, ceil_lin).astype(np.float32)

        post_m = measure_ebu_r128_loudness(out, sr=sr)
        if post_m["integrated_lufs"] > -65.0 and abs(post_m["integrated_lufs"] - eff_target) > 0.25:
            trim_db = float(np.clip(eff_target - post_m["integrated_lufs"], -12.0, 8.0))
            out_trimmed = out * float(10.0 ** (trim_db / 20.0))
            if float(np.max(np.abs(out_trimmed))) <= ceil_lin * 1.02:
                out = np.clip(out_trimmed, -ceil_lin, ceil_lin).astype(np.float32)

    out_metrics = measure_ebu_r128_loudness(out, sr=sr)
    return out.astype(np.float32), {
        "active": True,
        "mode": mode,
        "target_lufs": round(float(spec.target_lufs), 2),
        "retranscode_passes": passes,
        "input": in_metrics,
        "output": in_metrics,
    }


# ===========================================================================
# SUBSYSTEM 10 · 音视频同步与时钟系统 (Audio-Video Sync & Master Clock System)
# ===========================================================================
@dataclass
class AVSyncClockSpec:
    enabled: bool = False
    offset_ms: float = 0.0
    clock_drift_ppm: float = 0.0
    frame_drop_rate: float = 0.0
    drop_sync_mode: int = 0
    edit_splice_mode: int = 0
    edit_interval_s: float = 1.5

    @classmethod
    def from_params(cls, params: dict) -> "AVSyncClockSpec":
        p = params or {}
        aud = p.get("audio") if isinstance(p.get("audio"), dict) else p
        ply = p.get("player") if isinstance(p.get("player"), dict) else {}
        tape = p.get("tape") if isinstance(p.get("tape"), dict) else {}
        cod = p.get("codec") if isinstance(p.get("codec"), dict) else {}

        base_offset = float(aud.get("av_sync_offset_ms", 0.0))
        if int(round(float(ply.get("transfer_station_mode", 0.0)))) > 0:
            if float(ply.get("transfer_tbc_lock", ply.get("tbc_enabled", 0.0))) < 0.5:
                base_offset += float(ply.get("transfer_av_drift_ms", 0.0))

        base_drift = float(aud.get("av_clock_drift_ppm", 0.0))
        if abs(base_drift) < 1e-3 and abs(float(ply.get("clock_drift_ppm", 0.0))) > 1.0:
            base_drift = float(ply.get("clock_drift_ppm", 0.0)) * 0.5

        base_drop = float(aud.get("av_frame_drop_rate", 0.0))
        if base_drop < 1e-4:
            if int(round(float(tape.get("edit_mode", 0.0)))) == 3:
                base_drop = max(base_drop, float(tape.get("nle_drop_frame_prob", 0.15)))
            if float(cod.get("stream_enabled", 0.0)) > 0.5 and float(cod.get("stream_rebuffer_prob", 0.0)) > 0.1:
                base_drop = max(base_drop, float(cod.get("stream_rebuffer_prob", 0.0)) * 0.35)

        splice_m = int(round(float(aud.get("av_edit_splice_mode", 0.0))))
        if splice_m == 0 and int(round(float(tape.get("edit_mode", 0.0)))) in (1, 2):
            splice_m = 1

        drop_m = int(np.clip(round(float(aud.get("av_drop_sync_mode", 0.0))), 0, 2))
        edit_int = float(np.clip(aud.get("av_edit_interval_s", 1.5), 0.25, 10.0))

        en = (
            float(aud.get("av_sync_enabled", 0.0)) > 0.5
            or abs(base_offset) > 0.5
            or abs(base_drift) > 1.0
            or base_drop > 1e-4
            or splice_m > 0
        )
        return cls(
            enabled=en,
            offset_ms=float(np.clip(base_offset, -1000.0, 1000.0)),
            clock_drift_ppm=float(np.clip(base_drift, -5000.0, 5000.0)),
            frame_drop_rate=float(np.clip(base_drop, 0.0, 0.90)),
            drop_sync_mode=drop_m,
            edit_splice_mode=int(np.clip(splice_m, 0, 2)),
            edit_interval_s=edit_int,
        )

    def is_active(self) -> bool:
        return bool(self.enabled)


def compute_av_sync_frame_map(
    frame_index: int,
    fps: float,
    params: dict,
    seed: int = 20240101,
) -> dict[str, Any]:
    spec = AVSyncClockSpec.from_params(params)
    fps_safe = max(1.0, float(fps))
    nominal_pts_s = float(frame_index) / fps_safe

    drift_ms = nominal_pts_s * float(spec.clock_drift_ppm) * 1e-3

    rng = np.random.default_rng((int(seed) * 747796405 + int(frame_index) * 2891336453) & 0xFFFFFFFF)
    dropped = bool(spec.frame_drop_rate > 1e-4 and frame_index > 0 and rng.random() < spec.frame_drop_rate)
    effective_video_frame = max(0, frame_index - 1) if dropped else frame_index

    tape_cfg = (params or {}).get("tape") or {}
    in_f = int(round(float(tape_cfg.get("edit_in_frame", -1))))
    out_f = int(round(float(tape_cfg.get("edit_out_frame", -1))))
    periodic_frames = max(4, int(round(spec.edit_interval_s * fps_safe)))
    is_splice_frame = False
    if spec.edit_splice_mode > 0:
        if frame_index in (in_f, out_f) and frame_index > 0:
            is_splice_frame = True
        elif frame_index > 0 and (frame_index % periodic_frames == 0):
            is_splice_frame = True

    total_av_offset_ms = float(spec.offset_ms) + drift_ms
    if dropped and spec.drop_sync_mode == 0:
        total_av_offset_ms += 1000.0 / fps_safe

    video_pts_s = float(effective_video_frame) / fps_safe
    audio_pts_s = nominal_pts_s + total_av_offset_ms / 1000.0

    return {
        "frame_index": int(frame_index),
        "effective_video_frame": int(effective_video_frame),
        "video_pts_s": round(video_pts_s, 5),
        "audio_pts_s": round(audio_pts_s, 5),
        "av_offset_ms": round(total_av_offset_ms, 3),
        "clock_drift_ms": round(drift_ms, 3),
        "frame_dropped": bool(dropped),
        "edit_splice": bool(is_splice_frame),
    }


def apply_av_sync_clock_system(
    sig: np.ndarray,
    spec: AVSyncClockSpec,
    sr: int = 44100,
    fps: float = 29.97,
    start_s: float = 0.0,
    params: dict | None = None,
    seed: int = 20240101,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not spec.is_active():
        return sig.astype(np.float32, copy=False), {
            "active": False,
            "av_offset_ms": 0.0,
            "clock_drift_ppm": 0.0,
            "dropped_frames": 0,
            "splice_points": 0,
        }

    out = np.asarray(sig, dtype=np.float32).copy()
    n_samples = len(out)
    fps_safe = max(1.0, float(fps))

    offset_samples = (float(spec.offset_ms) / 1000.0) * float(sr)
    drift_rate = float(spec.clock_drift_ppm) * 1e-6
    sample_idx = np.arange(n_samples, dtype=np.float64)

    initial_drift_samples = float(start_s) * drift_rate * float(sr)
    src_pos = sample_idx * (1.0 - drift_rate) - offset_samples - initial_drift_samples

    valid_mask = (src_pos >= 0.0) & (src_pos <= float(n_samples - 1.0001))
    pos_clamped = np.clip(src_pos, 0.0, float(n_samples - 1.0001))
    i0 = np.floor(pos_clamped).astype(np.int32)
    i1 = np.minimum(i0 + 1, n_samples - 1)
    frac = (pos_clamped - i0)[:, None].astype(np.float32)
    shifted = out[i0] * (1.0 - frac) + out[i1] * frac
    out = np.where(valid_mask[:, None], shifted, 0.0).astype(np.float32)

    start_f = max(0, int(np.floor(start_s * fps_safe + 1e-6)))
    end_f = max(start_f, int(np.ceil((start_s + n_samples / float(sr)) * fps_safe - 1e-6)) - 1)
    dropped_count = 0
    splice_count = 0

    for f in range(start_f, end_f + 1):
        f_map = compute_av_sync_frame_map(f, fps_safe, params or {}, seed=seed)
        s0 = max(0, min(n_samples, int(round((f / fps_safe - start_s) * sr))))
        s1 = max(0, min(n_samples, int(round(((f + 1) / fps_safe - start_s) * sr))))
        if s1 <= s0:
            continue

        if f_map["frame_dropped"]:
            dropped_count += 1
            if int(spec.drop_sync_mode) == 1 and s0 >= (s1 - s0):
                span = s1 - s0
                out[s0:s1] = out[s0 - span : s0]
            elif int(spec.drop_sync_mode) == 2 and s1 - s0 > 16:
                span = s1 - s0
                half = max(4, span // 2)
                p_pll = np.linspace(0.0, float(half - 1), span, dtype=np.float32)
                p0_idx = np.floor(p_pll).astype(np.int32)
                p1_idx = np.minimum(p0_idx + 1, half - 1)
                pf = (p_pll - p0_idx)[:, None]
                src_seg = out[max(0, s0 - half) : max(0, s0 - half) + half]
                if len(src_seg) == half:
                    out[s0:s1] = src_seg[p0_idx] * (1.0 - pf) + src_seg[p1_idx] * pf

        if f_map["edit_splice"]:
            splice_count += 1
            if int(spec.edit_splice_mode) == 1:
                gap_len = min(s1 - s0, max(4, int(round(0.0035 * sr))))
                out[s0 : s0 + gap_len] = 0.0
                click_len = min(s1 - s0 - gap_len, max(8, int(round(0.004 * sr))))
                if click_len > 0:
                    t_c = np.arange(click_len, dtype=np.float32) / float(sr)
                    click_wave = 0.58 * np.exp(-t_c / 0.0009) * np.cos(2.0 * np.pi * 1450.0 * t_c)
                    out[s0 + gap_len : s0 + gap_len + click_len] += click_wave[:, None]
            elif int(spec.edit_splice_mode) == 2:
                fade_len = min(s1 - s0, max(8, int(round(0.005 * sr))))
                v_env = np.abs(np.linspace(-1.0, 1.0, fade_len, dtype=np.float32))[:, None]
                out[s0 : s0 + fade_len] *= v_env

    end_drift_ms = (start_s + n_samples / float(sr)) * float(spec.clock_drift_ppm) * 1e-3
    return out.astype(np.float32), {
        "active": True,
        "av_offset_ms": round(float(spec.offset_ms) + end_drift_ms, 2),
        "fixed_offset_ms": round(float(spec.offset_ms), 2),
        "clock_drift_ppm": round(float(spec.clock_drift_ppm), 2),
        "cumulative_drift_ms": round(end_drift_ms, 3),
        "dropped_frames": int(dropped_count),
        "drop_sync_mode": int(spec.drop_sync_mode),
        "splice_points": int(splice_count),
    }


# ===========================================================================
# Unified 10-Subsystem Orchestrator & Node-Graph Stages
# ===========================================================================
@dataclass
class AudioSubsystemsChainSpec:
    mic_frontend: MicPreampFrontendSpec
    mobile_speaker: MobileSpeakerSimSpec
    digital_sampling: DigitalSamplingSpec
    perceptual_codec: PerceptualCodecSpec
    telecom_voice: TelecomVoiceSpec
    acoustic_space: AcousticSpaceSpec
    system_chain: SystemAudioChainSpec
    ringtone_synth: MobileRingtoneSynthSpec
    loudness_proc: LoudnessProcessorSpec
    av_sync_clock: AVSyncClockSpec

    @classmethod
    def from_params(cls, params: dict) -> "AudioSubsystemsChainSpec":
        p = params or {}
        return cls(
            mic_frontend=MicPreampFrontendSpec.from_params(p),
            mobile_speaker=MobileSpeakerSimSpec.from_params(p),
            digital_sampling=DigitalSamplingSpec.from_params(p),
            perceptual_codec=PerceptualCodecSpec.from_params(p),
            telecom_voice=TelecomVoiceSpec.from_params(p),
            acoustic_space=AcousticSpaceSpec.from_params(p),
            system_chain=SystemAudioChainSpec.from_params(p),
            ringtone_synth=MobileRingtoneSynthSpec.from_params(p),
            loudness_proc=LoudnessProcessorSpec.from_params(p),
            av_sync_clock=AVSyncClockSpec.from_params(p),
        )


def run_audio_subsystems_front_end(
    sig: np.ndarray,
    params: dict,
    sr: int = 44100,
    seed: int = 20240101,
    start_s: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    chain_spec = AudioSubsystemsChainSpec.from_params(params)
    out = np.asarray(sig, dtype=np.float32).copy()
    telemetry: dict[str, Any] = {}

    if chain_spec.ringtone_synth.is_active():
        out, m08 = apply_mobile_ringtone_synth(out, chain_spec.ringtone_synth, sr=sr, seed=seed, start_s=start_s)
        telemetry["s08_ringtone_midi"] = m08

    if chain_spec.acoustic_space.is_active():
        out, m06 = apply_acoustic_space_simulator(out, chain_spec.acoustic_space, sr=sr, seed=seed, start_s=start_s)
        telemetry["s06_acoustic_space"] = m06

    if chain_spec.mic_frontend.is_active():
        out, m01 = apply_mic_preamp_frontend(out, chain_spec.mic_frontend, sr=sr, seed=seed, start_s=start_s)
        telemetry["s01_mic_frontend"] = m01

    if chain_spec.digital_sampling.is_active():
        out, m03 = apply_digital_sampling_converter(out, chain_spec.digital_sampling, sr=sr, seed=seed)
        telemetry["s03_digital_sampling"] = m03

    return out, telemetry


def run_audio_subsystems_back_end(
    sig: np.ndarray,
    params: dict,
    sr: int = 44100,
    fps: float = 29.97,
    seed: int = 20240101,
    start_s: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    chain_spec = AudioSubsystemsChainSpec.from_params(params)
    out = np.asarray(sig, dtype=np.float32).copy()
    telemetry: dict[str, Any] = {}

    if chain_spec.perceptual_codec.is_active():
        out, m04 = apply_perceptual_audio_codec(out, chain_spec.perceptual_codec, sr=sr, seed=seed)
        telemetry["s04_perceptual_codec"] = m04

    if chain_spec.telecom_voice.is_active():
        out, m05 = apply_telecom_voice_channel(out, chain_spec.telecom_voice, sr=sr, seed=seed, start_s=start_s)
        telemetry["s05_telecom_voice"] = m05

    out, m09 = apply_broadcast_platform_loudness(out, chain_spec.loudness_proc, sr=sr)
    telemetry["s09_loudness"] = m09

    if chain_spec.system_chain.is_active():
        out, m07 = apply_system_audio_chain(out, chain_spec.system_chain, sr=sr, seed=seed, start_s=start_s)
        telemetry["s07_system_chain"] = m07

    if chain_spec.mobile_speaker.is_active():
        out, m02 = apply_mobile_speaker_sim(out, chain_spec.mobile_speaker, sr=sr)
        telemetry["s02_mobile_speaker"] = m02

    if chain_spec.av_sync_clock.is_active():
        out, m10 = apply_av_sync_clock_system(
            out, chain_spec.av_sync_clock, sr=sr, fps=fps, start_s=start_s, params=params, seed=seed
        )
        telemetry["s10_av_sync_clock"] = m10
    else:
        telemetry["s10_av_sync_clock"] = {
            "active": False,
            "av_offset_ms": 0.0,
            "clock_drift_ppm": 0.0,
            "dropped_frames": 0,
            "splice_points": 0,
        }

    return out, telemetry


from ..core.pipeline import FrameBuffer, RenderContext, Stage


class AVSyncClockStage(Stage):
    """
    Node-Graph stage (`av_sync_clock`) that unifies Video Frame PTS, Frame Drops/Holds,
    Edit-Point Splice Flashes, and Audio Sample Clock Drift across the apparatus.
    """
    name = "av_sync_clock"

    def __init__(self, ctx: RenderContext):
        super().__init__(ctx)
        self._last_valid_rgb: np.ndarray | None = None

    def prepare(self) -> None:
        self._last_valid_rgb = None

    def apply(self, buf: FrameBuffer) -> FrameBuffer:
        sync_map = compute_av_sync_frame_map(
            buf.index, self.ctx.fps, self.ctx.params, seed=self.ctx.seed
        )
        buf.meta["av_sync_clock"] = sync_map
        spec = AVSyncClockSpec.from_params(self.ctx.params)
        if spec.is_active():
            if sync_map["frame_dropped"] and self._last_valid_rgb is not None:
                buf.rgb = self._last_valid_rgb.copy()
            else:
                self._last_valid_rgb = buf.rgb.copy()

            if sync_map["edit_splice"] and int(spec.edit_splice_mode) == 1:
                h = buf.rgb.shape[0]
                seam_row = max(2, min(h - 3, int(0.82 * h)))
                buf.rgb[seam_row - 1 : seam_row + 1] = np.clip(buf.rgb[seam_row - 1 : seam_row + 1] + 0.22, 0.0, 1.0)
        return buf
