"""
`atsim.codecs.audio` -- Physical analog & digital-tape audio simulation with frame-accurate A/V fault synchronization.

Why audio must share the video chain's physical state (`TapePlan`, `ConnectorStage`, `RFLink`, `DisplayStage`)
rather than running as an independent post-filter:

  1. **Transport Coupling (Capstan & Drum)**:
     - On VHS, Betamax, and 8mm/Hi8, the same capstan pinch-roller that pulls the tape past the video drum
       pulls it past the linear audio head (and sets the track pitch for AFM / Hi-Fi heads).
     - Therefore, capstan wow & flutter (`line_phase`, `frame_dy`, `walk_amplitude`, `transport_jitter`) and
       binder stiction (`TapePlan.stiction` grab-and-release episodes) modulate audio pitch/speed at the exact
       same millisecond that the video lines tear or jump.
     - During shuttle (fast-forward / rewind, `shuttle_speed > 0`), the tape speed changes and the heads cross
       azimuth tracks: analog audio warps in pitch and chops across guard bands in lockstep with the video
       shuttle bars, while MiniDV PCM produces choppy buffer-hold / click bursts synchronized with macroblock
       concealment.

  2. **Format-Specific Audio Recording Paths (`audio.mode`)**:
     - **0: Auto** -- Resolves automatically from the project's video format (`codec` & `tape` layers):
       - `codec.kind == 2` -> Video CD (MPEG-1 Layer II subband quantization, birdies, pre-echo)
       - `codec.dv_enabled > 0.5` -> MiniDV / Digital8 PCM (12-bit 32 kHz nonlinear / 16-bit 48 kHz)
       - Otherwise -> VHS Hi-Fi AFM Stereo with automatic fallback to VHS Linear Track on severe carrier loss
     - **1: VHS Linear Track** -- Longitudinal track at the upper edge of the 1/2-inch tape moving at
       33.35 mm/s (SP), 16.67 mm/s (LP), or 11.12 mm/s (EP). Very narrow head gap and slow tape speed
       limit high frequencies (~8 kHz in SP, ~4.5 kHz in LP, ~2.8 kHz in EP), head azimuth misalignment
       (`azimuth_loss`) rolls off highs and introduces L/R phase offset, magnetic hysteresis (`tanh` +
       even/odd harmonics) saturates peaks, and oxide particle hiss (`hiss_floor`) sits high.
     - **2: VHS Hi-Fi AFM Stereo** -- Depth-multiplexed FM carriers (1.3 MHz L, 1.7 MHz R) recorded by
       rotating heads on the video drum beneath the video layer. Wide 20 Hz - 20 kHz response and low hiss,
       but suffers from **59.94/50 Hz head-switching transient crackle** at the end of every video field,
       companding noise-reduction breathing, and **automatic fallback to the muffled linear track** whenever
       a deep tape dropout, head clog, crease, or tracking error causes the FM limiter to lose carrier lock.
     - **3: 8mm / Hi8 AFM Audio** -- 1.5 MHz FM audio carrier recorded alongside the video track on the
       rotating drum; tracking errors and dropouts cause brief FM discriminator static bursts.
     - **4: MiniDV / Digital8 PCM** -- 12-bit 32 kHz (nonlinear companded) or 16-bit 48 kHz uncompressed
       PCM recorded in dedicated helical track sectors. Immune to analog hiss and wow, but when tape dropouts
       or shuttle exceed ECC capacity, the decoder mutes, holds the last sample buffer, or emits digital
       framing clicks at the exact frames where video macroblocks conceal.
     - **5: Video CD (MPEG-1 Layer II)** -- 16-bit 44.1 kHz audio compressed via 32-subband psychoacoustic
       quantization: high-frequency "watery" spectral warble (birdies) and transient pre-echo smear.

  3. **Strict Frame- & Row-Accurate Fault Synchronization**:
     - Every video frame `f` spans `[f / fps, (f + 1) / fps]`, and scanline `r` (out of `H`) maps to the
       sub-frame instant `(f + r / H) / fps`.
     - **Dropouts (`TapePlan.dropouts`)**: A dropout at `(start_frame, frames, row, rows, severity)` occurs at
       the exact sub-frame timestamp `(frame + row / H) / fps`. On Hi-Fi/AFM it triggers an FM carrier-loss
       pop/crackle burst and forces fallback to the muffled linear track; on Linear it causes an oxide
       contact-loss amplitude dip and high-frequency dulling; on MiniDV it causes a PCM sample-hold/click.
     - **Head Clog (`TapePlan.clog_rows`) & Creases (`TapePlan.creases`)**: Produce periodic field-rate
       muffling / FM static bursts and low-frequency thumps (`< 120 Hz` head-contact shock) when the crease
       sweeps past the heads.
     - **Base Cracking & Breakage (`base_cracking`, `base_breakage`)**: Sub-surface magnetic-layer cracks
       produce sharp impulse clicks at the exact scanline positions of the video white/black spikes; severe
       base breakage produces control-track (CTL) servo unlock warble and white-noise snow bursts.
     - **Head-Switching Buzz (`head_switching`, `tracking_error`)**: 59.94 Hz (NTSC) / 50 Hz (PAL) pulse
       train and harmonics modulated per frame by `TapePlan.head_switch_amp[f]`.
     - **Connector & RF Intercarrier Faults (`connector` & `rf` layers)**:
       - **AV Ground Loop (`av_ground_loop`, `av_moisture`, `av_corrosion`)**: 50/60 Hz (`mains_hz`) hum +
         odd/even rectifier harmonics synchronized in phase with the rolling video hum bar.
       - **RF Intercarrier Vision-into-Sound Buzz (`rf_moisture`, `rf_corrosion`, `rf_present`, `rf`)**:
         In an RF TV link, audio rides on a 4.5 MHz (NTSC) / 5.5 MHz (PAL) FM sound carrier above the video
         carrier. When damp/corroded RF coaxial reflections or sync-slipped line-clamp surges (`slipped_state`,
         `f_mod != 2`) over-modulate the video carrier, video sync and bright streaking bars bleed into the
         sound IF as harsh 59.94/50 Hz + 15.734/15.625 kHz intercarrier buzz and FM static bursts.
       - **HDMI Bit Errors (`hdmi_present`, `hdmi_moisture`, `hdmi_corrosion`)**: High BER produces sharp
         PCM packet-drop clicks/gaps, and when `ber > 0.55` (`link == "no signal"`), audio drops out completely
         in lockstep with the black video frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfilt


@dataclass
class AudioSpec:
    """Resolved parameters of the `audio` layer."""
    enabled: bool = True
    mode: int = 0                       # 0=Auto, 1=Linear, 2=Hi-Fi, 3=8mm AFM, 4=MiniDV PCM, 5=VCD MP2, 6=PAL-D RF Mono, 7=TV-DX, 8=Film Optical, 9=Streaming Low-Bitrate, 10=Broadcast Studio
    synth_type: int = 0                 # 0=Retro Synth Chord+Beat, 1=1kHz+400Hz Reference Tone, 2=Voice+Percussion
    hifi_unlock_thresh: float = 0.32    # Dropout/tracking severity that forces Hi-Fi -> Linear fallback
    linear_bandwidth: float = 7500.0    # Nominal SP linear track bandwidth (Hz)
    azimuth_loss: float = 0.22          # Head azimuth misalignment HF rolloff & L/R phase skew (0..1)
    wow_flutter: float = 0.35           # Capstan speed instability / wow & flutter depth (0..1.5)
    stiction_pitch: float = 0.65        # Pitch drop/overshoot depth during capstan stiction (0..1.5)
    dropout_sensitivity: float = 0.55   # How strongly video dropouts/creases/clogs damage audio (0..1.5)
    head_switch_buzz: float = 0.30      # 50/59.94 Hz head-switching pulse & tracking buzz (0..1)
    ground_hum: float = 0.20            # 50/60 Hz mains ground-loop hum coupled to AV connector (0..1)
    rf_buzz: float = 0.30               # Intercarrier vision-into-sound buzz coupled to RF link (0..1)
    saturation: float = 0.30            # Magnetic tape hysteresis soft-clipping & harmonics (0..1.2)
    hiss_floor: float = 0.22            # Oxide particle / FM discriminator noise floor (0..1)
    speaker_sim: float = 0.35           # CRT TV small cabinet speaker bandpass resonance (0..1)
    flyback_whine: float = 0.15         # 15.734 kHz / 15.625 kHz CRT horizontal flyback whine (0..1)
    master_gain: float = 1.0            # Output master gain (0..2)
    # Microphone Recording System & Acoustic Space
    mic_type: int = 0                   # 0=Direct, 1=Dynamic, 2=Condenser, 3=Ribbon, 4=Electret Webcam (2000s PC/Webcam), 5=Carbon
    proximity_effect: float = 0.0       # Proximity bass boost <200 Hz (0..1)
    plosive_pop: float = 0.0            # Pop/plosive wind blasts on vocal attacks (0..1)
    mic_overdrive: float = 0.0          # Diaphragm & preamp overload saturation/clipping (0..1)
    broadcast_agc: float = 0.0          # Broadcast limiter/compressor with pumping dynamics (0..1)
    studio_room_reverb: float = 0.0     # Studio console / bedroom anchor room acoustics (0..1)
    # Film Optical Soundtrack
    film_optical_scratches: float = 0.20# Optical soundtrack dust pops & scratch clicks (0..1)
    film_exciter_hum: float = 0.15      # Exciter lamp AC ripple hum (50/100 Hz or 60/120 Hz) (0..1)
    film_slit_loss: float = 0.40        # Optical reading slit aperture HF rolloff (~6.8 kHz 35mm / ~4.8 kHz 16mm)

    @classmethod
    def from_params(cls, params: dict) -> "AudioSpec":
        p = params or {}
        cfg = p.get("audio") if isinstance(p.get("audio"), dict) else p
        fmt_cfg = p.get("format") if isinstance(p.get("format"), dict) else {}
        s_type = int(round(float(cfg.get("synth_type", 0.0))))
        if float(fmt_cfg.get("test_gen_enabled", 0.0)) > 0.5 and int(round(float(fmt_cfg.get("test_tone_mode", 0.0)))) > 0:
            s_type = 1
        return cls(
            enabled=float(cfg.get("enabled", 1.0)) > 0.5,
            mode=int(round(float(cfg.get("mode", 0.0)))),
            synth_type=s_type,
            hifi_unlock_thresh=float(np.clip(cfg.get("hifi_unlock_thresh", 0.32), 0.05, 1.0)),
            linear_bandwidth=float(np.clip(cfg.get("linear_bandwidth", 7500.0), 800.0, 16000.0)),
            azimuth_loss=float(np.clip(cfg.get("azimuth_loss", 0.22), 0.0, 1.0)),
            wow_flutter=float(np.clip(cfg.get("wow_flutter", 0.35), 0.0, 1.5)),
            stiction_pitch=float(np.clip(cfg.get("stiction_pitch", 0.65), 0.0, 1.5)),
            dropout_sensitivity=float(np.clip(cfg.get("dropout_sensitivity", cfg.get("afm_switch_pop", cfg.get("dropout_crack", 0.55))), 0.0, 1.5)),
            head_switch_buzz=float(np.clip(cfg.get("head_switch_buzz", 0.30), 0.0, 1.0)),
            ground_hum=float(np.clip(cfg.get("ground_hum", 0.20), 0.0, 1.0)),
            rf_buzz=float(np.clip(cfg.get("rf_buzz", 0.30), 0.0, 1.0)),
            saturation=float(np.clip(cfg.get("saturation", 0.30), 0.0, 1.2)),
            hiss_floor=float(np.clip(cfg.get("hiss_floor", cfg.get("hiss", 0.22)), 0.0, 1.0)),
            speaker_sim=float(np.clip(cfg.get("speaker_sim", 0.35), 0.0, 1.0)),
            flyback_whine=float(np.clip(cfg.get("flyback_whine", 0.15), 0.0, 1.0)),
            master_gain=float(np.clip(cfg.get("master_gain", 1.0), 0.0, 2.0)),
            mic_type=int(round(float(cfg.get("mic_type", 0.0)))),
            proximity_effect=float(np.clip(cfg.get("proximity_effect", 0.0), 0.0, 1.0)),
            plosive_pop=float(np.clip(cfg.get("plosive_pop", 0.0), 0.0, 1.0)),
            mic_overdrive=float(np.clip(cfg.get("mic_overdrive", 0.0), 0.0, 1.0)),
            broadcast_agc=float(np.clip(cfg.get("broadcast_agc", 0.0), 0.0, 1.0)),
            studio_room_reverb=float(np.clip(cfg.get("studio_room_reverb", 0.0), 0.0, 1.0)),
            film_optical_scratches=float(np.clip(cfg.get("film_optical_scratches", 0.20), 0.0, 1.0)),
            film_exciter_hum=float(np.clip(cfg.get("film_exciter_hum", 0.15), 0.0, 1.0)),
            film_slit_loss=float(np.clip(cfg.get("film_slit_loss", 0.40), 0.0, 1.0)),
        )


@dataclass
class AudioFaultFrame:
    """Per-video-frame summary of synchronized A/V fault intensities (for timeline inspection & verification)."""
    frame_index: int
    time_s: float
    pitch_deviation_pct: float = 0.0
    dropout_level: float = 0.0
    hifi_fallback: float = 0.0
    stiction_grab: float = 0.0
    head_switch_buzz: float = 0.0
    rf_sync_buzz: float = 0.0
    ground_hum: float = 0.0
    hdmi_drop: float = 0.0
    crack_spike: float = 0.0
    shuttle_chop: float = 0.0
    av_offset_ms: float = 0.0
    clock_drift_ms: float = 0.0
    frame_dropped: bool = False
    edit_splice: bool = False

    @property
    def afm_unlock(self) -> bool:
        return self.hifi_fallback > 0.15

    @property
    def pitch_shift_cents(self) -> float:
        return float(1200.0 * np.log2(max(0.1, 1.0 + self.pitch_deviation_pct / 100.0)))

    @property
    def pitch_cents(self) -> float:
        return self.pitch_shift_cents

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame_index,
            "frame_index": self.frame_index,
            "t": round(self.time_s, 4),
            "time_s": round(self.time_s, 4),
            "pitch_pct": round(self.pitch_deviation_pct, 3),
            "pitch_cents": round(self.pitch_shift_cents, 2),
            "pitch_shift_cents": round(self.pitch_shift_cents, 2),
            "dropout": round(self.dropout_level, 3),
            "dropout_level": round(self.dropout_level, 3),
            "hifi_fallback": round(self.hifi_fallback, 3),
            "afm_unlock": bool(self.afm_unlock),
            "stiction": round(self.stiction_grab, 3),
            "hs_buzz": round(self.head_switch_buzz, 3),
            "head_switch_buzz": round(self.head_switch_buzz, 3),
            "rf_buzz": round(self.rf_sync_buzz, 3),
            "hum": round(self.ground_hum, 3),
            "hum_buzz": round(max(self.ground_hum, self.rf_sync_buzz), 3),
            "hdmi_drop": round(self.hdmi_drop, 3),
            "crack": round(self.crack_spike, 3),
            "shuttle": round(self.shuttle_chop, 3),
            "av_offset_ms": round(self.av_offset_ms, 2),
            "clock_drift_ms": round(self.clock_drift_ms, 3),
            "frame_dropped": bool(self.frame_dropped),
            "edit_splice": bool(self.edit_splice),
        }


class AudioRun:
    """Result of simulating an audio window."""
    def __init__(
        self,
        samples: np.ndarray | None = None,
        clean_samples: np.ndarray | None = None,
        sample_rate: int = 44100,
        start_s: float = 0.0,
        duration_s: float = 0.0,
        resolved_mode: str = "vhs_hifi",
        frame_faults: list[AudioFaultFrame] | None = None,
        summary: dict[str, Any] | None = None,
        *,
        pcm: np.ndarray | None = None,
        effective_mode: str | None = None,
        fault_frames: list[AudioFaultFrame] | None = None,
        frames: list[AudioFaultFrame] | None = None,
    ):
        arr = pcm if pcm is not None else samples
        self.samples = np.asarray(arr, dtype=np.float32) if arr is not None else np.zeros((0, 2), dtype=np.float32)
        self.clean_samples = np.asarray(clean_samples, dtype=np.float32) if clean_samples is not None else self.samples.copy()
        self.sample_rate = int(sample_rate)
        self.start_s = float(start_s)
        self.duration_s = float(duration_s)
        self.resolved_mode = str(effective_mode if effective_mode is not None else resolved_mode)
        ff = fault_frames if fault_frames is not None else (frames if frames is not None else frame_faults)
        self.frame_faults: list[AudioFaultFrame] = list(ff) if ff is not None else []
        self.summary: dict[str, Any] = dict(summary) if summary is not None else {}

    @property
    def pcm(self) -> np.ndarray:
        return self.samples

    @property
    def effective_mode(self) -> str:
        return self.resolved_mode

    @property
    def frames(self) -> list[AudioFaultFrame]:
        return self.frame_faults

    @property
    def fault_frames(self) -> list[AudioFaultFrame]:
        return self.frame_faults


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------
def _lowpass_audio(sig: np.ndarray, cutoff_hz: float, sr: int, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sr
    fc = float(np.clip(cutoff_hz, 80.0, nyq * 0.96))
    if fc >= nyq * 0.95:
        return sig
    sos = butter(order, fc / nyq, btype="low", output="sos")
    return sosfilt(sos, sig, axis=0).astype(np.float32)


def _highpass_audio(sig: np.ndarray, cutoff_hz: float, sr: int, order: int = 2) -> np.ndarray:
    nyq = 0.5 * sr
    fc = float(np.clip(cutoff_hz, 15.0, nyq * 0.80))
    sos = butter(order, fc / nyq, btype="high", output="sos")
    return sosfilt(sos, sig, axis=0).astype(np.float32)


def _bandpass_audio(sig: np.ndarray, lo_hz: float, hi_hz: float, sr: int, order: int = 2) -> np.ndarray:
    nyq = 0.5 * sr
    lo = float(np.clip(lo_hz, 20.0, nyq * 0.85))
    hi = float(np.clip(hi_hz, lo + 100.0, nyq * 0.96))
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfilt(sos, sig, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Built-in reference audio synthesizer (when clip has no audio or is synthetic)
# ---------------------------------------------------------------------------
def synthesize_test_audio(duration_s: float, sample_rate: int = 44100,
                          start_s: float = 0.0, synth_type: int = 0,
                          seed: int = 20240101, kind: int | None = None,
                          fps: float = 29.97, **kwargs) -> np.ndarray:
    """
    Generate a clean stereo test audio signal `(N, 2)` in `float32` so any project (including
    built-in test patterns, still images, or silent clips) has rich, pitch-sensitive and
    transient-rich audio to make wow & flutter, dropouts, Hi-Fi unlock, and hum unmistakable.

    synth_type:
      0 = Warm Retro Synth Pad + Arpeggio + Crisp Transients (ideal for hearing pitch wow, azimuth & dropouts)
      1 = Broadcast Reference Calibration Tone (1 kHz L / 440 Hz R + 8 kHz pilot)
      2 = Speech-like Formant Cadence +Metronome Ticks (ideal for lip-sync / transient fault checks)
    """
    if kind is not None:
        synth_type = int(kind)
    n = max(1, int(round(duration_s * sample_rate)))
    t = start_s + np.arange(n, dtype=np.float64) / float(sample_rate)

    if synth_type == 1:
        # Broadcast calibration tone: 1000 Hz + 440 Hz + 6000 Hz high-frequency probe (to hear azimuth/LP cutoff)
        left = (0.42 * np.sin(2.0 * np.pi * 1000.0 * t)
                + 0.14 * np.sin(2.0 * np.pi * 6000.0 * t)
                + 0.06 * np.sin(2.0 * np.pi * 10000.0 * t))
        right = (0.42 * np.sin(2.0 * np.pi * 440.0 * t)
                 + 0.14 * np.sin(2.0 * np.pi * 6000.0 * t)
                 + 0.06 * np.sin(2.0 * np.pi * 10000.0 * t))
        return np.stack([left, right], axis=-1).astype(np.float32)

    if synth_type == 2:
        # Formant syllables + high-frequency sibilance / harmonic overtones + 2 Hz frame-sync clicks
        syl_env = np.clip(np.sin(2.0 * np.pi * 1.75 * t) * 1.4, 0.0, 1.0) ** 1.5
        f0 = 135.0 + 22.0 * np.sin(2.0 * np.pi * 0.45 * t)
        phase0 = 2.0 * np.pi * np.cumsum(f0) / float(sample_rate)
        glottal = (0.45 * np.sin(phase0)
                   + 0.32 * np.sin(2.0 * phase0)
                   + 0.22 * np.sin(5.0 * phase0)
                   + 0.16 * np.sin(9.0 * phase0)
                   + 0.12 * np.sin(18.0 * phase0)
                   + 0.16 * np.sin(2.0 * np.pi * 9800.0 * t)
                   + 0.14 * np.sin(2.0 * np.pi * 12800.0 * t))
        voice = syl_env * glottal * 0.42
        # Crisp 2 Hz metronome tick for A/V sync verification
        tick_phase = np.mod(t, 0.5)
        tick = np.exp(-tick_phase / 0.008) * (
            0.34 * np.sin(2.0 * np.pi * 2400.0 * t)
            + 0.22 * np.sin(2.0 * np.pi * 11200.0 * t)
        )
        left = voice + tick
        right = np.roll(voice, int(0.0004 * sample_rate)) + tick * 0.9
        return np.clip(np.stack([left, right], axis=-1), -0.95, 0.95).astype(np.float32)

    # Default (0): Warm Retro Synth Chord (Fmaj7 -> Em7 progression) + Arpeggio + Hi-Hat transients
    prog_step = (np.floor(t / 2.0).astype(np.int32)) % 2
    # Chord frequencies: Fmaj7 (174.61, 220.0, 261.63, 329.63) / Em7 (164.81, 196.0, 246.94, 293.66)
    f1 = np.where(prog_step == 0, 174.61, 164.81)
    f2 = np.where(prog_step == 0, 220.00, 196.00)
    f3 = np.where(prog_step == 0, 261.63, 246.94)
    f4 = np.where(prog_step == 0, 329.63, 293.66)

    def _rich_voice(freq: np.ndarray, detune: float = 1.0) -> np.ndarray:
        ph = 2.0 * np.pi * np.cumsum(freq * detune) / float(sample_rate)
        return (0.55 * np.sin(ph)
                + 0.25 * np.sin(2.0 * ph + 0.3)
                + 0.14 * np.sin(3.0 * ph + 0.7)
                + 0.08 * np.sin(4.0 * ph))

    pad_l = 0.18 * (_rich_voice(f1, 0.999) + _rich_voice(f3, 1.001) + 0.7 * _rich_voice(f4, 0.998))
    pad_r = 0.18 * (_rich_voice(f2, 1.001) + _rich_voice(f4, 0.999) + 0.7 * _rich_voice(f3, 1.002))

    # 4 Hz arpeggio notes with crisp attack and high harmonics (up to 9 kHz so bandwidth/azimuth changes shine)
    arp_notes = np.array([523.25, 659.25, 783.99, 987.77, 1318.51, 987.77, 783.99, 659.25], dtype=np.float64)
    note_idx = (np.floor(t * 4.0).astype(np.int32)) % len(arp_notes)
    f_arp = arp_notes[note_idx]
    note_t = np.mod(t, 0.25)
    arp_env = (1.0 - np.exp(-note_t / 0.004)) * np.exp(-note_t / 0.11)
    ph_arp = 2.0 * np.pi * np.cumsum(f_arp) / float(sample_rate)
    arp_sig = arp_env * (0.45 * np.sin(ph_arp)
                         + 0.22 * np.sin(2.0 * ph_arp)
                         + 0.12 * np.sin(4.0 * ph_arp)
                         + 0.08 * np.sin(7.0 * ph_arp))
    pan = 0.5 + 0.35 * np.sin(2.0 * np.pi * 0.5 * t)

    # Subtle 2 Hz brushed hi-hat transient (for hearing high-frequency loss above 5 kHz)
    rng = np.random.default_rng(int(seed) ^ 0x5A17C3)
    hat_t = np.mod(t, 0.5)
    hat_env = np.exp(-hat_t / 0.018) * 0.10
    white = rng.standard_normal(n)
    # Simple first-difference highpass for crispness
    hat_hp = np.diff(white, prepend=white[0]) * hat_env

    left = pad_l + arp_sig * pan + hat_hp
    right = pad_r + arp_sig * (1.0 - pan) + hat_hp * 0.92
    return np.clip(np.stack([left, right], axis=-1), -0.92, 0.92).astype(np.float32)


# ---------------------------------------------------------------------------
# Main AudioSimulator
# ---------------------------------------------------------------------------
class AudioSimulator:
    """
    Simulates analog & digital-tape audio degradation in strict lockstep with the video chain.
    """

    MODE_NAMES = {
        0: "auto",
        1: "vhs_linear",
        2: "vhs_hifi",
        3: "hi8_afm",
        4: "minidv_pcm",
        5: "vcd_mp2",
        6: "rf_pal_d_mono",
        7: "rf_tv_dx_tuner",
        8: "film_optical",
        9: "streaming_lowbitrate",
        10: "broadcast_studio",
        11: "perceptual_codec",
        12: "telecom_voip",
        13: "mobile_speaker_earpiece",
        14: "mobile_ringtone_midi",
        15: "system_player_chain",
    }

    def __init__(self, params_or_renderer: Any, fps: float = 29.97, seed: int = 20240101,
                 std_name: str = "ntsc", sample_rate: int = 44100):
        self._renderer = None
        self._tape_stage = None
        if hasattr(params_or_renderer, "params") and hasattr(params_or_renderer, "fps"):
            r = params_or_renderer
            self._renderer = r
            if getattr(r, "chain", None) is None and hasattr(r, "_build_chain"):
                r._build_chain()
            self._tape_stage = (getattr(r, "stage", None) or {}).get("tape")
            raw_params = r.params or {}
            fps = float(getattr(r, "fps", fps))
            seed = int(getattr(r, "seed", seed))
            std_obj = getattr(r, "std", None)
            if std_obj is not None and hasattr(std_obj, "name"):
                std_name = str(std_obj.name).lower()
        else:
            raw_params = params_or_renderer or {}
        _known_layers = {"camera", "film", "format", "codec", "connector", "rf", "tape", "player", "display", "overlay", "audio"}
        if isinstance(raw_params, dict) and "audio" not in raw_params and raw_params and not any(k in _known_layers for k in raw_params):
            self.params = {"audio": dict(raw_params)}
        else:
            self.params = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (raw_params or {}).items()}
        self.spec = AudioSpec.from_params(self.params)
        self.fps = float(max(1.0, fps))
        self.seed = int(seed)
        self.std_name = str(std_name).lower()
        self.sample_rate = int(sample_rate)

    def resolve_mode_name(self) -> str:
        m = int(self.spec.mode)
        if m in self.MODE_NAMES and m > 0:
            return self.MODE_NAMES[m]
        # Auto-resolve from film, mobile/streaming, codec, connector & tape layers
        film_cfg = self.params.get("film") or {}
        cam_cfg = self.params.get("camera") or {}
        if float(film_cfg.get("enabled", 0.0)) > 0.5 or float(cam_cfg.get("film_enabled", 0.0)) > 0.5:
            return "film_optical"
        codec_cfg = self.params.get("codec") or {}
        if (float(codec_cfg.get("stream_enabled", 0.0)) > 0.5
                or float(cam_cfg.get("ldme_mobile_enabled", 0.0)) > 0.5
                or float(codec_cfg.get("ldme_transfer_enabled", 0.0)) > 0.5):
            return "streaming_lowbitrate"
        if int(round(float(codec_cfg.get("kind", 0.0)))) == 2:
            return "vcd_mp2"
        if float(codec_cfg.get("dv_enabled", 0.0)) > 0.5:
            return "minidv_pcm"
        conn_cfg = self.params.get("connector") or {}
        tape_cfg = self.params.get("tape") or {}
        if (float(conn_cfg.get("rf_present", 0.0)) > 0.5
                and float(conn_cfg.get("av_present", 0.0)) > 0.5
                and float(tape_cfg.get("dropout_rate", 0.35)) <= 1e-4
                and float(tape_cfg.get("head_switching", 0.45)) <= 1e-4):
            return "rf_pal_d_mono"
        return "vhs_hifi"

    def _frame_params_at(self, frame_index: int, keyframes=None, environment=None,
                         ranges: dict | None = None, layer_of=None) -> dict:
        """Resolve per-frame parameters including environment & keyframe tracks."""
        if keyframes is None and environment is None:
            return self.params
        p = {lay: dict(vals) for lay, vals in self.params.items()}
        t = frame_index / self.fps
        if environment is not None and ranges is not None and layer_of is not None:
            from ..core.environment import apply_environment
            p, _ = apply_environment(p, environment, ranges, layer_of)
        if keyframes is not None and ranges is not None:
            p, _ = keyframes.apply(p, t, ranges)
        return p

    def render_window(self, clean_stereo: np.ndarray, start_s: float = 0.0, duration_s: float | None = None,
                      tape_stage=None, tape_plan=None, keyframes=None, environment=None,
                      ranges: dict | None = None, layer_of=None) -> AudioRun:
        """
        Process a stereo audio buffer `(N, 2)` spanning `[start_s, start_s + duration_s]`
        with exact frame-by-frame and scanline-by-scanline fault synchronization to `tape_stage`.
        """
        sr = self.sample_rate
        clean = np.asarray(clean_stereo, dtype=np.float32)
        if clean.ndim == 1:
            clean = np.stack([clean, clean], axis=-1)
        if duration_s is None:
            duration_s = max(1.0 / self.fps, len(clean) / float(sr))
        n_samples = max(1, int(round(duration_s * sr)))
        if len(clean) < n_samples:
            pad = np.zeros((n_samples - len(clean), 2), dtype=np.float32)
            clean = np.concatenate([clean, pad], axis=0)
        elif len(clean) > n_samples:
            clean = clean[:n_samples]
        clean = clean.copy()

        mode_name = self.resolve_mode_name()
        if not self.spec.enabled:
            return AudioRun(
                samples=np.clip(clean * self.spec.master_gain, -1.0, 1.0),
                clean_samples=clean,
                sample_rate=sr,
                start_s=start_s,
                duration_s=duration_s,
                resolved_mode="bypass",
                frame_faults=[],
                summary={"mode": "bypass", "fault_frames": 0},
            )

        if tape_stage is None:
            tape_stage = self._tape_stage
        start_frame = max(0, int(np.floor(start_s * self.fps + 1e-6)))
        end_frame = max(start_frame, int(np.ceil((start_s + duration_s) * self.fps - 1e-6)) - 1)
        if tape_plan is not None:
            plan = tape_plan
            H_vid = max(2, int(getattr(getattr(tape_stage, "ctx", None), "height", 240) or 240))
        elif tape_stage is not None:
            tape_stage._ensure(end_frame + 2)
            plan = tape_stage.plan
            H_vid = max(2, int(tape_stage.ctx.height))
        else:
            plan = None
            H_vid = 240

        # Time array and continuous frame/row coordinates for every audio sample
        t_arr = start_s + np.arange(n_samples, dtype=np.float64) / float(sr)
        frame_float = t_arr * self.fps
        frame_idx_arr = np.maximum(0, np.floor(frame_float).astype(np.int32))
        row_float = (frame_float - frame_idx_arr) * H_vid
        row_idx_arr = np.clip(np.floor(row_float).astype(np.int32), 0, H_vid - 1)

        # Per-sample envelope arrays (built from frame & scanline video events)
        pitch_shift_rate = np.zeros(n_samples, dtype=np.float64)   # fractional speed delta (dv/v)
        dropout_env = np.zeros(n_samples, dtype=np.float32)        # 0..1.5 oxide dropout / clog / crease
        crease_thump = np.zeros(n_samples, dtype=np.float32)       # low-frequency head-bump impulse
        hs_pulse_env = np.zeros(n_samples, dtype=np.float32)       # head-switch band envelope
        rf_buzz_env = np.zeros(n_samples, dtype=np.float32)        # RF intercarrier vision-into-sound buzz
        rf_static_env = np.zeros(n_samples, dtype=np.float32)      # RF sync-slip FM static burst
        hum_env = np.zeros(n_samples, dtype=np.float32)            # AV ground-loop hum amplitude
        hdmi_mute_env = np.zeros(n_samples, dtype=np.float32)      # 1.0 when HDMI link drops
        hdmi_ber_env = np.zeros(n_samples, dtype=np.float32)       # HDMI bit-error rate
        crack_clicks = np.zeros(n_samples, dtype=np.float32)       # sharp impulse spikes from base_cracking
        ctl_unlock_env = np.zeros(n_samples, dtype=np.float32)     # base_breakage CTL servo unlock
        shuttle_env = np.zeros(n_samples, dtype=np.float32)        # shuttle track-crossing chop mask
        speed_ratio_arr = np.ones(n_samples, dtype=np.float32)     # 1.0=SP, 0.5=LP, 0.333=EP

        frame_faults: list[AudioFaultFrame] = []

        # Pre-index dropouts by frame for fast lookup
        dropouts_by_frame: dict[int, list] = {}
        if plan is not None and plan.dropouts:
            for ev in plan.dropouts:
                if ev.start_frame + ev.frames < start_frame or ev.start_frame > end_frame:
                    continue
                for f in range(max(start_frame, ev.start_frame),
                               min(end_frame + 1, ev.start_frame + ev.frames)):
                    dropouts_by_frame.setdefault(f, []).append(ev)

        # Pre-index stiction episodes by frame
        stiction_by_frame: dict[int, list] = {}
        if plan is not None and getattr(plan, "stiction", None):
            for sev in plan.stiction:
                f0 = int(sev.start_frame)
                hold_f = int(getattr(sev, "hold_frames", getattr(sev, "frames", 4)))
                rel_f_cnt = int(getattr(sev, "release_frames", max(2, hold_f // 2)))
                f1 = int(f0 + hold_f + rel_f_cnt)
                if f1 < start_frame or f0 > end_frame:
                    continue
                for f in range(max(start_frame, f0), min(end_frame + 1, f1)):
                    stiction_by_frame.setdefault(f, []).append(sev)

        # Populate per-frame and per-scanline synchronized fault envelopes
        from ..codecs.tape_speed import by_index
        from ..codecs.connectors import CONNECTORS, impedance, reflection_coefficient

        for f in range(start_frame, end_frame + 1):
            # Sample range belonging to frame `f`
            s0 = int(round((f / self.fps - start_s) * sr))
            s1 = int(round(((f + 1) / self.fps - start_s) * sr))
            s0_c = max(0, min(n_samples, s0))
            s1_c = max(0, min(n_samples, s1))
            if s1_c <= s0_c:
                continue
            sl = slice(s0_c, s1_c)
            n_f = s1_c - s0_c
            u_f = np.linspace(0.0, 1.0, n_f, endpoint=False, dtype=np.float32)
            rows_f = np.clip((u_f * H_vid).astype(np.int32), 0, H_vid - 1)

            f_params = self._frame_params_at(f, keyframes, environment, ranges, layer_of)
            tape_cfg = f_params.get("tape") or {}
            conn_cfg = f_params.get("connector") or {}
            rf_cfg = f_params.get("rf") or {}
            player_cfg = f_params.get("player") or {}
            aud_spec = AudioSpec.from_params(f_params)

            ff = AudioFaultFrame(frame_index=f, time_s=f / self.fps)

            # 1. Tape Speed (SP / LP / EP + sp_lp_mix transition)
            sp_obj = by_index(tape_cfg.get("tape_speed", 0.0))
            sp_lp_mix = float(np.clip(tape_cfg.get("sp_lp_mix", 0.0), 0.0, 1.0))
            eff_ratio = float(sp_obj.ratio) * (1.0 - 0.5 * sp_lp_mix)
            speed_ratio_arr[sl] = eff_ratio

            # 2. Capstan Wow & Flutter coupled directly to TapePlan line_phase & frame_dy
            wf_amt = aud_spec.wow_flutter * (1.4 - 0.85 * float(np.clip(tape_cfg.get("tape_stability", 0.55), 0.0, 1.0)))
            wf_amt /= max(0.33, eff_ratio ** 0.5)
            p_dev = 0.0
            if plan is not None and plan.prepared_lines > f * H_vid:
                l0 = f * H_vid
                l1 = min(len(plan.line_phase), (f + 1) * H_vid)
                if l1 > l0:
                    lp_seg = plan.line_phase[l0:l1]
                    # Resample scanline phase velocity onto the audio samples of this frame
                    lp_interp = np.interp(u_f * (len(lp_seg) - 1), np.arange(len(lp_seg)), lp_seg)
                    pitch_shift_rate[sl] += wf_amt * 0.0022 * lp_interp
                    p_dev += float(np.mean(np.abs(lp_interp))) * wf_amt * 0.22
            if plan is not None and f < len(plan.frame_dy):
                f_dy = float(plan.frame_dy[f])
                pitch_shift_rate[sl] += wf_amt * 0.0035 * f_dy * np.sin(2.0 * np.pi * u_f)
                p_dev += abs(f_dy) * wf_amt * 0.35

            # Drum speed bearing error (`drum_speed` != 1.0)
            drum_sp = float(tape_cfg.get("drum_speed", 1.0))
            if abs(drum_sp - 1.0) > 1e-4:
                pitch_shift_rate[sl] += (drum_sp - 1.0) * 0.35
                p_dev += abs(drum_sp - 1.0) * 35.0

            # 2b. Module 4: Mechanical Transport & Servo State Machine Audio Coupling
            #     (Belt slip droop, pinch-roller eccentricity wow, reel-pack tension ripple,
            #      servo hunting, Pause/Stop mute, and Record-Pause splice pop)
            from ..codecs.transport_servo import TransportServoSpec, evaluate_transport_audio_mod
            ts_spec = TransportServoSpec.from_params(f_params)
            if ts_spec.is_active():
                ts_aud = evaluate_transport_audio_mod(tape_cfg, f, u_f, fps=self.fps, seed=self.seed)
                pitch_shift_rate[sl] += ts_aud["dv_over_v"]
                p_dev += float(ts_aud["pitch_pct"])
                if np.any(ts_aud["drop_gate"] > 1e-4):
                    dropout_env[sl] = np.maximum(dropout_env[sl], ts_aud["drop_gate"])
                    ff.dropout_level = max(ff.dropout_level, float(np.max(ts_aud["drop_gate"])))
                mute_arr = np.clip(1.0 - ts_aud["amp_gain"], 0.0, 1.0)
                if np.any(mute_arr > 1e-4):
                    hdmi_mute_env[sl] = np.maximum(hdmi_mute_env[sl], mute_arr)

            # 3. Capstan Stiction Grab & Release (`TapePlan.stiction`)
            if f in stiction_by_frame and aud_spec.stiction_pitch > 1e-4:
                for sev in stiction_by_frame[f]:
                    hold_f = float(max(1, getattr(sev, "hold_frames", getattr(sev, "frames", 4))))
                    rel_f_cnt = float(max(1, getattr(sev, "release_frames", max(2.0, hold_f * 0.5))))
                    rel_f = (f + u_f - sev.start_frame) / hold_f
                    # During grab (0 <= rel_f <= 1): tape slows down sharply (negative pitch shift)
                    # Just after release (1 < rel_f <= 1 + rel_f_cnt/hold_f): capstan spring-back overshoots pitch upward!
                    grab_mask = (rel_f >= 0.0) & (rel_f <= 1.0)
                    rel_mask = (rel_f > 1.0) & (rel_f <= 1.0 + rel_f_cnt / hold_f)
                    grab_env = np.where(grab_mask, np.maximum(0.0, np.sin(np.pi * np.clip(rel_f, 0.0, 1.0))) ** 1.2, 0.0)
                    overshoot_env = np.where(
                        rel_mask,
                        0.60 * np.exp(-(rel_f - 1.0) / max(0.15, 0.35 * rel_f_cnt / hold_f))
                        * np.sin(2.0 * np.pi * (rel_f - 1.0) / max(0.25, 0.75 * rel_f_cnt / hold_f)),
                        0.0,
                    )
                    st_depth = float(getattr(sev, "peak_px", getattr(sev, "magnitude", 24.0))) / 32.0 * aud_spec.stiction_pitch
                    pitch_shift_rate[sl] += (-0.068 * grab_env + 0.036 * overshoot_env) * st_depth
                    dropout_env[sl] = np.maximum(dropout_env[sl], (0.45 * grab_env * st_depth).astype(np.float32))
                    ff.stiction_grab = max(ff.stiction_grab, float(np.max(grab_env + np.abs(overshoot_env)) * st_depth))
                    p_dev += float(np.max(np.abs(-6.8 * grab_env + 3.6 * overshoot_env)) * st_depth)

            ff.pitch_deviation_pct = float(p_dev)

            # 4. Dropouts (`TapePlan.dropouts`), Head Clog (`clog_rows`), Creases (`creases`), Binder Loss
            do_sens = aud_spec.dropout_sensitivity
            if do_sens > 1e-4:
                # 4a. Discrete DropoutEvents mapped to exact scanline timestamp within the frame
                if f in dropouts_by_frame:
                    for ev in dropouts_by_frame[f]:
                        r0_frac = max(0.0, (ev.row - 2.0) / float(H_vid))
                        r1_frac = min(1.0, (ev.row + ev.rows + 4.0) / float(H_vid))
                        # Even a 2-scanline video dropout excites the audio head / FM discriminator for ~2-6 ms
                        min_span = max(0.06, (ev.rows / float(H_vid)) * 2.5)
                        mid_frac = 0.5 * (r0_frac + r1_frac)
                        half_w = max(0.5 * (r1_frac - r0_frac), 0.5 * min_span)
                        dist = np.abs(u_f - mid_frac) / max(1e-4, half_w)
                        win = np.clip(1.0 - dist * dist, 0.0, 1.0)
                        sev_val = float(ev.severity) * do_sens
                        dropout_env[sl] = np.maximum(dropout_env[sl], (win * sev_val).astype(np.float32))

                # 4b. Spatially fixed Head Clog (`plan.clog_rows`)
                if plan is not None and plan.clog_rows.size == H_vid and float(tape_cfg.get("head_clog", 0.0)) > 1e-4:
                    wobble = 0.75 + 0.45 * float(np.sin(f * 0.21 + float(plan.clog_rows.sum())))
                    clog_r = np.clip(plan.clog_rows[rows_f] * wobble * do_sens, 0.0, 1.4)
                    dropout_env[sl] = np.maximum(dropout_env[sl], clog_r.astype(np.float32))

                # 4c. Tape Creases (`plan.creases`) -> low-frequency mechanical head thump + contact loss
                if plan is not None and plan.creases and float(tape_cfg.get("crease_rate", 0.0)) > 1e-4:
                    for cr in plan.creases:
                        t_cr = f / max(1e-6, cr.period_frames) + cr.phase
                        pos_row = (0.5 + 0.5 * np.sin(t_cr * 2.0 * np.pi)) * (H_vid + 2.0 * cr.width) - cr.width + cr.offset
                        if -cr.width <= pos_row <= H_vid + cr.width:
                            center_u = float(np.clip(pos_row / float(H_vid), 0.02, 0.98))
                            w_u = max(0.04, (cr.width / float(H_vid)) * 1.5)
                            cr_env = np.exp(-0.5 * ((u_f - center_u) / w_u) ** 2) * float(cr.depth) * do_sens
                            dropout_env[sl] = np.maximum(dropout_env[sl], (cr_env * 0.85).astype(np.float32))
                            # Mechanical head-to-tape bump wave (damped 75 Hz thump)
                            thump_wave = cr_env * np.sin(2.0 * np.pi * (u_f - center_u) / max(0.03, w_u * 1.2))
                            crease_thump[sl] += (thump_wave * 0.45).astype(np.float32)

                # 4d. Binder Loss / Debris / Demagnetization
                binder = float(tape_cfg.get("binder_loss", 0.0))
                debris = float(tape_cfg.get("debris", 0.0))
                demag = float(tape_cfg.get("demagnetization", 0.0))
                if binder > 1e-4 or debris > 1e-4 or demag > 1e-4:
                    rng_bd = np.random.default_rng((self.seed * 131 + f * 997) & 0xFFFFFFFF)
                    patch = (0.35 * binder + 0.25 * demag) * (0.6 + 0.4 * np.sin(2.0 * np.pi * u_f + f * 0.37))
                    if debris > 1e-4:
                        n_strikes = int(rng_bd.poisson(max(1.0, debris * 6.0)))
                        for _ in range(n_strikes):
                            cu = float(rng_bd.uniform(0.0, 1.0))
                            patch = np.maximum(patch, debris * 0.75 * np.exp(-0.5 * ((u_f - cu) / 0.015) ** 2))
                    dropout_env[sl] = np.maximum(dropout_env[sl], (patch * do_sens).astype(np.float32))

            ff.dropout_level = float(np.max(dropout_env[sl])) if n_f > 0 else 0.0

            # 5. Base Cracking (`base_cracking`) & Base Breakage (`base_breakage`)
            cracking = float(tape_cfg.get("base_cracking", 0.0))
            if cracking > 1e-4:
                rng_crk = np.random.default_rng((self.seed * 7919 + f * 2654435761) & 0xFFFFFFFF)
                n_cracks = int(rng_crk.poisson(cracking * 3.5))
                for _ in range(n_cracks):
                    pos_s = int(rng_crk.integers(0, max(1, n_f - 12)))
                    pol = 1.0 if rng_crk.random() < 0.65 else -0.85
                    amp_c = float(pol * (0.35 + 0.55 * cracking))
                    # Sharp 0.4 ms magnetic-oxide crack impulse followed by brief exponential tail
                    tail_len = min(24, n_f - pos_s)
                    crack_clicks[s0_c + pos_s : s0_c + pos_s + tail_len] += (
                        amp_c * np.exp(-np.arange(tail_len, dtype=np.float32) / 4.5)
                        * np.cos(np.arange(tail_len, dtype=np.float32) * 0.9)
                    )
                    ff.crack_spike = max(ff.crack_spike, abs(amp_c))

            breakage = float(tape_cfg.get("base_breakage", 0.0))
            if breakage > 1e-4:
                # Severe physical tape snap / CTL loss episode
                brk_w = breakage * (0.65 + 0.35 * float(np.sin(f * 0.43)))
                ctl_unlock_env[sl] = np.maximum(ctl_unlock_env[sl], brk_w)
                dropout_env[sl] = np.maximum(dropout_env[sl], (brk_w * 0.95).astype(np.float32))
                pitch_shift_rate[sl] += -0.12 * brk_w * np.cos(2.0 * np.pi * u_f * 2.0)

            # 6. Shuttle (`shuttle_speed`, `shuttle_stale`)
            sh_speed = float(tape_cfg.get("shuttle_speed", 0.0))
            sh_stale = float(tape_cfg.get("shuttle_stale", 0.0))
            if sh_speed > 1e-4 and sh_stale > 1e-4:
                # During shuttle, heads cross multiple helical tracks per field, producing rhythmic
                # track-crossing bursts at `max(2, round(sh_speed))` bands per frame
                n_bands = max(2.0, min(10.0, sh_speed * 1.5))
                chop_wave = 0.5 + 0.5 * np.sin(2.0 * np.pi * n_bands * u_f + f * 1.1)
                sh_mask = np.clip((chop_wave - (1.0 - sh_stale)) / max(0.1, sh_stale), 0.0, 1.0)
                shuttle_env[sl] = sh_mask.astype(np.float32)
                # Varispeed pitch offset during analog shuttle
                sh_dir = -1.0 if float(tape_cfg.get("shuttle_reverse", 0.0)) > 0.5 else 1.0
                pitch_shift_rate[sl] += sh_dir * min(0.45, 0.08 * sh_speed) * (0.7 + 0.3 * chop_wave)
                ff.shuttle_chop = float(np.mean(sh_mask))

            # 7. Head-Switching Buzz & Tracking Error (`head_switching`, `tracking_error`, `tracking_noise_snow`)
            hs_amp_frame = float(plan.head_switch_amp[f]) if (plan is not None and f < len(plan.head_switch_amp)) else float(tape_cfg.get("head_switching", 0.45))
            hs_lines = float(tape_cfg.get("head_switch_lines", 12.0))
            tr_err = float(tape_cfg.get("tracking_error", 0.0)) + 0.6 * float(tape_cfg.get("tracking_noise_snow", 0.0))
            buzz_gain = aud_spec.head_switch_buzz * (0.45 * hs_amp_frame + 0.85 * tr_err + 0.30 * sp_lp_mix)
            if buzz_gain > 1e-4:
                # Video fields switch twice per frame (at u ~ 0.48..0.50 and u ~ 0.96..1.00)
                band_frac = max(0.025, min(0.18, hs_lines / max(1.0, float(H_vid) * 0.5)))
                in_sw1 = np.clip(1.0 - np.abs(u_f - (0.50 - 0.5 * band_frac)) / band_frac, 0.0, 1.0)
                in_sw2 = np.clip(1.0 - np.abs(u_f - (1.00 - 0.5 * band_frac)) / band_frac, 0.0, 1.0)
                sw_pulse = np.maximum(in_sw1, in_sw2)
                # Continuous tracking error also bleeds a sawtooth field-rate buzz across the whole field
                hs_pulse_env[sl] = (buzz_gain * (0.75 * sw_pulse + 0.25 * tr_err)).astype(np.float32)
                ff.head_switch_buzz = float(buzz_gain)

            # 8. AV Connector Ground-Loop Hum (`av_present`, `av_ground_loop`, `av_moisture`, `av_corrosion`)
            av_on = float(conn_cfg.get("av_present", 0.0)) > 0.5
            av_gnd = float(conn_cfg.get("av_ground_loop", 0.0))
            av_m = float(conn_cfg.get("av_moisture", 0.0))
            av_c = float(conn_cfg.get("av_corrosion", 0.0))
            eff_hum = aud_spec.ground_hum * (
                (av_gnd + 0.35 * av_m + 0.45 * av_c) if av_on else (0.65 * av_gnd + 0.25 * av_m + 0.30 * av_c)
            )
            if eff_hum > 1e-4:
                hum_env[sl] = float(np.clip(eff_hum, 0.0, 1.2))
                ff.ground_hum = float(hum_env[s0_c])

            # 8b. Shared Power Bus Coupling (Battery Load Sag Pitch Droop, Startup Spin-Up & PSU Ripple/EMI Hum)
            if (float(conn_cfg.get("psu_ripple", 0.0)) > 1e-4
                    or float(conn_cfg.get("battery_sag", 0.0)) > 1e-4
                    or float(conn_cfg.get("battery_level", 1.0)) < 0.999
                    or float(conn_cfg.get("startup_surge", 0.0)) > 1e-4
                    or float(conn_cfg.get("emi_crosstalk", 0.0)) > 1e-4):
                from ..codecs.power_bus import evaluate_power_bus
                pb = evaluate_power_bus(
                    f_params, frame_index=f, fps=self.fps, height=H_vid,
                    row_luma=None, stiction_loss=float(ff.stiction_grab),
                    seed=self.seed,
                )
                dp_rate = float(pb.audio_pitch_ratio) - 1.0
                if abs(dp_rate) > 1e-6:
                    pitch_shift_rate[sl] += dp_rate
                    ff.pitch_deviation_pct = float(ff.pitch_deviation_pct + dp_rate * 100.0)
                if pb.audio_hum_amp > 1e-5:
                    hum_env[sl] = np.clip(hum_env[sl] + float(pb.audio_hum_amp), 0.0, 1.5).astype(np.float32)
                    ff.ground_hum = float(hum_env[s0_c])

            # 8c. Analog Video Switcher Crosspoint Click & Hum Coupling (`switcher_mode`)
            if int(round(float(conn_cfg.get("switcher_mode", 0.0)))) > 0:
                from ..codecs.switcher import AnalogSwitcherSpec, compute_switcher_audio_transient
                sw_spec = AnalogSwitcherSpec.from_params(conn_cfg)
                t_f_samples = (f / self.fps) + u_f.astype(np.float64) / self.fps
                sw_aud = compute_switcher_audio_transient(sw_spec, t_f_samples, seed=self.seed)
                if np.any(np.abs(sw_aud["click_pop"]) > 1e-5):
                    crack_clicks[sl] = crack_clicks[sl] + sw_aud["click_pop"]
                if sw_aud["hum_add"] > 1e-5:
                    hum_env[sl] = np.clip(hum_env[sl] + float(sw_aud["hum_add"]), 0.0, 1.5).astype(np.float32)
                    ff.ground_hum = float(hum_env[s0_c])

            # 8d. Broadcast Patchbay & Distribution Amplifier Coupling (`da_enabled`, `patch_crosstalk`, `patch_contact_aging`)
            from ..codecs.patchbay_da import PatchbayDASpec, compute_patchbay_audio_transient
            da_spec = PatchbayDASpec.from_params(conn_cfg)
            if da_spec.is_active():
                t_f_samples = (f / self.fps) + u_f.astype(np.float64) / self.fps
                da_aud = compute_patchbay_audio_transient(da_spec, t_f_samples, seed=self.seed)
                if np.any(np.abs(da_aud["crackle"]) > 1e-5):
                    crack_clicks[sl] = crack_clicks[sl] + da_aud["crackle"]
                    ff.crack_spike = float(max(ff.crack_spike, float(np.max(np.abs(da_aud["crackle"])))))
                if np.any(np.abs(da_aud["crosstalk_tone"]) > 1e-5):
                    crack_clicks[sl] = crack_clicks[sl] + da_aud["crosstalk_tone"]
                if da_aud["hum_add"] > 1e-5:
                    hum_env[sl] = np.clip(hum_env[sl] + float(da_aud["hum_add"]), 0.0, 1.5).astype(np.float32)
                    ff.ground_hum = float(hum_env[s0_c])

            # 8e. Transfer & Dubbing Workstation Audio Sync Offset & Multi-Gen Hiss (`transfer_station_mode`)
            ply_cfg = f_params.get("player") or {}
            if int(round(float(ply_cfg.get("transfer_station_mode", 0.0)))) > 0:
                from ..codecs.transfer_workstation import TransferWorkstationSpec, compute_transfer_audio_effect
                tw_spec = TransferWorkstationSpec.from_params(ply_cfg)
                tw_aud = compute_transfer_audio_effect(tw_spec, self.sample_rate, s1_c - s0_c, seed=self.seed + f)
                if tw_aud["active"] and np.any(np.abs(tw_aud["noise_vec"]) > 1e-6):
                    crack_clicks[sl] = crack_clicks[sl] + tw_aud["noise_vec"]

            # 9. RF Coaxial Link & Damp Interface Sync-Slip Buzz (`rf_present`, `rf_moisture`, `rf_corrosion`, `rf`)
            rf_on = float(conn_cfg.get("rf_present", 0.0)) > 0.5
            rf_m = float(conn_cfg.get("rf_moisture", 0.0))
            rf_c = float(conn_cfg.get("rf_corrosion", 0.0))
            rf_dbm = float(rf_cfg.get("signal_dbm", -60.0))
            rf_weak = float(np.clip((-60.0 - rf_dbm) / 25.0, 0.0, 1.0))
            eff_rf_m = float(np.clip(rf_m + 0.45 * rf_c, 0.0, 1.0)) if (rf_on or rf_m > 1e-4 or rf_c > 1e-4) else 0.0
            if (eff_rf_m > 1e-4 or rf_weak > 1e-4) and aud_spec.rf_buzz > 1e-4:
                # Match `degrade_rf_signal` bistable sync slip (`f_mod = frame_index % 4`, slipped when `f_mod != 2`)
                f_mod = int(f) % 4
                slipped = (f_mod != 2) and (eff_rf_m >= 0.30)
                multi_band = (f_mod in (0, 3)) and slipped
                base_rf_buzz = aud_spec.rf_buzz * (0.55 * eff_rf_m + 0.45 * rf_weak)
                if slipped:
                    # On slipped frames with horizontal line-clamp surge bars (`sy_frac = 0.515, 0.692`),
                    # vision-into-sound intercarrier buzz spikes sharply at those scanline bands!
                    surge_env = np.ones_like(u_f) * 1.25
                    if not multi_band:
                        b1 = np.exp(-0.5 * ((u_f - 0.515) / 0.045) ** 2)
                        b2 = np.exp(-0.5 * ((u_f - 0.692) / 0.045) ** 2)
                        surge_env += 1.65 * (b1 + b2)
                    else:
                        # Multi-band chroma unlock frames (`f_mod in (0, 3)`)
                        b_top = np.exp(-0.5 * ((u_f - 0.09) / 0.05) ** 2)
                        b_bot = np.exp(-0.5 * ((u_f - 0.83) / 0.07) ** 2)
                        surge_env += 1.35 * (b_top + b_bot)
                    rf_buzz_env[sl] = (base_rf_buzz * surge_env).astype(np.float32)
                    rf_static_env[sl] = (base_rf_buzz * 0.65 * (0.4 + 0.6 * (surge_env - 1.0))).astype(np.float32)
                else:
                    # Frame `f_mod == 2`: V-sync top wrap-around & lower static bands (`0.66`, `0.80`)
                    sb = np.exp(-0.5 * ((u_f - 0.66) / 0.05) ** 2) + np.exp(-0.5 * ((u_f - 0.80) / 0.06) ** 2)
                    rf_buzz_env[sl] = (base_rf_buzz * (0.55 + 0.55 * sb)).astype(np.float32)
                    rf_static_env[sl] = (base_rf_buzz * 0.45 * sb).astype(np.float32)
                ff.rf_sync_buzz = float(np.max(rf_buzz_env[sl]))

            # 10. HDMI Digital Link Errors & Handshake Drop (`hdmi_present`, `hdmi_moisture`, `hdmi_corrosion`)
            hdmi_on = float(conn_cfg.get("hdmi_present", 0.0)) > 0.5
            hdmi_m = float(conn_cfg.get("hdmi_moisture", 0.0))
            hdmi_c = float(conn_cfg.get("hdmi_corrosion", 0.0))
            if hdmi_on and (hdmi_m > 1e-4 or hdmi_c > 1e-4):
                z_h = impedance(CONNECTORS["hdmi"], hdmi_m, hdmi_c)
                gamma_h = abs(reflection_coefficient(CONNECTORS["hdmi"].nominal_z, z_h))
                ber = float(np.clip(gamma_h * 3.4 + 0.42 * hdmi_c, 0.0, 1.0))
                if ber > 0.55:
                    # Complete link handshake drop -> NO SIGNAL (silent audio + brief unlock click at edge)
                    hdmi_mute_env[sl] = 1.0
                    ff.hdmi_drop = 1.0
                elif ber > 1e-4:
                    hdmi_ber_env[sl] = ber
                    ff.hdmi_drop = round(ber, 3)

            # Record Hi-Fi fallback fraction for this frame
            if mode_name == "vhs_hifi":
                unlk = np.clip((dropout_env[sl] + 0.6 * tr_err - aud_spec.hifi_unlock_thresh) / 0.22, 0.0, 1.0)
                ff.hifi_fallback = float(0.65 * np.max(unlk) + 0.35 * np.mean(unlk))

            # Subsystem 10: Per-Frame Bidirectional A/V Sync & Master Clock Telemetry
            av_map = compute_av_sync_frame_map(f, self.fps, f_params, seed=self.seed)
            ff.av_offset_ms = float(av_map["av_offset_ms"])
            ff.clock_drift_ms = float(av_map["clock_drift_ms"])
            ff.frame_dropped = bool(av_map["frame_dropped"])
            ff.edit_splice = bool(av_map["edit_splice"])

            frame_faults.append(ff)

        rng = np.random.default_rng((self.seed * 2654435761 + int(round(start_s * 1000))) & 0xFFFFFFFF)
        # ------------------------------------------------------------------
        # Step 0: 10-Subsystem Front-End (08 Ringtone/MIDI -> 06 Acoustic Space -> 01 Mic & Preamp -> 03 Digital Sampling)
        #         plus Broadcast Console AGC
        # ------------------------------------------------------------------
        # If mode explicitly selects one of the new modes (11..15), activate its default subsystem if not already set
        eff_params = {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.params.items()}
        aud_dict = dict(eff_params.get("audio") or {})
        if mode_name == "perceptual_codec" and int(round(float(aud_dict.get("pcodec_type", 0.0)))) == 0:
            aud_dict["pcodec_type"] = 1.0
        elif mode_name == "telecom_voip" and int(round(float(aud_dict.get("telecom_mode", 0.0)))) == 0:
            aud_dict["telecom_mode"] = 1.0
        elif mode_name == "mobile_speaker_earpiece" and int(round(float(aud_dict.get("mobile_spk_mode", 0.0)))) == 0:
            aud_dict["mobile_spk_mode"] = 2.0
        elif mode_name == "mobile_ringtone_midi" and int(round(float(aud_dict.get("ringtone_mode", 0.0)))) == 0:
            aud_dict["ringtone_mode"] = 1.0
        elif mode_name == "system_player_chain" and int(round(float(aud_dict.get("sys_audio_preset", 0.0)))) == 0:
            aud_dict["sys_audio_preset"] = 1.0
        eff_params["audio"] = aud_dict

        clean, front_sub_meta = run_audio_subsystems_front_end(
            clean, eff_params, sr=sr, seed=self.seed, start_s=start_s
        )
        clean, _ = self._apply_broadcast_console(clean, sr, rng, aud_spec)

        # ------------------------------------------------------------------
        # Step 1: Time-Domain Varispeed Resampling (Wow & Flutter + Stiction + Shuttle)
        # ------------------------------------------------------------------
        # Smooth the fractional speed deviation slightly (~1.5 ms) so frame boundaries are C1 continuous
        pitch_shift_rate = gaussian_filter1d(pitch_shift_rate, sigma=max(1.0, 0.0015 * sr))
        # Integrate instantaneous speed ratio `1 + pitch_shift_rate(t)` to get warped sample indices
        inst_speed = np.clip(1.0 + pitch_shift_rate, 0.25, 2.50)
        warped_pos = np.cumsum(inst_speed) - inst_speed[0]
        idx_float = np.clip(warped_pos, 0.0, float(n_samples - 1.001))
        idx0 = np.floor(idx_float).astype(np.int32)
        idx1 = np.minimum(idx0 + 1, n_samples - 1)
        frac = (idx_float - idx0)[:, None].astype(np.float32)
        sig = clean[idx0] * (1.0 - frac) + clean[idx1] * frac

        # ------------------------------------------------------------------
        # Step 2: Head Azimuth Misalignment (L/R micro-delay skew + HF comb/rolloff)
        # ------------------------------------------------------------------
        az = float(self.spec.azimuth_loss)
        if az > 1e-4 and mode_name in ("vhs_linear", "vhs_hifi", "hi8_afm"):
            # Azimuth tilt delays the right channel relative to left by up to 0.35 ms
            delay_samples = int(round(az * 0.00035 * sr))
            if delay_samples > 0 and delay_samples < n_samples:
                sig[delay_samples:, 1] = sig[:-delay_samples, 1]

        # ------------------------------------------------------------------
        # Step 3: Format-Specific Signal Path & Fault Coupling
        # ------------------------------------------------------------------
        rng = np.random.default_rng((self.seed * 2654435761 + int(round(start_s * 1000))) & 0xFFFFFFFF)
        mean_speed_ratio = float(np.clip(np.mean(speed_ratio_arr), 0.25, 1.0))

        # Pre-Tape RF Broadcast Intercarrier Tuner Cascade ("VHS 录像带电视杂录"):
        # When RF wireless reception is enabled (`rf.enabled > 0.5`) AND the recording medium
        # is a magnetic tape (`vhs_linear`, `vhs_hifi`, `hi8_afm`), the wireless TV broadcast
        # audio first passes through the VCR's FM intercarrier demodulator (where the FM
        # limiter + discriminator + 50us de-emphasis give audio an ~18-24 dB SNR advantage
        # over the AM-VSB video picture!) before being recorded onto the VHS tape track.
        rf_cfg_cascade = self.params.get("rf") or {}
        rf_enabled_cascade = float(rf_cfg_cascade.get("enabled", 0.0)) > 0.5
        rf_cascade_meta: dict = {}
        if rf_enabled_cascade and mode_name in ("vhs_linear", "vhs_hifi", "hi8_afm"):
            sig, rf_cascade_meta = self._apply_rf_broadcast_tuner_prestage(
                sig, t_arr, sr, n_samples, rng, rf_cfg_cascade
            )

        # Build the VHS Linear Track signal (used both in Mode 1 and as the fallback track in Mode 2)
        def _build_linear_track(inp: np.ndarray) -> np.ndarray:
            # Linear track cutoff scales directly with tape speed (SP=1.0, LP=0.5, EP=0.333) and azimuth loss
            fc_lin = min(6200.0, self.spec.linear_bandwidth) * (mean_speed_ratio ** 0.78) * (1.0 - 0.52 * az)
            lin = _highpass_audio(inp, 95.0, sr, order=2)
            lin = _lowpass_audio(lin, fc_lin, sr, order=6)
            # Right channel loses extra high frequencies under azimuth misalignment
            if az > 0.05:
                lin[:, 1] = _lowpass_audio(lin[:, 1:2], fc_lin * (1.0 - 0.35 * az), sr, order=2)[:, 0]
            # Magnetic oxide dropouts dull and attenuate the longitudinal track
            do_atten = np.clip(1.0 - 0.85 * dropout_env, 0.04, 1.0)[:, None]
            lin_muffled = _lowpass_audio(lin, max(450.0, fc_lin * 0.24), sr, order=2)
            drop_mix = np.clip(dropout_env, 0.0, 1.0)[:, None]
            lin = (lin * (1.0 - 0.72 * drop_mix) + lin_muffled * (0.72 * drop_mix)) * do_atten
            # Linear tape hiss (rises at slower tape speeds, roll-off above fc_lin)
            hiss_raw = rng.standard_normal((n_samples, 2)).astype(np.float32)
            hiss_shaped = _bandpass_audio(hiss_raw, 350.0, min(6200.0, fc_lin * 0.95), sr, order=3)
            hiss_amp = 0.024 * self.spec.hiss_floor / (mean_speed_ratio ** 0.45)
            return lin + hiss_shaped * hiss_amp

        if mode_name == "vhs_linear":
            sig = _build_linear_track(sig)
            if np.any(shuttle_env > 1e-4):
                # During analog shuttle, linear head reads high-pitched warble with guard-band dips
                sig = sig * (1.0 - 0.45 * shuttle_env[:, None])

        elif mode_name == "vhs_hifi":
            # Hi-Fi AFM track: wide 20 Hz - 18 kHz frequency response, compander breathing,
            # and automatic switching to the muffled Linear Track when FM carrier unlocks!
            hifi = _highpass_audio(sig, 25.0, sr, order=2)
            hifi = _lowpass_audio(hifi, 18000.0, sr, order=4)
            # Subtle compander noise-reduction breathing (hiss modulated by signal envelope)
            sig_env = gaussian_filter1d(np.mean(np.abs(hifi), axis=1), sigma=max(10, int(0.012 * sr)))
            fm_hiss = _bandpass_audio(rng.standard_normal((n_samples, 2)).astype(np.float32), 800.0, 15000.0, sr, order=2)
            hifi = hifi + fm_hiss * (0.0045 * self.spec.hiss_floor * (0.35 + 1.8 * sig_env))[:, None]

            # Evaluate FM carrier unlock envelope from dropouts + tracking error + shuttle
            unlock_drive = dropout_env + 0.85 * shuttle_env + 0.85 * ctl_unlock_env
            unlock_raw = np.clip((unlock_drive - self.spec.hifi_unlock_thresh) / 0.18, 0.0, 1.0)
            # Hold relay time constant (~8 ms attack/release) + derivative crackle burst at switching edges!
            unlock_env = gaussian_filter1d(unlock_raw, sigma=max(1.0, 0.006 * sr))
            switch_edge = np.abs(np.diff(unlock_env, prepend=unlock_env[0])) * float(sr) * 0.008
            switch_crackle = (rng.standard_normal((n_samples, 2)).astype(np.float32)
                              * np.clip(switch_edge, 0.0, 1.2)[:, None] * 0.28 * self.spec.dropout_sensitivity)

            if np.any(unlock_env > 1e-3):
                lin_fallback = _build_linear_track(sig)
                u_col = np.clip(unlock_env, 0.0, 1.0)[:, None]
                sig = hifi * (1.0 - u_col) + lin_fallback * u_col + switch_crackle
            else:
                sig = hifi

        elif mode_name == "hi8_afm":
            # 8mm / Hi8 AFM audio: 1.5 MHz single FM carrier on drum; dropouts cause direct FM static bursts
            afm = _highpass_audio(sig, 35.0, sr, order=2)
            afm = _lowpass_audio(afm, 15000.0 * (1.0 - 0.25 * az), sr, order=4)
            burst_env = np.clip(dropout_env * 0.85 + shuttle_env * 0.75 + ctl_unlock_env, 0.0, 1.0)[:, None]
            fm_static = _bandpass_audio(rng.standard_normal((n_samples, 2)).astype(np.float32), 500.0, 11000.0, sr, order=2)
            hiss_base = fm_static * (0.009 * self.spec.hiss_floor)
            sig = afm * (1.0 - 0.82 * burst_env) + fm_static * (burst_env * 0.32 * self.spec.dropout_sensitivity) + hiss_base

        elif mode_name == "minidv_pcm":
            # MiniDV 12-bit 32 kHz nonlinear PCM quantization + sample-hold / framing clicks on dropout/shuttle
            # 1. Downsample-and-hold at 32 kHz
            step_32k = max(1, int(round(sr / 32000.0)))
            if step_32k > 1:
                held_idx = (np.arange(n_samples) // step_32k) * step_32k
                sig = sig[held_idx]
            # 2. 12-bit nonlinear companded quantization (fewer steps at higher levels)
            levels = 1024.0
            companded = np.sign(sig) * np.log1p(15.0 * np.abs(sig)) / np.log1p(15.0)
            quant = np.round(companded * levels) / levels
            sig = np.sign(quant) * (np.expm1(np.abs(quant) * np.log1p(15.0)) / 15.0).astype(np.float32)

            # 3. Digital ECC failure during severe dropouts or shuttle -> 12 ms sample-block hold or mute + click
            ecc_fail = (dropout_env > 0.42) | (shuttle_env > 0.35) | (ctl_unlock_env > 0.35)
            if np.any(ecc_fail):
                blk_len = max(32, int(0.010 * sr))
                for b0 in range(0, n_samples, blk_len):
                    b1 = min(n_samples, b0 + blk_len)
                    if np.mean(ecc_fail[b0:b1]) > 0.30:
                        if (b0 // blk_len) % 2 == 0 and b0 > 0:
                            # Repeat last valid sample or short grain buffer (classic DV shuttle stutter)
                            grain_len = min(128, b0)
                            reps = int(np.ceil((b1 - b0) / float(grain_len)))
                            sig[b0:b1] = np.tile(sig[b0 - grain_len : b0], (reps, 1))[: b1 - b0] * 0.75
                        else:
                            # Brief PCM mute packet
                            sig[b0:b1] *= 0.05

        elif mode_name == "vcd_mp2":
            # Video CD MPEG-1 Layer II (1152-sample frame subband quantization, birdies & pre-echo)
            codec_cfg = self.params.get("codec") or {}
            bitrate = float(codec_cfg.get("bitrate", 1150.0))
            q_coarseness = float(np.clip(1200.0 / max(250.0, bitrate), 0.5, 4.0))
            sig = _lowpass_audio(sig, min(16500.0, 18000.0 / (q_coarseness ** 0.35)), sr, order=6)
            # STFT-like subband thresholding on 512-sample windows to produce authentic MP2 "watery" birdies & pre-echo
            win_len = 512
            hop = 256
            win = np.hanning(win_len).astype(np.float32)[:, None]
            out_mp2 = np.zeros_like(sig)
            w_norm = np.zeros((n_samples, 1), dtype=np.float32)
            for p0 in range(0, n_samples - win_len, hop):
                seg = sig[p0 : p0 + win_len] * win
                spec_f = np.fft.rfft(seg, axis=0)
                mag = np.abs(spec_f)
                # Psychoacoustic masking threshold per frame: drop quiet spectral bins
                thresh = np.max(mag, axis=0, keepdims=True) * (0.018 * q_coarseness)
                keep = np.clip((mag - thresh) / np.maximum(thresh * 0.5, 1e-6), 0.0, 1.0)
                # Quantize surviving spectral magnitudes
                q_step = max(1e-4, 0.012 * q_coarseness)
                spec_q = np.round(spec_f * keep / q_step) * q_step
                rec = np.fft.irfft(spec_q, n=win_len, axis=0).astype(np.float32)
                out_mp2[p0 : p0 + win_len] += rec * win
                w_norm[p0 : p0 + win_len] += win * win
            valid = w_norm[:, 0] > 1e-4
            sig[valid] = out_mp2[valid] / w_norm[valid]

        elif mode_name == "film_optical":
            # 8. Photographic Optical Soundtrack (Slit Loss + AC Exciter Lamp Hum + Grain Hiss + Scratches)
            sig = self._build_film_optical_track(sig, sr, rng, t_arr)

        elif mode_name == "streaming_lowbitrate":
            # 9. Low-Bitrate Web Streaming Codec (32-48kbps AAC/MP3 Watery Warble & Metallic Resonance)
            sig = self._build_streaming_lowbitrate_track(sig, sr, rng, t_arr)

        elif mode_name == "broadcast_studio":
            # 10. Broadcast Studio PGM Link (processed by pre-stage AGC & Reverb)
            sig = _highpass_audio(sig, 35.0, sr, order=2)
            sig = _lowpass_audio(sig, 15000.0, sr, order=4)

        elif mode_name == "rf_tv_dx_tuner":
            # 7. Taobao Cheap TV Capture Card (SAA7130 / TDA9887) FM Intercarrier Demodulator
            #    with TV-DX Atmospheric Ducting / Sporadic-E FM Quieting & QSB Fading:
            #    - PAL-D (+6.5 MHz) / PAL-I (+6.0 MHz) FM sound subcarrier (50 us de-emphasis, ~12 kHz BW).
            #    - FM Capture & Quieting threshold around C/N = 18 dB:
            #      When TV-DX atmospheric ducting (`dx_duct_gain_db`) lifts effective C/N above threshold,
            #      FM capture effect suppresses background hiss so distant stations sound surprisingly clean!
            #    - QSB slow ionospheric fading modulates C/N over time, causing noise floor to "breathe".
            #    - Intercarrier Vision-into-Sound 50 Hz / 15.625 kHz buzz when white OSD text overmodulates carrier.
            rf_cfg = self.params.get("rf") or {}
            p_mode = int(round(float(rf_cfg.get("propagation_mode", 1.0))))
            sig_dbm = float(rf_cfg.get("signal_dbm", -72.0))
            duct_gain = float(rf_cfg.get("dx_duct_gain_db", 24.0)) if p_mode in (1, 2, 3) else 0.0
            nf_db = float(rf_cfg.get("noise_figure_db", 7.5))
            qsb_depth = float(rf_cfg.get("qsb_depth_db", 4.5)) if p_mode > 0 else 0.0
            qsb_rate = float(rf_cfg.get("qsb_rate_hz", 0.35))
            buzz_lv = float((self.params.get("audio") or {}).get("rf_buzz", 0.12))

            noise_floor_dbm = -106.2 + nf_db
            qsb_wave = qsb_depth * (
                0.65 * np.sin(2.0 * np.pi * qsb_rate * t_arr)
                + 0.35 * np.cos(2.0 * np.pi * (qsb_rate * 1.73) * t_arr + 0.8)
            ).astype(np.float32)
            cnr_t = (sig_dbm + duct_gain + qsb_wave) - noise_floor_dbm

            # FM capture & quieting curve: below 14 dB is pure FM discriminator static; above 26 dB is quieted
            fm_quieting = np.clip((cnr_t - 13.0) / 15.0, 0.0, 1.0).astype(np.float32)
            voice_gate = np.clip((cnr_t - 10.0) / 10.0, 0.05, 1.0).astype(np.float32)[:, None]

            mono = 0.5 * (sig[:, 0] + sig[:, 1])
            mono = _highpass_audio(mono[:, None], 65.0, sr, order=2)[:, 0]
            mono = _lowpass_audio(mono[:, None], 11500.0, sr, order=4)[:, 0]
            sig = np.stack([mono, mono], axis=1) * voice_gate

            # FM discriminator triangulated hiss (modulated by QSB fading and suppressed by ducting gain)
            hiss_raw = rng.standard_normal((n_samples, 1)).astype(np.float32)
            hiss_shaped = _bandpass_audio(hiss_raw, 300.0, 9500.0, sr, order=2)
            hiss_env = ((1.0 - fm_quieting) ** 1.6 * 0.22 + 0.008 * (1.0 - 0.6 * fm_quieting))[:, None]

            # Vision-into-sound 50 Hz frame + 100 Hz harmonic intercarrier buzz on cheap TV card
            buzz_50 = (
                0.55 * np.sin(2.0 * np.pi * 50.0 * t_arr)
                + 0.30 * np.sin(2.0 * np.pi * 100.0 * t_arr + 0.4)
                + 0.15 * np.sin(2.0 * np.pi * 150.0 * t_arr + 1.1)
            ).astype(np.float32)[:, None]
            buzz_amt = buzz_lv * 0.025 * (0.4 + 0.6 * (1.0 - fm_quieting[:, None]))

            sig = np.tanh((sig + hiss_shaped * hiss_env + buzz_50 * buzz_amt) * 1.18) / 1.10

        elif mode_name == "rf_pal_d_mono":
            # PAL-D VHF Channel 6 (171 MHz, +6.5 MHz FM Sound Carrier) -> Panasonic NV-J27 Mono RF Tuner -> AV Capture Card
            # Calibrated against real hardware capture `10月16日.mp4` (Xiaomi Mi Box 720p60 -> HDMI-AV PAL-D -> VHF Ch6 -> NV-J27 -> PC):
            #   1. Stereo-to-Mono downmix at HDMI-AV / RF modulator (+6.5 MHz single FM carrier -> dual-mono L=R, corr=1.0000)
            #   2. 50 us FM de-emphasis (f_c = 1 / (2*pi*50e-6) = 3183.1 Hz) + 6.5 MHz ceramic IF & baseband rolloff (~8.5 kHz)
            #      and steep >14.2 kHz IF stopband (-110 dB floor in 14-18 kHz, -131 dB in 18-22 kHz)
            #   3. Quadrature FM discriminator noise floor + 50 Hz PAL vertical field sync & 150 Hz 3rd-harmonic intercarrier buzz
            mono = 0.5 * (sig[:, 0] + sig[:, 1])
            mono_col = mono[:, None]
            mono_col = _highpass_audio(mono_col, 32.0, sr, order=2)
            # 50 us de-emphasis shelf (3183.1 Hz corner)
            deemph = _lowpass_audio(mono_col, 3183.1, sr, order=1)
            mono_col = 0.78 * deemph + 0.22 * mono_col
            # Baseband & 6.5 MHz ceramic IF filter roll-off
            fc_rf = float(np.clip(self.spec.linear_bandwidth, 2500.0, 14000.0))
            mono_col = _lowpass_audio(mono_col, fc_rf, sr, order=4)
            mono_col = _lowpass_audio(mono_col, min(14200.0, 0.46 * sr), sr, order=6)
            # Quadrature FM detector de-emphasized noise floor (mono, before RCA split)
            fm_n_raw = rng.standard_normal((n_samples, 1)).astype(np.float32)
            fm_n = _bandpass_audio(fm_n_raw, 160.0, min(9000.0, fc_rf), sr, order=2)
            fm_n = _lowpass_audio(fm_n, 3183.1, sr, order=1)
            mono_col = mono_col + fm_n * (0.0038 * self.spec.hiss_floor)
            sig = np.repeat(mono_col, 2, axis=1)

        # ------------------------------------------------------------------
        # Step 4: Add Synchronized Physical Fault Signals
        # (Crease Thump, Base Cracks, CTL Breakage, Head-Switch Buzz, Mains Hum, RF Buzz, HDMI Drops)
        # ------------------------------------------------------------------
        # 4a. Low-frequency mechanical crease thump
        if np.any(np.abs(crease_thump) > 1e-5):
            thump_lp = _lowpass_audio(crease_thump[:, None], 110.0, sr, order=2)
            sig = sig + thump_lp

        # 4b. Base cracking sharp impulse clicks
        if np.any(np.abs(crack_clicks) > 1e-5):
            sig[:, 0] += crack_clicks
            sig[:, 1] += crack_clicks * 0.92

        # 4c. Base breakage CTL unlock static burst
        if np.any(ctl_unlock_env > 1e-4):
            brk_noise = _bandpass_audio(rng.standard_normal((n_samples, 2)).astype(np.float32), 200.0, 8000.0, sr, order=2)
            c_env = np.clip(ctl_unlock_env, 0.0, 1.0)[:, None]
            sig = sig * (1.0 - 0.75 * c_env) + brk_noise * (0.28 * c_env)

        # 4d. Head-Switching 59.94 Hz (NTSC) / 50 Hz (PAL) Pulse Train + Harmonics
        if np.any(hs_pulse_env > 1e-4):
            field_hz = 50.0 if "pal" in self.std_name else 59.94
            # Sawtooth + narrow switching spike at field rate
            ph_f = 2.0 * np.pi * field_hz * t_arr
            buzz_wave = (0.45 * np.sin(ph_f)
                         + 0.35 * np.sin(2.0 * ph_f + 0.4)
                         + 0.25 * np.sin(3.0 * ph_f + 0.9)
                         + 0.18 * np.sin(5.0 * ph_f)
                         + 0.12 * np.sin(9.0 * ph_f))
            sig = sig + (buzz_wave[:, None] * hs_pulse_env[:, None] * 0.14).astype(np.float32)

        # 4e. AV Connector Ground-Loop Mains Hum (`mains_hz`, 50 or 60 Hz + rectifier harmonics)
        if np.any(hum_env > 1e-4):
            conn_cfg = self.params.get("connector") or {}
            mains_hz = float(conn_cfg.get("mains_hz", 50.0))
            ph_m = 2.0 * np.pi * mains_hz * t_arr
            if mode_name == "rf_pal_d_mono":
                # Calibrated to `10月16日.mp4`: 50 Hz (-50.0 dB), 100 Hz (-58.4 dB), 150 Hz (-49.6 dB)
                hum_wave = (0.58 * np.sin(ph_m)
                            + 0.22 * np.sin(2.0 * ph_m + 0.3)
                            + 0.62 * np.sin(3.0 * ph_m + 0.7)
                            + 0.12 * np.sin(5.0 * ph_m))
                sig = sig + (hum_wave[:, None] * hum_env[:, None] * 0.045).astype(np.float32)
            else:
                # Fundamental + 2nd harmonic (rectified corrosion) + 3rd harmonic (transformer core saturation)
                hum_wave = (0.65 * np.sin(ph_m)
                            + 0.25 * np.sin(2.0 * ph_m + 0.3)
                            + 0.20 * np.sin(3.0 * ph_m)
                            + 0.08 * np.sin(5.0 * ph_m))
                sig = sig + (hum_wave[:, None] * hum_env[:, None] * 0.16).astype(np.float32)

        # 4f. RF Coaxial Link Intercarrier Vision-into-Sound Buzz & Sync-Slip Static
        if np.any(rf_buzz_env > 1e-4) or np.any(rf_static_env > 1e-4):
            field_hz = 50.0 if "pal" in self.std_name else 59.94
            # Vision-into-sound intercarrier buzz has a harsh rasp at field_hz + harmonics up to 1.5 kHz
            ph_rf = 2.0 * np.pi * field_hz * t_arr
            rasp = (0.42 * np.sin(ph_rf)
                    + 0.24 * np.sin(2.0 * ph_rf)
                    + 0.46 * np.sin(3.0 * ph_rf + 0.5)
                    + 0.22 * np.cos(4.0 * ph_rf)
                    + 0.18 * np.sin(8.0 * ph_rf)
                    + 0.12 * np.cos(16.0 * ph_rf))
            if mode_name == "rf_pal_d_mono":
                rf_n_mono = _bandpass_audio(rng.standard_normal((n_samples, 1)).astype(np.float32), 600.0, 8500.0, sr, order=2)
                rf_noise = np.repeat(rf_n_mono, 2, axis=1)
                sig = (sig * (1.0 - np.clip(0.20 * rf_static_env, 0.0, 0.50)[:, None])
                       + (rasp[:, None] * rf_buzz_env[:, None] * 0.038).astype(np.float32)
                       + rf_noise * (rf_static_env[:, None] * 0.045))
            else:
                rf_noise = _bandpass_audio(rng.standard_normal((n_samples, 2)).astype(np.float32), 600.0, 9000.0, sr, order=2)
                sig = (sig * (1.0 - np.clip(0.35 * rf_static_env, 0.0, 0.75)[:, None])
                       + (rasp[:, None] * rf_buzz_env[:, None] * 0.13).astype(np.float32)
                       + rf_noise * (rf_static_env[:, None] * 0.22))

        # 4g. HDMI Bit-Error Clicks & Link-Drop Mute
        if np.any(hdmi_ber_env > 1e-4):
            # Serial bit errors corrupt PCM words in short runs, producing digital splats/clicks and brief dropouts
            blk = max(64, int(0.008 * sr))
            for b0 in range(0, n_samples, blk):
                b1 = min(n_samples, b0 + blk)
                ber_local = float(np.mean(hdmi_ber_env[b0:b1]))
                if ber_local > 1e-4 and rng.random() < ber_local * 0.85:
                    if rng.random() < 0.55:
                        sig[b0:b1] = 0.0
                    else:
                        # Corrupted MSB burst
                        sig[b0 : min(b1, b0 + 24)] = rng.uniform(-0.65, 0.65, size=(min(24, b1 - b0), 2)).astype(np.float32)
        if np.any(hdmi_mute_env > 0.5):
            sig = sig * (1.0 - hdmi_mute_env[:, None])

        # ------------------------------------------------------------------
        # Step 5: Magnetic Tape Hysteresis / Quadrature Detector Soft-Clipping
        # ------------------------------------------------------------------
        sat = float(self.spec.saturation)
        if mode_name == "rf_pal_d_mono":
            # Panasonic NV-J27 quadrature FM detector & AV line-out soft ceiling (measured peak ~0.725 on 10月16日.mp4)
            drive = 1.0 + 1.35 * sat
            asym = 0.045 * sat
            driven = sig * drive + asym * (sig * sig)
            sig = 0.725 * (np.tanh(driven / 0.725) - np.tanh(asym / 0.725))
        elif sat > 1e-4 and mode_name in ("vhs_linear", "vhs_hifi", "hi8_afm"):
            drive = 1.0 + 2.4 * sat
            # Asymmetric bias nonlinearity adds warm 2nd harmonic before odd-order `tanh` tape ceiling
            asym = 0.08 * sat
            driven = sig * drive + asym * (sig * sig)
            sig = (np.tanh(driven) - np.tanh(asym)) / np.tanh(drive)

        # ------------------------------------------------------------------
        # Step 6: CRT TV Speaker Cabinet Resonance & Horizontal Line-Rate / Flyback Whine
        # ------------------------------------------------------------------
        display_cfg = self.params.get("display") or {}
        disp_on = float(display_cfg.get("enabled", 1.0)) > 0.5
        spk_mix = float(self.spec.speaker_sim) if disp_on else float(self.spec.speaker_sim) * 0.35
        if spk_mix > 1e-4:
            # Small 3-inch plastic TV cabinet speaker: rolls off bass below 190 Hz, resonant bump around 1.1 kHz,
            # and paper-cone rolloff above 5.8 kHz
            spk = _highpass_audio(sig, 185.0, sr, order=2)
            spk = _lowpass_audio(spk, 5800.0, sr, order=3)
            mid_bump = _bandpass_audio(spk, 750.0, 1800.0, sr, order=2)
            spk = spk + 0.42 * mid_bump
            sig = sig * (1.0 - spk_mix) + spk * spk_mix

        whine = float(self.spec.flyback_whine) if (disp_on or mode_name == "rf_pal_d_mono") else 0.0
        if whine > 1e-4:
            # 15,734.26 Hz (NTSC) or 15,625.0 Hz (PAL) horizontal line-rate sync / flyback whistle
            f_line = 15625.0 if "pal" in self.std_name else 15734.26
            if f_line < 0.48 * sr:
                fly_amp = (0.0022 * whine) if mode_name == "rf_pal_d_mono" else (0.018 * whine)
                fly_wave = np.sin(2.0 * np.pi * f_line * t_arr)[:, None].astype(np.float32)
                sig = sig + fly_wave * fly_amp

        # Step 6b: Option R Spatial Acoustic Room Field (Early Reflections, RT60 Tail, Spatial EMI Hum & HVAC Rumble)
        conn_cfg_env = self.params.get("connector") or {}
        if float(conn_cfg_env.get("env_field_enabled", 0.0)) > 0.5 or int(round(float(conn_cfg_env.get("env_field_mode", 0.0)))) > 0:
            from ..codecs.spatial_environment_field import SpatialEnvFieldSpec, compute_spatial_acoustic_audio
            env_sp = SpatialEnvFieldSpec.from_params(conn_cfg_env)
            sig, env_aud_meta = compute_spatial_acoustic_audio(
                sig, env_sp, sample_rate=sr, start_s=start_s, seed=self.seed
            )
        else:
            env_aud_meta = {}

        # Step 6c: Option S Network Streaming Audio Compression, Rebuffer Stalls & Packet-Loss Dropouts
        codec_cfg_str = self.params.get("codec") or {}
        if float(codec_cfg_str.get("stream_enabled", 0.0)) > 0.5 or int(round(float(codec_cfg_str.get("stream_mode", 0.0)))) > 0:
            from ..codecs.network_streaming_era import NetworkStreamingSpec, compute_streaming_audio_effect
            str_sp = NetworkStreamingSpec.from_params(codec_cfg_str)
            sig, str_aud_meta = compute_streaming_audio_effect(
                sig, str_sp, sample_rate=sr, start_s=start_s
            )
        else:
            str_aud_meta = {}

        # Step 6d: 10-Subsystem Back-End (04 Perceptual Codec -> 05 Telecom/VoIP -> 09 Loudness -> 07 System Chain -> 02 Mobile Speaker -> 10 A/V Sync Clock)
        sig, back_sub_meta = run_audio_subsystems_back_end(
            sig, eff_params, sr=sr, fps=self.fps, seed=self.seed, start_s=start_s
        )
        all_sub_meta = {**front_sub_meta, **back_sub_meta}

        # Final output gain & soft limiter
        sig = sig * float(self.spec.master_gain)
        if np.any(hdmi_mute_env > 0.5):
            sig = sig * (1.0 - hdmi_mute_env[:, None])
        sig = np.clip(sig, -0.99, 0.99).astype(np.float32)

        # Compute summary metrics for UI and verification
        fault_frame_count = sum(
            1 for f in frame_faults
            if (f.dropout_level > 0.05 or f.stiction_grab > 0.05 or f.head_switch_buzz > 0.05
                or f.rf_sync_buzz > 0.05 or f.ground_hum > 0.05 or f.hdmi_drop > 0.05
                or f.crack_spike > 0.05 or f.shuttle_chop > 0.05
                or abs(f.av_offset_ms) > 5.0 or f.frame_dropped or f.edit_splice)
        )
        loud_out = (all_sub_meta.get("s09_loudness") or {}).get("output") or {}
        av_sync_meta = all_sub_meta.get("s10_av_sync_clock") or {}
        summary = {
            "mode": mode_name,
            "sample_rate": sr,
            "start_s": round(start_s, 3),
            "duration_s": round(duration_s, 3),
            "speed_ratio": round(mean_speed_ratio, 3),
            "max_pitch_dev_pct": round(max((f.pitch_deviation_pct for f in frame_faults), default=0.0), 3),
            "max_dropout": round(max((f.dropout_level for f in frame_faults), default=0.0), 3),
            "hifi_fallback_frames": sum(1 for f in frame_faults if f.hifi_fallback > 0.1),
            "fault_frames": fault_frame_count,
            "rms_db": round(float(20.0 * np.log10(max(1e-6, float(np.sqrt(np.mean(sig ** 2)))))), 2),
            "integrated_lufs": loud_out.get("integrated_lufs", -24.0),
            "true_peak_dbtp": loud_out.get("true_peak_dbtp", -6.0),
            "lra_lu": loud_out.get("lra_lu", 4.0),
            "av_offset_ms": av_sync_meta.get("av_offset_ms", 0.0),
            "clock_drift_ppm": av_sync_meta.get("clock_drift_ppm", 0.0),
            "subsystems": all_sub_meta,
            "rf_tuner_cascaded": bool(rf_enabled_cascade and mode_name in ("vhs_linear", "vhs_hifi", "hi8_afm")),
            **rf_cascade_meta,
            **({"spatial_acoustic": env_aud_meta} if env_aud_meta else {}),
        }
        return AudioRun(
            samples=sig,
            clean_samples=clean,
            sample_rate=sr,
            start_s=start_s,
            duration_s=duration_s,
            resolved_mode=mode_name,
            frame_faults=frame_faults,
            summary=summary,
        )


    # ------------------------------------------------------------------
    # Acoustic & Electroacoustic Front-End Subsystems
    # ------------------------------------------------------------------
    def _apply_microphone_recording_system(
        self, sig: np.ndarray, sr: int, rng: np.random.Generator, spec: AudioSpec
    ) -> tuple[np.ndarray, dict]:
        """
        Microphone Recording System (换能器类型、近讲效应、防喷罩缺失破音与振膜/前级削顶失真).
        """
        m_type = int(spec.mic_type)
        if m_type == 0 and spec.proximity_effect <= 1e-4 and spec.plosive_pop <= 1e-4 and spec.mic_overdrive <= 1e-4:
            return sig, {"active": False}

        out = sig.copy()
        n_samples = len(out)

        # 1. Transducer Frequency Response Profile
        if m_type == 1:
            # Dynamic Mic (e.g. Shure SM58): 55 Hz HPF, +2.8 dB presence bump around 4.2 kHz, gentle 13.5 kHz roll-off
            out = _highpass_audio(out, 55.0, sr, order=2)
            pres = _bandpass_audio(out, 2800.0, 5800.0, sr, order=2)
            out = out + 0.38 * pres
            out = _lowpass_audio(out, 13800.0, sr, order=3)
        elif m_type == 2:
            # Large-Diaphragm Condenser: fast transients, flat 25 Hz - 18.5 kHz
            out = _highpass_audio(out, 25.0, sr, order=1)
            air = _highpass_audio(out, 9000.0, sr, order=1)
            out = out + 0.15 * air
            out = _lowpass_audio(out, 19500.0, sr, order=3)
        elif m_type == 3:
            # Ribbon Mic: smooth top-end roll-off above 10.5 kHz, warm low-mid presence
            out = _lowpass_audio(out, 10500.0, sr, order=3)
            warm = _bandpass_audio(out, 180.0, 650.0, sr, order=2)
            out = out + 0.25 * warm
        elif m_type == 4:
            # 2000s PC USB Webcam / Electret Mic (优酷女主播《稍息立正站好》同款):
            # Severe bass cut (<200 Hz), harsh plastic cavity resonance at 3.8 kHz (+6.5 dB), high preamp hiss
            out = _highpass_audio(out, 210.0, sr, order=3)
            res = _bandpass_audio(out, 3200.0, 4800.0, sr, order=3)
            out = out + 0.85 * res  # harsh piercing vocal resonance
            out = _lowpass_audio(out, 9200.0, sr, order=4)
            # Electret capsule thermal hiss floor
            capsule_hiss = rng.standard_normal(out.shape).astype(np.float32)
            out += 0.0055 * _highpass_audio(capsule_hiss, 1500.0, sr, order=2)
        elif m_type == 5:
            # Carbon Mic (Early Intercom / Telephone): 350 Hz - 3200 Hz steep bandpass
            out = _bandpass_audio(out, 340.0, 3250.0, sr, order=4)
            # Carbon granule granular noise
            out += 0.009 * rng.standard_normal(out.shape).astype(np.float32)

        # 2. Proximity Effect (近讲效应: 声源贴近定向话筒时 <180 Hz 显著提升 +3..+10 dB)
        prox = float(spec.proximity_effect)
        if prox > 1e-4 and m_type in (1, 2, 3, 4):
            bass = _lowpass_audio(out, 185.0, sr, order=2)
            out = out + (0.95 * prox) * bass

        # 3. Plosive Pop (防喷罩缺失引起的 P/B/T 爆破音低频风噪冲击)
        pop_amt = float(spec.plosive_pop)
        if pop_amt > 1e-4:
            # Detect sharp energy attack transients in vocal band
            env_fast = gaussian_filter1d(np.mean(np.abs(out), axis=1), sigma=max(2, int(0.004 * sr)))
            d_env = np.diff(env_fast, prepend=env_fast[0]) * sr
            pop_triggers = np.where((d_env > 18.0) & (env_fast > 0.08))[0]
            if len(pop_triggers) > 0:
                pop_wave = np.zeros(n_samples, dtype=np.float32)
                t_idx = 0
                while t_idx < len(pop_triggers):
                    pos = pop_triggers[t_idx]
                    dur = int(rng.uniform(0.065, 0.130) * sr)
                    end_p = min(n_samples, pos + dur)
                    t_pop = np.linspace(0.0, float(end_p - pos) / sr, end_p - pos, dtype=np.float32)
                    freq_pop = rng.uniform(42.0, 68.0)
                    decay = np.exp(-t_pop * rng.uniform(18.0, 32.0))
                    thump = pop_amt * rng.uniform(0.28, 0.65) * np.sin(2.0 * np.pi * freq_pop * t_pop) * decay
                    pop_wave[pos:end_p] += thump
                    t_idx += max(1, int(0.12 * sr))  # refractory interval
                out += pop_wave[:, None]

        # 4. Diaphragm & Preamp Overload (唱腔高亢或大声压导致的不对称谐波削顶破音)
        od = float(spec.mic_overdrive)
        if od > 1e-4:
            drive = 1.0 + 3.8 * od
            asym = 0.16 * od
            driven = out * drive + asym * (out * out)
            out = (np.tanh(driven) - np.tanh(asym)).astype(np.float32)

        return out, {"active": True, "mic_type": m_type}

    def _apply_broadcast_console(
        self, sig: np.ndarray, sr: int, rng: np.random.Generator, spec: AudioSpec
    ) -> tuple[np.ndarray, dict]:
        """
        Broadcast Audio Console & Room Reverb (演播室播出级压缩限幅器/抽吸效应与房间混响).
        """
        out = sig.copy()
        n_samples = len(out)
        agc_amt = float(spec.broadcast_agc)
        rev_amt = float(spec.studio_room_reverb)

        # 1. Broadcast AGC / Compressor (CBS Audimax / Volumax / Orban Optimod style)
        if agc_amt > 1e-4:
            # Fast peak detection (tau ~ 2.5 ms), adaptive release (tau ~ 320 ms)
            sig_mag = np.mean(np.abs(out), axis=1)
            sig_peak = gaussian_filter1d(sig_mag, sigma=max(2, int(0.0025 * sr)))
            sig_slow = gaussian_filter1d(sig_mag, sigma=max(10, int(0.32 * sr)))
            env = np.maximum(sig_peak * 0.75, sig_slow)
            thresh = 0.22
            gain_reduction = np.ones(n_samples, dtype=np.float32)
            over = env > thresh
            if np.any(over):
                # Pumping compression curve
                ratio = 4.5 * agc_amt
                gr = (thresh / np.maximum(1e-4, env)) ** (1.0 - 1.0 / (1.0 + ratio))
                gain_reduction = np.where(over, gr, 1.0).astype(np.float32)
                # Apply make-up gain so compressed audio sounds dynamically pumped and punchy
                makeup = 1.0 + 0.85 * agc_amt
                out = out * (gain_reduction[:, None] * makeup)

        # 2. Studio / Bedroom Livestreamer Room Reverb (早期反射梳状滤波器 + 漫射衰减)
        if rev_amt > 1e-4:
            delays_ms = [14.0, 26.5, 41.0]
            gains = [0.38, 0.26, 0.18]
            rev_accum = np.zeros_like(out)
            for d_ms, g_val in zip(delays_ms, gains):
                d_samp = int(round((d_ms / 1000.0) * sr))
                if d_samp < n_samples:
                    rev_accum[d_samp:] += out[:-d_samp] * (g_val * rev_amt)
            # Lowpass filtered room dampening
            rev_accum = _lowpass_audio(rev_accum, 4500.0, sr, order=2)
            out = out * (1.0 - 0.22 * rev_amt) + rev_accum

        return out, {"active": (agc_amt > 1e-4 or rev_amt > 1e-4)}

    # ------------------------------------------------------------------
    # Medium-Specific Soundtracks: Optical Film & Low-Bitrate Web
    # ------------------------------------------------------------------
    def _build_film_optical_track(
        self, inp: np.ndarray, sr: int, rng: np.random.Generator, t_arr: np.ndarray
    ) -> np.ndarray:
        """
        Film Optical Soundtrack (感光胶片光学音轨物理仿真: 缝隙孔径损耗、励磁灯交流声、银盐颗粒底噪与划痕爆音).
        """
        n_samples = len(inp)
        slit = float(self.spec.film_slit_loss)
        hum_amt = float(self.spec.film_exciter_hum)
        scr_amt = float(self.spec.film_optical_scratches)

        # 1. Optical Slit Aperture Loss (12.5 um slit): 35mm cutoff ~6.8 kHz; 16mm cutoff ~4.8 kHz
        fc_slit = 6800.0 * (1.0 - 0.32 * slit)
        mono = 0.5 * (inp[:, 0] + inp[:, 1])
        mono = _highpass_audio(mono[:, None], 42.0, sr, order=2)[:, 0]
        mono = _lowpass_audio(mono[:, None], fc_slit, sr, order=5)[:, 0]

        # 2. Optical Track Overmodulation Hard Clipping (双边可变面积音轨边界物理截断)
        track_drive = 1.15
        mono = np.clip(mono * track_drive, -0.86, 0.86)

        # 3. Exciter Lamp 50 Hz/100 Hz (PAL) or 60 Hz/120 Hz (NTSC) AC Filament Ripple Hum
        mains_hz = 50.0 if "pal" in self.std_name else 60.0
        if hum_amt > 1e-4:
            hum_wave = (
                0.012 * hum_amt * (
                    0.65 * np.sin(2.0 * np.pi * mains_hz * t_arr)
                    + 0.35 * np.sin(4.0 * np.pi * mains_hz * t_arr + 0.4)
                )
            ).astype(np.float32)
            mono += hum_wave

        # 4. Silver Halide Grain Transmittance Hiss (透过光学声带透光面积的光电管颗粒噪底)
        grain_noise = rng.standard_normal((n_samples,)).astype(np.float32)
        grain_hiss = _bandpass_audio(grain_noise[:, None], 850.0, min(fc_slit * 0.95, 0.46 * sr), sr, order=2)[:, 0]
        # Noise level is higher in quiet passages (clear base transmittance)
        amp_mod = 0.5 + 0.5 * np.clip(1.0 - np.abs(mono), 0.2, 1.0)
        mono += grain_hiss * (0.018 * self.spec.hiss_floor * amp_mod)

        # 5. Optical Dust & Scratch Clicks/Pops (灰尘阻挡光缝负脉冲与乳剂划破强光正脉冲)
        if scr_amt > 1e-4:
            n_pops = int(rng.poisson(scr_amt * 18.0 * (n_samples / float(sr))))
            if n_pops > 0:
                pop_positions = rng.integers(0, max(1, n_samples - 24), size=n_pops)
                for pos in pop_positions:
                    p_len = rng.integers(4, 20)
                    pol = 1.0 if rng.random() < 0.55 else -0.75
                    amp = float(pol * rng.uniform(0.18, 0.55) * scr_amt)
                    decay = np.exp(-np.arange(p_len) / 3.0).astype(np.float32)
                    mono[pos : pos + p_len] += amp * decay

        out = np.stack([mono, mono], axis=-1)
        return out

    def _build_streaming_lowbitrate_track(
        self, inp: np.ndarray, sr: int, rng: np.random.Generator, t_arr: np.ndarray
    ) -> np.ndarray:
        """
        Streaming Low-Bitrate Codec Artifacts (优酷/土豆/3GP 32-48kbps AAC/MP3 心理声学量化水下水花声与金属共鸣).
        Calibrated against: 2008 年优酷主播演唱《稍息立正站好》实测特征 (352x264, 15fps, 32kbps AAC).
        """
        n_samples = len(inp)
        # 1. Steep brickwall low-pass cutoff at 6.2 kHz (32kbps AAC/MP3 standard cutoff)
        out = _highpass_audio(inp, 75.0, sr, order=2)
        out = _lowpass_audio(out, 6200.0, sr, order=8)

        # 2. Perceptual Subband Quantization Warble (Birdies 水声水花杂音)
        win_len = 512
        hop = 256
        win = np.hanning(win_len).astype(np.float32)[:, None]
        out_str = np.zeros_like(out)
        w_norm = np.zeros((n_samples, 1), dtype=np.float32)
        for p0 in range(0, n_samples - win_len, hop):
            seg = out[p0 : p0 + win_len] * win
            spec_f = np.fft.rfft(seg, axis=0)
            mag = np.abs(spec_f)
            # High threshold in mid-high frequencies produces authentic watery chirping
            freqs = np.fft.rfftfreq(win_len, 1.0 / sr)[:, None]
            mask_curve = 0.025 + 0.065 * np.clip((freqs - 1800.0) / 3500.0, 0.0, 1.0)
            thresh = np.max(mag, axis=0, keepdims=True) * mask_curve
            keep = np.clip((mag - thresh) / np.maximum(thresh * 0.4, 1e-6), 0.0, 1.0)
            # Coarse quantization steps
            q_step = 0.035
            spec_q = np.round(spec_f * keep / q_step) * q_step
            rec = np.fft.irfft(spec_q, n=win_len, axis=0).astype(np.float32)
            out_str[p0 : p0 + win_len] += rec * win
            w_norm[p0 : p0 + win_len] += win * win
        valid = w_norm[:, 0] > 1e-4
        out[valid] = out_str[valid] / w_norm[valid]

        # 3. Metallic Hollow Resonant Chamber (32kbps starved bitpool phase dispersion)
        comb_delay = int(round(sr / 2150.0))  # 2.15 kHz metallic comb tone
        if comb_delay < n_samples:
            out[comb_delay:] += out[:-comb_delay] * 0.22

        return out

    def _apply_rf_broadcast_tuner_prestage(
        self, sig: np.ndarray, t_arr: np.ndarray, sr: int, n_samples: int,
        rng: np.random.Generator, rf_cfg: dict
    ) -> tuple[np.ndarray, dict]:
        """
        Wireless Analog TV Broadcast FM Intercarrier Sound Demodulator (pre-tape VCR tuner).

        Physical reason why wireless RF audio degradation is much smaller than video:
          1. Modulation & Limiter Capture Effect:
             - Video is AM-VSB (Vestigial Sideband Amplitude Modulation, 5-6 MHz wide). Every dB of
               RF path loss or thermal noise directly adds snow to the picture, and multipath echoes
               directly superimpose ghosts and cancel chroma phase.
             - Audio is FM (±50 kHz deviation on the +6.5 MHz PAL-D / +6.0 MHz PAL-I / +4.5 MHz NTSC
               sound-vision intercarrier, only ~180 kHz IF bandwidth). Before frequency discrimination,
               the IF stage passes through a hard amplitude limiter that strips off AM noise and
               multipath envelope fluctuations (FM Quieting & Capture Effect).
          2. Bandwidth & De-emphasis Processing Gain:
             - Narrower IF bandwidth (180 kHz vs 5 MHz) gives a 14.4 dB lower thermal noise floor.
             - FM Carson/discriminator processing gain + 50 us de-emphasis filter (f_c = 3183 Hz)
               attenuates triangular high-frequency phase noise, yielding an 18-24 dB SNR advantage
               over video above the ~13 dB CNR FM threshold!
        """
        p_mode = int(round(float(rf_cfg.get("propagation_mode", 0.0))))
        sig_dbm = float(rf_cfg.get("signal_dbm", -66.0))
        duct_gain = float(rf_cfg.get("dx_duct_gain_db", 0.0)) if p_mode in (1, 2, 3) else 0.0
        nf_db = float(rf_cfg.get("noise_figure_db", 7.0))
        multipath = float(rf_cfg.get("multipath_level", 0.15))
        noise_floor_dbm = -106.2 + nf_db
        cnr_db = (sig_dbm + duct_gain) - noise_floor_dbm

        # FM capture & quieting factor (1.0 = full quieting above 26 dB CNR; drops below 13 dB threshold)
        fm_quieting = float(np.clip((cnr_db - 13.0) / 14.0, 0.0, 1.0))
        snr_adv_db = float(np.clip(6.0 + 16.5 * fm_quieting, 4.0, 22.5))

        # Broadcast mono intercarrier demodulation + 50 us de-emphasis + 12 kHz ceramic IF filter
        mono = 0.5 * (sig[:, 0] + sig[:, 1])
        mono_lp = _lowpass_audio(mono[:, None], 3183.1, sr, order=1)[:, 0]
        mono = 0.42 * mono + 0.58 * mono_lp
        mono = _highpass_audio(mono[:, None], 55.0, sr, order=2)[:, 0]
        mono = _lowpass_audio(mono[:, None], 11500.0, sr, order=4)[:, 0]
        out = np.stack([mono, mono], axis=1)

        # Residual FM discriminator triangular hiss (heavily suppressed when above FM threshold)
        hiss_raw = rng.standard_normal((n_samples, 1)).astype(np.float32)
        hiss_shaped = _bandpass_audio(hiss_raw, 800.0, 10500.0, sr, order=2)
        hiss_amp = ((1.0 - fm_quieting) ** 1.8) * 0.14 + 0.0025 * (1.0 - 0.5 * fm_quieting)

        # Vision-into-sound 50 Hz/60 Hz intercarrier sync buzz (from multipath or white OSD overmodulation)
        f_field = 50.0 if "pal" in self.std_name else 59.94
        buzz_wave = (
            0.60 * np.sin(2.0 * np.pi * f_field * t_arr)
            + 0.28 * np.sin(4.0 * np.pi * f_field * t_arr + 0.4)
            + 0.12 * np.sin(6.0 * np.pi * f_field * t_arr + 1.1)
        ).astype(np.float32)[:, None]
        buzz_amp = (0.0015 + 0.012 * multipath + 0.008 * (1.0 - fm_quieting)) * float(np.clip(self.spec.rf_buzz + 0.25, 0.0, 1.5))

        # Channel-switch (换台) intercarrier FM unlock static burst, varactor sweep chirp & 50Hz V-Sync roll buzz
        sw_rate = float(rf_cfg.get("channel_switch_rate", 0.0))
        sw_dur = float(rf_cfg.get("channel_switch_dur_s", 0.28))
        lo_sw = float(rf_cfg.get("lo_sweep_mhz", 0.0))
        vs_roll = float(rf_cfg.get("vsync_roll", 0.0))
        if sw_rate > 1e-4 or lo_sw > 1e-4 or vs_roll > 1e-4:
            n = n_samples
            t = t_arr
            sw_env = np.zeros(n, dtype=np.float32)
            if sw_rate > 1e-4:
                period_s = 1.0 / max(0.05, sw_rate)
                dur_s = float(np.clip(sw_dur, 0.06, min(1.5, period_s * 0.85)))
                phase_s = (t + 0.18 * period_s) % period_s
                in_sw = phase_s < dur_s
                u = np.clip(phase_s / max(1e-4, dur_s), 0.0, 1.0)
                env_u = np.where(u < 0.15, u / 0.15,
                        np.where(u < 0.70, 1.0, np.exp(-4.5 * (u - 0.70) / 0.30)))
                sw_env = np.where(in_sw, env_u, 0.0).astype(np.float32)
                # Switching relay / tuner contact click transient at start of switch
                click_env = np.where(in_sw & (u < 0.04), np.exp(-u / 0.008) * np.sin(2.0 * np.pi * 920.0 * t), 0.0).astype(np.float32)
            else:
                click_env = np.zeros(n, dtype=np.float32)

            eff_sw = np.clip(sw_env + 0.35 * lo_sw, 0.0, 1.0)
            # 1) Unsquelched 6.5 MHz PAL-D FM discriminator static burst ("沙沙" white/pink noise crash)
            static_raw = rng.standard_normal((n, 2)).astype(np.float32)
            static_band = _bandpass_audio(static_raw, 320.0, 4800.0, sr, order=2)
            # 2) Varactor LO sweep heterodyne whistle & 50Hz V-Sync vertical roll frame buzz
            chirp_freq = 420.0 + 1650.0 * np.sin(2.0 * np.pi * 3.5 * t) * eff_sw
            chirp_phase = 2.0 * np.pi * np.cumsum(chirp_freq / float(sr))
            chirp_wave = (0.12 * lo_sw * eff_sw * np.sin(chirp_phase))[:, None].astype(np.float32)
            vroll_buzz = (0.14 * (vs_roll * 0.5 + 0.5 * eff_sw) * np.sign(np.sin(2.0 * np.pi * 50.0 * t)))[:, None].astype(np.float32)
            # Duck program audio during carrier loss and inject FM discriminator static + click + buzz
            duck = (1.0 - 0.86 * eff_sw)[:, None].astype(np.float32)
            out = out * duck + static_band * (0.34 * eff_sw)[:, None] + chirp_wave + vroll_buzz + (0.28 * click_env)[:, None]

        out = (out + hiss_shaped * hiss_amp + buzz_wave * buzz_amp).astype(np.float32)
        meta = {
            "rf_cnr_db": round(cnr_db, 2),
            "rf_fm_quieting": round(fm_quieting, 4),
            "rf_video_vs_audio_snr_advantage_db": round(snr_adv_db, 2),
        }
        return out, meta

    def simulate(self, clean_stereo: np.ndarray, start_s: float = 0.0,
                 duration_s: float | None = None, **kwargs) -> AudioRun:
        return self.render_window(clean_stereo, start_s=start_s, duration_s=duration_s, **kwargs)

    def process(self, clean_stereo: np.ndarray, start_s: float = 0.0,
                duration_s: float | None = None, tape_plan=None, tape_stage=None, **kwargs) -> AudioRun:
        return self.render_window(
            clean_stereo, start_s=start_s, duration_s=duration_s,
            tape_stage=tape_stage, tape_plan=tape_plan, **kwargs
        )

    def extract_fault_envelope(self, start_frame: int, n_frames: int, tape_plan=None) -> list[AudioFaultFrame]:
        dur_s = max(1, int(n_frames)) / self.fps
        dummy = np.zeros((max(16, int(round(dur_s * 8000))), 2), dtype=np.float32)
        orig_sr = self.sample_rate
        try:
            self.sample_rate = 8000
            run = self.render_window(dummy, start_s=start_frame / self.fps, duration_s=dur_s, tape_plan=tape_plan)
            return run.frame_faults
        finally:
            self.sample_rate = orig_sr

    def timeline(self, start_s: float = 0.0, duration_s: float = 1.0, tape_plan=None) -> dict[str, Any]:
        start_f = max(0, int(round(start_s * self.fps)))
        n_f = max(1, int(round(duration_s * self.fps)))
        faults = self.extract_fault_envelope(start_f, n_f, tape_plan=tape_plan)
        return {
            "fps": self.fps,
            "start_s": start_s,
            "duration_s": duration_s,
            "frames": [f.to_dict() for f in faults],
        }


# ===========================================================================
# 10-SUBSYSTEM PHYSICAL AUDIO & A/V SYNCHRONIZATION EXPANSION
# ===========================================================================
from scipy.signal import lfilter as _scipy_lfilter


def _peaking_eq_audio(sig: np.ndarray, center_hz: float, q: float, gain_db: float, sr: int) -> np.ndarray:
    """RBJ Audio EQ Cookbook peaking biquad filter."""
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
    return _scipy_lfilter(b, a, sig, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# SUBSYSTEM 01 · 麦克风与录音前端 (Microphone & Recording Front-End)
# ---------------------------------------------------------------------------
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
        vib_base = float(np.clip(aud.get("mic_vibration_hum", aud.get("mic_chassis_vibration", 0.0)), 0.0, 1.0))
        if m_type == 6 and vib_base < 0.15:
            vib_base = max(vib_base, 0.25 * float(np.clip(tape.get("walk_amplitude", 0.0) + tape.get("shuttle_speed", 0.0) * 0.1, 0.0, 1.0)))
        return cls(
            mic_type=int(np.clip(m_type, 0, 7)),
            polar_pattern=int(np.clip(round(float(aud.get("mic_polar_pattern", 1.0))), 0, 4)),
            off_axis_deg=float(np.clip(aud.get("mic_off_axis_deg", 0.0), 0.0, 180.0)),
            sensitivity_db=float(np.clip(aud.get("mic_sensitivity_db", 0.0), -24.0, 24.0)),
            self_noise_level=float(np.clip(aud.get("mic_self_noise", 0.0), 0.0, 1.0)),
            preamp_gain_db=float(np.clip(aud.get("mic_preamp_gain_db", 0.0), -18.0, 30.0)),
            agc_amount=float(np.clip(aud.get("mic_agc_strength", aud.get("mic_agc_amount", 0.0)), 0.0, 1.0)),
            agc_speed=float(np.clip(aud.get("mic_agc_speed", 0.5), 0.0, 1.0)),
            clip_mode=int(np.clip(round(float(aud.get("mic_clip_mode", 0.0))), 0, 2)),
            overdrive=float(np.clip(aud.get("mic_overdrive_clip", aud.get("mic_overdrive", 0.0)), 0.0, 1.0)),
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
        out = _highpass_audio(out, 55.0, sr, order=2)
        out = _peaking_eq_audio(out, 4200.0, q=1.2, gain_db=3.5, sr=sr)
        out = _peaking_eq_audio(out, 220.0, q=1.0, gain_db=-1.5, sr=sr)
        out = _lowpass_audio(out, 13500.0, sr, order=3)
        intrinsic_sens_db = -2.0
        intrinsic_ein = 0.0008
    elif m_type == 2:
        out = _highpass_audio(out, 25.0, sr, order=1)
        out = _peaking_eq_audio(out, 10500.0, q=0.8, gain_db=2.8, sr=sr)
        out = _lowpass_audio(out, 19500.0, sr, order=3)
        intrinsic_sens_db = +3.5
        intrinsic_ein = 0.0003
    elif m_type == 3:
        out = _highpass_audio(out, 35.0, sr, order=1)
        out = _peaking_eq_audio(out, 280.0, q=0.9, gain_db=3.8, sr=sr)
        out = _lowpass_audio(out, 9800.0, sr, order=3)
        intrinsic_sens_db = -4.5
        intrinsic_ein = 0.0014
    elif m_type == 4:
        out = _highpass_audio(out, 215.0, sr, order=3)
        out = _peaking_eq_audio(out, 3800.0, q=2.4, gain_db=7.5, sr=sr)
        out = _lowpass_audio(out, 9000.0, sr, order=4)
        intrinsic_sens_db = +1.5
        intrinsic_ein = 0.0065
    elif m_type == 5:
        out = _bandpass_audio(out, 340.0, 3200.0, sr, order=4)
        out = _peaking_eq_audio(out, 1150.0, q=1.8, gain_db=5.5, sr=sr)
        out = (out + 0.34 * (out * out) - 0.08).astype(np.float32)
        out = _highpass_audio(out, 300.0, sr, order=2)
        intrinsic_sens_db = +1.0
        intrinsic_ein = 0.0110
    elif m_type == 6:
        out = _highpass_audio(out, 135.0, sr, order=2)
        out = _peaking_eq_audio(out, 2650.0, q=1.8, gain_db=4.2, sr=sr)
        out = _lowpass_audio(out, 12500.0, sr, order=3)
        mid = 0.5 * (out[:, 0] + out[:, 1])
        side = 0.5 * (out[:, 0] - out[:, 1]) * 0.55
        out = np.stack([mid + side, mid - side], axis=-1)
        intrinsic_sens_db = 0.0
        intrinsic_ein = 0.0038
    elif m_type == 7:
        mono = 0.5 * (out[:, 0] + out[:, 1])
        mono_2d = np.stack([mono, mono], axis=-1)
        out = _highpass_audio(mono_2d, 270.0, sr, order=3)
        out = _peaking_eq_audio(out, 3150.0, q=2.8, gain_db=6.8, sr=sr)
        out = _lowpass_audio(out, 7200.0, sr, order=4)
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
        out = _lowpass_audio(out, fc_offaxis, sr, order=2)

    prox = float(spec.proximity_effect)
    if prox > 1e-4 and int(spec.polar_pattern) > 0:
        dir_factor = float((1.0 - alpha) * max(0.2, abs(np.cos(theta_rad))))
        bass = _lowpass_audio(out, 185.0, sr, order=2)
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
        rumble_lp = _bandpass_audio(raw_rumble, 24.0, 82.0, sr, order=2)
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
        pink_lp = _lowpass_audio(white_n, 450.0, sr, order=1)
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


# ---------------------------------------------------------------------------
# SUBSYSTEM 02 · 旧手机扬声器与听筒 (Legacy Mobile Speaker & Earpiece)
# ---------------------------------------------------------------------------
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
            drive_spl=float(np.clip(aud.get("mobile_spk_volume_drive", aud.get("mobile_spk_drive", 0.35)), 0.0, 1.0)),
            cavity_resonance=float(np.clip(aud.get("mobile_spk_resonance_q", aud.get("mobile_spk_cavity_res", 0.45)), 0.0, 1.0)),
            volume_limiter=float(np.clip(aud.get("mobile_spk_alc_limit", aud.get("mobile_spk_limiter", 0.40)), 0.0, 1.0)),
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
        x = _highpass_audio(x, max(240.0, f0 * 0.85), sr, order=3)
        x = _lowpass_audio(x, 3550.0, sr, order=5)
        x = _peaking_eq_audio(x, 1150.0, q=2.0, gain_db=4.5 * cav, sr=sr)
        x = _peaking_eq_audio(x, 2450.0, q=2.5, gain_db=3.0 * cav, sr=sr)
    elif mode == 2:
        x = _highpass_audio(x, f0, sr, order=4)
        x = _peaking_eq_audio(x, f0 * 1.08, q=3.2, gain_db=5.5, sr=sr)
        x = _peaking_eq_audio(x, 2450.0, q=3.0, gain_db=8.0 * cav, sr=sr)
        x = _peaking_eq_audio(x, 3850.0, q=3.4, gain_db=6.5 * cav, sr=sr)
        x = _lowpass_audio(x, 6400.0, sr, order=4)
    elif mode == 3:
        x = _highpass_audio(x, f0, sr, order=3)
        x = _peaking_eq_audio(x, f0 * 1.05, q=2.4, gain_db=4.2, sr=sr)
        x = _peaking_eq_audio(x, 1850.0, q=2.2, gain_db=5.5 * cav, sr=sr)
        x = _peaking_eq_audio(x, 4200.0, q=2.6, gain_db=5.0 * cav, sr=sr)
        x = _lowpass_audio(x, 9200.0, sr, order=4)
    else:
        x = _highpass_audio(x, f0, sr, order=3)
        x = _peaking_eq_audio(x, 980.0, q=2.5, gain_db=6.0 * cav, sr=sr)
        x = _peaking_eq_audio(x, 2750.0, q=2.8, gain_db=5.5 * cav, sr=sr)
        x = _lowpass_audio(x, 4400.0, sr, order=5)

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
        x = _highpass_audio(x, max(200.0, f0 * 0.72), sr, order=2)

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


# ---------------------------------------------------------------------------
# SUBSYSTEM 03 · 数字采样与音频格式转换 (Digital Sampling & Format Conversion)
# ---------------------------------------------------------------------------
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
            out = _lowpass_audio(out, nyq_target * 0.92, sr, order=8)
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


# ---------------------------------------------------------------------------
# SUBSYSTEM 04 · 感知音频编码 (Perceptual Audio Coding: MP3 / AAC / AMR / Speech)
# ---------------------------------------------------------------------------
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
        out = _bandpass_audio(out, 220.0, fc_cut, sr, order=4)
        win_len = 512
    elif c_type == 4:
        eff_br = float(np.clip(br, 8.0, 32.0))
        fc_cut = float(np.clip(2800.0 + 140.0 * eff_br, 3200.0, 7200.0))
        mono = 0.5 * (out[:, 0] + out[:, 1])
        out = np.stack([mono, mono], axis=-1)
        out = _bandpass_audio(out, 160.0, fc_cut, sr, order=4)
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
        out = _lowpass_audio(out, min(fc_cut * (0.95 ** g_idx), 0.47 * sr), sr, order=6)
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
            mid_band = _bandpass_audio(out, 2400.0, 4800.0, sr, order=2)
            sbr_hf = _highpass_audio(np.abs(mid_band) * np.sign(np.roll(mid_band, 1, axis=0)), 5000.0, sr, order=2)
            out = out + 0.16 * sbr_hf
        elif c_type in (3, 4):
            celp_step = max(2, int(round(sr / 8000.0)))
            held = out[(np.arange(n_samples) // celp_step) * celp_step]
            out = 0.65 * out + 0.35 * held
            out = _peaking_eq_audio(out, 1050.0, q=2.6, gain_db=3.2, sr=sr)
            out = _peaking_eq_audio(out, 2350.0, q=3.0, gain_db=3.8, sr=sr)
            if c_type == 3:
                out = _lowpass_audio(out, 3600.0, sr, order=5)

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


# ---------------------------------------------------------------------------
# SUBSYSTEM 05 · 移动电话与网络语音 (Mobile Telephony & VoIP Real-Time Speech)
# ---------------------------------------------------------------------------
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
        ploss_raw = aud.get("telecom_packet_loss_rate", None)
        ploss = float(np.clip(float(ploss_raw) * 100.0 if ploss_raw is not None else aud.get("telecom_packet_loss_pct", 14.0 if mode == 3 else (3.5 if mode > 0 else 0.0)), 0.0, 60.0))
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
        mono_col = _bandpass_audio(mono[:, None], 300.0, 3400.0, sr, order=4)
    else:
        mono_col = _bandpass_audio(mono[:, None], 55.0, 7000.0, sr, order=4)
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
            far_echo = _bandpass_audio(far_echo[:, None], 400.0, 2600.0, sr, order=2)[:, 0]
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
        cng_shaped = _bandpass_audio(cng_raw[:, None], 320.0, 2400.0, sr, order=2)[:, 0] * (0.018 * cng_lv)
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


# ---------------------------------------------------------------------------
# SUBSYSTEM 06 · 声学空间与麦克风位置 (Acoustic Space & Microphone Placement)
# ---------------------------------------------------------------------------
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

        raw_dist = float(aud.get("space_distance_m", aud.get("space_mic_distance_m", -1.0)))
        dist = d_dist if raw_dist < 0.0 else raw_dist

        raw_abs = float(aud.get("space_wall_hf_absorb", aud.get("space_wall_absorption", -1.0)))
        w_abs = d_abs if raw_abs < 0.0 else raw_abs

        raw_modes = float(aud.get("space_standing_waves", -1.0))
        sw_modes = d_modes if raw_modes < 0.0 else raw_modes

        return cls(
            preset=preset,
            room_length_m=float(np.clip(aud.get("space_room_len_m", aud.get("space_room_length_m", d_lx)), 1.2, 40.0)),
            room_width_m=float(np.clip(aud.get("space_room_wid_m", aud.get("space_room_width_m", d_ly)), 1.0, 40.0)),
            room_height_m=float(np.clip(aud.get("space_room_hei_m", aud.get("space_room_height_m", d_lz)), 1.8, 20.0)),
            rt60_s=float(np.clip(rt60, 0.05, 4.0)),
            wall_absorption=float(np.clip(w_abs, 0.02, 0.98)),
            standing_wave_modes=float(np.clip(sw_modes, 0.0, 1.0)),
            mic_distance_m=float(np.clip(dist, 0.05, 12.0)),
            occlusion=float(np.clip(aud.get("space_hand_occlusion", aud.get("space_occlusion", 0.0)), 0.0, 1.0)),
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
        direct_sig = _lowpass_audio(out, fc_air, sr, order=2) * direct_gain
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
            direct_sig = _peaking_eq_audio(direct_sig, fm, q=sw_q, gain_db=sw_gain_db, sr=sr)

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
    er_accum = _lowpass_audio(er_accum, fc_wall, sr, order=2)

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
        tail_accum = _lowpass_audio(tail_accum, max(1100.0, fc_wall * 0.65), sr, order=3)
        tail_accum = _highpass_audio(tail_accum, 95.0, sr, order=2)

    wet_sig = er_accum * (0.45 * reverb_send) + tail_accum * (0.55 * reverb_send)
    out = direct_sig + wet_sig

    occ = float(spec.occlusion)
    if occ > 1e-4:
        fc_occ = float(np.clip(16000.0 * ((1.0 - 0.88 * occ) ** 2.2), 320.0, 16000.0))
        out = _lowpass_audio(out, fc_occ, sr, order=4)
        out = _peaking_eq_audio(out, 280.0, q=1.6, gain_db=4.5 * occ, sr=sr) * (1.0 - 0.35 * occ)

    amb = float(spec.ambient_noise)
    if amb > 1e-4:
        raw_amb = rng.standard_normal((n_samples, 2)).astype(np.float32)
        if preset == 1:
            fan_hum = (0.008 * np.sin(2.0 * np.pi * 120.0 * t_arr) + 0.005 * np.sin(2.0 * np.pi * 238.0 * t_arr))[:, None].astype(np.float32)
            air_bed = _bandpass_audio(raw_amb, 180.0, 2200.0, sr, order=2) * 0.012
            out = out + amb * (fan_hum + air_bed)
        elif preset == 2:
            ballast = (0.010 * np.sin(2.0 * np.pi * 100.0 * t_arr) + 0.006 * np.sin(2.0 * np.pi * 300.0 * t_arr))[:, None].astype(np.float32)
            hall_bed = _bandpass_audio(raw_amb, 220.0, 1600.0, sr, order=2) * 0.016
            out = out + amb * (ballast + hall_bed)
        elif preset == 3:
            hvac = _lowpass_audio(raw_amb, 220.0, sr, order=2) * 0.022
            murmur = _bandpass_audio(raw_amb, 380.0, 1750.0, sr, order=2) * (0.015 * (0.6 + 0.4 * np.sin(2.0 * np.pi * 0.9 * t_arr))[:, None].astype(np.float32))
            out = out + amb * (hvac + murmur)
        elif preset == 4:
            wind_mod = np.maximum(0.0, np.sin(2.0 * np.pi * 0.85 * t_arr + 0.3) * np.cos(2.0 * np.pi * 0.37 * t_arr))[:, None].astype(np.float32)
            wind_rumble = _lowpass_audio(raw_amb, 130.0, sr, order=3) * (0.045 * wind_mod)
            traffic = _bandpass_audio(raw_amb, 150.0, 1200.0, sr, order=2) * 0.016
            out = out + amb * (wind_rumble + traffic)
        else:
            out = out + amb * (_bandpass_audio(raw_amb, 120.0, 3000.0, sr, order=2) * 0.014)

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


# ---------------------------------------------------------------------------
# SUBSYSTEM 07 · 播放器与系统音频链 (Media Player & OS System Audio Chain)
# ---------------------------------------------------------------------------
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
        sp_3d = float(np.clip(aud.get("sys_spatial_widener", aud.get("sys_spatial_3d", 0.45 if pr == 3 else 0.0)), 0.0, 1.0))
        und = float(np.clip(aud.get("sys_buffer_underrun", 0.25 if pr == 2 else 0.0), 0.0, 1.0))
        sw_pop = float(np.clip(aud.get("sys_switch_transient", aud.get("sys_device_switch_pop", 0.30 if pr == 4 else 0.0)), 0.0, 1.0))
        m_clip = float(np.clip(aud.get("sys_mixer_clip", 0.25 if pr in (1, 2, 3) else 0.0), 0.0, 1.0))
        m_gain = float(np.clip(aud.get("sys_mixer_volume", aud.get("sys_mixer_gain", 1.0)), 0.0, 2.5))
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
        out = 0.65 * out[idx_hold] + 0.35 * _lowpass_audio(out, 9800.0, sr, order=3)

    eq_p = int(spec.eq_preset)
    if eq_p == 1:
        out = _peaking_eq_audio(out, 110.0, q=1.1, gain_db=5.5, sr=sr)
        out = _peaking_eq_audio(out, 900.0, q=1.2, gain_db=-3.5, sr=sr)
        out = _peaking_eq_audio(out, 7500.0, q=1.2, gain_db=5.0, sr=sr)
    elif eq_p == 2:
        out = _peaking_eq_audio(out, 100.0, q=1.2, gain_db=-2.5, sr=sr)
        out = _peaking_eq_audio(out, 2200.0, q=1.1, gain_db=5.0, sr=sr)
    elif eq_p == 3:
        out = _highpass_audio(out, 95.0, sr, order=2)
        out = _peaking_eq_audio(out, 145.0, q=2.2, gain_db=7.0, sr=sr)
        out = _peaking_eq_audio(out, 3600.0, q=1.6, gain_db=3.2, sr=sr)
        out = _lowpass_audio(out, 12800.0, sr, order=3)

    sp_3d = float(spec.spatial_3d)
    if sp_3d > 1e-4:
        mid = 0.5 * (out[:, 0] + out[:, 1])
        side = 0.5 * (out[:, 0] - out[:, 1])
        side_delayed = np.roll(side, max(1, int(round(0.0006 * sr))))
        side_wide = side * (1.0 + 1.15 * sp_3d) + 0.45 * sp_3d * side_delayed
        out = np.stack([mid + side_wide, mid - side_wide], axis=-1).astype(np.float32)

    ch_m = int(spec.channel_map)
    if ch_m == 0 and int(spec.preset) == 3:
        ch_m = 3
    if ch_m == 1:
        out = out[:, ::-1].copy()
    elif ch_m == 2:
        out = np.stack([np.zeros_like(out[:, 1]), out[:, 1]], axis=-1).astype(np.float32)
    elif ch_m == 3:
        diff_lr = 0.65 * (out[:, 0] - 0.88 * out[:, 1])
        out = np.stack([diff_lr, -diff_lr * 0.92], axis=-1).astype(np.float32)
    elif ch_m == 4:
        mono_m = 0.5 * (out[:, 0] + out[:, 1])
        out = np.stack([mono_m, mono_m], axis=-1).astype(np.float32)

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


# ---------------------------------------------------------------------------
# SUBSYSTEM 08 · 铃声、MIDI 与移动音源 (Ringtones, MIDI & Mobile Sound Engines)
# ---------------------------------------------------------------------------
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
            mix=float(np.clip(aud.get("ringtone_wet_mix", aud.get("ringtone_mix", 1.0)), 0.0, 1.0)),
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
        synth_out = _highpass_audio(synth_out, 550.0, sr, order=3)
        synth_out = _peaking_eq_audio(synth_out, 2750.0, q=3.2, gain_db=6.5, sr=sr)
        synth_out = _lowpass_audio(synth_out, 6500.0, sr, order=3)

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
        synth_out = _lowpass_audio(synth_out, 11025.0, sr, order=4)

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


# ---------------------------------------------------------------------------
# SUBSYSTEM 09 · 广播与网络平台的响度处理 (Broadcast & Platform Loudness / EBU R 128)
# ---------------------------------------------------------------------------
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
        comp = float(np.clip(aud.get("loudness_comp_amount", aud.get("loudness_compressor", 0.78 if mode == 2 else (0.35 if mode in (1, 4) else 0.0))), 0.0, 1.0))
        tp_ceil = float(np.clip(aud.get("loudness_true_peak_limit_dbtp", aud.get("loudness_true_peak_dbtp", -1.0 if mode == 1 else (-0.3 if mode == 2 else -1.0))), -9.0, 0.0))
        passes = int(np.clip(round(float(aud.get("loudness_retranscode_gens", aud.get("loudness_retranscode_passes", 3.0 if mode == 4 else 1.0)))), 1, 6))
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

    k_sig = _peaking_eq_audio(arr, min(4500.0, 0.42 * sr), q=0.5, gain_db=4.0, sr=sr)
    k_sig = _highpass_audio(k_sig, 38.0, sr, order=2)

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
            b_low = _lowpass_audio(out, 220.0, sr, order=2)
            b_high = _highpass_audio(out, 3800.0, sr, order=2)
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
        "output": out_metrics,
    }


# ---------------------------------------------------------------------------
# SUBSYSTEM 10 · 音视频同步与时钟系统 (Audio-Video Sync & Master Clock System)
# ---------------------------------------------------------------------------
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

        base_offset = float(aud.get("av_fixed_offset_ms", aud.get("av_sync_offset_ms", 0.0)))
        if int(round(float(ply.get("transfer_station_mode", 0.0)))) > 0:
            if float(ply.get("transfer_tbc_lock", ply.get("tbc_enabled", 0.0))) < 0.5:
                base_offset += float(ply.get("transfer_av_drift_ms", 0.0))

        base_drift = float(aud.get("av_clock_drift_ppm", 0.0))
        if abs(base_drift) < 1e-3 and abs(float(ply.get("clock_drift_ppm", 0.0))) > 1.0:
            base_drift = float(ply.get("clock_drift_ppm", 0.0)) * 0.5

        base_drop = float(aud.get("av_drop_frame_prob", aud.get("av_frame_drop_rate", 0.0)))
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


# ---------------------------------------------------------------------------
# Unified 10-Subsystem Orchestrator & Node-Graph Stage
# ---------------------------------------------------------------------------
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
