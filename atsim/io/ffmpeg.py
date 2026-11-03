"""
Video I/O on top of FFmpeg.

We deliberately do NOT pull in a Python video library.  FFmpeg already knows
how to decode/encode every container we care about, and this project only needs
two things from it:

    read  : decode to raw RGB24 frames at a known size (and hand out audio)
    write : encode raw RGB24 frames back, copying or processing the audio

Frame-accurate seeking is done by decoding from the previous keyframe and
discarding, which is what makes scrubbing in the preview acceptable in speed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

FFMPEG = os.environ.get("ATSIM_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("ATSIM_FFPROBE", "ffprobe")


def _windows_candidates(name: str) -> list[str]:
    """
    Usual places FFmpeg ends up on Windows.

    A bare `ffmpeg` on PATH is not enough in practice.  The WebUI is often started
    from a shortcut or a .bat whose environment differs from the shell the user
    tested in -- and then FFmpeg is present but invisible, which is exactly the
    situation that produced "FileNotFoundError: [WinError 2]" on import while
    offline rendering (which pipes to `ffmpeg` and happened to resolve) worked.
    """
    exe = name + ".exe"
    roots = [
        r"C:\ffmpeg\bin", r"C:\Program Files\ffmpeg\bin",
        r"C:\Program Files (x86)\ffmpeg\bin",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links"),
        os.path.expandvars(r"%ProgramData%\chocolatey\bin"),
        os.path.expandvars(r"%USERPROFILE%\scoop\shims"),
    ]
    # a portable build extracted anywhere under a drive root, e.g. E:\ffmpeg-6.1.1...
    for drive in ("C:", "D:", "E:", "F:"):
        for entry in (r"\ffmpeg\bin", r"\ffmpeg-6.1.1-essentials_build\ffmpeg-6.1.1-essentials_build\bin"):
            roots.append(drive + entry)
    out = []
    for root in roots:
        cand = os.path.join(root, exe)
        if os.path.isfile(cand):
            out.append(cand)
    # also glob for a versioned portable directory
    for drive in ("C:", "D:", "E:", "F:"):
        for pattern in (r"\ffmpeg-*\ffmpeg-*\bin", r"\ffmpeg*\bin"):
            import glob as _glob
            for hit in _glob.glob(drive + pattern):
                cand = os.path.join(hit, exe)
                if os.path.isfile(cand):
                    out.append(cand)
    return out


def resolve_tool(name: str, env_value: str) -> str:
    """
    Turn a tool name into something that can actually be executed.

    Order: an explicit override, then PATH, then the well-known install
    locations.  Returning the bare name when nothing is found lets the caller
    produce a clear error rather than a mysterious crash.
    """
    found = shutil.which(env_value)
    if found:
        return found
    if sys.platform == "win32":
        for cand in _windows_candidates(env_value):
            return cand
    return env_value


def tool_paths() -> tuple[str, str]:
    return (resolve_tool("ffmpeg", FFMPEG), resolve_tool("ffprobe", FFPROBE))


def missing_tools() -> list[str]:
    """Which of the two tools could not be located, by friendly name."""
    missing = []
    for label, env_value in (("ffmpeg", FFMPEG), ("ffprobe", FFPROBE)):
        resolved = resolve_tool(label, env_value)
        if shutil.which(resolved) is None and not os.path.isfile(resolved):
            missing.append(label)
    return missing


def have_ffmpeg() -> bool:
    return not missing_tools()


def _tool_error(label: str) -> RuntimeError:
    return RuntimeError(
        f"找不到 {label}。本工具的导入/导出/渲染都需要 FFmpeg。\n"
        f"解决办法（任选其一）：\n"
        f"  1) 安装 FFmpeg 并让它出现在 PATH 中；\n"
        f"  2) 设置环境变量 ATSIM_FFMPEG / ATSIM_FFPROBE 指向可执行文件的完整路径，"
        f"例如 set ATSIM_FFPROBE=E:\\ffmpeg\\bin\\ffprobe.exe；\n"
        f"  3) 把 FFmpeg 解压到 C:\\ffmpeg\\bin（本工具会自动查找该位置）。\n"
        f"当前 ATSIM_FFMPEG={FFMPEG!r}，ATSIM_FFPROBE={FFPROBE!r}。")


def _creationflags() -> int:
    if sys.platform == "win32":
        return 0x08000000  # CREATE_NO_WINDOW
    return 0


def run(cmd: Sequence[str], **kw) -> subprocess.CompletedProcess:
    """
    Run a tool, translating "not installed" into something actionable.

    A missing executable surfaces as FileNotFoundError (WinError 2 on Windows),
    which as a user-facing message is useless: it does not say *what* is missing.
    """
    argv = list(cmd)
    if argv:
        resolved = resolve_tool(os.path.basename(argv[0]), argv[0])
        argv[0] = resolved
        if shutil.which(resolved) is None and not os.path.isfile(resolved):
            raise _tool_error(os.path.basename(argv[0]))
    try:
        return subprocess.run(argv, capture_output=True,
                              creationflags=_creationflags(), **kw)
    except FileNotFoundError:
        raise _tool_error(os.path.basename(argv[0])) from None


# ---------------------------------------------------------------------------
@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    has_audio: bool = False
    pix_fmt: str = ""
    codec: str = ""
    rotation: int = 0
    raw: dict = field(default_factory=dict)

    @property
    def aspect(self) -> float:
        return self.width / max(1, self.height)

    def to_dict(self) -> dict:
        return {
            "path": self.path, "width": self.width, "height": self.height,
            "fps": round(self.fps, 4), "frame_count": self.frame_count,
            "duration": round(self.duration, 3), "has_audio": self.has_audio,
            "codec": self.codec, "aspect": round(self.aspect, 4),
        }


def probe(path: str | Path) -> VideoInfo:
    cmd = [resolve_tool("ffprobe", FFPROBE), "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    proc = run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.decode('utf-8', 'replace')[:400]}")
    data = json.loads(proc.stdout.decode("utf-8", "replace"))
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if v is None:
        raise RuntimeError(f"no video stream in {path}")
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})

    num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    if fps <= 0:
        num, den = (v.get("r_frame_rate") or "25/1").split("/")
        fps = float(num) / float(den) if float(den) else 25.0

    duration = float(fmt.get("duration") or v.get("duration") or 0.0)
    frame_count = int(v.get("nb_frames") or 0)
    if frame_count <= 0 and duration > 0:
        frame_count = int(round(duration * fps))

    rotation = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rotation = int(float(sd["rotation"]))
            except (TypeError, ValueError):
                rotation = 0

    return VideoInfo(
        path=str(path), width=int(v.get("width", 0)), height=int(v.get("height", 0)),
        fps=fps, frame_count=frame_count, duration=duration, has_audio=a is not None,
        pix_fmt=v.get("pix_fmt", ""), codec=v.get("codec_name", ""), rotation=rotation,
        raw=data,
    )


# ---------------------------------------------------------------------------
class VideoReader:
    """
    Decodes a video to RGB float32 frames, in order, optionally at a reduced
    scale.  `scale` < 1.0 is how the preview proxy is produced: decode the full
    frame and let FFmpeg resample it, so the proxy keeps the source's detail
    statistics reasonably intact.
    """

    def __init__(self, path: str | Path, info: VideoInfo | None = None,
                 scale: float = 1.0, start: int = 0, count: int | None = None,
                 width: int | None = None, height: int | None = None,
                 in_rgb24: bool = True):
        self.path = str(path)
        self.info = info or probe(path)
        self.scale = float(scale)
        self.start = max(0, int(start))
        self.count = count
        if width and height:
            self.out_w, self.out_h = int(width), int(height)
        else:
            self.out_w = max(2, int(round(self.info.width * self.scale / 2)) * 2)
            self.out_h = max(2, int(round(self.info.height * self.scale / 2)) * 2)
        self.in_rgb24 = in_rgb24
        self._proc: subprocess.Popen | None = None

    # -- command -------------------------------------------------------
    def _cmd(self) -> list[str]:
        vf = [f"scale={self.out_w}:{self.out_h}:flags=lanczos"]
        cmd = [resolve_tool("ffmpeg", FFMPEG), "-v", "error", "-nostdin", "-threads", "2"]
        if self.start > 0:
            ss = self.start / max(1e-6, self.info.fps)
            cmd += ["-ss", f"{ss:.6f}"]
        cmd += ["-i", self.path, "-vf", ",".join(vf), "-f", "rawvideo",
                "-pix_fmt", "rgb24"]
        if self.count is not None:
            cmd += ["-frames:v", str(int(self.count))]
        cmd += ["-"]
        return cmd

    def __enter__(self) -> "VideoReader":
        self._proc = subprocess.Popen(
            self._cmd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=_creationflags())
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdout:
                    self._proc.stdout.close()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None

    # -- frames --------------------------------------------------------
    @property
    def frame_bytes(self) -> int:
        return self.out_w * self.out_h * 3

    def frames_u8(self) -> Iterator[np.ndarray]:
        assert self._proc is not None and self._proc.stdout is not None
        n = self.frame_bytes
        while True:
            raw = self._proc.stdout.read(n)
            if not raw or len(raw) < n:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(self.out_h, self.out_w, 3).copy()

    def frames(self) -> Iterator[np.ndarray]:
        assert self._proc is not None and self._proc.stdout is not None
        n = self.frame_bytes
        scale_f = np.float32(1.0 / 255.0)
        try:
            import cv2 as _cv2
            lut = getattr(VideoReader, "_U8_LUT", None)
            if lut is None:
                lut = (np.arange(256, dtype=np.float32) * scale_f).reshape(256, 1)
                VideoReader._U8_LUT = lut
            _have_cv = True
        except Exception:
            _cv2 = None
            lut = None
            _have_cv = False
        while True:
            raw = self._proc.stdout.read(n)
            if not raw or len(raw) < n:
                break
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.out_h, self.out_w, 3)
            if _have_cv:
                yield _cv2.LUT(arr, lut)
            else:
                yield arr.astype(np.float32) * scale_f

    def frame_at(self, index: int) -> np.ndarray | None:
        """Random access: decode from just before `index` and discard."""
        with VideoReader(self.path, self.info, self.scale, start=int(index), count=1,
                         width=self.out_w, height=self.out_h) as r:
            for f in r.frames():
                return f
        return None


def read_frames(path: str | Path, scale: float = 1.0,
                start: int = 0, count: int | None = None) -> list[np.ndarray]:
    info = probe(path)
    with VideoReader(path, info, scale, start, count) as r:
        return list(r.frames())


def extract_audio(path: str | Path, out_path: str | Path, fmt: str = "wav") -> bool:
    cmd = [resolve_tool("ffmpeg", FFMPEG), "-y", "-v", "error", "-nostdin", "-i", str(path),
           "-vn", "-acodec", "pcm_s16le" if fmt == "wav" else "copy", str(out_path)]
    return run(cmd).returncode == 0


def wav_encode(samples: np.ndarray, sample_rate: int = 44100) -> bytes:
    """
    Encode stereo or mono float32 [-1.0, 1.0] PCM into a standard 16-bit RIFF WAV byte buffer.
    Built with struct + NumPy so audio preview and muxing need zero external audio libraries.
    """
    import struct
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=-1)
    elif arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)
    else:
        arr = arr[:, :2]
    pcm16 = np.clip(np.round(arr * 32767.0), -32768, 32767).astype("<i2")
    raw = pcm16.tobytes()
    n_ch = 2
    sr = int(sample_rate)
    byte_rate = sr * n_ch * 2
    block_align = n_ch * 2
    data_size = len(raw)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        n_ch,
        sr,
        byte_rate,
        block_align,
        16,
        b"data",
        data_size,
    )
    return header + raw


def read_audio_pcm(path: str | Path, sample_rate: int = 44100,
                   start_s: float = 0.0, duration_s: float | None = None
                   ) -> np.ndarray | None:
    """
    Decode stereo float32 PCM from a media file via FFmpeg.
    Returns shape (N, 2) in [-1, 1], or None if the file has no audio stream or FFmpeg fails.
    """
    cmd = [resolve_tool("ffmpeg", FFMPEG), "-v", "error", "-nostdin"]
    if start_s > 1e-4:
        cmd += ["-ss", f"{float(start_s):.4f}"]
    cmd += ["-i", str(path)]
    if duration_s is not None and duration_s > 0:
        cmd += ["-t", f"{float(duration_s):.4f}"]
    cmd += [
        "-vn",
        "-ac", "2",
        "-ar", str(int(sample_rate)),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]
    try:
        r = run(cmd)
        if r.returncode != 0 or not r.stdout:
            return None
        n_floats = len(r.stdout) // 4
        n_frames = n_floats // 2
        if n_frames <= 0:
            return None
        arr = np.frombuffer(r.stdout[:n_frames * 8], dtype="<f4").reshape(n_frames, 2)
        return np.clip(arr.copy(), -1.0, 1.0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
class VideoWriter:
    """Pipes RGB frames into FFmpeg."""

    def __init__(self, out_path: str | Path, width: int, height: int, fps: float,
                 crf: int = 16, preset: str = "medium", codec: str = "libx264",
                 pix_fmt: str = "yuv420p", extra: Sequence[str] = ()):
        self.out_path = str(out_path)
        self.in_width, self.in_height = int(width), int(height)
        #: **The encoded frame is padded to EVEN dimensions, because `yuv420p` cannot store an odd one.**
        #:
        #: Chroma in `yuv420p` is subsampled 2x in both directions, so an odd width or height has no place to put the
        #: last chroma sample: libx264 refuses outright -- `height not divisible by 2 (334x485)`.  **The source of odd
        #: sizes is a still image**, whose dimensions are whatever the file happens to be, and the UI's `scale=0.5`
        #: turns even sizes odd as often as not (`720x540 -> 360x270` is fine, `334x485 -> 167x243` is not).
        #:
        #: Measured: every `/api/render` on a still produced `RuntimeError: ffmpeg encode failed` and a **0-byte file**,
        #: the same symptom as the silent-clip fault next to it -- two different causes, one shape of failure.
        #:
        #: The pad is one pixel at most, taken from the bottom/right edge, and it is applied to the **encoded** frame
        #: rather than by resizing the render: the picture the stages produced is unchanged, and only the container's
        #: frame is rounded up.  `write` pads the incoming array to match, so callers keep passing their own size.
        self.width = self.in_width + (self.in_width % 2)
        self.height = self.in_height + (self.in_height % 2)
        self.fps = float(fps)
        self.crf, self.preset, self.codec = crf, preset, codec
        self.pix_fmt = pix_fmt
        self.extra = list(extra)
        self._proc: subprocess.Popen | None = None
        self.frames_written = 0

    def _cmd(self, audio_from: str | None, audio_codec: str) -> list[str]:
        cmd = [resolve_tool("ffmpeg", FFMPEG), "-y", "-v", "error", "-nostdin",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{self.width}x{self.height}", "-r", f"{self.fps:.6f}",
               "-i", "-"]
        if audio_from:
            cmd += ["-i", audio_from]
        cmd += ["-map", "0:v:0"]
        if audio_from:
            #: **The audio map is OPTIONAL, and it has to be, because the source may carry no audio at all.**
            #:
            #: `-map 1:a:0` is a *requirement*: when the input has no audio stream ffmpeg refuses to start and says so
            #: -- `Stream map '1:a:0' matches no streams ... To ignore this, add a trailing '?'`.  Because the process
            #: never started, **nothing was ever written**, so the failure surfaced as a **0-byte output file** rather
            #: than as a usable video, and `close()` raised only at the end.
            #:
            #: Measured on the user's uploads, which are silent: every `/api/render` on one produced
            #: `RuntimeError: ffmpeg encode failed: Stream map '1:a:0' matches no streams` and left
            #: `runs/renders/<job>.mp4` at 0 bytes -- the two zero-length files sitting in that directory are the
            #: evidence.  The trailing `?` makes the map *conditional*: audio is carried when it exists and silently
            #: omitted when it does not, which is what "keep the audio if there is any" was supposed to mean.
            cmd += ["-map", "1:a:0?", "-c:a", audio_codec, "-shortest"]
        cmd += ["-c:v", self.codec, "-threads", "0", "-crf", str(self.crf), "-preset", self.preset,
                "-pix_fmt", self.pix_fmt]
        cmd += self.extra
        cmd += [self.out_path]
        return cmd

    def __enter__(self) -> "VideoWriter":
        return self

    def open(self, audio_from: str | None = None, audio_codec: str = "aac") -> "VideoWriter":
        self._proc = subprocess.Popen(
            self._cmd(audio_from, audio_codec), stdin=subprocess.PIPE,
            stderr=subprocess.PIPE, creationflags=_creationflags())
        return self

    def write_u8(self, arr: np.ndarray) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        if arr.shape[0] == self.in_height and arr.shape[1] == self.in_width:
            if (self.height, self.width) != (self.in_height, self.in_width):
                arr = np.pad(arr, ((0, self.height - self.in_height), (0, self.width - self.in_width), (0, 0)),
                             mode="edge")
        if arr.shape[0] != self.height or arr.shape[1] != self.width:
            raise ValueError(f"frame size {arr.shape[:2]} != {self.height, self.width}")
        self._proc.stdin.write(np.ascontiguousarray(arr, dtype=np.uint8).tobytes())
        self.frames_written += 1

    def write(self, rgb: np.ndarray) -> None:
        if rgb.dtype == np.uint8:
            self.write_u8(rgb)
            return
        arr = (np.clip(rgb, 0.0, 1.0) * np.float32(255.0) + np.float32(0.5)).astype(np.uint8)
        self.write_u8(arr)

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            err = self._proc.stderr.read() if self._proc.stderr else b""
            self._proc.wait(timeout=120)
            if self._proc.returncode != 0:
                raise RuntimeError(f"ffmpeg encode failed: {err.decode('utf-8', 'replace')[:600]}")
        finally:
            self._proc = None

    def __exit__(self, *exc) -> None:
        self.close()


def jpeg_bytes(rgb: np.ndarray, quality: int = 88) -> bytes:
    """Encode one frame to JPEG for the preview stream (no temp files)."""
    arr = np.clip(rgb, 0.0, 1.0)
    if arr.dtype != np.uint8:
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    h, w = arr.shape[:2]
    cmd = [resolve_tool("ffmpeg", FFMPEG), "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-i", "-", "-frames:v", "1",
           "-f", "image2", "-c:v", "mjpeg", "-q:v", str(max(2, min(31, int(31 - quality / 3.2)))),
           "-"]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(arr).tobytes(),
                          capture_output=True, creationflags=_creationflags())
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[:300])
    return proc.stdout
