"""
Frame sources: anything the renderer can pull frames from by index.

`ClipSource` reads a real video through FFmpeg and caches decoded frames in a
small LRU.  The cache matters because the renderer legitimately asks for a frame
twice: once when the chain needs its input and once for the camera's motion blur
(the previous field).  `SyntheticSource` generates test patterns with no I/O, so
the engine can be exercised and benchmarked without a clip.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import numpy as np

from .ffmpeg import VideoInfo, VideoReader, probe


class ClipSource:
    def __init__(self, path: str | Path, scale: float = 1.0,
                 cache_frames: int = 24, info: VideoInfo | None = None):
        self.path = str(path)
        self.info = info or probe(path)
        self.scale = float(scale)
        self.cache_frames = int(max(2, cache_frames))
        self.width = max(2, int(round(self.info.width * self.scale / 2)) * 2)
        self.height = max(2, int(round(self.info.height * self.scale / 2)) * 2)
        self.fps = self.info.fps or 25.0
        self.frame_count = self.info.frame_count
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._last_frame: np.ndarray | None = None
        self._seq_reader: VideoReader | None = None
        self._seq_iter: Iterator[np.ndarray] | None = None
        self._seq_next_idx: int = -1
        self._last_req_idx: int = -2
        self._lock = threading.Lock()

    def _close_seq(self) -> None:
        if self._seq_reader is not None:
            try:
                self._seq_reader.close()
            except Exception:
                pass
            self._seq_reader = None
            self._seq_iter = None
            self._seq_next_idx = -1

    def iter_sequential_u8(self, start: int = 0, count: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """Stream raw uint8 (H, W, 3) frames from a single continuous FFmpeg pipe."""
        with VideoReader(self.path, self.info, self.scale, start=start, count=count,
                         width=self.width, height=self.height) as r:
            last_u8: np.ndarray | None = None
            idx = int(start)
            for f_u8 in r.frames_u8():
                last_u8 = f_u8
                yield idx, f_u8
                idx += 1
            if count is not None and last_u8 is not None:
                while idx < start + count:
                    yield idx, last_u8
                    idx += 1

    # ------------------------------------------------------------------
    def frame(self, index: int) -> np.ndarray:
        index = int(max(0, index))
        with self._lock:
            hit = self._cache.get(index)
            if hit is not None:
                self._cache.move_to_end(index)
                self._last_req_idx = index
                return hit
            # Check if our persistent sequential FFmpeg pipe is positioned right at or just before `index`
            if self._seq_iter is not None and 0 <= (index - self._seq_next_idx) <= 4:
                got_seq: np.ndarray | None = None
                try:
                    while self._seq_next_idx <= index:
                        f_arr = next(self._seq_iter)
                        cur_i = self._seq_next_idx
                        self._seq_next_idx += 1
                        self._last_frame = f_arr
                        self._cache[cur_i] = f_arr
                        self._cache.move_to_end(cur_i)
                        while len(self._cache) > self.cache_frames:
                            self._cache.popitem(last=False)
                        if cur_i == index:
                            got_seq = f_arr
                except StopIteration:
                    self._close_seq()
                except Exception:
                    self._close_seq()
                if got_seq is not None:
                    self._last_req_idx = index
                    return got_seq

            is_sequential = (index == 0 or index == self._last_req_idx + 1)
            self._last_req_idx = index
            if is_sequential:
                self._close_seq()
                try:
                    sr = VideoReader(self.path, self.info, self.scale, start=index, count=None,
                                     width=self.width, height=self.height)
                    sr.__enter__()
                    sit = sr.frames()
                    first_f = next(sit)
                    self._seq_reader = sr
                    self._seq_iter = sit
                    self._seq_next_idx = index + 1
                    self._last_frame = first_f
                    self._cache[index] = first_f
                    self._cache.move_to_end(index)
                    while len(self._cache) > self.cache_frames:
                        self._cache.popitem(last=False)
                    return first_f
                except Exception:
                    self._close_seq()

        # Non-sequential random access (UI scrubbing): decode a small contiguous chunk
        chunk = min(16, max(1, self.cache_frames // 2))
        if self.frame_count > 0:
            chunk = max(1, min(chunk, self.frame_count - index))
        decoded: list[tuple[int, np.ndarray]] = []
        with VideoReader(self.path, self.info, self.scale, start=index, count=chunk,
                         width=self.width, height=self.height) as r:
            for offset, f in enumerate(r.frames()):
                decoded.append((index + offset, f))
                if len(decoded) >= chunk:
                    break
        if not decoded:
            with self._lock:
                if self._cache:
                    last_idx = max(self._cache)
                    fallback = self._cache[last_idx]
                    self._last_frame = fallback
                    self._cache[index] = fallback
                    self.frame_count = min(self.frame_count, max(1, last_idx + 1))
                    return fallback
                if self._last_frame is not None:
                    self._cache[index] = self._last_frame
                    self.frame_count = min(self.frame_count, max(1, index))
                    return self._last_frame
            for back_start in (max(0, index - 16), max(0, index - 48), 0):
                back_frames: list[np.ndarray] = []
                with VideoReader(self.path, self.info, self.scale, start=back_start, count=24,
                                 width=self.width, height=self.height) as r:
                    for f in r.frames():
                        back_frames.append(f)
                if back_frames:
                    fallback = back_frames[-1]
                    with self._lock:
                        self._last_frame = fallback
                        self._cache[index] = fallback
                    return fallback
            return np.zeros((self.height, self.width, 3), dtype=np.float32)

        got = decoded[0][1]
        with self._lock:
            self._last_frame = decoded[-1][1]
            for f_idx, f_arr in decoded:
                self._cache[f_idx] = f_arr
                self._cache.move_to_end(f_idx)
            while len(self._cache) > self.cache_frames:
                self._cache.popitem(last=False)
        return got

    def prefetch(self, start: int, count: int) -> None:
        """Warm the cache sequentially -- far faster than random seeks."""
        with VideoReader(self.path, self.info, self.scale, start=start, count=count,
                         width=self.width, height=self.height) as r:
            for i, f in enumerate(r.frames()):
                with self._lock:
                    self._cache[start + i] = f
                    while len(self._cache) > self.cache_frames:
                        self._cache.popitem(last=False)

    def warm(self, count: int = 8) -> None:
        try:
            self.prefetch(0, count)
        except Exception:
            pass

    def clear(self) -> None:
        with self._lock:
            self._close_seq()
            self._cache.clear()


# ---------------------------------------------------------------------------
class StillSource:
    """
    A single image presented as a clip.
    """

    def __init__(self, path: str | Path, scale: float = 1.0,
                 fps: float = 25.0, frames: int = 1):
        from .imagecodec import load_image
        self.path = str(path)
        rgb = load_image(self.path).astype(np.float32)
        if scale and abs(scale - 1.0) > 1e-6:
            h0, w0 = rgb.shape[:2]
            nh = max(2, int(round(h0 * scale)))
            nw = max(2, int(round(w0 * scale)))
            try:
                from PIL import Image as _PILImage
                pil = _PILImage.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
                rgb = np.asarray(pil.resize((nw, nh), _PILImage.LANCZOS), dtype=np.float32) / 255.0
            except Exception:
                step = max(1, int(round(1.0 / max(1e-6, scale))))
                rgb = rgb[::step, ::step]
        self._rgb = np.ascontiguousarray(rgb)
        self.height, self.width = self._rgb.shape[:2]
        self.fps = float(fps)
        self.frame_count = int(max(1, frames))
        self.scale = 1.0
        self.info = VideoInfo(path=self.path, width=self.width, height=self.height,
                              fps=self.fps, frame_count=self.frame_count,
                              duration=self.frame_count / max(1e-6, self.fps),
                              has_audio=False, codec="still")

    def frame(self, index: int) -> np.ndarray:
        return self._rgb.copy()

    def prefetch(self, start: int = 0, count: int = 8) -> None:
        return None

    def warm(self, count: int = 8) -> None:
        return None

    def clear(self) -> None:
        return None

    @property
    def is_still(self) -> bool:
        return True


# ---------------------------------------------------------------------------
_SYNTH_CHART_CACHE: dict[tuple[int, int], np.ndarray] = {}


class SyntheticSource:
    """
    A procedurally generated clip.  Used for the self-test and for the in-app
    "no file yet" demo project, so the pipeline can always be demonstrated.
    """

    def __init__(self, width: int = 720, height: int = 480, frames: int = 150,
                 fps: float = 30000.0 / 1001.0, variant: str = "chart"):
        self.width, self.height = int(width), int(height)
        self.frame_count = int(frames)
        self.fps = float(fps)
        self.variant = variant
        self.info = VideoInfo(path="<synthetic>", width=self.width, height=self.height,
                              fps=self.fps, frame_count=self.frame_count,
                              duration=self.frame_count / self.fps, has_audio=False,
                              codec="synthetic")
        self._base_chart: np.ndarray | None = None
        if self.variant == "chart":
            key = (self.height, self.width)
            base = _SYNTH_CHART_CACHE.get(key)
            if base is None:
                h, w = self.height, self.width
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                nx = xx / max(1, w - 1)
                ny = yy / max(1, h - 1)
                img = np.zeros((h, w, 3), dtype=np.float32)
                bars = np.array([
                    [1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.15, 0.15, 0.15],
                ], dtype=np.float32)
                seg = np.clip((nx * 8).astype(np.int32), 0, 7)
                img[: int(h * 0.45)] = bars[seg][: int(h * 0.45)]
                step = ((nx[..., None] > 0.5) * 1.0).repeat(3, axis=2)
                img[int(h * 0.45): int(h * 0.62)] = step[int(h * 0.45): int(h * 0.62)]
                r = np.sqrt((nx - 0.5) ** 2 + ((ny - 0.78) * w / h) ** 2)
                plate = 0.5 + 0.5 * np.cos(2 * np.pi * 60.0 * r ** 2)
                zone = plate[int(h * 0.62):, :, None] * np.array([1.0, 1.0, 1.0], dtype=np.float32)
                img[int(h * 0.62):] = zone * 0.9
                base = np.clip(img, 0.0, 1.0).astype(np.float32)
                _SYNTH_CHART_CACHE[key] = base
            self._base_chart = base

    def frame(self, index: int) -> np.ndarray:
        t = index / max(1e-6, self.fps)
        h, w = self.height, self.width
        if self.variant == "chart" and self._base_chart is not None:
            img = self._base_chart.copy()
            cx = int((0.1 + 0.8 * (0.5 + 0.5 * np.sin(t * 2 * np.pi))) * (w - 1))
            cy = int(0.70 * h)
            img[max(0, cy - 8):cy + 8, max(0, cx - 8):cx + 8] = np.array(
                [0.05, 0.05, 0.05], dtype=np.float32)
            return img

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = xx / max(1, w - 1)
        ny = yy / max(1, h - 1)
        img = np.zeros((h, w, 3), dtype=np.float32)
        ramp = (0.5 + 0.5 * np.sin(2 * np.pi * (nx * 3 + t * 0.4)))
        img[..., 0] = ramp
        img[..., 1] = 0.5 + 0.4 * np.cos(2 * np.pi * (ny * 5 - t * 0.3))
        img[..., 2] = 0.6 * ramp + 0.2
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def prefetch(self, start: int, count: int) -> None:
        pass

    def warm(self, count: int = 8) -> None:
        pass

    def clear(self) -> None:
        pass
