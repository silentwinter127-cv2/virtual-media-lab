"""
The renderer: turns a project (source clip + five-layer parameters + seed) into
frames, with the structure that makes interactive adjustment possible.

    ---------------------------------------------------------------------
    | camera | format | tape | player | display |     stage chain
    ---------------------------------------------------------------------
        ^         ^       ^        ^         ^
        |         |       |        |         |
      cached per stage, keyed by the hash of (seed + all upstream layers)
"""
from __future__ import annotations

import threading

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from ..core.pipeline import Chain, FrameBuffer, RenderContext, Stage
from ..core.profiles import SCHEMA
from ..core.signal import SignalFrame
from ..core.standards import get_standard
from ..stages.camera import CameraStage
from ..stages.film import FilmStage
from ..stages.codec import CodecStage
from ..stages.connector import ConnectorStage
from ..stages.display import DisplayStage
from ..stages.format import FormatStage
from ..stages.overlay import OverlayStage
from ..stages.player import PlayerStage
from ..stages.tape import TapeStage

STAGE_ORDER = ("camera", "film", "format", "codec", "connector", "tape", "player",
               "display", "overlay")
STAGE_LABELS = {
    "camera": "01 成像",
    "film": "01b 胶片",
    "format": "02 编码",
    "codec": "02b DV",
    "connector": "02c 接口/RF",
    "tape": "03 介质",
    "player": "04 读取",
    "display": "05 显示",
    "overlay": "06 字幕机",
}
# which parameter layers feed which stage
STAGE_LAYERS = {
    "camera": ("camera",),
    "film": ("film", "camera"),
    "format": ("format",),
    "codec": ("codec",),
    # the connectors and RF link sit between the source and the recorder, so a
    # tape or player change must not dirty them
    "connector": ("connector", "rf"),
    "tape": ("tape",),
    "player": ("player", "format"),
    "display": ("display",),
    # the generator reads only its own layer, so editing the text does not dirty
    # the camera or the tape
    "overlay": ("overlay",),
}


def _hash(obj) -> str:
    return hashlib.blake2s(
        json.dumps(obj, sort_keys=True, default=str).encode(), digest_size=10
    ).hexdigest()


def _fast_clone_meta(meta: dict | None) -> dict:
    """
    Fast clone of FrameBuffer.meta without generic Python `copy.deepcopy`.
    Clones mutable SignalFrame planes while keeping read-only tap RGB arrays
    and telemetry dicts by reference.
    """
    if not meta:
        return {}
    out: dict = {}
    for k, v in meta.items():
        if k in ("dropout_mask", "crease_mask", "line_phase"):
            continue
        if isinstance(v, SignalFrame):
            out[k] = v.copy()
        elif k == "bus_b_yiq" and isinstance(v, tuple) and len(v) == 3:
            out[k] = (v[0].copy(), v[1].copy(), v[2].copy())
        elif isinstance(v, dict):
            out[k] = dict(v)
        else:
            out[k] = v
    return out


@dataclass
class DirtyInfo:
    first_dirty_stage: str | None = None
    first_dirty_frame: int = 0
    cached_frames: int = 0

    def to_dict(self) -> dict:
        return {
            "first_dirty_stage": self.first_dirty_stage,
            "first_dirty_frame": self.first_dirty_frame,
            "cached_frames": self.cached_frames,
        }


class Renderer:
    """
    `source` must provide frames by index: `source.frame(index) -> np.ndarray`
    (H, W, 3) float32 in 0..1 (:class:`atsim.io.source.ClipSource`).
    """

    def __init__(self, source, width: int, height: int, fps: float,
                 params: dict[str, dict[str, float]] | None = None,
                 seed: int = 20240101, std_key: str = "ntsc",
                 preview: bool = False, scale: float = 1.0,
                 max_pump: int = 1200, total_frames: int | None = None,
                 streaming: bool = False, max_cache_frames: int = 8):
        self.source = source
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.seed = int(seed)
        self.std = get_standard(std_key)
        self.preview = bool(preview)
        self.scale = float(scale)
        self.max_pump = int(max_pump)
        self.streaming = bool(streaming)
        self.max_cache_frames = int(max(1, max_cache_frames))
        self.total_frames = int(total_frames or getattr(source, "frame_count", 0) or 0)
        self.schema = SCHEMA
        self.params = self.schema.resolve_all(params or {})

        self.chain: Chain | None = None
        self.stage: dict[str, Stage] = {}
        self._layer_hashes: dict[str, str] = self._compute_hashes()
        self._own_hashes: dict[str, str] = self._compute_own_hashes()
        self._osd_at: int = int(round(float(self.params.get("overlay", {}).get("at", 0.0)) or 0.0))
        self.dirty_from = STAGE_ORDER[0]
        self.dirty_frame = 0
        self._cache: dict[str, dict[int, np.ndarray]] = {s: {} for s in STAGE_ORDER}
        self._cache_buf: dict[str, dict[int, tuple]] = {s: {} for s in STAGE_ORDER}
        self._keyframes = None
        self._environment = None
        self._range_cache = None
        self._layer_cache = None
        self._prev_src: np.ndarray | None = None
        self._pumps = 0
        self.last_render_ms = 0.0
        self.stage_ms: dict[str, float] = {}
        self._render_lock = threading.RLock()

    # ------------------------------------------------------------------
    # hashing / invalidation
    # ------------------------------------------------------------------
    def _compute_hashes(self) -> dict[str, str]:
        h: dict[str, str] = {}
        acc: dict = {"seed": self.seed, "std": self.std.name, "w": self.width,
                     "h": self.height, "fps": round(self.fps, 6)}
        for stage_name in STAGE_ORDER:
            for layer in STAGE_LAYERS[stage_name]:
                acc[f"{stage_name}/{layer}"] = self.params.get(layer, {})
            h[stage_name] = _hash(acc)
        return h

    def _compute_own_hashes(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for stage_name in STAGE_ORDER:
            sub = {layer: self.params.get(layer, {}) for layer in STAGE_LAYERS[stage_name]}
            out[stage_name] = _hash(sub)
        return out

    def invalidate(self, params: dict[str, dict[str, float]] | None = None) -> DirtyInfo:
        if params is not None:
            self.params = self.schema.resolve_all(params)
        new = self._compute_hashes()
        new_own = self._compute_own_hashes()
        first: str | None = None
        for stage_name in STAGE_ORDER:
            if new[stage_name] != self._layer_hashes.get(stage_name):
                first = stage_name
                break
        if first is None:
            return DirtyInfo(None, 0, self._cached_count())

        idx = STAGE_ORDER.index(first)
        dirty_stage_names = set(STAGE_ORDER[idx:])
        new_osd_at = int(round(float(self.params.get("overlay", {}).get("at", 0.0)) or 0.0))
        if self.chain is None or new_osd_at != self._osd_at:
            self._build_chain()
        else:
            self.chain.ctx.params = self.params
            for idx_s, st in enumerate(self.chain.stages):
                if st.name in dirty_stage_names or new_own.get(st.name) != self._own_hashes.get(st.name):
                    new_st = self._make_stage(st.name, self.chain.ctx, new_osd_at)
                    new_st.prepare()
                    self.chain.stages[idx_s] = new_st
                    self.stage[st.name] = new_st
                    if st.name == "camera":
                        self._prev_src = None

        self._layer_hashes = new
        self._own_hashes = new_own
        idx = STAGE_ORDER.index(first)
        for stage_name in STAGE_ORDER[idx:]:
            self._cache[stage_name] = {}
            self._cache_buf[stage_name] = {}
        self.dirty_from = first
        frames = self._cache[STAGE_ORDER[idx - 1]] if idx > 0 else {}
        self.dirty_frame = min(frames) if frames else 0
        return DirtyInfo(first, self.dirty_frame, self._cached_count())

    def _cached_count(self) -> int:
        return len(self._cache["display"])

    def cached_prefix_length(self) -> int:
        """Length of the contiguous cached run starting at frame 0."""
        n = 0
        cache = self._cache["display"]
        while n in cache:
            n += 1
        return n

    # ------------------------------------------------------------------
    # chain construction
    # ------------------------------------------------------------------
    def _make_stage(self, name: str, ctx: RenderContext, osd_at: int) -> Stage:
        if name == "camera":
            return CameraStage(ctx)
        if name == "film":
            return FilmStage(ctx)
        if name == "format":
            return FormatStage(ctx)
        if name == "codec":
            return CodecStage(ctx)
        if name == "connector":
            return ConnectorStage(ctx)
        if name == "tape":
            return TapeStage(ctx)
        if name == "player":
            return PlayerStage(ctx)
        if name == "display":
            return DisplayStage(ctx)
        if name == "overlay":
            slot = OverlayStage.SLOT_PRE_TAPE if osd_at == 1 else OverlayStage.SLOT_TAIL
            return OverlayStage(ctx, slot=slot)
        raise KeyError(name)

    def _build_chain(self) -> None:
        ctx = RenderContext(
            std=self.std, params=self.params, seed=self.seed,
            frame_count=self.total_frames, fps=self.fps, width=self.width,
            height=self.height, preview=self.preview, scale=self.scale,
            source=self.source,
        )
        chain = Chain(ctx)
        chain.add(CameraStage(ctx)).add(FilmStage(ctx)).add(FormatStage(ctx)).add(CodecStage(ctx))
        chain.add(ConnectorStage(ctx))
        _osd_at = int(round(float(self.params.get("overlay", {}).get("at", 0.0)) or 0.0))
        self._osd_at = _osd_at
        if _osd_at == 1:
            chain.add(OverlayStage(ctx, slot=OverlayStage.SLOT_PRE_TAPE))
        chain.add(TapeStage(ctx)).add(PlayerStage(ctx)).add(DisplayStage(ctx))
        if _osd_at != 1:
            chain.add(OverlayStage(ctx, slot=OverlayStage.SLOT_TAIL))
        chain.prepare()
        self.chain = chain
        self.stage = {s.name: s for s in chain.stages}
        self._own_hashes = self._compute_own_hashes()
        self._prev_src = None

    def plan(self) -> dict:
        return {
            "stages": [
                {"name": s, "label": STAGE_LABELS[s],
                 "cached": len(self._cache.get(s, {})),
                 "dirty": STAGE_ORDER.index(s) >= STAGE_ORDER.index(self.dirty_from),
                 "hash": self._layer_hashes.get(s, "")[:8]}
                for s in STAGE_ORDER
            ],
            "cached_frames": self._cached_count(),
            "dirty_from": self.dirty_from,
            "dirty_frame": self.dirty_frame,
            "stage_ms": {k: round(v, 2) for k, v in self.stage_ms.items()},
            "last_render_ms": round(self.last_render_ms, 2),
        }

    def timeline(self) -> list[dict]:
        if self.chain is None:
            self._build_chain()
        assert self.chain is not None
        return self.chain.defect_timeline()

    def reset(self) -> None:
        self._cache = {s: {} for s in STAGE_ORDER}
        self._cache_buf = {s: {} for s in STAGE_ORDER}
        self.dirty_from = STAGE_ORDER[0]
        self.dirty_frame = 0
        self._build_chain()

    # ------------------------------------------------------------------
    # frame rendering
    # ------------------------------------------------------------------
    def _fit_source(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if (w, h) == (int(self.width), int(self.height)):
            return frame
        arr = np.asarray(frame, dtype=np.float32)
        ys = np.clip((np.arange(int(self.height)) + 0.5) * h / max(1, int(self.height)),
                     0, h - 1).astype(np.int32)
        xs = np.clip((np.arange(int(self.width)) + 0.5) * w / max(1, int(self.width)),
                     0, w - 1).astype(np.int32)
        out = arr[ys][:, xs]
        if out.ndim == 2 and frame.ndim == 3:
            out = np.repeat(out[:, :, None], frame.shape[2], axis=2)
        return out

    def set_time_varying(self, keyframes=None, environment=None) -> None:
        """
        Parameters that change during the clip: keyframe tracks, and the machine's condition.
        """
        if isinstance(keyframes, list):
            if keyframes:
                from ..core.keyframes import KeyframeSet, Track
                keyframes = KeyframeSet(tracks=[
                    t if isinstance(t, Track) else Track.from_dict(t)
                    for t in keyframes
                ])
            else:
                keyframes = None
        if isinstance(environment, dict):
            if environment:
                from ..core.environment import Environment
                environment = Environment(
                    warmth=float(environment.get("warmth", 1.0)),
                    age=float(environment.get("age", 0.0)),
                    humidity=float(environment.get("humidity", 0.0)),
                    cold_sensitivity=float(environment.get("cold_sensitivity", 1.0)),
                    age_sensitivity=float(environment.get("age_sensitivity", 1.0)),
                )
            else:
                environment = None
        self._keyframes = keyframes
        self._environment = environment

    def _apply_frame_params(self, index: int) -> None:
        if self._keyframes is None and self._environment is None:
            return
        params = {layer: dict(vals) for layer, vals in self.params.items()}
        t = index / max(1e-6, self.fps)
        if self._environment is not None:
            from ..core.environment import apply_environment
            params, _rec = apply_environment(params, self._environment, self._ranges(),
                                             self._layer_of)
        if self._keyframes is not None and hasattr(self._keyframes, "apply"):
            params, _rec = self._keyframes.apply(params, t, self._ranges())
        ctx = self.chain.ctx
        ctx.params = params
        for stage in self.chain.stages:
            stage.refresh_params()

    def _ranges(self) -> dict[str, tuple[float, float]]:
        if getattr(self, "_range_cache", None) is None:
            cache: dict[str, tuple[float, float]] = {}
            for layer in SCHEMA.layer_names():
                for key, spec in SCHEMA.layer(layer).specs.items():
                    cache[key] = (float(spec.minimum), float(spec.maximum))
            self._range_cache = cache
        return self._range_cache

    def _layer_of(self, key: str) -> str | None:
        if getattr(self, "_layer_cache", None) is None:
            cache: dict[str, str] = {}
            for layer in SCHEMA.layer_names():
                for k in SCHEMA.layer(layer).keys():
                    cache.setdefault(k, layer)
            self._layer_cache = cache
        return self._layer_cache.get(key)

    def _render_index(self, index: int, src_frame: np.ndarray | None) -> np.ndarray:
        if self.chain is None:
            self._build_chain()
        assert self.chain is not None

        self._apply_frame_params(index)

        start = 0
        buf: FrameBuffer | None = None
        for i in range(len(STAGE_ORDER) - 1, 0, -1):
            up_name = STAGE_ORDER[i - 1]
            cached = self._cache[up_name].get(index)
            if cached is not None:
                aux = self._cache_buf.get(up_name, {}).get(index)
                if aux is not None:
                    comp_c, meta_c = aux
                    comp_copy = comp_c.copy() if comp_c is not None else None
                    meta_copy = _fast_clone_meta(meta_c)
                else:
                    comp_copy = None
                    meta_copy = {"field": index % 2}
                buf = FrameBuffer(np.asarray(cached, dtype=np.float32).copy(), comp_copy, index,
                                  index / max(1e-6, self.fps), self.std,
                                  meta_copy)
                start = i
                break

        if start == 0:
            if src_frame is None:
                src_frame = self.source.frame(index)
            src_frame = self._fit_source(src_frame)
            cam: CameraStage = self.stage["camera"]  # type: ignore[assignment]
            if getattr(cam, "motion_blur", 0.0) > 1e-4:
                if self._prev_src is None and index > 0:
                    try:
                        self._prev_src = self.source.frame(index - 1)
                    except Exception:
                        self._prev_src = None
                if self._prev_src is not None:
                    self._prev_src = self._fit_source(self._prev_src)
                cam.set_previous(self._prev_src)
            else:
                cam.set_previous(None)
            buf = FrameBuffer(np.asarray(src_frame, dtype=np.float32), None, index,
                              index / max(1e-6, self.fps), self.std,
                              {"field": index % 2})

        t0 = time.perf_counter()
        capture_from = STAGE_ORDER.index(self.dirty_from)
        pending: list[tuple[str, np.ndarray, np.ndarray | None, dict]] = []
        for pos in range(start, len(STAGE_ORDER)):
            st = self.chain.stages[pos]
            t = time.perf_counter()
            buf = st.apply(buf)  # type: ignore[arg-type]
            dt = (time.perf_counter() - t) * 1000.0
            self.stage_ms[st.name] = self.stage_ms.get(st.name, 0.0) * 0.7 + dt * 0.3
            if not self.streaming and pos >= capture_from and pos < len(STAGE_ORDER) - 1:
                pending.append((st.name, buf.rgb,
                                buf.composite.copy() if buf.composite is not None else None,
                                _fast_clone_meta(buf.meta)))
        self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        if not self.streaming:
            limit_f = self.max_cache_frames
            for name, rgb, comp_c, meta_c in pending:
                c_rgb = self._cache[name]
                c_buf = self._cache_buf[name]
                c_rgb[index] = rgb
                c_buf[index] = (comp_c, meta_c)
                while len(c_rgb) > limit_f:
                    old_k = next(iter(c_rgb))
                    c_rgb.pop(old_k, None)
                    c_buf.pop(old_k, None)
            c_disp = self._cache["display"]
            c_disp[index] = buf.rgb  # type: ignore[union-attr]
            while len(c_disp) > limit_f:
                old_k = next(iter(c_disp))
                c_disp.pop(old_k, None)
        if src_frame is not None:
            self._prev_src = np.asarray(src_frame, dtype=np.float32)
        return buf.rgb  # type: ignore[union-attr]

    def needs_sequential_state(self) -> bool:
        cam_p = self.params.get("camera") or {}
        fmt_p = self.params.get("format") or {}
        tpe_p = self.params.get("tape") or {}
        return bool(
            float(cam_p.get("ae_enabled", 0.0)) > 0.5
            or float(cam_p.get("awb_enabled", 0.0)) > 0.5
            or float(cam_p.get("af_enabled", 0.0)) > 0.5
            or float(cam_p.get("nv_mode", 0.0)) > 0.5
            or float(fmt_p.get("chroma_lag", 0.0)) > 1e-4
            or float(tpe_p.get("shuttle_speed", 0.0)) > 1e-4
            or int(round(float(tpe_p.get("transport_state", 0.0)))) != 0
        )

    def render_frame_u8_with_src(self, index: int, src_u8: np.ndarray,
                                 prev_src_u8: np.ndarray | None = None) -> np.ndarray:
        """Fast worker entry point: takes uint8 source frame(s) and returns uint8 rendered RGB."""
        scale_f = np.float32(1.0 / 255.0)
        try:
            import cv2 as _cv2
            lut = getattr(Renderer, "_U8_LUT", None)
            if lut is None:
                lut = (np.arange(256, dtype=np.float32) * scale_f).reshape(256, 1)
                Renderer._U8_LUT = lut
            src_f32 = _cv2.LUT(src_u8, lut)
            self._prev_src = _cv2.LUT(prev_src_u8, lut) if prev_src_u8 is not None else None
        except Exception:
            src_f32 = src_u8.astype(np.float32) * scale_f
            self._prev_src = (prev_src_u8.astype(np.float32) * scale_f) if prev_src_u8 is not None else None
        rgb = self._render_index(int(index), src_f32)
        return (np.clip(rgb, 0.0, 1.0) * np.float32(255.0) + np.float32(0.5)).astype(np.uint8)

    def render_frames(self, indices: list[int], limit: int | None = None,
                      should_stop: Callable[[], bool] | None = None
                      ) -> Iterator[tuple[int, np.ndarray]]:
        wanted = sorted({int(i) for i in indices})
        if not wanted:
            return
        wanted_set = set(wanted)
        if self.chain is None:
            self._build_chain()

        if all(i in self._cache["display"] for i in wanted):
            for i in wanted:
                yield i, self._cache["display"][i]
            return

        top = max(wanted)
        first_w = min(wanted)
        start_idx = 0
        needs_multi = self.needs_sequential_state()
        if (self.preview and len(wanted) == 1) or self.streaming:
            if not needs_multi:
                start_idx = first_w
            else:
                start_idx = max(0, first_w - 6)
        pumped = 0
        for index in range(start_idx, top + 1):
            if should_stop is not None and should_stop():
                return
            if index in self._cache["display"]:
                if index in wanted_set:
                    yield index, self._cache["display"][index]
                continue
            if limit is not None and pumped >= limit:
                return
            src = None
            if STAGE_ORDER.index(self.dirty_from) == 0 \
                    and index not in self._cache["camera"]:
                src = self.source.frame(index)
            try:
                rgb = self._render_index(index, src)
            except Exception:
                for stage_name in STAGE_ORDER:
                    self._cache[stage_name].pop(index, None)
                raise
            pumped += 1
            self._pumps += 1
            if index in wanted_set:
                yield index, rgb

    def render_frame(self, index: int) -> np.ndarray:
        with self._render_lock:
            for _, rgb in self.render_frames([index]):
                return rgb
        raise RuntimeError(f"could not render frame {index}")

    # ------------------------------------------------------------------
    def still(self, index: int) -> np.ndarray:
        """Render one frame without disturbing the cache (used by A/B splits)."""
        saved = {s: dict(self._cache[s]) for s in STAGE_ORDER}
        saved_buf = {s: dict(self._cache_buf[s]) for s in STAGE_ORDER}
        try:
            return self.render_frame(index)
        finally:
            self._cache = saved
            self._cache_buf = saved_buf
