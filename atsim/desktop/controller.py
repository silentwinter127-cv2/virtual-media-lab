"""In-process low-memory controller and renderers for the Virtual Media Lab Desktop GUI.

Why the WebUI consumes heavy resources:
1. Browser engine (Chromium/Edge/Firefox) allocates 300-800 MB RAM for DOM, 2D canvases, and decoded image bitmaps.
2. Every slider tweak triggers HTTP loopback requests (`/api/frame`, `/api/scopes`, `/api/timeline`, `/api/audio_timeline`),
   serializing full frames to PNG/JPEG on the Python side (or spawning FFmpeg for JPEG) and decoding them again in browser threads.
3. Unbounded or large `Renderer` / `ClipSource` frame caches + multiple `GraphExecutor` instances retain float32 tensors.

How `DesktopBackendController` solves this:
1. Zero HTTP server, zero PNG/JPEG encoding during interactive preview: frames stay as raw `float32/uint8 (H, W, 3)` arrays in memory and convert directly to `PIL.Image` / `ImageTk.PhotoImage`.
2. Strict low-memory cap (`max_cache_frames=2`, single active `Renderer` / `GraphExecutor`, explicit `free_memory()` with `gc.collect()`).
3. Lazy on-demand computation: Broadcast Scopes (Waveform / Vectorscope / PSD) and Audio synthesis only compute when requested or when the scopes drawer is open.
"""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent.parent):
    if (_cand / "atsim").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

try:
    from ..core.profiles import SCHEMA
    from ..web.server import App
except (ImportError, ValueError):
    from atsim.core.profiles import SCHEMA
    from atsim.web.server import App


# Preset category tabs matching WebUI buildPresetTabs()
PRESET_CATEGORIES: List[Tuple[str, str]] = [
    ("factory", "整套配置"),
    ("digital", "古早数字/手机/流媒体"),
    ("custom", "★ 自定义预设"),
    ("film", "胶片/LOMO"),
    ("media", "记录媒介"),
    ("player", "播放设备"),
    ("display", "显示设备"),
    ("camera", "摄像机"),
]


def _load_selectors_from_webui() -> Dict[str, List[Tuple[int, str]]]:
    """Dynamically parse `const SELECTORS = { ... };` from `atsim/web/static/app.js`."""
    out: Dict[str, List[Tuple[int, str]]] = {}
    here = Path(__file__).resolve().parent
    app_js = here.parent / "web" / "static" / "app.js"
    for cand in (
        here.parent / "web" / "static" / "app.js",
        here / "atsim" / "web" / "static" / "app.js",
        here / "app.js",
    ):
        if cand.exists():
            app_js = cand
            break
    if app_js.exists():
        try:
            text = app_js.read_text(encoding="utf-8")
            s_idx = text.find("const SELECTORS = {")
            e_idx = text.find("\n};", s_idx)
            if s_idx != -1 and e_idx != -1:
                body = text[s_idx : e_idx + 3]
                for km in re.finditer(r'"([a-zA-Z0-9_.]+)"\s*:\s*\[((?:\s*\[[^\]]+\]\s*,?)+)\s*\]', body):
                    opts = [
                        (int(m.group(1)), m.group(2))
                        for m in re.finditer(r'\[\s*(-?\d+)\s*,\s*"([^"]+)"\s*\]', km.group(2))
                    ]
                    if opts:
                        out[km.group(1)] = opts
        except Exception:
            pass
    return out


SELECTORS: Dict[str, List[Tuple[int, str]]] = _load_selectors_from_webui()


def get_process_memory_mb() -> float:
    """Return current process RSS memory in MB (cross-platform: Linux, Windows, macOS)."""
    try:
        if sys.platform.startswith("linux") and os.path.exists("/proc/self/statm"):
            with open("/proc/self/statm", "r", encoding="utf-8") as f:
                parts = f.read().split()
                if len(parts) >= 2:
                    rss_pages = int(parts[1])
                    page_size = os.sysconf("SC_PAGE_SIZE")
                    return (rss_pages * page_size) / (1024.0 * 1024.0)
        elif sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return pmc.WorkingSetSize / (1024.0 * 1024.0)
        else:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return ru / (1024.0 * 1024.0)
            return ru / 1024.0
    except Exception:
        pass
    return 0.0


class DesktopBackendController:
    """Zero-HTTP, low-memory in-process controller for Virtual Media Lab Desktop GUI."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        default_clip: Optional[str] = None,
        low_memory: bool = True,
    ) -> None:
        self.low_memory = bool(low_memory)
        self.data_dir = Path(data_dir) if data_dir else (Path.cwd() / ".vml_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.app = App(data_dir=self.data_dir, default_clip=default_clip)

        # Schema & Catalog cached once in memory
        self.schema_dict: Dict[str, Any] = SCHEMA.schema_dict()
        self.catalog: Dict[str, Any] = self.app.store.catalog()

        # Interactive Desktop State (matching WebUI S object)
        self.t: float = 0.40
        self.preview_scale: str = "0.35" if self.low_memory else "0.5"
        self.export_scale: str = "1.0"
        self.export_quality: str = "high"
        self.ab_enabled: bool = False
        self.split_x: float = 0.50
        self.auto_refresh: bool = True
        self.use_graph: bool = False
        self.preset_tab: str = "factory"
        self.active_layer: str = "camera"

        # Quad bar state
        self.quad_corner: Optional[int] = None
        self.quad_border: Optional[int] = None
        self.quad_text: Optional[int] = None

        # Scopes & Audio state
        self.scopes_open: bool = False
        self.scope_line_y: float = 0.5
        self.scope_ab_mode: str = "spectrum"
        self.audio_clean: bool = False

        # Minimal single-frame memory buffers
        self.last_sim_u8: Optional[np.ndarray] = None
        self.last_src_u8: Optional[np.ndarray] = None
        self.last_comp_u8: Optional[np.ndarray] = None
        self.last_render_ms: float = 0.0
        self.last_frame_label: str = "静帧"
        self.last_audio_tl: Optional[Dict[str, Any]] = None
        self._lock = threading.RLock()

        self._enforce_low_memory_caches()

    def _enforce_low_memory_caches(self) -> None:
        """Bound internal Renderer, SessionStore, and GraphExecutor caches so RAM stays flat."""
        if not self.low_memory:
            return
        try:
            store = self.app.store
            if len(store._renderers) > 1:
                keys = list(store._renderers.keys())
                for k in keys[:-1]:
                    r_old = store._renderers.pop(k, None)
                    if r_old is not None:
                        r_old._cache.clear()
                        r_old._cache_buf.clear()
            for r in store._renderers.values():
                r.max_cache_frames = 2
                for st_cache in r._cache.values():
                    while len(st_cache) > 2:
                        oldest = next(iter(st_cache))
                        st_cache.pop(oldest, None)
                for st_buf in r._cache_buf.values():
                    while len(st_buf) > 2:
                        oldest = next(iter(st_buf))
                        st_buf.pop(oldest, None)
            if len(self.app._graph_execs) > 1:
                gkeys = list(self.app._graph_execs.keys())
                for k in gkeys[:-1]:
                    self.app._graph_execs.pop(k, None)
            self.app._clean.clear()
        except Exception:
            pass

    def free_memory(self) -> Dict[str, float]:
        """One-click memory release: drop all cached stage frames, renderers, and run GC."""
        before_mb = get_process_memory_mb()
        with self._lock:
            try:
                for r in self.app.store._renderers.values():
                    r._cache.clear()
                    r._cache_buf.clear()
                self.app.store._renderers.clear()
                for s in self.app.store._sources.values():
                    if hasattr(s, "clear"):
                        s.clear()
                self.app.store._sources.clear()
                self.app._graph_execs.clear()
                self.app._clean.clear()
                self.last_src_u8 = None
            except Exception:
                pass
            gc.collect()
        after_mb = get_process_memory_mb()
        return {
            "before_mb": round(before_mb, 1),
            "after_mb": round(after_mb, 1),
            "freed_mb": round(max(0.0, before_mb - after_mb), 1),
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """Return live RSS memory, render latency, and active source resolution."""
        info = self.app.current.source_info or {}
        h, w = (0, 0)
        if self.last_sim_u8 is not None:
            h, w = self.last_sim_u8.shape[:2]
        fps = float(self.app.current.fps or 29.97)
        frames = int(info.get("frame_count") or 120)
        dur = round(frames / max(1.0, fps), 2)
        return {
            "rss_mb": round(get_process_memory_mb(), 1),
            "render_ms": round(self.last_render_ms, 1),
            "preview_w": w,
            "preview_h": h,
            "source_name": Path(self.app.current.source).name if self.app.current.source else "内置测试图 (SMPTE)",
            "source_w": int(info.get("width") or 720),
            "source_h": int(info.get("height") or 480),
            "fps": round(fps, 2),
            "frame_count": frames,
            "duration": dur,
            "seed": int(self.app.current.seed),
            "preset": self.app.current.preset or "custom",
        }

    # --- Source & Overlay File Management ---
    def import_clip(self, file_path: str) -> Dict[str, Any]:
        """Import a video or image file directly without HTTP multipart encoding."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"找不到素材文件: {file_path}")
        dest = self.app.store.uploads / p.name
        if p.resolve() != dest.resolve():
            shutil.copy2(p, dest)
        res = self.app.set_project({"source": dest.name})
        self.free_memory()
        return res

    def import_overlay(self, file_path: str) -> Dict[str, Any]:
        """Import a switcher/PIP overlay image or video."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"找不到叠层文件: {file_path}")
        dest = self.app.store.uploads / f"overlay_{p.name}"
        if p.resolve() != dest.resolve():
            shutil.copy2(p, dest)
        res = self.app.set_project({
            "params": {
                "connector": {
                    "switcher_overlay_path": str(dest),
                    "switcher_b_source": 3.0,
                }
            }
        })
        return res

    def clear_overlay(self) -> Dict[str, Any]:
        return self.app.set_project({
            "params": {
                "connector": {
                    "switcher_overlay_path": "",
                    "switcher_b_source": 0.0,
                }
            }
        })

    # --- Presets & Parameters ---
    def get_preset_entries(self, tab_key: Optional[str] = None, search_query: str = "") -> List[Tuple[str, str, str]]:
        """Return `[(key, label, note), ...]` for the active preset tab, matching WebUI `presetEntries()`."""
        tab = tab_key or self.preset_tab
        self.catalog = self.app.store.catalog()
        entries: List[Tuple[str, str, str]] = []
        if tab == "factory":
            for k, v in (self.catalog.get("factory") or {}).items():
                entries.append((k, v.get("label", k), v.get("note") or "整套：摄像机+格式+介质+播放"))
        elif tab == "custom":
            for k, v in (self.catalog.get("custom") or {}).items():
                entries.append((k, v.get("label", k), v.get("note") or "用户自定义保存预设"))
        elif tab == "digital":
            profs = ((self.catalog.get("profiles") or {}).get("digital")) or {}
            for k, v in profs.items():
                entries.append((k, v.get("label", k), v.get("note") or "古早数字/手机/流媒体预设"))
            if not entries:
                for k, v in (self.catalog.get("factory") or {}).items():
                    if str(k).startswith("ldme_"):
                        entries.append((k, v.get("label", k), v.get("note") or "古早数字媒介配置"))
        else:
            grp = ((self.catalog.get("profiles") or {}).get(tab)) or {}
            for k, v in grp.items():
                entries.append((k, v.get("label", k), v.get("note") or ""))

        q = search_query.strip().lower()
        if q:
            entries = [
                (k, lbl, note)
                for (k, lbl, note) in entries
                if q in k.lower() or q in lbl.lower() or q in note.lower()
            ]
        return entries

    def apply_preset(self, key: str) -> Dict[str, Any]:
        with self._lock:
            self.quad_corner = None
            self.quad_border = None
            self.quad_text = None
            res = self.app.apply_preset(key)
            self._enforce_low_memory_caches()
            return res

    def save_custom_preset(self, label: str, note: str = "") -> Dict[str, Any]:
        with self._lock:
            res = self.app.save_custom_preset({"label": label, "note": note})
            self.catalog = self.app.store.catalog()
            return res

    def delete_custom_preset(self, key: str) -> Dict[str, Any]:
        with self._lock:
            res = self.app.delete_custom_preset({"key": key})
            self.catalog = self.app.store.catalog()
            return res

    def resolved_params(self) -> Dict[str, Dict[str, Any]]:
        return self.app.current.resolved_params()

    def set_param(self, layer: str, key: str, value: Any) -> Dict[str, Any]:
        with self._lock:
            res = self.app.set_levels(layer, {key: value})
            self._enforce_low_memory_caches()
            return res

    def set_seed(self, seed_val: int) -> Dict[str, Any]:
        with self._lock:
            return self.app.set_project({"seed": int(seed_val)})

    def selector_options_for(self, layer: str, sp: Dict[str, Any]) -> Optional[List[Tuple[int, str]]]:
        full_key = f"{layer}.{sp['key']}"
        if full_key == "overlay.font":
            fonts = [f for f in (self.catalog.get("fonts") or []) if f.get("available") is not False]
            if fonts:
                return [(i, f"{i} · {f.get('label', f.get('name', i))}") for i, f in enumerate(fonts)]
        return SELECTORS.get(full_key)

    # --- W-T Optical Zoom Helper ---
    def get_wt_readout(self, z_val: Optional[float] = None) -> str:
        cam = self.resolved_params().get("camera") or {}
        z = float(np.clip(z_val if z_val is not None else float(cam.get("wt_zoom", 0.0)), 0.0, 1.0))
        max_z = float(cam.get("wt_max_zoom", 4.0) or 4.0)
        mag = max_z ** z
        fnum = 1.4 + (2.9 - 1.4) * (z ** 1.15)
        state = "W 端光晕态" if z < 0.15 else ("中焦过渡态" if z < 0.55 else "T 端具体光源细节")
        return f"{mag:.2f}× · f/{fnum:.1f} · {state}"

    # --- Zero-Encode Frame Rendering (Returns Raw uint8 RGB & PIL.Image) ---
    def render_frame_numpy(self) -> Tuple[np.ndarray, float]:
        """Render the current frame in-process without any PNG/JPEG encoding overhead."""
        t0 = time.perf_counter()
        with self._lock:
            scale = self.app.resolve_scale(self.preview_scale, default=0.35)
            sim_f32, _slug, label_zh = self.app.render_still_with_quad(
                t=self.t,
                scale=scale,
                want="sim",
                corner=self.quad_corner,
                border=self.quad_border,
                text=self.quad_text,
                use_graph=self.use_graph,
            )
            sim_u8 = np.clip(sim_f32 * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
            self.last_sim_u8 = sim_u8
            self.last_frame_label = label_zh

            if self.ab_enabled:
                src_f32, _, _ = self.app.render_still_with_quad(
                    t=self.t,
                    scale=scale,
                    want="source",
                )
                src_u8 = np.clip(src_f32 * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
                if src_u8.shape[:2] != sim_u8.shape[:2]:
                    src_pil = Image.fromarray(src_u8).resize(
                        (sim_u8.shape[1], sim_u8.shape[0]), Image.Resampling.BILINEAR
                    )
                    src_u8 = np.asarray(src_pil, dtype=np.uint8)
                self.last_src_u8 = src_u8
                comp = sim_u8.copy()
                split_col = int(np.clip(self.split_x, 0.02, 0.98) * comp.shape[1])
                comp[:, split_col:, :] = src_u8[:, split_col:, :]
                c0 = max(0, split_col - 1)
                c1 = min(comp.shape[1], split_col + 1)
                comp[:, c0:c1, 0] = 0
                comp[:, c0:c1, 1] = 255
                comp[:, c0:c1, 2] = 102
                self.last_comp_u8 = comp
            else:
                self.last_comp_u8 = sim_u8

            self._enforce_low_memory_caches()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.last_render_ms = elapsed_ms
            return self.last_comp_u8, elapsed_ms

    def render_viewport_pil(self, canvas_w: int = 860, canvas_h: int = 500) -> Image.Image:
        """Render the current frame and fit it inside `(canvas_w, canvas_h)` as a PIL Image."""
        rgb, _ = self.render_frame_numpy()
        h, w = rgb.shape[:2]
        if w <= 0 or h <= 0:
            return Image.new("RGB", (max(64, canvas_w), max(64, canvas_h)), (11, 15, 24))
        fit_scale = min(max(64, canvas_w) / float(w), max(64, canvas_h) / float(h))
        new_w = max(1, int(round(w * fit_scale)))
        new_h = max(1, int(round(h * fit_scale)))
        img = Image.fromarray(rgb, mode="RGB")
        if (new_w, new_h) != (w, h):
            resample = Image.Resampling.NEAREST if fit_scale > 1.35 else Image.Resampling.BILINEAR
            img = img.resize((new_w, new_h), resample)

        if self.ab_enabled:
            draw = ImageDraw.Draw(img)
            sx = int(np.clip(self.split_x, 0.02, 0.98) * new_w)
            draw.rectangle([max(4, sx - 64), 8, max(4, sx - 6), 26], fill=(11, 15, 24))
            draw.text((max(8, sx - 58), 10), "A 仿真", fill=(0, 255, 102))
            draw.rectangle([min(new_w - 64, sx + 6), 8, min(new_w - 4, sx + 64), 26], fill=(11, 15, 24))
            draw.text((min(new_w - 58, sx + 10), 10), "B 原片", fill=(56, 189, 248))
        return img

    # --- Lazy Broadcast Scopes Rendering (Waveform / Vectorscope / PSD Spectrum) ---
    def render_scopes_pil(self, width: int = 860, height: int = 180) -> Tuple[Image.Image, Dict[str, Any]]:
        """Compute broadcast scopes in-process and render them to a composite PIL Image."""
        with self._lock:
            scale = min(0.5, self.app.resolve_scale(self.preview_scale, default=0.35))
            sc = self.app.scopes(
                t=self.t,
                scale=scale,
                line_y=self.scope_line_y,
                engine="graph" if self.use_graph else "chain",
            )
            self._enforce_low_memory_caches()

        canvas = Image.new("RGB", (max(320, width), max(120, height)), (8, 12, 20))
        draw = ImageDraw.Draw(canvas)
        panel_w = (canvas.width - 24) // 3
        panel_h = canvas.height - 28

        # 1. Waveform Monitor (IRE)
        wfm = sc.get("waveform") or {}
        luma_cols = wfm.get("luma_band") or {}
        p10 = luma_cols.get("p10") or []
        p50 = luma_cols.get("p50") or []
        p90 = luma_cols.get("p90") or []
        line_ire = wfm.get("line_ire") or []
        wx0, wy0 = 6, 22
        draw.rectangle([wx0, wy0, wx0 + panel_w, wy0 + panel_h], fill=(6, 14, 12), outline=(30, 58, 47))
        for ire_v in (-40, 0, 50, 100):
            gy = wy0 + panel_h - int(((ire_v + 45.0) / 160.0) * panel_h)
            gy = int(np.clip(gy, wy0 + 1, wy0 + panel_h - 1))
            draw.line([(wx0 + 2, gy), (wx0 + panel_w - 2, gy)], fill=(18, 42, 32), width=1)
        if len(p50) > 1:
            n = len(p50)
            pts_10, pts_50, pts_90 = [], [], []
            for i in range(n):
                px = wx0 + 4 + int((i / float(n - 1)) * (panel_w - 8))
                y10 = wy0 + panel_h - int(((float(p10[i]) + 45.0) / 160.0) * panel_h)
                y50 = wy0 + panel_h - int(((float(p50[i]) + 45.0) / 160.0) * panel_h)
                y90 = wy0 + panel_h - int(((float(p90[i]) + 45.0) / 160.0) * panel_h)
                pts_10.append((px, int(np.clip(y10, wy0 + 2, wy0 + panel_h - 2))))
                pts_50.append((px, int(np.clip(y50, wy0 + 2, wy0 + panel_h - 2))))
                pts_90.append((px, int(np.clip(y90, wy0 + 2, wy0 + panel_h - 2))))
            draw.line(pts_90, fill=(16, 120, 68), width=1)
            draw.line(pts_10, fill=(16, 120, 68), width=1)
            draw.line(pts_50, fill=(0, 255, 102), width=2)
        if len(line_ire) > 1:
            n_l = len(line_ire)
            pts_l = []
            for i in range(n_l):
                px = wx0 + 4 + int((i / float(n_l - 1)) * (panel_w - 8))
                yl = wy0 + panel_h - int(((float(line_ire[i]) + 45.0) / 160.0) * panel_h)
                pts_l.append((px, int(np.clip(yl, wy0 + 2, wy0 + panel_h - 2))))
            draw.line(pts_l, fill=(250, 204, 21), width=1)
        peak_ire = float(wfm.get("peak_white_ire", 100.0))
        sync_ire = float(wfm.get("sync_tip_ire", -40.0))
        draw.text((wx0 + 4, 4), f"IRE 波形 (同步 {sync_ire:.0f} | 峰值 {peak_ire:.0f} IRE)", fill=(0, 255, 102))

        # 2. Chroma Vectorscope
        vec = sc.get("vectorscope") or {}
        vx0 = 12 + panel_w
        vy0 = 22
        draw.rectangle([vx0, vy0, vx0 + panel_w, vy0 + panel_h], fill=(6, 14, 16), outline=(30, 58, 68))
        vs_r = min(panel_w, panel_h) // 2 - 8
        vcx = vx0 + panel_w // 2
        vcy = vy0 + panel_h // 2
        draw.ellipse([vcx - vs_r, vcy - vs_r, vcx + vs_r, vcy + vs_r], outline=(24, 68, 58), width=1)
        draw.line([(vcx - vs_r, vcy), (vcx + vs_r, vcy)], fill=(18, 46, 42), width=1)
        draw.line([(vcx, vcy - vs_r), (vcx, vcy + vs_r)], fill=(18, 46, 42), width=1)
        hist = vec.get("hist") or vec.get("density")
        if isinstance(hist, list) and len(hist) > 0:
            arr = np.asarray(hist, dtype=np.float32)
            if arr.ndim == 2:
                arr_norm = np.clip(arr / max(1e-6, float(arr.max())), 0.0, 1.0)
                vh, vw = arr_norm.shape
                v_rgb = np.zeros((vh, vw, 3), dtype=np.uint8)
                v_rgb[:, :, 1] = np.clip(arr_norm * 255.0 * 1.6, 0, 255).astype(np.uint8)
                v_rgb[:, :, 2] = np.clip(arr_norm * 180.0, 0, 255).astype(np.uint8)
                v_pil = Image.fromarray(v_rgb).resize((vs_r * 2, vs_r * 2), Image.Resampling.BILINEAR)
                canvas.paste(v_pil, (vcx - vs_r, vcy - vs_r))
                draw.ellipse([vcx - vs_r, vcy - vs_r, vcx + vs_r, vcy + vs_r], outline=(38, 110, 90), width=1)
        targets = vec.get("targets") or {}
        for _t_name, t_info in targets.items():
            if isinstance(t_info, dict):
                tx = vcx + int(float(t_info.get("u", 0.0)) * vs_r * 1.4)
                ty = vcy - int(float(t_info.get("v", 0.0)) * vs_r * 1.4)
                draw.rectangle([tx - 3, ty - 3, tx + 3, ty + 3], outline=(250, 204, 21))
        burst_deg = float(vec.get("burst_phase_deg", 180.0))
        acc_g = float(vec.get("acc_gain", 1.0))
        draw.text((vx0 + 4, 4), f"矢量示波器 (Burst {burst_deg:.1f}° | ACC {acc_g:.2f}×)", fill=(56, 189, 248))

        # 3. 0-6.5MHz Power Spectrum & A/B Metrics
        ab = sc.get("ab_compare") or {}
        sx0 = 18 + 2 * panel_w
        sy0 = 22
        draw.rectangle([sx0, sy0, sx0 + panel_w, sy0 + panel_h], fill=(10, 14, 24), outline=(35, 48, 72))
        spec = ab.get("spectrum") or {}
        freqs = spec.get("freq_mhz") or []
        sim_db = spec.get("sim_luma_db") or spec.get("luma_db") or []
        ref_db = spec.get("ref_luma_db") or []
        if len(freqs) > 1 and len(sim_db) == len(freqs):
            pts_sim, pts_ref = [], []
            for i, f_val in enumerate(freqs):
                px = sx0 + 4 + int((float(f_val) / 6.5) * (panel_w - 8))
                ny = np.clip((float(sim_db[i]) + 60.0) / 60.0, 0.0, 1.0)
                pts_sim.append((px, sy0 + panel_h - 4 - int(ny * (panel_h - 12))))
                if i < len(ref_db):
                    nr = np.clip((float(ref_db[i]) + 60.0) / 60.0, 0.0, 1.0)
                    pts_ref.append((px, sy0 + panel_h - 4 - int(nr * (panel_h - 12))))
            if len(pts_ref) >= 2:
                draw.line(pts_ref, fill=(100, 116, 139), width=1)
            if len(pts_sim) >= 2:
                draw.line(pts_sim, fill=(0, 255, 102), width=2)
        metrics = ab.get("metrics") or {}
        psnr = float(metrics.get("psnr_db", 0.0))
        dy_ire = float(metrics.get("mean_abs_dy_ire", 0.0))
        draw.text((sx0 + 4, 4), f"0–6.5MHz 功率谱 (PSNR {psnr:.1f}dB | ΔY {dy_ire:.1f}IRE)", fill=(244, 114, 182))

        return canvas, sc

    # --- Audio Synthesis & Oscilloscope Rendering ---
    def synthesize_audio_window(self, duration_s: float = 3.0) -> Tuple[bytes, Dict[str, Any]]:
        """Synthesize the 3s audio window and fetch synchronized waveform + A/V fault telemetry."""
        with self._lock:
            tl = self.app.audio_timeline(start_s=self.t, duration_s=duration_s)
            wav_bytes = self.app.audio_wav(
                start_s=self.t,
                duration_s=duration_s,
                mode="source" if self.audio_clean else "sim",
                sample_rate=44100,
                clean=self.audio_clean,
            )
            self.last_audio_tl = tl
            self._enforce_low_memory_caches()
            return wav_bytes, tl

    def export_full_audio_wav(self, out_path: str) -> str:
        """Render the full clip duration audio to a WAV file."""
        with self._lock:
            tel = self.get_telemetry()
            dur = max(1.0, float(tel.get("duration") or 4.0))
            wav_bytes = self.app.audio_wav(
                start_s=0.0,
                duration_s=dur,
                mode="source" if self.audio_clean else "sim",
                sample_rate=44100,
                clean=self.audio_clean,
            )
            Path(out_path).write_bytes(wav_bytes)
            self._enforce_low_memory_caches()
            return out_path

    def render_audio_scope_pil(self, width: int = 560, height: int = 44) -> Tuple[Image.Image, str]:
        """Render the dual-channel audio envelope & A/V sync fault bar to a PIL Image."""
        img = Image.new("RGB", (max(160, width), max(28, height)), (9, 13, 22))
        draw = ImageDraw.Draw(img)
        mid_y = img.height // 2
        draw.line([(0, mid_y), (img.width, mid_y)], fill=(30, 41, 59), width=1)

        tl = self.last_audio_tl
        if not tl:
            draw.text((10, mid_y - 7), "点击「▶ 试听伴音 (3s)」按需合成伴音与音画同步故障包络", fill=(100, 116, 139))
            return img, "伴音待命 (按需合成省内存)"

        ch = tl.get("waveform_channels") or {}
        peak_l = ch.get("peak_l") or []
        peak_r = ch.get("peak_r") or []
        n = len(peak_l)
        if n > 1:
            pts_l, pts_r = [], []
            for i in range(n):
                x = int((i / float(n - 1)) * (img.width - 1))
                pl = float(np.clip(peak_l[i], 0.0, 1.0))
                pr = float(np.clip(peak_r[i] if i < len(peak_r) else pl, 0.0, 1.0))
                pts_l.append((x, int(mid_y - pl * (mid_y - 3))))
                pts_r.append((x, int(mid_y + pr * (mid_y - 3))))
            draw.line(pts_l, fill=(0, 255, 102), width=1)
            draw.line(pts_r, fill=(56, 189, 248), width=1)

        frames = tl.get("fault_frames") or []
        nf = len(frames)
        if nf > 1:
            for i, ff in enumerate(frames):
                fx = int((i / float(nf - 1)) * (img.width - 1))
                drop = float(ff.get("dropout", 0.0))
                hsw = float(ff.get("head_switch", 0.0))
                wow = float(ff.get("wow_flutter", 0.0))
                if drop > 0.15:
                    draw.line([(fx, img.height - 8), (fx, img.height - 1)], fill=(239, 68, 68), width=2)
                elif hsw > 0.2:
                    draw.line([(fx, img.height - 6), (fx, img.height - 1)], fill=(34, 211, 238), width=1)
                elif wow > 0.2:
                    draw.line([(fx, img.height - 5), (fx, img.height - 1)], fill=(59, 130, 246), width=1)

        summ = tl.get("summary") or {}
        mode_lbl = str(tl.get("effective_mode") or "auto")
        drift_ms = float(summ.get("av_sync_offset_ms", summ.get("av_drift_ms", 0.0)) or 0.0)
        badge = f"模式: {mode_lbl} · 偏移: {drift_ms:+.1f}ms"
        return img, badge

    # --- Export Still, Quad 4-Corner Burst & ZIP ---
    def export_still_file(self, out_path: str) -> str:
        """Render and save a single still frame at `self.export_scale`."""
        with self._lock:
            scale = self.app.resolve_scale(self.export_scale, default=1.0)
            rgb_f32, _slug, _lbl = self.app.render_still_with_quad(
                t=self.t,
                scale=scale,
                want="sim",
                corner=self.quad_corner,
                border=self.quad_border,
                text=self.quad_text,
                use_graph=self.use_graph,
            )
            rgb_u8 = np.clip(rgb_f32 * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
            ext = Path(out_path).suffix.lower()
            img = Image.fromarray(rgb_u8, mode="RGB")
            if ext in (".jpg", ".jpeg"):
                q = 92 if self.export_quality in ("lossless", "high", "medium") else 82
                img.save(out_path, format="JPEG", quality=q)
            else:
                img.save(out_path, format="PNG")
            self.free_memory()
            return out_path

    def export_quad_burst_dir(self, out_dir: str, include_quad: bool = True) -> List[str]:
        """Batch export the 4 corners (and optional 2x2 overview) to `out_dir`."""
        os.makedirs(out_dir, exist_ok=True)
        with self._lock:
            scale = self.app.resolve_scale(self.export_scale, default=1.0)
            fmt = "jpg" if self.export_quality in ("medium", "draft") else "png"
            q = 92 if self.export_quality != "draft" else 82
            res = self.app.render_quad_burst(
                t=self.t,
                scale=scale,
                fmt=fmt,
                quality=q,
                border=self.quad_border if self.quad_border is not None else 0,
                text=self.quad_text if self.quad_text is not None else 0,
                include_quad=include_quad,
                use_graph=self.use_graph,
            )
            saved_paths: List[str] = []
            for shot in res.get("shots", []):
                p = Path(out_dir) / shot["filename"]
                p.write_bytes(shot["bytes"])
                saved_paths.append(str(p))
            self.free_memory()
            return saved_paths

    def export_quad_zip_file(self, zip_path: str) -> str:
        """Export 2x2 overview + 4 corners into a single ZIP archive."""
        with tempfile.TemporaryDirectory() as tmp_d:
            files = self.export_quad_burst_dir(tmp_d, include_quad=True)
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
                for f in files:
                    zf.write(f, arcname=Path(f).name)
        return zip_path
