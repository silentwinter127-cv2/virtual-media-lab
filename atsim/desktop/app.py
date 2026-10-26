"""Native Low-Overhead Desktop GUI for Virtual Media Lab (虚拟媒介实验室).

Provides a complete dark-Matrix native desktop interface matching the WebUI's
layout and capabilities while eliminating browser memory overhead, HTTP loopback
polling, and per-frame PNG/JPEG encoding latency.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent.parent):
    if (_cand / "atsim").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

try:
    from .controller import PRESET_CATEGORIES, SELECTORS, DesktopBackendController
except (ImportError, ValueError):
    try:
        from controller import PRESET_CATEGORIES, SELECTORS, DesktopBackendController
    except ImportError:
        from atsim.desktop.controller import PRESET_CATEGORIES, SELECTORS, DesktopBackendController


# Matrix Dark Palette matching WebUI app.css
BG_MAIN = "#0b0f18"
BG_PANEL = "#111827"
BG_CARD = "#172033"
BG_INPUT = "#0f172a"
FG_TEXT = "#e2e8f0"
FG_MUTED = "#94a3b8"
ACCENT_GREEN = "#00ff66"
ACCENT_CYAN = "#38bdf8"
ACCENT_AMBER = "#fbbf24"
ACCENT_RED = "#ef4444"
BORDER_COLOR = "#1e293b"


class NativeAudioPlayer:
    """Cross-platform zero-dependency WAV audio player for the Desktop GUI."""

    def __init__(self) -> None:
        self._temp_wav: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None

    def play_wav_bytes(self, wav_bytes: bytes) -> None:
        self.stop()
        fd, path = tempfile.mkstemp(prefix="vml_preview_", suffix=".wav")
        os.close(fd)
        Path(path).write_bytes(wav_bytes)
        self._temp_wav = path

        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        elif sys.platform == "darwin":
            self._proc = subprocess.Popen(
                ["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            for cmd in (
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                ["aplay", "-q", path],
                ["paplay", path],
            ):
                try:
                    self._proc = subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    break
                except FileNotFoundError:
                    continue

    def stop(self) -> None:
        if sys.platform == "win32":
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None
        if self._temp_wav and os.path.exists(self._temp_wav):
            try:
                os.unlink(self._temp_wav)
            except Exception:
                pass
            self._temp_wav = None


def build_node_graph_pil(
    ctrl: DesktopBackendController,
    width: int = 920,
    height: int = 560,
    selected_nodes: Optional[set] = None,
) -> Tuple[
    Image.Image,
    Dict[str, Tuple[int, int, int, int]],
    Dict[Tuple[str, str, str], Tuple[int, int]],
]:
    """Render the interactive Node Graph (`ctrl.app.graph`) to a PIL Image and return hit-test maps."""
    sel = selected_nodes or set()
    graph_obj = ctrl.app.graph

    img = Image.new("RGB", (max(400, width), max(300, height)), (9, 13, 22))
    draw = ImageDraw.Draw(img)

    for gx in range(0, img.width, 32):
        draw.line([(gx, 0), (gx, img.height)], fill=(15, 22, 36), width=1)
    for gy in range(0, img.height, 32):
        draw.line([(0, gy), (img.width, gy)], fill=(15, 22, 36), width=1)

    node_boxes: Dict[str, Tuple[int, int, int, int]] = {}
    port_centers: Dict[Tuple[str, str, str], Tuple[int, int]] = {}

    NW, NH = 152, 74
    for idx, (name, nd) in enumerate(graph_obj.nodes.items()):
        pos = nd.position or (idx * 180 + 30, 80)
        px, py = float(pos[0]), float(pos[1])
        if not getattr(nd, "px_position", True) and px < 25 and py < 25:
            px = 36 + px * 175
            py = 60 + py * 110
        x0 = int(np.clip(px, 12, max(24, img.width - NW - 16)))
        y0 = int(np.clip(py, 16, max(24, img.height - NH - 16)))
        ins = list(nd.spec.inputs or [])
        outs = list(nd.spec.outputs or [])
        box_h = max(NH, 34 + max(len(ins), len(outs)) * 18)
        node_boxes[name] = (x0, y0, x0 + NW, y0 + box_h)

        for i_idx, p_spec in enumerate(ins):
            p_name = p_spec.name if hasattr(p_spec, "name") else str(p_spec)
            port_centers[(name, p_name, "in")] = (x0, y0 + 34 + i_idx * 18)
        for o_idx, p_spec in enumerate(outs):
            p_name = p_spec.name if hasattr(p_spec, "name") else str(p_spec)
            port_centers[(name, p_name, "out")] = (x0 + NW, y0 + 34 + o_idx * 18)

    for ed in graph_obj.edges:
        p0 = port_centers.get((ed.src, ed.src_port, "out"))
        p1 = port_centers.get((ed.dst, ed.dst_port, "in"))
        if p0 and p1:
            mx = (p0[0] + p1[0]) // 2
            pts = [p0, (mx, p0[1]), (mx, p1[1]), p1]
            draw.line(pts, fill=(0, 255, 102), width=2)

    for name, nd in graph_obj.nodes.items():
        if name not in node_boxes:
            continue
        x0, y0, x1, y1 = node_boxes[name]
        enabled = bool(getattr(nd, "enabled", True))
        is_group = nd.kind == "node_group"
        is_sel = name in sel

        outline_col = (0, 255, 102) if is_sel else ((56, 189, 248) if is_group else (45, 62, 90))
        fill_col = (22, 34, 56) if enabled else (18, 22, 30)
        head_col = (16, 68, 52) if is_group else ((18, 48, 40) if enabled else (30, 35, 46))

        draw.rectangle([x0, y0, x1, y1], fill=fill_col, outline=outline_col, width=2 if is_sel else 1)
        draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y0 + 22], fill=head_col)

        lbl = str(nd.label or nd.spec.label or name)
        iters = int(getattr(nd, "iterations", 1) or 1)
        if iters > 1:
            lbl = f"{lbl} ×{iters}"
        draw.text((x0 + 6, y0 + 5), lbl[:18], fill=(230, 245, 255) if enabled else (120, 130, 145))

        for p_spec in (nd.spec.inputs or []):
            p_name = p_spec.name if hasattr(p_spec, "name") else str(p_spec)
            pc = port_centers.get((name, p_name, "in"))
            if pc:
                draw.ellipse([pc[0] - 4, pc[1] - 4, pc[0] + 4, pc[1] + 4], fill=(56, 189, 248), outline=(11, 15, 24))
                draw.text((pc[0] + 7, pc[1] - 6), p_name[:8], fill=(160, 185, 210))

        for p_spec in (nd.spec.outputs or []):
            p_name = p_spec.name if hasattr(p_spec, "name") else str(p_spec)
            pc = port_centers.get((name, p_name, "out"))
            if pc:
                draw.ellipse([pc[0] - 4, pc[1] - 4, pc[0] + 4, pc[1] + 4], fill=(0, 255, 102), outline=(11, 15, 24))
                draw.text((pc[0] - 46, pc[1] - 6), p_name[:7], fill=(160, 215, 185))

    return img, node_boxes, port_centers


def launch_desktop_gui(default_clip: Optional[str] = None, low_memory: bool = True) -> int:
    """Launch the native Tkinter/TTK Desktop GUI window."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, simpledialog, ttk
        from PIL import ImageTk
    except ImportError as exc:
        sys.stderr.write(
            f"[Virtual Media Lab Desktop] 无法加载图形界面库 tkinter ({exc})。\n"
            "• Windows / macOS 官方 Python 安装包默认自带 tkinter。\n"
            "• 若在精简版 Linux 下运行，请先执行: sudo apt-get install python3-tk\n"
            "• 若需在无头终端验证桌面端控制器，请运行: python desktop.py --self-test\n"
        )
        return 1

    ctrl = DesktopBackendController(default_clip=default_clip, low_memory=low_memory)
    audio_player = NativeAudioPlayer()

    root = tk.Tk()
    root.title("虚拟媒介实验室 · Virtual Media Lab Desktop (原生低内存直连工作站)")
    root.geometry("1520x920")
    root.minsize(1180, 740)
    root.configure(bg=BG_MAIN)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(".", background=BG_MAIN, foreground=FG_TEXT, fieldbackground=BG_INPUT)
    style.configure("TFrame", background=BG_MAIN)
    style.configure("Panel.TFrame", background=BG_PANEL)
    style.configure("Card.TFrame", background=BG_CARD)
    style.configure("TLabel", background=BG_MAIN, foreground=FG_TEXT)
    style.configure("Panel.TLabel", background=BG_PANEL, foreground=FG_TEXT)
    style.configure("Muted.TLabel", background=BG_PANEL, foreground=FG_MUTED)
    style.configure("Badge.TLabel", background=BG_CARD, foreground=ACCENT_GREEN, padding=(6, 2))
    style.configure(
        "TButton",
        background=BG_CARD,
        foreground=FG_TEXT,
        bordercolor=BORDER_COLOR,
        focuscolor=ACCENT_GREEN,
        padding=(8, 4),
    )
    style.map(
        "TButton",
        background=[("active", "#1e293b"), ("pressed", "#0f5132")],
        foreground=[("active", ACCENT_GREEN)],
    )
    style.configure(
        "Primary.TButton",
        background="#065f46",
        foreground="#ffffff",
        bordercolor=ACCENT_GREEN,
        padding=(9, 4),
    )
    style.map("Primary.TButton", background=[("active", "#047857")])
    style.configure("Tab.TButton", background=BG_INPUT, foreground=FG_MUTED, padding=(6, 3))
    style.configure("TabActive.TButton", background="#064e3b", foreground=ACCENT_GREEN, padding=(6, 3))
    style.configure("TCheckbutton", background=BG_PANEL, foreground=FG_TEXT)
    style.map("TCheckbutton", background=[("active", BG_PANEL)])
    style.configure(
        "TCombobox",
        fieldbackground=BG_INPUT,
        background=BG_CARD,
        foreground=FG_TEXT,
        arrowcolor=ACCENT_GREEN,
    )
    root.option_add("*TCombobox*Listbox.background", BG_INPUT)
    root.option_add("*TCombobox*Listbox.foreground", FG_TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", "#065f46")
    root.option_add("*TCombobox*Listbox.selectForeground", ACCENT_GREEN)

    photo_refs: Dict[str, Any] = {}
    debounce_state: Dict[str, Any] = {"timer": None, "playing": False, "play_timer": None}

    # 1. TOP HEADER BAR
    header = ttk.Frame(root, style="Panel.TFrame", padding=(10, 6))
    header.pack(side=tk.TOP, fill=tk.X)

    brand_box = ttk.Frame(header, style="Panel.TFrame")
    brand_box.pack(side=tk.LEFT)
    tk.Label(
        brand_box,
        text="● 虚拟媒介实验室",
        bg=BG_PANEL,
        fg=ACCENT_GREEN,
        font=("Microsoft YaHei UI", 12, "bold"),
    ).pack(side=tk.LEFT, padx=(0, 8))
    tk.Label(
        brand_box,
        text="Desktop 原生直连引擎 (零HTTP轮询·零编码损耗)",
        bg=BG_PANEL,
        fg=FG_MUTED,
        font=("Microsoft YaHei UI", 9),
    ).pack(side=tk.LEFT, padx=(0, 12))

    telemetry_lbl = tk.Label(
        header,
        text="RAM: -- MB | 帧耗时: -- ms",
        bg=BG_CARD,
        fg=ACCENT_CYAN,
        font=("Consolas", 9, "bold"),
        padx=8,
        pady=3,
    )
    telemetry_lbl.pack(side=tk.RIGHT, padx=(6, 0))

    actions_bar = ttk.Frame(header, style="Panel.TFrame")
    actions_bar.pack(side=tk.RIGHT)

    # 2. SIGNAL CHAIN BAR
    chain_bar = tk.Frame(root, bg="#0d1320", height=30, pady=3, padx=8)
    chain_bar.pack(side=tk.TOP, fill=tk.X)

    # 3. MAIN 3-COLUMN WORKSPACE
    main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg=BORDER_COLOR, sashwidth=4, bd=0)
    main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    panel_left = ttk.Frame(main_pane, style="Panel.TFrame", width=310, padding=6)
    panel_center = ttk.Frame(main_pane, style="TFrame", padding=6)
    panel_right = ttk.Frame(main_pane, style="Panel.TFrame", width=400, padding=6)

    main_pane.add(panel_left, minsize=260, width=310)
    main_pane.add(panel_center, minsize=520)
    main_pane.add(panel_right, minsize=340, width=410)

    status_var = tk.StringVar(value="就绪 · 已启用原生低内存模式 (帧缓存上限=2，示波器与伴音按需计算)")
    status_bar = tk.Label(
        root,
        textvariable=status_var,
        bg=BG_PANEL,
        fg=ACCENT_GREEN,
        anchor="w",
        font=("Microsoft YaHei UI", 9),
        padx=10,
        pady=3,
    )
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def toast(msg: str) -> None:
        status_var.set(msg)

    # CENTER STAGE WIDGETS
    viewer_frame = tk.Frame(panel_center, bg="#05080f", bd=1, relief=tk.SOLID)
    viewer_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 4))

    viewport_canvas = tk.Canvas(viewer_frame, bg="#05080f", highlightthickness=0, cursor="crosshair")
    viewport_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    badge_var = tk.StringVar(value="内置测试图 (SMPTE)")
    badge_lbl = tk.Label(
        viewer_frame,
        textvariable=badge_var,
        bg="#0b0f18",
        fg=ACCENT_GREEN,
        font=("Consolas", 9),
        padx=6,
        pady=2,
    )
    badge_lbl.place(x=8, y=8)

    scrub_bar = ttk.Frame(panel_center, style="Panel.TFrame", padding=(6, 4))
    scrub_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 3))

    btn_play = ttk.Button(scrub_bar, text="▶ 播放", width=7)
    btn_play.pack(side=tk.LEFT, padx=(0, 6))

    time_var = tk.DoubleVar(value=ctrl.t)
    time_slider = tk.Scale(
        scrub_bar,
        variable=time_var,
        from_=0.0,
        to=10.0,
        resolution=0.05,
        orient=tk.HORIZONTAL,
        showvalue=False,
        bg=BG_PANEL,
        fg=FG_TEXT,
        highlightthickness=0,
        troughcolor=BG_INPUT,
        activebackground=ACCENT_GREEN,
        length=220,
    )
    time_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

    time_lbl = tk.Label(
        scrub_bar,
        text="t = 0.40 s",
        bg=BG_PANEL,
        fg=ACCENT_CYAN,
        font=("Consolas", 10, "bold"),
        width=11,
    )
    time_lbl.pack(side=tk.LEFT, padx=4)

    preview_scale_map = [
        ("0.25", "预览: 0.25x 极速 (超省内存)"),
        ("0.35", "预览: 0.35x 流畅 (默认省内存)"),
        ("0.5", "预览: 0.50x 标准"),
        ("sd720", "预览: 720p 标清"),
        ("0.75", "预览: 0.75x 高清"),
        ("1.0", "预览: 1.00x 原画"),
    ]
    preview_scale_cb = ttk.Combobox(
        scrub_bar, values=[lbl for _, lbl in preview_scale_map], state="readonly", width=22
    )
    preview_scale_cb.current(1 if ctrl.low_memory else 2)
    preview_scale_cb.pack(side=tk.LEFT, padx=4)

    ab_var = tk.BooleanVar(value=ctrl.ab_enabled)
    ab_chk = ttk.Checkbutton(scrub_bar, text="A/B 对比", variable=ab_var)
    ab_chk.pack(side=tk.LEFT, padx=4)

    auto_var = tk.BooleanVar(value=ctrl.auto_refresh)
    auto_chk = ttk.Checkbutton(scrub_bar, text="自动刷新", variable=auto_var)
    auto_chk.pack(side=tk.LEFT, padx=4)

    graph_mode_var = tk.BooleanVar(value=ctrl.use_graph)
    graph_chk = ttk.Checkbutton(scrub_bar, text="用节点图渲染", variable=graph_mode_var)
    graph_chk.pack(side=tk.LEFT, padx=4)

    btn_free_mem = ttk.Button(scrub_bar, text="🧹 释放内存")
    btn_free_mem.pack(side=tk.LEFT, padx=(6, 0))

    # Quad Bar
    quad_bar = ttk.Frame(panel_center, style="Panel.TFrame", padding=(6, 3))
    quad_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 3))

    tk.Label(
        quad_bar,
        text="❖ 分屏/分框/文字",
        bg=BG_PANEL,
        fg=ACCENT_GREEN,
        font=("Microsoft YaHei UI", 9, "bold"),
    ).pack(side=tk.LEFT, padx=(0, 6))

    quad_corner_btns: List[ttk.Button] = []
    for _c_idx, c_text in [
        (0, "⊞ 2×2四分屏"),
        (1, "↖ 左上1"),
        (2, "↗ 右上2"),
        (3, "↙ 左下3"),
        (4, "↘ 右下4"),
    ]:
        b = ttk.Button(quad_bar, text=c_text, style="Tab.TButton")
        b.pack(side=tk.LEFT, padx=1)
        quad_corner_btns.append(b)

    quad_border_cb = ttk.Combobox(
        quad_bar,
        values=["分框: 无分框 (无缝)", "分框: 细十字分框线", "分框: 标准演播室分框"],
        state="readonly",
        width=17,
    )
    quad_border_cb.current(2)
    quad_border_cb.pack(side=tk.LEFT, padx=4)

    quad_text_cb = ttk.Combobox(
        quad_bar,
        values=["文字: 不要文字 (纯净)", "文字: 仅角落标签", "文字: 标签+底部遥测栏"],
        state="readonly",
        width=18,
    )
    quad_text_cb.current(0)
    quad_text_cb.pack(side=tk.LEFT, padx=4)

    btn_quad_still = ttk.Button(quad_bar, text="⬇ 导出当前帧")
    btn_quad_still.pack(side=tk.LEFT, padx=2)
    btn_quad_burst = ttk.Button(quad_bar, text="📸 批量四连拍", style="Primary.TButton")
    btn_quad_burst.pack(side=tk.LEFT, padx=2)
    btn_quad_zip = ttk.Button(quad_bar, text="📦 四连拍ZIP")
    btn_quad_zip.pack(side=tk.LEFT, padx=2)

    # W-T Optical Zoom Bar
    wt_bar = ttk.Frame(panel_center, style="Panel.TFrame", padding=(6, 3))
    wt_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 3))

    tk.Label(
        wt_bar,
        text="W–T 光学变焦",
        bg=BG_PANEL,
        fg=ACCENT_CYAN,
        font=("Microsoft YaHei UI", 9, "bold"),
    ).pack(side=tk.LEFT, padx=(0, 6))

    btn_zoom_w = ttk.Button(wt_bar, text="◀ W 广角", width=8)
    btn_zoom_w.pack(side=tk.LEFT, padx=2)

    wt_zoom_var = tk.DoubleVar(value=0.0)
    wt_slider = tk.Scale(
        wt_bar,
        variable=wt_zoom_var,
        from_=0.0,
        to=1.0,
        resolution=0.01,
        orient=tk.HORIZONTAL,
        showvalue=False,
        bg=BG_PANEL,
        fg=FG_TEXT,
        highlightthickness=0,
        troughcolor=BG_INPUT,
        activebackground=ACCENT_CYAN,
        length=160,
    )
    wt_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

    btn_zoom_t = ttk.Button(wt_bar, text="T 放大 ▶", width=8)
    btn_zoom_t.pack(side=tk.LEFT, padx=2)

    wt_readout_lbl = tk.Label(
        wt_bar,
        text="1.00× · f/1.4 · W 端光晕态",
        bg=BG_CARD,
        fg=ACCENT_GREEN,
        font=("Consolas", 9),
        padx=6,
        pady=2,
    )
    wt_readout_lbl.pack(side=tk.LEFT, padx=4)

    wt_dyn_cb = ttk.Combobox(
        wt_bar,
        values=["手动固定 W–T", "W→T 电动推进放大", "T→W 电动拉远缩小", "W↔T 往复变焦"],
        state="readonly",
        width=16,
    )
    wt_dyn_cb.current(0)
    wt_dyn_cb.pack(side=tk.LEFT, padx=4)

    wt_osd_var = tk.BooleanVar(value=False)
    wt_osd_chk = ttk.Checkbutton(wt_bar, text="W–T 指示条", variable=wt_osd_var)
    wt_osd_chk.pack(side=tk.LEFT, padx=4)

    # Audio Monitor Bar
    audio_bar = ttk.Frame(panel_center, style="Panel.TFrame", padding=(6, 4))
    audio_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 3))

    btn_audio_play = ttk.Button(audio_bar, text="▶ 试听伴音 (3s)", style="Primary.TButton")
    btn_audio_play.pack(side=tk.LEFT, padx=2)
    btn_audio_stop = ttk.Button(audio_bar, text="⏹", width=3)
    btn_audio_stop.pack(side=tk.LEFT, padx=2)
    btn_audio_win_wav = ttk.Button(audio_bar, text="⬇ 试听段WAV")
    btn_audio_win_wav.pack(side=tk.LEFT, padx=2)
    btn_audio_full_wav = ttk.Button(audio_bar, text="⬇ 全片WAV")
    btn_audio_full_wav.pack(side=tk.LEFT, padx=2)

    audio_clean_var = tk.BooleanVar(value=False)
    audio_clean_chk = ttk.Checkbutton(audio_bar, text="原始伴音对比", variable=audio_clean_var)
    audio_clean_chk.pack(side=tk.LEFT, padx=4)

    audio_scope_canvas = tk.Canvas(
        audio_bar, width=320, height=38, bg="#090d16", highlightthickness=1, highlightbackground=BORDER_COLOR
    )
    audio_scope_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

    audio_badge_lbl = tk.Label(
        audio_bar,
        text="伴音待命",
        bg=BG_CARD,
        fg=ACCENT_GREEN,
        font=("Consolas", 9),
        padx=6,
        pady=2,
    )
    audio_badge_lbl.pack(side=tk.RIGHT, padx=2)

    # Collapsible Broadcast Scopes Drawer
    scopes_drawer = ttk.Frame(panel_center, style="Panel.TFrame", padding=6)
    scopes_head = ttk.Frame(scopes_drawer, style="Panel.TFrame")
    scopes_head.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
    tk.Label(
        scopes_head,
        text="📊 广播级实时示波与校准台 (Tektronix 1740A IRE / Vectorscope / 0–6.5MHz PSD — 收起时零CPU占用)",
        bg=BG_PANEL,
        fg=ACCENT_GREEN,
        font=("Microsoft YaHei UI", 9, "bold"),
    ).pack(side=tk.LEFT)

    scope_line_var = tk.DoubleVar(value=0.5)
    tk.Label(scopes_head, text="扫描行 Y:", bg=BG_PANEL, fg=FG_MUTED).pack(side=tk.LEFT, padx=(12, 2))
    scope_line_slider = tk.Scale(
        scopes_head,
        variable=scope_line_var,
        from_=0.0,
        to=1.0,
        resolution=0.01,
        orient=tk.HORIZONTAL,
        showvalue=False,
        bg=BG_PANEL,
        highlightthickness=0,
        troughcolor=BG_INPUT,
        length=100,
    )
    scope_line_slider.pack(side=tk.LEFT, padx=4)

    scopes_canvas = tk.Canvas(
        scopes_drawer, height=175, bg="#080c14", highlightthickness=1, highlightbackground=BORDER_COLOR
    )
    scopes_canvas.pack(side=tk.TOP, fill=tk.X, expand=True)

    def update_telemetry_ui() -> None:
        tel = ctrl.get_telemetry()
        telemetry_lbl.configure(
            text=f"RAM: {tel['rss_mb']:.0f} MB | 帧耗时: {tel['render_ms']:.0f} ms | {tel['preview_w']}×{tel['preview_h']}"
        )
        badge_var.set(
            f"{tel['source_name']} ({tel['source_w']}×{tel['source_h']} @ {tel['fps']}fps) · {ctrl.last_frame_label}"
        )
        if float(time_slider.cget("to")) != max(1.0, tel["duration"]):
            time_slider.configure(to=max(1.0, tel["duration"]))

    def refresh_audio_scope_ui() -> None:
        w = max(240, audio_scope_canvas.winfo_width())
        h = max(34, audio_scope_canvas.winfo_height())
        pil_img, badge_txt = ctrl.render_audio_scope_pil(width=w, height=h)
        tk_img = ImageTk.PhotoImage(pil_img)
        photo_refs["audio_scope"] = tk_img
        audio_scope_canvas.delete("all")
        audio_scope_canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        audio_badge_lbl.configure(text=badge_txt)

    def refresh_scopes_ui() -> None:
        if not ctrl.scopes_open:
            return
        w = max(480, scopes_canvas.winfo_width())
        h = max(160, scopes_canvas.winfo_height())
        pil_img, _ = ctrl.render_scopes_pil(width=w, height=h)
        tk_img = ImageTk.PhotoImage(pil_img)
        photo_refs["scopes"] = tk_img
        scopes_canvas.delete("all")
        scopes_canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

    def refresh_viewport(force: bool = False) -> None:
        if not force and not ctrl.auto_refresh:
            return
        cw = max(320, viewport_canvas.winfo_width())
        ch = max(240, viewport_canvas.winfo_height())
        try:
            pil_img = ctrl.render_viewport_pil(canvas_w=cw, canvas_h=ch)
            tk_img = ImageTk.PhotoImage(pil_img)
            photo_refs["viewport"] = tk_img
            viewport_canvas.delete("all")
            ox = max(0, (cw - pil_img.width) // 2)
            oy = max(0, (ch - pil_img.height) // 2)
            viewport_canvas.create_image(ox, oy, anchor=tk.NW, image=tk_img)
            viewport_canvas._img_bounds = (ox, oy, pil_img.width, pil_img.height)  # type: ignore[attr-defined]
            update_telemetry_ui()
            if ctrl.scopes_open:
                refresh_scopes_ui()
        except Exception as exc:
            toast(f"渲染异常: {exc}")

    def schedule_refresh(delay_ms: int = 90) -> None:
        if debounce_state["timer"] is not None:
            try:
                root.after_cancel(debounce_state["timer"])
            except Exception:
                pass
        debounce_state["timer"] = root.after(delay_ms, lambda: refresh_viewport(force=False))

    def on_viewport_mouse(event: Any) -> None:
        bounds = getattr(viewport_canvas, "_img_bounds", None)
        if not bounds:
            return
        ox, oy, iw, ih = bounds
        if iw <= 0 or ih <= 0:
            return
        if event.state & 0x20000 or event.state & 0x0008:
            ry = float(np.clip((event.y - oy) / float(ih), 0.0, 1.0))
            ctrl.scope_line_y = ry
            scope_line_var.set(ry)
            if ctrl.scopes_open:
                refresh_scopes_ui()
            toast(f"已选择示波器单行扫描线 Y = {ry:.3f}")
            return
        if ctrl.ab_enabled:
            rx = float(np.clip((event.x - ox) / float(iw), 0.02, 0.98))
            ctrl.split_x = rx
            schedule_refresh(30)

    viewport_canvas.bind("<Button-1>", on_viewport_mouse)
    viewport_canvas.bind("<B1-Motion>", on_viewport_mouse)
    viewport_canvas.bind("<Configure>", lambda _e: schedule_refresh(120))

    # LEFT DOCK: PRESETS
    preset_tabs_frame = tk.Frame(panel_left, bg=BG_PANEL)
    preset_tabs_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

    search_preset_var = tk.StringVar(value="")
    search_row = ttk.Frame(panel_left, style="Panel.TFrame")
    search_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
    tk.Label(search_row, text="🔍", bg=BG_PANEL, fg=FG_MUTED).pack(side=tk.LEFT)
    search_preset_entry = tk.Entry(
        search_row,
        textvariable=search_preset_var,
        bg=BG_INPUT,
        fg=FG_TEXT,
        insertbackground=ACCENT_GREEN,
        relief=tk.FLAT,
    )
    search_preset_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, ipady=3)

    save_cp_row = ttk.Frame(panel_left, style="Panel.TFrame")
    save_cp_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
    cp_name_var = tk.StringVar(value="")
    cp_entry = tk.Entry(
        save_cp_row,
        textvariable=cp_name_var,
        bg=BG_INPUT,
        fg=FG_TEXT,
        insertbackground=ACCENT_GREEN,
        relief=tk.FLAT,
    )
    cp_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4), ipady=3)
    btn_save_cp = ttk.Button(save_cp_row, text="+ 保存预设", style="Primary.TButton")
    btn_save_cp.pack(side=tk.RIGHT)

    preset_list_frame = ttk.Frame(panel_left, style="Panel.TFrame")
    preset_list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    preset_listbox = tk.Listbox(
        preset_list_frame,
        bg=BG_INPUT,
        fg=FG_TEXT,
        selectbackground="#065f46",
        selectforeground=ACCENT_GREEN,
        activestyle="none",
        highlightthickness=1,
        highlightbackground=BORDER_COLOR,
        font=("Microsoft YaHei UI", 9),
    )
    preset_scroll = ttk.Scrollbar(preset_list_frame, orient=tk.VERTICAL, command=preset_listbox.yview)
    preset_listbox.configure(yscrollcommand=preset_scroll.set)
    preset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    preset_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    preset_note_lbl = tk.Label(
        panel_left,
        text="点击上方预设立即切换整套模拟链路或单设备模块",
        bg=BG_CARD,
        fg=FG_MUTED,
        wraplength=280,
        justify=tk.LEFT,
        anchor="w",
        padx=6,
        pady=4,
        font=("Microsoft YaHei UI", 8),
    )
    preset_note_lbl.pack(side=tk.TOP, fill=tk.X, pady=(4, 4))

    btn_del_cp = ttk.Button(panel_left, text="🗑 删除所选自定义预设")

    graph_quick_frame = ttk.Frame(panel_left, style="Panel.TFrame")
    graph_quick_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))
    tk.Label(
        graph_quick_frame,
        text="整套接线图预设 (Node Graph):",
        bg=BG_PANEL,
        fg=ACCENT_CYAN,
        font=("Microsoft YaHei UI", 8, "bold"),
    ).pack(side=tk.TOP, anchor="w")
    graph_presets_info = ctrl.app.graph_state().get("presets") or {}
    graph_preset_keys = list(graph_presets_info.keys())
    graph_preset_cb = ttk.Combobox(
        graph_quick_frame,
        values=[f"{k} · {graph_presets_info[k]['label']}" for k in graph_preset_keys],
        state="readonly",
    )
    if graph_preset_keys:
        graph_preset_cb.current(0)
    graph_preset_cb.pack(side=tk.TOP, fill=tk.X, pady=2)

    current_preset_entries: List[Tuple[str, str, str]] = []

    def render_preset_tabs() -> None:
        for w in preset_tabs_frame.winfo_children():
            w.destroy()
        for idx, (tab_k, tab_lbl) in enumerate(PRESET_CATEGORIES):
            r_idx = idx // 2
            c_idx = idx % 2
            st_name = "TabActive.TButton" if ctrl.preset_tab == tab_k else "Tab.TButton"
            b = ttk.Button(
                preset_tabs_frame,
                text=tab_lbl,
                style=st_name,
                command=lambda k=tab_k: switch_preset_tab(k),
            )
            b.grid(row=r_idx, column=c_idx, sticky="ew", padx=1, pady=1)
        preset_tabs_frame.columnconfigure(0, weight=1)
        preset_tabs_frame.columnconfigure(1, weight=1)

    def render_preset_list() -> None:
        nonlocal current_preset_entries
        current_preset_entries = ctrl.get_preset_entries(ctrl.preset_tab, search_preset_var.get())
        preset_listbox.delete(0, tk.END)
        active_p = ctrl.app.current.preset
        sel_idx = None
        for i, (k, lbl, _note) in enumerate(current_preset_entries):
            prefix = "★ " if k == active_p else "  "
            preset_listbox.insert(tk.END, f"{prefix}{lbl}")
            if k == active_p:
                sel_idx = i
        if sel_idx is not None:
            preset_listbox.selection_set(sel_idx)
            preset_listbox.see(sel_idx)
        if ctrl.preset_tab == "custom":
            btn_del_cp.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        else:
            btn_del_cp.pack_forget()

    def switch_preset_tab(tab_key: str) -> None:
        ctrl.preset_tab = tab_key
        render_preset_tabs()
        render_preset_list()

    def on_preset_select(_event: Any = None) -> None:
        sel = preset_listbox.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(current_preset_entries):
            key, lbl, note = current_preset_entries[idx]
            preset_note_lbl.configure(text=f"【{lbl}】\n{note}")
            ctrl.apply_preset(key)
            sync_all_controls_from_project()
            refresh_viewport(force=True)
            toast(f"已加载预设：{lbl} ({key})")

    preset_listbox.bind("<<ListboxSelect>>", on_preset_select)
    search_preset_var.trace_add("write", lambda *_: render_preset_list())

    def on_save_custom_preset() -> None:
        name = cp_name_var.get().strip()
        if not name:
            name = simpledialog.askstring("保存自定义预设", "请输入自定义预设名称：", parent=root) or ""
        if not name.strip():
            return
        ctrl.save_custom_preset(label=name.strip(), note="桌面端保存预设")
        cp_name_var.set("")
        switch_preset_tab("custom")
        toast(f"已保存自定义预设「{name.strip()}」")

    def on_delete_custom_preset() -> None:
        sel = preset_listbox.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(current_preset_entries):
            key, lbl, _ = current_preset_entries[idx]
            ctrl.delete_custom_preset(key)
            render_preset_list()
            toast(f"已删除自定义预设「{lbl}」")

    btn_save_cp.configure(command=on_save_custom_preset)
    btn_del_cp.configure(command=on_delete_custom_preset)

    def on_graph_preset_selected(_event: Any = None) -> None:
        idx = graph_preset_cb.current()
        if 0 <= idx < len(graph_preset_keys):
            g_key = graph_preset_keys[idx]
            ctrl.app.graph_load_preset(g_key)
            ctrl._enforce_low_memory_caches()
            sync_all_controls_from_project()
            refresh_viewport(force=True)
            toast(f"已加载整套接线图预设：{g_key}")

    graph_preset_cb.bind("<<ComboboxSelected>>", on_graph_preset_selected)

    # RIGHT DOCK: LAYER TABS & SCROLLABLE PARAMETERS
    layer_tabs_frame = tk.Frame(panel_right, bg=BG_PANEL)
    layer_tabs_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

    param_search_var = tk.StringVar(value="")
    p_search_row = ttk.Frame(panel_right, style="Panel.TFrame")
    p_search_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
    tk.Label(p_search_row, text="🔍 筛选参数:", bg=BG_PANEL, fg=FG_MUTED).pack(side=tk.LEFT)
    p_search_entry = tk.Entry(
        p_search_row,
        textvariable=param_search_var,
        bg=BG_INPUT,
        fg=FG_TEXT,
        insertbackground=ACCENT_GREEN,
        relief=tk.FLAT,
    )
    p_search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, ipady=3)

    param_outer = ttk.Frame(panel_right, style="Panel.TFrame")
    param_outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    param_canvas = tk.Canvas(param_outer, bg=BG_PANEL, highlightthickness=0)
    param_scrollbar = ttk.Scrollbar(param_outer, orient=tk.VERTICAL, command=param_canvas.yview)
    param_inner = ttk.Frame(param_canvas, style="Panel.TFrame")

    param_window_id = param_canvas.create_window((0, 0), window=param_inner, anchor="nw")
    param_canvas.configure(yscrollcommand=param_scrollbar.set)
    param_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    param_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    param_inner.bind(
        "<Configure>",
        lambda _e: param_canvas.configure(scrollregion=param_canvas.bbox("all")),
    )
    param_canvas.bind(
        "<Configure>",
        lambda e: param_canvas.itemconfigure(param_window_id, width=e.width),
    )

    def render_chain_and_layer_tabs() -> None:
        for w in layer_tabs_frame.winfo_children():
            w.destroy()
        for w in chain_bar.winfo_children():
            w.destroy()

        layers_list = ctrl.schema_dict.get("layers") or []
        layer_labels = ctrl.catalog.get("layers") or {}

        tk.Label(
            chain_bar,
            text="信号链阶段:",
            bg="#0d1320",
            fg=FG_MUTED,
            font=("Microsoft YaHei UI", 8, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 6))

        for idx, l_obj in enumerate(layers_list):
            l_key = l_obj["layer"]
            lbl = str(layer_labels.get(l_key) or l_key)
            is_active = l_key == ctrl.active_layer
            st_name = "TabActive.TButton" if is_active else "Tab.TButton"

            b = ttk.Button(
                layer_tabs_frame,
                text=lbl,
                style=st_name,
                command=lambda k=l_key: switch_layer_tab(k),
            )
            b.grid(row=idx // 2, column=idx % 2, sticky="ew", padx=1, pady=1)

            cb = tk.Button(
                chain_bar,
                text=lbl,
                bg="#064e3b" if is_active else "#162032",
                fg=ACCENT_GREEN if is_active else FG_TEXT,
                activebackground="#047857",
                activeforeground="#ffffff",
                relief=tk.FLAT,
                font=("Microsoft YaHei UI", 8),
                padx=6,
                pady=1,
                command=lambda k=l_key: switch_layer_tab(k),
            )
            cb.pack(side=tk.LEFT, padx=2)
            if idx < len(layers_list) - 1:
                tk.Label(chain_bar, text="→", bg="#0d1320", fg=FG_MUTED).pack(side=tk.LEFT)

        layer_tabs_frame.columnconfigure(0, weight=1)
        layer_tabs_frame.columnconfigure(1, weight=1)

    def render_params_panel() -> None:
        for w in param_inner.winfo_children():
            w.destroy()

        layers_list = ctrl.schema_dict.get("layers") or []
        layer_obj = next((l for l in layers_list if l["layer"] == ctrl.active_layer), None)
        if not layer_obj:
            return

        resolved = ctrl.resolved_params().get(ctrl.active_layer) or {}
        q = param_search_var.get().strip().lower()

        for g_name, specs in (layer_obj.get("groups") or {}).items():
            filtered_specs = [
                sp
                for sp in specs
                if not q
                or q in sp["key"].lower()
                or q in str(sp.get("label", "")).lower()
                or q in g_name.lower()
            ]
            if not filtered_specs:
                continue

            g_frame = tk.Frame(param_inner, bg=BG_CARD, bd=1, relief=tk.SOLID, padx=6, pady=5)
            g_frame.pack(side=tk.TOP, fill=tk.X, pady=4, padx=2)

            tk.Label(
                g_frame,
                text=f"■ {g_name}",
                bg=BG_CARD,
                fg=ACCENT_CYAN,
                font=("Microsoft YaHei UI", 9, "bold"),
                anchor="w",
            ).pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

            for sp in filtered_specs:
                p_key = sp["key"]
                p_lbl = sp.get("label", p_key)
                unit = sp.get("unit") or ""
                kind = sp.get("kind", "number")
                cur_val = resolved.get(p_key, sp.get("default", 0.0))

                row = tk.Frame(g_frame, bg=BG_CARD, pady=2)
                row.pack(side=tk.TOP, fill=tk.X)

                head = tk.Frame(row, bg=BG_CARD)
                head.pack(side=tk.TOP, fill=tk.X)

                unit_str = f" ({unit})" if unit and unit not in ("0-1", "bool", "text", "enum") else ""
                tk.Label(
                    head,
                    text=f"{p_lbl}{unit_str}",
                    bg=BG_CARD,
                    fg=FG_TEXT,
                    font=("Microsoft YaHei UI", 8),
                    anchor="w",
                ).pack(side=tk.LEFT)

                val_lbl = tk.Label(
                    head,
                    text=str(cur_val) if kind == "text" else f"{float(cur_val):.3g}",
                    bg=BG_CARD,
                    fg=ACCENT_GREEN,
                    font=("Consolas", 8, "bold"),
                )
                val_lbl.pack(side=tk.RIGHT)

                sel_opts = ctrl.selector_options_for(ctrl.active_layer, sp)
                if kind == "text":
                    t_var = tk.StringVar(value=str(cur_val or ""))
                    ent = tk.Entry(
                        row,
                        textvariable=t_var,
                        bg=BG_INPUT,
                        fg=FG_TEXT,
                        insertbackground=ACCENT_GREEN,
                        relief=tk.FLAT,
                    )
                    ent.pack(side=tk.TOP, fill=tk.X, pady=2, ipady=2)

                    def _commit_text(
                        _e: Any = None,
                        lk: str = ctrl.active_layer,
                        pk: str = p_key,
                        tv: tk.StringVar = t_var,
                        vl: tk.Label = val_lbl,
                    ) -> None:
                        v = tv.get()
                        vl.configure(text=v[:18])
                        ctrl.set_param(lk, pk, v)
                        schedule_refresh(80)

                    ent.bind("<Return>", _commit_text)
                    ent.bind("<FocusOut>", _commit_text)

                elif sel_opts is not None:
                    opt_labels = [lbl_o for _, lbl_o in sel_opts]
                    opt_vals = [v_o for v_o, _ in sel_opts]
                    cb = ttk.Combobox(row, values=opt_labels, state="readonly")
                    try:
                        c_int = int(round(float(cur_val)))
                        idx_match = opt_vals.index(c_int) if c_int in opt_vals else 0
                    except Exception:
                        idx_match = 0
                    if opt_labels:
                        cb.current(idx_match)
                    cb.pack(side=tk.TOP, fill=tk.X, pady=2)

                    def _on_sel(
                        _e: Any,
                        lk: str = ctrl.active_layer,
                        pk: str = p_key,
                        w_cb: ttk.Combobox = cb,
                        ovs: List[int] = opt_vals,
                        vl: tk.Label = val_lbl,
                    ) -> None:
                        ix = w_cb.current()
                        if 0 <= ix < len(ovs):
                            v_num = float(ovs[ix])
                            vl.configure(text=f"{int(v_num)}")
                            ctrl.set_param(lk, pk, v_num)
                            schedule_refresh(40)

                    cb.bind("<<ComboboxSelected>>", _on_sel)

                elif unit == "bool" or (
                    sp.get("step") == 1 and sp.get("min") == 0 and sp.get("max") == 1
                ):
                    b_var = tk.BooleanVar(value=float(cur_val) > 0.5)
                    val_lbl.configure(text="开" if b_var.get() else "关")

                    def _on_bool(
                        lk: str = ctrl.active_layer,
                        pk: str = p_key,
                        bv: tk.BooleanVar = b_var,
                        vl: tk.Label = val_lbl,
                    ) -> None:
                        v_num = 1.0 if bv.get() else 0.0
                        vl.configure(text="开" if bv.get() else "关")
                        ctrl.set_param(lk, pk, v_num)
                        schedule_refresh(40)

                    chk = ttk.Checkbutton(row, text="启用 / 开关", variable=b_var, command=_on_bool)
                    chk.pack(side=tk.TOP, anchor="w")

                else:
                    p_min = float(sp.get("min", 0.0))
                    p_max = float(sp.get("max", 1.0))
                    p_step = float(sp.get("step") or ((p_max - p_min) / 200.0))
                    s_var = tk.DoubleVar(value=float(cur_val))
                    sc_w = tk.Scale(
                        row,
                        variable=s_var,
                        from_=p_min,
                        to=p_max,
                        resolution=p_step,
                        orient=tk.HORIZONTAL,
                        showvalue=False,
                        bg=BG_CARD,
                        fg=FG_TEXT,
                        highlightthickness=0,
                        troughcolor=BG_INPUT,
                        activebackground=ACCENT_GREEN,
                    )
                    sc_w.pack(side=tk.TOP, fill=tk.X)

                    def _on_slide(
                        val_s: str,
                        lk: str = ctrl.active_layer,
                        pk: str = p_key,
                        vl: tk.Label = val_lbl,
                    ) -> None:
                        vf = float(val_s)
                        vl.configure(text=f"{vf:.3g}")
                        ctrl.set_param(lk, pk, vf)
                        if lk == "camera" and pk == "wt_zoom":
                            wt_zoom_var.set(vf)
                            wt_readout_lbl.configure(text=ctrl.get_wt_readout(vf))
                        schedule_refresh(90)

                    sc_w.configure(command=_on_slide)

    def switch_layer_tab(layer_key: str) -> None:
        ctrl.active_layer = layer_key
        render_chain_and_layer_tabs()
        render_params_panel()

    param_search_var.trace_add("write", lambda *_: render_params_panel())

    def sync_all_controls_from_project() -> None:
        res_p = ctrl.resolved_params()
        cam = res_p.get("camera") or {}
        wt_z = float(cam.get("wt_zoom", 0.0))
        wt_zoom_var.set(wt_z)
        wt_readout_lbl.configure(text=ctrl.get_wt_readout(wt_z))
        dyn_m = int(round(float(cam.get("zoom_dynamic_mode", 0.0))))
        if 0 <= dyn_m < 4:
            wt_dyn_cb.current(dyn_m)
        wt_osd_var.set(float(cam.get("zoom_osd", 0.0)) > 0.5)
        render_chain_and_layer_tabs()
        render_params_panel()
        render_preset_list()

    # WIRE CENTER STAGE CONTROLS
    def on_time_changed(_val: Any = None) -> None:
        ctrl.t = float(time_var.get())
        time_lbl.configure(text=f"t = {ctrl.t:.2f} s")
        schedule_refresh(60)

    time_slider.configure(command=on_time_changed)

    def toggle_playback() -> None:
        debounce_state["playing"] = not debounce_state["playing"]
        if debounce_state["playing"]:
            btn_play.configure(text="⏸ 暂停")
            _step_playback()
        else:
            btn_play.configure(text="▶ 播放")
            if debounce_state["play_timer"]:
                root.after_cancel(debounce_state["play_timer"])

    def _step_playback() -> None:
        if not debounce_state["playing"]:
            return
        tel = ctrl.get_telemetry()
        dur = max(1.0, float(tel.get("duration") or 5.0))
        nxt = ctrl.t + 0.10
        if nxt > dur:
            nxt = 0.0
        ctrl.t = nxt
        time_var.set(nxt)
        time_lbl.configure(text=f"t = {nxt:.2f} s")
        refresh_viewport(force=True)
        debounce_state["play_timer"] = root.after(110, _step_playback)

    btn_play.configure(command=toggle_playback)

    def on_preview_scale_changed(_e: Any = None) -> None:
        ix = preview_scale_cb.current()
        if 0 <= ix < len(preview_scale_map):
            ctrl.preview_scale = preview_scale_map[ix][0]
            ctrl.free_memory()
            refresh_viewport(force=True)

    preview_scale_cb.bind("<<ComboboxSelected>>", on_preview_scale_changed)

    ab_chk.configure(
        command=lambda: (
            setattr(ctrl, "ab_enabled", bool(ab_var.get())),
            refresh_viewport(force=True),
        )
    )
    auto_chk.configure(command=lambda: setattr(ctrl, "auto_refresh", bool(auto_var.get())))
    graph_chk.configure(
        command=lambda: (
            setattr(ctrl, "use_graph", bool(graph_mode_var.get())),
            refresh_viewport(force=True),
        )
    )

    def on_free_memory_clicked() -> None:
        info = ctrl.free_memory()
        update_telemetry_ui()
        toast(f"🧹 已清理全部阶段帧缓存与中间张量：当前内存 {info['after_mb']} MB (释放 {info['freed_mb']} MB)")

    btn_free_mem.configure(command=on_free_memory_clicked)

    def set_quad_corner(c_idx: int) -> None:
        ctrl.quad_corner = c_idx
        for i, b in enumerate(quad_corner_btns):
            b.configure(style="TabActive.TButton" if i == c_idx else "Tab.TButton")
        refresh_viewport(force=True)

    for idx_c, b_c in enumerate(quad_corner_btns):
        b_c.configure(command=lambda ci=idx_c: set_quad_corner(ci))

    quad_border_cb.bind(
        "<<ComboboxSelected>>",
        lambda _e: (
            setattr(ctrl, "quad_border", quad_border_cb.current()),
            refresh_viewport(force=True),
        ),
    )
    quad_text_cb.bind(
        "<<ComboboxSelected>>",
        lambda _e: (
            setattr(ctrl, "quad_text", quad_text_cb.current()),
            refresh_viewport(force=True),
        ),
    )

    def apply_wt_zoom(z_val: float) -> None:
        z = float(np.clip(z_val, 0.0, 1.0))
        wt_zoom_var.set(z)
        wt_readout_lbl.configure(text=ctrl.get_wt_readout(z))
        ctrl.set_param("camera", "wt_zoom", z)
        if ctrl.active_layer == "camera":
            render_params_panel()
        schedule_refresh(60)

    wt_slider.configure(command=lambda v: apply_wt_zoom(float(v)))
    btn_zoom_w.configure(command=lambda: apply_wt_zoom(wt_zoom_var.get() - 0.08))
    btn_zoom_t.configure(command=lambda: apply_wt_zoom(wt_zoom_var.get() + 0.08))

    wt_dyn_cb.bind(
        "<<ComboboxSelected>>",
        lambda _e: (
            ctrl.set_param("camera", "zoom_dynamic_mode", float(wt_dyn_cb.current())),
            refresh_viewport(force=True),
        ),
    )
    wt_osd_chk.configure(
        command=lambda: (
            ctrl.set_param("camera", "zoom_osd", 1.0 if wt_osd_var.get() else 0.0),
            refresh_viewport(force=True),
        )
    )

    def on_audio_play() -> None:
        ctrl.audio_clean = bool(audio_clean_var.get())
        toast("正在合成 3s 物理伴音与音画同步故障包络…")
        root.update_idletasks()
        try:
            wav_bytes, _tl = ctrl.synthesize_audio_window(duration_s=3.0)
            refresh_audio_scope_ui()
            audio_player.play_wav_bytes(wav_bytes)
            update_telemetry_ui()
            toast("▶ 正在播放 3s 伴音试听窗口")
        except Exception as exc:
            toast(f"伴音试听失败: {exc}")

    def on_audio_export_win() -> None:
        ctrl.audio_clean = bool(audio_clean_var.get())
        out = filedialog.asksaveasfilename(
            parent=root,
            title="导出 3s 试听段 WAV",
            defaultextension=".wav",
            initialfile=f"vml_audio_t{ctrl.t:.2f}s.wav",
            filetypes=[("WAV Audio", "*.wav")],
        )
        if not out:
            return
        wav_bytes, _ = ctrl.synthesize_audio_window(duration_s=3.0)
        Path(out).write_bytes(wav_bytes)
        refresh_audio_scope_ui()
        toast(f"已导出 3s 试听段伴音：{out}")

    def on_audio_export_full() -> None:
        ctrl.audio_clean = bool(audio_clean_var.get())
        out = filedialog.asksaveasfilename(
            parent=root,
            title="导出全片完整伴音 WAV",
            defaultextension=".wav",
            initialfile="vml_full_audio.wav",
            filetypes=[("WAV Audio", "*.wav")],
        )
        if not out:
            return
        ctrl.export_full_audio_wav(out)
        toast(f"已导出全片完整伴音：{out}")

    btn_audio_play.configure(command=on_audio_play)
    btn_audio_stop.configure(command=audio_player.stop)
    btn_audio_win_wav.configure(command=on_audio_export_win)
    btn_audio_full_wav.configure(command=on_audio_export_full)

    def on_audio_scope_click(event: Any) -> None:
        w = max(1, audio_scope_canvas.winfo_width())
        frac = float(np.clip(event.x / float(w), 0.0, 1.0))
        tl = ctrl.last_audio_tl
        if tl:
            new_t = float(tl.get("start_s", ctrl.t)) + frac * float(tl.get("duration_s", 3.0))
            ctrl.t = new_t
            time_var.set(new_t)
            time_lbl.configure(text=f"t = {new_t:.2f} s")
            refresh_viewport(force=True)

    audio_scope_canvas.bind("<Button-1>", on_audio_scope_click)

    def toggle_scopes_drawer() -> None:
        ctrl.scopes_open = not ctrl.scopes_open
        if ctrl.scopes_open:
            scopes_drawer.pack(side=tk.TOP, fill=tk.X, pady=(0, 3))
            root.update_idletasks()
            refresh_scopes_ui()
            toast("已展开广播级实时示波器 (Tektronix 1740A IRE / Vectorscope / PSD)")
        else:
            scopes_drawer.pack_forget()
            ctrl.free_memory()
            toast("已收起广播示波器（后台停止计算示波器以节省 CPU 与内存）")

    scope_line_slider.configure(
        command=lambda v: (
            setattr(ctrl, "scope_line_y", float(v)),
            refresh_scopes_ui() if ctrl.scopes_open else None,
        )
    )

    # INTERACTIVE NODE GRAPH EDITOR WINDOW
    def open_node_graph_window() -> None:
        win = tk.Toplevel(root)
        win.title("设备接线画布 (Node Graph Editor · 支持打包节点组 Ctrl+G / 解包 / 串联复制)")
        win.geometry("1220x740")
        win.configure(bg=BG_MAIN)

        selected_nodes: set = set()
        drag_state: Dict[str, Any] = {"node": None, "port_from": None}
        node_boxes_cache: Dict[str, Tuple[int, int, int, int]] = {}
        port_centers_cache: Dict[Tuple[str, str, str], Tuple[int, int]] = {}

        top_bar = ttk.Frame(win, style="Panel.TFrame", padding=6)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(
            top_bar,
            text="🕸 设备接线画布 (拖拽移动节点 · 点击输出端口再点击输入端口连线 · 右键点击输入端口断开)",
            bg=BG_PANEL,
            fg=ACCENT_GREEN,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 10))

        grp_name_var = tk.StringVar(value="VHS")
        tk.Entry(top_bar, textvariable=grp_name_var, width=10, bg=BG_INPUT, fg=FG_TEXT).pack(
            side=tk.LEFT, padx=3
        )

        body_pane = tk.PanedWindow(win, orient=tk.HORIZONTAL, bg=BORDER_COLOR, sashwidth=4)
        body_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left_pal = ttk.Frame(body_pane, style="Panel.TFrame", width=230, padding=6)
        g_canvas = tk.Canvas(body_pane, bg="#090d16", highlightthickness=0)
        right_insp = ttk.Frame(body_pane, style="Panel.TFrame", width=300, padding=6)

        body_pane.add(left_pal, minsize=200, width=230)
        body_pane.add(g_canvas, minsize=500)
        body_pane.add(right_insp, minsize=250, width=300)

        tk.Label(left_pal, text="添加设备节点 (双击加入):", bg=BG_PANEL, fg=ACCENT_CYAN).pack(
            side=tk.TOP, anchor="w", pady=(0, 4)
        )
        cat_list = ctrl.app.graph_state().get("catalog") or []
        pal_lb = tk.Listbox(
            left_pal, bg=BG_INPUT, fg=FG_TEXT, selectbackground="#065f46", selectforeground=ACCENT_GREEN
        )
        pal_lb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        for item in cat_list:
            pal_lb.insert(tk.END, f"{item['kind']} · {item['label']}")

        insp_title = tk.Label(
            right_insp,
            text="节点参数检查器",
            bg=BG_PANEL,
            fg=ACCENT_GREEN,
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        insp_title.pack(side=tk.TOP, anchor="w", pady=(0, 6))
        insp_box = ttk.Frame(right_insp, style="Panel.TFrame")
        insp_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        def redraw_graph() -> None:
            nonlocal node_boxes_cache, port_centers_cache
            cw = max(480, g_canvas.winfo_width())
            ch = max(340, g_canvas.winfo_height())
            pil_img, node_boxes_cache, port_centers_cache = build_node_graph_pil(
                ctrl, width=cw, height=ch, selected_nodes=selected_nodes
            )
            tk_img = ImageTk.PhotoImage(pil_img)
            photo_refs["node_graph"] = tk_img
            g_canvas.delete("all")
            g_canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

        def populate_inspector(node_name: Optional[str]) -> None:
            for w in insp_box.winfo_children():
                w.destroy()
            if not node_name or node_name not in ctrl.app.graph.nodes:
                tk.Label(insp_box, text="点击画布上的节点查看/编辑属性", bg=BG_PANEL, fg=FG_MUTED).pack(
                    side=tk.TOP, anchor="w"
                )
                return
            nd = ctrl.app.graph.nodes[node_name]
            insp_title.configure(text=f"节点: {nd.label or nd.spec.label} ({node_name})")

            en_var = tk.BooleanVar(value=bool(nd.enabled))
            ttk.Checkbutton(
                insp_box,
                text="启用该节点 (Enabled)",
                variable=en_var,
                command=lambda: (
                    ctrl.app.graph_node_params({"name": node_name, "enabled": en_var.get()}),
                    redraw_graph(),
                    refresh_viewport(force=True),
                ),
            ).pack(side=tk.TOP, anchor="w", pady=4)

            tk.Label(insp_box, text="串联迭代次数 (Iterations 1..25):", bg=BG_PANEL, fg=FG_TEXT).pack(
                side=tk.TOP, anchor="w", pady=(6, 2)
            )
            it_var = tk.IntVar(value=int(getattr(nd, "iterations", 1) or 1))
            it_spin = tk.Spinbox(
                insp_box,
                from_=1,
                to=25,
                textvariable=it_var,
                bg=BG_INPUT,
                fg=FG_TEXT,
                command=lambda: (
                    ctrl.app.graph_node_params({"name": node_name, "iterations": int(it_var.get())}),
                    redraw_graph(),
                    refresh_viewport(force=True),
                ),
            )
            it_spin.pack(side=tk.TOP, fill=tk.X, pady=2)

            p_layer = nd.spec.params_layer
            if p_layer:
                ttk.Button(
                    insp_box,
                    text=f"👉 跳转到「{p_layer}」详细参数面板",
                    style="Primary.TButton",
                    command=lambda: switch_layer_tab(p_layer),
                ).pack(side=tk.TOP, fill=tk.X, pady=10)

        def on_add_node_from_palette(_e: Any = None) -> None:
            sel = pal_lb.curselection()
            if not sel:
                return
            ix = int(sel[0])
            if 0 <= ix < len(cat_list):
                kind = cat_list[ix]["kind"]
                ctrl.app.graph_add({"kind": kind, "x": 180.0, "y": 140.0})
                redraw_graph()

        pal_lb.bind("<Double-Button-1>", on_add_node_from_palette)

        def on_canvas_down(event: Any) -> None:
            x, y = event.x, event.y
            for (n_name, p_name, p_dir), (px, py) in port_centers_cache.items():
                if (x - px) ** 2 + (y - py) ** 2 <= 81:
                    if p_dir == "out":
                        drag_state["port_from"] = (n_name, p_name)
                        toast(f"已选中输出端口 {n_name}.{p_name}，请点击目标节点的输入端口完成连线")
                    elif p_dir == "in" and drag_state.get("port_from"):
                        src_n, src_p = drag_state["port_from"]
                        res = ctrl.app.graph_connect(
                            {"src": src_n, "src_port": src_p, "dst": n_name, "dst_port": p_name}
                        )
                        drag_state["port_from"] = None
                        if res.get("error"):
                            toast(f"连线失败: {res['error']}")
                        else:
                            redraw_graph()
                            refresh_viewport(force=True)
                    return

            hit_node = None
            for n_name, (x0, y0, x1, y1) in reversed(list(node_boxes_cache.items())):
                if x0 <= x <= x1 and y0 <= y <= y1:
                    hit_node = n_name
                    break

            if hit_node:
                if event.state & 0x0001:
                    if hit_node in selected_nodes:
                        selected_nodes.remove(hit_node)
                    else:
                        selected_nodes.add(hit_node)
                else:
                    if hit_node not in selected_nodes:
                        selected_nodes.clear()
                        selected_nodes.add(hit_node)
                drag_state["node"] = hit_node
                populate_inspector(hit_node)
            else:
                selected_nodes.clear()
                drag_state["node"] = None
                populate_inspector(None)
            redraw_graph()

        def on_canvas_drag(event: Any) -> None:
            n_name = drag_state.get("node")
            if n_name:
                ctrl.app.graph_move({"name": n_name, "x": max(10, event.x - 70), "y": max(10, event.y - 20)})
                redraw_graph()

        def on_canvas_right_click(event: Any) -> None:
            x, y = event.x, event.y
            for (n_name, p_name, p_dir), (px, py) in port_centers_cache.items():
                if p_dir == "in" and (x - px) ** 2 + (y - py) ** 2 <= 100:
                    ctrl.app.graph_disconnect({"dst": n_name, "dst_port": p_name})
                    redraw_graph()
                    refresh_viewport(force=True)
                    toast(f"已断开输入端口 {n_name}.{p_name}")
                    return

        g_canvas.bind("<Button-1>", on_canvas_down)
        g_canvas.bind("<B1-Motion>", on_canvas_drag)
        g_canvas.bind("<Button-3>", on_canvas_right_click)
        g_canvas.bind("<Configure>", lambda _e: redraw_graph())

        ttk.Button(
            top_bar,
            text="☑ 选VHS四件套",
            command=lambda: (
                selected_nodes.clear(),
                selected_nodes.update(
                    [
                        name
                        for name, nd in ctrl.app.graph.nodes.items()
                        if nd.kind in ("camera", "format", "tape", "player")
                    ]
                ),
                redraw_graph(),
            ),
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            top_bar,
            text="📦 打包节点组",
            style="Primary.TButton",
            command=lambda: (
                ctrl.app.graph_group_nodes({"nodes": list(selected_nodes), "label": grp_name_var.get() or "VHS"}),
                selected_nodes.clear(),
                redraw_graph(),
                refresh_viewport(force=True),
            ),
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            top_bar,
            text="🔓 解包",
            command=lambda: (
                ctrl.app.graph_ungroup_node({"name": next(iter(selected_nodes), "")}),
                selected_nodes.clear(),
                redraw_graph(),
                refresh_viewport(force=True),
            ),
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            top_bar,
            text="📋 复制",
            command=lambda: (
                ctrl.app.graph_duplicate_node({"name": next(iter(selected_nodes), ""), "chain": False}),
                redraw_graph(),
            ),
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            top_bar,
            text="⛓ 串联复制",
            command=lambda: (
                ctrl.app.graph_chain_duplicate({"name": next(iter(selected_nodes), "")}),
                redraw_graph(),
                refresh_viewport(force=True),
            ),
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            top_bar,
            text="🗑 删除所选",
            command=lambda: (
                [ctrl.app.graph_remove({"name": n}) for n in list(selected_nodes)],
                selected_nodes.clear(),
                redraw_graph(),
                refresh_viewport(force=True),
            ),
        ).pack(side=tk.LEFT, padx=2)

        win.after(80, redraw_graph)

    # HEADER ACTION BUTTONS
    def on_import_clip() -> None:
        path = filedialog.askopenfilename(
            parent=root,
            title="导入视频或图像素材",
            filetypes=[
                ("Media Files", "*.mp4 *.mov *.avi *.mkv *.webm *.png *.jpg *.jpeg *.bmp *.tif"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            ctrl.import_clip(path)
            sync_all_controls_from_project()
            refresh_viewport(force=True)
            toast(f"已导入素材：{Path(path).name}")
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc), parent=root)

    def on_import_overlay() -> None:
        path = filedialog.askopenfilename(
            parent=root,
            title="导入切换台/画中画叠层图像或视频",
            filetypes=[("Media Files", "*.png *.jpg *.jpeg *.mp4 *.mov *.avi"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            ctrl.import_overlay(path)
            sync_all_controls_from_project()
            refresh_viewport(force=True)
            toast(f"已导入叠层素材：{Path(path).name}")
        except Exception as exc:
            messagebox.showerror("导入叠层失败", str(exc), parent=root)

    def on_export_still() -> None:
        ext = ".jpg" if ctrl.export_quality in ("medium", "draft") else ".png"
        out = filedialog.asksaveasfilename(
            parent=root,
            title="导出当前帧静帧图像",
            defaultextension=ext,
            initialfile=f"vml_still_t{ctrl.t:.2f}s{ext}",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")],
        )
        if not out:
            return
        toast("正在渲染并导出静帧…")
        root.update_idletasks()
        ctrl.export_still_file(out)
        update_telemetry_ui()
        toast(f"已导出静帧：{out}")

    def on_export_quad_burst() -> None:
        out_dir = filedialog.askdirectory(parent=root, title="选择批量四连拍输出文件夹")
        if not out_dir:
            return
        toast("正在批量渲染 2×2 四分屏 + 四个独立角落静帧…")
        root.update_idletasks()
        saved = ctrl.export_quad_burst_dir(out_dir, include_quad=True)
        update_telemetry_ui()
        toast(f"📸 已批量导出 {len(saved)} 张静帧至：{out_dir}")

    def on_export_quad_zip() -> None:
        out_zip = filedialog.asksaveasfilename(
            parent=root,
            title="导出四连拍 ZIP 压缩包",
            defaultextension=".zip",
            initialfile=f"vml_quad_burst_t{ctrl.t:.2f}s.zip",
            filetypes=[("ZIP Archive", "*.zip")],
        )
        if not out_zip:
            return
        toast("正在打包四连拍 ZIP…")
        root.update_idletasks()
        ctrl.export_quad_zip_file(out_zip)
        update_telemetry_ui()
        toast(f"📦 已导出四连拍 ZIP：{out_zip}")

    def on_export_config_json() -> None:
        out = filedialog.asksaveasfilename(
            parent=root,
            title="导出工程配置 JSON",
            defaultextension=".json",
            initialfile="vml_config.json",
            filetypes=[("JSON Config", "*.json")],
        )
        if not out:
            return
        cfg = ctrl.app.export_config()
        Path(out).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        toast(f"已导出设置文件：{out}")

    def on_import_config_json() -> None:
        path = filedialog.askopenfilename(
            parent=root, title="导入工程配置 JSON", filetypes=[("JSON Config", "*.json")]
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            res = ctrl.app.import_config(payload)
            ctrl.free_memory()
            sync_all_controls_from_project()
            refresh_viewport(force=True)
            n_app = (res.get("imported") or {}).get("params_applied", 0)
            toast(f"已导入 {n_app} 项配置参数")
        except Exception as exc:
            messagebox.showerror("导入配置失败", str(exc), parent=root)

    def on_offline_render_mp4() -> None:
        out_mp4 = filedialog.asksaveasfilename(
            parent=root,
            title="离线渲染导出 MP4 视频 (流式低内存模式)",
            defaultextension=".mp4",
            initialfile="vml_render.mp4",
            filetypes=[("MP4 Video", "*.mp4")],
        )
        if not out_mp4:
            return
        scale = ctrl.app.resolve_scale(ctrl.export_scale, default=1.0)
        crf_map = {"lossless": 12, "high": 16, "medium": 20, "draft": 24}
        crf = crf_map.get(ctrl.export_quality, 16)
        job_id = ctrl.app.start_render(scale=scale, crf=crf, keep_audio=True)

        def _poll_job() -> None:
            job = ctrl.app.jobs.get(job_id) or {}
            st = job.get("state")
            prog = float(job.get("progress") or 0.0) * 100.0
            msg = job.get("message") or ""
            if st in ("queued", "running"):
                toast(f"🎬 离线渲染中: {prog:.1f}% ({msg})")
                root.after(400, _poll_job)
            elif st == "done" and job.get("path"):
                shutil.copy2(job["path"], out_mp4)
                ctrl.free_memory()
                update_telemetry_ui()
                toast(f"✅ 离线渲染完成，已保存至：{out_mp4}")
            else:
                toast(f"❌ 离线渲染失败: {msg}")

        _poll_job()

    btn_quad_still.configure(command=on_export_still)
    btn_quad_burst.configure(command=on_export_quad_burst)
    btn_quad_zip.configure(command=on_export_quad_zip)

    ttk.Button(actions_bar, text="📂 导入素材", command=on_import_clip).pack(side=tk.LEFT, padx=2)
    ttk.Button(actions_bar, text="🖼 导入叠层", command=on_import_overlay).pack(side=tk.LEFT, padx=2)

    tk.Label(actions_bar, text="种子:", bg=BG_PANEL, fg=FG_MUTED).pack(side=tk.LEFT, padx=(6, 2))
    seed_var = tk.IntVar(value=ctrl.app.current.seed)
    seed_spin = tk.Spinbox(
        actions_bar,
        from_=0,
        to=999999999,
        textvariable=seed_var,
        width=9,
        bg=BG_INPUT,
        fg=ACCENT_GREEN,
        relief=tk.FLAT,
        command=lambda: (ctrl.set_seed(seed_var.get()), schedule_refresh(60)),
    )
    seed_spin.pack(side=tk.LEFT, padx=2)
    seed_spin.bind("<Return>", lambda _e: (ctrl.set_seed(seed_var.get()), refresh_viewport(force=True)))

    ttk.Button(actions_bar, text="🕸 接线画布", command=open_node_graph_window).pack(side=tk.LEFT, padx=3)
    ttk.Button(actions_bar, text="📊 广播示波器", command=toggle_scopes_drawer).pack(side=tk.LEFT, padx=3)
    ttk.Button(actions_bar, text="⬇ 导出静帧", command=on_export_still).pack(side=tk.LEFT, padx=2)
    ttk.Button(actions_bar, text="📸 批量四连拍", style="Primary.TButton", command=on_export_quad_burst).pack(
        side=tk.LEFT, padx=2
    )
    ttk.Button(actions_bar, text="📦 四连拍ZIP", command=on_export_quad_zip).pack(side=tk.LEFT, padx=2)
    ttk.Button(actions_bar, text="导出JSON", command=on_export_config_json).pack(side=tk.LEFT, padx=2)
    ttk.Button(actions_bar, text="导入JSON", command=on_import_config_json).pack(side=tk.LEFT, padx=2)
    ttk.Button(actions_bar, text="🎬 离线渲染", style="Primary.TButton", command=on_offline_render_mp4).pack(
        side=tk.LEFT, padx=(4, 0)
    )

    render_preset_tabs()
    sync_all_controls_from_project()
    root.after(100, lambda: (refresh_viewport(force=True), refresh_audio_scope_ui()))

    root.protocol("WM_DELETE_WINDOW", lambda: (audio_player.stop(), root.destroy()))
    root.mainloop()
    return 0


def run_self_test() -> int:
    """Headless self-test verifying all DesktopBackendController & rendering pipelines."""
    print("[Desktop Self-Test] Initializing DesktopBackendController (low_memory=True)...")
    ctrl = DesktopBackendController(low_memory=True)

    tel0 = ctrl.get_telemetry()
    assert tel0["source_w"] > 0 and tel0["source_h"] > 0, "Invalid source dimensions"

    # 1. Test zero-encode viewport rendering & A/B split compositing
    ctrl.ab_enabled = True
    ctrl.split_x = 0.45
    vp_img = ctrl.render_viewport_pil(canvas_w=640, canvas_h=360)
    assert isinstance(vp_img, Image.Image) and vp_img.width > 0 and vp_img.height > 0

    # 2. Test preset switching & parameter mutation
    entries = ctrl.get_preset_entries("factory")
    assert len(entries) > 5, "Expected factory presets"
    ctrl.apply_preset("family_archive_1992")
    ctrl.set_param("camera", "wt_zoom", 0.5)
    assert "×" in ctrl.get_wt_readout(0.5)

    # 3. Test Quad 2x2 and individual corner rendering
    ctrl.quad_corner = 0
    rgb_q0, ms_q0 = ctrl.render_frame_numpy()
    assert rgb_q0.ndim == 3 and rgb_q0.shape[2] == 3

    # 4. Test lazy Broadcast Scopes rendering
    sc_img, sc_data = ctrl.render_scopes_pil(width=720, height=160)
    assert isinstance(sc_img, Image.Image) and "waveform" in sc_data and "vectorscope" in sc_data

    # 5. Test Audio synthesis & Audio Oscilloscope rendering
    wav_bytes, tl = ctrl.synthesize_audio_window(duration_s=1.0)
    assert wav_bytes[:4] == b"RIFF" and len(wav_bytes) > 1000
    aud_img, badge = ctrl.render_audio_scope_pil(width=480, height=40)
    assert isinstance(aud_img, Image.Image) and len(badge) > 0

    # 6. Test Node Graph PIL renderer
    g_img, n_boxes, p_centers = build_node_graph_pil(ctrl, width=800, height=450)
    assert isinstance(g_img, Image.Image) and len(n_boxes) > 0 and len(p_centers) > 0

    # 7. Test one-click memory cleanup
    mem_res = ctrl.free_memory()
    assert "after_mb" in mem_res and len(ctrl.app.store._renderers) == 0

    print(
        f"[Desktop Self-Test] PASS — Frame: {ms_q0:.1f} ms | RAM after cleanup: {mem_res['after_mb']:.1f} MB | Selectors: {len(SELECTORS)}"
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Virtual Media Lab — Native Low-Memory Desktop GUI")
    parser.add_argument("--clip", type=str, default=None, help="启动时自动加载的视频或图片路径")
    parser.add_argument("--no-low-memory", action="store_true", help="关闭激进低内存缓存限制")
    parser.add_argument("--self-test", action="store_true", help="运行无头自检测试并退出")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()
    return launch_desktop_gui(default_clip=args.clip, low_memory=not args.no_low_memory)


if __name__ == "__main__":
    raise SystemExit(main())
