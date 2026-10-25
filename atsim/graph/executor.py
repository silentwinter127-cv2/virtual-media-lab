"""
Graph executor: run a node graph (including Blender-style `node_group` subgraphs
and `overlay` character-generator nodes) and produce frames.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from ..core.pipeline import Chain, FrameBuffer, RenderContext, Stage
from ..core.rng import derive_seed
from ..core.signal import SignalFrame, composite_encode
from ..core.standards import VideoStandard
from ..stages.display import CRTReshootStage, CRTMagnetEnvStage
from ..render.renderer import (STAGE_ORDER, STAGE_LAYERS, STAGE_LABELS,
                               CameraStage, CodecStage, ConnectorStage,
                               DisplayStage, FormatStage, OverlayStage,
                               PlayerStage, TapeStage, _hash)
from ..stages.player import SyncTBCStage
from ..stages.tape import TransportServoStage
from ..stages.connector import VideoSwitcherStage
from ..codecs.patchbay_da import DistAmpStage, PatchRouterStage
from ..codecs.test_signal_gen import TestGeneratorStage
from ..codecs.multiview_wall import MultiviewWallStage
from ..codecs.studio_ccu_genlock import CCURackStage
from ..codecs.master_playout import MasterPlayoutStage
from ..codecs.transfer_workstation import TransferStationStage
from ..codecs.edit_controller_nle import LinearEditorStage, NLETimelineStage
from ..codecs.playout_automation_v2 import AutomationControllerStage
from ..codecs.satellite_ird import SatDishLNBStage, SatIRDReceiverStage
from ..codecs.tv_transmitter_chain import TVExciterPAStage
from ..codecs.film_optics import FilmOpticsStage
from ..codecs.photochemical_film import PhotochemicalFilmStage
from ..codecs.film_telecine import FilmTelecineStage
from ..codecs.inversion_calibration_lab import InversionAnalyzerStage
from ..codecs.archival_archaeology import ArchivalVaultStage
from ..codecs.spatial_environment_field import EnvChamberStage
from ..codecs.network_streaming_era import StreamCDNClientStage
from ..codecs.mobile_isp_optics import MobileCameraISPStage
from ..codecs.legacy_digital_media import FileTransferStage, LCDFlashPlayerStage
from ..codecs.perception_and_operator import HumanOperatorStage
from ..codecs.media_ecosystem_evolution import EcosystemSimulatorStage
from ..codecs.audio import AVSyncClockStage
from .graph import CATALOG_BY_KIND, Graph, Node

# node kind -> the stage class that implements it (None = handled specially)
STAGE_FOR_KIND: dict[str, type[Stage] | None] = {
    "camera": CameraStage,
    "encoder": FormatStage,
    "dv_codec": CodecStage,
    "connector": ConnectorStage,
    "power_supply": ConnectorStage,
    "video_switcher": VideoSwitcherStage,
    "dist_amp": DistAmpStage,
    "patch_router": PatchRouterStage,
    "tape": TapeStage,
    "transport_servo": TransportServoStage,
    "vhs_gen": TapeStage,
    "vhs_dub_loop": TapeStage,
    "vhs_iterative_loop": TapeStage,
    "deck": PlayerStage,
    "sync_tbc": SyncTBCStage,
    "overlay": OverlayStage,
    "display": DisplayStage,
    "offscreen_camera": CRTReshootStage,
    "crt_magnet_env": CRTMagnetEnvStage,
    "av_sync_clock": AVSyncClockStage,
    "test_generator": TestGeneratorStage,
    "multiview_wall": MultiviewWallStage,
    "ccu_rack": CCURackStage,
    "master_playout": MasterPlayoutStage,
    "transfer_station": TransferStationStage,
    "linear_editor": LinearEditorStage,
    "nle_timeline": NLETimelineStage,
    "automation_controller": AutomationControllerStage,
    "sat_dish_lnb": SatDishLNBStage,
    "sat_ird_receiver": SatIRDReceiverStage,
    "tv_exciter_pa": TVExciterPAStage,
    "film_optics": FilmOpticsStage,
    "photochemical_film": PhotochemicalFilmStage,
    "film_scanner_telecine": FilmTelecineStage,
    "inversion_analyzer": InversionAnalyzerStage,
    "archival_vault": ArchivalVaultStage,
    "env_chamber": EnvChamberStage,
    "stream_cdn_client": StreamCDNClientStage,
    "mobile_camera_isp": MobileCameraISPStage,
    "file_transfer_node": FileTransferStage,
    "lcd_flash_player": LCDFlashPlayerStage,
    "human_operator": HumanOperatorStage,
    "ecosystem_simulator": EcosystemSimulatorStage,
    "source": None,
    "hdmi_out": None,
    "hdmi_to_av": None,
    "rf_modulator": None,
    "rf_tuner": None,
    "node_group": None,
}


@dataclass
class GraphRun:
    """Result of executing a graph for one frame."""
    rgb: np.ndarray
    node_frames: dict[str, np.ndarray] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class GraphExecutor:
    """
    Executes a `Graph` over a frame source, supporting:
      * Per-node isolated `RenderContext`, per-node parameter overrides, and `iterations`
      * First-class `overlay` (`字幕机`) nodes anywhere in the graph, PLUS automatic
        project-level `overlay` support when no explicit `overlay` node is present
      * Blender-style `node_group` (`节点组`) subgraphs (including duplicated/chained groups)
    """

    def __init__(self, graph: Graph, std: VideoStandard, params: dict,
                 source, width: int, height: int, fps: float,
                 seed: int = 20240101, scale: float = 1.0,
                 max_cache_frames: int = 8):
        self.graph = graph
        self.std = std
        self.source = source
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.seed = int(seed)
        self.scale = float(scale)
        self.max_cache_frames = int(max(1, max_cache_frames))
        self.base_params = {k: dict(v) for k, v in params.items()}
        self.params = {k: dict(v) for k, v in params.items()}
        self.osd_report: dict = {}

        self.ctx = RenderContext(std=std, params=self.params, seed=self.seed,
                                 frame_count=int(getattr(source, "frame_count", 0) or 0),
                                 fps=fps, width=int(width),
                                 height=int(height), preview=False, scale=scale,
                                 source=source)
        self.stages: dict[str, list[Stage]] = {}
        self.group_stages: dict[str, dict[tuple[int, str], list[Stage]]] = {}
        self._auto_osd_pre: OverlayStage | None = None
        self._auto_osd_tail: OverlayStage | None = None
        self._build_node_stages()

        self.order = graph.order()
        self.hashes = graph.node_hashes(base_params=self.base_params)
        self._cache: dict[str, dict[int, np.ndarray]] = {n: {} for n in graph.nodes}
        self._buf_cache: dict[str, dict[int, FrameBuffer]] = {n: {} for n in graph.nodes}
        self._cache_key: dict[str, dict[int, str]] = {n: {} for n in graph.nodes}
        self._prev_src: np.ndarray | None = None
        self.node_ms: dict[str, float] = {}
        self.skipped: dict[str, int] = {}
        self.last_rgb: np.ndarray | None = None

    def _has_explicit_overlay_node(self) -> bool:
        """Check whether the graph (or any enabled `node_group`) contains an `overlay` node."""
        for node in self.graph.nodes.values():
            if node.kind == "overlay":
                return True
            if node.kind == "node_group" and isinstance(node.subgraph, dict):
                for snd in (node.subgraph.get("nodes") or []):
                    if snd.get("kind") == "overlay":
                        return True
        return False

    def _node_params(self, kind: str, node_params: dict | None, node_enabled: bool = True) -> dict:
        """Build an isolated parameter dictionary for a node or subnode."""
        np_dict = {k: dict(v) for k, v in self.base_params.items()}
        spec = CATALOG_BY_KIND.get(kind)
        layer_name = (spec.params_layer if spec else None) or kind
        if kind == "vhs_gen":
            np_dict.setdefault("tape", {})
            if "generation_passes" not in (node_params or {}) and float(np_dict["tape"].get("generation_passes", 1.0)) <= 1.0:
                np_dict["tape"]["generation_passes"] = 2.0
        elif kind == "vhs_dub_loop":
            np_dict.setdefault("tape", {})
            if "generation_count" not in (node_params or {}) and float(np_dict["tape"].get("generation_count", 1.0)) <= 1.0:
                np_dict["tape"]["generation_count"] = 5.0
        if node_params and layer_name:
            np_dict.setdefault(layer_name, {}).update(node_params)
        if kind == "overlay":
            np_dict.setdefault("overlay", {})
            # When an explicit `overlay` node is enabled on the graph and has text,
            # ensure `overlay.enabled` is active unless explicitly set to 0 in `node_params`.
            if node_enabled and "enabled" not in (node_params or {}):
                if str(np_dict["overlay"].get("text", "")).strip():
                    np_dict["overlay"]["enabled"] = 1.0
        if kind == "display":
            np_dict.setdefault("display", {})
            if node_enabled and "enabled" not in (node_params or {}):
                np_dict["display"]["enabled"] = 1.0
                if "mv_enabled" not in (node_params or {}):
                    _quad_node_kinds = {
                        "stream_cdn_client", "photochemical_film", "film_scanner_telecine",
                        "transfer_station", "inversion_analyzer", "archival_vault",
                        "env_chamber", "sat_ird_receiver", "tv_exciter_pa",
                        "human_operator", "ecosystem_simulator", "crt_magnet_env",
                    }
                    has_upstream_quad = any(
                        getattr(n, "enabled", True) and getattr(n, "kind", "") in _quad_node_kinds
                        for n in self.graph.nodes.values()
                    )
                    if has_upstream_quad:
                        np_dict["display"]["mv_enabled"] = 0.0
            elif not node_enabled:
                np_dict["display"]["enabled"] = 0.0
                np_dict["display"]["mv_enabled"] = 0.0
                np_dict["display"]["crt_magnet_mode"] = 0.0
                np_dict["display"]["kv1214_enabled"] = 0.0
                np_dict["display"]["percept_enabled"] = 0.0
                np_dict["display"]["percept_mode"] = 0.0
                np_dict["display"]["reshoot_enabled"] = 0.0
        if kind == "test_generator":
            np_dict.setdefault("format", {})
            if node_enabled:
                np_dict["format"]["test_gen_enabled"] = 1.0
                if float(np_dict["format"].get("test_pattern_mode", 0.0)) <= 0.0:
                    np_dict["format"]["test_pattern_mode"] = 1.0
        if kind == "multiview_wall":
            np_dict.setdefault("display", {})
            if node_enabled:
                np_dict["display"]["mv_enabled"] = 1.0
        if kind == "ccu_rack":
            np_dict.setdefault("camera", {})
            if node_enabled:
                np_dict["camera"]["ccu_enabled"] = 1.0
        if kind == "master_playout":
            np_dict.setdefault("overlay", {})
            if node_enabled:
                np_dict["overlay"]["playout_enabled"] = 1.0
        if kind == "transfer_station":
            np_dict.setdefault("player", {})
            if node_enabled and float(np_dict["player"].get("transfer_station_mode", 0.0)) <= 0.0:
                np_dict["player"]["transfer_station_mode"] = 4.0
        if kind == "linear_editor":
            np_dict.setdefault("tape", {})
            if node_enabled and int(round(float(np_dict["tape"].get("edit_mode", 0.0)))) not in (1, 2):
                np_dict["tape"]["edit_mode"] = 1.0
        if kind == "nle_timeline":
            np_dict.setdefault("tape", {})
            if node_enabled and int(round(float(np_dict["tape"].get("edit_mode", 0.0)))) not in (3, 4):
                np_dict["tape"]["edit_mode"] = 3.0
        if kind == "automation_controller":
            np_dict.setdefault("overlay", {})
            if node_enabled:
                np_dict["overlay"]["auto_v2_enabled"] = 1.0
                if int(round(float(np_dict["overlay"].get("auto_v2_mode", 0.0)))) <= 0:
                    np_dict["overlay"]["auto_v2_mode"] = 3.0
        if kind == "sat_ird_receiver":
            np_dict.setdefault("rf", {})
            if node_enabled and int(round(float(np_dict["rf"].get("sat_ird_mode", 0.0)))) <= 0:
                np_dict["rf"]["sat_ird_mode"] = 3.0
        if kind == "tv_exciter_pa":
            np_dict.setdefault("rf", {})
            if node_enabled and int(round(float(np_dict["rf"].get("tx_chain_mode", 0.0)))) <= 0:
                np_dict["rf"]["tx_chain_mode"] = 3.0
        if kind == "photochemical_film":
            np_dict.setdefault("film", {})
            if node_enabled:
                np_dict["film"]["enabled"] = 1.0
                if int(round(float(np_dict["film"].get("stock", 0.0)))) <= 0:
                    np_dict["film"]["stock"] = 7.0
        if kind == "film_scanner_telecine":
            np_dict.setdefault("camera", {})
            if node_enabled:
                np_dict["camera"]["film_enabled"] = 1.0
                if int(round(float(np_dict["camera"].get("film_stock", 0.0)))) <= 0:
                    np_dict["camera"]["film_stock"] = 2.0
        if kind == "inversion_analyzer":
            np_dict.setdefault("player", {})
            if node_enabled:
                np_dict["player"]["inv_lab_enabled"] = 1.0
                if int(round(float(np_dict["player"].get("inv_lab_mode", 0.0)))) <= 0:
                    np_dict["player"]["inv_lab_mode"] = 3.0
        if kind == "archival_vault":
            np_dict.setdefault("player", {})
            if node_enabled:
                np_dict["player"]["archival_enabled"] = 1.0
                if int(round(float(np_dict["player"].get("archival_mode", 0.0)))) <= 0:
                    np_dict["player"]["archival_mode"] = 4.0
        if kind == "env_chamber":
            np_dict.setdefault("connector", {})
            if node_enabled:
                np_dict["connector"]["env_field_enabled"] = 1.0
                if int(round(float(np_dict["connector"].get("env_field_mode", 0.0)))) <= 0:
                    np_dict["connector"]["env_field_mode"] = 5.0
        if kind == "stream_cdn_client":
            np_dict.setdefault("codec", {})
            if node_enabled:
                np_dict["codec"]["stream_enabled"] = 1.0
                if int(round(float(np_dict["codec"].get("stream_mode", 0.0)))) <= 0:
                    np_dict["codec"]["stream_mode"] = 6.0
        if kind == "mobile_camera_isp":
            np_dict.setdefault("camera", {})
            if node_enabled:
                np_dict["camera"]["ldme_mobile_enabled"] = 1.0
                if int(round(float(np_dict["camera"].get("ldme_mobile_profile", 0.0)))) <= 0:
                    np_dict["camera"]["ldme_mobile_profile"] = 2.0
        if kind == "file_transfer_node":
            np_dict.setdefault("codec", {})
            if node_enabled:
                np_dict["codec"]["ldme_transfer_enabled"] = 1.0
                if int(round(float(np_dict["codec"].get("ldme_transfer_method", 0.0)))) <= 0:
                    np_dict["codec"]["ldme_transfer_method"] = 4.0
        if kind == "lcd_flash_player":
            np_dict.setdefault("display", {})
            if node_enabled:
                np_dict["display"]["ldme_display_enabled"] = 1.0
                if int(round(float(np_dict["display"].get("ldme_display_device", 0.0)))) <= 0:
                    np_dict["display"]["ldme_display_device"] = 2.0
        if kind == "human_operator":
            np_dict.setdefault("display", {})
            if node_enabled:
                np_dict["display"]["percept_enabled"] = 1.0
                if int(round(float(np_dict["display"].get("percept_mode", 0.0)))) <= 0:
                    np_dict["display"]["percept_mode"] = 4.0
        if kind == "ecosystem_simulator":
            np_dict.setdefault("overlay", {})
            if node_enabled:
                np_dict["overlay"]["eco_enabled"] = 1.0
                if int(round(float(np_dict["overlay"].get("eco_mode", 0.0)))) <= 0:
                    np_dict["overlay"]["eco_mode"] = 4.0
        if kind == "crt_magnet_env":
            np_dict.setdefault("display", {})
            if node_enabled and int(round(float(np_dict["display"].get("crt_magnet_mode", 0.0)))) <= 0:
                np_dict["display"]["crt_magnet_mode"] = 5.0
        if kind == "av_sync_clock":
            np_dict.setdefault("audio", {})
            if node_enabled and "av_sync_enabled" not in (node_params or {}):
                np_dict["audio"]["av_sync_enabled"] = 1.0
        return np_dict

    def _make_stage_instance(self, kind: str, n_ctx: RenderContext) -> Stage | None:
        cls = STAGE_FOR_KIND.get(kind)
        if cls is None:
            return None
        if kind == "overlay":
            st = OverlayStage(n_ctx, slot=OverlayStage.SLOT_GRAPH)
        else:
            st = cls(n_ctx)
        st.prepare()
        return st

    def _build_node_stages(self, only_nodes: set[str] | None = None) -> None:
        """Create isolated Stage instance(s) per node, per subnode, and per iteration pass."""
        if only_nodes is None:
            self.stages = {}
            self.group_stages = {}
        frame_cnt = int(getattr(self.source, "frame_count", 0) or 0)
        for name, node in self.graph.nodes.items():
            if only_nodes is not None and name not in only_nodes:
                continue
            if node.kind == "node_group" and isinstance(node.subgraph, dict):
                g_iters = max(1, min(25, int(getattr(node, "iterations", 1) or 1)))
                g_map: dict[tuple[int, str], list[Stage]] = {}
                for g_it in range(g_iters):
                    for snd in (node.subgraph.get("nodes") or []):
                        s_name = str(snd.get("name", ""))
                        s_kind = str(snd.get("kind", ""))
                        s_enabled = bool(snd.get("enabled", True))
                        s_iters = max(1, min(25, int(snd.get("iterations", 1) or 1)))
                        if STAGE_FOR_KIND.get(s_kind) is None:
                            continue
                        s_np = self._node_params(s_kind, snd.get("params"), node_enabled=s_enabled)
                        s_passes: list[Stage] = []
                        for s_it in range(s_iters):
                            s_seed = derive_seed(self.seed, "group", name, g_it, s_name, s_it)
                            s_ctx = RenderContext(
                                std=self.std, params={k: dict(v) for k, v in s_np.items()},
                                seed=s_seed, frame_count=frame_cnt, fps=self.fps,
                                width=int(self.width), height=int(self.height),
                                preview=False, scale=self.scale, source=self.source,
                            )
                            st = self._make_stage_instance(s_kind, s_ctx)
                            if st is not None:
                                s_passes.append(st)
                        g_map[(g_it, s_name)] = s_passes
                self.group_stages[name] = g_map
                continue

            if STAGE_FOR_KIND.get(node.kind) is None:
                continue
            iters = max(1, min(25, int(getattr(node, "iterations", 1) or 1)))
            pass_stages: list[Stage] = []
            np_dict = self._node_params(node.kind, node.params, node_enabled=node.enabled)
            for it in range(iters):
                node_seed = derive_seed(self.seed, "node", name, it)
                n_ctx = RenderContext(
                    std=self.std, params={k: dict(v) for k, v in np_dict.items()},
                    seed=node_seed, frame_count=frame_cnt, fps=self.fps,
                    width=int(self.width), height=int(self.height),
                    preview=False, scale=self.scale, source=self.source,
                )
                st = self._make_stage_instance(node.kind, n_ctx)
                if st is not None:
                    if hasattr(st, "connected_loads"):
                        n_out_edges = sum(1 for ed in self.graph.edges if ed.src == name)
                        if n_out_edges >= 2:
                            st.connected_loads = max(int(st.spec.load_count), n_out_edges)
                    pass_stages.append(st)
            self.stages[name] = pass_stages

        # Build fallback project-level OverlayStage instances for when the user enables
        # `字幕机` in the left panel without adding an explicit `overlay` node to the canvas.
        ov_ctx = RenderContext(
            std=self.std, params={k: dict(v) for k, v in self.base_params.items()},
            seed=derive_seed(self.seed, "graph_auto_osd"),
            frame_count=frame_cnt, fps=self.fps,
            width=int(self.width), height=int(self.height),
            preview=False, scale=self.scale, source=self.source,
        )
        self._auto_osd_pre = OverlayStage(ov_ctx, slot=OverlayStage.SLOT_PRE_TAPE)
        self._auto_osd_tail = OverlayStage(ov_ctx, slot=OverlayStage.SLOT_TAIL)

    def sync_params(self, params: dict) -> None:
        """Update base parameters and rebuild node stages if hashes changed."""
        self.base_params = {k: dict(v) for k, v in params.items()}
        self.params = {k: dict(v) for k, v in params.items()}
        self.ctx.params = self.params
        self._rflink = None
        self._fmt = None
        new_hashes = self.graph.node_hashes(base_params=self.base_params)
        if new_hashes != self.hashes or set(self.graph.nodes) != set(self._cache):
            self.order = self.graph.order()
            for n in list(self._cache):
                if n not in self.graph.nodes:
                    self._cache.pop(n, None)
                    self._buf_cache.pop(n, None)
                    self._cache_key.pop(n, None)
                elif self.hashes.get(n) != new_hashes.get(n):
                    self._cache[n].clear()
                    self._buf_cache[n].clear()
                    self._cache_key[n].clear()
            for n in self.graph.nodes:
                self._cache.setdefault(n, {})
                self._buf_cache.setdefault(n, {})
                self._cache_key.setdefault(n, {})
            dirty_set = {n for n in self.graph.nodes if self.hashes.get(n) != new_hashes.get(n) or (n not in self.stages and n not in self.group_stages)}
            self._build_node_stages(only_nodes=dirty_set if dirty_set else None)
            self.hashes = new_hashes

    # ------------------------------------------------------------------
    def _rf(self):
        if getattr(self, "_rflink", None) is None:
            from ..codecs.rf import RFLink, RFParams
            p = self.ctx.layer("rf") or {}
            self._rflink = RFLink(self.std, RFParams(
                channel_mhz=float(p.get("channel_mhz", 6.0)),
                carrier_offset_mhz=float(p.get("carrier_offset_mhz", 1.25)),
                noise_figure_db=float(p.get("noise_figure_db", 8.0)),
                signal_dbm=float(p.get("signal_dbm", -60.0)),
                sync_loss_threshold_db=float(p.get("sync_loss_threshold_db", 30.0)),
                sync_loss_amount=float(p.get("sync_loss_amount", 1.0)),
                adjacent_channel=float(p.get("adjacent_channel", 0.0)),
                broadcast_noise=float(p.get("broadcast_noise", 0.0)),
                channel_switch_rate=float(p.get("channel_switch_rate", 0.0)),
                channel_switch_dur_s=float(p.get("channel_switch_dur_s", 0.28)),
                lo_sweep_mhz=float(p.get("lo_sweep_mhz", 0.0)),
                vsync_roll=float(p.get("vsync_roll", 0.0)),
                vbi_bar_vis=float(p.get("vbi_bar_vis", 0.0)),
                flagging=float(p.get("flagging", 0.0)),
                hanover_bars=float(p.get("hanover_bars", 0.0)),
            ), seed=self.seed)
        return self._rflink

    def _decompose_for_rf(self, comp: np.ndarray):
        from ..core.signal import demod_chroma, split_composite
        window = float(np.clip(
            float(self.ctx.layer("format").get("chroma_bandwidth", 0.75)) * 1.15e6,
            0.35e6, 1.2e6))
        y_band, chroma_band = split_composite(comp, self.std, comb_depth=0.0,
                                              chroma_bandwidth_hz=window)
        i, q = demod_chroma(chroma_band / 0.63, self.std,
                            i_bw_hz=1.5e6, q_bw_hz=1.5e6)
        return y_band, i, q

    def _format_clone(self) -> FormatStage:
        if getattr(self, "_fmt", None) is None:
            self._fmt = FormatStage(self.ctx)
            self._fmt.prepare()
        return self._fmt

    def _clone_buffer(self, buf: FrameBuffer) -> FrameBuffer:
        rgb_c = np.asarray(buf.rgb, dtype=np.float32).copy()
        comp_c = buf.composite.copy() if buf.composite is not None else None
        meta_c = {}
        for k, v in (buf.meta or {}).items():
            if k in ("dropout_mask", "crease_mask"):
                continue
            if isinstance(v, SignalFrame):
                sf = v.copy()
                if getattr(v, "_rf_direct", False):
                    sf._rf_direct = True
                meta_c[k] = sf
            elif isinstance(v, np.ndarray):
                meta_c[k] = v.copy()
            elif isinstance(v, dict):
                meta_c[k] = copy.deepcopy(v)
            else:
                meta_c[k] = v
        return FrameBuffer(rgb_c, comp_c, buf.index, buf.time, buf.std, meta_c)

    def _prepare_stage_input(self, kind: str, buf: FrameBuffer) -> FrameBuffer:
        """Ensure `buf` carries the representation expected by `kind` before a pass."""
        if kind in ("tape", "vhs_gen", "vhs_dub_loop", "vhs_iterative_loop", "transport_servo", "deck", "connector", "power_supply", "video_switcher", "dist_amp", "patch_router") and buf.composite is None:
            sig = buf.meta.get("signal")
            if sig is None:
                sig = SignalFrame.from_rgb(buf.rgb, self.std)
                buf.meta["signal"] = sig
            buf.composite = composite_encode(sig.y, sig.i, sig.q, self.std, chroma_gain=0.63)
        elif kind in ("encoder", "camera") and buf.meta.get("signal") is not None:
            if not buf.meta.get("from_display", False):
                buf.rgb = buf.meta["signal"].to_rgb()
        elif kind == "dv_codec" and buf.meta.get("signal") is None:
            buf.meta["signal"] = SignalFrame.from_rgb(buf.rgb, self.std)
        return buf

    def _picture_from_output(self, kind: str, out: FrameBuffer) -> np.ndarray:
        """Extract an RGB picture from `out` even when a stage only updates `composite` or `signal`."""
        if kind in ("display", "camera", "source", "overlay", "multiview_wall", "test_generator", "ccu_rack", "master_playout", "transfer_station", "linear_editor", "nle_timeline", "automation_controller", "sat_dish_lnb", "sat_ird_receiver", "tv_exciter_pa", "photochemical_film", "film_scanner_telecine", "inversion_analyzer", "archival_vault", "env_chamber", "stream_cdn_client", "human_operator", "ecosystem_simulator", "crt_magnet_env", "offscreen_camera"):
            if kind == "overlay" and out.meta.get("signal") is not None and not out.meta.get("from_display", False):
                return out.meta["signal"].to_rgb()
            return out.rgb
        sig = out.meta.get("signal")
        if sig is not None and kind not in ("tape", "vhs_gen", "vhs_dub_loop", "vhs_iterative_loop", "transport_servo"):
            return sig.to_rgb()
        if out.composite is not None and kind in ("tape", "vhs_gen", "vhs_dub_loop", "vhs_iterative_loop", "transport_servo"):
            y, i, q = self._decompose_for_rf(out.composite)
            sig_c = SignalFrame(y.astype(np.float32), i.astype(np.float32), q.astype(np.float32))
            if sig is not None and getattr(sig, "_rf_direct", False):
                sig_c._rf_direct = True
            out.meta["signal"] = sig_c
            out.rgb = sig_c.to_rgb()
            return out.rgb
        if sig is not None:
            return sig.to_rgb()
        return out.rgb

    def _run_stage_passes(self, kind: str, pass_stages: list[Stage],
                          buf: FrameBuffer, index: int,
                          auto_pre_tape_done: list[bool]) -> tuple[FrameBuffer, np.ndarray]:
        """Run 1..N iteration passes of a stage list on `buf`."""
        # If no explicit `overlay` node exists in the graph and the user enabled `字幕机` with
        # `at == 1` (pre-tape), apply `_auto_osd_pre` right before the first tape stage runs.
        if (
            not auto_pre_tape_done[0]
            and kind in ("tape", "vhs_gen", "vhs_dub_loop", "vhs_iterative_loop", "transport_servo")
            and not self._has_explicit_overlay_node()
            and self._auto_osd_pre is not None
            and self._auto_osd_pre.enabled
            and self._auto_osd_pre.text
            and self._auto_osd_pre.at == 1
        ):
            buf = self._auto_osd_pre.apply(buf)
            if getattr(self._auto_osd_pre, "report", None):
                self.osd_report = dict(self._auto_osd_pre.report)
            auto_pre_tape_done[0] = True

        out = buf
        for st in pass_stages:
            out = self._prepare_stage_input(kind, out)
            if kind == "camera" and hasattr(st, "set_previous"):
                if self._prev_src is None and index > 0:
                    try:
                        self._prev_src = np.asarray(self.source.frame(index - 1), dtype=np.float32)
                    except Exception:
                        self._prev_src = None
                st.set_previous(self._prev_src)
            if kind != "display":
                out.meta.pop("from_display", None)
            out = st.apply(out)
            if kind == "display" and getattr(st, "enabled", False):
                out.meta["from_display"] = True
            if kind == "overlay" and getattr(st, "report", None):
                self.osd_report = dict(st.report)
            picture = self._picture_from_output(kind, out)
            out.rgb = np.asarray(picture, dtype=np.float32)
        picture = self._picture_from_output(kind, out)
        out.rgb = np.asarray(picture, dtype=np.float32)
        return out, out.rgb

    def _run_converter_kind(self, kind: str, buf: FrameBuffer, index: int) -> tuple[FrameBuffer, np.ndarray]:
        """Execute special converter nodes (`hdmi_out`, `hdmi_to_av`, `rf_modulator`, `rf_tuner`)."""
        if kind == "hdmi_out":
            sig = buf.meta.get("signal")
            if sig is None:
                sig = SignalFrame.from_rgb(buf.rgb, self.std)
            out = FrameBuffer(buf.rgb, None, index, buf.time, self.std,
                              {**buf.meta, "signal": sig, "format": "hdmi"})
            return out, sig.to_rgb()

        if kind == "hdmi_to_av":
            sig = buf.meta.get("signal")
            if sig is None:
                sig = SignalFrame.from_rgb(buf.rgb, self.std)
            fmt = self._format_clone()
            comp = fmt.apply(FrameBuffer(sig.to_rgb(), None, index, buf.time,
                                         self.std, dict(buf.meta)))
            pic = comp.meta["signal"].to_rgb()
            comp.rgb = pic
            return comp, pic

        if kind == "rf_modulator":
            sig = buf.meta.get("signal")
            if sig is None:
                sig = SignalFrame.from_rgb(buf.rgb, self.std)
            comp = buf.composite
            if comp is None:
                comp = composite_encode(sig.y, sig.i, sig.q, self.std,
                                        chroma_gain=0.63)
            tx = self._rf().modulate(comp, index)
            out = FrameBuffer(sig.to_rgb(), tx, index, buf.time, self.std,
                              {**buf.meta, "format": "rf"})
            out.meta["signal"] = sig
            return out, sig.to_rgb()

        if kind == "rf_tuner":
            rf = self._rf()
            comp = buf.composite
            if comp is None:
                sig0 = buf.meta.get("signal") or SignalFrame.from_rgb(buf.rgb, self.std)
                comp = composite_encode(sig0.y, sig0.i, sig0.q, self.std,
                                        chroma_gain=0.63)
            base, info = rf.demodulate(
                comp, index,
                np.random.default_rng((int(self.seed) * 8191 + index * 131) & 0xFFFFFFFF))
            y, i, q = self._decompose_for_rf(base)
            if getattr(rf.p, "color_killer_threshold_db", 0.0) > 0.0 and info.get("cnr_db", 99.0) < rf.p.color_killer_threshold_db:
                ck = float(np.clip((rf.p.color_killer_threshold_db - info["cnr_db"]) / 4.0, 0.0, 1.0))
                i = i * (1.0 - ck)
                q = q * (1.0 - ck)
                info["color_killer"] = round(ck, 3)
            ph = rf.chroma_phase_error()
            if ph is not None:
                ph = np.asarray(ph, dtype=np.float32).reshape(-1, 1)
                c, s = np.cos(ph), np.sin(ph)
                i, q = i * c - q * s, i * s + q * c
            hb = rf.hanover_bar_phase() if hasattr(rf, "hanover_bar_phase") else None
            if hb is not None and str(getattr(self.std, "color_system", "PAL")).upper() == "PAL":
                hb_arr = np.asarray(hb, dtype=np.float32).reshape(-1, 1)
                alt = np.where((np.arange(i.shape[0]) % 2 == 0)[:, None], 1.0, -1.0).astype(np.float32)
                theta_h = hb_arr * alt
                ch, sh = np.cos(theta_h), np.sin(theta_h)
                i, q = i * ch - q * sh, i * sh + q * ch
                sat_boost = 1.0 + 0.45 * np.minimum(1.0, np.abs(hb_arr))
                i, q = i * sat_boost, q * sat_boost
            sig_rf = SignalFrame(y.astype(np.float32),
                                 i.astype(np.float32),
                                 q.astype(np.float32))
            conn_cfg = self.ctx.layer("connector") or {}
            sq = float(conn_cfg.get("aspect_squeeze", 0.0))
            if sq > 1e-4:
                from ..codecs.connectors import apply_aspect_squeeze
                rgb_sq, sq_info = apply_aspect_squeeze(sig_rf.to_rgb(), sq)
                sig_rf = SignalFrame.from_rgb(rgb_sq, self.std)
                info.update(sq_info)
                sig_rf._rf_direct = True
            rf_m = float(conn_cfg.get("rf_moisture", 0.0))
            rf_c = float(conn_cfg.get("rf_corrosion", 0.0))
            rf_sh = float(conn_cfg.get("rf_shielding", 0.0))
            meta_out = {**buf.meta, "format": "composite"}
            if rf_m > 1e-4 or rf_c > 1e-4:
                from ..codecs.connectors import CONNECTORS, degrade_rf_signal
                rgb_damp, c_info = degrade_rf_signal(
                    sig_rf.to_rgb(), CONNECTORS["rf"], rf_m, rf_c, rf_sh, index, int(self.seed) + 23)
                sig_rf = SignalFrame.from_rgb(rgb_damp, self.std)
                info.update(c_info)
                sig_rf._rf_direct = True
            pic = sig_rf.to_rgb()
            out = FrameBuffer(pic, base, index, buf.time, self.std, meta_out)
            out.meta["signal"] = sig_rf
            out.meta["rf"] = info
            return out, pic

        raise RuntimeError(f"no implementation for node kind {kind!r}")

    def _execute_node_group(self, group_name: str, node: Node,
                            in_buffers: dict[str, FrameBuffer],
                            index: int, auto_pre_tape_done: list[bool]) -> tuple[FrameBuffer, np.ndarray]:
        """Execute a Blender-style `node_group`'s internal subgraph across `1..node.iterations`."""
        sub = node.subgraph or {}
        sub_nodes = sub.get("nodes") or []
        sub_edges = sub.get("edges") or []
        sub_inputs = sub.get("inputs") or []
        sub_outputs = sub.get("outputs") or []

        default_in_buf = next(iter(in_buffers.values()))
        if not node.enabled or not sub_nodes:
            pic = self._picture_from_output("camera", default_in_buf)
            return default_in_buf, pic

        # Build topological order of the internal subgraph
        s_names = [str(nd["name"]) for nd in sub_nodes]
        s_by_name = {str(nd["name"]): nd for nd in sub_nodes}
        indeg = {n: 0 for n in s_names}
        for ed in sub_edges:
            if ed.get("src") in indeg and ed.get("dst") in indeg:
                indeg[ed["dst"]] += 1
        q = [n for n in s_names if indeg[n] == 0]
        topo: list[str] = []
        while q:
            cur = q.pop(0)
            topo.append(cur)
            for ed in sub_edges:
                if ed.get("src") == cur and ed.get("dst") in indeg:
                    indeg[ed["dst"]] -= 1
                    if indeg[ed["dst"]] == 0:
                        q.append(ed["dst"])
        for n in s_names:
            if n not in topo:
                topo.append(n)

        g_iters = max(1, min(25, int(getattr(node, "iterations", 1) or 1)))
        g_stage_map = self.group_stages.get(group_name, {})
        cur_buf = self._clone_buffer(default_in_buf)
        cur_pic = cur_buf.rgb

        # Map target internal nodes from `sub_inputs`
        entry_targets: dict[str, str] = {}
        for pin in sub_inputs:
            t_node = str(pin.get("target_node", ""))
            p_name = str(pin.get("name", "in"))
            if t_node:
                entry_targets[t_node] = p_name

        for g_it in range(g_iters):
            local_bufs: dict[str, FrameBuffer] = {}
            for s_name in topo:
                snd = s_by_name[s_name]
                s_kind = str(snd.get("kind", ""))
                s_enabled = bool(snd.get("enabled", True))

                # Determine input buffer for `s_name`
                inc_edges = [ed for ed in sub_edges if ed.get("dst") == s_name and ed.get("src") in local_bufs]
                if inc_edges:
                    s_buf = self._clone_buffer(local_bufs[inc_edges[0]["src"]])
                elif s_name in entry_targets and g_it == 0:
                    port_key = entry_targets[s_name]
                    s_buf = self._clone_buffer(in_buffers.get(port_key, cur_buf))
                else:
                    s_buf = self._clone_buffer(cur_buf)

                if not s_enabled:
                    local_bufs[s_name] = s_buf
                    cur_buf = s_buf
                    cur_pic = s_buf.rgb
                    continue

                pass_stages = g_stage_map.get((g_it, s_name))
                if pass_stages:
                    s_out, s_pic = self._run_stage_passes(s_kind, pass_stages, s_buf, index, auto_pre_tape_done)
                else:
                    s_out, s_pic = self._run_converter_kind(s_kind, s_buf, index)
                local_bufs[s_name] = s_out
                cur_buf = s_out
                cur_pic = s_pic

            if sub_outputs:
                out_src = str(sub_outputs[0].get("source_node", ""))
                if out_src in local_bufs:
                    cur_buf = local_bufs[out_src]
                    cur_pic = cur_buf.rgb

        return cur_buf, cur_pic

    def _frame_for(self, name: str, index: int, src_rgb: np.ndarray | None,
                   auto_pre_tape_done: list[bool]) -> tuple[FrameBuffer, np.ndarray]:
        """Produce this node's output `(FrameBuffer, picture)` for one frame."""
        node = self.graph.nodes[name]
        kind = node.kind
        ins = self.graph.inputs_of(name)

        in_buffers: dict[str, FrameBuffer] = {}
        buf: FrameBuffer | None = None
        if ins:
            for port_name, edge in ins.items():
                upstream = edge.src
                cached_sig = self._cache.get(upstream, {}).get(index)
                b = self._buffer_from_cache(upstream, index, cached_sig, src_rgb)
                if b is None:
                    raise RuntimeError(f"{name}: upstream {upstream} has no frame {index}")
                in_buffers[port_name] = b
            buf = in_buffers.get("in_a") or in_buffers.get("in") or next(iter(in_buffers.values()))
            if "in_b" in in_buffers and buf is not None:
                b_buf = in_buffers["in_b"]
                sig_b = b_buf.meta.get("signal")
                if sig_b is None:
                    if b_buf.composite is not None:
                        yb, ib, qb = self._decompose_for_rf(b_buf.composite)
                        sig_b = SignalFrame(yb.astype(np.float32), ib.astype(np.float32), qb.astype(np.float32))
                    else:
                        sig_b = SignalFrame.from_rgb(np.clip(b_buf.rgb, 0.0, 1.0), self.std)
                buf = self._clone_buffer(buf)
                buf.meta["bus_b_yiq"] = (sig_b.y, sig_b.i, sig_b.q)

        if kind == "source":
            rgb = src_rgb if src_rgb is not None else self.source.frame(index)
            out = FrameBuffer(np.asarray(rgb, dtype=np.float32), None, index,
                              index / max(1e-6, self.fps), self.std,
                              {"field": index % 2})
            return out, out.rgb

        if kind == "test_generator" and buf is None:
            blank = np.zeros((int(self.height), int(self.width), 3), dtype=np.float32)
            buf = FrameBuffer(blank, None, index, index / max(1e-6, self.fps),
                              self.std, {"field": index % 2})

        if kind == "multiview_wall" and buf is not None:
            buf = self._clone_buffer(buf)
            mv_feeds = []
            port_defs = [
                ("in_a", "CH1 CAM/SRC", "PVW"),
                ("in_b", "CH2 TEST/AUX", "AUX"),
                ("in_c", "CH3 VTR-RAW", "PVW"),
                ("in_d", "CH4 PGM-AIR", "PGM"),
            ]
            for idx_p, (pname, default_lbl, tally_role) in enumerate(port_defs):
                if pname in in_buffers:
                    b_in = in_buffers[pname]
                    ed = ins.get(pname)
                    up_name = ed.src.upper() if ed else default_lbl
                    src_port = ed.src_port if ed else "out"
                    pbuses = b_in.meta.get("playout_buses") if isinstance(b_in.meta, dict) else None
                    if pbuses and src_port == "clean" and "clean_feed_rgb" in pbuses:
                        b_rgb = np.clip(pbuses["clean_feed_rgb"], 0.0, 1.0)
                        lbl = f"CH{idx_p+1} CLEAN-FD"
                    elif pbuses and src_port == "pvw" and "pvw_rgb" in pbuses:
                        b_rgb = np.clip(pbuses["pvw_rgb"], 0.0, 1.0)
                        lbl = f"CH{idx_p+1} PVW-NEXT"
                    else:
                        lbl = f"CH{idx_p+1} {up_name[:9]}"
                        b_sig = b_in.meta.get("signal")
                        b_rgb = np.clip(b_in.rgb if b_in.rgb is not None else (b_sig.to_rgb() if b_sig is not None else buf.rgb), 0.0, 1.0)
                    is_flt = (b_in.meta.get("format") == "no_signal")
                    mv_feeds.append({"label": lbl, "rgb": b_rgb, "tally": tally_role, "faulted": is_flt})
            if mv_feeds:
                while len(mv_feeds) < 4:
                    mv_feeds.append(dict(mv_feeds[-1], label=f"CH{len(mv_feeds)+1} MIRROR", tally="AUX"))
                buf.meta["mv_feeds"] = mv_feeds

        if buf is None:
            raise RuntimeError(f"节点 {name} 没有输入，无法渲染")

        if not node.enabled:
            picture = self._picture_from_output(kind, buf)
            buf.rgb = np.asarray(picture, dtype=np.float32)
            return buf, buf.rgb

        if kind == "node_group":
            return self._execute_node_group(name, node, in_buffers, index, auto_pre_tape_done)

        pass_stages = self.stages.get(name)
        if pass_stages is not None:
            return self._run_stage_passes(kind, pass_stages, buf, index, auto_pre_tape_done)

        return self._run_converter_kind(kind, buf, index)

    def _buffer_from_cache(self, upstream: str, index: int,
                           cached: np.ndarray | None,
                           src_rgb: np.ndarray | None) -> FrameBuffer | None:
        buf_hit = self._buf_cache.get(upstream, {}).get(index)
        if buf_hit is not None:
            return self._clone_buffer(buf_hit)
        if cached is None:
            return None
        up = self.graph.nodes[upstream]
        buf = FrameBuffer(np.asarray(cached, dtype=np.float32).copy(), None, index,
                          index / max(1e-6, self.fps), self.std,
                          {"field": index % 2})
        if up.kind in ("encoder", "test_generator", "hdmi_to_av", "rf_tuner", "tape", "vhs_gen", "vhs_dub_loop", "deck", "power_supply", "connector", "dist_amp", "patch_router", "video_switcher"):
            sig = SignalFrame.from_rgb(buf.rgb, self.std)
            if up.kind == "rf_tuner":
                _cc = self.ctx.layer("connector") or {}
                if float(_cc.get("rf_moisture", 0.0)) > 1e-4 or float(_cc.get("rf_corrosion", 0.0)) > 1e-4:
                    sig._rf_direct = True
            buf.composite = composite_encode(sig.y, sig.i, sig.q, self.std,
                                             chroma_gain=0.63)
            buf.meta["signal"] = sig
        else:
            buf.meta["signal"] = SignalFrame.from_rgb(buf.rgb, self.std)
        return buf

    def invalidate(self) -> list[str]:
        old = self.hashes
        self.hashes = self.graph.node_hashes(base_params=self.base_params)
        dirty = [n for n in self.graph.nodes if old.get(n) != self.hashes[n]]
        if dirty:
            for n in dirty:
                self._cache.get(n, {}).clear()
                self._buf_cache.get(n, {}).clear()
                self._cache_key.get(n, {}).clear()
            self._build_node_stages(only_nodes=set(dirty))
        return dirty

    def node_is_cached(self, name: str, index: int) -> bool:
        return (index in self._cache.get(name, {})
                and self._cache_key.get(name, {}).get(index) == self.hashes.get(name))

    # ------------------------------------------------------------------
    def pending_nodes(self) -> list[str]:
        out: list[str] = []
        for name, node in self.graph.nodes.items():
            if not node.effective_inputs:
                continue
            if not self.graph.inputs_of(name):
                out.append(name)
        return out

    def render_frame(self, index: int, use_cache: bool = True) -> GraphRun:
        run = GraphRun(rgb=np.zeros((self.height, self.width, 3), np.float32))

        pending = self.pending_nodes()
        if pending:
            names = "、".join(pending)
            run.errors.append(
                f"以下节点没有接入输入，无法渲染：{names}。"
                f"请在画布上把它们连起来，或删除它们。")
            return run

        src_rgb = None
        auto_pre_tape_done = [False]
        for name in self.order:
            self._cache.setdefault(name, {})
            self._buf_cache.setdefault(name, {})
            self._cache_key.setdefault(name, {})
            if use_cache and self.node_is_cached(name, index):
                run.node_frames[name] = self._cache[name][index]
                run.trace.append(f"{name}(cached)")
                self.skipped[name] = self.skipped.get(name, 0) + 1
                continue
            import time
            t0 = time.perf_counter()
            try:
                buf, picture = self._frame_for(name, index, src_rgb, auto_pre_tape_done)
            except Exception as exc:
                run.errors.append(f"{name}: {type(exc).__name__}: {exc}")
                return run
            if self.graph.nodes[name].kind == "source":
                self._prev_src = np.asarray(picture, dtype=np.float32)
            self._cache[name][index] = picture
            self._buf_cache[name][index] = self._clone_buffer(buf)
            self._cache_key[name][index] = self.hashes[name]
            while len(self._cache[name]) > self.max_cache_frames:
                old_k = next(iter(self._cache[name]))
                self._cache[name].pop(old_k, None)
                self._buf_cache[name].pop(old_k, None)
                self._cache_key[name].pop(old_k, None)
            run.node_frames[name] = picture
            iters = max(1, int(getattr(self.graph.nodes[name], "iterations", 1) or 1))
            run.trace.append(f"{name}×{iters}" if iters > 1 else name)
            self.node_ms[name] = self.node_ms.get(name, 0.0) * 0.7 + \
                (time.perf_counter() - t0) * 1000.0 * 0.3

        outs = self.graph.output_nodes()
        if outs:
            sink = outs[0]
            final_rgb = self._cache[sink][index]
            # If no explicit `overlay` node exists in the graph and the user enabled
            # `字幕机` in the left panel (`at == 0`, post-chain OSD), draw it on the final frame.
            if (
                not self._has_explicit_overlay_node()
                and self._auto_osd_tail is not None
                and self._auto_osd_tail.enabled
                and self._auto_osd_tail.text
                and self._auto_osd_tail.at != 1
            ):
                tail_buf = FrameBuffer(
                    np.asarray(final_rgb, dtype=np.float32).copy(), None,
                    index, index / max(1e-6, self.fps), self.std,
                    {"field": index % 2, "from_display": True},
                )
                tail_out = self._auto_osd_tail.apply(tail_buf)
                if getattr(self._auto_osd_tail, "report", None):
                    self.osd_report = dict(self._auto_osd_tail.report)
                final_rgb = tail_out.rgb
                self._cache[sink][index] = final_rgb
                run.node_frames[sink] = final_rgb
            run.rgb = final_rgb
        return run

    def render_frames(self, indices: list[int]
                      ) -> Iterator[tuple[int, np.ndarray]]:
        wanted = sorted({int(i) for i in indices})
        if not wanted:
            return
        outs = self.graph.output_nodes()
        if not outs:
            return
        sink = outs[0]
        for index in range(0, max(wanted) + 1):
            if self.node_is_cached(sink, index):
                if index in wanted:
                    yield index, self._cache[sink][index]
                continue
            run = self.render_frame(index)
            if run.errors:
                raise RuntimeError("; ".join(run.errors))
            if index in wanted:
                yield index, run.rgb

    def describe(self) -> dict:
        d = self.graph.describe(base_params=self.base_params)
        d["node_ms"] = {k: round(v, 2) for k, v in self.node_ms.items()}
        d["cached_frames"] = {k: len(v) for k, v in self._cache.items()}
        return d
