"""
The WebUI server.

Deliberately built on the standard library's HTTP server: this project's value is
in the signal model, and a framework dependency would buy nothing here while
adding install friction.  Long work (renders) runs on worker threads and reports
progress through a job table the page polls, so the HTTP loop never blocks.

Endpoints
    GET  /                    the app
    GET  /api/catalog         presets, device profiles, layer labels
    GET  /api/schema          the full parameter schema with units/provenance
    GET  /api/project         current project
    GET  /api/export          the whole configuration as a downloadable JSON file
    POST /api/import          apply an exported configuration (resets first, so the file is complete)
    POST /api/project         update project (params / seed / preset / source)
    POST /api/preset          apply a factory preset or a device profile
    POST /api/upload          import a clip
    GET  /api/frame           one rendered still (PNG or JPEG)  ?t=&scale=&mode=
    GET  /api/timeline        the generated defect timeline
    GET  /api/render          start a full-resolution render job
    GET  /api/job             job status / progress
    GET  /api/download        the finished render
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..core.params import merge_values
from ..core.profiles import SCHEMA
from ..io.ffmpeg import VideoWriter, have_ffmpeg, jpeg_bytes
from ..render.renderer import Renderer, STAGE_ORDER
from .session import Project, SessionStore

WEB_DIR = Path(__file__).parent / "static"


def _png(rgb: np.ndarray) -> bytes:
    """
    Encode a frame as PNG.

    Uses the built-in codec rather than Pillow.  Pillow was the only third-party
    dependency in the whole path, and on a machine that lacks it the program
    started, accepted a clip, and then failed at the first preview frame -- the
    worst place for a missing library to surface.
    """
    from ..io.imagecodec import png_encode
    return png_encode(rgb)


def _jpeg(rgb: np.ndarray, quality: int = 88) -> bytes:
    """
    Encode a frame as JPEG through FFmpeg.

    Worth knowing the cost: this spawns one FFmpeg per call, measured at ~50 ms
    for a 360x240 frame, against ~6 ms for the built-in PNG encoder.  So JPEG is
    the right choice for a *file the user keeps* and the wrong choice for a
    preview frame crossing localhost -- see `_auto` below, which picks per use.
    """
    from ..io.imagecodec import jpeg_encode
    return jpeg_encode(rgb, quality=quality)


def _auto(rgb: np.ndarray, prefer_jpeg: bool, quality: int = 88) -> tuple[bytes, str]:
    """
    Pick an encoder and report which was used.

    PNG is the default for interactive frames: the bytes travel over loopback, so
    the larger file costs nothing measurable, while avoiding a per-frame process
    spawn removes roughly half the preview's latency.  JPEG is kept for exports,
    where the file is kept and size matters more than milliseconds.
    """
    if prefer_jpeg:
        return _jpeg(rgb, quality), "image/jpeg"
    return _png(rgb), "image/png"


def _neutral_params(params: dict) -> dict:
    """Every device neutralised, for the canvas A/B reference."""
    from ..core.params import merge_values
    return merge_values(params, {
        "camera": {"sensor_noise": 0.0, "fixed_pattern": 0.0, "chroma_aberration": 0.0,
                   "motion_blur": 0.0, "sharpness": 1.0, "exposure": 0.0,
                   "white_balance_temp": 0.0, "ccd_smear": 0.0, "halation": 0.0,
                   "ccd_gain": 0.0, "color_cast": 0.0, "highlight_tint": 0.0,
                   "lens_wear": 0.0, "lens_flare": 0.0, "sensor_type": 0.0},
        "film": {"enabled": 0.0, "stock": 0.0},
        "format": {"edge_boost": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                   "color_noise": 0.0, "burst_phase_error": 0.0,
                   "luma_bandwidth": 5.0, "chroma_bandwidth": 1.5,
                   "chroma_vertical_averaging": 0.0},
        "codec": {"dv_enabled": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0,
                 "micro_jitter": 0.0, "vertical_instability": 0.0, "head_switching": 0.0,
                 "dropout_rate": 0.0, "wear": 0.0, "crease_rate": 0.0,
                 "noise_floor": 0.0, "head_clog": 0.0, "debris": 0.0,
                 "surface_damage": 0.0, "chroma_loss": 0.0,
                 "transport_state": 0.0, "reel_progress": 0.50, "tension_ripple": 0.0,
                 "belt_slip": 0.0, "roller_eccentricity": 0.0, "servo_hunt": 0.0},
        "connector": {"psu_ripple": 0.0, "battery_level": 1.0, "battery_sag": 0.0,
                      "load_coupling": 0.0, "startup_surge": 0.0, "emi_crosstalk": 0.0, "switcher_mode": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "tbc_residual": 0.0,
                   "transport_jitter": 0.0, "head_wear": 0.0, "frequency_response": 0.0,
                   "line_delay": 0.0, "output_gain": 0.0, "output_bias": 0.0,
                   "cross_color": 0.0, "cross_luma": 0.0,
                   "sync_clip": 0.0, "h_hold": 0.0, "v_hold": 0.0,
                   "clock_drift_ppm": 0.0, "clock_jitter": 0.0,
                   "std_mismatch": 0.0, "pulse_cross": 0.0},
        "display": {"enabled": 0.0, "reshoot_enabled": 0.0, "reshoot_moire": 0.0, "reshoot_tilt_deg": 0.0, "reshoot_perspective": 0.0, "reshoot_ambient_glare": 0.0, "reshoot_wb_shift": 0.0},
    })


def _neutral_checked(neutral: dict) -> dict:
    """
    Fail loudly if the A/B reference names a parameter that does not exist.

    A hand-written neutral set is exactly the kind of code that rots: parameters
    get added, the dict is not updated, and because unknown keys are ignored the
    comparison silently stops being a comparison.  That had already happened here
    -- five keys named parameters that do not exist, and several newer ones were
    missing entirely.  Checking at construction turns it into an immediate error.
    """
    from ..core.profiles import SCHEMA
    known = {lay: set(SCHEMA.layer(lay).keys()) for lay in SCHEMA.layer_names()}
    for layer, values in neutral.items():
        if layer not in known:
            raise KeyError(f"neutral reference names unknown layer {layer!r}")
        unknown = sorted(k for k in values if k not in known[layer])
        if unknown:
            raise KeyError(
                f"neutral reference names unknown parameter(s) in {layer!r}: {unknown}")
    return neutral


# --- node position units ---------------------------------------------------
# The canvas stores node positions in *pixels*.  Presets are authored in grid columns because
# that is readable in source, so the conversion happens once, here, as a preset is applied.
#
# There used to be three units in circulation at the same time -- preset columns, dragged
# pixels, and a newly added node's pixels sent without the flag that marks them as such -- and
# every resulting bug was an attempt to disambiguate them on the client.  That cannot work
# while the units differ, so the ambiguity is removed instead of being guessed at.


class App:
    """Holds all mutable state; one instance per server."""

    def __init__(self, data_dir: Path, default_clip: str | None = None):
        self.store = SessionStore(data_dir)
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self._clean: dict[tuple, Renderer] = {}
        # what the character generator last drew.  Declared here so the payload can read it before any
        # frame has been rendered, rather than raising on a missing attribute.
        self.osd_report: dict = {}
        self.current = Project(name="新工程", seed=20240101)
        if default_clip:
            p = Path(default_clip)
            if p.exists():
                from ..io.ffmpeg import probe
                self.current.source = str(p)
                self.current.source_info = probe(p).to_dict()
                self.current.fps = self.current.source_info.get("fps") or self.current.fps
        self.store.apply_preset(self.current, "family_archive_1992")
        self.store.save(self.current)
        # the apparatus the user wires up; starts as the classic VHS path, which
        # matches the default preset above
        from ..graph import GRAPH_PRESETS, Graph
        # Normalise here as well as in the preset route: the graph is built from a
        # preset, so it arrives in grid columns, and a server that hands the canvas
        # columns renders every node on top of the next one.
        self.graph: Graph = (GRAPH_PRESETS["vhs_consumer"]["build"]()
                             .normalise_positions())
        self._graph_execs: dict[tuple, object] = {}
        # renderer revision counter: lets the client tell whether a tweak needs
        # a full re-render of the preview or can reuse a cached prefix
        self._revision = 0

    # ------------------------------------------------------------------
    def project_dict(self) -> dict:
        info = self.current.source_info or {}
        return {
            "project": self.current.to_dict(),
            "params": self.current.resolved_params(),
            "source_info": info,
            "standard": self.current.standard,
            # The per-frame state, so the interface can show what is set.  `unit_report` is included
            # because the point of an individual machine is that its deviations can be named -- a
            # realism claim about one unit is only checkable if its numbers can be read out.
            "environment": dict(self.current.environment or {}),
            "keyframes": list(self.current.keyframes or []),
            "unit_identities": dict(self.current.unit_identities or {}),
            "unit_report": self.current.unit_report(),
            # what the character generator last drew, and which characters its font could not draw.
            # Empty until a frame has been rendered, which the interface handles by falling back to its
            # own coverage check.
            "osd_report": dict(getattr(self, "osd_report", {}) or {}),
        }

    def set_project(self, payload: dict) -> dict:
        with self.lock:
            p = self.current
            # A payload may name layers in two ways and both must work.  The nested form is the server's own
            # convention (`set_levels` wraps its argument that way), and the bare form is what a caller
            # reasonably sends when it wants to set a layer.  Measured before this: the bare form left
            # `overlay.enabled` at 0.0 with no error, so a caller had no way to tell it had failed.
            #
            # The bare form is detected by the payload naming any *known layer* at the top level.  A payload
            # that only carries `params`/`seed`/`standard` and so on has none, so the nested path is used.
            _layers = {l for l in self.store.schema.layer_names()} if hasattr(self.store, "schema") else set()
            if not _layers:
                from ..core.profiles import SCHEMA as _S
                _layers = set(_S.layer_names())
            _bare = [k for k in payload if k in _layers]
            if "params" in payload and isinstance(payload["params"], dict):
                merged = dict(payload["params"])
                # a caller that nested twice leaves a layer literally called `params`; unfurl it so the
                # values reach the layer they name
                if isinstance(merged.get("params"), dict):
                    inner = dict(merged.pop("params"))
                    inner.update(merged)
                    merged = inner
                p.params = merge_values(p.params or {}, merged)
            elif _bare:
                p.params = merge_values(p.params or {}, {k: payload[k] for k in _bare})
            # **`and payload["seed"] is not None`, because a key that exists holding None is not a value.**  Measured:
            # `import_config` builds its payload with `"seed": payload.get("seed")`, so **any imported configuration that
            # does not carry a `seed` key of its own reaches this line as None and raises**
            # `TypeError: int() argument must be ... not 'NoneType'` -- and the import fails entirely, with the values
            # never applied.  **The user's report "导入失败 配置里没有 params" was this**, not the missing `params`, which
            # was a second and separate fault in the file itself.
            #
            # The same guard is applied to the other optional fields below: each is set only when the payload actually
            # carries a value, so a complete-looking payload with None-filled gaps cannot overwrite a project with
            # `str(None)` or an empty name.
            if payload.get("seed") is not None:
                p.seed = int(payload["seed"])
            if payload.get("standard") is not None:
                p.standard = str(payload["standard"])
            if "environment" in payload and isinstance(payload["environment"], dict):
                p.environment = dict(payload["environment"])
            if "keyframes" in payload and isinstance(payload["keyframes"], list):
                p.keyframes = list(payload["keyframes"])
            if "unit_identities" in payload and isinstance(payload["unit_identities"], dict):
                # an empty string means the nominal model, and the key is the device class
                p.unit_identities = {str(k): str(v) for k, v in payload["unit_identities"].items()}
            if payload.get("name") is not None:
                p.name = str(payload["name"])
            if "source" in payload:
                p.source = payload["source"]
                if p.source:
                    from ..io.ffmpeg import probe
                    try:
                        info = probe(self.store.clip_path(p.source))
                        p.source_info = info.to_dict()
                        p.fps = info.fps or p.fps
                    except Exception as exc:
                        p.source_info = {"error": str(exc)}
                else:
                    p.source_info = None
            if payload.get("reset_params"):
                p.params = {}
            # Sync modified layer keys to any top-level graph node that already overrides that key
            if getattr(self, "graph", None) is not None:
                touched_layers = {}
                if "params" in payload and isinstance(payload["params"], dict):
                    touched_layers = payload["params"]
                elif _bare:
                    touched_layers = {k: payload[k] for k in _bare if isinstance(payload[k], dict)}
                from ..graph import CATALOG_BY_KIND
                for node in self.graph.nodes.values():
                    lay = node.spec.params_layer
                    if lay and lay in touched_layers and isinstance(touched_layers[lay], dict) and node.params:
                        for k, v in touched_layers[lay].items():
                            if k in node.params:
                                node.params[k] = v
                    if node.kind == "node_group" and isinstance(node.subgraph, dict):
                        for sub in (node.subgraph.get("nodes") or []):
                            if not isinstance(sub, dict) or not isinstance(sub.get("params"), dict):
                                continue
                            sub_spec = CATALOG_BY_KIND.get(str(sub.get("kind") or ""))
                            sub_lay = sub_spec.params_layer if sub_spec else None
                            if sub_lay and sub_lay in touched_layers and isinstance(touched_layers[sub_lay], dict):
                                for k, v in touched_layers[sub_lay].items():
                                    if k in sub["params"]:
                                        sub["params"][k] = v
            structural_change = (
                "source" in payload
                or payload.get("seed") is not None
                or payload.get("standard") is not None
                or bool(payload.get("reset_params"))
            )
            if structural_change:
                self._graph_execs.clear()
                self.store.drop(p.id)
            self.store.save(p)
            self._revision += 1
        return self.project_dict()

    # ------------------------------------------------------------------ export / import
    def export_config(self) -> dict:
        """
        The whole configuration as one JSON-serialisable object, **as a diff against the schema defaults**.

        A full dump of every layer is 400-odd numbers and says nothing about which of them a reproduction depended
        on. The diff says exactly that, and it is small enough to read.

        Everything needed to reproduce a frame is included: the changed parameters, the seed, the standard, the
        environment, the keyframe tracks, the unit identities, and what the source was. A version and a timestamp go
        with it, so an old file can be recognised instead of being applied half-way.
        """
        from ..core.profiles import SCHEMA
        import datetime as _dt

        with self.lock:
            p = self.current
            defaults = SCHEMA.resolve_all({})
            changed: dict[str, dict[str, object]] = {}
            for layer, values in (p.params or {}).items():
                base = defaults.get(layer) or {}
                diff = {}
                for k, v in values.items():
                    d = base.get(k)
                    # a NaN default is how the schema writes "off", and a NaN is not equal to itself
                    if isinstance(d, float) and isinstance(v, float) and d != d and v != v:
                        continue
                    if isinstance(d, float) and isinstance(v, float):
                        if abs(d - v) <= 1e-9:
                            continue
                    elif d == v:
                        continue
                    diff[k] = v
                if diff:
                    changed[layer] = diff
            return {
                "kind": "atsim.config",
                "version": 1,
                "exported": _dt.datetime.now().isoformat(timespec="seconds"),
                "name": p.name,
                "standard": p.standard,
                "seed": p.seed,
                "fps": p.fps,
                "environment": dict(p.environment or {}),
                "keyframes": list(p.keyframes or []),
                "unit_identities": dict(p.unit_identities or {}),
                # recorded but not imported: the path is machine-specific, and it is here so a reader can tell
                # which clip the file was made from
                "source": p.source,
                "source_info": p.source_info,
                "params": changed,
                "param_count": sum(len(v) for v in changed.values()),
            }

    def import_config(self, payload: dict) -> dict:
        """
        Apply an exported configuration.  **The parameters are reset first, so the file means the same thing here as
        it did where it was written.**

        `set_project` already merges correctly, so the only thing an import has to add is determinism. Applying a
        file on top of whatever happens to be loaded would leave every value the file does not mention at its
        current setting, and the same file would then give different results on two machines -- which defeats the
        point of exporting one.
        """
        if not isinstance(payload, dict):
            raise ValueError("配置必须是一个 JSON 对象")
        kind = payload.get("kind")
        if kind not in (None, "atsim.config"):
            raise ValueError("这不是本程序的配置导出文件（kind=%r）" % (kind,))
        version = payload.get("version")
        if isinstance(version, int) and version > 1:
            raise ValueError("配置版本 %d 比本程序支持的版本新" % version)
        params = payload.get("params")
        if not isinstance(params, dict):
            raise ValueError("配置里没有 params")

        # reset first, then merge -- so the file is complete on its own
        self.set_project({"reset_params": True})
        # **Only the keys the file actually carries are forwarded.**  Passing `payload.get(k)` for every optional field
        # puts `None` under each key the file omits, and `set_project` then has to defend against a value that is not
        # there -- which is how a configuration without a `seed` failed to import at all.  **A key that is absent should
        # be absent**, so the forwarding is built from what is present rather than from a fixed list.
        forward = {"params": params}
        for opt in ("seed", "standard", "environment", "keyframes", "unit_identities", "name"):
            if payload.get(opt) is not None:
                forward[opt] = payload[opt]
        applied = self.set_project(forward)
        applied["imported"] = {
            "from": payload.get("exported"),
            "version": version,
            "params_applied": sum(len(v) for v in params.values() if isinstance(v, dict)),
        }
        return applied

    def apply_preset(self, key: str) -> dict:
        with self.lock:
            self.store.apply_preset(self.current, key)
            custom_spec = self.store.custom_presets.get(key)
            if custom_spec and isinstance(custom_spec.get("graph"), dict):
                try:
                    from ..graph import Graph
                    self.graph = Graph.from_dict(custom_spec["graph"])
                except Exception:
                    pass
            elif getattr(self, "graph", None) is not None:
                preset_layers = self.store.preset_layers(key) or {}
                for node in self.graph.nodes.values():
                    lay = node.spec.params_layer
                    if lay and (lay in preset_layers or not preset_layers):
                        node.params.clear()
                    if node.kind == "display" and "display" in (self.current.params or {}):
                        node.enabled = float(self.current.params["display"].get("enabled", 0.0)) > 0.5
            self._graph_execs.clear()
            self.store.drop(self.current.id)
            self.store.save(self.current)
            self._revision += 1
        out = self.project_dict()
        out["graph_state"] = self.graph_state()
        return out

    def save_custom_preset(self, payload: dict) -> dict:
        label = str(payload.get("label") or payload.get("name") or "").strip()
        note = str(payload.get("note") or "").strip()
        key = payload.get("key")
        with self.lock:
            entry = self.store.save_custom_preset(
                self.current, label=label, note=note, key=key,
                graph_dict=self.graph.to_dict() if getattr(self, "graph", None) else None,
            )
            self.current.preset = entry["key"]
            self.store.save(self.current)
        return {
            "ok": True,
            "preset": entry,
            "catalog": self.store.catalog(),
            "project": self.project_dict(),
        }

    def delete_custom_preset(self, payload: dict) -> dict:
        key = str(payload.get("key") or "")
        with self.lock:
            removed = self.store.delete_custom_preset(key)
            if self.current.preset == key:
                self.current.preset = ""
                self.store.save(self.current)
        return {
            "ok": removed,
            "catalog": self.store.catalog(),
            "project": self.project_dict(),
        }

    def set_levels(self, layer: str, values: dict) -> dict:
        return self.set_project({"params": {layer: values}})

    # ------------------------------------------------------------------
    def renderer(self, preview: bool, scale: float):
        with self.lock:
            return self.store.renderer_for(self.current, preview=preview, scale=scale)

    def frame(self, t: float, scale: float, quality: int,
              clean: bool = False) -> bytes:
        r, src = self.renderer(True, scale)
        idx = int(round(t * self.current.fps))
        idx = max(0, min(idx, max(0, getattr(src, "frame_count", 1) - 1)))
        if clean:
            # A/B reference: the same frame with every stage neutralised, so the
            # comparison shows what the chain actually did rather than comparing
            # against a differently-scaled decode.
            r = self._clean_renderer(scale)
        rgb = r.render_frame(idx)
        # the character generator fills its report while drawing, so this is the moment it is known.
        # Captured here rather than in the stage so that every path which renders a preview updates it
        # and the interface has one place to read from.
        try:
            st = (r.stage or {}).get("overlay")
            if st is not None and getattr(st, "report", None):
                self.osd_report = dict(st.report)
        except Exception:                                             # noqa: BLE001
            pass
        return _jpeg(rgb, quality) if quality < 100 else _png(rgb)


    def _has_active_graph_node(self, kind: str) -> bool:
        """Check whether an enabled node of `kind` exists in `self.graph` (including inside `node_group`s)."""
        if getattr(self, "graph", None) is None:
            return False
        for node in self.graph.nodes.values():
            if not getattr(node, "enabled", True):
                continue
            if node.kind == kind:
                return True
            if node.kind == "node_group" and isinstance(node.subgraph, dict):
                for sub in (node.subgraph.get("nodes") or []):
                    if isinstance(sub, dict) and sub.get("enabled", True) is not False and sub.get("kind") == kind:
                        return True
        return False

    def _detect_quad_lab_and_override(
        self,
        base_params: dict,
        corner: int | None = None,
        border: int | None = None,
        text: int | None = None,
        force_quad_if_none: bool = False,
        use_graph: bool = False,
    ) -> tuple[dict, str, dict[int, tuple[str, str]]]:
        """
        Detect which 2x2 Quad Lab is active in `base_params` (or in `self.graph` when `use_graph=True`,
        or activate Multiview Wall if `force_quad_if_none` is True on a single-screen preset) and return
        `(overridden_params, lab_kind, corner_meta_map)`.
        """
        from ..core.params import merge_values
        p = {k: dict(v) for k, v in (base_params or {}).items()}
        rf = p.get("rf") or {}
        ply = p.get("player") or {}
        cam = p.get("camera") or {}
        dsp = p.get("display") or {}
        cn = p.get("connector") or {}
        flm = p.get("film") or {}
        cod = p.get("codec") or {}
        ov = p.get("overlay") or {}

        g_has = (lambda k: self._has_active_graph_node(k)) if use_graph else (lambda _k: False)

        if float(rf.get("tx_chain_mode", 0.0)) >= 0.5 or g_has("tv_exciter_pa"):
            lab = "tx_chain"
            c_map = {
                0: ("0_quad_2x2", "2×2 四分屏对照"),
                1: ("1_top_left_baseband_ref", "左上1 · 主控直通基带参考"),
                2: ("2_top_right_tx_hpa_probe", "右上2 · 发射机高功放监测"),
                3: ("3_bottom_left_analog_offair", "左下3 · 模拟开路接收终端"),
                4: ("4_bottom_right_digital_dtt", "右下4 · 数字地面DTT接收"),
            }
        elif float(rf.get("sat_ird_mode", 0.0)) >= 0.5 or g_has("sat_ird_receiver"):
            lab = "sat_ird"
            c_map = {
                0: ("0_quad_2x2", "2×2 四分屏对照"),
                1: ("1_top_left_clear_sky_ref", "左上1 · 晴空卫星参考"),
                2: ("2_top_right_analog_fm_sparkles", "右上2 · 模拟C波段FM打火噪点"),
                3: ("3_bottom_left_digital_dvbs_cliff", "左下3 · 数字DVB-S悬崖解码"),
                4: ("4_bottom_right_cn_cliff_curve", "右下4 · 载噪比悬崖响应图"),
            }
        elif float(ply.get("transfer_station_mode", 0.0)) >= 0.5 or g_has("transfer_station"):
            lab = "transfer"
            c_map = {
                0: ("0_quad_2x2", "2×2 四分屏对照"),
                1: ("1_top_left_gen0_master", "左上1 · Gen-0母带基准"),
                2: ("2_top_right_multi_gen_no_tbc", "右上2 · 无TBC多代模拟复制"),
                3: ("3_bottom_left_multi_gen_with_tbc", "左下3 · 带TBC多代模拟复制"),
                4: ("4_bottom_right_digital_transcode", "右下4 · 4:1:1→4:2:0数字转码"),
            }
        elif (float(flm.get("enabled", 0.0)) >= 0.5 or g_has("photochemical_film")) and int(round(float(flm.get("bloom_veil_mode", 0.0)))) == 5:
            lab = "lomo_opt"
            c_map = {
                0: ("0_quad_2x2", "2×2 四分屏对照"),
                1: ("1_top_left_trioplan_bubble_split_veil", "左上1 · 肥皂泡散景与双色温光雾"),
                2: ("2_top_right_golden_hour_solar_bloom", "右上2 · 黄金时刻逆光太阳耀斑洪流"),
                3: ("3_bottom_left_uncoated_pastel_milk_veil", "左下3 · 无镀膜老镜头奶油白雾柔焦"),
                4: ("4_bottom_right_tilt_shift_warm_corona", "右下4 · 移轴微距浅景深与暖金日冕"),
            }
        elif (float(flm.get("enabled", 0.0)) >= 0.5 and int(round(float(flm.get("process_mode", 0.0)))) == 5) or g_has("photochemical_film"):
            lab = "lomo_darkroom"
            c_map = {
                0: ("0_quad_2x2", "2×2 四分屏对照"),
                1: ("1_top_left_e6_in_c41_xpro", "左上1 · E-6正冲负青绿暖金"),
                2: ("2_top_right_c41_in_e6_heart", "右上2 · C-41负冲正玫瑰桃心散景"),
                3: ("3_bottom_left_expired_svema_fog", "左下3 · 过期胶卷高底雾粉彩"),
                4: ("4_bottom_right_polaroid_sx70_leak", "右下4 · 宝丽来浓暗角与暖阳漏光"),
            }
        elif float(flm.get("enabled", 0.0)) >= 0.5 or float(cam.get("film_enabled", 0.0)) >= 0.5 or float(cam.get("telecine_pulldown", 0.0)) >= 0.5 or g_has("film_scanner_telecine"):
            lab = "telecine"
            c_map = {
                0: ("0_quad_2x2", "2×2 四分屏对照"),
                1: ("1_top_left_16mm_film_stock", "左上1 · 16mm胶片原片"),
                2: ("2_top_right_rank_cintel_telecine", "右上2 · 电视飞点胶转磁"),
                3: ("3_bottom_left_home_vhs_dub", "左下3 · 家用VHS电视录屏"),
                4: ("4_bottom_right_web_stream_reencode", "右下4 · 网络流媒体重压"),
            }
        elif float(ply.get("inv_lab_enabled", 0.0)) >= 0.5 or g_has("inversion_analyzer"):
            lab = "inversion"
            c_map = {
                0: ("0_quad_2x2", "2×2 四分屏对照"),
                1: ("1_top_left_degraded_obs", "左上1 · 退化观测画面"),
                2: ("2_top_right_residual_heatmap", "右上2 · 反演残差热力图"),
                3: ("3_bottom_left_hypothesis_ranking", "左下3 · 多解竞争假设排序"),
                4: ("4_bottom_right_fadgi_scorecard", "右下4 · FADGI三级审计评分"),
            }
        elif float(ply.get("archival_enabled", 0.0)) >= 0.5 or float(ply.get("archival_mode", 0.0)) >= 0.5 or g_has("archival_vault"):
            lab = "archival"
            c_map = {
                0: ("0_quad_2x2", "2×2 四分屏对照"),
                1: ("1_top_left_original_carrier", "左上1 · 原始载体残件"),
                2: ("2_top_right_preservation_master", "右上2 · 档案保存母版"),
                3: ("3_bottom_left_overcleaned_dnr", "左下3 · 过度清洗商业修复版"),
                4: ("4_bottom_right_web_lineage_audit", "右下4 · 流媒体版与重要属性损耗"),
            }
        elif float(cn.get("env_field_enabled", 0.0)) >= 0.5 or float(cn.get("env_field_mode", 0.0)) >= 0.5 or g_has("env_chamber"):
            lab = "env_field"
            c_map = {
                0: ("0_quad_2x2", "2×2 四分屏对照"),
                1: ("1_top_left_thermal_dew_map", "左上1 · 温湿度露点场与结露画面"),
                2: ("2_top_right_emi_ground_map", "右上2 · 电磁干扰场与地环路画面"),
                3: ("3_bottom_left_optical_acoustic", "左下3 · 光学与声学空间场耦合"),
                4: ("4_bottom_right_device_telemetry", "右下4 · 设备局域坐标物理遥测"),
            }
        elif float(cod.get("stream_enabled", 0.0)) >= 0.5 and (int(round(float(cod.get("stream_mode", 0.0)))) == 5 or float(cod.get("stream_quad_corner", 0.0)) > 0 or g_has("stream_cdn_client")):
            lab = "streaming"
            c_map = {
                0: ("0_quad_2x2", "2×2 四分屏对照"),
                1: ("1_top_left_ingest_source", "左上1 · 采集源与TBC母本"),
                2: ("2_top_right_realvideo_1999", "右上2 · 1999拨号RealVideo流媒体"),
                3: ("3_bottom_left_web20_reencode", "左下3 · Web2.0平台多代转码"),
                4: ("4_bottom_right_abr_net_telemetry", "右下4 · 现代ABR与网络缓冲遥测"),
            }
        elif float(dsp.get("percept_enabled", 0.0)) >= 0.5 or float(dsp.get("percept_mode", 0.0)) >= 0.5 or g_has("human_operator"):
            lab = "perception"
            c_map = {
                0: ("0_quad_2x2", "2×2 四分屏对照"),
                1: ("1_top_left_raw_hardware", "左上1 · 未干预物理故障输出"),
                2: ("2_top_right_viewer_csf", "右上2 · 观众视网膜CSF感知画面"),
                3: ("3_bottom_left_operator_closed_loop", "左下3 · 机房操作者闭环矫正画面"),
                4: ("4_bottom_right_jnd_operator_panel", "右下4 · JND显著度与操作轨迹面板"),
            }
        elif float(ov.get("eco_enabled", 0.0)) >= 0.5 or float(ov.get("eco_mode", 0.0)) >= 0.5 or g_has("ecosystem_simulator"):
            lab = "ecosystem"
            c_map = {
                0: ("0_quad_2x2", "2×2 四分屏对照"),
                1: ("1_top_left_1992_golden_era", "左上1 · 1992模拟黄金期基准"),
                2: ("2_top_right_selected_year_actual", "右上2 · 所选年份真实生态状态"),
                3: ("3_bottom_left_counterfactual_scenario", "左下3 · 反事实情景推演状态"),
                4: ("4_bottom_right_ecosystem_trajectories", "右下4 · 1975-2035七维演化曲线"),
            }
        elif int(round(float(dsp.get("crt_magnet_mode", 0.0)))) == 5 or g_has("crt_magnet_env"):
            lab = "crt_magnet"
            c_map = {
                0: ("0_quad_2x2", "2×2 四分屏对照"),
                1: ("1_top_left_strong_magnet_faceplate", "左上1 · 强钕铁硼磁铁贴面黑洞与彩虹环"),
                2: ("2_top_right_rotating_vortex", "右上2 · 强磁极旋转洛伦兹涡流"),
                3: ("3_bottom_left_shadow_mask_remanence", "左下3 · 撤磁后铁磁荫罩永久剩磁色斑"),
                4: ("4_bottom_right_ptc_degauss_coil", "右下4 · 内置PTC消磁线圈衰减振荡净化"),
            }
        elif float(dsp.get("mv_enabled", 0.0)) >= 0.5 or g_has("multiview_wall") or force_quad_if_none:
            lab = "multiview"
            c_map = {
                0: ("0_quad_2x2", "2×2 四分屏对照"),
                1: ("1_top_left_ch1_cam_src", "左上1 · CH1摄像机源"),
                2: ("2_top_right_ch2_vtr_raw", "右上2 · CH2磁带直出"),
                3: ("3_bottom_left_ch3_tbc_out", "左下3 · CH3时基校正输出"),
                4: ("4_bottom_right_ch4_pgm_air", "右下4 · CH4主播出PGM"),
            }
        else:
            lab = "single"
            c_map = {
                0: ("still", "单画面静帧"),
                1: ("1_top_left", "左上1"),
                2: ("2_top_right", "右上2"),
                3: ("3_bottom_left", "左下3"),
                4: ("4_bottom_right", "右下4"),
            }

        patch: dict[str, dict[str, float]] = {}
        if lab == "tx_chain":
            patch["rf"] = {"tx_chain_mode": 3.0}
            if corner is not None:
                patch["rf"]["tx_quad_corner"] = float(corner)
            if border is not None:
                patch["rf"]["tx_show_border"] = float(border)
            if text is not None:
                patch["rf"]["tx_show_text"] = float(text)
        elif lab == "sat_ird":
            patch["rf"] = {"sat_ird_mode": 3.0}
            if corner is not None:
                patch["rf"]["sat_quad_corner"] = float(corner)
            if border is not None:
                patch["rf"]["sat_show_border"] = float(border)
            if text is not None:
                patch["rf"]["sat_ird_osd"] = float(text)
        elif lab == "transfer":
            patch["player"] = {"transfer_station_mode": 4.0}
            if corner is not None:
                patch["player"]["transfer_quad_corner"] = float(corner)
            if border is not None:
                patch["player"]["transfer_show_border"] = float(border)
            if text is not None:
                patch["player"]["transfer_show_text"] = float(text)
        elif lab == "lomo_opt":
            patch["film"] = {"enabled": 1.0, "bloom_veil_mode": 5.0}
            if corner is not None:
                patch["film"]["quad_corner"] = float(corner)
            if border is not None:
                patch["film"]["show_border"] = float(border)
            if text is not None:
                patch["film"]["show_text"] = float(text)
        elif lab == "lomo_darkroom":
            patch["film"] = {"enabled": 1.0, "process_mode": 5.0}
            if corner is not None:
                patch["film"]["quad_corner"] = float(corner)
            if border is not None:
                patch["film"]["show_border"] = float(border)
            if text is not None:
                patch["film"]["show_text"] = float(text)
        elif lab == "telecine":
            if float(flm.get("enabled", 0.0)) >= 0.5 and float(flm.get("telecine_pulldown", 0.0)) >= 0.5:
                patch["film"] = {"enabled": 1.0, "telecine_pulldown": 3.0}
                if corner is not None:
                    patch["film"]["quad_corner"] = float(corner)
                if border is not None:
                    patch["film"]["show_border"] = float(border)
                if text is not None:
                    patch["film"]["show_text"] = float(text)
            else:
                patch["camera"] = {"film_enabled": 1.0, "telecine_pulldown": 3.0}
                if corner is not None:
                    patch["camera"]["telecine_quad_corner"] = float(corner)
                if border is not None:
                    patch["camera"]["telecine_show_border"] = float(border)
                if text is not None:
                    patch["camera"]["telecine_show_text"] = float(text)
        elif lab == "inversion":
            patch["player"] = {"inv_lab_enabled": 1.0, "inv_lab_mode": 3.0}
            if corner is not None:
                patch["player"]["inv_quad_corner"] = float(corner)
            if border is not None:
                patch["player"]["inv_show_border"] = float(border)
            if text is not None:
                patch["player"]["inv_show_text"] = float(text)
        elif lab == "archival":
            patch["player"] = {"archival_enabled": 1.0, "archival_mode": 4.0}
            if corner is not None:
                patch["player"]["archival_quad_corner"] = float(corner)
            if border is not None:
                patch["player"]["archival_show_border"] = float(border)
            if text is not None:
                patch["player"]["archival_show_text"] = float(text)
        elif lab == "env_field":
            patch["connector"] = {"env_field_enabled": 1.0, "env_field_mode": 5.0}
            if corner is not None:
                patch["connector"]["env_quad_corner"] = float(corner)
            if border is not None:
                patch["connector"]["env_show_border"] = float(border)
            if text is not None:
                patch["connector"]["env_show_text"] = float(text)
        elif lab == "streaming":
            patch["codec"] = {"stream_enabled": 1.0, "stream_mode": 5.0}
            if corner is not None:
                patch["codec"]["stream_quad_corner"] = float(corner)
            if border is not None:
                patch["codec"]["stream_show_border"] = float(border)
            if text is not None:
                patch["codec"]["stream_show_text"] = float(text)
        elif lab == "perception":
            patch["display"] = {"percept_enabled": 1.0, "percept_mode": 4.0}
            if corner is not None:
                patch["display"]["percept_quad_corner"] = float(corner)
            if border is not None:
                patch["display"]["percept_show_border"] = float(border)
            if text is not None:
                patch["display"]["percept_show_text"] = float(text)
        elif lab == "ecosystem":
            patch["overlay"] = {"eco_enabled": 1.0, "eco_mode": 4.0}
            if corner is not None:
                patch["overlay"]["eco_quad_corner"] = float(corner)
            if border is not None:
                patch["overlay"]["eco_show_border"] = float(border)
            if text is not None:
                patch["overlay"]["eco_show_text"] = float(text)
        elif lab == "crt_magnet":
            patch["display"] = {"crt_magnet_mode": 5.0}
            if corner is not None:
                patch["display"]["crt_magnet_quad_corner"] = float(corner)
            if border is not None:
                patch["display"]["crt_magnet_show_border"] = float(border)
            if text is not None:
                patch["display"]["crt_magnet_show_text"] = float(text)
        elif lab == "multiview":
            patch["display"] = {"mv_enabled": 1.0, "mv_layout": 0.0}
            if corner is not None:
                patch["display"]["mv_quad_corner"] = float(corner)
            if border is not None:
                patch["display"]["mv_show_border"] = float(border)
            if text is not None:
                patch["display"]["mv_tally_umd"] = float(text)

        merged = merge_values(p, patch) if patch else p
        return merged, lab, c_map

    def _render_frame_via_graph_with_params(self, idx: int, scale: float, params: dict) -> np.ndarray:
        """Render frame `idx` through `self.graph` using `params` as `base_params`."""
        from ..graph import GraphExecutor
        from ..core.standards import get_standard
        src = self.store.source_for((self.current.id, round(float(scale), 3)), self.current)
        idx = max(0, min(idx, max(0, getattr(src, "frame_count", 1) - 1)))
        ex = GraphExecutor(
            self.graph,
            get_standard(self.current.standard),
            params,
            src,
            int(getattr(src, "width", 720)),
            int(getattr(src, "height", 480)),
            self.current.fps,
            seed=self.current.seed,
            scale=1.0,
        )
        run = ex.render_frame(idx)
        if run.errors:
            raise RuntimeError("; ".join(run.errors))
        return np.clip(run.rgb, 0.0, 1.0).astype(np.float32)

    def render_still_with_quad(
        self,
        t: float,
        scale: float,
        want: str = "sim",
        corner: int | None = None,
        border: int | None = None,
        text: int | None = None,
        use_graph: bool = False,
    ) -> tuple[np.ndarray, str, str]:
        """
        Render a still frame at `(t, scale)` with optional `(corner, border, text)` overrides.
        Returns `(rgb_frame, file_slug, label_zh)`.
        """
        r, src = self.renderer(True, scale)
        idx = int(round(t * self.current.fps))
        idx = max(0, min(idx, max(0, getattr(src, "frame_count", 1) - 1)))
        if want == "source":
            frame = np.clip(src.frame(idx), 0.0, 1.0).astype(np.float32)
            return frame, "source", "原始素材"

        base_params = self.current.resolved_params()
        has_override = (corner is not None) or (border is not None) or (text is not None)
        merged, lab, c_map = self._detect_quad_lab_and_override(
            base_params,
            corner=corner,
            border=border,
            text=text,
            force_quad_if_none=bool(corner is not None and corner > 0),
            use_graph=use_graph,
        )
        eff_corner = int(corner) if corner is not None else 0
        if not has_override:
            rf = base_params.get("rf") or {}
            ply = base_params.get("player") or {}
            cam = base_params.get("camera") or {}
            dsp = base_params.get("display") or {}
            flm = base_params.get("film") or {}
            cn = base_params.get("connector") or {}
            cod = base_params.get("codec") or {}
            ov = base_params.get("overlay") or {}
            if lab == "tx_chain":
                eff_corner = int(round(float(rf.get("tx_quad_corner", 0.0))))
            elif lab == "sat_ird":
                eff_corner = int(round(float(rf.get("sat_quad_corner", 0.0))))
            elif lab == "transfer":
                eff_corner = int(round(float(ply.get("transfer_quad_corner", 0.0))))
            elif lab in ("lomo_opt", "lomo_darkroom"):
                eff_corner = int(round(float(flm.get("quad_corner", 0.0))))
            elif lab == "telecine":
                eff_corner = int(round(float(cam.get("telecine_quad_corner", flm.get("quad_corner", 0.0)))))
            elif lab == "inversion":
                eff_corner = int(round(float(ply.get("inv_quad_corner", 0.0))))
            elif lab == "archival":
                eff_corner = int(round(float(ply.get("archival_quad_corner", 0.0))))
            elif lab == "env_field":
                eff_corner = int(round(float(cn.get("env_quad_corner", 0.0))))
            elif lab == "streaming":
                eff_corner = int(round(float(cod.get("stream_quad_corner", 0.0))))
            elif lab == "perception":
                eff_corner = int(round(float(dsp.get("percept_quad_corner", 0.0))))
            elif lab == "ecosystem":
                eff_corner = int(round(float(ov.get("eco_quad_corner", 0.0))))
            elif lab == "crt_magnet":
                eff_corner = int(round(float(dsp.get("crt_magnet_quad_corner", 0.0))))
            elif lab == "multiview":
                eff_corner = int(round(float(dsp.get("mv_quad_corner", 0.0))))
            if use_graph:
                frame = self._render_frame_via_graph_with_params(idx, scale, base_params)
            else:
                frame = r.render_frame(idx)
        else:
            if use_graph:
                frame = self._render_frame_via_graph_with_params(idx, scale, merged)
            else:
                r.invalidate(merged)
                try:
                    frame = r.render_frame(idx)
                finally:
                    r.invalidate(base_params)
        slug, label_zh = c_map.get(eff_corner, ("still", "静帧"))
        return np.clip(frame, 0.0, 1.0).astype(np.float32), slug, label_zh

    def render_quad_burst(
        self,
        t: float,
        scale: float,
        fmt: str = "png",
        quality: int = 95,
        border: int = 0,
        text: int = 0,
        include_quad: bool = True,
        use_graph: bool = False,
    ) -> dict:
        """
        Batch-render the 4 individual corners (`1=左上`, `2=右上`, `3=左下`, `4=右下`) at full frame
        resolution (`W×H`) with `show_text = text` (default `0`, pure textless) and `show_border = 0`,
        plus the optional `0_quad_2x2` overview sheet (`corner=0`, `show_border=border`, `show_text=text`).
        """
        import base64
        r, src = self.renderer(True, scale)
        idx = int(round(t * self.current.fps))
        idx = max(0, min(idx, max(0, getattr(src, "frame_count", 1) - 1)))
        base_params = self.current.resolved_params()
        ext = "jpg" if fmt.lower() in ("jpg", "jpeg") else "png"
        ctype = "image/jpeg" if ext == "jpg" else "image/png"
        t_slug = f"{t:.2f}".replace(".", "_") + "s"

        corners_to_render = ([0, 1, 2, 3, 4] if include_quad else [1, 2, 3, 4])
        shots = []
        lab_name = "multiview"
        try:
            for c_idx in corners_to_render:
                b_val = int(border) if c_idx == 0 else 0
                merged, lab_name, c_map = self._detect_quad_lab_and_override(
                    base_params,
                    corner=c_idx,
                    border=b_val,
                    text=int(text),
                    force_quad_if_none=True,
                    use_graph=use_graph,
                )
                if use_graph:
                    frame = self._render_frame_via_graph_with_params(idx, scale, merged)
                else:
                    r.invalidate(merged)
                    frame = np.clip(r.render_frame(idx), 0.0, 1.0).astype(np.float32)
                raw_bytes = _jpeg(frame, int(quality)) if ext == "jpg" else _png(frame)
                slug, label_zh = c_map.get(c_idx, (f"c{c_idx}", f"角落{c_idx}"))
                fname = f"atsim_{t_slug}_{slug}.{ext}"
                shots.append({
                    "corner": c_idx,
                    "slug": slug,
                    "filename": fname,
                    "label": label_zh,
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "bytes": raw_bytes,
                    "b64": base64.b64encode(raw_bytes).decode("ascii"),
                })
        finally:
            if not use_graph:
                r.invalidate(base_params)

        return {
            "ok": True,
            "lab": lab_name,
            "t": round(float(t), 3),
            "scale": round(float(scale), 4),
            "format": ext,
            "content_type": ctype,
            "show_border": int(border),
            "show_text": int(text),
            "shots": shots,
        }

    def source_frame(self, t: float, scale: float, quality: int = 88) -> bytes:
        """
        The untouched source frame, for the A/B reference.

        This used to be produced by running the whole chain with every parameter
        neutralised, which has two problems.  It cannot actually be clean -- the
        neutral set has to name every parameter the chain can touch, and as soon
        as one is missed the "original" is silently distorted (the camera's
        highlight roll-off alone shifted it by 0.07, which the player's
        composite separation then amplified to 0.31).  And it costs a full
        luma/chroma separation and re-encode to produce a frame that is supposed
        to be unprocessed.

        The honest reference is the decoded clip itself, which is also what the
        chain is fed, so the comparison isolates the chain's contribution exactly.
        """
        r, src = self.renderer(True, scale)
        idx = int(round(t * self.current.fps))
        idx = max(0, min(idx, max(0, getattr(src, "frame_count", 1) - 1)))
        rgb = np.clip(src.frame(idx), 0.0, 1.0).astype(np.float32)
        return _jpeg(rgb, quality) if quality < 100 else _png(rgb)

    def sweep(self, layer: str, key: str, values: list[float], t: float,
              scale: float = 0.35, quality: int = 82,
              second: tuple[str, str, list[float]] | None = None) -> dict:
        """
        Render the current project once per parameter value, for one-page review.

        The point is to make "does this control behave the way it is documented"
        answerable by *looking*, on the user's own material, without running a
        script or dragging sliders one at a time.  Each cell is genuinely re-rendered
        through the same chain rather than reusing the preview, so the grid shows
        the mechanism instead of one cached picture.

        A second parameter may be swept as well, giving a small matrix rather than a
        strip -- useful where two mechanisms interact, such as tape speed with
        tracking error.
        """
        import base64
        from ..core.params import merge_values
        from ..io.imagecodec import jpeg_encode

        base = self.current.params or {}
        idx = int(round(t * self.current.fps))
        combos: list[tuple[float, ...]] = [(float(v),) for v in values]
        if second is not None and second[2]:
            combos = [(float(v), float(w)) for v in values for w in second[2]]

        tiles: list[dict] = []
        for combo in combos:
            params = merge_values(base, {layer: {key: float(combo[0])}})
            if second is not None and len(combo) > 1:
                params = merge_values(params, {second[0]: {second[1]: float(combo[1])}})
            r, src = self.renderer(True, scale)
            r.invalidate(params)
            last = max(0, int(getattr(src, "frame_count", 1)) - 1)
            frame = r.render_frame(min(idx, last))
            tile = {
                "value": float(combo[0]),
                "jpeg": base64.b64encode(jpeg_encode(frame, quality=quality)).decode("ascii"),
                "mean": round(float(frame.mean()), 4),
            }
            if second is not None and len(combo) > 1:
                tile["value2"] = float(combo[1])
            tiles.append(tile)

        cols = len(second[2]) if (second is not None and second[2]) else len(values)
        return {
            "layer": layer, "key": key, "t": float(t), "cols": max(1, int(cols)),
            "second": None if second is None else {"layer": second[0], "key": second[1]},
            "tiles": tiles,
        }

    def _clean_renderer(self, scale: float) -> Renderer:
        key = (self.current.id, "clean", round(float(scale), 3))
        r = self._clean.get(key)
        src = self.store.source_for((self.current.id, round(float(scale), 3)), self.current)
        if r is None:
            # The neutral set must cover every parameter the chain can add, or the
            # "original" side is not actually clean.  Several keys had been added
            # after this dict was written and were silently leaking into the
            # reference (recording speed, pre-emphasis, ringing, filter family),
            # and five more named parameters that do not exist at all -- the
            # simulator ignores unknown keys, so nothing complained while the
            # reference quietly kept whatever the project had set.
            #
            # Built here, then verified against the schema in `_neutral_checked`
            # so a typo or a newly added parameter fails loudly instead of
            # silently degrading the comparison.
            neutral = {
                "camera": {"sensor_noise": 0.0, "noise_grain": 0.0, "fixed_pattern": 0.0,
                           "chroma_aberration": 0.0, "motion_blur": 0.0,
                           "sharpness": 1.0, "exposure": 0.0, "white_balance_temp": 0.0,
                           "ccd_smear": 0.0, "halation": 0.0, "ccd_gain": 0.0,
                           "color_cast": 0.0, "highlight_tint": 0.0,
                           "lens_wear": 0.0, "lens_flare": 0.0},
                "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0,
                         "micro_jitter": 0.0, "vertical_instability": 0.0,
                         "head_switching": 0.0, "dropout_rate": 0.0, "wear": 0.0,
                         "crease_rate": 0.0, "noise_floor": 0.0,
                         "tape_speed": 0.0,          # SP
                         "head_clog": 0.0, "debris": 0.0, "surface_damage": 0.0,
                         "chroma_loss": 0.0, "tracking_error": 0.0,
                         "tracking_noise_rows": 0.0, "tracking_noise_wave": 0.0,
                         "tracking_noise_snow": 0.0,
                         "transport_state": 0.0, "reel_progress": 0.50, "tension_ripple": 0.0,
                         "belt_slip": 0.0, "roller_eccentricity": 0.0, "servo_hunt": 0.0},
                "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "tbc_residual": 0.0,
                           "transport_jitter": 0.0, "head_wear": 0.0,
                           "frequency_response": 0.0, "line_delay": 0.0,
                           "output_gain": 0.0, "output_bias": 0.0,
                           "preemphasis": 0.0, "ringing": 0.0},
                "display": {"enabled": 0.0, "reshoot_enabled": 0.0, "reshoot_moire": 0.0, "reshoot_tilt_deg": 0.0, "reshoot_perspective": 0.0, "reshoot_ambient_glare": 0.0, "reshoot_wb_shift": 0.0},
                "format": {"edge_boost": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                           "color_noise": 0.0, "chroma_vertical_averaging": 0.0,
                           "comb_depth": 0.0, "burst_phase_error": 0.0,
                           "luma_bandwidth": 5.0, "chroma_bandwidth": 1.5,
                           "filter_kind": 0.0},    # gaussian: no group delay
                "codec": {"dv_enabled": 0.0},
                "rf": {"enabled": 0.0, "adjacent_channel": 0.0, "sync_loss_amount": 0.0, "multipath_level": 0.0, "cochannel_level": 0.0, "if_saw_ringing": 0.0, "front_end_overload": 0.0},
                "audio": {"enabled": 0.0},
            }
            neutral = _neutral_checked(neutral)
            r = Renderer(src, getattr(src, "width", 720), getattr(src, "height", 480),
                         self.current.fps, neutral, seed=self.current.seed,
                         std_key=self.current.standard, preview=True, scale=scale)
            self._clean[key] = r
        return r

    # ------------------------------------------------------------------
    # the node graph
    # ------------------------------------------------------------------
    def graph_state(self) -> dict:
        from ..graph import CATALOG, TYPES, GRAPH_PRESETS
        issues = self.graph.validate()
        base_p = self.current.resolved_params()
        return {
            "graph": self.graph.to_dict(),
            "describe": self.graph.describe(base_params=base_p),
            "catalog": [n.to_dict() for n in CATALOG if n.kind != "node_group"],
            "types": TYPES,
            "presets": {k: {"label": v["label"], "note": v["note"]}
                        for k, v in GRAPH_PRESETS.items()},
            "issues": [i.to_dict() for i in issues],
            "project": self.project_dict(),
        }

    def graph_set(self, payload: dict) -> dict:
        from ..graph import Graph
        with self.lock:
            if "graph" in payload:
                self.graph = Graph.from_dict(payload["graph"])
            self._graph_execs.clear()
        return self.graph_state()

    def graph_load_preset(self, key: str) -> dict:
        from ..graph import GRAPH_PRESETS
        spec = GRAPH_PRESETS.get(key)
        if spec is None:
            return {"error": f"unknown graph preset {key!r}"}
        with self.lock:
            self.graph = spec["build"]()
            self._graph_execs.clear()
            # The preset declares grid columns; store pixels, so that everything downstream --
            # the canvas, a later drag, the renderer -- agrees on one unit.  The conversion
            # lives on the graph so that this route and the start-up constructor cannot drift.
            self.graph.normalise_positions()
            # a graph preset may also declare the reception conditions it was
            # designed around; without this the RF loop loaded a clean signal and
            # showed none of the artefacts it exists to demonstrate
            # Root-Cause Fix: Reset project params to clean baseline before applying a graph
            # preset! Previously, `merge_values(self.current.params or {}, spec["params"])` left
            # residual parameters from earlier presets (such as `display.percept_enabled=1,
            # percept_mode=2` from `complete_media_ecology_end_to_end_suite` or VHS `tape` jitter
            # from `family_archive_1992`) stuck in `self.current.params`, causing bottom-half
            # sinusoidal operator-hunting ripples (`sin(rows * 0.25 + t * 6.0)`) on LOMO presets.
            from ..core.params import merge_values
            from .session import _preset_baseline
            base_params = _preset_baseline()
            has_tape_node = any(
                n.kind in ("tape", "transport_servo", "vhs_gen", "vhs_dub_loop", "vhs_iterative_loop")
                for n in self.graph.nodes.values()
            )
            if not has_tape_node:
                base_params = merge_values(base_params, {
                    "format": {
                        "luma_bandwidth": 6.0, "chroma_bandwidth": 2.0,
                        "chroma_vertical_averaging": 0.0, "cross_color": 0.0,
                        "cross_luma": 0.0, "edge_boost": 0.0, "color_noise": 0.0,
                        "burst_phase_error": 0.0, "filter_kind": 0.0,
                    },
                    "tape": {
                        "tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0,
                        "micro_jitter": 0.0, "vertical_instability": 0.0, "head_switching": 0.0,
                        "dropout_rate": 0.0, "wear": 0.0, "crease_rate": 0.0, "noise_floor": 0.0,
                        "tracking_error": 0.0, "row_timebase": 0.0, "stripe_moire": 0.0,
                    },
                    "player": {
                        "tbc_enabled": 1.0, "tbc_strength": 1.0, "tbc_residual": 0.0,
                        "transport_jitter": 0.0, "head_wear": 0.0, "frequency_response": 0.0,
                        "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0,
                    },
                })
            if spec.get("params"):
                base_params = merge_values(base_params, spec["params"])
            self.current.params = base_params
            self.store.drop(self.current.id)
            self.store.save(self.current)
        return self.graph_state()

    def graph_add(self, payload: dict) -> dict:
        from ..graph import CATALOG_BY_KIND
        kind = str(payload.get("kind", ""))
        if kind not in CATALOG_BY_KIND:
            return {"error": f"unknown node kind {kind!r}"}
        with self.lock:
            base = kind
            n = 1
            name = base
            while name in self.graph.nodes:
                n += 1
                name = f"{base}{n}"
            self.graph.add(name, kind,
                           position=(float(payload.get("x", 0.0)),
                                     float(payload.get("y", 0.0))))
            # The caller sends `40 + random*260`, which is pixels.  `graph_move` marks its
            # position as pixels and `graph_add` did not, so a node from the palette came back
            # unflagged and could be read as a grid column -- column 300 is 67200 px to the
            # right, which is where a node would vanish to.
            self.graph.nodes[name].px_position = True
            self._graph_execs.clear()
        return self.graph_state()

    def graph_connect(self, payload: dict) -> dict:
        """Add an edge, but only if the types actually match."""
        with self.lock:
            e = None
            try:
                e = self.graph.connect(str(payload["src"]), str(payload["src_port"]),
                                       str(payload["dst"]), str(payload["dst_port"]))
                issues = [i for i in self.graph.validate() if i.severity == "error"]
                if issues:
                    # reject the edge rather than leaving a broken graph behind
                    self.graph.edges.remove(e)
                    return {"error": issues[0].message, **self.graph_state()}
            except KeyError as exc:
                if e is not None and e in self.graph.edges:
                    self.graph.edges.remove(e)
                return {"error": f"invalid connection: {exc}", **self.graph_state()}
            self._graph_execs.clear()
        return self.graph_state()

    def graph_disconnect(self, payload: dict) -> dict:
        with self.lock:
            self.graph.disconnect(str(payload["dst"]), str(payload["dst_port"]))
            self._graph_execs.clear()
        return self.graph_state()

    def graph_remove(self, payload: dict) -> dict:
        with self.lock:
            self.graph.remove(str(payload["name"]))
            self._graph_execs.clear()
        return self.graph_state()

    def graph_move(self, payload: dict) -> dict:
        """
        Record a dragged position.

        The `px` flag matters: a node placed by a preset carries *grid* units,
        while a dragged node carries pixels.  Storing the distinction is what
        stops the canvas from re-scaling a dragged coordinate on the next load,
        which previously threw the node far off screen.
        """
        with self.lock:
            node = self.graph.nodes.get(str(payload.get("name", "")))
            if node is None:
                return {"error": "no such node", **self.graph_state()}
            node.position = (float(payload.get("x", 0.0)),
                             float(payload.get("y", 0.0)))
            node.px_position = True
        return {"ok": True}

    def graph_node_params(self, payload: dict) -> dict:
        """Per-node parameter overrides, label, and iteration count, synced cleanly with project state."""
        with self.lock:
            node = self.graph.nodes.get(str(payload.get("name", "")))
            if node is None:
                return {"error": "no such node"}
            if "enabled" in payload:
                node.enabled = bool(payload.get("enabled", node.enabled))
                if node.kind == "display":
                    self.current.params = merge_values(
                        self.current.params or {},
                        {"display": {"enabled": 1.0 if node.enabled else 0.0}},
                    )
            if "label" in payload and payload["label"] is not None:
                node.label = str(payload["label"]).strip()
                if node.kind == "node_group" and isinstance(node.subgraph, dict) and node.label:
                    node.subgraph["label"] = node.label
            if "iterations" in payload and payload["iterations"] is not None:
                node.iterations = max(1, min(25, int(round(float(payload["iterations"])))))
            if isinstance(payload.get("params"), dict):
                clean_p = {}
                for k, v in payload["params"].items():
                    if isinstance(v, str):
                        node.params[k] = v
                        clean_p[k] = v
                    elif isinstance(v, (int, float)):
                        node.params[k] = float(v)
                        clean_p[k] = float(v)
                layer = node.spec.params_layer
                if layer and clean_p:
                    same_layer_nodes = [
                        n for n in self.graph.nodes.values()
                        if n.spec.params_layer == layer
                    ]
                    if len(same_layer_nodes) <= 1:
                        self.current.params = merge_values(
                            self.current.params or {},
                            {layer: clean_p},
                        )
                        self.store.save(self.current)
        return self.graph_state()

    def graph_group_nodes(self, payload: dict) -> dict:
        """Pack multiple selected nodes into a Blender-style `node_group` (e.g. 'VHS')."""
        with self.lock:
            raw_names = payload.get("nodes") or []
            if not isinstance(raw_names, list):
                return {"error": "nodes must be a list", **self.graph_state()}
            label = str(payload.get("label") or "VHS").strip() or "VHS"
            grp_name = payload.get("name")
            try:
                g_node = self.graph.group_nodes(
                    [str(x) for x in raw_names],
                    label=label,
                    group_name=str(grp_name) if grp_name else None,
                    base_params=self.current.resolved_params(),
                )
            except Exception as exc:
                return {"error": str(exc), **self.graph_state()}
            self._graph_execs.clear()
            st = self.graph_state()
            st["created_group"] = g_node.name
            return st

    def graph_ungroup_node(self, payload: dict) -> dict:
        """Unpack a `node_group` back into individual nodes in the parent graph."""
        with self.lock:
            name = str(payload.get("name") or "")
            try:
                restored = self.graph.ungroup_node(name)
            except Exception as exc:
                return {"error": str(exc), **self.graph_state()}
            self._graph_execs.clear()
            st = self.graph_state()
            st["restored_nodes"] = restored
            return st

    def graph_duplicate_node(self, payload: dict) -> dict:
        """Duplicate a node or `node_group` (either in parallel or chained in series)."""
        with self.lock:
            src_name = str(payload.get("name") or "")
            chain = bool(payload.get("chain", False))
            new_label = payload.get("label")
            try:
                dup = self.graph.duplicate_node(
                    src_name, chain_in_series=chain,
                    new_label=str(new_label) if new_label else None,
                )
            except Exception as exc:
                return {"error": str(exc), **self.graph_state()}
            self._graph_execs.clear()
            st = self.graph_state()
            st["duplicated_node"] = dup.name
            return st

    def graph_subnode_params(self, payload: dict) -> dict:
        """Update parameters, enabled state, iterations, or position of an internal node inside a `node_group`."""
        with self.lock:
            grp_name = str(payload.get("group") or "")
            sub_name = str(payload.get("name") or "")
            updated = self.graph.update_subnode(grp_name, sub_name, payload)
            if updated is None:
                return {"error": f"找不到节点组 {grp_name!r} 内的子节点 {sub_name!r}", **self.graph_state()}
            self._graph_execs.clear()
            return self.graph_state()

    def graph_chain_duplicate(self, payload: dict) -> dict:
        """Append a duplicate of an effector node or `node_group` in series right after `name`."""
        payload_copy = dict(payload or {})
        payload_copy["chain"] = True
        return self.graph_duplicate_node(payload_copy)

    def graph_frame(self, t: float, scale: float, quality: int,
                    clean: bool = False) -> bytes:
        """Render one frame by executing the user's graph."""
        from ..graph import GraphExecutor
        from ..core.standards import get_standard
        idx = int(round(t * self.current.fps))
        with self.lock:
            key = (self.current.id, "graph", round(float(scale), 3),
                   self.current.seed, self.current.standard, clean)
            ex = self._graph_execs.get(key)
            src = self.store.source_for((self.current.id, round(float(scale), 3)),
                                        self.current)
            params = self.current.resolved_params()
            if clean:
                params = _neutral_params(params)
            if ex is None:
                ex = GraphExecutor(self.graph, get_standard(self.current.standard),
                                   params, src, int(getattr(src, "width", 720)),
                                   int(getattr(src, "height", 480)), self.current.fps,
                                   seed=self.current.seed, scale=1.0)
                self._graph_execs[key] = ex
            else:
                ex.graph = self.graph
                ex.sync_params(params)
        idx = max(0, min(idx, max(0, getattr(src, "frame_count", 1) - 1)))
        run = ex.render_frame(idx)
        if run.errors:
            raise RuntimeError("; ".join(run.errors))
        if getattr(ex, "osd_report", None):
            self.osd_report = dict(ex.osd_report)
        return _jpeg(run.rgb, quality) if quality < 100 else _png(run.rgb)

    def graph_perf(self) -> dict:
        """Per-node timing and cache hits, for the canvas readout."""
        out = {}
        for key, ex in self._graph_execs.items():
            out[str(key)] = ex.describe()
        return out

    def scopes(self, t: float = 0.0, scale: float = 0.5,
               line_y: float | int | None = None,
               engine: str = "chain",
               node_a: str | None = None,
               node_b: str | None = None) -> dict:
        """
        Compute Broadcast Waveform Monitor (IRE Scope), Chroma Vectorscope,
        and Node-Level / Chain A/B Difference Heatmap & Power Spectrum.
        """
        from ..codecs.scopes import compute_all_scopes
        from ..core.standards import get_standard
        from ..graph import GraphExecutor

        std = get_standard(self.current.standard)
        params = self.current.resolved_params()
        idx = int(round(t * self.current.fps))

        available_nodes: list[dict] = [{"id": "source", "label": "原始素材 (Source)"}]
        label_a = "最终输出 (Output)"
        label_b = "原始素材 (Source)"

        if engine == "graph":
            with self.lock:
                key = (self.current.id, "graph", round(float(scale), 3),
                       self.current.seed, self.current.standard, False)
                ex = self._graph_execs.get(key)
                src = self.store.source_for((self.current.id, round(float(scale), 3)),
                                            self.current)
                if ex is None:
                    ex = GraphExecutor(self.graph, std, params, src,
                                       int(getattr(src, "width", 720)),
                                       int(getattr(src, "height", 480)), self.current.fps,
                                       seed=self.current.seed, scale=1.0)
                    self._graph_execs[key] = ex
                else:
                    ex.graph = self.graph
                    ex.sync_params(params)
            idx = max(0, min(idx, max(0, getattr(src, "frame_count", 1) - 1)))
            run = ex.render_frame(idx)
            src_rgb = np.clip(src.frame(idx), 0.0, 1.0).astype(np.float32)
            for n_name in ex.order:
                nd = self.graph.nodes.get(n_name)
                n_lbl = (nd.label or nd.spec.label) if nd else n_name
                available_nodes.append({"id": n_name, "label": f"{n_lbl} ({n_name})"})
            if node_a and node_a in run.node_frames:
                sim_rgb = np.clip(run.node_frames[node_a], 0.0, 1.0).astype(np.float32)
                label_a = f"节点 {node_a}"
            elif node_a == "source":
                sim_rgb = src_rgb
                label_a = "原始素材 (Source)"
            else:
                sim_rgb = np.clip(run.rgb, 0.0, 1.0).astype(np.float32)

            if node_b and node_b in run.node_frames:
                ref_rgb = np.clip(run.node_frames[node_b], 0.0, 1.0).astype(np.float32)
                label_b = f"节点 {node_b}"
            else:
                ref_rgb = src_rgb
        else:
            r, src = self.renderer(True, scale)
            idx = max(0, min(idx, max(0, getattr(src, "frame_count", 1) - 1)))
            out_rgb = np.clip(r.render_frame(idx), 0.0, 1.0).astype(np.float32)
            src_rgb = np.clip(src.frame(idx), 0.0, 1.0).astype(np.float32)
            from ..render.renderer import STAGE_LABELS
            for st_name in STAGE_ORDER:
                available_nodes.append({"id": st_name, "label": STAGE_LABELS.get(st_name, st_name)})

            def _pick_stage(name_or_empty: str | None, default_arr: np.ndarray, default_lbl: str):
                if not name_or_empty:
                    return default_arr, default_lbl
                if name_or_empty == "source":
                    return src_rgb, "原始素材 (Source)"
                aux = (r._cache_buf.get(name_or_empty) or {}).get(idx)
                if aux is not None:
                    comp_c, meta_c = aux
                    if comp_c is not None and name_or_empty in ("format", "connector", "tape"):
                        from ..core.signal import SignalFrame, demod_chroma, split_composite
                        y_b, c_b = split_composite(comp_c, std, comb_depth=0.0, chroma_bandwidth_hz=0.85e6)
                        i_b, q_b = demod_chroma(c_b / 0.63, std, i_bw_hz=1.3e6, q_bw_hz=0.6e6)
                        return np.clip(SignalFrame(y_b, i_b, q_b).to_rgb(), 0.0, 1.0), STAGE_LABELS.get(name_or_empty, name_or_empty)
                    sig_c = (meta_c or {}).get("signal")
                    if sig_c is not None and hasattr(sig_c, "to_rgb"):
                        return np.clip(sig_c.to_rgb(), 0.0, 1.0), STAGE_LABELS.get(name_or_empty, name_or_empty)
                cached = (r._cache.get(name_or_empty) or {}).get(idx)
                if cached is not None:
                    return np.clip(np.asarray(cached, dtype=np.float32), 0.0, 1.0), STAGE_LABELS.get(name_or_empty, name_or_empty)
                return default_arr, default_lbl

            sim_rgb, label_a = _pick_stage(node_a, out_rgb, "最终输出 (Output)")
            ref_rgb, label_b = _pick_stage(node_b, src_rgb, "原始素材 (Source)")

        res = compute_all_scopes(
            sim_rgb=sim_rgb,
            ref_rgb=ref_rgb,
            std=std,
            params=params,
            line_y=line_y,
            label_a=label_a,
            label_b=label_b,
        )
        res["t"] = round(float(t), 4)
        res["frame_index"] = int(idx)
        res["engine"] = engine
        res["available_nodes"] = available_nodes
        return res

    def timeline(self) -> dict:
        r, _ = self.renderer(True, 0.5)
        return {
            "stages": r.plan()["stages"],
            "defects": r.timeline(),
            "seed": self.current.seed,
            "fps": self.current.fps,
        }

    def _source_audio_window(self, start_s: float, duration_s: float,
                             sample_rate: int, spec,
                             project: Project | None = None) -> tuple[np.ndarray, bool]:
        """
        Fetch or synthesize a stereo float32 PCM window [start_s, start_s + duration_s).
        Returns (pcm_array, is_synthetic).
        """
        from ..codecs.audio import synthesize_test_audio
        from ..io.ffmpeg import read_audio_pcm
        p = project if project is not None else self.current
        if p.source and int(round(spec.synth_type)) == 0:
            clip = self.store.clip_path(p.source)
            if clip.exists() and (p.source_info or {}).get("has_audio", True):
                pcm = read_audio_pcm(clip, sample_rate=sample_rate,
                                     start_s=max(0.0, float(start_s)),
                                     duration_s=max(0.1, float(duration_s)))
                if pcm is not None and len(pcm) > 0 and float(np.max(np.abs(pcm))) > 1e-5:
                    target_n = max(1, int(round(duration_s * sample_rate)))
                    if len(pcm) < target_n:
                        pad = np.zeros((target_n - len(pcm), 2), dtype=np.float32)
                        pcm = np.concatenate([pcm, pad], axis=0)
                    else:
                        pcm = pcm[:target_n]
                    return pcm, False
        # Preserve synth_type=0 (retro synth chord+beat) when falling back so preview & export match:
        kind = int(round(spec.synth_type))
        pcm = synthesize_test_audio(duration_s, sample_rate=sample_rate,
                                    kind=kind, seed=p.seed, start_s=start_s)
        return pcm, True

    def audio_window(self, start_s: float = 0.0, duration_s: float = 4.0,
                     mode: str = "sim", sample_rate: int = 44100):
        """
        Render an audio window [start_s, start_s + duration_s) synchronized with the
        exact video TapeStage defect plan at the same frame indices.
        """
        from ..codecs.audio import AudioSimulator, AudioSpec
        r, src = self.renderer(True, 0.5)
        fps = float(self.current.fps or 29.97)
        start_s = max(0.0, float(start_s))
        duration_s = float(np.clip(duration_s, 0.2, 60.0))
        start_frame = max(0, int(round(start_s * fps)))
        end_frame = start_frame + max(1, int(np.ceil(duration_s * fps))) + 2

        tape_stage = (r.stage or {}).get("tape")
        if tape_stage is not None and hasattr(tape_stage, "_ensure"):
            tape_stage._ensure(end_frame)
        tape_plan = getattr(tape_stage, "plan", None)

        params = self.current.resolved_params()
        spec = AudioSpec.from_params(params.get("audio", {}))
        pcm_in, is_synth = self._source_audio_window(start_s, duration_s, sample_rate, spec)
        sim = AudioSimulator(params, seed=self.current.seed, fps=fps,
                             sample_rate=sample_rate)
        if mode == "source":
            faults = sim.extract_fault_envelope(start_frame, max(1, end_frame - start_frame), tape_plan)
            from ..codecs.audio import AudioRun
            return AudioRun(
                pcm=pcm_in,
                sample_rate=sample_rate,
                start_s=start_s,
                duration_s=duration_s,
                effective_mode="source_clean",
                fault_frames=faults,
                summary={"effective_mode": "source_clean", "synthetic_source": is_synth},
            )
        run = sim.process(pcm_in, start_s=start_s, tape_plan=tape_plan)
        run.summary["synthetic_source"] = is_synth
        return run

    def audio_wav(self, start_s: float = 0.0, duration_s: float = 4.0,
                  mode: str = "sim", sample_rate: int = 44100,
                  clean: bool = False) -> bytes:
        from ..io.ffmpeg import wav_encode
        if clean or mode in ("source", "original", "clean"):
            mode = "source"
        run = self.audio_window(start_s=start_s, duration_s=duration_s,
                                mode=mode, sample_rate=sample_rate)
        return wav_encode(run.pcm, sample_rate=run.sample_rate)

    def audio_timeline(self, start_s: float = 0.0, duration_s: float = 4.0) -> dict:
        """
        Return the synchronized audio waveform envelope and per-frame A/V fault telemetry
        for the WebUI oscilloscope and fault-sync strip.
        """
        run = self.audio_window(start_s=start_s, duration_s=duration_s,
                                mode="sim", sample_rate=22050)
        bins = 180
        pcm = run.pcm
        n = len(pcm)
        step = max(1, n // bins)
        trimmed = pcm[:step * bins].reshape(bins, step, 2) if n >= bins else pcm[None, :, :]
        peak_l = np.max(np.abs(trimmed[:, :, 0]), axis=1).round(4).tolist()
        peak_r = np.max(np.abs(trimmed[:, :, 1]), axis=1).round(4).tolist()
        rms_b = np.sqrt(np.mean(trimmed ** 2, axis=(1, 2))).round(4).tolist()
        wf_list = [
            {"peak": round(float(max(pl, pr)), 4), "rms": round(float(rm), 4),
             "peak_l": round(float(pl), 4), "peak_r": round(float(pr), 4)}
            for pl, pr, rm in zip(peak_l, peak_r, rms_b)
        ]
        ff_dicts = [f.to_dict() for f in run.fault_frames]
        return {
            "start_s": round(run.start_s, 3),
            "duration_s": round(run.duration_s, 3),
            "fps": float(self.current.fps or 29.97),
            "effective_mode": run.effective_mode,
            "summary": run.summary,
            "waveform": wf_list,
            "waveform_channels": {"bins": len(peak_l), "peak_l": peak_l, "peak_r": peak_r, "rms": rms_b},
            "frames": ff_dicts,
            "fault_frames": ff_dicts,
        }

    # ------------------------------------------------------------------
    def resolve_scale(self, raw_scale: str | float, default: float = 1.0) -> float:
        s_val = str(raw_scale).strip().lower()
        if s_val in ("sd", "sd720", "720"):
            info = self.current.source_info or {}
            w = float(info.get("width") or 720.0)
            return round(float(np.clip(720.0 / max(1.0, w), 0.1, 2.0)), 4)
        try:
            return round(float(np.clip(float(s_val), 0.1, 2.0)), 4)
        except (TypeError, ValueError):
            return default

    def start_render(self, scale: float = 1.0, crf: int = 16,
                     keep_audio: bool = True, preset: str = "medium") -> str:
        job_id = uuid.uuid4().hex[:10]
        self.jobs[job_id] = {"state": "queued", "progress": 0.0, "message": "排队中",
                             "path": None, "started": time.time(),
                             "scale": scale, "crf": crf, "preset": preset}
        th = threading.Thread(target=self._run_render, args=(job_id, scale, crf, keep_audio, preset),
                              daemon=True)
        th.start()
        return job_id

    def _run_render(self, job_id: str, scale: float, crf: int, keep_audio: bool,
                    preset: str = "medium") -> None:
        import gc
        job = self.jobs[job_id]
        sim_wav_path = None
        r = None
        src = None
        try:
            project = Project.from_dict(self.current.to_dict())
            store = SessionStore(self.store.data_dir)
            store.projects[project.id] = project
            # The source already delivers frames at the requested scale, so the
            # renderer must be sized to *its* dimensions.  Scaling the source's
            # nominal size again leaves the stage buffers at a different shape
            # than the frames flowing through them.
            src = store.source_for((project.id, scale), project)
            width = int(getattr(src, "width", 720))
            height = int(getattr(src, "height", 480))
            resolved_p = project.resolved_params()
            # Enable streaming=True for batch export so intermediate stages do not
            # accumulate unbounded float32 frame caches in RAM/pagefile ([WinError 1455]).
            r = Renderer(src, width, height, project.fps, resolved_p,
                         seed=project.seed, std_key=project.standard,
                         preview=(scale < 1.0), scale=1.0,
                         streaming=True, max_cache_frames=2)
            r.set_time_varying(project.keyframe_tracks(), project.environment_state())
            total = max(1, int(getattr(src, "frame_count", 1)))
            out_path = self.store.renders / f"{job_id}.mp4"
            audio_src = None
            if keep_audio:
                aud_cfg = resolved_p.get("audio", {}) or {}
                aud_enabled = float(aud_cfg.get("enabled", 1.0)) > 0.5
                if aud_enabled:
                    try:
                        dur_s = float(total) / max(1.0, float(project.fps))
                        wav_bytes = self.audio_wav(start_s=0.0, duration_s=dur_s, mode="sim", sample_rate=44100)
                        sim_wav_path = self.store.renders / f"{job_id}_audio.wav"
                        sim_wav_path.write_bytes(wav_bytes)
                        audio_src = str(sim_wav_path)
                        del wav_bytes
                        gc.collect()
                    except Exception:
                        if project.source:
                            clip = store.clip_path(project.source)
                            if clip.exists():
                                audio_src = str(clip)
                elif project.source:
                    clip = store.clip_path(project.source)
                    if clip.exists():
                        audio_src = str(clip)

            job.update(state="running", message="渲染中", total=total)
            with VideoWriter(out_path, width, height, project.fps, crf=crf,
                             preset=preset) as vw:
                vw.open(audio_from=audio_src if audio_src else None)
                for i, rgb in r.render_frames(list(range(total))):
                    vw.write(rgb)
                    if i % 5 == 0 or i == total - 1:
                        job["progress"] = (i + 1) / total
                        job["message"] = f"渲染中 {i + 1}/{total}"
                        if job.get("cancel"):
                            break
                    if i % 25 == 24:
                        gc.collect()
            job.update(state="done", progress=1.0, message="完成", path=str(out_path))
        except Exception as exc:
            job.update(state="error", message=f"{type(exc).__name__}: {exc}",
                       trace=traceback.format_exc()[-2000:])
        finally:
            if r is not None:
                r._cache.clear()
                r._cache_buf.clear()
            if src is not None and hasattr(src, "clear"):
                try:
                    src.clear()
                except Exception:
                    pass
            if sim_wav_path is not None and sim_wav_path.exists():
                try:
                    sim_wav_path.unlink()
                except Exception:
                    pass
            gc.collect()
# ---------------------------------------------------------------------------
def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "virtual-media-lab"

        def log_message(self, fmt, *args):  # quieter console
            if "/api/frame" in (self.path or "") or "/api/scopes" in (self.path or ""):
                return
            print(f"  {self.address_string()} {fmt % args}")

        # -- helpers ---------------------------------------------------
        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send(self, code: int, body: bytes, ctype: str,
                  extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            raw = self.rfile.read(n)
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return {}

        # -- routes ----------------------------------------------------
        def do_GET(self) -> None:
            u = urlparse(self.path)
            q = parse_qs(u.query)
            path = u.path
            try:
                if path in ("/", "/index.html"):
                    return self._static("index.html")
                # The connector test bench: every interface degraded side by side against a
                # clean baseline, so the differences between them can be read directly
                # instead of inferred from one picture at a time.
                if path in ("/connectors", "/connectors.html"):
                    return self._static("connectors.html")
                if path.startswith("/static/"):
                    return self._static(path[len("/static/"):])
                if path == "/api/catalog":
                    return self._json({"catalog": app.store.catalog()})
                if path == "/api/schema":
                    return self._json(SCHEMA.schema_dict())
                if path == "/api/project":
                    return self._json(app.project_dict())
                if path == "/api/export":
                    # a download rather than a JSON reply, so the browser saves it as a file
                    cfg = app.export_config()
                    body = json.dumps(cfg, ensure_ascii=False, indent=2, default=str).encode("utf-8")
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    return self._send(200, body, "application/json; charset=utf-8", {
                        "Content-Disposition":
                            'attachment; filename="atsim-config-%s.json"' % stamp})
                if path == "/api/scopes":
                    t = float((q.get("t") or ["0"])[0])
                    scale = app.resolve_scale((q.get("scale") or ["0.5"])[0], default=0.5)
                    raw_line = (q.get("line") or [None])[0]
                    line_val = float(raw_line) if raw_line is not None and raw_line != "" else None
                    engine = str((q.get("engine") or ["chain"])[0])
                    node_a = str((q.get("node_a") or [""])[0]).strip() or None
                    node_b = str((q.get("node_b") or ["source"])[0]).strip() or "source"
                    return self._json(app.scopes(t=t, scale=scale, line_y=line_val,
                                                 engine=engine, node_a=node_a, node_b=node_b))
                if path == "/api/timeline":
                    return self._json(app.timeline())
                if path == "/api/audio":
                    t = float((q.get("t") or ["0"])[0])
                    dur = float((q.get("dur") or q.get("duration") or ["4.0"])[0])
                    mode = str((q.get("mode") or ["sim"])[0])
                    sr = int(np.clip(int((q.get("sr") or ["44100"])[0]), 8000, 48000))
                    wav_bytes = app.audio_wav(start_s=t, duration_s=dur, mode=mode, sample_rate=sr)
                    dl = (q.get("download") or ["0"])[0] in ("1", "true", "yes")
                    headers = {}
                    if dl:
                        headers["Content-Disposition"] = f'attachment; filename="atsim_audio_{t:.2f}s.wav"'
                    return self._send(200, wav_bytes, "audio/wav", headers)
                if path == "/api/audio_timeline":
                    t = float((q.get("t") or ["0"])[0])
                    dur = float((q.get("dur") or q.get("duration") or ["4.0"])[0])
                    return self._json(app.audio_timeline(start_s=t, duration_s=dur))
                if path == "/api/frame":
                    t = float((q.get("t") or ["0"])[0])
                    scale = app.resolve_scale((q.get("scale") or ["0.5"])[0], default=0.5)
                    qual = int((q.get("q") or ["85"])[0])
                    clean = (q.get("mode") or [""])[0] == "original"
                    source_mode = (q.get("mode") or [""])[0] == "source"
                    use_graph = (q.get("engine") or ["chain"])[0] == "graph"
                    t0 = time.time()
                    if source_mode:
                        # the A/B reference: the decoded clip itself
                        data = app.source_frame(t, scale, qual)
                    else:
                        data = (app.graph_frame(t, scale, qual, clean=clean) if use_graph
                                else app.frame(t, scale, qual, clean=clean))
                    ctype = "image/png" if qual >= 100 else "image/jpeg"
                    return self._send(200, data, ctype, {
                        "X-Render-Ms": f"{(time.time() - t0) * 1000:.0f}"})
                if path == "/api/still":
                    # Export one frame as a full-resolution still, optionally overriding
                    # `corner` (0=2x2 quad, 1..4=individual corner), `border` (0..2), and `text` (0..2).
                    try:
                        t = float((q.get("t") or ["0"])[0])
                    except (ValueError, TypeError):
                        t = 0.0
                    want = (q.get("mode") or ["sim"])[0]
                    fmt = (q.get("format") or ["png"])[0].lower()
                    scale = app.resolve_scale((q.get("scale") or ["1.0"])[0], default=1.0)
                    corner_q = None
                    if "corner" in q and q["corner"] and q["corner"][0] not in ("", "null", "undefined"):
                        try:
                            corner_q = int(float(q["corner"][0]))
                        except (ValueError, TypeError):
                            corner_q = None
                    border_q = None
                    if "border" in q and q["border"] and q["border"][0] not in ("", "null", "undefined"):
                        try:
                            border_q = int(float(q["border"][0]))
                        except (ValueError, TypeError):
                            border_q = None
                    text_q = None
                    if "text" in q and q["text"] and q["text"][0] not in ("", "null", "undefined"):
                        try:
                            text_q = int(float(q["text"][0]))
                        except (ValueError, TypeError):
                            text_q = None
                    use_graph = (q.get("engine") or ["chain"])[0] == "graph"
                    try:
                        frame, slug, _lbl = app.render_still_with_quad(
                            t, scale, want=want, corner=corner_q, border=border_q, text=text_q,
                            use_graph=use_graph,
                        )
                    except Exception:
                        r, src = app.renderer(True, scale)
                        idx = int(round(t * app.current.fps))
                        idx = max(0, min(idx, max(0, getattr(src, "frame_count", 1) - 1)))
                        frame = np.clip(r.render_frame(idx), 0.0, 1.0).astype(np.float32)
                        slug, _lbl = "still", "静帧" 
                    if fmt in ("jpg", "jpeg"):
                        data = _jpeg(frame, int((q.get("q") or ["92"])[0]))
                        ctype, ext = "image/jpeg", "jpg"
                    else:
                        data = _png(frame)
                        ctype, ext = "image/png", "png"
                    t_slug = f"{t:.2f}".replace(".", "_") + "s"
                    return self._send(200, data, ctype, {
                        "Content-Disposition":
                            f'attachment; filename="atsim_{t_slug}_{slug}.{ext}"'})
                if path == "/api/still_quad_burst":
                    # Batch 4-Corner Burst Still Export: renders all 4 corners (1..4) at full frame
                    # resolution with no text (`text=0`) and no border (`border=0`), plus optional 2x2 quad.
                    import zipfile
                    t = float((q.get("t") or ["0"])[0])
                    fmt = (q.get("format") or ["png"])[0].lower()
                    qual = int((q.get("q") or ["95"])[0])
                    scale = app.resolve_scale((q.get("scale") or ["1.0"])[0], default=1.0)
                    border_q = int(float((q.get("border") or ["0"])[0]))
                    text_q = int(float((q.get("text") or ["0"])[0]))
                    out_mode = (q.get("mode") or ["json"])[0].lower()
                    inc_quad = (q.get("include_quad") or ["1"])[0] not in ("0", "false", "no")
                    use_graph = (q.get("engine") or ["chain"])[0] == "graph"
                    burst = app.render_quad_burst(
                        t, scale, fmt=fmt, quality=qual,
                        border=border_q, text=text_q, include_quad=inc_quad,
                        use_graph=use_graph,
                    )
                    if out_mode == "zip":
                        zbuf = io.BytesIO()
                        manifest = {
                            "kind": "atsim.quad_burst",
                            "lab": burst["lab"],
                            "t": burst["t"],
                            "scale": burst["scale"],
                            "format": burst["format"],
                            "show_border": burst["show_border"],
                            "show_text": burst["show_text"],
                            "files": [],
                        }
                        with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                            for s in burst["shots"]:
                                zf.writestr(s["filename"], s["bytes"])
                                manifest["files"].append({
                                    "corner": s["corner"],
                                    "filename": s["filename"],
                                    "label": s["label"],
                                    "width": s["width"],
                                    "height": s["height"],
                                })
                            zf.writestr(
                                "burst_manifest.json",
                                json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
                            )
                        t_slug = f"{t:.2f}".replace(".", "_") + "s"
                        return self._send(200, zbuf.getvalue(), "application/zip", {
                            "Content-Disposition":
                                f'attachment; filename="atsim_quad_burst_{t_slug}.zip"'})
                    # JSON mode for browser 4-file burst download
                    json_shots = [
                        {k: v for k, v in s.items() if k != "bytes"}
                        for s in burst["shots"]
                    ]
                    return self._json({
                        "ok": True,
                        "lab": burst["lab"],
                        "t": burst["t"],
                        "scale": burst["scale"],
                        "format": burst["format"],
                        "content_type": burst["content_type"],
                        "show_border": burst["show_border"],
                        "show_text": burst["show_text"],
                        "shots": json_shots,
                    })
                if path == "/api/graph":
                    return self._json(app.graph_state())
                if path == "/api/graph/perf":
                    return self._json(app.graph_perf())
                if path == "/api/render":
                    scale = app.resolve_scale((q.get("scale") or ["1.0"])[0], default=1.0)
                    crf = int(np.clip(int((q.get("crf") or ["16"])[0]), 0, 51))
                    preset = str((q.get("preset") or ["medium"])[0]).strip().lower()
                    if preset not in ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"):
                        preset = "medium"
                    keep_audio = (q.get("audio") or ["1"])[0] not in ("0", "false", "no")
                    return self._json({"job": app.start_render(scale, crf, keep_audio=keep_audio, preset=preset)})

                if path == "/api/faults":
                    from ..diagnostics import as_dicts, coverage, STAGE_LABELS
                    return self._json({"faults": as_dicts(),
                                       "coverage": coverage(),
                                       "stage_labels": STAGE_LABELS})
                if path == "/api/job":
                    jid = (q.get("id") or [""])[0]
                    return self._json(app.jobs.get(jid, {"state": "unknown"}))
                if path == "/api/download":
                    jid = (q.get("id") or [""])[0]
                    job = app.jobs.get(jid) or {}
                    p = Path(job.get("path") or "")
                    if not p.exists():
                        return self._json({"error": "not ready"}, 404)
                    return self._send(200, p.read_bytes(), "video/mp4", {
                        "Content-Disposition": f'attachment; filename="{p.name}"'})
                if path == "/api/projects":
                    return self._json({"projects": [
                        {k: v for k, v in p.to_dict().items()
                         if k in ("id", "name", "modified", "preset", "seed")}
                        for p in app.store.projects.values()]})
                return self._json({"error": "not found"}, 404)
            except Exception as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}",
                                   "trace": traceback.format_exc()[-1500:]}, 500)

        def do_POST(self) -> None:
            u = urlparse(self.path)
            path = u.path
            try:
                if path == "/api/sweep":
                    b = self._body()
                    layer = str(b.get("layer", "tape"))
                    key = str(b.get("key", ""))
                    vals = [float(v) for v in (b.get("values") or [])]
                    if not key or not vals:
                        return self._json({"error": "需要 layer、key 与 values"})
                    if len(vals) > 24:
                        return self._json({"error": "一次最多 24 个取值"})
                    sec = None
                    if b.get("second"):
                        s2 = b["second"]
                        v2 = [float(v) for v in (s2.get("values") or [])]
                        if s2.get("key") and v2 and len(vals) * len(v2) <= 36:
                            sec = (str(s2.get("layer", layer)), str(s2["key"]), v2)
                    t0 = time.time()
                    out = app.sweep(layer, key, vals, float(b.get("t", 0.0)),
                                    scale=float(b.get("scale", 0.35)), second=sec)
                    out["ms"] = int((time.time() - t0) * 1000)
                    return self._json(out)
                if path == "/api/project":
                    return self._json(app.set_project(self._body()))
                if path == "/api/preset":
                    b = self._body()
                    return self._json(app.apply_preset(str(b.get("key", ""))))
                if path == "/api/preset/save":
                    return self._json(app.save_custom_preset(self._body()))
                if path == "/api/preset/delete":
                    return self._json(app.delete_custom_preset(self._body()))
                if path == "/api/upload":
                    n = int(self.headers.get("Content-Length") or 0)
                    raw_name = self.headers.get("X-Filename") or "clip.mp4"
                    # Accept stills as well as clips.  A photograph is a clip of
                    # length one downstream, so the only thing that has to differ
                    # is how the file is probed -- and `source_for` decides that
                    # from the extension.
                    from urllib.parse import unquote
                    from ..io.imagecodec import image_size, is_image_path
                    name = Path(unquote(raw_name)).name
                    dest = app.store.uploads / name
                    with open(dest, "wb") as fh:
                        remaining = n
                        while remaining > 0:
                            chunk = self.rfile.read(min(1 << 20, remaining))
                            if not chunk:
                                break
                            fh.write(chunk)
                            remaining -= len(chunk)
                    if is_image_path(dest):
                        size = image_size(dest)
                        if size is None:
                            return self._json({"error": f"无法读取图片：{name}"})
                        w, h = size
                        from ..io.ffmpeg import VideoInfo
                        info = VideoInfo(path=str(dest), width=w, height=h,
                                         fps=app.current.fps, frame_count=1,
                                         duration=1.0 / max(1.0, app.current.fps),
                                         has_audio=False, codec="still")
                        app.set_project({"source": str(dest)})
                        return self._json({"ok": True, "still": True,
                                           "info": info.to_dict(),
                                           "project": app.project_dict()})
                    from ..io.ffmpeg import probe
                    info = probe(dest)
                    app.set_project({"source": str(dest)})
                    return self._json({"ok": True, "info": info.to_dict(),
                                       "project": app.project_dict()})
                if path == "/api/upload_overlay":
                    n = int(self.headers.get("Content-Length") or 0)
                    raw_name = self.headers.get("X-Filename") or "overlay.png"
                    from urllib.parse import unquote
                    name = "overlay_" + Path(unquote(raw_name)).name
                    dest = app.store.uploads / name
                    with open(dest, "wb") as fh:
                        remaining = n
                        while remaining > 0:
                            chunk = self.rfile.read(min(1 << 20, remaining))
                            if not chunk:
                                break
                            fh.write(chunk)
                            remaining -= len(chunk)
                    cur_cn = dict((app.current.params or {}).get("connector") or {})
                    cur_mode = float(cur_cn.get("switcher_mode", 0.0))
                    patch_cn = {
                        "switcher_overlay_path": str(dest),
                        "switcher_b_source": 3.0,
                    }
                    if cur_mode < 0.5:
                        patch_cn["switcher_mode"] = 5.0
                    proj_out = app.set_project({"params": {"connector": patch_cn}})
                    return self._json({"ok": True, "overlay_path": str(dest), "project": proj_out})
                if path == "/api/clear_overlay":
                    proj_out = app.set_project({"params": {"connector": {
                        "switcher_overlay_path": "",
                        "switcher_b_source": 0.0,
                    }}})
                    return self._json({"ok": True, "project": proj_out})
                if path == "/api/graph":
                    return self._json(app.graph_set(self._body()))
                if path == "/api/graph/preset":
                    return self._json(app.graph_load_preset(str(self._body().get("key", ""))))
                if path == "/api/graph/add":
                    return self._json(app.graph_add(self._body()))
                if path == "/api/graph/connect":
                    return self._json(app.graph_connect(self._body()))
                if path == "/api/graph/disconnect":
                    return self._json(app.graph_disconnect(self._body()))
                if path == "/api/graph/remove":
                    return self._json(app.graph_remove(self._body()))
                if path == "/api/graph/move":
                    return self._json(app.graph_move(self._body()))
                if path == "/api/graph/node":
                    return self._json(app.graph_node_params(self._body()))
                if path == "/api/graph/group":
                    return self._json(app.graph_group_nodes(self._body()))
                if path == "/api/graph/ungroup":
                    return self._json(app.graph_ungroup_node(self._body()))
                if path == "/api/graph/duplicate":
                    return self._json(app.graph_duplicate_node(self._body()))
                if path == "/api/graph/group/node":
                    return self._json(app.graph_subnode_params(self._body()))
                if path == "/api/graph/chain_duplicate":
                    return self._json(app.graph_chain_duplicate(self._body()))
                if path == "/api/import":
                    b = self._body()
                    # accept either the parsed object or the raw text of a file
                    if isinstance(b.get("config"), dict):
                        b = b["config"]
                    elif isinstance(b.get("text"), str):
                        b = json.loads(b["text"])
                    return self._json(app.import_config(b))
                if path == "/api/save":
                    b = self._body()
                    if b.get("name"):
                        app.current.name = str(b["name"])
                    p = app.store.save(app.current)
                    return self._json({"ok": True, "project": p.to_dict()})
                if path == "/api/load":
                    b = self._body()
                    p = app.store.projects.get(str(b.get("id", "")))
                    if p is None:
                        return self._json({"error": "no such project"}, 404)
                    with app.lock:
                        app.current = p
                        app.store.drop(p.id)
                    return self._json(app.project_dict())
                return self._json({"error": "not found"}, 404)
            except Exception as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}",
                                   "trace": traceback.format_exc()[-1500:]}, 500)

        # -- static ----------------------------------------------------
        def _static(self, rel: str) -> None:
            p = (WEB_DIR / rel).resolve()
            if not str(p).startswith(str(WEB_DIR.resolve())) or not p.is_file():
                return self._json({"error": "not found"}, 404)
            ctype = {
                ".html": "text/html; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".svg": "image/svg+xml",
            }.get(p.suffix, "application/octet-stream")
            self._send(200, p.read_bytes(), ctype)

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8931, data_dir: str | Path = "runs",
          open_browser: bool = False, default_clip: str | None = None) -> None:
    # Report the tools at startup, with the paths actually resolved.  Discovering
    # a missing FFmpeg only when the user tries to import a clip produces a
    # failure that looks like the *file* is at fault ("系统找不到指定的文件"),
    # when in fact the tool that reads it could not be run.
    from ..io.ffmpeg import missing_tools, tool_paths
    ffmpeg_path, ffprobe_path = tool_paths()
    missing = missing_tools()
    app = App(Path(data_dir), default_clip=default_clip)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{port}/"
    print("=" * 70)
    print("  Virtual Media Lab  --  虚拟媒介实验室")
    print("=" * 70)
    print(f"  WebUI:  {url}")
    print(f"  数据目录: {Path(data_dir).resolve()}")
    if app.current.source:
        print(f"  素材:   {app.current.source}")
    else:
        print("  素材:   未导入（正在使用内置测试图；可用「导入素材」按钮添加视频）")
    if missing:
        print(f"  !! 缺少工具: {', '.join(missing)} —— 导入/导出/离线渲染会失败")
        print(f"     已尝试的路径: ffmpeg={ffmpeg_path}")
        print(f"                   ffprobe={ffprobe_path}")
        print("     设置 ATSIM_FFMPEG / ATSIM_FFPROBE 指向可执行文件，或装到 C:\\ffmpeg\\bin")
    else:
        print(f"  ffmpeg: {ffmpeg_path}")
        print(f"  ffprobe:{ffprobe_path}")
    print("  按 Ctrl+C 停止")
    print("=" * 70)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止。")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="atsim-web", description="Virtual Media Lab (虚拟媒介实验室) WebUI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8931)
    ap.add_argument("--data-dir", default="runs")
    ap.add_argument("--clip", default=None, help="video file to load at startup")
    ap.add_argument("--open", action="store_true", help="open a browser window")
    args = ap.parse_args(argv)
    serve(args.host, args.port, args.data_dir, args.open, args.clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
