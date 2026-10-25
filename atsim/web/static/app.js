/* Virtual Media Lab (虚拟媒介实验室) WebUI
 *
 * The page is deliberately schema-driven: every control is generated from
 * /api/schema, so adding a parameter to the Python schema makes it appear here
 * with its unit, its physical range and its provenance.  That is what keeps the
 * UI and the signal model from drifting apart.
 */
"use strict";

const S = {
  schema: null,
  catalog: null,
  project: null,
  params: {},
  layer: "format",
  presetTab: "factory",
  t: 0.4,
  previewScale: "0.5",
  previewQuality: 88,
  exportScale: "1.0",
  exportQuality: "high",
  ab: true,
  auto: true,
  busy: false,
  pending: false,
  token: 0,
  frameUrl: null,
  origUrl: null,
  dirtyStages: {},
};

const $ = (id) => document.getElementById(id);
const fmt = (v) => {
  if (v === null || v === undefined) return "";
  const n = typeof v === "number" ? v : Number(v);
  if (isNaN(n)) return String(v);
  const a = Math.abs(n);
  return a >= 100 ? n.toFixed(0)
    : a >= 10 ? n.toFixed(1)
    : a >= 1 ? n.toFixed(2) : n.toFixed(3);
};

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.error || j.trace || msg; } catch (e) {}
    throw new Error(msg);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r.blob();
}

/* ------------------------------------------------------------------ boot */
async function boot() {
  S.schema = await api("/api/schema");
  S.catalog = (await api("/api/catalog")).catalog;
  applyProject(await api("/api/project"));
  buildLayerTabs();
  buildPresetTabs();
  renderPresetList();
  renderParams();
  wireChrome();
  refreshPreview();
  loadTimelineMeta();
  // the canvas lives in canvas.js; it needs the schema before it can render a
  // node inspector, so it is initialised after the schema arrives
  if (typeof initCanvas === "function") {
    initCanvas();
    await loadGraph();
    // open the canvas only once the graph is loaded, otherwise it would draw
    // from an empty state
    if (location.hash === "#canvas" && typeof window.showCanvas === "function")
      window.showCanvas();
  }
}

function applyProject(d) {
  S.project = d.project;
  S.params = d.params;
  $("seed").value = d.project.seed;
  const info = d.source_info || {};
  $("badge").textContent = d.project.source
    ? `${(info.width||"?")}×${(info.height||"?")} · ${(info.fps||0).toFixed(3)} fps · ${info.frame_count||"?"} 帧`
    : "内置测试图（未导入素材）";
  // 维度面板：环境 / 设备个体 / 关键帧时间线。放在最后，
  // 所以面板反映的是服务端刚刚返回的状态——只有一份真相。
  syncWTBar();
  syncQuadBar();
  const clrOvBtn = $("btnClearOverlay");
  if (clrOvBtn) {
    const ovPath = (S.params && S.params.connector && S.params.connector.switcher_overlay_path) || "";
    clrOvBtn.classList.toggle("hidden", !ovPath);
  }
  if (window.SessionUI) SessionUI.syncFromProject(d);
}

function updateWTReadout(zVal) {
  const cam = (S.params && S.params.camera) || {};
  const z = Math.min(1, Math.max(0, zVal !== undefined ? zVal : (cam.wt_zoom || 0)));
  const maxZ = cam.wt_max_zoom || 4.0;
  const mag = Math.pow(maxZ, z);
  const fnum = 1.4 + (2.9 - 1.4) * Math.pow(z, 1.15);
  const state = z < 0.15 ? "W 端光晕态" : (z < 0.55 ? "中焦过渡态" : "T 端具体光源细节");
  const rd = $("wtReadout");
  if (rd) rd.textContent = `${mag.toFixed(2)}× · f/${fnum.toFixed(1)} · ${state}`;
}

function syncWTBar() {
  const cam = (S.params && S.params.camera) || {};
  const sl = $("wtZoomSlider");
  if (sl && document.activeElement !== sl) sl.value = cam.wt_zoom !== undefined ? cam.wt_zoom : 0;
  const dm = $("wtDynMode");
  if (dm) dm.value = String(Math.round(cam.zoom_dynamic_mode || 0));
  const osd = $("wtOsdToggle");
  if (osd) osd.checked = (cam.zoom_osd || 0) > 0.5;
  updateWTReadout(cam.wt_zoom || 0);
}

/* --------------------------------------------------------------- chrome */
function wireChrome() {
  $("btnImport").onclick = () => $("clipFile").click();
  $("clipFile").onchange = uploadClip;
  const btnOv = $("btnImportOverlay");
  const ovFile = $("overlayFile");
  if (btnOv && ovFile) {
    btnOv.onclick = () => ovFile.click();
    ovFile.onchange = uploadOverlay;
  }
  const btnClrOv = $("btnClearOverlay");
  if (btnClrOv) {
    btnClrOv.onclick = clearOverlay;
  }
  const still = $("btnStill");
  if (still) still.onclick = exportStill;
  const stillIn = $("btnQuadStillInline");
  if (stillIn) stillIn.onclick = exportStill;
  const burstBtn = $("btnQuadBurst");
  if (burstBtn) burstBtn.onclick = exportQuadBurst;
  const burstIn = $("btnQuadBurstInline");
  if (burstIn) burstIn.onclick = exportQuadBurst;
  const zipBtn = $("btnQuadZip");
  if (zipBtn) zipBtn.onclick = exportQuadZip;
  const zipIn = $("btnQuadZipInline");
  if (zipIn) zipIn.onclick = exportQuadZip;
  $("seed").onchange = () => setProject({ seed: parseInt($("seed").value, 10) || 0 });
  $("timeSlider").oninput = () => {
    S.t = parseFloat($("timeSlider").value);
    $("timeLabel").textContent = `t = ${S.t.toFixed(2)} s`;
    renderParams();
    if (window.SessionUI) {
      if (typeof SessionUI.renderKeyframes === "function") SessionUI.renderKeyframes();
      if (typeof SessionUI.renderEnv === "function") SessionUI.renderEnv();
    }
    refreshPreview();
  };
  const pScale = $("previewScale");
  if (pScale) {
    pScale.onchange = () => {
      S.previewScale = pScale.value || "0.5";
      refreshPreview();
    };
  }
  const pQual = $("previewQuality");
  if (pQual) {
    pQual.onchange = () => {
      S.previewQuality = parseInt(pQual.value, 10) || 88;
      refreshPreview();
    };
  }
  const eScale = $("exportScale");
  if (eScale) {
    eScale.onchange = () => { S.exportScale = eScale.value || "1.0"; };
  }
  const eQual = $("exportQuality");
  if (eQual) {
    eQual.onchange = () => { S.exportQuality = eQual.value || "high"; };
  }
  $("abToggle").onchange = () => { S.ab = $("abToggle").checked; layoutAB(); if (S.ab) loadOriginal(); };
  $("autoRefresh").onchange = () => { S.auto = $("autoRefresh").checked; };
  $("btnRender").onclick = startRender;
  $("btnSave").onclick = () => api("/api/save", jsonPost({ name: S.project.name }));

  // **Configuration export and import.**  Export is a navigation rather than a fetch: the route sends the file with
  // a Content-Disposition header and a timestamped name, so the browser saves it without any code here.  Import has
  // to go through a file input, since a page cannot read a path the user did not choose.
  //
  // The whole point of the pair is that a reproduction can be handed over, so the import reports what it applied
  // rather than failing silently -- an import that half-applies is worse than one that refuses.
  const btnExport = $("btnExport");
  if (btnExport) {
    btnExport.onclick = () => {
      const a = document.createElement("a");
      a.href = "/api/export";
      a.download = "";
      document.body.appendChild(a);
      a.click();
      a.remove();
      toast("已导出设置文件");
    };
  }
  const btnImportCfg = $("btnImportCfg");
  const cfgFile = $("cfgFile");
  if (btnImportCfg && cfgFile) {
    btnImportCfg.onclick = () => cfgFile.click();
    cfgFile.onchange = async () => {
      const f = cfgFile.files && cfgFile.files[0];
      if (!f) return;
      try {
        const text = await f.text();
        // parsed here only to fail early with a clear message; the server parses it again
        const parsed = JSON.parse(text);
        if (parsed.kind && parsed.kind !== "atsim.config") {
          throw new Error(`这不是本程序的配置文件（kind=${parsed.kind}）`);
        }
        const r = await api("/api/import", jsonPost({ config: parsed }));
        if (r && r.error) throw new Error(r.error);
        const n = (r && r.imported && r.imported.params_applied) || 0;
        toast(`已导入 ${n} 项设置${parsed.exported ? "（导出于 " + parsed.exported + "）" : ""}`);
      } catch (err) {
        toast("导入失败：" + (err && err.message ? err.message : err));
      } finally {
        cfgFile.value = "";
      }
    };
  }
  $("btnTimeline").onclick = showTimeline;
  $("tlClose").onclick = () => $("modal").classList.add("hidden");
  $("jobClose").onclick = () => $("jobbar").classList.add("hidden");
  wireWTBar();
  wireQuadBar();
  initDivider();
  layoutAB();
  initAudioMonitor();
  if (window.BroadcastScopes) BroadcastScopes.init();
}

function wireWTBar() {
  const sl = $("wtZoomSlider");
  const btnW = $("btnZoomW");
  const btnT = $("btnZoomT");
  const dm = $("wtDynMode");
  const osd = $("wtOsdToggle");
  if (!sl) return;

  const applyZ = (nv, debounce) => {
    const z = Math.min(1, Math.max(0, parseFloat(nv) || 0));
    sl.value = z;
    S.params.camera = S.params.camera || {};
    S.params.camera.wt_zoom = z;
    updateWTReadout(z);
    if (S.layer === "camera") renderParams();
    setParam("camera", "wt_zoom", z, debounce);
  };

  sl.oninput = () => applyZ(sl.value, true);
  sl.onchange = () => applyZ(sl.value, false);

  let rockerTimer = null;
  const startRocker = (dir) => {
    clearInterval(rockerTimer);
    const stepOnce = () => {
      const cur = parseFloat(sl.value) || 0;
      const next = Math.min(1, Math.max(0, cur + dir * 0.04));
      applyZ(next, true);
      if (next <= 0 || next >= 1) stopRocker();
    };
    stepOnce();
    rockerTimer = setInterval(stepOnce, 75);
  };
  const stopRocker = () => {
    if (rockerTimer) {
      clearInterval(rockerTimer);
      rockerTimer = null;
      applyZ(sl.value, false);
    }
  };
  if (btnW) {
    btnW.onmousedown = () => startRocker(-1);
    btnW.ontouchstart = (e) => { e.preventDefault(); startRocker(-1); };
  }
  if (btnT) {
    btnT.onmousedown = () => startRocker(1);
    btnT.ontouchstart = (e) => { e.preventDefault(); startRocker(1); };
  }
  window.addEventListener("mouseup", stopRocker);
  window.addEventListener("touchend", stopRocker);

  if (dm) {
    dm.onchange = () => {
      const v = parseFloat(dm.value) || 0;
      S.params.camera = S.params.camera || {};
      S.params.camera.zoom_dynamic_mode = v;
      setProject({ params: { camera: { zoom_dynamic_mode: v } } });
    };
  }
  if (osd) {
    osd.onchange = () => {
      const v = osd.checked ? 1.0 : 0.0;
      S.params.camera = S.params.camera || {};
      S.params.camera.zoom_osd = v;
      setProject({ params: { camera: { zoom_osd: v } } });
    };
  }
}

function jsonPost(obj) {
  return { method: "POST", headers: { "Content-Type": "application/json" },
           body: JSON.stringify(obj) };
}

S.abSplit = 0.5;

function initDivider() {
  const v = $("viewer"), d = $("divider");
  let drag = false;
  const move = (e) => {
    if (!drag || !e) return;
    const a = $("previewImg");
    const box = (a && a.clientWidth > 10) ? a.getBoundingClientRect() : v.getBoundingClientRect();
    const x = Math.min(Math.max((e.clientX - box.left) / Math.max(1, box.width), 0), 1);
    S.abSplit = x;
    syncAB();
  };
  d.onmousedown = (e) => { e.preventDefault(); drag = true; };
  window.addEventListener("mouseup", () => drag = false);
  window.addEventListener("mousemove", move);
  d.ontouchstart = () => { drag = true; };
  d.ontouchmove = (e) => { if (e.touches && e.touches[0]) move(e.touches[0]); };
  window.addEventListener("resize", () => syncAB());
  if (typeof ResizeObserver !== "undefined" && v) {
    const ro = new ResizeObserver(() => syncAB());
    ro.observe(v);
  }
}

function layoutAB() {
  $("originalImg").style.display = S.ab ? "block" : "none";
  $("divider").style.display = S.ab ? "block" : "none";
  $("hint").style.display = S.ab ? "block" : "none";
  syncAB();
}

/**
 * Force the two images AND the A/B divider onto one identical centered rectangle,
 * dynamically re-centering whenever the viewer resizes (fixing collapsed drawer UI bug).
 */
function syncAB() {
  const v = $("viewer"), a = $("previewImg"), b = $("originalImg"), d = $("divider");
  if (!v || !a || !b) return;
  const vw = v.clientWidth, vh = v.clientHeight;
  const rw = S.frameW || a.naturalWidth || 4;
  const rh = S.frameH || a.naturalHeight || 3;
  if (!vw || !vh || !rw || !rh) return;
  const scale = Math.min(vw / rw, vh / rh);
  const w = Math.max(1, Math.round(rw * scale));
  const h = Math.max(1, Math.round(rh * scale));
  const left = Math.round((vw - w) / 2);
  const top = Math.round((vh - h) / 2);
  for (const el of [a, b]) {
    el.style.width = w + "px";
    el.style.height = h + "px";
    el.style.left = left + "px";
    el.style.top = top + "px";
    el.style.right = "auto";
    el.style.bottom = "auto";
    el.style.objectFit = "fill";
  }
  const split = (typeof S.abSplit === "number") ? S.abSplit : 0.5;
  b.style.clipPath = `inset(0 0 0 ${(split * 100).toFixed(2)}%)`;
  if (d) {
    d.style.top = top + "px";
    d.style.height = h + "px";
    d.style.bottom = "auto";
    d.style.left = Math.round(left + w * split) + "px";
  }
}
window.syncAB = syncAB;

/* --------------------------------------------------------------- tabs */
function buildLayerTabs() {
  // Drive the tabs from the *schema*, not a hardcoded list: the schema is the
  // single source of truth for which device layers exist, so adding a layer
  // (as happened with `codec`) cannot leave the UI out of step with the engine.
  const order = (S.schema.layers || []).map((l) => l.layer);
  const labels = S.catalog.layers || {};
  $("layerTabs").innerHTML = "";
  order.forEach((k) => {
    const b = document.createElement("button");
    // never trust a label to exist: a missing one must degrade, not crash
    b.textContent = (labels[k] || k).replace(/^(\d+)\s*/, "$1 ");
    b.className = k === S.layer ? "on" : "";
    b.onclick = () => { S.layer = k; buildLayerTabs(); renderParams(); };
    $("layerTabs").appendChild(b);
  });
  if (!order.includes(S.layer) && order.length) S.layer = order[0];
  renderChainBar();
}

function buildPresetTabs() {
  const tabs = [
    ["factory", "整套配置"],
    ["digital", "古早数字/手机/流媒体"],
    ["custom", "自定义预设"],
    ["film", "胶片/LOMO"],
    ["media", "记录媒介"],
    ["player", "播放设备"],
    ["display", "显示设备"],
    ["camera", "摄像机"]
  ];
  $("presetTabs").innerHTML = "";
  tabs.forEach(([k, label]) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.className = k === S.presetTab ? "on" : "";
    b.onclick = () => { S.presetTab = k; buildPresetTabs(); renderPresetList(); };
    $("presetTabs").appendChild(b);
  });
}

function presetEntries() {
  if (S.presetTab === "factory") {
    return Object.entries(S.catalog.factory || {}).map(([k, v]) => [k, v.label, v.note || "整套：摄像机+格式+介质+播放（默认无CRT）"]);
  }
  if (S.presetTab === "custom") {
    return Object.entries(S.catalog.custom || {}).map(([k, v]) => [k, v.label, v.note || "用户自定义保存预设"]);
  }
  if (S.presetTab === "digital") {
    const profs = (S.catalog.profiles || {}).digital || {};
    const pList = Object.entries(profs).map(([k, v]) => [k, v.label, v.note || "古早数字/手机/流媒体预设"]);
    if (pList.length > 0) return pList;
    return Object.entries(S.catalog.factory || {})
      .filter(([k]) => k.startsWith("ldme_"))
      .map(([k, v]) => [k, v.label, v.note || "古早数字媒介配置"]);
  }
  const g = (S.catalog.profiles || {})[S.presetTab] || {};
  return Object.entries(g).map(([k, v]) => [k, v.label, v.note || ""]);
}

async function saveCustomPreset(labelInput, noteInput) {
  const label = (labelInput ? labelInput.value : "").trim() || prompt("请输入自定义预设名称：", S.project?.name || "我的 VHS 预设");
  if (!label) return;
  const note = (noteInput ? noteInput.value : "").trim();
  try {
    const res = await api("/api/preset/save", jsonPost({ label, note }));
    if (res.catalog) S.catalog = res.catalog;
    if (res.project) applyProject(res.project);
    S.presetTab = "custom";
    buildPresetTabs();
    renderPresetList();
    toast(`已保存自定义预设「${label}」`);
  } catch (e) {
    toast("保存预设失败：" + e.message);
  }
}

async function deleteCustomPreset(key, label, ev) {
  if (ev) ev.stopPropagation();
  try {
    const res = await api("/api/preset/delete", jsonPost({ key }));
    if (res.catalog) S.catalog = res.catalog;
    if (res.project) applyProject(res.project);
    renderPresetList();
    toast(`已删除预设「${label}」`);
  } catch (e) {
    toast("删除预设失败：" + e.message);
  }
}

function renderPresetList() {
  const box = $("presetList");
  box.innerHTML = "";

  // Inline bar to save current parameters + node graph as a user custom preset
  const saveBar = document.createElement("div");
  saveBar.className = "custom-preset-bar";
  saveBar.innerHTML = `
    <input type="text" id="customPresetName" class="cp-input" placeholder="输入名称保存当前参数为预设…" />
    <button type="button" id="btnSaveCustomPreset" class="cp-btn">+ 保存预设</button>
  `;
  const nameIn = saveBar.querySelector("#customPresetName");
  const saveBtn = saveBar.querySelector("#btnSaveCustomPreset");
  saveBtn.onclick = () => saveCustomPreset(nameIn, null);
  nameIn.onkeydown = (e) => { if (e.key === "Enter") saveCustomPreset(nameIn, null); };
  box.appendChild(saveBar);

  const entries = presetEntries();
  if (!entries.length && S.presetTab === "custom") {
    const empty = document.createElement("div");
    empty.className = "phelp";
    empty.style.padding = "8px 4px";
    empty.textContent = "暂无自定义预设。调节任意参数或节点连线后，在上方输入名称点击「+ 保存预设」即可永久保存。";
    box.appendChild(empty);
    return;
  }

  entries.forEach(([key, label, note]) => {
    const d = document.createElement("div");
    d.className = "card" + (S.project.preset === key ? " on" : "");
    d.innerHTML = `<div class="card-head"><div class="t"></div></div><div class="n"></div>`;
    d.querySelector(".t").textContent = label;
    d.querySelector(".n").textContent = note;
    if (key.startsWith("custom_") || S.presetTab === "custom") {
      const del = document.createElement("button");
      del.className = "cp-del";
      del.title = "删除此自定义预设";
      del.textContent = "×";
      del.onclick = (ev) => deleteCustomPreset(key, label, ev);
      d.querySelector(".card-head").appendChild(del);
    }
    d.onclick = () => applyPreset(key);
    box.appendChild(d);
  });
}

/* --------------------------------------------------------------- params */
function renderParams() {
  const layer = (S.schema.layers || []).find((l) => l.layer === S.layer);
  const box = $("paramList");
  box.innerHTML = "";
  if (!layer) return;
  Object.entries(layer.groups).forEach(([gname, specs]) => {
    const g = document.createElement("div");
    g.className = "pgroup";
    const h = document.createElement("h3");
    h.textContent = gname;
    g.appendChild(h);
    specs.forEach((sp) => g.appendChild(paramRow(sp)));
    box.appendChild(g);
  });
}

// **Parameters that are selectors rather than quantities.**  A night-vision mode is one of four named functions of the
// camera, so a slider reading "1.7" tells the user nothing -- and the schema has no `choices` field, so the names live
// here.  The control still calls `setParam(layer, key, value)` with a number, so the API, the settings file, the
// keyframes and the render path are all unchanged: only the widget differs.
const SELECTORS = {
  "rf.tx_quad_corner": [
    [0, "0 · 2×2 四分屏全览"],
    [1, "1 · 左上角（主控直通基带）"],
    [2, "2 · 右上角（发射机天线探头）"],
    [3, "3 · 左下角（模拟开路接收）"],
    [4, "4 · 右下角（数字地面 DTT 接收）"],
  ],
  "rf.tx_show_border": [
    [0, "0 · 无分框（无缝拼接）"],
    [1, "1 · 细十字分框线"],
    [2, "2 · 标准演播室分框"],
  ],
  "rf.tx_show_text": [
    [0, "0 · 不要文字（纯净画面）"],
    [1, "1 · 仅显示角落标签"],
    [2, "2 · 显示角落标签 + 底部遥测栏"],
  ],
  "rf.sat_ird_osd": [
    [0, "0 · 不要文字（纯净画面）"],
    [1, "1 · 仅显示角落标签"],
    [2, "2 · 显示完整 C/N 与 TS 统计文字"],
  ],
  "rf.sat_quad_corner": [
    [0, "0 · 2×2 四分屏全览"],
    [1, "1 · 左上角（晴空基带基准）"],
    [2, "2 · 右上角（模拟 C 波段 FM）"],
    [3, "3 · 左下角（数字 DVB-S 悬崖）"],
    [4, "4 · 右下角（C/N 悬崖曲线面板）"],
  ],
  "rf.sat_show_border": [
    [0, "0 · 无分框（无缝拼接）"],
    [1, "1 · 细十字分框线"],
    [2, "2 · 标准演播室分框"],
  ],
  "player.transfer_quad_corner": [
    [0, "0 · 2×2 四分屏全览"],
    [1, "1 · 左上角（Gen-0 母带基准）"],
    [2, "2 · 右上角（路径 A 无 TBC 多代模拟复制）"],
    [3, "3 · 左下角（路径 B 带 TBC 多代模拟复制）"],
    [4, "4 · 右下角（路径 C 数字格式转码链）"],
  ],
  "player.transfer_show_border": [
    [0, "0 · 无分框（无缝拼接）"],
    [1, "1 · 细十字分框线"],
    [2, "2 · 标准演播室分框"],
  ],
  "player.transfer_show_text": [
    [0, "0 · 不要文字（纯净画面）"],
    [1, "1 · 仅显示角落标签"],
    [2, "2 · 显示角落标签 + 底部统计栏"],
  ],
  "player.inv_quad_corner": [
    [0, "0 · 2×2 四分屏全览"],
    [1, "1 · 左上角"],
    [2, "2 · 右上角"],
    [3, "3 · 左下角"],
    [4, "4 · 右下角"],
  ],
  "player.inv_show_border": [
    [0, "0 · 无分框（无缝拼接）"],
    [1, "1 · 细十字分框线"],
    [2, "2 · 标准演播室分框"],
  ],
  "player.inv_show_text": [
    [0, "0 · 不要文字（纯净画面）"],
    [1, "1 · 仅显示角落标签"],
    [2, "2 · 显示角落标签 + 底部统计栏"],
  ],
  "camera.telecine_quad_corner": [
    [0, "0 · 2×2 四分屏全览"],
    [1, "1 · 左上角（16mm/35mm 胶片原片）"],
    [2, "2 · 右上角（电视胶转磁母带）"],
    [3, "3 · 左下角（家用 VHS 磁带录屏）"],
    [4, "4 · 右下角（早期网络流媒体重压）"],
  ],
  "camera.telecine_show_border": [
    [0, "0 · 无分框（无缝拼接）"],
    [1, "1 · 细十字分框线"],
    [2, "2 · 标准演播室分框"],
  ],
  "camera.telecine_show_text": [
    [0, "0 · 不要文字（纯净画面）"],
    [1, "1 · 仅显示角落标签"],
    [2, "2 · 显示角落标签 + 底部状态栏"],
  ],
  "display.mv_quad_corner": [
    [0, "0 · 完整多画面监视墙"],
    [1, "1 · 左上角窗口 (CH1)"],
    [2, "2 · 右上角窗口 (CH2)"],
    [3, "3 · 左下角窗口 (CH3)"],
    [4, "4 · 右下角窗口 (CH4)"],
    [5, "5 · 第 5 窗口 (CH5)"],
    [6, "6 · 第 6 窗口 (CH6)"],
  ],
  "display.mv_show_border": [
    [0, "0 · 无分框（无缝拼接）"],
    [1, "1 · 细十字分框线"],
    [2, "2 · 演播室金属机架边框"],
  ],
  "film.stock": [
    [0, "0 · 关闭独立胶片光化学模块"],
    [1, "1 · Eastman Color 35mm 5247 (经典电影负片+2383印片)"],
    [2, "2 · Kodak Ektachrome 16mm 7240 (电视台新闻反转片)"],
    [3, "3 · Fujicolor 16mm 8514 (第4感青层翡翠纪实片)"],
    [4, "4 · 早期醋酸片基暖红棕褪色档案拷贝"],
    [5, "5 · Agfa Vista / Optima 200 (琥珀暖金与深可可红暗部)"],
    [6, "6 · Kodak Ektachrome E100VS / Elite Chrome 100 (高饱和度反转片)"],
    [7, "7 · Lomography Color Negative 400 (玫瑰洋红/丁香紫暗部与奶油高光)"],
    [8, "8 · Expired Svema / ORWO Vintage Film (高底雾 D_min 粉彩鼠尾草绿)"],
    [9, "9 · Polaroid SX-70 / 600 Time-Zero Integral Film (即时显影药浆与奶油高光)"],
    [10, "10 · Lomography LomoChrome Purple XR 100-400 (绿→紫二向色耦合平移)"],
    [11, "11 · 90s Tungsten Acid Olive-Chartreuse (90年代港式钨丝灯酸橄榄黄绿迫冲)"],
    [12, "12 · Prussian Cyan-Navy & Electric Violet Rain (普鲁士深青蓝与电光紫雨夜乳剂)"],
    [13, "13 · Fujifilm Pro 400H Pastel Candy High-Key (富士高调粉彩马卡龙专业负片)"],
    [14, "14 · Vintage Brass-Olive & Maroon-Plum Petzval (复古黄铜橄榄与深紫红暗部乳剂)"],
    [15, "15 · LOMO LC-A Minitar-1 Turquoise-Emerald X-Pro (松石青绿高反差正冲负)"],
    [16, "16 · Meyer Trioplan Golden-Violet"],
    [17, "17 · Kodak Vision3 500T 5219 (现代电影彩色负片 · 橙色遮罩+Remjet)"],
    [18, "18 · Kodak Double-X 5222 (经典35mm全色黑白电影负片 · D-96银盐颗粒)"],
    [19, "19 · Kodachrome 64 (历史K-14外偶法彩色反转片 · 高锐度档案级)"],
    [20, "20 · Kodak Ektachrome 100D (现代E-6内偶法彩色反转片 · 直接正像)"],
  ],
  "film.optical_print_gen": [
    [0, "0 · 原始摄影负片直扫 (OCN Direct Scan)"],
    [1, "1 · 一代光学印片 (1st Gen Workprint)"],
    [2, "2 · 二代中间片印片 (2nd Gen OCN → IP → IN)"],
    [3, "3 · 三代院线发行拷贝 (3rd Gen OCN → IP → IN → 2383 Release Print)"],
  ],
  "film.shutter_blades": [
    [2, "2 · 传统双叶片旋转快门 (24fps × 2 = 48Hz 光脉冲)"],
    [3, "3 · 影院三叶片旋转快门 (24fps × 3 = 72Hz 低闪烁光脉冲)"],
  ],
  "film.process_mode": [
    [0, "0 · 标准暗房冲洗 (C-41 / E-6 Standard)"],
    [1, "1 · 正冲负 E-6 in C-41 Xpro (翡翠青绿阴影 + 金盏花暖金高光)"],
    [2, "2 · 负冲正 C-41 in E-6 Xpro (丁香紫/玫瑰洋红阴影 + 奶油高光)"],
    [3, "3 · 迫冲/减冲显影 (Push / Pull Process)"],
    [4, "4 · 留银冲洗 (Bleach Bypass · 高反差金属银保留)"],
    [5, "5 · 2×2 LOMO 光化学暗房四象限对照实验室"],
  ],
  "film.xpro_tint_mode": [
    [0, "0 · 自动跟随冲洗工艺与胶卷型号"],
    [1, "1 · 暗部翡翠青绿 + 高光金盏花暖金"],
    [2, "2 · 暗部丁香紫/玫瑰红 + 高光奶油白"],
    [3, "3 · 暗部可可暖褐 + 中间调琥珀蜜糖"],
  ],
  "film.bokeh_shape": [
    [0, "0 · 关闭 / 标准圆形光圈"],
    [1, "1 · Heart Bokeh 粉红桃心散景光斑 (Praktica / R2M 异形光圈)"],
    [2, "2 · Starburst 四角星芒插片散景"],
    [3, "3 · Diamond 菱形叶片孔径散景"],
    [4, "4 · Petzval Swirl 匹兹伐切向猫眼旋焦散景"],
    [5, "5 · Soap-Bubble Ring 梅耶过校正球差肥皂泡亮环散景"],
    [6, "6 · Hexagonal Fairy-Light 六边形机械光圈奶油彩灯散景"],
  ],
  "film.bloom_veil_mode": [
    [0, "0 · 关闭专项光学光晕/白雾"],
    [1, "1 · 黄金时刻大逆光太阳耀斑洪流"],
    [2, "2 · 无镀膜老镜头高调奶油白雾眩光与柔焦"],
    [3, "3 · 点光源双层暖金日冕与防光晕层红橙背散射"],
    [4, "4 · 斜向双色温冷紫-暖金光学分色光雾"],
    [5, "5 · 2×2 LOMO 光学光晕与散景四象限对照实验室"],
  ],
  "film.light_leak_pattern": [
    [0, "0 · 无漏光"],
    [1, "1 · 右上角金琥珀斜向暖阳漏光"],
    [2, "2 · 左侧铰链海绵老化红橙背散射漏光条"],
    [3, "3 · 横贯画面粉红/洋红桃心光斑串"],
    [4, "4 · 左下与右上双象限梦幻奶油光雾"],
  ],
  "film.telecine_pulldown": [
    [0, "0 · 24fps 逐行直扫 (Direct 24p / 纯胶片扫描)"],
    [1, "1 · NTSC 3:2 Pulldown (24fps→29.97i AA-BB-BC-CD-DD 隔行梳状)"],
    [2, "2 · PAL 25fps 加速胶转磁 (+4.167% 加速与 +70.67 cents 升调)"],
    [3, "3 · 2×2 跨媒介历史对照实验室 (胶片→胶转磁→VHS→流媒体)"],
  ],
  "film.quad_corner": [
    [0, "0 · 2×2 四分屏同屏对照"],
    [1, "1 · 左上角全屏 (E-6正冲负青绿暖金 / 胶片原片)"],
    [2, "2 · 右上角全屏 (C-41负冲正桃心散景 / 3:2胶转磁)"],
    [3, "3 · 左下角全屏 (过期胶卷高底雾粉彩 / 家用VHS录屏)"],
    [4, "4 · 右下角全屏 (宝丽来浓暗角暖阳漏光 / 流媒体重压)"],
  ],
  "film.show_border": [
    [0, "0 · 无分框（无缝拼接）"],
    [1, "1 · 细十字分框线"],
    [2, "2 · 暗房金属外框"],
  ],
  "film.show_text": [
    [0, "0 · 不要文字（纯净画面）"],
    [1, "1 · 仅显示四角短标签"],
    [2, "2 · 显示标签 + 底部暗房遥测栏"],
  ],
  "camera.film_stock": [
    [0, "0 · 关闭电影胶片系统"],
    [1, "1 · Eastman Color 35mm 5247 (经典电影负片+印片)"],
    [2, "2 · Kodak Ektachrome 16mm 7240 (电视台新闻反转片)"],
    [3, "3 · Fujicolor 16mm 8514 (高感光翡翠青暗部纪实片)"],
    [4, "4 · 早期醋酸片基暖红棕褪色档案拷贝"],
  ],
  "camera.telecine_pulldown": [
    [0, "0 · 24fps 逐行直扫 (Direct 24p)"],
    [1, "1 · NTSC 3:2 Pulldown (24fps→29.97i AA-BB-BC-CD-DD 隔行梳状)"],
    [2, "2 · PAL 25fps 加速胶转磁 (+4.167% 加速与 +70.67 cents 升调)"],
    [3, "3 · 2×2 跨媒介历史对照实验室 (胶片→胶转磁→VHS→流媒体)"],
  ],
  "player.inv_lab_mode": [
    [0, "0 · 关闭参数反演与误差预算台"],
    [1, "1 · 参数反演求解器与多解竞争假设 HUD"],
    [2, "2 · 三级参数分离审计 (Physical / Device / Artistic) HUD"],
    [3, "3 · 2×2 参数反演、残差热力图与 FADGI 误差预算实验室"],
  ],
  "player.archival_mode": [
    [0, "0 · 跟随档案库开关"],
    [1, "1 · 原始载体残件 (Original Carrier Artifact · 阿伦尼乌斯温湿度老化)"],
    [2, "2 · 档案保存母版 (Preservation Master · 全画幅无损含 VBI/VITC)"],
    [3, "3 · 过度清洗商业修复版 (Over-Cleaned DNR · 塑料蜡像感与证据链断裂)"],
    [4, "4 · 2×2 多版本谱系树对照实验室 (ArchivalLineageTree + 重要属性损耗表)"],
  ],
  "connector.env_field_mode": [
    [0, "0 · 跟随环境场开关"],
    [1, "1 · 温湿度与结露粘带耦合 (冷机抱入暖房镜片起雾与磁鼓结露)"],
    [2, "2 · 变压器/发射塔近场电磁耦合与接地环路工频干扰"],
    [3, "3 · 房间声学混响 RT60 与窗户/钨丝灯混合光源耦合"],
    [4, "4 · 四场全耦合 + 机房空间坐标探针 HUD"],
    [5, "5 · 2×2 二维空间物理环境场与多维耦合四分屏实验室"],
  ],
  "overlay.auto_v2_mode": [
    [0, "0 · 关闭 / 仅跟随总控开关"],
    [1, "1 · 时间驱动定时顺播 (Time-Driven · 自动 5s VTR Preroll 预卷锁相切播)"],
    [2, "2 · 条件/GPI 信号驱动 (Condition-Driven · 实时监测黑场/静帧/失锁触发跳转)"],
    [3, "3 · 主备冗余倒换闭环 (Redundant Main/Backup · 看门狗超时无缝切备机/垫片)"],
  ],
  "overlay.auto_v2_fault_inject": [
    [0, "0 · 正常顺播无异常 (Normal Schedule)"],
    [1, "1 · MISSING_ASSET (素材缺失 / 磁带未上架 -> 自动降级切应急垫片)"],
    [2, "2 · NOT_READY (磁带预卷未就绪 / 粘带失锁超时 -> 延迟切入或垫片兜底)"],
    [3, "3 · SIGNAL_LOSS (在播主信号突然断流/无同步 -> 看门狗触发主备倒换)"],
    [4, "4 · OVERRIDDEN (导播手动紧急强切 Emergency Take 接管自动化母线)"],
    [5, "5 · FALLBACK (主源静帧/黑场超时 -> 自动无缝倒换至 BACKUP 备路)"],
    [6, "6 · HELD (直播延时挂起 Hold -> 保持当前事件等待手动释放)"],
  ],
  "overlay.auto_v2_console_view": [
    [0, "0 · 主播出母线画面（顶部叠加状态机与故障溯源状态栏）"],
    [1, "1 · 2×2 播出自动化编排与三级故障因果反查控制台视图"],
  ],
  "rf.sat_ird_mode": [
    [0, "0 · 关闭 (不经过卫星接收链路)"],
    [1, "1 · 模拟 C 波段 FM 卫星接收机 (低 C/N 下爆发鱼骨黑白打火噪点 Sparkles)"],
    [2, "2 · 数字 DVB-S QPSK 卫星接收解码器 (QEF 零雪花 vs 悬崖区 MPEG-2 宏块崩坏/静帧)"],
    [3, "3 · 模拟 C 波段 FM vs 数字 DVB-S 同屏悬崖效应对照实验室 (2×2 四象限)"],
  ],
  "rf.sat_ird_osd": [
    [0, "0 · 仅输出解码画面"],
    [1, "1 · 叠加卫星 IRD 工程遥测 HUD (C/N · Eb/N0 · Viterbi/RS 误码率 · 打火率)"],
  ],
  "rf.tx_chain_mode": [
    [0, "0 · 关闭 (不经过电视台大功率发射机激励链)"],
    [1, "1 · 模拟大功率电视发射机 (VSB 残留边带激励 + 高功放 AM-AM/AM-PM + 馈线驻波 + 开路接收)"],
    [2, "2 · 数字地面电视发射机 (COFDM 激励 + 高功放非线性 MER 退化 + 数字接收悬崖效应)"],
    [3, "3 · 端到端开路广播 2×2 四象限对照测试台 (直通参考 / 天线口 / 模拟接收 / 数字接收)"],
  ],
  "player.transfer_station_mode": [
    [0, "0 · 关闭 (正常播放链路)"],
    [1, "1 · A. 模拟磁带数字化采集站 (VHS/Betacam → TBC → BT.601 采集卡)"],
    [2, "2 · B. 磁带多代复制站 (1..8代模拟复制 · TBC抑制抖动 vs 不可逆带宽色度退化)"],
    [3, "3 · C. 数字格式转码站 (4:2:2 → DV25 4:1:1 → VCD 4:2:0 跨格式重编码)"],
    [4, "4 · 四路复制链对照实验室 (2×2 同屏四路复制 + TransferEvent 溯源遥测)"],
  ],
  "player.transfer_tbc_mode": [
    [0, "0 · 无 TBC 直通 (Free-Run · 采集滑移与多代抖动发散)"],
    [1, "1 · 行时基校正器 (Line TBC · 逐行同步与色同步重锁定)"],
    [2, "2 · 全帧同步 TBC + Genlock (Frame Sync · 锁定行场时基与音频基准)"],
  ],
  "player.transfer_codec_target": [
    [0, "0 · BT.601 YUV422 8-bit 采集直出"],
    [1, "1 · DV25 (4:1:1 横向 1/4 色度抽样 + 25Mbps 8×8 DCT)"],
    [2, "2 · MPEG-1 Video CD (352×240 CIF + 4:2:0 色度 + 1.15Mbps)"],
    [3, "3 · 4:2:2 → DV25 (4:1:1) → VCD (4:2:0) 跨格式级联转码 (1/8 色度交集)"],
  ],
  "tape.edit_mode": [
    [0, "0 · 关闭 (无剪辑介入)"],
    [1, "1 · 线性组合编辑 (Assemble Edit · 切点全宽抹音闪黑与尾端 CTL 断轨)"],
    [2, "2 · 线性插入编辑 (Insert Edit · 保留连续控制轨 + 旋转飞抹磁头替换)"],
    [3, "3 · 早期非线编 · 实时预览 (NLE Preview · 半分辨率代理降级与缓存掉帧重复)"],
    [4, "4 · 早期非线编 · 最终渲染导出 (NLE Export · 无掉帧全分辨率 + 转场二次 DCT 重编码)"],
  ],
  "tape.edit_flying_erase": [
    [0, "0 · 无飞抹磁头 / 固定抹音头老化 (切点斜向旧轨残留重影 + 180° 彩虹差拍)"],
    [1, "1 · 旋转飞抹磁头正常 (Flying Erase Head · 逐轨干净替换)"],
  ],
  "tape.edit_console_osd": [
    [0, "0 · 仅输出节目画面"],
    [1, "1 · 叠加 Sony BVE-9100 线性编辑 / NLE 多轨时间线工程状态栏"],
  ],
  "camera.zoom_dynamic_mode": [
    [0, "手动固定 W–T 位置"],
    [1, "W→T 电动推进放大（光晕→具体光源）"],
    [2, "T→W 电动拉远缩小（具体光源→光晕）"],
    [3, "W↔T 往复变焦测试"],
  ],
  "camera.ccu_cam_id": [
    [0, "0 · CAM-1 Sony BVP-370 (演播室 D65 标准基准座机)"],
    [1, "1 · CAM-2 Ikegami HL-55 (偏暖高反差采访机位 · 需 CCU 降红升蓝配平)"],
    [2, "2 · CAM-3 BTS LDK-90 (偏冷暗部微青全景机位 · 需 CCU 升红降蓝配平)"],
    [3, "3 · CAM-4 老旧游机位 (CCD 暗电流基座抬升 + 绿偏 · 需黑平衡压暗)"],
  ],
  "camera.genlock_mode": [
    [0, "0 · 锁定演播室主同步且自动线长定时 (Blackburst Genlock Locked)"],
    [1, "1 · 脐带线缆长度相移未补偿 (Cable Delay Uncompensated · 6.51°/m 色相偏转)"],
    [2, "2 · 外同步断线 / 内晶振自由漂移 (Free-Run Crystal Drift · 色相连续旋转+滚屏)"],
    [3, "3 · BNC 同步头接触不良间歇掉锁冲击 (Intermittent Genlock Unlock)"],
  ],
  "overlay.playout_output_bus": [
    [0, "0 · 主播出母线 PGM（含 DSK 台标 Bug、实时时钟与底部游走字幕）"],
    [1, "1 · 干净净画面母线 Clean Feed（100% 不含 DSK 台标与字幕 · 供转播/归档）"],
    [2, "2 · 预监母线 PVW（下一条节目源 + 安全框 + 预卷就绪指示）"],
    [3, "3 · 主控双总线导播控制台视图（左 PVW + 右 PGM + Clean 小窗 + As-Run 播出日志）"],
  ],
  "overlay.playout_state": [
    [0, "0 · 正常节目单在播 (ON-AIR Program)"],
    [1, "1 · 录像带 5秒预卷倒计时锁相 (VTR Preroll Countdown -00:05..-00:01 -> Take)"],
    [2, "2 · 信号中断/故障 -> 应急切入「PLEASE STAND BY / 请稍候」备用垫片"],
    [3, "3 · 播出单自动顺播循环演示 (Auto-Cycle: Preroll -> On-Air -> Next)"],
  ],
  "format.test_pattern_mode": [
    [0, "0 · 关闭（使用导入视频/内置默认图）"],
    [1, "1 · SMPTE RP-219 75%彩条 + 反向蓝条 + -I/White/+Q + PLUGE(-4/0/+4 IRE)"],
    [2, "2 · EBU Tech 3087 100/0/75/0 PAL 标准彩条 + 线性亮度斜坡"],
    [3, "3 · FCC 0.5–5.0 MHz 多波群频响测试卡 (Multiburst)"],
    [4, "4 · 5阶调制阶梯 (微分增益DG/微分相位DP) + 2T脉冲与12.5T色亮延迟脉冲"],
    [5, "5 · Philips PM5544 / 电视台圆环几何与分辨率楔形卡（带动态扫描秒针）"],
  ],
  "format.test_tone_mode": [
    [0, "0 · 跟随原片伴音"],
    [1, "1 · 1 kHz (-18 dBFS / 0 VU) 广播正弦基准音"],
    [2, "2 · 3.15 kHz IEC 60386 走带抖晃 (Wow & Flutter) 测量载波"],
    [3, "3 · 20 Hz–15 kHz 对数扫频测试音"],
  ],
  "display.mv_layout": [
    [0, "0 · 2×2 导播室四分屏监视墙 (CAM / VTR-RAW / TBC-OUT / PGM)"],
    [1, "1 · 3×2 总控机房六画面监视桥（含配线架与银白荧光单色监看管）"],
    [2, "2 · 1+3 导播主副非对称监视墙（左侧大屏主监 PGM + 右侧三路预监）"],
  ],
  "display.mv_fault_channel": [
    [0, "0 · 全部监视通道正常锁相"],
    [1, "1 · CH1 一号监视窗失锁/断流"],
    [2, "2 · CH2 二号监视窗失锁/断流"],
    [3, "3 · CH3 三号监视窗失锁/断流"],
    [4, "4 · CH4 四号监视窗失锁/断流"],
    [5, "5 · CH5 五号监视窗失锁/断流 (3×2模式)"],
    [6, "6 · CH6 六号监视窗失锁/断流 (3×2模式)"],
  ],
  "display.mv_fault_mode": [
    [0, "0 · 按通道设备类型自动分配失锁响应"],
    [1, "1 · 纯黑无同步底噪 (Black No-Sync)"],
    [2, "2 · RF 高频头 AGC 满增益白雪花 (RF Snow)"],
    [3, "3 · 场行自由振荡斜向滚屏消隐带 (Free-Run Roll)"],
    [4, "4 · 数字 TBC 末帧死锁残影冻结 (Frame Freeze)"],
    [5, "5 · JVC/松下录像机本机蓝屏 + NO SIGNAL 字符替换 (VTR Blue OSD)"],
  ],
  "connector.da_termination_mode": [
    [0, "0 · 标准 75Ω 单端接匹配 (1.0 Vpp / 0 dB)"],
    [1, "1 · 未端接高阻开路环通 (High-Z Open · ~2.0 Vpp / +6 dB 过曝削波 + 正向重影)"],
    [2, "2 · 违规无源三通双/多端接 (Passive T-Split 75Ω/N · 压降发暗 + 同步顶压缩 + 负向重影)"],
    [3, "3 · 有源分配放大器隔离驱动 (Active DA 1-in-N-out · 75Ω 恒定电平 + 电缆均衡)"],
  ],
  "connector.patch_format_mismatch": [
    [0, "0 · 正常信号格式匹配 (Composite / 正常跳线)"],
    [1, "1 · S-Video 亮度 (Y) 误插复合口（色同步丢失→纯黑白画面）"],
    [2, "2 · 分量 YPbPr / 绿通道误插复合口（无副载波→绿/品红偏色与同步滑移）"],
    [3, "3 · SDI / 数字码流误插模拟复合口（满屏高频数字雪花杂波与失锁滚道）"],
  ],
  "connector.switcher_mode": [
    [0, "0 · 关闭（直通主信号 Bus A）"],
    [1, "1 · 非同步信号源硬切瞬态 (Non-Genlocked Hot Switch)"],
    [2, "2 · 模拟 T-Bar 叠化 (Analog Mix / Dissolve)"],
    [3, "3 · 模拟电压斜坡划像 (Analog Ramp Wipe)"],
    [4, "4 · 模拟亮度键控 (Analog Luma Key)"],
    [5, "5 · 模拟色度键控 (Analog Chroma Key)"],
  ],
  "connector.switcher_b_source": [
    [0, "0 · 自适应自定义叠层 / 副机位画面 (Camera B)"],
    [1, "1 · 自定义文字与字幕底栏叠层 (Lower-Third + Text)"],
    [2, "2 · 自定义画中画小窗叠层 (PIP Window + Title)"],
    [3, "3 · 已导入的自定义叠层图 / 副视频 (Uploaded Overlay)"],
    [4, "4 · 自定义色调 / 渐变光效叠层 (Custom Color / Gradient)"],
    [5, "5 · 经典 SMPTE/EBU 演播室彩条测试卡"],
  ],
  "connector.switcher_overlay_style": [
    [0, "0 · 自动适配（上传图 / 纯净副画面 / 画中画）"],
    [1, "1 · 自定义文字 + 字幕底栏 (Lower-Third + Text)"],
    [2, "2 · 仅自定义文字 / 台标角标 (Text / Bug Only)"],
    [3, "3 · 自定义画中画小窗 + 标题 (PIP Window + Title)"],
    [4, "4 · 全画幅纯净副机位画面 (Full-Frame Camera B)"],
    [5, "5 · 经典演播室徽标与字幕条 (Classic Emblem + Banner)"],
  ],
  "connector.switcher_wipe_pattern": [
    [0, "0 · 水平拉幕划像 (Left → Right)"],
    [1, "1 · 垂直拉幕划像 (Top → Bottom)"],
    [2, "2 · 45° 对角斜切划像 (Diagonal)"],
    [3, "3 · 中心菱形光圈划像 (Diamond Iris)"],
    [4, "4 · 中心圆形光圈划像 (Circle Iris)"],
  ],
  "tape.transport_state": [
    [0, "0 · 正常走带播放 (Play · 伺服锁相)"],
    [1, "1 · 静态暂停定格 (Pause/Still · 跨轨双峰雪花带 + 场微颤)"],
    [2, "2 · 压带轮啮合与 CTL 重锁定瞬态 (Play↔Pause 起停过渡)"],
    [3, "3 · 无飞抹磁头暂停起录切口 (Record-Pause 彩虹叠印切口)"],
    [4, "4 · 停机卸带瞬态 (Stop/Unthread · 磁鼓包角坍塌雪花帷幕)"],
  ],
  "camera.nv_mode": [
    [0, "关"],
    [1, "NightShot（红外 + 绿单色）"],
    [2, "Super NightShot（帧积累）"],
    [3, "Color Slow Shutter（慢快门保留彩色）"],
  ],
  "display.mask_type": [
    [0, "荫栅 Aperture Grille（特丽珑连续竖条 + 阻尼丝）"],
    [1, "槽孔荫罩 Slot Mask（家用彩电交错砖形槽）"],
    [2, "点状荫罩 Dot Triad（球面管三角圆点阵）"],
  ],
  "rf.tv_standard": [
    [0, "PAL-D/K（中国大陆 8MHz / +6.5MHz 伴音）"],
    [1, "PAL-I（港澳珠三角 8MHz / +6.0MHz 伴音）"],
    [2, "NTSC-M（6MHz / +4.5MHz 伴音）"],
  ],
  "rf.propagation_mode": [
    [0, "本地视距传播 (LOS)"],
    [1, "常规超视距弱信号 (Troposcatter)"],
    [2, "TV-DX 大气波导现象 (Atmospheric Ducting 异常清晰)"],
    [3, "电离层 Es 突发层反射 (Sporadic-E Skywave)"],
  ],
  "player.std_mismatch": [
    [0, "0 · 制式正常匹配（跟随工程 NTSC/PAL）"],
    [1, "1 · PAL (50Hz) 信号误入 NTSC (59.94Hz) 系统（9.94Hz 滚屏 + VBI 同步带 + 差拍消色）"],
    [2, "2 · NTSC (59.94Hz) 信号误入 PAL (50Hz) 系统（反向滚屏 + PAL 倒相汉诺威色条）"],
  ],
  "rf.tuner_type": [
    [0, "传统铁壳高频头 (Can Tuner)"],
    [1, "淘宝廉价电视卡硅调谐器 (SAA713x / TDA9887)"],
    [2, "高选择性 DX 接收机 (High-Q DX Tuner)"],
  ],
  "audio.mode": [
    [0, "0 · 自动（跟随媒介制式：胶片/流媒体/磁带）"],
    [1, "1 · VHS 纵向线性磁迹"],
    [2, "2 · VHS Hi-Fi 调频深层记录"],
    [3, "3 · 8mm / Hi8 AFM 调频伴音"],
    [4, "4 · MiniDV PCM 数字音频"],
    [5, "5 · VCD MP2 子带量化伴音"],
    [6, "6 · PAL-D RF 模拟电视射频伴音"],
    [7, "7 · TV-DX 电视卡 FM 伴音解调"],
    [8, "8 · 胶片感光光学音轨 (Optical Soundtrack)"],
    [9, "9 · 网络流媒体低码率音频 (优酷/土豆/3GP)"],
    [10, "10 · 广播电台/演播室播出前级系统"],
    [11, "11 · 旧手机免提外放与听筒 (Mobile Speaker/Earpiece)"],
    [12, "12 · 移动电话与 VoIP 网络语音 (GSM/AMR/窄带电话)"],
    [13, "13 · 铃声、MIDI 与和弦音源 (Nokia蜂鸣器/Yamaha FM)"],
    [14, "14 · 早期廉价录音笔 / MP3 随身听 ADPCM 内录"],
    [15, "15 · 10 大音频子系统自由组合工作台"],
  ],
  "audio.mic_type": [
    [0, "0 · 直通 / 线路输入 (Direct Line In)"],
    [1, "1 · 动圈话筒 (Shure SM58 风格)"],
    [2, "2 · 大振膜电容话筒 (演播室高灵敏度)"],
    [3, "3 · 铝带话筒 (温暖平滑 8 字指向)"],
    [4, "4 · 驻极体电脑摄像头麦克风 (桌面散装麦克风)"],
    [5, "5 · 碳粒话筒 (老式座机电话/对讲窄频)"],
    [6, "6 · 家用 DV 机顶立体声驻极体麦 (风噪/走带电机耦合)"],
    [7, "7 · 早期直板功能机针孔驻极体/MEMS 麦 (强AGC/极易爆音)"],
  ],
  "audio.mic_polar_pattern": [
    [0, "0 · 全指向 (Omnidirectional)"],
    [1, "1 · 心形指向 (Cardioid)"],
    [2, "2 · 超心形指向 (Supercardioid)"],
    [3, "3 · 8字双指向 (Figure-8)"],
    [4, "4 · 强指向枪式话筒 (Shotgun 干涉管)"],
  ],
  "audio.mic_clip_mode": [
    [0, "0 · 模拟运放软削波 (Soft Op-Amp Saturation)"],
    [1, "1 · 廉价声卡/手机 ADC 硬削波 (Hard Brickwall Clip)"],
    [2, "2 · 定点寄存器溢出折返爆音 (Integer Wrap-Around)"],
  ],
  "audio.mobile_spk_mode": [
    [0, "0 · 关闭"],
    [1, "1 · 电话听筒模式 (贴耳密闭受话器 300-3400Hz)"],
    [2, "2 · 功能机背面免提单声道外放 (Nokia/山寨机大嗓门薄亮刺耳)"],
    [3, "3 · 早期智能手机微型扬声器外放 (iPhone 3G/4 微型腔体)"],
    [4, "4 · 对讲机 / 语音信箱免提公放 (强中频共振)"],
  ],
  "audio.adc_resample_filter": [
    [0, "0 · 标准抗混叠 Sinc 低通重采样"],
    [1, "1 · 早期声卡线性插值重采样 (高频暗淡+镜像残留)"],
    [2, "2 · 无抗混叠零阶保持抽取 (Zero-Order Hold 金属折返混叠)"],
  ],
  "audio.adc_dither_mode": [
    [0, "0 · 无抖动直接截断 (Truncation 谐波颗粒失真)"],
    [1, "1 · 矩形概率抖动 (RPDF Dither)"],
    [2, "2 · 三角概率抖动 (TPDF 标准抖动)"],
    [3, "3 · 噪声整形抖动 (Noise-Shaped TPDF)"],
  ],
  "audio.adc_channel_mode": [
    [0, "0 · 立体声正常 (Stereo)"],
    [1, "1 · 强制单声道合并 (L+R Mono Sum)"],
    [2, "2 · 伪立体声反相抵消 (L/R 极性接反，人声中置抵消空洞)"],
    [3, "3 · 单边声道丢失 (仅左声道复制)"],
  ],
  "audio.pcodec_type": [
    [0, "0 · 关闭"],
    [1, "1 · MP3 (MPEG-1 Layer III 32子带+MDCT 水声/预回声)"],
    [2, "2 · AAC-LC / HE-AAC (早期流媒体 3GP/FLV SBR 高频重建)"],
    [3, "3 · AMR-NB (3GPP 移动语音代数码激励线性预测 4.75-12.2kbps)"],
    [4, "4 · 早期流媒体低码率语音编码 (Speex / Nellymoser Asao 风格)"],
  ],
  "audio.telecom_mode": [
    [0, "0 · 关闭"],
    [1, "1 · PSTN 固话 / 2G GSM 窄带语音 (300-3400Hz)"],
    [2, "2 · 早期 2G GSM 边缘弱信号 (217Hz TDMA 蜂音 + 突发丢包)"],
    [3, "3 · 早期 Skype / QQ 语音通话 (VAD 静音门 + 丢包机器音拖尾)"],
    [4, "4 · 现代会议软件劣质网络语音 (强降噪水声 + 回声抑制抽吸)"],
  ],
  "audio.telecom_wideband": [
    [0, "0 · 窄带语音 Narrowband (300-3400Hz @ 8kHz)"],
    [1, "1 · 宽带高清语音 Wideband HD Voice (50-7000Hz @ 16kHz)"],
  ],
  "audio.telecom_plc_mode": [
    [0, "0 · 零填充硬静音断字 (Zero-Fill Hard Drop)"],
    [1, "1 · 基音周期重复冻结 (Pitch-Waveform Repeat 机器音拖尾)"],
    [2, "2 · 帧间交叉淡化插值 (Crossfade Interpolation)"],
  ],
  "audio.space_preset": [
    [0, "0 · 直通干声 (Dry Anechoic)"],
    [1, "1 · 小卧室桌面近讲 (早期优酷/播客桌面反射+电脑风扇底噪)"],
    [2, "2 · 空旷瓷砖走廊 / 楼梯间 (强颤动回声 Flutter Echo + 高频反射)"],
    [3, "3 · 教室 / 会议室远距离拾音 (低直达混响比 DRR + 空洞驻波)"],
    [4, "4 · 户外开放街区环境 (无混响尾音 + 远距空气高频吸收 + 街区风噪)"],
  ],
  "audio.sys_audio_preset": [
    [0, "0 · 现代标准直通 (Clean OS Mixer)"],
    [1, "1 · Windows XP KMixer 44.1→48kHz 劣质重采样 + AC97 板载声卡"],
    [2, "2 · 网页 Flash Player 播放器 (22.05kHz 降采样 + 音量推满削波)"],
    [3, "3 · 3.5mm 耳机插头地线虚接 (L-R 人声消除伴奏卡拉OK效应)"],
    [4, "4 · 廉价多媒体音箱 3D 环绕扩展 + 微笑曲线 EQ"],
  ],
  "audio.sys_src_mode": [
    [0, "0 · 高质量重采样 (Transparent SRC)"],
    [1, "1 · Windows KMixer 44.1→48kHz 线性插值劣化"],
    [2, "2 · 网页 Flash Player 22.05kHz 降采样"],
    [3, "3 · 11.025kHz 极低采样率缓冲"],
  ],
  "audio.sys_channel_map": [
    [0, "0 · 正常立体声 (L / R)"],
    [1, "1 · 左右声道反接 (R / L Swap)"],
    [2, "2 · 左声道单边断线 (仅右声道)"],
    [3, "3 · 3.5mm 插头地线松动脱落 (L-R 差模串扰，中置人声消失)"],
    [4, "4 · 强制单声道混音 (Mono Mixdown)"],
  ],
  "audio.sys_eq_preset": [
    [0, "0 · 平直 (Flat)"],
    [1, "1 · 搖滚 / 微笑曲线 (V-Shape Bass+Treble Boost)"],
    [2, "2 · 廉价塑料小音箱中频凸起 (Plastic Boxy)"],
    [3, "3 · 深夜对白增强 (Vocal Mid Boost)"],
  ],
  "audio.ringtone_mode": [
    [0, "0 · 关闭"],
    [1, "1 · Nokia 单音压电蜂鸣器铃声 (Monophonic Piezo Square Buzzer)"],
    [2, "2 · 早期 4/16 和弦 FM 合成铃声 (Yamaha MA-2/MA-3 OPL 芯片)"],
    [3, "3 · 低比特 PCM 采样波表和弦铃声 (8-bit ADPCM SoundFont)"],
    [4, "4 · 输入音频实时转写为复古和弦/蜂鸣器旋律 (Audio-to-Ringtone)"],
  ],
  "audio.loudness_mode": [
    [0, "0 · 关闭"],
    [1, "1 · 电视台 / 调频广播多段压限器 (Loudness War 高响度砖墙压限)"],
    [2, "2 · 视频平台自动响度归一化 (YouTube / B站 -14 LUFS 峰值惩罚)"],
    [3, "3 · 播客 / 移动流媒体响度标准 (-16 LUFS + 真峰值限制)"],
    [4, "4 · 广电 EBU R 128 标准 (-23 LUFS 宽动态)"],
  ],
  "audio.av_drop_sync_mode": [
    [0, "0 · 视频掉帧冻结但音频照常连续走带 (造成画面与声音错位)"],
    [1, "1 · 音视频硬切同跳 (视频掉帧处音频同步丢块咔哒跳帧)"],
    [2, "2 · 锁相环 PLL 柔性追赶重采样 (恢复同步时产生短暂音高滑音)"],
  ],
  "audio.av_edit_splice_mode": [
    [0, "0 · 关闭"],
    [1, "1 · 切点短暂静音缺口 (Mute Gap 8-25ms)"],
    [2, "2 · 切点非过零点硬切爆音 + 直流阶跃 (Zero-Crossing Click & DC Pop)"],
    [3, "3 · 切点相位跳变与磁迹重叠擦除瞬态 (Splice Overlap Transient)"],
  ],
  "overlay.style": [
    [0, "0 · 自定义（使用上方RGB参数）"],
    [1, "1 · 绿字黑边（OSD 经典）"],
    [2, "2 · 红字黑边（REC 经典）"],
    [3, "3 · 白字黑边"],
    [4, "4 · 黄字黑边"],
    [5, "5 · 青字黑边"],
    [6, "6 · 80年代纪录片·金黄标宋立体字"],
    [7, "7 · 80年代纪录片·象牙白光学叠印字"],
    [8, "8 · 广播级字幕机·琥珀黄滚屏字"],
    [9, "9 · 80年代专题片·深红描金标题字"],
  ],
  "overlay.anim_mode": [
    [0, "0 · 静态常显"],
    [1, "1 · 逐字打字机显字"],
    [2, "2 · 逐行顺序显字"],
    [3, "3 · 水平光栅扫描擦除"],
    [4, "4 · 百叶窗隔行扫描显字"],
    [5, "5 · 纵向滚屏（片尾字幕）"],
    [6, "6 · 光学叠化淡入淡出"],
    [7, "7 · 横向滑入定格"],
  ],
  "codec.stream_mode": [
    [0, "0 · 跟随总开关"],
    [1, "1 · 早期拨号/宽带流媒体 (RealVideo G2 / WMV9)"],
    [2, "2 · Web 2.0 早期视频网站多代转码"],
    [3, "3 · 现代自适应流媒体 (HLS/DASH ABR + 缓冲)"],
    [4, "4 · UDP/RTP 实时丢包切片撕裂"],
    [5, "5 · 2×2 跨时代网络流媒体对照实验室"],
    [6, "6 · 优酷/土豆 2006–2009 早期 Flash 流畅档 (FLV 320×240)"],
    [7, "7 · 优酷/土豆 2009–2012 标清 H.264 档 (480×360)"],
    [8, "8 · 移动互联网 2012–2015 多屏 H.264 档 (640×360)"],
    [9, "9 · 旧手机传输与剪辑多代压缩链 (V0→V1→V2→V3)"],
    [10, "10 · 测试一：同一场景，不同手机"],
    [11, "11 · 测试二：同一视频，不同转码链"],
    [12, "12 · 测试三：同一文件，不同播放设备"],
  ],
  "camera.ldme_mobile_profile": [
    [0, "0 · 关闭手机摄影管线"],
    [1, "1 · Nokia 6600 (2003 VGA 640×480)"],
    [2, "2 · Nokia N73 (2006 蔡司 320万像素)"],
    [3, "3 · Nokia N95 (2007 蔡司 500万像素)"],
    [4, "4 · Nokia 5800 XpressMusic (2008 3GP 视频)"],
    [5, "5 · iPhone 2007 (初代 200万像素)"],
    [6, "6 · iPhone 3G (2008 200万像素)"],
    [7, "7 · iPhone 3GS (2009 VGA 视频管线)"],
    [8, "8 · iPhone 4S (2011 800万 BSI CMOS)"],
  ],
  "camera.ldme_capture_mode": [
    [0, "0 · 静态照片管线 capture() (JPEG 4:2:0 + EXIF)"],
    [1, "1 · 动态视频管线 record() (3GP / MOV 帧间编码)"],
  ],
  "codec.ldme_transfer_method": [
    [0, "0 · 直通无额外传输处理"],
    [1, "1 · 蓝牙传输 (位级无损复制，不加虚假压缩)"],
    [2, "2 · USB / 电脑导入 (无损复制，更新文件系统时间戳)"],
    [3, "3 · 彩信 / 早期社交软件发送 (强制缩放 + 重编码 + 剥离EXIF)"],
    [4, "4 · 电脑剪辑软件导出 + 多代有损压缩链 (V0→V1→V2→V3)"],
  ],
  "codec.ldme_source_origin": [
    [0, "0 · 高清源文件被平台缩放转码为低清"],
    [1, "1 · 上传时原本就是手机低清源文件"],
  ],
  "display.ldme_display_device": [
    [0, "0 · 关闭早期 LCD 与 Flash 播放器模型"],
    [1, "1 · 早期手机 TFT 液晶屏 (16-bit RGB565 65K色 + 28ms慢响应)"],
    [2, "2 · 2007 桌面 6-bit TN 液晶显示器 (FRC抖动 + 视角伽马 + CCFL)"],
    [3, "3 · 经典 CRT 监视器对照模式"],
  ],
  "display.ldme_flash_scale_mode": [
    [0, "0 · 原始比例居中"],
    [1, "1 · Flash 窗口非整数倍双线性放大"],
    [2, "2 · Flash 最近邻像素化放大 (smoothing=false)"],
    [3, "3 · 4:3 视频强行全屏拉伸为 16:9"],
  ],
  "codec.stream_quad_corner": [
    [0, "0 · 2×2 四分屏全览"],
    [1, "1 · 左上角（采集源与TBC母本）"],
    [2, "2 · 右上角（1999 RealVideo/WMV）"],
    [3, "3 · 左下角（Web 2.0 平台多代转码）"],
    [4, "4 · 右下角（现代 ABR 与网络缓冲遥测）"],
  ],
  "codec.stream_show_border": [
    [0, "0 · 无分框（无缝拼接）"],
    [1, "1 · 细十字分框线"],
    [2, "2 · 标准演播室分框"],
  ],
  "codec.stream_show_text": [
    [0, "0 · 不要文字（纯净画面）"],
    [1, "1 · 仅显示角落标签"],
    [2, "2 · 显示角落标签 + 底部遥测栏"],
  ],
  "display.percept_mode": [
    [0, "0 · 跟随总开关"],
    [1, "1 · 观众视网膜心理物理感知视图 (Mannos-Sakrison CSF)"],
    [2, "2 · 机房值班员闭环矫正画面 (PID 跟踪微调 + CCU 配平)"],
    [3, "3 · 物理误差 vs 人眼感知显著度 (JND) 热力图"],
    [4, "4 · 2×2 感知与操作者行为闭环实验室"],
  ],
  "display.percept_quad_corner": [
    [0, "0 · 2×2 四分屏全览"],
    [1, "1 · 左上角（未干预物理故障输出）"],
    [2, "2 · 右上角（观众视网膜 CSF 感知画面）"],
    [3, "3 · 左下角（机房操作者闭环矫正画面）"],
    [4, "4 · 右下角（JND 显著度与操作轨迹面板）"],
  ],
  "display.percept_show_border": [
    [0, "0 · 无分框（无缝拼接）"],
    [1, "1 · 细十字分框线"],
    [2, "2 · 标准演播室分框"],
  ],
  "display.percept_show_text": [
    [0, "0 · 不要文字（纯净画面）"],
    [1, "1 · 仅显示角落标签"],
    [2, "2 · 显示角落标签 + 底部遥测栏"],
  ],
  "overlay.eco_mode": [
    [0, "0 · 跟随总开关"],
    [1, "1 · 历史年代媒介生态实况投射 (按所选年份驱动)"],
    [2, "2 · 备件与磁头断崖危机模拟 (Spare Collapse Cliff)"],
    [3, "3 · 历史反事实情景推演输出 (Counterfactual)"],
    [4, "4 · 2×2 媒介技术生态演化实验室 (1975–2035)"],
  ],
  "overlay.eco_scenario": [
    [0, "0 · 真实历史基准轨迹 (Baseline History)"],
    [1, "1 · 反事实 A：无损数字归档提前 10 年普及 (Early Digital)"],
    [2, "2 · 反事实 B：磁头备件于 2005 年提前断供 (Spare Collapse)"],
    [3, "3 · 反事实 C：全球公共档案强制冷库与开源备件计划"],
  ],
  "overlay.eco_quad_corner": [
    [0, "0 · 2×2 四分屏全览"],
    [1, "1 · 左上角（1992 模拟黄金期基准）"],
    [2, "2 · 右上角（所选年份真实生态状态）"],
    [3, "3 · 左下角（反事实情景推演状态）"],
    [4, "4 · 右下角（1975–2035 七维演化曲线）"],
  ],
  "overlay.eco_show_border": [
    [0, "0 · 无分框（无缝拼接）"],
    [1, "1 · 细十字分框线"],
    [2, "2 · 标准演播室分框"],
  ],
  "overlay.eco_show_text": [
    [0, "0 · 不要文字（纯净画面）"],
    [1, "1 · 仅显示角落标签"],
    [2, "2 · 显示角落标签 + 底部遥测栏"],
  ],
  "display.crt_magnet_mode": [
    [0, "0 · 关闭强磁环境"],
    [1, "1 · 强钕铁硼永磁体贴面（电子束黑洞+十字折叠+同心彩虹环）"],
    [2, "2 · 强磁极旋转洛伦兹涡流（动态旋转色环与几何扭曲）"],
    [3, "3 · 撤走磁铁后的铁磁荫罩永久剩磁 + PTC 消磁线圈净化"],
    [4, "4 · 外部 50/60Hz 变压器交流漏磁扭动 + 阳极高压电晕"],
    [5, "5 · 2×2 CRT 强磁场、铁磁剩磁与 PTC 消磁线圈四象限实验室"],
  ],
  "display.crt_magnet_quad_corner": [
    [0, "0 · 显示完整 2×2 四分屏对照"],
    [1, "1 · 仅显示左上角（强磁铁贴面黑洞 + 十字折叠 + 彩虹环）"],
    [2, "2 · 仅显示右上角（磁极旋转洛伦兹涡流扭曲）"],
    [3, "3 · 仅显示左下角（撤磁后铁磁荫罩永久剩磁色斑）"],
    [4, "4 · 仅显示右下角（内置 PTC 消磁线圈交变振荡净化后）"],
  ],
  "display.crt_magnet_show_border": [
    [0, "0 · 无分框线（无缝拼接）"],
    [1, "1 · 细十字分框线"],
    [2, "2 · 标准演播室分框"],
  ],
  "display.crt_magnet_show_text": [
    [0, "0 · 不要文字（纯净画面）"],
    [1, "1 · 仅显示角落标签"],
    [2, "2 · 显示角落标签 + 底部遥测栏"],
  ],
  "display.kv1214_ab_split": [
    [0, "0 · 全屏输出 Sony Trinitron KV-1214 效果"],
    [1, "1 · 数字基准原片 vs Sony KV-1214 显像管左右同屏对照"],
    [2, "2 · 2×2 四象限（数字原片 / KV-1214全貌 / 垂直荫栅微距 / 弯曲摩尔纹放大）"],
  ],
};

function selectorOptionsFor(layer, sp) {
  const fullKey = `${layer}.${sp.key}`;
  if (fullKey === "overlay.font") {
    if (window.SessionUI && typeof SessionUI.getOsdFonts === "function") {
      return SessionUI.getOsdFonts().map((f, i) => [i, `${i} · ${f[1]}`]);
    }
    if (S.catalog && Array.isArray(S.catalog.fonts)) {
      return S.catalog.fonts.filter((f) => f.available !== false).map((f, i) => [i, `${i} · ${f.label}`]);
    }
  }
  return SELECTORS[fullKey] || null;
}

function formatReadout(sp, v) {
  if (v === null || v === undefined) return "";
  if (sp.kind === "text") return String(v);
  const n = typeof v === "number" ? v : Number(v);
  if (isNaN(n)) return String(v);
  const isBool = sp.unit === "bool" || (sp.step === 1 && sp.min === 0 && sp.max === 1 && sp.unit === "0/1");
  if (isBool) return n > 0.5 ? "开" : "关";
  if (sp.step === 1) return Math.round(n).toFixed(0);
  return fmt(n);
}

function paramRow(sp) {
  const layerName = S.layer;
  const cur = (S.params[layerName] || {})[sp.key];
  const baseVal = cur === undefined ? sp.default : cur;
  const kfInfo = (sp.kind !== "text" && window.SessionUI && typeof SessionUI.trackInfo === "function")
    ? SessionUI.trackInfo(layerName, sp.key, S.t)
    : { hasTrack: false, atPoint: false, pointCount: 0, value: null };
  const val = (kfInfo.hasTrack && kfInfo.value !== null) ? kfInfo.value : baseVal;

  const row = document.createElement("div");
  row.className = "prow" + (kfInfo.hasTrack ? " prow-kf" : "");

  const head = document.createElement("div");
  head.className = "plabel";
  const name = document.createElement("span");
  name.textContent = sp.label + (sp.unit && sp.unit !== "0-1" && sp.unit !== "bool" && sp.unit !== "text" ? ` (${sp.unit})` : "");

  const rightBox = document.createElement("span");
  rightBox.className = "pright";

  const vv = document.createElement("span");
  vv.className = "v";
  const inPhys = sp.phys_min === null || sp.phys_max === null || sp.kind === "text"
    || (val >= sp.phys_min - 1e-9 && val <= sp.phys_max + 1e-9);
  if (!inPhys) vv.classList.add("artistic");
  vv.textContent = formatReadout(sp, val);
  rightBox.appendChild(vv);

  // 为所有可调节数值参数添加关键帧控制按钮（◆ 打关键帧 / 跳上一帧 / 跳下一帧 / 打开曲线）
  if (sp.kind !== "text" && window.SessionUI && typeof SessionUI.upsertKeyframe === "function") {
    if (kfInfo.hasTrack) {
      const prevBtn = document.createElement("button");
      prevBtn.className = "kfNav";
      prevBtn.textContent = "‹";
      prevBtn.title = "跳到上一个关键帧";
      prevBtn.onclick = (ev) => {
        ev.stopPropagation();
        const tTarget = SessionUI.stepKeyframe(layerName, sp.key, -1);
        if (tTarget != null) {
          S.t = tTarget;
          if ($("timeSlider")) $("timeSlider").value = String(tTarget);
          if ($("timeLabel")) $("timeLabel").textContent = `t = ${tTarget.toFixed(2)} s`;
          renderParams();
          refreshPreview();
        }
      };
      const nextBtn = document.createElement("button");
      nextBtn.className = "kfNav";
      nextBtn.textContent = "›";
      nextBtn.title = "跳到下一个关键帧";
      nextBtn.onclick = (ev) => {
        ev.stopPropagation();
        const tTarget = SessionUI.stepKeyframe(layerName, sp.key, 1);
        if (tTarget != null) {
          S.t = tTarget;
          if ($("timeSlider")) $("timeSlider").value = String(tTarget);
          if ($("timeLabel")) $("timeLabel").textContent = `t = ${tTarget.toFixed(2)} s`;
          renderParams();
          refreshPreview();
        }
      };
      rightBox.append(prevBtn, nextBtn);
    }

    const kfBtn = document.createElement("button");
    kfBtn.className = "kfBtn" + (kfInfo.hasTrack ? (kfInfo.atPoint ? " on at" : " on") : "");
    kfBtn.textContent = "◆";
    kfBtn.title = kfInfo.hasTrack
      ? (kfInfo.atPoint
          ? `当前时刻 (t=${S.t.toFixed(2)}s) 已打关键帧（点击删除该点，Shift+点击删除整条轨道，双击打开曲线面板）`
          : `已有 ${kfInfo.pointCount} 个关键帧（点击在 t=${S.t.toFixed(2)}s 插入关键帧，Shift+点击删除整条轨道，双击打开曲线面板）`)
      : `在当前时刻 t=${S.t.toFixed(2)}s 为「${sp.label}」打关键帧`;
    kfBtn.onclick = (ev) => {
      ev.stopPropagation();
      if (ev.shiftKey && kfInfo.hasTrack) {
        SessionUI.removeKeyframe(layerName, sp.key, null, true);
        renderParams();
        return;
      }
      const curInputVal = r ? (sp.kind === "text" ? r.value : parseFloat(r.value)) : val;
      if (kfInfo.atPoint) {
        SessionUI.removeKeyframe(layerName, sp.key, S.t, false);
      } else {
        const defaultCurve = (sp.step === 1) ? "hold" : "linear";
        SessionUI.upsertKeyframe(layerName, sp.key, S.t, curInputVal, defaultCurve);
      }
      renderParams();
    };
    kfBtn.ondblclick = (ev) => {
      ev.stopPropagation();
      if (typeof SessionUI.focusTrack === "function") SessionUI.focusTrack(layerName, sp.key);
    };
    rightBox.appendChild(kfBtn);
  }

  head.append(name, rightBox);

  const opts = selectorOptionsFor(layerName, sp);
  let r;
  if (sp.kind === "text") {
    r = document.createElement("input");
    r.type = "text";
    r.className = "ssText";
    r.style.width = "100%";
    r.value = val == null ? "" : String(val);
    r.onchange = () => {
      S.params[layerName] = S.params[layerName] || {};
      S.params[layerName][sp.key] = r.value;
      setProject({ params: { [layerName]: { [sp.key]: r.value } } });
    };
  } else if (opts) {
    r = document.createElement("select");
    r.className = "psel";
    for (const [v, label] of opts) {
      const o = document.createElement("option");
      o.value = String(v);
      o.textContent = label;
      if (Math.abs(v - Math.round(val)) < 1e-6) o.selected = true;
      r.appendChild(o);
    }
    vv.textContent = formatReadout(sp, val);
    r.onchange = () => {
      const nv = parseFloat(r.value);
      vv.textContent = formatReadout(sp, nv);
      if (kfInfo.hasTrack && window.SessionUI) {
        SessionUI.upsertKeyframe(layerName, sp.key, S.t, nv, "hold");
      }
      S.params[layerName] = S.params[layerName] || {};
      S.params[layerName][sp.key] = nv;
      setProject({ params: { [layerName]: { [sp.key]: nv } } });
    };
  } else {
    r = document.createElement("input");
    r.type = "range";
    r.min = sp.min; r.max = sp.max;
    r.step = sp.step === 1 ? 1 : (sp.max - sp.min) / 200;
    r.value = val;
    r.oninput = () => {
      const nv = parseFloat(r.value);
      vv.textContent = formatReadout(sp, nv);
      const stillPhys = sp.phys_min === null || sp.phys_max === null
        || (nv >= sp.phys_min - 1e-9 && nv <= sp.phys_max + 1e-9);
      vv.classList.toggle("artistic", !stillPhys);
      setParam(layerName, sp.key, nv, true);
    };
    r.onchange = () => setParam(layerName, sp.key, parseFloat(r.value), false);
  }

  row.append(head, r);
  if (sp.help) {
    const p = document.createElement("div");
    p.className = "phelp";
    p.textContent = sp.help;
    row.appendChild(p);
  }
  if (sp.phys_min !== null && sp.phys_max !== null) {
    const s = document.createElement("div");
    s.className = "src";
    s.textContent = `物理区间 ${fmt(sp.phys_min)}–${fmt(sp.phys_max)} · 来源：${sp.source}`;
    row.appendChild(s);
  }
  return row;
}

/* --------------------------------------------------------------- actions */
let saveTimer = null;
function setParam(layer, key, value, debounce) {
  S.params[layer] = S.params[layer] || {};
  S.params[layer][key] = value;
  const hasKf = window.SessionUI && typeof SessionUI.trackInfo === "function"
    && SessionUI.trackInfo(layer, key, S.t).hasTrack;
  if (!S.auto) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    if (hasKf) {
      SessionUI.upsertKeyframe(layer, key, S.t, value);
    } else {
      setProject({ params: { [layer]: { [key]: value } } });
    }
  }, debounce ? 130 : 0);
}

async function setProject(payload) {
  try {
    applyProject(await api("/api/project", jsonPost(payload)));
    renderParams();
    if (typeof renderNodeInspector === "function" && typeof G !== "undefined" && G.state) {
      renderNodeInspector();
    }
    refreshPreview();
    loadTimelineMeta();
  } catch (e) { toast(e.message); }
}

async function applyPreset(key) {
  try {
    const d = await api("/api/preset", jsonPost({ key }));
    applyProject(d);
    if (d.graph_state && typeof applyGraphState === "function") {
      applyGraphState(d.graph_state);
      if (typeof renderNodeInspector === "function") renderNodeInspector();
    }
    renderParams();
    renderPresetList();
    refreshPreview();
    loadTimelineMeta();
  } catch (e) { toast(e.message); }
}

async function uploadClip(ev) {
  const f = ev.target.files[0];
  if (!f) return;
  $("badge").textContent = `导入中… ${(f.size / 1048576).toFixed(1)} MB`;
  try {
    const r = await fetch("/api/upload", {
      method: "POST", headers: { "X-Filename": encodeURIComponent(f.name) }, body: f });
    const j = await r.json();
    if (j.error) throw new Error(j.error);
    applyProject(j.project);
    refreshPreview();
  } catch (e) { toast("导入失败：" + e.message); }
}

async function uploadOverlay(ev) {
  const f = ev.target.files && ev.target.files[0];
  if (!f) return;
  toast(`正在导入自定义叠层: ${f.name}…`);
  try {
    const r = await fetch("/api/upload_overlay", {
      method: "POST", headers: { "X-Filename": encodeURIComponent(f.name) }, body: f });
    const j = await r.json();
    if (j.error) throw new Error(j.error);
    applyProject(j.project);
    renderParams();
    refreshPreview();
    toast(`已应用自定义叠层: ${f.name}`);
  } catch (e) {
    toast("导入叠层失败：" + e.message);
  } finally {
    ev.target.value = "";
  }
}

async function clearOverlay() {
  try {
    const r = await api("/api/clear_overlay", jsonPost({}));
    if (r && r.project) applyProject(r.project);
    renderParams();
    refreshPreview();
    toast("已清除自定义叠层图，恢复参数化自定义叠层");
  } catch (e) {
    toast("清除叠层失败：" + e.message);
  }
}

/**
 * Export the frame at the current time cursor as a full-resolution still.
 *
 * Rendered at scale 1.0, not the preview scale: the point of exporting a still is
 * to have the real output, and the preview's 0.5 proxy would quietly halve the
 * resolution of anything the user files away.
 */
const EXPORT_PROFILES = {
  lossless: { format: "png", q: 100, crf: 12, preset: "slow", label: "极高画质" },
  high:     { format: "png", q: 95,  crf: 16, preset: "medium", label: "标准高画质" },
  medium:   { format: "jpg", q: 92,  crf: 20, preset: "medium", label: "均衡画质" },
  draft:    { format: "jpg", q: 82,  crf: 24, preset: "fast", label: "快速导出" },
};

async function exportStill() {
  const btn = $("btnStill");
  const btnIn = $("btnQuadStillInline");
  const was = btn ? btn.textContent : "";
  const wasIn = btnIn ? btnIn.textContent : "";
  if (btn) { btn.disabled = true; btn.textContent = "导出中…"; }
  if (btnIn) { btnIn.disabled = true; btnIn.textContent = "导出中…"; }
  try {
    const prof = EXPORT_PROFILES[S.exportQuality] || EXPORT_PROFILES.high;
    const sc = encodeURIComponent(S.exportScale || "1.0");
    const qCorner = $("quadCornerSel"), qBorder = $("quadBorderSel"), qText = $("quadTextSel");
    const active = detectActiveQuadModule();
    let extra = "";
    if (active.isActive && qCorner && qBorder && qText) {
      extra = `&corner=${encodeURIComponent(qCorner.value)}&border=${encodeURIComponent(qBorder.value)}&text=${encodeURIComponent(qText.value)}`;
    }
    const engine = (typeof G !== "undefined" && G.perNode) ? "&engine=graph" : "";
    const tVal = typeof S.t === "number" ? S.t : (parseFloat(S.t) || 0.0);
    const url = `/api/still?t=${tVal}&scale=${sc}&format=${prof.format}&q=${prof.q}${extra}${engine}`;
    const r = await fetch(url);
    if (!r.ok) {
      let errTxt = `HTTP ${r.status}`;
      try {
        const j = await r.json();
        if (j.error) errTxt = j.error;
      } catch (_) {}
      throw new Error(errTxt);
    }
    const disp = r.headers.get("content-disposition") || "";
    const m = disp.match(/filename="([^"]+)"/);
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    const ext = prof.format === "jpg" ? "jpg" : "png";
    a.download = (m && m[1]) ? m[1] : `atsim_${tVal.toFixed(2).replace(".", "_")}s.${ext}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    toast("已导出静帧 " + a.download);
  } catch (e) {
    toast("导出失败：" + e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = was || "导出静帧"; }
    if (btnIn) { btnIn.disabled = false; btnIn.textContent = wasIn || "⬇ 导出当前帧"; }
  }
}

async function exportQuadBurst() {
  const btn = $("btnQuadBurst");
  const btnIn = $("btnQuadBurstInline");
  const was = btn ? btn.textContent : "";
  const wasIn = btnIn ? btnIn.textContent : "";
  if (btn) { btn.disabled = true; btn.textContent = "四连拍中…"; }
  if (btnIn) { btnIn.disabled = true; btnIn.textContent = "四连拍中…"; }
  try {
    const prof = EXPORT_PROFILES[S.exportQuality] || EXPORT_PROFILES.high;
    const sc = encodeURIComponent(S.exportScale || "1.0");
    const qText = $("quadTextSel");
    const tVal = qText ? parseInt(qText.value, 10) || 0 : 0;
    const engine = (typeof G !== "undefined" && G.perNode) ? "&engine=graph" : "";
    const url = `/api/still_quad_burst?t=${S.t}&scale=${sc}&format=${prof.format}&q=${prof.q}&border=0&text=${tVal}&include_quad=0&mode=json${engine}`;
    const res = await api(url);
    if (!res || !res.shots || !res.shots.length) throw new Error("未返回四连拍图像");
    const ctype = res.content_type || (prof.format === "jpg" ? "image/jpeg" : "image/png");
    for (let i = 0; i < res.shots.length; i++) {
      const s = res.shots[i];
      const bin = atob(s.b64);
      const bytes = new Uint8Array(bin.length);
      for (let j = 0; j < bin.length; j++) bytes[j] = bin.charCodeAt(j);
      const blob = new Blob([bytes], { type: ctype });
      await new Promise((r) => setTimeout(r, i === 0 ? 0 : 140));
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = s.filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 6000);
    }
    toast(`已完成批量四连拍（共 ${res.shots.length} 张全尺寸静帧：${res.shots.map((x) => x.label).join(" / ")}）`);
  } catch (e) {
    toast("批量四连拍失败：" + e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = was || "📸 批量四连拍"; }
    if (btnIn) { btnIn.disabled = false; btnIn.textContent = wasIn || "📸 批量四连拍 (4张)"; }
  }
}

async function exportQuadZip() {
  const btn = $("btnQuadZip");
  const btnIn = $("btnQuadZipInline");
  const was = btn ? btn.textContent : "";
  const wasIn = btnIn ? btnIn.textContent : "";
  if (btn) { btn.disabled = true; btn.textContent = "打包中…"; }
  if (btnIn) { btnIn.disabled = true; btnIn.textContent = "打包中…"; }
  try {
    const prof = EXPORT_PROFILES[S.exportQuality] || EXPORT_PROFILES.high;
    const sc = encodeURIComponent(S.exportScale || "1.0");
    const qBorder = $("quadBorderSel"), qText = $("quadTextSel");
    const bVal = qBorder ? parseInt(qBorder.value, 10) || 0 : 0;
    const tVal = qText ? parseInt(qText.value, 10) || 0 : 0;
    const engine = (typeof G !== "undefined" && G.perNode) ? "&engine=graph" : "";
    const url = `/api/still_quad_burst?t=${S.t}&scale=${sc}&format=${prof.format}&q=${prof.q}&border=${bVal}&text=${tVal}&include_quad=1&mode=zip${engine}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `atsim_quad_burst_${S.t.toFixed(2).replace(".", "_")}s.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    toast("已打包下载四连拍 ZIP (含 2×2 四分屏 + 4 个独立角落静帧)");
  } catch (e) {
    toast("四连拍 ZIP 导出失败：" + e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = was || "📦 四连拍ZIP"; }
    if (btnIn) { btnIn.disabled = false; btnIn.textContent = wasIn || "📦 四连拍ZIP"; }
  }
}

/* --------------------------------------------------------------- preview */
let _auxRefreshTimer = null;
function scheduleAuxRefresh() {
  if (_auxRefreshTimer) clearTimeout(_auxRefreshTimer);
  _auxRefreshTimer = setTimeout(async () => {
    _auxRefreshTimer = null;
    try {
      const tl = await api("/api/timeline");
      paintTimeline(tl);
    } catch (_) {}
    refreshAudioTimeline();
    if (window.BroadcastScopes) BroadcastScopes.refresh();
    if (typeof refreshGraphPerf === "function") {
      try { await refreshGraphPerf(); } catch (_) {}
    }
  }, 140);
}

async function refreshPreview() {
  if (S.busy) { S.pending = true; return; }
  S.busy = true;
  const my = ++S.token;
  const started = performance.now();
  const engine = (typeof G !== "undefined" && G.perNode) ? "&engine=graph" : "";
  const pScale = encodeURIComponent(S.previewScale || "0.5");
  const pQual = S.previewQuality || 88;
  const wantOrigKey = `${(S.project && S.project.source) || ""}|${Number(S.t || 0).toFixed(4)}|${pScale}|${pQual}`;
  try {
    const framePromise = api(`/api/frame?t=${S.t}&scale=${pScale}&q=${pQual}&_=${my}${engine}`);
    const origPromise = (S.ab && S.origKey !== wantOrigKey) ? loadOriginal(wantOrigKey) : Promise.resolve();
    const [blob] = await Promise.all([framePromise, origPromise]);
    if (my !== S.token) return;
    if (S.frameUrl) URL.revokeObjectURL(S.frameUrl);
    S.frameUrl = URL.createObjectURL(blob);
    const pimg = $("previewImg");
    pimg.onload = () => {
      S.frameW = pimg.naturalWidth; S.frameH = pimg.naturalHeight;
      syncAB();
    };
    pimg.src = S.frameUrl;
    const ms = performance.now() - started;
    const via = (typeof G !== "undefined" && G.perNode) ? "节点图" : "固定链路";
    $("badge").textContent =
      `${via} · 预览 ${S.frameW || "?"}×${S.frameH || "?"} · ` +
      `${ms.toFixed(0)} ms/帧 · t=${S.t.toFixed(2)}s`;
    scheduleAuxRefresh();
  } catch (e) {
    const b = $("badge");
    if (b) b.textContent = "错误：" + e.message.slice(0, 120);
    else console.error("refreshPreview failed:", e);
  } finally {
    S.busy = false;
    if (S.pending) { S.pending = false; refreshPreview(); }
  }
}

async function loadOriginal(expectedKey) {
  const pScale = encodeURIComponent(S.previewScale || "0.5");
  const pQual = S.previewQuality || 88;
  const key = expectedKey || `${(S.project && S.project.source) || ""}|${Number(S.t || 0).toFixed(4)}|${pScale}|${pQual}`;
  if (S.origUrl && S.origKey === key) return;
  try {
    const blob = await api(`/api/frame?t=${S.t}&scale=${pScale}&q=${pQual}&mode=source`);
    if (S.origUrl) URL.revokeObjectURL(S.origUrl);
    S.origUrl = URL.createObjectURL(blob);
    S.origKey = key;
    const oimg = $("originalImg");
    oimg.onload = () => syncAB();
    oimg.src = S.origUrl;
  } catch (e) { /* the A/B overlay is optional */ }
}

function paintTimeline(tl) {
  const track = $("tlTrack");
  track.innerHTML = "";
  const stages = tl.stages || [];
  stages.forEach((s, i) => {
    const seg = document.createElement("div");
    seg.className = "tl-seg";
    seg.style.left = (i / stages.length * 100) + "%";
    seg.style.width = (100 / stages.length) + "%";
    seg.style.background = s.dirty ? "rgba(56,189,248,.32)" : "rgba(52,211,153,.20)";
    seg.style.borderRight = "1px solid var(--line)";
    seg.title = `${s.label} · 缓存 ${s.cached} 帧${s.dirty ? " · 需重算" : ""}`;
    seg.textContent = s.label;
    seg.style.fontSize = "10px";
    seg.style.paddingLeft = "6px";
    seg.style.lineHeight = "16px";
    seg.style.color = "#cbd5e1";
    track.appendChild(seg);
  });
  renderChainBar(stages);

  const def = $("tlDefects");
  def.innerHTML = "";
  const dens = ((tl.defects || []).find((d) => d.kind === "tape_plan") || {}).dropout_density || [];
  const max = Math.max(1, ...dens.map((d) => d.count));
  const total = Math.max(1, (tl.defects || []).find((d) => d.kind === "tape_plan")?.prepared_frames || 1);
  dens.forEach((d) => {
    const seg = document.createElement("div");
    seg.className = "tl-seg";
    const x = (d.second / (total / (tl.fps || 30))) * 100;
    seg.style.left = Math.min(99, x) + "%";
    seg.style.width = "2px";
    seg.style.background = "#f87171";
    seg.style.opacity = 0.35 + 0.65 * (d.count / max);
    seg.title = `第 ${d.second} 秒：${d.count} 次失落`;
    def.appendChild(seg);
  });
  const head = document.createElement("div");
  head.className = "tl-tick";
  head.style.left = (S.t / (total / (tl.fps || 30)) * 100) + "%";
  def.appendChild(head);
}

function renderChainBar(stages) {
  const bar = $("chainbar");
  bar.innerHTML = "";
  (stages || []).forEach((s) => {
    const d = document.createElement("div");
    d.className = "chainnode" + (s.dirty ? " dirty" : " cached");
    d.innerHTML = `<span class="nm"></span><span class="cnt"></span>`;
    d.querySelector(".nm").textContent = s.label;
    d.querySelector(".cnt").textContent = s.dirty ? "待重算" : `${s.cached} 帧`;
    d.title = `参数哈希 ${s.hash} · ${s.dirty ? "参数已变，需要重算" : "已缓存"}`;
    bar.appendChild(d);
  });
}

/* --------------------------------------------------------------- render */
let jobTimer = null;
async function startRender() {
  const prof = EXPORT_PROFILES[S.exportQuality] || EXPORT_PROFILES.high;
  const sc = encodeURIComponent(S.exportScale || "1.0");
  $("jobbar").classList.remove("hidden");
  $("jobDownload").classList.add("hidden");
  $("jobFill").style.width = "0%";
  $("jobText").textContent = `启动渲染 (${S.exportScale || "1.0"} · CRF ${prof.crf})…`;
  try {
    const { job } = await api(`/api/render?scale=${sc}&crf=${prof.crf}&preset=${prof.preset}`);
    clearInterval(jobTimer);
    jobTimer = setInterval(() => pollJob(job), 700);
    pollJob(job);
  } catch (e) { $("jobText").textContent = "失败：" + e.message; }
}

async function pollJob(id) {
  try {
    const j = await api("/api/job?id=" + id);
    $("jobFill").style.width = ((j.progress || 0) * 100).toFixed(1) + "%";
    $("jobText").textContent = j.message || j.state;
    if (j.state === "done") {
      clearInterval(jobTimer);
      const a = $("jobDownload");
      a.href = "/api/download?id=" + id;
      a.classList.remove("hidden");
    } else if (j.state === "error") {
      clearInterval(jobTimer);
      $("jobText").innerHTML = `<span class="err">${(j.message || "").slice(0, 200)}</span>`;
    }
  } catch (e) { /* keep polling */ }
}

/* --------------------------------------------------------------- misc */
async function loadTimelineMeta() { /* defect strip refreshes with the frame */ }

async function showTimeline() {
  const tl = await api("/api/timeline");
  const plan = (tl.defects || []).find((d) => d.kind === "tape_plan") || {};
  $("tlSeed").textContent = `seed = ${tl.seed}`;
  const rows = (plan.dropout_events || []).slice(0, 400);
  $("tlBody").innerHTML = `
    <p class="note">共 ${plan.dropout_count || 0} 次失落事件，${(plan.creases || []).length} 条折痕；
    下方列出前 ${rows.length} 条。</p>
    <table><thead><tr><th>开始 (s)</th><th>持续 (s)</th><th>行</th><th>高度</th>
    <th>列</th><th>宽度</th><th>场</th><th>严重度</th></tr></thead><tbody>
    ${rows.map((e) => `<tr><td>${e.start_s}</td><td>${e.duration_s}</td><td>${e.row}</td>
      <td>${e.rows}</td><td>${e.col}</td><td>${e.cols}</td><td>${e.field}</td>
      <td>${e.severity}</td></tr>`).join("")}
    </tbody></table>`;
  $("modal").classList.remove("hidden");
}

function toast(msg) {
  const b = $("badge");
  b.textContent = msg;
  b.style.color = "#00ff66";
  setTimeout(() => { b.style.color = ""; }, 2500);
}


/* --------------------------------------------------------------- audio monitor & A/V sync scope */
let _audioTl = null;
let _audioRaf = null;
let _audioWinStart = 0.0;
let _audioWinDur = 3.0;

function initAudioMonitor() {
  const btn = $("btnAudioPlay");
  const player = $("audioPlayer");
  const scope = $("audioScope");
  if (!btn || !player || !scope) return;

  const btnWin = $("btnAudioExportWin");
  if (btnWin) {
    btnWin.onclick = async () => {
      const clean = $("audioClean") && $("audioClean").checked ? "original" : "sim";
      const start = Math.max(0, Math.floor(S.t / 1.5) * 1.5);
      const orig = btnWin.textContent;
      btnWin.disabled = true;
      btnWin.textContent = "导出中…";
      try {
        const r = await fetch(`/api/audio?t=${start.toFixed(3)}&dur=${_audioWinDur}&mode=${clean}&_=${Date.now()}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const blob = await r.blob();
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `atsim_audio_${start.toFixed(1).replace(".", "_")}s_${_audioWinDur}s.wav`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 4000);
        toast("已导出试听段伴音 " + a.download);
      } catch (e) {
        toast("伴音导出失败：" + e.message);
      } finally {
        btnWin.disabled = false;
        btnWin.textContent = orig;
      }
    };
  }
  const btnFull = $("btnAudioExportFull");
  if (btnFull) {
    btnFull.onclick = async () => {
      const clean = $("audioClean") && $("audioClean").checked ? "original" : "sim";
      const orig = btnFull.textContent;
      btnFull.disabled = true;
      btnFull.textContent = "合成中…";
      try {
        const info = (S.project && S.project.source_info) || {};
        const dur = Math.min(30.0, Math.max(3.0, parseFloat(info.duration) || 6.0));
        const r = await fetch(`/api/audio?t=0&dur=${dur.toFixed(3)}&mode=${clean}&_=${Date.now()}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const blob = await r.blob();
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `atsim_audio_full_${dur.toFixed(1).replace(".", "_")}s.wav`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 4000);
        toast("已导出全片伴音 " + a.download);
      } catch (e) {
        toast("全片伴音导出失败：" + e.message);
      } finally {
        btnFull.disabled = false;
        btnFull.textContent = orig;
      }
    };
  }

  btn.onclick = () => {
    if (!player.paused) {
      player.pause();
      return;
    }
    const clean = $("audioClean") && $("audioClean").checked ? "original" : "sim";
    _audioWinDur = 3.0;
    // Align preview start time with refreshAudioTimeline & exported window
    _audioWinStart = Math.max(0, Math.floor(S.t / 1.5) * 1.5);
    player.src = `/api/audio?t=${_audioWinStart.toFixed(3)}&dur=${_audioWinDur}&mode=${clean}&_=${Date.now()}`;
    btn.textContent = "加载伴音…";
    player.play().then(() => {
      btn.textContent = "⏹ 停止试听";
      pumpAudioSyncCursor();
    }).catch((err) => {
      btn.textContent = "▶ 试听伴音 (3s)";
      toast("音频播放失败：" + err.message);
    });
  };

  player.onended = () => {
    btn.textContent = "▶ 试听伴音 (3s)";
    if (_audioRaf) cancelAnimationFrame(_audioRaf);
    drawAudioScope();
  };
  player.onpause = () => {
    btn.textContent = "▶ 试听伴音 (3s)";
    if (_audioRaf) cancelAnimationFrame(_audioRaf);
    drawAudioScope();
  };

  scope.onclick = (ev) => {
    const r = scope.getBoundingClientRect();
    const frac = Math.min(Math.max((ev.clientX - r.left) / Math.max(1, r.width), 0), 1);
    const tNew = _audioWinStart + frac * _audioWinDur;
    S.t = Math.max(0, Math.min(10, tNew));
    const slider = $("timeSlider");
    if (slider) slider.value = S.t;
    const lbl = $("timeLabel");
    if (lbl) lbl.textContent = `t = ${S.t.toFixed(2)} s`;
    refreshPreview();
  };
}

function pumpAudioSyncCursor() {
  const player = $("audioPlayer");
  if (!player || player.paused) return;
  const curT = _audioWinStart + (player.currentTime || 0);
  drawAudioScope(curT);
  if ($("audioSyncPlay") && $("audioSyncPlay").checked && !S.busy) {
    if (Math.abs(curT - S.t) >= (1.0 / ((_audioTl && _audioTl.fps) || 29.97))) {
      S.t = Math.min(10.0, curT);
      const slider = $("timeSlider");
      if (slider) slider.value = S.t;
      const lbl = $("timeLabel");
      if (lbl) lbl.textContent = `t = ${S.t.toFixed(2)} s`;
      refreshPreview();
    }
  }
  _audioRaf = requestAnimationFrame(pumpAudioSyncCursor);
}

async function refreshAudioTimeline() {
  const scope = $("audioScope");
  if (!scope) return;
  try {
    _audioWinDur = 3.0;
    const start = Math.max(0, Math.floor(S.t / 1.5) * 1.5);
    _audioWinStart = start;
    _audioTl = await api(`/api/audio_timeline?t=${start.toFixed(2)}&dur=${_audioWinDur}`);
    drawAudioScope(S.t);
  } catch (e) {
    console.warn("audio_timeline error:", e);
  }
}

function drawAudioScope(cursorT) {
  const scope = $("audioScope");
  if (!scope || !_audioTl) return;
  const ctx = scope.getContext("2d");
  const w = scope.width, h = scope.height;
  ctx.clearRect(0, 0, w, h);

  ctx.fillStyle = "#0b0f18";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#1e293b";
  ctx.beginPath();
  ctx.moveTo(0, 11); ctx.lineTo(w, 11);
  ctx.moveTo(0, 22); ctx.lineTo(w, 22);
  ctx.stroke();

  const wf = _audioTl.waveform || [];
  if (wf.length > 0) {
    ctx.strokeStyle = "#34d399";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < wf.length; i++) {
      const x = (i / wf.length) * w;
      const pk = Math.min(1.0, wf[i].peak || 0);
      const y0 = 11 - pk * 10;
      const y1 = 11 + pk * 10;
      ctx.moveTo(x, y0);
      ctx.lineTo(x, y1);
    }
    ctx.stroke();
  }

  const frames = _audioTl.frames || [];
  let curFrameFault = null;
  const tRef = cursorT !== undefined ? cursorT : S.t;
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    const x0 = (i / Math.max(1, frames.length)) * w;
    const bw = Math.max(2, w / Math.max(1, frames.length));
    if (Math.abs((f.time_s || 0) - tRef) < (0.65 / (_audioTl.fps || 30))) {
      curFrameFault = f;
    }
    if ((f.dropout || 0) > 0.02 || f.afm_unlock) {
      ctx.fillStyle = f.afm_unlock ? "rgba(217, 97, 76, 0.92)" : "rgba(217, 97, 76, 0.55)";
      ctx.fillRect(x0, 23, bw, 5);
    }
    if ((f.head_switch_buzz || 0) > 0.05) {
      ctx.fillStyle = "rgba(232, 163, 61, 0.75)";
      ctx.fillRect(x0, 28, bw, 3);
    }
    if (Math.abs(f.pitch_cents || 0) > 4.0) {
      ctx.fillStyle = "rgba(104, 168, 214, 0.78)";
      ctx.fillRect(x0, 31, bw, 3);
    }
    if ((f.hum_buzz || 0) > 0.02 || f.hdmi_drop) {
      ctx.fillStyle = "rgba(186, 120, 214, 0.80)";
      ctx.fillRect(x0, 23, bw, 11);
    }
  }

  const frac = ((tRef - _audioWinStart) / Math.max(0.001, _audioWinDur));
  if (frac >= 0 && frac <= 1) {
    const cx = frac * w;
    ctx.strokeStyle = "#00ff66";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, h);
    ctx.stroke();
  }

  const badge = $("audioFaultBadge");
  if (badge) {
    const modeLabel = (_audioTl.summary && _audioTl.summary.mode) || "auto";
    if (curFrameFault) {
      const tags = [];
      if (curFrameFault.hdmi_drop) tags.push("HDMI断流静音");
      if (curFrameFault.afm_unlock) tags.push("AFM解锁→线性轨");
      else if ((curFrameFault.dropout || 0) > 0.03) tags.push(`失落 ${(curFrameFault.dropout * 100).toFixed(0)}%`);
      if (Math.abs(curFrameFault.pitch_cents || 0) >= 2.0) {
        const c = curFrameFault.pitch_cents;
        tags.push(`音高 ${c >= 0 ? "+" : ""}${c.toFixed(1)}c`);
      }
      if ((curFrameFault.head_switch_buzz || 0) > 0.1) tags.push("切头脉冲");
      if ((curFrameFault.hum_buzz || 0) > 0.02) tags.push("哼声/RF蜂音");
      if (Math.abs(curFrameFault.av_offset_ms || 0) >= 5.0) {
        const avm = curFrameFault.av_offset_ms;
        tags.push(`AV偏移 ${avm >= 0 ? "+" : ""}${avm.toFixed(0)}ms`);
      }
      if (curFrameFault.av_drop_event) tags.push("掉帧失步");
      if (curFrameFault.av_splice_transient) tags.push("切点瞬态");
      if (_audioTl.summary && typeof _audioTl.summary.integrated_lufs === "number") {
        tags.push(`${_audioTl.summary.integrated_lufs.toFixed(1)} LUFS`);
      }
      badge.textContent = `${modeLabel} · ` + (tags.length ? tags.join(" | ") : "音画同步正常");
      badge.style.color = (curFrameFault.afm_unlock || curFrameFault.hdmi_drop) ? "#d9614c" : "";
    } else {
      badge.textContent = `${modeLabel} · t=${tRef.toFixed(2)}s`;
      badge.style.color = "";
    }
  }
}

boot().catch((e) => {
  // Log the real stack before replacing the DOM: a bare message here hides
  // which line actually failed, which cost real debugging time once already.
  console.error("BOOT FAILED:", e, e && e.stack);
  document.body.innerHTML =
    `<pre class="err" style="padding:24px">启动失败：${e.message}\n\n${(e.stack || "").split("\n").slice(0, 8).join("\n")}</pre>`;
});


function _isGraphEngineActive() {
  if (typeof G !== "undefined" && G.perNode) return true;
  const eng = document.getElementById("engineGraph");
  return !!(eng && eng.checked);
}

function _findActiveGraphNode(kind) {
  if (!_isGraphEngineActive() || typeof G === "undefined" || !G.state || !G.state.graph) return null;
  const nodes = G.state.graph.nodes || [];
  const searchList = (list) => {
    for (const nd of list) {
      if (nd.enabled === false) continue;
      if (nd.kind === kind) return nd;
      if (nd.kind === "node_group" && nd.subgraph && Array.isArray(nd.subgraph.nodes)) {
        const found = searchList(nd.subgraph.nodes);
        if (found) return found;
      }
    }
    return null;
  };
  return searchList(nodes);
}

function _hasActiveGraphNode(kind) {
  return !!_findActiveGraphNode(kind);
}

function detectActiveQuadModule() {
  const p = S.params || {};
  const rf = p.rf || {}, ply = p.player || {}, cam = p.camera || {}, dsp = p.display || {}, cn = p.connector || {};
  const flm = p.film || {}, cod = p.codec || {}, osd = p.overlay || {};

  // When Node Graph rendering (`☑ 用节点图渲染`) is active, prioritize any active 2x2 quad lab node in the graph first
  if (_isGraphEngineActive()) {
    if (_hasActiveGraphNode("stream_cdn_client")) {
      return {
        isActive: true,
        nodeKind: "stream_cdn_client",
        layer: "codec",
        cornerKey: "stream_quad_corner",
        borderKey: "stream_show_border",
        textKey: "stream_show_text",
        labels: [
          "分屏: 2×2 四分屏 (网络流媒体传播实验室)",
          "角落 1: 左上 · 采集源与TBC母本",
          "角落 2: 右上 · 1999拨号RealVideo",
          "角落 3: 左下 · Web2.0平台多代转码",
          "角落 4: 右下 · 现代ABR与缓冲遥测",
        ],
      };
    }
    if (_hasActiveGraphNode("photochemical_film")) {
      const nd = _findActiveGraphNode("photochemical_film");
      const bvm = (nd && nd.params && nd.params.bloom_veil_mode !== undefined)
        ? nd.params.bloom_veil_mode : (flm.bloom_veil_mode || 0);
      const isOpticalLab = Math.round(bvm) === 5;
      return {
        isActive: true,
        nodeKind: "photochemical_film",
        layer: "film",
        cornerKey: "quad_corner",
        borderKey: "show_border",
        textKey: "show_text",
        labels: isOpticalLab
          ? [
              "分屏: 2×2 LOMO 光学光晕与散景实验室",
              "角落 1: 左上 · 肥皂泡散景+斜向紫金光雾",
              "角落 2: 右上 · 黄金时刻逆光太阳耀斑洪流",
              "角落 3: 左下 · 无镀膜高调奶油白雾柔焦",
              "角落 4: 右下 · 移轴浅景深+烛火双层光冕",
            ]
          : [
              "分屏: 2×2 LOMO 光化学暗房实验室",
              "角落 1: 左上 · E-6正冲负青绿暖金",
              "角落 2: 右上 · C-41负冲正玫瑰桃心散景",
              "角落 3: 左下 · 过期胶卷高底雾粉彩",
              "角落 4: 右下 · 宝丽来浓暗角与暖阳漏光",
            ],
      };
    }
    if (_hasActiveGraphNode("film_scanner_telecine")) {
      return {
        isActive: true,
        nodeKind: "film_scanner_telecine",
        layer: "camera",
        cornerKey: "telecine_quad_corner",
        borderKey: "telecine_show_border",
        textKey: "telecine_show_text",
        labels: [
          "分屏: 2×2 四分屏 (跨媒介胶转磁实验室)",
          "角落 1: 左上 · 16mm 胶片原片",
          "角落 2: 右上 · 电视飞点胶转磁",
          "角落 3: 左下 · 家用 VHS 电视录屏",
          "角落 4: 右下 · 网络流媒体重压",
        ],
      };
    }
    if (_hasActiveGraphNode("transfer_station")) {
      return {
        isActive: true,
        nodeKind: "transfer_station",
        layer: "player",
        cornerKey: "transfer_quad_corner",
        borderKey: "transfer_show_border",
        textKey: "transfer_show_text",
        labels: [
          "分屏: 2×2 四分屏 (多代复制实验室)",
          "角落 1: 左上 · Gen-0 母带基准",
          "角落 2: 右上 · 无 TBC 多代模拟复制",
          "角落 3: 左下 · 带 TBC 多代模拟复制",
          "角落 4: 右下 · 4:1:1→4:2:0 数字转码",
        ],
      };
    }
    if (_hasActiveGraphNode("inversion_analyzer")) {
      return {
        isActive: true,
        nodeKind: "inversion_analyzer",
        layer: "player",
        cornerKey: "inv_quad_corner",
        borderKey: "inv_show_border",
        textKey: "inv_show_text",
        labels: [
          "分屏: 2×2 四分屏 (FADGI 反演实验室)",
          "角落 1: 左上 · 退化观测画面",
          "角落 2: 右上 · 反演残差热力图",
          "角落 3: 左下 · 多解竞争假设排序",
          "角落 4: 右下 · FADGI 三级审计评分",
        ],
      };
    }
    if (_hasActiveGraphNode("archival_vault")) {
      return {
        isActive: true,
        nodeKind: "archival_vault",
        layer: "player",
        cornerKey: "archival_quad_corner",
        borderKey: "archival_show_border",
        textKey: "archival_show_text",
        labels: [
          "分屏: 2×2 四分屏 (档案谱系树实验室)",
          "角落 1: 左上 · 原始载体残件",
          "角落 2: 右上 · 档案保存母版",
          "角落 3: 左下 · 过度清洗商业修复版",
          "角落 4: 右下 · 流媒体版与重要属性损耗",
        ],
      };
    }
    if (_hasActiveGraphNode("env_chamber")) {
      return {
        isActive: true,
        nodeKind: "env_chamber",
        layer: "connector",
        cornerKey: "env_quad_corner",
        borderKey: "env_show_border",
        textKey: "env_show_text",
        labels: [
          "分屏: 2×2 四分屏 (空间物理环境场实验室)",
          "角落 1: 左上 · 温湿度露点场与结露",
          "角落 2: 右上 · 电磁干扰场与地环路",
          "角落 3: 左下 · 光学与声学空间场耦合",
          "角落 4: 右下 · 设备局域坐标物理遥测",
        ],
      };
    }
    if (_hasActiveGraphNode("tv_exciter_pa")) {
      return {
        isActive: true,
        nodeKind: "tv_exciter_pa",
        layer: "rf",
        cornerKey: "tx_quad_corner",
        borderKey: "tx_show_border",
        textKey: "tx_show_text",
        modeKey: "tx_chain_mode",
        labels: [
          "分屏: 2×2 四分屏 (发射机测试台)",
          "角落 1: 左上 · 主控直通基带参考",
          "角落 2: 右上 · 发射机高功放监测",
          "角落 3: 左下 · 模拟开路接收终端",
          "角落 4: 右下 · 数字地面 DTT 接收",
        ],
      };
    }
    if (_hasActiveGraphNode("sat_ird_receiver")) {
      return {
        isActive: true,
        nodeKind: "sat_ird_receiver",
        layer: "rf",
        cornerKey: "sat_quad_corner",
        borderKey: "sat_show_border",
        textKey: "sat_ird_osd",
        labels: [
          "分屏: 2×2 四分屏 (卫星悬崖实验室)",
          "角落 1: 左上 · 晴空卫星参考",
          "角落 2: 右上 · 模拟 C 波段打火噪点",
          "角落 3: 左下 · 数字 DVB-S 悬崖解码",
          "角落 4: 右下 · 载噪比悬崖响应曲线",
        ],
      };
    }
    if (_hasActiveGraphNode("human_operator")) {
      return {
        isActive: true,
        nodeKind: "human_operator",
        layer: "display",
        cornerKey: "percept_quad_corner",
        borderKey: "percept_show_border",
        textKey: "percept_show_text",
        labels: [
          "分屏: 2×2 四分屏 (人类视觉感知与操作闭环)",
          "角落 1: 左上 · 未干预物理故障输出",
          "角落 2: 右上 · 观众视网膜CSF感知",
          "角落 3: 左下 · 机房操作者闭环矫正",
          "角落 4: 右下 · JND显著度与操作面板",
        ],
      };
    }
    if (_hasActiveGraphNode("ecosystem_simulator")) {
      return {
        isActive: true,
        nodeKind: "ecosystem_simulator",
        layer: "overlay",
        cornerKey: "eco_quad_corner",
        borderKey: "eco_show_border",
        textKey: "eco_show_text",
        labels: [
          "分屏: 2×2 四分屏 (媒介技术生态演化实验室)",
          "角落 1: 左上 · 1992模拟黄金期基准",
          "角落 2: 右上 · 所选年份真实生态状态",
          "角落 3: 左下 · 反事实情景推演状态",
          "角落 4: 右下 · 1975-2035七维演化曲线",
        ],
      };
    }
    if (_hasActiveGraphNode("crt_magnet_env")) {
      return {
        isActive: true,
        nodeKind: "crt_magnet_env",
        layer: "display",
        cornerKey: "crt_magnet_quad_corner",
        borderKey: "crt_magnet_show_border",
        textKey: "crt_magnet_show_text",
        labels: [
          "分屏: 2×2 四分屏 (CRT 强磁场与消磁实验室)",
          "角落 1: 左上 · 强钕铁硼磁铁贴面黑洞与彩虹环",
          "角落 2: 右上 · 强磁极旋转洛伦兹涡流",
          "角落 3: 左下 · 撤磁后铁磁荫罩永久剩磁色斑",
          "角落 4: 右下 · 内置 PTC 消磁线圈交变振荡净化",
        ],
      };
    }
    if (_hasActiveGraphNode("multiview_wall")) {
      return {
        isActive: true,
        nodeKind: "multiview_wall",
        layer: "display",
        cornerKey: "mv_quad_corner",
        borderKey: "mv_show_border",
        textKey: "mv_tally_umd",
        labels: [
          "分屏: 2×2 四分屏 (演播室多画面监视墙)",
          "角落 1: 左上 · CH1 摄像机源",
          "角落 2: 右上 · CH2 磁带直出 (VTR)",
          "角落 3: 左下 · CH3 时基校正 (TBC)",
          "角落 4: 右下 · CH4 主播出 (PGM)",
        ],
      };
    }
  }

  if (Math.round(rf.tx_chain_mode || 0) >= 1) {
    return {
      isActive: true,
      layer: "rf",
      cornerKey: "tx_quad_corner",
      borderKey: "tx_show_border",
      textKey: "tx_show_text",
      modeKey: "tx_chain_mode",
      labels: [
        "分屏: 2×2 四分屏 (发射机测试台)",
        "角落 1: 左上 · 主控直通基带参考",
        "角落 2: 右上 · 发射机高功放监测",
        "角落 3: 左下 · 模拟开路接收终端",
        "角落 4: 右下 · 数字地面 DTT 接收",
      ],
    };
  }
  if (Math.round(rf.sat_ird_mode || 0) >= 1) {
    return {
      isActive: true,
      layer: "rf",
      cornerKey: "sat_quad_corner",
      borderKey: "sat_show_border",
      textKey: "sat_ird_osd",
      labels: [
        "分屏: 2×2 四分屏 (卫星悬崖实验室)",
        "角落 1: 左上 · 晴空卫星参考",
        "角落 2: 右上 · 模拟 C 波段打火噪点",
        "角落 3: 左下 · 数字 DVB-S 悬崖解码",
        "角落 4: 右下 · 载噪比悬崖响应曲线",
      ],
    };
  }
  if (Math.round(ply.transfer_station_mode || 0) >= 1) {
    return {
      isActive: true,
      layer: "player",
      cornerKey: "transfer_quad_corner",
      borderKey: "transfer_show_border",
      textKey: "transfer_show_text",
      labels: [
        "分屏: 2×2 四分屏 (多代复制实验室)",
        "角落 1: 左上 · Gen-0 母带基准",
        "角落 2: 右上 · 无 TBC 多代模拟复制",
        "角落 3: 左下 · 带 TBC 多代模拟复制",
        "角落 4: 右下 · 4:1:1→4:2:0 数字转码",
      ],
    };
  }
  if ((flm.enabled || 0) > 0.5 && (Math.round(flm.bloom_veil_mode || 0) === 5 || Math.round(flm.process_mode || 0) === 5 || Math.round(flm.telecine_pulldown || 0) === 3)) {
    const isOpticalLab = Math.round(flm.bloom_veil_mode || 0) === 5;
    return {
      isActive: true,
      layer: "film",
      cornerKey: "quad_corner",
      borderKey: "show_border",
      textKey: "show_text",
      labels: isOpticalLab
        ? [
            "分屏: 2×2 LOMO 光学光晕与散景实验室",
            "角落 1: 左上 · 肥皂泡散景+斜向紫金光雾",
            "角落 2: 右上 · 黄金时刻逆光太阳耀斑洪流",
            "角落 3: 左下 · 无镀膜高调奶油白雾柔焦",
            "角落 4: 右下 · 移轴浅景深+烛火双层光冕",
          ]
        : [
            "分屏: 2×2 LOMO 光化学暗房实验室",
            "角落 1: 左上 · E-6正冲负青绿暖金",
            "角落 2: 右上 · C-41负冲正玫瑰桃心散景",
            "角落 3: 左下 · 过期胶卷高底雾粉彩",
            "角落 4: 右下 · 宝丽来浓暗角与暖阳漏光",
          ],
    };
  }
  if ((cam.film_enabled || 0) > 0.5 && Math.round(cam.telecine_pulldown || 0) === 3) {
    return {
      isActive: true,
      layer: "camera",
      cornerKey: "telecine_quad_corner",
      borderKey: "telecine_show_border",
      textKey: "telecine_show_text",
      labels: [
        "分屏: 2×2 四分屏 (跨媒介胶转磁实验室)",
        "角落 1: 左上 · 16mm 胶片原片",
        "角落 2: 右上 · 电视飞点胶转磁",
        "角落 3: 左下 · 家用 VHS 电视录屏",
        "角落 4: 右下 · 网络流媒体重压",
      ],
    };
  }
  if ((ply.inv_lab_enabled || 0) > 0.5 || (ply.inv_lab_mode || 0) >= 1) {
    return {
      isActive: true,
      layer: "player",
      cornerKey: "inv_quad_corner",
      borderKey: "inv_show_border",
      textKey: "inv_show_text",
      labels: [
        "分屏: 2×2 四分屏 (FADGI 反演实验室)",
        "角落 1: 左上 · 退化观测画面",
        "角落 2: 右上 · 反演残差热力图",
        "角落 3: 左下 · 多解竞争假设排序",
        "角落 4: 右下 · FADGI 三级审计评分",
      ],
    };
  }
  if ((ply.archival_enabled || 0) > 0.5 || (ply.archival_mode || 0) >= 1) {
    return {
      isActive: true,
      layer: "player",
      cornerKey: "archival_quad_corner",
      borderKey: "archival_show_border",
      textKey: "archival_show_text",
      labels: [
        "分屏: 2×2 四分屏 (档案谱系树实验室)",
        "角落 1: 左上 · 原始载体残件",
        "角落 2: 右上 · 档案保存母版",
        "角落 3: 左下 · 过度清洗商业修复版",
        "角落 4: 右下 · 流媒体版与重要属性损耗",
      ],
    };
  }
  if ((cn.env_field_enabled || 0) > 0.5 || (cn.env_field_mode || 0) >= 1) {
    return {
      isActive: true,
      layer: "connector",
      cornerKey: "env_quad_corner",
      borderKey: "env_show_border",
      textKey: "env_show_text",
      labels: [
        "分屏: 2×2 四分屏 (空间物理环境场实验室)",
        "角落 1: 左上 · 温湿度露点场与结露",
        "角落 2: 右上 · 电磁干扰场与地环路",
        "角落 3: 左下 · 光学与声学空间场耦合",
        "角落 4: 右下 · 设备局域坐标物理遥测",
      ],
    };
  }
  if ((cod.stream_enabled || 0) > 0.5 || (cod.stream_mode || 0) >= 1) {
    return {
      isActive: true,
      layer: "codec",
      cornerKey: "stream_quad_corner",
      borderKey: "stream_show_border",
      textKey: "stream_show_text",
      labels: [
        "分屏: 2×2 四分屏 (网络流媒体传播实验室)",
        "角落 1: 左上 · 采集源与TBC母本",
        "角落 2: 右上 · 1999拨号RealVideo",
        "角落 3: 左下 · Web2.0平台多代转码",
        "角落 4: 右下 · 现代ABR与缓冲遥测",
      ],
    };
  }
  if ((dsp.percept_enabled || 0) > 0.5 || (dsp.percept_mode || 0) >= 1) {
    return {
      isActive: true,
      layer: "display",
      cornerKey: "percept_quad_corner",
      borderKey: "percept_show_border",
      textKey: "percept_show_text",
      labels: [
        "分屏: 2×2 四分屏 (人类视觉感知与操作闭环)",
        "角落 1: 左上 · 未干预物理故障输出",
        "角落 2: 右上 · 观众视网膜CSF感知",
        "角落 3: 左下 · 机房操作者闭环矫正",
        "角落 4: 右下 · JND显著度与操作面板",
      ],
    };
  }
  if ((osd.eco_enabled || 0) > 0.5 || (osd.eco_mode || 0) >= 1) {
    return {
      isActive: true,
      layer: "overlay",
      cornerKey: "eco_quad_corner",
      borderKey: "eco_show_border",
      textKey: "eco_show_text",
      labels: [
        "分屏: 2×2 四分屏 (媒介技术生态演化实验室)",
        "角落 1: 左上 · 1992模拟黄金期基准",
        "角落 2: 右上 · 所选年份真实生态状态",
        "角落 3: 左下 · 反事实情景推演状态",
        "角落 4: 右下 · 1975-2035七维演化曲线",
      ],
    };
  }
  if (Math.round(dsp.crt_magnet_mode || 0) === 5) {
    return {
      isActive: true,
      layer: "display",
      cornerKey: "crt_magnet_quad_corner",
      borderKey: "crt_magnet_show_border",
      textKey: "crt_magnet_show_text",
      labels: [
        "分屏: 2×2 四分屏 (CRT 强磁场与消磁实验室)",
        "角落 1: 左上 · 强钕铁硼磁铁贴面黑洞与彩虹环",
        "角落 2: 右上 · 强磁极旋转洛伦兹涡流",
        "角落 3: 左下 · 撤磁后铁磁荫罩永久剩磁色斑",
        "角落 4: 右下 · 内置 PTC 消磁线圈交变振荡净化",
      ],
    };
  }
  if ((dsp.mv_enabled || 0) > 0.5) {
    return {
      isActive: true,
      layer: "display",
      cornerKey: "mv_quad_corner",
      borderKey: "mv_show_border",
      textKey: "mv_tally_umd",
      labels: [
        "分屏: 2×2 四分屏 (演播室多画面监视墙)",
        "角落 1: 左上 · CH1 摄像机源",
        "角落 2: 右上 · CH2 磁带直出 (VTR)",
        "角落 3: 左下 · CH3 时基校正 (TBC)",
        "角落 4: 右下 · CH4 主播出 (PGM)",
      ],
    };
  }
  return {
    isActive: false,
    layer: "display",
    cornerKey: "mv_quad_corner",
    borderKey: "mv_show_border",
    textKey: "mv_tally_umd",
    labels: [
      "分屏: 2×2 四分屏 (全链路四阶段同屏)",
      "角落 1: 左上角 · 01 成像源 (全屏)",
      "角落 2: 右上角 · 03 磁带直出 (全屏)",
      "角落 3: 左下角 · 04 TBC 校正 (全屏)",
      "角落 4: 右下角 · 05 最终输出 (全屏)",
    ],
  };
}

function syncQuadBar() {
  const qCorner = $("quadCornerSel"), qBorder = $("quadBorderSel"), qText = $("quadTextSel");
  if (!qCorner || !qBorder || !qText) return;
  const active = detectActiveQuadModule();
  const lay = Object.assign({}, (S.params && S.params[active.layer]) || {});
  if (active.nodeKind) {
    const gNode = _findActiveGraphNode(active.nodeKind);
    if (gNode && gNode.params) {
      Object.assign(lay, gNode.params);
    }
  }
  let cornerVal = Math.round(lay[active.cornerKey] || 0);
  if (active.modeKey && Math.round(lay[active.modeKey] || 0) >= 4) {
    cornerVal = Math.round(lay[active.modeKey]) - 3;
  }
  cornerVal = Math.min(4, Math.max(0, cornerVal));

  // Update option labels to reflect the active quad lab
  if (active.labels && qCorner.options.length === 5) {
    for (let i = 0; i < 5; i++) {
      qCorner.options[i].textContent = active.labels[i];
    }
  }
  qCorner.value = String(cornerVal);
  qBorder.value = String(Math.round(lay[active.borderKey] !== undefined ? lay[active.borderKey] : 2));
  const rawTextVal = Math.round(lay[active.textKey] !== undefined ? lay[active.textKey] : 0);
  qText.value = active.textKey === "mv_tally_umd" ? (rawTextVal >= 1 ? "1" : "0") : String(Math.min(2, Math.max(0, rawTextVal)));

  for (let i = 0; i <= 4; i++) {
    const b = $("btnQuadC" + i);
    if (b) {
      b.classList.toggle("on", i === cornerVal && active.isActive);
      if (active.labels && active.labels[i]) {
        b.title = active.labels[i];
      }
    }
  }
}

function wireQuadBar() {
  const qCorner = $("quadCornerSel"), qBorder = $("quadBorderSel"), qText = $("quadTextSel");
  if (!qCorner || !qBorder || !qText) return;
  const applyAll = (forceCornerVal) => {
    const active = detectActiveQuadModule();
    const cVal = forceCornerVal !== undefined ? forceCornerVal : (parseFloat(qCorner.value) || 0);
    qCorner.value = String(cVal);
    const bVal = parseFloat(qBorder.value) || 0;
    const tVal = parseFloat(qText.value) || 0;
    const payload = {
      params: {
        rf: {
          tx_quad_corner: cVal, tx_show_border: bVal, tx_show_text: tVal,
          sat_quad_corner: cVal, sat_show_border: bVal, sat_ird_osd: tVal,
        },
        player: {
          transfer_quad_corner: cVal, transfer_show_border: bVal, transfer_show_text: tVal,
          inv_quad_corner: cVal, inv_show_border: bVal, inv_show_text: tVal,
          archival_quad_corner: cVal, archival_show_border: bVal, archival_show_text: tVal,
        },
        connector: {
          env_quad_corner: cVal, env_show_border: bVal, env_show_text: tVal,
        },
        camera: {
          telecine_quad_corner: cVal, telecine_show_border: bVal, telecine_show_text: tVal,
        },
        film: {
          quad_corner: cVal, show_border: bVal, show_text: tVal,
        },
        codec: {
          stream_quad_corner: cVal, stream_show_border: bVal, stream_show_text: tVal,
        },
        display: {
          mv_quad_corner: cVal, mv_show_border: bVal, mv_tally_umd: Math.min(1, tVal),
          percept_quad_corner: cVal, percept_show_border: bVal, percept_show_text: tVal,
          crt_magnet_quad_corner: cVal, crt_magnet_show_border: bVal, crt_magnet_show_text: tVal,
          ldme_flash_show_text: tVal, stream_show_text: tVal,
        },
        overlay: {
          eco_quad_corner: cVal, eco_show_border: bVal, eco_show_text: tVal,
        },
      }
    };
    const rfLay = (S.params && S.params.rf) || {};
    if (Math.round(rfLay.tx_chain_mode || 0) >= 4) {
      payload.params.rf.tx_chain_mode = 3;
    }
    const usingGraph = _isGraphEngineActive();
    // If no quad lab is active yet and not in a quad-node graph, turn on Multiview 2x2 Wall
    if (!active.isActive && !usingGraph) {
      payload.params.display.mv_enabled = 1.0;
      payload.params.display.mv_layout = 0.0;
    } else if (usingGraph && active.nodeKind && active.nodeKind !== "multiview_wall") {
      // Clear accidental mv_enabled from previous fallback clicks when a graph quad node is active
      payload.params.display.mv_enabled = 0.0;
    }
    setProject(payload);
  };
  qCorner.onchange = () => applyAll();
  qBorder.onchange = () => applyAll();
  qText.onchange = () => applyAll();
  for (let i = 0; i <= 4; i++) {
    const b = $("btnQuadC" + i);
    if (b) {
      b.onclick = () => applyAll(i);
    }
  }
}
