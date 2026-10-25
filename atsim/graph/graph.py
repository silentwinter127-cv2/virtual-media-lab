"""
The node graph.

The linear chain this project started with (`camera -> format -> tape -> player
-> display`) is a special case of a graph, and it is the wrong shape for the
thing users actually want to describe: an *apparatus*.  A real signal path is a
sequence of devices wired together, and the interesting questions are about the
wiring:

    camera -> encoder -> tape -> deck -> TV
    HDMI source -> AV converter -> RF modulator -> tuner -> TV
    camera -> encoder -> (two tapes, spliced) -> deck
    source -> [VHS 节点组 (camera -> encoder -> tape -> deck)] -> 字幕机 -> TV

Supports Blender-style Node Groups (`node_group`): multiple connected nodes can
be packed into a named Node Group (e.g. "VHS"), duplicated (in parallel or in
series), opened/expanded to inspect and edit internal node parameters, or
ungrouped back into the parent graph.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# signal types carried by edges
# ---------------------------------------------------------------------------
TYPES: dict[str, dict] = {
    "rgb": {
        "label": "RGB 画面",
        "note": "成像系统的输出：光，尚无任何视频编码",
    },
    "composite": {
        "label": "复合视频",
        "note": "亮度＋色度已调制的单一信号。只有复合格式的损伤机制对它有效",
    },
    "rf": {
        "label": "射频",
        "note": "经调制的 RF 载波（决定频道与邻频干扰），与复合基带不同",
    },
    "dv": {
        "label": "DV 数据流",
        "note": "数字分量：4:1:1/4:2:0 子采样＋DCT 量化后的码流",
    },
    "yiq": {
        "label": "YIQ 分量",
        "note": "分离后的亮度与色度，显示前的形态",
    },
    "hdmi": {
        "label": "HDMI",
        "note": "未压缩数字视频",
    },
    "audio": {"label": "音频", "note": "旁路用途"},
}

# Picture-like baseband signal types that can be converted losslessly at node
# group or stage boundaries (e.g. chaining a VHS group whose deck outputs `yiq`
# into a second VHS group whose camera expects `rgb`, or connecting `deck` to `overlay`).
_PICTURE_TYPES = {"rgb", "yiq", "hdmi"}
_VIDEO_TYPES = {"rgb", "composite", "rf", "dv", "yiq", "hdmi"}


# ---------------------------------------------------------------------------
@dataclass
class Port:
    name: str
    type: str
    label: str = ""


@dataclass
class NodeSpec:
    """A node type: what it accepts, what it produces, what it can be told."""
    kind: str
    label: str
    group: str
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    device: str = ""            # physical device class, for provenance in the UI
    note: str = ""
    params_layer: str | None = None   # which schema layer configures it

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "label": self.label, "group": self.group,
            "device": self.device, "note": self.note,
            "params_layer": self.params_layer,
            "inputs": [{"name": p.name, "type": p.type, "label": p.label} for p in self.inputs],
            "outputs": [{"name": p.name, "type": p.type, "label": p.label} for p in self.outputs],
        }


def _p(name: str, type_: str, label: str = "") -> Port:
    return Port(name, type_, label or name)


# ---------------------------------------------------------------------------
# The node catalogue.
# ---------------------------------------------------------------------------
CATALOG: tuple[NodeSpec, ...] = (
    NodeSpec(
        kind="source", label="素材源", group="输入",
        inputs=(), outputs=(_p("out", "rgb", "画面"),),
        device="数字文件", note="把解码后的画面接入仿真链路",
        params_layer=None,
    ),
    NodeSpec(
        kind="test_generator", label="测试信号发生器", group="输入",
        inputs=(),
        outputs=(_p("out", "composite", "复合测试信号"), _p("rgb", "rgb", "测试图卡")),
        device="Tektronix TSG-170A / Philips PM5544 广播级测试信号发生器",
        note="合成 SMPTE RP-219 彩条+PLUGE、EBU 彩条、FCC 0.5-5.0MHz 多波群、5阶调制阶梯+2T脉冲与 PM5544 圆环测试卡，支持与 TBC 及示波器构成工程闭环校准台",
        params_layer="format",
    ),
    NodeSpec(
        kind="camera", label="成像系统", group="设备",
        inputs=(_p("in", "rgb"),), outputs=(_p("out", "rgb"),),
        device="摄像机/镜头/传感器",
        note="镜头、传感器、曝光与色彩科学。发生在任何视频信号存在之前",
        params_layer="camera",
    ),
    NodeSpec(
        kind="ccu_rack", label="演播室 CCU 机架", group="设备",
        inputs=(_p("in", "rgb", "摄像机头输入"),),
        outputs=(_p("out", "rgb", "CCU 配平输出"),),
        device="Sony CCU-370 / Ikegami MA-79 摄像机控制单元与 Blackburst 同步机架",
        note="通过演播室脐带线缆集中遥调多机位光圈 Iris、总黑电平、R/B 白平衡增益、R/B 黑平衡基座与高光膝点 Knee，并模拟线长引起的副载波相移与外同步掉锁",
        params_layer="camera",
    ),
    NodeSpec(
        kind="encoder", label="信号编码", group="设备",
        inputs=(_p("in", "rgb"),), outputs=(_p("out", "composite"), _p("dv", "dv")),
        device="编码器 / 摄像机输出级",
        note="把画面变成可记录的信号：复合调制的限带与串扰，或 DV 的量化",
        params_layer="format",
    ),
    NodeSpec(
        kind="dv_codec", label="DV 编解码", group="设备",
        inputs=(_p("in", "dv"),), outputs=(_p("out", "dv"),),
        device="MiniDV / DVCAM 记录器",
        note="4:1:1 或 4:2:0 子采样、8x8 DCT、按宏块分配码率的量化",
        params_layer="codec",
    ),
    NodeSpec(
        kind="tape", label="记录介质", group="设备",
        inputs=(_p("in", "composite"), _p("dv", "dv")),
        outputs=(_p("out", "composite"), _p("dv", "dv")),
        device="磁带 / 磁头",
        note="时基误差、失落、折痕、磁头切换、堵塞与划伤。对复合信号与 DV 码流施加不同的损伤机制",
        params_layer="tape",
    ),
    NodeSpec(
        kind="deck", label="播放设备", group="设备",
        inputs=(_p("in", "composite"), _p("dv", "dv")),
        outputs=(_p("out", "yiq"),),
        device="录像机 / 摄像机回放 / TBC",
        note="读取、时基校正、解码与输出。复合分支做梳状分离，DV 分支做解码",
        params_layer="player",
    ),
    NodeSpec(
        kind="overlay", label="字幕机", group="设备",
        inputs=(_p("in", "rgb", "输入信号"),),
        outputs=(_p("out", "rgb", "输出信号"),),
        device="字符发生器 / OSD 芯片 / 光学字幕机",
        note="支持任意位置接入：接在记录介质前则字幕随磁带一同磨损失落，接在播放/显示后则为机内 OSD 叠字",
        params_layer="overlay",
    ),
    NodeSpec(
        kind="display", label="显示设备", group="设备",
        inputs=(_p("in", "yiq"),), outputs=(_p("out", "rgb"),),
        device="CRT / 监视器",
        note="扫描、光栅、伽马、过扫描、几何、屏幕反光",
        params_layer="display",
    ),
    NodeSpec(
        kind="offscreen_camera", label="屏摄重拍相机", group="设备",
        inputs=(_p("in", "rgb"),), outputs=(_p("out", "rgb"),),
        device="CMOS/CCD 屏摄重拍相机",
        note="把视频信号送进虚拟 CRT 后再用虚拟相机重新拍摄：快门-场扫描差拍滚带、像素-荫罩二维摩尔纹、离轴透视与室内环境光反射",
        params_layer="display",
    ),
    NodeSpec(
        kind="master_playout", label="主控播出台 (Playout)", group="设备",
        inputs=(_p("in_a", "rgb", "主节目源 PGM"), _p("in_b", "rgb", "预监/垫片源 PVW")),
        outputs=(
            _p("out", "rgb", "主控母线输出"),
            _p("clean", "rgb", "干净输出 Clean Feed"),
            _p("pvw", "rgb", "预监输出 PVW"),
        ),
        device="电视台总控主播出台 (PVW/PGM + Clean Feed + DSK 下游键控 + As-Run 日志)",
        note="支持 PGM 主播出母线（叠印 DSK 台标/时钟/游走字幕）、100% 无台标的 Clean Feed 干净输出、PVW 预监母线、5秒录像带预卷倒计时及断流自动切「请稍候」备用垫片",
        params_layer="overlay",
    ),
    NodeSpec(
        kind="transfer_station", label="转录与复制工作站", group="设备",
        inputs=(_p("in", "rgb", "输入信号"),),
        outputs=(_p("out", "rgb", "转录输出"),),
        device="模拟数字化采集站 / 磁带多代复制站 / 数字格式转码站 / 四路复制链对照实验室",
        note="支持 VHS/Betacam 数字化采集、1..8 代磁带复制（对比 TBC 对时基抖动的抑制与不可逆带宽/色度损失）、4:2:2→4:1:1→4:2:0 跨格式转码及 2×2 四路复制对照实验室，生成完整 TransferEvent 溯源链",
        params_layer="player",
    ),
    NodeSpec(
        kind="linear_editor", label="磁带线性编辑台", group="设备",
        inputs=(_p("in_a", "rgb", "录机主轨 Shot A"), _p("in_b", "rgb", "源机素材 Shot B")),
        outputs=(_p("out", "rgb", "编辑输出"),),
        device="Sony BVE-9100 / RM-450 双机磁带线性编辑控制器 (Assemble / Insert Edit)",
        note="协调源机与录机 5 秒预卷伺服锁相、模拟 CTL 脉冲寻址 ±1..3 帧入出点机械滑移、组合编辑 (Assemble) 切点闪黑与尾端断轨，以及旋转飞抹磁头 (Flying Erase Head) 插入编辑",
        params_layer="tape",
    ),
    NodeSpec(
        kind="nle_timeline", label="早期非线编时间线", group="设备",
        inputs=(_p("in_a", "rgb", "视频轨 V1"), _p("in_b", "rgb", "叠加轨 V2")),
        outputs=(_p("out", "rgb", "时间线输出"),),
        device="2000年代初 DV/MPEG 非线性编辑工作站 (Premiere 6 / Canopus DV Storm)",
        note="非破坏性多轨时间线 (EditSession/EditDecision)，对比实时预览下的半分辨率代理降级/缓冲区掉帧卡顿与最终渲染导出的二次 8×8 DCT 重编码差异",
        params_layer="tape",
    ),
    NodeSpec(
        kind="multiview_wall", label="多画面监视墙", group="设备",
        inputs=(
            _p("in_a", "rgb", "CH1 信号源"),
            _p("in_b", "rgb", "CH2 信号源"),
            _p("in_c", "rgb", "CH3 信号源"),
            _p("in_d", "rgb", "CH4 主监 PGM"),
        ),
        outputs=(_p("out", "rgb"),),
        device="演播室 CRT 监视墙 (SONY PVM / JVC / Ikegami 4-6路阵列)",
        note="支持 2×2 四分屏、3×2 六分屏与 1+3 导播主副墙，每台监视器具备独立 CRT 荧光粉/偏转老化性格、Tally 指示灯、UMD 源名及物理无信号失锁响应",
        params_layer="display",
    ),
    # --- converters: how a consumer signal path is expressed -------------
    NodeSpec(
        kind="connector", label="链路接口", group="转换",
        inputs=(_p("in", "composite"),), outputs=(_p("out", "composite"),),
        device="HDMI / AV / S-Video / RF 物理接口",
        note="接口受潮、氧化整流与 50/60Hz 地环路交流声滚动横条",
        params_layer="connector",
    ),
    NodeSpec(
        kind="hdmi_out", label="HDMI 输出", group="转换",
        inputs=(_p("in", "rgb"),), outputs=(_p("out", "hdmi"),),
        device="播放器/电脑 HDMI 口",
        note="未压缩数字输出。到这里为止没有任何模拟损伤",
        params_layer=None,
    ),
    NodeSpec(
        kind="hdmi_to_av", label="HDMI→AV 转换器", group="转换",
        inputs=(_p("in", "hdmi"),), outputs=(_p("out", "composite"),),
        device="消费级转换盒",
        note="把数字画面编码成复合基带。廉价转换盒自己的限带与色度处理就在这里引入",
        params_layer="format",
    ),
    NodeSpec(
        kind="rf_modulator", label="RF 调制器", group="转换",
        inputs=(_p("in", "composite"),), outputs=(_p("out", "rf"),),
        device="射频调制器",
        note="把复合基带搬到 VHF/UHF 载波上：频道宽度、图像载波位置、邻频差拍都在这里",
        params_layer="rf",
    ),
    NodeSpec(
        kind="rf_tuner", label="RF 调谐器", group="转换",
        inputs=(_p("in", "rf"),), outputs=(_p("out", "composite"),),
        device="电视/录像机高频头",
        note="解调回基带：载噪比、调谐器噪声系数、行 AFC 失锁与邻频干扰",
        params_layer="rf",
    ),
    NodeSpec(
        kind="vhs_gen", label="VHS 翻录代数单元", group="设备",
        inputs=(_p("in", "composite"),), outputs=(_p("out", "composite"),),
        device="双机转录 (VCR Play -> VCR Rec)",
        note="单次或多次物理重编码与磁带重放世代劣化（带宽指数缩窄、FM底噪正交累加、时基抖动复合、色度垂直下移与群延迟右移、边缘振铃过冲、峰值削波）",
        params_layer="tape",
    ),
    NodeSpec(
        kind="vhs_dub_loop", label="VHS 多代循环迭代池", group="设备",
        inputs=(_p("in", "composite"),), outputs=(_p("out", "composite"),),
        device="多代转录迭代环 (N-Gen Dubbing Loop)",
        note="在单个节点内循环执行 N 代 VHS 复合重录→磁带损伤→重放梳状分离的物理迭代（可通过节点 iterations 或 tape.generation_count 控制代数）",
        params_layer="tape",
    ),
    NodeSpec(
        kind="power_supply", label="共享电源与电气母线", group="设备",
        inputs=(_p("in", "composite"),), outputs=(_p("out", "composite"),),
        device="整流滤波电源 / 镍镉电池内阻 / 共地电机母线",
        note="电解电容干涸 100/120Hz 全波纹波 (Hum Snake)、电池 Shepherd 放电曲线与内阻动态负载压降、冷机上电冲击及电机/行包 EMI 串扰",
        params_layer="connector",
    ),
    NodeSpec(
        kind="dist_amp", label="75Ω 分配放大器 / 三通分路器", group="设备",
        inputs=(_p("in", "composite", "复合输入 (75Ω/Loop)"),),
        outputs=(
            _p("out", "composite", "输出 1 (Out 1)"),
            _p("out_2", "composite", "输出 2 (Out 2)"),
            _p("out_3", "composite", "输出 3 (Out 3 / 监看)"),
        ),
        device="广播级 1×3 视频分配放大器 (Active DA) / BNC 无源三通分路器",
        note="将一路复合视频一分为多路送往下游录像机、切换台与监视器；可切换标准 75Ω、未端接高阻开路 (+6dB 过曝削波)、无源三通双/多端接 (37.5Ω/25Ω 压降发暗与同步顶压缩) 及有源 DA 隔离驱动",
        params_layer="connector",
    ),
    NodeSpec(
        kind="patch_router", label="广播跳线配线架 / 矩阵路由", group="转换",
        inputs=(
            _p("in_a", "composite", "主跳线输入 A (Bus A)"),
            _p("in_b", "composite", "相邻母线 B (串扰源/Aux)"),
        ),
        outputs=(
            _p("out", "composite", "主跳线输出 (PGM)"),
            _p("out_mon", "composite", "监看环出 (MON)"),
        ),
        device="高密度视频跳线配线架 (Bantam/BNC Patchbay) & 模拟矩阵切点",
        note="模拟演播室机房跳线配线架与矩阵路由切点：支持相邻通道电容耦合串扰 (in_b → in_a)、常闭簧片氧化接触噪波，以及 S-Video/YPbPr/SDI 跨制式误插故障表征",
        params_layer="connector",
    ),
    NodeSpec(
        kind="video_switcher", label="模拟视频切换台 / 混合器", group="设备",
        inputs=(_p("in_a", "composite", "主信号 A (Program)"), _p("in_b", "composite", "副信号 B (Preset/Key)")),
        outputs=(_p("out", "composite", "混合复合输出"),),
        device="模拟特技切换台 (Panasonic WJ-AVE5 / WJ-MX12 / Sony XV-AL100)",
        note="支持双路复合输入 (in_a + in_b) 或单路接入内置副源：非同步信号源硬切瞬态 (Hot Switch)、模拟 T-Bar 叠化、斜坡发生器划像 (Wipe) 及亮键/色键 (Luma/Chroma Key)",
        params_layer="connector",
    ),
    NodeSpec(
        kind="transport_servo", label="机械走带与伺服", group="设备",
        inputs=(_p("in", "composite"),), outputs=(_p("out", "composite"),),
        device="供收带盘张力机构 / 主导轴压带轮 / CTL 伺服状态机",
        note="供带盘/收带盘半径与张力演化 (BOT→EOT)、走带操作状态机 (Play/Pause/起停瞬态/无飞抹暂停起录切口/卸带)、方形橡胶皮带老化打滑与压带轮硬化偏心",
        params_layer="tape",
    ),
    NodeSpec(
        kind="sync_tbc", label="同步与时钟系统", group="设备",
        inputs=(_p("in", "yiq"),), outputs=(_p("out", "yiq"),),
        device="LM1881 同步分离 / H-AFC & V-Hold 振荡器 / Pulse Cross",
        note="行场同步脉冲削波、H-Hold 顶部弯钩、V-Hold 注入锁定滚屏与 VBI 消隐带、晶振时钟频偏/抖动、PAL↔NTSC 异制式失配及 Pulse Cross 脉冲十字检视",
        params_layer="player",
    ),
    NodeSpec(
        kind="automation_controller", label="播出自动化编排引擎 v2", group="设备",
        inputs=(
            _p("in_a", "composite", "主播出源 A (Primary)"),
            _p("in_b", "composite", "备机/预卷源 B (Backup/VTR)"),
        ),
        outputs=(
            _p("out", "composite", "自动化主输出 (PGM)"),
            _p("clean", "composite", "净画面母线 (Clean)"),
        ),
        device="总控事件与条件驱动播出自动化编排器 (PlayoutEvent 状态机 + 6类异常分支 + 根因反查链)",
        note="维护完整 PlayoutEvent 状态流转 (SCHEDULED→PREPARING→READY→ON_AIR→COMPLETED)、6 类分级广播异常分支 (MISSING_ASSET/NOT_READY/SIGNAL_LOSS/OVERRIDDEN/FALLBACK/HELD) 与故障因果溯源链",
        params_layer="overlay",
    ),
    NodeSpec(
        kind="sat_dish_lnb", label="卫星天线与 LNB 高频头", group="转换",
        inputs=(_p("in", "composite", "上行/转发器信源"),),
        outputs=(_p("out", "rf", "L-Band 卫星中频 (950-2150MHz)"),),
        device="C/Ku 波段抛物面卫星接收天线 + DRO 低噪声变频器 (LNB)",
        note="建模卫星 EIRP、自由空间损耗、指向误差、ITU-R P.618 暴雨衰减 (Rain Fade)、春秋季日凌热噪爆发 (Sun Outage) 与 LNB 本振温漂/相位噪声",
        params_layer="rf",
    ),
    NodeSpec(
        kind="sat_ird_receiver", label="卫星综合解码接收机 (IRD)", group="设备",
        inputs=(_p("in", "rf", "卫星中频输入 (IF)"),),
        outputs=(_p("out", "composite", "解码基带视频输出"),),
        device="广播级模拟 C 波段 FM 卫星接收机 / 数字 DVB-S (QPSK+Viterbi+RS) 综合接收解码器 (IRD)",
        note="支持模拟卫星 FM 门限打火噪点 (Threshold Sparkles 鱼骨流星点) 与数字 DVB-S 悬崖效应 (Digital Cliff Effect：QEF 零雪花 vs 悬崖区 MPEG-2 宏块崩坏、PTS 失步与死锁静帧) 及同屏实验室",
        params_layer="rf",
    ),
    NodeSpec(
        kind="tv_exciter_pa", label="电视发射机激励与高功放链", group="转换",
        inputs=(_p("in", "composite", "主控播出基带输入"),),
        outputs=(_p("out", "rf", "大功率开路射频发射输出"),),
        device="电视台残留边带 (VSB) / COFDM 激励器 + 速调管/固态高功放 (HPA) + 腔体双工器与铁塔天馈系统",
        note="建模高功放 AM-AM 微分增益压缩 (DG%)、AM-PM 微分相位旋转 (DP°)、VSB 腔体双工器群延迟振铃、920kHz 图像-色度-伴音三载波互调与铁塔馈线驻波反射 (VSWR)，支持端到端四路开路广播对照测试台",
        params_layer="rf",
    ),
    NodeSpec(
        kind="film_optics", label="胶片光学台", group="设备",
        inputs=(_p("in", "rgb", "光学入射"),),
        outputs=(_p("out", "rgb", "乳剂像面"),),
        device="独立胶片镜头光学与抗晕光结构",
        note="独立胶片光学模块 (`film_optics`)：物理区分片基反射晕光 (Halation)、镜头内部炫光与传感器电荷溢出，建模 Remjet 碳黑抗晕光背层吸收、醋酸片基老化背层染料重显及新月镜/移轴/匹兹伐旋焦/肥皂泡散景",
        params_layer="film",
    ),
    NodeSpec(
        kind="photochemical_film", label="胶片光化学冲印台", group="设备",
        inputs=(_p("in", "rgb", "光学像面输入"),),
        outputs=(_p("out", "rgb", "胶片冲印输出"),),
        device="16mm/35mm/120 LOMO 卤化银胶片与 C-41/E-6 冲印暗房",
        note="独立胶片光化学模块 (`01b 胶片`)：建模 10 款电影/LOMO/宝丽来胶片乳剂、H&D 特性曲线、层间抑制效应 (DIR)、正冲负/负冲正 (Xpro)、过期底雾 D_min、Redscale 反卷、弯月镜暗角、粉红桃心散景、海绵漏光与二次重曝",
        params_layer="film",
    ),
    NodeSpec(
        kind="film_scanner_telecine", label="电影胶片与胶转磁台 (16/35mm Telecine)", group="设备",
        inputs=(_p("in", "rgb", "场景/胶片输入"),),
        outputs=(_p("out", "rgb", "胶转磁视频输出"),),
        device="16mm/35mm 卤化银胶片感光冲洗 + 机械片门放映 + Rank Cintel 飞点胶转磁机",
        note="H&D 特性曲线、三层减色法耦合染料、红橙防光晕散射、中密度卤化银颗粒团簇、青/黄非对称褪色、醋酸片基收缩、片门二维抖动与 24fps→29.97i (3:2 Pulldown) / 25i (+4.167%) 胶转磁转换",
        params_layer="camera",
    ),
    NodeSpec(
        kind="inversion_analyzer", label="参数反演与三级误差预算台 (Inversion & FADGI)", group="校准",
        inputs=(_p("in", "yiq", "待测退化信号"),),
        outputs=(_p("out", "yiq", "反演与审计台输出"),),
        device="多解置信区间参数反演求解器 + 三级参数分离审计 + FADGI 1★-4★ 数字化评价仪",
        note="实时反推退化画面的 -3dB 带宽、行时基抖动、片门平移与色度误差置信区间，对滤波器/磁头磨损/光学散焦给出竞争性概率排序，并分离统计 Physical / Device / Artistic 三级参数贡献",
        params_layer="player",
    ),
    NodeSpec(
        kind="archival_vault", label="档案保存与媒介考古库 (Archival Vault)", group="校准",
        inputs=(_p("in", "yiq", "待归档/修复信号"),),
        outputs=(_p("out", "yiq", "档案库多版本谱系输出"),),
        device="四阶段档案生命周期模拟器 + FADGI/NFSA 重要属性损耗评估 + 多版本谱系树 (ArchivalLineageTree)",
        note="模拟『原始载体残件 → 档案保存母版 (Preservation Master) → 过度清洗商业修复版 (Over-Cleaned DNR) → 流媒体传播版』，量化验证『保存成功 ≠ 信息完整』",
        params_layer="player",
    ),
    NodeSpec(
        kind="env_chamber", label="空间物理环境场耦合舱 (Environment Field)", group="接口",
        inputs=(_p("in", "composite", "设备链路输入"),),
        outputs=(_p("out", "composite", "四场物理耦合输出"),),
        device="二维机房空间环境场 (EnvironmentField: 温湿度露点结露 + 近场电磁干扰 + 声学混响 + 混合光源)",
        note="每台设备按二维机房坐标 (x, y) 独立读取局域温湿度、电磁场、光照频谱与声学驻波，复现冷机抱入暖房瞬态结露与变压器近场干扰",
        params_layer="connector",
    ),
    NodeSpec(
        kind="stream_cdn_client", label="网络流媒体与平台转码终端 (Stream CDN Client)", group="转换",
        inputs=(_p("in", "yiq", "采集/播出基带输入"),),
        outputs=(_p("out", "yiq", "流媒体解码输出"),),
        device="RealVideo/WMV + Web 2.0 平台多代转码 + HLS/DASH 自适应流媒体与分组网络传输模拟器",
        note="模拟 GOP 关键帧呼吸脉冲、TCP 预缓冲转圈卡顿、UDP/RTP 丢包切片撕裂、ABR 分辨率阶跃切换与平台多代上传重编码",
        params_layer="codec",
    ),
    NodeSpec(
        kind="human_operator", label="人类视觉感知与机房操作者闭环 (Viewer & Operator)", group="设备",
        inputs=(_p("in", "rgb", "监视器画面输入"),),
        outputs=(_p("out", "rgb", "感知/操作闭环输出"),),
        device="Mannos-Sakrison 心理物理 CSF 视觉感知模型 + 机房值班工程师 PID 手动跟踪与 CCU 配平闭环",
        note="量化『物理实测误差 vs 人眼视角 CPD 主观清晰度/年代感』差异，并复现操作者反应延迟、过调猎振与高压误切事故",
        params_layer="display",
    ),
    NodeSpec(
        kind="ecosystem_simulator", label="媒介技术生态演化与历史模拟器 (Media Ecosystem)", group="校准",
        inputs=(_p("in", "rgb", "信号链画面输入"),),
        outputs=(_p("out", "rgb", "历史生态投射输出"),),
        device="七维宏观媒介技术史动力学模型 (1975–2035 设备存量/备件断崖/技师消亡/反事实情景推演)",
        note="推演 1975–2035 年间备件耗尽与格式迁移引发的存量档案可访问性断崖，并将历史/反事实状态投射为当前信号链老化分布",
        params_layer="overlay",
    ),
    NodeSpec(
        kind="crt_magnet_env", label="CRT 强磁场与特殊环境 (Strong Magnet & Degauss)", group="设备",
        inputs=(_p("in", "rgb", "CRT 画面输入"),),
        outputs=(_p("out", "rgb", "强磁偏转/消磁输出"),),
        device="钕铁硼强永磁体近场偶极子 F=-e(v×B) + 铁磁荫罩剩磁 + PTC 衰减交变消磁线圈",
        note="模拟强磁铁贴近 CRT 时的中心电子束排空黑洞、十字枕形折叠、同心彩虹色纯度环、撤磁后铁磁荫罩永久剩磁色斑及内置 PTC 消磁线圈动态净化",
        params_layer="display",
    ),
    NodeSpec(
        kind="mobile_camera_isp", label="早期手机与 iPhone 成像系统", group="设备",
        inputs=(_p("in", "rgb", "光学场景输入"),),
        outputs=(_p("out", "rgb", "手机 ISP 编码输出"),),
        device="Nokia 6600 / N73 / N95 / 5800 & iPhone 2007 / 3G / 3GS / 4S 六步成像管线",
        note="独立早期手机与早期 iPhone 摄像成像模块：1.微型镜头光学 → 2.传感器泊松-高斯采样与饱和 → 3.Bayer去马赛克 → 4.白平衡与色彩矩阵 → 5.双边降噪涂抹与USM锐化光晕 → 6.JPEG/3GP/MOV编码 (区分 capture 拍照与 record 视频管线)",
        params_layer="camera",
    ),
    NodeSpec(
        kind="file_transfer_node", label="文件传输与多代重编码节点", group="转换",
        inputs=(_p("in", "rgb", "源文件输入"),),
        outputs=(_p("out", "rgb", "传输/重编码输出"),),
        device="蓝牙 OBEX / USB 导入 / 彩信与早期社交软件压缩 / 桌面剪辑导出多代级联",
        note="严格区分无损文件复制（蓝牙/USB 仅按策略保留或更新时间戳，绝不凭空添加图像压缩）与有损应用处理（彩信/社交软件缩放重编码剥离EXIF、桌面剪辑导出多代有损压缩 V0→V1→V2→V3）",
        params_layer="codec",
    ),
    NodeSpec(
        kind="lcd_flash_player", label="早期 LCD 显示器与 Flash 播放器", group="设备",
        inputs=(_p("in", "rgb", "解码视频/图像输入"),),
        outputs=(_p("out", "rgb", "LCD/播放器显示输出"),),
        device="功能机 RGB565 TFT 液晶 / 2007 桌面 6-bit TN 液晶显示器 + Flash Player 窗口缩放与缓冲系统",
        note="严格区分 LCD 与 CRT 物理机制：建模 6-bit+FRC 灰阶量化、液晶分子响应时间保持拖影、TN 垂直视角伽马偏移、CCFL 冷白背光与 RGB 竖条子像素晶格，以及 Flash 播放器非整数倍窗口缩放与网络缓冲卡顿",
        params_layer="display",
    ),
    NodeSpec(
        kind="av_sync_clock", label="音视频同步与时钟系统", group="音频",
        inputs=(_p("in", "rgb", "视频画面输入"),),
        outputs=(_p("out", "rgb", "时钟同步画面输出"),),
        device="10 · 音视频同步与时钟系统 (PTS / 采样率晶振漂移 / 掉帧重发 / 剪辑点瞬态)",
        note="建模视频 PTS 与音频采样时钟对位、固定唇音偏移 (ms)、晶振时钟漂移 (ppm) 线性累积跑偏、视频掉帧重复帧 vs 音频 FIFO 滑移/PLL 变调以及剪辑点静音/爆音瞬态，串联摄像机、DV、磁带、广电、剪辑、网络流媒体与全部 10 大音频子系统",
        params_layer="audio",
    ),
    NodeSpec(
        kind="node_group", label="节点组", group="节点组",
        inputs=(_p("in", "rgb", "输入"),),
        outputs=(_p("out", "yiq", "输出"),),
        device="Blender 风格打包节点组 (Node Group)",
        note="由多个设备节点打包而成的复合节点组，可整体复制、串联、解包，或双击/点选展开修改内部各个节点的参数",
        params_layer=None,
    ),
)

CATALOG_BY_KIND: dict[str, NodeSpec] = {n.kind: n for n in CATALOG}


# ---------------------------------------------------------------------------
@dataclass
class Node:
    name: str
    kind: str
    params: dict = field(default_factory=dict)
    position: tuple[float, float] = (0.0, 0.0)
    enabled: bool = True
    px_position: bool = False
    iterations: int = 1
    label: str = ""
    subgraph: dict | None = None

    @property
    def spec(self) -> NodeSpec:
        return CATALOG_BY_KIND[self.kind]

    @property
    def display_label(self) -> str:
        if self.label:
            return self.label
        return self.spec.label

    @property
    def effective_inputs(self) -> tuple[Port, ...]:
        if self.kind == "node_group" and isinstance(self.subgraph, dict):
            ins = self.subgraph.get("inputs") or []
            if ins:
                return tuple(
                    Port(name=str(p.get("name", "in")),
                         type=str(p.get("type", "rgb")),
                         label=str(p.get("label") or p.get("name", "in")))
                    for p in ins
                )
        return self.spec.inputs

    @property
    def effective_outputs(self) -> tuple[Port, ...]:
        if self.kind == "node_group" and isinstance(self.subgraph, dict):
            outs = self.subgraph.get("outputs") or []
            if outs:
                return tuple(
                    Port(name=str(p.get("name", "out")),
                         type=str(p.get("type", "yiq")),
                         label=str(p.get("label") or p.get("name", "out")))
                    for p in outs
                )
        return self.spec.outputs


@dataclass
class Edge:
    src: str
    src_port: str
    dst: str
    dst_port: str


@dataclass
class Issue:
    severity: str          # "error" | "warning"
    message: str
    node: str | None = None

    def to_dict(self) -> dict:
        return {"severity": self.severity, "message": self.message, "node": self.node}


def _ports_compatible(src_node: Node, src_type: str, dst_node: Node, dst_type: str) -> bool:
    """
    Determine whether an edge from `(src_node, src_type)` to `(dst_node, dst_type)` is valid.

    - Exact type matches are always valid.
    - `overlay` (`字幕机`) and `connector` can sit at any point in the video signal chain
      (accepting and forwarding `rgb`, `composite`, `yiq`, `hdmi`, `dv`).
    - Baseband picture formats (`rgb`, `yiq`, `hdmi`) are mutually compatible across
      `node_group` boundaries and between `deck` (`yiq`), `camera`/`encoder` (`rgb`), and
      `display` (`yiq`), because `GraphExecutor` automatically converts between `SignalFrame`
      (YIQ) and `rgb` when crossing stage boundaries.
    """
    if src_type == dst_type:
        return True
    if src_node.kind == "overlay" or dst_node.kind == "overlay":
        return (src_type in _VIDEO_TYPES) and (dst_type in _VIDEO_TYPES)
    if src_node.kind in ("connector", "dist_amp", "patch_router") or dst_node.kind in ("connector", "dist_amp", "patch_router"):
        return (src_type in _VIDEO_TYPES) and (dst_type in _VIDEO_TYPES)
    if src_node.kind in ("test_generator", "multiview_wall", "ccu_rack", "master_playout", "transfer_station", "linear_editor", "nle_timeline", "automation_controller", "sat_dish_lnb", "sat_ird_receiver", "tv_exciter_pa", "photochemical_film", "film_scanner_telecine", "inversion_analyzer", "archival_vault", "env_chamber", "crt_magnet_env", "av_sync_clock") or dst_node.kind in ("test_generator", "multiview_wall", "ccu_rack", "master_playout", "transfer_station", "linear_editor", "nle_timeline", "automation_controller", "sat_dish_lnb", "sat_ird_receiver", "tv_exciter_pa", "photochemical_film", "film_scanner_telecine", "inversion_analyzer", "archival_vault", "env_chamber", "crt_magnet_env", "av_sync_clock"):
        return (src_type in _VIDEO_TYPES) and (dst_type in _VIDEO_TYPES)
    if src_type in _PICTURE_TYPES and dst_type in _PICTURE_TYPES:
        return True
    return False


# ---------------------------------------------------------------------------
class Graph:
    """A validated DAG of device nodes (supporting nested `node_group` subgraphs)."""

    def __init__(self, nodes: Iterable[Node] = (), edges: Iterable[Edge] = (),
                 seed: int = 20240101):
        self.nodes: dict[str, Node] = {n.name: n for n in nodes}
        self.edges: list[Edge] = list(edges)
        self.seed = int(seed)

    # -- construction ---------------------------------------------------
    def add(self, name: str, kind: str, params: dict | None = None,
            position: tuple[float, float] | None = None,
            enabled: bool = True, iterations: int = 1,
            label: str = "", subgraph: dict | None = None) -> Node:
        if kind not in CATALOG_BY_KIND:
            raise KeyError(f"unknown node kind {kind!r}")
        it = max(1, int(iterations))
        p_dict = dict(params or {})
        if it > 1 and "iterations" not in p_dict:
            p_dict["iterations"] = float(it)
        n = Node(
            name=name, kind=kind, params=p_dict,
            position=position or (0.0, 0.0), enabled=bool(enabled),
            iterations=it, label=str(label or ""),
            subgraph=copy.deepcopy(subgraph) if isinstance(subgraph, dict) else None,
        )
        if kind == "display" and "enabled" not in p_dict:
            n.enabled = False
        self.nodes[name] = n
        return n

    def connect(self, src: str, src_port: str, dst: str, dst_port: str) -> Edge:
        # Remove any existing connection to the same destination port first
        self.disconnect(dst, dst_port)
        e = Edge(src, src_port, dst, dst_port)
        self.edges.append(e)
        return e

    def disconnect(self, dst: str, dst_port: str) -> None:
        self.edges = [e for e in self.edges
                      if not (e.dst == dst and e.dst_port == dst_port)]

    def remove(self, name: str) -> None:
        self.nodes.pop(name, None)
        self.edges = [e for e in self.edges if e.src != name and e.dst != name]

    def to_dict(self) -> dict:
        out_nodes = []
        for n in self.nodes.values():
            nd = {
                "name": n.name,
                "kind": n.kind,
                "params": copy.deepcopy(n.params),
                "position": list(n.position),
                "enabled": bool(n.enabled),
                "px": bool(getattr(n, "px_position", False)),
                "iterations": int(max(1, getattr(n, "iterations", 1))),
            }
            if n.label:
                nd["label"] = n.label
            if n.subgraph is not None:
                nd["subgraph"] = copy.deepcopy(n.subgraph)
            out_nodes.append(nd)
        return {
            "seed": self.seed,
            "nodes": out_nodes,
            "edges": [
                {"src": e.src, "src_port": e.src_port,
                 "dst": e.dst, "dst_port": e.dst_port}
                for e in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Graph":
        g = cls(seed=int(d.get("seed", 20240101)))
        for nd in d.get("nodes", []):
            kind = nd["kind"]
            g.add(
                nd["name"], kind, nd.get("params"),
                tuple(nd.get("position", (0.0, 0.0))),
                enabled=bool(nd.get("enabled", True)),
                iterations=int(max(1, int(nd.get("iterations", 1)))),
                label=str(nd.get("label", "")),
                subgraph=nd.get("subgraph"),
            )
            node = g.nodes[nd["name"]]
            node.enabled = bool(nd.get("enabled", True))
            node.px_position = bool(nd.get("px", False))
            node.iterations = int(max(1, int(nd.get("iterations", 1))))
        for ed in d.get("edges", []):
            g.connect(ed["src"], ed["src_port"], ed["dst"], ed["dst_port"])
        return g

    # -- queries --------------------------------------------------------
    def upstream(self, name: str) -> list[str]:
        """Direct predecessors of a node."""
        return [e.src for e in self.edges if e.dst == name]

    def ancestors(self, name: str) -> set[str]:
        """Every node that the given node's output depends on."""
        seen: set[str] = set()
        stack = [name]
        while stack:
            cur = stack.pop()
            for up in self.upstream(cur):
                if up not in seen:
                    seen.add(up)
                    stack.append(up)
        return seen

    def downstream(self, name: str) -> set[str]:
        out: set[str] = set()
        stack = [name]
        while stack:
            cur = stack.pop()
            for e in self.edges:
                if e.src == cur and e.dst not in out:
                    out.add(e.dst)
                    stack.append(e.dst)
        return out

    def inputs_of(self, name: str) -> dict[str, Edge]:
        return {e.dst_port: e for e in self.edges if e.dst == name}

    def order(self) -> list[str]:
        """Topological order (Kahn).  Raises if the graph has a cycle."""
        indeg: dict[str, int] = {n: 0 for n in self.nodes}
        for e in self.edges:
            if e.dst in indeg:
                indeg[e.dst] += 1
        queue = [n for n in self.nodes if indeg[n] == 0]
        out: list[str] = []
        while queue:
            cur = queue.pop(0)
            out.append(cur)
            for e in self.edges:
                if e.src != cur or e.dst not in indeg:
                    continue
                indeg[e.dst] -= 1
                if indeg[e.dst] == 0:
                    queue.append(e.dst)
        if len(out) != len(self.nodes):
            raise ValueError("graph has a cycle")
        return out

    # -- Blender-style Node Group operations ----------------------------
    def group_nodes(self, names: list[str], label: str = "VHS",
                    group_name: str | None = None,
                    base_params: dict | None = None) -> Node:
        """
        Pack multiple selected nodes (e.g. `camera`, `encoder`, `tape`, `deck`) into a
        single Blender-style `node_group` node (e.g. labeled `"VHS"`).

        - Preserves the internal nodes, their topological wiring, and a snapshot of their
          parameters so the group is self-contained and can be duplicated or edited.
        - Automatically exposes boundary `inputs` and `outputs` and re-routes all external
          edges into and out of the new Node Group.
        """
        topo = self.order()
        raw_valid = [n for n in topo if n in set(names) and n in self.nodes]
        # Never pack the `source` node into an effector group if other nodes are selected
        non_source = [n for n in raw_valid if self.nodes[n].kind != "source"]
        selected = non_source if non_source else raw_valid
        if not selected:
            raise ValueError("请先在画布上选择至少一个要打包的设备节点（不含素材源）")
        sel_set = set(selected)

        # Check convexity: ensure packing `sel_set` will not create a cycle in the parent graph
        # (i.e. an unselected node lying on a path between two selected nodes).
        outside_Reachable_from_sel: set[str] = set()
        for s_name in selected:
            for d_name in self.downstream(s_name):
                if d_name not in sel_set:
                    outside_Reachable_from_sel.add(d_name)
        for out_n in outside_Reachable_from_sel:
            if self.downstream(out_n) & sel_set:
                raise ValueError("所选节点之间夹有未选中的中间节点，直接打包会形成环路，请把中间节点一并选入")

        # Build internal nodes list with snapshotted parameters and clean internal layout
        sub_nodes: list[dict] = []
        xs, ys = [], []
        for idx, n_name in enumerate(selected):
            orig = self.nodes[n_name]
            px_x, px_y = float(orig.position[0]), float(orig.position[1])
            xs.append(px_x)
            ys.append(px_y)
            layer = orig.spec.params_layer
            merged_p = {}
            if base_params and layer and isinstance(base_params.get(layer), dict):
                merged_p.update(copy.deepcopy(base_params[layer]))
            if orig.params:
                merged_p.update(copy.deepcopy(orig.params))
            sub_nd = {
                "name": orig.name,
                "kind": orig.kind,
                "label": orig.display_label,
                "params": merged_p,
                "position": [32.0 + (idx + 1) * self.GRID_COL_STEP, 96.0],
                "px": True,
                "enabled": bool(orig.enabled),
                "iterations": int(max(1, getattr(orig, "iterations", 1))),
            }
            if orig.subgraph is not None:
                sub_nd["subgraph"] = copy.deepcopy(orig.subgraph)
            sub_nodes.append(sub_nd)

        # Internal edges (both ends inside `sel_set`)
        internal_edges = [
            {"src": e.src, "src_port": e.src_port, "dst": e.dst, "dst_port": e.dst_port}
            for e in self.edges
            if e.src in sel_set and e.dst in sel_set
        ]

        # Incoming external edges (src outside, dst inside)
        incoming_ext = [e for e in self.edges if e.src not in sel_set and e.dst in sel_set]
        # Outgoing external edges (src inside, dst outside)
        outgoing_ext = [e for e in self.edges if e.src in sel_set and e.dst not in sel_set]

        group_inputs: list[dict] = []
        in_port_map: dict[tuple[str, str], str] = {}
        for e in incoming_ext:
            key = (e.dst, e.dst_port)
            if key not in in_port_map:
                dst_node = self.nodes[e.dst]
                p_type = "rgb"
                for p in dst_node.effective_inputs:
                    if p.name == e.dst_port:
                        p_type = p.type
                        break
                port_name = "in" if not group_inputs else f"in_{len(group_inputs) + 1}"
                in_port_map[key] = port_name
                group_inputs.append({
                    "name": port_name,
                    "type": p_type,
                    "label": f"{dst_node.display_label}.{e.dst_port}",
                    "target_node": e.dst,
                    "target_port": e.dst_port,
                })

        if not group_inputs:
            # Expose the first internal node's primary input port
            first_node = self.nodes[selected[0]]
            first_in = first_node.effective_inputs[0] if first_node.effective_inputs else Port("in", "rgb", "in")
            group_inputs.append({
                "name": "in",
                "type": first_in.type,
                "label": f"{first_node.display_label}.{first_in.name}",
                "target_node": first_node.name,
                "target_port": first_in.name,
            })

        group_outputs: list[dict] = []
        out_port_map: dict[tuple[str, str], str] = {}
        for e in outgoing_ext:
            key = (e.src, e.src_port)
            if key not in out_port_map:
                src_node = self.nodes[e.src]
                p_type = "yiq"
                for p in src_node.effective_outputs:
                    if p.name == e.src_port:
                        p_type = p.type
                        break
                port_name = "out" if not group_outputs else f"out_{len(group_outputs) + 1}"
                out_port_map[key] = port_name
                group_outputs.append({
                    "name": port_name,
                    "type": p_type,
                    "label": f"{src_node.display_label}.{e.src_port}",
                    "source_node": e.src,
                    "source_port": e.src_port,
                })

        if not group_outputs:
            # Expose the last internal node's primary output port
            last_node = self.nodes[selected[-1]]
            last_out = last_node.effective_outputs[0] if last_node.effective_outputs else Port("out", "rgb", "out")
            group_outputs.append({
                "name": "out",
                "type": last_out.type,
                "label": f"{last_node.display_label}.{last_out.name}",
                "source_node": last_node.name,
                "source_port": last_out.name,
            })

        # Generate a clean unique node name for the group
        import re as _re
        clean_label = (label or "").strip() or "VHS"
        if not group_name:
            slug = _re.sub(r"[^a-zA-Z0-9_]+", "_", clean_label.lower()).strip("_") or "group"
            base_id = slug if slug not in self.nodes or slug in sel_set else f"{slug}_group"
            cand = base_id
            idx_g = 2
            while cand in self.nodes and cand not in sel_set:
                cand = f"{base_id}_{idx_g}"
                idx_g += 1
            group_name = cand

        cx = sum(xs) / max(1, len(xs))
        cy = sum(ys) / max(1, len(ys))

        subgraph_dict = {
            "label": clean_label,
            "nodes": sub_nodes,
            "edges": internal_edges,
            "inputs": group_inputs,
            "outputs": group_outputs,
        }

        # Remove packed nodes and their edges
        for n_name in selected:
            self.remove(n_name)

        g_node = self.add(
            group_name, "node_group", params={},
            position=(cx, cy), enabled=True, iterations=1,
            label=clean_label, subgraph=subgraph_dict,
        )
        g_node.px_position = True

        # Rewire incoming and outgoing external edges to `g_node`
        for e in incoming_ext:
            g_port = in_port_map.get((e.dst, e.dst_port), "in")
            if e.src in self.nodes:
                self.connect(e.src, e.src_port, group_name, g_port)

        for e in outgoing_ext:
            g_port = out_port_map.get((e.src, e.src_port), "out")
            if e.dst in self.nodes:
                self.connect(group_name, g_port, e.dst, e.dst_port)

        return g_node

    def ungroup_node(self, group_name: str) -> list[str]:
        """
        Unpack a `node_group` node back into the parent graph, restoring its internal
        nodes and reconnecting incoming and outgoing edges.
        """
        g_node = self.nodes.get(group_name)
        if g_node is None or g_node.kind != "node_group" or not isinstance(g_node.subgraph, dict):
            raise ValueError(f"节点 {group_name!r} 不是可解包的节点组")

        sub = g_node.subgraph
        sub_nodes = sub.get("nodes") or []
        sub_edges = sub.get("edges") or []
        sub_inputs = {p.get("name", "in"): p for p in (sub.get("inputs") or [])}
        sub_outputs = {p.get("name", "out"): p for p in (sub.get("outputs") or [])}

        incoming = [e for e in self.edges if e.dst == group_name]
        outgoing = [e for e in self.edges if e.src == group_name]

        gx, gy = float(g_node.position[0]), float(g_node.position[1])
        self.remove(group_name)

        name_map: dict[str, str] = {}
        restored_names: list[str] = []
        n_count = max(1, len(sub_nodes))
        start_x = gx - ((n_count - 1) * 185.0) * 0.5
        for idx, nd in enumerate(sub_nodes):
            old_name = str(nd["name"])
            new_name = old_name
            k = 2
            while new_name in self.nodes:
                new_name = f"{old_name}_{k}"
                k += 1
            name_map[old_name] = new_name
            node = self.add(
                new_name,
                str(nd["kind"]),
                params=copy.deepcopy(nd.get("params") or {}),
                position=(start_x + idx * 185.0, gy),
                enabled=bool(nd.get("enabled", True)),
                iterations=int(max(1, int(nd.get("iterations", 1)))),
                label=str(nd.get("label", "")),
                subgraph=copy.deepcopy(nd.get("subgraph")) if nd.get("subgraph") else None,
            )
            node.enabled = bool(nd.get("enabled", True))
            node.px_position = True
            restored_names.append(new_name)

        for ed in sub_edges:
            s_new = name_map.get(ed["src"])
            d_new = name_map.get(ed["dst"])
            if s_new and d_new:
                self.connect(s_new, ed["src_port"], d_new, ed["dst_port"])

        for e in incoming:
            pin = sub_inputs.get(e.dst_port) or (next(iter(sub_inputs.values()), None))
            if pin and e.src in self.nodes:
                t_node = name_map.get(str(pin.get("target_node", "")))
                t_port = str(pin.get("target_port", "in"))
                if t_node and t_node in self.nodes:
                    self.connect(e.src, e.src_port, t_node, t_port)

        for e in outgoing:
            pout = sub_outputs.get(e.src_port) or (next(iter(sub_outputs.values()), None))
            if pout and e.dst in self.nodes:
                s_node = name_map.get(str(pout.get("source_node", "")))
                s_port = str(pout.get("source_port", "out"))
                if s_node and s_node in self.nodes:
                    self.connect(s_node, s_port, e.dst, e.dst_port)

        return restored_names

    def duplicate_node(self, src_name: str, chain_in_series: bool = False,
                       new_label: str | None = None) -> Node:
        """
        Duplicate a node or `node_group` (deep-copying its `params` and `subgraph`).
        If `chain_in_series=True`, inserts the duplicate immediately downstream of `src_name`.
        """
        node = self.nodes.get(src_name)
        if node is None:
            raise KeyError(f"no such node {src_name!r}")
        if node.kind == "source":
            raise ValueError("素材源节点不支持复制")

        base = src_name.rstrip("0123456789_") or node.kind
        idx = 2
        new_name = f"{base}_{idx}"
        while new_name in self.nodes:
            idx += 1
            new_name = f"{base}_{idx}"

        nx = float(node.position[0]) + (210.0 if node.px_position else 1.0)
        ny = float(node.position[1]) + (32.0 if node.px_position else 0.0)
        lbl = new_label if new_label is not None else (
            f"{node.label} #{idx}" if node.kind == "node_group" and node.label else node.label
        )
        dup_sub = copy.deepcopy(node.subgraph) if node.subgraph is not None else None
        if isinstance(dup_sub, dict) and lbl:
            dup_sub["label"] = lbl

        dup = self.add(
            new_name,
            node.kind,
            params=copy.deepcopy(node.params or {}),
            position=(nx, ny),
            enabled=bool(node.enabled),
            iterations=int(getattr(node, "iterations", 1)),
            label=lbl,
            subgraph=dup_sub,
        )
        dup.enabled = bool(node.enabled)
        dup.px_position = bool(node.px_position)

        if chain_in_series and node.effective_inputs and node.effective_outputs:
            out_ports = list(node.effective_outputs)
            in_ports = list(dup.effective_inputs)
            matched_out, matched_in = None, None
            for op in out_ports:
                for ip in in_ports:
                    if _ports_compatible(node, op.type, dup, ip.type):
                        matched_out, matched_in = op.name, ip.name
                        break
                if matched_out:
                    break
            if matched_out and matched_in:
                downstream = [e for e in list(self.edges)
                              if e.src == src_name and e.src_port == matched_out]
                for e in downstream:
                    self.edges.remove(e)
                self.connect(src_name, matched_out, new_name, matched_in)
                for e in downstream:
                    self.connect(new_name, matched_out, e.dst, e.dst_port)
        elif not chain_in_series:
            # Connect the same upstream inputs as `src_name` so the copied node/group is ready to use
            for port_name, e in self.inputs_of(src_name).items():
                self.connect(e.src, e.src_port, new_name, port_name)

        return dup

    def update_subnode(self, group_name: str, subnode_name: str, payload: dict) -> dict | None:
        """
        Modify an internal node inside a `node_group` (supports slash-separated `group_path`
        for nested groups).
        """
        parts = [p for p in str(group_name).split("/") if p]
        if not parts:
            return None
        root = self.nodes.get(parts[0])
        if root is None or root.kind != "node_group" or not isinstance(root.subgraph, dict):
            return None
        cur_sub = root.subgraph
        for p in parts[1:]:
            found_next = None
            for nd in (cur_sub.get("nodes") or []):
                if nd.get("name") == p and nd.get("kind") == "node_group" and isinstance(nd.get("subgraph"), dict):
                    found_next = nd["subgraph"]
                    break
            if found_next is None:
                return None
            cur_sub = found_next

        for nd in (cur_sub.get("nodes") or []):
            if nd.get("name") == subnode_name:
                if "enabled" in payload:
                    nd["enabled"] = bool(payload["enabled"])
                if "label" in payload and payload["label"] is not None:
                    nd["label"] = str(payload["label"])
                if "iterations" in payload and payload["iterations"] is not None:
                    nd["iterations"] = max(1, min(25, int(round(float(payload["iterations"])))))
                if "x" in payload and "y" in payload:
                    nd["position"] = [float(payload["x"]), float(payload["y"])]
                    nd["px"] = True
                if isinstance(payload.get("params"), dict):
                    nd_params = nd.setdefault("params", {})
                    for k, v in payload["params"].items():
                        if isinstance(v, str):
                            nd_params[k] = v
                        elif isinstance(v, (int, float)):
                            nd_params[k] = float(v)
                return nd
        return None

    # -- validation -----------------------------------------------------
    def validate(self) -> list[Issue]:
        issues: list[Issue] = []
        try:
            self.order()
        except ValueError as exc:
            issues.append(Issue("error", str(exc)))
            return issues

        for name, node in self.nodes.items():
            ins = self.inputs_of(name)
            eff_inputs = node.effective_inputs
            allowed = {p.name: p.type for p in eff_inputs}
            for port_name in ins:
                if port_name not in allowed:
                    issues.append(Issue("error", f"节点没有输入端口 {port_name!r}", name))
            for p in eff_inputs:
                e = ins.get(p.name)
                if e is None:
                    if not ins:
                        issues.append(Issue("warning", f"输入 {p.label} 未连接", name))
                    continue
                src_node = self.nodes.get(e.src)
                if src_node is None:
                    issues.append(Issue("error", f"上游节点 {e.src!r} 不存在", name))
                    continue
                outs = {q.name: q.type for q in src_node.effective_outputs}
                if e.src_port not in outs:
                    issues.append(Issue("error",
                                        f"{e.src} 没有输出端口 {e.src_port!r}", name))
                    continue
                src_type = outs[e.src_port]
                if not _ports_compatible(src_node, src_type, node, p.type):
                    issues.append(Issue(
                        "error",
                        f"类型不匹配：{e.src}.{e.src_port} 是 {TYPES.get(src_type, {}).get('label', src_type)}，"
                        f"而 {name}.{p.name} 需要 {TYPES.get(p.type, {}).get('label', p.type)}",
                        name))
            if not node.enabled and self.downstream(name):
                issues.append(Issue("warning", "该节点已关闭，下游将收到未处理的信号", name))

        sinks = [n for n in self.nodes if not self.downstream(n)]
        if not sinks:
            issues.append(Issue("error", "图里没有输出节点"))
        return issues

    def output_nodes(self) -> list[str]:
        return [n for n in self.nodes if not self.downstream(n)]

    # -- cache keys -----------------------------------------------------
    def _params_of(self, name: str, base_params: dict | None = None) -> dict:
        node = self.nodes[name]
        layer = node.spec.params_layer
        base_layer = {}
        if base_params and layer and isinstance(base_params.get(layer), dict):
            base_layer = base_params[layer]
        elif base_params and node.kind == "node_group" and isinstance(node.subgraph, dict):
            for snd in (node.subgraph.get("nodes") or []):
                skind = snd.get("kind", "")
                slayer = CATALOG_BY_KIND.get(skind).params_layer if skind in CATALOG_BY_KIND else None
                if slayer and isinstance(base_params.get(slayer), dict):
                    base_layer[slayer] = base_params[slayer]
        return {
            "kind": node.kind,
            "label": node.label,
            "base": base_layer,
            "params": node.params,
            "subgraph": node.subgraph,
            "iterations": int(max(1, node.iterations)),
            "enabled": node.enabled,
        }

    def node_hashes(self, base_params: dict | None = None) -> dict[str, str]:
        """
        A stable hash per node over itself AND all of its ancestors (including
        relevant `base_params` layers when supplied).
        """
        hashes: dict[str, str] = {}
        ov_global = (base_params or {}).get("overlay") if base_params else None
        for name in self.order():
            payload: dict[str, Any] = {"seed": self.seed}
            if ov_global:
                payload["ov_global"] = ov_global
            payload["self"] = self._params_of(name, base_params=base_params)
            for up in sorted(self.ancestors(name)):
                payload[f"up:{up}"] = self._params_of(up, base_params=base_params)
                payload[f"uph:{up}"] = hashes.get(up, "")
            hashes[name] = hashlib.blake2s(
                json.dumps(payload, sort_keys=True, default=str).encode(),
                digest_size=10).hexdigest()
        return hashes

    GRID_COL_STEP = 224.0
    GRID_ROW_STEP = 108.0

    def normalise_positions(self) -> "Graph":
        """Convert this graph's grid columns to pixels, and mark every node as being in pixels."""
        for n in self.nodes.values():
            if not n.px_position:
                col, row = n.position
                n.position = (32.0 + float(col) * self.GRID_COL_STEP,
                              72.0 + float(row) * self.GRID_ROW_STEP)
                n.px_position = True
        return self

    def describe(self, base_params: dict | None = None) -> dict:
        h = self.node_hashes(base_params=base_params)
        nodes_desc = []
        for n in self.nodes.values():
            spec_d = n.spec.to_dict()
            spec_d["label"] = n.display_label
            spec_d["inputs"] = [{"name": p.name, "type": p.type, "label": p.label}
                                for p in n.effective_inputs]
            spec_d["outputs"] = [{"name": p.name, "type": p.type, "label": p.label}
                                 for p in n.effective_outputs]
            nd = {
                **spec_d,
                "name": n.name,
                "enabled": n.enabled,
                "iterations": int(getattr(n, "iterations", 1)),
                "position": [float(n.position[0]), float(n.position[1])],
                "px": bool(getattr(n, "px_position", False)),
                "params": copy.deepcopy(n.params),
                "hash": h.get(n.name, "")[:8],
                "upstream": sorted(self.upstream(n.name)),
            }
            if n.subgraph is not None:
                nd["subgraph"] = copy.deepcopy(n.subgraph)
            nodes_desc.append(nd)
        return {
            "order": self.order(),
            "nodes": nodes_desc,
            "edges": [
                {**e.__dict__, "type": self._edge_type(e)} for e in self.edges
            ],
            "outputs": self.output_nodes(),
            "types": TYPES,
        }

    def _edge_type(self, e: Edge) -> str:
        src_node = self.nodes.get(e.src)
        if src_node is None:
            return "?"
        for p in src_node.effective_outputs:
            if p.name == e.src_port:
                return p.type
        return "?"


# ---------------------------------------------------------------------------
def vhs_consumer() -> Graph:
    """The classic consumer camcorder -> tape -> TV path, as a graph."""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", position=(1, 0))
    g.add("encoder", "encoder", position=(2, 0))
    g.add("tape", "tape", position=(3, 0))
    g.add("deck", "deck", position=(4, 0))
    g.add("display", "display", position=(5, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    g.connect("encoder", "out", "tape", "in")
    g.connect("tape", "out", "deck", "in")
    g.connect("deck", "out", "display", "in")
    return g


def vhs_node_group_demo() -> Graph:
    """
    A pre-packed Blender-style 'VHS' Node Group (`camera -> encoder -> tape -> deck`)
    wired to a Subtitle Generator (`overlay`) and CRT (`display`).
    """
    g = vhs_consumer()
    g.normalise_positions()
    vhs_grp = g.group_nodes(["camera", "encoder", "tape", "deck"], label="VHS", group_name="vhs")
    vhs_grp.position = (256.0, 72.0)
    ov = g.add("overlay", "overlay", params={"enabled": 1.0, "text": "VHS NODE GROUP"}, position=(480.0, 72.0))
    ov.px_position = True
    if "display" in g.nodes:
        g.nodes["display"].position = (704.0, 72.0)
        g.nodes["display"].px_position = True
    g.connect("source", "out", "vhs", "in")
    g.connect("vhs", "out", "overlay", "in")
    g.connect("overlay", "out", "display", "in")
    return g


def minidv_camcorder() -> Graph:
    """A MiniDV camcorder: the digital branch, no composite carrier anywhere."""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", position=(1, 0))
    g.add("encoder", "encoder", position=(2, 0))
    g.add("codec", "dv_codec", position=(3, 0))
    g.add("tape", "tape", position=(4, 0))
    g.add("deck", "deck", position=(5, 0))
    g.add("display", "display", position=(6, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    g.connect("encoder", "dv", "codec", "in")
    g.connect("codec", "out", "tape", "dv")
    g.connect("tape", "dv", "deck", "dv")
    g.connect("deck", "out", "display", "in")
    return g


def hdmi_av_rf_vhs_loop() -> Graph:
    """HDMI -> AV -> RF -> Tuner -> VHS Tape -> Deck -> Display."""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("hdmi_out", "hdmi_out", position=(1, 0))
    g.add("hdmi_to_av", "hdmi_to_av", position=(2, 0))
    g.add("rf_modulator", "rf_modulator", position=(3, 0))
    g.add("rf_tuner", "rf_tuner", position=(4, 0))
    g.add("tape", "tape", position=(5, 0))
    g.add("deck", "deck", position=(6, 0))
    g.add("display", "display", position=(7, 0))
    g.connect("source", "out", "hdmi_out", "in")
    g.connect("hdmi_out", "out", "hdmi_to_av", "in")
    g.connect("hdmi_to_av", "out", "rf_modulator", "in")
    g.connect("rf_modulator", "out", "rf_tuner", "in")
    g.connect("rf_tuner", "out", "tape", "in")
    g.connect("tape", "out", "deck", "in")
    g.connect("deck", "out", "display", "in")
    return g


def vhs_5gen_cascade() -> Graph:
    """5-generation physical VHS dubbing cascade (Gen 1 -> Gen 2 -> Gen 3 -> Gen 4 -> Gen 5)."""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", position=(1, 0))
    g.add("encoder", "encoder", position=(2, 0))
    g.add("gen1", "vhs_gen", position=(3, 0))
    g.add("gen2", "vhs_gen", position=(4, 0))
    g.add("gen3", "vhs_gen", position=(5, 0))
    g.add("gen4", "vhs_gen", position=(6, 0))
    g.add("gen5", "vhs_gen", position=(7, 0))
    g.add("deck", "deck", position=(8, 0))
    g.add("display", "display", position=(9, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    g.connect("encoder", "out", "gen1", "in")
    g.connect("gen1", "out", "gen2", "in")
    g.connect("gen2", "out", "gen3", "in")
    g.connect("gen3", "out", "gen4", "in")
    g.connect("gen4", "out", "gen5", "in")
    g.connect("gen5", "out", "deck", "in")
    g.connect("deck", "out", "display", "in")
    return g


def vhs_iterative_loop() -> Graph:
    """N-generation VHS iterative dubbing loop node."""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", position=(1, 0))
    g.add("encoder", "encoder", position=(2, 0))
    g.add("dub_loop", "vhs_dub_loop", params={"generation_count": 5.0}, position=(3, 0))
    g.add("deck", "deck", position=(4, 0))
    g.add("display", "display", position=(5, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    g.connect("encoder", "out", "dub_loop", "in")
    g.connect("dub_loop", "out", "deck", "in")
    g.connect("deck", "out", "display", "in")
    return g


default_graph = vhs_consumer
minidv_graph = minidv_camcorder
rf_loop_graph = hdmi_av_rf_vhs_loop

def crt_offscreen_reshoot_graph() -> Graph:
    """Signal chain feeding a physical CRT monitor, then re-photographed by an off-screen camera."""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", position=(1, 0))
    g.add("encoder", "encoder", position=(2, 0))
    g.add("tape", "tape", position=(3, 0))
    g.add("deck", "deck", position=(4, 0))
    g.add("display", "display", position=(5, 0))
    g.add("reshoot", "offscreen_camera", position=(6, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    g.connect("encoder", "out", "tape", "in")
    g.connect("tape", "out", "deck", "in")
    g.connect("deck", "out", "display", "in")
    g.connect("display", "out", "reshoot", "in")
    return g


GRAPH_PRESETS: dict[str, dict] = {
    "vhs_consumer": {
        "label": "家用 VHS 摄像链路",
        "note": "经典的 1990 年代家用模拟磁带链路：摄像机 -> 复合编码 -> 磁带 -> 录像机 -> 显示器",
        "build": vhs_consumer,
    },
    "vhs_node_group_demo": {
        "label": "📦 VHS 节点组 + 字幕机（Blender 风格打包示例）",
        "note": "将「成像系统→信号编码→记录介质→播放设备」打包为一个“VHS”节点组，并接入「字幕机」节点；双击或点选 VHS 节点组可展开修改内部参数或串联复制",
        "build": vhs_node_group_demo,
    },
    "minidv_camcorder": {
        "label": "MiniDV 数字分量链路",
        "note": "跳过复合视频编码，走 4:1:1 / 4:2:0 DCT 数字分量磁带通路",
        "build": minidv_camcorder,
        "params": {"codec": {"dv_enabled": 1.0}},
    },
    "hdmi_av_rf_vhs_loop": {
        "label": "HDMI -> AV -> RF -> VHS 闭路转录",
        "note": "现代信号经 HDMI2AV 与 RF 调制器进入 VHS 录像机高频头",
        "build": hdmi_av_rf_vhs_loop,
    },
    "vhs_5gen_cascade": {
        "label": "VHS 五代物理串接转录链（Gen 1 → Gen 5）",
        "note": "5 台 VHS 磁带转录节点物理串联，直观观察每一代带宽收窄、色度右移、时基抖动与切换带阶梯的指数级累积",
        "build": vhs_5gen_cascade,
    },
    "vhs_iterative_loop": {
        "label": "VHS 多代循环迭代转录机（N 代闭环）",
        "note": "使用「VHS N代循环转录」节点在单节点内执行 1~20 代真实的 VCR 播放→AV线→VCR 录制迭代回路",
        "build": vhs_iterative_loop,
        "params": {"tape": {"generation_count": 5.0}},
    },
}


GRAPH_PRESETS["j27_channel_switch_horror"] = {
    "label": "松下 NV-J27 高频头换台瞬态与走带切点故障链路",
    "note": "PAL-D VHF 6频道 (171.25MHz) 变容二极管换台瞬态（本振扫频奈奎斯特斜坡负片浮雕反相、场同步失锁 VBI 锤头黑条滚屏、行 AFC 顶部旗帜弯折、PAL 爬行百叶窗彩条）+ 松下 NV-J27 走带暂停切点与 1H 延迟线 DOC 失落补偿",
    "build": rf_loop_graph,
    "params": {
        "rf": {
            "enabled": 1.0, "tv_standard": 0.0, "channel_mhz": 8.0, "carrier_offset_mhz": 1.25,
            "signal_dbm": -62.0, "noise_figure_db": 8.5,
            "channel_switch_rate": 0.85, "channel_switch_dur_s": 0.48,
            "lo_sweep_mhz": 1.45, "vsync_roll": 0.82, "vbi_bar_vis": 0.88,
            "flagging": 0.72, "hanover_bars": 0.68,
            "multipath_level": 0.24, "multipath_delay_us": 1.65,
            "if_saw_ringing": 0.38, "sync_loss_threshold_db": 31.0, "sync_loss_amount": 1.15,
        },
        "tape": {
            "head_switching": 0.88, "head_switch_lines": 12.0, "tracking_error": 0.48,
            "doc_hold": 0.82, "edit_glitch_rate": 0.45, "edit_glitch_severity": 0.78,
        },
        "display": {"enabled": 1.0, "scanlines": 0.35, "spot_bloom": 0.48},
    },
}


def power_sync_causal_lab() -> Graph:
    """Camera -> Encoder -> Shared Power Bus -> Tape -> Deck -> Sync & Clock System -> Display."""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", position=(1, 0))
    g.add("encoder", "encoder", position=(2, 0))
    g.add("power_supply", "power_supply", position=(3, 0))
    g.add("tape", "tape", position=(4, 0))
    g.add("deck", "deck", position=(5, 0))
    g.add("sync_tbc", "sync_tbc", position=(6, 0))
    g.add("display", "display", position=(7, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    g.connect("encoder", "out", "power_supply", "in")
    g.connect("power_supply", "out", "tape", "in")
    g.connect("tape", "out", "deck", "in")
    g.connect("deck", "out", "sync_tbc", "in")
    g.connect("sync_tbc", "out", "display", "in")
    return g


GRAPH_PRESETS["power_sync_causal_lab"] = {
    "label": "⚡ 共享电源母线 + 同步时钟系统（跨设备负载耦合与脉冲十字实验台）",
    "note": "串联「共享电源与电气母线」与「同步与时钟系统」节点：观察老化电池内阻压降与 100Hz 纹波如何同时引起相机曝光下坠、走带伺服扭曲、同步分离门限漂移及伴音变调",
    "build": power_sync_causal_lab,
    "params": {
        "connector": {
            "psu_ripple": 0.42, "battery_level": 0.22, "battery_sag": 0.58,
            "load_coupling": 1.0, "startup_surge": 0.45, "emi_crosstalk": 0.32,
        },
        "player": {
            "sync_clip": 0.24, "h_hold": 0.14, "v_hold": 0.0,
            "clock_drift_ppm": 45.0, "clock_jitter": 0.12, "pulse_cross": 0.0,
        },
        "display": {"enabled": 1.0, "scanlines": 0.32, "breathing": 0.30},
    },
}

GRAPH_PRESETS["crt_offscreen_reshoot_lab"] = {
        "label": "虚拟相机重拍 CRT 屏摄链路（VHS → CRT 显像管 → 屏摄重拍相机）",
        "note": "把视频信号先送入虚拟球面 CRT 显像管发光，再用独立快门与拜耳传感器的虚拟相机重新拍摄：复现快门-场扫描差拍滚带、像素-荫罩二维摩尔纹、离轴透视与室内环境光反射",
        "build": crt_offscreen_reshoot_graph,
        "params": {
            "display": {
                "enabled": 1.0, "scanlines": 0.36, "mask_type": 1.0, "mask_pitch": 2.8,
                "mask_depth": 0.38, "phosphor_rgb": 0.68, "curvature": 0.24, "bezel": 0.35,
                "reshoot_enabled": 1.0, "reshoot_shutter_hz": 120.0, "reshoot_fps_offset": 1.10,
                "reshoot_rolling_shutter": 0.35, "reshoot_moire": 0.55, "reshoot_tilt_deg": 2.2,
                "reshoot_perspective": 0.20, "reshoot_ambient_glare": 0.28, "reshoot_wb_shift": 0.28,
            },
        },
   }

def vhs_transport_servo_lab() -> Graph:
    """Camera -> Encoder -> Tape -> Transport & Servo State Machine -> Deck -> Display."""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", position=(1, 0))
    g.add("encoder", "encoder", position=(2, 0))
    g.add("tape", "tape", position=(3, 0))
    g.add("transport", "transport_servo", position=(4, 0))
    g.add("deck", "deck", position=(5, 0))
    g.add("display", "display", position=(6, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    g.connect("encoder", "out", "tape", "in")
    g.connect("tape", "out", "transport", "in")
    g.connect("transport", "out", "deck", "in")
    g.connect("deck", "out", "display", "in")
    return g


GRAPH_PRESETS["vhs_transport_servo_lab"] = {
    "label": "⚙️ 机械传动与伺服状态机实验台（带盘张力演化 + 走带状态机 + 皮带打滑/压带轮偏心）",
    "note": "串联独立的「机械走带与伺服」节点：可切换 Play / Pause 静态定格噪波带 / 无飞抹磁头暂停起录切口，或调节带盘位置 (BOT→EOT)、方形皮带打滑与压带轮偏心抖晃",
    "build": vhs_transport_servo_lab,
    "params": {
        "tape": {
            "transport_state": 0.0, "reel_progress": 0.92, "tension_ripple": 0.78,
            "belt_slip": 0.65, "roller_eccentricity": 0.70, "servo_hunt": 0.52,
        },
        "display": {"enabled": 0.0},
    },
}


def vhs_analog_switcher_lab() -> Graph:
    """
    Dual-branch multi-input graph preset for Candidate C (Module 6):
      Branch A: source -> camera -> encoder -> tape -> switcher.in_a
      Branch B: source -> camera_b (telephoto warm Cam B) -> encoder_b -> switcher.in_b
      Output:   switcher.out -> deck -> sync_tbc -> display
    """
    g = Graph()
    g.add("source", "source", position=(0, 0.5))
    g.add("camera", "camera", position=(1, 0))
    g.add("encoder", "encoder", position=(2, 0))
    g.add("tape", "tape", position=(3, 0))
    g.add("camera_b", "camera", params={"wt_zoom": 0.55, "color_cast": -0.35, "exposure": 0.15}, position=(1, 1.2))
    g.add("encoder_b", "encoder", position=(2, 1.2))
    g.add("switcher", "video_switcher", position=(4, 0.5))
    g.add("deck", "deck", position=(5, 0.5))
    g.add("sync_tbc", "sync_tbc", position=(6, 0.5))
    g.add("display", "display", position=(7, 0.5))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    g.connect("encoder", "out", "tape", "in")
    g.connect("tape", "out", "switcher", "in_a")
    g.connect("source", "out", "camera_b", "in")
    g.connect("camera_b", "out", "encoder_b", "in")
    g.connect("encoder_b", "out", "switcher", "in_b")
    g.connect("switcher", "out", "deck", "in")
    g.connect("deck", "out", "sync_tbc", "in")
    g.connect("sync_tbc", "out", "display", "in")
    return g


GRAPH_PRESETS["vhs_analog_switcher_lab"] = {
    "label": "多路模拟视频混合器与切换台实验台（in_a + in_b 双路复合输入 + 划像/叠化/硬切/亮色键）",
    "note": "双分支多输入节点图：主路 (Camera A → 编码 → VHS 磁带) 接入切换台 in_a，副路 (Camera B 望远暖调机位 → 编码) 接入切换台 in_b，支持非同步硬切瞬态、T-Bar 叠化、斜坡划像与模拟亮/色键合成",
    "build": vhs_analog_switcher_lab,
    "params": {
        "connector": {
            "switcher_mode": 3.0, "switcher_tbar": 0.52, "switcher_wipe_pattern": 2.0,
            "switcher_genlock_err": 0.65, "switcher_key_jitter": 0.42,
            "switcher_subcarrier_spill": 0.62, "switcher_pedestal_add": 0.25,
        },
        "display": {"enabled": 0.0},
    },
}


def broadcast_router_da_bench() -> Graph:
    """
    Candidate E studio routing & 1-in-N-out distribution topology:
      source -> camera -> encoder -> dist_amp (1-in, 3-out DA / T-Split):
        * Branch 1 (Out 1): dist_amp.out   -> tape -> transport_servo -> switcher.in_a
        * Branch 2 (Out 2): dist_amp.out_2 -> patch_router.in_a -> switcher.in_b
      switcher.out -> deck -> sync_tbc -> display
    """
    g = Graph()
    g.add("source", "source", position=(0, 0.5))
    g.add("camera", "camera", position=(1, 0.5))
    g.add("encoder", "encoder", position=(2, 0.5))
    g.add("dist_amp", "dist_amp", position=(3, 0.5))
    g.add("tape", "tape", position=(4, 0.0))
    g.add("transport", "transport_servo", position=(5, 0.0))
    g.add("patch_router", "patch_router", position=(4.5, 1.15))
    g.add("switcher", "video_switcher", position=(6, 0.5))
    g.add("deck", "deck", position=(7, 0.5))
    g.add("sync_tbc", "sync_tbc", position=(8, 0.5))
    g.add("display", "display", position=(9, 0.5))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    g.connect("encoder", "out", "dist_amp", "in")
    g.connect("dist_amp", "out", "tape", "in")
    g.connect("tape", "out", "transport", "in")
    g.connect("transport", "out", "switcher", "in_a")
    g.connect("dist_amp", "out_2", "patch_router", "in_a")
    g.connect("patch_router", "out", "switcher", "in_b")
    g.connect("switcher", "out", "deck", "in")
    g.connect("deck", "out", "sync_tbc", "in")
    g.connect("sync_tbc", "out", "display", "in")
    return g


GRAPH_PRESETS["broadcast_router_da_bench"] = {
    "label": "广播级配线架与 75Ω 分配放大器工作台（1进多出扇出 + 磁带/直通双分支汇入切换台）",
    "note": "编码器输出经「75Ω 分配放大器 (dist_amp)」一分为二：一路送入 VHS 磁带与走带伺服 (Branch A)，另一路经「广播跳线配线架 (patch_router)」直通送入切换台副路 (Branch B)，可直观对比 75Ω 标准匹配、未端接高阻开路 (+6dB)、违规三通双端接 (37.5Ω) 与有源 DA 隔离驱动",
    "build": broadcast_router_da_bench,
    "params": {
        "connector": {
            "da_enabled": 1.0, "da_termination_mode": 3.0, "da_load_count": 2.0,
            "da_cable_len_m": 25.0, "da_gain_db": 0.0, "da_eq_boost": 0.35,
            "patch_crosstalk": 0.22, "patch_contact_aging": 0.06, "patch_format_mismatch": 0.0,
            "switcher_mode": 3.0, "switcher_tbar": 0.50, "switcher_wipe_pattern": 0.0,
        },
        "display": {"enabled": 0.0},
    },
}


def test_generator_tbc_calibration_lab() -> Graph:
    """
    Candidate G: Broadcast Test Signal Generator -> 75-Ohm DA -> VHS Tape -> Deck -> Sync/TBC -> Display.
    """
    g = Graph()
    g.add("test_gen", "test_generator", params={"test_gen_enabled": 1.0, "test_pattern_mode": 1.0, "test_amplitude_ire": 100.0}, position=(0, 0))
    g.add("dist_amp", "dist_amp", params={"da_enabled": 1.0, "da_termination_mode": 3.0, "da_eq_boost": 0.35}, position=(1, 0))
    g.add("tape", "tape", params={"tape_stability": 0.62, "line_jitter": 0.28, "head_switching": 0.72, "head_switch_lines": 10.0}, position=(2, 0))
    g.add("deck", "deck", params={"tbc_enabled": 1.0, "tbc_strength": 0.90}, position=(3, 0))
    g.add("sync_tbc", "sync_tbc", params={"sync_clip": 0.08, "clock_jitter": 0.06}, position=(4, 0))
    g.add("display", "display", params={"enabled": 0.0}, position=(5, 0))
    g.connect("test_gen", "out", "dist_amp", "in")
    g.connect("dist_amp", "out", "tape", "in")
    g.connect("tape", "out", "deck", "in")
    g.connect("deck", "out", "sync_tbc", "in")
    g.connect("sync_tbc", "out", "display", "in")
    return g


def broadcast_multiview_workstation() -> Graph:
    """
    Candidate F + G: Multi-branch studio signal chain feeding the 4-input `multiview_wall` CRT array:
      - CH1 (`in_a`): Camera / Source clean feed
      - CH2 (`in_b`): Test Signal Generator (SMPTE Bars + PLUGE) via Patch Router
      - CH3 (`in_c`): Raw VHS Tape + Transport Servo playback (uncorrected)
      - CH4 (`in_d`): TBC-stabilized Program output (`sync_tbc`)
    """
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", position=(1, 0))
    g.add("encoder", "encoder", position=(2, 0))
    g.add("tape", "tape", params={"tape_stability": 0.52, "line_jitter": 0.38, "head_switching": 0.85, "head_switch_lines": 14.0}, position=(3, 0))
    g.add("deck", "deck", params={"tbc_enabled": 1.0, "tbc_strength": 0.92}, position=(4, 0))
    g.add("sync_tbc", "sync_tbc", params={"sync_clip": 0.05}, position=(5, 0))

    g.add("test_gen", "test_generator", params={"test_gen_enabled": 1.0, "test_pattern_mode": 1.0}, position=(1, 1))
    g.add("patch_router", "patch_router", params={"da_enabled": 1.0, "patch_crosstalk": 0.18}, position=(3, 1))

    g.add("multiview_wall", "multiview_wall", params={"mv_enabled": 1.0, "mv_layout": 0.0, "mv_unit_variance": 0.65, "mv_tally_umd": 1.0}, position=(6, 0.5))

    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    g.connect("encoder", "out", "tape", "in")
    g.connect("tape", "out", "deck", "in")
    g.connect("deck", "out", "sync_tbc", "in")

    g.connect("test_gen", "out", "patch_router", "in_a")

    # Wire all 4 channels into the studio monitor wall
    g.connect("camera", "out", "multiview_wall", "in_a")
    g.connect("patch_router", "out", "multiview_wall", "in_b")
    g.connect("tape", "out", "multiview_wall", "in_c")
    g.connect("sync_tbc", "out", "multiview_wall", "in_d")
    return g


GRAPH_PRESETS["test_generator_tbc_calibration_lab"] = {
    "label": "📏 广播测试信号发生器与 TBC 闭环校准台（TSG-170A → DA → VHS → TBC → 示波器）",
    "note": "内置广播级测试信号发生器（SMPTE 彩条+PLUGE / EBU / FCC 多波群 / 5阶阶梯+2T脉冲 / PM5544）直入 75Ω 分配放大器、VHS 磁带与数字 TBC，配合实时示波器测量频响衰减、微分增益/微分相位与 TBC 抑抖比",
    "build": test_generator_tbc_calibration_lab,
    "params": {
        "format": {"test_gen_enabled": 1.0, "test_pattern_mode": 1.0, "test_amplitude_ire": 100.0},
        "connector": {"da_enabled": 1.0, "da_termination_mode": 3.0, "da_eq_boost": 0.35},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 0.90},
        "display": {"enabled": 0.0},
    },
}

GRAPH_PRESETS["broadcast_multiview_workstation"] = {
    "label": "演播室多画面监视墙与独立 CRT 阵列（摄像机 + 测试卡 + 磁带原片 + TBC主监 四路同屏）",
    "note": "4 路信号（CH1 摄像机源、CH2 配线架测试信号卡、CH3 VHS 磁带未校正重放、CH4 TBC 锁相主播出）汇入「多画面监视墙 (multiview_wall)」，每台监视器拥有独立 CRT 阴极老化色温、欠扫描边框、Tally 灯与 UMD 铭牌",
    "build": broadcast_multiview_workstation,
    "params": {
        "display": {
            "enabled": 1.0, "mv_enabled": 1.0, "mv_layout": 0.0,
            "mv_unit_variance": 0.65, "mv_fault_channel": 0.0,
            "mv_fault_mode": 0.0, "mv_tally_umd": 1.0,
        },
    },
}


def studio_ccu_genlock_switcher_bench() -> Graph:
    """
    Candidate H: Dual Studio Cameras (CAM-1 Reference + CAM-2 Warm ENG matched via CCU Rack & Genlock)
    feeding Production Video Switcher -> Master Playout -> Multiview Monitor Wall.
    """
    g = Graph()
    g.add("source", "source", position=(0, 0.5))
    g.add("cam_a", "camera", params={"sensor_noise": 0.03, "sharpness": 0.95}, position=(1, 0))
    g.add("ccu_a", "ccu_rack", params={"ccu_enabled": 1.0, "ccu_cam_id": 0.0, "genlock_mode": 0.0, "genlock_cable_len_m": 20.0}, position=(2, 0))
    g.add("enc_a", "encoder", position=(3, 0))

    g.add("cam_b", "camera", params={"sensor_noise": 0.04, "sharpness": 0.92}, position=(1, 1.2))
    g.add("ccu_b", "ccu_rack", params={
        "ccu_enabled": 1.0, "ccu_cam_id": 1.0, "ccu_iris_ev": 0.10,
        "ccu_r_gain": 0.89, "ccu_b_gain": 1.15, "ccu_r_flare": -0.020, "ccu_b_flare": 0.015,
        "genlock_mode": 0.0, "genlock_cable_len_m": 45.0, "genlock_sc_phase_deg": 0.0,
    }, position=(2, 1.2))
    g.add("enc_b", "encoder", position=(3, 1.2))

    g.add("video_switcher", "video_switcher", params={"switcher_mode": 3.0, "switcher_tbar": 0.50, "switcher_wipe_pattern": 0.0}, position=(4, 0.6))
    g.add("deck", "deck", params={"tbc_enabled": 1.0, "tbc_strength": 0.92}, position=(5, 0.6))
    g.add("display", "display", params={"enabled": 0.0}, position=(6, 0.6))

    g.connect("source", "out", "cam_a", "in")
    g.connect("cam_a", "out", "ccu_a", "in")
    g.connect("ccu_a", "out", "enc_a", "in")
    g.connect("enc_a", "out", "video_switcher", "in_a")

    g.connect("source", "out", "cam_b", "in")
    g.connect("cam_b", "out", "ccu_b", "in")
    g.connect("ccu_b", "out", "enc_b", "in")
    g.connect("enc_b", "out", "video_switcher", "in_b")

    g.connect("video_switcher", "out", "deck", "in")
    g.connect("deck", "out", "display", "in")
    return g


def master_control_playout_automation() -> Graph:
    """
    Candidate I (+ F/G/H): Complete Virtual TV Station Master Control Room (MCR) Playout Chain:
      - Main Program Path: Source -> Studio CCU -> Encoder -> VHS Tape -> Deck -> Master Playout (`in_a`)
      - Preview / Standby Path: Test Signal Generator -> Master Playout (`in_b`)
      - Master Playout fans out `PGM (with DSK)`, `Clean Feed (no DSK)`, and `PVW` simultaneously into `multiview_wall`!
    """
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("ccu_rack", "ccu_rack", params={"ccu_enabled": 1.0, "ccu_cam_id": 0.0, "genlock_mode": 0.0}, position=(1, 0))
    g.add("encoder", "encoder", position=(2, 0))
    g.add("tape", "tape", params={"tape_stability": 0.72, "line_jitter": 0.18, "head_switching": 0.65, "head_switch_lines": 10.0}, position=(3, 0))
    g.add("deck", "deck", params={"tbc_enabled": 1.0, "tbc_strength": 0.92}, position=(4, 0))

    g.add("test_gen", "test_generator", params={"test_gen_enabled": 1.0, "test_pattern_mode": 5.0}, position=(3, 1.2))
    g.add("master_playout", "master_playout", params={
        "playout_enabled": 1.0, "playout_output_bus": 0.0,
        "playout_state": 0.0, "playout_dsk_enabled": 1.0, "playout_dsk_crawl": 1.0,
    }, position=(5, 0.5))
    g.add("multiview_wall", "multiview_wall", params={
        "mv_enabled": 1.0, "mv_layout": 0.0, "mv_unit_variance": 0.55, "mv_tally_umd": 1.0,
    }, position=(6, 0.5))

    g.connect("source", "out", "ccu_rack", "in")
    g.connect("ccu_rack", "out", "encoder", "in")
    g.connect("encoder", "out", "tape", "in")
    g.connect("tape", "out", "deck", "in")
    g.connect("deck", "out", "master_playout", "in_a")
    g.connect("test_gen", "out", "master_playout", "in_b")

    # Wire Camera/CCU, Preview Bus, Clean Feed, and On-Air PGM into the 4-channel Monitor Wall
    g.connect("ccu_rack", "out", "multiview_wall", "in_a")
    g.connect("master_playout", "pvw", "multiview_wall", "in_b")
    g.connect("master_playout", "clean", "multiview_wall", "in_c")
    g.connect("master_playout", "out", "multiview_wall", "in_d")
    return g


GRAPH_PRESETS["studio_ccu_genlock_switcher_bench"] = {
    "label": "演播室多机位 CCU 集中调色与 Blackburst 主同步配平工作台（双机位 CCU → 切换台划像）",
    "note": "1号标准机 (Sony BVP-370) 与 2号暖调采访机 (Ikegami HL-55) 分别经独立 CCU 机架（遥调光圈、黑平衡 Flare、白平衡 Gain 与脐带线副载波相位补偿）配平后送入模拟切换台同屏划像对比",
    "build": studio_ccu_genlock_switcher_bench,
    "params": {
        "camera": {"ccu_enabled": 1.0, "genlock_mode": 0.0},
        "connector": {"switcher_mode": 3.0, "switcher_tbar": 0.50, "switcher_wipe_pattern": 0.0},
        "display": {"enabled": 0.0},
    },
}

GRAPH_PRESETS["master_control_playout_automation"] = {
    "label": "电视台总控主播出台全链路（CCU → 磁带 → 主控双总线 PVW/Clean/PGM+DSK → 四分屏监视墙）",
    "note": "主控播出台 (master_playout) 同时向演播室四分屏监视墙输出 CH1 摄像机 CCU、CH2 预监母线 (PVW)、CH3 无台标干净母线 (Clean Feed) 与 CH4 带 DSK 台标/游走字幕的主播出母线 (PGM)",
    "build": master_control_playout_automation,
    "params": {
        "overlay": {
            "playout_enabled": 1.0, "playout_output_bus": 0.0,
            "playout_state": 0.0, "playout_dsk_enabled": 1.0, "playout_dsk_crawl": 1.0,
        },
        "display": {"enabled": 1.0, "mv_enabled": 1.0, "mv_layout": 0.0, "mv_unit_variance": 0.55},
    },
}


def transfer_replication_comparison_lab() -> Graph:
    """
    Candidate J: 4-Way Replication Chain Comparison Lab (`source -> camera -> encoder -> tape -> deck -> transfer_station -> display`).
    """
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", position=(1, 0))
    g.add("encoder", "encoder", position=(2, 0))
    g.add("tape", "tape", params={"tape_stability": 0.82, "line_jitter": 0.12, "head_switching": 0.45, "head_switch_lines": 6.0}, position=(3, 0))
    g.add("deck", "deck", params={"tbc_enabled": 1.0, "tbc_strength": 0.90}, position=(4, 0))
    g.add("transfer_station", "transfer_station", params={
        "transfer_station_mode": 4.0, "transfer_generations": 3.0,
        "transfer_tbc_mode": 2.0, "transfer_codec_target": 3.0,
        "transfer_adc_crop": 0.45, "transfer_av_drift_ms": 35.0,
    }, position=(5, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    g.connect("encoder", "out", "tape", "in")
    g.connect("tape", "out", "deck", "in")
    g.connect("deck", "out", "transfer_station", "in")
    return g


def linear_and_nle_editing_suite() -> Graph:
    """
    Candidate K (+ J/F): Dual-Branch Editing & Post-Production Suite comparing:
      - CH1: Raw Camera / Studio Feed
      - CH2: Sony BVE-9100 Tape-to-Tape Linear Editor (`linear_editor`, Assemble/Insert Edit with Pre-roll & CTL slip)
      - CH3: Early 2000s Non-Linear Editor (`nle_timeline`, Multi-track V1/V2/Title timeline)
      - CH4: Transfer & Dubbing Comparison Workstation (`transfer_station`)
    all monitored side-by-side on the 2x2 `multiview_wall`!
    """
    g = Graph()
    g.add("source", "source", position=(0, 0.5))
    g.add("camera", "camera", position=(1, 0))
    g.add("test_gen", "test_generator", params={"test_gen_enabled": 1.0, "test_pattern_mode": 1.0}, position=(1, 1.2))
    g.add("linear_editor", "linear_editor", params={
        "edit_mode": 1.0, "edit_point_slip_frames": 2.0,
        "edit_flying_erase": 1.0, "edit_preroll_sec": 5.0, "edit_console_osd": 1.0,
    }, position=(3, 0))
    g.add("nle_timeline", "nle_timeline", params={
        "edit_mode": 3.0, "edit_nle_complexity": 0.70, "edit_console_osd": 1.0,
    }, position=(3, 1.2))
    g.add("transfer_station", "transfer_station", params={
        "transfer_station_mode": 2.0, "transfer_generations": 3.0, "transfer_tbc_mode": 1.0,
    }, position=(4, 1.2))
    g.add("multiview_wall", "multiview_wall", params={
        "mv_enabled": 1.0, "mv_layout": 0.0, "mv_unit_variance": 0.45, "mv_tally_umd": 1.0,
    }, position=(5, 0.5))

    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "linear_editor", "in_a")
    g.connect("test_gen", "out", "linear_editor", "in_b")
    g.connect("camera", "out", "nle_timeline", "in_a")
    g.connect("test_gen", "out", "nle_timeline", "in_b")
    g.connect("nle_timeline", "out", "transfer_station", "in")

    g.connect("camera", "out", "multiview_wall", "in_a")
    g.connect("linear_editor", "out", "multiview_wall", "in_b")
    g.connect("nle_timeline", "out", "multiview_wall", "in_c")
    g.connect("transfer_station", "out", "multiview_wall", "in_d")
    return g


GRAPH_PRESETS["transfer_replication_comparison_lab"] = {
    "label": "四路复制链对照实验室（母带直出 vs 无TBC多代复制 vs 带TBC多代复制 vs 数字跨格式转码）",
    "note": "同一段信号同时经过四条复制链（母带基准、无 TBC 模拟多代复制、带 TBC 模拟多代复制、4:2:2→DV25 4:1:1→MPEG-1 VCD 4:2:0 数字转码），同屏对比时基抖动、带宽滚降、色度损失与 TransferEvent 溯源链",
    "build": transfer_replication_comparison_lab,
    "params": {
        "player": {
            "transfer_station_mode": 4.0, "transfer_generations": 3.0,
            "transfer_tbc_mode": 2.0, "transfer_codec_target": 3.0,
        },
        "display": {"enabled": 0.0},
    },
}

GRAPH_PRESETS["linear_and_nle_editing_suite"] = {
    "label": "双机线性编辑台 vs 早期非线编系统与转录工作站同屏监看台（BVE-9100 + NLE + 转录站 → 监视墙）",
    "note": "摄像机主信号与测试图 B-Roll 同时送入「磁带线性编辑台 (linear_editor)」与「早期非线编时间线 (nle_timeline)」并经「转录工作站 (transfer_station)」汇总至演播室四分屏监视墙同屏对比",
    "build": linear_and_nle_editing_suite,
    "params": {
        "tape": {"edit_mode": 1.0, "edit_point_slip_frames": 2.0, "edit_console_osd": 1.0},
        "display": {"enabled": 1.0, "mv_enabled": 1.0, "mv_layout": 0.0},
    },
}


def satellite_to_automation_playout_suite() -> Graph:
    """
    模块 + M 联合节点图预设：
    卫星天线/LNB (`sat_dish_lnb`) → 卫星 IRD 接收机 (`sat_ird_receiver`) 作为主直播源 `in_a`，
    测试信号/本地备带 (`test_generator` → `tape`) 作为备机预卷源 `in_b`，
    双路汇入播出自动化编排引擎 (`automation_controller`)，并送入演播室四分屏监视墙 (`multiview_wall`)。
    """
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("sat_dish_lnb", "sat_dish_lnb", params={
        "sat_cn_db": 9.8, "sat_rain_fade_db": 4.2, "sat_dish_mispoint_deg": 0.25, "sat_lnb_drift_khz": 45.0,
    }, position=(1, -0.6))
    g.add("sat_ird_receiver", "sat_ird_receiver", params={
        "sat_ird_mode": 2.0, "sat_cn_db": 9.8, "sat_rain_fade_db": 4.2, "sat_ird_osd": 1.0,
    }, position=(2, -0.6))
    g.add("test_gen", "test_generator", params={
        "test_gen_enabled": 1.0, "test_pattern_mode": 1.0,
    }, position=(1, 1.0))
    g.add("backup_vtr", "tape", params={
        "tape_stability": 0.85, "line_jitter": 0.08, "head_switching": 0.55,
    }, position=(2, 1.0))
    g.add("automation_controller", "automation_controller", params={
        "auto_v2_enabled": 1.0, "auto_v2_mode": 3.0, "auto_v2_fault_inject": 3.0,
        "auto_v2_watchdog_sec": 1.0, "auto_v2_console_view": 0.0,
    }, position=(3, 0.2))
    g.add("multiview_wall", "multiview_wall", params={
        "mv_enabled": 1.0, "mv_layout": 0.0, "mv_tally_umd": 1.0,
    }, position=(4, 0.2))

    g.connect("source", "out", "sat_dish_lnb", "in")
    g.connect("sat_dish_lnb", "out", "sat_ird_receiver", "in")
    g.connect("test_gen", "out", "backup_vtr", "in")
    g.connect("sat_ird_receiver", "out", "automation_controller", "in_a")
    g.connect("backup_vtr", "out", "automation_controller", "in_b")

    g.connect("sat_ird_receiver", "out", "multiview_wall", "in_a")
    g.connect("backup_vtr", "out", "multiview_wall", "in_b")
    g.connect("automation_controller", "clean", "multiview_wall", "in_c")
    g.connect("automation_controller", "out", "multiview_wall", "in_d")
    return g


def end_to_end_tv_transmitter_broadcast_bench() -> Graph:
    """
    模块 + M + N 端到端全链路开路广播测试台：
    编辑台 (`linear_editor`) → 播出自动化 (`automation_controller`) → 电视台大功率发射机激励与高功放链 (`tv_exciter_pa`)
    → 电视调谐器 (`rf_tuner`) → CRT 显像管 (`display`)。
    """
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", position=(1, 0))
    g.add("linear_editor", "linear_editor", params={
        "edit_mode": 2.0, "edit_flying_erase": 1.0,
    }, position=(2, 0))
    g.add("automation_controller", "automation_controller", params={
        "auto_v2_enabled": 1.0, "auto_v2_mode": 1.0, "auto_v2_fault_inject": 0.0,
    }, position=(3, 0))
    g.add("tv_exciter_pa", "tv_exciter_pa", params={
        "tx_chain_mode": 3.0, "tx_pa_drive": 0.65, "tx_pa_am_pm_deg": 14.0,
        "tx_filter_gd_ns": 95.0, "tx_feeder_vswr": 1.35, "tx_sound_intermod": 0.38,
    }, position=(4, 0))
    g.add("display", "display", params={"enabled": 0.0}, position=(5, 0))

    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "linear_editor", "in_a")
    g.connect("linear_editor", "out", "automation_controller", "in_a")
    g.connect("automation_controller", "out", "tv_exciter_pa", "in")
    g.connect("tv_exciter_pa", "out", "display", "in")
    return g


GRAPH_PRESETS["satellite_to_automation_playout_suite"] = {
    "label": "卫星下行接收 (Satellite IRD) → 总控播出自动化主备冗余倒换与四分屏监看台",
    "note": "卫星天线与 LNB → 卫星 IRD 接收机作为主信号源，本地录像带作为备机预卷源，汇入事件/条件驱动播出自动化编排引擎 (`automation_controller`)，并在四分屏监视墙同屏监看卫星信号、备机信号、Clean 净画面与 PGM 主输出",
    "build": satellite_to_automation_playout_suite,
    "params": {
        "rf": {"sat_ird_mode": 2.0, "sat_cn_db": 9.8, "sat_rain_fade_db": 4.2, "sat_ird_osd": 1.0},
        "overlay": {"auto_v2_enabled": 1.0, "auto_v2_mode": 3.0, "auto_v2_fault_inject": 3.0},
        "display": {"enabled": 1.0, "mv_enabled": 1.0, "mv_layout": 0.0},
    },
}

GRAPH_PRESETS["end_to_end_tv_transmitter_broadcast_bench"] = {
    "label": "端到端电视台开路广播测试台（编辑台 → 播出自动化 → 大功率发射机激励/HPA → 开路信道 → 接收机对照）",
    "note": "从节目编辑、总控自动化播出到电视台大功率发射机（VSB 残留边带激励 + 速调管/固态高功放 AM-AM/AM-PM + 腔体双工器 + 铁塔馈线驻波）与开路模拟/数字接收终端同屏四象限对照",
    "build": end_to_end_tv_transmitter_broadcast_bench,
    "params": {
        "rf": {
            "tx_chain_mode": 3.0, "tx_pa_drive": 0.65, "tx_pa_am_pm_deg": 14.0,
            "tx_filter_gd_ns": 95.0, "tx_feeder_vswr": 1.35, "tx_sound_intermod": 0.38,
        },
        "display": {"enabled": 0.0},
    },
}


def film_telecine_to_vhs_and_inversion_lab() -> Graph:
    """
    Option P + Option O combined node graph preset:
    16mm/35mm Film & Rank Cintel Telecine (`film_scanner_telecine`) -> Encoder -> VHS Tape -> Deck ->
    Parameter Inversion & FADGI Error Budget Analyzer (`inversion_analyzer`) -> Display.
    """
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("film_scanner_telecine", "film_scanner_telecine", params={
        "film_enabled": 1.0, "film_stock": 2.0, "film_halation": 0.40,
        "film_grain_amount": 0.32, "film_gate_weave": 0.28, "telecine_pulldown": 1.0,
    }, position=(1, 0))
    g.add("encoder", "encoder", params={"luma_bandwidth": 3.1, "chroma_bandwidth": 0.72}, position=(2, 0))
    g.add("tape", "tape", params={"line_jitter": 0.24, "dropout_rate": 0.18, "noise_floor": 0.12}, position=(3, 0))
    g.add("deck", "deck", params={"tbc_enabled": 0.0, "head_wear": 0.22}, position=(4, 0))
    g.add("inversion_analyzer", "inversion_analyzer", params={
        "inv_lab_enabled": 1.0, "inv_lab_mode": 3.0, "inv_fadgi_target": 3.0,
    }, position=(5, 0))
    g.add("display", "display", params={"enabled": 0.0}, position=(6, 0))

    g.connect("source", "out", "film_scanner_telecine", "in")
    g.connect("film_scanner_telecine", "out", "encoder", "in")
    g.connect("encoder", "out", "tape", "in")
    g.connect("tape", "out", "deck", "in")
    g.connect("deck", "out", "inversion_analyzer", "in")
    g.connect("inversion_analyzer", "out", "display", "in")
    return g


GRAPH_PRESETS["film_telecine_to_vhs_and_inversion_lab"] = {
    "label": "16mm/35mm 电影胶片 → 飞点胶转磁 (3:2 Pulldown) → 家用 VHS 录放 → 参数反演与 FADGI 误差预算台",
    "note": "从 16mm/35mm 卤化银感光、片门二维抖动与 Rank Cintel 3:2 Pulldown 胶转磁播出，进入 VHS 磁带录放，最终送入多解置信区间参数反演与 FADGI 三级参数审计台",
    "build": film_telecine_to_vhs_and_inversion_lab,
    "params": {
        "camera": {
            "film_enabled": 1.0, "film_stock": 2.0, "film_halation": 0.40,
            "film_grain_amount": 0.32, "film_gate_weave": 0.28, "telecine_pulldown": 1.0,
        },
        "format": {"luma_bandwidth": 3.1, "chroma_bandwidth": 0.72},
        "tape": {"line_jitter": 0.24, "dropout_rate": 0.18, "noise_floor": 0.12},
        "player": {"tbc_enabled": 0.0, "head_wear": 0.22, "inv_lab_enabled": 1.0, "inv_lab_mode": 3.0},
        "display": {"enabled": 0.0},
    },
}


def archival_and_spatial_environment_suite() -> Graph:
    """
    Option Q + Option R Joint Ecosystem Graph:
    Source -> Camera -> Encoder -> Spatial Environment Chamber (`env_chamber`) ->
    VHS Tape -> Deck -> Archival Vault & Lineage Tree (`archival_vault`) -> Display.
    """
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", position=(1, 0))
    g.add("encoder", "encoder", position=(2, 0))
    g.add("env_chamber", "env_chamber", params={
        "env_field_enabled": 1.0, "env_field_mode": 4.0,
        "env_room_temp_c": 25.0, "env_room_rh_pct": 75.0,
        "env_device_init_temp_c": 5.0, "env_emi_source_power": 0.65,
    }, position=(3, 0))
    g.add("tape", "tape", position=(4, 0))
    g.add("deck", "deck", position=(5, 0))
    g.add("archival_vault", "archival_vault", params={
        "archival_enabled": 1.0, "archival_mode": 4.0,
        "archival_storage_years": 30.0, "archival_temp_c": 24.0,
        "archival_rh_pct": 65.0, "archival_dnr_overrestore": 0.78,
    }, position=(6, 0))
    g.add("display", "display", position=(7, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    g.connect("encoder", "out", "env_chamber", "in")
    g.connect("env_chamber", "out", "tape", "in")
    g.connect("tape", "out", "deck", "in")
    g.connect("deck", "out", "archival_vault", "in")
    g.connect("archival_vault", "out", "display", "in")
    return g


def complete_media_ecology_end_to_end_suite() -> Graph:
    """Option S + T + U Complete Audiovisual Media Ecology End-to-End Suite."""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", position=(1, 0))
    g.add("encoder", "encoder", position=(2, 0))
    g.add("tape", "tape", position=(3, 0))
    g.add("deck", "deck", position=(4, 0))
    g.add("stream_cdn_client", "stream_cdn_client", params={
        "stream_enabled": 1.0, "stream_mode": 3.0,
        "stream_bitrate_kbps": 420.0, "stream_bw_fluctuation": 0.45,
        "stream_packet_loss": 0.06, "stream_gop_len": 15.0,
    }, position=(5, 0))
    g.add("display", "display", position=(6, 0))
    g.add("human_operator", "human_operator", params={
        "percept_enabled": 1.0, "percept_mode": 2.0,
        "percept_viewing_dist_h": 3.5, "operator_skill": 0.82, "operator_stress": 0.30,
    }, position=(7, 0))
    g.add("ecosystem_simulator", "ecosystem_simulator", params={
        "eco_enabled": 1.0, "eco_mode": 4.0,
        "eco_year": 2018.0, "eco_scenario": 1.0,
    }, position=(8, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    g.connect("encoder", "out", "tape", "in")
    g.connect("tape", "out", "deck", "in")
    g.connect("deck", "out", "stream_cdn_client", "in")
    g.connect("stream_cdn_client", "out", "display", "in")
    g.connect("display", "out", "human_operator", "in")
    g.connect("human_operator", "out", "ecosystem_simulator", "in")
    return g


GRAPH_PRESETS["complete_media_ecology_end_to_end_suite"] = {
    "label": "完整视听媒介生态全链路总成：磁带录放 → 网络流媒体传播 → 人类感知与操作闭环 → 媒介生态演化模拟",
    "note": "打通从模拟摄像/磁带介质、互联网流媒体平台转码与 ABR 传输、观众心理物理 CSF 感知与机房值班员闭环，直至 1975–2035 宏观媒介生态演化与反事实情景推演",
    "build": complete_media_ecology_end_to_end_suite,
    "params": {
        "codec": {
            "stream_enabled": 1.0, "stream_mode": 3.0,
            "stream_bitrate_kbps": 420.0, "stream_bw_fluctuation": 0.45,
        },
        "display": {
            "enabled": 1.0, "percept_enabled": 1.0, "percept_mode": 2.0,
            "percept_viewing_dist_h": 3.5, "operator_skill": 0.82,
        },
        "overlay": {
            "eco_enabled": 1.0, "eco_mode": 4.0,
            "eco_year": 2018.0, "eco_scenario": 1.0,
        },
    },
}

GRAPH_PRESETS["archival_and_spatial_environment_suite"] = {
    "label": "空间物理环境场耦合舱 → 磁带录放 → 档案保存生命周期与多版本谱系树实验室",
    "note": "摄像机/录像机在二维空间物理环境场 (env_chamber) 中经历局域结露与近场电磁耦合，录制磁带后送入档案保存库 (archival_vault) 进行四阶段生命周期与多版本谱系树对照",
    "build": archival_and_spatial_environment_suite,
    "params": {
        "connector": {
            "env_field_enabled": 1.0, "env_field_mode": 4.0,
            "env_room_temp_c": 25.0, "env_room_rh_pct": 75.0,
            "env_device_init_temp_c": 5.0, "env_emi_source_power": 0.65,
        },
        "tape": {"line_jitter": 0.22, "head_switching": 0.78, "head_switch_lines": 10.0, "dropout_rate": 0.25},
        "player": {
            "tbc_enabled": 1.0, "archival_enabled": 1.0, "archival_mode": 4.0,
            "archival_storage_years": 30.0, "archival_dnr_overrestore": 0.78,
        },
        "display": {"enabled": 0.0},
    },
}


def lomo_photochemical_darkroom_lab() -> Graph:
    """
    Standalone Photochemical Film & LOMO Darkroom Lab node-graph preset:
    Source -> Camera (LOMO lens) -> Photochemical Film (`photochemical_film`) -> Display.
    """
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", params={
        "sensor_noise": 0.0, "sharpness": 0.96, "halation": 0.0, "ccd_smear": 0.0,
    }, position=(1, 0))
    g.add("photochemical_film", "photochemical_film", params={
        "enabled": 1.0, "stock": 7.0, "process_mode": 2.0,
        "xpro_strength": 0.85, "xpro_tint_mode": 2.0,
        "halation": 0.42, "grain_amount": 0.26, "lomo_vignette": 0.42,
        "bokeh_shape": 1.0, "bokeh_amount": 0.65,
        "light_leak_amount": 0.48, "light_leak_pattern": 3.0,
    }, position=(2, 0))
    g.add("display", "display", params={"enabled": 0.0}, position=(3, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "photochemical_film", "in")
    g.connect("photochemical_film", "out", "display", "in")
    return g


def crt_strong_magnet_lab_graph() -> Graph:
    """Ref: CRT Strong Magnet, Shadow-Mask Remanence & PTC Degaussing Coil Lab."""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", params={"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0}, position=(1, 0))
    g.add("encoder", "encoder", params={"luma_bandwidth": 4.8, "chroma_bandwidth": 1.4}, position=(2, 0))
    g.add("deck", "deck", params={"tbc_enabled": 1.0, "transport_jitter": 0.0, "head_wear": 0.0}, position=(3, 0))
    g.add("display", "display", params={
        "enabled": 1.0, "gamma": 2.25, "scanlines": 0.28, "mask_type": 1.0, "mask_pitch": 2.6,
    }, position=(4, 0))
    g.add("crt_magnet_env", "crt_magnet_env", params={
        "crt_magnet_mode": 5.0, "crt_magnet_strength": 0.85,
        "crt_magnet_x": 0.52, "crt_magnet_y": 0.48, "crt_magnet_z_dist": 0.12,
        "crt_magnet_angle_deg": 25.0, "crt_magnet_polarity": 1.0,
        "crt_magnet_remanence": 0.65, "crt_degauss_trigger": 0.75,
    }, position=(5, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    g.connect("encoder", "out", "deck", "in")
    g.connect("deck", "out", "display", "in")
    g.connect("display", "out", "crt_magnet_env", "in")
    return g


def sony_trinitron_kv1214_graph() -> Graph:
    """Ref vs Ref: Sony Trinitron KV-1214 (1978 Showa 12-inch Cylindrical Trinitron TV) 4K Moire & Grille calibration."""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", params={"sensor_noise": 0.0, "sharpness": 1.02, "halation": 0.08}, position=(1, 0))
    g.add("encoder", "encoder", params={
        "luma_bandwidth": 4.2, "chroma_bandwidth": 1.15, "edge_boost": 0.28, "chroma_gain": 1.06,
    }, position=(2, 0))
    g.add("deck", "deck", params={
        "tbc_enabled": 1.0, "transport_jitter": 0.03, "head_wear": 0.04, "preemphasis": 0.22,
    }, position=(3, 0))
    g.add("display", "display", params={
        "enabled": 1.0, "gamma": 2.28, "contrast": 0.62, "brightness": 0.47, "saturation": 0.56,
        "scanlines": 0.34, "spot_sigma": 0.38, "scan_pitch": 2.0, "spot_bloom": 0.44,
        "mask_type": 0.0, "mask_pitch": 2.35, "mask_depth": 0.42, "phosphor_rgb": 0.78,
        "kv1214_enabled": 1.0, "kv1214_grille_pitch_px": 2.35, "kv1214_grille_contrast": 0.68,
        "kv1214_moire_amount": 0.62, "kv1214_moire_freq": 0.052, "kv1214_moire_tilt_deg": 1.8,
        "kv1214_cylindrical_bow": 0.34, "kv1214_ntscj_9300k": 0.78,
        "kv1214_single_damper": 0.65, "kv1214_bezel_frame": 0.45,
    }, position=(4, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "encoder", "in")
    g.connect("encoder", "out", "deck", "in")
    g.connect("deck", "out", "display", "in")
    return g


def lomo_optical_halation_bokeh_lab_graph() -> Graph:
    """Standalone Film Optics & Photochemical pipeline graph (single-screen full-frame)."""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", params={
        "sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
    }, position=(1, 0))
    g.add("film_optics", "film_optics", params={
        "bloom_veil_mode": 4.0, "bloom_veil_amount": 0.85,
        "soft_focus_mist": 0.55, "dual_tone_split": 0.78,
        "bokeh_shape": 5.0, "bokeh_amount": 0.85,
        "halation": 0.45, "remjet_backing": 0.0, "lomo_vignette": 0.22,
    }, position=(2, 0))
    g.add("photochemical_film", "photochemical_film", params={
        "enabled": 1.0, "stock": 16.0, "process_mode": 0.0,
        "grain_amount": 0.16,
    }, position=(3, 0))
    g.add("display", "display", params={"enabled": 0.0}, position=(4, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "film_optics", "in")
    g.connect("film_optics", "out", "photochemical_film", "in")
    g.connect("photochemical_film", "out", "display", "in")
    return g


def benchmark_exp1_film_stocks_graph() -> Graph:
    """基准实验一：同一场景、不同胶片与冲洗动力学（独立胶片光学 -> 光化学显影）。"""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", params={"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0}, position=(1, 0))
    g.add("film_optics", "film_optics", params={"halation": 0.35, "remjet_backing": 0.88}, position=(2, 0))
    g.add("photochemical_film", "photochemical_film", params={
        "enabled": 1.0, "stock": 17.0, "process_mode": 0.0,
        "dev_temp_c": 41.1, "dev_time_norm": 1.0, "bath_activity": 1.0,
        "orange_mask": 1.0, "grain_amount": 0.22,
    }, position=(3, 0))
    g.add("display", "display", params={"enabled": 0.0}, position=(4, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "film_optics", "in")
    g.connect("film_optics", "out", "photochemical_film", "in")
    g.connect("photochemical_film", "out", "display", "in")
    return g


def benchmark_exp2_optical_printing_graph() -> Graph:
    """基准实验二：同一原片、不同复制链（原始负片 -> 光学印片中间片 -> 发行拷贝）。"""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", params={"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0}, position=(1, 0))
    g.add("film_optics", "film_optics", params={"halation": 0.38, "remjet_backing": 0.85}, position=(2, 0))
    g.add("photochemical_film", "photochemical_film", params={
        "enabled": 1.0, "stock": 17.0, "process_mode": 0.0,
        "optical_print_gen": 2.0, "gate_weave": 0.18, "grain_amount": 0.28,
    }, position=(3, 0))
    g.add("display", "display", params={"enabled": 0.0}, position=(4, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "film_optics", "in")
    g.connect("film_optics", "out", "photochemical_film", "in")
    g.connect("photochemical_film", "out", "display", "in")
    return g


def benchmark_exp3_aging_restoration_graph() -> Graph:
    """基准实验三：同一胶片、不同保存历史与湿门扫描数字修复。"""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("camera", "camera", params={"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0}, position=(1, 0))
    g.add("film_optics", "film_optics", params={"halation": 0.36, "aged_backing_dye": 0.45}, position=(2, 0))
    g.add("photochemical_film", "photochemical_film", params={
        "enabled": 1.0, "stock": 4.0, "process_mode": 0.0,
        "storage_temp_c": 32.0, "storage_rh": 78.0, "vinegar_syndrome": 0.82,
        "dye_fade": 0.68, "shrinkage": 0.62, "emulsion_damage": 0.55,
        "scratches_dust": 0.48, "wet_gate": 0.95, "scanner_diffuse": 0.92, "restoration_strength": 0.88,
    }, position=(3, 0))
    g.add("display", "display", params={"enabled": 0.0}, position=(4, 0))
    g.connect("source", "out", "camera", "in")
    g.connect("camera", "out", "film_optics", "in")
    g.connect("film_optics", "out", "photochemical_film", "in")
    g.connect("photochemical_film", "out", "display", "in")
    return g


GRAPH_PRESETS["benchmark_exp1_film_stocks"] = {
    "label": "基准实验一 同场景不同胶片与冲洗",
    "note": "基准实验一（感光与显影动力学）：独立胶片光学台 (`film_optics`) 串联胶片光化学冲印台 (`photochemical_film`)，比较 Kodak Vision3 500T、Double-X 5222、Kodachrome 64、Ektachrome 100D 及 ECN-2 推拉冲",
    "build": benchmark_exp1_film_stocks_graph,
    "params": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        "film": {
            "enabled": 1.0, "stock": 17.0, "process_mode": 0.0,
            "dev_temp_c": 41.1, "dev_time_norm": 1.0, "bath_activity": 1.0,
            "orange_mask": 1.0, "remjet_backing": 0.88, "halation": 0.35, "grain_amount": 0.22,
        },
        "display": {"enabled": 0.0},
    },
}

GRAPH_PRESETS["benchmark_exp2_optical_printing"] = {
    "label": "基准实验二 同原片多代光学印片链",
    "note": "基准实验二（多代复制链）：对比原始负片直扫 (OCN)、一代光学印片、二代中间片 (IP->IN)、三代院线发行拷贝 (2383) 与飞点胶转磁",
    "build": benchmark_exp2_optical_printing_graph,
    "params": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        "film": {
            "enabled": 1.0, "stock": 17.0, "process_mode": 0.0,
            "optical_print_gen": 2.0, "gate_weave": 0.18, "grain_amount": 0.28,
        },
        "display": {"enabled": 0.0},
    },
}

GRAPH_PRESETS["benchmark_exp3_aging_restoration"] = {
    "label": "基准实验三 醋酸老化与湿门数字修复",
    "note": "基准实验三（保存历史与可逆修复）：模拟醋酸综合征自催化降解、背层抗晕光染料重显、湿门扫描划痕抑制与 Disney Research / FIAF 四层证据分离可逆修复",
    "build": benchmark_exp3_aging_restoration_graph,
    "params": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        "film": {
            "enabled": 1.0, "stock": 4.0, "process_mode": 0.0,
            "storage_temp_c": 32.0, "storage_rh": 78.0, "vinegar_syndrome": 0.82,
            "dye_fade": 0.68, "shrinkage": 0.62, "emulsion_damage": 0.55,
            "scratches_dust": 0.48, "wet_gate": 0.95, "scanner_diffuse": 0.92, "restoration_strength": 0.88,
        },
        "display": {"enabled": 0.0},
    },
}

GRAPH_PRESETS["lomo_optical_halation_bokeh_lab"] = {
    "label": "胶片光学散景与光晕",
    "note": "独立胶片光学模块 (`film_optics`) 全屏单画面预设：Trioplan 肥皂泡亮环散景、紫金分色光雾与片基抗晕光散射",
    "build": lomo_optical_halation_bokeh_lab_graph,
    "params": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        "film": {
            "enabled": 1.0, "stock": 16.0, "process_mode": 0.0,
            "bloom_veil_mode": 4.0, "bloom_veil_amount": 0.85,
            "soft_focus_mist": 0.55, "dual_tone_split": 0.78,
            "bokeh_shape": 5.0, "bokeh_amount": 0.85,
            "halation": 0.45, "grain_amount": 0.16, "lomo_vignette": 0.22,
            "scratches_dust": 0.0, "gate_weave": 0.0, "shutter_flicker": 0.0,
        },
        "display": {"enabled": 0.0},
    },
}

GRAPH_PRESETS["lomo_photochemical_darkroom_lab"] = {
    "label": "胶片暗房交叉冲洗",
    "note": "独立胶片光化学冲印台 (`photochemical_film`) 全屏单画面预设：C-41/E-6 交叉冲洗、桃心异形散景与机壳漏光",
    "build": lomo_photochemical_darkroom_lab,
    "params": {
        "camera": {"sensor_noise": 0.0, "sharpness": 0.96, "halation": 0.0, "ccd_smear": 0.0},
        "film": {
            "enabled": 1.0, "stock": 7.0, "process_mode": 2.0,
            "xpro_strength": 0.85, "xpro_tint_mode": 2.0,
            "halation": 0.42, "grain_amount": 0.26, "lomo_vignette": 0.42,
            "bokeh_shape": 1.0, "bokeh_amount": 0.65,
            "light_leak_amount": 0.48, "light_leak_pattern": 3.0,
            "scratches_dust": 0.0, "gate_weave": 0.0, "shutter_flicker": 0.0,
        },
        "display": {"enabled": 0.0},
    },
}

GRAPH_PRESETS["crt_strong_magnet_physics_lab"] = {
    "label": "CRT 强磁偏转与消磁线圈",
    "note": "还原强钕铁硼磁铁贴近 CRT 时的洛伦兹力电子束排空黑洞、十字枕形内凹折叠、多阶同心色纯度彩虹环、撤磁后铁磁荫罩永久剩磁色斑及 PTC 衰减交变消磁线圈闭环",
    "build": crt_strong_magnet_lab_graph,
    "params": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0},
        "format": {"luma_bandwidth": 4.8, "chroma_bandwidth": 1.4, "cross_color": 0.08, "cross_luma": 0.05, "color_noise": 0.02},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "head_switching": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {
            "enabled": 1.0, "gamma": 2.25, "scanlines": 0.28, "spot_sigma": 0.38, "scan_pitch": 2.0,
            "mask_type": 1.0, "mask_pitch": 2.6, "mask_depth": 0.35, "phosphor_rgb": 0.65,
            "crt_magnet_mode": 5.0, "crt_magnet_strength": 0.85,
            "crt_magnet_x": 0.52, "crt_magnet_y": 0.48, "crt_magnet_z_dist": 0.12,
            "crt_magnet_angle_deg": 25.0, "crt_magnet_polarity": 1.0,
            "crt_magnet_remanence": 0.65, "crt_degauss_trigger": 0.75,
        },
    },
}

GRAPH_PRESETS["sony_trinitron_kv1214_showa"] = {
    "label": "Sony Trinitron KV-1214 特丽珑",
    "note": "12寸柱面特丽珑 4K 屏摄实测标定：单枪三束粗节距垂直荫栅 (HFx/HFy≈1.48)、单根钨丝阻尼线、单向水平柱面玻壳、NTSC-J 9300K 青蓝白点与弧形屏摄摩尔纹",
    "build": sony_trinitron_kv1214_graph,
    "params": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.02, "halation": 0.08},
        "format": {
            "filter_kind": 1.0, "filter_sections": 2.0,
            "luma_bandwidth": 4.2, "chroma_bandwidth": 1.15,
            "chroma_vertical_averaging": 0.35, "separation_kind": 2.0, "comb_depth": 0.68,
            "cross_color": 0.22, "cross_luma": 0.16, "edge_boost": 0.28,
            "color_noise": 0.06, "chroma_gain": 1.06,
        },
        "tape": {"tape_stability": 0.96, "line_jitter": 0.03, "walk_amplitude": 0.02, "micro_jitter": 0.02, "head_switching": 0.0, "dropout_rate": 0.0, "noise_floor": 0.01},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.03, "head_wear": 0.04, "preemphasis": 0.22},
        "display": {
            "enabled": 1.0, "gamma": 2.28, "contrast": 0.62, "brightness": 0.47, "saturation": 0.56,
            "scanlines": 0.34, "spot_sigma": 0.38, "scan_pitch": 2.0, "spot_bloom": 0.44,
            "mask_type": 0.0, "mask_pitch": 2.35, "mask_depth": 0.42, "phosphor_rgb": 0.78,
            "kv1214_enabled": 1.0, "kv1214_grille_pitch_px": 2.35, "kv1214_grille_contrast": 0.68,
            "kv1214_moire_amount": 0.62, "kv1214_moire_freq": 0.052, "kv1214_moire_tilt_deg": 1.8,
            "kv1214_cylindrical_bow": 0.34, "kv1214_ntscj_9300k": 0.78,
            "kv1214_single_damper": 0.65, "kv1214_bezel_frame": 0.45,
        },
    },
}


def ldme_full_archaeology_chain_graph() -> Graph:
    """第九节完整数字媒介考古链路：场景 → Nokia N73 拍摄 → USB 导入与桌面剪辑导出 → 优酷式 FLV 转码 → Flash 播放器缩放与 2007 旧 LCD 显示。"""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("mobile_camera_isp", "mobile_camera_isp", params={
        "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 2.0, "ldme_capture_mode": 0.0,
    }, position=(1, 0))
    g.add("file_transfer_node", "file_transfer_node", params={
        "ldme_transfer_enabled": 1.0, "ldme_transfer_method": 4.0, "ldme_transfer_generations": 2.0,
        "ldme_nle_bitrate_kbps": 420.0,
    }, position=(2, 0))
    g.add("stream_cdn_client", "stream_cdn_client", params={
        "stream_enabled": 1.0, "stream_mode": 6.0, "stream_bitrate_kbps": 350.0,
        "stream_gop_len": 15.0, "ldme_source_origin": 1.0,
    }, position=(3, 0))
    g.add("lcd_flash_player", "lcd_flash_player", params={
        "ldme_display_enabled": 1.0, "ldme_display_device": 2.0,
        "ldme_lcd_bits": 6.0, "ldme_lcd_response_ms": 16.0,
        "ldme_lcd_view_angle": 12.0, "ldme_lcd_subpixel": 0.35,
        "ldme_flash_scale_mode": 1.0,
    }, position=(4, 0))
    g.connect("source", "out", "mobile_camera_isp", "in")
    g.connect("mobile_camera_isp", "out", "file_transfer_node", "in")
    g.connect("file_transfer_node", "out", "stream_cdn_client", "in")
    g.connect("stream_cdn_client", "out", "lcd_flash_player", "in")
    return g


def ldme_test1_mobile_camera_graph() -> Graph:
    """LDME 测试一：同一场景，不同手机（独立手机成像节点 -> 零失真直出监看）。"""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("mobile_camera_isp", "mobile_camera_isp", params={
        "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 2.0, "ldme_capture_mode": 0.0,
    }, position=(1, 0))
    g.add("display", "display", params={"enabled": 0.0}, position=(2, 0))
    g.connect("source", "out", "mobile_camera_isp", "in")
    g.connect("mobile_camera_isp", "out", "display", "in")
    return g


def ldme_test2_transcode_cascade_graph() -> Graph:
    """LDME 测试二：同一视频，不同转码链（文件传输与多代重编码 -> 优酷/土豆流媒体转码）。"""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("file_transfer_node", "file_transfer_node", params={
        "ldme_transfer_enabled": 1.0, "ldme_transfer_method": 4.0, "ldme_transfer_generations": 2.0,
    }, position=(1, 0))
    g.add("stream_cdn_client", "stream_cdn_client", params={
        "stream_enabled": 1.0, "stream_mode": 6.0, "stream_bitrate_kbps": 350.0,
        "stream_gop_len": 15.0,
    }, position=(2, 0))
    g.add("display", "display", params={"enabled": 0.0}, position=(3, 0))
    g.connect("source", "out", "file_transfer_node", "in")
    g.connect("file_transfer_node", "out", "stream_cdn_client", "in")
    g.connect("stream_cdn_client", "out", "display", "in")
    return g


def ldme_test3_display_comparison_graph() -> Graph:
    """LDME 测试三：同一文件，不同播放设备（固定标清 H.264 流媒体源 -> 早期 LCD 与 Flash 播放器）。"""
    g = Graph()
    g.add("source", "source", position=(0, 0))
    g.add("stream_cdn_client", "stream_cdn_client", params={
        "stream_enabled": 1.0, "stream_mode": 7.0, "stream_bitrate_kbps": 700.0,
    }, position=(1, 0))
    g.add("lcd_flash_player", "lcd_flash_player", params={
        "ldme_display_enabled": 1.0, "ldme_display_device": 2.0,
        "ldme_lcd_bits": 6.0, "ldme_lcd_response_ms": 16.0, "ldme_flash_scale_mode": 1.0,
    }, position=(2, 0))
    g.connect("source", "out", "stream_cdn_client", "in")
    g.connect("stream_cdn_client", "out", "lcd_flash_player", "in")
    return g


GRAPH_PRESETS["ldme_n73_youku_flash_lcd_chain"] = {
    "label": "诺基亚拍摄经剪辑上传优酷全链路",
    "note": "串联完整古早数字媒介管线：场景 → Nokia N73 拍摄 (`mobile_camera_isp`) → USB 导入与桌面剪辑导出 (`file_transfer_node`) → 优酷式 FLV 转码 (`stream_cdn_client`) → Flash 播放器缩放与 2007 桌面 TN 液晶显示 (`lcd_flash_player`)",
    "build": ldme_full_archaeology_chain_graph,
    "params": {
        "camera": {"ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 2.0, "ldme_capture_mode": 0.0},
        "codec": {
            "ldme_transfer_enabled": 1.0, "ldme_transfer_method": 4.0, "ldme_transfer_generations": 2.0,
            "stream_enabled": 1.0, "stream_mode": 6.0, "stream_bitrate_kbps": 350.0, "ldme_source_origin": 1.0,
        },
        "display": {
            "enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 2.0,
            "ldme_lcd_bits": 6.0, "ldme_lcd_response_ms": 16.0, "ldme_flash_scale_mode": 1.0,
        },
    },
}

GRAPH_PRESETS["ldme_bench1_mobile_cameras"] = {
    "label": "测试一 同场景不同手机摄影",
    "note": "基准测试一（同一场景，不同手机）：通过 `ldme_mobile_profile` 切换比较 Nokia 6600、Nokia N73、Nokia N95、Nokia 5800、iPhone 2007、iPhone 3G、iPhone 3GS 与 iPhone 4S 的传感器、镜头、白平衡、降噪涂抹与锐化光晕差异",
    "build": ldme_test1_mobile_camera_graph,
    "params": {
        "camera": {"ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 2.0, "ldme_capture_mode": 0.0},
        "display": {"enabled": 0.0},
    },
}

GRAPH_PRESETS["ldme_bench2_transcode_chains"] = {
    "label": "测试二 同视频不同转码与传输链",
    "note": "基准测试二（同一视频，不同转码链）：对比单次 FLV 320×240 转码、多代桌面剪辑重编码级联 (V0→V1→V2→V3)、标清 H.264 480×360 以及低帧率 3GP 上传转码",
    "build": ldme_test2_transcode_cascade_graph,
    "params": {
        "codec": {
            "ldme_transfer_enabled": 1.0, "ldme_transfer_method": 4.0, "ldme_transfer_generations": 2.0,
            "stream_enabled": 1.0, "stream_mode": 6.0, "stream_bitrate_kbps": 350.0,
        },
        "display": {"enabled": 0.0},
    },
}

GRAPH_PRESETS["ldme_bench3_display_devices"] = {
    "label": "测试三 同文件不同播放与显示设备",
    "note": "基准测试三（同一文件，不同播放设备）：固定解码后的流媒体画面，切换比较早期手机 RGB565 TFT 液晶屏、2007 桌面 6-bit TN 液晶显示器 (含 Flash 窗口缩放) 与经典 CRT 监视器",
    "build": ldme_test3_display_comparison_graph,
    "params": {
        "codec": {"stream_enabled": 1.0, "stream_mode": 7.0, "stream_bitrate_kbps": 700.0},
        "display": {
            "enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 2.0,
            "ldme_lcd_bits": 6.0, "ldme_lcd_response_ms": 16.0, "ldme_flash_scale_mode": 1.0,
        },
    },
}


def audio_10_subsystems_av_sync_graph() -> Graph:
    """10 Audio Subsystems + Unified A/V Clock Synchronization End-to-End Graph."""
    g = Graph()
    g.add("src", "source", position=(40, 180))
    g.add("cam", "mobile_camera_isp", params={"ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 1.0}, position=(220, 180))
    g.add("av_sync", "av_sync_clock", params={
        "av_sync_enabled": 1.0,
        "av_sync_offset_ms": 45.0,
        "av_clock_drift_ppm": 180.0,
        "av_frame_drop_rate": 0.04,
        "av_drop_sync_mode": 0.0,
        "av_edit_interval_s": 1.2,
        "av_edit_splice_mode": 2.0,
    }, position=(430, 180))
    g.add("disp", "display", position=(650, 180))
    g.connect("src", "out", "cam", "in")
    g.connect("cam", "out", "av_sync", "in")
    g.connect("av_sync", "out", "disp", "in")
    return g


GRAPH_PRESETS["audio_10_subsystems_av_sync_suite"] = {
    "label": "音频10大子系统与音视频时钟同步全链路台",
    "note": "覆盖 01 麦克风与录音前端、02 旧手机扬声器与听筒、03 数字采样与格式转换、04 感知音频编码、05 移动电话与网络语音、06 声学空间与麦克风位置、07 播放器与系统音频链、08 铃声/MIDI与移动音源、09 广播与网络平台响度处理、10 音视频同步与时钟系统 (PTS/晶振漂移/掉帧对位/剪辑点瞬态)",
    "build": audio_10_subsystems_av_sync_graph,
    "params": {
        "audio": {
            "enabled": 1.0,
            "mic_type": 4.0, "mic_agc_amount": 0.55,
            "space_preset": 1.0, "space_mic_distance_m": 1.2,
            "adc_enabled": 1.0, "adc_sample_rate_hz": 22050.0, "adc_bit_depth": 12.0,
            "pcodec_type": 1.0, "pcodec_bitrate_kbps": 64.0,
            "sys_audio_preset": 1.0,
            "loudness_mode": 2.0, "loudness_target_lufs": -14.0,
            "av_sync_enabled": 1.0, "av_sync_offset_ms": 45.0, "av_clock_drift_ppm": 180.0,
        },
        "display": {"enabled": 0.0},
    },
}
