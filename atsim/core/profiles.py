"""
The five layers of the virtual signal chain, as a parameter schema.

    A camera   -> "what shot it"        (before a signal exists)
    B format   -> "how the signal was made"  (bandwidth, composite coding)
    C tape     -> "what the medium did to it" (transport, dropout, wear)
    D player   -> "how it was read back"      (TBC, head, output level)
    E display  -> "how it was watched"        (CRT, gamma, overscan, glass)

Values are all nominal 0..1 unless a physically meaningful unit is given.
Bounds are UI ranges; `phys_min`/`phys_max` mark the physically plausible zone,
so the UI can warn when a value has left the calibrated envelope (the
"physical model as floor, artistic deviation allowed" decision).
"""
from __future__ import annotations

from .params import ParamSchema, merge_values

LAYER_LABELS = {
    "camera": "01 成像系统 (摄像机)",
    "film": "01b 胶片光化学 (胶片/LOMO)",
    "format": "02 信号编码 (格式)",
    "connector": "02c 链路接口",
    "tape": "03 记录介质 (磁带)",
    "player": "04 读取解码 (播放设备)",
    "display": "05 显示设备 (CRT)",
    "overlay": "06 字幕机",
    "audio": "07 音频系统 (10大声学子系统与音画同步)",
}


# **The night-vision functions by name.**  `nv_mode` is a selector, not a slider -- a number between 0 and 3 gives the
# user no way to know which function they are choosing -- so the schema carries the four names and the interface
# presents them.  The mechanism still reads the number, so a settings file and the API are unchanged.
# **The four functions, named here so the label can carry them.**  `ParamSpec` has no `choices` field,
# so the selector's names live in the label and the help text rather than in the parameter machinery.


def build_schema() -> ParamSchema:
    s = ParamSchema()

    # ------------------------------------------------------------------ A
    cam = s.layer("camera")
    g = "传感器"
    cam.define("sensor_noise", "传感器噪声", 0.06, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.35, source="CCD/CMOS 消费级摄像机估算",
               help="光子散粒噪声与读出噪声的合成电平。")
    cam.define("noise_grain", "噪声颗粒尺度", 1.4, 1.0, 6.0, unit="px", group=g,
               phys_min=1.0, phys_max=3.0, source="传感器像素间距估算",
               help="噪声的空间相关长度；1.0 为逐像素白噪声。")
    cam.define("fixed_pattern", "固定图案噪声", 0.05, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.25, source="CCD 列固定图案噪声估算",
               help="不随时间变化、固定在传感器位置上的噪点。")
    cam.define("dynamic_range", "动态范围", 8.0, 4.0, 14.0, unit="stops", group=g,
               phys_min=6.0, phys_max=12.0, source="1990 年代 CCD 约 8 档",
               help="高光滚降起点；越小越容易过曝成一片死白。")
    g = "曝光与白平衡"
    # --- automatic control loops ---------------------------------------------
    # A handycam has no operator setting exposure or white balance; it has loops that measure the
    # picture, move a control, and take time doing it.  That is where a consumer camcorder's
    # character comes from, and a static offset cannot express it: a static cast cannot drift.
    #
    # All three are off by default, so a camera with no automatic control renders exactly as before.
    g = "自动控制"
    cam.define("ae_enabled", "自动曝光 (AE)", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0,
               help="**这是消费级摄像机味道的主要来源。** 手持机没有人在调曝光，"
                    "它有一个环：测画面亮度、动一个控制、而且要花时间。"
                    "于是「突然进暗处」的表现是**慢慢变亮**，而不是瞬间变亮。"
                    "静止的偏移做不出这件事——静止的偏移不会漂移。")
    cam.define("ae_target", "AE 目标亮度", 0.46, 0.2, 0.7, unit="0-1", group=g,
               phys_min=0.35, phys_max=0.55,
               help="环要维持的画面平均亮度（中央加权，因为真实测光不是均匀平均）。"
                    "偏高会过曝，偏低会欠曝——这正是不同机器" +
                    "对同一场景给出不同曝光的原因。")
    cam.define("ae_tau_s", "AE 时间常数", 1.1, 0.05, 6.0, unit="s", group=g,
               phys_min=0.3, phys_max=3.0,
               help="环响应的时间常数。数值来自实测的渐进曲线：一个时间常数处走到约 63%。"
                    "**注意这不是「收敛速度」而是「逼近平衡的速率」**——"
                    "环在测量停止变化时停在目标上，而不是永远差 1/e。")
    cam.define("ae_limit_stops", "AE 调节范围", 2.0, 0.5, 4.0, unit="stop", group=g,
               phys_min=1.0, phys_max=3.0,
               help="环能推多远。真实 AE 有范围限制，超出就放弃——"
                    "这就是为什么极暗场景下画面仍然很暗，而不是被硬拉亮。")
    cam.define("awb_enabled", "自动白平衡 (AWB)", 0.0, 0.0, 1.0, unit="bool", group=g,
               step=1.0, phys_min=0.0, phys_max=1.0,
               help="**色彩漂移的来源。** 环把画面的平均色度归零，所以灯泡下发黄、"
                    "一两秒后变中性，镜头转向窗户又摆回去。"
                    "静态色偏做不出这个——而这正是 90 年代家庭录像最典型的样子。")
    cam.define("awb_tau_s", "AWB 时间常数", 1.6, 0.05, 8.0, unit="s", group=g,
               phys_min=0.5, phys_max=4.0)
    cam.define("awb_limit", "AWB 调节范围", 0.42, 0.05, 0.8, unit="0-1", group=g,
               phys_min=0.2, phys_max=0.6,
               help="每通道的校正范围。真实的 AWB 有上限，超出就**放弃并保留色偏**——"
                    "这就是为什么在强色光下拍的素材仍然是那个颜色。")
    cam.define("af_enabled", "自动对焦 (AF)", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0,
               help="对比度检测 AF：**它测不了焦点，只能测对比度**，所以它迈一步、看对比度有没有变好、"
                    "变差就反向。低对比度场景下它**找不到峰，会一直找**——"
                    "「清晰 → 模糊 → 寻找焦点 → 清晰」就是从这里来的。")
    cam.define("af_tau_s", "AF 步进间隔", 0.9, 0.05, 4.0, unit="s", group=g,
               phys_min=0.2, phys_max=2.0,
               help="每次试探之间的间隔。真实 AF 不会逐帧狂搜，它按一个节奏试。")
    cam.define("autopresettle_s", "开机已适应时长", 0.0, 0.0, 60.0, unit="s", group=g,
               phys_min=0.0, phys_max=30.0,
               help="**这段录像开始之前，相机已经适应这个场景多久了。**"
                    "0 = 录像从开机开始，此时相机最不稳定；"
                    "设成 10 秒则三个环都已基本稳定，画面上看不出开场摆动。"
                    "它按解析衰减计算，所以「已适应」不是代码里的特例，只是剩余暂态更短。")
    cam.define("exposure", "曝光", 0.0, -2.0, 2.0, unit="stops", group=g,
               phys_min=-1.0, phys_max=1.0, source="自动曝光残余误差")
    cam.define("white_balance_temp", "白平衡色温偏移", 0.0, -1.0, 1.0, unit="0-1", group=g,
               phys_min=-0.5, phys_max=0.5, source="消费级 AWB 偏差",
               help="负值偏冷，正值偏暖（室内灯光下常见的偏色）。")
    g = "镜头"
    cam.define("sharpness", "镜头锐度", 0.75, 0.0, 2.0, unit="0-2", group=g,
               phys_min=0.4, phys_max=1.2, source="消费级变焦镜头 MTF")
    cam.define("chroma_aberration", "色差", 0.05, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.3, source="低端镜头横向色差估算")
    cam.define("axis_offset", "光轴偏心", 0.0, 0.0, 0.6, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.3,
               help="**真实镜头的装配不会正好对中传感器**，所以光轴会偏离画面中心。"
                    "**所有径向像差都是相对光轴、而不是相对画面中心度量的**——"
                    "所以偏心会把「像差为零的那一点」挪走：**画面正中的高光就获得了半径，不再是零。**"
                    "这就是为什么正中的灯在完美对中的镜头下不该有色边，而真机上能看到。")
    cam.define("coma", "彗差", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="**轴外的一点会成彗星状**：通过镜头外缘的光线放大率与通过中心的不同，"
                    "所以它不是点，而是**亮核加一条拖尾**。拖尾是**单向的、朝光轴外的**，并随视场半径增大。"
                    "这就是为什么街灯落在画面角落时像一颗抹开的泪滴。")
    cam.define("astigmatism", "像散", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="**轴外光线在切向平面和径向平面上的焦距不同**，一点被成成一条短线，"
                    "而且**两个焦面上短线的方向是翻转的**。"
                    "签名是**方向性**：模糊沿某一个轴更强，而像场弯曲和球差都是圆的。")
    cam.define("spherical_aberration", "球差", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="**通过镜头外缘的光线聚焦更近**，所以焦点是软的、而且**光晕与亮度有关**："
                    "亮点长出大而圆的晕，暗点仍然紧实。"
                    "**这就是大光圈全开软、收小光圈变锐的原因**，也是它和像场弯曲（随半径）的区别。")
    cam.define("lens_distortion", "镜头畸变", 0.0, -1.0, 1.0, unit="-1..1", group=g,
               phys_min=-0.4, phys_max=0.4,
               help="**镜头本身的几何畸变**：简易镜头是桶形（负）、长焦是枕形（正）。"
                    "注意相机端原本**完全没有几何项**——从镜头出来的画面四边是绝对直的。"
                    "它和「05 显示设备」的管面曲面共用径向几何，但**符号相反**：那是显像管端面，这是玻璃。")
    cam.define("field_curvature", "像场弯曲", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="**简易镜头把平面景物成在曲面上**，所以中心合焦的那个平面并不是四角合焦的平面。"
                    "结果就是**同一个对焦设定下、中间锐利而四周发软**——这也是为什么当年的机器"
                    "中心永远比角落好看，跟光圈多大无关。"
                    "模糊量是**视场半径**的函数：中心为零、向外增大，因为点扩散函数随视场角的平方增长。")
    cam.define("longitudinal_ca", "纵向色差", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="**镜头把三个波长聚焦在轴上的不同距离**，所以在感光面上总有一个合焦、另外两个不合焦。"
                    "**和横向色差正好相反：它不产生任何位移**——红边和蓝边在同一位置、只是都更宽，"
                    "所以彩色边出现在高光的**四周**，而不是偏向某一侧。"
                    "这是它和「横向色差」的本质区别：一个是位移，一个是散焦。")
    # --- 操作者行为（手持、颠簸、变焦）----------------------------------------
    # 这一层不是器件属性，而是**拿着它的人**的属性。它是让素材读起来像"某件事的录像"
    # 而不是"一张被加了故障的图"的那一层。
    #
    # **手持抖动不是噪声。** 最直观的实现是每帧一个随机偏移，那是错的：人的抖动是**窄带且有节奏**的，
    # 而这个节奏才是它读起来像手、而不像故障的原因。手是几个准周期分量的叠加：
    #   0.4-1.5 Hz  姿态摇摆（全身缓慢平衡修正）——幅度最大
    #   1.5-3 Hz    走路与呼吸
    #   4-8 Hz      手臂与肩部肌活动
    #   8-12 Hz     生理性震颤（运动系统固有，始终存在）
    # 幅度随频率下降，**震颤约为摇摆的百分之一**，而这个**比例**才是眼睛读到的东西。
    g = "操作者行为"
    cam.define("shake_intensity", "手持抖动", 0.0, 0.0, 4.0, unit="x", group=g,
               phys_min=0.0, phys_max=2.0,
               help="四个窄带分量按实测比例叠加：摇摆 0.9 Hz、走呼吸 2.1 Hz、臂肩 5.5 Hz、"
                    "生理震颤 9.5 Hz，幅度依次为画面的 1.0%/0.42%/0.16%/0.08%。"
                    "**1.0 约合 5.6 px 标准差**（720 宽）。"
                    "关键是**节奏**而不是幅度——所以它是振子而不是滤波过的随机游走，"
                    "频谱集中度实测 0.22（平坦谱为 0.008）。")
    cam.define("bump_rate_hz", "颠簸发生率", 0.0, 0.0, 3.0, unit="次/秒", group=g,
               phys_min=0.0, phys_max=1.0,
               help="被撞到：一个台阶、一扇门、别人碰到手肘。**这些是事件，不是连续运动**，"
                    "所以和失落到粘带一样是事件列表——稀疏的尖峰叠在一个安静的底上"
                    "（实测 80 分位偏差为 0.00 px）。")
    cam.define("bump_amount", "颠簸强度", 0.0, 0.0, 2.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               help="幅度取自一个**有长尾的分布**：大多数颠簸很小，少数很大。"
                    "形状是**猛撞后指数恢复**（先快后慢），不是阶跃也不是对称脉冲。")
    cam.define("zoom_travel", "变焦行程", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               help="变焦马达的行程。**消费机的变焦不是线性的**：小直流电机带齿轮组，"
                    "先要克服静摩擦（慢）、中段扫过去（快）、松手时**直接断电停住**——"
                    "末尾是停住而不是减速，因为机构摩擦力足够保持位置。")
    cam.define("zoom_kind", "变焦类型", 1.0, 0.0, 2.0, unit="0-2", group=g, step=1.0,
               phys_min=0.0, phys_max=2.0,
               help="0=线性；1=**消费机马达**（克服静摩擦 → 扫过 → 断电停住）；"
                    "2=专业伺服（两端都平滑）。中段斜率 0.91、末端 1.70——"
                    "断电的马达**仍然有速度**，这正是它和缓入缓出的区别。")
    cam.define("wt_zoom", "W-T 变焦位置", 0.0, 0.0, 1.0, unit="W..T", group=g,
               phys_min=0.0, phys_max=1.0, source="SONY DCR-TRV310E 20x 光学变焦镜头 (f=3.6–72mm, f/1.4–2.9)",
               help="0.0 = W（广角端，小细节融合成柔和光晕）；1.0 = T（望远放大端）。"
                    "**真实的光学变焦发生在传感器采样与 OLPF 之前**：向 T 端推近时，不仅视场角收窄、"
                    "有效光圈从 f/1.4 降至 f/2.9（触发 AGC 增益补偿）、枕形畸变与纵向/横向色差随焦距上升，"
                    "而且**原本在广角端因光学 PSF 与传感器扩散而融合成一片的光晕，在放大后会分离并解析为"
                    "更具体的光源细节**（如独立的 LED 灯珠、双灯管结构与机箱冲孔网眼）。")
    cam.define("wt_max_zoom", "T 端最大倍率", 4.0, 1.0, 12.0, unit="×", group=g,
               phys_min=1.5, phys_max=8.0, source="光学变焦常用取景段 1x–8x",
               help="T 端（wt_zoom=1.0）对应的光学放大倍率 M = wt_max_zoom ** wt_zoom。")
    cam.define("zoom_center_x", "变焦光轴中心 X", 0.5, 0.1, 0.9, unit="0-1", group=g,
               phys_min=0.25, phys_max=0.75,
               help="W-T 放大缩小的光学中心横坐标（对准发光主体时可观察光晕向具体光源结构解析的过程）。")
    cam.define("zoom_center_y", "变焦光轴中心 Y", 0.5, 0.1, 0.9, unit="0-1", group=g,
               phys_min=0.25, phys_max=0.75,
               help="W-T 放大缩小的光学中心纵坐标。")
    cam.define("zoom_dynamic_mode", "W-T 动态变焦轨迹", 0.0, 0.0, 3.0, unit="模式", group=g, step=1.0,
               phys_min=0.0, phys_max=3.0,
               help="0=固定在当前 W-T 位置；1=W→T 电动推进放大；2=T→W 电动拉远缩小；3=W↔T 往复变焦。")
    cam.define("zoom_speed", "电动变焦速度", 0.35, 0.05, 1.5, unit="1/s", group=g,
               phys_min=0.15, phys_max=0.85,
               help="W-T 摇杆驱动直流变焦马达的速度；快速变焦期间会伴随自动对焦（AF）短暂呼吸失焦。")
    cam.define("zoom_resolve_detail", "放大解析光源细节", 0.75, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.3, phys_max=1.2,
               help="控制从 W 端（光晕融合）向 T 端（具体光源细节：独立灯珠、灯管间隙、散热网孔）的光学解析度跃迁强度。")
    cam.define("zoom_osd", "显示 W-T 变焦条 OSD", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0,
               help="在取景器左上侧绘制索尼 Handycam 标志性的 W [━━━|━━━] T 白色变焦指示条。")
    cam.define("motion_blur", "运动模糊", 0.15, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.05, phys_max=0.6, source="1/60s 快门")
    cam.define("ccd_smear", "CCD 垂直拖尾", 0.35, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="隔行 CCD 读出期间的电荷泄漏",
               help="强光在垂直方向拉出的长条拖影。消费级 CCD 在夜间拍摄灯光时非常明显。")
    # --- 光学与传感器（笔记第 5 节）--------------------------------------------
    # 这三个是**属于「这一台相机」而不是属于介质或链路**的故障，也是让素材能被认出
    # 来自某一台机器的东西。
    g = "光学与传感器"

    cam.define("lens_fog", "镜头起雾", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.8,
               help="**结在前镜片上的水汽**——和它容易混淆的三种故障都不同："
                    "灰尘是**挡住**光、边缘锐利；镀膜磨损会散射但留下有结构的雾翳；"
                    "散焦是**整幅图像被卷积**、边缘对称展宽。"
                    "起雾是**高光泽扩散**：水膜表面自身就是个透镜，"
                    "所以亮部向暗部**平滑而无边界**地扩散，**黑位先抬且抬得最多**，"
                    "同时高光长出一圈很宽很软的裙边。"
                    "从冷车抱进暖房的手持机起雾，是最典型也最少被建模的故障之一，"
                    "而且它会在几分钟里随玻璃升温而恢复——所以它同时属于环境维度。")

    cam.define("bloom_amount", "电荷溢出", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.6,
               help="**饱和像元的电荷沿垂直转移通道溢出**，所以亮源产生的是**垂直条纹，不是圆形光晕**。"
                    "这正是它和 halation 的区别：圆形光晕是光学效应（镜头或盖板玻璃散射），"
                    "垂直条纹是电荷输运效应。CCD 手持机对着太阳拍会出现这条竖纹，"
                    "也就是笔记把它单列的原因。")
    cam.define("bloom_threshold", "溢出阈值", 0.82, 0.5, 1.0, unit="0-1", group=g,
               phys_min=0.7, phys_max=0.95,
               help="像元到多少亮度才算溢出。这是阱容对应的电平——真实 CCD 大约在 80% 左右饱和。")
    cam.define("dead_sites", "坏点数量", 0.0, 0.0, 8.0, unit="个", group=g, step=1.0,
               phys_min=0.0, phys_max=4.0,
               help="**这一台相机传感器上的失效像元数量。**"
                    "CCD 是几十万个独立结构组成的大阵列，其中一些会失效；"
                    "失效分两类且看起来不同：**死点**每帧都是黑的，**亮点（卡住）**每帧都是满值。"
                    "两者位置**永远固定**——这正是要点：会动的缺陷是噪声，不动的缺陷是**那台相机的记忆**。"
                    "真实机器上几个坏点是正常的，超过几个属于不良品。"
                    "**数量按传感器面积缩放**，所以预览和全分辨率看到的是同一台传感器。")
    cam.define("stuck_fraction", "亮点占比", 0.45, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.2, phys_max=0.7,
               help="失效像元里「卡在满值」而非「读出全黑」的比例。两者不等概率："
                    "短路读出为黑，复位失效为满。")
    cam.define("halation", "高光光晕/泛光", 0.45, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.9, source="CCD 过饱和溢出 + 镜头内反射",
               help="过曝区域的扩展光晕，会带出紫红色边缘。与显示阶段的辉光不同：它发生在传感器上。")
    cam.define("halation_source_colour", "光晕跟随光源色", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               help="**散射光带的是光源自己的颜色**，所以蓝色细线周围的光晕应该是蓝白的。"
                    "0 = 固定的镜片镀膜色调（原行为），1 = 完全跟随炸白区的自身颜色。"
                    "实测：蓝色细线在固定色调的光晕下回到 B−R +0.000（自身 +0.024），"
                    "而且没有任何参数值能恢复——因为提高饱和度只能放大已存在的差，"
                    "无法恢复被中性光除掉的差。")
    cam.define("shadow_depth", "暗部与中间调压深", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               help="**只压暗中间调和暗部，亮部增益恰好为 1**（gain(L) = 1 − a·(1−L)²）。"
                    "用途：提高亮细线与其周围的对比，**而不给线本身加任何光**——"
                    "因为加光会同时抬高线和周围、从而减小它们的差，"
                    "也就稀释了线的颜色（这正是灯杆蓝色丢失的原因）。"
                    "实测：原来靠调低泛光来提对比，结果对比从 0.2185 掉到 0.2069，方向相反。")
    # --- the night-vision modes ------------------------------------------------
    cam.define("nv_mode", "夜视模式 0关/1NightShot/2Super/3ColorSlow", 0.00, 0.0, 3.0, unit="", group=g,
               phys_min=0.0, phys_max=3.0, source="通用 DV / Digital 8 / DSC-F717 的三种夜视模式（含 DCR-TRV 慢快门对应关系）",
               help="**从物理底层重建的索尼夜视功能（保持通用 DV 的三种夜视模式）。**"
                "**默认 0（关闭）：模式是被选择的功能，不是一个一直在运行的机制。**"
                "0 = 关；1 = NightShot（拉下夜视拨杆：拆掉红外截止滤镜 + 开红外照明，正常快门，显示为绿单色）；"
                "2 = Super NightShot（红外夜视 + 慢快门/多帧累积，所以【更亮且更少噪】；"
                "在 **Sony DCR-TRV 系列（如 DCR-TRV310E）** 上，拉下夜视键为 `NightShot`，"
                "按一下机身 `SLOW SHUTTER` 键显示 `NIGHTSHOT 1`，再按一下显示 `NIGHTSHOT 2`——"
                "两者即本模式下不同档位的慢快门/帧累积，可通过 `nv_accumulate` / `nv_shutter_frames` 控制档位）；"
                "3 = Color Slow Shutter（保留彩色、只放慢快门，不拆红外滤镜）。"
                "对照实拍 NightShot 素材（遮除 OSD）分段实测："
                "TRV `NightShot`（`c_0063..c_0091`）mean 0.278、G 占比 0.377、R/G 0.731；"
                "TRV `NightShot 1`（`c_0109..c_0135`）mean 0.268、G 占比 0.415、R/G 0.556；"
                "TRV `NightShot 2`（`c_0136..c_0168`）mean 0.363、G 占比 0.440、R/G 0.464、B/G 0.812。")
    cam.define("nv_illuminator", "红外照明强度", 1.20, 0.0, 4.0, unit="0-4", group=g,
               phys_min=0.0, phys_max=0.6,
               help="**平方反比、且在锥形光束内**——所以近场被照亮、走廊远处归黑。"
                    "这是手持 NightShot 画面最好认的特征，而它来自灯而不是场景。"
                    "实测：`illuminator=1.0` 时场从近场 7.41 降到地平线 0.11，58/58 步单调下降。")
    cam.define("nv_beam_x", "照明光轴 X", 0.5, 0.0, 1.0, unit="x", group=g,
               phys_min=0.0, phys_max=1.0, help="红外灯瞄准的位置（画面横向）。")
    cam.define("nv_beam_y", "照明光轴 Y", 0.75, 0.0, 1.0, unit="y", group=g,
               phys_min=0.0, phys_max=1.0, help="红外灯瞄准的位置（画面纵向；机位在画面底部）。")
    cam.define("nv_beam_angle", "光束角", 0.55, 0.1, 2.0, unit="rad", group=g,
               phys_min=0.15, phys_max=1.5, help="红外灯的锥形光束宽度。")
    cam.define("nv_ir_albedo", "近红外反射占比", 0.35, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.9,
               help="**NightShot 的物理核心：拆掉红外截止滤镜后，画面由【近红外反射率】主导，**"
                    "而近红外反射率与可见光颜色【无关】——这是为什么皮肤、植被、白布会发亮，"
                    "而深色衣物和沥青变黑，这是任何调色都做不出来的。")
    cam.define("nv_ir_response", "红外响应", 1.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               help="红外截止滤镜被移除的程度（0 = 滤镜在位）。**这是近红外响应的总闸门**——"
                    "设为 0 时 `ir_highlights` 直接返回原图。\n"
                    "**我一度以为「输入已是夜视输出时应当关掉它」，而实测推翻了这个假设**："
                    "把真实 NightShot 帧喂进链路，关掉它在**每一点上都更差**——\n"
                    "  t=78s  实拍 mean 0.387 R/G 0.367 | ir 1.0 → 0.423/0.645 | ir 0.0 → 0.433/0.665\n"
                    "  t=60s  实拍 mean 0.242 R/G 0.587 | ir 1.0 → 0.265/0.687 | ir 0.0 → 0.282/0.711\n"
                    "**所以 1.0 保留为默认。**\n"
                    "**而这次测量暴露的真实偏差是**：均值一致地高约 **12%**、R/G 一致地偏高——"
                    "输入帧已经是夜视输出（mean 与 R/G 本来是对的），**而链路在它上面又叠了一层色调曲线**。"
                    "那个偏差**不在这一项**，尚未定位。")
    cam.define("nv_ir_focus_shift", "红外焦点偏移 (px)", 0.5, 0.0, 12.0, unit="px", group=g,
               phys_min=0.0, phys_max=6.0,
               help="**note.txt 第六条机制，而它自己说「这是很多人不知道的」。**\n"
                    "镜头是为【可见光】聚焦设计的，而 700–1000nm 折射率不同，**近红外的像面落在可见光像面之后**。\n"
                    "**后果不是画面变糊**——可见光像仍然锐利——**而是【近红外那一层的贡献】变软**："
                    "植被与被灯照亮的表面失去细节结构，可见光边缘依旧清晰。0 = 与可见光共焦。\n"
                    "**第一次实现被实测推翻并回退**：模糊施加在 `ir` 上（spread 归一化【之前】），"
                    "而归一化那一步除以 `lum_new.std()`——**模糊先缩小了分母，因子就变大去补偿**，"
                    "shift 2.0 时把均值从 0.3750 推到 **0.9697**。**归一化恰好把焦点偏移本该去掉的细节又装了回去。**\n"
                    "**第二次修正**：施加在 spread 归一化【之后】时，`lum_new = lum_new - (lum_new - blur(lum_new))` "
                    "**恰好等于 `blur(lum_new)`**——即整幅图被替换成它自己的模糊副本，**可见光部分也在内**，"
                    "而它自己上方两行的注释写的是「画面不会变软，变软的只是红外贡献」。现在只对 `m * ir` 做散焦再混合。\n"
                    "**默认值 1.5 也太大，这是第三次修正，而且是量出来的**。逐级测（照片输入，HF = mean|dx|）：\n"
                    "进入 0.02326（100%）→ add_illuminator 0.02422（104%）→ **ir_highlights 0.01710（74%）** → "
                    "monochrome_green 0.01713（74%）。**整条夜视链里只有这一步在削细节，26% 全部来自这个焦点偏移。**\n"
                    "把本参数降到 0.5：**HF 回到 0.02321（100%），粒度 lag-1 从 0.220 降到 0.092**（输入是 0.083）。\n"
                    "1.5px 在 720×540 上是把红外层糊掉约 3 个像素的尺度，对「近红外略失焦」而言过强；"
                    "0.5px 保留机制的量级而不吃掉整幅图的细节。")
    cam.define("nv_nir_material", "近红外材质判定 (1开/0关)", 1.0, 0.0, 1.0, group=g,
               phys_min=0.0, phys_max=1.0,
               help="**用视觉模型判定场景材质，再按材质的近红外反射率成像**，而不是把 RGB 转成灰度。"
                    "近红外反射率与可见光亮度【无关**（植被 2.10、皮肤 1.55、深色衣物 0.28——7.5 倍差距），"
                    "所以亮度曲线表达不出这个差异，而这是普通夜视滤镜最难自然模拟的部分。"
                    "**本机没有可用视觉模型时自动退回亮度估计**，那不是降级而是文档化的回退路径。")
    #: **`nv_nir_cell_px` replaces `nv_nir_grid`, and the reason is that a cell COUNT is not a physical quantity.**
    #: The old parameter took 8 cells across whatever the frame was: 66x50 px per cell at the 533x400 the figures were
    #: calibrated at, and **180x135 px -- 7.5x the area -- on the clip's own 1440x1080**.  `IR_ALBEDO` runs 0.28 to
    #: 2.10, so a 7.5x step at a cell edge is stamped onto the picture as a band, and the plateau of that band is one
    #: cell wide: the user photographed exactly that, rectangles a sixth of the frame across, while the mechanism's own
    #: figures "passed" because they were taken where a cell was a quarter of the area.
    cam.define("nv_nir_cell_px", "材质判定格子边长", 66.0, 16.0, 512.0, unit="px", group=g,
               phys_min=32.0, phys_max=256.0, source="标定分辨率 533/8 = 66 px、400/8 = 50 px，取较短边 66",
               help="**把画面按【固定像素边长】切块逐块判定材质**，而不是按固定块数。\n"
                    "**改这次的原因**：旧参数 `nv_nir_grid` 是「横切 8 块」，"
                    "而 8 块在不同分辨率下是**不同的物理尺寸**——533×400 时每块 66×50 px，"
                    "1440×1080 时每块 **180×135 px（面积 7.5 倍）**。"
                    "`IR_ALBEDO` 的范围是 0.28–2.10，**边界上 7.5 倍的台阶会被印成一条带，带的平台宽度就是一个格子**，"
                    "所以用户看到的是「一块一块的光」。\n"
                    "默认 **66 px** 复现标定时的物理尺度（533/8 = 66、400/8 = 50，取较短边）。"
                    "格子按两个方向分别算，所以**任何画幅比例下格子都是方的**——"
                    "单一块数会给出 4:3 上「高比宽多三分之一」的格子，那样量化的形状就取决于画幅比例了，"
                    "而画幅比例不是关于场景的物理陈述。\n"
                    "实测代价（720×540）：8×8 约 10.8 s、12×12 约 10.1 s、16×16 约 16.5 s、24×24 约 33 s，"
                    "而分析**每片段只跑一次**。\n"
                    "**更细并不自动更好，这是量出来的**：模型判「不确定」的格子占比 8×8 时 9%、16×16 时 11%、"
                    "32×32 时 11.5%——格子越小，分类器拿到的上下文越少。所以暴露的是格子边长，不是块数。")
    cam.define("nv_nir_grid_max", "材质判定最大块数", 32.0, 2.0, 64.0, group=g, step=1.0,
               phys_min=8.0, phys_max=48.0, source="32×32 时判定耗时接近一分钟，且 unsure 占比不再改善",
               help="格子边长的上限之外，再封一个块数上限，避免超大画面上判定耗时失控。"
                    "实测标注耗时 8×8 约 10.8 s、16×16 约 16.5 s、24×24 约 33 s、32×32 约 55 s，"
                    "而不确定占比在 16×16 之后就不再下降。")
    # ---- the DSP's output level ---------------------------------------------------------------------------------
    #: **Two parameters, because the measurement needs two and one brightness cannot supply them.**  Against the
    #: clip's own frames at native resolution the model was wrong in a specific, *asymmetric* way -- p1 0.0368 after
    #: the guided albedo (it had been 0.0162), p50 0.2431 vs 0.1619, p99 0.9041 vs 0.8451.  A gain moves p50 and p99
    #: together but pushes p1 the wrong way, and a gamma sets one ratio while leaving the rest to follow.  The two
    #: independent operations are a pedestal (setup) and a gain, and both exist in a camcorder's DSP.
    #:
    #: **The schema defaults are NEUTRAL and the calibrated values live on the NightShot modes.**  The first version
    #: put 0.75 and 0.009 here, and that pushed a NightShot calibration onto every other mode: five checks failed, all
    #: about an ordinary visible-light picture ("a 2.5x source peaks at 0.837 with 0 pixels at or above 1.0", "the
    #: sensor's white point follows its dynamic range" -- 0.837 at every range).  **The values were solved from
    #: NightShot material, so they are the gain and black level THAT MODE settles on**, which is what `MODE_PRESETS`
    #: is for.  A project that has not chosen a mode is untouched, and so is every visible-light measurement here.
    cam.define("dsp_gain", "DSP 增益", 1.0, 0.1, 4.0, unit="x", group=g,
               phys_min=0.5, phys_max=2.0,
               source="NightShot 两个模式取 0.55 档曝光（等效 0.75 增益）；其余模式中立",
               help="**DSP 稳定下来的增益**，乘在信号上、加黑电平之前。\n"
                    "**增益不该砍白点，这是本参数最重要的一条**：`y = g·x` 会把整条链的上限压成 `g`——"
                    "纯白输入逐级实测 max **0.7590**，而 DV 帧到 **1.0000**；DV 帧有 **6.71%** 的像素在 0.90 以上，"
                    "模型是 **0.00%**。**这就是「对比度太差」和「风扇发光部分应该要更亮」**——"
                    "风扇是全画面最亮的东西，而模型没有地方让它去。\n"
                    "**所以这个压暗量现在由 `nv_exposure_trim` 承载**（加在 `_roll_off`【之前】，"
                    "而 `_roll_off` 会把传感器饱和点映回 1.0，白点因此保住）。\n"
                    "本参数保持 1.0（不压），除非用户明确要用增益本身调电平。")
    cam.define("nv_highlight_knee", "高光肩部膝点", 0.70, 0.5, 0.99, unit="0-1", group=g,
               phys_min=0.5, phys_max=0.99,
               source="实拍帧在 0.83 附近就不再往上放；膝点取 0.70 让肩部从那里开始",
               help="**肩部从哪里开始弯。**\n"
                    "上面的量在实拍帧上定的是「最亮处到多亮」（`p95` 0.8322），这个定「从哪里开始弯」。\n"
                    "**它一度完全无效，而原因是这个项目已经记过多次的那个形状**："
                    "`_read_params` 读 `cfg.get(\"nv_highlight_knee\", 0.85)`，而 `cfg` 是**解析后**的参数集——"
                    "**schema 没有声明的键永远到不了这里**，所以 0.85 / 0.75 / 0.65 三个值给出**逐位相同**的画面。"
                    "与早前 `nv_exposure_trim` 的故障同形，也与模式预设覆盖那个找过六次的故障同形："
                    "**一个参数存在于一层而不存在于另一层。**\n"
                    "膝点越低，进入压缩的高光越多、画面越平；越高则越接近直接裁剪。0.70 是让肩部覆盖实拍帧"
                    "真正开始收敛的那个区间。")
    cam.define("nv_highlight_shape", "高光肩部指数形状", 2.5, 1.05, 4.0, unit="p", group=g,
               phys_min=1.05, phys_max=3.0,
               source="实拍帧亮度 p95 0.847 反解：p=1.6 把峰值 0.95 压到 0.834",
               help="**肩部曲线的形状指数，大于 1 时压缩发生在肩部的【顶端】。**\n"
                    "**为什么形状比大小更要紧**（沿相机链路逐级实测）：\n"
                    "  `_expose` 后        luma p95 0.5330   peak p95 0.5526\n"
                    "  `_roll_off` 后      luma p95 0.5955   peak p95 0.6442\n"
                    "  `_dsp_level` 后     luma p95 0.9423   peak p95 1.0000\n"
                    "**`_dsp_level` 把十分之一的画面顶到天花板**（峰值 >0.95 的像素占 10.18%），而这正是肩部要收回来的。\n"
                    "第一版用 `s**2`：它的导数 `2s` **在 `s = 1` 处为 0**——"
                    "所以修正最强的地方在肩部中段，**恰好在画面被钉住的地方完全失效**。"
                    "实测：峰值 0.95（`s = 0.83`）出来只有 **0.907**，压了 0.043，而需要 0.118；全画幅平均改动 **0.0039**。\n"
                    "`s**p`（`p > 1`）有同样两个不动点、同样在恒等线之下，但导数 `p·s^(p-1)` **在 `s = 1` 处最大**"
                    "——**咬得最狠的地方就在顶端**。`p = 1.6` 把峰值 0.95 压到 **0.834**，"
                    "对应实拍帧自己的亮度 p95 **0.847**，而 `s = 0` 仍精确落在膝点上。")
    cam.define("nv_highlight_compress", "高光肩部压缩", 1.0, 0.0, 6.0, unit="x", group=g,
               phys_min=0.0, phys_max=0.8,
               source="实拍帧 p95 0.8322 反解；模型 0.9387、>0.95 占 4.76% 对实拍 1.04%",
               help="**高光肩部：决定「最亮的东西有多亮」的那一步。**\n"
                    "**为什么需要它。** 在把黑位与中间调都标定到实拍 NightShot 2 帧之后，模型只剩下顶端不一致：\n"
                    "  统计量    模型      实拍\n"
                    "  p95      0.9387    0.8322   ← 亮端顶到天花板\n"
                    "  >0.95    4.76%     1.04%    ← 4.6 倍的画面被钉在那里\n"
                    "  暗部<0.15 20.6%    19.9%    ← **这一项已经吻合**\n"
                    "**而现有参数一个都动不了前两项**，这是量出来的、不是假设：\n"
                    "  `nv_highlight_saturation` 2.0 → 0.8：p95 只从 0.9387 到 0.9297（变化 2.5 倍、p95 动 0.009）"
                    "——它是闸门以上的**驱动量**（去饱和机制），不是肩部；\n"
                    "  `dsp_gamma` 0.60 → 0.80：暗部占比 20.6% → **48.9%**——同一条曲线的两端，救高光就塌暗部；\n"
                    "  `dynamic_range` 定的是膝点而不是上限。\n"
                    "**所以模型有了正确的黑位和中间调之后就没有余量了**——这正是「光感太差、不够通透」的物理含义："
                    "最亮的地方全都顶到天花板，画面就没有余地表现「这个比那个更亮」。\n"
                    "**机制**：膝点以上把超出量走 `1 - exp(-x)` 而不是直接相加，"
                    "因此**像素的次序被保住**、膝点自身的值精确不动、膝点以下完全不受影响——"
                    "所以刚标定好的黑位与中间调不会被它扰动。作用在**亮度**上，让高光均匀去饱和而不偏色。\n"
                    "膝点 **0.85** 取自实拍帧：那以上实拍就基本不再放东西了。")
    cam.define("dsp_gamma", "DSP 伽马（暗部抬升）", 0.60, 0.3, 2.0, unit="g", group=g,
               phys_min=0.4, phys_max=1.0,
               source="实拍帧的 p25/p50 反解：0.052^g=0.151 要 g=0.588，0.188^g=0.271 要 g=0.634",
               help="**DSP 的传输曲线 `y = x^g`**，0 与 1 固定，`g < 1` 拉伸暗部、压缩亮部。\n"
                    "**为什么需要它，而基座电平不够**（对实拍 NightShot 2 帧实测）：\n"
                    "  分位      模型      实拍      比值\n"
                    "  p1       0.0186    0.0471    0.39x\n"
                    "  p5       0.0247    0.0565    0.44x\n"
                    "  p25      0.0522    0.1511    0.35x\n"
                    "  p50      0.1881    0.2710    0.69x\n"
                    "  p75      0.7027    0.4841    1.45x\n"
                    "**模型暗部低 2–3 倍、亮部高 1.4 倍**——对比度放错了位置。"
                    "`note.txt` 说的正是这个：NightShot 的黑不是纯黑，而是**深绿灰**。\n"
                    "**基座电平修不了它**：`dsp_black` 加的是常数，抬了黑电平但暗部的【比例】没动；"
                    "增益则会连已经过高的亮部一起抬。`x ** g` 在 `g < 1` 时把低端拉开、把高端压回去，"
                    "而且 0 和 1 不动——这是标准的摄像机传输函数，不是调色。\n"
                    "**0.60 是从实拍帧自己的分位反解的**：`0.052^g = 0.151` 要 `g = 0.588`，"
                    "`0.188^g = 0.271` 要 `g = 0.634`，两者相差 0.05 以内，取 0.60。")
    cam.define("dsp_contrast", "DSP 对比度曲线", 0.70, -0.8, 2.0, unit="k-1", group=g,
               phys_min=-0.5, phys_max=1.0,
               source="DV 帧上扫出来的最优：p1 0.0344 对目标 0.0353、std 0.2988 对 0.2896；0 为恒等",
               help="**DSP 的色调曲线**，形状是绕 0.5 的幂律 `y = 0.5·(x/0.5)^k`，所以 0.0、0.5、1.0 三点不动，"
                    "**动的只有两端**——这正是对比度控制该做的事，也是那个年代每台消费级摄像机的对比度旋钮。\n"
                    "**为什么需要它**（新照片对 DV 帧实测）：\n"
                    "  DV 帧    p1 0.0353   p50 0.1833   p99 1.0000   std 0.2896\n"
                    "  模型     p1 0.0447   p50 0.2052   p99 0.6585   std 0.1782\n"
                    "**模型的中灰偏高、暗端偏高、白端偏低**——整段量程是被【从两端往中间挤】，不是整体平移，"
                    "而幂律曲线修的就是这个形状。\n"
                    "**0.25 是在 DV 帧上扫出来的**（其余参数默认）：\n"
                    "  k−1  p1      p5      p50     p99     std\n"
                    "  0.00 0.0517  0.0615  0.2028  1.0000  0.2794\n"
                    "  0.15 0.0404  0.0481  0.1781  1.0000  0.2919\n"
                    "  **0.25 0.0344  0.0413  0.1636  1.0000  0.2988**   ← 取这个：暗端 0.0344 对目标 0.0353，"
                    "对比度 0.2988 对 0.2896，白点保持 1.0000\n"
                    "  0.35 0.0295  0.0358  0.1505  1.0000  0.3048\n"
                    "  0.50 0.0241  0.0292  0.1328  1.0000  0.3123\n"
                    "**0.35 以上暗端就低于目标了**（0.0295 对 0.0353），所以 0.25 是唯一同时对上暗端与对比度的点。\n"
                    "**它跑在增益之前**，所以两者按「先色调、后电平」的顺序复合，与摄像机的实际顺序一致。")
    cam.define("dsp_black", "DSP 黑电平（setup）", 0.0, -0.2, 0.3, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.10,
               source="NightShot 两个模式取 0.009（反解自素材 p1 与 p50）；其余模式中立",
               help="**基座电平（setup / pedestal）**：摄像机不把最暗的场景像素记在码值零上，"
                    "而是放在消隐电平之上——**这样噪声底不会被削掉，画面也有一个黑参考**。\n"
                    "**这就是素材第 1 百分位是 0.037 而不是 0.000 的原因**：量的是摄像机的输出，"
                    "而摄像机的输出有黑电平。Sony 自己的 DV 黑电平是码值 16（全范围意义上约 0.063），"
                    "比这份素材略高。\n"
                    "**顺序是「先增益、后黑电平」，这是物理顺序**：基座加在已经放大过的信号上，即在数字化之后。"
                    "反过来会把基座也一起缩放，于是 `dsp_black` 在每个增益下含义都不同、两个控制分不开。\n"
                    "**与 `dsp_gain` 一样，0.009 放在 NightShot 的模式预设里。**")
    #: **Kept so an existing project still loads, and inert because the cell size now decides.**  A stale value here
    #: would otherwise silently override the physical parameter, which is the "preset that removes a control" fault
    #: this file records three times.
    cam.define("nv_nir_grid", "材质判定网格（已废弃，由格子边长决定）", 8.0, 2.0, 32.0, group=g,
               phys_min=2, phys_max=32,
               help="**已废弃**：块数是随分辨率变化的物理量，所以判定改由 `nv_nir_cell_px`（格子边长，单位像素）"
                    "决定，这个参数不再被读取。保留是为了让旧工程仍能导入，并让找到它的人读到原因。")
    cam.define("nv_nir_depth", "近红外距离衰减 (1开/0关)", 1.0, 0.0, 1.0, group=g,
               phys_min=0.0, phys_max=1.0,
               help="**用视觉模型推断相对深度，让红外灯的衰减按【距离】而不是按【半径】。**"
                    "note 的第一条机制是「物体距离增加后反射光减弱——近处物体可能很亮，远处背景却仍然很暗」，"
                    "而纯径向场只能压暗画面四角（实测默认档中心到边缘只差 21%），做不到这件事。"
                    "灯光照度按 1/d² 衰减，所以有逐像素深度就是一除。**" 
                    "深度是【相对值】（模型的零点任意），所以只用于衰减的形状，不代入任何绝对距离。**")
    cam.define("nv_nir_margin_floor", "材质判定置信下限", 0.02, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="块的首选与次选概率差低于此值时判为「不确定」，该块退回亮度估计。"
                    "**实测参考照片的 margin 从 0.692 到 0.030**，所以这不是假想情况："
                    "0.03 的判定不能把植被的 2.10 贴到水泥上。")
    cam.define("nv_cast_variation", "绿投射的空间变化", 0.0, 0.0, 1.0, group=g,
               phys_min=0.0, phys_max=0.8,
               help="**实拍的绿不是均匀的，而固定投影必然是均匀的。**\n"
                    "实测模型的色调散度是 **0.1714**，而真实三段是 **0.5519 / 0.1170 / 0.1129**"
                    "——**平均颜色对了，分布不对**，这正是「实拍图片并不是均匀的绿」。\n"
                    "物理上变化的是【有多少红外到达传感器】：红外灯的光随距离衰减，"
                    "所以远离灯的表面近红外回报更弱、颜色更多来自环境可见光——"
                    "**为权重拟合的那个蓝色泄漏应当在远场更弱、近场更强。**\n"
                    "变化依据是视觉模型给出的相对深度（已挂在 spec 上）；"
                    "**亮度不受影响**：这是逐像素缩放权重后再按自身亮度归一化。")
    cam.define("nv_shot_noise", "散粒噪声 (随信号√)", 0.10, 0.0, 0.5, group=g,
               phys_min=0.0, phys_max=0.3,
               help="**随【信号强度】变化的噪声——与读出噪声物理行为相反，两者都必须有。**\n"
                    "note 说：「加入随曝光和信号强度变化的噪声、暗部读出噪声、固定图样噪声，以及高亮区域的饱和。"
                    "**不要简单地对整张图叠加均匀颗粒；暗部和高光的表现应该不同。**」\n"
                    "读出噪声是【固定绝对值】，所以它在暗部【相对】最大；"
                    "散粒噪声按信号平方根增长，所以在绝对值上亮部最大。\n"
                    "**只有其一的模型必然在某一端是错的**，而这一点单个标量测不出来——"
                    "实测把 read_noise 从 0 提到 0.25，暗/亮时间噪声比反而从 0.281 升到 0.448。\n"
                    "逐像素标准差取 shot_noise × √lum：泊松到达的方差正比于均值，所以偏差按平方根。")
    cam.define("nv_shutter_frames", "慢快门积分帧数 (1=不积分)", 1.0, 1.0, 32.0, unit="帧", group=g,
               phys_min=1.0, phys_max=16.0,
               help="**一次曝光跨越几帧——这是「慢快门」而不是「多帧累积」。**"
                    "两者算出来的均值相同，物理意义不同：note 第四条说「根据曝光时间对运动物体进行时间积分，"
                    "而不是只复制几张半透明图层。快门越慢，运动物体在一次曝光期间移动得越远，残影就越明显」。"
                    "**多帧求和是多次独立曝光**（信号相干相加、噪声按平方根相加，画面【更干净】）；"
                    "**慢快门是一次曝光开着**（运动主体被拖出残影，且不会回来）。"
                    "把前者当成后者，就是为什么模型的模式 2 没有拖影而素材的 NIGHTSHOT 2 明显有。")
    cam.define("nv_green_mix", "绿单色显示", 1.00, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               help="1 = 纯绿单色取景，0 = 保留原色。"
                    "**是【单轴增益】而不是去饱和**——实测素材绿段的通道差大而色相散度小"
                    "（0.06-0.18 对彩色照片的 0.37），这正是单色取景器接到彩色录像机的结果。")
    cam.define("nv_green_gain", "绿通道增益", 1.0, 0.5, 1.6, unit="x", group=g,
               phys_min=0.7, phys_max=1.4,
               help="绿单色显示里绿通道的增益，**默认 1.0**。\n"
                    "**1.0 是唯一使目标比值精确成立的取值，而这不是巧合，是自由度算出来的。**\n"
                    "投影满足 `w @ LUMA = 1`（这条约束让 `mono` 精确承载画面亮度）；"
                    "在这个约束下 `w` 只剩【两个】自由比值，而目标 R/G 0.5434 与 B/G 0.8525 正好用掉两个。"
                    "**再乘一个绿增益就是第三个比值变化，它没有地方可去，只能去改那两个目标。**实测：\n"
                    "  w = [0.5434, 1, 0.8525] → /(w@LUMA) → [0.6418, 1.1811, 1.0069]   R/G 0.5434  B/G 0.8525\n"
                    "  再 ×[1, 1.15, 1] 再 /(w@LUMA) → [0.5814, 1.2303, 0.9120]        R/G 0.4725  B/G 0.7413\n"
                    "**旧代码在增益之后又做了一次按亮度归一化，那正是把目标比值打散的一步**（已删除）。\n"
                    "所以 1.0 以上是「把绿枪推得更狠」的**真实增益**：比值和亮度一起动，"
                    "这就是显示器绿枪被推狠时发生的事；而那套由目标解出的原色比例，是显示器**被造出来时**"
                    "就定下的东西。\n"
                    "**旧的默认 1.15 是反解出来的，而它反解的依据是三条素材里最淡的那一条**——"
                    "同一段三帧原图重测 R/G 是 0.517/0.540/0.574、B/G 是 0.840/0.842/0.876，"
                    "所以 1.15 相对这段自己的中心偏高 33%（红）与 8%（蓝）。")
    cam.define("nv_exposure_trim", "夜视曝光补偿 (档)", -0.85, -3.0, 3.0, unit="stops", group=g,
               phys_min=-1.0, phys_max=0.0,
               help="**夜视模式自己的曝光补偿，不是全局 `exposure`。**\n"
                    "**它现在同时承载了原来 `dsp_gain = 0.75` 的压暗量**，因为那个增益加在 `_roll_off`【之后】，"
                    "等于给整条链压了一个 `g` 的硬上限——**模型永远到不了满量程**：纯白输入逐级实测 max 0.7590，"
                    "而 DV 帧到 1.0000；DV 帧有 6.71% 的像素在 0.90 以上，模型是 0.00%。"
                    "**增益不该砍白点**——摄像机把增益调低并不失去记录白色的能力，是曝光开大去补偿。\n"
                    "移到这里在电平上等价（`2 ** -0.55 = 0.683` 对 `0.75`，同一量级），"
                    "**而 `_roll_off` 会把传感器饱和点映回 1.0，所以白点保住了**"
                    "（实测 p99 与 max 都到 1.0000，过曝 8.45% 对素材 6.71%）。\n"
                    "**标定用的是 DV 帧，而与照片的折中必须写出来**：照片自身的最大亮度只有 0.77，"
                    "**它没有高光可丢**，而两张输入要的 trim 不同——\n"
                    "  trim −0.10 → DV帧 p50 0.2713 / 过曝 18.01%   照片 0.2695 / 8.91%\n"
                    "  trim −0.55 → DV帧 p50 0.2028 / 过曝  8.45%   照片 0.1966 / 0.00%  ← 取这个\n"
                    "  trim −0.70 → DV帧 p50 0.1826 / 过曝  5.11%   照片 0.1791 / 0.00%（p50 正中目标但超出声明范围）\n"
                    "**−0.70 让 p50 精确命中目标的 0.1833，但超出 `[-1.0, 0]` 的物理范围；"
                    "取 −0.55 是因为它在范围内，而照片连十二分之一档都没动**（0.2144 → 0.1966）。\n"
                    "**更早的 −0.10 是在两张照片上一起扫出来的**（对基准素材 NIGHTSHOT 0：mean 0.2346 / p90 0.5870 / "
                    "p99 0.9511；trim −0.10 合并误差 0.153/0.221，trim −0.50 是 0.835/0.806）——"
                    "**那一轮是在白点被上限卡住的代码上标的，所以白点修好之后它必须重标，这就是上面这次重标**。\n"
                    "**它必须加在【AE 环路之后】**：环路里 `self.exposure = ...update(...)` 会【替换】这个值，"
                    "此前在构造函数里设置会被一行之后丢掉——实测 −0.10/−0.6/−1.0/−1.4/−1.8 输出逐位相同。\n"
                    "**而且它必须在 schema 中声明**：未声明时 `SCHEMA.resolve_all` 会在阶段看到它之前丢掉它"
                    "（实测：传 −1.2 到达阶段的是 −0.1）——**这是本工作第二次踩到同一个机制。**")
    cam.define("saturation_stops", "饱和点 (低于白点几档)", 0.30, 0.0, 3.0, unit="stops", group=g,
               phys_min=0.0, phys_max=2.0,
               help="**传感器的饱和点——它决定画面亮部能否被打爆。**\n"
                    "实测：模型最亮的像素是 **0.8686，全画面没有任何像素被裁切**，"
                    "而素材到 **1.0、0.66% 的像素在 0.98 以上**。\n"
                    "逐调用追踪证明 `_roll_off` **不是**原因——它裁切正确，位置就在它的白点上。"
                    "**原因是没有任何值到达那个白点**：默认 `dynamic_range` 为 8 时白点正好是 1.0，"
                    "而画面经曝光与白平衡后的最大值只有 0.87，所以 `lin >= white` 那一支从不触发，"
                    "肩部坐在它本该压缩的内容【上方】。\n"
                    "0 保持曲线原样；大于 0 则把白点降低相应档位，"
                    "于是本该停在 0.95 的高光会被钉在 1.0 上——**真实传感器就是这样过曝的**。\n"
                    "**传感器的饱和点是传感器自身的属性，不是它动态范围的属性**——"
                    "所以它是一个独立控制，而不是另一个 `dynamic_range`（后者调【低】反而把白点推得更高）。\n"
                    "**注意：新参数必须在 schema 中声明**，否则 `SCHEMA.resolve_all` 会在阶段看到它之前丢掉它"
                    "（实测：`merge_values` 后是 0.6，`resolve_all` 后是 ABSENT）。\n"
                    "**实测限度**：默认 0.30 把 max 从 0.8686 抬到 0.8857、p99 从 0.8061 到 0.8334，"
                    "而 clipped 仍是 0.0000。素材是 max 1.0000、p99 0.9511、clip 0.00657。"
                    "**所以这一项只部分闭合**：真实照片的最亮像素在 0.87 附近，"
                    "而曲线把能量重新分配进肩部而不是让它裁切——需要光源使画面自身有更宽的动态范围"
                    "（例如更强的照明或更亮的高光），才可能真正打爆。")
    cam.define("nv_highlight_saturation", "高光饱和驱动 (只作用于闸门以上)", 2.0, 0.0, 8.0, unit="x", group=g,
               phys_min=1.0, phys_max=3.0,
               help="**取景器的驱动量，只作用在【闸门以上】——这才是高光变白的机制。**\n"
                    "显示把画面构成 `lum × w`，所以某通道在 `lum × w == 1` 处饱和："
                    "归一化后的权重是 [0.6069 1.2021 0.9904]，**绿在 lum 0.832 处饱和**，蓝 1.010，红 1.648。\n"
                    "**闸门以下投影精确成立**，所以中间调保持实测的 R:G:B，为素材拟合的标定完全不动；"
                    "**闸门以上最强通道被钉住而其余继续爬升**，于是该像素**去饱和走向白色**，"
                    "其亮度得以离开比例自身的上限。**这正是 note 六条里唯一没有实现的那条：「高亮区域的饱和」。**\n"
                    "**为什么必须加闸门**：第一版把驱动乘在【每个】像素上——实测 gain 1.0→1.6 时"
                    "均值从 0.2370 升到 0.3493，即整体提亮，而那正是早前 `nv_highlight_saturation` 不可用的原因。"
                    "**在没有东西饱和的地方施加增益，得到的不是高光而是更亮的照片。**\n"
                    "**加闸门后实测**（目标是素材的 mean 0.2346 / p99 0.9511 / max 1.0000 / clip 0.00657）：\n"
                    "  gain 1.00 → mean 0.2370  p99 0.8061  max 0.8686  评分 0.3135\n"
                    "  gain 2.00 → mean 【0.2383】 p99 0.8454  max 【0.9799】 评分 【0.1663】\n"
                    "**均值几乎不动（+0.5%）而 max 升了 0.111——这是第一次高光被抬起而没有整体提亮。**\n"
                    "gain 2.0 以上不再变化，因为闸门以上的像素已全部裁切。")
    #: **This parameter is now inert, and the reason is that its premise was wrong.**  See the block comment in
    #: `monochrome_green`: NightShot routes the sensor's output as ONE LUMINANCE onto the green display, so a
    #: visible-light chroma that has been through that step is not in the signal any more, and the picture's green is
    #: the DISPLAY's.  It is kept in the schema so that an existing profile or project that sets it still loads, and
    #: so that a reader who finds it can read why it does nothing instead of concluding that parameters are broken.
    cam.define("nv_chroma_residue", "绿显示残留色度", 0.15, 0.0, 1.0, group=g,
               phys_min=0.0, phys_max=0.5,
               source="实拍 NightShot 2 帧上扫出来的：residue 0.15 让色度到目标的 104%、R/G 差 0.011",
               help="**输入的色度残留，叠加在绿显示的色偏【之外】。**\n"
                    "**它此前被记录为「已废弃，无效」，而原因是写法**：\n"
                    "    out = mono + (rgb - lum) * r      ← 残差项带着【显示自己的权重】\n"
                    "于是输入的彩色被拉到单色通路已经在用的那组比例上，两者分不开——实测 `0.00 / 0.35 / 0.95` "
                    "都到达 0.350 且输出逐位相同。\n"
                    "**物理陈述其实更简单**：NightShot 的画面不是纯单色的——**场景的彩色残留会留进录制**，"
                    "这正是实拍帧的暗部读起来是「深绿灰」而不是中性灰、也正是它的色度【比纯投影更大】的原因。"
                    "所以场景色度加在显示比例**之外**，而显示比例只作用于亮度。\n"
                    "`note.txt` 要的正是这条（它的第一个方案：「提取输入影像的色度信息 … 将色度残留叠加到单色亮度图」），"
                    "并且点出了朴素写法的失败：它应当经过摄像机自己的矩阵，而不是贴上去的 RGB 效果。\n"
                    "**实测（实拍 NightShot 2 帧为目标）：**\n"
                    "  实拍帧           色度 0.10128   R/G 0.4739   B/G 0.8123\n"
                    "  residue 0.00     色度 0.09839   R/G 0.5172   B/G 0.7622\n"
                    "  **residue 0.15   色度 0.10583   R/G 0.4850   B/G 0.7565**   ← 取这个\n"
                    "  residue 0.30     色度 0.11306   R/G 0.4545   B/G 0.7511\n"
                    "**色度上升而 R/G 下降**——画面多了彩色、同时离开绿色，这正是「太绿」与「没有彩色」两个抱怨的方向。"
                    "把绿色原色推得更亮则两者同向，所以 `green_gain` 修不了它，上一轮去拉它反而让画面更绿。")
    cam.define("nv_exposure_steps", "夜视曝光补偿", 0.00, 0.0, 3.0, unit="EV", group=g,
               phys_min=0.0, phys_max=2.5,
               help="**模式自己的曝光提升，走相机正常的曝光通路**——"
                    "所以在高光滚降、增益噪声和白平衡上都和普通曝光一样。"
                    "在末端乘一个系数只会让画面变亮，而没有真的被曝光过。")
    cam.define("nv_accumulate", "帧积累", 8.00, 1.0, 16.0, unit="frames", group=g,
               phys_min=1.0, phys_max=16.0,
               help="**Super NightShot 的关键：信号相干相加、噪声按平方根相加，"
                    "所以信噪比按 sqrt(N) 改善。**单次 N 倍曝光则会同时放大暗电流和散粒噪声，"
                    "两者物理上不同。实测：8 帧使高频能量降到 1/2.878（理论 sqrt(8)=2.83）。")
    cam.define("nv_read_noise", "夜视读出噪声", 0.0, 0.0, 0.2, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.1, help="夜视增益档带来的读出噪声。")
    cam.define("nv_chroma_noise", "彩噪 (0纯亮度噪/1各通道独立)", 0.15, 0.0, 1.0, group=g,
               phys_min=0.0, phys_max=1.0,
               help="**低照度下噪点是有颜色的**：实测三个 NIGHTSHOT 帧的暗部逐通道高频能量为"
                    "0.00736 / 0.00659 / 0.00836，通道间相差 27%。"
                    "read_noise 是单一标量，**无论开多大都做不出这个通道差**——这是性质缺失而不是没调好。"
                    "本项按 read_noise 的比例为每个通道独立抽样并减去三通道均值，"
                    "所以它**只加颜色、不改变噪声对亮度的贡献**。")
    cam.define("nv_illuminator_visible", "照明可见泄漏", 0.02, 0.0, 0.25, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.15,
               help="红外灯自身的微弱可见泄漏——**所以近场有紫偏、远场没有**，"
                    "这是给【灯】上色而不是给画面上色。")
    cam.define("source_glow", "光源色散射晕", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.8,
               help="**光源的光在空气和镜片里散射，散射光带的是光源自己的颜色**——"
                    "橙灯周围是橙晕、白灯周围是白晕。"
                    "这是它和「高光光晕」的本质区别：后者的色调是固定的（偏冷），"
                    "也和「镜头光斑」不同，后者的色调是镀膜色。**两者都不跟随光源。**"
                    "实测参考素材：炸白核心下方 60 行和 100 行处 B−R 到 −0.170 和 −0.510，"
                    "而模型只有 −0.014 和 −0.034——差 15 倍，这是近场最大的缺口。")
    cam.define("overflow_drain", "溢出暗环（深度）", 0.0, 0.0, 0.8, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.4,
               help="**光点溢出时，反溢出电路要抽走多余电荷，而它是从邻居那里抽的**，"
                    "所以高光外圈被压到**低于背景**——真实画面里炸白点周围那圈暗边。"
                    "注意方向和光晕相反：光晕向外**加亮**，这个向外**压暗**，"
                    "所以任何光晕参数都做不出它。深度是**该点自身亮度的一个比例**，"
                    "不是减一个常数：亮背景上暗环更深，全黑背景没有电荷可抽、也就没有暗环。")
    cam.define("overflow_width", "溢出暗环（宽度）", 2.0, 1.0, 6.0, unit="像素", group=g,
               phys_min=1.0, phys_max=3.0,
               help="暗环宽度，单位是**感光点**——抽电荷的电路作用于相邻的光电二极管，"
                    "不是画面上的某个距离，所以这个宽度随传感器的像素尺寸走。")
    cam.define("ccd_gain", "增益档位", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_max=1.0, source="AGC 在低照度下提升增益",
               help="低照度时自动增益提升，会同时放大读出噪声。这是夜间画面颗粒感的来源。")
    g = "色彩科学"
    cam.define("color_cast", "色彩倾向", 0.0, -1.0, 1.0, unit="0-1", group=g,
               phys_min=-0.5, phys_max=0.5, source="各品牌 CCD 滤色片与 DSP 色调",
               help="**正值偏冷（偏青蓝）——Sony 消费级常见；负值偏暖（偏红）。**\n"
                    "**此前的注释把这两个方向写反了**，而代码一直是正确的：`_color_cast` 施加的增益是\n"
                    "    `[1 − 0.16c,  1 + 0.03c,  1 + 0.20c]`   （R, G, B）\n"
                    "所以 c > 0 把 R 压低、B 抬高（偏冷），c < 0 把 R 抬高、B 压低（偏暖）。\n"
                    "**用户看出来的正是这个矛盾**：界面文字说「负值偏青蓝」，而滑块与画面给的是相反的方向。\n"
                    "这会整体改变画面的时代感。")
    cam.define("highlight_tint", "高光染色", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, source="CCD 过曝时的通道非线性",
               help="过曝区域向品红/紫偏移——参考素材里灯光边缘的那圈紫边。")
    cam.define("vcd_color_tone", "港式 VCD 冷暖分色与柔光肩部", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.2, source="90年代末港式 VCD / 胶转磁短片实测色彩科学",
               help="复现 90 年代末香港 VCD 纪实/MV 短片的标志性色彩曲线：暗部黑位微抬并带冷青灰底色（p05≈0.18）、"
                    "中间调人物肤色与紫荆铜雕呈暖金琥珀色（R/G≈1.11, B/G≈0.93）、玻璃幕墙与阴影呈翡翠青绿色（B/G≈1.04），"
                    "同时高光在 0.82 处平滑软肩滚降而不生硬死白。")
    cam.define("ccd_flash_falloff", "SONY CCD 机顶氙气闪光灯与金属高光响应", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.2, source="SONY Cyber-shot 1/2.7英寸 Super HAD CCD 内存卡提取原图 (1024×768 药板闪光实拍)",
               help="建模早期 SONY Cyber-shot CCD 数码相机近距机顶氙气闪光灯直闪的物理特征："
                    "1. 中心亮（Y≈0.44）向四角遵循反平方律与广角 cos^4 衰减至暗部（Y≈0.075，中心/四角比 5.84x）；"
                    "2. 暗部深色桌面保留高饱和暖红棕阶调（R/G≈1.31, B/G≈0.74）；"
                    "3. 铝塑反光面中亮调呈冷银灰（R/G≈0.98, B/G≈1.05）并伴随微透镜高对比边缘蓝紫边与满阱锐利白斑；"
                    "4. 机械快门遮光读出，完全无摄像模式下的垂直拖尾光柱。")
    g = "镜头"
    cam.define("lens_wear", "镜头磨损/镀膜老化", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, source="镜片镀膜磨损与内部起雾",
               help="镀膜磨损不是模糊：它把光散射掉，因此同时造成对比度下降、黑位抬升（雾化），"
                    "并且因为抗反射层正是抑制鬼影的，鬼影会随之加重。")
    cam.define("lens_flare", "镜头光斑/眩光", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.7, source="镜头内部多次反射",
               help="按光圈叶片形状成像的几何鬼影与星芒。注意与「高光光晕」不同："
                    "后者是传感器过曝的圆形溢光，这个是光学孔径形状。")
    cam.define("flare_blades", "光圈叶片数", 6.0, 3.0, 10.0, unit="片", group=g,
               phys_min=5.0, phys_max=9.0, step=1.0, source="常见 6/7/8/9 片光圈",
               help="星芒的射线数。6 片给出六角星芒。")
    cam.define("spike_spread", "衍射星芒分色", 0.6, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="**孔径的衍射星芒不是一张图样**：条纹间距随波长变化，"
                    "所以红、绿、蓝形成星芒的**角度和长度都略有不同**，"
                    "结果是**彩色的射线**——高光向外散出分立的红绿蓝条纹。"
                    "消色差的星芒无论调多大都做不出它：一个几何图形乘上一个颜色，"
                    "三个通道拿到的是同一组射线。0 = 消色差，1 = 明显的分色。")
    cam.define("light_streak_amount", "实拍光条强度", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.2, source="SONY DCR-TRV310E 实拍高光光条与列溢出",
               help="建模 SONY DCR-TRV310E 实拍画面中强光源产生的**光条细节**：包含隔行 Super HAD CCD "
                    "垂直读出通道的电荷泄漏光柱（核心颈部膨大+蓝青色长尾）以及前组镜片/光圈边缘产生的"
                    "定向光学眩光光条。与 W-T 变焦联动：广角端光晕拉出宽光带，放大后分解为每颗灯珠独立的细光条。")
    cam.define("light_streak_angle", "光条延伸角度", 0.0, -90.0, 90.0, unit="deg", group=g,
               phys_min=-45.0, phys_max=45.0,
               help="定向光学眩光光条的倾角（0° 为水平光条，90° 为垂直光条，配合 CCD 垂直拖尾形成十字或斜向光束）。")
    cam.define("light_streak_length", "光条延伸长度", 0.55, 0.1, 1.5, unit="0-1", group=g,
               phys_min=0.2, phys_max=1.2,
               help="控制高光光条向画面外侧延伸的衰减尺度。")
    cam.define("coat_ghost", "变焦镜组镀膜鬼影与彩虹色散弧", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.2, source="单层 MgF2 镀膜双反射鬼影链与广角端色散弧实测标定",
               help="建模短片开头逆光摇摄时的多层光学细节：1990 年代单层 MgF2 增透膜针对绿光（550nm）优化，"
                    "内部镜片二次反射会强烈反射红光与蓝紫光，形成中心对称的品红/蓝紫色反转重影（B-G≈+0.11）、"
                    "穿轴斜向焦散粉紫光带，以及广角端对角象限的青-绿-品红彩虹色散弧（B-R≈+0.10）。")
    cam.define("flare_defocus", "闪光逐通道离焦", 0.5, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="**纵向色差让三个通道聚焦在不同距离上**，所以孔径的**离焦像在蓝通道比红通道更大**。"
                    "结果是亮光源**周围和下方**出现彩色光晕，而不是光源本身被染色——"
                    "参考素材里路灯下方那条蓝纹、灯杆上的蓝色拖尾、以及炸白灯周围的硬边环，都指向它。"
                    "它和「01 成像系统」的 `纵向色差` 是同一套物理，"
                    "但那个作用于**画面**，这个作用于**闪光**——闪光已经是离开孔径焦点之后的光。")
    g = "传感器类型"
    cam.define("sensor_type", "3CCD 传感器", 0.0, 0.0, 1.0, unit="bool", group=g,
               phys_min=0.0, phys_max=1.0, step=1.0, source="3CCD 机型（如 Panasonic NV-GS78）",
               help="三片传感器经棱镜分光，每片独立读出原色。特征不是「更好的单 CCD」，"
                    "而是噪点更低（读出噪声表现为亮度颗粒而非彩色斑点）、色彩分离干净、"
                    "细密纹理上无伪色摩尔纹。")

    g = "演播室 CCU 机架控制与主同步 Genlock"
    cam.define("ccu_enabled", "启用演播室 CCU 与 Genlock 控制", 0.0, 0.0, 1.0, unit="bool", group=g,
               step=1.0, phys_min=0.0, phys_max=1.0,
               source="Sony CCU-370 / Ikegami MA-79 演播室集中式摄像机控制单元与 SPG 黑场主同步发生器",
               help="开启后接入演播室集中式 CCU 机架遥调（Iris、Master Pedestal、R/B 白平衡增益、R/B 黑平衡基座、动态高光膝点 Knee）与全馆 Blackburst 外同步锁相。")
    cam.define("ccu_cam_id", "演播室摄像机机位特性", 0.0, 0.0, 3.0, unit="0-3", group=g,
               step=1.0, phys_min=0.0, phys_max=3.0,
               source="多机位同场照明下的不同品牌/老化个体色差模型",
               help="0=CAM-1 Sony BVP-370 (D65 基准演播室座机)｜1=CAM-2 Ikegami HL-55 (暖调高反差采访机)｜"
                    "2=CAM-3 BTS LDK-90 (偏冷暗部微青全景机)｜3=CAM-4 老旧游机 (阴极/CCD 暗电流抬升 + 绿偏)。"
                    "视频工程师可通过下方 CCU R/B Gain 与 R/B Flare 旋钮将多台异构机位配平到完全一致。")
    cam.define("ccu_iris_ev", "CCU 遥控光圈推杆 (Iris)", 0.0, -2.5, 2.5, unit="stops", group=g,
               phys_min=-1.5, phys_max=1.5,
               source="演播室 RCP 遥控面板纵推光圈杆 (f/1.4 – f/16)",
               help="机房视觉工程师远程实时推拉摄像机镜头光圈，控制画面整体曝光进入高光 Knee 膝点的深度。")
    cam.define("ccu_master_pedestal_ire", "总黑电平基座 (Master Pedestal)", 0.0, -15.0, 25.0, unit="IRE", group=g,
               phys_min=-5.0, phys_max=12.0,
               source="RCP 主黑电平旋钮（校准 7.5 IRE / 0 IRE 消隐基准黑位）",
               help="整体平移三通道暗部黑位基座；配合波形监视器与 PLUGE 测试卡可精确消除暗部死黑截断或灰雾。")
    cam.define("ccu_r_gain", "红通道白平衡增益 (R Gain)", 1.0, 0.5, 1.6, unit="x", group=g,
               phys_min=0.75, phys_max=1.30,
               source="CCU 红路视放白平衡配平旋钮",
               help="调节高光与中间调红通道增益，用于多机位高光白平衡配平。")
    cam.define("ccu_b_gain", "蓝通道白平衡增益 (B Gain)", 1.0, 0.5, 1.6, unit="x", group=g,
               phys_min=0.75, phys_max=1.30,
               source="CCU 蓝路视放白平衡配平旋钮",
               help="调节高光与中间调蓝通道增益，用于多机位高光白平衡配平。")
    cam.define("ccu_r_flare", "红通道黑平衡基座 (R Black / Flare)", 0.0, -0.25, 0.25, unit="-0.25..0.25", group=g,
               phys_min=-0.12, phys_max=0.12,
               source="CCU 红路暗部黑平衡与镜筒杂散光补偿旋钮",
               help="独立微调暗部红通道基座，消除镜头内部杂散光（Flare）或不同机位暗部偏红/偏青。")
    cam.define("ccu_b_flare", "蓝通道黑平衡基座 (B Black / Flare)", 0.0, -0.25, 0.25, unit="-0.25..0.25", group=g,
               phys_min=-0.12, phys_max=0.12,
               source="CCU 蓝路暗部黑平衡与镜筒杂散光补偿旋钮",
               help="独立微调暗部蓝通道基座，消除暗部冷蓝或暖黄偏色。")
    cam.define("ccu_knee_point", "动态高光膝点起点 (Knee Point)", 0.82, 0.50, 0.98, unit="0-1", group=g,
               phys_min=0.72, phys_max=0.92,
               source="广播级摄像机高光动态压缩起点 (80%–95% IRE)",
               help="高于此电平的演播室聚光灯与高反光区域进入 Knee 线性压缩段，防止 100 IRE 硬削波死白。")
    cam.define("ccu_knee_slope", "动态高光膝点斜率 (Knee Slope)", 0.35, 0.05, 1.0, unit="0-1", group=g,
               phys_min=0.15, phys_max=0.65,
               source="广播级摄像机 400%–600% 过曝动态余量压缩斜率",
               help="斜率越小，高光压缩越强（聚光灯下人物额头与白衬衫细节保留越多）。")
    cam.define("genlock_mode", "演播室主同步锁相模式 (Genlock)", 0.0, 0.0, 3.0, unit="0-3", group=g,
               step=1.0, phys_min=0.0, phys_max=3.0,
               source="Tektronix SPG-170A 黑场同步发生器 (Blackburst) 与多机位外同步锁相环",
               help="0=锁定演播室主同步且自动补偿线长 (BB Locked & Timed)｜"
                    "1=脐带线缆长度相移未补偿 (Cable Delay Uncompensated：线长直接引起副载波色相旋转与水平位移)｜"
                    "2=外同步断线/内晶振自由漂移 (Free-Run Crystal Drift：色相连续翻滚 + 行场滑移滚屏)｜"
                    "3=BNC 同步头接触不良间歇掉锁冲击 (Intermittent Genlock Drop)。")
    cam.define("genlock_cable_len_m", "摄像机脐带/同轴线缆长度", 15.0, 0.0, 150.0, unit="m", group=g,
               phys_min=2.0, phys_max=100.0,
               source="Belden 8281 / 演播室三同轴 Triax 线缆传播速度 v_p = 0.66c (5.05 ns/m)",
               help="同轴/脐带线每增加 1 米引入 5.05 ns 延迟，在 NTSC 3.58 MHz 下产生 6.51°/m 的副载波相移（PAL 4.43 MHz 下为 8.06°/m），"
                    "超过 15 米未配平时在切换台上切机位会出现明显色相跳变与右侧消隐黑边。")
    cam.define("genlock_sc_phase_deg", "CCU 副载波相位补偿 (SC Phase)", 0.0, -180.0, 180.0, unit="deg", group=g,
               phys_min=-180.0, phys_max=180.0,
               source="CCU 面板 Subcarrier Phase 360° 粗/细调旋钮",
               help="看着矢量示波器旋转此旋钮，可将因长电缆延迟而偏转的摄像机色同步矢量精确拉回标准靶框。")
    cam.define("genlock_h_phase_px", "CCU 水平行相位微调 (H Phase)", 0.0, -24.0, 24.0, unit="px", group=g,
               phys_min=-12.0, phys_max=12.0,
               source="CCU 面板 H-Phase 水平行定时对齐旋钮",
               help="微调摄像机相对于演播室主同步的行触发时刻，消除多机位切换时的画面左右跳动与消隐边错位。")

    g_ldme_cam = "早期手机与早期 iPhone 摄影管线 (LDME)"
    cam.define("ldme_mobile_enabled", "启用早期手机/iPhone 成像管线", 0.0, 0.0, 1.0, unit="bool", step=1.0, group=g_ldme_cam,
               phys_min=0.0, phys_max=1.0, source="LDME 子系统 A/B (2003–2012 诺基亚与早期 iPhone 六步成像管线)",
               help="开启『光学成像 → 传感器采样 → Bayer去马赛克 → 白平衡与色彩矩阵 → 降噪与USM锐化 → JPEG/3GP/MOV编码』完整手机成像链。")
    cam.define("ldme_mobile_profile", "早期手机与 iPhone 机型预设", 0.0, 0.0, 8.0, unit="机型", step=1.0, group=g_ldme_cam,
               phys_min=0.0, phys_max=8.0, source="Nokia 6600 / N73 / N95 / 5800 & iPhone 2007 / 3G / 3GS / 4S",
               help="0=关闭｜1=Nokia 6600 (VGA)｜2=Nokia N73 (蔡司320万)｜3=Nokia N95 (蔡司500万)｜4=Nokia 5800 (3GP视频)｜"
                    "5=iPhone 2007 (初代200万)｜6=iPhone 3G｜7=iPhone 3GS (VGA视频管线)｜8=iPhone 4S (800万BSI)。")
    cam.define("ldme_capture_mode", "成像工作路径 (拍照 capture / 录像 record)", 0.0, 0.0, 1.0, unit="路径", step=1.0, group=g_ldme_cam,
               phys_min=0.0, phys_max=1.0, source="camera.capture() 静态照片 vs camera.record() 动态视频管线",
               help="0=静态照片路径 capture()（全传感器采样 + JPEG 4:2:0 + EXIF）｜1=动态视频路径 record()（视频分辨率/帧率限制 + 卷帘快门 + 3GP/MOV 帧间编码）。")
    cam.define("ldme_optics_blur", "微型镜头模糊与场曲缩放", 1.0, 0.0, 2.5, unit="x", group=g_ldme_cam,
               phys_min=0.2, phys_max=2.0, source="微型塑胶/蔡司镜头中心解析力与离轴边缘软化")
    cam.define("ldme_vignette", "手机镜头暗角强度缩放", 1.0, 0.0, 2.5, unit="x", group=g_ldme_cam,
               phys_min=0.0, phys_max=2.0, source="微型光圈孔径 cos^4(θ) 暗角")
    cam.define("ldme_sensor_noise", "CMOS 泊松-高斯噪声缩放", 1.0, 0.0, 3.0, unit="x", group=g_ldme_cam,
               phys_min=0.0, phys_max=2.5, source="小尺寸传感器光子散粒噪声与读出噪声")
    cam.define("ldme_bayer_demosaic", "Bayer 去马赛克伪影强度", 1.0, 0.0, 2.5, unit="x", group=g_ldme_cam,
               phys_min=0.0, phys_max=2.0, source="RGGB 拜耳阵列低复杂度插值边缘拉链与伪彩")
    cam.define("ldme_isp_denoise", "ISP 降噪涂抹强度缩放", 1.0, 0.0, 2.5, unit="x", group=g_ldme_cam,
               phys_min=0.0, phys_max=2.0, source="早期手机 ISP 色度抹除与亮度保边涂抹")
    cam.define("ldme_isp_sharpen", "ISP 锐化与边缘光晕缩放", 1.0, 0.0, 2.5, unit="x", group=g_ldme_cam,
               phys_min=0.0, phys_max=2.0, source="早期手机 USM 孔径增强过冲与亮边振铃")
    cam.define("ldme_jpeg_quality", "JPEG 量化品质覆盖 (0为机型默认)", 0.0, 0.0, 100.0, unit="Q", step=1.0, group=g_ldme_cam,
               phys_min=0.0, phys_max=95.0, source="8×8 DCT 量化表与 4:2:0 色度子采样")
    cam.define("ldme_use_real_jpeg", "调用真实 Pillow JPEG 编码器", 0.0, 0.0, 1.0, unit="bool", step=1.0, group=g_ldme_cam,
               phys_min=0.0, phys_max=1.0, source="Pillow 真实 JPEG 字节流编码与解码回读")

    # 物理建模：工频荧光灯频闪、低色温白平衡失效、JPEG 磨损衰减与夜景星芒 (旧手机照片实测标定)
    cam.define("ldme_flicker_enabled", "启用 CMOS 卷帘快门工频频闪条纹", 0.0, 0.0, 1.0, unit="bool", step=1.0, group=g_ldme_cam,
               phys_min=0.0, phys_max=1.0, source="50Hz/60Hz 工频荧光灯/交流LED卷帘快门曝光条纹 (image.jpg)",
               help="物理建模旧手机 CMOS 卷帘快门在 50Hz/60Hz 工频荧光灯下的明暗与青绿/橙黄光谱交替条纹。")
    cam.define("ldme_flicker_mains_hz", "工频交流电网频率", 50.0, 45.0, 65.0, unit="Hz", step=5.0, group=g_ldme_cam,
               phys_min=45.0, phys_max=65.0, source="中国/欧洲/香港 50Hz (100Hz 纹波) vs 北美/日本 60Hz (120Hz 纹波)")
    cam.define("ldme_flicker_depth", "工频频闪明暗条纹深度", 0.0, 0.0, 1.0, unit="0-1", group=g_ldme_cam,
               phys_min=0.0, phys_max=1.0, source="交流电光照强度纹波调制深度")
    cam.define("ldme_flicker_color_drift", "荧光灯波峰波谷色温漂移", 0.65, 0.0, 1.0, unit="0-1", group=g_ldme_cam,
               phys_min=0.0, phys_max=1.0, source="荧光灯汞放电峰青绿 (546nm) 与荧光粉余辉暖橙 (611nm) 交替漂移")
    cam.define("ldme_flicker_num_bands", "全画幅垂直频闪条纹数量", 5.2, 0.5, 20.0, unit="条", step=0.1, group=g_ldme_cam,
               phys_min=0.5, phys_max=20.0, source="由逐行读出时间与曝光快门决定 (实拍约4-8条)")

    cam.define("ldme_tungsten_enabled", "启用钨丝灯/低色温照明与白平衡失效", 0.0, 0.0, 1.0, unit="bool", step=1.0, group=g_ldme_cam,
               phys_min=0.0, phys_max=1.0, source="普朗克黑体辐射与早期手机 ISP AWB 蓝增益钳位 (b0ae5b91...jpg)",
               help="物理建模室内低色温钨丝灯下蓝光匮乏、ISP 蓝通道增益钳位导致的暖黄偏色与餐厅混光芥末青黄。")
    cam.define("ldme_color_temp_k", "环境光源辐射色温", 5500.0, 1800.0, 9500.0, unit="K", step=100.0, group=g_ldme_cam,
               phys_min=1800.0, phys_max=9500.0, source="2700K 钨丝灯/白炽灯、3200K 卤素灯、5500K 日光")
    cam.define("ldme_awb_mode", "ISP 自动白平衡失效模式", 0.0, 0.0, 3.0, unit="模式", step=1.0, group=g_ldme_cam,
               phys_min=0.0, phys_max=3.0, source="0=标准自动｜1=钨丝灯暖黄失效｜2=餐厅混光芥末青黄｜3=蓝通道强增益噪点")

    cam.define("ldme_jpeg_decay_enabled", "启用 JPEG 磨损与多代有损压缩衰减", 0.0, 0.0, 1.0, unit="bool", step=1.0, group=g_ldme_cam,
               phys_min=0.0, phys_max=1.0, source="早期优酷土豆平台转码与网络论坛搬运 JPEG 磨损 (digicamfx)")
    cam.define("ldme_jpeg_decay_gens", "JPEG 重复压缩磨损代数", 1.0, 1.0, 10.0, unit="代", step=1.0, group=g_ldme_cam,
               phys_min=1.0, phys_max=10.0, source="多代 2D 8×8 DCT 量化与色度衰减累积")
    cam.define("ldme_jpeg_decay_quality", "JPEG 磨损基础量化品质", 75.0, 5.0, 95.0, unit="Q", step=1.0, group=g_ldme_cam,
               phys_min=5.0, phys_max=95.0, source="标准 IJG 亮度与色度量化表品质因子 Q")
    cam.define("ldme_jpeg_decay_grid_shift", "跨代 8×8 DCT 网格错位漂移", 2.0, 0.0, 7.0, unit="px", step=1.0, group=g_ldme_cam,
               phys_min=0.0, phys_max=7.0, source="跨代图像裁剪/缩放诱发 8×8 网格错位与双重蚊式振铃噪点")

    cam.define("ldme_night_bloom", "夜景高光漫射与星芒溢出强度", 0.0, 0.0, 2.5, unit="x", group=g_ldme_cam,
               phys_min=0.0, phys_max=2.5, source="微型镜头车灯星芒与点光源垂直光柱溢出 (image_d61d18.png)")

    # ------------------------------------------------------------------ B
    fmt = s.layer("format")
    gf = "滤波器家族"
    fmt.define("filter_kind", "滤波器类型", 1.0, 0.0, 2.0, unit="0-2", group=gf,
               step=1.0, source="消费级电路为恒 K 型级联；测试仪器用巴特沃斯",
               help="0=高斯（零相位，理想，无色边）｜1=恒 K 型（真实电路，有群延迟）"
                    "｜2=巴特沃斯（拐点更陡）。群延迟使色度相对亮度右移，"
                    "这是硬边出现色边的成因之一——零相位滤波做不出这个效应。")
    fmt.define("filter_sections", "级联级数", 3.0, 1.0, 5.0, unit="级", group=gf,
               phys_min=2.0, phys_max=4.0, step=1.0, source="恒 K 型通常为 3 级级联")
    g = "色度通道增益"
    fmt.define("chroma_gain", "色度增益（ACC）", 1.0, 0.0, 3.0, unit="×", group=g,
               phys_min=0.7, phys_max=1.4, source="ACC/APC 自动色度控制的失控范围",
               help="色度通道的整体幅度，也是「饱和度」真正的物理来源：自动色度控制"
                    "（ACC）以副载波色同步的幅度为基准把色度拉回标准电平，"
                    "一旦基准本身出错，色度就会被系统性地放大或压死。"
                    "1.0 = 标准；2 以上是色度卡故障式的过饱和；0.1 以下近乎黑白"
                    "（色度消隐/killer 动作）。参考素材的色度故障片饱和度达 0.87，"
                    "而真实 DV 只有 0.21。")
    g = "模拟复合视频"
    fmt.define("luma_bandwidth", "亮度带宽", 3.0, 0.1, 6.0, unit="MHz", group=g,
               phys_min=2.0, phys_max=5.0, source="VHS 约 3.0 MHz / S-VHS 约 5.0 MHz",
               help="决定细节衰减。VHS 白电平区带宽约 3 MHz，色度区更低。")
    fmt.define("chroma_bandwidth", "色度带宽", 0.75, 0.25, 2.0, unit="MHz", group=g,
               phys_min=0.3, phys_max=1.5, source="VHS 色度欠载记录约 0.6-1.0 MHz",
               help="色度以欠载方式记录，带宽远低于亮度，是 VHS 颜色发糊的根源。")
    fmt.define("chroma_vertical_averaging", "色度垂直平均", 0.85, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.5, phys_max=1.0, source="色度约 40 行，接收机逐行平均",
               help="相邻行色度混合的强度，造成垂直方向的彩边。")
    # The receiver's Y/C separator.  Four real designs of different eras, and the choice
    # is what decides the shape of the coloured fringes on hard edges: a cheap set uses a
    # band split, a better one a glass delay line, a good one two lines or a 2-D comb.
    fmt.define("separation_kind", "Y/C 分离方式", 0.0, 0.0, 5.0, unit="0-4", group=g,
               step=1.0, phys_min=0.0, phys_max=4.0,
               source="消费级接收机的四种分离方案（同年代分层）",
               help="**这是器件选择，不是效果旋钮。** 0=带通分离（廉价机，无延迟线）；"
                    "1=盒式孔径（小横向孔径，载波只被部分抵消，所以彩边粗而宽）；"
                    "2=1H 延迟线（一行玻璃延迟线，抽头相隔一行，副载波自然反相抵消）；"
                    "3=2H 延迟线（0.5×本行 + 0.25×上一行 + 0.25×下一行，垂直孔径更宽，"
                    "串色更少但垂直亮度更糊）；4=二维梳状（横向 ±1 像素、纵向 ±1 行，"
                    "**对角抽头与中心同相位**，所以不需要延迟线就能去掉载波，四者中垂直拖尾最小）。"
                    "梳状滤波器的相位抵消是**自动产生**的，不是靠系数调出来的——"
                    "这就是为什么真实接收机在这几种方案之间选择，"
                    "而**选哪一种决定了边缘彩边的形态**。")
    fmt.define("chroma_vert_blend", "垂直色度反馈", 0.5, 0.0, 0.99, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.9,
               help="**递归垂直色度反馈**——每行的色度是上一行**输出**的一半加当前输入的一半，"
                    "也就是沿列的一阶 IIR，而不是邻行平均。"
                    "这正是 ntsc-rs 的 `chroma_vert_blend`，也是它自己承认的缺陷来源："
                    "输出依赖像素上方的整段历史，所以颜色会拖出几十行。"
                    "**对称平均做不到这件事**——平均在两条行内就消失，而且它是对称的，不会漂移。"
                    "实测参考片：行间色度相关从 lag1 的 +0.98 掉到 lag16 的 +0.20，"
                    "相干长度约 20-30 行且**没有固定周期**（因为反馈是按列独立的，每列历史不同）。"
                    "注意：**只在帧内递归**，不跨帧携带状态——跨帧会把相邻帧耦合起来，"
                    "从而破坏「任意顺序请求帧、缓存仍然正确」这条前提。"
                    "**需要跨帧的那一条另有其人**：`色度时间残留 (C(t-1))`。"
                    "两者都是色度通道的记忆，但一个沿【垂直方向】在帧内递归，一个沿【时间方向】跨帧递归，"
                    "分开放是因为它们的代价不同——前者不影响缓存前提，后者要求按顺序推进。")
    # ---- temporal chroma lag ------------------------------------------------------------------------------------
    # **Where this lives, and why it is not an `nv_*` parameter.**  The note asks for it as part of the chroma-damage
    # model (its §2) and calls the NightShot requirement separately (its second half), so the mechanism belongs to the
    # chroma channel -- the same channel in a VHS deck, a Hi8 camcorder and a NightShot DV path.  Putting it in the
    # night-vision block would have repeated the fault REVIEW.md §3.4 names: confining a sensor/processing parameter
    # to one *display* mode, when the physics does not know what mode the camera is in.  So it is a `format` parameter
    # and it is live on every branch.
    fmt.define("chroma_lag", "色度时间残留 (C(t-1))", 0.0, 0.0, 0.9, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.35,
               source="窄带色度通道的冲激响应记忆（估算）；note.txt §2 的 0.3 权重",
               help="**色度滞后一帧再混回来**，是唯一一条纯粹关于时间的机制："
                    "色度通道的带宽只有亮度的几分之一（VHS 约 0.6-1.0 MHz），"
                    "窄带通道的冲激响应本来就横跨若干行的时间，"
                    "所以颜色比亮度晚到、并在物体移走后残留——"
                    "Digital8 NightShot 的运动拖影、VHS 的彩色尾巴都出自这里。"
                    "0 = 关闭（默认，逐帧输出与从前逐位相同）；"
                    "0.3 = note.txt §2 的 `0.7·C(t) + 0.3·C(t-1)`。"
                    "**实现是递归的**（每帧再混入前一帧的【输出】），"
                    "等效于一条一阶 IIR——0.3 的尾巴衰减到 5% 需要 3 帧，"
                    "这正是「拖影」而不是「双影」的原因。"
                    "**代价与前提**：它必须按顺序推进（与整条链的时基/噪声状态同一条规矩），"
                    "所以第一帧、以及参数改动后的第一帧没有历史、等同于关闭；"
                    "任意顺序跳帧取样不会看到这条机制。")
    fmt.define("chroma_lag_frames", "色度残留记忆深度", 4.0, 1.0, 8.0, unit="帧", group=g,
               step=1.0, phys_min=1.0, phys_max=4.0, source="窄带通道的冲激响应长度",
               help="保留多少帧的色度历史。**它不改变稳态的尾巴**（尾巴由 `色度时间残留` 的递归系数决定），"
                    "决定的是这台机器「记得多久」——历史表填满后，"
                    "一条衰减到 0.3^4 ≈ 0.8% 的尾巴本来也看不见了。"
                    "默认 4 是在实测里能覆盖整条可见尾巴的最小值。")
    fmt.define("comb_depth", "梳状滤波深度", 0.55, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.2, phys_max=0.9, source="NTSC 接收机梳状滤波器",
               help="1H 延迟梳状滤波的权重；越深串色越少，但垂直亮度分辨率下降。")
    fmt.define("cross_color", "交叉色彩 (亮度→色度)", 0.45, 0.0, 1.5, unit="0-1", group=g,
               phys_max=1.0, source="复合信号分离残余",
               help="高频亮度细节被误判为色度，形成彩虹状彩边。")
    fmt.define("cross_luma", "交叉亮度 (色度→亮度)", 0.30, 0.0, 1.5, unit="0-1", group=g,
               phys_max=1.0, source="复合信号分离残余",
               help="色度残留进亮度，形成爬行点状花纹 (dot crawl)。")
    fmt.define("edge_boost", "轮廓增强", 0.25, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, source="摄像机/录像机轮廓补偿电路",
               help="录像机的边缘增强，会在高对比边缘产生白边。")
    fmt.define("color_noise", "色度噪声", 0.25, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, source="色度欠载通道信噪比",
               help="色度通道噪声远高于亮度通道，是彩点噪声的来源。")
    fmt.define("burst_phase_error", "色同步相位误差", 5.0, 0.0, 45.0, unit="deg", group=g,
               phys_max=20.0, source="色同步与色度相位误差",
               help="整场的色相旋转/漂移。")

    g = "测试信号发生器与工程校准台"
    fmt.define("test_gen_enabled", "启用广播级测试信号发生器", 0.0, 0.0, 1.0, unit="bool", group=g,
               step=1.0, phys_min=0.0, phys_max=1.0,
               source="Tektronix TSG-170A / Philips PM5544 标准电视测试信号发生器",
               help="开启后由机内标准测试信号发生器替代外部画面源，直通送入后续模拟复合编码、配线架、磁带、TBC 与示波器，构成工程闭环校准台。")
    fmt.define("test_pattern_mode", "标准测试图卡模式", 0.0, 0.0, 5.0, unit="0-5", group=g,
               step=1.0, phys_min=0.0, phys_max=5.0,
               source="SMPTE RP-219 / EBU Tech 3087 / FCC Multiburst / CCIR Rec.569 / Philips PM5544",
               help="0=关闭（使用外部素材源）｜1=SMPTE RP-219 75%彩条 + 反向蓝条 + -I/White/+Q + PLUGE(-4/0/+4 IRE)｜"
                    "2=EBU 100/0/75/0 PAL 标准彩条 + 线性亮度斜坡｜3=FCC 0.5–5.0 MHz 多波群频响测试卡 (Multiburst)｜"
                    "4=5阶调制阶梯 (微分增益DG/微分相位DP) + 2T正弦平方脉冲与12.5T色亮群延迟脉冲｜"
                    "5=Philips PM5544 / 电视台圆环几何与中心分辨率楔形卡（带动态场扫描秒针）。")
    fmt.define("test_tone_mode", "校准音频测试信号", 0.0, 0.0, 3.0, unit="0-3", group=g,
               step=1.0, phys_min=0.0, phys_max=3.0,
               source="EBU R68 / SMPTE RP-155 广播音频基准电平与抖晃测量规范",
               help="0=跟随原片伴音｜1=1 kHz (-18 dBFS / 0 VU) 正弦基准音｜2=3.15 kHz IEC 60386 走带抖晃 (Wow & Flutter) 测量载波｜"
                    "3=20 Hz–15 kHz 对数扫频音（测量线性磁迹高频滚降与方位角损耗）。")
    fmt.define("test_amplitude_ire", "测试信号峰值电平", 100.0, 50.0, 120.0, unit="IRE", group=g,
               phys_min=75.0, phys_max=110.0,
               source="IEEE Std 205 / Tektronix VM700A 信号发生器输出幅度微调",
               help="100 IRE 为标准满幅度（714 mVpp）；调高至 110–120 IRE 可测试磁带 FM 白削波、同步压低与视放过载。")

    # ----------------------------------------------------- B2 数字编码分支
    cod = s.layer("codec")
    g = "编码分支"
    cod.define("codec_kind", "编码器选择", 0.0, 0.0, 2.0, unit="0-2", group=g,
               step=1.0, phys_min=0.0, phys_max=2.0,
               source="0=不编码（模拟）｜1=DV25｜2=MPEG-1（Video CD）",
               help="选择数字编码分支。0 为模拟通路（VHS/Hi8），不经任何数字编码；"
                    "1 为 DV25；2 为 MPEG-1，也就是 Video CD 用的编码。"
                    "三种通路的伪影结构不同，不能互相近似：DV 是 4:1:1 + 固定码率，"
                    "MPEG-1 是 4:2:0 + 粗量化 + 运动补偿预测。")
    g = "MPEG-1 / Video CD"
    cod.define("vcd_quality", "VCD 码率质量", 1.0, 0.25, 2.0, unit="x", group=g,
               phys_min=0.6, phys_max=1.4,
               source="VCD 固定 1150 kbit/s，352x240",
               help="量化步长的缩放。VCD 的码率只有 DV25 的约二十分之一，"
                    "所以量化粗得多——高频被**成结构地丢弃**，而不是平滑衰减，"
                    "这是它看起来又软又带有块感的原因。")
    cod.define("vcd_width", "VCD 宽度", 352.0, 176.0, 720.0, unit="px", group=g,
               step=1.0, phys_min=352.0, phys_max=352.0,
               source="VCD（MPEG-1）352x240，525/60",
               help="编码前的水平尺寸。VCD 标准固定 352，物理区间也只允许这一个值——"
                    "它不是为了调节画面，而是为了说明 MPEG-1 是在**四分之一分辨率**上"
                    "做运动估计与量化的，这本身就是它伪影粗的一个原因。")
    cod.define("vcd_height", "VCD 高度", 240.0, 120.0, 576.0, unit="px", group=g,
               step=1.0, phys_min=240.0, phys_max=288.0,
               source="VCD（MPEG-1）352x240，525/60；625/50 为 352x288",
               help="编码前的垂直尺寸。525/60 为 240，625/50 为 288。")
    cod.define("vcd_motion_comp", "运动补偿强度", 1.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               help="P 帧从上一帧的**已解码亮度**做运动预测的程度。"
                    "这是 VCD 最独特的伪影来源：预测失败的地方（遮挡、硬切、运动边缘）"
                    "残差大、误差可见，而预测成功的地方画面很干净——"
                    "所以伪影不是逐帧独立的噪声，而是**跟着预测成败走**。"
                    "取 0 退化为逐帧帧内编码，也就失去了这个特征。")
    cod.define("vcd_bitrate_kbps", "VCD 码率", 1150.0, 600.0, 2000.0, unit="kbps",
               group=g, step=50.0, phys_min=1100.0, phys_max=1200.0,
               source="VCD 标准 1150 kbit/s",
               help="参考资料值。当前实现用它设定量化尺度，并未真正做码率控制。")
    cod.define("vcd_hbi_border", "模拟采集 HBI 边缘振铃与亮边", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0, source="模拟复合母带采集压制 VCD 时的左右行消隐振铃边",
               help="复现模拟母带经采集卡转压 VCD 时的 4:3 画幅左右边界特征：左侧低通抗混叠滤波器阶跃振铃暗槽与过冲，"
                    "以及右侧行消隐（HBI）钳位亮白细边。")
    g = "DV / MiniDV 编码"
    cod.define("dv_enabled", "启用 DV 编码分支", 0.0, 0.0, 1.0, unit="bool", group=g,
               phys_min=0.0, phys_max=1.0, step=1.0, source="MiniDV / DVCAM / DVCPRO 机型",
               help="开启后画面走数字分量分支（4:1:1 子采样 + DCT 量化），不再经过复合视频载波。"
                    "MiniDV 机型如 Sony DCR-TRV310E 必须开启此项。")
    cod.define("chroma_mode", "色度子采样", 1.0, 0.0, 1.0, unit="bool", group=g,
               phys_min=0.0, phys_max=1.0, step=1.0, source="525/60 用 4:1:1，625/50 用 4:2:0",
               help="开=按制式选择（NTSC 4:1:1 / PAL 4:2:0），关=固定 4:1:1。"
                    "4:1:1 只损失横向色度分辨率，4:2:0 两个方向都损失。")
    cod.define("dv_quality", "DV 码率质量", 1.0, 0.25, 2.0, unit="x", group=g,
               phys_min=0.7, phys_max=1.2, source="DV25 固定 25 Mbit/s",
               help="1.0 为标准 DV25。调低等于码率更紧（块状伪影与蚊式噪声更重）。")
    cod.define("dv_base_step", "DV 量化基准", 0.0125, 0.002, 0.06, unit="0-1", group=g,
               source="DV 量化表量级",
               help="量化步长基准。暗部块状伪影的强度主要由它决定。")

    g_str = "网络视听传播、平台转码与流媒体协议"
    cod.define("stream_enabled", "启用网络流媒体与平台转码系统", 0.0, 0.0, 1.0, unit="bool", group=g_str, step=1.0,
               phys_min=0.0, phys_max=1.0, source="RealVideo / WMV / Web 2.0 FLV / HLS DASH 自适应流媒体模型",
               help="打通『广播电视 → 录像带 → 数字采集 → 早期流媒体/平台多代上传转码 → 分组网络传输 → 客户端缓冲与二次录屏』跨时代传播链路。")
    cod.define("stream_mode", "流媒体时代与协议模式", 0.0, 0.0, 12.0, unit="0-12", group=g_str, step=1.0,
               phys_min=0.0, phys_max=12.0, source="1999 RealVideo / 优酷土豆 2006–2015 / LDME 基准实验",
               help="0=跟随总开关｜1=早期拨号/宽带流媒体 (RealVideo G2 / WMV9 · 160x120~320x240 强环路滤波塑料涂抹与帧率抽帧)｜"
                    "2=Web 2.0 早期视频网站 (2006 360p FLV/H.263 · 4:2:0 粗量化 + 关键帧 GOP 呼吸脉冲 + 平台角标水印)｜"
                    "3=现代自适应流媒体 (HLS / DASH H.264 ABR · 720p↔240p↔1080p 阶跃切换与 TCP 预缓冲转圈卡顿 / UDP 丢包切片撕裂)｜"
                    "4=平台多代搬运重压与二次录屏 (代际色度模糊、块效应叠加、双水印与录屏摩尔纹)｜"
                    "5=2×2 跨时代网络流媒体对照实验室 (同屏对比：采集源母本 vs 1999 RealVideo vs 2007 平台多代转码 vs 现代 ABR/UDP 丢包 + 缓冲遥测)。")
    cod.define("stream_bitrate_kbps", "流媒体目标视频码率", 380.0, 16.0, 8000.0, unit="kbps", group=g_str,
               phys_min=28.0, phys_max=4500.0, source="拨号 28.8kbps RealVideo 至高清 H.264 码率包络")
    cod.define("stream_bw_fluctuation", "网络可用带宽波动幅度", 0.45, 0.0, 1.0, unit="0-1", group=g_str,
               phys_min=0.05, phys_max=0.85, source="TCP 锯齿拥塞窗口与无线信道吞吐起伏",
               help="控制分组网络吞吐率随时间波动的剧烈程度；波动过大将触发 ABR 分辨率阶跃降级或客户端缓冲区欠载转圈 (Rebuffering Stall)。")
    cod.define("stream_rtt_ms", "网络往返延迟 (RTT)", 85.0, 5.0, 800.0, unit="ms", group=g_str,
               phys_min=10.0, phys_max=350.0, source="跨洋/骨干网与移动网络 RTT 延迟")
    cod.define("stream_packet_loss", "分组丢包率 (Packet Loss)", 0.08, 0.0, 0.50, unit="0-0.5", group=g_str,
               phys_min=0.0, phys_max=0.25, source="UDP/RTP 实时流切片丢失与参考帧误差传播",
               help="在 UDP/RTP 流中引发水平宏块条带丢失与跨 P 帧运动向量绿/品红错位扩散；在 TCP/HLS 流中引发重传吞吐坍塌与缓冲卡顿。")
    cod.define("stream_gop_len", "关键帧间隔 (GOP 长度)", 15.0, 4.0, 120.0, unit="帧", group=g_str, step=1.0,
               phys_min=8.0, phys_max=60.0, source="帧间预测 P 帧误差累积与 I 帧全帧刷新周期",
               help="P 帧期间运动预测残差逐帧退化模糊，到达 I 帧瞬间画面突然清晰跳变（经典早期流媒体 GOP 呼吸脉冲效应）。")
    cod.define("stream_platform_generations", "平台搬运重编码代数", 2.0, 0.0, 5.0, unit="代", group=g_str, step=1.0,
               phys_min=0.0, phys_max=4.0, source="跨平台下载→二创剪辑→二次上传 CRF 重编码代际损失")
    cod.define("stream_quad_corner", "流媒体实验室画面角落选择 (0四分屏/1左上/2右上/3左下/4右下)", 0.0, 0.0, 4.0, unit="角落", group=g_str,
               step=1.0, phys_min=0.0, phys_max=4.0, source="2×2 四分屏单角全屏拆选",
               help="0=显示完整 2×2 四分屏对照；1=左上角（采集源/TBC母本）；2=右上角（1999 RealVideo/WMV）；3=左下角（Web 2.0 平台多代转码）；4=右下角（ABR 阶跃与网络缓冲遥测）。")
    cod.define("stream_show_border", "流媒体实验室分框样式 (0无分框/1细十字线/2标准分框)", 2.0, 0.0, 2.0, unit="分框", group=g_str,
               step=1.0, phys_min=0.0, phys_max=2.0, source="四分屏分框线显示控制")
    cod.define("stream_show_text", "流媒体实验室文字叠加 (0不要文字/1仅标签/2完整遥测)", 0.0, 0.0, 2.0, unit="文字", group=g_str,
               step=1.0, phys_min=0.0, phys_max=2.0, source="画面文字叠加开关（默认 0 纯净无字画面）")
    cod.define("ldme_transfer_enabled", "启用文件传输与多代压缩节点", 0.0, 0.0, 1.0, unit="bool", group=g_str, step=1.0,
               phys_min=0.0, phys_max=1.0, source="LDME 子系统 D (03 · 文件与传输)",
               help="模拟旧手机到电脑、彩信/社交软件与视频网站之间的文件传输与多代有损重编码历史。")
    cod.define("ldme_transfer_method", "文件传输与重编码方式", 0.0, 0.0, 5.0, unit="方式", group=g_str, step=1.0,
               phys_min=0.0, phys_max=5.0, source="蓝牙OBEX / USB导入 / 彩信社交压缩 / 桌面剪辑 / 多代JPEG磨损",
               help="0=直通｜1=蓝牙传输 (位级无损复制)｜2=USB/电脑导入 (保留像素)｜"
                    "3=彩信/早期社交软件发送 (强制缩放+重编码+剥离EXIF)｜4=电脑剪辑软件导出+多代视频转码 (V0→V1→V2→V3)｜"
                    "5=早期优酷土豆/论坛搬运多代 JPEG 磨损衰减 (2D 8×8 DCT、IJG Q表、网格错位与双重蚊式振铃噪点)。")
    cod.define("ldme_transfer_generations", "多代重编码级联次数", 2.0, 1.0, 5.0, unit="代", group=g_str, step=1.0,
               phys_min=1.0, phys_max=4.0, source="V_0 -> V_1 -> V_2 -> V_3 多代有损编码级联")
    cod.define("ldme_mms_width", "彩信/社交发送最大宽度", 320.0, 128.0, 1280.0, unit="px", group=g_str, step=16.0,
               phys_min=160.0, phys_max=640.0, source="MMS 100KB/300KB 网关分辨率限制")
    cod.define("ldme_mms_quality", "彩信/社交重编码品质", 52.0, 15.0, 95.0, unit="Q", group=g_str, step=1.0,
               phys_min=30.0, phys_max=80.0, source="应用层二次 JPEG/3GP 压缩品质")
    cod.define("ldme_nle_bitrate_kbps", "桌面剪辑软件导出码率", 420.0, 80.0, 4000.0, unit="kbps", group=g_str,
               phys_min=150.0, phys_max=2000.0, source="早期桌面剪辑软件导出低码率 MP4/WMV/AVI")
    cod.define("ldme_source_origin", "上传源文件清晰度来源", 0.0, 0.0, 1.0, unit="来源", group=g_str, step=1.0,
               phys_min=0.0, phys_max=1.0, source="区分『原本就是低清上传』与『高清源被平台降为低清』",
               help="0=高清源文件被平台缩放转码为低清｜1=上传时原本就是手机低清源文件（携带前置传感器噪点与早期块伪影进入平台转码器）。")
    cod.define("ldme_benchmark_variant", "LDME 基准实验子项选择", 1.0, 1.0, 6.0, unit="子项", group=g_str, step=1.0,
               phys_min=1.0, phys_max=6.0, source="LDME 第七节 F1 三组标准测试子项切换")
    cod.define("ldme_use_real_ffmpeg", "调用真实 FFmpeg 编解码器实验", 0.0, 0.0, 1.0, unit="bool", group=g_str, step=1.0,
               phys_min=0.0, phys_max=1.0, source="FFmpeg flv1 / h263p / mpeg4 / libx264 真实容器编解码")
    cod.define("ldme_jpeg_decay_quality", "JPEG 磨损基础量化品质", 75.0, 5.0, 95.0, unit="Q", step=1.0, group=g_str,
               phys_min=5.0, phys_max=95.0, source="标准 IJG 亮度与色度量化表品质因子 Q")
    cod.define("ldme_jpeg_decay_grid_shift", "跨代 8×8 DCT 网格错位漂移", 2.0, 0.0, 7.0, unit="px", step=1.0, group=g_str,
               phys_min=0.0, phys_max=7.0, source="跨代图像裁剪/缩放诱发 8×8 网格错位与双重蚊式振铃噪点")

    # ------------------------------------------------------------ RF 链路（无线电系统）
    rf = s.layer("rf")
    g = "发射与制式"
    rf.define("enabled", "启用独立 RF 无线电链路", 0.0, 0.0, 1.0, unit="bool", step=1.0, group=g,
              source="独立 RF 射频信号链（发射 -> 传播 -> 调谐 -> 解调）",
              help="在固定链路中开启独立无线电模拟电视接收子系统（节点图中的 RF 调制器/调谐器始终使用本层参数）。")
    rf.define("tv_standard", "模拟电视射频制式", 0.0, 0.0, 2.0, unit="enum", step=1.0, group=g,
              phys_min=0.0, phys_max=2.0, source="GB 3174 PAL-D (6.5MHz) / HK PAL-I (6.0MHz) / FCC M/NTSC (4.5MHz)",
              help="0=中国大陆 PAL-D/K (8MHz 频道, +6.5MHz 伴音)，1=香港/珠三角外溢 PAL-I (8MHz 频道, +6.0MHz 伴音)，2=NTSC-M (6MHz 频道, +4.5MHz 伴音)。")
    rf.define("channel_mhz", "频道带宽", 6.0, 5.0, 8.0, unit="MHz", group=g,
              phys_min=6.0, phys_max=8.0, source="M/NTSC 6 MHz，PAL-D/I 8 MHz",
              help="单个电视频道的宽度。决定了残留边带滤波与视频信息能占用的最高频率。")
    rf.define("carrier_offset_mhz", "图像载波位置", 1.25, 0.5, 2.5, unit="MHz", group=g,
              phys_min=1.0, phys_max=1.5, source="图像载波距频道下沿偏移")

    g = "大气波导与电离层传播 (TV-DX)"
    rf.define("propagation_mode", "无线电传播模式", 0.0, 0.0, 3.0, unit="enum", step=1.0, group=g,
              phys_min=0.0, phys_max=3.0, source="ITU-R P.452 对流层波导 / ITU-R P.534 电离层 Es 突发层",
              help="0=常规视距/绕射传播，1=对流层大气波导 (TV-DX 逆温层波导，令超视距远方频道比平时异常清晰)，"
                   "2=电离层活跃突发 E 层 (Sporadic-E 天波超远距传播，伴随明显多径与慢衰落)，"
                   "3=大气波导 + 电离层活跃复合传播。")
    rf.define("distance_km", "发射台距离", 45.0, 5.0, 1200.0, unit="km", group=g,
              phys_min=10.0, phys_max=800.0, source="ITU-R P.1546 地面广播传播距离",
              help="接收点与发射台之间的地面距离。超过视距（~55 km）后常规传播迅速衰减，而大气波导与 Es 电离层反射可在 150–800 km 维持高信噪比。")
    rf.define("signal_dbm", "基准接收电平", -70.0, -95.0, -30.0, unit="dBm", group=g,
              phys_min=-85.0, phys_max=-40.0, source="天线常规输入电平",
              help="无异常传播时的常规天线接收电平。远距离台站在常规状态下往往只有 −80 ~ −72 dBm（满屏雪花），"
                   "而在开启 TV-DX 大气波导或电离层活跃增益后会跃升至 −55 ~ −45 dBm，画面变得比一般时候清晰得多。")
    rf.define("dx_duct_gain_db", "TV-DX 波导/电离层异常增益", 0.0, 0.0, 36.0, unit="dB", group=g,
              phys_min=0.0, phys_max=32.0, source="华南沿海春夏逆温层大气波导 / 电离层 Es 层反射增益",
              help="当大气波导形成或电离层活跃时，原本散逸到太空中或被地球曲率遮挡的电磁波被波导捕获，"
                   "带来 15~30 dB 的额外路径增益，使平时满是雪花的远距弱台突然变得清晰锐利。")
    rf.define("qsb_depth_db", "QSB 慢衰落深度", 0.0, 0.0, 24.0, unit="dB", group=g,
              phys_min=0.0, phys_max=18.0, source="电离层电子密度起伏与波导湍流衰落",
              help="TV-DX 远距离接收特有的信号呼吸感：电离层云团或大气波导界面缓慢漂移引起的周期性电平起伏。")
    rf.define("qsb_rate_hz", "QSB 衰落速率", 0.35, 0.05, 2.5, unit="Hz", group=g,
              phys_min=0.1, phys_max=1.5, source="VHF/UHF 远距慢衰落周期 (0.1~1.5 Hz)")
    rf.define("multipath_level", "天波/波导多径重影强度", 0.0, 0.0, 0.85, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.65, source="r(t)=Σ a_k s(t-τ_k)exp(jφ_k)",
              help="多条波导路径或地面/海面反射造成的延迟副本。经解调后自然形成右侧重影与色度副载波相位干涉。")
    rf.define("multipath_delay_us", "多径相对延迟", 1.8, 0.2, 8.0, unit="μs", group=g,
              phys_min=0.4, phys_max=6.0, source="路径差 Δd = c·Δτ (1 μs ≈ 300 m 路径差)")
    rf.define("multipath_phase_deg", "多径载波初相", 35.0, -180.0, 180.0, unit="deg", group=g,
              phys_min=-180.0, phys_max=180.0, source="反射路径射频相位 φ_1")
    rf.define("multipath_drift_hz", "多径相位漂移率", 0.22, 0.0, 2.0, unit="Hz", group=g,
              phys_min=0.0, phys_max=1.2, source="电离层反射高度微变引起的多普勒相移",
              help="多径反射点移动导致重影极性（正像/负像）与彩色重影色调随时间缓慢交替。")

    g = "淘宝电视卡高频头与同邻频干扰"
    rf.define("tuner_type", "接收机/电视卡前端模型", 1.0, 0.0, 2.0, unit="enum", step=1.0, group=g,
              phys_min=0.0, phys_max=2.0, source="Philips SAA7130/7134 PCI 电视卡 / 硅高频头 USB 电视棒",
              help="0=标准电视调谐器，1=淘宝几十块廉价电视卡 (SAA713x + TDA9887：窄锯齿声表面波中频滤波、群延时振铃、阶梯 AGC 与高色度串扰)，2=高选择性 DX 接收机。")
    rf.define("noise_figure_db", "高频头噪声系数", 8.0, 2.0, 20.0, unit="dB", group=g,
              phys_min=4.0, phys_max=14.0, source="消费级硅高频头 / 铁壳高频头 NF",
              help="廉价电视卡高频头通常在 8~11 dB 左右，且容易串入 USB/PCI 总线底噪。")
    rf.define("tuning_offset_khz", "本振调谐频偏", 0.0, -250.0, 250.0, unit="kHz", group=g,
              phys_min=-150.0, phys_max=150.0, source="廉价电视卡步进锁相环 (PLL) 频偏 / 奈奎斯特斜坡失配",
              help="负向频偏会切掉高频细节与色度副载波；正向频偏会令高频边缘过冲振铃并加重伴音载波串扰。")
    rf.define("if_saw_ringing", "中频 SAW 滤波器振铃", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.75, source="廉价声表面波滤波器 (SAW) 群延时起伏与驻波",
              help="淘宝电视卡标志性的硬边缘右侧明暗回波（台标与字幕右侧的紧邻轮廓伴影）。")
    rf.define("front_end_overload", "前端过载与三阶互调", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.8, source="LNA 1dB 压缩点与混频器 IM3 交调",
              help="当大气波导使远距信号或本地强台过强时，廉价电视卡 LNA 进入非线性区，导致高光压缩与细密交调网纹。")
    rf.define("cochannel_interference", "同频干扰（百叶窗效应）", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6, source="TV-DX 远距离同频道第二个电台载波差拍",
              help="大气波导常把几百公里外的同频电台一并送入天线，两个图像载波的微小频差形成水平百叶窗明暗横纹。")
    rf.define("cochannel_offset_khz", "同频台载波偏置", 10.0, 0.5, 25.0, unit="kHz", group=g,
              phys_min=1.0, phys_max=20.0, source="精密偏置电视发射机行频倍数差 (10 kHz / 15.625 kHz)")
    rf.define("adjacent_channel", "邻频干扰与伴音差拍", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_max=0.6, source="相邻频道载波与本频道图像/伴音差拍",
              help="廉价电视卡邻频抑制不足时，斜向游走的人字形/网状差拍条纹。")
    rf.define("sync_loss_threshold_db", "行同步失锁阈值", 30.0, 18.0, 44.0, unit="dB", group=g,
              phys_min=22.0, phys_max=38.0, source="行 AFC 保持范围")
    rf.define("sync_loss_amount", "失锁撕裂强度", 1.0, 0.0, 2.0, unit="0-2", group=g,
              phys_min=0.5, phys_max=1.5)
    rf.define("color_killer_threshold_db", "消色器门限", 21.0, 10.0, 32.0, unit="dB", group=g,
              phys_min=16.0, phys_max=26.0, source="PAL/NTSC 色同步锁定与自动消色器 (Color Killer)",
              help="当衰落谷底 C/N 低于此门限时，解码芯片自动切断色度通道以防满屏彩雪，画面短暂退化为黑白。")

    g = "换台瞬态与模拟恐怖失锁 (松下 NV-J27 高频头实测标定)"
    rf.define("channel_switch_rate", "换台切台触发频率", 0.0, 0.0, 2.0, unit="次/秒", group=g,
              phys_min=0.0, phys_max=1.2, source="松下 NV-J27 变容二极管调谐换台瞬态实测标定",
              help="模拟电视或录像机高频头换台（或模拟恐怖频道劫持切台）的事件发生率。"
                   "每次换台会依次经历：本振扫频穿越邻频/奈奎斯特斜坡 → 中频无载波满增益宽带雪花爆发 → "
                   "耦合电容直流恢复充放电闪白/黑屏 → 行场同步与色同步重新拉入锁定。")
    rf.define("channel_switch_dur_s", "单次换台失锁持续时长", 0.42, 0.08, 1.80, unit="s", group=g,
              phys_min=0.15, phys_max=0.90, source="变容二极管 PLL 锁相环与 AGC 时间常数 (实测 0.25~0.65s)",
              help="单次换台从本振脱锁、雪花爆发到新频道行场同步重新锁定的持续时间。")
    rf.define("lo_sweep_mhz", "本振扫频与负像浮雕反相", 0.0, 0.0, 1.5, unit="0-1", group=g,
              phys_min=0.0, phys_max=1.0, source="SAW 声表面波滤波器 38.0MHz 奈奎斯特斜坡正交失真与包络过调制反相",
              help="换台瞬间变容二极管电压跳变使本振扫过中频 SAW 滤波器边缘：残留边带正交失真产生强烈的"
                   "高对比微分边缘浮雕感，且图像载波跌落导致包络检波器跨过零点发生明暗负片反相（暗部变亮白、高光变死黑）。")
    rf.define("vsync_roll", "场同步失锁垂直滚屏", 0.0, 0.0, 1.5, unit="0-1", group=g,
              phys_min=0.0, phys_max=1.0, source="50Hz 场同步积分器失步与自由振荡频差 Δf_v",
              help="换台或弱信号下场同步脉冲被雪花淹没，50Hz 场振荡器自由运行导致画面整场上下翻滚。")
    rf.define("vbi_bar_vis", "场消隐黑带 (VBI 锤头同步带)", 0.65, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.2, phys_max=1.0, source="625行 PAL 场消隐期 25 行结构 (前均衡+场同步宽脉冲+VITC/图文亮斑)",
              help="当发生垂直滚屏时，原本藏在屏幕上下边框外的 25 行场消隐带（含半行均衡脉冲缺口与场逆程图文/时间码白点）"
                   "滚入可见画面中央，是模拟恐怖换台最核心的真实物理标志。")
    rf.define("flagging", "行 AFC 顶部旗帜弯折", 0.0, 0.0, 1.5, unit="0-1", group=g,
              phys_min=0.0, phys_max=1.0, source="松下 NV-J27 磁头切换瞬态与电视机行 AFC 时间常数失配",
              help="场消隐或磁头切换后行同步相位跳变，行 AFC 锁相环在画面顶部 15~35 行内按指数曲线拉入锁定，"
                   "导致画面顶部像迎风飘扬的旗帜一样向一侧剧烈歪斜。")
    rf.define("hanover_bars", "PAL 汉诺威盲窗错色条", 0.0, 0.0, 1.5, unit="0-1", group=g,
              phys_min=0.0, phys_max=1.0, source="PAL 逐行倒相 (V轴 ±90°) 色同步摆动门失步与 1H 延迟线相位失配",
              help="PAL-D 解码器在换台或弱信号下色同步相位误差未通过 1H 延迟线抵消，导致奇数行与偶数行出现"
                   "饱和度极高、色调互补交替的 2 行周期水平盲窗彩条（Hanover Bars）。")

    # ------------------------------------------------------------------ C
    tpe = s.layer("tape")
    g = "录制速度"
    tpe.define("tape_speed", "录制速度", 0.0, 0.0, 2.0, unit="0-2", group=g,
               step=1.0, source="VHS 标准 SP/LP/EP（同机同一盘磁带的速度档）",
               help="0=SP 标准｜1=LP 长时｜2=EP 超长。注意这不是「把质量调低」："
                    "磁鼓转速不变，所以写入的 FM 偏离几乎不变，下降的是走带速度。"
                    "由此同时产生四种后果——重放输出下降（LP 约 −6 dB、EP 约 −10 dB）、"
                    "磁迹变窄使失落概率上升（约 2×/3×）、保护带变窄使跟踪敏感度剧增"
                    "（约 2.2×/3.4×）、控制轨迹信号变弱使时基更不稳（约 1.9×/2.8×）。"
                    "色度欠载方式不变，所以色度带宽不随速度改变——这也是为什么"
                    "LP/EP 素材里亮度电平基本不变，坏的是色度分层与稳定度。")
    tpe.define("sp_lp_mix", "SP/LP 混合交界包络", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.8,
               source="COSMAG E-180（180分 SP / 360分 LP 混合录制）+ 夏普 VC-A62DT 4视频磁头切换",
               help="同一盘磁带上 SP（49 μm 磁迹）与 LP（24.5 μm 磁迹）混合录制或跨档自动切换时，"
                    "4 视频磁头（2×SP + 2×LP）读取窄保护带磁迹产生的 FM 包络起伏与 629 kHz 色度相位颤动。")
    g = "走带与时基误差"
    tpe.define("tape_stability", "走带稳定性", 0.55, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0, source="磁带盒与走带机构机械状态",
               help="1.0 为机械完美。这是逐行水平位移的主要来源。")
    tpe.define("line_jitter", "行抖动", 0.35, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="行频相位抖动",
               help="每行独立的水平位移，造成竖线边缘的锯齿状波动。")
    tpe.define("walk_amplitude", "低频漂移", 0.30, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="走带速度缓慢波动",
               help="缓慢相关的过程，造成整幅画面缓慢的水平摆动。")
    tpe.define("micro_jitter", "微小抖动", 0.20, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.5, source="磁鼓相位噪声",
               help="亚像素级的行内不齐，由插值相位误差表现。")
    tpe.define("vertical_instability", "垂直不稳定", 0.08, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.4, source="场同步裕度不足",
               help="整场的垂直跳动，偶发出现。")
    g = "磁头切换"
    g = "走带模式（快进/倒带）"
    g = "磁粉与霉变"
    tpe.define("binder_loss", "黏合剂失效（霉变/脱粉）", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.7, source="潮湿存放导致黏合剂吸水、磁粉层脱落",
               help="磁粉层脱落的面积比例。这不是「脏」而是**涂层本身缺失**，"
                    "所以有两个复合后果：受损区域固定在磁带表面、反复出现在同一位置"
                    "（不像噪声那样每帧重掷）；且**色度先死**——色度是低幅度的欠载"
                    "载波，骑在亮度上，涂层变薄时它先消失而亮度还能撑一阵。"
                    "参考素材的「发霉」样本实测饱和度崩到 0.053、高频升高 7 倍、"
                    "38% 画面接近全黑——是「高频且暗」，不是单纯变糊。")
    g = "涂层脱落"
    tpe.define("shed_grain", "脱粉颗粒", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.7, source="参考素材发霉格：高频为正常格的 3 倍",
               help="磁粉脱落使磁头读到更粗糙的表面，重放信号带有宽带颗粒。"
                    "它与「粒子噪声底」的区别有两点：**强得多且更细**"
                    "（参考发霉格高频 0.116 而正常格 0.039；噪声底拉满只能到 0.035，"
                    "差六倍，所以那种样子根本做不出来），以及**只作用于亮度**——"
                    "涂层脱落会直接毁掉微弱的色度载波，颗粒若带上彩色就错了。"
                    "它与「色度增益」「消隐电平」共同构成发霉格的三要素："
                    "去饱和 + 变暗 + 高频颗粒。")
    g = "存档带物理失效（COSMAG E-180 标定）"
    tpe.define("demagnetization", "消磁（剩磁衰减）", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.9,
               source="COSMAG E-180 存档带采录失败段（消磁）：Wallace 剩磁损耗与色度消隐",
               help="磁粉层剩磁 M_r 随岁月或外部磁场消退。按 Wallace 电磁感应定律，高频 FM 亮度载波"
                    "（3.8–4.8 MHz）与微弱的 629 kHz 欠载色度先跌破限幅器与 ACC 阈值，"
                    "导致画面细节发虚、反差坍塌、伴随限幅器雪花噪波并触发色度消隐（去饱和）。")
    tpe.define("base_cracking", "带基龟裂（PET 微裂纹）", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.9,
               source="COSMAG E-180 存档带采录失败段（带基龟裂）：磁头掠过微裂纹的间距损耗",
               help="老化聚酯（PET）带基产生横跨 5.9° 螺旋磁迹的脆性微裂纹网。当夏普 VC-A62DT 的 4 颗"
                    "视频磁头高速掠过裂纹台阶时，磁头-磁带间距 d 瞬间跳变（54.6·d/λ dB 间距损耗），"
                    "形成斜向/横向双极性 FM 失锁黑白尖峰裂纹与局部行相位滑移。")
    tpe.define("base_breakage", "带基破损（折损/断带失步）", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.9,
               source="COSMAG E-180 存档带采录失败段（带基破损）：下沿 1×时间轴 CTL 磁头失锁",
               help="带基边缘破损、撕裂或接带折痕同时破坏螺旋视频轨与磁带下沿 25 Hz 时间轴控制轨（CTL）。"
                    "夏普 VC-A62DT 的 1 颗时间轴磁头失去 CTL 脉冲后，主导轴伺服瞬间失步，"
                    "在破损横带处产生剧烈的行同步剪切、场跳动与全宽原生 FM 雪花。")

    g = "多世代翻录损耗（VHS 反复迭代标定）"
    tpe.define("generation_passes", "VHS 世代翻录次数", 1.0, 1.0, 25.0, unit="代", group=g,
               step=1.0, phys_min=1.0, phys_max=20.0,
               source="多世代 VHS 连续翻录实验标定（第 1 代 → 第 20 代）",
               help="模拟磁带对磁带（VCR → VCR）反复翻录迭代的世代数。1 代为母带直出；"
                    "随着世代数增加（如 5 代、8 代、15 代、18 代），每次翻录都会累积："
                    "① 629 kHz 欠载色度低通与 1H 梳状滤波导致的色度向右下方跨界偏移与洇色（每代约 +1.35 px / +1.15 行）；"
                    "② FM 预加重与去加重失配导致的垂直边缘多级黑白振铃光环（Zebra Ringing）与高光黄绿偏色；"
                    "③ 行时基抖动按 √N 累积放大、AGC 黑电平抬升、右侧 HBI 钳位噪波边加宽及底部多层磁头切换带堆叠；"
                    "④ 约第 19–20 代时行同步脉冲圆化跌破 AFC 门限，触发消色器（Color Killer）切入黑白高反差斜纹失锁状态。")
    tpe.define("gen_chroma_bleed", "世代色度偏移与下垂洇色", 0.85, 0.0, 2.0, unit="0-2", group=g,
               phys_min=0.3, phys_max=1.5,
               source="VHS 629 kHz 欠载色度群延迟 + 1H 玻璃延迟线逐代级联",
               help="控制每经过一代翻录时，色度平面相对亮度的向右群延迟偏移、向下行平均下垂及色度饱和度滚降强度。")
    tpe.define("gen_ringing", "世代孔径过冲与多级振铃黑边", 0.85, 0.0, 2.0, unit="0-2", group=g,
               phys_min=0.3, phys_max=1.5,
               source="VHS 录放高频预加重/孔径补偿级联二阶过冲（实测 Gen 18 边缘 hf_x 升至 2.6 倍）",
               help="控制每代翻录在人物轮廓、条纹与明暗硬边界两侧激发出的一阶/二阶黑白振铃条纹（Ringing Halo）及右侧 HBI 噪波边强度。")
    tpe.define("gen_sync_collapse", "世代同步崩溃与消色门限", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               source="实拍第 19–20 代（60.5s–66.5s）行同步圆化失锁与 Color Killer 动作",
               help="当世代数达到 19–20 代或手动调高此项时，模拟行同步与色同步彻底崩溃：自动消色器关闭彩色、"
                    "画面陷入黑白高反差 FM 差拍斜纹与剧烈行撕裂，仅隐约保留人物剪影。")

    g = "亮度电平"
    tpe.define("luma_pedestal", "消隐电平偏移", 0.0, -0.5, 0.3, unit="0-1", group=g,
               phys_max=0.05, phys_min=-0.35,
               source="直流恢复/消隐钳位失常（S-VHS 劣化带实测约 -0.30）",
               help="亮度基底（消隐电平）的整体偏移。它与「磁粉脱落」是两回事："
                    "亮度是**基带电平**，绝对值直接承载明暗，必须钳位到参考；"
                    "色度是**围绕该参考的调制幅度**，参考错了它照样活着。"
                    "所以信号丢失让画面**又暗又灰**（亮度色度一起没），"
                    "而钳位失效让画面**又暗又极其鲜艳**——"
                    "参考素材实测饱和度升到 0.94、均值掉到 0.09。")
    tpe.define("drum_speed", "磁鼓转速", 1.0, 0.80, 1.05, unit="x", group=g,
               phys_min=0.97, phys_max=1.03,
               source="松下肩扛式 VHS / VHS-C 磁鼓轴承油变质（十台九台）",
               help="磁鼓额定转速的比值。注意它与走带误差是**不同轴**的问题："
                    "走带（主导轴）误差让画面横向抖动，磁鼓转速误差改变磁头"
                    "向下扫过磁迹的快慢——鼓慢了，相邻输出行就取自磁带上几乎"
                    "相同的位置，画面出现**微量重复的行**。垂直方向被平滑、"
                    "水平方向不变，这正是「横向颗粒」的来源，而不是随机噪声。"
                    "继续使用会导致轴承不可逆磨损。")

    g = "清洁度与污染"
    tpe.define("head_clog", "磁头堵塞", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="磁头积垢",
               help="堵塞的磁头拖出一条固定的丢失信号带，画面出现起格或条纹。")
    tpe.define("clog_bands", "堵塞条带数", 3.0, 1.0, 8.0, unit="条", group=g, step=1.0)
    tpe.define("debris", "磁粉颗粒", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="脱落的磁粉与灰尘",
               help="松散颗粒短暂撞击磁带，形成断续的短促丢失。")
    tpe.define("surface_damage", "表面划伤", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="走带划伤",
               help="划伤损坏一条窄的、缓慢漂移的行带。")
    tpe.define("chroma_loss", "色度丢失行", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.4, source="色度欠载载波弱",
               help="随机整行丢失彩色（色度载波弱于亮度）。")

    tpe.define("row_timebase", "逐行时基", 0.0, 0.0, 3.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.5,
               source="未标定。曾引用 JVC SR-S388E 素材的三个读数，该证据已撤回（见 help）",
               help="整行而非整场的水平位移，横向的「撕裂」感来源。"
                    "**量级未标定，且这不是遗漏而是结论。** 曾据 JVC SR-S388E 采集的三个读数"
                    "（t=1000s 时 std 0.19 px、1380s 时 0.46、2000s 时 1.03）给出 0.02/0.04/0.08，"
                    "但同一素材按每 5 秒重采样后发现：**相邻采样的逐行散布在约 0.1 与 2.9 px 之间"
                    "以 6x、12x、14x 的幅度翻转，且与画面内容无关**。具有这种行为的量无法用三个"
                    "读数概括，每个读数都只是一个闪烁信号的采样。所用追踪器后来也被测出非线性"
                    "（平滑场恢复 0.47x、快变场 3.11x），所以它给出的数字本来就不是绝对像素位移。"
                    "**因此本参数保持 0.0，并且在拿到状态稳定的素材之前不应标定**——"
                    "素材显示的是**间歇性**，那由 stiction_rate 单独建模。"
                    "与行抖动不同：行抖动是每行独立，这里是整行整体移动。")
    tpe.define("stripe_moire", "卷曲条纹强度", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.6,
               source="残余载波与视频通路的差频干涉（偏磁/伴音副载波泄漏、邻频、时钟不同步）",
               help="沿画面弯曲的条纹。它与「整行时基误差」是**不同现象**，这一点是量出来的："
                    "逐行位移让跨行相位变化达到完好基线的 3990 倍，"
                    "但**行内曲率仍停在地板**（0.00001 rad/px²）——"
                    "位移只是把内容横向搬走，不改变空间频率，所以直线条纹依然是直线。"
                    "要让条纹弯曲必须有**差频**：残余载波与视频通路频率略有差异，"
                    "两者相位差随位置变化，条纹因此弯曲。")
    tpe.define("stripe_beat", "条纹间距", 0.055, 0.01, 0.25, unit="cyc/px", group=g,
               phys_min=0.02, phys_max=0.12,
               help="差频决定的条纹疏密。数值越小条纹越宽。")
    tpe.define("stripe_curve", "条纹弯曲程度", 0.35, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.8,
               help="条纹沿画面方向的弯曲量。由相位差随位置变化的快慢决定；"
                    "取 0 得到一组平行的直条纹，与「整行位移」外观不同。")

    g = "磁带仓受潮（粘带）"
    tpe.define("stiction_rate", "粘带发生率", 0.0, 0.0, 4.0, unit="次/秒", group=g,
               phys_min=0.0, phys_max=1.2,
               source="实测：相邻场逐行位移在 0.1 与 3 px 之间以数十倍幅度翻转，"
                      "且与画面内容无关",
               help="磁带黏合剂吸水，在主导轴与压带轮之间形成水膜，产生黏附。"
                    "走带机构继续拉，张力升高到黏附突然失效，磁带猛跳一下，然后再粘住。"
                    "**这是间歇性故障**，不是平滑劣化：事件之间磁带表现正常。"
                    "模型里它是一个**事件列表**（和失落、折痕一样），"
                    "而不是逐帧场——把场平滑掉恰好会抹掉让它看起来像故障的那些突变。")
    tpe.define("stiction_amount", "粘带强度", 0.6, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.8,
               help="每次黏附的强度。位移在抓取过程中**渐增**、释放时**回弹**，"
                    "所以是锯齿波——不是阶跃（那会像接头），也不是正弦（那会像抖动）。"
                    "空间上位移呈斜坡，所以受影响的行是渐变的，而不是一条整齐的裂口。")
    tpe.define("stiction_stretch", "沿带加重", 0.0, -0.8, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.8,
               help="发生率沿带长的偏移。湿气分布不均、靠近卷芯的部分更潮，"
                    "而且播放最多的位置磨损最重，所以同一盘带前后段的故障密度不同。"
                    "注意这影响的是**事件发生率**而不是每帧幅度——故障在事件里，不在信号里。")
    g = "整行色度错误"
    tpe.define("row_chroma_error", "整行错色发生率", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               source="熟悉设备的用户描述：解析出错误的颜色，而且是整行全错",
               help="整条扫描线解出**错误的颜色**——不是没有颜色。"
                    "它与「色度丢失行」是两回事：那个把色度清零，得到灰线；"
                    "这个旋转色相并改变色度增益，线仍然是饱和的，只是颜色错了。"
                    "机制上这是**参考错误**：色度副载波的相位由每行起始的色同步恢复，"
                    "所以某一行的参考错了，会把那一行上**所有**颜色旋转同一个角度，"
                    "而不是各自独立地坏掉。")
    tpe.define("row_chroma_phase", "整行色相误差", 0.0, 0.0, 90.0, unit="deg", group=g,
               phys_min=0.0, phys_max=45.0,
               help="受影响行的色相旋转幅度。因为它是旋转而不是缩放，"
                    "饱和度得以保留——这正是「错色」与「去色」在画面上的区别。")
    tpe.define("row_chroma_gain", "整行色度增益误差", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="受影响行的色度幅度误差。色同步幅度经 ACC 环路决定色度增益，"
                    "所以弱的色同步会让整行过饱和或欠饱和。与色相误差相互独立。")
    g = "失落与磁头切换"
    tpe.define("dropout_rate", "失落率", 0.35, 0.0, 4.0, unit="次/秒", group=g,
               phys_max=1.5, source="磁粉脱落点的密度",
               help="Poisson 到达过程。每次失落都有确定的位置与长度，"
                    "因此相同种子会重现完全相同的缺陷时间线。")
    tpe.define("dropout_len", "失落长度", 2.5, 0.5, 30.0, unit="行", group=g,
               phys_max=8.0)
    tpe.define("dropout_color", "失落补色残留", 0.45, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, source="失落补偿电路残留")
    tpe.define("dropout_softness", "失落边缘柔和度", 0.35, 0.0, 1.0, unit="0-1", group=g)
    tpe.define("doc_hold", "1H 延迟线失落补偿与彗星尾", 0.55, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.2, phys_max=0.9, source="松下 NV-J27 1H 延迟线 DOC 失落补偿器与 FM 限幅器失锁尖峰",
               help="真实 VHS 录像机在磁粉脱落处的物理行为：并非单纯叠加灰白噪声，而是前端 FM 限幅器在包络跌落瞬间"
                    "先爆发出极亮/极黑的零交越尖峰并向右拖出 RC 充放电彗星尾，随后包络检波器触发开关，强行复制"
                    "上一扫描行 (y-1) 的无彩色基带信号进行修补。")
    tpe.define("edit_glitch_rate", "暂停/覆录拼接故障率", 0.0, 0.0, 2.0, unit="次/秒", group=g,
               phys_min=0.0, phys_max=0.8, source="松下 NV-J27 无飞抹磁头覆录拼接与 25Hz CTL 控制轨跳变",
               help="模拟恐怖视频中常见的分镜突变：消费级录像机暂停后再录或换台录制时，新旧螺旋磁迹重叠（无飞抹磁头）"
                    "且下沿 25Hz CTL 控制脉冲相位断裂，引发走带主导轴伺服短暂猎振、底部切换带向上蹿升与局部雪花切口。")
    tpe.define("edit_glitch_severity", "拼接伺服失锁强度", 0.70, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.2, phys_max=1.0, source="主导轴伺服猎振包络与双磁迹差拍",
               help="每次暂停/覆录拼接事件中 CTL 失锁导致的场跳动、行撕裂与跨轨雪花带的剧烈程度。")
    tpe.define("head_switching", "磁头切换强度", 0.65, 0.0, 1.2, unit="0-1", group=g,
               phys_max=0.8, source="IEC 60774-1 / SMPTE 170M 旋转磁鼓双磁头切换瞬态",
               help="画面底部的磁头切换细条（含 1–2 行 RF 切换接缝脉冲与行 AFC 锁相环指数回拉钩状偏移）。"
                    "0.5–0.7 为标准家用机切换痕迹，超过 0.9 为重度包络失锁。")
    tpe.define("head_switch_lines", "切换带行数", 6.0, 2.0, 40.0, unit="行/场", group=g,
               phys_min=4.0, phys_max=10.0, source="IEC 60774-1 规定场同步前 6.5±1.5 H 切换（约 5–8 行/场）",
               help="每场末尾距垂直消隐的切换行数。标准 VHS 规定在 V-sync 前 6.5 H 切换（6 行/场 → 隔行交织后"
                    "在 480 行画面底部占约 11 行 ≈ 2.3%）。")
    tpe.define("head_switch_mid_line", "行中途切换", 0.25, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, source="切换点落在行中间的情况")

    g = "介质磨损"
    tpe.define("wear", "磁带磨损", 0.25, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.7, source="磁带表面磨损与走带次数",
               help="磨损使磁介质输出下降、表面变粗糙，亮度与色度信噪比一起变差。")
    tpe.define("crease_rate", "折痕发生率", 0.20, 0.0, 3.0, unit="次/秒", group=g,
               phys_max=1.0, source="磁带折痕与走带受压",
               help="折痕降低磁头与磁带的接触，所以它同时软化并削弱亮度与色度。")
    tpe.define("noise_floor", "粒子噪声底", 0.18, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.5, source="磁介质粒子噪声",
               help="磁介质本身带来的宽带噪声。注意它与「脱粉颗粒」不同：这个是一般"
                    "粒子噪声底，强度上限只有脱粉颗粒的约六分之一。")

    g = "跟踪失谐"
    tpe.define("tracking_error", "跟踪误差", 0.22, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, source="25/30Hz CTL 控制磁迹与磁鼓切换点的相位差",
               help="录像机前面板「Tracking（循迹）」旋钮对应的相位偏移：决定磁头切换接缝处的初始行相位跳变幅度与 H-AFC 回拉斜率。")
    tpe.define("tracking_noise_rows", "失控行数", 0.0, 0.0, 240.0, unit="行", group=g,
               phys_min=0.0, phys_max=120.0, source="控制轨迹与磁头不同步的深度",
               help="底部有多少行失去同步。与磁头切换带的区别：切换带是固定硬边"
                    "细条（约 8 行），跟踪失谐是从某行开始向下渐强的斜坡。")
    tpe.define("tracking_noise_wave", "失控摆动", 0.0, 0.0, 6.0, unit="px", group=g,
               phys_min=0.0, phys_max=3.0, help="失控行在底部的最大横向位移。")
    tpe.define("tracking_noise_snow", "失控噪点", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, help="失控区的雪花与彩色丢失，越靠下越重。")

    g = "走带模式（快进/倒带）"
    tpe.define("shuttle_speed", "走带倍速", 0.0, 0.0, 8.0, unit="x", group=g,
               phys_min=0.0, phys_max=5.0, step=0.5,
               source="NV-GS 系列倒带/快进搜索速度",
               help="0 = 正常播放。非零时磁头掠过磁迹的速度与记录速率不符，"
                    "只能取到部分宏块/扫描线，由解码器或时基电路用缓冲内容补齐。")
    tpe.define("shuttle_stale", "残留比例", 0.0, 0.0, 0.9, unit="比例", group=g,
               phys_max=0.5, source="参考素材实测 15-24%（均值约 19%）")
    tpe.define("shuttle_cluster", "残留成片程度", 0.55, 0.0, 1.0, unit="0-1", group=g)
    tpe.define("shuttle_carry", "残留保持率", 0.45, 0.0, 0.95, unit="0-1", group=g)
    tpe.define("shuttle_mode", "残留粒度", 0.0, 0.0, 2.0, unit="0-2", group=g, step=1.0,
               source="VHS/Hi8/Video8 为行记录；MiniDV 为宏块",
               help="0=自动（跟随编码层）|1=强制按行|2=强制按宏块。这不是偏好而是"
                    "格式属性：模拟格式沿磁迹记录连续波形，没有可寻址的块结构。")
    tpe.define("shuttle_span", "残留段长", 8.0, 1.0, 60.0, unit="行/段", group=g,
               phys_min=2.0, phys_max=30.0, step=1.0)
    tpe.define("shuttle_wave", "残留行位移", 14.0, 0.0, 60.0, unit="px", group=g,
               phys_min=0.0, phys_max=30.0,
               help="失效行的横向位移，在一段内逐行递增——台阶状边缘的来源。")
    tpe.define("shuttle_reverse", "方向（0 快进/1 倒带）", 0.0, 0.0, 1.0, unit="0/1",
               group=g, step=1.0)
    tpe.define("shuttle_block", "宏块边长", 16.0, 8.0, 32.0, unit="px", group=g,
               step=8.0, phys_min=16.0, phys_max=16.0, source="DV 宏块为 16x16 亮度像素")
    tpe.define("shuttle_guard_bars", "跨轨噪波条（Cue/Review 搜索）", 0.0, 0.0, 1.0, unit="0-1",
               group=g, phys_min=0.0, phys_max=1.0,
               source="螺旋扫描磁鼓倍速搜索 RF 包络周期塌陷实测标定",
               help="模拟磁带倍速快进/倒带搜索（Cue/Review）时，旋转磁头斜跨 |N-1| 条磁迹，"
                    "RF 包络周期性穿过相邻磁迹保护带/方位角零点，产生垂直等距向下滚动的"
                    "双峰解调雪花横条、行同步黑槽切口与行剪切。")
    tpe.define("shuttle_bar_luma", "噪波条灰白平台电平", 0.565, 0.20, 0.95, unit="IRE",
               group=g, phys_min=0.45, phys_max=0.82,
               source="FM 限幅器失锁零交越自激电平（5x 搜索实测 0.562，9x 搜索实测 0.750）",
               help="RF 包络跌破 FM 限幅器门限时，解调器锁定在限幅器自激中值形成的灰白横条亮度。")
    tpe.define("shuttle_PIP_bars", "画中画子窗轨数倍频", 0.0, 0.0, 1.0, unit="bool",
               group=g, step=1.0, phys_min=0.0, phys_max=1.0,
               source="切换台上半幅子窗（Y ∈ [0, 0.562H]，9x 倍速 8 条主+副噪波条）",
               help="开启后仅在画面上半幅子窗（Y ∈ [0, 0.562H]）施加约 9x 倍速的密集跨轨噪波条"
                    "（周期 129.3 px），下半幅保持正常重放（对应模拟切换台分屏子窗合成结构）。")

    g = "机械传动与伺服状态机 (Transport & Servo)"
    tpe.define("transport_state", "走带操作状态机", 0.0, 0.0, 4.0, unit="0-4", group=g, step=1.0,
               phys_min=0.0, phys_max=4.0,
               source="VHS 走带机构与主导轴/磁鼓双闭环伺服状态机 (IEC 60774-1 / 松下 NV-J27 机械手册)",
               help="0 = 正常播放 (Play)｜1 = 静态暂停定格 (Pause/Still：双磁头在 5°56' 静止扫描角下跨越相邻磁迹保护带，产生固定双峰雪花横条与奇偶场微颤，伴音静音)｜"
                    "2 = 压带轮啮合与 CTL 重锁定瞬态 (Play↔Pause Transient：压带轮啮合起动、主导轴加减速与 25/30Hz CTL 控制脉冲二阶欠阻尼重锁定过程)｜"
                    "3 = 无飞抹磁头暂停起录切口 (Record-Pause Assembly Cut：旧磁迹未抹除斜向叠印 + 180° 色度副载波彩虹干涉带 + CTL 脉冲断裂纵向跳帧)｜"
                    "4 = 停机卸带瞬态 (Stop/Unthread：导带柱退回、磁鼓包角坍塌引起的上下雪花帷幕与行失锁)。")
    tpe.define("reel_progress", "带盘进度位置 (带头0→带尾1)", 0.50, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               source="VHS E-180 磁带盒供带盘 R_s(p) 与收带盘 R_t(p) 半径连续演化 (R_hub=13mm, R_max=41mm)",
               help="0.0 = 带头 (BOT，供带盘满卷 R_s=41mm、收带盘空卷 R_t=13mm，高后张力导致画面顶部向右弯钩 Skew)；"
                    "0.50 = 中段平衡区（供收带盘半径相等 R_s=R_t≈29.8mm，走带最稳）；"
                    "1.0 = 带尾 (EOT，供带盘接近卷芯 R_s=13mm、角速度飙升 3.15 倍，低背张力引发画面顶部向左歪斜与底部导带柱卷边)。")
    tpe.define("tension_ripple", "供收带盘张力纹波与卷边", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               source="张力杆弹簧-阻尼谐振 (f_pole) 与带盘偏心扭矩脉动 (ω_s = v_tape / R_s(p))",
               help="控制带盘旋转偏心扭矩脉动、张力杆弹簧谐振以及带头/带尾边缘卷曲褶皱 (Edge Curl) 的耦合强度。")
    tpe.define("belt_slip", "方形橡胶皮带老化打滑", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               source="主导轴飞轮氯丁橡胶方皮带松弛老化与粘滑掉速",
               help="老式录像机橡胶传动皮带松弛硬化后，在负载峰值处发生间歇性指数打滑掉速与回弹：导致走带速度下坠、底部循迹雪花带间歇蹿升，并引发伴音音高同步下坠。")
    tpe.define("roller_eccentricity", "压带轮硬化偏心抖晃", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               source="合成橡胶压带轮 (D≈14.6mm) 长期压迫变形引起的 ~1.45Hz (SP) 周期性摆动",
               help="压带轮表面硬化或压出凹痕后，每转一周产生一次周期性线速度调制（SP 下约 1.45Hz 及二次谐波），造成整幅画面平滑的 S 形横向呼吸摆动与伴音周期性低频抖晃 (Wow)。")
    tpe.define("servo_hunt", "主导轴/磁鼓伺服猎振", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.9,
               source="主导轴 FG/PG 锁相环与 25/30Hz CTL 相位比较器欠阻尼猎振 (~3.2Hz)",
               help="伺服环路滤波电容老化导致相位裕度下降，在 3.25Hz 附近产生持续的欠阻尼正弦猎振：画面左右周期性摆动，同时磁头切换带与跨轨雪花条上下呼吸起伏。")
    tpe.define("servo_lock_tau", "伺服重锁定时间常数", 0.45, 0.08, 2.0, unit="s", group=g,
               phys_min=0.15, phys_max=1.2,
               source="压带轮啮合后主导轴 CTL 锁相环建立锁定的时间常数 τ_lock",
               help="在状态 2（暂停↔播放瞬态）与状态 3（暂停起录切口）中，控制跨轨噪波条滚出画面、行时基恢复平稳以及伴音从静音滑升至正常音高所需的衰减时间（秒）。")
    tpe.define("pause_noise_pos", "暂停静态噪波带垂直位置", 0.62, 0.08, 0.92, unit="0-1", group=g,
               phys_min=0.15, phys_max=0.88,
               source="暂停定格时磁带停靠相位对应的方位角零点高度",
               help="在状态 1（Pause 静态定格）下，旋转磁头跨越相邻磁迹保护带形成的双峰雪花噪波横条在画面上的垂直落点（可用前面板 Tracking 旋钮微调移动）。")

    g = "线性编辑机与早期非线编"
    tpe.define("edit_mode", "编辑系统工作模式", 0.0, 0.0, 4.0, unit="mode", group=g, step=1.0,
               phys_min=0.0, phys_max=4.0,
               source="Sony RM-450 / BVE-9100 双机磁带线性编辑控制器 vs 2000年代初 Canopus DV Storm / Premiere 6 非线编系统",
               help="0=关闭｜1=线性组合编辑 (Assemble Edit：全磁鼓重写控制轨与视频轨，入点产生静止抹音头闪黑与 H-AFC 弯钩，出点留下新旧 CTL 控制轨断裂与雪花撕裂)｜"
                    "2=线性插入编辑 (Insert Edit：保留目标带原有连续 CTL 控制轨，由旋转飞抹磁头替换画面，无尾端断轨)｜"
                    "3=早期非线编 · 实时时间线预览 (NLE Real-Time Preview：零入点偏移，但多轨转场/字幕负载触发半分辨率代理降级与解码缓冲区欠载掉帧重复)｜"
                    "4=早期非线编 · 最终渲染导出 (NLE Final Export：全分辨率逐帧合成无掉帧，但对转场与叠字区间执行二次 DV25 8×8 DCT 重编码)。")
    tpe.define("edit_point_slip_frames", "线性编辑入出点机械偏移", 2.0, 0.0, 4.0, unit="帧", group=g, step=1.0,
               phys_min=0.0, phys_max=3.0,
               source="磁带压带轮啮合与 29.97/25Hz CTL 控制脉冲计数滑移容差 (±1..3 frames)",
               help="在线性编辑模式 (1/2) 下，双机预卷啮合与控制轨寻址造成的实际切点相对理想 EDL 入出点的帧偏移；非线编模式 (3/4) 采用文件样本寻址，恒为 0 帧偏移。")
    tpe.define("edit_flying_erase", "旋转飞抹磁头 (Flying Erase Head)", 1.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0,
               source="高档编辑录像机旋转上磁鼓飞抹磁头 (FEH) vs 家用机固定全宽抹音头",
               help="在线性插入编辑 (Mode 2) 下：1=旋转飞抹磁头正常（紧贴视频写磁头前缘逐条抹除旧螺旋轨，切点干净）；0=无飞抹磁头或飞抹老化（新画面直接叠写在未抹净的旧斜向磁迹上，切点出现斜向重影与 180° 色度副载波彩虹差拍）。")
    tpe.define("edit_preroll_sec", "双机同步预卷时长 (Pre-roll)", 5.0, 1.0, 8.0, unit="s", group=g,
               phys_min=3.0, phys_max=7.0,
               source="Sony BVE-9100 / RM-450 标准 5 秒倒带预卷 (Pre-roll) 伺服锁定时间",
               help="源录像机与录制机提前倒带并加速建立主导轴/磁鼓相位锁定的预卷时间（秒）；预卷过短会导致切点附近残留伺服抖动。")
    tpe.define("edit_nle_complexity", "非线编多轨合成负载复杂度", 0.65, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.90,
               source="2000年代初 IDE 磁盘吞吐与单核 CPU 实时解码多轨 DV 流 + Alpha 字幕的缓冲区负载",
               help="在模式 3（NLE 实时预览）下决定预览分辨率降级倍率与缓冲区欠载掉帧/卡顿概率；在模式 4（NLE 最终导出）下决定转场渲染段二次 DCT 重编码的量化损失。")
    tpe.define("edit_console_osd", "显示编辑台/时间线工程监看 HUD", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0,
               help="在画面顶部叠加 Sony BVE-9100 线性编辑双机时间码/CTL/飞抹状态栏，或早期 NLE 时间线缓存/掉帧/渲染状态栏。")

    g = "广播链路信噪比"
    rf.define("broadcast_noise", "广播噪声", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.7,
               source="1992 地方电视台开路弱信号录像：高频为基线 2 倍而饱和仅 0.178",
               help="接收端在解调前的宽带噪声。它与磁带噪声的关键区别是**同时淹没"
                    "亮度与色度**——实测参考素材高频升到 0.024 而饱和度只有 0.178，"
                    "即画面被噪声抹平而不是偏色。磁带劣化则相反：色度先死、"
                    "亮度还撑得住（参考 S-VHS 钳位失常带：sat 0.94 而 mean 0.09）。"
                    "两者用不同的参数，因为它们本来就不是同一个环节的故障。")

    g = "卫星接收机 Satellite IRD 与模拟/数字悬崖效应"
    rf.define("sat_ird_mode", "卫星接收工作站模式 (0关/1模拟C波段FM/2数字DVB-S/3同屏悬崖实验室)", 0.0, 0.0, 3.0, unit="模式", group=g,
              step=1.0, phys_min=0.0, phys_max=3.0,
              source="ITU-R BO.652 模拟卫星 FM 门限效应 vs ETSI EN 300 421 DVB-S QPSK+Viterbi+RS 数字悬崖效应",
              help="0=关闭｜1=模拟 C 波段 FM 卫星电视（低于 11.5 dB 门限时爆发标志性鱼骨黑白打火噪点 Sparkles，但画面内容渐进退化仍可辨认）｜"
                   "2=数字 DVB-S QPSK 卫星综合解码器（高于 6.5 dB 门限时 Reed-Solomon 纠错令画面 100% 干净无雪花，跌入 4.4–6.5 dB 悬崖区瞬间爆发 MPEG-2 宏块撕裂、PTS 失步与静帧冻结，低于 4.4 dB 直接服务丢失）｜"
                   "3=模拟 C 波段 FM vs 数字 DVB-S 同屏悬崖效应对照实验室（2×2 四象限：原信号 + 模拟 FM + 数字 DVB-S + 载噪比响应曲线与 TS 统计仪表盘）。")
    rf.define("sat_cn_db", "卫星晴空基准载噪比 (C/N)", 12.0, 0.0, 26.0, unit="dB", group=g,
              phys_min=3.0, phys_max=20.0,
              source="4 GHz C 波段 / 12 GHz Ku 波段抛物面天线下行链路功率预算",
              help="晴空且天线对准时的卫星下行中频基准载噪比 C/N0（dB）。调低此值或增大雨衰即可观察模拟 FM 渐进打火噪点与数字 DVB-S 悬崖效应的本质分野。")
    rf.define("sat_rain_fade_db", "暴雨衰减与日凌干扰 (Rain Fade / Sun Outage)", 0.0, 0.0, 20.0, unit="dB", group=g,
              phys_min=0.0, phys_max=15.0,
              source="ITU-R P.618 降雨衰减模型与黑体太阳噪声温度 T_sun 抬升",
              help="模拟强对流暴雨吸收散射或春分/秋分日凌（太阳进入抛物面天线主瓣）引发的载噪比骤降与系统热噪底抬升。")
    rf.define("sat_dish_mispoint_deg", "抛物面天线指向偏差角", 0.0, 0.0, 4.0, unit="deg", group=g,
              phys_min=0.0, phys_max=2.5,
              source="抛物面天线主瓣高斯方向图 G(θ) ≈ -12·(θ / θ_3dB)² dB",
              help="天线方位角/俯仰角受强风或机械松动偏离卫星轨位的角度（半功率波束宽度 θ_3dB = 1.6°）。")
    rf.define("sat_lnb_drift_khz", "高频头 LNB 本振温度频偏", 0.0, -500.0, 500.0, unit="kHz", group=g,
              phys_min=-250.0, phys_max=250.0,
              source="室外 LNB 介质谐振器 (DRO) 昼夜温差本振频率漂移",
              help="LNB 本振频偏导致模拟 FM 解调器工作点偏移（产生直流基座偏移、50Hz 截断削波与相位噪声），并在数字 DVB-S 下消耗载波恢复环路裕量。")
    rf.define("sat_ird_osd", "显示卫星画面文字与遥测 HUD (0不要文字/1仅标签/2完整HUD)", 0.0, 0.0, 2.0, unit="文字", group=g,
              step=1.0, phys_min=0.0, phys_max=2.0,
              help="默认 0=不要文字（纯净画面）；1=仅显示四角通道名；2=显示完整 C/N、BER 与 MPEG-TS 遥测文字。")
    rf.define("sat_quad_corner", "卫星对照角落选择 (0四分屏/1左上/2右上/3左下/4右下)", 0.0, 0.0, 4.0, unit="角落", group=g,
              step=1.0, phys_min=0.0, phys_max=4.0,
              source="卫星悬崖对照分屏角落拆选",
              help="在 2×2 卫星对照模式（sat_ird_mode=3）下拆成每个角落单独全屏选择：0=2×2四分屏，1=左上角（晴空基带基准），2=右上角（模拟C波段FM），3=左下角（数字DVB-S），4=右下角（C/N悬崖曲线）。")
    rf.define("sat_show_border", "卫星对照分框样式 (0无分框/1细十字线/2标准分框)", 1.0, 0.0, 2.0, unit="分框", group=g,
              step=1.0, phys_min=0.0, phys_max=2.0,
              source="分屏分框显示控制",
              help="0=无分框无缝拼接，1=细十字分框线，2=标准演播室分框。")

    g = "电视发射机激励链与端到端开路广播测试台"
    rf.define("tx_quad_corner", "发射测试台角落选择 (0四分屏/1左上/2右上/3左下/4右下)", 0.0, 0.0, 4.0, unit="角落", group=g,
              step=1.0, phys_min=0.0, phys_max=4.0,
              source="端到端开路广播测试台角落拆选",
              help="在 2×2 四象限测试台（tx_chain_mode=3）下拆成每个角落单独全屏选择：0=2×2四分屏全览，1=左上角（1.主控直通基带），2=右上角（2.发射机天线探头），3=左下角（3.模拟开路接收），4=右下角（4.数字地面 DTT 接收）。")
    rf.define("tx_show_border", "发射测试台分框样式 (0无分框/1细十字线/2标准分框)", 1.0, 0.0, 2.0, unit="分框", group=g,
              step=1.0, phys_min=0.0, phys_max=2.0,
              source="分屏分框显示控制",
              help="控制 2×2 四分屏之间的分框线：0=无分框无缝拼接，1=细十字分框线，2=标准演播室分框。")
    rf.define("tx_show_text", "发射测试台画面文字 (0不要文字/1仅角落标签/2标签+底部遥测栏)", 0.0, 0.0, 2.0, unit="文字", group=g,
              step=1.0, phys_min=0.0, phys_max=2.0,
              source="纯净画面无文字开关",
              help="默认 0=不要文字（纯净画面，无顶部黑条与底部遥测文字）；1=仅显示四角通道标签；2=显示四角通道标签与底部工程遥测栏。")
    rf.define("tx_chain_mode", "电视发射机激励链工作模式 (0关/1模拟VSB+HPA/2数字COFDM/3四分屏/4..7单选角落)", 0.0, 0.0, 7.0, unit="模式", group=g,
              step=1.0, phys_min=0.0, phys_max=7.0,
              source="ITU-R BT.470 / FCC Part 73 模拟残留边带发射机 & DVB-T COFDM 数字发射机激励链规范",
              help="0=关闭｜1=模拟大功率电视发射机激励与天馈链（高稳晶振 → VSB 残留边带激励器 → 速调管/固态高功放 HPA AM-AM 微分增益与 AM-PM 微分相位 → 920kHz/1.57MHz 图像-色度-伴音互调 → 腔体双工器群延迟与铁塔馈线 VSWR 重影 → 开路信道与接收机）｜"
                   "2=数字地面电视 (DTT COFDM) 发射与接收链（功放过激励削波抬升 MER 底噪，触发数字接收机 QEF/Cliff 悬崖效应）｜"
                   "3=端到端开路广播 2×2 四象限对照测试台（同屏对比：主控直通基带参考、发射机天线口监测、模拟开路接收终端、数字地面接收终端及失真贡献分解表）。")
    rf.define("tx_pa_drive", "高功放 (HPA) 激励过载深度 (AM-AM 微分增益)", 0.35, 0.0, 1.5, unit="0-1.5", group=g,
              phys_min=0.05, phys_max=1.10,
              source="电子管/速调管与 LDMOS 固态高功放 Rapp 饱和包络模型",
              help="控制末级高功放逼近 1 dB 压缩点的驱动深度：由于模拟电视采用负极性调幅（黑电平载波功率高、白电平接近零载波），高功放饱和会非线性压缩高光色度副载波幅度（产生微分增益 DG%）并压扁同步顶。")
    rf.define("tx_pa_am_pm_deg", "高功放 AM-PM 微分相位失真", 8.0, 0.0, 45.0, unit="deg", group=g,
              phys_min=0.0, phys_max=28.0,
              source="速调管/行波管束流速度调制与固态功放结电容随包络电压变化的 AM-PM 相移",
              help="亮度从黑电平过渡到白电平时，高功放寄生相移导致 3.58/4.43 MHz 色度副载波发生随亮度变化的相位旋转（微分相位 DP°，表现为高光区域色相偏转）。")
    rf.define("tx_filter_gd_ns", "VSB 残留边带与腔体双工器群延迟", 65.0, 0.0, 400.0, unit="ns", group=g,
              phys_min=15.0, phys_max=250.0,
              source="声表面波 VSB 成形滤波器与大功率波导腔体合成器 (Cavity Diplexer) 边缘群延迟峰值",
              help="发射机带通滤波器在通带边缘的陡峭群延迟斜率：导致 3.58/4.43 MHz 色度包络相对宽带亮度向右延迟错位，并在垂直硬边处激发 2T K 因子振铃过冲。")
    rf.define("tx_feeder_vswr", "铁塔主馈线驻波比 (VSWR)", 1.15, 1.0, 3.5, unit="VSWR", group=g,
              phys_min=1.0, phys_max=2.2,
              source="发射塔长馈管终端天线失配反射系数 Γ = (VSWR - 1) / (VSWR + 1)",
              help="天线振子结冰、进水或阻抗失配在数百米铁塔长馈线内产生往返反射，在发射端即形成固定延迟的馈线驻波重影与频响波纹。")
    rf.define("tx_sound_intermod", "伴音-色度-图像三载波互调 (920kHz 差拍)", 0.25, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.75,
              source="共用功放 (Common Amplification) 图像 f_v、色度 f_sc 与伴音 f_a 三阶互调产物",
              help="当图像与伴音共用末级功放或双工器隔离度下降时，伴音载波 (+4.5/+6.0 MHz) 与色度副载波 (3.58/4.43 MHz) 产生 920 kHz / 1.57 MHz 差拍斜向网纹，同时高亮度扫描行侵入伴音鉴频器产生 50/60 Hz 画面蜂音 (Intercarrier Buzz)。")

    # ------------------------------------------------------------------
    # Physical connectors between the source and the recorder.  Kept per-connector
    # rather than as one "damage" control, because in a real chain one joint in the run
    # is corroded and the others are not, and locating the fault is the point.
    cn = s.layer("connector")
    g = "链路中的接口"
    cn.define("hdmi_present", "HDMI 接口", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
              phys_min=0.0, phys_max=1.0, source="HDMI 转 AV 转换器的输入",
              help="高速差分数字接口。损坏的表现与模拟接口**完全不同**："
                   "数字没有「轻微雪花」这个状态，位要么对要么错，所以看到的是"
                   "马赛克、局部冻结、闪黑；再差就是握手失败直接「无信号」。")
    cn.define("hdmi_moisture", "HDMI 受潮", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6,
              help="水在缝隙里把介电常数从空气的 1 拉到约 80，单位长度电容上升，"
                   "差分阻抗偏离 100Ω。差分对对此极其敏感，眼图很快闭合。")
    cn.define("hdmi_corrosion", "HDMI 氧化", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6,
              help="氧化层是差的导体，接触电阻上升，结点更像开路而不是匹配负载。")
    cn.define("aspect_squeeze", "16:9→4:3 二次压扁", 0.0, 0.0, 1.5, unit="次", group=g,
              phys_min=0.0, phys_max=1.0, source="机顶盒 720p60 16:9 固定输出 → HDMI-AV 压入 4:3 行宽",
              help="机顶盒（如小米盒子 720p60）固定输出 16:9 宽屏，经 HDMI-AV 与 PAL-D 调制进入 4:3 录像机"
                   "（松下 NV-J27）时被横向压缩为 3/4（0.75），导致原本 4:3 的画面被再次横向压扁为 9:16"
                   "（两侧各留 12.5% 消隐黑边）。1.0 = 一次 16:9→4:3 横向压扁（×0.75）。")
    g = "AV / RCA 接口"
    cn.define("av_present", "AV 接口", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
              phys_min=0.0, phys_max=1.0, source="RCA 复合视频（黄莲花头）")
    cn.define("av_moisture", "AV 受潮", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6)
    cn.define("av_corrosion", "AV 氧化", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6)
    cn.define("av_ground_loop", "地环路", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.5,
              help="RCA 的**外壳就是地**。外壳接触电阻大，线缆参考电位与信号源不同，"
                   "差值就是市电上的交流声——表现为一条**缓慢滚动**的亮暗条。"
                   "滚动是这类故障的全部特征；静止的条是另一种故障。")
    cn.define("mains_hz", "市电频率", 50.0, 45.0, 65.0, unit="Hz", group=g,
              phys_min=50.0, phys_max=60.0,
              help="滚动的速度取决于市电与场频之**差**，所以 50Hz 与 60Hz 地区的滚动方向"
                   "与速度不同。这解释了同一台设备在不同地区观感不同。")
    g = "S-Video 接口"
    cn.define("svideo_present", "S-Video 接口", 0.0, 0.0, 1.0, unit="bool", group=g,
              step=1.0, phys_min=0.0, phys_max=1.0)
    cn.define("svideo_moisture", "S-Video 受潮", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6)
    cn.define("svideo_corrosion", "S-Video 氧化", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6)
    cn.define("svideo_leakage", "Y/C 串扰", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.5,
              help="Y 与 C 在相隔几毫米的两个针脚上，受潮形成漏电通路。"
                   "结果是**边缘上的彩色镶边**而不是整体偏色：亮度的陡变出现在色度通道里，"
                   "反过来也一样，所以镶边正好落在边缘上——"
                   "这就是为什么它读起来像坏磁带，而源头是湿了的接口。")
    g = "RF / 天线接口"
    cn.define("rf_present", "RF 接口", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
              phys_min=0.0, phys_max=1.0, source="75Ω 同轴天线接口")
    cn.define("rf_moisture", "RF 受潮", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6,
              help="水改变介电常数，阻抗偏离 75Ω，能量被反射回去。反射波与前进波干涉，"
                   "而干涉是相加还是相消取决于**电长度**，所以损伤是**频率选择性的**——"
                   "咬住某些频率而放过另一些，不是整体衰减。")
    cn.define("rf_corrosion", "RF 氧化", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.6,
              help="氧化层加潮气可以像一个粗糙的二极管：一个方向比另一个方向更容易导通，"
                   "正弦波被削掉一边。削顶会产生**谐波**，所以腐蚀的 RF 接点会干扰"
                   "**别的频道**——干扰是在接点上产生的，不是从外面收进来的。")
    cn.define("rf_shielding", "屏蔽损失", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.5,
              help="屏蔽层破损会让外部射频进入。与「氧化」的区别：氧化是接点**自己产生**"
                   "干扰，屏蔽损失是**外部**干扰进来。")

    g = "电源与电气母线 (Power & Electrical Bus)"
    cn.define("psu_ripple", "电源滤波纹波 (PSU Ripple)", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.65,
              source="整流滤波电解电容 ESR 升高与容量干涸导致的 50/60Hz 及 100/120Hz 全波纹波",
              help="模拟线性/开关电源滤波电解电容老化干涸：在共享直流母线上注入 100/120Hz 全波整流充放电锯齿波与 "
                   "50/60Hz 半波分量。由于电容放电斜率 dV/dt = -I_load/C，动态负载越重纹波越深，同时造成画面滚动交流哼声带、"
                   "垂直边缘 S 形「交流声蛇形扭曲 (Hum Snake)」与伴音 50/100Hz 整流蜂音。")
    cn.define("battery_level", "电池剩余电量 (Battery SoC)", 1.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.10, phys_max=1.0,
              source="NiCd / NiMH / Li-ion 摄像机与便携录像机电池 Shepherd 电化学放电曲线",
              help="1.0 为满电标称电压；放电至 0.35 以下进入 Shepherd 曲线陡降膝点，开路电压 V_oc 下跌且电池内阻 R_int 急剧攀升。")
    cn.define("battery_sag", "电池内阻压降 (Dynamic Load Sag)", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.75,
              source="高内阻老化电池在红外灯/变焦马达/走带伺服/高亮画面冲击下的母线压降 V_bus = V_oc - I_load * R_int",
              help="供电内阻系数。当红外夜视灯、W-T 电动变焦马达、主导轴粘带/快进伺服或 CRT 大面积高光抽取引发瞬态大电流时，"
                   "母线电压 V_bus 动态塌陷，导致摄像曝光变暗与底噪激增、走带伺服掉速（画面撕裂+音频变调下坠）、同步分离门限漂移及显像管画幅收缩。")
    cn.define("load_coupling", "跨设备负载耦合度 (Load Coupling)", 0.0, 0.0, 1.5, unit="0-1", group=g,
              phys_min=0.0, phys_max=1.0,
              source="成像/伺服装载/显像管高压共地共母线动态耦合",
              help="控制摄像机（红外灯+变焦电机）、走带机构（主导轴/磁鼓伺服）与显示器（阴极束流）通过公共电源母线相互拉扯电压的耦合强度。")
    cn.define("startup_surge", "冷机上电冲击瞬态 (Power-On Surge)", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.80,
              source="开机瞬间主滤波电容充电浪涌、显像管灯丝预热与主导轴伺服锁相建立过程",
              help="建模开机前 2.4 秒的冷机上电瞬态：滤波电容浪涌压降、稳压器过冲闪亮、以及主导轴/磁鼓从慢速爬升锁相的动态过程。")
    cn.define("emi_crosstalk", "电机/行包电磁串扰 (EMI Crosstalk)", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.65,
              source="主导轴/变焦直流电机换向器火花与行输出变压器对视频/音频母线的传导耦合",
              help="直流有刷电机（变焦马达、主导轴电机）换向火花与 CRT 行输出高压包脉冲沿共享电源线串入视频基带（细密水平火花纹）与伴音通道。")

    g = "多路模拟切换台与特技混合器 (Analog Video Switcher / SEG)"
    cn.define("switcher_mode", "切换台工作模式", 0.0, 0.0, 5.0, unit="mode", group=g, step=1.0,
              phys_min=0.0, phys_max=5.0,
              source="松下 WJ-AVE5 / WJ-MX12 / 索尼 XV-AL100 模拟视频特技切换台 (SEG)",
              help="0=关闭(直通 Bus A)｜1=非同步信号源硬切瞬态 (Non-Genlocked Hot Switch：无帧同步器硬切引发下游 H-AFC 弯钩撕裂、V-Hold 场跳动滚屏与色同步相位突变)｜"
                   "2=模拟 T-Bar 叠化 (Mix / Dissolve：双路消隐基座叠加漂移与非锁相副载波差拍网纹)｜"
                   "3=模拟电压斜坡划像 (Analog Ramp Wipe：水平/垂直/对角/菱形/圆形划像与比较器彩边)｜"
                   "4=模拟亮度键控 (Analog Luma Key：高速电压比较器切键、700ns 群延迟错位与副载波拉链边)｜"
                   "5=模拟色度键控 (Analog Chroma Key：窄带色差蓝幕抠像、色度群延迟黑边与 3.58/4.43MHz 副载波残留镶边)。")
    cn.define("switcher_b_source", "副路/叠层信号源 (Bus B) 选择", 0.0, 0.0, 5.0, unit="src", group=g, step=1.0,
              phys_min=0.0, phys_max=5.0,
              source="切换台 Preset / Ext Key 输入母线与自定义叠层发生器",
              help="当节点图中 `video_switcher` 节点的 `in_b` 端口已接入第二条上游信号链时，自动使用外部 `in_b` 信号；"
                   "单输入或固定链路下可自由切换自定义叠层源：0=自适应自定义叠层/副机位 (Camera B)｜1=自定义文字与字幕底栏叠层｜"
                   "2=自定义画中画小窗叠层 (PIP Window)｜3=自定义上传叠层图/副视频｜4=自定义色调/光效渐变叠层｜5=经典 SMPTE 演播室彩条。")
    cn.define("switcher_overlay_style", "自定义叠层样式", 0.0, 0.0, 5.0, unit="style", group=g, step=1.0,
              phys_min=0.0, phys_max=5.0,
              source="切换台自定义图文/画中画/外接图层合成器",
              help="0=自动适配（已导入叠层图时使用上传图，划像时使用纯净副机位，键控时使用画中画小窗）｜"
                   "1=自定义文字 + 字幕底栏 (Lower-Third + Text)｜2=仅自定义文字/角标 (Text / Bug Only)｜"
                   "3=自定义画中画小窗 + 标题 (PIP Window + Title)｜4=全画幅纯净副机位画面 (Full-Frame Camera B)｜"
                   "5=经典演播室徽标与字幕条 (Classic Emblem + Banner)。")
    cn.define("switcher_overlay_text", "自定义叠层文本", "STUDIO B · LIVE", 0.0, 1.0, unit="text", kind="text", group=g,
              source="切换台内置字幕/角标叠层字符发生器（支持中英文自定义输入，留空则不显示文字）",
              help="可输入任意中英文自定义叠层标题、台标或注释文本；清空此项即可隐藏叠层文字。")
    cn.define("switcher_overlay_path", "自定义叠层图/视频路径", "", 0.0, 1.0, unit="text", kind="text", group=g,
              source="顶部「🖼 导入叠层」按钮上传的自定义图片(PNG/JPG，支持透明通道)或视频路径",
              help="可通过顶部工具栏「🖼 导入叠层」一键上传图片或视频作为划像/色键/亮键的自定义叠层源；清空即可恢复参数化叠层。")
    cn.define("switcher_overlay_x", "自定义叠层水平位置 X", 0.50, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.05, phys_max=0.95,
              help="控制自定义叠层（画中画窗口、字幕条、角标、上传图层或菱形/圆形划像中心）的水平中心坐标 (0=最左, 0.5=居中, 1=最右)。")
    cn.define("switcher_overlay_y", "自定义叠层垂直位置 Y", 0.78, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.05, phys_max=0.95,
              help="控制自定义叠层（画中画窗口、字幕条、角标、上传图层或菱形/圆形划像中心）的垂直中心坐标 (0=最上, 0.5=居中, 1=最下)。")
    cn.define("switcher_overlay_scale", "自定义叠层缩放尺寸", 1.00, 0.15, 2.50, unit="x", group=g,
              phys_min=0.30, phys_max=2.00,
              help="控制自定义叠层窗口、字幕条、文字或上传叠层图像的缩放比例。")
    cn.define("switcher_overlay_opacity", "自定义叠层不透明度", 0.92, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.10, phys_max=1.0,
              help="控制自定义叠层图元、画中画窗口或划像副画面叠层的混合不透明度。")
    cn.define("switcher_overlay_r", "自定义叠层主色 R", 0.95, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=1.0,
              help="自定义叠层边框、装饰条、文字或渐变叠层的红色分量 (R)。")
    cn.define("switcher_overlay_g", "自定义叠层主色 G", 0.86, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=1.0,
              help="自定义叠层边框、装饰条、文字或渐变叠层的绿色分量 (G)。")
    cn.define("switcher_overlay_b", "自定义叠层主色 B", 0.28, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=1.0,
              help="自定义叠层边框、装饰条、文字或渐变叠层的蓝色分量 (B)。")
    cn.define("switcher_tbar", "T-Bar 推杆位置 / 划像进度", 0.50, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=1.0,
              source="模拟特技切换台机械 T-Bar 电位器推杆 (0.0 = Bus A .. 1.0 = Bus B)",
              help="在叠化 (Mode 2) 下控制 Bus A 与 Bus B 的混合比例；在划像 (Mode 3) 下控制斜坡比较器的擦除进度；在硬切 (Mode 1) 下控制场中切换行位置。")
    cn.define("switcher_wipe_pattern", "模拟划像图形 (Wipe Pattern)", 0.0, 0.0, 4.0, unit="pat", group=g, step=1.0,
              phys_min=0.0, phys_max=4.0,
              source="模拟行/场锯齿波与抛物波函数发生器",
              help="0=水平拉幕划像 (Left→Right)｜1=垂直拉幕划像 (Top→Bottom)｜2=45°对角斜切划像｜3=中心菱形光圈 (Diamond Iris)｜4=中心圆形光圈 (Circle Iris)。")
    cn.define("switcher_hot_cut_rate", "非同步硬切频率", 1.25, 0.10, 6.0, unit="次/秒", group=g,
              phys_min=0.25, phys_max=3.5,
              source="多机位无锁相导播切换或磁带拼接跳切频率",
              help="在模式 1 (Hot Switch) 下，主副信号源之间发生非同步硬切的速率。")
    cn.define("switcher_genlock_err", "异源时基与副载波失配度 (Genlock Error)", 0.65, 0.0, 1.5, unit="0-1", group=g,
              phys_min=0.0, phys_max=1.2,
              source="两台独立晶振摄像机/录像机之间的行相移 Δτ_H、场计数差 ΔY 与副载波频偏 Δφ_sc",
              help="0 为两路信号已通过帧同步器/黑场发生器严格锁相 (Genlocked)；大于 0 时，硬切瞬间爆发 H-AFC 顶部弯钩撕裂、V-Hold 场跳动滚屏（露出 VBI 消隐带）与 APC 副载波色相突变，叠化时产生非相干副载波差拍网纹。")
    cn.define("switcher_key_clip", "模拟键控比较器门限 (Key Clip)", 0.42, 0.05, 0.95, unit="IRE", group=g,
              phys_min=0.15, phys_max=0.85,
              source="LM311 / 差分对高速键控电压比较器参考电位",
              help="亮键 (Luma Key) 或色键 (Chroma Key) 的触发电压门限。")
    cn.define("switcher_key_gain", "模拟键控放大斜率 (Key Gain)", 0.65, 0.05, 1.5, unit="0-1", group=g,
              phys_min=0.20, phys_max=1.2,
              source="模拟线性键控器边缘放大器增益",
              help="控制划像与键控过渡带的陡峭程度：越低边缘越柔和（半透明混合），越高越接近硬切。")
    cn.define("switcher_key_jitter", "比较器行触发抖动 (Comparator Jitter)", 0.35, 0.0, 1.5, unit="0-1", group=g,
              phys_min=0.0, phys_max=1.0,
              source="模拟斜坡发生器与键控比较器受电源纹波/热噪声调制产生的逐行切点抖动",
              help="使划像边界与键控轮廓在相邻扫描行之间发生细腻的模拟锯齿颤动。")
    cn.define("switcher_subcarrier_spill", "副载波残留拉链边与群延迟镶边", 0.45, 0.0, 1.5, unit="0-1", group=g,
              phys_min=0.0, phys_max=1.2,
              source="0.6MHz 色度窄带群延迟 (~700ns) 错位与 3.58/4.43MHz 副载波未完全陷波残留",
              help="还原 80/90 年代模拟复合键控最标志性的两大物理缺陷：1. 色度窄带滤波器比宽带亮度慢约 700ns（约 5 像素），导致抠像蒙版相对前景画面整体右移，在左侧露出暗黑/蓝幕切边、右侧溢出亮边；2. 复合信号中的副载波调制比较器门限，在边缘形成 4 像素周期的彩色拉链锯齿 (Subcarrier Zipper)。")
    cn.define("switcher_pedestal_add", "双母线黑电平叠加与溢色漂移", 0.25, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.75,
              source="无后肩重钳位的模拟电阻混合母线 7.5 IRE 双基座叠加",
              help="在 T-Bar 半叠化（50/50 Mix）或键控叠加时，两路信号的消隐黑电平基座相加导致画面暗部发灰抬升。")

    g = "广播级配线架与 75Ω 分配放大器 (Patchbay, DA & 75Ω Termination)"
    cn.define("da_enabled", "启用配线架与分配放大器", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
              phys_min=0.0, phys_max=1.0,
              source="广播级演播室 75Ω 视频分配放大器 (Grass Valley / Leitch / Extron MDA) 与跳线配线架",
              help="开启后在信号链中启用 75Ω 源-负载端接分压、同轴电缆反射/趋肤效应、有源分配放大器均衡与配线架交叉点物理模型。")
    cn.define("da_termination_mode", "75Ω 端接与分配模式", 0.0, 0.0, 3.0, unit="mode", group=g, step=1.0,
              phys_min=0.0, phys_max=3.0,
              source="SMPTE 170M / ITU-R BT.470 75Ω 源内阻与终端匹配规范 (V_load = 2*V_nom * R_L / (75 + R_L))",
              help="0=标准 75Ω 单端接匹配 (1.0 Vpp / 0 dB, 反射系数 Γ=0)｜"
                   "1=未端接高阻开路环通 (High-Z Open Loop-Through, R_L≈10kΩ：电压翻倍至 ~2.0 Vpp / +6 dB，导致高光硬削波死白、色彩过饱和及正向延迟重影 Γ≈+0.98)｜"
                   "2=违规三通无源并接 / 双端接或多端接 (Passive T-Split, R_L=75Ω/N：双端接 37.5Ω 跌落 -3.52 dB、三端接 25Ω 跌落 -6.02 dB，画面昏暗发灰、-40 IRE 同步顶被压扁引发下游 H-AFC 撕裂，并产生负向反相重影 Γ<0)｜"
                   "3=有源分配放大器隔离驱动 (Active DA 1-in-N-out：高阻环通输入 + 独立 75Ω 背端接运放驱动，支持多路无损分发与电缆高频均衡补偿)。")
    cn.define("da_load_count", "并接/分配负载路数 N", 2.0, 1.0, 6.0, unit="路", group=g, step=1.0,
              phys_min=1.0, phys_max=4.0,
              source="无源三通并接终端数 N 或有源 DA 扇出通道数",
              help="在模式 2（违规无源三通并接）下决定并联 75Ω 负载个数 N（R_L = 75Ω/N，N=2 为 37.5Ω 双端接，N=3 为 25Ω 三端接）；在模式 3（有源 DA）下表示有源扇出路数。")
    cn.define("da_cable_len_m", "同轴线缆长度 (RG-59/8281)", 15.0, 1.0, 150.0, unit="m", group=g,
              phys_min=2.0, phys_max=100.0,
              source="Belden 8281 / RG-59 75Ω 同轴电缆传播速度 v_p≈0.66c (往返延迟 10.1 ns/m) 与趋肤效应高频衰减",
              help="决定阻抗失配时的往返反射重影水平间距 Δx，以及长距离传输时的趋肤效应高频滚降。")
    cn.define("da_gain_db", "有源 DA 增益微调", 0.0, -6.0, 6.0, unit="dB", group=g,
              phys_min=-3.0, phys_max=3.0,
              source="广播级分配放大器前面板 Gain 电位器 (-6 dB .. +6 dB)",
              help="有源分配放大器 (Mode 3) 或基准通道的电压增益微调（dB）。")
    cn.define("da_eq_boost", "有源 DA 电缆均衡高频补偿", 0.0, 0.0, 1.5, unit="0-1", group=g,
              phys_min=0.0, phys_max=1.0,
              source="有源分配放大器前面板 Cable EQ 高频峰值补偿网络",
              help="补偿长距离同轴电缆的高频亮度与 3.58/4.43MHz 色度副载波衰减；在短电缆上过度开启会产生硬边高频过冲振铃。")
    cn.define("patch_crosstalk", "配线架/矩阵交叉点电容串扰", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.65,
              source="高密度跳线配线架 (Patchbay) 或模拟矩阵切换器相邻通道寄生电容耦合 C_xp * dV_B/dt",
              help="相邻跳线孔或矩阵交叉点中的强信号向本通道发生微分电容串扰：在画面中浮现相邻母线（Bus B 或演播室彩条母线）的边缘微分浮雕轮廓与副载波差拍斜纹。")
    cn.define("patch_contact_aging", "跳线插孔触点氧化与接触噪波", 0.0, 0.0, 1.0, unit="0-1", group=g,
              phys_min=0.0, phys_max=0.65,
              source="Bantam / 1/4\" 常闭跳线插孔 (Normalled Jack) 簧片硫化氧化与屏蔽地松动",
              help="老化跳线插孔引入接触电阻压降、屏蔽壳松动工频微哼声，以及间歇性的行向接触火花噪波与局部丢色。")
    cn.define("patch_format_mismatch", "跳线信号格式误插", 0.0, 0.0, 3.0, unit="mode", group=g, step=1.0,
              phys_min=0.0, phys_max=3.0,
              source="配线架跨信号制式误跳线故障表征 (Composite / S-Video Y / YPbPr / SDI)",
              help="0=正常格式匹配｜1=S-Video 亮度 (Y) 误插复合输入（丢失色同步与色度，画面变为锐利纯黑白并带残留亮边点爬行）｜"
                   "2=分量 YPbPr / RGB 绿通道误插复合输入（无副载波基带色差导致绿/品红偏色与行同步滑移）｜"
                   "3=SDI / 数字码流误插模拟复合监视器（满屏高频数字雪花杂波与失锁滚道）。")

    g_env = "二维空间物理环境场与多维耦合"
    cn.define("env_field_enabled", "启用二维空间物理环境场 (EnvironmentField)", 0.0, 0.0, 1.0, unit="bool", group=g_env, step=1.0,
              phys_min=0.0, phys_max=1.0, source="空间连续二维环境场 (温湿度/电磁/声学/光学四场耦合)",
              help="每台设备根据自身在机房/拍摄空间中的二维坐标 (x, y) 读取局域温度、相对湿度、光照频谱与电磁干扰场，而非共用单一全局标量。")
    cn.define("env_field_mode", "空间环境场视图与测试模式", 0.0, 0.0, 5.0, unit="0-5", group=g_env, step=1.0,
              phys_min=0.0, phys_max=5.0, source="四场物理耦合与 2×2 空间热力图实验室",
              help="0=跟随总开关｜1=温湿度与结露粘带耦合 (冷机抱入暖房镜片起雾与磁鼓结露粘带)｜2=变压器/发射塔空间电磁场与接地环路耦合｜"
                   "3=房间声学混响/驻波与混合光源杂散光耦合｜4=四场全耦合 + 机房环境场 HUD 探针｜5=2×2 空间物理环境场与设备坐标四分屏实验室。")
    cn.define("env_room_temp_c", "机房/暖通空调基准温度", 24.0, -10.0, 45.0, unit="°C", group=g_env,
              phys_min=10.0, phys_max=35.0, source="ASHRAE 演播室/客厅温控基准")
    cn.define("env_room_rh_pct", "机房/环境相对湿度 (RH)", 65.0, 15.0, 98.0, unit="%", group=g_env,
              phys_min=30.0, phys_max=85.0, source="Magnus-Tetens 露点方程输入")
    cn.define("env_device_init_temp_c", "设备入场初始机壳温度 (冷机抱入暖房)", 6.0, -15.0, 40.0, unit="°C", group=g_env,
              phys_min=-5.0, phys_max=30.0, source="牛顿冷却定律与机壳热惯性 τ_th",
              help="当冷机温度低于局域空气露点温度 T_dew(x, y) 时，摄像机前镜片瞬间结露起雾、录像机磁鼓发生高危水膜粘带，并随设备升温跨帧自然消散。")
    cn.define("env_emi_source_power", "局域变压器/发射塔电磁辐射源功率", 0.55, 0.0, 2.0, unit="0-2", group=g_env,
              phys_min=0.0, phys_max=1.2, source="近场 1/r² 磁偶极子辐射与接地环路感应电动势")
    cn.define("env_shielding_eff", "机箱/同轴线缆电磁屏蔽效能", 0.50, 0.0, 0.98, unit="0-1", group=g_env,
              phys_min=0.20, phys_max=0.95, source="铜编织网屏蔽层与机架接地抑制比")
    cn.define("env_light_mix", "窗户日光 vs 钨丝灯/荧光灯空间混合比", 0.35, 0.0, 1.0, unit="0-1", group=g_env,
              phys_min=0.0, phys_max=1.0, source="二维空间色温梯度与 CRT 玻壳杂散光漫反射")
    cn.define("env_acoustic_rt60", "房间混响时间 RT60 与驻波耦合", 0.48, 0.05, 2.5, unit="s", group=g_env,
              phys_min=0.15, phys_max=1.6, source="Sabine 室内混响方程与早期反射梳状滤波")
    cn.define("env_cam_x", "摄像机空间坐标 X", 0.22, 0.0, 1.0, unit="0-1", group=g_env,
              phys_min=0.0, phys_max=1.0, source="二维机房平面横坐标 (0=左窗侧, 1=右机柜侧)")
    cn.define("env_cam_y", "摄像机空间坐标 Y", 0.30, 0.0, 1.0, unit="0-1", group=g_env,
              phys_min=0.0, phys_max=1.0, source="二维机房平面纵坐标 (0=上空调侧, 1=下潮湿死角/电源柜侧)")
    cn.define("env_vtr_x", "录像机 (VTR) 空间坐标 X", 0.72, 0.0, 1.0, unit="0-1", group=g_env,
              phys_min=0.0, phys_max=1.0, source="录像机在二维环境场中的横坐标")
    cn.define("env_vtr_y", "录像机 (VTR) 空间坐标 Y", 0.68, 0.0, 1.0, unit="0-1", group=g_env,
              phys_min=0.0, phys_max=1.0, source="录像机在二维环境场中的纵坐标")
    cn.define("env_crt_x", "CRT 监视器空间坐标 X", 0.80, 0.0, 1.0, unit="0-1", group=g_env,
              phys_min=0.0, phys_max=1.0, source="CRT 监视器在二维环境场中的横坐标")
    cn.define("env_crt_y", "CRT 监视器空间坐标 Y", 0.28, 0.0, 1.0, unit="0-1", group=g_env,
              phys_min=0.0, phys_max=1.0, source="CRT 监视器在二维环境场中的纵坐标")
    cn.define("env_quad_corner", "空间环境场分屏角落选择", 0.0, 0.0, 4.0, unit="0-4", group=g_env, step=1.0,
              phys_min=0.0, phys_max=4.0, source="四分屏单象限全屏放大检视",
              help="0=2×2 四分屏同屏对照｜1=左上角全屏 (温湿度露点场+结露画面)｜2=右上角全屏 (电磁干扰场+地环路画面)｜"
                   "3=左下角全屏 (光学/声学场耦合画面)｜4=右下角全屏 (各设备局域物理遥测仪表盘)。")
    cn.define("env_show_border", "空间环境场分框线显示", 2.0, 0.0, 2.0, unit="0-2", group=g_env, step=1.0,
              phys_min=0.0, phys_max=2.0, source="四分屏边框模式 (0=无边框, 1=细十字线, 2=工业外框)")
    cn.define("env_show_text", "空间环境场文字叠加显示", 0.0, 0.0, 2.0, unit="0-2", group=g_env, step=1.0,
              phys_min=0.0, phys_max=2.0, source="0=隐藏全部文字(纯净画面)｜1=紧凑角标｜2=完整物理遥测文字")

    # ------------------------------------------------------------------ D
    ply = s.layer("player")
    g = "时基校正与走带"
    ply.define("transport_jitter", "走带抖动", 0.45, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="主导轴伺服残差")
    ply.define("tbc_enabled", "时基校正开启", 0.0, 0.0, 1.0, unit="bool", group=g,
               step=1.0, source="TBC / 数字时基校正器")
    ply.define("tbc_strength", "时基校正强度", 1.0, 0.0, 1.0, unit="0-1", group=g)
    ply.define("tbc_residual", "校正残差", 0.0, 0.0, 1.0, unit="0-1", group=g)
    ply.define("stabilizer_stages", "多级串接画面稳定（H96+M9E）", 0.0, 0.0, 2.0, unit="级", group=g,
               step=1.0, phys_min=0.0, phys_max=2.0,
               source="夏普 VC-A62DT（主采录）→ 夏普 VC-H96（一级稳定）→ 夏普 VC-M9E（二级稳定）→ 天创恒达 TC-UB530",
               help="0=单机直出｜1=接入二级录像机（夏普 VC-H96，一级画面稳定）｜2=再串接三级录像机"
                    "（夏普 VC-M9E，二级画面稳定）。每一级串联录像机的行 AFC 与同步再生电路会逐级滤除"
                    "COSMAG E-180 磁带的时基漂移与 CTL 失步抖动，同时引入两级 75Ω 复合缓冲与 1H 延迟线"
                    "的轻微色度右移（约 0.85 px/级）及黑电平钳位，确保天创恒达 TC-UB530 采集卡与 OBS Studio 32.0 不丢帧。")
    ply.define("line_delay", "行延迟", 0.0, -8.0, 8.0, unit="px", group=g,
               phys_min=-3.0, phys_max=3.0, source="色度相对亮度的群延迟")
    g = "磁头与频响"
    ply.define("head_wear", "磁头磨损", 0.30, 0.0, 1.0, unit="0-1", group=g,
               phys_max=0.6, source="磁头尖磨损",
               help="磨损的磁头损失高频并降低色度信噪比。")
    ply.define("frequency_response", "频响倾斜", 0.15, -1.0, 1.0, unit="0-1", group=g,
               phys_max=0.5)
    ply.define("ringing", "振铃强度", 0.0, 0.0, 1.0, unit="0-1", group=g, phys_max=0.5)
    ply.define("ringing_freq", "振铃频率", 0.5, 0.1, 0.95, unit="xNyquist", group=g)
    ply.define("preemphasis", "高频预加重", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.6, source="记录前的轮廓补偿（aperture correction）",
               help="记录时提升高频以抗噪，播放时本应还原——做不到完美就会在硬边上留下过冲。")
    g = "分离与输出"
    ply.define("luma_bandwidth", "亮度带宽", 3.0, 0.1, 6.0, unit="MHz", group=g,
               phys_min=2.0, phys_max=5.0)
    ply.define("chroma_bandwidth", "色度带宽", 0.75, 0.25, 2.0, unit="MHz", group=g,
               phys_min=0.4, phys_max=1.2)
    # A deck's own separator.  Kept separate from the format layer's copy because a
    # camcorder and a deck are two different receivers, and each chose its own design.
    ply.define("separation_kind", "Y/C 分离方式", 0.0, 0.0, 5.0, unit="0-4", group=g,
               step=1.0, phys_min=0.0, phys_max=4.0,
               source="消费级接收机的四种分离方案（同年代分层）",
               help="**器件选择，不是效果旋钮。** 0=带通分离（廉价机，无延迟线）；"
                    "1=盒式孔径（小横向孔径，载波只被部分抵消，彩边粗而宽）；"
                    "2=1H 延迟线（抽头相隔一行，副载波自然反相抵消）；"
                    "3=2H 延迟线（垂直孔径更宽，串色更少但垂直亮度更糊）；"
                    "4=二维梳状（横向 ±1 像素、纵向 ±1 行，**对角抽头与中心同相位**，"
                    "不需要延迟线即可去载波，四者中垂直拖尾最小）。"
                    "**这是播放设备的属性**，与 format 层的同名参数相互独立——"
                    "摄像机和录像机是两台不同的接收机。")
    ply.define("chroma_vert_blend", "垂直色度反馈", 0.5, 0.0, 0.99, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.9,
               help="递归垂直色度反馈，说明见 format 层的同名参数。"
                    "**播放端也需要它**，因为哪个阶段做分离取决于接线；"
                    "参数只放在 format 层而播放端读自己的层，设置就会静默失效——"
                    "这个项目在 separation_kind 上已经犯过一次同样的错。")
    ply.define("comb_depth", "梳状滤波深度", 0.55, 0.0, 1.0, unit="0-1", group=g)
    ply.define("cross_color", "交叉色彩", 0.45, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="亮度串入色度（点爬行）")
    ply.define("cross_luma", "交叉亮度", 0.30, 0.0, 1.5, unit="0-1", group=g,
               phys_max=0.8, source="色度串入亮度")
    ply.define("color_noise", "色度噪声", 0.25, 0.0, 1.0, unit="0-1", group=g)
    ply.define("burst_phase_error", "色同步相位误差", 5.0, 0.0, 45.0, unit="deg", group=g,
               phys_max=20.0)
    ply.define("output_gain", "输出增益", 0.0, -1.0, 1.0, unit="0-1", group=g,
               phys_min=-0.3, phys_max=0.3)
    ply.define("output_bias", "黑电平偏移", 0.0, -0.5, 0.5, unit="0-1", group=g,
               phys_min=-0.15, phys_max=0.15, source="直流恢复/黑电平设定")

    g = "本机输出（无信号时）"
    ply.define("blue_screen", "蓝屏/OSD 替换", 0.0, 0.0, 1.0, unit="bool", group=g,
               step=1.0, phys_max=1.0,
               source="JVC SR-S388E 实测：信号过差时丢弃磁带画面并输出自身蓝底白字",
               help="这不是「画面变差」，而是**机器放弃显示磁带**并输出自己的图形。"
                    "实测参考素材的蓝底帧饱和度 0.8947（纯蓝场几乎全是色度），"
                    "而本工具生成的蓝屏饱和度 0.8947，两者吻合。"
                    "与任何程度的噪声都不同：噪声仍然是在显示磁带，蓝屏则不是。")
    ply.define("blue_osd", "OSD 内容", 0.0, 0.0, 2.0, unit="0-2", group=g,
               step=1.0, source="JVC SR-S388E 机内字符发生器",
               help="0=NO SIGNAL｜1=SEARCH / CHAPTER｜2=PLAY。"
                    "真实的字符发生器只有固定的若干条信息，不接受任意文字，"
                    "所以这里也用编号选择而不是自由输入。")

    g = "同步与时钟系统 (Sync & Clock)"
    ply.define("sync_clip", "同步脉冲压缩/削波 (Sync Clip)", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.75,
               source="复合视频 -40 IRE 同步顶幅度衰减与同步分离比较器门限偏移 (LM1881 / TA8690)",
               help="建模 -40 IRE 行/场同步脉冲被磁带饱和、多代翻录或电源压降削波压缩。同步顶变浅会导致：1. 垂直消隐后行 AFC 锁相环恢复变慢，产生画面顶部弯钩畸变 (Flagging)；2. 高亮扫描行通过交流耦合电容压低同步顶（Sync Crushing），造成亮部水平撕裂与色同步选通门错位；3. 超过门限时触发场失锁滚屏。")
    ply.define("h_hold", "行同步保持失调 (H-Hold)", 0.0, -1.0, 1.0, unit="-1..1", group=g,
               phys_min=-0.45, phys_max=0.45,
               source="行频压控振荡器 (H-VCO 15.625/15.734 kHz) 自由振荡频偏与鉴相器牵引范围",
               help="调节行频振荡器自由谐振频率偏置。小幅偏移在画面顶部产生经典的行 AFC 牵引弯钩 (Flagging)；超出锁相保持范围时行同步彻底失锁，画面斜向撕裂并暴露行消隐带 (HBI)。")
    ply.define("v_hold", "场同步保持失调 (V-Hold)", 0.0, -1.0, 1.0, unit="-1..1", group=g,
               phys_min=-0.50, phys_max=0.50,
               source="场频振荡器 (50/59.94 Hz) Adler 注入锁定方程与失锁垂直滚屏",
               help="调节场频振荡器自由周期偏置。当偏置或同步削波突破锁定门限时，画面按 Adler 注入锁定动力学发生非匀速纵向滚屏（接近锁相点时减速犹豫、滑过时加速），并在上下场交界处完整显现包含均衡脉冲、场同步齿脉冲与 VITC 时间码的物理垂直消隐带 (VBI Bar)。")
    ply.define("clock_drift_ppm", "采样时钟频偏 (Clock Drift)", 0.0, -500.0, 500.0, unit="ppm", group=g,
               phys_min=-150.0, phys_max=150.0,
               source="收发两端石英晶振频偏 (ITU-R BT.601 ±50 ppm 容限与廉价采集卡晶振老化)",
               help="录制端与回放/采集端采样时钟的百万分率频偏 (ppm)：引起逐行累积的水平比例伸缩斜切、副载波色同步相位沿扫描线连续漂移，以及大频偏下数字 FIFO / TBC 缓冲区的周期性滑帧跳变。")
    ply.define("clock_jitter", "采样时钟相位抖动 (Clock Jitter)", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.60,
               source="PLL 锁相环相位噪声与 ADC 采样孔径抖动",
               help="高频采样时钟相位噪声与孔径抖动，造成边缘亚像素毛刺与色度副载波高频相位噪波。")
    ply.define("std_mismatch", "跨制式失配 (PAL↔NTSC)", 0.0, 0.0, 2.0, unit="0-2", group=g, step=1.0,
               phys_min=0.0, phys_max=2.0,
               source="50Hz PAL (15625Hz/4.43MHz) 与 59.94Hz NTSC (15734Hz/3.58MHz) 异制式重放",
               help="0=制式正常匹配｜1=PAL (50Hz) 信号误入 NTSC (59.94Hz) 系统（9.94Hz 差拍场滚屏 + 行频斜切 + 4.43/3.58MHz 副载波失锁消色与彩虹斜纹）｜2=NTSC (59.94Hz) 信号误入 PAL (50Hz) 系统（反向场滚屏 + PAL 逐行倒相开关失配引发的汉诺威百叶窗色条 Hanover Bars）。")
    ply.define("pulse_cross", "脉冲十字监视 (Pulse Cross HBI/VBI)", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0,
               source="广播级监视器 (Sony PVM / Ikegami) Pulse-Cross 行场半周期延迟检视模式",
               help="开启后将行相位延迟半行 (W/2)、场相位延迟半场 (H/2) 并自动抬升消隐基底亮度，把真实的水平消隐带 (HBI：前肩、-40 IRE 行同步顶、后肩与 8–10 周期色同步脉冲串) 与垂直消隐带 (VBI：前后均衡脉冲、场同步阔脉冲开槽与 VITC 时间码) 移至画面正中央十字交叉处，供直观检视同步与色同步波形。")

    g = "转录与复制工作站"
    ply.define("transfer_station_mode", "转录/复制工作站模式", 0.0, 0.0, 4.0, unit="mode", group=g, step=1.0,
               phys_min=0.0, phys_max=4.0,
               source="模拟数字化采集站 (Ingest) / 多代磁带复制站 (Dubbing) / 数字格式转码站 (Transcode) / 四路复制链对照实验室",
               help="0=关闭｜1=A.模拟磁带数字化采集站 (VHS/Video8/Betacam → TBC → BT.601 采集卡 ADC，保留源缺陷并建模采样锁定、HBI 裁切与音画时钟漂移)｜"
                    "2=B.磁带多代复制站 (源机 → 分配/TBC → 目标机 1..8 代复制，区分可被 TBC 抑制的时基抖动与不可逆累积的带宽/色度/底噪非线性退化)｜"
                    "3=C.数字格式转码站 (4:2:2 → 4:1:1 DV25 → 4:2:0 MPEG-1 VCD 多代色度抽样交集损失与 8×8 DCT 重编码块伪影)｜"
                    "4=四路复制链对照实验室 (2×2 同屏对比：Gen-0 母带、无 TBC 多代模拟复制、带 TBC 多代模拟复制、数字跨格式转码 + TransferEvent 溯源遥测)。")
    ply.define("transfer_generations", "转录/复制累计代数 (Gen Index)", 3.0, 1.0, 8.0, unit="代", group=g, step=1.0,
               phys_min=1.0, phys_max=6.0,
               source="TransferEvent.generation_index (1..8 代连续复制或转码)",
               help="信号经历的连续磁带复制或数字重编码代数。每一代均生成独立的 TransferEvent 历史记录。")
    ply.define("transfer_tbc_mode", "转录链路时基校正模式", 1.0, 0.0, 2.0, unit="mode", group=g, step=1.0,
               phys_min=0.0, phys_max=2.0,
               source="TransferEvent.timebase_mode (无 TBC 自由运行 / 行 TBC / 全帧同步 TBC + Genlock)",
               help="0=无 TBC (Free-Run：采集卡 PLL 随磁带行抖动滑移，多代复制时行时基误差逐代加速发散)｜"
                    "1=行 TBC (Line TBC：逐行重锁定水平同步与色同步相位，大幅抑制几何扭曲，但带宽滚降与色度退化照常累积)｜"
                    "2=全帧同步 TBC + Genlock (Full-Frame Synchronizer：锁定行场时基与音频采样基准)。")
    ply.define("transfer_codec_target", "数字转码目标格式链", 3.0, 0.0, 3.0, unit="mode", group=g, step=1.0,
               phys_min=0.0, phys_max=3.0,
               source="ITU-R BT.601 4:2:2 / IEC 61834 DV25 4:1:1 / ISO 11172-2 MPEG-1 VCD 4:2:0",
               help="0=BT.601 YUV422 8-bit 采集直出｜1=DV25 (4:1:1 横向 1/4 色度抽样 + 25Mbps 8×8 DCT)｜"
                    "2=MPEG-1 VCD (352×240 CIF + 4:2:0 色度 + 1.15Mbps 运动补偿)｜"
                    "3=4:2:2 → DV25 (4:1:1) → MPEG-1 VCD (4:2:0) 跨格式级联转码（色度分辨率跌至 1/4×1/2 = 1/8）。")
    ply.define("transfer_adc_crop", "采集卡消隐裁切与 ADC 钳位", 0.45, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.10, phys_max=0.85,
               source="BT878 / SAA7134 / TC-UB530 采集卡抗混叠滤波器群延迟振铃与 BT.601 16..235 量化范围钳位",
               help="控制数字化采集站左右行消隐 (HBI) 窗口裁切边缘振铃，以及 8-bit 16..235 量化动态范围截断强度。")
    ply.define("transfer_av_drift_ms", "音画时钟同步漂移 (A/V Sync Offset)", 35.0, -250.0, 250.0, unit="ms", group=g,
               phys_min=-120.0, phys_max=120.0,
               source="独立声卡 48kHz 晶振与视频采集卡行锁时钟未同步导致的音画滑移 (TransferEvent.audio_sync_offset)",
               help="每次转录或复制引入的音画同步时间偏移（毫秒），随复制代数累积并同步反映在伴音时间轴与 TransferEvent 记录中。")
    ply.define("transfer_quad_corner", "复制实验室角落选择 (0四分屏/1左上/2右上/3左下/4右下)", 0.0, 0.0, 4.0, unit="角落", group=g,
               step=1.0, phys_min=0.0, phys_max=4.0,
               source="四路复制链对照角落拆选",
               help="在四路复制对照模式（transfer_station_mode=4）下拆成每个角落单独全屏选择：0=2×2四分屏，1=左上角（Gen-0母带），2=右上角（路径A无TBC多代模拟复制），3=左下角（路径B带TBC多代模拟复制），4=右下角（路径C数字转码链）。")
    ply.define("transfer_show_border", "复制实验室分框样式 (0无分框/1细十字线/2标准分框)", 1.0, 0.0, 2.0, unit="分框", group=g,
               step=1.0, phys_min=0.0, phys_max=2.0,
               source="分屏分框显示控制",
               help="0=无分框无缝拼接，1=细十字分框线，2=标准分框。")
    ply.define("transfer_show_text", "复制实验室画面文字 (0不要文字/1仅标签/2标签+状态栏)", 0.0, 0.0, 2.0, unit="文字", group=g,
               step=1.0, phys_min=0.0, phys_max=2.0,
               source="纯净画面无文字开关",
               help="默认 0=不要文字（纯净画面）；1=仅显示四角标签；2=显示四角标签与底部统计栏。")

    # ------------------------------------------------------------------ E
    dsp = s.layer("display")
    g = "显像管"
    dsp.define("enabled", "显示设备接入", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0)
    # **The minimum was 1.5, which made the parameter one-sided.**  A gamma is a curve that can go either
    # way, and the stage applies `rgb ** (1/g)`, so a value below 1 darkens the mid-tones.  Measured, asking
    # for 0.50 / 0.80 / 1.00 / 1.25 all left the stage holding **1.5** because the schema raised them, so
    # the lower half of the range the name implies was unreachable rather than mis-implemented.
    dsp.define("gamma", "伽马", 2.35, 0.4, 3.2, unit="x", group=g,
               phys_min=0.4, phys_max=2.8, source="消费级 CRT 典型 2.2-2.4")
    dsp.define("contrast", "对比度", 0.55, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("brightness", "亮度", 0.5, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("saturation", "饱和度", 0.5, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("scanlines", "扫描线", 0.35, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("interlace_flicker", "隔行闪烁", 0.25, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("geometry_warp", "几何形变", 0.15, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("trapezoid_x", "梯形（宽）", 0.0, -0.5, 0.5, unit="0-1", group=g,
               phys_min=-0.15, phys_max=0.15,
               help="**偏转线圈的扫描与显像管端面不成直角**，所以上边缘与下边缘的宽度可以不等。"
                    "这就是梯形失真（keystone），磁轭被撞歪的机器上很明显。正值下宽上窄。"
                    "和桶形/枕形畸变是**互相独立**的两种故障。")
    dsp.define("trapezoid_y", "梯形（高）", 0.0, -0.5, 0.5, unit="0-1", group=g,
               phys_min=-0.15, phys_max=0.15,
               help="左右两侧高度不等。与横向梯形是两个独立的故障，所以是两个参数。")
    dsp.define("convergence", "会聚误差", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.6,
               help="**三个电子束从三支枪以不同角度出发、穿过荫罩**，所以红绿蓝三幅像的大小"
                    "并不完全一致。误差在**中心为零、向边缘增大**——这就是真实显像管"
                    "**边缘有彩色镶边、中心却干净**的原因。模型把各通道绕中心做略微不同的缩放，"
                    "而不是整体平移：整体平移会让整幅画面均匀镶边，那是廉价近似的做法。")
    dsp.define("curvature", "管面曲面（鱼眼）", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.8,
               help="**显像管端面是球面**，所以画面绕中心被放大、**边缘向外鼓出**。"
                    "映射是随半径增大的径向放大：**中心几乎不动，边缘明显**——"
                    "这正是「鱼眼」和「整体缩放」的区别。和梯形（磁轭歪）是两个独立故障："
                    "一台机器可以管面是弯的、而磁轭对得很正。")
    dsp.define("spot_sigma", "电子束光点", 0.42, 0.10, 2.5, unit="行", group=g,
               phys_min=0.25, phys_max=1.0,
               help="**扫描线不是画上去的暗条纹，是电子束两次扫过之间的空隙**，"
                    "而空隙的形状取决于**光点的有限尺寸**。这里是高斯光点的宽度（单位：行）。"
                    "光点小于行距时空隙变黑、线条清晰；光点接近行距时相邻两次扫描重叠、"
                    "**扫描线结构就消失了**——**这就是调焦会改变扫描线可见度的原因**，"
                    "而固定周期的余弦永远做不出这件事。")
    dsp.define("scan_pitch", "扫描行距", 2.0, 1.05, 6.0, unit="px", group=g,
               phys_min=1.05, phys_max=3.0,
               help="两次扫描之间的像素距离。**这是显示器的属性，不是信号的**："
                    "480 行的画面并不必然对应 480 像素的面板。")
    dsp.define("mask_pitch", "荫罩节距", 0.0, 0.0, 8.0, unit="px", group=g,
               phys_min=0.0, phys_max=4.0,
               help="**荫罩是彩色显像管里那块金属栅格，它的影子就印在玻璃上。**"
                    "这不是只有水平方向的效果：栅格的图案是**竖直**的，与扫描线成直角。"
                    "缺了它，显像管看起来就只是「一幅画加几条横线」。节距是**玻璃的属性、"
                    "不是信号的**，所以同一幅画面在更细的管子上会显示更细的栅格。"
                    "0 表示关闭。")
    dsp.define("mask_type", "荫罩型式", 0.0, 0.0, 2.0, unit="0-2", group=g, step=1.0,
               phys_min=0.0, phys_max=2.0,
               help="0 **栅条式**（特丽珑：连续的竖直细线）；"
                    "1 **槽孔式**（竖直缝交替错开，呈砖块状）；"
                    "2 **点孔式**（最常见的荫罩，孔在错开的二维点阵上）。"
                    "**图案本身就是管子的身份**，所以这是三个选项而不是一个强度。")
    dsp.define("mask_depth", "荫罩深度", 0.35, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.1, phys_max=0.6,
               help="栅格影子的深浅。真实的取值取决于管子多亮、以及观看距离；"
                    "它**不改变画面平均亮度**（掩膜按均值归一化），"
                    "否则这个参数就成了名字骗人的亮度控制。")
    dsp.define("breathing", "高压呼吸", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="**末级阳极由行输出变压器供电，而它的调整率是有限的。**"
                    "画面越亮、束电流越大、电压越塌，而**偏转灵敏度随阳极电压下降而上升**，"
                    "于是**画面变大**。所以从暗场景切到亮场景的瞬间，画幅会胀一下。"
                    "这是一个**动态**故障，所以它读的是当前帧的亮度而不是一个设定值——"
                    "也是第一个逐帧变化的几何项。")
    dsp.define("overscan", "过扫描裁切", 0.0, 0.0, 0.15, unit="比例", group=g,
               phys_min=0.0, phys_max=0.10, source="家用电视典型 5%-10% 过扫描",
               help="画面被放大后越出显像管边框的比例。真实的过扫描会把画面底部那条"
                    "磁头切换带推出可见区（每边 4 行以上就明显削弱，12 行以上基本看不见）"
                    "——而这条带正是 VHS 最有辨识度的痕迹。默认 0 以便看得见它。")
    dsp.define("vignette", "暗角", 0.25, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("bloom", "泛光", 0.35, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("defocus", "散焦", 0.20, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("reflection", "屏幕反射", 0.10, 0.0, 1.0, unit="0-1", group=g)
    dsp.define("color_temp", "色温", 0.5, 0.0, 1.0, unit="0-1", group=g,
               source="冷/暖白平衡")

    g = "CRT 接口受潮"
    dsp.define("crt_moisture", "显像管受潮", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5, source="尾部接口与视放电路受潮",
               help="CRT 的信号通路极短且全模拟：接头 → 视放 → 阴极，偏转由轭驱动。"
                    "受潮影响的是**视放电路**，不是到达它的信号，"
                    "所以产生的故障和磁带损伤完全不同。")
    dsp.define("crt_corrosion", "CRT 接口氧化", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.5,
               help="接触电阻升高并向地漏电，改变视放偏置，**黑电平整体移动**——"
                    "画面变亮或变暗，而不是变脏。所以受潮的机器看起来是**发白**，"
                    "不是发噪。")
    dsp.define("crt_age", "显像管老化", 0.3, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.8,
               help="三个阴极各有独立视放、各自老化，所以旧机器受潮时出现**偏向某色**"
                    "而不只是变暗。同一个潮湿房间里，新机器看起来发白、旧机器发绿，"
                    "差别就在这里——这是**显示器**的属性，不是录制里的偏色。")
    dsp.define("spot_bloom", "束斑动态发散", 0.0, 0.0, 1.5, unit="0-1", group="电子光学与荧光粉",
               phys_min=0.0, phys_max=1.0,
               help="强阴极束流下空间电荷排斥使电子束斑变宽，高光处扫描线向黑隙扩张并桥接，"
                    "暗部仍保持清晰扫描黑缝——这是真实 CRT 与静态正弦条纹滤镜的核心区别。")
    dsp.define("phosphor_rgb", "RGB 荧光粉分色度", 0.0, 0.0, 1.0, unit="0-1", group="光栅与荫罩",
               phys_min=0.0, phys_max=1.0,
               help="0 为亮度荫罩，>0 为真实红/绿/蓝独立亚像素条纹或点阵（含黑矩阵与特丽珑水平阻尼丝阴影）。")
    dsp.define("glass_halation", "玻壳内反射光晕", 0.0, 0.0, 1.2, unit="0-1", group="电子光学与荧光粉",
               phys_min=0.0, phys_max=0.8,
               help="荧光粉发光在厚铅玻璃面板内发生全反射，在强高光周围形成一圈宽广柔和的二次反射光环（Halation Ring）。")
    dsp.define("phosphor_saturation", "荧光粉发光饱和", 0.0, 0.0, 1.2, unit="0-1", group="电子光学与荧光粉",
               phys_min=0.0, phys_max=0.8,
               help="P22 荧光粉在强电子束轰击下发光效率趋于饱和，高光核心向炽白退饱和。")
    dsp.define("corner_defocus", "边角像散失焦", 0.0, 0.0, 1.2, unit="0-1", group="光栅与荫罩",
               phys_min=0.0, phys_max=0.8,
               help="电子束斜射入屏幕四角时偏转线圈产生像散，使边角束斑沿切向/径向拉长失焦，中心仍保持锐利。")
    dsp.define("convergence_y", "垂直会聚误差", 0.0, -1.0, 1.0, unit="0-1", group="光栅与荫罩",
               phys_min=-0.5, phys_max=0.5,
               help="三支电子枪在垂直方向的会聚偏离，随距中心高度增加而在上下边缘产生红/蓝彩色镶边。")
    dsp.define("ehv_sag", "行高压负载下垂", 0.0, 0.0, 1.2, unit="0-1", group="光栅与荫罩",
               phys_min=0.0, phys_max=0.8,
               help="单行扫描遇到大面积高光时行输出高压（EHT）瞬态下垂，阳极电压下降使该行偏转灵敏度升高、向左右两侧外扩。")
    dsp.define("phosphor_persistence", "荧光粉余辉", 0.0, 0.0, 0.9, unit="0-1", group="电子光学与荧光粉",
               phys_min=0.0, phys_max=0.65,
               help="P22 荧光粉各通道衰减时间常数不同（红粉 Y2O2S:Eu 余辉慢于绿/蓝），运动高光拖出微弱暖色余辉。")
    dsp.define("bezel", "球面框体与边角遮罩", 0.0, 0.0, 1.0, unit="0-1", group="环境与框体",
               phys_min=0.0, phys_max=0.8,
               help="显像管超椭圆黑框、内侧暗角阴影与球面玻璃菲涅尔边缘高光。")

    g = "屏摄重拍系统 (Camera-off-CRT Reshoot)"
    dsp.define("reshoot_enabled", "启用虚拟相机屏摄重拍", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0,
               source="用虚拟 CMOS/CCD 相机重新拍摄正在扫描发光的 CRT 荧光屏（快门积分窗口 vs 垂直场扫描脉冲）",
               help="开启后，不再直接输出理想整帧光栅，而是用一台具备独立快门速度、帧率与卷帘读出时序的虚拟相机对 CRT 荧光屏进行二次拍摄："
                    "快门曝光窗口 [t, t+τ_shutter] 与 CRT 50/59.94Hz 电子束逐行轰击及 P22 荧光粉毫秒级指数余辉发生时域差拍，"
                    "自然生成相机屏摄特有的明暗滚动扫描横带（Roll Bar）。")
    dsp.define("reshoot_shutter_hz", "重拍相机快门倒数 (1/T)", 60.0, 20.0, 500.0, unit="Hz", group=g,
               phys_min=25.0, phys_max=250.0,
               source="相机快门曝光时间 τ_shutter = 1 / reshoot_shutter_hz（如 1/60s、1/100s、1/125s、1/250s）",
               help="快门速度越快（如 125Hz 即 1/125s），单次曝光内电子束仅扫过屏幕的一部分（D = f_CRT / f_shutter < 1），"
                    "形成明显的亮带与未扫到的暗谷；当快门慢于场周期（如 1/50s 拍 59.94Hz NTSC）时，局部区域在一帧内被电子束扫过两次，"
                    "形成一条二次曝光亮横带。")
    dsp.define("reshoot_fps_offset", "相机与场频差拍偏置 (Δf)", 0.60, -15.0, 15.0, unit="Hz", group=g,
               phys_min=-10.0, phys_max=10.0,
               source="重拍相机帧率 f_cam 与 CRT 实际场频 f_CRT 的微差（如 60.00Hz 手机拍摄 59.94Hz NTSC 电视）",
               help="决定屏摄扫描横带在画面上纵向滚动的速度与方向。Δf = 0 时扫描带静止在固定行位置；"
                    "Δf > 0 或 < 0 时扫描横带以 |Δf| Hz 的差拍周期平滑向上或向下滚动。")
    dsp.define("reshoot_rolling_shutter", "CMOS 卷帘快门读出斜率", 0.35, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.85,
               source="CMOS 传感器逐行曝光读出延迟 τ_readout（0=CCD 全局快门，>0=CMOS 卷帘快门）",
               help="0.0 为 CCD 全局快门（整幅画面同一毫秒开启快门）；>0 为手机/数码相机 CMOS 卷帘快门（自上而下逐行开启曝光），"
                    "与同样自上而下扫描的 CRT 电子束发生多普勒式收窄/展宽效应。")
    dsp.define("reshoot_moire", "像素-荫罩二维摩尔纹", 0.0, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=1.0,
               source="相机传感器拜耳像素网格与球面 CRT 荧光粉荫罩/扫描线重采样空间频率混叠 (Δk = k_CRT - k_sensor)",
               help="当相机镜头对准带有球面曲率与垂直荫罩（特丽珑荫栅 / 槽孔 / 三角点阵）的 CRT 屏幕时，"
                    "球面透视放大率变化使屏幕空间频率沿径向连续漂移，与相机拜耳 CFA 采样网格差拍产生标志性的彩色环形/双曲水波摩尔纹。")
    dsp.define("reshoot_tilt_deg", "重拍机位旋转倾角", 0.0, -15.0, 15.0, unit="deg", group=g,
               phys_min=-8.0, phys_max=8.0,
               source="手持或三脚架相机相对电视机水平扫描线的滚转角 θ_tilt",
               help="微小机位倾角（如 1.5°–4°）会改变相机像素列与 CRT 竖直荫罩条纹的夹角，使摩尔纹条纹的周期与旋转方向发生灵敏变化。")
    dsp.define("reshoot_perspective", "屏摄离轴透视与微距景深", 0.0, -1.0, 1.0, unit="-1..1", group=g,
               phys_min=-0.6, phys_max=0.6,
               source="相机微偏离屏幕法线拍摄时的三维单应性透视梯形畸变、电视边框入镜与远端浅景深虚化",
               help="模拟从略微偏左/偏右机位近距离翻拍电视屏幕：引入三维透视近大远小畸变、露出显示器深灰塑料外框边缘，并在远离焦平面的一侧产生微距浅景深柔化。")
    dsp.define("reshoot_ambient_glare", "房间环境光与荧光粉漫反", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.75,
               source="未激发 P22 荧光粉涂层（灰白粉末反射率 ~25%）对室内环境光的漫反射黑位抬升与曲面玻璃灯具倒影",
               help="CRT 在开灯房间内无法呈现纯黑：未发光的灰白色荧光粉层会漫反射室内环境光，把屏幕黑位冲淡成灰绿/暖灰底色，"
                    "同时凸面铅玻璃表面映出窗户或吸顶灯的菲涅尔高光倒影。")
    dsp.define("reshoot_wb_shift", "屏摄 9300K 色温偏移与过曝", 0.0, -1.0, 1.0, unit="-1..1", group=g,
               phys_min=-0.5, phys_max=0.85,
               source="中日消费级彩电 9300K (D93) 荧光粉白点在 5200K 相机白平衡下的冷青偏色与点测光中心过曝",
               help="正值模拟 5200K 日光/室内白平衡相机拍摄 9300K 冷白 CRT 荧光屏时的典型冷青蓝偏色与高光自动曝光溢出；负值为暖调偏移。")

    g = "多画面监视墙与独立 CRT 阵列"
    dsp.define("mv_enabled", "启用演播室多画面监视墙", 0.0, 0.0, 1.0, unit="bool", group=g,
               step=1.0, phys_min=0.0, phys_max=1.0,
               source="90年代电视台导播室 / 总控机房 (MCR) 多路 CRT 监视器墙",
               help="开启后将显示阶段切换为演播室多路 CRT 监视墙，同屏监看「摄像机/信号源 (CH1)」、「磁带原始重放 TBC Off (CH2)」、"
                    "「TBC 时基校正输出 (CH3)」与「主监播出 PGM (CH4)」，或节点图接入的 4–6 路独立信号源。")
    dsp.define("mv_layout", "监视墙布局模式", 0.0, 0.0, 2.0, unit="0-2", group=g,
               step=1.0, phys_min=0.0, phys_max=2.0,
               source="演播室 2×2 四分屏 / 总控 3×2 六分屏 / 导播 1+3 主副监看墙",
               help="0=2×2 导播室四分屏监视墙 (CAM / VTR-RAW / TBC-OUT / PGM)｜"
                    "1=3×2 总控机房六画面监视桥（含配线架与绿粉/单色波形监看管）｜"
                    "2=1+3 导播主副非对称监视墙（左侧大屏主监 PGM + 右侧三路预监 PVW）。")
    dsp.define("mv_unit_variance", "各监视器独立 CRT 个体老化差异", 0.55, 0.0, 1.2, unit="0-1", group=g,
               phys_min=0.15, phys_max=0.95,
               source="SONY PVM-14L5 (D65) / JVC TM-H150CG (欠扫描) / Ikegami 老化绿偏管 / 家用 9300K 彩电同墙个体差",
               help="关键物理区分：监视墙上每一台 CRT 拥有独立的阴极老化色温（D65 标准、绿枪偏高、9300K 冷蓝或 单色黑白同步监看管）、"
                    "独立伽马、独立几何枕形/欠扫描边框与高压呼吸——监视器自身的偏色只改变你在该窗口看到的外观，绝不污染底层视频信号本身。")
    dsp.define("mv_fault_channel", "注入通道失锁/断流故障槽位", 0.0, 0.0, 6.0, unit="0-6", group=g,
               step=1.0, phys_min=0.0, phys_max=6.0,
               source="总控监视墙单通道断流/失锁演练",
               help="0=全部通道正常｜1..6=指定第 1..6 号监视器窗口发生信号中断或同步失锁（触发下方设定的硬件失锁物理响应，并点亮红色闪烁 FAULT Tally 灯）。")
    dsp.define("mv_fault_mode", "通道无信号/失锁物理响应模式", 0.0, 0.0, 5.0, unit="0-5", group=g,
               step=1.0, phys_min=0.0, phys_max=5.0,
               source="各类模拟/数字监视器与录像机在丢失同步脉冲时的真实硬件行为",
               help="0=按通道设备类型自动分配｜1=纯黑无同步底噪 (Black No-Sync)｜2=RF 高频头 AGC 满增益白雪花 (RF Snow)｜"
                    "3=场行自由振荡斜向滚屏消隐带 (Free-Run Roll)｜4=数字 TBC 末帧死锁残影冻结 (Frame Freeze)｜"
                    "5=JVC/松下录像机本机蓝屏 + NO SIGNAL 字符替换 (VTR Blue OSD)。")
    dsp.define("mv_tally_umd", "显示监视墙文字标牌与 Tally (0不要文字/1显示)", 0.0, 0.0, 1.0, unit="bool", group=g,
               step=1.0, phys_min=0.0, phys_max=1.0,
               source="演播室监视桥顶部 Tally 灯箱与底部 UMD 源名显示器（默认关闭以保持纯净无文字画面）",
               help="默认 0=不要文字与标牌（纯净多画面）；1=在每台监视器外框顶部点亮 Tally 灯并在底部内嵌 UMD 通道铭牌。")
    dsp.define("mv_quad_corner", "监视墙单窗角落选择 (0全墙/1左上/2右上/3左下/4右下)", 0.0, 0.0, 6.0, unit="窗口", group=g,
               step=1.0, phys_min=0.0, phys_max=6.0,
               source="监视墙单路窗口拆选",
               help="0=完整多画面监视墙；1..4=单独全屏查看第 1..4 号角落窗口（3×2 模式下支持 1..6）。")
    dsp.define("mv_show_border", "监视墙分框样式 (0无分框/1细十字线/2机架边框)", 1.0, 0.0, 2.0, unit="分框", group=g,
               step=1.0, phys_min=0.0, phys_max=2.0,
               source="监视墙分框与金属隔断控制",
               help="0=无分框无缝拼接，1=细十字分框线，2=演播室机架金属边框。")

    g_prc = "人类视觉感知与机房操作者闭环"
    dsp.define("percept_enabled", "启用人类视觉感知与机房操作者闭环系统", 0.0, 0.0, 1.0, unit="bool", group=g_prc, step=1.0,
               phys_min=0.0, phys_max=1.0, source="Mannos-Sakrison 空间对比敏感度函数 (CSF) + 机房值班员 PID 认知控制模型",
               help="量化『设备物理输出 ≠ 观众主观视觉体验』，并模拟机房值班工程师基于波形监视器与报警做出手动跟踪微调、CCU 配平与高压误切的行为闭环。")
    dsp.define("percept_mode", "视觉感知与操作者行为模式", 0.0, 0.0, 4.0, unit="0-4", group=g_prc, step=1.0,
               phys_min=0.0, phys_max=4.0, source="心理物理学 CSF 滤波与人因控制回路",
               help="0=跟随总开关｜1=观众心理物理感知视图 (基于观看距离 CPD、屏幕尼特与环境照度的 CSF 空间滤波，解释为何远看 CRT 比近看像素截图更清晰柔和)｜"
                    "2=机房值班员闭环矫正后画面 (操作者实时监看并推拉 Tracking 旋钮与 CCU 增益，含认知反应延迟、欠阻尼过调震荡与高压误切事故)｜"
                    "3=物理误差 vs 人眼感知显著度 (JND) 差异热力图｜"
                    "4=2×2 感知与操作者行为闭环实验室 (同屏对比：未干预物理输出 vs 观众视网膜 CSF 感知画面 vs 操作者闭环矫正画面 vs JND 显著度与评分面板)。")
    dsp.define("percept_viewing_dist_h", "观众观看距离 (屏幕高度倍数 H)", 3.5, 1.0, 10.0, unit="×H", group=g_prc,
               phys_min=1.5, phys_max=7.0, source="ITU-R BT.500 标准电视观看距离 (客厅 3.5~6H / 桌面近看 1.2~2H)",
               help="决定每视角一度对应的空间频率 (CPD)：距离越远 CPD 越高，人眼 CSF 自动滤除扫描线、荫罩与高频彩噪，提升主观清晰度与舒适度。")
    dsp.define("percept_display_nits", "显示器峰值绝对亮度", 120.0, 20.0, 600.0, unit="cd/m²", group=g_prc,
               phys_min=60.0, phys_max=350.0, source="SMPTE RP-166 演播室监视器 100 nits / 家用彩电 150~250 nits",
               help="影响人眼明视觉/暗视觉对比敏感度峰值与 Ferry-Porter 临界闪烁融合频率 (CFF)。")
    dsp.define("percept_ambient_lux", "观看环境照度", 45.0, 1.0, 1000.0, unit="lux", group=g_prc,
               phys_min=5.0, phys_max=400.0, source="暗室 5 lux / 演播室 30 lux / 客厅日光 250 lux",
               help="环境光在显像管玻璃表面的漫反射会抬升视觉黑位 (Veiling Glare) 并改变暗部噪波的感知可见度。")
    dsp.define("operator_skill", "机房操作者熟练度", 0.78, 0.10, 1.0, unit="0-1", group=g_prc,
               phys_min=0.35, phys_max=0.98, source="广播机房值班工程师人因工程参数",
               help="熟练工程师能快速平滑收敛跟踪误差与色偏；新手则反应迟缓且容易在最佳点两侧反复过调猎振 (Overshoot Hunting)。")
    dsp.define("operator_stress", "机房时间压力与疲劳度", 0.35, 0.0, 1.0, unit="0-1", group=g_prc,
               phys_min=0.05, phys_max=0.85, source="Yerkes-Dodson 压力唤醒与误操作概率模型",
               help="高直播压力与多路报警并发时，操作者增益激进导致二阶欠阻尼过调震荡，甚至偶发按错切换台交叉点的误切事故 ( Panic Mis-cut )。")
    dsp.define("percept_quad_corner", "感知实验室画面角落选择 (0四分屏/1左上/2右上/3左下/4右下)", 0.0, 0.0, 4.0, unit="角落", group=g_prc,
               step=1.0, phys_min=0.0, phys_max=4.0, source="2×2 四分屏单角全屏拆选",
               help="0=显示完整 2×2 四分屏对照；1=左上角（未干预物理输出）；2=右上角（观众视网膜 CSF 感知画面）；3=左下角（操作者闭环矫正画面）；4=右下角（JND 感知显著度与人因轨迹面板）。")
    dsp.define("percept_show_border", "感知实验室分框样式 (0无分框/1细十字线/2标准分框)", 2.0, 0.0, 2.0, unit="分框", group=g_prc,
               step=1.0, phys_min=0.0, phys_max=2.0, source="四分屏分框线显示控制")
    dsp.define("percept_show_text", "感知实验室文字叠加 (0不要文字/1仅标签/2完整遥测)", 0.0, 0.0, 2.0, unit="文字", group=g_prc,
               step=1.0, phys_min=0.0, phys_max=2.0, source="画面文字叠加开关（默认 0 纯净无字画面）")

    g_ldme_lcd = "早期 LCD 液晶面板与 Flash 播放器 (LDME)"
    dsp.define("ldme_display_enabled", "启用早期 LCD 与 Flash 播放器模型", 0.0, 0.0, 1.0, unit="bool", step=1.0, group=g_ldme_lcd,
               phys_min=0.0, phys_max=1.0, source="LDME 子系统 E (05 · 播放与显示，严格区分 LCD 与 CRT)",
               help="开启早期手机 TFT 液晶或 2007 桌面 6-bit TN 液晶显示器与 Flash Player 窗口缩放/缓冲模型（不混入 CRT 扫描线）。")
    dsp.define("ldme_display_device", "数字显示设备类型", 0.0, 0.0, 3.0, unit="设备", step=1.0, group=g_ldme_lcd,
               phys_min=0.0, phys_max=3.0, source="手机 RGB565 TFT LCD vs 桌面 6-bit TN LCD vs CRT 对照",
               help="0=关闭｜1=早期手机 TFT 液晶屏 (176×208 RGB565 65K色 + 28ms慢响应 + 像素晶格)｜"
                    "2=2007 桌面 6-bit TN 液晶显示器 (64级+FRC抖动 + 液晶保持拖影 + 视角伽马偏移 + CCFL冷白背光)｜3=经典 CRT 监视器同源对照。")
    dsp.define("ldme_lcd_bits", "LCD 面板通道位深", 6.0, 5.0, 8.0, unit="bit", step=1.0, group=g_ldme_lcd,
               phys_min=5.0, phys_max=8.0, source="5-bit (RGB565) / 6-bit TN + Hi-FRC / 8-bit")
    dsp.define("ldme_lcd_response_ms", "液晶分子响应时间", 16.0, 2.0, 45.0, unit="ms", group=g_ldme_lcd,
               phys_min=5.0, phys_max=35.0, source="Sample-and-Hold 液晶上升/下降时间拖影 (区别于 CRT 荧光粉余辉)")
    dsp.define("ldme_lcd_view_angle", "TN 面板垂直观看偏角", 12.0, -35.0, 35.0, unit="deg", group=g_ldme_lcd,
               phys_min=-25.0, phys_max=25.0, source="早期 TN 液晶垂直视角伽马偏移与暗部反相")
    dsp.define("ldme_lcd_subpixel", "LCD RGB 竖条子像素晶格强度", 0.35, 0.0, 1.0, unit="0-1", group=g_ldme_lcd,
               phys_min=0.0, phys_max=0.8, source="规整矩形 RGB 子像素黑矩阵网格 (无隔行扫描线)")
    dsp.define("ldme_lcd_ccfl", "早期 CCFL 背光冷偏与暗部漏光", 0.25, 0.0, 1.0, unit="0-1", group=g_ldme_lcd,
               phys_min=0.0, phys_max=0.8, source="CCFL 冷阴极荧光灯管背光光谱与液晶暗态漏光")
    dsp.define("ldme_flash_scale_mode", "Flash 播放器与浏览器缩放方式", 1.0, 0.0, 3.0, unit="缩放", step=1.0, group=g_ldme_lcd,
               phys_min=0.0, phys_max=3.0, source="Flash Player Stage.scaleMode 与非整数窗口拉伸",
               help="0=原始比例居中｜1=浏览器非整数倍双线性窗口放大｜2=最近邻硬像素块放大 (smoothing=false)｜3=4:3 视频强行全屏拉伸为 16:9。")
    dsp.define("ldme_flash_chrome", "显示早期 Flash 播放器控制栏与缓冲条", 0.0, 0.0, 1.0, unit="bool", step=1.0, group=g_ldme_lcd,
               phys_min=0.0, phys_max=1.0, source="Flash Player 9/10 底部进度条与缓冲区水位指示")
    dsp.define("ldme_net_throughput_kbps", "客户端网络下载吞吐量", 520.0, 40.0, 5000.0, unit="kbps", group=g_ldme_lcd,
               phys_min=64.0, phys_max=2500.0, source="当吞吐量低于视频平均码率时触发缓冲区耗尽与暂停重缓冲")

    # --- Ref: CRT 强磁场、铁磁荫罩剩磁、PTC 消磁线圈与特殊环境物理系统 ---
    g_mag = "CRT 强磁场与特殊环境 (强磁/剩磁/消磁线圈)"
    dsp.define("crt_magnet_mode", "CRT 强磁环境模式 (0关/1永磁体贴面/2旋转涡流/3撤磁剩磁+PTC消磁/4工频磁蛇/5四象限实验室)", 0.0, 0.0, 5.0, unit="模式", group=g_mag,
               step=1.0, phys_min=0.0, phys_max=5.0,
               source="强永磁体近场洛伦兹偏转 F=-e(v×B) 与铁磁荫罩剩磁物理标定",
               help="0=关闭；1=强钕铁硼永磁体贴近/按压屏幕（中心电子束排空黑洞 + 十字枕形内凹折叠 + 同心彩虹纯度环）；"
                    "2=强磁铁在屏面动态游走与极轴旋转（洛伦兹旋度涡流扭曲 + 彩虹色环实时旋转）；"
                    "3=撤走磁铁后的铁磁荫罩永久剩磁色斑 + 电视内置 PTC 热敏电阻消磁线圈（Degaussing Coil）衰减振荡消磁；"
                    "4=外部大功率变压器 50/60Hz 漏磁扭动 + 潮湿阳极高压电晕放电；"
                    "5=2×2 CRT 强磁场与消磁物理四象限对照实验室。")
    dsp.define("crt_magnet_strength", "永磁体表磁强度 (NdFeB Tesla)", 0.0, 0.0, 1.5, unit="T", group=g_mag,
               phys_min=0.0, phys_max=1.2,
               source="N52 钕铁硼强磁柱近场磁偶极子感应强度 (0.1–1.2 Tesla)",
               help="控制永磁体对 CRT 电子束的洛伦兹力偏转幅度、中心电子束排空黑洞半径以及同心色纯度彩虹环阶数。")
    dsp.define("crt_magnet_x", "磁铁屏面横坐标 X", 0.52, 0.05, 0.95, unit="0-1", group=g_mag,
               phys_min=0.1, phys_max=0.9, help="强磁铁在显像管玻壳正面的归一化水平位置。")
    dsp.define("crt_magnet_y", "磁铁屏面纵坐标 Y", 0.48, 0.05, 0.95, unit="0-1", group=g_mag,
               phys_min=0.1, phys_max=0.9, help="强磁铁在显像管玻壳正面的归一化垂直位置。")
    dsp.define("crt_magnet_z_dist", "磁铁距玻壳距离 Z", 0.14, 0.04, 0.60, unit="d/W", group=g_mag,
               phys_min=0.06, phys_max=0.45,
               help="磁铁距荧光屏的距离：越靠近（如 0.08），中心磁通密度按 1/r³ 剧增，形成边缘锐利的黑洞与多层彩虹环；拉远则呈宽缓枕形凹陷。")
    dsp.define("crt_magnet_angle_deg", "磁铁磁矩极轴倾角", 25.0, -180.0, 180.0, unit="deg", group=g_mag,
               phys_min=-180.0, phys_max=180.0,
               help="磁偶极子在屏幕平面内的极轴朝向角；旋转磁铁时十字内凹瓣与彩虹纯度环随之同步旋转。")
    dsp.define("crt_magnet_polarity", "磁铁极性 (N极/S极)", 1.0, -1.0, 1.0, unit="极性", group=g_mag,
               phys_min=-1.0, phys_max=1.0,
               help="+1.0 为 N 极朝向屏幕（顺时针洛伦兹旋度），-1.0 为 S 极朝向屏幕（逆时针洛伦兹旋度与反相色环）。")
    dsp.define("crt_magnet_motion", "磁铁动态逼近/旋转轨迹", 0.0, 0.0, 1.0, unit="0-1", group=g_mag,
               phys_min=0.0, phys_max=1.0,
               help="开启后模拟手持强磁铁从远处逼近屏幕、贴面旋转并划过屏幕的连续物理过程。")
    dsp.define("crt_magnet_remanence", "荫罩/荫栅铁磁剩磁色斑 (Remanence)", 0.0, 0.0, 1.0, unit="0-1", group=g_mag,
               phys_min=0.0, phys_max=0.9,
               source="低碳钢/殷钢荫罩 Jiles-Atherton 磁滞回线剩磁 M_r",
               help="强磁铁撤走后，铁磁性金属荫罩或支撑框被永久磁化留下的静态彩色纯度污斑（需启动消磁线圈方可消除）。")
    dsp.define("crt_degauss_trigger", "启动 PTC 消磁线圈 (Degaussing Coil)", 0.0, 0.0, 1.0, unit="0-1", group=g_mag,
               phys_min=0.0, phys_max=1.0,
               source="CRT 玻壳外围双盘 PTC 热敏电阻 50/60Hz 衰减交变消磁回路",
               help="模拟按下显示器 Degauss 消磁按钮或冷机开机瞬间：50/60Hz 强交变包络令全屏剧烈抖动翻滚并伴随嗡鸣，在 ~2.5 秒内按指数衰减将荫罩剩磁色斑彻底归零。")
    dsp.define("crt_ac_hum_field", "外部工频交流漏磁 (AC Hum Snake)", 0.0, 0.0, 1.0, unit="0-1", group=g_mag,
               phys_min=0.0, phys_max=0.8,
               help="模拟 CRT 紧贴未屏蔽大功率电源变压器/音箱分频器时，50/60Hz 交变漏磁场引起的全屏S形扭动与滚动彩带。")
    dsp.define("crt_corona_discharge", "高压阳极帽受潮电晕与打火", 0.0, 0.0, 1.0, unit="0-1", group=g_mag,
               phys_min=0.0, phys_max=0.8,
               help="高湿/强电离环境下 CRT 25kV 阳极高压帽沿玻壳表面发生局部电晕暗紫辉光与行同步瞬态打火撕裂。")
    dsp.define("crt_magnet_quad_corner", "强磁实验室角落选择 (0四分屏/1左上/2右上/3左下/4右下)", 0.0, 0.0, 4.0, unit="角落", group=g_mag,
               step=1.0, phys_min=0.0, phys_max=4.0,
               help="0=显示 2×2 四分屏对照；1=左上（强磁贴面黑洞+彩虹环）；2=右上（磁极旋转涡流场）；3=左下（撤磁后铁磁荫罩剩磁色斑）；4=右下（PTC 消磁线圈交变振荡净化后）。")
    dsp.define("crt_magnet_show_border", "强磁实验室分框样式 (0无分框/1细十字/2标准分框)", 2.0, 0.0, 2.0, unit="分框", group=g_mag,
               step=1.0, phys_min=0.0, phys_max=2.0)
    dsp.define("crt_magnet_show_text", "强磁实验室文字叠加 (0不要文字/1仅标签/2完整遥测)", 0.0, 0.0, 2.0, unit="文字", group=g_mag,
               step=1.0, phys_min=0.0, phys_max=2.0)

    # --- Sony Trinitron KV-1214 (1978–1981 12 寸柱面特丽珑彩电) 专项物理模型 ---
    g_kv = "Sony Trinitron KV-1214 12寸柱面特丽珑模型"
    dsp.define("kv1214_enabled", "启用 Sony Trinitron KV-1214 专项光学引擎", 0.0, 0.0, 1.0, unit="bool", group=g_kv,
               step=1.0, phys_min=0.0, phys_max=1.0,
               source="12寸柱面特丽珑显像管 4K 屏摄实测对位标定",
               help="开启 1978–1981 年 Sony Trinitron KV-1214（12 英寸单枪三束柱面特丽珑彩电）全套物理光学："
                    "NTSC-J 9300K 青蓝荧光白点 + Y2O2S:Eu 红粉暖光晕、垂直连续荫栅 + 单根中心钨丝阻尼线、"
                    "单向水平柱面玻壳曲率、以及 4K 屏摄弯曲摩尔纹。")
    dsp.define("kv1214_grille_pitch_px", "KV-1214 垂直荫栅节距 (Aperture Grille Pitch)", 2.35, 1.4, 5.0, unit="px", group=g_kv,
               phys_min=1.6, phys_max=3.8,
               source="12 英寸小尺寸特丽珑显像管粗节距垂直条状荧光粉栅格 (HFx/HFy = 1.48)",
               help="控制垂直连续 R/G/B 荧光条与碳黑矩阵竖缝的物理节距；决定垂直条纹锐度与屏摄摩尔纹的空间基频。")
    dsp.define("kv1214_grille_contrast", "KV-1214 垂直荫栅与黑矩阵深度", 0.68, 0.0, 1.0, unit="0-1", group=g_kv,
               phys_min=0.25, phys_max=0.90,
               help="垂直荫栅条纹与碳黑保护缝的调制深度；与隔行扫描线共同形成 KV-1214 标志性的『竖向强于横向 (HFx/HFy ≈ 1.48)』微结构。")
    dsp.define("kv1214_moire_amount", "4K 屏摄每根摩尔纹强度 (Curved Moiré Fringes)", 0.62, 0.0, 1.2, unit="0-1", group=g_kv,
               phys_min=0.0, phys_max=1.0,
               source="4K 近距屏摄柱面荫栅 × 相机拜耳阵列二维空间差拍干涉",
               help="复现 4K 数码相机微距拍摄柱面特丽珑屏幕时，单向水平弯曲的垂直荫栅投影与传感器像素网格发生差拍干涉产生的弧形彩虹/明暗摩尔纹。")
    dsp.define("kv1214_moire_freq", "摩尔纹差拍空间频率", 0.052, 0.015, 0.15, unit="cyc/px", group=g_kv,
               phys_min=0.025, phys_max=0.10,
               help="控制屏摄距离微变引起的弧形摩尔纹疏密周期。")
    dsp.define("kv1214_moire_tilt_deg", "屏摄相机微倾角", 1.8, -12.0, 12.0, unit="deg", group=g_kv,
               phys_min=-6.0, phys_max=6.0,
               help="相机传感器水平轴相对特丽珑垂直荫栅的微小旋转角，决定摩尔纹条带的斜向弯曲走向。")
    dsp.define("kv1214_cylindrical_bow", "柱面显像管水平弧度 (Vertical Flat / Horizontal Cylinder)", 0.34, 0.0, 1.0, unit="0-1", group=g_kv,
               phys_min=0.10, phys_max=0.65,
               source="Sony Trinitron 专利垂直平直、水平圆柱面玻壳几何",
               help="特丽珑显像管在垂直方向笔直无曲率、仅在水平方向呈圆柱弯曲（区别于普通球面 CRT 的双向鱼眼鼓出）。")
    dsp.define("kv1214_ntscj_9300k", "NTSC-J 9300K 青蓝白点与红粉光晕", 0.78, 0.0, 1.2, unit="0-1", group=g_kv,
               phys_min=0.30, phys_max=1.0,
               source="NTSC-J 9300K 色温偏移实测标定：数字源 R/G=0.897, B/G=0.922 → 屏摄 R/G=0.778, B/G=1.106",
               help="还原日本本土 NTSC-J 9300K 色温基准下冷青蓝高光与冷调暗部基底，同时保留 Y2O2S:Eu 稀土红粉在暖色区域的浓郁发光与长程红橙晕染。")
    dsp.define("kv1214_single_damper", "12寸单根钨丝阻尼线阴影 (Single Damper Wire)", 0.65, 0.0, 1.0, unit="0-1", group=g_kv,
               phys_min=0.20, phys_max=0.90,
               help="12 英寸小型特丽珑显像管（KV-1214）在屏幕中部偏下 (y ≈ 0.52H) 仅使用单根水平钨丝阻尼线（14寸以上才用双线），在高亮背景上可见极细水平暗丝。")
    dsp.define("kv1214_bezel_frame", "KV-1214 银灰/木纹复古前框与暗室辉光", 0.45, 0.0, 1.0, unit="0-1", group=g_kv,
               phys_min=0.0, phys_max=0.85,
               help="还原暗室屏摄下的 Sony KV-1214 柱面屏幕内凹黑框、暗室边缘环境辉光与厚铅玻璃表面微反光。")
    dsp.define("kv1214_ab_split", "数字源画面 vs KV-1214 显像管同屏对照模式 (0关闭/1左右分屏/2四象限)", 0.0, 0.0, 2.0, unit="模式", group=g_kv,
               step=1.0, phys_min=0.0, phys_max=2.0,
               help="0=全屏输出 KV-1214 效果；1=左半屏数字源原片 vs 右半屏 KV-1214 显像管效果实时对照；2=2×2 四象限细节微距与摩尔纹放大对照。")

    # --- 字幕机（字符发生器）----------------------------------------------------
    # **字符发生器不是排版引擎。** OSD 芯片有字模 ROM、每字一个固定字格、一个位置寄存器；
    # 它把字形画进格子，**不测量、不调字距、不换行、不重排**。这不是简化，而是这台设备实际做的事，
    # 也正是摄像机叠字看起来那样的原因：等距、定尺寸、永不比例化。
    #
    # 用户给的三个字体**全部是点阵来源**（实测：VCR-OSD-Mono 204 字形仅 ASCII、
    # 柳宋 7577 字形含 6763 汉字、A-POS 22169 字形含 20902 汉字），
    # 而且**西文与汉字的前进宽度正好 1:2**（1200:2400、504:1008）——
    # 所以一个汉字占两个半格，这就是那个年代的半角/全角约定。
    osd = s.layer("overlay")
    g = "字幕机"
    osd.define("at", "发生器位置", 0.0, 0.0, 1.0, unit="0-1", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0,
               help="**0 = 画面成形之后**（摄像机自身的 OSD）：字幕写在已成形的画面上，"
                    "所以之后发生的磁带损伤落在画面上、**不落在字上**。"
                    "**1 = 介质之前**（字幕机往磁带上录）："
                    "录像机的失落、磁头磨损、时基误差**连字一起毁**。"
                    "两者都是真实设备；差别正好在磁带很差的时候看得出来。")
    osd.define("enabled", "启用字幕机", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0)
    osd.define("font", "字体", 0.0, 0.0, 27.0, unit="0-27", group=g, step=1.0,
               phys_min=0.0, phys_max=27.0,
               help="字符发生器与光学字幕字库选择（共 28 款点阵/等宽/电视/卡拉OK/80年代纪录片字体）："
                    "0 = VCR OSD Mono、1 = 柳宋、2 = A-POS 点阵、3 = OCR-A、4 = OCR-B、"
                    "5 = Ming-Imperial、6 = OCR-A 扩展、7 = 中易黑体、8 = 中易宋体、"
                    "9 = MS 明朝、10 = MS 哥特、11 = 细明体、12 = Consolas、13 = Courier New、"
                    "14 = Century Gothic、15 = Sony Handycam 窄高体、16 = Panasonic VCR 扁宽点阵、"
                    "17 = LD/VCD 卡拉OK 大粗宋、18 = LD/VCD 卡拉OK 综艺粗黑、19 = CCTV 监控时间码、"
                    "20 = 80年代纪录片大标宋、21 = 80年代纪录片长宋、22 = 80年代电视专题片粗黑、"
                    "23 = 80年代新闻片扁黑、24 = 80年代文献片仿宋、25 = 80年代历史片硬楷、"
                    "26 = 80年代片头魏碑体、27 = 80年代广播级字幕机点阵宋。")
    osd.define("text", "字幕文本", "", 0.0, 1.0, unit="text", kind="text", group=g,
               help="可多行（换行即多行）。**不换行、不重排**——一行超出右边缘的部分直接丢失，"
                    "因为真实字符发生器没有断行逻辑。汉字占两格、西文占一格。")
    osd.define("cell_h", "字格高度", 24.0, 8.0, 96.0, unit="px", group=g, step=1.0,
               phys_min=12.0, phys_max=48.0,
               help="字格像素高度；**半格宽度由字体自己的前进宽度换算而来**，不是猜测值。"
                    "把格子当成长度的一半会让字形与格子错位、相邻字符串重叠——第一版就是这样。")
    osd.define("halign", "水平对齐", 0.0, 0.0, 2.0, unit="0-2", group=g, step=1.0,
               phys_min=0.0, phys_max=2.0,
               help="0 左、1 居中、2 右，都在安全区内。**字符发生器没有排版引擎，但有位置寄存器**——"
                    "OSD 芯片本来就能把一行放在安全区的左/中/右。对齐不需要断行，所以它不是排版功能，"
                    "而是这台设备本来就有的控件。**再也不用打空格来挪字幕了。**")
    osd.define("valign", "垂直对齐", 0.0, 0.0, 2.0, unit="0-2", group=g, step=1.0,
               phys_min=0.0, phys_max=2.0,
               help="0 上、1 中、2 下。多行时整个文字块一起对齐，所以**行间距由字格高度决定**，"
                    "不需要靠空行来撑。")
    osd.define("pos", "位置预设", 0.0, 0.0, 8.0, unit="0-8", group=g, step=1.0,
               phys_min=0.0, phys_max=8.0,
               help="一次设好水平与垂直，省得调两个参数：0 左上、1 上中、2 右上、3 左中、4 正中、"
                    "5 右中、6 左下、7 下中、8 右下。**8（右下）是家用录像最常见的 OSD 位置。**")
    osd.define("col", "列偏移", 0.0, 0.0, 60.0, unit="格", group=g, step=1.0)
    osd.define("row", "行偏移", 0.0, 0.0, 20.0, unit="行", group=g, step=1.0)
    osd.define("margin", "安全区边距", 0.03, 0.0, 0.25, unit="比例", group=g,
               phys_min=0.02, phys_max=0.12)
    osd.define("shadow", "投影强度", 0.5, 0.0, 1.0, unit="0-1", group=g,
               help="字下方偏移一份暗影。这是消费级发生器为了保证亮场景可读性用的最省的做法。")
    osd.define("outline", "字形描边", 1.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0,
               help="**每个字形外面一圈深色边。默认开启，这是真实摄像机的做法。**"
                    "它让字幕在任意画面上都可读，**而且不遮挡画面**——"
                    "这正是摄像机自身 OSD 用的办法，也是之前白字压白画面时字幕消失的原因。"
                    "一像素的膨胀就是一圈边，代价是三次平移取最大，而不是卷积。")
    osd.define("background", "底色块", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               help="在文字后面烧一个实心黑框。**默认随位置**："
                    "「画面成形之后」（摄像机自身 OSD）**不画框**——真实手持机不会遮住一块画面；"
                    "「介质之前」（字幕机录到磁带）**默认画框**，因为那个位置的字要压在磁带上、"
                    "底下是什么画面无法预期。显式设定会覆盖这个默认。")
    osd.define("opacity", "不透明度", 1.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.5, phys_max=1.0)
    osd.define("red", "字色 R", 0.92, 0.0, 1.0, unit="0-1", group=g)
    osd.define("green", "字色 G", 0.94, 0.0, 1.0, unit="0-1", group=g)
    osd.define("blue", "字色 B", 0.90, 0.0, 1.0, unit="0-1", group=g)

    # --- style: the outline has its own pen, and the glyph can be sheared -----------------
    osd.define("italic", "斜体（剪切）", 0.0, -0.8, 0.8, unit="shear", group=g,
               phys_min=0.0, phys_max=0.4,
               help="**点阵发生器没有真正的斜体，它是把字形剪切。**"
                    "所以斜着的 OSD 文字边缘是台阶状的——剪切作用在**阈值化之后**的掩膜上，"
                    "像素结构因此保留下来。正值向右倾。")
    osd.define("stretch_x", "字宽（拉伸）", 1.0, 0.4, 2.5, unit="x", group=g,
               phys_min=0.75, phys_max=1.35,
               help="横向缩放字形。**点阵发生器画瘦高字靠的是改字格，不是滤波**，"
                    "所以这里是**在掩膜上重采样**，硬边缘保留下来。1.0 是字体本身的比例。"
                    "和下面的「字高」是**两个独立轴**：真实机器上横向由时钟、纵向由行数决定。")
    osd.define("stretch_y", "字高（拉伸）", 1.0, 0.4, 2.5, unit="x", group=g,
               phys_min=0.75, phys_max=1.35,
               help="纵向缩放字形。与「字宽」独立。把两个字轴分开，"
                    "才能做出当年窄体/宽体字幕机那种字形。")
    osd.define("letter_gap", "字距", 0.0, -4.0, 12.0, unit="px", group=g, step=1.0,
               phys_min=-1.0, phys_max=4.0,
               help="**字符之间**额外增加的像素。这与「字宽」是不同的控制："
                    "真实发生器的字格是固定的，而字符之间的空隙取决于 ROM 被读出的方式。"
                    "负值让字挤在一起，那是当年为了在有限字格里塞下更多字的做法。")
    osd.define("line_gap", "行距", 0.0, -8.0, 24.0, unit="px", group=g, step=1.0,
               phys_min=-2.0, phys_max=8.0,
               help="**行与行之间**额外增加的像素。")
    osd.define("outline_red", "描边色 R", 0.02, 0.0, 1.0, unit="0-1", group=g)
    osd.define("outline_green", "描边色 G", 0.03, 0.0, 1.0, unit="0-1", group=g)
    osd.define("outline_blue", "描边色 B", 0.05, 0.0, 1.0, unit="0-1", group=g)
    osd.define("shadow_x", "阴影横向偏移", 2.0, -8.0, 8.0, unit="px", group=g, step=1.0)
    osd.define("shadow_y", "阴影纵向偏移", 2.0, -8.0, 8.0, unit="px", group=g, step=1.0)
    osd.define("style", "样式模板", 0.0, 0.0, 9.0, unit="0-9", group=g, step=1.0,
               phys_min=0.0, phys_max=9.0,
               help="一次设好字色、描边色和斜体：**0 自定义**、1 绿字黑边（OSD 经典）、"
                    "2 红字黑边（REC 经典）、3 白字黑边、4 黄字黑边、5 青字黑边、"
                    "6 80年代纪录片·金黄标宋立体字、7 80年代纪录片·象牙白光学叠印字、"
                    "8 广播级字幕机·琥珀黄滚屏字、9 80年代专题片·深红描金标题字。"
                    "**0 以外的值会覆盖上面那些颜色参数。**")
    osd.define("edge_softness", "边缘柔化（模拟键控上升沿）", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.65,
               help="**硬件点阵字符发生器是 1-bit 硬边缘（0.0），而 80 年代光学印片字幕或模拟线性键控器（Chroma/Luma Keyer）具有有限的模拟带宽与上升时间。**"
                    "横向比纵向略宽的高斯过渡带还原模拟字幕机键控边缘。")
    osd.define("glow", "光学晕染（字幕漫射辉光）", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.65,
               help="**光学印片机叠印字幕或高亮度视频键控在摄像管/胶片乳剂上产生的暖色漫射光晕（Halation）。**"
                    "使高亮字幕周围呈现柔和的胶转磁/模拟电视光感。")
    osd.define("bevel", "立体浮雕（硬件斜角高光）", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.80,
               help="**80年代末广播级硬件字幕机与大型专题片片头常见的 3D 斜角立体浮雕效果。**"
                    "字形左上方受光提亮、右下方背光压暗。")
    osd.define("anim_mode", "字幕动画模式", 0.0, 0.0, 7.0, unit="0-7", group=g, step=1.0,
               phys_min=0.0, phys_max=7.0,
               help="80/90年代硬件字符发生器与光学片头动画：**0 静态常显**、1 逐字打字机显字（电传/纪实解说）、"
                    "2 逐行顺序显字、3 水平光栅扫描擦除（模拟电子束横向拉幕）、"
                    "4 百叶窗隔行扫描显字（硬件偶数行/奇数行分步切入）、"
                    "5 纵向滚屏（片尾演职员表自下而上滚屏）、6 光学叠化淡入淡出、7 横向滑入定格。")
    osd.define("anim_speed", "动画速度倍率", 1.0, 0.1, 4.0, unit="x", group=g,
               phys_min=0.4, phys_max=2.5,
               help="控制打字机显字、扫描擦除、滚屏及淡入淡出的时间速率。")
    osd.define("anim_in_s", "入场延迟时间", 0.15, 0.0, 5.0, unit="s", group=g,
               phys_min=0.0, phys_max=2.0,
               help="字幕动画开始前的画面停留时间（秒）。")
    osd.define("anim_cycle_s", "动画周期时长", 0.0, 0.0, 30.0, unit="s", group=g,
               phys_min=0.0, phys_max=15.0,
               help="0 表示单次播放并保持定格；大于 0 时按该秒数循环演示字幕进出场动画。")

    g = "主控播出自动化 Playout 与双总线 DSK / Clean Feed"
    osd.define("playout_enabled", "启用主控播出台 (Master Playout)", 0.0, 0.0, 1.0, unit="bool", group=g,
               step=1.0, phys_min=0.0, phys_max=1.0,
               source="电视台总控/主控播出机房 (MCR Master Control Playout Desk)",
               help="开启后接入主控双总线播出台：同时维护「主播出母线 PGM（带 DSK 台标/时钟/游走字幕）」、「干净净画面母线 Clean Feed（无台标）」与「预监母线 PVW」，并支持磁带预卷倒计时、故障自动切备用垫片与 As-Run 播出日志。")
    osd.define("playout_output_bus", "主控输出母线选择", 0.0, 0.0, 3.0, unit="0-3", group=g,
               step=1.0, phys_min=0.0, phys_max=3.0,
               source="广播级主控切换台 PGM / Clean Feed / PVW / 导播双总线控制台",
               help="0=主播出母线 PGM（含 DSK 台标 Bug、实时时间码时钟与底部游走字幕）｜"
                    "1=干净净画面输出 Clean Feed（经切换与调色但 100% 不含 DSK 台标与字幕，供异地转播与母带归档）｜"
                    "2=预监母线 PVW（显示下一条待播源、安全框与预卷就绪状态）｜"
                    "3=主控双总线导播控制台视图（左 PVW + 右 PGM + 底部 Clean Feed 小窗与实时 As-Run 播出运行日志表）。")
    osd.define("playout_state", "主控播出单状态机 (Playlist State)", 0.0, 0.0, 3.0, unit="0-3", group=g,
               step=1.0, phys_min=0.0, phys_max=3.0,
               source="90年代电视台自动化播出表状态机 (Preroll -> On-Air -> Emergency Failover)",
               help="0=正常节目播出 (ON-AIR Program)｜"
                    "1=录像带 5秒预卷倒计时锁相 (VTR Preroll Leader Countdown -00:05..-00:01 -> Take)｜"
                    "2=信号中断/故障 -> 应急切入「PLEASE STAND BY / 请稍候」备用垫片 (Emergency Standby Slate)｜"
                    "3=播出单自动顺播循环演示 (Auto-Cycle: Preroll -> On-Air -> Next)。")
    osd.define("playout_dsk_enabled", "下游键控器 (DSK 台标与主控时钟)", 1.0, 0.0, 1.0, unit="bool", group=g,
               step=1.0, phys_min=0.0, phys_max=1.0,
               source="主控切换台最末级 Downstream Keyer (DSK)",
               help="开启后在 PGM 母线右上角键入三色台标与实时广播时钟（Clean Feed 母线始终绕过 DSK 保持纯净）。")
    osd.define("playout_dsk_crawl", "DSK 底部游走新闻/通告字幕条", 1.0, 0.0, 1.0, unit="bool", group=g,
               step=1.0, phys_min=0.0, phys_max=1.0,
               source="主控字幕机底部横向游走字幕条 (Lower-Third Ticker Crawl)",
               help="在 PGM 母线底部叠加半透明深蓝底栏与横向滚动的节目预告/应急抢修通告。")
    osd.define("playout_failover_auto", "上游断流自动应急接管 (Auto-Failover)", 1.0, 0.0, 1.0, unit="bool", group=g,
               step=1.0, phys_min=0.0, phys_max=1.0,
               source="总控同步检测与黑场告警自动倒换矩阵",
               help="当检测到上游画面断流（如 HDMI 握手失败或磁带完全失锁黑屏）时，自动切入「PLEASE STAND BY」应急备用垫片并在 As-Run 日志记录告警。")

    g = "事件驱动与条件驱动播出自动化编排引擎"
    osd.define("auto_v2_enabled", "启用事件与条件驱动播出自动化编排引擎 v2", 0.0, 0.0, 1.0, unit="bool", group=g,
               step=1.0, phys_min=0.0, phys_max=1.0,
               source="广播总控事件驱动播出自动化系统 (Harris ADC / Grass Valley Maestro)",
               help="开启后激活完整 PlayoutEvent 状态机（SCHEDULED -> PREPARING -> READY -> ON_AIR -> COMPLETED）与 6 类分级异常分支及故障因果反查链。")
    osd.define("auto_v2_mode", "播出自动化触发与倒换策略", 0.0, 0.0, 3.0, unit="模式", group=g,
               step=1.0, phys_min=0.0, phys_max=3.0,
               source="广播播出单定时触发、GPI/信号条件触发与主备冗余自动倒换规范",
               help="0=跟随开关｜1=时间驱动定时顺播 (Time-Driven：按播出单在切点前自动向录像机发送 5s Preroll 指令并锁相切播)｜"
                    "2=条件/GPI 信号驱动 (Condition-Driven：实时监测视频黑场、静帧死锁、同步丢失与静音，触发条件即执行跳转)｜"
                    "3=主备冗余倒换闭环 (Redundant Main/Backup Closed-Loop：主源故障超时自动无缝倒换至备机/垫片路并记录完整因果溯源链)。")
    osd.define("auto_v2_fault_inject", "注入播出单异常分支 (6类标准广播事故)", 0.0, 0.0, 6.0, unit="故障", group=g,
               step=1.0, phys_min=0.0, phys_max=6.0,
               source="广播电视安全播出事故分级与应急预案",
               help="0=正常顺播无故障｜1=MISSING_ASSET（磁带未上架/素材离线 -> 自动切备用垫片）｜"
                    "2=NOT_READY（录像机预卷粘带失锁/伺服未锁定 -> 延迟切入或垫片兜底）｜"
                    "3=SIGNAL_LOSS（在播主路射频/同步突然断流 -> 看门狗超时切垫片）｜"
                    "4=OVERRIDDEN（导播手动紧急强切 Emergency Take 接管自动化母线）｜"
                    "5=FALLBACK（主路触发静帧/黑场告警 -> 自动倒换至 BACKUP 备路）｜"
                    "6=HELD（直播延时/挂起等待导播手动确认释放）。")
    osd.define("auto_v2_watchdog_sec", "黑场/静帧/失锁看门狗门限", 1.0, 0.2, 5.0, unit="s", group=g,
               phys_min=0.4, phys_max=3.0,
               source="总控视频静帧与同步丢失自动告警延时门限",
               help="主播出母线容忍无同步、黑场或静帧的最长持续秒数；超出该门限后自动化引擎立即触发主备倒换并生成 CausalFaultRecord 根因链。")
    osd.define("auto_v2_console_view", "播出自动化四分屏编排与故障反查控制台", 0.0, 0.0, 1.0, unit="视图", group=g,
               step=1.0, phys_min=0.0, phys_max=1.0,
               help="0=输出主播出母线画面（顶部带状态机与故障溯源简报栏）｜1=切换为 2×2 播出自动化编排控制台（同屏监看：PGM 主输出、PVW/备机预卷、PlayoutEvent 五态状态机表、以及根因→传播→现象三级故障反查树）。")

    g_eco = "媒介技术生态演化与历史情景模拟器"
    osd.define("eco_enabled", "启用媒介技术生态演化与历史情景模拟器", 0.0, 0.0, 1.0, unit="bool", group=g_eco, step=1.0,
               phys_min=0.0, phys_max=1.0, source="MediaEcosystemSimulator 七维宏观媒介技术史动力学模型",
               help="联动追踪 1975–2035 年间『设备存量、介质存量、跨代兼容性、备件与技师维护能力、迁移成本、制度约束与机构惯性』，并将所选年代的生态状态直接投射为当前信号链的物理老化分布。")
    osd.define("eco_mode", "媒介生态演化与反事实推演模式", 0.0, 0.0, 4.0, unit="0-4", group=g_eco, step=1.0,
               phys_min=0.0, phys_max=4.0, source="历史年代状态投射与反事实情景对照",
               help="0=跟随总开关｜1=历史年代媒介生态实况投射 (按 eco_year 所处年代的机具磨损、备件枯竭与磁带老化直接驱动画面)｜"
                    "2=备件与磁头断崖危机模拟 (模拟最后一批原厂旋转磁鼓/飞抹磁头备件耗尽、老师傅退休引发的存量磁带可访问性断崖)｜"
                    "3=历史反事实情景对照输出 (对比基准历史 vs 提前十年无损归档/延迟淘汰模拟机具的画质与存活率差异)｜"
                    "4=2×2 媒介技术生态演化实验室 (同屏对比：1992 黄金期基准 vs 当前年代真实生态状态 vs 反事实情景推演 vs 1975-2035 七维演化曲线与断崖预警图)。")
    osd.define("eco_year", "所选历史/未来推演年份", 2008.0, 1975.0, 2035.0, unit="年", group=g_eco, step=1.0,
               phys_min=1980.0, phys_max=2030.0, source="1975–2035 视听媒介技术更替时间轴",
               help="1985-1995 为模拟磁带黄金期（机具新、备件充足）；2005-2015 为数字转轨期；2020 以后模拟回放机具与备件加速消亡进入档案抢救窗口危机。")
    osd.define("eco_scenario", "历史反事实推演情景选择", 0.0, 0.0, 3.0, unit="0-3", group=g_eco, step=1.0,
               phys_min=0.0, phys_max=3.0, source="媒介技术史反事实情景库",
               help="0=真实历史基准轨迹 (Baseline History)｜1=反事实 A：数字格式与无损档案采集提前 10 年普及 (Early Digital Shift)｜"
                    "2=反事实 B：原厂磁头备件与特约维修站于 2005 年提前断供 (Premature Spare Collapse)｜"
                    "3=反事实 C：全球公共视听档案强制冷库保存与备件开源复刻计划 (Global Archival Mandate)。")
    osd.define("eco_spare_depletion", "磁头备件与技师消亡速率倍率", 1.0, 0.2, 2.0, unit="×", group=g_eco,
               phys_min=0.4, phys_max=1.6, source="专用旋转磁鼓/主导轴电机停产与维修技师代际断层速率")
    osd.define("eco_institution_inertia", "机构工作流惯性与迁移阻力", 0.55, 0.0, 1.0, unit="0-1", group=g_eco,
               phys_min=0.15, phys_max=0.85, source="电视台/档案馆旧格式路径依赖与预算摩擦系数")
    osd.define("eco_quad_corner", "生态模拟器画面角落选择 (0四分屏/1左上/2右上/3左下/4右下)", 0.0, 0.0, 4.0, unit="角落", group=g_eco,
               step=1.0, phys_min=0.0, phys_max=4.0, source="2×2 四分屏单角全屏拆选",
               help="0=显示完整 2×2 四分屏对照；1=左上角（1992 模拟黄金期基准）；2=右上角（所选年份真实生态回放状态）；3=左下角（反事实情景推演状态）；4=右下角（1975-2035 七维生态演化曲线与断崖预警图）。")
    osd.define("eco_show_border", "生态模拟器分框样式 (0无分框/1细十字线/2标准分框)", 2.0, 0.0, 2.0, unit="分框", group=g_eco,
               step=1.0, phys_min=0.0, phys_max=2.0, source="四分屏分框线显示控制")
    osd.define("eco_show_text", "生态模拟器文字叠加 (0不要文字/1仅标签/2完整遥测)", 0.0, 0.0, 2.0, unit="文字", group=g_eco,
               step=1.0, phys_min=0.0, phys_max=2.0, source="画面文字叠加开关（默认 0 纯净无字画面）")

    # ------------------------------------------------------------------ G
    # Audio simulation layer: physical analog/digital-tape audio & A/V fault sync
    aud = s.layer("audio")
    g = "伴音制式与音源"
    aud.define("enabled", "启用音频模拟", 1.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0, source="系统架构",
               help="开启后，导入素材的原声（或无声素材的内置参考音源）会经过与画面完全同源的模拟/磁带音频链路处理。")
    aud.define("mode", "伴音轨制式", 0.0, 0.0, 15.0, unit="0-15", group=g, step=1.0,
               phys_min=0.0, phys_max=15.0,
               source="IEC 60774 / IEC 61834 / GB 3174 / ISO 11172-3 / 3GPP TS 26.071 / ITU-R BS.1770",
               help="0=自动跟随视频制式，1=VHS 纵向线性磁迹，2=VHS Hi-Fi 深层调频立体声，"
                    "3=8mm/Hi8 AFM 调频伴音，4=MiniDV 12-bit PCM 数字伴音，5=VCD MP2 子带量化伴音，"
                    "6=PAL-D RF 伴音，7=淘宝电视卡 TV-DX 差拍伴音，8=胶片感光光学声带，9=流媒体低码率伴音，"
                    "10=广播演播室调音台，11=感知音频编码(MP3/AAC/AMR)，12=移动电话与网络语音(GSM/VoIP)，"
                    "13=旧手机扬声器与听筒外放，14=早期手机铃声/MIDI与和弦音源，15=电脑播放器与系统音频链。")
    aud.define("synth_type", "无声素材测试音源", 0.0, 0.0, 2.0, unit="0-2", group=g, step=1.0,
               phys_min=0.0, phys_max=2.0, source="内置参考信号发生器",
               help="当使用内置测试图、静帧或无音轨视频时自动生成的测试音源："
                    "0=复古合成器弦乐+琶音+节拍，1=广播校准双音(1kHz/440Hz+高频导频)，2=人声共振峰+同步节拍器。")
    aud.define("master_gain", "输出主增益", 1.0, 0.0, 2.0, unit="x", group=g,
               phys_min=0.5, phys_max=1.5, source="工程标定",
               help="模拟链路末端的音频输出电平。")

    g = "磁头频响与Hi-Fi切换"
    aud.define("hifi_unlock_thresh", "Hi-Fi 失锁回退阈值", 0.32, 0.05, 1.0, unit="0-1", group=g,
               phys_min=0.20, phys_max=0.55, source="JVC HR-D725U Hi-Fi 选轨电路手册",
               help="VHS Hi-Fi 旋转磁头遇到磁粉脱落、折痕或循迹失锁超过该阈值时，"
                    "FM 限幅器瞬间失锁爆音并自动切回沉闷的纵向线性伴音轨。")
    aud.define("linear_bandwidth", "线性轨高频截止", 7500.0, 800.0, 16000.0, unit="Hz", group=g,
               phys_min=2500.0, phys_max=10000.0, source="VHS 纵向音频规范 (SP ~8kHz, EP ~3kHz)",
               help="VHS 纵向线性音频轨在 SP 速度下的标称上限频率；切换到 LP/EP 慢录时会自动随带速按比例缩窄。")
    aud.define("azimuth_loss", "磁头方位角偏斜", 0.22, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.65, source="Wallace 磁头间隙与方位角损耗公式",
               help="音频磁头缝隙与磁迹倾角不匹配：导致高频滚降，并在左右声道之间产生微秒级相位偏斜。")

    g = "走带机构与音画故障同步"
    aud.define("wow_flutter", "主导轴抖晃 (Wow & Flutter)", 0.35, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.05, phys_max=0.85, source="IEC 60386 抖晃测量规范",
               help="与视频行相位误差 (line_phase) 及场垂直抖动 (frame_dy) 严格同源的音高抖晃："
                    "画面横移撕裂的同一毫秒，音高同步发生漂移。")
    aud.define("stiction_pitch", "粘滑卡滞变调深度", 0.65, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.20, phys_max=1.0, source="AMPEX 粘滑综合征机械张力模型",
               help="磁带受潮粘连主导轴 (stiction) 时的瞬态音高骤降与释放回弹超调，与画面横向撕裂帧精确对齐。")
    aud.define("dropout_sensitivity", "失落/折痕/糊头耦合度", 0.55, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.20, phys_max=1.0, source="音画同源介质缺陷模型",
               help="画面出现白斑失落 (dropout)、折痕 (crease)、糊头 (head_clog) 或基带开裂 (base_cracking) 时，"
                    "伴音在同一扫描行时刻同步产生的瞬态闷音、FM 爆音、低频撞击声与裂纹脉冲。")

    g = "接口串扰与电磁干扰"
    aud.define("head_switch_buzz", "磁头切换脉冲哼声", 0.30, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.05, phys_max=0.70, source="VHS 旋转磁鼓换向瞬态串扰",
               help="视频磁鼓每场切换 (59.94Hz NTSC / 50Hz PAL) 及循迹误差串入音频通道产生的周期性脉冲锯齿嗡声。")
    aud.define("ground_hum", "AV 接地环路交流声", 0.20, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.65, source="IEC 接地环路与氧化接触整流模型",
               help="与 AV 复合接口受潮/锈蚀及画面滚动亮带同源的 50Hz/60Hz 电源工频交流哼声及整流谐波。")
    aud.define("rf_buzz", "RF 伴音载波串扰 (Intercarrier Buzz)", 0.30, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.80, source="NTSC/PAL 差拍伴音 (4.5/5.5MHz) 鉴频模型",
               help="RF 射频调制/高频头受潮失锁时，画面亮带过调制与行钳位奔涌串入伴音载波产生的粗粝蜂鸣与静噪爆发。")

    g = "磁饱和、底噪与扬声器"
    aud.define("saturation", "磁饱和非线性失真", 0.30, 0.0, 1.2, unit="0-1", group=g,
               phys_min=0.05, phys_max=0.75, source="Jiles-Atherton 磁滞回线近似",
               help="磁粉剩磁饱和曲线 (tanh + 偶次/奇次谐波)：大动态电平软削波并带来模拟磁带的温暖厚重感。")
    aud.define("hiss_floor", "磁粉颗粒底噪 (Tape Hiss)", 0.22, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.05, phys_max=0.65, source="磁带热磁噪声模型",
               help="氧化铁/金属磁粉颗粒与 FM 鉴频器的本底嘶声；在 LP/EP 慢速模式下自动随信噪比损失抬升。")
    aud.define("speaker_sim", "CRT 电视机腔体箱体声", 0.35, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.85, source="消费级 CRT 电视声学测量",
               help="模拟老式 CRT 电视机 3 英寸塑料腔体扬声器的频响（切除 185Hz 以下低频、1.1kHz 中频共鸣峰与 5.8kHz 以上纸盆滚降）。")
    aud.define("flyback_whine", "CRT 行输出高压包啸叫", 0.15, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.50, source="CRT 行逆程变压器声学辐射",
               help="CRT 显像管开启时，15,734Hz (NTSC) / 15,625Hz (PAL) 行扫描高压包磁芯振动串入伴音的微弱高频啸叫。")

    # --- 01 · 麦克风与录音前端 (Microphone & Preamp Front-End) ---
    g_mic = "01 · 麦克风与录音前端 (Microphone & Preamp)"
    aud.define("mic_type", "麦克风换能器型号", 0.0, 0.0, 7.0, unit="0-7", group=g_mic, step=1.0,
               phys_min=0.0, phys_max=7.0, source="电声换能器与振膜声学阻抗模型",
               help="0=直通线路输入｜1=动圈话筒(SM58)｜2=大振膜电容话筒(U87)｜3=复古铝带话筒(RCA44)｜"
                    "4=驻极体电脑摄像头/耳麦话筒(3.8kHz塑料腔体峰)｜5=老式电话碳粒话筒｜"
                    "6=便携摄像机内置立体声驻极体话筒(带走带电机耦合)｜7=早期手机底部针孔驻极体/MEMS麦克风。")
    aud.define("mic_polar_pattern", "麦克风指向性图案", 1.0, 0.0, 4.0, unit="0-4", group=g_mic, step=1.0,
               phys_min=0.0, phys_max=4.0, source="一阶/高阶压力梯度指向性方程 G(θ)=α+(1-α)cos(θ)",
               help="0=全指向(Omni)｜1=心形指向(Cardioid)｜2=超心形(Supercardioid)｜3=8字双向(Figure-8)｜4=干涉管强指向枪式话筒(Shotgun)。")
    aud.define("mic_off_axis_deg", "声源偏轴入射角 θ", 0.0, 0.0, 180.0, unit="°", group=g_mic,
               phys_min=0.0, phys_max=150.0, source="指向性振膜孔径衍射高频滚降模型",
               help="声源偏离话筒主轴的角度：偏轴不仅按指向性曲线衰减声压，还会因振膜孔径相位抵消导致高频陡峭滚降。")
    aud.define("mic_sensitivity_db", "换能器灵敏度偏置", 0.0, -24.0, 24.0, unit="dB", group=g_mic,
               phys_min=-18.0, phys_max=18.0, source="IEC 60268-4 话筒开路灵敏度")
    aud.define("mic_self_noise", "话筒本底热噪与 JFET 1/f 噪声", 0.0, 0.0, 1.0, unit="0-1", group=g_mic,
               phys_min=0.0, phys_max=0.80, source="等效输入噪声 (EIN) 与驻极体场效应管闪烁噪声")
    aud.define("mic_preamp_gain_db", "话放前级增益", 0.0, -18.0, 30.0, unit="dB", group=g_mic,
               phys_min=-12.0, phys_max=24.0, source="麦克风前置放大器增益级")
    aud.define("mic_agc_amount", "前端自动增益控制 (AGC) 深度", 0.0, 0.0, 1.0, unit="0-1", group=g_mic,
               phys_min=0.0, phys_max=0.95, source="摄像机/手机录音前端快启慢释 AGC 抽吸电路",
               help="在安静间隙自动拉高增益（将房间底噪与走带电机声一并放大），遇到突发大声压时瞬态下压产生呼吸抽吸。")
    aud.define("mic_agc_speed", "前端 AGC 响应速度", 0.5, 0.0, 1.0, unit="0-1", group=g_mic,
               phys_min=0.0, phys_max=1.0, source="AGC 检波时间常数 (0=慢速摄像机, 1=快速录音笔/手机)")
    aud.define("mic_clip_mode", "前级削波失真拓扑", 0.0, 0.0, 2.0, unit="0-2", group=g_mic, step=1.0,
               phys_min=0.0, phys_max=2.0, source="前级过载削波物理机制",
               help="0=场效应管/变压器非对称软饱和(Soft Clip)｜1=廉价运放电源轨硬削顶(Hard Rail Clip)｜2=低成本 ADC 整型溢出翻转爆音(Wrap Glitch)。")
    aud.define("proximity_effect", "麦克风近讲效应", 0.0, 0.0, 1.0, unit="0-1", group=g_mic,
               phys_min=0.0, phys_max=1.0, source="近场球面波声压差模型",
               help="贴近定向话筒时 180Hz 以下低频显著提升 (+9dB)。")
    aud.define("plosive_pop", "防喷罩缺失破音", 0.0, 0.0, 1.0, unit="0-1", group=g_mic,
               phys_min=0.0, phys_max=1.0, source="气流冲击振膜低频风噪模型",
               help="演唱爆破音 (P/B/T) 强气流冲击振膜引起的低频风噪轰鸣与瞬态过载。")
    aud.define("mic_overdrive", "话筒与前级过载削顶", 0.0, 0.0, 1.0, unit="0-1", group=g_mic,
               phys_min=0.0, phys_max=1.0, source="前级放大器轨电压削顶模型",
               help="超出麦克风前级线性动态产生的谐波削顶破音。")
    aud.define("mic_chassis_vibration", "机身电机与手持震动耦合", 0.0, 0.0, 1.0, unit="0-1", group=g_mic,
               phys_min=0.0, phys_max=0.90, source="机壳固体声传导：走带电机/变焦马达啸叫与手指摩擦低频轰鸣")

    # --- 02 · 旧手机扬声器与听筒 (Legacy Mobile Speaker & Earpiece) ---
    g_mspk = "02 · 旧手机扬声器与听筒 (Mobile Speaker & Earpiece)"
    aud.define("mobile_spk_mode", "手机声学放音终端模式", 0.0, 0.0, 4.0, unit="0-4", group=g_mspk, step=1.0,
               phys_min=0.0, phys_max=4.0, source="微型受话器与聚酯振膜微扬声器 Thiele-Small 模型",
               help="0=关闭(全频直出)｜1=手机听筒模式(Earpiece 贴耳受话器 300-3500Hz)｜"
                    "2=功能机免提外放(Nokia 6600/N73 13×18mm 微型扬声器 f0=780Hz 高通与盆分裂峰)｜"
                    "3=早期音乐手机/直板机外放(Walkman/iPhone 初代外放 f0=520Hz)｜4=语音信箱与免提座机喇叭。")
    aud.define("mobile_spk_f0_hz", "微型扬声器低频谐振截止 f0", 680.0, 200.0, 1600.0, unit="Hz", group=g_mspk,
               phys_min=240.0, phys_max=1200.0, source="微型扬声器顺性与等效振动质量谐振频率 f0")
    aud.define("mobile_spk_drive", "最大声压级与振膜非线性破音", 0.35, 0.0, 1.0, unit="0-1", group=g_mspk,
               phys_min=0.0, phys_max=0.95, source="大振幅下音圈磁场 Bl(x) 与折环劲度 Kms(x) 非线性谐波失真")
    aud.define("mobile_spk_cavity_res", "塑料机壳后腔与出音孔共鸣", 0.45, 0.0, 1.0, unit="0-1", group=g_mspk,
               phys_min=0.0, phys_max=0.95, source="微型塑料背腔赫姆霍兹谐振与壳体短延迟反射")
    aud.define("mobile_spk_limiter", "硬件防烧音量限幅器 (ALC)", 0.40, 0.0, 1.0, unit="0-1", group=g_mspk,
               phys_min=0.0, phys_max=0.95, source="手机功放芯片防过热动态限幅压缩")

    # --- 03 · 数字采样与音频格式转换 (Digital Sampling & Format Conversion) ---
    g_adc = "03 · 数字采样与音频格式转换 (Digital Sampling & SRC)"
    aud.define("adc_enabled", "启用数字采样与格式转换器", 0.0, 0.0, 1.0, unit="bool", group=g_adc, step=1.0,
               phys_min=0.0, phys_max=1.0, source="PCM 模数采样定理与定点量化误差模型")
    aud.define("adc_sample_rate_hz", "数字目标采样率 fs", 44100.0, 4000.0, 48000.0, unit="Hz", group=g_adc,
               phys_min=6000.0, phys_max=48000.0, source="常见数字采样率 (8kHz语音 / 11.025kHz多媒体 / 22.05kHz Flash / 32kHz DV / 44.1kHz CD)")
    aud.define("adc_bit_depth", "量化位深 (Bit Depth)", 16.0, 4.0, 24.0, unit="bit", group=g_adc, step=1.0,
               phys_min=6.0, phys_max=24.0, source="均匀 PCM 定点量化阶数 2^B")
    aud.define("adc_resample_filter", "重采样抗混叠与插值滤波器", 0.0, 0.0, 2.0, unit="0-2", group=g_adc, step=1.0,
               phys_min=0.0, phys_max=2.0, source="数字抽取与内插重构核",
               help="0=Sinc 理想抗混叠滤波｜1=线性插值(Linear SRC 高频滚降与部分折叠混叠)｜2=零阶保持阶梯采样(Zero-Order Hold 无抗混叠强烈镜像金属混叠)。")
    aud.define("adc_dither_mode", "量化抖动与噪声整形 (Dither)", 0.0, 0.0, 3.0, unit="0-3", group=g_adc, step=1.0,
               phys_min=0.0, phys_max=3.0, source="Lipshitz & Vanderkooy 量化抖动理论",
               help="0=无抖动硬截断(产生相关奇次谐波与尾音门限截断)｜1=矩形分布 RPDF｜2=三角分布 TPDF(消除非线性失真转为平稳白噪底)｜3=一阶高频噪声整形 TPDF。")
    aud.define("adc_channel_mode", "声道混合与相位故障", 0.0, 0.0, 3.0, unit="0-3", group=g_adc, step=1.0,
               phys_min=0.0, phys_max=3.0, source="立体声下混与转接头反相抵消模型",
               help="0=保持立体声｜1=单声道相加 (L+R)/2｜2=反相相消梳状下混 (L-0.85R 典型转接线虚焊中置人声消失)｜3=仅左声道复制。")
    aud.define("adc_clip_level", "ADC 满刻度削波上限", 1.0, 0.20, 1.0, unit="FS", group=g_adc,
               phys_min=0.35, phys_max=1.0, source="A/D 转换器参考电压上限")
    aud.define("adc_clock_jitter", "采样时钟孔径抖动 (Jitter)", 0.0, 0.0, 1.0, unit="0-1", group=g_adc,
               phys_min=0.0, phys_max=0.80, source="采样时钟相位噪声 Δt(n)·dx/dt")
    aud.define("adc_clock_drift_ppm", "重采样时钟频偏 (Drift)", 0.0, -2000.0, 2000.0, unit="ppm", group=g_adc,
               phys_min=-1000.0, phys_max=1000.0, source="独立晶振频率公差 (ppm)")

    # --- 04 · 感知音频编码 (Perceptual Audio Coding: MP3 / AAC / AMR / Speech) ---
    g_pcodec = "04 · 感知音频编码 (Perceptual Audio Codecs)"
    aud.define("pcodec_type", "感知音频编码器类型", 0.0, 0.0, 4.0, unit="0-4", group=g_pcodec, step=1.0,
               phys_min=0.0, phys_max=4.0, source="ISO/IEC 11172-3 (MP3) / 14496-3 (AAC) / 3GPP TS 26.071 (AMR-NB)",
               help="0=关闭｜1=MP3 (32子带+MDCT混合滤波组，低码率预回声与水声Birdies明显)｜"
                    "2=AAC-LC / HE-AAC (含低码率SBR高频频带复制金属光泽)｜"
                    "3=AMR-NB (3GPP 4.75-12.2kbps 手机3GP代数码本激励线性预测语音编码)｜"
                    "4=极低码率语音编码 (RealAudio G2 / CELP 8-24kbps)。")
    aud.define("pcodec_bitrate_kbps", "感知编码目标码率", 64.0, 4.75, 320.0, unit="kbps", group=g_pcodec,
               phys_min=4.75, phys_max=320.0, source="编码器比特池分配速率 (相同码率下不同编码器的心理声学掩蔽损伤截然不同)")
    aud.define("pcodec_pre_echo", "瞬态涂抹与预回声 (Pre-Echo)", 0.35, 0.0, 1.0, unit="0-1", group=g_pcodec,
               phys_min=0.0, phys_max=0.95, source="长窗 MDCT 块内量化噪声向打击乐瞬态前方扩散效应")
    aud.define("pcodec_birdies", "频带空洞与水声伪影 (Birdies)", 0.40, 0.0, 1.0, unit="0-1", group=g_pcodec,
               phys_min=0.0, phys_max=0.95, source="Bark 临界频带比特饥饿引发的谱线时变闪烁杂音")
    aud.define("pcodec_generations", "多代重复有损重编码次数", 1.0, 1.0, 8.0, unit="代", group=g_pcodec, step=1.0,
               phys_min=1.0, phys_max=6.0, source="跨平台多次下载重传引起的级联 MDCT 帧移位与频带塌陷")

    # --- 05 · 移动电话与网络语音 (Mobile Telephony & VoIP Speech) ---
    g_tel = "05 · 移动电话与网络语音 (Mobile & VoIP Speech)"
    aud.define("telecom_mode", "实时语音通信链路模式", 0.0, 0.0, 3.0, unit="0-3", group=g_tel, step=1.0,
               phys_min=0.0, phys_max=3.0, source="GSM 06.10 / 3GPP AMR / ITU-T G.711 & G.722 / WebRTC",
               help="0=关闭｜1=2G GSM/AMR 移动蜂窝通话 (含 217Hz TDMA 射频突发蜂音与 DTX 舒适噪声)｜"
                    "2=早期网络语音 (Skype / QQ语音：谱减法降噪音乐噪声 + 回声消除残差 + 抖动缓冲变速)｜"
                    "3=弱信号基站边缘/高丢包会议语音 (连续丢包 PLC 机械电音冻结 + VAD 吞字)。")
    aud.define("telecom_wideband", "语音带宽制式 (0窄带/1宽带)", 0.0, 0.0, 1.0, unit="0-1", group=g_tel, step=1.0,
               phys_min=0.0, phys_max=1.0, source="0=ITU-T G.712 窄带电话 (300-3400Hz)，1=G.722/AMR-WB 高清宽带语音 (50-7000Hz)")
    aud.define("telecom_vad_threshold", "语音活动检测 (VAD) 静音门限", 0.25, 0.0, 1.0, unit="0-1", group=g_tel,
               phys_min=0.0, phys_max=0.85, source="DTX 不连续传输静音抑制门限（过高会吞掉字头与尾音）")
    aud.define("telecom_cng_level", "舒适噪声生成 (CNG) 电平", 0.20, 0.0, 1.0, unit="0-1", group=g_tel,
               phys_min=0.0, phys_max=0.80, source="RFC 3389 / GSM 06.12 静音期背景合成舒适噪声")
    aud.define("telecom_packet_loss_pct", "RTP/无线帧丢包率", 0.0, 0.0, 60.0, unit="%", group=g_tel,
               phys_min=0.0, phys_max=45.0, source="Gilbert-Elliott 突发信道丢包模型")
    aud.define("telecom_plc_mode", "丢包隐藏算法 (PLC)", 1.0, 0.0, 1.0, unit="0-1", group=g_tel, step=1.0,
               phys_min=0.0, phys_max=1.0, source="0=零填充静音断续，1=ITU G.711 App.I 基音周期波形重复隐藏(连续丢包呈机械电音)")
    aud.define("telecom_aec_residue", "声学回声消除 (AEC) 残差与双讲压制", 0.0, 0.0, 1.0, unit="0-1", group=g_tel,
               phys_min=0.0, phys_max=0.90, source="自适应滤波器非线性残余回声中心削波与双讲抽吸")
    aud.define("telecom_ns_strength", "谱减法背景降噪 (NS) 强度", 0.0, 0.0, 1.0, unit="0-1", group=g_tel,
               phys_min=0.0, phys_max=0.95, source="Boll 谱减法过减因子引入的孤立谱峰『音乐噪声』(Musical Noise)")
    aud.define("telecom_jitter_ms", "网络自适应抖动缓冲变速深度", 0.0, 0.0, 150.0, unit="ms", group=g_tel,
               phys_min=0.0, phys_max=120.0, source="NetEQ 自适应抖动缓冲时间伸缩调整")
    aud.define("telecom_latency_ms", "单向传输与编解码延迟", 0.0, 0.0, 600.0, unit="ms", group=g_tel,
               phys_min=0.0, phys_max=500.0, source="端到端语音分组排队与交织延迟")

    # --- 06 · 声学空间与麦克风位置 (Acoustic Space & Microphone Placement) ---
    g_spc = "06 · 声学空间与麦克风位置 (Acoustic Space & Placement)"
    aud.define("space_preset", "物理声学空间环境预设", 0.0, 0.0, 4.0, unit="0-4", group=g_spc, step=1.0,
               phys_min=0.0, phys_max=4.0, source="三维矩形房间简正驻波模态 + 镜像源早期反射 + Sabine 混响模型",
               help="0=消声直通｜1=主播卧室/书桌近讲(3.4×2.8×2.6m 强桌面反射与低频驻波+机箱风扇)｜"
                    "2=瓷砖走廊/楼梯间(12.5×1.7×3.0m 平行硬墙颤动回声 Flutter Echo + 长混响)｜"
                    "3=大学教室/阶梯讲堂(9.6×7.2×3.4m 远距拾音多抽头反射+空调底噪)｜"
                    "4=室外街道/露天场地(无房间驻波、地面与远处楼体单次反射、空气高频吸收与风噪车流)。")
    aud.define("space_room_length_m", "房间长度 Lx", 4.2, 1.2, 40.0, unit="m", group=g_spc,
               phys_min=1.5, phys_max=30.0, source="空间纵向几何尺寸 (决定轴向简正模态 f=c/2Lx)")
    aud.define("space_room_width_m", "房间宽度 Ly", 3.4, 1.0, 40.0, unit="m", group=g_spc,
               phys_min=1.2, phys_max=30.0, source="空间横向几何尺寸 (决定侧墙早期反射与颤动回声周期)")
    aud.define("space_room_height_m", "房间高度 Lz", 2.7, 1.8, 20.0, unit="m", group=g_spc,
               phys_min=2.0, phys_max=15.0, source="空间垂直几何尺寸")
    aud.define("space_rt60_s", "混响时间 RT60 (-1自动跟随预设)", -1.0, -1.0, 4.0, unit="s", group=g_spc,
               phys_min=-1.0, phys_max=3.5, source="Sabine / Eyring 室内声能衰减 60dB 时间")
    aud.define("space_wall_absorption", "墙面高频吸声系数 (-1自动)", -1.0, -1.0, 0.98, unit="α", group=g_spc,
               phys_min=-1.0, phys_max=0.95, source="建筑材料吸声率 (0.1=釉面瓷砖/水泥, 0.5=石膏板/窗帘, 0.9=声学海绵)")
    aud.define("space_standing_waves", "房间低频简正驻波共振 (-1自动)", -1.0, -1.0, 1.0, unit="0-1", group=g_spc,
               phys_min=-1.0, phys_max=0.95, source="Schroeder 频率以下三维轴向驻波共振峰 (决非简单低频 EQ)")
    aud.define("space_mic_distance_m", "声源至麦克风距离 (-1自动)", -1.0, -1.0, 12.0, unit="m", group=g_spc,
               phys_min=-1.0, phys_max=10.0, source="临界距离直达声/混响声比 (DRR) 与 ISO 9613 大气高频吸收")
    aud.define("space_occlusion", "物理遮挡与口袋/手掌闷音", 0.0, 0.0, 1.0, unit="0-1", group=g_spc,
               phys_min=0.0, phys_max=0.95, source="手指按住麦克风孔、手机揣入口袋或隔门拾音的绕射低通与腔体闷响")
    aud.define("space_ambient_noise", "环境特征背景噪声电平", 0.0, 0.0, 1.0, unit="0-1", group=g_spc,
               phys_min=0.0, phys_max=0.90, source="按空间类型合成真实环境声床 (卧室机箱风扇/走廊镇流器/教室空调低语/室外阵风车流)")

    # --- 07 · 播放器与系统音频链 (Media Player & OS System Audio Chain) ---
    g_sys = "07 · 播放器与系统音频链 (Player & OS Audio Chain)"
    aud.define("sys_audio_preset", "播放终端与系统混音链预设", 0.0, 0.0, 4.0, unit="0-4", group=g_sys, step=1.0,
               phys_min=0.0, phys_max=4.0, source="Windows KMixer / AC'97 / Flash SoundMixer / 桌面有源音箱实测模型",
               help="0=直通无损｜1=Windows 98/XP KMixer + AC'97 板载声卡(44.1→48kHz线性重采样与总线底噪)｜"
                    "2=浏览器 Flash Player 网页视频音频链(22.05kHz重采样+环形缓冲欠载卡顿)｜"
                    "3=千禧年百元桌面 2.0 多媒体塑料有源音箱(145Hz倒相管轰鸣+SRS 3D扩展)｜"
                    "4=接触不良 3.5mm 耳机孔/外接转接线(共地阻抗串扰导致人声中置抵消与切口爆音)。")
    aud.define("sys_mixer_gain", "系统混音器音量增益", 1.0, 0.0, 2.5, unit="×", group=g_sys,
               phys_min=0.1, phys_max=2.0, source="操作系统主混音器数字增益")
    aud.define("sys_src_mode", "系统/浏览器重采样器模式", 0.0, 0.0, 2.0, unit="0-2", group=g_sys, step=1.0,
               phys_min=0.0, phys_max=2.0, source="0=无损直通，1=WinXP KMixer 44.1→48kHz 线性插值，2=Flash Player 22.05kHz 快速抽取")
    aud.define("sys_channel_map", "输出声道映射与接线状态", 0.0, 0.0, 3.0, unit="0-3", group=g_sys, step=1.0,
               phys_min=0.0, phys_max=3.0, source="0=正常立体声，1=左右声道反接，2=3.5mm 地线虚焊(L-R差模去人声)，3=单声道左路")
    aud.define("sys_eq_preset", "播放器均衡器与低音增强预设", 0.0, 0.0, 3.0, unit="0-3", group=g_sys, step=1.0,
               phys_min=0.0, phys_max=3.0, source="0=平直，1=摇滚 V 型微笑曲线，2=流行人声突出，3=Mega Bass 桌面箱低音增强")
    aud.define("sys_spatial_3d", "3D 环绕立体声扩展与激励 (SRS/BBE)", 0.0, 0.0, 1.0, unit="0-1", group=g_sys,
               phys_min=0.0, phys_max=0.95, source="差模侧向信号加宽与哈斯效应延迟激励")
    aud.define("sys_buffer_underrun", "音频环形缓冲区欠载卡顿率", 0.0, 0.0, 1.0, unit="0-1", group=g_sys,
               phys_min=0.0, phys_max=0.85, source="DMA / 浏览器音频线程阻塞导致 32ms 缓冲片重复卡碟颤音")
    aud.define("sys_device_switch_pop", "输出设备热切换瞬态冲击", 0.0, 0.0, 1.0, unit="0-1", group=g_sys,
               phys_min=0.0, phys_max=0.90, source="声卡隔直电容充放电冲击与继电器切换静音间隙")
    aud.define("sys_mixer_clip", "系统混音器定点溢出硬削波", 0.0, 0.0, 1.0, unit="0-1", group=g_sys,
               phys_min=0.0, phys_max=0.90, source="16-bit 混音母线无动态余量饱和削顶")

    # --- 08 · 铃声、MIDI 与移动音源 (Ringtones, MIDI & Mobile Sound Engines) ---
    g_ring = "08 · 铃声、MIDI 与移动音源 (Ringtones & Mobile Synth)"
    aud.define("ringtone_mode", "移动合成音源与铃声引擎", 0.0, 0.0, 4.0, unit="0-4", group=g_ring, step=1.0,
               phys_min=0.0, phys_max=4.0, source="Nokia RTTTL 压电蜂鸣器 / Yamaha MA-2 & MA-3 FM 芯片 / 早期低位深波表合成",
               help="0=关闭(使用正常伴音)｜1=单音压电蜂鸣器(Nokia RTTTL 纯方波单声部经典铃声)｜"
                    "2=4/16和弦 FM 手机合成芯片(Yamaha MA-2/MA-3 正弦查找表调制与阶梯包络)｜"
                    "3=早期低位深 PCM 波表 MIDI 音源(8-bit/12-bit 循环采样 SoundFont)｜"
                    "4=输入伴音实时转译为手机和弦/单音铃声(实时追踪主频峰并按最大复音数抢占发声)。")
    aud.define("ringtone_polyphony", "最大同时发声复音数 (Polyphony)", 4.0, 1.0, 64.0, unit="声部", group=g_ring, step=1.0,
               phys_min=1.0, phys_max=64.0, source="硬件通道上限 (1=单音蜂鸣器, 4=早期4和弦, 16=16和弦, 64=64和弦；超限触发旧音截断 Voice Stealing)")
    aud.define("ringtone_tick_quant_ms", "音序器时基 Tick 量化网格", 25.0, 0.0, 150.0, unit="ms", group=g_ring,
               phys_min=0.0, phys_max=120.0, source="MIDI / RTTTL 硬件定时器中断粒度")
    aud.define("ringtone_env_steps", "硬件音量包络寄存器阶数", 16.0, 2.0, 128.0, unit="阶", group=g_ring, step=1.0,
               phys_min=4.0, phys_max=128.0, source="音源芯片包络发生器离散衰减步进 (低阶数产生特征性拉链噪声)")
    aud.define("ringtone_wavetable_bits", "音源查找表/波表位深", 8.0, 4.0, 16.0, unit="bit", group=g_ring, step=1.0,
               phys_min=4.0, phys_max=16.0, source="FM 正弦表或 PCM 波表 ROM 采样位宽")
    aud.define("ringtone_mix", "铃声/MIDI 音源混合比", 1.0, 0.0, 1.0, unit="0-1", group=g_ring,
               phys_min=0.0, phys_max=1.0, source="合成音源与原始伴音的干湿混合度")

    # --- 09 · 广播与网络平台的响度处理 (Broadcast & Platform Loudness / EBU R 128) ---
    g_loud = "09 · 广播与网络平台的响度处理 (Loudness & EBU R 128)"
    aud.define("loudness_mode", "响度标准化与动态压限模式", 0.0, 0.0, 4.0, unit="0-4", group=g_loud, step=1.0,
               phys_min=0.0, phys_max=4.0, source="ITU-R BS.1770-4 / EBU R 128 / ATSC A/85 / 流媒体平台响度规范",
               help="0=仅开启 K 加权 LUFS 与真峰值测量表(不改变音频)｜"
                    "1=EBU R 128 广播标准响度归一化(-23.0 LUFS, -1.0 dBTP 真峰值限幅)｜"
                    "2=FM 电台/电视商业广告『响度战争』多段压限器(Orban Optimod 风格 -9.5 LUFS 高密度抽吸轰炸)｜"
                    "3=网络视频平台响度归一化(YouTube/B站 -14.0 LUFS 自动增益惩罚与真峰值削顶)｜"
                    "4=跨平台多次搬运重转码响度漂移与动态塌陷。")
    aud.define("loudness_target_lufs", "目标综合响度 (Integrated LUFS)", -14.0, -32.0, -5.0, unit="LUFS", group=g_loud,
               phys_min=-28.0, phys_max=-6.0, source="ITU-R BS.1770-4 K-加权门限综合响度目标值")
    aud.define("loudness_compressor", "广播多段压缩与响度激励深度", 0.0, 0.0, 1.0, unit="0-1", group=g_loud,
               phys_min=0.0, phys_max=0.95, source="三频段分频动态压缩与自动增益补偿 (减小 LRA 响度范围)")
    aud.define("loudness_true_peak_dbtp", "真峰值砖墙限幅上限 (True Peak)", -1.0, -9.0, 0.0, unit="dBTP", group=g_loud,
               phys_min=-6.0, phys_max=0.0, source="4倍过采样采样间真峰值 (Inter-sample True Peak) 上限")
    aud.define("loudness_retranscode_passes", "平台级联归一化重转码轮数", 1.0, 1.0, 6.0, unit="轮", group=g_loud, step=1.0,
               phys_min=1.0, phys_max=6.0, source="不同平台目标响度不一致引发的多代动态挤压")

    # --- 10 · 音视频同步与时钟系统 (Audio-Video Sync & Master Clock System) ---
    g_avsync = "10 · 音视频同步与时钟系统 (A/V Sync & Clock System)"
    aud.define("av_sync_enabled", "启用音视频同步与主时钟系统", 0.0, 0.0, 1.0, unit="bool", group=g_avsync, step=1.0,
               phys_min=0.0, phys_max=1.0, source="MPEG PTS/DTS 时间戳、声卡/采集卡独立晶振与主控同步模型",
               help="打通摄像机、DV、磁带走带、线性/非线性编辑、广播播出与网络流媒体的统一音画时钟耦合系统。")
    aud.define("av_sync_offset_ms", "固定音画唇音同步偏移 (Lip-Sync Offset)", 0.0, -1000.0, 1000.0, unit="ms", group=g_avsync,
               phys_min=-500.0, phys_max=500.0, source="ITU-R BT.1359 音画同步感知容限 (+值表示声音滞后于画面，-值表示声音超前)")
    aud.define("av_clock_drift_ppm", "音视频独立晶振时钟漂移 (Clock Drift)", 0.0, -5000.0, 5000.0, unit="ppm", group=g_avsync,
               phys_min=-2000.0, phys_max=2000.0, source="音频采样时钟与视频帧率时钟未锁相 (Non-Genlocked) 导致的随时间线性累积滑移")
    aud.define("av_frame_drop_rate", "视频掉帧/重复帧概率", 0.0, 0.0, 0.90, unit="0-1", group=g_avsync,
               phys_min=0.0, phys_max=0.75, source="USB 摄像头带宽瓶颈、早期 NLE 预览掉帧或流媒体卡顿引发的画面停顿")
    aud.define("av_drop_sync_mode", "视频掉帧时伴音同步策略", 0.0, 0.0, 2.0, unit="0-2", group=g_avsync, step=1.0,
               phys_min=0.0, phys_max=2.0, source="音视频缓冲区追赶与重同步机制",
               help="0=时钟解耦自由滑移(画面卡住但声音继续播放，导致后续音画错位累积)｜"
                    "1=硬锁帧 FIFO 滑移(视频掉帧时伴音同步重复/丢弃对应帧切片，产生跳帧咔哒声但保持同步)｜"
                    "2=PLL 锁相环平滑变调重采样(动态微调音频速度追赶视频 PTS)。")
    aud.define("av_edit_splice_mode", "编辑切点伴音拼接瞬态", 0.0, 0.0, 2.0, unit="0-2", group=g_avsync, step=1.0,
               phys_min=0.0, phys_max=2.0, source="磁带组合/插入编辑与早期非线编切点声学瞬态",
               help="0=无缝拼接｜1=非过零硬切口(4ms 控制轨断流静音间隙 + 偏磁阶跃咔哒声与相位跳变)｜2=NLE 5ms V型交叉淡化凹陷。")
    aud.define("av_edit_interval_s", "周期性编辑切点间隔", 1.5, 0.25, 10.0, unit="s", group=g_avsync,
               phys_min=0.5, phys_max=10.0, source="时间线分镜头切换周期")

    g_bcast = "广播音响与调音台 (Broadcast Console)"
    aud.define("broadcast_agc", "广播级AGC限幅压缩器", 0.0, 0.0, 1.0, unit="0-1", group=g_bcast,
               phys_min=0.0, phys_max=1.0, source="CBS Audimax/Volumax 广播压限器模型",
               help="演播室播出级动态压制 (快启慢释自适应抽吸效应/Pumping Breathing)。")
    aud.define("studio_room_reverb", "录播间/主播间空间混响", 0.0, 0.0, 1.0, unit="0-1", group=g_bcast,
               phys_min=0.0, phys_max=1.0, source="室内声学早期反射与混响模型",
               help="房间墙面早期反射与短RT60混响尾音。")

    g_opt = "胶片感光光学声带 (Film Optical Soundtrack)"
    aud.define("film_optical_scratches", "光学声带划痕爆音", 0.20, 0.0, 1.0, unit="0-1", group=g_opt,
               phys_min=0.0, phys_max=1.0, source="声带物理磨损光强跃变模型",
               help="走片磨损划伤声带或沾染尘埃通过光缝产生的瞬态光跃变爆音 (Clicks & Pops)。")
    aud.define("film_exciter_hum", "励磁灯交流声", 0.15, 0.0, 1.0, unit="0-1", group=g_opt,
               phys_min=0.0, phys_max=1.0, source="钨丝灯交流纹波光强调制模型",
               help="老式放映机钨丝励磁灯交流供电引起的 50Hz/100Hz (或 60Hz/120Hz) 光强调制蜂鸣。")
    aud.define("film_slit_loss", "光学光缝高频截止", 0.40, 0.0, 1.0, unit="0-1", group=g_opt,
               phys_min=0.0, phys_max=1.0, source="微型狭缝孔径损耗 sinc 模型",
               help="声头 12.5 微米微型光缝孔径效应导致的高频物理截止 (~6.8kHz 35mm / ~4.8kHz 16mm)。")


    # --- 电影胶片（16mm / 35mm）化学成像与电视胶转磁系统 ---
    g = "电影胶片化学成像与电视胶转磁 (Film & Telecine)"
    cam.define("film_enabled", "启用电影胶片与胶转磁系统", 0.0, 0.0, 1.0, unit="bool", group=g, step=1.0,
               phys_min=0.0, phys_max=1.0, source="Rank Cintel Mk III 飞点胶转磁与 16mm/35mm 卤化银感光模型",
               help="开启后，成像前端切换为 16mm/35mm 卤化银化学胶片感光、三层彩色耦合染料、醋酸片基老化、机械片门二维抖动与飞点扫描胶转磁系统。")
    cam.define("film_stock", "电影胶片型号与冲洗工艺", 0.0, 0.0, 10.0, unit="0-10", group=g, step=1.0,
               phys_min=0.0, phys_max=10.0, source="Kodak Eastman 5247 / Ektachrome 7240 / Fujicolor 8514",
               help="0=关闭｜1=Eastman Color 35mm 5247 负片+印片｜2=Kodak Ektachrome 16mm 7240 新闻反转片｜"
                    "3=Fujicolor 16mm 8514 纪实胶片｜4=早期醋酸片基暖红棕褪色档案拷贝。")
    cam.define("film_gamma", "H&D 特性曲线反差斜率", 1.0, 0.5, 2.0, unit="γ", group=g,
               phys_min=0.75, phys_max=1.45, source="Hurter–Driffield D(log E) 感光特性曲线",
               help="控制卤化银感光乳剂在线性曝光区的反差斜率，同时保留胶片趾部与肩部的软饱和滚降。")
    cam.define("film_halation", "胶片防光晕层红橙背散射", 0.35, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.90, source="片基背面反射穿透底层感红乳剂",
               help="强光穿透三层乳剂后在片基背面反射回底层感红层，在高光边缘产生标志性的暖红/橙红光晕环（区别于 CCD 垂直拖尾）。")
    cam.define("film_grain_amount", "卤化银/染料云颗粒强度", 0.28, 0.0, 1.2, unit="0-1", group=g,
               phys_min=0.05, phys_max=0.75, source="中密度区曝光依赖染料云团簇模型",
               help="胶片颗粒并非均匀白噪声，而是空间随机团簇，在中密度区（∝√(D(D_max-D))）最强、全黑与高光饱和区自然收敛。")
    cam.define("film_grain_size", "胶片颗粒团簇尺度", 1.45, 0.6, 4.0, unit="px", group=g,
               phys_min=0.8, phys_max=2.8, source="35mm 约 1.0-1.4 px / 16mm 约 1.5-2.4 px")
    cam.define("film_dye_fade", "三层耦合染料非对称褪色", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.85, source="Eastmancolor 青/黄染料水解速率快于品红染料",
               help="模拟档案电影拷贝中青色与黄色染料先褪去、品红染料残留导致的经典暖红棕/洋红偏色。")
    cam.define("film_shrinkage", "醋酸片基酸败收缩 (Vinegar Syndrome)", 0.0, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.75, source="醋酸纤维素脱乙酰化收缩与齿孔节距失配",
               help="老化醋酸片基收缩引起的片门张力翘曲与边缘几何呼吸。")
    cam.define("film_scratches_dust", "纵向走片划痕与单帧灰尘", 0.15, 0.0, 1.2, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.70, source="压片滑轨持续划痕与片门单帧灰尘/毛发")
    cam.define("film_gate_weave", "片门机械二维抖动 (Gate Weave)", 0.25, 0.0, 1.5, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.85, source="间歇输片爪定位公差（整幅刚性平移，区别于磁带逐行时基抖动）",
               help="电影放映机/胶转磁片门定位误差：整幅画面刚性二维平移 (Δx_f, Δy_f)，同一帧内所有扫描行位移完全相同。")
    cam.define("film_shutter_flicker", "旋转翼板快门闪烁", 0.12, 0.0, 1.0, unit="0-1", group=g,
               phys_min=0.0, phys_max=0.55, source="双叶/三叶机械旋转快门与弧光灯波动")
    cam.define("telecine_pulldown", "胶转磁扫描与下拉制式 (Pulldown)", 0.0, 0.0, 3.0, unit="0-3", group=g, step=1.0,
               phys_min=0.0, phys_max=3.0, source="Rank Cintel 24fps->29.97i (3:2 Pulldown) / 24fps->25i (PAL +4.167%)",
               help="0=24fps 逐行直扫｜1=NTSC 3:2 Pulldown (AA-BB-BC-CD-DD 五帧周期，在 BC/CD 帧产生跨胶片帧隔行梳状伪影)｜"
                    "2=PAL 25fps 加速胶转磁 (+4.167% 加速与 +70.67 cents 伴音升调)｜3=2×2 跨媒介历史对照实验室 (胶片→胶转磁→VHS→流媒体)。")
    cam.define("telecine_quad_corner", "胶转磁对照角落选择 (0四分屏/1左上/2右上/3左下/4右下)", 0.0, 0.0, 4.0, unit="角落", group=g, step=1.0,
               phys_min=0.0, phys_max=4.0, source="跨媒介四分屏角落拆选",
               help="在 2×2 跨媒介对照模式（telecine_pulldown=3）下拆成每个角落单独全屏选择：0=2×2四分屏，1=左上角（胶片原片），2=右上角（3:2胶转磁母带），3=左下角（家用VHS磁带录屏），4=右下角（早期流媒体重压）。")
    cam.define("telecine_show_border", "胶转磁对照分框样式 (0无分框/1细十字线/2标准分框)", 1.0, 0.0, 2.0, unit="分框", group=g, step=1.0,
               phys_min=0.0, phys_max=2.0, source="分屏分框显示控制",
               help="0=无分框无缝拼接，1=细十字分框线，2=标准分框。")
    cam.define("telecine_show_text", "胶转磁对照画面文字 (0不要文字/1仅标签/2标签+状态栏)", 0.0, 0.0, 2.0, unit="文字", group=g, step=1.0,
               phys_min=0.0, phys_max=2.0, source="纯净画面无文字开关",
               help="默认 0=不要文字（纯净画面）；1=仅显示四角标签；2=显示四角标签与底部状态栏。")

    # --- 测量、误差预算、参数反演与三级参数审计系统 ---
    g_inv = "参数反演、误差预算与三级参数审计 (Inversion & FADGI Lab)"
    ply.define("inv_lab_enabled", "启用参数反演与误差预算台", 0.0, 0.0, 1.0, unit="bool", group=g_inv, step=1.0,
               phys_min=0.0, phys_max=1.0, source="FADGI 数字化质量指南与多解置信椭圆反演器",
               help="开启后，实时反推退化画面的等效带宽、行时基抖动、片门抖动与色度误差，并审计 Physical/Device/Artistic 三级参数贡献。")
    ply.define("inv_lab_mode", "反演与审计台视图模式", 0.0, 0.0, 3.0, unit="0-3", group=g_inv, step=1.0,
               phys_min=0.0, phys_max=3.0, source="可验证视听物理实验室",
               help="0=关闭｜1=参数反演与多解竞争假设 HUD｜2=三级参数分离审计 (Physical/Device/Artistic) HUD｜"
                    "3=2×2 参数反演、残差热力图与 FADGI 星级误差预算实验室。")
    ply.define("inv_quad_corner", "反演实验室角落选择 (0四分屏/1左上/2右上/3左下/4右下)", 0.0, 0.0, 4.0, unit="角落", group=g_inv, step=1.0,
               phys_min=0.0, phys_max=4.0, source="反演实验室角落拆选",
               help="在 2×2 参数反演与误差预算实验室下拆成每个角落单独全屏选择：0=2×2四分屏，1=左上角，2=右上角，3=左下角，4=右下角。")
    ply.define("inv_show_border", "反演实验室分框样式 (0无分框/1细十字线/2标准分框)", 1.0, 0.0, 2.0, unit="分框", group=g_inv, step=1.0,
               phys_min=0.0, phys_max=2.0, source="分屏分框显示控制",
               help="0=无分框无缝拼接，1=细十字分框线，2=标准分框。")
    ply.define("inv_show_text", "反演实验室画面文字 (0不要文字/1仅标签/2标签+统计栏)", 0.0, 0.0, 2.0, unit="文字", group=g_inv, step=1.0,
               phys_min=0.0, phys_max=2.0, source="纯净画面无文字开关",
               help="默认 0=不要文字（纯净画面）；1=仅显示四角标签；2=显示四角标签与底部统计栏。")
    ply.define("inv_fadgi_target", "FADGI 目标合规星级 (1-4★)", 3.0, 1.0, 4.0, unit="★", group=g_inv, step=1.0,
               phys_min=1.0, phys_max=4.0, source="FADGI 1★(Access) .. 4★(Archival Master)")
    ply.define("inv_inject_probe", "注入垂直边缘与多波群反演探针", 0.0, 0.0, 1.0, unit="bool", group=g_inv, step=1.0,
               phys_min=0.0, phys_max=1.0, source="在线盲反演辅助参考探针")
    ply.define("inv_confidence", "反演置信区间水平", 0.90, 0.50, 0.99, unit="0-1", group=g_inv,
               phys_min=0.68, phys_max=0.95, source="参数反演置信椭圆覆盖率")

    g_arch = "档案保存生命周期与媒介考古溯源"
    ply.define("archival_enabled", "启用档案保存生命周期与媒介考古系统", 0.0, 0.0, 1.0, unit="bool", group=g_arch, step=1.0,
               phys_min=0.0, phys_max=1.0, source="IASA-TC 04 / FADGI / NFSA 视听档案生命周期模型",
               help="模拟『原始记录 → 生产传播 → 载体保存与阿伦尼乌斯加速老化/设备消亡风险 → 抢救数字化与再编目』四阶段档案生命周期。")
    ply.define("archival_mode", "档案生命周期与版本谱系模式", 0.0, 0.0, 4.0, unit="0-4", group=g_arch, step=1.0,
               phys_min=0.0, phys_max=4.0, source="ArchivalLineageTree 多版本并存谱系树",
               help="0=跟随总开关｜1=原始载体残件 (Original Carrier Artifact · 含温湿度水解/剩磁衰减与完整原始 VITC/隔行/过扫描特征)｜"
                    "2=档案保存母版 (Preservation Master · 10-bit 无损全画幅含 VBI/HBI 完整采集，重要属性保留率 98%+)｜"
                    "3=过度清洗商业修复版 (Over-Cleaned Restoration · 激进 3D-DNR 塑料蜡像感 + 过扫描裁切抹除磁头切换带 + 丢弃隔行与 VITC，演示『保存成功≠信息完整』)｜"
                    "4=2×2 多版本谱系树对照实验室 (同屏对比：原始载体残件 vs 保存母版 vs 过度清洗修复版 vs 网络流媒体版 + FADGI/NFSA 重要属性损耗表)。")
    ply.define("archival_storage_years", "库房保存年限", 28.0, 0.0, 60.0, unit="年", group=g_arch,
               phys_min=1.0, phys_max=50.0, source="IASA-TC 04 磁带/胶片保存周期")
    ply.define("archival_temp_c", "档案库房平均温度", 23.0, 5.0, 45.0, unit="°C", group=g_arch,
               phys_min=8.0, phys_max=35.0, source="Arrhenius 化学动力学速率温度项 (ISO 18923)")
    ply.define("archival_rh_pct", "档案库房平均相对湿度 (RH)", 62.0, 15.0, 95.0, unit="%", group=g_arch,
               phys_min=25.0, phys_max=85.0, source="聚氨酯黏合剂水解与醋酸综合症湿度催化项")
    ply.define("archival_obsolescence", "回放机具与备件消亡指数", 0.55, 0.0, 1.0, unit="0-1", group=g_arch,
               phys_min=0.0, phys_max=0.95, source="UNESCO 磁带危机备件存量与技师校准能力衰减",
               help="模拟原厂录像机停产、旋转磁头备件枯竭与专用接口消亡导致的访问成本倍增及非标回放失真。")
    ply.define("archival_dnr_overrestore", "商业修复 3D-DNR 与锐化过度清洗强度", 0.72, 0.0, 1.0, unit="0-1", group=g_arch,
               phys_min=0.0, phys_max=0.90, source="过度数字化去噪导致的纹理抹平与运动鬼影",
               help="模拟将原始介质颗粒当作噪声一并抹除的『塑料蜡像感』、硬边光环以及裁切消隐区导致的考古证据链断裂。")
    ply.define("archival_quad_corner", "档案谱系树分屏角落选择", 0.0, 0.0, 4.0, unit="0-4", group=g_arch, step=1.0,
               phys_min=0.0, phys_max=4.0, source="四分屏单象限全屏放大检视",
               help="0=2×2 四分屏同屏对照｜1=左上角全屏 (原始载体残件)｜2=右上角全屏 (档案保存母版)｜"
                    "3=左下角全屏 (过度清洗修复版)｜4=右下角全屏 (流媒体传播版与重要属性损耗审计表)。")
    ply.define("archival_show_border", "档案谱系树分框线显示", 2.0, 0.0, 2.0, unit="0-2", group=g_arch, step=1.0,
               phys_min=0.0, phys_max=2.0, source="四分屏边框模式 (0=无边框, 1=细十字线, 2=档案库金属外框)")
    ply.define("archival_show_text", "档案谱系树文字叠加显示", 0.0, 0.0, 2.0, unit="0-2", group=g_arch, step=1.0,
               phys_min=0.0, phys_max=2.0, source="0=隐藏全部文字(纯净画面)｜1=紧凑角标｜2=完整考古元数据与损耗表")


    # ------------------------------------------------------------------ A2: 01b 胶片光化学 (胶片/LOMO)
    flm = s.layer("film")
    gf_chem = "卤化银感光乳剂与 H&D 特性曲线 (Photochemical Emulsion)"
    flm.define("enabled", "启用独立胶片光化学与 LOMO 暗房模块", 0.0, 0.0, 1.0, unit="bool", group=gf_chem, step=1.0,
               phys_min=0.0, phys_max=1.0, source="独立 01b 胶片光化学层 (`atsim/codecs/photochemical_film.py`)",
               help="开启后，信号链 01b 胶片光化学阶段对画面执行完整的三层减色法卤化银 H&D 特性曲线、层间抑制效应 (DIR)、C-41/E-6 负冲正正冲负交叉冲洗、过期底雾、LOMO 弯月镜暗角、异形桃心散景、海绵老化漏光及飞点胶转磁仿真。")
    flm.define("stock", "胶片乳剂型号", 0.0, 0.0, 20.0, unit="0-20", group=gf_chem, step=1.0,
               phys_min=0.0, phys_max=20.0, source="Kodak Vision3 500T / Double-X 5222 / Kodachrome 64 / Ektachrome 100D 感光曲线标定",
               help="0=关闭｜1=Eastman Color 35mm 5247 (电影负片+印片)｜2=Kodak Ektachrome 16mm 7240 (反转片)｜"
                    "3=Fujicolor 16mm 8514 (第4感青层纪实片)｜4=早期醋酸片基暖棕褪色拷贝｜"
                    "5=Agfa Vista / Optima 200 (琥珀暖金高饱和)｜"
                    "6=Kodak Ektachrome E100VS / Elite Chrome 100 (高反差反转片)｜"
                    "7=Lomography Color Negative 400 (玫瑰洋红/丁香紫暗部与奶油高光)｜"
                    "8=Expired Svema / ORWO Vintage Film (高底雾 D_min 粉彩鼠尾草绿)｜"
                    "9=Polaroid SX-70 / 600 Time-Zero Integral Film (即时显影药浆与 TiO2 奶油高光)｜"
                    "10=Lomography LomoChrome Purple XR 100-400 (绿→紫二向色耦合平移)｜"
                    "11=90s Tungsten Acid Olive-Chartreuse (90年代钨丝灯酸橄榄黄绿迫冲胶片)｜"
                    "12=Prussian Cyan-Navy & Electric Violet Rain (普鲁士深青蓝与电光紫雨景)｜"
                    "13=Fujifilm Pro 400H Pastel Candy High-Key (富士高调粉彩马卡龙奶油白)｜"
                    "14=Vintage Brass-Olive & Maroon-Plum Petzval (复古黄铜橄榄与深紫红暗部)｜"
                    "15=LOMO LC-A Minitar-1 Turquoise-Emerald X-Pro (松石青绿高反差交叉冲洗)｜"
                    "16=Meyer Trioplan Soap-Bubble Golden-Violet Dream (梅耶肥皂泡散景丁香紫与金白光雾)。")
    flm.define("gamma", "H&D 特性曲线反差斜率 (γ)", 1.0, 0.5, 2.0, unit="γ", group=gf_chem,
               phys_min=0.75, phys_max=1.55, source="Hurter–Driffield D(log E) 感光特性曲线",
               help="控制卤化银感光乳剂在线性曝光区的反差斜率，同时保留趾部 (Toe) 与肩部 (Shoulder) 的软饱和滚降。")
    flm.define("halation", "防光晕层红橙背散射 (Anti-Halation Bounce)", 0.35, 0.0, 1.5, unit="0-1", group=gf_chem,
               phys_min=0.0, phys_max=0.95, source="片基背面全反射先穿透底层感红乳剂 (Red Layer)",
               help="强光穿透三层乳剂后在醋酸/聚酯片基背面反射回底层感红层，在高光边缘产生标志性的暖红/橙红光晕环。")
    flm.define("grain_amount", "卤化银/染料云颗粒强度", 0.28, 0.0, 1.2, unit="0-1", group=gf_chem,
               phys_min=0.05, phys_max=0.80, source="中密度区曝光依赖染料云团簇模型 ∝√(D(D_max-D))",
               help="胶片颗粒在中间调达到峰值，而在全黑趾部与过曝肩部自然收敛。")
    flm.define("grain_size", "染料云颗粒团簇尺度", 1.45, 0.6, 4.0, unit="px", group=gf_chem,
               phys_min=0.8, phys_max=2.8, source="35mm 约 1.1-1.4 px / 16mm & 高感 LOMO 约 1.5-2.4 px")

    gf_dark = "LOMO 暗房冲洗工艺、正冲负/负冲正 (Xpro) 与反卷曝光"
    flm.define("process_mode", "暗房冲洗工艺模式 (标准/E6→C41/C41→E6/迫冲/留银/2×2对照)", 0.0, 0.0, 5.0, unit="0-5", group=gf_dark, step=1.0,
               phys_min=0.0, phys_max=5.0, source="C-41 / E-6 交叉冲洗化学耦合动力学实测标定",
               help="0=标准冲洗 (C-41/E-6)｜1=正冲负 E-6 in C-41 Xpro（反转片入负片药水：无橙色掩膜致 γ 陡增 1.38×、暗部祖母绿/青色深谷 + 高光暖金分离）｜"
                    "2=负冲正 C-41 in E-6 Xpro（彩色负片入反转药水：未抵消橙色掩膜令暗部呈丁香紫/洋红、高光呈奶油暖黄）｜"
                    "3=迫冲/减冲 Push/Pull｜4=留银冲洗 Bleach Bypass（跳过漂白保留金属银颗粒与高反差去饱和）｜"
                    "5=2×2 LOMO 光化学暗房四象限对照实验室。")
    flm.define("xpro_strength", "交叉冲洗 (Xpro) 分色与 S 曲线强度", 0.78, 0.0, 1.5, unit="0-1", group=gf_dark,
               phys_min=0.0, phys_max=1.20, source="LOMO Xpro RGB 分位数实测标定",
               help="控制正冲负/负冲正时的三通道独立 γ 失配与暗部/高光互补色分色深度。")
    flm.define("xpro_tint_mode", "Xpro 冷暖分色极性 (0自动/1青绿金盏花/2紫红玫瑰/3琥珀可可)", 0.0, 0.0, 3.0, unit="0-3", group=gf_dark, step=1.0,
               phys_min=0.0, phys_max=3.0, source="LOMO 经典三大分色方向",
               help="0=跟随冲洗工艺自动选择｜1=暗部翡翠青绿 + 高光暖金｜2=暗部丁香紫/玫瑰红 + 高光奶油白｜3=暗部可可暖褐 + 中间调琥珀金。")
    flm.define("expired_fog", "过期胶片热底雾抬升与感蓝层衰减 (D_min Fog)", 0.0, 0.0, 1.0, unit="0-1", group=gf_dark,
               phys_min=0.0, phys_max=0.90, source="向日葵实测黑位 p02=0.406 与高感顶层感蓝卤化银宇宙射线热雾化",
               help="模拟过期胶卷存放期间顶层高感卤化银热雾化导致的基底最小密度 D_min 抬升（死黑变成粉彩鼠尾草绿/奶油灰雾）、最高密度 D_max 下降及颗粒粗化。")
    flm.define("redscale_flip", "Redscale 胶片片基反卷曝光", 0.0, 0.0, 1.0, unit="0-1", group=gf_dark,
               phys_min=0.0, phys_max=1.0, source="LOMO Redscale XR 反装片基先穿透感红层模型",
               help="将彩色胶片反转片基面朝向镜头曝光：光线先经过底层感红乳剂，被中间层吸收后再有少量黄绿光到达感绿层，感蓝层几乎无光，形成炽烈红/橙/金黄影调。")
    flm.define("push_pull_ev", "迫冲/减冲显影时长偏移 (Push/Pull EV)", 0.0, -2.0, 3.0, unit="EV", group=gf_dark,
               phys_min=-1.5, phys_max=2.5, source="延长/缩短第一显影时间改变反差 γ 与有效感光度")

    gf_lomo = "LOMO 塑料弯月镜光学、异形散景、海绵漏光与重曝"
    flm.define("lomo_vignette", "LOMO 弯月镜头 cos^4 θ 机械暗角", 0.0, 0.0, 1.2, unit="0-1", group=gf_lomo,
               phys_min=0.0, phys_max=0.95, source="LOMO LC-A Minitar-1 32mm / Holga 120N 60mm 单片塑料弯月镜边缘照度跌落",
               help="建模 LOMO LC-A 与 Holga 120N 的强烈渐晕暗角：四角不仅照度按 cos^4 θ 跌落，且落入胶片趾部产生更浓的色彩偏移。")
    flm.define("lomo_edge_smear", "弯月单镜片边缘场曲与切向慧差抹开", 0.0, 0.0, 1.2, unit="0-1", group=gf_lomo,
               phys_min=0.0, phys_max=0.90, source="未校正匹兹伐场曲 (Petzval Field Curvature) 与边缘色散",
               help="中心保持锐利合焦，而四角随视场半径 r^2 发生各向异性柔化与红蓝倍率色差分离。")
    flm.define("bokeh_shape", "异形光圈散景光斑模板 (0圆/1桃心/2星芒/3菱形/4旋焦/5肥皂泡亮环/6六边形彩灯)", 0.0, 0.0, 6.0, unit="0-6", group=gf_lomo, step=1.0,
               phys_min=0.0, phys_max=6.0, source="异形孔径 PSF 卷积：桃心散景 + 肥皂泡球差亮环散景 + 六边形机械光圈散景",
               help="0=关闭/标准圆光圈｜1=Heart Bokeh 桃心散景光斑｜"
                    "2=Starburst 四角星芒插片｜3=Diamond 菱形叶片孔径｜4=Petzval Swirl 匹兹伐旋焦散景｜"
                    "5=Soap-Bubble Ring 肥皂泡亮边圆环散景（三片式 Trioplan 过校正球差亮环）｜"
                    "6=Hexagonal Fairy-Light 六边形奶油彩灯散景（6片机械光圈暖金光斑）。")
    flm.define("bokeh_amount", "异形散景光斑与梦幻焦外强度", 0.0, 0.0, 1.2, unit="0-1", group=gf_lomo,
               phys_min=0.0, phys_max=0.85, source="高光点源与异形光圈孔径 PSF 卷积及焦外光斑簇合成")
    flm.define("bloom_veil_mode", "LOMO 光学眩光、逆光太阳洪流与高调白雾模式 (0关/1黄金逆光洪流/2奶油白雾柔焦/3烛火双层光冕/4斜向冷暖分色光雾/5四象限实验室)", 0.0, 0.0, 5.0, unit="0-5", group=gf_lomo, step=1.0,
               phys_min=0.0, phys_max=5.0, source="无镀膜/单层镀膜老镜头逆光耀斑、Mie 散射白雾与防光晕层背散射物理标定",
               help="0=关闭｜1=黄金时刻逆光太阳耀斑洪流（顶部大面积明黄过曝核向全画幅倾泻琥珀金光晕，暗部保持深棕红）｜"
                    "2=高调奶油白雾眩光与无镀膜柔焦（无增透膜老镜头全画幅 Mie 散射白雾抬升黑位 p02≈0.39）｜"
                    "3=点光源双层烛火光冕与乳剂防光晕红环（炽白核心 + 暖金内日冕 + 红橙防光晕背散射外环 + 焦外弥散圆斑）｜"
                    "4=斜向双色温冷紫-暖金光学光雾（左上奶油柠檬白 + 中央丁香紫阴影谷 + 右侧玫瑰琥珀散景）｜"
                    "5=2×2 LOMO 光学光晕与散景四象限对照实验室。")
    flm.define("bloom_veil_amount", "光学光晕、逆光耀斑与镜头白雾强度", 0.0, 0.0, 1.5, unit="0-1", group=gf_lomo,
               phys_min=0.0, phys_max=1.20, source="无镀膜/单层镀膜 LOMO 镜组内反射眩光与高光扩散包络",
               help="控制所选 `bloom_veil_mode` 光学光晕/逆光洪流/奶油白雾/烛火光冕的总注入强度。")
    flm.define("soft_focus_mist", "前组柔焦镜 / 凡士林漫射白雾 (Soft-Focus Mist)", 0.0, 0.0, 1.2, unit="0-1", group=gf_lomo,
               phys_min=0.0, phys_max=0.95, source="大光圈球差柔焦核 + 前镜片微水汽/柔焦网漫反射",
               help="保留画面底层清晰结构轮廓的同时，将高光与中间调笼罩在一层梦幻奶油柔焦薄雾中。")
    flm.define("dual_tone_split", "斜向冷紫-暖金光学分色光雾强度", 0.0, 0.0, 1.2, unit="0-1", group=gf_lomo,
               phys_min=0.0, phys_max=0.95, source="斜向逆光入射角引起的镀膜透射/反射互补色场",
               help="在画面上叠加随入射角连续变化的左上奶油暖白、中轴丁香紫阴影与右侧暖金玫瑰光学分色场。")
    flm.define("tilt_shift_dof", "移轴镜头 / 微距浅景深平面散焦 (Tilt-Shift Planar DOF)", 0.0, 0.0, 1.2, unit="0-1", group=gf_lomo,
               phys_min=0.0, phys_max=0.95, source="沙姆定律 (Scheimpflug Principle) 倾斜焦平面与焦外散景膨胀",
               help="仅在 `focus_plane_y` 水平/微倾焦平面附近保持极度锐利合焦（如岩石裂缝与烛芯、近景纸船），向远景与前景迅速过渡为奶油级大光斑散焦。")
    flm.define("focus_plane_y", "移轴/微距合焦平面纵向位置 Y", 0.65, 0.1, 0.9, unit="0-1", group=gf_lomo,
               phys_min=0.15, phys_max=0.85, source="0.0=画面顶部合焦，0.65=中下部主体合焦，1.0=画面底部合焦")
    flm.define("swirl_amount", "匹兹伐镜头切向猫眼旋焦强度 (Petzval Swirl Astigmatism)", 0.0, 0.0, 1.2, unit="0-1", group=gf_lomo,
               phys_min=0.0, phys_max=0.95, source="19世纪 Petzval 匹兹伐人像镜无场曲校正切向像散与机械渐晕猫眼散景",
               help="画面中心主体（如复古街灯）保持锐利，而外围树叶与背景散景光斑沿切向圆弧方向旋转拉长，形成强烈的漩涡状散景包络。")
    flm.define("light_leak_amount", "塑料机壳与老化遮光海绵漏光强度", 0.0, 0.0, 1.2, unit="0-1", group=gf_lomo,
               phys_min=0.0, phys_max=0.90, source="实拍漏光梯度与片基正背面漏光分色物理",
               help="模拟 Holga / Diana F+ 塑料卡扣缝隙或老化遮光海绵漏光：从乳剂正面漏入呈奶油白/金黄/粉洋红，从片基背面穿透防光晕层漏入呈炽烈橙红。")
    flm.define("light_leak_pattern", "漏光空间分布形态 (0无/1右上暖阳/2左侧红边/3粉心光轨/4四角漫射)", 0.0, 0.0, 4.0, unit="0-4", group=gf_lomo, step=1.0,
               phys_min=0.0, phys_max=4.0, source="LOMO 常见 4 类漏光几何位置",
               help="0=无漏光｜1=右上角金琥珀斜向暖阳漏光｜"
                    "2=左侧铰链垂直红橙背散射漏光条（35mm 后盖海绵老化）｜"
                    "3=横贯画面粉红/洋红桃心光斑串｜"
                    "4=左下与右上双象限梦幻奶油光雾。")
    flm.define("double_exposure", "未过片二次重曝叠印 (Double Exposure / MX)", 0.0, 0.0, 1.0, unit="0-1", group=gf_lomo,
               phys_min=0.0, phys_max=0.80, source="卤化银潜影空间对数曝光量叠加 E_tot = E_1 + α·E_2",
               help="在胶片线性曝光域（进入 H&D 特性曲线之前）叠加第二幅错位旋转底片的曝光潜影，还原 LOMO 机械快门原地重曝。")

    gf_proj = "胶片老化、放映机械与 Rank Cintel 飞点胶转磁"
    flm.define("dye_fade", "三层耦合染料非对称褪色", 0.0, 0.0, 1.0, unit="0-1", group=gf_proj,
               phys_min=0.0, phys_max=0.85, source="Eastmancolor 青/黄染料水解速率快于品红染料")
    flm.define("shrinkage", "醋酸片基酸败收缩 (Vinegar Syndrome)", 0.0, 0.0, 1.0, unit="0-1", group=gf_proj,
               phys_min=0.0, phys_max=0.75, source="醋酸纤维素脱乙酰化收缩与齿孔节距失配")
    flm.define("scratches_dust", "纵向走片划痕与单帧灰尘", 0.0, 0.0, 1.2, unit="0-1", group=gf_proj,
               phys_min=0.0, phys_max=0.70, source="压片滑轨持续划痕与片门单帧灰尘")
    flm.define("gate_weave", "片门机械二维抖动 (Gate Weave)", 0.0, 0.0, 1.5, unit="0-1", group=gf_proj,
               phys_min=0.0, phys_max=0.85, source="间歇输片爪定位公差（整幅刚性平移）")
    flm.define("shutter_flicker", "旋转翼板快门闪烁", 0.0, 0.0, 1.0, unit="0-1", group=gf_proj,
               phys_min=0.0, phys_max=0.55, source="双叶/三叶机械旋转快门与弧光灯波动")
    flm.define("telecine_pulldown", "胶转磁扫描与下拉制式 (Pulldown)", 0.0, 0.0, 3.0, unit="0-3", group=gf_proj, step=1.0,
               phys_min=0.0, phys_max=3.0, source="Rank Cintel 24fps->29.97i (3:2 Pulldown) / 24fps->25i (PAL +4.167%)",
               help="0=24fps 逐行直扫｜1=NTSC 3:2 Pulldown｜2=PAL 25fps 加速胶转磁｜3=2×2 跨媒介历史对照实验室。")
    flm.define("quad_corner", "胶片暗房/胶转磁四分屏角落选择", 0.0, 0.0, 4.0, unit="角落", group=gf_proj, step=1.0,
               phys_min=0.0, phys_max=4.0, source="2×2 四分屏角落拆选",
               help="0=2×2四分屏对照｜1=左上角全屏｜2=右上角全屏｜3=左下角全屏｜4=右下角全屏。")
    flm.define("show_border", "胶片暗房四分屏分框样式", 1.0, 0.0, 2.0, unit="分框", group=gf_proj, step=1.0,
               phys_min=0.0, phys_max=2.0, source="0=无分框, 1=细十字分框线, 2=暗房金属外框")
    flm.define("show_text", "胶片暗房四分屏画面文字", 0.0, 0.0, 2.0, unit="文字", group=gf_proj, step=1.0,
               phys_min=0.0, phys_max=2.0, source="默认 0=不要文字（纯净画面）；1=仅显示四角标签；2=显示标签与底部暗房遥测栏")

    # 新增：显影动力学、抗晕光结构、光学印片代际、醋酸综合征动力学、湿门扫描与可逆数字修复参数
    flm.define("dev_temp_c", "显影药液温度 T", 38.0, 15.0, 50.0, unit="°C", group=gf_dark,
               phys_min=18.0, phys_max=45.0, source="Kodak ECN-2 (41.1°C) / C-41 & E-6 (38.0°C) / D-96 (20.0°C) 显影动力学",
               help="控制显影反应速率与各感光层反差演化 D_c = f_c(H_c, t, T, a)。")
    flm.define("dev_time_norm", "显影时间倍率 t", 1.0, 0.4, 2.0, unit="×", group=gf_dark,
               phys_min=0.6, phys_max=1.6, source="显影时间对潜影银还原密度与层间扩散的影响")
    flm.define("bath_activity", "药液活性与补充状态 a", 1.0, 0.4, 1.4, unit="0-1.4", group=gf_dark,
               phys_min=0.6, phys_max=1.25, source="显影液新鲜度、溴化物累积与 DIR 层间边缘效应强度")
    flm.define("orange_mask", "负片橙色遮罩校准度", 1.0, 0.0, 1.5, unit="0-1.5", group=gf_chem,
               phys_min=0.5, phys_max=1.2, source="彩色负片偶合剂橙色遮罩与扫描光源正像转换校准")
    flm.define("remjet_backing", "Remjet 碳黑抗晕光背层 (-1默认/0去除/1完整)", -1.0, -1.0, 1.0, unit="-1..1", group=gf_lomo,
               phys_min=-1.0, phys_max=1.0, source="Kodak 电影胶片 Remjet 碳黑背层吸收率（0=去除背层产生强烈高光红晕）")
    flm.define("aged_backing_dye", "老化片基背层抗晕光染料重显", 0.0, 0.0, 1.0, unit="0-1", group=gf_lomo,
               phys_min=0.0, phys_max=0.85, source="FilmCare.org 醋酸片基产酸老化后背层脱色抗晕光染料重新显色")
    flm.define("optical_print_gen", "光学印片复制代际 (0直扫/1一代/2中间片/3发行拷贝)", 0.0, 0.0, 3.0, unit="代", group=gf_proj, step=1.0,
               phys_min=0.0, phys_max=3.0, source="Optical Printer 负片→翻正片(IP)→翻底片(IN)→发行拷贝(2383) 多代光化学复制链")
    flm.define("vinegar_syndrome", "醋酸综合征自催化退化强度", 0.0, 0.0, 1.0, unit="0-1", group=gf_proj,
               phys_min=0.0, phys_max=0.95, source="NARA / CCI 醋酸纤维素脱乙酰自催化产酸、聚合物解聚、收缩与脆化动力学")
    flm.define("storage_temp_c", "档案保存环境温度", 20.0, -10.0, 50.0, unit="°C", group=gf_proj,
               phys_min=-5.0, phys_max=40.0, source="ISO 18911 / IPI 胶片库房保存温度")
    flm.define("storage_rh", "档案保存相对湿度", 45.0, 10.0, 95.0, unit="%RH", group=gf_proj,
               phys_min=20.0, phys_max=85.0, source="水分催化醋酸片基脱乙酰水解与彩色染料褪色")
    flm.define("emulsion_damage", "乳剂与黏结层开裂/起皱/银镜", 0.0, 0.0, 1.0, unit="0-1", group=gf_proj,
               phys_min=0.0, phys_max=0.85, source="明胶乳剂微裂纹、网状起皱 (Reticulation) 与高密度边缘银镜 (Silver Mirroring)")
    flm.define("shutter_blades", "放映机快门叶片数 (2叶48Hz / 3叶72Hz)", 2.0, 2.0, 3.0, unit="叶", group=gf_proj, step=1.0,
               phys_min=2.0, phys_max=3.0, source="SMPTE 24fps 电影放映机双叶片(48次/秒光脉冲)与三叶片(72次/秒光脉冲)")
    flm.define("sprocket_wear", "齿孔与间歇爪机械磨损", 0.0, 0.0, 1.0, unit="0-1", group=gf_proj,
               phys_min=0.0, phys_max=0.85, source="16mm/35mm 齿孔撕裂磨损引发的纵向片门跳格与走片不稳")
    flm.define("wet_gate", "湿门扫描划痕抑制 (Wet-Gate)", 0.0, 0.0, 1.0, unit="0-1", group=gf_proj,
               phys_min=0.0, phys_max=1.0, source="FIAF 数字化指南：折射率匹配液填充片基划痕消除光散射")
    flm.define("scanner_diffuse", "扫描仪积分球漫射照明度", 0.5, 0.0, 1.0, unit="0-1", group=gf_proj,
               phys_min=0.0, phys_max=1.0, source="0=准直聚光镜光源凸显表面划痕，1=漫射积分球光源抑制划痕阴影")
    flm.define("restoration_strength", "可逆数字修复与四层证据分离强度", 0.0, 0.0, 1.0, unit="0-1", group=gf_proj,
               phys_min=0.0, phys_max=1.0, source="Disney Research / FIAF 红外缺陷检测、褪色恢复、稳像去闪与四层数字证据分离")

    return s



# ---------------------------------------------------------------------------
# per-unit spread: which parameters differ between two machines of one model
# ---------------------------------------------------------------------------
# The value of this table is as much in what it omits as in what it contains. A spread is
# declared only where there is a reason to expect unit variation:
#
#   * alignment and gain tolerances -- an amplifier's gain is set by resistors with a
#     tolerance, so the same model comes out a few per cent apart; a servo loop is trimmed;
#   * component tolerances whose effect is known -- a CCD's dark current varies per die,
#     which is sensor noise; an electrolytic capacitor's ESR varies, which is supply ripple;
#   * wear, which differs per unit because it depends on use rather than on manufacture.
#
# Artistic parameters (`halation`, `lens_flare`, `highlight_tint`) are left at zero on purpose.
# An invented spread is worse than none: it looks like provenance and stops the next person
# from checking. The fraction is one sigma, clipped to two when a unit is drawn, because an
# alignment tolerance is a bound and beyond twice it the machine is broken rather than a unit
# of the model.
UNIT_SPREADS: dict[str, tuple[float, str]] = {
    # camcorder sensor -- per-die variation, which is what "this camera" means
    "sensor_noise": (0.30, "CCD dark current and read noise vary per die"),
    "fixed_pattern": (0.25, "per-die fixed-pattern non-uniformity"),
    # camcorder lens and signal path -- alignment tolerances
    "chroma_aberration": (0.20, "lens centring tolerance"),
    "sharpness": (0.12, "aperture-correction trim"),
    "ccd_gain": (0.15, "video amplifier gain trim"),
    "ccd_smear": (0.12, "transfer efficiency varies per device"),
    "dynamic_range": (0.06, "well capacity and clip point"),
    "color_cast": (0.05, "white-balance null trim; a small residual cast is normal"),
    "exposure": (0.10, "AE reference trim"),
    "white_balance_temp": (0.12, "AWB reference trim"),
    # deck -- servo trims and head wear
    "head_wear": (0.35, "wear depends on use, not on the model"),
    "transport_jitter": (0.20, "capstan servo gain trim"),
    "frequency_response": (0.15, "replay equaliser trim"),
    "luma_bandwidth": (0.05, "replay amplifier bandwidth trim"),
    "chroma_bandwidth": (0.05, "chroma amplifier bandwidth trim"),
    "comb_depth": (0.08, "delay-line and summing network tolerances"),
    "cross_color": (0.10, "separation residue depends on the comb's alignment"),
    "cross_luma": (0.10, "as above"),
    "burst_phase_error": (0.60, "APC loop and crystal tolerance; a few degrees is normal"),
    "tbc_strength": (0.08, "corrector loop gain trim"),
    "tbc_residual": (0.25, "corrector residual after alignment"),
    "ringing": (0.20, "replay filter Q"),
    "color_noise": (0.20, "chroma amplifier noise figure"),
    # display -- convergence, geometry and drive trims
    "gamma": (0.04, "gun and video-drive tracking"),
    "geometry_warp": (0.20, "yoke and convergence adjustment"),
    "defocus": (0.15, "focus and astigmatism adjustment"),
    "contrast": (0.08, "video drive preset"),
    "brightness": (0.06, "cut-off preset"),
    "saturation": (0.08, "chroma drive preset"),
    "color_temp": (0.06, "grey-scale tracking"),
    "scanlines": (0.10, "beam spot size and focus"),
    "vignette": (0.12, "shadow-mask and yoke geometry"),
    "interlace_flicker": (0.10, "field timing and phosphor persistence"),
    "crt_age": (0.30, "hours of use, not the model"),
    "bloom": (0.10, "high-voltage regulation per set"),
    # audio -- fixed head azimuth screw, capstan roller eccentricity, preamp noise
    "azimuth_loss": (0.10, "fixed audio head azimuth screw alignment"),
    "wow_flutter": (0.12, "capstan roller eccentricity and belt tension"),
    "hiss_floor": (0.08, "playback preamp transistor noise figure"),
    "head_switch_buzz": (0.10, "rotary transformer cross-coupling tolerance"),
    "hifi_unlock_thresh": (0.08, "AFM DOC envelope comparator threshold"),
    "linear_bandwidth": (0.06, "audio head gap wear"),
}


def _apply_unit_spreads(s: "ParamSchema") -> None:
    """
    Set each declared spread on its parameter, and check the name exists.

    Setting the field rather than editing the define calls keeps the declaration in one readable
    table -- with the reason next to the number -- and cannot break the file, which two earlier
    attempts at splicing did.
    """
    for key, (spread, _why) in UNIT_SPREADS.items():
        found = False
        for layer in s.layer_names():
            specs = s.layer(layer).specs
            if key in specs:
                specs[key].unit_spread = float(spread)
                found = True
        if not found:
            raise KeyError("UNIT_SPREADS names a parameter that does not exist: %r" % key)


SCHEMA = build_schema()


# ---------------------------------------------------------------------------
# device profile libraries
#
# A profile is a named set of layer values describing a real device or medium, kept
# separate from the factory presets so the two can be mixed: a preset is a whole
# scenario, a profile is one part of the chain.
#
# Values marked in `source` as estimates are exactly that -- derived from reference
# stills and the failure descriptions supplied with them, not from a bench
# measurement.  Marking them keeps the difference visible.
# ---------------------------------------------------------------------------

MEDIA_PROFILES = {
    "vhs_sp": {"label": "VHS SP 标准", "note": "磁迹最宽、重放输出最高的一档", "layers": {
        "tape": {"tape_speed": 0.0}}},
    "vhs_lp": {"label": "VHS LP 长时", "note": "走带 1/2，重放输出约 -6 dB", "layers": {
        "tape": {"tape_speed": 1.0}}},
    "vhs_ep": {"label": "VHS EP 超长", "note": "走带 1/3，重放输出约 -10 dB", "layers": {
        "tape": {"tape_speed": 2.0}}},
    "vhs_gen_5": {
        "label": "VHS 第 5 代翻录带（色度右移下垂 + 初级轮廓振铃）",
        "note": "5 次磁带迭代翻录：条纹与肤色开始向右下方溢出边界，垂直边缘出现预加重黑白伴影（第 5 代 VCR→VCR 翻录实测标定）",
        "layers": {
            "format": {"luma_bandwidth": 2.6, "chroma_bandwidth": 0.55, "edge_boost": 0.35, "color_noise": 0.28},
            "tape": {"generation_passes": 5.0, "gen_chroma_bleed": 0.85, "gen_ringing": 0.85,
                     "tape_stability": 0.52, "line_jitter": 0.32, "noise_floor": 0.20, "head_switching": 0.80}}},
    "vhs_gen_12": {
        "label": "VHS 第 12 代翻录带（重度振铃黑边 + 色度分层脱节）",
        "note": "12 次磁带迭代翻录：多级二阶过冲在人物轮廓外拉出粗黑边与白晕，色度大幅下垂右移，黑电平抬升泛绿（第 12 代 VCR→VCR 翻录实测标定）",
        "layers": {
            "format": {"luma_bandwidth": 2.1, "chroma_bandwidth": 0.42, "edge_boost": 0.48, "color_noise": 0.42},
            "tape": {"generation_passes": 12.0, "gen_chroma_bleed": 0.95, "gen_ringing": 0.95,
                     "tape_stability": 0.35, "line_jitter": 0.52, "noise_floor": 0.30, "head_switching": 0.92}}},
    "vhs_worn": {"label": "磨损 VHS 带", "note": "多次走带后的磁介质", "layers": {
        "tape": {"wear": 0.55, "dropout_rate": 0.9, "noise_floor": 0.32,
                 "crease_rate": 0.35, "chroma_loss": 0.22}}},
    "cosmag_e180_pal_splp_good": {
        "label": "COSMAG E-180（PAL/SP/LP 混合 · 258分钟采录成功段）",
        "note": "标称 180分钟(SP)/360分钟(LP) 混合录制磁带中成功抢救的 258 分钟段落：SP/LP 磁迹切换包络、轻度岁月磨损与偶发磁粉失落",
        "layers": {
            "format": {"luma_bandwidth": 2.95, "chroma_bandwidth": 0.62, "chroma_gain": 0.94, "color_noise": 0.20},
            "tape": {"tape_speed": 0.0, "sp_lp_mix": 0.42, "tape_stability": 0.68, "line_jitter": 0.22,
                     "walk_amplitude": 0.18, "row_timebase": 0.08, "wear": 0.24, "dropout_rate": 0.38,
                     "noise_floor": 0.16, "crease_rate": 0.14, "demagnetization": 0.08,
                     "base_cracking": 0.04, "binder_loss": 0.03, "head_switching": 0.78,
                     "head_switch_lines": 6.0, "tracking_error": 0.24}}},
    "cosmag_e180_pal_splp_fail": {
        "label": "COSMAG E-180（PAL/SP/LP 混合 · 61分钟采录失败段：消磁/龟裂/发霉/破损）",
        "note": "61 分钟高危失效段落：同时爆发剩磁衰减（消磁）、PET 带基脆性微裂纹（带基龟裂）、潮湿黏合剂霉变脱粉（发霉）与边缘破损 CTL 控制轨失步（带基破损）",
        "layers": {
            "format": {"luma_bandwidth": 2.35, "chroma_bandwidth": 0.48, "chroma_gain": 0.38, "color_noise": 0.42},
            "tape": {"tape_speed": 1.0, "sp_lp_mix": 0.75, "tape_stability": 0.28, "line_jitter": 0.65,
                     "walk_amplitude": 0.55, "row_timebase": 0.45, "wear": 0.68, "dropout_rate": 2.10,
                     "noise_floor": 0.38, "crease_rate": 0.65, "binder_loss": 0.34, "shed_grain": 0.38,
                     "demagnetization": 0.52, "base_cracking": 0.62, "base_breakage": 0.58,
                     "stiction_rate": 0.85, "stiction_amount": 0.65,
                     "row_chroma_error": 0.26, "row_chroma_phase": 40.0,
                     "head_switching": 0.95, "head_switch_lines": 10.0, "tracking_error": 0.62}}},
    "svhs": {"label": "S-VHS", "note": "亮度带宽约 5 MHz，色度仍欠载", "layers": {
        "format": {"luma_bandwidth": 5.0, "chroma_bandwidth": 0.75}}},
    "hi8": {"label": "Hi8", "note": "Sony 1/4 英寸 CCD + 模拟行记录（4.2 MHz FM 亮度 + 743 kHz 欠载色度），无宏块结构", "layers": {
        "camera": {"sensor_type": 0.0, "sensor_noise": 0.09, "ccd_gain": 0.68,
                   "sharpness": 0.80, "chroma_aberration": 0.05, "dynamic_range": 7.5,
                   "color_cast": 0.45, "white_balance_temp": 0.0,
                   "ccd_smear": 0.28, "halation": 0.28},
        "format": {"luma_bandwidth": 4.2, "chroma_bandwidth": 0.75,
                   "chroma_vertical_averaging": 0.55, "comb_depth": 0.65,
                   "cross_color": 0.28, "cross_luma": 0.18, "color_noise": 0.22,
                   "edge_boost": 0.24},
        "tape": {"shuttle_mode": 1.0, "tape_stability": 0.76, "wear": 0.15,
                 "dropout_rate": 0.15, "noise_floor": 0.14, "line_jitter": 0.14,
                 "walk_amplitude": 0.10, "micro_jitter": 0.08,
                 "head_switching": 0.35, "head_switch_lines": 6.0,
                 "tracking_error": 0.15}}},
    "digital8": {"label": "Digital8", "note": "Sony 1/4 英寸 CCD + 在 Hi8 8mm 磁带上记录 DV25 (4:1:1 + DCT) 数字分量流", "layers": {
        "camera": {"sensor_type": 0.0, "sensor_noise": 0.08, "ccd_gain": 0.65,
                   "sharpness": 0.82, "chroma_aberration": 0.04, "dynamic_range": 7.5,
                   "color_cast": 0.45, "white_balance_temp": 0.0,
                   "ccd_smear": 0.25, "halation": 0.25},
        "format": {"luma_bandwidth": 5.0, "chroma_bandwidth": 2.0,
                   "chroma_vertical_averaging": 0.0, "color_noise": 0.04, "edge_boost": 0.18},
        "codec": {"dv_enabled": 1.0, "dv_quality": 0.90, "dv_base_step": 0.015},
        "tape": {"shuttle_mode": 2.0, "dropout_rate": 0.0, "noise_floor": 0.03,
                 "line_jitter": 0.03, "walk_amplitude": 0.02, "micro_jitter": 0.03,
                 "head_switching": 0.0}}},
    "video8": {"label": "Video8", "note": "8mm 家用机，带宽低于 Hi8", "layers": {
        "format": {"luma_bandwidth": 3.0}, "tape": {"shuttle_mode": 1.0}}},
    "minidv": {"label": "MiniDV", "note": "数字分支：4:1:1 + DCT", "layers": {
        "codec": {"dv_enabled": 1.0, "dv_quality": 1.0},
        "tape": {"shuttle_mode": 2.0}}},
    "mouldy": {"label": "发霉脱粉带", "note": "黏合剂吸水、磁粉层脱落", "layers": {
        "format": {"chroma_gain": 0.12},
        "tape": {"shed_grain": 0.75, "luma_pedestal": -0.16, "binder_loss": 0.35,
                 "dropout_rate": 2.2, "surface_damage": 0.0}}},
    "shed_coating": {"label": "严重脱粉", "note": "涂层大面积缺失", "layers": {
        "format": {"chroma_gain": 0.10},
        "tape": {"binder_loss": 0.7, "shed_grain": 0.85, "luma_pedestal": -0.18,
                 "noise_floor": 0.6}}},
}

PLAYER_PROFILES = {
    "deck_consumer": {"label": "消费级录像机", "note": "无 TBC，走带残差明显", "layers": {
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.45,
                   "frequency_response": 0.15, "comb_depth": 0.55}}},
    "deck_with_tbc": {"label": "带 TBC 的专业机", "note": "时基校正接上，抖动大幅下降", "layers": {
        "player": {"tbc_enabled": 1.0, "tbc_strength": 0.9, "tbc_residual": 0.12,
                   "transport_jitter": 0.15}}},
    "deck_worn_heads": {"label": "磁头磨损的录像机", "note": "高频损失 + 色度信噪比下降", "layers": {
        "player": {"head_wear": 0.65, "frequency_response": 0.35,
                   "color_noise": 0.45}}},
    "deck_panasonic_m9000": {
        "label": "松下 M9000（磁鼓轴承油变质）",
        "note": "肩扛式 VHS 通病：磁鼓转速不足 → 行微量重复 → 横向颗粒",
        "layers": {"tape": {"drum_speed": 0.97},
                   "player": {"tbc_enabled": 0.0, "transport_jitter": 0.5}}},
    "deck_jvc_srs388e": {"label": "JVC SR-S388E（S-VHS 编辑机）",
                         "note": "信号过差时会丢弃磁带画面并输出自身蓝底 + OSD",
                         "layers": {"player": {"tbc_enabled": 1.0, "tbc_strength": 0.95}}},
    "deck_sharp_vca62dt_only": {
        "label": "夏普 VC-A62DT（主采录机直出：4视频磁头 + 1音频磁头 + 1时间轴磁头）",
        "note": "主采录机单机工作：4 颗旋转视频磁头（2×SP + 2×LP）负责读取混合磁迹，1 颗固定音频磁头，1 颗 25Hz 时间轴控制磁头（CTL）；未接后续画面稳定机时，老带时基摆动直接暴露",
        "layers": {
            "player": {"tbc_enabled": 0.0, "stabilizer_stages": 0.0, "transport_jitter": 0.32,
                       "head_wear": 0.16, "preemphasis": 0.24, "ringing": 0.16, "ringing_freq": 0.50,
                       "luma_bandwidth": 3.1, "chroma_bandwidth": 0.65, "separation_kind": 2.0,
                       "comb_depth": 0.65, "cross_color": 0.28, "cross_luma": 0.18, "color_noise": 0.18}}},
    "deck_sharp_vca62dt_h96_stage1": {
        "label": "夏普 VC-A62DT → 夏普 VC-H96（二级采录 · 一级画面稳定）",
        "note": "主采录机 VC-A62DT 复合信号进入二级录像机夏普 VC-H96 进行一级行同步再生与水平 AFC 钳位稳定",
        "layers": {
            "player": {"tbc_enabled": 1.0, "tbc_strength": 0.72, "tbc_residual": 0.08,
                       "stabilizer_stages": 1.0, "transport_jitter": 0.14, "head_wear": 0.14,
                       "preemphasis": 0.22, "ringing": 0.15, "ringing_freq": 0.50,
                       "luma_bandwidth": 3.0, "chroma_bandwidth": 0.62, "separation_kind": 2.0,
                       "comb_depth": 0.72, "cross_color": 0.22, "cross_luma": 0.14, "color_noise": 0.15}}},
    "deck_sharp_vca62dt_h96_m9e_cascade": {
        "label": "夏普 VC-A62DT → 夏普 VC-H96（一级稳定）→ 夏普 VC-M9E（二级稳定）三机级联",
        "note": "完整三级录像机时基接力：主采录机 VC-A62DT（4视频+1音频+1时间轴磁头）→ 二级 VC-H96（一级画面稳定）→ 三级 VC-M9E（二级画面稳定），将行同步抖动压缩 ~90% 后送入天创恒达 TC-UB530 采集卡",
        "layers": {
            "player": {"tbc_enabled": 1.0, "tbc_strength": 0.92, "tbc_residual": 0.03,
                       "stabilizer_stages": 2.0, "transport_jitter": 0.06, "head_wear": 0.12,
                       "preemphasis": 0.20, "ringing": 0.14, "ringing_freq": 0.50,
                       "luma_bandwidth": 2.9, "chroma_bandwidth": 0.60, "separation_kind": 2.0,
                       "comb_depth": 0.78, "cross_color": 0.18, "cross_luma": 0.12, "color_noise": 0.12}}},
    "deck_panasonic_j27": {
        "label": "松下 NV-J27（PAL-D 捷克/中国制式录像机 · VHF 6频道/AV输出）",
        "note": "PAL-D 制式（6.5 MHz 伴音间隔、VHF 6 频道 171.25 MHz 图像载波）高频头解调与 1H 玻璃延迟线 Y/C 分离，经 AV 复合端子进入电脑 USB 采集卡",
        "layers": {
            "player": {"tbc_enabled": 0.0, "transport_jitter": 0.14, "head_wear": 0.08,
                       "preemphasis": 0.28, "ringing": 0.22, "ringing_freq": 0.52,
                       "luma_bandwidth": 3.4, "chroma_bandwidth": 0.68,
                       "separation_kind": 2.0, "comb_depth": 0.68,
                       "cross_color": 0.32, "cross_luma": 0.22, "color_noise": 0.14}}},
    "deck_panasonic_nv180": {
        "label": "松下 NV-180（1986 背包便携式 VHS 录像机）",
        "note": "双磁头斜进扫描便携机：石英锁相平稳走带 + 底部 7 行磁头切换水平跳变与右侧 HBI 黑边阶跃 + 550 kHz 欠载色度群延迟拖尾与高频预加重过冲（对照 1986 松下初代 CCD + NV-180 实拍标定）",
        "layers": {
            "tape": {"tape_speed": 0.0, "tape_stability": 0.88, "line_jitter": 0.04,
                     "walk_amplitude": 0.04, "micro_jitter": 0.03, "row_timebase": 0.02,
                     "head_switching": 0.90, "head_switch_lines": 4.0, "tracking_error": 0.52,
                     "dropout_rate": 0.08, "noise_floor": 0.06, "wear": 0.08},
            "player": {"tbc_enabled": 1.0, "tbc_strength": 0.88, "tbc_residual": 0.04,
                       "transport_jitter": 0.05, "head_wear": 0.10,
                       "preemphasis": 0.32, "ringing": 0.18, "ringing_freq": 0.48,
                       "luma_bandwidth": 2.8, "chroma_bandwidth": 0.55, "comb_depth": 0.72,
                       "cross_color": 0.18, "cross_luma": 0.12, "color_noise": 0.10}}},
}

DISPLAY_PROFILES = {
    "crt_consumer": {
        "label": "家用球面 CRT 电视（交错槽孔荫罩 Slot Mask）",
        "note": "90 年代 21 寸球面彩电：交错槽孔荫罩、高光束斑发散、玻壳内反射光晕与边角像散",
        "layers": {"display": {
            "enabled": 1.0, "gamma": 2.35, "scanlines": 0.35, "spot_sigma": 0.42, "scan_pitch": 2.0,
            "spot_bloom": 0.48, "bloom": 0.35, "glass_halation": 0.32, "phosphor_saturation": 0.35,
            "mask_type": 1.0, "mask_pitch": 3.0, "mask_depth": 0.38, "phosphor_rgb": 0.55,
            "geometry_warp": 0.15, "curvature": 0.22, "corner_defocus": 0.32,
            "convergence": 0.22, "convergence_y": 0.12, "ehv_sag": 0.28, "breathing": 0.18,
            "phosphor_persistence": 0.22, "overscan": 0.0, "vignette": 0.25,
            "interlace_flicker": 0.25, "reflection": 0.14, "bezel": 0.35,
        }},
    },
    "sony_trinitron_kv1214": {
        "label": "Sony Trinitron KV-1214（1978 12寸柱面特丽珑 · 4K屏摄摩尔纹标定）",
        "note": "12寸柱面特丽珑 4K 屏摄实测标定：单枪三束粗节距垂直荫栅 (HFx/HFy≈1.48)、单根水平钨丝阻尼线、水平单向柱面玻壳、NTSC-J 9300K 青蓝白点 (R/G≈0.78, B/G≈1.11) 与清晰弧形屏摄摩尔纹",
        "layers": {"display": {
            "enabled": 1.0, "gamma": 2.28, "contrast": 0.62, "brightness": 0.47, "saturation": 0.56,
            "scanlines": 0.34, "spot_sigma": 0.38, "scan_pitch": 2.0, "spot_bloom": 0.44,
            "bloom": 0.28, "glass_halation": 0.34, "phosphor_saturation": 0.36,
            "mask_type": 0.0, "mask_pitch": 2.35, "mask_depth": 0.42, "phosphor_rgb": 0.78,
            "geometry_warp": 0.08, "curvature": 0.10, "corner_defocus": 0.18,
            "convergence": 0.14, "convergence_y": 0.05, "ehv_sag": 0.18, "breathing": 0.12,
            "phosphor_persistence": 0.22, "overscan": 0.02, "vignette": 0.20,
            "interlace_flicker": 0.18, "reflection": 0.10, "bezel": 0.28,
            "kv1214_enabled": 1.0, "kv1214_grille_pitch_px": 2.35, "kv1214_grille_contrast": 0.68,
            "kv1214_moire_amount": 0.62, "kv1214_moire_freq": 0.052, "kv1214_moire_tilt_deg": 1.8,
            "kv1214_cylindrical_bow": 0.34, "kv1214_ntscj_9300k": 0.78,
            "kv1214_single_damper": 0.65, "kv1214_bezel_frame": 0.45, "kv1214_ab_split": 0.0,
        }},
    },
    "crt_strong_magnet_faceplate": {
        "label": "CRT 遇强磁铁贴面实验（钕铁硼强磁黑洞 + 十字枕形折叠 + 同心彩虹纯度环）",
        "note": "强永磁体近场磁偏转与铁磁荫罩剩磁物理标定：洛伦兹力 F=-e(v×B) 电子束径向/切向偏转、中心电子束排空黑洞、十字内凹几何折叠与多阶同心色纯度彩虹环",
        "layers": {"display": {
            "enabled": 1.0, "gamma": 2.30, "scanlines": 0.32, "mask_type": 1.0, "mask_pitch": 2.8,
            "crt_magnet_mode": 1.0, "crt_magnet_strength": 0.85,
            "crt_magnet_x": 0.52, "crt_magnet_y": 0.48, "crt_magnet_z_dist": 0.12,
            "crt_magnet_angle_deg": 25.0, "crt_magnet_polarity": 1.0,
            "crt_magnet_remanence": 0.28, "crt_degauss_trigger": 0.0,
        }},
    },
    "crt_trinitron": {
        "label": "Sony 特丽珑 PVM 监视器（单枪三束荫栅 Aperture Grille）",
        "note": "柱面特丽珑显像管：连续垂直 RGB 荧光条、双水平钨丝阻尼线阴影、高会聚精度与锐利扫描线",
        "layers": {"display": {
            "enabled": 1.0, "gamma": 2.2, "scanlines": 0.28, "spot_sigma": 0.36, "scan_pitch": 2.0,
            "spot_bloom": 0.32, "bloom": 0.22, "glass_halation": 0.20, "phosphor_saturation": 0.25,
            "mask_type": 0.0, "mask_pitch": 2.5, "mask_depth": 0.32, "phosphor_rgb": 0.65,
            "geometry_warp": 0.08, "curvature": 0.08, "corner_defocus": 0.12,
            "convergence": 0.08, "convergence_y": 0.04, "ehv_sag": 0.12, "breathing": 0.08,
            "phosphor_persistence": 0.16, "overscan": 0.0, "vignette": 0.18,
            "interlace_flicker": 0.18, "reflection": 0.08, "bezel": 0.22,
        }},
    },
    "crt_arcade_dot": {
        "label": "80 年代球面点状荫罩彩电（Delta 三角点阵 Dot Mask）",
        "note": "早期球面荧光屏：交错圆孔 RGB 三角点阵、明显球面曲率、边角会聚偏离与高压呼吸效应",
        "layers": {"display": {
            "enabled": 1.0, "gamma": 2.45, "scanlines": 0.42, "spot_sigma": 0.40, "scan_pitch": 2.2,
            "spot_bloom": 0.62, "bloom": 0.45, "glass_halation": 0.45, "phosphor_saturation": 0.42,
            "mask_type": 2.0, "mask_pitch": 3.4, "mask_depth": 0.48, "phosphor_rgb": 0.72,
            "geometry_warp": 0.22, "curvature": 0.38, "corner_defocus": 0.48,
            "convergence": 0.38, "convergence_y": 0.24, "ehv_sag": 0.45, "breathing": 0.32,
            "phosphor_persistence": 0.32, "overscan": 0.02, "vignette": 0.32,
            "interlace_flicker": 0.28, "reflection": 0.20, "bezel": 0.55,
        }},
    },
    "capture_card": {"label": "采集卡直入", "note": "不经显像管，几何与扫描线不参与",
                     "layers": {"display": {"enabled": 0.0}}},
    "crt_reshoot_fast_shutter": {
        "label": "CRT 屏摄重拍 · 1/125s 高速快门扫描滚带 (Shutter-Field Beat Roll Bar)",
        "note": "用 1/125s 快门相机重拍 59.94Hz CRT：单次快门仅捕获约 48% 场扫描高度，呈现边缘带 P22 红粉余辉的明暗滚动扫描横带",
        "layers": {"display": {
            "enabled": 1.0, "gamma": 2.30, "scanlines": 0.36, "spot_sigma": 0.40, "scan_pitch": 2.0,
            "spot_bloom": 0.45, "bloom": 0.32, "glass_halation": 0.30, "mask_type": 1.0, "mask_pitch": 2.8,
            "mask_depth": 0.36, "phosphor_rgb": 0.65, "curvature": 0.20, "bezel": 0.32,
            "reshoot_enabled": 1.0, "reshoot_shutter_hz": 125.0, "reshoot_fps_offset": 1.20,
            "reshoot_rolling_shutter": 0.35, "reshoot_moire": 0.22, "reshoot_ambient_glare": 0.15,
            "reshoot_wb_shift": 0.25,
        }},
    },
    "crt_reshoot_angled_moire": {
        "label": "CRT 屏摄重拍 · 斜角微距二维彩色摩尔纹 (2D Subpixel-Mask Moiré)",
        "note": "相机微倾 2.8° 离轴近拍球面槽孔/特丽珑 CRT：传感器拜耳网格与球面荫罩/扫描线混叠出彩色环形水波摩尔纹",
        "layers": {"display": {
            "enabled": 1.0, "gamma": 2.25, "scanlines": 0.38, "spot_sigma": 0.38, "scan_pitch": 2.0,
            "spot_bloom": 0.38, "bloom": 0.28, "glass_halation": 0.25, "mask_type": 0.0, "mask_pitch": 2.6,
            "mask_depth": 0.42, "phosphor_rgb": 0.75, "curvature": 0.26, "bezel": 0.38,
            "reshoot_enabled": 1.0, "reshoot_shutter_hz": 60.0, "reshoot_fps_offset": 0.45,
            "reshoot_rolling_shutter": 0.25, "reshoot_moire": 0.88, "reshoot_tilt_deg": 2.8,
            "reshoot_perspective": 0.28, "reshoot_ambient_glare": 0.18, "reshoot_wb_shift": 0.22,
        }},
    },
    "crt_reshoot_living_room_cam": {
        "label": "CRT 屏摄重拍 · 开灯客厅手持 DV/手机翻拍（环境光漫反 + 9300K 冷青过曝 + 慢滚带）",
        "note": "开灯房间内手持相机翻拍 9300K 家用球面彩电：未发光荧光粉层漫反射室内光抬升黑位、玻璃映出窗户高光、高光冷青过曝并伴随慢速差拍亮带",
        "layers": {"display": {
            "enabled": 1.0, "gamma": 2.35, "scanlines": 0.32, "spot_sigma": 0.42, "scan_pitch": 2.0,
            "spot_bloom": 0.52, "bloom": 0.42, "glass_halation": 0.38, "mask_type": 1.0, "mask_pitch": 3.0,
            "mask_depth": 0.35, "phosphor_rgb": 0.60, "curvature": 0.28, "bezel": 0.45,
            "reshoot_enabled": 1.0, "reshoot_shutter_hz": 50.0, "reshoot_fps_offset": 0.85,
            "reshoot_rolling_shutter": 0.45, "reshoot_moire": 0.35, "reshoot_tilt_deg": -1.6,
            "reshoot_perspective": -0.22, "reshoot_ambient_glare": 0.62, "reshoot_wb_shift": 0.52,
        }},
    },
    "tc_ub530_obs_mkv": {
        "label": "天创恒达 TC-UB530 采集卡 → 神舟 HASEE K680E-G6D1（OBS Studio 32.0 MKV）",
        "note": "天创恒达 TC-UB530 USB 复合采集卡（PAL 720×576@25fps BT.601 YUV422 ADC）直入神舟 K680E-G6D1 笔记本，由 OBS Studio 32.0 封装为 MKV 数字母带",
        "layers": {"display": {"enabled": 0.0}}},
}

CAMERA_PROFILES = {
    "ccd_3ccd_consumer": {"label": "3CCD 民用机", "note": "色度信噪比更好、无彩色摩尔纹",
                          "layers": {"camera": {"sensor_type": 1.0, "chroma_aberration": 0.02,
                                                "sensor_noise": 0.05}}},
    "ccd_single": {"label": "单 CCD（典型）", "layers": {"camera": {
        "sensor_type": 0.0, "sensor_noise": 0.06, "chroma_aberration": 0.05}}},
    "cmos_early": {"label": "早期 CMOS", "note": "固定图案噪声更重", "layers": {
        "camera": {"sensor_type": 2.0, "fixed_pattern": 0.12, "sensor_noise": 0.10}}},
    "lens_worn": {"label": "镜头磨损/发霉", "layers": {"camera": {
        "lens_wear": 0.55, "halation": 0.18, "lens_flare": 0.22}}},
    "sony_cctv_bnc": {
        "label": "Sony 工业枪机 CCD（C 口镜头 + BNC 复合输出）",
        "note": "C/CS 口大光圈镜头光轴偏心赛德尔彗差（三瓣散射扇面）+ 自动光圈叶片衍射暗槽 + CCD 防晕溢出漏极（VOD/HOD）U 型前缘黑边 + 100 IRE 白削波平台与 2D 孔径轮廓过冲",
        "layers": {"camera": {
            "sensor_type": 0.0, "sensor_noise": 0.015, "ccd_gain": 0.08,
            "sharpness": 1.18, "dynamic_range": 7.8,
            "coma": 0.75, "axis_offset": 0.50, "overflow_drain": 0.65,
            "chroma_aberration": 0.12, "longitudinal_ca": 0.08,
            "ccd_smear": 0.0, "halation": 0.0}}},
    "sony_cybershot_ccd_memorystick": {
        "label": "SONY Cyber-shot CCD 数码相机（Memory Stick 内存卡直出 · 1024×768 氙气闪光实拍）",
        "note": "早期索尼 DSC-P/S 系列 1/2.7 英寸 Super HAD CCD 静态拍照模式：机械快门遮光读出（零垂直拖尾光柱）、机顶氙气闪光灯近距直闪中心亮四角压暗（5.84x 衰减）、深暖棕木桌暗部阶调（R/G≈1.31）、铝塑药板冷银反光（B/G≈1.05）与微透镜边缘轻微蓝紫边（实拍传感器与闪光灯场分布标定）",
        "layers": {"camera": {
            "sensor_type": 0.0, "sensor_noise": 0.018, "noise_grain": 1.15, "fixed_pattern": 0.01,
            "dynamic_range": 7.6, "exposure": 0.0, "white_balance_temp": -0.04,
            "color_cast": 0.06, "highlight_tint": 0.15, "ccd_flash_falloff": 0.85,
            "sharpness": 1.18, "chroma_aberration": 0.05, "longitudinal_ca": 0.04,
            "halation": 0.06, "halation_source_colour": 0.65, "ccd_smear": 0.0, "motion_blur": 0.0}}},
    "panasonic_ccd_1986": {
        "label": "1986 松下初代单板 CCD 肩扛摄像机",
        "note": "初代 1/2 英寸补色马赛克（Ye/Cy/Mg/Gr）行间转移 CCD：6.8 档动态范围、天空与水面高光青蓝偏色（Cy+Gr 晚饱和）、暖色中间调保留与高光晕染（对照 1986 松下初代 CCD + NV-180 实拍标定）",
        "layers": {"camera": {
            "sensor_type": 0.0, "sensor_noise": 0.025, "fixed_pattern": 0.02,
            "dynamic_range": 6.8, "exposure": 0.08, "white_balance_temp": -0.06,
            "color_cast": 0.12, "highlight_tint": 0.68,
            "sharpness": 1.12, "chroma_aberration": 0.05, "longitudinal_ca": 0.04,
            "halation": 0.22, "halation_source_colour": 0.70, "ccd_smear": 0.12}}},
}


FILM_PROFILES = {
    # ==================== 基准实验一：同一场景、不同胶片与显影动力学 ====================
    "bench1_vision3_500t_std": {
        "label": "Kodak Vision3 500T 标准冲洗",
        "note": "基准实验一：Kodak Vision3 500T 5219 现代彩色电影负片，标准 ECN-2 41.1°C 显影、负片橙色遮罩正像转换与完整 Remjet 碳黑抗晕光背层",
        "layers": {"film": {
            "enabled": 1.0, "stock": 17.0, "process_mode": 0.0, "push_pull_ev": 0.0,
            "dev_temp_c": 41.1, "dev_time_norm": 1.0, "bath_activity": 1.0, "orange_mask": 1.0,
            "gamma": 1.0, "halation": 0.35, "remjet_backing": 0.88, "grain_amount": 0.22, "grain_size": 1.25,
        }},
    },
    "bench1_vision3_500t_push2": {
        "label": "Kodak Vision3 500T 推冲两档",
        "note": "基准实验一：Kodak Vision3 500T 欠曝后延长 ECN-2 显影时间推冲 +2 EV，红/绿/蓝三层反差斜率非均匀陡化并产生高光暖黄与暗部青蓝交叉偏色，颗粒显著粗化",
        "layers": {"film": {
            "enabled": 1.0, "stock": 17.0, "process_mode": 3.0, "push_pull_ev": 2.0,
            "dev_temp_c": 42.5, "dev_time_norm": 1.45, "bath_activity": 1.12, "orange_mask": 1.0,
            "gamma": 1.16, "halation": 0.42, "remjet_backing": 0.85, "grain_amount": 0.38, "grain_size": 1.65,
        }},
    },
    "bench1_vision3_500t_pull1": {
        "label": "Kodak Vision3 500T 拉冲一档",
        "note": "基准实验一：Kodak Vision3 500T 过曝后缩短 ECN-2 显影时间拉冲 -1 EV，降低底雾 D_min 与直线段反差，扩展高光肩部宽容度并细化染料云颗粒",
        "layers": {"film": {
            "enabled": 1.0, "stock": 17.0, "process_mode": 0.0, "push_pull_ev": -1.0,
            "dev_temp_c": 40.0, "dev_time_norm": 0.76, "bath_activity": 0.92, "orange_mask": 1.0,
            "gamma": 0.88, "halation": 0.28, "remjet_backing": 0.90, "grain_amount": 0.16, "grain_size": 1.15,
        }},
    },
    "bench1_double_x_5222_bw": {
        "label": "Kodak Double-X 5222 黑白负片",
        "note": "基准实验一：Kodak Eastman Double-X 5222 经典 35mm 全色黑白电影负片，D-96 20°C 显影还原金属银颗粒团簇、Eberhard 边缘微反差与深沉银盐阶调",
        "layers": {"film": {
            "enabled": 1.0, "stock": 18.0, "process_mode": 0.0, "push_pull_ev": 0.0,
            "dev_temp_c": 20.0, "dev_time_norm": 1.0, "bath_activity": 1.05,
            "gamma": 1.12, "halation": 0.36, "remjet_backing": 0.35, "grain_amount": 0.34, "grain_size": 1.48,
        }},
    },
    "bench1_kodachrome_64": {
        "label": "Kodachrome 64 反转片",
        "note": "基准实验一：历史性 Kodachrome 64 (K-14 外偶法工艺) 彩色反转片，高 D_max 光学密度、高锐度低颗粒与浓郁暖红钴蓝直接正像还原",
        "layers": {"film": {
            "enabled": 1.0, "stock": 19.0, "process_mode": 0.0,
            "dev_temp_c": 38.0, "dev_time_norm": 1.0, "bath_activity": 1.05,
            "gamma": 1.18, "halation": 0.24, "remjet_backing": 0.92, "grain_amount": 0.16, "grain_size": 1.05,
        }},
    },
    "bench1_ektachrome_100d": {
        "label": "Kodak Ektachrome 100D 反转片",
        "note": "基准实验一：现代 Kodak Ektachrome 100D (E-6 内偶法工艺) 彩色反转片，无负片橙色遮罩直接形成正像，高光洁净、青蓝还原清澈",
        "layers": {"film": {
            "enabled": 1.0, "stock": 20.0, "process_mode": 0.0,
            "dev_temp_c": 38.0, "dev_time_norm": 1.0, "bath_activity": 1.0,
            "gamma": 1.12, "halation": 0.28, "remjet_backing": 0.55, "grain_amount": 0.18, "grain_size": 1.12,
        }},
    },

    # ==================== 基准实验二：同一原片、不同复制链 ====================
    "bench2_ocn_direct_scan": {
        "label": "原始负片直扫",
        "note": "基准实验二：原始摄影负片 (OCN) 经漫射积分球光源直接 24p 数字化扫描，无中间片印片损耗，完整保留原始动态范围与最高 MTF 解析力",
        "layers": {"film": {
            "enabled": 1.0, "stock": 17.0, "process_mode": 0.0, "optical_print_gen": 0.0,
            "gamma": 1.0, "halation": 0.32, "remjet_backing": 0.88, "grain_amount": 0.20,
            "scanner_diffuse": 0.80, "telecine_pulldown": 0.0,
        }},
    },
    "bench2_optical_print_gen1": {
        "label": "一代光学印片",
        "note": "基准实验二：原始负片经光学印片机 (Optical Printer) 一代印制工作拷贝，引入中继镜头轻微 MTF 衰减、印片配光偏移与首代步进片门定位微晃动",
        "layers": {"film": {
            "enabled": 1.0, "stock": 17.0, "process_mode": 0.0, "optical_print_gen": 1.0,
            "gamma": 1.04, "halation": 0.35, "remjet_backing": 0.85, "grain_amount": 0.24,
            "gate_weave": 0.10, "telecine_pulldown": 0.0,
        }},
    },
    "bench2_optical_print_gen2": {
        "label": "二代中间片印片",
        "note": "基准实验二：原始负片 (OCN) → 翻正片 (Interpositive) → 翻底片 (Internegative) 二代光化学复制链，累积两代 H&D 反差压缩与双层颗粒团簇",
        "layers": {"film": {
            "enabled": 1.0, "stock": 17.0, "process_mode": 0.0, "optical_print_gen": 2.0,
            "gamma": 1.08, "halation": 0.38, "remjet_backing": 0.82, "grain_amount": 0.28,
            "gate_weave": 0.18, "telecine_pulldown": 0.0,
        }},
    },
    "bench2_release_print_gen3": {
        "label": "三代发行拷贝",
        "note": "基准实验二：OCN → IP → IN → 影院发行正片拷贝 (2383 Release Print) 三代完整院线印片链，搭配 24fps 双叶片快门 (48Hz) 影院放映机械微抖动",
        "layers": {"film": {
            "enabled": 1.0, "stock": 17.0, "process_mode": 0.0, "optical_print_gen": 3.0,
            "gamma": 1.12, "halation": 0.42, "remjet_backing": 0.80, "grain_amount": 0.32,
            "gate_weave": 0.24, "shutter_flicker": 0.12, "shutter_blades": 2.0, "telecine_pulldown": 0.0,
        }},
    },
    "bench2_flying_spot_telecine": {
        "label": "飞点胶转磁转录",
        "note": "基准实验二：历史电影拷贝经 Rank Cintel 飞点扫描仪执行 24fps→29.97i NTSC 3:2 Pulldown 胶转磁转录，呈现跨胶片帧隔行梳状交错与电视视放轮廓",
        "layers": {"film": {
            "enabled": 1.0, "stock": 1.0, "process_mode": 0.0, "optical_print_gen": 1.0,
            "gamma": 1.06, "halation": 0.38, "grain_amount": 0.26,
            "gate_weave": 0.22, "shutter_flicker": 0.08, "telecine_pulldown": 1.0,
        }},
    },

    # ==================== 基准实验三：同一胶片、不同保存历史与数字化修复 ====================
    "bench3_cold_vault_archive": {
        "label": "低温冷库保存",
        "note": "基准实验三：2°C / 30%RH 恒温恒湿冷库保存历史，醋酸片基游离酸度极低，聚合物完整性 >98%，三层耦合染料与齿孔尺寸几乎无退化",
        "layers": {"film": {
            "enabled": 1.0, "stock": 17.0, "process_mode": 0.0,
            "storage_temp_c": 2.0, "storage_rh": 30.0, "vinegar_syndrome": 0.02,
            "dye_fade": 0.02, "shrinkage": 0.01, "emulsion_damage": 0.0, "scratches_dust": 0.02,
            "scanner_diffuse": 0.75, "wet_gate": 0.0, "restoration_strength": 0.0,
        }},
    },
    "bench3_ambient_natural_aging": {
        "label": "常温库房自然老化",
        "note": "基准实验三：22°C / 55%RH 常温库房保存数十年，青/黄染料中度水解褪色向暖红偏移，片基轻度收缩并伴随走片纵向划痕与齿孔磨损",
        "layers": {"film": {
            "enabled": 1.0, "stock": 4.0, "process_mode": 0.0,
            "storage_temp_c": 22.0, "storage_rh": 55.0, "vinegar_syndrome": 0.35,
            "dye_fade": 0.38, "shrinkage": 0.24, "emulsion_damage": 0.18,
            "scratches_dust": 0.28, "sprocket_wear": 0.22, "gate_weave": 0.20,
            "scanner_diffuse": 0.35, "wet_gate": 0.0, "restoration_strength": 0.0,
        }},
    },
    "bench3_humid_vinegar_syndrome": {
        "label": "湿热库房醋酸综合征",
        "note": "基准实验三：32°C / 78%RH 湿热环境触发醋酸纤维素自催化脱乙酰降解，游离酸度突破自催化点导致片基剧烈收缩屈曲、乳剂起皱与边缘银镜、背层抗晕光染料重新显色",
        "layers": {"film": {
            "enabled": 1.0, "stock": 4.0, "process_mode": 0.0,
            "storage_temp_c": 32.0, "storage_rh": 78.0, "vinegar_syndrome": 0.82,
            "dye_fade": 0.68, "shrinkage": 0.62, "emulsion_damage": 0.55, "aged_backing_dye": 0.58,
            "scratches_dust": 0.48, "sprocket_wear": 0.45, "gate_weave": 0.35,
            "scanner_diffuse": 0.20, "wet_gate": 0.0, "restoration_strength": 0.0,
        }},
    },
    "bench3_wet_gate_restoration": {
        "label": "湿门扫描与数字修复",
        "note": "基准实验三：对重度老化胶片启用折射率匹配液湿门扫描 (Wet-Gate) + 漫射积分球光源消除片基划痕散射，并执行 Disney Research / FIAF 可逆数字修复与四层数字证据分离记录",
        "layers": {"film": {
            "enabled": 1.0, "stock": 4.0, "process_mode": 0.0,
            "storage_temp_c": 32.0, "storage_rh": 78.0, "vinegar_syndrome": 0.82,
            "dye_fade": 0.68, "shrinkage": 0.62, "emulsion_damage": 0.55, "aged_backing_dye": 0.58,
            "scratches_dust": 0.48, "sprocket_wear": 0.45, "gate_weave": 0.35,
            "scanner_diffuse": 0.92, "wet_gate": 0.95, "restoration_strength": 0.88,
        }},
    },

    # ==================== 独立胶片光学与经典暗房预设（纯净命名 · 单屏输出） ====================
    "optics_remjet_removed_halation": {
        "label": "去碳层高光红晕",
        "note": "独立胶片光学模块：去除电影负片背面 Remjet 碳黑抗晕光层 (`remjet_backing=0`)，强光穿透乳剂在片基界面全反射回感红层，在点光源周围形成典型双层红橙晕光环",
        "layers": {"film": {
            "enabled": 1.0, "stock": 17.0, "process_mode": 0.0,
            "remjet_backing": 0.0, "halation": 0.88, "bloom_veil_mode": 3.0, "bloom_veil_amount": 0.68,
            "grain_amount": 0.24, "lomo_vignette": 0.25,
        }},
    },
    "lomo_xpro_e6_in_c41_marigold": {
        "label": "正冲负翡翠暖金",
        "note": "Kodak Ektachrome E100VS 反转片投入 C-41 负片药水交叉冲洗：高反差 S 曲线、翡翠青绿阴影与金盏花暖金高光",
        "layers": {"film": {
            "enabled": 1.0, "stock": 6.0, "process_mode": 1.0, "xpro_strength": 0.92,
            "xpro_tint_mode": 1.0, "gamma": 1.18, "halation": 0.36, "grain_amount": 0.24,
            "lomo_vignette": 0.32, "lomo_edge_smear": 0.25,
        }},
    },
    "lomo_xpro_c41_in_e6_heart_bokeh": {
        "label": "负冲正桃心散景",
        "note": "Lomography CN 400 负片投入 E-6 药水：丁香紫/洋红暗部与奶油暖白高光，配合桃心异形光圈散景与漏光轨",
        "layers": {"film": {
            "enabled": 1.0, "stock": 7.0, "process_mode": 2.0, "xpro_strength": 0.85,
            "xpro_tint_mode": 2.0, "gamma": 1.10, "halation": 0.42, "grain_amount": 0.26,
            "lomo_vignette": 0.42, "lomo_edge_smear": 0.30, "bokeh_shape": 1.0, "bokeh_amount": 0.65,
            "light_leak_amount": 0.48, "light_leak_pattern": 3.0,
        }},
    },
    "lomo_agfa_warm_amber_holga": {
        "label": "琥珀暖调弯月镜",
        "note": "Agfa Vista 200 高感红乳剂琥珀暖金中间调与深可可暗部，叠加 Holga 单片弯月镜柔焦与暖雾漏光",
        "layers": {"film": {
            "enabled": 1.0, "stock": 5.0, "process_mode": 0.0, "xpro_strength": 0.45,
            "xpro_tint_mode": 3.0, "gamma": 1.08, "halation": 0.45, "grain_amount": 0.25,
            "lomo_vignette": 0.38, "lomo_edge_smear": 0.35, "light_leak_amount": 0.32, "light_leak_pattern": 4.0,
        }},
    },
    "lomo_expired_svema_sunflower": {
        "label": "过期胶卷高底雾",
        "note": "存放十余年的过期胶卷：顶层卤化银热雾化将死黑基底抬升至粉彩鼠尾草绿，高光软肩压缩并伴随粗颗粒团簇",
        "layers": {"film": {
            "enabled": 1.0, "stock": 8.0, "process_mode": 0.0, "expired_fog": 0.78,
            "gamma": 0.86, "halation": 0.32, "grain_amount": 0.42, "grain_size": 1.85,
            "lomo_vignette": 0.24, "light_leak_amount": 0.22, "light_leak_pattern": 4.0,
        }},
    },
    "lomo_polaroid_sx70_qlight": {
        "label": "宝丽来即时显影",
        "note": "Polaroid SX-70 一体式相纸碱液药浆扩散与 TiO2 反射层奶油高光滚降、复古青棕阴影及右上暖阳漏光",
        "layers": {"film": {
            "enabled": 1.0, "stock": 9.0, "process_mode": 0.0, "xpro_strength": 0.35,
            "xpro_tint_mode": 3.0, "gamma": 1.12, "halation": 0.38, "grain_amount": 0.20,
            "lomo_vignette": 0.68, "lomo_edge_smear": 0.32, "light_leak_amount": 0.55, "light_leak_pattern": 1.0,
        }},
    },
    "lomo_redscale_xr_reverse": {
        "label": "反装片基红金显影",
        "note": "将彩色负片片基朝向镜头反卷曝光：光线先穿透底层感红层再经防光晕层滤波到达感绿层，呈现深红-琥珀-金黄阶梯",
        "layers": {"film": {
            "enabled": 1.0, "stock": 7.0, "process_mode": 0.0, "redscale_flip": 0.90,
            "gamma": 1.15, "halation": 0.55, "grain_amount": 0.34, "lomo_vignette": 0.45,
        }},
    },
    "eastman_5247_35mm_telecine": {
        "label": "Eastman 5247 胶转磁",
        "note": "经典 35mm 电影负片+2383印片特性曲线、防光晕红橙背散射、片门机械二维抖动与 NTSC 3:2 下拉",
        "layers": {"film": {
            "enabled": 1.0, "stock": 1.0, "process_mode": 0.0, "gamma": 1.05,
            "halation": 0.38, "grain_amount": 0.25, "gate_weave": 0.25, "telecine_pulldown": 1.0,
        }},
    },
    "lomo_trioplan_soap_bubble_dandelion": {
        "label": "肥皂泡球差散景",
        "note": "三片式 Meyer Trioplan 过校正球差形成锐利亮边肥皂泡圆环散景，配合斜向柠檬白-丁香紫-暖金分色光雾",
        "layers": {"film": {
            "enabled": 1.0, "stock": 16.0, "process_mode": 0.0, "gamma": 1.02,
            "halation": 0.45, "grain_amount": 0.16, "bokeh_shape": 5.0, "bokeh_amount": 0.88,
            "bloom_veil_mode": 4.0, "bloom_veil_amount": 0.85, "soft_focus_mist": 0.55,
            "dual_tone_split": 0.82, "lomo_vignette": 0.22,
        }},
    },
    "lomo_golden_hour_solar_flare_flood": {
        "label": "黄金逆光阳炎",
        "note": "顶部正逆光太阳形成纯明黄过曝核心，向全画幅倾泻指数衰减金琥珀色米氏散射阳炎，前景维持深可可暖褐暗部",
        "layers": {"film": {
            "enabled": 1.0, "stock": 5.0, "process_mode": 0.0, "xpro_strength": 0.55,
            "xpro_tint_mode": 3.0, "gamma": 1.16, "halation": 0.68, "grain_amount": 0.20,
            "bloom_veil_mode": 1.0, "bloom_veil_amount": 0.92, "lomo_vignette": 0.48,
        }},
    },
    "lomo_highkey_milky_veiling_glare": {
        "label": "无镀膜高调柔焦",
        "note": "强逆光进入无镀膜单镜片引发全画幅米氏散射乳白光雾抬升黑位，配合柔焦水汽与富士 Pro 400H 粉彩乳剂",
        "layers": {"film": {
            "enabled": 1.0, "stock": 13.0, "process_mode": 0.0, "expired_fog": 0.45,
            "gamma": 0.84, "halation": 0.48, "grain_amount": 0.14,
            "bloom_veil_mode": 2.0, "bloom_veil_amount": 0.88, "soft_focus_mist": 0.72,
        }},
    },
    "lomo_tilt_shift_candle_flame_aura": {
        "label": "移轴双层烛火光冕",
        "note": "点光源双层烛火日冕与防光晕红环 + 沙姆定律移轴焦平面浅景深 + 冷青板岩暗部",
        "layers": {"film": {
            "enabled": 1.0, "stock": 2.0, "process_mode": 0.0, "gamma": 1.12,
            "halation": 0.62, "grain_amount": 0.22, "bloom_veil_mode": 3.0, "bloom_veil_amount": 0.90,
            "tilt_shift_dof": 0.82, "focus_plane_y": 0.68, "lomo_vignette": 0.45,
        }},
    },
    "lomo_wong_kar_wai_tungsten_olive": {
        "label": "钨丝灯迫冲绿调",
        "note": "日光型电影负片在室内荧光灯/钨丝灯下未加色温滤镜直接迫冲：酸橄榄黄绿偏色、墨绿暗部与柔化高光",
        "layers": {"film": {
            "enabled": 1.0, "stock": 11.0, "process_mode": 3.0, "push_pull_ev": 1.2,
            "gamma": 1.15, "halation": 0.42, "grain_amount": 0.30, "grain_size": 1.65,
            "lomo_vignette": 0.36, "lomo_edge_smear": 0.24, "soft_focus_mist": 0.28,
        }},
    },
    "lomo_prussian_rain_paper_boats": {
        "label": "普鲁士蓝雨滴散景",
        "note": "冷调普鲁士深蓝阴影与高饱和品红/电光紫分离，配合移轴浅景深与焦外六边形雨滴散景光斑",
        "layers": {"film": {
            "enabled": 1.0, "stock": 12.0, "process_mode": 1.0, "xpro_strength": 0.65,
            "gamma": 1.18, "halation": 0.38, "grain_amount": 0.22,
            "bokeh_shape": 6.0, "bokeh_amount": 0.72, "tilt_shift_dof": 0.75, "focus_plane_y": 0.62,
            "lomo_vignette": 0.52,
        }},
    },
    "lomo_macaron_fairy_lights_pastel": {
        "label": "六边形彩灯散景",
        "note": "富士 Pro 400H 粉彩乳剂叠加 6 片机械光圈形成的大尺寸暖金六边形彩灯散景与柔焦白雾",
        "layers": {"film": {
            "enabled": 1.0, "stock": 13.0, "process_mode": 0.0, "gamma": 0.92,
            "halation": 0.46, "grain_amount": 0.15, "bokeh_shape": 6.0, "bokeh_amount": 0.88,
            "bloom_veil_mode": 4.0, "bloom_veil_amount": 0.55, "soft_focus_mist": 0.38,
        }},
    },
    "lomo_petzval_swirl_street_lantern": {
        "label": "匹兹伐切向旋焦",
        "note": "中心保持锐利合焦，四周背景高光光斑沿切向圆弧旋转拉长形成猫眼旋焦，配以复古黄铜橄榄与紫褐暗部",
        "layers": {"film": {
            "enabled": 1.0, "stock": 14.0, "process_mode": 0.0, "gamma": 1.08,
            "halation": 0.40, "grain_amount": 0.24, "bokeh_shape": 4.0, "bokeh_amount": 0.75,
            "swirl_amount": 0.88, "lomo_vignette": 0.46, "lomo_edge_smear": 0.35,
        }},
    },
    "lomo_lca_turquoise_bridge_xpro": {
        "label": "广角弯月镜正冲负",
        "note": "LOMO LC-A Minitar-1 32mm 广角弯月镜浓重四角暗角、边缘场曲抹开与松石青绿高反差交叉冲洗",
        "layers": {"film": {
            "enabled": 1.0, "stock": 15.0, "process_mode": 1.0, "xpro_strength": 0.88,
            "xpro_tint_mode": 1.0, "gamma": 1.24, "halation": 0.44, "grain_amount": 0.28,
            "lomo_vignette": 0.78, "lomo_edge_smear": 0.48,
        }},
    },
}


# ---------------------------------------------------------------------------
# 古早数字媒介与网络流媒体预设库 (LDME: 手机摄影、优酷土豆、文件传输与早期显示)
# ---------------------------------------------------------------------------
DIGITAL_PROFILES = {
    "nokia_6600": {
        "label": "诺基亚 6600 (2003 VGA手机摄影)",
        "note": "2003 年 Nokia 6600 VGA 拍照管线，低像素空间采样限制、小塑料镜头边缘模糊、AWB 微漂移与有损 JPEG 4:2:0 压缩",
        "layers": {
            "camera": {"ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 1.0, "ldme_capture_mode": 0.0, "noise_grain": 0.0, "sensor_noise": 0.08},
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0, "luma_bandwidth": 3.8, "chroma_bandwidth": 1.2},
            "codec": {"stream_enabled": 0.0, "dv_enabled": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 1.0, "ldme_lcd_bits": 5.0, "ldme_lcd_response_ms": 28.0, "ldme_lcd_view_angle": 16.0, "ldme_lcd_subpixel": 0.50},
            "audio": {"enabled": 1.0, "mode": 9.0, "mic_type": 4.0, "proximity_effect": 0.35, "mic_overdrive": 0.25},
        },
    },
    "nokia_n73": {
        "label": "诺基亚 N73 (2006 卡尔蔡司拍照手机)",
        "note": "2006 年 Nokia N73 320 万像素卡尔蔡司镜头，ISP 降噪细节涂抹、USM 锐化边缘光晕与 JPEG 4:2:0 块效应",
        "layers": {
            "camera": {"ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 2.0, "ldme_capture_mode": 0.0, "noise_grain": 0.0, "sensor_noise": 0.05},
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0, "luma_bandwidth": 4.8, "chroma_bandwidth": 1.8},
            "codec": {"stream_enabled": 0.0, "dv_enabled": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 1.0, "ldme_lcd_bits": 6.0, "ldme_lcd_response_ms": 18.0, "ldme_lcd_view_angle": 12.0, "ldme_lcd_subpixel": 0.35},
            "audio": {"enabled": 1.0, "mode": 9.0, "mic_type": 1.0, "proximity_effect": 0.20},
        },
    },
    "nokia_n95": {
        "label": "诺基亚 N95 (2007 早期智能旗舰)",
        "note": "2007 年 Nokia N95 500 万像素卡尔蔡司管线，蔡司色彩矩阵、自动曝光微延迟与边缘过冲",
        "layers": {
            "camera": {"ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 3.0, "ldme_capture_mode": 0.0, "noise_grain": 0.0, "sensor_noise": 0.04},
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0, "luma_bandwidth": 5.2, "chroma_bandwidth": 2.0},
            "codec": {"stream_enabled": 0.0, "dv_enabled": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 1.0, "ldme_lcd_bits": 6.0, "ldme_lcd_response_ms": 16.0, "ldme_lcd_view_angle": 10.0, "ldme_lcd_subpixel": 0.30},
            "audio": {"enabled": 1.0, "mode": 9.0, "mic_type": 1.0, "proximity_effect": 0.15},
        },
    },
    "nokia_5800": {
        "label": "诺基亚 5800 XpressMusic (2008 3GP视频录制)",
        "note": "2008 年 Nokia 5800 3GP 动态视频录制管线 (record)，640×352 H.263 视频编码与卷帘快门",
        "layers": {
            "camera": {"ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 4.0, "ldme_capture_mode": 1.0, "noise_grain": 0.0, "sensor_noise": 0.06},
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0, "luma_bandwidth": 4.2, "chroma_bandwidth": 1.5},
            "codec": {"stream_enabled": 0.0, "dv_enabled": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 1.0, "ldme_lcd_bits": 6.0, "ldme_lcd_response_ms": 18.0, "ldme_lcd_view_angle": 12.0, "ldme_lcd_subpixel": 0.35},
            "audio": {"enabled": 1.0, "mode": 9.0, "mic_type": 4.0, "proximity_effect": 0.30},
        },
    },
    "iphone_2007": {
        "label": "初代 iPhone (2007 200万定焦摄影)",
        "note": "2007 年初代 iPhone 200 万像素定焦拍照管线 (capture)，有限动态范围高光硬削波与自然偏暖肤色",
        "layers": {
            "camera": {"ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 5.0, "ldme_capture_mode": 0.0, "noise_grain": 0.0, "sensor_noise": 0.05},
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0, "luma_bandwidth": 4.5, "chroma_bandwidth": 1.6},
            "codec": {"stream_enabled": 0.0, "dv_enabled": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 1.0, "ldme_lcd_bits": 6.0, "ldme_lcd_response_ms": 16.0, "ldme_lcd_view_angle": 10.0, "ldme_lcd_subpixel": 0.30},
            "audio": {"enabled": 1.0, "mode": 9.0, "mic_type": 1.0, "proximity_effect": 0.20},
        },
    },
    "iphone_3g": {
        "label": "iPhone 3G (2008 移动网络传播)",
        "note": "2008 年 iPhone 3G 200 万像素拍照管线，早期小型 CMOS 暗部信噪比与定焦镜头物理特性",
        "layers": {
            "camera": {"ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 6.0, "ldme_capture_mode": 0.0, "noise_grain": 0.0, "sensor_noise": 0.05},
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0, "luma_bandwidth": 4.5, "chroma_bandwidth": 1.6},
            "codec": {"stream_enabled": 0.0, "dv_enabled": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 1.0, "ldme_lcd_bits": 6.0, "ldme_lcd_response_ms": 16.0, "ldme_lcd_view_angle": 10.0, "ldme_lcd_subpixel": 0.30},
            "audio": {"enabled": 1.0, "mode": 9.0, "mic_type": 1.0, "proximity_effect": 0.20},
        },
    },
    "iphone_3gs": {
        "label": "iPhone 3GS (2009 VGA视频录制)",
        "note": "2009 年 iPhone 3GS 独立 VGA 视频录制管线 (record)，640×480 @ 30fps H.264 编码与连续曝光阶跃",
        "layers": {
            "camera": {"ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 7.0, "ldme_capture_mode": 1.0, "noise_grain": 0.0, "sensor_noise": 0.04},
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0, "luma_bandwidth": 5.0, "chroma_bandwidth": 2.0},
            "codec": {"stream_enabled": 0.0, "dv_enabled": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 1.0, "ldme_lcd_bits": 6.0, "ldme_lcd_response_ms": 14.0, "ldme_lcd_view_angle": 8.0, "ldme_lcd_subpixel": 0.25},
            "audio": {"enabled": 1.0, "mode": 9.0, "mic_type": 1.0, "proximity_effect": 0.15},
        },
    },
    "iphone_4s": {
        "label": "iPhone 4S (2011 1080p高清视频)",
        "note": "2011 年 iPhone 4S 800 万像素背照式 BSI CMOS + A5 ISP 管线，边缘解析力与高光宽容度大幅提升",
        "layers": {
            "camera": {"ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 8.0, "ldme_capture_mode": 0.0, "noise_grain": 0.0, "sensor_noise": 0.02},
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0, "luma_bandwidth": 6.0, "chroma_bandwidth": 2.5},
            "codec": {"stream_enabled": 0.0, "dv_enabled": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
            "display": {"enabled": 0.0, "ldme_display_enabled": 0.0},
            "audio": {"enabled": 1.0, "mode": 9.0, "mic_type": 2.0, "proximity_effect": 0.10},
        },
    },
    "youku_2007_flv": {
        "label": "早期优酷土豆 Flash 流畅档 (320×240 FLV)",
        "note": "2006-2009 年中文视频网站经典转码档位，320×240 分辨率、Sorenson Spark (FLV1) 编码、350 kbps 码率与低码率音频",
        "layers": {
            "camera": {"noise_grain": 0.0, "sensor_noise": 0.04},
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0, "luma_bandwidth": 4.0, "chroma_bandwidth": 1.4},
            "codec": {"stream_enabled": 1.0, "stream_mode": 6.0, "stream_quad_corner": 0.0, "stream_show_border": 0.0, "stream_show_text": 0.0, "stream_bitrate_kbps": 350.0, "stream_fps": 25.0, "stream_gop_len": 15.0, "ldme_source_origin": 0.0, "dv_enabled": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 2.0, "ldme_flash_scale_mode": 1.0, "ldme_lcd_bits": 6.0, "ldme_lcd_subpixel": 0.35},
            "audio": {"enabled": 1.0, "mode": 9.0, "mic_type": 4.0, "proximity_effect": 0.25, "mic_overdrive": 0.30},
        },
    },
    "youku_2010_sd": {
        "label": "优酷标清 H.264 视频档 (480×360)",
        "note": "2009-2012 年 Web2.0 平台标清转码档，480×360 分辨率、H.264 Baseline 编码、700 kbps 码率",
        "layers": {
            "camera": {"noise_grain": 0.0, "sensor_noise": 0.03},
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0, "luma_bandwidth": 5.0, "chroma_bandwidth": 1.8},
            "codec": {"stream_enabled": 1.0, "stream_mode": 7.0, "stream_quad_corner": 0.0, "stream_show_border": 0.0, "stream_show_text": 0.0, "stream_bitrate_kbps": 700.0, "stream_fps": 25.0, "stream_gop_len": 25.0, "ldme_source_origin": 0.0, "dv_enabled": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 2.0, "ldme_flash_scale_mode": 1.0, "ldme_lcd_bits": 6.0, "ldme_lcd_subpixel": 0.25},
            "audio": {"enabled": 1.0, "mode": 9.0, "mic_type": 1.0, "proximity_effect": 0.15},
        },
    },
    "youku_2008_webcam_anchor": {
        "label": "优酷 2008 网络主播视听参考款 (稍息立正站好同款)",
        "note": "实测校准：352×264 @ 15fps、73.5kbps 极低码率宏块、廉价 USB 摄像头驻极体麦克风近讲、唱腔高音振膜与前级过载削顶、爆破音破音与 32kbps 心理声学水下金属杂音",
        "layers": {
            "camera": {"ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 1.0, "ldme_capture_mode": 1.0, "noise_grain": 0.0, "sensor_noise": 0.09, "ae_enabled": 1.0, "ae_tau_s": 2.2},
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0, "luma_bandwidth": 3.6, "chroma_bandwidth": 1.2},
            "codec": {"stream_enabled": 1.0, "stream_mode": 6.0, "stream_quad_corner": 0.0, "stream_show_border": 0.0, "stream_show_text": 0.0, "stream_bitrate_kbps": 80.0, "stream_fps": 15.0, "stream_gop_len": 12.0, "ldme_source_origin": 1.0, "dv_enabled": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 2.0, "ldme_flash_scale_mode": 1.0, "ldme_lcd_bits": 6.0, "ldme_lcd_subpixel": 0.40},
            "audio": {"enabled": 1.0, "mode": 9.0, "mic_type": 4.0, "proximity_effect": 0.65, "plosive_pop": 0.55, "mic_overdrive": 0.70, "broadcast_agc": 0.45, "studio_room_reverb": 0.35},
        },
    },
    "mms_compress": {
        "label": "手机彩信与极度压缩 (MMS 160×120 30KB)",
        "note": "早期 2G/2.5G 移动网络彩信体积硬约束 (<30KB)，160×120 极限压缩、粗糙色度块与重度量化",
        "layers": {
            "camera": {"ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 2.0, "ldme_capture_mode": 0.0, "noise_grain": 0.0, "sensor_noise": 0.06},
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0, "luma_bandwidth": 3.2, "chroma_bandwidth": 1.0},
            "codec": {"stream_enabled": 0.0, "ldme_transfer_enabled": 1.0, "ldme_transfer_method": 3.0, "ldme_mms_width": 320.0, "ldme_mms_quality": 45.0, "dv_enabled": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 1.0, "ldme_lcd_bits": 5.0, "ldme_lcd_subpixel": 0.45},
            "audio": {"enabled": 1.0, "mode": 9.0, "mic_type": 5.0, "proximity_effect": 0.30},
        },
    },
    "phone_tft_lcd": {
        "label": "早期手机 TFT 液晶显示 (6-bit 偏角偏色)",
        "note": "2000年代彩色功能机 6-bit TN/STN 液晶屏，视线偏离垂直法线时反差翻转、Gamma 阶跃与分子拖尾",
        "layers": {
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 1.0, "ldme_lcd_bits": 5.0, "ldme_lcd_response_ms": 28.0, "ldme_lcd_view_angle": 16.0, "ldme_lcd_subpixel": 0.50, "ldme_flash_scale_mode": 0.0},
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
        },
    },
    "desktop_tn_flash": {
        "label": "早期桌面 TN 液晶显示器 (Flash 播放器缩放)",
        "note": "2000年代初期桌面 17/19 英寸 TN 液晶显示器，6-bit+FRC 抖动灰阶、冷阴极荧光灯 (CCFL) 背光偏蓝与 Flash 全屏缩放模糊",
        "layers": {
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 2.0, "ldme_lcd_bits": 6.0, "ldme_lcd_response_ms": 16.0, "ldme_lcd_view_angle": 12.0, "ldme_lcd_subpixel": 0.35, "ldme_lcd_ccfl": 0.28, "ldme_flash_scale_mode": 1.0},
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
        },
    },
    "digicam_early_phone_flicker_tungsten": {
        "label": "旧手机荧光灯频闪与钨丝灯白平衡失效",
        "note": "物理建模旧手机实拍图片（image.jpg、image (1).jpg 与 b0ae5b91...jpg）：CMOS 卷帘快门 50Hz 工频荧光灯水平青绿/橙黄交替条纹 + 餐厅低色温钨丝灯/混光 AWB 蓝增益硬钳位芥末青黄偏色",
        "layers": {
            "camera": {
                "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 2.0, "ldme_capture_mode": 0.0,
                "ldme_flicker_enabled": 1.0, "ldme_flicker_mains_hz": 50.0, "ldme_flicker_depth": 0.45, "ldme_flicker_color_drift": 0.65, "ldme_flicker_num_bands": 5.2,
                "ldme_tungsten_enabled": 1.0, "ldme_color_temp_k": 2700.0, "ldme_awb_mode": 2.0,
                "sensor_noise": 0.04,
            },
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0},
            "codec": {"stream_enabled": 0.0, "dv_enabled": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 1.0, "ldme_lcd_bits": 6.0, "ldme_lcd_subpixel": 0.35},
        },
    },
    "youku_tudou_multi_gen_jpeg_wear": {
        "label": "早期优酷土豆多代 JPEG 磨损衰减",
        "note": "2006–2009 优酷土豆视频截图与论坛搬运多代 JPEG 有损磨损：真实 2D 8×8 DCT 量化、4:2:0 色度衰减与跨代 8×8 网格错位双重蚊式振铃噪点",
        "layers": {
            "camera": {
                "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 1.0, "ldme_capture_mode": 0.0,
                "ldme_jpeg_decay_enabled": 1.0, "ldme_jpeg_decay_gens": 4.0, "ldme_jpeg_decay_quality": 45.0, "ldme_jpeg_decay_grid_shift": 3.0,
            },
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0},
            "codec": {"stream_enabled": 0.0, "ldme_transfer_enabled": 1.0, "ldme_transfer_method": 5.0, "ldme_transfer_generations": 4.0, "ldme_jpeg_decay_quality": 45.0, "ldme_jpeg_decay_grid_shift": 3.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 2.0, "ldme_lcd_bits": 6.0, "ldme_lcd_subpixel": 0.35},
        },
    },
    "digicam_night_starburst_ccd": {
        "label": "早期数码相机夜景车灯星芒与粗糙宏块",
        "note": "物理建模旧相机夜景实拍（image_d61d18.png）：微型镜头夜景车灯星芒溢出、点光源垂直光柱、1/4 英寸 CCD 满井饱和硬削波与暗部深空粗糙 DCT 块效应",
        "layers": {
            "camera": {
                "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 1.0, "ldme_capture_mode": 0.0,
                "ldme_night_bloom": 1.25,
                "ldme_jpeg_decay_enabled": 1.0, "ldme_jpeg_decay_gens": 2.0, "ldme_jpeg_decay_quality": 35.0,
                "sensor_noise": 0.05,
            },
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0},
            "codec": {"stream_enabled": 0.0, "dv_enabled": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 2.0, "ldme_lcd_bits": 6.0, "ldme_lcd_subpixel": 0.35},
        },
    },
    "digicam_y2k_ccd_sensor": {
        "label": "千禧年 Y2K 卡片机 CCD 饱和与动态范围",
        "note": "参考 digicamfx.com 经典小底数码卡片机：CCD 高饱和暖调肤色、小尺寸传感器动态范围硬截断与低像素微细节",
        "layers": {
            "camera": {
                "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 3.0, "ldme_capture_mode": 0.0,
                "ldme_night_bloom": 0.40, "ldme_jpeg_quality": 68.0, "sensor_noise": 0.035,
            },
            "format": {"filter_kind": 1.0, "chroma_gain": 1.0, "chroma_vertical_averaging": 0.0},
            "codec": {"stream_enabled": 0.0, "dv_enabled": 0.0},
            "tape": {"tape_stability": 1.0, "dropout_rate": 0.0, "head_switching": 0.0, "noise_floor": 0.0},
            "display": {"enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 2.0, "ldme_lcd_bits": 6.0, "ldme_lcd_subpixel": 0.30},
        },
    },
}

AUDIO_PROFILES = {
    "audio_01_webcam_electret_agc_clip": {
        "label": "01 · 电脑摄像头驻极体话筒 + 前端 AGC 抽吸与运放硬削顶",
        "note": "还原千禧年 PC 摄像头/耳麦近讲拾音：切除 215Hz 以下低频、3.8kHz 塑料腔体尖峰、安静间隙 AGC 自动拉起风扇底噪与高声压运放轨硬削波",
        "layers": {
            "audio": {
                "enabled": 1.0, "mic_type": 4.0, "mic_polar_pattern": 1.0, "proximity_effect": 0.55,
                "plosive_pop": 0.48, "mic_agc_amount": 0.72, "mic_clip_mode": 1.0, "mic_overdrive": 0.62,
                "mic_self_noise": 0.28,
            },
        },
    },
    "audio_02_nokia_handsfree_micro_speaker": {
        "label": "02 · 旧功能机免提微型扬声器外放 (Nokia 6600/N73 破音腔体)",
        "note": "13×18mm 聚酯振膜微扬声器：f0=780Hz 高通切除低音、2.45kHz/3.85kHz 盆分裂与塑料背腔共鸣、大音量折环非线性破音与防烧 ALC 限幅",
        "layers": {
            "audio": {
                "enabled": 1.0, "mode": 13.0, "mobile_spk_mode": 2.0, "mobile_spk_f0_hz": 780.0,
                "mobile_spk_drive": 0.68, "mobile_spk_cavity_res": 0.75, "mobile_spk_limiter": 0.60,
            },
        },
    },
    "audio_03_8bit_11khz_zoh_sampler": {
        "label": "03 · 早期 8-bit / 11.025kHz 零阶保持数字采样与反相梳状下混",
        "note": "无抗混叠零阶保持阶梯重采样 (ZOH) 叠加 8-bit 无抖动截断量化与转接头反相声道抵消",
        "layers": {
            "audio": {
                "enabled": 1.0, "adc_enabled": 1.0, "adc_sample_rate_hz": 11025.0, "adc_bit_depth": 8.0,
                "adc_resample_filter": 2.0, "adc_dither_mode": 0.0, "adc_clock_jitter": 0.25,
            },
        },
    },
    "audio_04_mp3_32kbps_multigen_birdies": {
        "label": "04 · 32kbps MP3 心理声学编码水声 (Birdies) 与预回声 (3代重压)",
        "note": "Bark 临界频带比特饥饿导致未掩蔽子带成片归零（频带空洞闪烁水声）与打击乐长窗 MDCT 预回声",
        "layers": {
            "audio": {
                "enabled": 1.0, "mode": 11.0, "pcodec_type": 1.0, "pcodec_bitrate_kbps": 32.0,
                "pcodec_pre_echo": 0.72, "pcodec_birdies": 0.78, "pcodec_generations": 3.0,
            },
        },
    },
    "audio_05_gsm_voip_packet_loss_plc": {
        "label": "05 · 移动蜂窝/早期网络语音 (VAD 吞字 + CNG 舒适噪声 + 丢包 PLC 电音)",
        "note": "300-3400Hz 窄带语音、DTX 静音抑制切掉尾音并填入合成舒适噪声、谱减法音乐噪声及突发丢包基音重复机械声",
        "layers": {
            "audio": {
                "enabled": 1.0, "mode": 12.0, "telecom_mode": 3.0, "telecom_wideband": 0.0,
                "telecom_vad_threshold": 0.38, "telecom_cng_level": 0.30, "telecom_packet_loss_pct": 16.0,
                "telecom_plc_mode": 1.0, "telecom_aec_residue": 0.35, "telecom_ns_strength": 0.55,
                "telecom_jitter_ms": 38.0,
            },
        },
    },
    "audio_06_tiled_corridor_flutter_echo": {
        "label": "06 · 瓷砖走廊与楼梯间声学空间 (平行墙颤动回声 + 轴向驻波 + 口袋遮挡)",
        "note": "12.5×1.7×3.0m 狭长硬质瓷砖走廊：9.9ms 周期性平行墙颤动回声、1.35s 长混响、低频简正模态与远距离拾音",
        "layers": {
            "audio": {
                "enabled": 1.0, "space_preset": 2.0, "space_rt60_s": 1.35, "space_wall_absorption": 0.12,
                "space_standing_waves": 0.68, "space_mic_distance_m": 2.2, "space_ambient_noise": 0.25,
            },
        },
    },
    "audio_07_winxp_kmixer_flash_underrun": {
        "label": "07 · WinXP KMixer + 浏览器 Flash 播放器音频链 (缓冲欠载卡碟 + 3.5mm 虚焊)",
        "note": "模拟早期电脑看视频：Flash 22.05kHz 重采样、CPU 卡顿引发 32ms 环形缓冲重复颤音、Mega Bass 音箱共鸣与混音器硬削波",
        "layers": {
            "audio": {
                "enabled": 1.0, "mode": 15.0, "sys_audio_preset": 2.0, "sys_src_mode": 2.0,
                "sys_eq_preset": 3.0, "sys_spatial_3d": 0.45, "sys_buffer_underrun": 0.35, "sys_mixer_clip": 0.35,
            },
        },
    },
    "audio_08_nokia_rtttl_and_4chord_fm": {
        "label": "08 · 诺基亚单音蜂鸣器 (RTTTL) 与 Yamaha 4和弦 FM 手机音源",
        "note": "严格 1 声部压电陶瓷方波蜂鸣器或 4 声部 FM 合成器：硬件声部抢占截断 (Voice Stealing)、25ms Tick 量化与 16 阶包络步进",
        "layers": {
            "audio": {
                "enabled": 1.0, "mode": 14.0, "ringtone_mode": 1.0, "ringtone_polyphony": 1.0,
                "ringtone_tick_quant_ms": 28.0, "ringtone_env_steps": 12.0, "ringtone_mix": 1.0,
            },
        },
    },
    "audio_09_broadcast_loudness_war_optimod": {
        "label": "09 · 电视广告『响度战争』多段压限 (-9.5 LUFS) vs 平台 -14 LUFS 归一化",
        "note": "ITU-R BS.1770 K-加权响度测量、三频段高密度广播多段压缩与 -0.3 dBTP 真峰值砖墙限幅",
        "layers": {
            "audio": {
                "enabled": 1.0, "loudness_mode": 2.0, "loudness_target_lufs": -9.5,
                "loudness_compressor": 0.82, "loudness_true_peak_dbtp": -0.3,
            },
        },
    },
    "audio_10_av_clock_drift_and_splice_sync": {
        "label": "10 · 音视频时钟漂移 (ppm)、掉帧 FIFO 追赶与磁带切点瞬态同步",
        "note": "模拟声卡与采集卡未锁相引起的线性累积唇音失步、视频掉帧时伴音 FIFO 重复切片以及非过零硬切口咔哒声",
        "layers": {
            "audio": {
                "enabled": 1.0, "av_sync_enabled": 1.0, "av_sync_offset_ms": 85.0,
                "av_clock_drift_ppm": 650.0, "av_frame_drop_rate": 0.18, "av_drop_sync_mode": 1.0,
                "av_edit_splice_mode": 1.0, "av_edit_interval_s": 1.2,
            },
        },
    },
}

PROFILE_LIBRARY = {
    "digital": DIGITAL_PROFILES,
    "film": FILM_PROFILES,
    "media": MEDIA_PROFILES,
    "player": PLAYER_PROFILES,
    "display": DISPLAY_PROFILES,
    "camera": CAMERA_PROFILES,
    "audio": AUDIO_PROFILES,
}

# NOTE on `rf.broadcast_noise` absolute level:
#   Three calibration attempts against the 1992 off-air reference all compared the
#   model against something it is not (a bright pattern against a near-black frame,
#   twice, then a dark pattern with different content).  The reference is also the
#   *sum* of the RF link, a VHS recording and a capture, and its original scene
#   detail is unknown, so its absolute noise level is underdetermined by that
#   material.  The mechanism's direction and character are verified (see
#   tools/verify.py); its absolute level is a modelling choice, not a measurement.
# The reference camcorder this project was largely calibrated against.
SONY_DCR_TRV310E = {"label": "Sony DCR-TRV310E", "layers": {
    "camera": {"sensor_type": 0.0, "sensor_noise": 0.09, "ccd_gain": 0.68,
               "sharpness": 0.82, "chroma_aberration": 0.04, "dynamic_range": 7.5,
               "color_cast": 0.45, "white_balance_temp": 0.0,
               "ccd_smear": 0.58, "halation": 0.44, "halation_source_colour": 0.65,
               "light_streak_amount": 0.38, "overflow_drain": 0.24},
    "format": {"luma_bandwidth": 5.0, "chroma_bandwidth": 2.0,
               "chroma_vertical_averaging": 0.0, "color_noise": 0.04, "edge_boost": 0.18},
    "codec": {"dv_enabled": 1.0, "dv_quality": 0.90, "dv_base_step": 0.015},
    "tape": {"dropout_rate": 0.0, "noise_floor": 0.02, "line_jitter": 0.02,
             "walk_amplitude": 0.02, "micro_jitter": 0.02, "head_switching": 0.0,
             "shuttle_mode": 2.0},
    "player": {"tbc_enabled": 1.0, "head_wear": 0.05, "transport_jitter": 0.05,
               "luma_bandwidth": 5.0, "chroma_bandwidth": 2.0},
    "display": {"enabled": 0.0},
}}

CAMCORDER_PRESETS = {
    "sony_dcr_trv310e": SONY_DCR_TRV310E,
    "panasonic_nv_gs78": {"label": "Panasonic NV-GS78", "layers": {
        "camera": {"sensor_type": 0.0, "sensor_noise": 0.08, "sharpness": 0.80,
                   "dynamic_range": 7.0},
        "format": {"luma_bandwidth": 4.0, "chroma_bandwidth": 0.7},
        "codec": {"dv_enabled": 1.0},
        "display": {"enabled": 0.0}}},
    "panasonic_m9000": {"label": "Panasonic M9000（肩扛 VHS）", "layers": PLAYER_PROFILES[
        "deck_panasonic_m9000"]["layers"] | {"display": {"enabled": 0.0}}},
    "panasonic_1986_nv180": {
        "label": "1986 松下初代 CCD 肩扛机 + NV-180 背包录像机",
        "layers": {
            **CAMERA_PROFILES["panasonic_ccd_1986"]["layers"],
            "format": {"filter_kind": 2.0, "filter_sections": 2.0,
                       "luma_bandwidth": 2.8, "chroma_bandwidth": 0.55,
                       "chroma_vertical_averaging": 0.85, "comb_depth": 0.72,
                       "cross_color": 0.18, "cross_luma": 0.12, "color_noise": 0.10,
                       "edge_boost": 0.35, "chroma_gain": 1.05},
            **PLAYER_PROFILES["deck_panasonic_nv180"]["layers"],
            "display": {"enabled": 0.0},
        }},
}

FACTORY_PRESETS = {
    "sony_ccd_memorystick_still": {
        "label": "SONY CCD 数码相机 · 内存卡提取原片直出 (Cyber-shot 1024×768 氙气闪光灯 + 铝塑高光与暗木暖阶)",
        "note": "专为 SONY Cyber-shot CCD 相机 Memory Stick 内存卡提取的 1024×768 JPG 原片校准：保留拍照机械快门（无录像模式的垂直拖尾涂抹）、机顶氙气闪光灯近距中心聚光与四角 5.84x 衰减、暗部木纹暖红棕阶调（R/G=1.308）、铝塑药板冷银高光（B/G=1.048）与 JPEG 8×8 精细微反差",
        "layers": {
            "camera": {
                "sensor_type": 0.0, "sensor_noise": 0.018, "noise_grain": 1.15, "fixed_pattern": 0.01,
                "dynamic_range": 7.6, "exposure": 0.0, "white_balance_temp": -0.04,
                "color_cast": 0.06, "highlight_tint": 0.15, "ccd_flash_falloff": 0.85,
                "sharpness": 1.18, "chroma_aberration": 0.05, "longitudinal_ca": 0.04,
                "halation": 0.06, "halation_source_colour": 0.65, "ccd_smear": 0.0, "motion_blur": 0.0,
            },
            "format": {
                "filter_kind": 0.0, "luma_bandwidth": 5.8, "chroma_bandwidth": 1.9,
                "chroma_vertical_averaging": 0.0, "comb_depth": 0.98,
                "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.01,
                "edge_boost": 0.16, "chroma_gain": 1.02,
            },
            "codec": {"dv_enabled": 0.0, "codec_kind": 0.0},
            "tape": {
                "tape_speed": 0.0, "tape_stability": 1.0, "wear": 0.0,
                "dropout_rate": 0.0, "noise_floor": 0.0, "line_jitter": 0.0,
                "walk_amplitude": 0.0, "micro_jitter": 0.0, "vertical_instability": 0.0,
                "head_switching": 0.0, "tracking_error": 0.0, "crease_rate": 0.0,
            },
            "player": {
                "tbc_enabled": 1.0, "tbc_strength": 1.0, "tbc_residual": 0.0,
                "transport_jitter": 0.0, "head_wear": 0.0, "frequency_response": 0.0,
                "luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "comb_depth": 0.98,
                "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0,
            },
            "display": {"enabled": 0.0},
        }},
    "family_archive_1992": {
        "label": "家庭录像（1992 录像机 → VHS 存档）",
        "note": "摄像机 + 格式 + 介质 + 播放 + 显示 五层齐备",
        "layers": {
            "camera": {"sensor_noise": 0.08, "noise_grain": 0.0, "dynamic_range": 8.0,
                       "sharpness": 0.80, "motion_blur": 0.18, "sensor_type": 0.0},
            "format": {"luma_bandwidth": 3.0, "chroma_bandwidth": 0.7,
                       "chroma_vertical_averaging": 0.9, "comb_depth": 0.45,
                       "cross_color": 0.50, "cross_luma": 0.35, "color_noise": 0.30,
                       "edge_boost": 0.25, "chroma_gain": 1.0},
            "tape": {"tape_speed": 0.0, "tape_stability": 0.55, "wear": 0.25,
                     "dropout_rate": 0.35, "noise_floor": 0.18, "head_switching": 0.65,
                     "head_switch_lines": 6.0, "tracking_error": 0.22},
            "player": {"transport_jitter": 0.45, "tbc_enabled": 0.0, "head_wear": 0.30,
                       "frequency_response": 0.15},
            "display": {"enabled": 0.0, "gamma": 2.35, "scanlines": 0.35, "bloom": 0.35, "overscan": 0.0,
                        "vignette": 0.25, "interlace_flicker": 0.25, "geometry_warp": 0.15},
        }},
    "rental_tape_worn": {
        "label": "旧出租带（重度磨损 + 无 TBC）",
        "note": "多次走带、划伤、失落密集，且播放机无时基校正",
        "layers": {
            "camera": {"sensor_noise": 0.09, "sharpness": 0.82, "motion_blur": 0.20},
            "format": {"luma_bandwidth": 2.8, "chroma_bandwidth": 0.6,
                       "cross_color": 0.60, "cross_luma": 0.42, "color_noise": 0.38},
            "tape": {"tape_speed": 1.0, "tape_stability": 0.35, "wear": 0.65,
                     "crease_rate": 0.45, "dropout_rate": 1.4, "noise_floor": 0.34,
                     "surface_damage": 0.35, "chroma_loss": 0.18,
                     "head_switching": 0.95, "tracking_error": 0.6},
            "player": {"transport_jitter": 0.7, "tbc_enabled": 0.0, "head_wear": 0.55,
                       "frequency_response": 0.30},
            "display": {"enabled": 0.0, "gamma": 2.35, "scanlines": 0.4, "bloom": 0.32, "overscan": 0.0,
                        "vignette": 0.30, "interlace_flicker": 0.30},
        }},
    "pro_transfer_svhs": {
        "label": "专业转录（S-VHS + TBC）",
        "note": "编辑机 + 时基校正，尽量还原母带",
        "layers": {
            "format": {"luma_bandwidth": 5.0, "chroma_bandwidth": 0.85,
                       "cross_color": 0.28, "cross_luma": 0.18, "color_noise": 0.16},
            "tape": {"tape_speed": 0.0, "tape_stability": 0.8, "wear": 0.12,
                     "dropout_rate": 0.15, "noise_floor": 0.10, "head_switching": 0.5},
            "player": {"transport_jitter": 0.15, "tbc_enabled": 1.0, "tbc_strength": 0.9,
                       "head_wear": 0.10, "frequency_response": 0.05},
            "display": {"enabled": 0.0},
        }},
    "mouldy_archive": {
        "label": "发霉存档带",
        "note": "潮湿存放：黏合剂吸水、磁粉层脱落。色度先死、亮度塌陷、高频颗粒（潮湿霉变脱粉磁带实测标定）",
        "layers": {
            "camera": {"ccd_smear": 0.0, "halation": 0.05, "fixed_pattern": 0.0},
            "format": {"chroma_gain": 0.04, "color_noise": 0.01, "cross_color": 0.02},
            "tape": {"binder_loss": 0.35, "shed_grain": 0.32, "luma_pedestal": -0.06,
                     "dropout_rate": 1.8, "noise_floor": 0.25, "dropout_color": 0.02},
            "player": {"cross_color": 0.02, "color_noise": 0.01, "head_wear": 0.10},
            "display": {"enabled": 0.0},
        }},
    "svhs_disc_rot": {
        "label": "S-VHS 消隐钳位失常劣化带转录",
        "note": "磁迹严重劣化：直流恢复/消隐钳位失常导致亮度基底塌陷而色度过饱和保留（实测 sat 0.92 / mean 0.20），伴随整行色相错乱与跟踪撕裂（S-VHS 劣化磁带直流恢复失常实测标定）",
        "layers": {
            "camera": {"ccd_smear": 0.0, "halation": 0.10, "sensor_noise": 0.04},
            "format": {"luma_bandwidth": 4.6, "chroma_gain": 1.65, "color_noise": 0.32},
            "tape": {"tape_stability": 0.4, "wear": 0.65, "luma_pedestal": -0.22,
                     "dropout_rate": 1.6, "noise_floor": 0.28, "tracking_error": 0.6,
                     "row_chroma_error": 0.22, "row_chroma_phase": 38.0,
                     "row_chroma_gain": 0.25},
            "player": {"transport_jitter": 0.55, "head_wear": 0.45},
            "display": {"enabled": 0.0},
        }},
    "broadcast_1992_weak": {
        "label": "1992 电视台弱信号录像",
        "note": "广播链路信噪比不足：宽带噪声把亮度与色度一起淹没（与磁带劣化的"
                "「色度先死」相反）。绝对噪声电平是建模选择而非实测，见 profiles.py 中的说明",
        "layers": {
            "rf": {"signal_dbm": -74.0, "adjacent_channel": 0.20,
                   "broadcast_noise": 0.45},
            "tape": {"noise_floor": 0.30, "dropout_rate": 0.6},
            "display": {"enabled": 0.0},
        }},
    "vcd_352x240": {
        "label": "Video CD（MPEG-1 352x240）",
        "note": "VCD 编码：352x240 四分之一分辨率重采样、4:2:0 色度、粗量化、运动补偿预测。"
                "伪影跟着**预测成败**走，不是逐帧独立的噪声——"
                "这是它区别于 DV 与模拟通路的地方（早期 MPEG-1 VCD 实测标定）。",
        "layers": {
            "camera": {"sensor_noise": 0.02, "sharpness": 1.0, "ccd_smear": 0.0,
                       "halation": 0.05, "chroma_aberration": 0.0, "fixed_pattern": 0.0},
            "format": {"luma_bandwidth": 3.5, "chroma_bandwidth": 0.8,
                       "chroma_vertical_averaging": 0.35, "cross_color": 0.15,
                       "cross_luma": 0.10, "color_noise": 0.08},
            "codec": {"codec_kind": 2.0, "vcd_quality": 0.85,
                      "vcd_width": 352.0, "vcd_height": 240.0,
                      "vcd_motion_comp": 1.0, "vcd_bitrate_kbps": 1150.0},
            "tape": {"tape_speed": 0.0, "tape_stability": 0.90, "wear": 0.05,
                     "dropout_rate": 0.0, "noise_floor": 0.04,
                     "head_switching": 0.0},
            "player": {"tbc_enabled": 1.0, "transport_jitter": 0.08,
                       "head_wear": 0.08},
            "display": {"enabled": 0.0},
        }},
    "vcd_crt_offscreen": {
        "label": "VCD 碟机 → 家用球面 CRT 电视屏摄",
        "note": "MPEG-1 352x240 4:2:0 运动补偿块伪影 + 家用球面 CRT 显像管（管面曲率、轻微梯形失真、高光泛光与扫描线，早期 MPEG-1 VCD 屏摄实测标定）",
        "layers": {
            "camera": {"sensor_noise": 0.02, "sharpness": 1.0, "ccd_smear": 0.0,
                       "halation": 0.08, "chroma_aberration": 0.02, "fixed_pattern": 0.0},
            "format": {"luma_bandwidth": 3.5, "chroma_bandwidth": 0.8,
                       "chroma_vertical_averaging": 0.35, "cross_color": 0.15,
                       "cross_luma": 0.10, "color_noise": 0.08},
            "codec": {"codec_kind": 2.0, "vcd_quality": 0.85,
                      "vcd_width": 352.0, "vcd_height": 240.0,
                      "vcd_motion_comp": 1.0, "vcd_bitrate_kbps": 1150.0},
            "tape": {"tape_speed": 0.0, "tape_stability": 0.92, "wear": 0.0,
                     "dropout_rate": 0.0, "noise_floor": 0.02, "head_switching": 0.0},
            "player": {"tbc_enabled": 1.0, "transport_jitter": 0.05, "head_wear": 0.05},
            "display": {"enabled": 0.0, "gamma": 1.05, "contrast": 0.52, "brightness": 0.48,
                        "saturation": 0.60, "scanlines": 0.04, "spot_sigma": 0.95,
                        "bloom": 0.22, "curvature": 0.18, "trapezoid_x": 0.03,
                        "vignette": 0.16, "interlace_flicker": 0.02, "reflection": 0.03},
        }},
    "zibo_1992_rf_weak": {
        "label": "1992 地方电视台弱信号录像（RF 雪花 + 行失锁 + VHS 录放）",
        "note": "开路电视射频载噪比骤降：中频宽带雪花与 AGC 黑电平抬升将亮度与色度一并淹没（sat 降至 ~0.23，hf 升至 ~0.58），伴随行 AFC 失锁撕裂与 VHS 底部磁头切换带（开路电视弱信号 VHS 录放实测标定）",
        "layers": {
            "camera": {"sensor_noise": 0.02, "ccd_smear": 0.0, "halation": 0.08,
                       "chroma_aberration": 0.02, "fixed_pattern": 0.0, "sharpness": 0.92},
            "format": {"luma_bandwidth": 2.8, "chroma_bandwidth": 0.65,
                       "chroma_gain": 0.48, "chroma_vertical_averaging": 0.65,
                       "comb_depth": 0.35, "cross_color": 0.28, "cross_luma": 0.18,
                       "color_noise": 0.52},
            "rf": {"signal_dbm": -74.0, "adjacent_channel": 0.22,
                   "broadcast_noise": 0.45, "sync_loss_threshold_db": 32.0,
                   "sync_loss_amount": 1.0},
            "tape": {"tape_speed": 0.0, "tape_stability": 0.48, "wear": 0.32,
                     "shed_grain": 0.06, "luma_pedestal": 0.06, "row_timebase": 0.32,
                     "line_jitter": 0.55, "dropout_rate": 0.85, "noise_floor": 0.26,
                     "row_chroma_error": 0.24, "row_chroma_phase": 42.0,
                     "row_chroma_gain": 0.25, "head_switching": 0.75,
                     "head_switch_lines": 12.0, "tracking_error": 0.40},
            "player": {"transport_jitter": 0.50, "tbc_enabled": 0.0, "head_wear": 0.35,
                       "cross_color": 0.25, "color_noise": 0.45},
            "display": {"enabled": 0.0},
        }},
    "panasonic_1986_ccd_nv180": {
        "label": "1986 松下初代 CCD 肩扛摄像机 + NV-180 背包录像机（实拍标定）",
        "note": "1986 年初代补色马赛克 CCD 肩扛机经多芯脐带线接入松下 NV-180 便携式 VHS 背包机：6.8 档动态范围与天空高光青蓝偏色（R/G≈0.60, B/G≈1.03）、2.8 MHz 亮度孔径预加重白边过冲、550 kHz 欠载色度群延迟右移拖尾（~5 px）、底部 7 行磁头切换水平跳变与右侧 31-49 px HBI 黑边阶跃及底行半行消隐截断",
        "layers": {
            "camera": {
                "sensor_type": 0.0, "sensor_noise": 0.025, "fixed_pattern": 0.02,
                "dynamic_range": 6.8, "exposure": 0.08, "white_balance_temp": -0.06,
                "color_cast": 0.12, "highlight_tint": 0.68,
                "sharpness": 1.12, "chroma_aberration": 0.05, "longitudinal_ca": 0.04,
                "halation": 0.22, "halation_source_colour": 0.70, "ccd_smear": 0.12,
            },
            "format": {
                "filter_kind": 2.0, "filter_sections": 2.0,
                "luma_bandwidth": 2.8, "chroma_bandwidth": 0.55,
                "chroma_vertical_averaging": 0.85, "comb_depth": 0.72,
                "cross_color": 0.18, "cross_luma": 0.12, "color_noise": 0.10,
                "edge_boost": 0.35, "chroma_gain": 1.05,
            },
            "tape": {
                "tape_speed": 0.0, "tape_stability": 0.88, "line_jitter": 0.04,
                "walk_amplitude": 0.04, "micro_jitter": 0.03, "row_timebase": 0.02,
                "head_switching": 0.90, "head_switch_lines": 4.0, "tracking_error": 0.52,
                "dropout_rate": 0.08, "noise_floor": 0.06, "wear": 0.08,
            },
            "player": {
                "tbc_enabled": 1.0, "tbc_strength": 0.88, "tbc_residual": 0.04,
                "transport_jitter": 0.05, "head_wear": 0.10,
                "preemphasis": 0.32, "ringing": 0.18, "ringing_freq": 0.48,
                "luma_bandwidth": 2.8, "chroma_bandwidth": 0.55, "comb_depth": 0.72,
                "cross_color": 0.18, "cross_luma": 0.12, "color_noise": 0.10,
            },
            "display": {"enabled": 0.0},
        }},
    "beijing_youth_vhs_1990s": {
        "label": "1990 年代 VHS 原版出版带片头（PAL SP 欠载色度与高光金晕）",
        "note": "PAL VHS SP 原版出版带：3.0 MHz 亮度与 629 kHz 欠载色度横向洇色、高光金色光晕、偶发整行错色与底部磁头切换带（1990 年代 PAL VHS SP 出版带实测标定）",
        "layers": {
            "camera": {"sensor_noise": 0.03, "ccd_smear": 0.0, "halation": 0.22,
                       "halation_source_colour": 0.85, "chroma_aberration": 0.03,
                       "fixed_pattern": 0.0, "sharpness": 0.88},
            "format": {"luma_bandwidth": 3.0, "chroma_bandwidth": 0.68,
                       "chroma_vertical_averaging": 0.85, "comb_depth": 0.50,
                       "cross_color": 0.35, "cross_luma": 0.22, "color_noise": 0.18,
                       "edge_boost": 0.22, "chroma_gain": 1.02},
            "tape": {"tape_speed": 0.0, "tape_stability": 0.68, "wear": 0.18,
                     "dropout_rate": 0.25, "noise_floor": 0.14, "line_jitter": 0.22,
                     "row_chroma_error": 0.06, "row_chroma_phase": 28.0,
                     "head_switching": 0.75, "head_switch_lines": 12.0,
                     "tracking_error": 0.32},
            "player": {"transport_jitter": 0.28, "tbc_enabled": 0.0, "head_wear": 0.20,
                       "frequency_response": 0.10},
            "display": {"enabled": 0.0},
        }},
    "curving_stripes": {
        "label": "卷曲条纹（差频干涉）",
        "note": "沿画面**弯曲**的条纹，由残余载波与视频通路的差频产生。"
                "与「整行时基误差」是不同现象：位移只把内容横向搬走，"
                "行内曲率保持在地板（0.00001）；这个让行内曲率升到 240 倍。"
                "条纹随读取过程淡入淡出与迁移，不是贴在画面上的固定纹理。",
        "layers": {
            "tape": {"stripe_moire": 0.55, "stripe_beat": 0.05,
                     "stripe_curve": 0.5, "row_timebase": 0.35,
                     "tape_stability": 0.5, "wear": 0.4,
                     "dropout_rate": 0.4, "noise_floor": 0.20},
            "player": {"transport_jitter": 0.5, "tbc_enabled": 0.0,
                       "head_wear": 0.4},
            "display": {"enabled": 0.0},
        }},
    "worn_transport_row_timebase": {
        "label": "走带时基故障（整行位移）",
        "note": "沿画面平滑变化的逐行水平位移：竖直边缘会**弯曲**而不是碎裂，"
                "而且位移形态逐帧迁移。与「行抖动」区分——后者在同一根轴上振荡、"
                "形态每帧重复。亮度与色度同移，不产生凭空偏色。",
        "layers": {
            "tape": {"row_timebase": 0.12, "line_jitter": 0.25,
                     "tape_stability": 0.5, "wear": 0.35,
                     "dropout_rate": 0.4, "noise_floor": 0.20},
            "player": {"transport_jitter": 0.5, "tbc_enabled": 0.0},
            "display": {"enabled": 0.0},
        }},
    "worn_deck_row_chroma": {
        "label": "老化色度解码（整行错色）",
        "note": "整条扫描线解出错误的颜色：色相被旋转、色度增益偏移，"
                "线仍然是饱和的。与「色度丢失行」不同——那个得到灰线，这个得到错色。"
                "误差随读取过程演化，不是逐帧重掷。",
        "layers": {
            "tape": {"row_chroma_error": 0.16, "row_chroma_phase": 42.0,
                     "row_chroma_gain": 0.22, "chroma_loss": 0.03,
                     "tape_stability": 0.45, "wear": 0.45,
                     "dropout_rate": 0.5, "noise_floor": 0.22},
            "format": {"chroma_gain": 1.0},
            "player": {"transport_jitter": 0.55, "tbc_enabled": 0.0,
                       "head_wear": 0.45},
            "display": {"enabled": 0.0},
        }},
    "digital8_trv310e": {
        "label": "Sony DCR-TRV310E（Digital8 自动白平衡实拍）",
        "note": "1/4 英寸 互补色 CFA CCD + 非均匀信号相关噪波（纯黑天光消隐钳位 + 暗部/光晕裙边散粒AGC与8x8块噪波 + 高光核满阱平顶）+ 多瓣扇形高光溢出与上下非对称伴随暗槽垂直拖尾（低照度实拍标定）",
        "layers": {
            "camera": {"sensor_type": 0.0, "sensor_noise": 0.09, "ccd_gain": 0.68,
                       "sharpness": 0.82, "chroma_aberration": 0.04, "dynamic_range": 7.5,
                       "color_cast": 0.45, "white_balance_temp": 0.0,
                       "ccd_smear": 0.58, "halation": 0.44, "halation_source_colour": 0.65,
                       "light_streak_amount": 0.38, "overflow_drain": 0.24},
            "format": {"luma_bandwidth": 5.0, "chroma_bandwidth": 2.0,
                       "chroma_vertical_averaging": 0.0, "color_noise": 0.04, "edge_boost": 0.18},
            "codec": {"dv_enabled": 1.0, "dv_quality": 0.90, "dv_base_step": 0.015},
            "tape": {"dropout_rate": 0.0, "noise_floor": 0.03, "line_jitter": 0.03,
                     "walk_amplitude": 0.02, "micro_jitter": 0.03, "head_switching": 0.0,
                     "shuttle_mode": 2.0},
            "player": {"tbc_enabled": 1.0, "head_wear": 0.05, "transport_jitter": 0.05,
                       "luma_bandwidth": 5.0, "chroma_bandwidth": 2.0},
            "display": {"enabled": 0.0},
        }},
    "digital8_trv310e_night_street": {
        "label": "Sony DCR-TRV310E（夜间路灯/走廊实拍标定：非均匀噪波 + 多瓣高光溢出 + 伴随暗槽垂直拖尾）",
        "note": "SONY DCR-TRV310E 夜间实拍标定：纯黑天光（L<0.03）经光学黑钳位与DV死区抑制近乎无噪，中暗部路面/树冠与路灯高光裙边（L∈[0.12,0.85]）呈现强烈横向CDS条纹、4:1:1互补色CFA红绿/蓝黄色斑与8×8宏块颗粒，高光过曝核（L>0.90）满阱平顶；路灯周围呈6瓣不对称扇形星芒裙边与走廊小灯竖菱形水滴光斑，隔行CCD垂直溢出向下呈细锐青白针状亮线并在两侧伴随VOD/HOD抽吸暗槽（深度-0.15~-0.28），向上呈宽软快衰减羽流",
        "layers": {
            "camera": {"sensor_type": 0.0, "sensor_noise": 0.10, "ccd_gain": 0.72,
                       "sharpness": 0.84, "chroma_aberration": 0.05, "dynamic_range": 7.4,
                       "color_cast": 0.42, "white_balance_temp": -0.05,
                       "ccd_smear": 0.68, "halation": 0.48, "halation_source_colour": 0.70,
                       "light_streak_amount": 0.46, "light_streak_length": 0.72,
                       "overflow_drain": 0.28, "overflow_width": 2.0},
            "format": {"luma_bandwidth": 5.0, "chroma_bandwidth": 2.0,
                       "chroma_vertical_averaging": 0.0, "color_noise": 0.04, "edge_boost": 0.18},
            "codec": {"dv_enabled": 1.0, "dv_quality": 0.88, "dv_base_step": 0.016},
            "tape": {"dropout_rate": 0.0, "noise_floor": 0.02, "line_jitter": 0.02,
                     "walk_amplitude": 0.02, "micro_jitter": 0.02, "head_switching": 0.0,
                     "shuttle_mode": 2.0},
            "player": {"tbc_enabled": 1.0, "head_wear": 0.04, "transport_jitter": 0.04,
                       "luma_bandwidth": 5.0, "chroma_bandwidth": 2.0},
            "display": {"enabled": 0.0},
        }},
    "hk_vhs_commercial_early": {
        "label": "PAL VHS 底部磁头切换带精细模型（双极性换向尖峰 + H-AFC 隔行回拉 + APC 黄绿相移）",
        "note": "早年 PAL 25fps VHS 广播录带实测标定：画面底部约4~6行（切换带）完整保留广告底图与色彩，在奇偶场各自的磁头切换水平坐标 x_sw 处产生双极性黑白FM限幅尖峰短横簇（宽10~32px）与行内水平跳变折口，横向错位在切换接缝行（bot-2）达峰值（均值7.9px、p90=18px）并随 H-AFC 锁相环向底边指数回拉且带奇偶场梳状错位齿，同时 629kHz 欠载色度经 1H 延迟线产生标志性黄绿/橄榄色（dRGB≈(-0.043,-0.005,-0.055)）APC 相位阶跃偏色",
        "standard": "pal",
        "layers": {
            "camera": {"sensor_noise": 0.025, "sharpness": 0.90, "dynamic_range": 8.0,
                       "halation": 0.18, "halation_source_colour": 0.75, "ccd_smear": 0.0},
            "format": {"luma_bandwidth": 3.0, "chroma_bandwidth": 0.65,
                       "chroma_vertical_averaging": 0.82, "separation_kind": 2.0,
                       "comb_depth": 0.62, "cross_color": 0.28, "cross_luma": 0.18,
                       "color_noise": 0.16, "edge_boost": 0.24, "chroma_gain": 1.02},
            "tape": {"tape_speed": 0.0, "tape_stability": 0.72, "wear": 0.16,
                     "dropout_rate": 0.18, "noise_floor": 0.12, "line_jitter": 0.16,
                     "walk_amplitude": 0.12, "micro_jitter": 0.08,
                     "head_switching": 0.82, "head_switch_lines": 3.0,
                     "tracking_error": 0.36, "head_switch_mid_line": 0.35},
            "player": {"tbc_enabled": 0.0, "transport_jitter": 0.20, "head_wear": 0.16,
                       "preemphasis": 0.22, "ringing": 0.14, "ringing_freq": 0.50,
                       "luma_bandwidth": 3.0, "chroma_bandwidth": 0.65,
                       "separation_kind": 2.0, "comb_depth": 0.62},
            "display": {"enabled": 0.0},
        }},
    "digital8_trv310e_indoor": {
        "label": "Sony DCR-TRV310E（室内 3200K 锁定 · 深蓝实拍）",
        "note": "锁定室内钨丝灯白平衡 ([F✋] white_balance_temp = -0.50) 下的深钴蓝冷调 + 低照度 AGC 噪波",
        "layers": {
            "camera": {"sensor_type": 0.0, "sensor_noise": 0.08, "ccd_gain": 0.65,
                       "sharpness": 0.82, "chroma_aberration": 0.04, "dynamic_range": 7.5,
                       "color_cast": 0.45, "white_balance_temp": -0.50,
                       "ccd_smear": 0.25, "halation": 0.25},
            "format": {"luma_bandwidth": 5.0, "chroma_bandwidth": 2.0,
                       "chroma_vertical_averaging": 0.0, "color_noise": 0.04, "edge_boost": 0.18},
            "codec": {"dv_enabled": 1.0, "dv_quality": 0.90, "dv_base_step": 0.015},
            "tape": {"dropout_rate": 0.0, "noise_floor": 0.03, "line_jitter": 0.03,
                     "walk_amplitude": 0.02, "micro_jitter": 0.03, "head_switching": 0.0,
                     "shuttle_mode": 2.0},
            "player": {"tbc_enabled": 1.0, "head_wear": 0.05, "transport_jitter": 0.05,
                       "luma_bandwidth": 5.0, "chroma_bandwidth": 2.0},
            "display": {"enabled": 0.0},
        }},
    "cosmag_e180_sharp_cascade_258m": {
        "label": "COSMAG E-180(PAL/SP/LP) → 夏普A62DT(4磁头) → H96(一级稳定) → M9E(二级稳定) → TC-UB530 → K680E(OBS 32.0 MKV · 258分钟成功段)",
        "note": "COSMAG E-180（180分SP/360分LP混合，成功采录 258 分钟）经夏普 VC-A62DT（4视频+1音频+1时间轴磁头）读取，串接夏普 VC-H96（一级画面稳定）与夏普 VC-M9E（二级画面稳定）两级同步再生锁相，再经天创恒达 TC-UB530 采集卡在神舟 HASEE K680E-G6D1 上由 OBS Studio 32.0 录为 MKV：画面行同步极稳，保留三机级联 2.9 MHz 柔和带宽、1.7 px 级联色度右移、SP/LP 混合包络及底部 6 行被锁定的磁头切换带",
        "layers": {
            "camera": {
                "sensor_noise": 0.02, "noise_grain": 1.2, "fixed_pattern": 0.0,
                "sharpness": 0.96, "dynamic_range": 8.5, "halation": 0.12,
                "halation_source_colour": 0.75, "ccd_smear": 0.0,
            },
            "format": {
                "filter_kind": 1.0, "filter_sections": 3.0,
                "luma_bandwidth": 2.95, "chroma_bandwidth": 0.62,
                "chroma_vertical_averaging": 0.82, "separation_kind": 2.0,
                "comb_depth": 0.72, "cross_color": 0.22, "cross_luma": 0.14,
                "edge_boost": 0.22, "color_noise": 0.16, "chroma_gain": 0.95,
            },
            "connector": {
                "av_present": 1.0, "av_moisture": 0.04, "av_corrosion": 0.02,
                "av_ground_loop": 0.03, "mains_hz": 50.0,
            },
            "tape": {
                "tape_speed": 0.0, "sp_lp_mix": 0.42, "tape_stability": 0.68,
                "line_jitter": 0.22, "walk_amplitude": 0.18, "micro_jitter": 0.08,
                "row_timebase": 0.08, "wear": 0.24, "dropout_rate": 0.38,
                "noise_floor": 0.16, "crease_rate": 0.14,
                "demagnetization": 0.08, "base_cracking": 0.04, "binder_loss": 0.03,
                "head_switching": 0.78, "head_switch_lines": 6.0, "tracking_error": 0.22,
            },
            "player": {
                "tbc_enabled": 1.0, "tbc_strength": 0.92, "tbc_residual": 0.03,
                "stabilizer_stages": 2.0, "transport_jitter": 0.06, "head_wear": 0.12,
                "preemphasis": 0.20, "ringing": 0.14, "ringing_freq": 0.50,
                "luma_bandwidth": 2.9, "chroma_bandwidth": 0.60,
                "separation_kind": 2.0, "comb_depth": 0.78,
                "cross_color": 0.18, "cross_luma": 0.12, "color_noise": 0.12,
            },
            "display": {"enabled": 0.0},
        }},
    "cosmag_e180_sharp_cascade_61m_fail": {
        "label": "COSMAG E-180(PAL/SP/LP) → 夏普A62DT → H96 → M9E → TC-UB530 → OBS 32.0 MKV（61分钟采录失败段：消磁+带基龟裂+发霉+带基破损）",
        "note": "还原 61 分钟采录失败段落的四种复合介质绝症：1.消磁（剩磁衰减致 FM 载波跌入雪花、ACC 压死色度）；2.带基龟裂（PET 微裂纹横跨螺旋轨引发斜向黑白限幅尖峰与行错位）；3.发霉（黏合剂吸水脱粉形成固定暗斑与宽带粗颗粒）；4.带基破损（边缘撕裂导致 1×时间轴磁头丢失 25Hz CTL 脉冲，突破 H96+M9E 两级稳定极限产生局部横带撕裂与原生雪花）",
        "layers": {
            "camera": {
                "sensor_noise": 0.03, "noise_grain": 1.4, "fixed_pattern": 0.0,
                "sharpness": 0.88, "dynamic_range": 8.0, "halation": 0.10, "ccd_smear": 0.0,
            },
            "format": {
                "filter_kind": 1.0, "filter_sections": 3.0,
                "luma_bandwidth": 2.35, "chroma_bandwidth": 0.48,
                "chroma_vertical_averaging": 0.88, "separation_kind": 2.0,
                "comb_depth": 0.65, "cross_color": 0.28, "cross_luma": 0.20,
                "edge_boost": 0.15, "color_noise": 0.38, "chroma_gain": 0.40,
            },
            "connector": {
                "av_present": 1.0, "av_moisture": 0.08, "av_corrosion": 0.05,
                "av_ground_loop": 0.06, "mains_hz": 50.0,
            },
            "tape": {
                "tape_speed": 1.0, "sp_lp_mix": 0.75, "tape_stability": 0.28,
                "line_jitter": 0.65, "walk_amplitude": 0.55, "micro_jitter": 0.25,
                "row_timebase": 0.45, "wear": 0.68, "dropout_rate": 2.10,
                "noise_floor": 0.38, "crease_rate": 0.65,
                "binder_loss": 0.34, "shed_grain": 0.38, "luma_pedestal": -0.06,
                "demagnetization": 0.52, "base_cracking": 0.62, "base_breakage": 0.58,
                "stiction_rate": 0.85, "stiction_amount": 0.65,
                "row_chroma_error": 0.26, "row_chroma_phase": 40.0,
                "head_switching": 0.95, "head_switch_lines": 10.0, "tracking_error": 0.62,
            },
            "player": {
                "tbc_enabled": 1.0, "tbc_strength": 0.78, "tbc_residual": 0.14,
                "stabilizer_stages": 2.0, "transport_jitter": 0.22, "head_wear": 0.32,
                "preemphasis": 0.18, "ringing": 0.16, "ringing_freq": 0.48,
                "luma_bandwidth": 2.4, "chroma_bandwidth": 0.50,
                "separation_kind": 2.0, "comb_depth": 0.65,
                "cross_color": 0.25, "cross_luma": 0.18, "color_noise": 0.35,
            },
            "display": {"enabled": 0.0},
        }},
    "panasonic_j27_channel_switch_horror": {
        "label": "松下 NV-J27 换台瞬态与磁带覆录拼接 (PAL-D VHF 换台失锁 + VBI 滚屏 + 1H DOC)",
        "note": "针对老式电视高频头换台瞬态与磁带覆录接缝物理建模的整套预设：松下 NV-J27 (PAL-D, +6.5MHz 伴音) 高频头变容二极管换台扫频、"
                "SAW 奈奎斯特斜坡正交浮雕与包络负片反相、中频 AGC 满增益雪花爆发与直流恢复闪白黑屏、50Hz 场同步失锁垂直滚屏"
                "（暴露 25 行 VBI 场消隐锤头黑带与 VITC 白点）、行 AFC 顶部旗帜弯折、PAL 汉诺威双行盲窗错色条，以及磁带覆录"
                "CTL 控制轨跳变、1H 延迟线 DOC 失落补偿彗星尾与底部磁头切换带",
        "standard": "pal",
        "layers": {
            "camera": {
                "sensor_noise": 0.05, "noise_grain": 1.4, "fixed_pattern": 0.03,
                "chroma_aberration": 0.08, "motion_blur": 0.20, "sharpness": 0.88,
                "exposure": -0.08, "white_balance_temp": -0.08, "ccd_smear": 0.22,
                "halation": 0.32, "halation_source_colour": 0.75, "dynamic_range": 7.4,
                "shadow_depth": 0.28, "dsp_contrast": 0.22,
            },
            "format": {
                "filter_kind": 1.0, "filter_sections": 3.0,
                "luma_bandwidth": 2.85, "chroma_bandwidth": 0.58,
                "chroma_vertical_averaging": 0.85, "separation_kind": 2.0,
                "comb_depth": 0.65, "cross_color": 0.42, "cross_luma": 0.28,
                "edge_boost": 0.34, "color_noise": 0.26, "chroma_gain": 1.08,
                "chroma_lag": 0.22,
            },
            "connector": {
                "av_present": 1.0, "av_moisture": 0.08, "av_corrosion": 0.05,
                "av_ground_loop": 0.14, "mains_hz": 50.0,
                "rf_present": 0.0,
            },
            "rf": {
                "enabled": 1.0, "tv_standard": 0.0, "channel_mhz": 8.0, "carrier_offset_mhz": 1.25,
                "propagation_mode": 0.0, "distance_km": 38.0,
                "signal_dbm": -62.0, "noise_figure_db": 8.5,
                "tuning_offset_khz": 28.0, "if_saw_ringing": 0.38,
                "sync_loss_threshold_db": 29.0, "sync_loss_amount": 0.95,
                "color_killer_threshold_db": 20.5,
                "adjacent_channel": 0.22, "broadcast_noise": 0.16,
                "channel_switch_rate": 0.55, "channel_switch_dur_s": 0.48,
                "lo_sweep_mhz": 0.82, "vsync_roll": 0.85, "vbi_bar_vis": 0.88,
                "flagging": 0.72, "hanover_bars": 0.68,
            },
            "tape": {
                "tape_speed": 0.0, "sp_lp_mix": 0.25, "tape_stability": 0.48,
                "line_jitter": 0.34, "walk_amplitude": 0.26, "micro_jitter": 0.18,
                "vertical_instability": 0.12, "row_timebase": 0.16,
                "head_switching": 0.88, "head_switch_lines": 12.0, "head_switch_mid_line": 0.35,
                "tracking_error": 0.48, "tracking_noise_rows": 36.0,
                "tracking_noise_wave": 1.8, "tracking_noise_snow": 0.28,
                "dropout_rate": 0.85, "dropout_len": 3.2, "dropout_color": 0.42,
                "doc_hold": 0.78, "edit_glitch_rate": 0.35, "edit_glitch_severity": 0.75,
                "wear": 0.34, "crease_rate": 0.22, "noise_floor": 0.22,
                "row_chroma_error": 0.14, "row_chroma_phase": 34.0,
            },
            "player": {
                "tbc_enabled": 0.0, "transport_jitter": 0.36, "head_wear": 0.24,
                "preemphasis": 0.32, "ringing": 0.26, "ringing_freq": 0.50,
                "luma_bandwidth": 2.9, "chroma_bandwidth": 0.60,
                "separation_kind": 2.0, "comb_depth": 0.65,
                "cross_color": 0.38, "cross_luma": 0.26, "color_noise": 0.22,
            },
            "display": {
                "enabled": 1.0, "gamma": 2.38, "contrast": 0.58, "brightness": 0.48,
                "saturation": 0.52, "scanlines": 0.36, "spot_sigma": 0.40,
                "spot_bloom": 0.52, "bloom": 0.38, "glass_halation": 0.36,
                "phosphor_saturation": 0.38, "mask_type": 1.0, "mask_pitch": 3.0,
                "mask_depth": 0.38, "phosphor_rgb": 0.58, "geometry_warp": 0.16,
                "curvature": 0.24, "corner_defocus": 0.32, "convergence": 0.24,
                "ehv_sag": 0.35, "breathing": 0.28, "vignette": 0.28,
                "interlace_flicker": 0.26, "bezel": 0.32,
            },
            "audio": {
                "enabled": 1.0, "mode": 6.0, "wow_flutter": 0.42,
                "stiction_pitch": 0.68, "dropout_sensitivity": 0.65,
                "azimuth_loss": 0.28, "hiss_floor": 0.32,
                "head_switch_buzz": 0.38, "ground_hum": 0.24, "rf_buzz": 0.58,
                "linear_bandwidth": 5400.0, "saturation": 0.36,
                "speaker_sim": 0.45, "flyback_whine": 0.22,
            },
        }},
    "mibox_vhf6_panasonic_j27_pal_d": {
        "label": "小米盒子 720p60 → HDMI-AV → PAL-D VHF 6频道(171.25MHz) → 松下 NV-J27 → AV采集卡（16:9→4:3 二次压扁）",
        "note": "小米盒子（720P 60Hz 强制 16:9 输出）→ HDMI-AV 转换盒 → PAL-D 射频调制（VHF 6 频道：168–176 MHz，图像载波 171.25 MHz，伴音载波 177.75 MHz，6.5 MHz 伴音间隔）经闭路电视同轴线进入松下 NV-J27 录像机高频头，4:3 原图被 16:9→4:3 二次横向压缩为 ×0.75（9:16 瘦长比例 + 两侧 12.5% 消隐黑边），再由录像机 AV 端子输出至电脑采集卡",
        "layers": {
            "camera": {
                "sensor_noise": 0.0, "noise_grain": 1.0, "fixed_pattern": 0.0,
                "chroma_aberration": 0.0, "motion_blur": 0.18, "sharpness": 1.04,
                "exposure": 0.0, "white_balance_temp": 0.0, "ccd_smear": 0.0,
                "halation": 0.08, "halation_source_colour": 0.70, "dynamic_range": 9.5,
            },
            "format": {
                "filter_kind": 2.0, "filter_sections": 2.0,
                "luma_bandwidth": 3.4, "chroma_bandwidth": 0.68,
                "chroma_vertical_averaging": 0.78, "separation_kind": 2.0,
                "comb_depth": 0.68, "cross_color": 0.32, "cross_luma": 0.22,
                "edge_boost": 0.30, "color_noise": 0.14, "chroma_gain": 1.04,
            },
            "connector": {
                "hdmi_present": 1.0, "hdmi_moisture": 0.0, "hdmi_corrosion": 0.0,
                "aspect_squeeze": 1.0,
                "av_present": 1.0, "av_moisture": 0.04, "av_corrosion": 0.02,
                "av_ground_loop": 0.06, "mains_hz": 50.0,
                "rf_present": 1.0, "rf_moisture": 0.04, "rf_corrosion": 0.02,
                "rf_shielding": 0.18,
            },
            "rf": {
                "channel_mhz": 8.0, "carrier_offset_mhz": 1.25,
                "signal_dbm": -54.0, "noise_figure_db": 7.5,
                "sync_loss_threshold_db": 28.0, "sync_loss_amount": 0.4,
                "adjacent_channel": 0.18, "broadcast_noise": 0.04,
            },
            "tape": {
                "tape_speed": 0.0, "tape_stability": 0.92, "line_jitter": 0.05,
                "walk_amplitude": 0.04, "micro_jitter": 0.04, "vertical_instability": 0.01,
                "head_switching": 0.0, "dropout_rate": 0.0, "noise_floor": 0.04, "wear": 0.0,
            },
            "player": {
                "tbc_enabled": 0.0, "transport_jitter": 0.14, "head_wear": 0.08,
                "preemphasis": 0.28, "ringing": 0.22, "ringing_freq": 0.52,
                "luma_bandwidth": 3.4, "chroma_bandwidth": 0.68,
                "separation_kind": 2.0, "comb_depth": 0.68,
                "cross_color": 0.32, "cross_luma": 0.22, "color_noise": 0.14,
            },
            "display": {"enabled": 0.0},
        }},
    "hdmi_av_rf_hdmi_damp": {
        "label": "HDMI→AV→RF→HDMI 回路（RF 接口受潮实拍）",
        "note": "HDMI 数字源 → AV 复合转换盒 → RF 射频调制（频道 86）→ 受潮同轴 RF 接口（介电常数骤升 + 氧化整流）→ 高频头解调 → HDMI 采集：双稳态行同步延迟（右侧 HBI 黑边内缩与撕裂）、色同步门偏移反相（蓝屏翻转为炽烈红橙与横向失锁彩带）、斜向 6.4×10.2px 驻波差拍网格纹及行钳位充放电横条",
        "layers": {
            "camera": {"sensor_noise": 0.0, "noise_grain": 0.0, "sharpness": 1.0,
                       "chroma_aberration": 0.0, "ccd_smear": 0.0, "halation": 0.0},
            "format": {"luma_bandwidth": 4.2, "chroma_bandwidth": 1.1,
                       "chroma_vertical_averaging": 0.25, "comb_depth": 0.45,
                       "cross_color": 0.35, "cross_luma": 0.22, "color_noise": 0.12,
                       "edge_boost": 0.20, "chroma_gain": 1.0},
            "codec": {"dv_enabled": 0.0},
            "connector": {"hdmi_present": 1.0, "hdmi_moisture": 0.0, "hdmi_corrosion": 0.0,
                          "av_present": 1.0, "av_moisture": 0.12, "av_corrosion": 0.08, "av_ground_loop": 0.08,
                          "rf_present": 1.0, "rf_moisture": 0.80, "rf_corrosion": 0.35, "rf_shielding": 0.20},
            "rf": {"channel_mhz": 6.0, "carrier_offset_mhz": 1.25, "signal_dbm": -68.0,
                   "noise_figure_db": 10.0, "sync_loss_threshold_db": 32.0,
                   "sync_loss_amount": 1.0, "adjacent_channel": 0.35, "broadcast_noise": 0.08},
            "tape": {"tape_speed": 0.0, "tape_stability": 1.0, "wear": 0.0,
                     "dropout_rate": 0.0, "noise_floor": 0.0, "line_jitter": 0.0,
                     "walk_amplitude": 0.0, "micro_jitter": 0.0, "vertical_instability": 0.0,
                     "head_switching": 0.0, "tracking_error": 0.0, "crease_rate": 0.0},
            "player": {"transport_jitter": 0.0, "tbc_enabled": 1.0, "tbc_strength": 1.0,
                       "tbc_residual": 0.0, "head_wear": 0.0, "frequency_response": 0.0},
            "display": {"enabled": 0.0},
        }},
    "sony_cctv_composite": {
        "label": "Sony 工业枪机 CCD（BNC 转 AV 复合视频采集卡直采）",
        "note": "Sony Color Video Camera 工业枪机（外绑 12V 锂电 + DC 自动光圈 C 口镜头）经 BNC-RCA 复合视频线直入 USB 采集卡：赛德尔彗差三瓣扇形、自动光圈叶片衍射暗缺口、CCD 溢出漏极（VOD/HOD）前缘黑弧、上缘暖橙色散与右下钴蓝彗尾",
        "layers": {
            "camera": {"sensor_type": 0.0, "sensor_noise": 0.012, "ccd_gain": 0.05,
                       "sharpness": 1.16, "dynamic_range": 7.8,
                       "coma": 0.75, "axis_offset": 0.50, "overflow_drain": 0.65,
                       "chroma_aberration": 0.08, "longitudinal_ca": 0.05,
                       "color_cast": 0.0, "white_balance_temp": 0.0,
                       "ccd_smear": 0.0, "halation": 0.0},
            "format": {"luma_bandwidth": 5.0, "chroma_bandwidth": 1.75,
                       "filter_kind": 0.0, "chroma_vertical_averaging": 0.08,
                       "comb_depth": 0.95, "cross_color": 0.04, "cross_luma": 0.02,
                       "color_noise": 0.015, "edge_boost": 0.18, "chroma_gain": 1.38},
            "codec": {"dv_enabled": 0.0},
            "tape": {"tape_speed": 0.0, "tape_stability": 1.0, "wear": 0.0,
                     "dropout_rate": 0.0, "noise_floor": 0.008, "line_jitter": 0.0,
                     "walk_amplitude": 0.0, "micro_jitter": 0.0, "vertical_instability": 0.0,
                     "head_switching": 0.0, "tracking_error": 0.0, "crease_rate": 0.0},
            "player": {"transport_jitter": 0.0, "tbc_enabled": 1.0, "tbc_strength": 1.0,
                       "tbc_residual": 0.0, "head_wear": 0.0, "frequency_response": -0.15,
                       "luma_bandwidth": 5.0, "chroma_bandwidth": 1.75,
                       "comb_depth": 0.95, "cross_color": 0.04, "cross_luma": 0.02},
            "display": {"enabled": 0.0},
        }},
    "hi8_trv": {
        "label": "Sony CCD-TRV（Hi8 模拟 8mm 摄像机）",
        "note": "同代 Sony 1/4 英寸 互补色 CFA CCD + 低照度 AGC 噪波 + Hi8 高带模拟 FM 亮度 (4.2 MHz) 与 743 kHz 欠载色度记录",
        "layers": {
            "camera": {"sensor_type": 0.0, "sensor_noise": 0.09, "ccd_gain": 0.68,
                       "sharpness": 0.80, "chroma_aberration": 0.05, "dynamic_range": 7.5,
                       "color_cast": 0.45, "white_balance_temp": 0.0,
                       "ccd_smear": 0.28, "halation": 0.28},
            "format": {"luma_bandwidth": 4.2, "chroma_bandwidth": 0.75,
                       "chroma_vertical_averaging": 0.55, "comb_depth": 0.65,
                       "cross_color": 0.28, "cross_luma": 0.18, "color_noise": 0.22,
                       "edge_boost": 0.24, "chroma_gain": 1.0},
            "codec": {"dv_enabled": 0.0},
            "tape": {"tape_speed": 0.0, "tape_stability": 0.76, "wear": 0.15,
                     "dropout_rate": 0.15, "noise_floor": 0.14, "line_jitter": 0.14,
                     "walk_amplitude": 0.10, "micro_jitter": 0.08,
                     "head_switching": 0.35, "head_switch_lines": 6.0,
                     "tracking_error": 0.15, "shuttle_mode": 1.0},
            "player": {"transport_jitter": 0.18, "tbc_enabled": 0.0, "head_wear": 0.18,
                       "frequency_response": 0.08, "luma_bandwidth": 4.2,
                       "chroma_bandwidth": 0.75, "comb_depth": 0.65,
                       "cross_color": 0.28, "cross_luma": 0.18},
            "display": {"enabled": 0.0},
        }},
    "minidv_clean": {
        "label": "MiniDV 干净",
        "layers": {
            "camera": {"sharpness": 0.98, "chroma_aberration": 0.0, "sensor_noise": 0.02, "ccd_smear": 0.0, "halation": 0.05},
            "format": {"luma_bandwidth": 5.2, "chroma_bandwidth": 1.8, "chroma_vertical_averaging": 0.0, "color_noise": 0.02},
            "codec": {"dv_enabled": 1.0, "dv_quality": 1.0},
            "tape": {"dropout_rate": 0.0, "noise_floor": 0.01, "line_jitter": 0.01, "walk_amplitude": 0.01, "head_switching": 0.0},
            "player": {"tbc_enabled": 1.0, "transport_jitter": 0.02, "head_wear": 0.02, "luma_bandwidth": 5.2, "chroma_bandwidth": 1.8},
            "display": {"enabled": 0.0},
        }},
    "minidv_shuttle": {
        "label": "MiniDV / Digital8 快进/倒带搜索",
        "note": "解码器用缓冲区旧帧补齐——成片、带锯齿边缘的 16x16 宏块残留",
        "layers": {
            "camera": {"sensor_type": 0.0, "sensor_noise": 0.08, "ccd_gain": 0.65,
                       "sharpness": 0.82, "chroma_aberration": 0.04, "dynamic_range": 7.5,
                       "color_cast": 0.45, "white_balance_temp": 0.0,
                       "ccd_smear": 0.25, "halation": 0.25},
            "format": {"luma_bandwidth": 5.0, "chroma_bandwidth": 2.0,
                       "chroma_vertical_averaging": 0.0, "color_noise": 0.04, "edge_boost": 0.18},
            "codec": {"dv_enabled": 1.0, "dv_quality": 0.90, "dv_base_step": 0.015},
            "tape": {"shuttle_speed": 5.0, "shuttle_stale": 0.19,
                     "shuttle_cluster": 0.55, "shuttle_carry": 0.45,
                     "shuttle_mode": 2.0, "dropout_rate": 0.0, "noise_floor": 0.03,
                     "line_jitter": 0.03, "walk_amplitude": 0.02, "micro_jitter": 0.03,
                     "head_switching": 0.0},
            "player": {"tbc_enabled": 1.0, "head_wear": 0.05, "transport_jitter": 0.05,
                       "luma_bandwidth": 5.0, "chroma_bandwidth": 2.0},
            "display": {"enabled": 0.0},
        }},
    "capital_mv_cue_4bars": {
        "label": "VHS 全屏 5x 快进跨轨搜索扫描线（4条双峰方位角噪波条）",
        "note": "螺旋扫描磁鼓 5x 倍速搜索（Cue/Review）跨越 |5-1|=4 条磁迹：垂直等距 (281 px) 向下滚动 (68 px/帧) 的 4 条灰白限幅器噪波条 (L=0.565) + 尾部 H-Sync 电压黑槽切口 (L=0.25) 与副峰回弹 + 629 kHz 欠载色度消隐",
        "layers": {
            "camera": {"sensor_noise": 0.03, "sharpness": 0.90, "dynamic_range": 8.0},
            "format": {"luma_bandwidth": 3.2, "chroma_bandwidth": 0.65,
                       "chroma_vertical_averaging": 0.65, "comb_depth": 0.70,
                       "color_noise": 0.14, "edge_boost": 0.20, "chroma_gain": 0.95},
            "codec": {"dv_enabled": 0.0},
            "tape": {"shuttle_speed": 5.0, "shuttle_stale": 0.22,
                     "shuttle_span": 10.0, "shuttle_wave": 16.0,
                     "shuttle_mode": 1.0, "shuttle_guard_bars": 0.95,
                     "shuttle_bar_luma": 0.565, "shuttle_PIP_bars": 0.0,
                     "tape_stability": 0.72, "line_jitter": 0.16,
                     "walk_amplitude": 0.12, "noise_floor": 0.12,
                     "dropout_rate": 0.08, "head_switching": 0.45,
                     "head_switch_lines": 8.0, "tracking_error": 0.20},
            "player": {"tbc_enabled": 0.0, "transport_jitter": 0.18,
                       "head_wear": 0.15, "luma_bandwidth": 3.2, "chroma_bandwidth": 0.65},
            "display": {"enabled": 0.0},
        }},
    "capital_mv_pip_9bars": {
        "label": "VHS 分屏子窗 9x 倍速搜索扫描线（8条主副跨轨噪波条）",
        "note": "DVE 切换台上半幅子窗（Y ∈ [0, 0.562H]）输入 9x 倍速搜索画面：主条周期 129.3 px、含方位角副条共 8 条水平噪波带以 21 px/帧匀速下滚，峰值 L=0.750，行同步错位右移 +48 px，下半幅红衣队列保持正常锁相播放",
        "layers": {
            "camera": {"sensor_noise": 0.03, "sharpness": 0.92, "dynamic_range": 8.0},
            "format": {"luma_bandwidth": 3.4, "chroma_bandwidth": 0.68,
                       "chroma_vertical_averaging": 0.60, "comb_depth": 0.72,
                       "color_noise": 0.12, "edge_boost": 0.22, "chroma_gain": 1.0},
            "codec": {"dv_enabled": 0.0},
            "tape": {"shuttle_speed": 5.0, "shuttle_stale": 0.24,
                     "shuttle_span": 8.0, "shuttle_wave": 18.0,
                     "shuttle_mode": 1.0, "shuttle_guard_bars": 0.95,
                     "shuttle_bar_luma": 0.750, "shuttle_PIP_bars": 1.0,
                     "tape_stability": 0.75, "line_jitter": 0.14,
                     "walk_amplitude": 0.10, "noise_floor": 0.10,
                     "dropout_rate": 0.05, "head_switching": 0.0},
            "player": {"tbc_enabled": 0.0, "transport_jitter": 0.15,
                       "head_wear": 0.12, "luma_bandwidth": 3.4, "chroma_bandwidth": 0.68},
            "display": {"enabled": 0.0},
        }},
    "hi8_shuttle": {
        "label": "Hi8 快进/倒带搜索",
        "note": "Sony 1/4 英寸 CCD 色彩与低照度噪波 + 模拟螺旋扫描跨磁迹方位角失锁噪波条、行同步斜坡剪切与 743 kHz 欠载色度消隐",
        "layers": {
            "camera": {"sensor_type": 0.0, "sensor_noise": 0.09, "ccd_gain": 0.68,
                       "sharpness": 0.80, "chroma_aberration": 0.05, "dynamic_range": 7.5,
                       "color_cast": 0.45, "white_balance_temp": 0.0,
                       "ccd_smear": 0.28, "halation": 0.28},
            "format": {"luma_bandwidth": 4.2, "chroma_bandwidth": 0.75,
                       "chroma_vertical_averaging": 0.55, "comb_depth": 0.65,
                       "cross_color": 0.28, "cross_luma": 0.18, "color_noise": 0.22,
                       "edge_boost": 0.24, "chroma_gain": 1.0},
            "codec": {"dv_enabled": 0.0},
            "tape": {"tape_speed": 0.0, "tape_stability": 0.72, "wear": 0.16,
                     "dropout_rate": 0.18, "noise_floor": 0.15, "line_jitter": 0.18,
                     "walk_amplitude": 0.14, "micro_jitter": 0.10,
                     "head_switching": 0.45, "head_switch_lines": 8.0,
                     "tracking_error": 0.22,
                     "shuttle_speed": 5.0, "shuttle_stale": 0.19,
                     "shuttle_span": 8.0, "shuttle_wave": 14.0,
                     "shuttle_cluster": 0.55, "shuttle_carry": 0.45,
                     "shuttle_mode": 1.0},
            "player": {"transport_jitter": 0.22, "tbc_enabled": 0.0, "head_wear": 0.18,
                       "frequency_response": 0.10, "luma_bandwidth": 4.2,
                       "chroma_bandwidth": 0.75, "comb_depth": 0.65,
                       "cross_color": 0.25, "cross_luma": 0.16},
            "display": {"enabled": 0.0},
        }},

    # --- demonstrations of the dimensions added later -------------------------
    # Every one of these defaults to off -- an unrequested dimension must not move an existing preset --
    # so without a preset that switches them on, nobody would find them.
    "cold_start_handheld": {
        "label": "冷机开机 · 手持（新维度演示）",
        "note": "一台冷机在手里拍的：三个自动环还在收敛、手在抖、录像机电容已经老化。"
                "注意画面亮度在前两秒里还在变——那是 AE 在追目标，不是曝光参数在动。",
        "kind": "composite",
        "layers": {
            "camera": {
                # the loops, all settling from switch-on
                "ae_enabled": 1.0, "ae_tau_s": 0.9, "ae_target": 0.44,
                "awb_enabled": 1.0, "awb_tau_s": 1.6,
                "af_enabled": 1.0, "af_tau_s": 0.5,
                # the hand
                "shake_intensity": 0.85, "bump_rate_hz": 0.5, "bump_amount": 0.7,
                # a consumer CCD, as these machines were
                "sensor_type": 0.0, "sensor_noise": 0.075, "fixed_pattern": 0.05,
                "sharpness": 0.8, "chroma_aberration": 0.06, "ccd_smear": 0.4,
                "halation": 0.5, "dynamic_range": 7.2, "lens_wear": 0.12,
                # the note's optical and sensor faults, all belonging to this one camera.
                # `lens_fog` is deliberately *not* set here: it is a state that clears as the glass
                # warms, so it is driven by the timeline below, alongside the warmth.
                "bloom_amount": 0.35, "bloom_threshold": 0.80,
                "dead_sites": 2.0, "stuck_fraction": 0.5,
            },
            # head wear is the machine's, not the medium's: it is on the player layer, which is also
            # where its per-unit spread lives
            "tape": {"dropout_rate": 0.9, "tracking_error": 0.16},
            "player": {"head_wear": 0.34, "burst_phase_error": 1.4, "frequency_response": 0.85},
            "display": {"enabled": 0.0, "bloom": 0.4, "scanlines": 0.35, "brightness": 0.46, "contrast": 1.05},
        },
        # the machine's condition and the timeline that moves it, together
        "environment": {"warmth": 0.0, "age": 0.65, "humidity": 0.2},
        "timeline": [
            {"layer": "environment", "key": "warmth", "absolute": True,
             "points": [{"t": 0.0, "value": 0.0, "curve": "exp"},
                        {"t": 6.0, "value": 1.0, "curve": "hold"}]},
            # the fog clears as the front element warms: the same physical process as the warmth track
            # above, acting on the glass instead of on the electronics.  It starts heavy and falls away
            # over about eight seconds.
            {"layer": "camera", "key": "lens_fog", "absolute": True,
             "points": [{"t": 0.0, "value": 0.55, "curve": "exp"},
                        {"t": 8.0, "value": 0.02, "curve": "hold"}]},
        ],
    },
    "warm_archival": {
        "label": "已预热的存档带（对照组）",
        "note": "同一台机器、同一盘带，但机器已经热了、手是稳的、自动环已经收敛。"
                "和上面那条对照着看，能分出哪些是机器的状态、哪些是介质本身的缺陷。",
        "kind": "composite",
        "layers": {
            "camera": {
                "ae_enabled": 1.0, "ae_tau_s": 0.9, "ae_target": 0.44,
                "awb_enabled": 1.0, "awb_tau_s": 1.6,
                "sensor_type": 0.0, "sensor_noise": 0.075, "sharpness": 0.8,
                "chroma_aberration": 0.06, "ccd_smear": 0.4, "halation": 0.5,
                "dynamic_range": 7.2, "lens_wear": 0.12,
                # the same camera, so the same failed sites and the same iris -- but warm, so no fog
                "bloom_amount": 0.35, "bloom_threshold": 0.80,
                "dead_sites": 2.0, "stuck_fraction": 0.5,
            },
            # head wear is the machine's, not the medium's: it is on the player layer, which is also
            # where its per-unit spread lives
            "tape": {"dropout_rate": 0.9, "tracking_error": 0.16},
            "player": {"head_wear": 0.34, "burst_phase_error": 1.4, "frequency_response": 0.85},
            "display": {"enabled": 0.0, "bloom": 0.4, "scanlines": 0.35, "brightness": 0.46, "contrast": 1.05},
        },
        "environment": {"warmth": 1.0, "age": 0.65, "humidity": 0.2},
        "timeline": [],
    },
}


def resolve_preset(layers: dict) -> dict:
    """Expand a preset through the schema, filling unspecified values."""
    return SCHEMA.resolve_all(layers)


# ---------------------------------------------------------------------------
# compatibility aliases
#
# The tooling refers to these profiles by their earlier names.  Aliases keep both
# spellings working; each one points at the canonical entry above.  Two names for
# one thing is a cost, so the mapping is explicit and kept in one place.
# ---------------------------------------------------------------------------
MEDIA_PROFILES["camcorder_hi8"] = MEDIA_PROFILES["hi8"]
MEDIA_PROFILES["clean_passthrough"] = {
    "label": "理想直通（无介质损伤）",
    "note": "用来取得「链路什么都没做」的对照",
    "layers": {"tape": {"wear": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0,
                        "head_switching": 0.0, "crease_rate": 0.0,
                        "line_jitter": 0.0, "walk_amplitude": 0.0,
                        "micro_jitter": 0.0, "vertical_instability": 0.0},
               "player": {"transport_jitter": 0.0, "head_wear": 0.0,
                          "frequency_response": 0.0, "tbc_enabled": 1.0},
               "display": {"enabled": 0.0}},
}
PLAYER_PROFILES["vcr_consumer"] = PLAYER_PROFILES["deck_consumer"]
DISPLAY_PROFILES["none"] = {"label": "不经显示设备",
                            "note": "采集卡直入：几何、扫描线、泛光都不参与",
                            "layers": {"display": {"enabled": 0.0}}}
CAMERA_PROFILES["ccd_camcorder_1990"] = CAMERA_PROFILES["ccd_single"]
CAMERA_PROFILES["clean_digital"] = {
    "label": "理想数字采集（无相机劣化）",
    "note": "用作对照：相机层全部中和",
    "layers": {"camera": {"sensor_noise": 0.0, "noise_grain": 0.0,
                          "fixed_pattern": 0.0, "chroma_aberration": 0.0,
                          "motion_blur": 0.0, "sharpness": 1.0, "exposure": 0.0,
                          "white_balance_temp": 0.0, "ccd_smear": 0.0,
                          "halation": 0.0, "ccd_gain": 0.0, "color_cast": 0.0,
                          "highlight_tint": 0.0, "lens_wear": 0.0, "lens_flare": 0.0},
               "format": {"edge_boost": 0.0, "cross_color": 0.0, "cross_luma": 0.0,
                          "color_noise": 0.0, "chroma_vertical_averaging": 0.0,
                          "comb_depth": 0.0, "burst_phase_error": 0.0,
                          "luma_bandwidth": 5.0, "chroma_bandwidth": 1.5,
                          "filter_kind": 0.0},
               "display": {"enabled": 0.0}},
}
FACTORY_PRESETS["guangdong_tv_dx_ducting_clear"] = {
    "label": "广东无线模拟电视接收测试 · TV-DX 大气波导超视距清晰接收（淘宝电视卡录制）",
    "note": "PAL-D/I + TV-DX 大气波导与电离层 Es 异常传播增益 + 廉价电视卡采集",
    "standard": "pal",
    "layers": {
        "camera": {"sharpness": 1.04, "sensor_noise": 0.01, "dynamic_range": 9.5},
        "format": {
            "luma_bandwidth": 4.3, "chroma_bandwidth": 1.15,
            "chroma_vertical_averaging": 0.35, "edge_boost": 0.28,
            "color_noise": 0.06, "chroma_gain": 1.06,
        },
        "codec": {"dv_enabled": 0.0},
        "connector": {"rf_present": 0.0},
        "rf": {
            "enabled": 1.0, "tv_standard": 0.0, "channel_mhz": 8.0, "carrier_offset_mhz": 1.25,
            "propagation_mode": 1.0, "distance_km": 185.0,
            "signal_dbm": -74.0, "dx_duct_gain_db": 24.5,
            "qsb_depth_db": 3.2, "qsb_rate_hz": 0.28,
            "multipath_level": 0.14, "multipath_delay_us": 1.45,
            "multipath_phase_deg": 28.0,
            "tuner_type": 1.0, "noise_figure_db": 8.5, "tuning_offset_khz": 18.0,
            "front_end_overload": 0.10,
            "cochannel_interference": 0.10, "cochannel_offset_khz": 10.5,
            "adjacent_channel": 0.08, "sync_loss_threshold_db": 26.0,
            "sync_loss_amount": 0.6, "color_killer_threshold_db": 20.0,
            "broadcast_noise": 0.06,
        },
        "tape": {
            "tape_stability": 0.98, "line_jitter": 0.02, "walk_amplitude": 0.01,
            "micro_jitter": 0.01, "vertical_instability": 0.0,
            "head_switching": 0.0, "head_switch_lines": 0.0,
            "dropout_rate": 0.0, "wear": 0.0, "crease_rate": 0.0, "noise_floor": 0.01,
        },
        "player": {
            "tbc_enabled": 1.0, "tbc_strength": 0.96, "tbc_residual": 0.04,
            "head_wear": 0.0, "frequency_response": 0.05,
            "cross_color": 0.24, "cross_luma": 0.22, "separation_kind": 0.0,
        },
        "display": {"enabled": 0.0},
        "audio": {"enabled": 1.0, "mode": 7.0, "hiss_floor": 0.08, "head_switch_buzz": 0.12},
    },
}
FACTORY_PRESETS["guangdong_tv_dx_normal_weak"] = {
    "label": "广东无线模拟电视接收测试 · 常规超视距弱台对照（无大气波导，雪花与消色临界）",
    "note": "同距离无大气波导时，超视距绕射损耗导致载噪比跌落、雪花剧增且接近消色门限",
    "standard": "pal",
    "layers": {
        "camera": {"sharpness": 1.0, "sensor_noise": 0.01},
        "format": {
            "luma_bandwidth": 3.8, "chroma_bandwidth": 1.0,
            "chroma_vertical_averaging": 0.45, "edge_boost": 0.20,
            "color_noise": 0.15, "chroma_gain": 1.0,
        },
        "codec": {"dv_enabled": 0.0},
        "connector": {"rf_present": 0.0},
        "rf": {
            "enabled": 1.0, "tv_standard": 0.0, "channel_mhz": 8.0, "carrier_offset_mhz": 1.25,
            "propagation_mode": 0.0, "distance_km": 185.0,
            "signal_dbm": -74.0, "dx_duct_gain_db": 0.0,
            "qsb_depth_db": 2.5, "qsb_rate_hz": 0.25,
            "multipath_level": 0.22, "multipath_delay_us": 2.1,
            "tuner_type": 1.0, "noise_figure_db": 8.5, "tuning_offset_khz": 18.0,
            "cochannel_interference": 0.08,
            "adjacent_channel": 0.15, "sync_loss_threshold_db": 26.0,
            "sync_loss_amount": 0.85, "color_killer_threshold_db": 20.0,
            "broadcast_noise": 0.22,
        },
        "tape": {
            "tape_stability": 0.96, "line_jitter": 0.04, "walk_amplitude": 0.02,
            "micro_jitter": 0.02, "vertical_instability": 0.01,
            "head_switching": 0.0, "head_switch_lines": 0.0,
            "dropout_rate": 0.0, "wear": 0.0, "crease_rate": 0.0, "noise_floor": 0.02,
        },
        "player": {
            "tbc_enabled": 1.0, "tbc_strength": 0.90, "tbc_residual": 0.10,
            "cross_color": 0.28, "cross_luma": 0.25,
        },
        "display": {"enabled": 0.0},
        "audio": {"enabled": 1.0, "mode": 7.0, "hiss_floor": 0.45, "head_switch_buzz": 0.28},
    },
}
FACTORY_PRESETS["vhs_tv_zalu"] = {
    "label": "VHS 录像带电视杂录（无线模拟电视 RF 接收 + 家用 VHS 磁带录制）",
    "note": "先经模拟电视 RF 高频头接收解调（AM 画面带雪花/多径重影，FM 伴音受限幅静噪保护损耗极小），再录入手持/家用 VHS 磁带叠加走带抖晃与磁头切换",
    "standard": "pal",
    "layers": {
        "camera": {"sharpness": 1.0, "sensor_noise": 0.01, "dynamic_range": 9.0},
        "format": {
            "luma_bandwidth": 2.9, "chroma_bandwidth": 0.65,
            "chroma_vertical_averaging": 0.85, "edge_boost": 0.28,
            "color_noise": 0.22, "chroma_gain": 1.04,
        },
        "codec": {"dv_enabled": 0.0},
        "connector": {"rf_present": 0.0},
        "rf": {
            "enabled": 1.0, "tv_standard": 0.0, "channel_mhz": 8.0, "carrier_offset_mhz": 1.25,
            "propagation_mode": 0.0, "distance_km": 42.0,
            "signal_dbm": -66.0, "dx_duct_gain_db": 0.0,
            "multipath_level": 0.18, "multipath_delay_us": 1.35,
            "multipath_phase_deg": 32.0,
            "tuner_type": 0.0, "noise_figure_db": 7.0, "tuning_offset_khz": 12.0,
            "adjacent_channel": 0.12, "sync_loss_threshold_db": 26.0,
            "sync_loss_amount": 0.35, "color_killer_threshold_db": 18.0,
            "broadcast_noise": 0.14,
        },
        "tape": {
            "tape_stability": 0.58, "line_jitter": 0.28, "walk_amplitude": 0.22,
            "micro_jitter": 0.18, "vertical_instability": 0.06,
            "head_switching": 0.48, "head_switch_lines": 10.0,
            "dropout_rate": 0.25, "wear": 0.28, "crease_rate": 0.15, "noise_floor": 0.18,
        },
        "player": {
            "tbc_enabled": 1.0, "tbc_strength": 0.65, "tbc_residual": 0.28,
            "head_wear": 0.22, "frequency_response": -0.20,
            "cross_color": 0.28, "cross_luma": 0.24, "separation_kind": 0.0,
        },
        "display": {"enabled": 0.0, "scanlines": 0.28, "bloom": 0.25, "overscan": 0.04},
        "audio": {
            "enabled": 1.0, "mode": 1.0, "wow_flutter": 0.26,
            "azimuth_loss": 0.22, "hiss_floor": 0.22,
            "head_switch_buzz": 0.20, "rf_buzz": 0.15,
            "linear_bandwidth": 5800.0, "saturation": 0.24,
        },
    },
}
CAMERA_PROFILES["docu_1989_telecine"] = {
    "label": "1989 年代广播级专题纪录片 · 胶转磁 / 三管摄像机光影 (16mm Telecine + 3-Tube ENG)",
    "note": "80年代末电视专题片经典光影：暖调钨丝灯/日光胶片色温偏移、高光暖色漫射光晕（Halation）、厚重暗部趾区（Shadow Depth）与柔和肩部压缩",
    "values": {
        "exposure": 0.12, "white_balance_temp": 0.22,
        "dynamic_range": 7.4, "saturation_stops": 0.0,
        "halation": 0.42, "halation_source_colour": 0.65,
        "source_glow": 0.018, "shadow_depth": 0.34,
        "dsp_gamma": 0.88, "dsp_contrast": 0.28, "dsp_black": 0.018,
        "nv_highlight_compress": 1.35, "nv_highlight_knee": 0.76, "nv_highlight_shape": 2.2,
        "color_cast": 0.14, "highlight_tint": 0.22,
        "field_curvature": 0.16, "chroma_aberration": 0.12, "longitudinal_ca": 0.10,
        "lens_wear": 0.12, "sensor_noise": 0.035, "noise_grain": 1.6,
        "ccd_gain": 0.08, "sharpness": 1.08, "motion_blur": 0.18,
    },
}
MEDIA_PROFILES["umatic_sp_1989_pal"] = {
    "label": "1989 年代 PAL 广播母带 (U-matic SP / Betacam 复合播出带)",
    "note": "电视台专题纪录片播出母带：亮度带宽 3.8 MHz、色度带宽 1.0 MHz，带轻微模拟孔阑校正过冲与字幕边缘副载波串色",
    "standard": "pal",
    "values": {
        "format": {
            "luma_bandwidth": 3.8, "chroma_bandwidth": 1.0,
            "chroma_vertical_averaging": 0.65, "edge_boost": 0.34,
            "color_noise": 0.14, "chroma_gain": 1.08,
            "chroma_lag": 0.12, "filter_kind": 1.0, "filter_sections": 3.0,
        },
        "codec": {"dv_enabled": 0.0},
        "tape": {
            "tape_stability": 0.82, "line_jitter": 0.12, "walk_amplitude": 0.10,
            "micro_jitter": 0.08, "vertical_instability": 0.02,
            "head_switching": 0.22, "head_switch_lines": 6.0,
            "dropout_rate": 0.08, "dropout_len": 0.22,
            "wear": 0.12, "crease_rate": 0.04, "noise_floor": 0.09,
        },
    },
}
FACTORY_PRESETS["docu_1989_broadcast"] = {
    "label": "1989 年代电视专题纪录片 · 胶转磁光影与纪实解说字幕 (PAL 广播母带)",
    "note": "复现 1989 年代大型电视专题纪录片的胶转磁暖调高反差光影、高光晕染、暗部沉稳层次，以及录在磁带之前（at=1）的象牙白长宋解说字幕与打字机逐字显字动画",
    "standard": "pal",
    "layers": {
        "camera": dict(CAMERA_PROFILES["docu_1989_telecine"]["values"]),
        "format": dict(MEDIA_PROFILES["umatic_sp_1989_pal"]["values"]["format"]),
        "codec": {"dv_enabled": 0.0},
        "connector": {"av_present": 1.0, "av_moisture": 0.04, "av_corrosion": 0.05},
        "tape": dict(MEDIA_PROFILES["umatic_sp_1989_pal"]["values"]["tape"]),
        "player": {
            "tbc_enabled": 1.0, "tbc_strength": 0.85, "tbc_residual": 0.12,
            "head_wear": 0.10, "frequency_response": -0.08,
            "preemphasis": 0.28, "ringing": 0.22,
            "cross_color": 0.24, "cross_luma": 0.20, "separation_kind": 2.0,
        },
        "display": {
            "enabled": 0.0, "scanlines": 0.22, "bloom": 0.32, "defocus": 0.12,
            "gamma": 2.30, "color_temp": 0.18, "contrast": 0.58,
            "brightness": 0.49, "saturation": 0.54, "overscan": 0.03, "vignette": 0.22,
        },
        "overlay": {
            "enabled": 1.0, "at": 1.0, "font": 21.0,
            "text": "一九八九年 · 专题纪实\n岁月的河床与山川光影",
            "cell_h": 28.0, "halign": 1.0, "valign": 2.0,
            "margin": 0.06, "style": 7.0,
            "outline": 1.0, "shadow": 0.65, "shadow_x": 2.0, "shadow_y": 2.0,
            "edge_softness": 0.35, "glow": 0.28, "bevel": 0.15,
            "anim_mode": 1.0, "anim_speed": 1.15, "anim_in_s": 0.10, "anim_cycle_s": 0.0,
            "letter_gap": 2.0, "line_gap": 6.0,
        },
        "audio": {
            "enabled": 1.0, "mode": 1.0, "wow_flutter": 0.12,
            "azimuth_loss": 0.10, "hiss_floor": 0.14,
            "linear_bandwidth": 8500.0, "saturation": 0.18,
        },
    },
}
FACTORY_PRESETS["docu_1989_title_card"] = {
    "label": "1989 年代大型纪录片 · 片头大标宋金字立体叠印与光栅扫描字幕",
    "note": "复现 80 年代末专题纪录片片头：大标宋金黄立体浮雕字（bevel+深褐描边+光学辉光）、水平光栅扫描拉幕显字，以及 PAL 复合信号在金字边缘产生的细腻副载波串色",
    "standard": "pal",
    "layers": {
        "camera": dict(CAMERA_PROFILES["docu_1989_telecine"]["values"]),
        "format": dict(MEDIA_PROFILES["umatic_sp_1989_pal"]["values"]["format"]),
        "codec": {"dv_enabled": 0.0},
        "connector": {"av_present": 1.0, "av_moisture": 0.05, "av_corrosion": 0.06},
        "tape": dict(MEDIA_PROFILES["umatic_sp_1989_pal"]["values"]["tape"]),
        "player": {
            "tbc_enabled": 1.0, "tbc_strength": 0.82, "tbc_residual": 0.14,
            "head_wear": 0.12, "frequency_response": -0.10,
            "preemphasis": 0.32, "ringing": 0.26,
            "cross_color": 0.28, "cross_luma": 0.22, "separation_kind": 2.0,
        },
        "display": {
            "enabled": 0.0, "scanlines": 0.24, "bloom": 0.36, "defocus": 0.14,
            "gamma": 2.32, "color_temp": 0.20, "contrast": 0.60,
            "brightness": 0.49, "saturation": 0.56, "overscan": 0.03, "vignette": 0.25,
        },
        "overlay": {
            "enabled": 1.0, "at": 1.0, "font": 20.0,
            "text": "第一集  岁月与山河\n专题文献纪录片 · 一九八九",
            "cell_h": 36.0, "halign": 1.0, "valign": 1.0,
            "margin": 0.08, "style": 6.0,
            "outline": 1.0, "shadow": 0.78, "shadow_x": 3.0, "shadow_y": 3.0,
            "edge_softness": 0.28, "glow": 0.42, "bevel": 0.62,
            "anim_mode": 3.0, "anim_speed": 1.0, "anim_in_s": 0.10, "anim_cycle_s": 0.0,
            "letter_gap": 3.0, "line_gap": 10.0,
        },
        "audio": {
            "enabled": 1.0, "mode": 1.0, "wow_flutter": 0.12,
            "azimuth_loss": 0.10, "hiss_floor": 0.14,
            "linear_bandwidth": 8500.0, "saturation": 0.18,
        },
    },
}

CAMERA_PROFILES["hk_mv_zoom_1999"] = {
    "label": "1999 香港纪实/MV 广角变焦摄像机（逆光镀膜鬼影 + 冷暖分色）",
    "note": "单层 MgF2 镀膜蓝紫/品红双反射鬼影链、左下彩虹色散弧、CCD 垂直溢光与港式金琥珀/翡翠青分色",
    "values": {
        "sensor_noise": 0.12, "noise_grain": 1.3, "dynamic_range": 7.5,
        "sharpness": 0.92, "chroma_aberration": 0.42, "longitudinal_ca": 0.35,
        "field_curvature": 0.28, "coma": 0.25, "astigmatism": 0.18,
        "spherical_aberration": 0.32, "lens_wear": 0.36, "lens_flare": 0.58,
        "flare_blades": 6.0, "spike_spread": 0.72, "flare_defocus": 0.55,
        "coat_ghost": 0.85, "ccd_smear": 0.32, "halation": 0.38,
        "halation_source_colour": 0.65, "vcd_color_tone": 0.90,
        "ae_enabled": 1.0, "ae_tau_s": 0.85,
    },
}

MEDIA_PROFILES["vcd_pal_mpeg1_1999"] = {
    "label": "Video CD（PAL 352×288 · 1150 kbps · 模拟复合母带转压）",
    "note": "经模拟复合视频采集后压制的 MPEG-1 VCD：单场垂直抽样、4:2:0 8×8 离散余弦块伪影与左右 HBI 采集振铃边",
    "values": {
        "format": {
            "luma_bandwidth": 3.2, "chroma_bandwidth": 0.85,
            "chroma_vertical_averaging": 0.65, "comb_depth": 0.45,
            "cross_color": 0.28, "cross_luma": 0.22, "edge_boost": 0.32,
            "color_noise": 0.16, "burst_phase_error": 3.0,
        },
        "codec": {
            "codec_kind": 2.0, "vcd_quality": 0.95,
            "vcd_width": 352.0, "vcd_height": 288.0,
            "vcd_motion_comp": 1.0, "vcd_bitrate_kbps": 1150.0,
            "vcd_hbi_border": 0.85,
        },
        "tape": {
            "tape_stability": 0.88, "line_jitter": 0.08, "walk_amplitude": 0.05,
            "micro_jitter": 0.05, "vertical_instability": 0.02,
            "head_switching": 0.0, "dropout_rate": 0.0, "wear": 0.08,
            "crease_rate": 0.0, "noise_floor": 0.06,
        },
    },
}

FACTORY_PRESETS["hk_vcd_bauhinia_1999"] = {
    "label": "1999 香港风光/MV 实拍 VCD（逆光镀膜鬼影 · 琥珀青分色 · MPEG-1）",
    "standard": "pal",
    "layers": {
        "camera": dict(CAMERA_PROFILES["hk_mv_zoom_1999"]["values"]),
        "format": dict(MEDIA_PROFILES["vcd_pal_mpeg1_1999"]["values"]["format"]),
        "codec": dict(MEDIA_PROFILES["vcd_pal_mpeg1_1999"]["values"]["codec"]),
        "connector": {"av_present": 1.0, "av_moisture": 0.04, "av_corrosion": 0.04},
        "tape": dict(MEDIA_PROFILES["vcd_pal_mpeg1_1999"]["values"]["tape"]),
        "player": {
            "tbc_enabled": 1.0, "tbc_strength": 0.90, "tbc_residual": 0.08,
            "head_wear": 0.06, "frequency_response": -0.08,
            "preemphasis": 0.22, "ringing": 0.24,
            "cross_color": 0.22, "cross_luma": 0.18, "separation_kind": 2.0,
        },
        "display": {"enabled": 0.0},
    },
}


FACTORY_PRESETS["vhs_gen_1_master"] = {
    "label": "VHS 迭代翻录实验 · 第 1 代母带（清晰基准）",
    "note": "首代 VHS 录制直采（关闭 CRT 显示）：保留 3.0 MHz 柔和亮度、轻微 629 kHz 色度水平柔和度与底部单层磁头切换带",
    "standard": "ntsc",
    "layers": {
        "camera": {"sensor_noise": 0.03, "sharpness": 0.95, "dynamic_range": 8.5, "halation": 0.08, "ccd_smear": 0.0},
        "format": {"luma_bandwidth": 3.1, "chroma_bandwidth": 0.72, "chroma_vertical_averaging": 0.75,
                   "comb_depth": 0.55, "cross_color": 0.25, "cross_luma": 0.18, "edge_boost": 0.22,
                   "color_noise": 0.16, "chroma_gain": 1.0},
        "tape": {"generation_passes": 1.0, "gen_chroma_bleed": 0.85, "gen_ringing": 0.85, "gen_sync_collapse": 0.0,
                 "tape_speed": 0.0, "tape_stability": 0.72, "line_jitter": 0.18, "walk_amplitude": 0.14,
                 "dropout_rate": 0.12, "noise_floor": 0.12, "head_switching": 0.65, "head_switch_lines": 10.0,
                 "tracking_error": 0.22},
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.20, "head_wear": 0.12, "preemphasis": 0.18,
                   "ringing": 0.12, "luma_bandwidth": 3.1, "chroma_bandwidth": 0.72},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["vhs_gen_8_dub"] = {
    "label": "VHS 迭代翻录实验 · 第 8 代翻录（色度下垂洇色 + 边缘振铃显现）",
    "note": "经过 8 代 VCR→VCR 反复翻录：红蓝条纹衫与面部色块向右下跨界漂移约 10 px，人物头发与衣领右侧浮现明显黑白预加重振铃伴影，黑位抬升、高光偏黄绿",
    "standard": "ntsc",
    "layers": {
        "camera": {"sensor_noise": 0.04, "sharpness": 0.92, "dynamic_range": 8.0, "halation": 0.12, "ccd_smear": 0.0},
        "format": {"luma_bandwidth": 2.4, "chroma_bandwidth": 0.50, "chroma_vertical_averaging": 0.88,
                   "comb_depth": 0.65, "cross_color": 0.38, "cross_luma": 0.28, "edge_boost": 0.38,
                   "color_noise": 0.32, "chroma_gain": 0.88},
        "tape": {"generation_passes": 8.0, "gen_chroma_bleed": 0.90, "gen_ringing": 0.90, "gen_sync_collapse": 0.0,
                 "tape_speed": 0.0, "tape_stability": 0.46, "line_jitter": 0.38, "walk_amplitude": 0.30,
                 "dropout_rate": 0.45, "noise_floor": 0.22, "head_switching": 0.82, "head_switch_lines": 14.0,
                 "tracking_error": 0.38},
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.38, "head_wear": 0.26, "preemphasis": 0.32,
                   "ringing": 0.28, "luma_bandwidth": 2.5, "chroma_bandwidth": 0.52},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["vhs_gen_18_heavy"] = {
    "label": "VHS 迭代翻录实验 · 第 18 代重度损耗（多重斑马振铃 + 色彩坍塌）",
    "note": "经过 18 代反复翻录：多级 FM 孔径过冲形成强烈的多重黑白斑马纹轮廓光环，色度严重脱节下垂且粉紫/黄绿偏色，右侧 HBI 钳位白噪带加宽，行同步剧烈蛇形摆动",
    "standard": "ntsc",
    "layers": {
        "camera": {"sensor_noise": 0.05, "sharpness": 0.88, "dynamic_range": 7.5, "halation": 0.18, "ccd_smear": 0.0},
        "format": {"luma_bandwidth": 1.8, "chroma_bandwidth": 0.36, "chroma_vertical_averaging": 0.95,
                   "comb_depth": 0.75, "cross_color": 0.52, "cross_luma": 0.42, "edge_boost": 0.55,
                   "color_noise": 0.48, "chroma_gain": 0.65},
        "tape": {"generation_passes": 18.0, "gen_chroma_bleed": 1.05, "gen_ringing": 1.10, "gen_sync_collapse": 0.0,
                 "tape_speed": 0.0, "tape_stability": 0.25, "line_jitter": 0.68, "walk_amplitude": 0.55,
                 "dropout_rate": 1.10, "noise_floor": 0.35, "head_switching": 0.95, "head_switch_lines": 20.0,
                 "tracking_error": 0.58},
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.62, "head_wear": 0.45, "preemphasis": 0.48,
                   "ringing": 0.45, "luma_bandwidth": 1.9, "chroma_bandwidth": 0.38},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["vhs_gen_20_sync_collapse"] = {
    "label": "VHS 迭代翻录实验 · 第 20 代同步崩溃与消色（黑白斜纹失锁）",
    "note": "第 20 代极限翻录：行同步脉冲圆化跌破 AFC 锁相门限，自动消色器（Color Killer）切断色度变为黑白，全屏爆发高反差 FM 差拍斜纹与水平行撕裂，仅隐约可见人物剪影",
    "standard": "ntsc",
    "layers": {
        "camera": {"sensor_noise": 0.06, "sharpness": 0.85, "dynamic_range": 7.0, "halation": 0.20, "ccd_smear": 0.0},
        "format": {"luma_bandwidth": 1.5, "chroma_bandwidth": 0.30, "chroma_vertical_averaging": 0.95,
                   "comb_depth": 0.80, "cross_color": 0.60, "cross_luma": 0.50, "edge_boost": 0.65,
                   "color_noise": 0.55, "chroma_gain": 0.20},
        "tape": {"generation_passes": 20.0, "gen_chroma_bleed": 1.15, "gen_ringing": 1.25, "gen_sync_collapse": 0.92,
                 "tape_speed": 0.0, "tape_stability": 0.12, "line_jitter": 0.95, "walk_amplitude": 0.80,
                 "dropout_rate": 1.80, "noise_floor": 0.48, "head_switching": 1.0, "head_switch_lines": 24.0,
                 "tracking_error": 0.78},
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.85, "head_wear": 0.58, "preemphasis": 0.55,
                   "ringing": 0.52, "luma_bandwidth": 1.6, "chroma_bandwidth": 0.32},
        "display": {"enabled": 0.0},
    },
}


FACTORY_PRESETS["smpte_pluge_calibration_bench"] = {
    "label": "SMPTE RP-219 彩条与 PLUGE 广播级闭环校准台 (TSG-170A → VHS → TBC → 示波器)",
    "note": "机内测试信号发生器输出标准 SMPTE 75% 彩条、反向蓝条、-I/100%白/+Q 与 PLUGE (-4/0/+4 IRE) 黑位脉冲，经 VHS 磁带录放与数字 TBC 后送入波形/矢量示波器，自动闭环测量黑位截断、微分增益 (DG) 与微分相位 (DP)",
    "standard": "ntsc",
    "layers": {
        "format": {
            "test_gen_enabled": 1.0, "test_pattern_mode": 1.0, "test_tone_mode": 1.0,
            "test_amplitude_ire": 100.0, "luma_bandwidth": 3.2, "chroma_bandwidth": 0.75,
            "chroma_gain": 1.0, "burst_phase_error": 3.0,
        },
        "tape": {
            "tape_speed": 0.0, "tape_stability": 0.68, "line_jitter": 0.25,
            "walk_amplitude": 0.20, "wear": 0.15, "dropout_rate": 0.15,
            "noise_floor": 0.12, "head_switching": 0.65, "head_switch_lines": 10.0,
        },
        "player": {
            "tbc_enabled": 1.0, "tbc_strength": 0.90, "tbc_residual": 0.08,
            "transport_jitter": 0.15, "head_wear": 0.12,
        },
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["fcc_multiburst_vhs_freq_sweep"] = {
    "label": "FCC 0.5–5.0 MHz 多波群频响测试与 TBC 抑抖校准台 (Multiburst)",
    "note": "注入 0.5 / 1.0 / 2.0 / 3.0 / 4.2 / 5.0 MHz 标准正弦多波群图卡，直接测量滤波器、同轴电缆与磁带通道在各频点的 dB 衰减及 -3 dB 实测带宽截止点",
    "standard": "ntsc",
    "layers": {
        "format": {
            "test_gen_enabled": 1.0, "test_pattern_mode": 3.0, "test_tone_mode": 3.0,
            "test_amplitude_ire": 100.0, "luma_bandwidth": 3.0, "chroma_bandwidth": 0.75,
            "edge_boost": 0.15,
        },
        "tape": {
            "tape_speed": 0.0, "tape_stability": 0.72, "line_jitter": 0.22,
            "wear": 0.18, "dropout_rate": 0.0, "noise_floor": 0.08, "head_switching": 0.0,
        },
        "player": {
            "tbc_enabled": 1.0, "tbc_strength": 0.92, "tbc_residual": 0.06,
            "head_wear": 0.15, "luma_bandwidth": 3.0,
        },
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["broadcast_multiview_quad_wall"] = {
    "label": "演播室 2×2 四分屏 CRT 监视墙（CAM / VTR-RAW / TBC-OUT / PGM 同屏监看）",
    "note": "还原 90 年代导播室四分屏监视墙：4 台 CRT 分别监看摄像机源、磁带原始重放 (TBC Off 含底部切头撕裂)、TBC 锁相输出与主播出 PGM，且每台监视器具有独立 CRT 色温/欠扫描/荧光粉老化性格与硬件 Tally/UMD 标牌",
    "standard": "ntsc",
    "layers": {
        "camera": {"sensor_noise": 0.05, "sharpness": 0.88, "halation": 0.25},
        "format": {"luma_bandwidth": 3.1, "chroma_bandwidth": 0.72, "cross_color": 0.35},
        "tape": {
            "tape_speed": 0.0, "tape_stability": 0.52, "line_jitter": 0.38,
            "walk_amplitude": 0.28, "wear": 0.24, "dropout_rate": 0.35,
            "noise_floor": 0.16, "head_switching": 0.85, "head_switch_lines": 14.0,
            "tracking_error": 0.42,
        },
        "player": {
            "tbc_enabled": 1.0, "tbc_strength": 0.92, "tbc_residual": 0.06,
            "transport_jitter": 0.20, "head_wear": 0.18,
        },
        "display": {
            "enabled": 1.0, "mv_enabled": 1.0, "mv_layout": 0.0,
            "mv_unit_variance": 0.65, "mv_fault_channel": 0.0,
            "mv_fault_mode": 0.0, "mv_tally_umd": 0.0,
        },
    },
}
FACTORY_PRESETS["mcr_six_bridge_fault_wall"] = {
    "label": "总控机房 3×2 六画面监视桥（含通道失锁雪花/蓝屏/自由滚屏与 单色同步监看管）",
    "note": "还原总控机房 (MCR) 6 窗口监视桥：包含 D65 特丽珑主监、JVC 欠扫描机房监看、池上老化绿偏管、单色黑白同步检查管与 9300K 家用接收机，并在第 2 通道演示硬件失锁告警状态",
    "standard": "pal",
    "layers": {
        "format": {"luma_bandwidth": 3.2, "chroma_bandwidth": 0.75},
        "tape": {"tape_stability": 0.58, "line_jitter": 0.32, "head_switching": 0.80, "head_switch_lines": 12.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 0.90},
        "display": {
            "enabled": 1.0, "mv_enabled": 1.0, "mv_layout": 1.0,
            "mv_unit_variance": 0.75, "mv_fault_channel": 2.0,
            "mv_fault_mode": 5.0, "mv_tally_umd": 0.0,
        },
    },
}


FACTORY_PRESETS["studio_ccu_multi_cam_match"] = {
    "label": "演播室集中式 CCU 多机位色彩配平与 Blackburst 主同步工作台",
    "note": "模拟演播室机房视频工程师通过 CCU 遥调 2 号偏暖采访机 (Ikegami HL-55) 的光圈、总黑电平、R/B 白平衡与黑平衡基座，并通过 35 米脐带电缆连接主同步发生器 (BB Genlock)",
    "standard": "ntsc",
    "layers": {
        "camera": {
            "ccu_enabled": 1.0, "ccu_cam_id": 1.0, "ccu_iris_ev": 0.15,
            "ccu_master_pedestal_ire": 2.5, "ccu_r_gain": 0.92, "ccu_b_gain": 1.12,
            "ccu_r_flare": -0.015, "ccu_b_flare": 0.012,
            "ccu_knee_point": 0.80, "ccu_knee_slope": 0.30,
            "genlock_mode": 0.0, "genlock_cable_len_m": 35.0,
            "genlock_sc_phase_deg": 0.0, "genlock_h_phase_px": 0.0,
        },
        "format": {"luma_bandwidth": 4.2, "chroma_bandwidth": 1.1},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 0.95},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["genlock_cable_phase_and_unlock"] = {
    "label": "演播室 45m 脐带线缆未补偿相移与外同步掉锁故障对照台",
    "note": "模拟 45 米同轴脐带线未做 SC-Phase / H-Phase 补偿时引起的 ~293° 副载波色相偏转与水平消隐错位；可将 genlock_mode 切至 2 观察内晶振自由漂移滚屏",
    "standard": "ntsc",
    "layers": {
        "camera": {
            "ccu_enabled": 1.0, "ccu_cam_id": 0.0,
            "genlock_mode": 1.0, "genlock_cable_len_m": 45.0,
            "genlock_sc_phase_deg": 0.0, "genlock_h_phase_px": 0.0,
        },
        "format": {"test_gen_enabled": 1.0, "test_pattern_mode": 1.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["mcr_master_playout_on_air"] = {
    "label": "电视台总控主播出台 (PGM + DSK 台标/时钟/游走字幕 + Clean Feed 净输出)",
    "note": "还原电视台主控播出链路：支持一键切换监看「主播出母线 PGM（含 DSK 台标与游走字幕）」、「干净净画面 Clean Feed（无台标）」与「主控双总线导播控制台视图 (bus=3)」",
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.03, "sharpness": 0.95},
        "format": {"luma_bandwidth": 3.6, "chroma_bandwidth": 0.90},
        "tape": {"tape_stability": 0.78, "line_jitter": 0.14, "head_switching": 0.45, "head_switch_lines": 8.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 0.92},
        "display": {"enabled": 0.0},
        "overlay": {
            "playout_enabled": 1.0, "playout_output_bus": 3.0,
            "playout_state": 0.0, "playout_dsk_enabled": 1.0,
            "playout_dsk_crawl": 1.0, "playout_failover_auto": 1.0,
        },
    },
}
FACTORY_PRESETS["mcr_emergency_failover_desk"] = {
    "label": "主控播出故障应急接管与「PLEASE STAND BY / 请稍候」垫片切换台",
    "note": "模拟磁带断流或手动应急切入备用彩条垫片，同时保持下游键控器 (DSK) 台标、广播时钟与抢修游走通告正常在播，并实时生成广播级 As-Run 事件日志",
    "standard": "pal",
    "layers": {
        "display": {"enabled": 0.0},
        "overlay": {
            "playout_enabled": 1.0, "playout_output_bus": 0.0,
            "playout_state": 2.0, "playout_dsk_enabled": 1.0,
            "playout_dsk_crawl": 1.0, "playout_failover_auto": 1.0,
        },
    },
}

CAMCORDER_PRESETS["sony_trv310e"] = CAMCORDER_PRESETS["sony_dcr_trv310e"]
FACTORY_PRESETS["pro_transfer"] = FACTORY_PRESETS["pro_transfer_svhs"]
FACTORY_PRESETS["rental_tape_horror"] = FACTORY_PRESETS["rental_tape_worn"]


# ==================== 三大可重复基准实验与独立胶片光学预设（纯净命名 · 全屏单画面输出） ====================
for _fp_key in (
    "bench1_vision3_500t_std",
    "bench1_vision3_500t_push2",
    "bench1_vision3_500t_pull1",
    "bench1_double_x_5222_bw",
    "bench1_kodachrome_64",
    "bench1_ektachrome_100d",
    "bench2_ocn_direct_scan",
    "bench2_optical_print_gen1",
    "bench2_optical_print_gen2",
    "bench2_release_print_gen3",
    "bench2_flying_spot_telecine",
    "bench3_cold_vault_archive",
    "bench3_ambient_natural_aging",
    "bench3_humid_vinegar_syndrome",
    "bench3_wet_gate_restoration",
    "optics_remjet_removed_halation",
):
    FACTORY_PRESETS[_fp_key] = {
        "label": FILM_PROFILES[_fp_key]["label"],
        "note": FILM_PROFILES[_fp_key]["note"],
        "standard": "ntsc" if _fp_key == "bench2_flying_spot_telecine" else "pal",
        "layers": {
            "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
            "film": dict(FILM_PROFILES[_fp_key]["layers"]["film"]),
        "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {"enabled": 0.0},
        },
    }

FACTORY_PRESETS["lomo_xpro_cyan_gold_marigold"] = {
    "label": "正冲负翡翠暖金",
    "note": FILM_PROFILES["lomo_xpro_e6_in_c41_marigold"]["note"],
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        "film": dict(FILM_PROFILES["lomo_xpro_e6_in_c41_marigold"]["layers"]["film"]),
        "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["lomo_xpro_violet_heart_bokeh"] = {
    "label": "负冲正桃心散景",
    "note": FILM_PROFILES["lomo_xpro_c41_in_e6_heart_bokeh"]["note"],
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        "film": dict(FILM_PROFILES["lomo_xpro_c41_in_e6_heart_bokeh"]["layers"]["film"]),
        "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["lomo_expired_svema_sunflower_fog"] = {
    "label": "过期胶卷高底雾",
    "note": FILM_PROFILES["lomo_expired_svema_sunflower"]["note"],
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 0.95, "halation": 0.0, "ccd_smear": 0.0},
        "film": dict(FILM_PROFILES["lomo_expired_svema_sunflower"]["layers"]["film"]),
        "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["lomo_polaroid_sx70_light_leak"] = {
    "label": "宝丽来即时显影",
    "note": FILM_PROFILES["lomo_polaroid_sx70_qlight"]["note"],
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 0.96, "halation": 0.0, "ccd_smear": 0.0},
        "film": dict(FILM_PROFILES["lomo_polaroid_sx70_qlight"]["layers"]["film"]),
        "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["lomo_trioplan_soap_bubble_dandelion"] = {
    "label": "肥皂泡球差散景",
    "note": FILM_PROFILES["lomo_trioplan_soap_bubble_dandelion"]["note"],
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        "film": dict(FILM_PROFILES["lomo_trioplan_soap_bubble_dandelion"]["layers"]["film"]),
        "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["lomo_golden_hour_solar_flare_flood"] = {
    "label": "黄金逆光阳炎",
    "note": FILM_PROFILES["lomo_golden_hour_solar_flare_flood"]["note"],
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        "film": dict(FILM_PROFILES["lomo_golden_hour_solar_flare_flood"]["layers"]["film"]),
        "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["lomo_highkey_milky_veiling_glare"] = {
    "label": "无镀膜高调柔焦",
    "note": FILM_PROFILES["lomo_highkey_milky_veiling_glare"]["note"],
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 0.96, "halation": 0.0, "ccd_smear": 0.0},
        "film": dict(FILM_PROFILES["lomo_highkey_milky_veiling_glare"]["layers"]["film"]),
        "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["lomo_tilt_shift_candle_flame_aura"] = {
    "label": "移轴双层烛火光冕",
    "note": FILM_PROFILES["lomo_tilt_shift_candle_flame_aura"]["note"],
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.02, "halation": 0.0, "ccd_smear": 0.0},
        "film": dict(FILM_PROFILES["lomo_tilt_shift_candle_flame_aura"]["layers"]["film"]),
        "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["lomo_wong_kar_wai_tungsten_olive"] = {
    "label": "钨丝灯迫冲绿调",
    "note": FILM_PROFILES["lomo_wong_kar_wai_tungsten_olive"]["note"],
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 0.96, "halation": 0.0, "ccd_smear": 0.0},
        "film": dict(FILM_PROFILES["lomo_wong_kar_wai_tungsten_olive"]["layers"]["film"]),
        "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["lomo_prussian_rain_paper_boats"] = {
    "label": "普鲁士蓝雨滴散景",
    "note": FILM_PROFILES["lomo_prussian_rain_paper_boats"]["note"],
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.02, "halation": 0.0, "ccd_smear": 0.0},
        "film": dict(FILM_PROFILES["lomo_prussian_rain_paper_boats"]["layers"]["film"]),
        "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["lomo_macaron_fairy_lights_pastel"] = {
    "label": "六边形彩灯散景",
    "note": FILM_PROFILES["lomo_macaron_fairy_lights_pastel"]["note"],
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 0.98, "halation": 0.0, "ccd_smear": 0.0},
        "film": dict(FILM_PROFILES["lomo_macaron_fairy_lights_pastel"]["layers"]["film"]),
        "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["lomo_petzval_swirl_street_lantern"] = {
    "label": "匹兹伐切向旋焦",
    "note": FILM_PROFILES["lomo_petzval_swirl_street_lantern"]["note"],
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        "film": dict(FILM_PROFILES["lomo_petzval_swirl_street_lantern"]["layers"]["film"]),
        "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["lomo_lca_turquoise_bridge_xpro"] = {
    "label": "广角弯月镜正冲负",
    "note": FILM_PROFILES["lomo_lca_turquoise_bridge_xpro"]["note"],
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        "film": dict(FILM_PROFILES["lomo_lca_turquoise_bridge_xpro"]["layers"]["film"]),
        "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["lomo_optical_halation_bokeh_2x2_lab"] = {
    "label": "胶片光学散景与光晕",
    "note": "独立胶片光学模块单屏全画幅展示：Trioplan 肥皂泡亮环散景 + 斜向紫金双色温光雾 + 抗晕光背散射",
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        "film": dict(FILM_PROFILES["lomo_trioplan_soap_bubble_dandelion"]["layers"]["film"]),
        "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["lomo_photochemical_darkroom_2x2_lab"] = {
    "label": "胶片暗房交叉冲洗",
    "note": "独立胶片光化学暗房单屏全画幅展示：E-6 反转片投入 C-41 药水交叉冲洗 + 弯月镜暗角",
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        "film": dict(FILM_PROFILES["lomo_xpro_e6_in_c41_marigold"]["layers"]["film"]),
        "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
        "display": {"enabled": 0.0},
    },
}

# resolve_preset is defined above; expose it from the package's public names
__all__ = [n for n in dir() if not n.startswith("_")]


# Attach per-unit tolerances (UNIT_SPREADS) to SCHEMA parameter specs
FACTORY_PRESETS["sync_pulse_cross_monitor"] = {
    "label": "同步与时钟系统 · 广播级监视器脉冲十字检视 (Pulse Cross HBI/VBI 波形)",
    "note": "开启广播级监视器 Pulse Cross 模式：将水平消隐 (HBI：前肩、-40 IRE 行同步顶、色同步脉冲串) 与垂直消隐 (VBI：均衡脉冲、场同步开槽、VITC 时间码) 移至画面中央十字，直观观测行 AFC 弯钩与同步脉冲形态",
    "standard": "ntsc",
    "layers": {
        "camera": {"sensor_noise": 0.02, "sharpness": 0.95, "dynamic_range": 8.5, "halation": 0.08, "ccd_smear": 0.0},
        "format": {"luma_bandwidth": 3.2, "chroma_bandwidth": 0.75, "comb_depth": 0.60, "color_noise": 0.12},
        "tape": {"tape_stability": 0.68, "line_jitter": 0.22, "head_switching": 0.70, "head_switch_lines": 6.0, "tracking_error": 0.25},
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.22, "sync_clip": 0.22, "h_hold": 0.12,
                   "clock_drift_ppm": 35.0, "clock_jitter": 0.12, "pulse_cross": 1.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["sync_pal_ntsc_mismatch_roll"] = {
    "label": "同步与时钟系统 · PAL 50Hz 磁带误入 NTSC 59.94Hz 录像机（跨制式滚屏 + VBI 同步带 + 差拍消色）",
    "note": "模拟将 50Hz PAL 录制的磁带放入 59.94Hz NTSC 机器重放：9.94Hz 场差拍纵向滚屏（上下场交界处显现完整的 VBI 均衡/开槽脉冲与 VITC 时间码带）、15625 vs 15734Hz 行频斜切，以及 4.43MHz 副载波失锁消色与斜向彩虹差拍",
    "standard": "ntsc",
    "layers": {
        "camera": {"sensor_noise": 0.03, "sharpness": 0.92, "dynamic_range": 8.0, "halation": 0.10, "ccd_smear": 0.0},
        "format": {"luma_bandwidth": 3.0, "chroma_bandwidth": 0.68, "comb_depth": 0.50, "color_noise": 0.18},
        "tape": {"tape_stability": 0.58, "line_jitter": 0.28, "head_switching": 0.75, "head_switch_lines": 6.0, "tracking_error": 0.30},
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.30, "std_mismatch": 1.0, "sync_clip": 0.30,
                   "h_hold": 0.16, "v_hold": 0.28, "clock_drift_ppm": 95.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["power_dying_battery_camcorder"] = {
    "label": "电源与电气系统 · 老化电池欠压与高负载动态塌陷（红外灯/变焦/走带伺服共母线耦合）",
    "note": "模拟电池电量濒临耗尽 (SoC=0.16) 且内阻升高 (battery_sag=0.72)：开启红外灯/变焦与走带伺服抽取引发母线电压 V_bus 动态塌陷，导致曝光变暗与底噪激增、主导轴伺服掉速（行撕裂+伴音音高下坠）、同步分离门限偏移引发画面顶部弯钩与场抖动",
    "standard": "ntsc",
    "layers": {
        "camera": {"sensor_noise": 0.06, "sharpness": 0.86, "dynamic_range": 7.5, "wt_zoom": 0.45,
                   "zoom_dynamic_mode": 3.0, "zoom_speed": 0.50, "halation": 0.20},
        "format": {"luma_bandwidth": 3.0, "chroma_bandwidth": 0.70, "color_noise": 0.22},
        "connector": {"battery_level": 0.16, "battery_sag": 0.72, "load_coupling": 1.15,
                      "psu_ripple": 0.18, "startup_surge": 0.45, "emi_crosstalk": 0.38},
        "tape": {"tape_stability": 0.52, "line_jitter": 0.32, "stiction_rate": 0.85, "stiction_amount": 0.55,
                 "head_switching": 0.75, "head_switch_lines": 6.0, "tracking_error": 0.32},
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.35, "sync_clip": 0.25, "h_hold": 0.14},
        "display": {"enabled": 1.0, "scanlines": 0.30, "bloom": 0.28, "breathing": 0.35},
    },
}
FACTORY_PRESETS["power_dried_capacitor_psu_hum"] = {
    "label": "电源与电气系统 · 线性电源滤波电容干涸（100Hz 全波纹波蛇形扭曲 + 冷机上电浪涌 + 电机 EMI）",
    "note": "模拟老式录像机/监视器电源滤波电解电容干涸 (ESR 剧增)：100Hz 全波整流充放电锯齿波与 50Hz 半波分量同时调制画面黑电平与行 AFC 锁相环，在垂直边缘形成标志性的 S 形交流声扭曲 (Hum Snake) 与滚动哼声带，伴随电机换向火花纹与伴音 100Hz 蜂鸣",
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.04, "sharpness": 0.90, "dynamic_range": 8.0},
        "format": {"luma_bandwidth": 3.0, "chroma_bandwidth": 0.68, "color_noise": 0.20},
        "connector": {"av_present": 1.0, "av_ground_loop": 0.28, "mains_hz": 50.0,
                      "psu_ripple": 0.65, "battery_level": 0.85, "battery_sag": 0.25,
                      "load_coupling": 0.85, "startup_surge": 0.60, "emi_crosstalk": 0.45},
        "tape": {"tape_stability": 0.60, "line_jitter": 0.25, "head_switching": 0.70, "head_switch_lines": 6.0},
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.28, "sync_clip": 0.18, "h_hold": 0.08},
        "display": {"enabled": 0.0},
    },
}

FACTORY_PRESETS["crt_reshoot_shutter_roll_125"] = {
    "label": "虚拟相机屏摄系统 · 1/125s 高速快门重拍 CRT（扫描亮带下滚 + P22 余辉拖尾）",
    "note": "把 VHS 信号送入虚拟球面 CRT 后，再用 1/125s 快门的虚拟 CMOS 相机重拍：单次快门仅积分约半场扫描周期，形成前缘炽亮、后缘带 P22 红粉慢余辉的明暗滚动扫描横带",
    "standard": "ntsc",
    "layers": {
        "camera": {"sensor_noise": 0.04, "sharpness": 0.90, "dynamic_range": 8.0, "halation": 0.15},
        "format": {"luma_bandwidth": 3.1, "chroma_bandwidth": 0.72, "comb_depth": 0.55, "color_noise": 0.18},
        "tape": {"tape_stability": 0.65, "line_jitter": 0.22, "head_switching": 0.75, "head_switch_lines": 8.0, "tracking_error": 0.25},
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.25},
        "display": dict(DISPLAY_PROFILES["crt_reshoot_fast_shutter"]["layers"]["display"]),
    },
}
FACTORY_PRESETS["crt_reshoot_rainbow_moire_offaxis"] = {
    "label": "虚拟相机屏摄系统 · 离轴斜拍特丽珑/槽孔 CRT（二维彩色水波摩尔纹 + 透视外框）",
    "note": "模拟相机带 2.8° 滚转倾角与离轴透视近距离拍摄球面 CRT：相机拜耳像素阵列与球面 RGB 荧光粉荫罩/扫描线发生空间混叠，浮现出真实的彩色环形水波摩尔纹与远端浅景深",
    "standard": "ntsc",
    "layers": {
        "camera": {"sensor_noise": 0.03, "sharpness": 0.94, "dynamic_range": 8.5, "halation": 0.12},
        "format": {"luma_bandwidth": 3.4, "chroma_bandwidth": 0.78, "comb_depth": 0.60, "color_noise": 0.14},
        "tape": {"tape_stability": 0.72, "line_jitter": 0.18, "head_switching": 0.65, "head_switch_lines": 6.0},
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.20},
        "display": dict(DISPLAY_PROFILES["crt_reshoot_angled_moire"]["layers"]["display"]),
    },
}
FACTORY_PRESETS["crt_reshoot_living_room_offscreen"] = {
    "label": "虚拟相机屏摄系统 · 开灯客厅手持翻拍 9300K 彩电（环境光漫反冲淡 + 冷青过曝 + 双扫亮带）",
    "note": "模拟在开灯房间内用 1/50s 快门相机手持翻拍 59.94Hz 9300K 家用彩电：约 20% 区域被电子束二次扫过形成滚动亮带，未发光荧光粉层漫反射室内光冲淡黑位，伴随曲面玻璃窗户倒影与 9300K 冷青过曝",
    "standard": "ntsc",
    "layers": {
        "camera": {"sensor_noise": 0.05, "sharpness": 0.88, "dynamic_range": 7.8, "shake_intensity": 0.35},
        "format": {"luma_bandwidth": 3.0, "chroma_bandwidth": 0.70, "comb_depth": 0.50, "color_noise": 0.22},
        "tape": {"tape_stability": 0.60, "line_jitter": 0.25, "head_switching": 0.80, "head_switch_lines": 10.0, "tracking_error": 0.30},
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.30},
        "display": dict(DISPLAY_PROFILES["crt_reshoot_living_room_cam"]["layers"]["display"]),
    },
}


MEDIA_PROFILES["vhs_bot_eot_tension"] = {
    "label": "VHS 带尾高危区 (EOT 供带盘近卷芯 · 高角速度张力脉动 + 边缘卷边)",
    "note": "模拟磁带播放至接近带尾 (reel_progress=0.94)：供带盘半径缩至 ~15mm、转速飙升 3 倍，背张力偏低导致画面顶部向左斜拉、底部导带柱边缘褶皱失落与行频抖晃加剧",
    "layers": {
        "tape": {
            "reel_progress": 0.94, "tension_ripple": 0.78,
            "tape_stability": 0.45, "line_jitter": 0.38,
            "head_switching": 0.85, "head_switch_lines": 10.0, "tracking_error": 0.38,
        },
    },
}

PLAYER_PROFILES["deck_aged_belt_roller"] = {
    "label": "皮带老化 + 压带轮硬化偏心的旧录像机",
    "note": "方形氯丁橡胶皮带松弛打滑 + 压带轮长期压迫偏心 (~1.45Hz S形行摆动与伴音低频抖晃) + 主导轴伺服 3.2Hz 欠阻尼猎振",
    "layers": {
        "tape": {
            "belt_slip": 0.68, "roller_eccentricity": 0.72,
            "servo_hunt": 0.52, "tension_ripple": 0.45,
        },
        "player": {
            "tbc_enabled": 0.0, "transport_jitter": 0.48, "head_wear": 0.28,
        },
    },
}

FACTORY_PRESETS["transport_pause_still_bar"] = {
    "label": "机械传动与伺服 · 静态暂停定格 (Pause/Still 跨轨方位角双峰噪波横带 + 场微颤)",
    "note": "复现家用 VHS 录像机按下 Pause 键后的定格画面：主导轴停转（v_tape=0），旋转磁鼓以 5°56' 静止倾角斜跨在两条反方位角磁迹之间，在画面中下部产生一条带 H-AFC 钩状剪切与色度消隐的固定双峰雪花横条，伴随单场重复微颤与伴音静音",
    "standard": "ntsc",
    "layers": {
        "camera": {"sensor_noise": 0.04, "sharpness": 0.90, "dynamic_range": 8.0},
        "format": {"luma_bandwidth": 3.0, "chroma_bandwidth": 0.70, "comb_depth": 0.55, "color_noise": 0.18},
        "tape": {
            "transport_state": 1.0, "pause_noise_pos": 0.62, "reel_progress": 0.35,
            "tension_ripple": 0.28, "tape_stability": 0.62, "line_jitter": 0.24,
            "head_switching": 0.80, "head_switch_lines": 8.0, "tracking_error": 0.30,
        },
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.25, "head_wear": 0.20},
        "display": {"enabled": 0.0},
    },
}

FACTORY_PRESETS["transport_rec_pause_rainbow_cut"] = {
    "label": "机械传动与伺服 · 无飞抹磁头暂停起录切口 (Record-Pause 彩虹叠印切口 + CTL 失步跳帧)",
    "note": "复现无飞抹磁头 (Flying Erase Head) 的消费级摄像机/录像机暂停后再录制的分镜接缝：全宽固定抹音磁头与旋转视频磁鼓相距数厘米，起录点处新旧螺旋磁迹斜向重叠产生 180° 色度副载波彩虹干涉条与旧画面残影，同时 25/30Hz CTL 控制脉冲相位断裂引发纵向跳帧与伺服猎振",
    "standard": "ntsc",
    "layers": {
        "camera": {"sensor_noise": 0.04, "sharpness": 0.90, "dynamic_range": 8.0},
        "format": {"luma_bandwidth": 3.0, "chroma_bandwidth": 0.68, "comb_depth": 0.50, "color_noise": 0.22},
        "tape": {
            "transport_state": 3.0, "servo_lock_tau": 0.55, "reel_progress": 0.20,
            "tension_ripple": 0.45, "servo_hunt": 0.42, "tape_stability": 0.55,
            "line_jitter": 0.30, "head_switching": 0.85, "head_switch_lines": 10.0,
        },
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.32, "head_wear": 0.22},
        "display": {"enabled": 0.0},
    },
}

FACTORY_PRESETS["switcher_nongenlock_hot_cut"] = {
    "label": "模拟切换台 · 非同步信号源硬切瞬态（无锁相场跳动 + VBI 消隐带 + H-AFC 撕裂）",
    "note": "两路未经过帧同步器 (Non-Genlocked) 的复合视频源在切换台硬切瞬间，下游录像机/电视机失去连续同步基准，爆发场中切口、H-AFC 弯钩撕裂、V-Hold 场跳动滚屏（露出 VBI 消隐带）与色同步副载波相位突变",
    "standard": "ntsc",
    "layers": {
        "camera": {"sharpness": 0.95, "sensor_noise": 0.03},
        "format": {"luma_bandwidth": 3.2, "chroma_bandwidth": 0.72, "comb_depth": 0.55},
        "connector": {
            "switcher_mode": 1.0, "switcher_b_source": 0.0, "switcher_tbar": 0.42,
            "switcher_hot_cut_rate": 1.50, "switcher_genlock_err": 0.85,
            "switcher_subcarrier_spill": 0.55, "switcher_pedestal_add": 0.25,
        },
        "tape": {"tape_stability": 0.65, "line_jitter": 0.22, "head_switching": 0.75, "head_switch_lines": 10.0},
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.28},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["switcher_tbar_wipe_seg"] = {
    "label": "模拟特技切换台 · 斜坡发生器菱形划像与比较器彩边",
    "note": "使用模拟行/场锯齿波发生器生成菱形光圈划像 (Diamond Iris Wipe) 并支持完全自定义叠层（可导入自定义叠层图/视频，或在「02c 接口/RF」层自由调节叠层样式、文字、位置、缩放与色彩），伴随比较器行触发抖动与 3.58MHz 切换瞬态彩色镶边",
    "standard": "ntsc",
    "layers": {
        "camera": {"sharpness": 0.96, "sensor_noise": 0.02},
        "format": {"luma_bandwidth": 3.4, "chroma_bandwidth": 0.75, "comb_depth": 0.60},
        "connector": {
            "switcher_mode": 3.0, "switcher_b_source": 0.0, "switcher_tbar": 0.52,
            "switcher_wipe_pattern": 3.0, "switcher_key_gain": 0.72,
            "switcher_key_jitter": 0.48, "switcher_subcarrier_spill": 0.68,
            "switcher_overlay_style": 4.0, "switcher_overlay_text": "",
            "switcher_overlay_x": 0.50, "switcher_overlay_y": 0.50,
            "switcher_overlay_scale": 1.00, "switcher_overlay_opacity": 0.95,
        },
        "tape": {"tape_stability": 0.78, "line_jitter": 0.14, "head_switching": 0.55, "head_switch_lines": 8.0},
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.16},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["switcher_analog_chroma_luma_key"] = {
    "label": "模拟演播室色键/亮键自定义叠层合成",
    "note": "模拟 80/90 年代复合视频色键/亮键器 (Chroma/Luma Keyer) 并支持完全自定义前景叠层（可通过顶部「🖼 导入叠层」上传任意图片/视频，或在「02c 接口/RF」层自定义画中画小窗、字幕条、文字、XY坐标、缩放与颜色），展现 700ns 色度群延迟错位黑边与 3.58MHz 副载波拉链锯齿",
    "standard": "ntsc",
    "layers": {
        "camera": {"sharpness": 0.96, "sensor_noise": 0.02},
        "format": {"luma_bandwidth": 3.4, "chroma_bandwidth": 0.72, "comb_depth": 0.55},
        "connector": {
            "switcher_mode": 5.0, "switcher_b_source": 2.0, "switcher_tbar": 1.0,
            "switcher_key_clip": 0.46, "switcher_key_gain": 0.78,
            "switcher_key_jitter": 0.42, "switcher_subcarrier_spill": 0.75,
            "switcher_pedestal_add": 0.28,
            "switcher_overlay_style": 3.0, "switcher_overlay_text": "STUDIO B · LIVE",
            "switcher_overlay_x": 0.66, "switcher_overlay_y": 0.36,
            "switcher_overlay_scale": 0.95, "switcher_overlay_opacity": 0.94,
            "switcher_overlay_r": 0.32, "switcher_overlay_g": 0.86, "switcher_overlay_b": 0.96,
        },
        "tape": {"tape_stability": 0.80, "line_jitter": 0.12, "head_switching": 0.45, "head_switch_lines": 8.0},
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.14},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["transport_worn_belt_roller_eot"] = {
    "label": "机械传动与伺服 · 带尾高张力 + 老化皮带打滑与压带轮偏心 (EOT + Belt Slip + Roller Wow)",
    "note": "综合演示三大机械传动故障：磁带行至带尾 (reel_progress=0.92) 供带盘高速旋转引发顶部斜拉与底部卷边失落 + 方形橡胶皮带间歇打滑掉速 + 硬化压带轮 1.45Hz 偏心 S 形摆动与主导轴伺服 3.25Hz 猎振（音画同步变调抖晃）",
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.05, "sharpness": 0.88, "dynamic_range": 7.8},
        "format": {"luma_bandwidth": 2.9, "chroma_bandwidth": 0.65, "comb_depth": 0.55, "color_noise": 0.24},
        "tape": {
            "transport_state": 0.0, "reel_progress": 0.92, "tension_ripple": 0.82,
            "belt_slip": 0.72, "roller_eccentricity": 0.75, "servo_hunt": 0.58,
            "tape_stability": 0.42, "line_jitter": 0.36, "head_switching": 0.88,
            "head_switch_lines": 10.0, "tracking_error": 0.42,
        },
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.42, "head_wear": 0.28},
        "display": {"enabled": 0.0},
        "audio": {"enabled": 1.0, "mode": 1.0, "wow_flutter": 0.55},
    },
}

FACTORY_PRESETS["broadcast_unterminated_highz"] = {
    "label": "广播级配线与负载 · 未端接高阻开路环通 (High-Z Open · +6dB 翻倍过曝削波 + 正向反射重影)",
    "note": "模拟监视器或录像机环通口 (Loop-Through) 忘记拨下 75Ω 端接开关（R_L≈10kΩ 高阻开路）：负载端失去 75Ω 分压，信号电压翻倍至 ~2.0 Vpp (+5.95 dB)，导致中高亮区域直接撞击 100 IRE 满度硬削波死白、色同步与色度过冲，且电缆末端全反射 (Γ=+0.98) 在硬边缘右侧形成正向重影",
    "standard": "ntsc",
    "layers": {
        "camera": {"sharpness": 0.96, "sensor_noise": 0.02},
        "format": {"luma_bandwidth": 3.6, "chroma_bandwidth": 0.80, "comb_depth": 0.65},
        "connector": {
            "da_enabled": 1.0, "da_termination_mode": 1.0, "da_load_count": 1.0,
            "da_cable_len_m": 28.0, "da_gain_db": 0.0, "da_eq_boost": 0.0,
            "patch_crosstalk": 0.08, "patch_contact_aging": 0.05, "patch_format_mismatch": 0.0,
        },
        "tape": {"tape_stability": 0.85, "line_jitter": 0.10, "head_switching": 0.35, "head_switch_lines": 6.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 0.85, "transport_jitter": 0.10},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["broadcast_passive_tsplit_double_term"] = {
    "label": "广播级配线与负载 · 违规三通无源并接/三端接 (Passive T-Split 25Ω · -6dB 发暗 + 同步顶腰斩失锁 + 负向重影)",
    "note": "模拟不用有源分配放大器，直接用 BNC 三通头将一路复合信号硬分给 N=3 台均开启 75Ω 端接的设备（等效负载跌至 R_L=25Ω）：分压使画面幅度腰斩至 0.50x (-6.02 dB)、暗淡发灰，同时 -40 IRE 行同步顶被压扁至 -20 IRE 引发下游同步分离器饥饿抖动与顶部弯钩，且负反射系数 (Γ=-0.50) 在亮边缘右侧刻出暗黑负向重影",
    "standard": "ntsc",
    "layers": {
        "camera": {"sharpness": 0.94, "sensor_noise": 0.02},
        "format": {"luma_bandwidth": 3.4, "chroma_bandwidth": 0.75, "comb_depth": 0.60},
        "connector": {
            "da_enabled": 1.0, "da_termination_mode": 2.0, "da_load_count": 3.0,
            "da_cable_len_m": 32.0, "da_gain_db": 0.0, "da_eq_boost": 0.0,
            "patch_crosstalk": 0.12, "patch_contact_aging": 0.10, "patch_format_mismatch": 0.0,
        },
        "tape": {"tape_stability": 0.75, "line_jitter": 0.16, "head_switching": 0.55, "head_switch_lines": 8.0},
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.22},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["broadcast_patchbay_active_da"] = {
    "label": "广播级配线与负载 · 有源分配放大器 1进4出隔离驱动 + 电缆均衡补偿 + 矩阵串扰 (Active DA + Patchbay)",
    "note": "使用广播级 1×4 有源分配放大器 (Active DA, Mode 3) 隔离驱动 4 路 75Ω 终端（电平精准保持 1.0 Vpp / 0 dB、零阻抗反射），开启长同轴电缆高频均衡补偿 (da_eq_boost=0.55)，并叠加高密度跳线配线架相邻通道的微分电容串扰",
    "standard": "ntsc",
    "layers": {
        "camera": {"sharpness": 0.98, "sensor_noise": 0.02},
        "format": {"luma_bandwidth": 4.2, "chroma_bandwidth": 0.95, "comb_depth": 0.75},
        "connector": {
            "da_enabled": 1.0, "da_termination_mode": 3.0, "da_load_count": 4.0,
            "da_cable_len_m": 45.0, "da_gain_db": 0.2, "da_eq_boost": 0.55,
            "patch_crosstalk": 0.28, "patch_contact_aging": 0.08, "patch_format_mismatch": 0.0,
        },
        "tape": {"tape_stability": 0.88, "line_jitter": 0.08, "head_switching": 0.30, "head_switch_lines": 6.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 0.90, "transport_jitter": 0.08},
        "display": {"enabled": 0.0},
    },
}


FACTORY_PRESETS["transfer_replication_comparison_lab"] = {
    "label": "四路复制链对照实验室 (母带 vs 无TBC多代复制 vs 带TBC多代复制 vs 4:1:1→4:2:0数字转码)",
    "note": "将同一段源素材同时送入四条复制/转录链路进行 2×2 实验室同屏对照：左上 Gen-0 母带基准、右上无 TBC 三代磁带复制（严重水平行抖动累积）、左下带全帧 TBC 三代磁带复制（几何稳定但带宽/色度/底噪不可逆退化）、右下 4:2:2→DV25(4:1:1)→VCD(4:2:0) 跨代数字转码（零时基抖动但呈现 1/8 色度抽样台阶与 8×8 DCT 块伪影），底部实时显示 PSNR/带宽/抖动与 TransferEvent 溯源链",
    "standard": "ntsc",
    "layers": {
        "camera": {"sharpness": 0.98, "sensor_noise": 0.02},
        "format": {"luma_bandwidth": 4.2, "chroma_bandwidth": 0.95, "comb_depth": 0.72},
        "tape": {"tape_stability": 0.85, "line_jitter": 0.08, "head_switching": 0.35, "head_switch_lines": 6.0},
        "player": {
            "tbc_enabled": 1.0, "tbc_strength": 0.90,
            "transfer_station_mode": 4.0, "transfer_generations": 3.0,
            "transfer_tbc_mode": 2.0, "transfer_codec_target": 3.0,
            "transfer_adc_crop": 0.45, "transfer_av_drift_ms": 35.0,
        },
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["transfer_analog_ingest_station"] = {
    "label": "模拟磁带数字化采集工作站 (VHS/Video8 → 行TBC → BT.601 8-bit 采集卡 + 音画时钟漂移记录)",
    "note": "模拟磁带经播放机与行 TBC 进入视频采集卡数字化：采集卡不凭空修复母带原有磁粉失落与底部磁头切换带，同时引入 BT.601 16..235 量化范围钳位、左右消隐裁切振铃边与独立声卡晶振的音画同步漂移 (TransferEvent)",
    "standard": "pal",
    "layers": {
        "camera": {"sharpness": 0.92, "sensor_noise": 0.03},
        "format": {"luma_bandwidth": 3.1, "chroma_bandwidth": 0.65, "comb_depth": 0.65},
        "tape": {"tape_stability": 0.65, "line_jitter": 0.24, "dropout_rate": 0.45, "head_switching": 0.82, "head_switch_lines": 8.0},
        "player": {
            "tbc_enabled": 1.0, "tbc_strength": 0.85,
            "transfer_station_mode": 1.0, "transfer_generations": 1.0,
            "transfer_tbc_mode": 1.0, "transfer_adc_crop": 0.60, "transfer_av_drift_ms": 42.0,
        },
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["linear_editor_assemble_vs_insert"] = {
    "label": "Sony BVE-9100 磁带双机线性编辑台 (5s 预卷伺服 + CTL 入点滑移 + 组合编辑闪黑与尾端断轨)",
    "note": "模拟源录像机 (Player VTR) 与录制机 (Recorder VTR) 双机预卷线性剪辑：包含压带轮啮合造成的 ±2 帧入出点机械滑移、Assemble Edit 切入点的全宽抹音头闪黑与 H-AFC 弯钩、切出点新旧 CTL 控制轨断裂雪花撕裂，以及顶部 BVE-9100 工程状态栏",
    "standard": "ntsc",
    "layers": {
        "camera": {"sharpness": 0.94, "sensor_noise": 0.02},
        "format": {"luma_bandwidth": 3.4, "chroma_bandwidth": 0.72, "comb_depth": 0.65},
        "tape": {
            "tape_stability": 0.75, "line_jitter": 0.14, "head_switching": 0.65, "head_switch_lines": 8.0,
            "edit_mode": 1.0, "edit_point_slip_frames": 2.0,
            "edit_flying_erase": 1.0, "edit_preroll_sec": 5.0, "edit_console_osd": 1.0,
        },
        "player": {"tbc_enabled": 1.0, "tbc_strength": 0.85},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["early_2000s_nle_preview_vs_export"] = {
    "label": "2000年代初 DV 非线性编辑工作站 (多轨时间线 · 实时预览缓存掉帧代理 vs 最终导出二次重编码)",
    "note": "模拟 Windows 2000 / Premiere 6.0 / Canopus DV Storm 早期非线编系统：时间线寻址零帧偏移，但在实时预览 (edit_mode=3) 下多轨叠化转场与字幕触发半分辨率代理降级与缓冲区欠载掉帧重复；切换至 edit_mode=4（最终渲染导出）即可消除掉帧并观察转场段的二次 DV25 DCT 重编码伪影",
    "standard": "ntsc",
    "layers": {
        "camera": {"sharpness": 0.98, "sensor_noise": 0.02},
        "format": {"luma_bandwidth": 4.8, "chroma_bandwidth": 1.5},
        "codec": {"dv_enabled": 1.0, "dv_quality": 0.95},
        "tape": {
            "tape_stability": 0.95, "line_jitter": 0.02, "head_switching": 0.0,
            "edit_mode": 3.0, "edit_point_slip_frames": 0.0,
            "edit_nle_complexity": 0.75, "edit_console_osd": 1.0,
        },
        "player": {"tbc_enabled": 1.0, "tbc_strength": 0.95},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["playout_automation_v2_failover"] = {
    "label": "事件与条件驱动播出自动化编排控制台 (PlayoutEvent 状态机 · 6类异常分支 · 根因反查链)",
    "note": "展示总控播出自动化编排引擎 v2 的 2×2 控制台视图，包含 SCHEDULED→PREPARING→READY→ON_AIR→COMPLETED 状态机、VTR 预卷伺服与看门狗联动、主备冗余无缝倒换以及「物理根因→节点传播→画面现象」三级反查树",
    "standard": "pal",
    "layers": {
        "camera": {"sharpness": 0.98, "sensor_noise": 0.01},
        "format": {"luma_bandwidth": 4.5, "chroma_bandwidth": 1.2},
        "overlay": {
            "auto_v2_enabled": 1.0, "auto_v2_mode": 3.0,
            "auto_v2_fault_inject": 3.0, "auto_v2_watchdog_sec": 1.0,
            "auto_v2_console_view": 1.0,
        },
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["satellite_cband_vs_dvbs_cliff_lab"] = {
    "label": "卫星接收机 (Satellite IRD) 模拟 C 波段 FM 打火噪点 vs 数字 DVB-S 悬崖效应对照实验室",
    "note": "在同一套卫星抛物面天线与雨衰/LNB 链路上，同屏对比模拟 C 波段 FM 卫星电视（低于门限时渐进出现鱼骨黑白打火噪点 Sparkles）与数字 DVB-S QPSK+Viterbi+RS（高于 6.5dB 完全无雪花，跌入 4.4–6.5dB 悬崖区瞬间爆发 MPEG-2 宏块崩坏与静帧冻结）",
    "standard": "pal",
    "layers": {
        "camera": {"sharpness": 0.98, "sensor_noise": 0.01, "ccd_smear": 0.0, "halation": 0.05},
        "format": {"luma_bandwidth": 5.0, "chroma_bandwidth": 1.5, "chroma_vertical_averaging": 0.0, "comb_depth": 0.92, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "rf": {
            "sat_ird_mode": 3.0, "sat_cn_db": 9.8,
            "sat_rain_fade_db": 4.2, "sat_dish_mispoint_deg": 0.25,
            "sat_lnb_drift_khz": 45.0, "sat_ird_osd": 0.0,
            "sat_quad_corner": 0.0, "sat_show_border": 2.0,
        },
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "vertical_instability": 0.0, "head_switching": 0.0, "dropout_rate": 0.0, "wear": 0.0, "crease_rate": 0.0, "noise_floor": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "tbc_residual": 0.0, "transport_jitter": 0.0, "head_wear": 0.0, "frequency_response": 0.0, "comb_depth": 0.92, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "display": {"enabled": 0.0},
    },
}
FACTORY_PRESETS["tv_transmitter_end_to_end_broadcast_lab"] = {
    "label": "电视台大功率发射机激励链与端到端开路广播测试台 (VSB+HPA 微分增益/相位 vs 地面接收)",
    "note": "构建「主控播出输出 → 发射机激励器/高功放/双工器/铁塔馈线 → 开路传播信道 → 模拟/数字接收机」端到端四象限对照测试台，实时量化高功放 AM-AM 微分增益 (DG%)、AM-PM 微分相位 (DP°)、滤波器群延迟 (ns)、920kHz 伴音互调与信道多径各自的独立失真贡献",
    "standard": "pal",
    "layers": {
        "camera": {"sharpness": 0.98, "sensor_noise": 0.01, "ccd_smear": 0.0, "halation": 0.05},
        "format": {"luma_bandwidth": 5.0, "chroma_bandwidth": 1.5, "chroma_vertical_averaging": 0.0, "comb_depth": 0.92, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "rf": {
            "tx_chain_mode": 3.0, "tx_pa_drive": 0.65,
            "tx_pa_am_pm_deg": 14.0, "tx_filter_gd_ns": 95.0,
            "tx_feeder_vswr": 1.35, "tx_sound_intermod": 0.38,
            "signal_dbm": -62.0, "multipath_level": 0.16,
            "tx_quad_corner": 0.0, "tx_show_border": 2.0, "tx_show_text": 0.0,
        },
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "vertical_instability": 0.0, "head_switching": 0.0, "dropout_rate": 0.0, "wear": 0.0, "crease_rate": 0.0, "noise_floor": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "tbc_residual": 0.0, "transport_jitter": 0.0, "head_wear": 0.0, "frequency_response": 0.0, "comb_depth": 0.92, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "display": {"enabled": 0.0},
    },
}

UNIT_SPREADS.update({
    "wow_flutter": (0.22, "capstan belt eccentricity and flywheel bearing wear vary per deck"),
    "azimuth_loss": (0.28, "audio head azimuth screw alignment tolerance"),
    "hiss_floor": (0.15, "replay head preamplifier noise figure tolerance"),
    "head_switch_buzz": (0.12, "audio head shield and switching transient tolerance"),
    "clock_drift_ppm": (0.25, "quartz crystal cut and load-capacitor ageing tolerance"),
    "psu_ripple": (0.30, "electrolytic filter capacitor ESR and capacitance ageing"),
    "h_hold": (0.18, "horizontal RC oscillator timing capacitor trim"),
    "v_hold": (0.18, "vertical hold potentiometer and RC timing drift"),
    "switcher_key_jitter": (0.18, "analog comparator reference ripple and ramp capacitor tolerance vary per switcher"),
    "switcher_subcarrier_spill": (0.16, "subcarrier trap notch tuning and delay-line matching vary per unit"),
    "switcher_pedestal_add": (0.14, "back-porch clamp diode offset varies per mixer bus"),
    "da_gain_db": (0.12, "distribution amplifier feedback resistor tolerance varies per channel"),
    "patch_crosstalk": (0.20, "jackfield normalled spring capacitance varies per bay"),
    "transfer_av_drift_ms": (0.22, "audio and video capture crystal oscillator tolerance varies per workstation"),
    "edit_preroll_sec": (0.12, "capstan and drum servo preroll lockup settling time varies per deck pair"),
    "sat_lnb_drift_khz": (0.25, "DRO local oscillator dielectric resonator temperature drift per LNB unit"),
    "tx_pa_am_pm_deg": (0.15, "high-power amplifier tube/LDMOS bias and phase compensation trim"),
    "tx_feeder_vswr": (0.08, "antenna feeder flange and dehydrator impedance tolerance"),
})
FACTORY_PRESETS["archival_lineage_4way_lab"] = {
    "label": "2×2 档案保存生命周期与多版本并存谱系树实验室 (Original / Preservation / Over-Cleaned / Web Stream)",
    "note": "同屏对比同一作品的『1.原始载体残件 (温湿度阿伦尼乌斯加速老化) → 2.档案保存母版 (全画幅无损含VBI/VITC，重要属性保留率98%+) → 3.过度清洗商业修复版 (3D-DNR塑料感与裁切导致考古证据链断裂) → 4.流媒体传播版与重要属性损耗表』",
    "standard": "ntsc",
    "layers": {
        "format": {"luma_bandwidth": 3.0, "chroma_bandwidth": 0.68, "edge_boost": 0.25},
        "tape": {"tape_stability": 0.65, "line_jitter": 0.25, "head_switching": 0.78, "head_switch_lines": 10.0, "dropout_rate": 0.30, "noise_floor": 0.16},
        "player": {
            "tbc_enabled": 1.0, "tbc_strength": 0.85,
            "archival_enabled": 1.0, "archival_mode": 4.0,
            "archival_storage_years": 30.0, "archival_temp_c": 25.0, "archival_rh_pct": 66.0,
            "archival_obsolescence": 0.58, "archival_dnr_overrestore": 0.78,
        },
        "display": {"enabled": 0.0},
    },
}

FACTORY_PRESETS["archival_overrestored_vs_preservation"] = {
    "label": "『保存成功 ≠ 信息完整』过度清洗商业修复版检视 (3D-DNR 塑料蜡像感 + 消隐裁切 + 场序破坏)",
    "note": "演示激进数字去噪、假锐化光环、色彩重映射与过扫描裁切如何抹除原始 VITC 时间码、底部磁头切换带与隔行运动结构，导致媒介考古证据链断裂",
    "standard": "ntsc",
    "layers": {
        "format": {"luma_bandwidth": 3.1, "chroma_bandwidth": 0.70},
        "tape": {"line_jitter": 0.20, "head_switching": 0.80, "head_switch_lines": 10.0, "noise_floor": 0.15},
        "player": {
            "tbc_enabled": 1.0,
            "archival_enabled": 1.0, "archival_mode": 3.0,
            "archival_storage_years": 25.0, "archival_temp_c": 22.0, "archival_rh_pct": 58.0,
            "archival_dnr_overrestore": 0.85, "archival_show_text": 2.0,
        },
        "display": {"enabled": 0.0},
    },
}

FACTORY_PRESETS["spatial_env_4way_coupling_lab"] = {
    "label": "2×2 二维空间物理环境场与多维耦合实验室 (温湿度露点场 / 电磁干扰场 / 声学光学场 / 局域坐标遥测)",
    "note": "构建连续二维机房空间场 EnvironmentField(x, y, t)：摄像机、录像机与 CRT 按各自空间坐标 (x, y) 独立感知局域温湿度结露、变压器近场电磁干扰、窗户混合色温与房间混响 RT60",
    "standard": "ntsc",
    "layers": {
        "camera": {"sharpness": 0.95, "sensor_noise": 0.04},
        "format": {"luma_bandwidth": 3.4, "chroma_bandwidth": 0.80},
        "connector": {
            "env_field_enabled": 1.0, "env_field_mode": 5.0,
            "env_room_temp_c": 25.0, "env_room_rh_pct": 76.0, "env_device_init_temp_c": 5.0,
            "env_emi_source_power": 0.75, "env_shielding_eff": 0.35,
            "env_light_mix": 0.55, "env_acoustic_rt60": 0.65,
            "env_cam_x": 0.20, "env_cam_y": 0.28,
            "env_vtr_x": 0.78, "env_vtr_y": 0.74,
            "env_crt_x": 0.82, "env_crt_y": 0.25,
        },
        "tape": {"line_jitter": 0.12, "noise_floor": 0.08},
        "player": {"tbc_enabled": 1.0},
        "display": {"enabled": 0.0},
    },
}

FACTORY_PRESETS["spatial_cold_camcorder_dew_condensation"] = {
    "label": "冷机抱入潮湿暖房瞬态结露与变压器近场电磁干扰实况 (Dew Condensation & Near-Field EMI)",
    "note": "模拟 4°C 冰冷摄像机与录像机刚抱入 26°C / 82% RH 潮湿房间并靠近大功率电源柜：镜片瞬间结露起雾、磁鼓水膜粘带撕裂与工频电磁驻波滚道（随时间升温自然消退）",
    "standard": "ntsc",
    "layers": {
        "connector": {
            "env_field_enabled": 1.0, "env_field_mode": 4.0,
            "env_room_temp_c": 26.0, "env_room_rh_pct": 82.0, "env_device_init_temp_c": 4.0,
            "env_emi_source_power": 0.80, "env_shielding_eff": 0.30,
            "env_light_mix": 0.45, "env_acoustic_rt60": 0.72,
            "env_cam_x": 0.75, "env_cam_y": 0.80,
            "env_vtr_x": 0.82, "env_vtr_y": 0.78,
            "env_crt_x": 0.30, "env_crt_y": 0.25,
            "env_show_text": 2.0,
        },
        "display": {"enabled": 0.0},
    },
}

UNIT_SPREADS.update({
    "archival_obsolescence": (0.18, "remaining head life and alignment skill vary across archival transfer houses"),
    "env_shielding_eff": (0.15, "chassis gasket contact and braid ground impedance vary per unit"),
    "env_device_init_temp_c": (0.12, "thermal mass and enclosure ventilation vary per unit"),
})

FACTORY_PRESETS["network_streaming_era_lab"] = {
    "label": "2×2 网络视听传播、平台多代转码与流媒体协议对照实验室 (RealVideo / Web2.0 / HLS ABR)",
    "note": "同屏对比：采集源/TBC母本 vs 1999 拨号 RealVideo G2 强涂抹抽帧 vs 2007 Web 2.0 平台多代转码与双水印 vs 现代 HLS/DASH ABR 分辨率阶跃与 UDP 丢包切片撕裂",
    "standard": "ntsc",
    "layers": {
        "format": {"luma_bandwidth": 3.2, "chroma_bandwidth": 0.72},
        "codec": {
            "stream_enabled": 1.0, "stream_mode": 5.0,
            "stream_bitrate_kbps": 320.0, "stream_bw_fluctuation": 0.55,
            "stream_packet_loss": 0.10, "stream_gop_len": 12.0,
            "stream_platform_generations": 3.0,
            "stream_quad_corner": 0.0, "stream_show_border": 2.0, "stream_show_text": 0.0,
        },
        "tape": {"line_jitter": 0.15, "head_switching": 0.55, "head_switch_lines": 8.0, "dropout_rate": 0.12},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 0.85},
        "display": {"enabled": 0.0},
    },
}

FACTORY_PRESETS["perception_and_operator_closed_loop_lab"] = {
    "label": "2×2 人类视觉感知 (CSF) 模型与机房操作者行为闭环实验室",
    "note": "同屏对比：未干预物理故障画面 vs 观众视网膜 Mannos-Sakrison CSF 空间滤波感知画面 vs 机房值班员 PID 手动跟踪/CCU 配平闭环画面 vs JND 感知显著度与操作轨迹面板",
    "standard": "ntsc",
    "layers": {
        "format": {"luma_bandwidth": 3.0, "chroma_bandwidth": 0.65},
        "tape": {"line_jitter": 0.35, "tracking_error": 0.58, "head_switching": 0.85, "noise_floor": 0.22},
        "player": {"tbc_enabled": 0.0, "transport_jitter": 0.35},
        "display": {
            "enabled": 1.0, "scanlines": 0.38, "mask_pitch": 2.8, "mask_depth": 0.35,
            "percept_enabled": 1.0, "percept_mode": 4.0,
            "percept_viewing_dist_h": 4.0, "percept_display_nits": 120.0, "percept_ambient_lux": 45.0,
            "operator_skill": 0.78, "operator_stress": 0.38,
            "percept_quad_corner": 0.0, "percept_show_border": 2.0, "percept_show_text": 0.0,
        },
    },
}

FACTORY_PRESETS["media_ecosystem_historical_simulator_lab"] = {
    "label": "2×2 媒介技术生态演化、备件断崖与历史反事实情景模拟器 (1975–2035)",
    "note": "同屏对比：1992 模拟黄金期基准 vs 所选历史年份真实生态回放状态 vs 反事实情景推演状态 vs 1975–2035 七维生态演化曲线与可访问性断崖预警图",
    "standard": "pal",
    "layers": {
        "format": {"luma_bandwidth": 3.0, "chroma_bandwidth": 0.65},
        "tape": {"line_jitter": 0.22, "head_switching": 0.75, "dropout_rate": 0.20, "noise_floor": 0.15},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 0.85},
        "display": {"enabled": 0.0},
        "overlay": {
            "eco_enabled": 1.0, "eco_mode": 4.0,
            "eco_year": 2018.0, "eco_scenario": 1.0,
            "eco_spare_depletion": 1.15, "eco_institution_inertia": 0.55,
            "eco_quad_corner": 0.0, "eco_show_border": 2.0, "eco_show_text": 0.0,
        },
    },
}

UNIT_SPREADS.update({
    "stream_bw_fluctuation": (0.18, "CDN edge node and last-mile access throughput variance"),
    "stream_packet_loss": (0.22, "router queue drop and wireless burst loss variance"),
    "percept_display_nits": (0.12, "CRT phosphor luminance and backlight unit tolerance"),
    "operator_skill": (0.15, "individual broadcast engineer reaction time and tracking finesse"),
    "eco_spare_depletion": (0.14, "regional VTR head stock and technician retirement variance"),
    "eco_institution_inertia": (0.16, "organizational budget friction and legacy workflow inertia"),
})
FACTORY_PRESETS["sony_trinitron_kv1214_showa"] = {
    "label": "Sony Trinitron KV-1214 1978 12寸柱面特丽珑彩电 (4K 屏摄摩尔纹 + NTSC-J 9300K)",
    "note": "12寸柱面特丽珑 4K 屏摄实测标定：NTSC-J RF/复合输入 (3.8 MHz 亮度 + 1.0 MHz 色度 + 高对比荧光边缘过冲) → Sony KV-1214 12寸单枪三束垂直荫栅、单根钨丝阻尼线、水平单向柱面曲率、9300K 青蓝白点与清晰弯曲摩尔纹",
    "standard": "ntsc",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.02, "halation": 0.08, "ccd_smear": 0.0},
        "format": {
            "filter_kind": 1.0, "filter_sections": 2.0,
            "luma_bandwidth": 4.2, "chroma_bandwidth": 1.15,
            "chroma_vertical_averaging": 0.35, "separation_kind": 2.0, "comb_depth": 0.68,
            "cross_color": 0.22, "cross_luma": 0.16, "edge_boost": 0.28,
            "color_noise": 0.06, "chroma_gain": 1.06,
        },
        "tape": {
            "tape_stability": 0.96, "line_jitter": 0.03, "walk_amplitude": 0.02,
            "micro_jitter": 0.02, "vertical_instability": 0.0,
            "head_switching": 0.0, "dropout_rate": 0.0, "wear": 0.0, "noise_floor": 0.01,
        },
        "player": {
            "tbc_enabled": 1.0, "tbc_strength": 0.95, "transport_jitter": 0.03,
            "head_wear": 0.04, "preemphasis": 0.22, "ringing": 0.16,
            "luma_bandwidth": 4.2, "chroma_bandwidth": 1.15,
        },
        **DISPLAY_PROFILES["sony_trinitron_kv1214"]["layers"],
    },
}

FACTORY_PRESETS["crt_strong_magnet_physics_lab"] = {
    "label": "CRT 遇强磁铁、铁磁荫罩剩磁与 PTC 消磁线圈 2×2 物理实验室",
    "note": "强永磁体近场磁偏转与铁磁荫罩剩磁物理标定：同屏对比 1.强钕铁硼磁铁贴面（中心电子束排空黑洞+十字枕形折叠+同心彩虹环）、2.强磁极旋转涡流扭曲、3.撤走磁铁后的铁磁荫罩永久剩磁色斑、4.启动内置 PTC 热敏电阻消磁线圈（Degaussing Coil）衰减交变磁场消除剩磁",
    "standard": "ntsc",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
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
            "crt_magnet_quad_corner": 0.0, "crt_magnet_show_border": 2.0, "crt_magnet_show_text": 0.0,
        },
    },
}

for _lay in SCHEMA.layer_names():
    for _key, _spec in SCHEMA.layer(_lay).specs.items():
        if _key in UNIT_SPREADS:
            _sp, _note = UNIT_SPREADS[_key]
            _spec.unit_spread = float(_sp)
            _spec.unit_note = str(_note)



UNIT_SPREADS.update({
    "film_gate_weave": (0.18, "projector intermittent claw wear and registration pin tolerance"),
    "film_shutter_flicker": (0.14, "rotary shutter governor and lamp ballast ripple"),
    "film_grain_amount": (0.10, "emulsion batch and developer temperature variance"),
})

FACTORY_PRESETS["film_16mm_telecine_ntsc_32"] = {
    "label": "16mm 新闻反转片 → Rank Cintel 飞点胶转磁 (NTSC 3:2 Pulldown → 电视播出)",
    "note": "Kodak Ektachrome 16mm 卤化银特性曲线、红橙防光晕背散射、中密度曝光依赖颗粒、片门二维机械抖动与 24fps→29.97i 3:2 Pulldown 隔行交错梳状伪影",
    "standard": "ntsc",
    "layers": {
        "camera": {
            "film_enabled": 1.0, "film_stock": 2.0, "film_gamma": 1.12,
            "film_halation": 0.42, "film_grain_amount": 0.34, "film_grain_size": 1.65,
            "film_dye_fade": 0.12, "film_shrinkage": 0.08, "film_scratches_dust": 0.22,
            "film_gate_weave": 0.32, "film_shutter_flicker": 0.15, "telecine_pulldown": 1.0,
            "sensor_noise": 0.01, "sharpness": 1.02,
        },
        "format": {"luma_bandwidth": 4.2, "chroma_bandwidth": 1.1, "edge_boost": 0.20, "color_noise": 0.06},
        "tape": {"tape_stability": 0.92, "line_jitter": 0.05, "dropout_rate": 0.05, "head_switching": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 0.92},
        "display": {"enabled": 0.0},
    },
}

FACTORY_PRESETS["film_to_tv_to_home_vhs_lineage"] = {
    "label": "2×2 跨媒介历史对照实验室（16mm胶片 → 电视胶转磁 → 家用VHS录屏 → 流媒体重压）",
    "note": "同屏对比同一画面的四代跨媒介历史传播痕迹：16mm化学颗粒与片门抖动 → 3:2 Pulldown胶转磁 → 家用VHS欠载色度与切头带叠加 → 流媒体8×8块量化",
    "standard": "ntsc",
    "layers": {
        "camera": {
            "film_enabled": 1.0, "film_stock": 1.0, "film_gamma": 1.08,
            "film_halation": 0.38, "film_grain_amount": 0.30, "film_grain_size": 1.45,
            "film_dye_fade": 0.25, "film_scratches_dust": 0.18,
            "film_gate_weave": 0.28, "film_shutter_flicker": 0.12, "telecine_pulldown": 3.0,
            "telecine_quad_corner": 0.0, "telecine_show_border": 2.0, "telecine_show_text": 0.0,
            "sensor_noise": 0.0, "ccd_smear": 0.0, "halation": 0.0,
        },
        "format": {"luma_bandwidth": 5.5, "chroma_bandwidth": 1.8, "chroma_vertical_averaging": 0.0, "comb_depth": 0.95, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0, "edge_boost": 0.0},
        "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "vertical_instability": 0.0, "head_switching": 0.0, "dropout_rate": 0.0, "wear": 0.0, "crease_rate": 0.0, "noise_floor": 0.0},
        "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "tbc_residual": 0.0, "transport_jitter": 0.0, "head_wear": 0.0, "frequency_response": 0.0, "comb_depth": 0.95, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
        "display": {"enabled": 0.0},
    },
}

FACTORY_PRESETS["inversion_fadgi_calibration_lab"] = {
    "label": "2×2 参数反演、多解竞争假设与三级参数审计实验室 (FADGI 误差预算台)",
    "note": "实时反推退化画面的 -3dB 带宽、行时基抖动与色度相位置信区间，对滤波器/磁头磨损/光学散焦给出竞争性概率排序，并分离统计 Physical / Device / Artistic 三级参数贡献与 FADGI 合规星级",
    "standard": "ntsc",
    "layers": {
        "format": {"luma_bandwidth": 2.8, "chroma_bandwidth": 0.68, "filter_kind": 1.0, "edge_boost": 0.28},
        "tape": {"tape_stability": 0.62, "line_jitter": 0.32, "noise_floor": 0.14, "dropout_rate": 0.20},
        "player": {
            "tbc_enabled": 0.0, "head_wear": 0.28,
            "inv_lab_enabled": 1.0, "inv_lab_mode": 3.0, "inv_fadgi_target": 3.0,
        },
        "display": {"enabled": 0.0},
    },
}

for _lay in SCHEMA.layer_names():
    for _key, _spec in SCHEMA.layer(_lay).specs.items():
        if _key in UNIT_SPREADS:
            _sp, _note = UNIT_SPREADS[_key]
            _spec.unit_spread = float(_sp)
            _spec.unit_note = str(_note)


# ==================== 古早数字媒介系统 (LDME)：早期手机、iPhone、优酷土豆转码与三大基准测试（纯净命名 · 全屏单画面） ====================

CAMERA_PROFILES.update({
    "nokia_6600": {
        "label": "Nokia 6600",
        "note": "2003 年 VGA (640×480) 早期拍照手机：微型塑料定焦镜头边缘软化与暗角、自动白平衡微漂移、Bayer 去马赛克伪影与有损 JPEG 4:2:0 压缩",
        "layers": {"camera": {
            "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 1.0, "ldme_capture_mode": 0.0,
            "sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
        }},
    },
    "nokia_n73": {
        "label": "Nokia N73",
        "note": "2006 年 320 万像素卡尔蔡司 Tessar 镜头多媒体手机：强力色度/亮度降噪造成的细节涂抹感、USM 锐化边缘亮晕与 JPEG 4:2:0 编码",
        "layers": {"camera": {
            "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 2.0, "ldme_capture_mode": 0.0,
            "sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
        }},
    },
    "nokia_n95": {
        "label": "Nokia N95",
        "note": "2007 年 500 万像素卡尔蔡司光学机皇：鲜艳色彩矩阵、暗部泊松-高斯噪声与边缘锐化过冲",
        "layers": {"camera": {
            "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 3.0, "ldme_capture_mode": 0.0,
            "sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
        }},
    },
    "nokia_5800_3gp": {
        "label": "Nokia 5800 XpressMusic",
        "note": "2008 年过渡期多媒体手机 3GP 视频录制管线 (`record()`)：640×352 H.263/MPEG-4 视频编码、帧率受限与 CMOS 卷帘快门",
        "layers": {"camera": {
            "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 4.0, "ldme_capture_mode": 1.0,
            "sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
        }},
    },
    "iphone_2007": {
        "label": "iPhone 2007",
        "note": "2007 年初代 iPhone (200 万像素 1600×1200 定焦)：有限曝光动态范围导致亮部高光硬削波、自然偏暖自动白平衡与克制锐化",
        "layers": {"camera": {
            "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 5.0, "ldme_capture_mode": 0.0,
            "sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
        }},
    },
    "iphone_3g": {
        "label": "iPhone 3G",
        "note": "2008 年 iPhone 3G (200 万像素定焦)：延续初代小型传感器成像限制，改进镜头镀膜与自动白平衡稳定性",
        "layers": {"camera": {
            "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 6.0, "ldme_capture_mode": 0.0,
            "sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
        }},
    },
    "iphone_3gs": {
        "label": "iPhone 3GS",
        "note": "2009 年 iPhone 3GS 独立视频录制管线 (`record()`)：640×480 @ 30fps MOV/H.264 录像、连续自动曝光阶跃与 CMOS 卷帘快门",
        "layers": {"camera": {
            "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 7.0, "ldme_capture_mode": 1.0,
            "sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
        }},
    },
    "iphone_4s": {
        "label": "iPhone 4S",
        "note": "2011 年 iPhone 4S (800 万像素背照式 BSI CMOS + A5 ISP)：显著提升解析力、暗光信噪比与高光宽容度",
        "layers": {"camera": {
            "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 8.0, "ldme_capture_mode": 0.0,
            "sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
        }},
    },
})

DISPLAY_PROFILES.update({
    "lcd_feature_phone_rgb565": {
        "label": "早期手机 TFT 液晶屏",
        "note": "2004–2008 功能机 176×208 / 240×320 65K 色 (RGB565 16-bit) TFT 面板：阶梯量化色带、28ms 慢速液晶响应拖影、窄视角伽马反相与方形像素晶格（无 CRT 扫描线）",
        "layers": {"display": {
            "enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 1.0,
            "ldme_lcd_bits": 5.0, "ldme_lcd_response_ms": 28.0, "ldme_lcd_view_angle": 16.0,
            "ldme_lcd_subpixel": 0.50, "ldme_flash_scale_mode": 0.0,
        }},
    },
    "lcd_desktop_2007_tn": {
        "label": "2007 桌面 TN 液晶显示器",
        "note": "2007 年主流 17/19 寸 6-bit TN 液晶显示器 + Flash 播放器非整数倍窗口缩放：64 级灰阶 FRC 抖动、16ms 样本保持拖影、垂直视角伽马偏移、CCFL 冷白背光与 RGB 竖条子像素",
        "layers": {"display": {
            "enabled": 0.0, "ldme_display_enabled": 1.0, "ldme_display_device": 2.0,
            "ldme_lcd_bits": 6.0, "ldme_lcd_response_ms": 16.0, "ldme_lcd_view_angle": 12.0,
            "ldme_lcd_subpixel": 0.35, "ldme_lcd_ccfl": 0.28, "ldme_flash_scale_mode": 1.0, "ldme_net_throughput_kbps": 950.0,
        }},
    },
})

_CLEAN_ANALOG_BYPASS = {
    "format": {"luma_bandwidth": 5.8, "chroma_bandwidth": 1.9, "cross_color": 0.0, "cross_luma": 0.0, "color_noise": 0.0},
    "tape": {"tape_stability": 1.0, "line_jitter": 0.0, "walk_amplitude": 0.0, "micro_jitter": 0.0, "dropout_rate": 0.0, "noise_floor": 0.0, "wear": 0.0, "head_switching": 0.0},
    "player": {"tbc_enabled": 1.0, "tbc_strength": 1.0, "transport_jitter": 0.0, "head_wear": 0.0},
}

# --- 测试一：同一场景，不同手机（纯净命名 · 单画面全幅输出） ---
for _p_key, _p_id, _cap_m, _p_label, _p_note in (
    ("ldme_nokia_6600", 1.0, 0.0, "Nokia 6600", "测试一（同一场景，不同手机）：2003 年 Nokia 6600 VGA 拍照管线，展现低像素空间采样限制、小塑胶镜头边缘模糊、AWB 微漂移与有损 JPEG 4:2:0 压缩"),
    ("ldme_nokia_n73", 2.0, 0.0, "Nokia N73", "测试一（同一场景，不同手机）：2006 年 Nokia N73 320 万像素卡尔蔡司管线，展现 ISP 强降噪细节涂抹、USM 锐化边缘光晕与 4:2:0 块效应"),
    ("ldme_nokia_n95", 3.0, 0.0, "Nokia N95", "测试一（同一场景，不同手机）：2007 年 Nokia N95 500 万像素卡尔蔡司管线，展现高饱和蔡司色彩矩阵与锐化过冲"),
    ("ldme_nokia_5800_3gp", 4.0, 1.0, "Nokia 5800 XpressMusic", "测试一（同一场景，不同手机）：2008 年 Nokia 5800 3GP 动态视频录制管线 (`record()`)，展现 640×352 H.263 视频编码、帧率受限与卷帘快门"),
    ("ldme_iphone_2007", 5.0, 0.0, "iPhone 2007", "测试一（同一场景，不同手机）：2007 年初代 iPhone 200 万像素定焦拍照管线 (`capture()`)，展现有限自动曝光高光硬削波与自然偏暖肤色还原"),
    ("ldme_iphone_3g", 6.0, 0.0, "iPhone 3G", "测试一（同一场景，不同手机）：2008 年 iPhone 3G 200 万像素拍照管线，展现早期小型 CMOS 在暗部信噪比与固定焦距上的物理特征"),
    ("ldme_iphone_3gs_video", 7.0, 1.0, "iPhone 3GS", "测试一（同一场景，不同手机）：2009 年 iPhone 3GS 独立 VGA 视频录制管线 (`record()`)，展现 640×480 @ 30fps H.264 编码、连续曝光阶跃与卷帘快门"),
    ("ldme_iphone_4s", 8.0, 0.0, "iPhone 4S", "测试一（同一场景，不同手机）：2011 年 iPhone 4S 800 万像素背照式 BSI CMOS + A5 ISP 管线，展现显著提升的边缘解析力与高光宽容度"),
):
    FACTORY_PRESETS[_p_key] = {
        "label": _p_label,
        "note": _p_note,
        "standard": "pal",
        "layers": {
            "camera": {
                "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": _p_id, "ldme_capture_mode": _cap_m,
                "sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
            },
            **_CLEAN_ANALOG_BYPASS,
            "display": {"enabled": 0.0},
        },
    }

# --- 测试二：同一视频，不同转码链 & 2006–2015 优酷/土豆平台流媒体阶段（纯净命名 · 单画面全幅输出） ---
FACTORY_PRESETS["ldme_youku_2007_flv_single"] = {
    "label": "早期优酷土豆 Flash 流畅档",
    "note": "测试二（同一视频，不同转码链 · 阶段一 2006–2009）：FLV 容器 + Sorenson Spark (`flv1`) 320×240 @ 25fps、350kbps 视频与 48kbps MP3 音频的单次平台转码损伤",
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        **_CLEAN_ANALOG_BYPASS,
        "codec": {
            "stream_enabled": 1.0, "stream_mode": 6.0, "stream_bitrate_kbps": 350.0,
            "stream_gop_len": 15.0, "ldme_source_origin": 0.0,
        },
        "display": {"enabled": 0.0},
    },
}

FACTORY_PRESETS["ldme_platform_2010_sd_h264"] = {
    "label": "标清 H.264 视频档",
    "note": "测试二（同一视频，不同转码链 · 阶段二 2009–2012）：高清源被平台下采样转码为 480×360 @ 25fps、700kbps H.264 标清档，体现环路去块滤波对 16×16 宏块边界的平滑与纹理柔化",
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        **_CLEAN_ANALOG_BYPASS,
        "codec": {
            "stream_enabled": 1.0, "stream_mode": 7.0, "stream_bitrate_kbps": 700.0,
            "stream_gop_len": 25.0, "ldme_source_origin": 0.0,
        },
        "display": {"enabled": 0.0},
    },
}

FACTORY_PRESETS["ldme_mobile_2013_multiscreen"] = {
    "label": "移动互联网多屏 H.264 档",
    "note": "阶段三（2012–2015 移动互联网与多设备播放）：640×360 @ 25fps H.264 MP4/HLS 多终端适配流",
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        **_CLEAN_ANALOG_BYPASS,
        "codec": {
            "stream_enabled": 1.0, "stream_mode": 8.0, "stream_bitrate_kbps": 600.0,
            "stream_gop_len": 25.0, "ldme_source_origin": 0.0,
        },
        "display": {"enabled": 0.0},
    },
}

FACTORY_PRESETS["ldme_multigen_nle_youku_reupload"] = {
    "label": "多代剪辑导出与平台重传链",
    "note": "测试二（同一视频，不同转码链）：模拟 V0 经桌面剪辑软件压缩为低码率 MP4 (V1)，上传优酷转码为 FLV (V2)，再被下载二次上传重压 (V3) 的非对齐宏块与运动预测误差累积",
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        **_CLEAN_ANALOG_BYPASS,
        "codec": {
            "stream_enabled": 1.0, "stream_mode": 9.0,
            "ldme_transfer_enabled": 1.0, "ldme_transfer_method": 4.0,
            "ldme_transfer_generations": 3.0, "stream_platform_generations": 3.0,
            "stream_bitrate_kbps": 350.0, "ldme_source_origin": 1.0,
        },
        "display": {"enabled": 0.0},
    },
}

FACTORY_PRESETS["ldme_lowfps_3gp_to_flv"] = {
    "label": "低帧率 3GP 上传转码 FLV",
    "note": "测试二（同一视频，不同转码链）：区分『原本就是手机 15fps 3GP 低清直传』与『高清源被降为低清』，展现低帧率时间采样顿挫与两次异构编码器块伪影叠加",
    "standard": "pal",
    "layers": {
        "camera": {
            "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 4.0, "ldme_capture_mode": 1.0,
            "sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
        },
        **_CLEAN_ANALOG_BYPASS,
        "codec": {
            "stream_enabled": 1.0, "stream_mode": 6.0, "stream_bitrate_kbps": 320.0,
            "stream_gop_len": 12.0, "ldme_source_origin": 1.0,
        },
        "display": {"enabled": 0.0},
    },
}

FACTORY_PRESETS["ldme_mms_social_gateway"] = {
    "label": "彩信与早期社交软件传图",
    "note": "子系统 D（03 · 文件与传输）：模拟彩信 (MMS) 与早期社交软件对手机照片执行强制 320×240 缩放、高压缩比 JPEG 重编码并剥离全部 EXIF 元数据",
    "standard": "pal",
    "layers": {
        "camera": {
            "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 2.0, "ldme_capture_mode": 0.0,
            "sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
        },
        **_CLEAN_ANALOG_BYPASS,
        "codec": {
            "stream_enabled": 1.0, "stream_mode": 9.0,
            "ldme_transfer_enabled": 1.0, "ldme_transfer_method": 3.0,
            "ldme_mms_width": 320.0, "ldme_mms_quality": 50.0,
        },
        "display": {"enabled": 0.0},
    },
}

# --- 测试三：同一文件，不同播放设备（纯净命名 · 单画面全幅输出） ---
FACTORY_PRESETS["ldme_display_phone_tft_lcd"] = {
    "label": "早期手机 TFT 液晶显示",
    "note": "测试三（同一文件，不同播放设备）：固定同一份解码后的流媒体画面，输出到早期手机 16-bit RGB565 TFT 液晶屏（65K 色量化阶梯、28ms 液晶保持拖影、窄视角偏移与方形像素晶格）",
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        **_CLEAN_ANALOG_BYPASS,
        "codec": {"stream_enabled": 1.0, "stream_mode": 7.0, "stream_bitrate_kbps": 700.0},
        **DISPLAY_PROFILES["lcd_feature_phone_rgb565"]["layers"],
    },
}

FACTORY_PRESETS["ldme_display_desktop_tn_lcd"] = {
    "label": "桌面 TN 液晶与 Flash 播放",
    "note": "测试三（同一文件，不同播放设备）：固定同一份解码后的流媒体画面，经 Flash 播放器非整数倍窗口缩放后输出到 2007 年桌面 6-bit TN 液晶显示器（64 级 FRC 抖动、液晶拖影、视角伽马偏移与 CCFL 冷白背光）",
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        **_CLEAN_ANALOG_BYPASS,
        "codec": {"stream_enabled": 1.0, "stream_mode": 7.0, "stream_bitrate_kbps": 700.0},
        **DISPLAY_PROFILES["lcd_desktop_2007_tn"]["layers"],
    },
}

FACTORY_PRESETS["ldme_display_crt_monitor_ref"] = {
    "label": "经典 CRT 监视器同源对照",
    "note": "测试三（同一文件，不同播放设备）：固定同一份解码后的流媒体画面，输出到球面 CRT 监视器（扫描线、荧光粉非线性发光与孔阑荫罩），用于与手机/桌面 LCD 进行物理机制对照",
    "standard": "pal",
    "layers": {
        "camera": {"sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0},
        **_CLEAN_ANALOG_BYPASS,
        "codec": {"stream_enabled": 1.0, "stream_mode": 7.0, "stream_bitrate_kbps": 700.0},
        **DISPLAY_PROFILES["crt_consumer"]["layers"],
    },
}

# --- 第九节终极全链路实验：场景 → Nokia N73 拍摄 → USB 导入 → 编辑软件导出低码率视频 → 优酷式转码 → Flash 缩放 → 旧 LCD 显示 ---
FACTORY_PRESETS["ldme_n73_usb_nle_youku_flash_lcd"] = {
    "label": "诺基亚拍摄经剪辑上传优酷全链路",
    "note": "完整复现第九节终极考古实验：场景 → Nokia N73 拍摄 → USB 导入电脑 → 剪辑软件导出低码率视频 → 优酷式 FLV 转码 → Flash 播放器窗口缩放 → 2007 桌面 6-bit TN 液晶显示",
    "standard": "pal",
    "layers": {
        "camera": {
            "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 2.0, "ldme_capture_mode": 0.0,
            "sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
        },
        **_CLEAN_ANALOG_BYPASS,
        "codec": {
            "stream_enabled": 1.0, "stream_mode": 9.0,
            "ldme_transfer_enabled": 1.0, "ldme_transfer_method": 4.0,
            "ldme_transfer_generations": 2.0, "stream_platform_generations": 2.0,
            "stream_bitrate_kbps": 350.0, "ldme_source_origin": 1.0,
        },
        **DISPLAY_PROFILES["lcd_desktop_2007_tn"]["layers"],
    },
}

FACTORY_PRESETS["digicam_early_phone_flicker_tungsten"] = {
    "label": "旧手机荧光灯频闪与钨丝灯白平衡失效",
    "note": "物理建模旧手机实拍图片（image.jpg、image (1).jpg 与 b0ae5b91...jpg）：CMOS 卷帘快门 50Hz 工频荧光灯水平青绿/橙黄交替条纹 + 餐厅低色温钨丝灯/混光 AWB 蓝增益硬钳位芥末青黄偏色",
    "standard": "pal",
    "layers": {
        "camera": {
            "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 2.0, "ldme_capture_mode": 0.0,
            "ldme_flicker_enabled": 1.0, "ldme_flicker_mains_hz": 50.0, "ldme_flicker_depth": 0.45, "ldme_flicker_color_drift": 0.65, "ldme_flicker_num_bands": 5.2,
            "ldme_tungsten_enabled": 1.0, "ldme_color_temp_k": 2700.0, "ldme_awb_mode": 2.0,
            "sensor_noise": 0.04, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
        },
        **_CLEAN_ANALOG_BYPASS,
        "codec": {"stream_enabled": 0.0, "dv_enabled": 0.0},
        **DISPLAY_PROFILES["lcd_feature_phone_rgb565"]["layers"],
    },
}

FACTORY_PRESETS["youku_tudou_multi_gen_jpeg_wear"] = {
    "label": "早期优酷土豆多代 JPEG 磨损衰减",
    "note": "2006–2009 优酷土豆视频截图与论坛搬运多代 JPEG 有损磨损：真实 2D 8×8 DCT 量化、4:2:0 色度衰减与跨代 8×8 网格错位双重蚊式振铃噪点",
    "standard": "pal",
    "layers": {
        "camera": {
            "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 1.0, "ldme_capture_mode": 0.0,
            "ldme_jpeg_decay_enabled": 1.0, "ldme_jpeg_decay_gens": 4.0, "ldme_jpeg_decay_quality": 45.0, "ldme_jpeg_decay_grid_shift": 3.0,
            "sensor_noise": 0.0, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
        },
        **_CLEAN_ANALOG_BYPASS,
        "codec": {
            "stream_enabled": 0.0, "ldme_transfer_enabled": 1.0, "ldme_transfer_method": 5.0,
            "ldme_transfer_generations": 4.0, "ldme_jpeg_decay_quality": 45.0, "ldme_jpeg_decay_grid_shift": 3.0,
        },
        **DISPLAY_PROFILES["lcd_desktop_2007_tn"]["layers"],
    },
}

FACTORY_PRESETS["digicam_night_starburst_ccd"] = {
    "label": "早期数码相机夜景车灯星芒与粗糙宏块",
    "note": "物理建模旧相机夜景实拍（image_d61d18.png）：微型镜头夜景车灯星芒溢出、点光源垂直光柱、1/4 英寸 CCD 满井饱和硬削波与暗部深空粗糙 DCT 块效应",
    "standard": "pal",
    "layers": {
        "camera": {
            "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 1.0, "ldme_capture_mode": 0.0,
            "ldme_night_bloom": 1.25,
            "ldme_jpeg_decay_enabled": 1.0, "ldme_jpeg_decay_gens": 2.0, "ldme_jpeg_decay_quality": 35.0,
            "sensor_noise": 0.05, "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
        },
        **_CLEAN_ANALOG_BYPASS,
        "codec": {"stream_enabled": 0.0, "dv_enabled": 0.0},
        **DISPLAY_PROFILES["lcd_desktop_2007_tn"]["layers"],
    },
}

FACTORY_PRESETS["digicam_y2k_ccd_sensor"] = {
    "label": "千禧年 Y2K 卡片机 CCD 饱和与动态范围",
    "note": "参考 digicamfx.com 经典小底数码卡片机：CCD 高饱和暖调肤色、小尺寸传感器动态范围硬截断与低像素微细节",
    "standard": "pal",
    "layers": {
        "camera": {
            "ldme_mobile_enabled": 1.0, "ldme_mobile_profile": 3.0, "ldme_capture_mode": 0.0,
            "ldme_night_bloom": 0.40, "ldme_jpeg_quality": 68.0, "sensor_noise": 0.035,
            "sharpness": 1.0, "halation": 0.0, "ccd_smear": 0.0,
        },
        **_CLEAN_ANALOG_BYPASS,
        "codec": {"stream_enabled": 0.0, "dv_enabled": 0.0},
        **DISPLAY_PROFILES["lcd_desktop_2007_tn"]["layers"],
    },
}

for _aud_k, _aud_spec in AUDIO_PROFILES.items():
    FACTORY_PRESETS[_aud_k] = {
        "label": _aud_spec["label"],
        "note": _aud_spec["note"],
        "layers": _aud_spec["layers"],
    }
