# Virtual Media Lab — 虚拟媒介实验室 (`atsim`)

**基于真实物理信号链、光电化学机理与媒介考古学的视听仿真工作站**

`Virtual Media Lab`（核心包名 `atsim`）不是一套给画面叠加表面噪点或 VHS 贴图的滤镜插件，而是一座按照真实广播电视工程、光化学洗印、早期数字编解码与消费级模拟/数字硬件电路构建的**九层物理信号链仿真引擎与交互式节点图工作站**。

从 1970 年代的盘式磁带、Sony Trinitron 显像管、35mm 电影胶片洗印与醋酸综合症，到 1990 年代的 VHS 多代翻录、VCD MPEG-1、卫视锅盖天线（C 波段模拟 FM / DVB-S），再到 2000 年代初的诺基亚 Symbian / 初代 iPhone 手机摄像头（荧光灯卷帘频闪、钨丝灯白平衡失效、车灯星芒与 CCD 垂直拖尾）以及优酷/土豆早期 Flash 流媒体与包浆表情包的多代 JPEG 8×8 DCT 磨损——每一个视觉与听觉特征都直接产生于底层物理方程、电路传递函数与离散有状态随机过程，而非静态图层叠加。

## 一、快速上手与环境配置

### 1. 环境依赖

本项目核心采用纯 Python 标准库提供 HTTP/WebUI 服务，无需安装 Flask、FastAPI 或 Node.js 前端构建链：

- **Python**：`3.11+`
- **Python 核心依赖**：`numpy`、`scipy`、`Pillow`、`fonttools`（`fonttools` 用于硬件字幕机 TrueType 字体表 `cmap`/`hmtx` 度量解析；最新版 `atsim/codecs/fonts.py` 已支持无 `fonttools` 自动降级回退至 `Pillow`）
- **系统多媒体工具**（需位于系统 `PATH` 中，用于视频流读写与离线长片编码）：`ffmpeg`、`ffprobe`

```bash
pip install numpy scipy Pillow fonttools
```

### 2. 启动与管理 WebUI 服务（推荐使用 `server.py`）

请优先使用项目根目录提供的进程守护脚本 `server.py`（Windows 下可直接双击 `server.bat` 或运行 `run_webui.bat`）。它会将启动的 PID 与端口记录在 `.server/pid.json` 中，`stop` 命令仅精确终止由它启动的进程，绝不会误杀机器上的其他 Python 任务：

```bash
# 启动 WebUI 服务（默认端口 8947）
python server.py start

# 启动服务并自动在浏览器中打开黑绿终端主题工作台
python server.py start --open

# 指定自定义端口与工程运行数据目录
python server.py start --port 9000 --data-dir runs

# 查看当前服务运行状态、PID 与端口响应
python server.py status

# 实时查看服务端日志（存储于 .server/server.log）
python server.py logs

# 安全停止或重启当前 WebUI 服务
python server.py stop
python server.py restart
```

如果需要直接通过模块命令行启动并载入外部视频或图片素材，也可以使用：

```bash
# 直接以模块方式启动并打开浏览器
python -m atsim.web.server --port 8947 --open

# 启动时直接载入本地视频或静态图片素材
python -m atsim.web.server --clip my_footage.mp4 --open
```

### 3. 运行物理自检与标定套件

```bash
# 快速运行全链路物理与契约回归验证（约 15 秒）
python tools/verify.py --quick

# 运行完整 100+ 项物理机理、状态机与测量仪器校验
python tools/verify.py

# 运行引擎确定性、帧间时域平滑度与多级缓存失效自检
python tools/selftest.py

# 测量模拟复合信号通道的 MHz 频响与梳状滤波器标定曲线
python tools/measure_response.py
```

## 二、核心设计哲学：为什么它不是「VHS 滤镜」

1. **伪影产生于真实的信号合并与分离残差**：在模拟复合信号层（`02 信号编码`），引擎真正将基带亮度 $Y$ 与正交调制的色度副载波 $C(f_{sc})$ 相加为单路复合波形；到了播放解码层（`04 读取解码`），再通过带通或 1H/2H/2D 梳状滤波器尝试将二者重新分离。**交叉色彩（彩虹纹）与交叉亮度（边缘爬行点）不是人为加上去的特效，而是因为亮度和色度真的加在了一起、解码器没能彻底分干净所留下的数学残差。**
2. **时基误差是逐扫描行的连续积分过程**：磁带走带与旋转磁鼓的速度波动遵循 Ornstein-Uhlenbeck (OU) 阻尼随机过程，水平采样相位误差是速度波动的连续积分。相邻两条扫描线间隔仅约 $63.5\,\mu\text{s}$，因此呈现真实的竖线锯齿波纹、S 形扭曲与锁相环（PLL）慢速摆动，而非全画面生硬平移。
3. **色度副载波解耦与欠载记录（Color-Under）**：物理磁带机采用色度降频欠载记录并在读取时通过自动相位控制（APC）重建副载波。本引擎严格在基带（$Y/I/Q$ 平面）施加时基位移，再调制副载波，避免了直接对副载波做亚像素位移导致色相随机旋转的常见仿真错误。
4. **离散确定性事件与跨帧状态记忆**：磁带失落（Dropout）、折痕（Crease）、粘滞滑移（Stick-Slip）、数字流媒体 TCP 缓冲卡顿、胶片醋酸综合症演变等均为带确定性随机种子（Seed）的有状态过程。同一工程配置在任意时刻重复渲染，输出的每一帧像素与伴音采样 100% 确定性一致。
5. **测量仪器优先原则（Suspect the Instrument First）**：所有物理机制均配备专属无混杂测量探针（见 `tools/verify.py`），确保每一项参数具有物理单位（MHz、IRE、dB、$\mu\text{s}$、K、nm）与真实可测的物理响应区间。

## 三、九层全链路信号架构

引擎默认提供从光学成像到显像管重拍及伴音合成的 **9 层全链路流水线**，每一层均可独立开关、旁路或在节点图中任意自由拓扑连接：

| 链路层级 | 核心模块 (`atsim/stages/` & `atsim/codecs/`) | 物理处理对象 | 核心仿真机制与代表性硬件模型 |
| :--- | :--- | :--- | :--- |
| **01 成像系统** (`camera`) | `camera.py`, `lens.py`, `optics.py`, `nightvision.py`, `mobile_isp_optics.py`, `studio_ccu_genlock.py` | 场景光子 $\rightarrow$ 传感器线性 RGB | W–T 连续光学变焦（$f=3.6\text{--}72\text{mm}$，呼吸效应与球差随焦距耦合）、CCD 垂直高光溢出拖尾（Smear）、CMOS 卷帘快门、索尼 NightShot 红外夜视（0–3 档）、50/60Hz 荧光灯卷帘频闪横纹、2700K 钨丝灯与混合光源白平衡失效、早期诺基亚/iPhone 6 级手机 ISP、演播室多机位 CCU 与 Genlock 台标同步 |
| **01b 胶片光化学** (`film`) | `film.py`, `film_optics.py`, `photochemical_film.py`, `film_telecine.py` | 光潜影 $\rightarrow$ 卤化银密度 $D(\log H)$ | 20 款真实胶片（Kodak Vision3 500T、Double-X 5222、Kodachrome 64、LOMO、宝丽来等）、纯物理反光晕层（Remjet Halation vs 镜头眩光 vs 电荷溢出）、Trioplan 肥皂泡/心形/星形散景、Petzval 旋焦、暗房冲印（ECN-2/C-41/E-6 交叉冲洗、增感/减感、漂白旁路）、自催化醋酸综合症（Vinegar Syndrome）、湿门扫描与 Rank Cintel 飞点胶转磁 |
| **02 信号编码** (`format`) | `format.py`, `signal.py`, `analog_iir.py`, `test_signal_gen.py` | RGB $\rightarrow$ 复合基带 / 分量信号 | NTSC-M / PAL-B,D / SECAM 制式、Constant-K 与 Butterworth 模拟滤波器群延迟振铃、色度欠载（Color-Under）、跨帧色度时间残留 `chroma_lag`（一阶 IIR 拖尾）、广播级测试信号发生器（TSG-170A / PM5544 / EBU / Multiburst / Zone Plate） |
| **02b 数字与射频** (`codec` / `rf`) | `codec.py`, `dv.py`, `mpeg1.py`, `legacy_digital_media.py`, `network_streaming_era.py`, `rf.py`, `satellite_ird.py`, `tv_transmitter_chain.py` | 离散块 DCT / 比特流 / VSB-AM & FM 射频 | MiniDV DV25（4:1:1/4:2:0 DCT）、MPEG-1 VCD 光盘划痕与宏块崩坏、早期手机 3GP/MMS 彩信、1999 RealVideo / 2006 优酷土豆早期 FLV/H.263 流媒体、**多代 JPEG 8×8 DCT 4:2:0 错格磨损与蚊式振铃**、甚高频 RF 调制、TV-DX 对流层远距离串台、C 波段模拟卫视 FM 鱼骨火点与 DVB-S 悬崖效应 |
| **02c 链路接口** (`connector`) | `connector.py`, `connectors.py`, `switcher.py`, `patchbay_da.py`, `power_bus.py`, `spatial_environment_field.py` | 线缆阻抗、接地环路、切换台与供电母线 | AV/RCA 50/60Hz 接地环路滚道、S-Video 亮色寄生电容串扰、75Ω 跳线盘双端接暗化与无端接驻波重影、Panasonic WJ-AVE5 模拟特技切换台（硬切非同步跳帧、划像、Luma/Chroma Key）、全机架公共电源母线压降与镍镉电池内阻崩溃、2D 空间温湿度/电磁场耦合 |
| **03 记录介质** (`tape`) | `tape.py`, `tape_speed.py`, `stiction.py`, `material.py`, `transport_servo.py`, `edit_controller_nle.py` | 磁粉剩磁 $M(B)$ 与机械走带运动学 | VHS SP/LP/EP 走带模式、逐行 OU 时基抖动、磁头切换噪波带（Head-Switch）、泊松失落事件 + 1H 延迟线补偿、粘滞滑移（Stiction）、霉斑与粘合剂脱落、COSMAG 劣质磁带掉粉/开裂/断带、**1–20 代 VHS 级联翻录劣化**、快进/倒带（Cue/Review）横跨噪波条、BVE-9100 线性编辑机与早期 NLE 非编时间线 |
| **04 读取解码** (`player`) | `player.py`, `sync_clock.py`, `transfer_workstation.py`, `inversion_calibration_lab.py`, `archival_archaeology.py` | 复合波形 $\rightarrow$ 分离 $Y/I/Q$ 与行场同步 | 1H/2H/2D 自适应梳状滤波器、行级与全帧数字时基校正器（TBC）、夏普三机串联稳像、LM1881 同步分离与 Adler 方程 H-Hold/V-Hold 失锁滚屏、Pulse Cross 十字脉冲消隐检视、多代跨介质转录工作站、**参数反演与三级误差预算校准实验室**、档案修复室（原带 vs 档案母带 vs 过度降噪对比） |
| **05 显示设备** (`display`) | `display.py`, `crt.py`, `crt_reshoot.py`, `multiview_wall.py`, `perception_and_operator.py` | 电子束扫描发光 $\rightarrow$ 屏幕与相机重拍 | 荫栅（Aperture Grille / Trinitron）、槽孔（Slot Mask）与圆点荫罩（Dot Triad）、P22 荧光粉余辉、高压包（EHT）负载跌落呼吸效应、**相机重拍 CRT 屏幕**（快门滚道黑带与栅距拍频摩尔纹）、**强磁铁洛伦兹力旋涡畸变与 PTC 消磁线圈**、Sony KV-1214 (1978) 4K 级还原、演播室多画面监视墙、人眼 CSF 感知与操作员 PID 闭环排障 |
| **06 字幕与总控** (`overlay`) | `overlay.py`, `osd.py`, `fonts.py`, `master_playout.py`, `playout_automation_v2.py`, `media_ecosystem_evolution.py` | 硬件字符点阵与播出自动化 | 28 款经典点阵/硬件/纪录片字体（支持入带前参与模拟劣化或出屏后干净叠加）、广播总控播出通道（PGM / Clean Feed / PVW / DSK 下游键 / 紧急切黑）、事件驱动播出自动化引擎 v2（6 类故障分支与因果审计报告）、1975–2035 跨时代媒介生态演化模拟 |
| **07 同步伴音** (`audio`) | `audio.py` | 模拟磁迹 / 调频伴音 / 数字音频帧 | VHS 线性磁迹、VHS Hi-Fi AFM 旋转磁头切换爆音、8mm AFM、MiniDV 12-bit PCM、VCD MP2 块丢失水泡声、PAL-D 6.5MHz 伴音载波蜂鸣（Intercarrier Buzz），与画面帧率严格同步的抖晃（Wow & Flutter）、粘滞降调与 CRT 15.7kHz 行频啸叫 |

## 四、专题物理模型与操作指南

### 1. 早期流媒体与旧手机多代 JPEG 磨损（包浆 / 块效应 / 蚊式噪点）

在早期诺基亚手机拍照、MMS 彩信、论坛传图以及 2006 年前后优酷/土豆网的多次转码链条中，画面会经历极为显著的 **JPEG / 帧内 8×8 DCT 磨损**。本系统在两个独立层级提供了真实的 YCbCr 4:2:0 8×8 离散余弦变换（DCT）量化与代际错格重采样实现，你可以根据场景任选其一或叠加使用：

| 操作路径 | 适用场景 | 关键参数与调节方式 |
| :--- | :--- | :--- |
| **路径 A：直接加载一键预设** | 最快体验早期优酷/土豆、手机彩信或社交平台传图包浆 | 在 WebUI 左侧预设栏顶部点击 **「古早数字/手机/流媒体」** 分类标签，点击 **`早期优酷土豆多代 JPEG 磨损衰减`**、**`诺基亚拍摄经剪辑上传优酷全链路`** 或 **`彩信与早期社交软件传图`** |
| **路径 B：`01 成像系统` 手机 ISP 机内 JPEG 磨损** | 模拟旧手机/数码相机机内低质量 JPG 存储及社交软件多次转发重存 | 展开右侧 **`01 成像系统`** $\rightarrow$ 开启 `mobile_isp_enabled`（旧手机与初代 iPhone ISP 总开关）与 `jpeg_decay_enabled`（JPEG 磨损与多代重压缩开关） $\rightarrow$ 调低 `jpeg_quality`（推荐 `15–35`）并调高 `jpeg_generations`（重压缩代数 `3–10` 代）及 `jpeg_chroma_subsampling`（`1.0` 为 4:2:0 色度半分辨率块化） |
| **路径 C：`02b 数字编码` 流媒体与网络代际 JPEG 磨损** | 模拟视频剪辑导出、平台二次转码、论坛盗图加水印重传 | 展开右侧 **`02b 数字编码`** $\rightarrow$ 将 `codec_family` 设为 `streaming`（网络流媒体时代） $\rightarrow$ 开启 `stream_jpeg_wear_enabled` $\rightarrow$ 调节 `stream_jpeg_quality`（`15–40`）、`stream_jpeg_generations`（`2–8` 代）与 `stream_jpeg_mosquito`（高频边缘振铃与蚊式噪点强度） |

- **跨代微移位错格效应（Grid Shift）**：当重压缩代数 `generations >= 2` 时，引擎不仅重复量化频域系数，还会在偶数代自动引入 `1px` 像素级网格偏移（模拟截图、缩放、加贴纸或平台非对齐重编码），旧的 8×8 块边界变成高频阶跃信号并在新一代 DCT 中激发剧烈的**蚊式噪点（Mosquito Noise）**与**块内阶梯伪影**。
- **纯净无文字控制**：所有流媒体与手机预设均支持顶部工具栏的 **「文字：不要文字（纯净）」** 一键净化，开启后不仅移除分屏标题，还会同步抑制流媒体画面中央的 `BUFFERING...` 缓冲提示文字与重传水印横幅。

### 2. 旧手机荧光灯卷帘频闪、钨丝灯白平衡失效与夜景车灯星芒

针对千禧年早期手机（如 Nokia 6600/N73/N95、初代 iPhone 2007/3G）在室内餐饮与夜景街拍中的典型物理缺陷，`atsim/codecs/mobile_isp_optics.py` 进行了全链路光学与 ISP 建模：

- **50Hz / 60Hz 荧光灯卷帘快门频闪（Fluorescent Flicker Banding）**：电感镇流器荧光灯以两倍电网频率（100Hz / 120Hz）发光调制，同时伴随荧光粉余辉色温漂移；配合早期 CMOS 传感器长达 $30\text{--}60\,\text{ms}$ 的逐行卷帘曝光时间（`mobile_readout_ms`），在画面上形成明暗与冷暖交替的水平/垂直条纹（通过 `mobile_flicker_mode`、`mobile_flicker_intensity`、`mobile_flicker_cycles` 控制）。
- **2700K 钨丝灯与混合光源白平衡崩溃（Tungsten AWB Failure）**：早期手机采用灰世界（Gray-World）简单均值白平衡，且蓝通道模拟增益上限极低（防止蓝色散粒噪声爆炸）。在暖黄餐厅钨丝灯或黄绿节能灯下，蓝通道增益触顶截断（`mobile_awb_mode = 1..4`），导致白色餐盘与食物呈现浓重的暖橙色或荧光黄绿色偏，高光区域更因红/绿通道先行饱和而发生不可逆的色相偏转。
- **夜景车灯衍射星芒与 CCD 垂直拖尾（Night Starburst & Vertical Smear）**：微缩塑料镜头孔径边缘衍射与镜片擦拭微划痕使夜间强点光源产生十字/多芒星耀斑（`mobile_starburst_strength`、`mobile_starburst_blades`），配合无机械快门的行间转移 CCD 垂直寄存器电荷泄漏（`mobile_ccd_smear`），形成贯穿全画幅的亮白/品红垂直光柱。

### 3. 35mm / 16mm 光化学胶片、LOMO 暗房与醋酸综合症

`01b 胶片光化学` 层将光学散景与暗房化学反应完全解耦为两个高精度物理子系统：

- **纯物理光晕与特种镜头散景（`film_optics.py`）**：严格区分三类常被混淆的高光机制——① **反光晕层失效（Remjet Halation）**：强光穿透感绿/感红层后在透明片基背面以临界角全反射回感红层，产生紧贴高光边缘的纯正红橙色光环；② **镜头鬼影眩光（Lens Flare）**：多片无镀膜玻璃多次反射并在光圈叶片处翻转成中心对称的多边形色斑；③ **传感器电荷溢出（Blooming）**：沿 CCD 垂直转移列溢出。同时内置 **Meyer Trioplan 肥皂泡亮边散景**、**Petzval 匹兹瓦尔旋焦猫眼散景**与心形/星形光圈片。
- **20 款胶片乳剂与暗房交叉冲洗（`photochemical_film.py`）**：涵盖电影负片（Kodak Vision3 500T 5219 / 250D 5207 / 50D 5203、Double-X 5222）、反转片（Kodachrome 64、Ektachrome 100D、Velvia 50）、消费负片（Portra 400、Cinestill 800T）及 LOMO Purple / 宝丽来即时显影，支持 **E-6 正片负冲（C-41）**、**C-41 负片正冲（E-6）**、**银留存（Bleach Bypass）** 与 **四代光学印片链路（`OCN -> IP -> IN -> 2383`）**。
- **自催化醋酸综合症（Vinegar Syndrome）与 FIAF 档案修复**：模拟三醋酸纤维素片基在温湿度作用下的自催化水解脱乙酰反应（自由酸浓度指数攀升 $\rightarrow$ 片基非线性几何收缩 $\rightarrow$ 齿孔撕裂步进失锁 $\rightarrow$ 偶氮青/品红染料选择性褪色 $\rightarrow$ 乳剂沟道鼓包与结晶析出），并提供湿门折射率匹配液（Wet-Gate）物理消划痕与 FIAF 四级可逆数字修复流水线。

### 4. CRT 显像管物理、相机翻拍摩尔纹与强磁铁消磁实验

`05 显示设备` 层不仅模拟电子束与荧光粉，还还原了显像管外围的高压电路、电磁场偏转与光学重拍几何：

- **Sony Trinitron KV-1214 (1978) 4K 级物理还原**：支持高达 $3840\times 2160$ 的子像素级荫栅（Aperture Grille）垂直连续荧光条、单枪三束（One-Gun Three-Beam）汇聚特性、深色玻璃面板内反射光晕与高压包（EHT）随亮场负载跌落的画面呼吸膨胀。
- **强磁铁洛伦兹力旋涡与 PTC 消磁线圈**：支持在屏幕任意归一化坐标 $(x,y)$ 放置钕铁硼强磁铁并调节磁极距离 $z$。电子束在洛伦兹力 $\mathbf{F} = -e(\mathbf{v} \times \mathbf{B})$ 作用下产生带旋涡扭曲的几何位移，且因错位轰击相邻 RGB 荧光条而产生剧烈的彩虹色纯斑（Purity Ring）；触发消磁开关后，模拟 50/60Hz 交变衰减磁场与 PTC 热敏电阻热惯性，完整还原画面震荡发散再恢复纯净的消磁过程。
- **相机翻拍 CRT 屏幕（Camera-off-CRT Reshoot）**：当使用数码相机或摄像机直接拍摄 CRT 屏幕时，相机快门时间与 CRT 场扫描周期拍频形成移动的水平快门滚道黑带（Shutter Roll Bar），而镜头透视投影网格与显像管荫罩间距在非整数倍缩放下产生真实的彩色空间拍频摩尔纹（Moiré Pattern）。

## 五、交互式 WebUI 工作台指南

启动 WebUI 后，界面采用经典《The Matrix》黑底荧光绿终端主题，核心功能区布局如下：

| 工作区模块 | 位置与入口 | 核心操作与工作流说明 |
| :--- | :--- | :--- |
| **分类预设库与档案室** | 左侧边栏 | 支持按 **「整套配置」**、**「古早数字/手机/流媒体」**、**「胶片/LOMO」**、**「自定义预设」** 或单层设备档案（摄像机 / 记录介质 / 播放设备 / 显示设备）一键切换；点击「保存预设」可将当前调校固化至 `profiles/` |
| **实时监视器与 A/B 对比** | 中央主视窗 | 支持多档代理分辨率（`0.25x` 极速至 `4.00x` 4K 荫罩级）、拖拽时间轴实时预览；勾选 **「A/B 对比」** 可拖动绿色垂直分割线实时对比原始素材与仿真画面 |
| **❖ 分屏 / 分框 / 文字工具栏** | 预览窗正下方第一栏 | 提供 **「⊞ 2×2 四分屏」** 与 **「↖ 左上 1 / ↗ 右上 2 / ↙ 左下 3 / ↘ 右下 4」** 单角全屏检视的一键瞬时切换；内置 15 种实验室专题四分屏模式（如网络流媒体传播实验室、光化学胶片冲印、CRT 强磁铁消磁、VHS 代际损失等）；支持 **「文字：不要文字（纯净）」** 一键消除全画面角标与缓冲文字；提供 **「导出当前帧」**、**「批量四连拍 (4张)」** 与 **「四连拍 ZIP」** 打包下载 |
| **W–T 光学变焦推杆** | 预览窗下方第二栏 | 复刻手持 DV 电动变焦摇杆（`◀ W 广角` / `T 放大 ▶`），支持 `1.0x–20.0x` 连续平滑变焦与 6 种自动推拉脚本，实时联动景深、球差、呼吸效应与推拉径向模糊 |
| **伴音试听与声画故障示波器** | 预览窗下方第三栏 | 点击 **「▶ 试听伴音 (3s)」** 即可在线生成并播放与当前画面时基抖动、磁带失落、磁头切换严格同步的退化伴音，支持勾选「原始伴音对比」或一键导出 3 秒试听段 / 全片 WAV |
| **广播级实时示波与校准台** | 预览窗下方折叠抽屉 | 复刻 **Tektronix 1740A** 波形监视器（复合 IRE 含 HBI/色同步、LUMA 亮度、RGB 分量、行选切 Line Select）、**NTSC/PAL 矢量示波器（Vectorscope）**（含 75% 彩条靶框与 I/Q 轴）以及 **0–6.5 MHz 射频功率谱密度（PSD）与 2D 差值热力图** |
| **Blender 式节点图合成器** | 底部「节点图编辑器」抽屉 | 勾选顶部「用节点图渲染」即可切换至任意有向无环图（DAG）模式；支持自由添加 9 类物理阶段节点与混合节点、拖拽端口连线、多级串联（如多台磁带机连续翻录）、**`Ctrl+G` 打包节点组（Node Group）**、**`Tab` 进入/退出子画布编辑** 与 **`Shift+D` 复制节点** |
| **关键帧时间线与机房环境** | 底部抽屉与顶部工具按钮 | 支持为任意连续物理参数打点设置关键帧（`linear` / `smooth` / `step` 插值 + LFO 周期调制）；支持调节机房环境温湿度（触发磁带结露保护与粘滞）及设备个体老化公差种子（Unit Identity） |

## 六、高性能分级缓存与极速渲染引擎

为了在纯 CPU 环境下依然保持流畅的交互帧率，引擎实现了多重无损计算加速：

1. **阶段级 DAG 哈希缓存（Stage-Level Prefix Caching）**：渲染器 `SessionRenderer` 对信号链每一阶段的参数与上游输入计算确定性签名。当你仅调节后端参数（如 `05 显示设备` 的 CRT 扫描线或磁铁位置）时，前端耗时的 `01 成像`、`02 编码`、`03 介质` 与 `04 解码` 阶段直接命中内存缓存，预览响应缩短至毫秒级。
2. **8×8 DCT 矩阵乘法向量化（Vectorized Separable Matrix Multiply）**：在多代 JPEG 磨损与 MPEG-1 / DV25 编解码中，采用预缓存正交余弦基矩阵 $C$ 的三维张量乘法 `(C @ blocks) @ C.T` 替代通用 `scipy.fft.dctn` 调度，将多代 8×8 块变换提速 5–10 倍且保持浮点数值完全等价。
3. **2×2 四分屏单角按需渲染**：在「❖ 分屏 / 分框 / 文字」工具栏中点击 `↖ 左上 1` 至 `↘ 右下 4` 任意单角按钮时，后端仅执行该象限对应的单条物理子链路（耗时降为四分屏的 $1/4$），方便在调参时高速预览并在导出时切换回完整四分屏或一键四连拍。
4. **静态源帧预缩放缓存**：内置测试图与上传的静态图片按目标分辨率自动缓存，省去每次拖动滑块时的重复图像重采样开销。

## 七、Python API 与脚本调用示例

除了通过 WebUI 交互操作外，你也可以在 Python 脚本或批处理流水线中直接调用 `atsim` 的信号链或节点图引擎：

```python
import numpy as np
from PIL import Image
from atsim.core.profiles import default_chain_config
from atsim.core.pipeline import FrameBuffer, RenderContext
from atsim.core.standards import NTSC
from atsim.stages import build_default_chain

# 1. 构建默认全链路配置并开启旧手机荧光灯频闪 + 多代 JPEG 磨损
cfg = default_chain_config()
cfg["camera"]["mobile_isp_enabled"] = 1
cfg["camera"]["mobile_device_profile"] = 1       # Nokia N73 (2006)
cfg["camera"]["mobile_flicker_mode"] = 1         # 50Hz 荧光灯卷帘频闪
cfg["camera"]["mobile_flicker_intensity"] = 0.65
cfg["camera"]["jpeg_decay_enabled"] = 1          # 开启多代 JPEG 磨损
cfg["camera"]["jpeg_quality"] = 22.0
cfg["camera"]["jpeg_generations"] = 5

# 2. 载入输入图像（归一化至 0.0 - 1.0 float32 RGB）
img = Image.open("input.png").convert("RGB")
rgb = np.asarray(img, dtype=np.float32) / 255.0

# 3. 执行全链路物理仿真
chain = build_default_chain(NTSC)
ctx = RenderContext(
    standard=NTSC,
    width=rgb.shape[1],
    height=rgb.shape[0],
    fps=29.97,
    seed=42,
    total_frames=1,
)
fb = FrameBuffer(rgb=rgb.copy(), frame_index=0, time_sec=0.0)
out_fb = chain.process(fb, cfg, ctx)

# 4. 保存渲染结果
out_u8 = np.clip(out_fb.rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
Image.fromarray(out_u8).save("output_simulated.png")
```

## 八、工程目录结构

```text
Virtual-Media-Lab/
├── server.py                              # WebUI 进程守护管理脚本 (start/stop/restart/status/logs)
├── server.bat / run_webui.bat             # Windows 一键启动脚本
├── settings.json                          # 默认工程配置快照
├── PHYSICAL_MODELING_AND_WEAR_GUIDE.md    # 频闪、钨丝灯白平衡与多代 JPEG 磨损物理建模专题说明
├── ROADMAP_BACKLOG.md                     # 媒介考古学全模块演进路线图与架构设计文档
├── profiles/                              # 用户保存的自定义整套/单层设备预设 (.json)
├── atsim/                                 # 核心物理仿真引擎包
│   ├── core/
│   │   ├── standards.py                   # NTSC / PAL / SECAM 行频、副载波与 MHz↔像素换算基准
│   │   ├── params.py                      # 参数元数据系统（单位、物理合理区间、艺术区间、来源说明）
│   │   ├── profiles.py                    # 9 层完整参数表、硬件档案库与整套预设配置
│   │   ├── rng.py                         # 确定性随机数派生、OU 阻尼过程、时空相关噪声
│   │   ├── signal.py                      # YIQ 变换、复合调制/解调、零相位 FFT 带通、逐行三次插值位移
│   │   ├── pipeline.py                    # Stage / Chain / FrameBuffer / RenderContext 核心流水线
│   │   ├── keyframes.py                   # 关键帧曲线插值 (Linear/Smooth/Step) 与 LFO 调制器
│   │   ├── environment.py                 # 机房环境温湿度、气压、电网波动与结露保护状态机
│   │   └── units.py                       # 设备个体公差身份 (Unit Identity) 确定性微扰生成
│   ├── stages/                            # 9 层主干信号链阶段封装
│   │   ├── camera.py                      # 01 成像系统阶段
│   │   ├── film.py                        # 01b 胶片光化学与暗房阶段
│   │   ├── format.py                      # 02 模拟复合信号编码阶段
│   │   ├── codec.py                       # 02b 数字编解码与网络流媒体阶段
│   │   ├── connector.py                   # 02c 物理接口、跳线盘、切换台与供电母线阶段
│   │   ├── tape.py                        # 03 磁带介质、走带伺服与多代翻录阶段
│   │   ├── player.py                      # 04 读取解码、TBC、同步锁相与档案修复阶段
│   │   ├── display.py                     # 05 CRT 显示器、相机重拍、强磁铁与多画面墙阶段
│   │   └── overlay.py                     # 06 硬件字幕机、总控播出与媒介生态演化阶段
│   ├── codecs/                            # 专项物理模型与硬件子系统实现
│   │   ├── mobile_isp_optics.py           # 旧手机/iPhone ISP、荧光灯频闪、钨丝灯白平衡、多代 JPEG 磨损
│   │   ├── network_streaming_era.py       # 1999-2020 流媒体演进、FLV/H.263、流媒体多代 JPEG 磨损
│   │   ├── legacy_digital_media.py        # 千禧年早期数字媒介全链路与 2×2 四分屏对比实验室
│   │   ├── film_optics.py                 # 纯物理反光晕层 (Halation)、镜头眩光、肥皂泡/旋焦散景
│   │   ├── photochemical_film.py          # 20 款胶片乳剂、交叉冲洗、光学印片与醋酸综合症
│   │   ├── film_telecine.py               # Rank Cintel 3:2 Pulldown 飞点胶转磁与编织抖动
│   │   ├── crt.py / crt_reshoot.py        # Trinitron/荫罩显像管、KV-1214 4K、强磁铁消磁、相机重拍屏幕
│   │   ├── dv.py / mpeg1.py               # MiniDV DV25 与 VCD MPEG-1 离散余弦变换编解码
│   │   ├── rf.py / satellite_ird.py       # 射频多径重影、TV-DX 远距离串台、C波段模拟卫视与 DVB-S
│   │   ├── tv_transmitter_chain.py        # 大功率模拟电视发射机 VSB 残留边带与速调管非线性
│   │   ├── transport_servo.py             # 磁带机机械穿带、主导轴/磁鼓伺服与暂停静帧磁迹磨损
│   │   ├── edit_controller_nle.py         # BVE-9100 线性编辑控制器与早期 2000s 非编时间线
│   │   ├── sync_clock.py                  # LM1881 同步分离、Adler 注入锁定方程与 Pulse Cross
│   │   ├── transfer_workstation.py        # 跨介质多代转录工作站与代际谱系追踪
│   │   ├── inversion_calibration_lab.py   # 退化参数逆向反演与三级 FADGI 误差预算校准实验室
│   │   ├── archival_archaeology.py        # 档案修复室（原始劣化带 vs 档案母带 vs 过度降噪）
│   │   ├── master_playout.py              # 电视台总控播出通道 (PGM/Clean/PVW/DSK/As-Run Log)
│   │   ├── playout_automation_v2.py       # 事件驱动播出自动化 v2 与故障因果审计报告
│   │   ├── media_ecosystem_evolution.py   # 1975-2035 跨时代视听媒介生态演化模拟器
│   │   ├── scopes.py                      # Tektronix 1740A 波形/矢量示波器与射频 PSD 频谱仪
│   │   └── audio.py                       # 画面帧同步模拟/磁带/VCD/RF 伴音退化与 WAV 合成器
│   ├── graph/                             # Blender 式有向无环图 (DAG) 节点与节点组执行引擎
│   ├── io/                                # FFmpeg 视频流封装、合成测试图源与高速 JPEG/PNG 编码
│   ├── render/                            # 多级前缀哈希缓存渲染器与参数扫描对比矩阵
│   └── web/                               # 零外部框架依赖的 HTTP API 服务端与黑绿主题前端
└── tools/
    ├── verify.py                          # 全模块物理机理与契约回归验证套件 (--quick 快速模式)
    ├── selftest.py                        # 引擎确定性、帧间平滑度与缓存一致性自检
    ├── measure_response.py                # 模拟复合通道频率响应标定测量工具
    └── testimage.py                       # 标准测试场景与预设对比图生成脚本
```

## 九、常见问题排查（FAQ）

1. **在其他电脑上启动时报 `ModuleNotFoundError: No module named 'fontTools'` 怎么办？**
   - **最快一行解决**：在终端执行 `pip install fonttools` 即可立即启动。
   - **根因与代码级免依赖修复**：`atsim/codecs/fonts.py` 原本在顶层直接导入了 `from fontTools.ttLib import TTFont` 用于解析 TrueType 字体的 `cmap` 与 `hmtx` 表。最新更新的 `atsim/codecs/fonts.py` 已将 `fontTools` 改为可选依赖（未安装时自动回退至 `Pillow` 的 `ImageFont` 度量），并补齐了跨机器的 Windows（`C:\Windows\Fonts\consola.ttf` / `simsun.ttc` 等）、macOS 与 Linux 系统字体回退链，即使目标电脑没有安装 `fonttools` 或自定义字体文件也能直接启动运行。
2. **为什么选择了分屏预设并将文字设为「不要文字（纯净）」，画面中间之前还会有 `BUFFERING...`？**
   在早期的流媒体模式中，`BUFFERING...` 是流媒体播放器内部生成的 OSD 图层。现已实现全链路联动：当顶部工具栏或参数面板将分屏文字模式设为 **「不要文字（纯净）」**（`text=0`）时，引擎会自动穿透抑制流媒体缓冲提示文字、重传平台水印横幅与所有分屏角标，输出 100% 纯净的物理磨损画面。
3. **如何在保持画面效果完全不变的前提下获得最快的调参响应？**
   - 日常调参时将预览分辨率保持在 `0.50x 标准` 或 `0.25x 极速`；
   - 使用 `❖ 分屏 / 分框 / 文字` 栏中的 **「↖ 左上 1 ~ ↘ 右下 4」** 单角检视按钮（仅计算 1/4 链路），调好后再切回 `⊞ 2×2 四分屏` 或点击 **「批量四连拍 (4张)」** 导出高清原图；
   - 调节后端显示层（CRT、扫描线、磁铁消磁）时无需关闭前端阶段，渲染器会自动复用上游阶段缓存。


## 🔊 音频系统扩展：10 大物理音频子系统与音画同步时钟架构

本工程在 `atsim/codecs/audio.py`、`atsim/core/profiles.py`、`atsim/graph/graph.py`、`atsim/graph/executor.py` 与 WebUI (`atsim/web/server.py`, `atsim/web/static/app.js`) 中完整内置了 **10 个物理级音频子系统** 与 **全链路音画同步时钟引擎**：

| 编号 | 子系统名称 | 核心物理与工程建模机制 | 关键参数前缀 / 节点 |
| :--- | :--- | :--- | :--- |
| **01** | **麦克风与录音前端** (`MicPreampFrontendSpec`) | 8 类换能器频响（直通、动圈 SM58、电容 U87、铝带 RCA44、驻极体电脑麦、碳粒电话麦、DV 机顶麦、直板功能机针孔 MEMS 麦）、5 种指向性极坐标图（全指向/心形/超心形/8字/枪式）与离轴高频孔径衍射滚降、等效自噪声底（EIN + JFET $1/f$ 粉噪）、前置放大增益、前端 AGC 呼吸抽吸效应、运放软削波/ADC 硬削波/整型溢出折返、近讲效应、喷麦气流冲击与机身马达/手持摩擦耦合 | `audio.mic_*`, `audio.proximity_effect`, `audio.plosive_pop` |
| **02** | **旧手机扬声器与听筒** (`MobileSpeakerSimSpec`) | Thiele-Small 机电高通谐振频点 ($f_0$)、最大声压级与音圈大振幅非线性奇次/偶次谐波失真、纸盆分割振动、塑料机壳腔体箱音共振，以及听筒模式（贴耳窄带 300–3400 Hz）vs 背面免提外放（Nokia/山寨机大嗓门薄亮刺耳 + ALC 压限） | `audio.mobile_spk_*` |
| **03** | **数字采样与音频格式转换** (`DigitalSamplingSpec`) | 采样率抽取与奈奎斯特折返混叠、重采样滤波器对比（Sinc 抗混叠 / 早期声卡线性插值 / Zero-Order Hold 无抗混叠阶梯）、比特深度量化（4–24 bit）、抖动对比（无抖动截断 / RPDF / TPDF / 噪声整形 TPDF）、立体声/单声道求和/L-R 反相抵消空洞、ADC 满量程削波、采样时钟漂移 (ppm) 与孔径抖动 (Jitter) | `audio.adc_*` |
| **04** | **感知音频编码** (`PerceptualCodecSpec`) | 覆盖 MP3、AAC-LC/HE-AAC（含 SBR 高频重建伪影）、AMR-NB（3GPP 代数码激励线性预测）与流媒体低码率语音编码；基于 Bark 临界频带同时掩蔽效应分配比特（验证**同码率 $\neq$ 同听感损伤**）、建模高频空洞与水声“鸟鸣”伪影 (Birdies)、瞬态涂抹与预回声 (Pre-echo)，以及 1–8 代重复转码世代累积劣化 | `audio.pcodec_*` |
| **05** | **移动电话与网络语音** (`TelecomVoiceSpec`) | 窄带语音 (300–3400 Hz) 与高清宽带语音 (50–7000 Hz) 切换、语音活动检测 (VAD) 截字、DTX 静音抑制与舒适噪声生成 (CNG)、Gilbert-Elliott 突发丢包与丢包隐藏（PLC 零填充断字 vs 基音周期冻结机器音拖尾）、声学回声消除 (AEC) 残余与双讲抽吸、谱减法降噪音乐水声 (Musical Noise)、217 Hz GSM TDMA 射频蜂音与自适应抖动缓冲变速 | `audio.telecom_*` |
| **06** | **声学空间与麦克风位置** (`AcousticSpaceSpec`) | 三维矩形房间轴向驻波简正模 ($f_{n_x,n_y,n_z}$)、六壁镜像源早期反射与走廊平行墙颤动回声 (Flutter Echo)、墙面高频吸声系数、Sabine/Eyring 混响时间 ($RT_{60}$)、声源-麦克风距离（直达混响比 DRR + ISO 9613 空气吸收）、手捂/口袋物理遮挡闷音与典型环境底噪（小卧室/空旷走廊/教室/户外街区） | `audio.space_*` |
| **07** | **播放器与系统音频链** (`SystemAudioChainSpec`) | 操作系统混音器增益与推满削波、Windows XP KMixer (44.1→48 kHz 线性插值 SRC) 与网页 Flash Player (22.05 kHz 降采样)、声道映射故障与 **3.5mm 耳机插头地线松动虚接（L-R 差模串扰导致中置人声消失的伴奏卡拉OK效应）**、系统图形 EQ 预设、3D 环绕扩展、DMA 环形缓冲区欠载卡顿 (Buffer Underrun) 与输出设备切换爆音 | `audio.sys_*` |
| **08** | **铃声、MIDI 与移动音源** (`MobileRingtoneSynthSpec`) | Nokia 单音压电蜂鸣器方波铃声 (Monophonic Piezo Buzzer)、Yamaha MA-2/MA-3 (OPL) 4/16 和弦 FM 合成芯片、严格最大复音数限制 (1/4/16/64) 与硬件抢音截断 (Voice Stealing)、低比特 PCM 波表、音序器节拍时间量化抖动、阶梯包络以及实时输入音频转写复古铃声旋律 | `audio.ringtone_*` |
| **09** | **广播与网络平台的响度处理** (`LoudnessProcessorSpec`) | 完整实现 ITU-R BS.1770-4 / EBU R 128 K 加权综合响度测量 (Integrated LUFS)、响度范围 (LRA)、峰值因子 (Crest Factor) 与 4 倍过采样真峰值检测 (True Peak dBTP)；模拟电台/电视多段响度大战压限器 (Optimod 风格)、视频平台自动响度归一化 (-14 / -16 / -23 LUFS 惩罚削减) 与多平台二次上传重编码动态坍塌 | `audio.loudness_*` |
| **10** | **音视频同步与时钟系统** (`AVSyncClockSpec` / `AVSyncClockStage`) | 统一视频帧时间戳 (PTS) 与音频采样时钟对位、固定唇音偏移 (ms)、晶振时钟漂移 (ppm) 引发的随时间线性累积音画跑偏、视频掉帧/重复帧 vs 音频 FIFO 滑移或锁相环 PLL 变调追赶、以及磁带/非编切点瞬态（静音缺口 / 非过零点硬切爆音 / 相位跳变），将摄像、DV、磁带、广电、剪辑、流媒体与音频系统彻底收口为统一整体 | `audio.av_*`, 节点 `av_sync_clock` |


## 🖥️ 原生低内存桌面端工作站 (Native Desktop GUI)

针对浏览器 WebUI 在长时间高分辨率预览、连续示波器轮询及多窗口 Canvas 渲染时容易出现的**内存与 CPU 占用过高**问题，工程提供了与 WebUI 布局及功能 1:1 对齐的**原生桌面端工作站 (`desktop.py` / `run_desktop.bat`)**：

### 为什么桌面端能大幅降低资源占用？
1. **零浏览器开销与零 HTTP 轮询**：不再启动本地 HTTP 服务器与 Chromium/Edge 浏览器进程（直接省去 300–800 MB 浏览器 DOM/Canvas 内存）。
2. **零预览帧 PNG/JPEG 编码与解码损耗**：WebUI 每次拖动滑块都需要将帧编码为 PNG/JPEG 并通过 HTTP 传输给浏览器解码；桌面端直接在进程内将 `Renderer` / `GraphExecutor` 输出的 `uint8 (H, W, 3)` 内存数组直传给 `PIL.ImageTk.PhotoImage` 显存位图。
3. **严格受限的 LRU 帧缓存 (`max_cache_frames=2`) 与一键内存释放**：默认限制单实例 `Renderer` 与最多 2 帧阶段缓存，顶部状态栏实时显示进程 `RAM (MB)` 与 `帧耗时 (ms)`，并提供 `🧹 释放内存` 按钮一键清空所有阶段缓存与触发 `gc.collect()`。
4. **按需惰性计算 (Lazy Evaluation)**：`📊 广播示波器`（Tektronix 1740A IRE 波形、矢量示波器、0–6.5MHz 功率谱）与 `▶ 试听伴音 (3s)` 仅在展开抽屉或点击试听时计算，收起时后台零 CPU 占用。

### 启动方式
```bash
# Windows 双击一键启动：
run_desktop.bat

# 或命令行启动（默认开启低内存模式）：
python desktop.py

# 启动时直接加载指定视频或图片素材：
python desktop.py --clip path/to/video.mp4

# 运行桌面端无头自检测试：
python desktop.py --self-test
```
