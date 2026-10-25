"""
The font registry: what each typeface can draw, and what its cell is.

A **character generator is not a typesetting engine.** An OSD chip has a font ROM, a fixed cell per
character, and a position register; it draws glyphs into cells and does not measure, kern, wrap or
reflow. That distinction decides everything here, and it is not a simplification -- it is what the
apparatus actually does, and it is why camcorder overlay text looks the way it does: evenly spaced,
fixed size, never proportional.

**All three of the user's fonts are bitmap-derived**, which the inspection established rather than the
file names:

    VCR-OSD-Mono   204 glyphs, ASCII only, one advance (1200), 10 distinct x values per glyph
    Lyusung       7577 glyphs, 6763 CJK, latin 1200 and CJK 2400, 12-13 distinct x values
    A-POS        22169 glyphs, 20902 CJK, latin 504 and CJK 1008, 11 distinct x values, 48% empty

Two facts follow and both matter:

  * **the advance ratio is exactly 1:2** in all three (1200:2400, 504:1008), so a cell is half-width for
    latin and full-width for CJK. That is the half/full-width convention of the era, and it is why a
    subtitle can mix scripts without a layout pass: every character occupies one or two half-cells;
  * **the outlines are grid-aligned**, so they must be rasterised with **nearest-neighbour** scaling. Any
    interpolation softens the pixel steps and the glyph stops looking like the character generator it came
    from. This is not a preference; it is the whole visual character of the font.

Coverage is a hard constraint the user gave, so it is recorded here and enforced: the OSD font has no CJK
at all, so Chinese in it must be reported rather than silently rendered as boxes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from fontTools.ttLib import TTFont
except ImportError:
    TTFont = None  # type: ignore[assignment,misc]


@dataclass
class FontSpec:
    """One typeface, as a character generator would see it: cells, coverage, and a file."""

    key: str
    label: str
    path: str
    #: which scripts this font can draw. Enforced, not advisory: a font with no CJK must not be
    #: silently asked for Chinese.
    scripts: tuple[str, ...]
    #: fallback paths (e.g. Linux / cross-platform equivalents matching the font's visual style & script coverage)
    fallbacks: tuple[str, ...] = ()
    #: default horizontal/vertical stretch of the character generator ROM (e.g. tall OSD, wide karaoke)
    rom_stretch_x: float = 1.0
    rom_stretch_y: float = 1.0
    #: optical phototypesetting / stroke-weight bias (-0.5 thin FangSong .. +0.5 heavy title WeiBei/CuHei)
    stroke_weight: float = 0.0
    #: horizontal serif / stroke-end flare (simulates 1980s optical title card / BiaoSong & WeiBei terminals)
    serif_flare: float = 0.0
    #: hardware character-generator dot-matrix quantization grid (e.g. 24 for 24x24 CJK ROM, 0 for native)
    rom_grid: int = 0
    #: units of advance for a half-width cell, from the font itself
    half_advance: int = 0
    full_advance: int = 0
    units_per_em: int = 0
    glyph_count: int = 0
    #: `hhea.ascent` in font units, which is where the baseline sits in a cell
    ascent: int = 0
    descent: int = 0
    covered: set[int] = field(default_factory=set)

    def ascent_px(self, cell_h: int) -> int:
        """
        Where the baseline sits, in pixels, for a cell of `cell_h`.

        The same conversion the advance uses, so the baseline and the cell width come from one source.
        A font whose ascent exceeds its em -- and all three here do to some degree -- places the baseline
        below the top of the cell by this amount, and a glyph is drawn upward from it.
        """
        upm = max(1, self.units_per_em)
        if not self.ascent:
            return int(cell_h * 0.8)
        return int(round(self.ascent * cell_h / upm))

    def can_draw(self, ch: str) -> bool:
        c = ord(ch)
        if c >= 0x80 and "cjk" not in self.scripts:
            return False
        return c in self.covered

    @property
    def missing_scripts(self) -> tuple[str, ...]:
        return tuple(s for s in ("latin", "cjk") if s not in self.scripts)

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "scripts": list(self.scripts),
                "units_per_em": self.units_per_em, "glyph_count": self.glyph_count,
                "half_advance": self.half_advance, "full_advance": self.full_advance,
                "ascent": self.ascent, "covers": len(self.covered),
                "rom_stretch_x": self.rom_stretch_x, "rom_stretch_y": self.rom_stretch_y,
                "stroke_weight": self.stroke_weight, "serif_flare": self.serif_flare,
                "rom_grid": self.rom_grid}


#: The full font registry: original 15 typefaces + 5 camcorder/VCR/LD/CCTV typefaces + 8 late-1980s
#: broadcast documentary & hardware character-generator typefaces (28 total).
#: Every entry declares cross-platform fallback paths matching its script coverage (Latin-only vs CJK+Latin)
#: and visual family (Mono, Song/Mincho, Hei/Gothic, BiaoSong, ChangSong, FangSong, KaiShu, WeiBei, 24x24 ROM)
#: so all 28 fonts are genuinely distinct and usable on both Windows and Linux.
FONTS: dict[str, FontSpec] = {
    "osd": FontSpec(
        key="osd", label="VCR OSD Mono（仅英文/数字）",
        path=r"C:\Users\Administrator\Downloads\字体\字体\VCR-OSD-Mono\VCR-OSD-Mono\VCR-OSD-MONO-1-001-1.ttf",
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\consolab.ttf"),
            os.path.expandvars(r"C:\WINDOWS\Fonts\consola.ttf"),
            os.path.expandvars(r"C:\WINDOWS\Fonts\courbd.ttf"),
            os.path.expandvars(r"C:\WINDOWS\Fonts\cour.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ),
        scripts=("latin",)),
    "lyusung": FontSpec(
        key="lyusung", label="柳宋（中英文数字）",
        path=r"C:\Users\Administrator\Downloads\字体\字体\柳宋_猫啃网\柳宋\Lyusung-210618.ttf",
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\simsun.ttc"),
            os.path.expandvars(r"C:\WINDOWS\Fonts\simkai.ttf"),
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ),
        scripts=("latin", "cjk")),
    "apos": FontSpec(
        key="apos", label="A-POS 点阵（中英文数字）",
        path=r"C:\USERS\ADMINISTRATOR\APPDATA\LOCAL\MICROSOFT\WINDOWS\FONTS\A-POS点阵修改后.TTF",
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\simsun.ttc"),
            os.path.expandvars(r"C:\WINDOWS\Fonts\simhei.ttf"),
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        ),
        scripts=("latin", "cjk")),
    "ocr_a": FontSpec(
        key="ocr_a", label="OCR-A（机器可读，OSD 经典）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\OCR-A BT.ttf"),
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\OCRAEXT.TTF"),
            os.path.expandvars(r"C:\WINDOWS\Fonts\consola.ttf"),
            "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ),
        scripts=("latin",)),
    "ocr_b": FontSpec(
        key="ocr_b", label="OCR-B（机器可读）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\OCR-B 10 BT.ttf"),
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\consola.ttf"),
            os.path.expandvars(r"C:\WINDOWS\Fonts\cour.ttf"),
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ),
        scripts=("latin",)),
    "ming": FontSpec(
        key="ming", label="Ming-Imperial（中文点阵明体）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\Ming-Imperial-1.ttf"),
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\mingliub.ttc"),
            os.path.expandvars(r"C:\WINDOWS\Fonts\simsun.ttc"),
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        ),
        scripts=("latin", "cjk")),
    "ocr_a_ext": FontSpec(
        key="ocr_a_ext", label="OCR-A 扩展（机器可读宽体）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\OCRAEXT.TTF"),
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\consolab.ttf"),
            os.path.expandvars(r"C:\WINDOWS\Fonts\consola.ttf"),
            "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        ),
        scripts=("latin",)),
    "simhei": FontSpec(
        key="simhei", label="中易黑体 SimHei（电视台粗黑字幕）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\simhei.ttf"),
        fallbacks=(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ),
        scripts=("latin", "cjk")),
    "simsun": FontSpec(
        key="simsun", label="中易宋体 SimSun（90年代电视字幕机）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\simsun.ttc"),
        fallbacks=(
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ),
        scripts=("latin", "cjk")),
    "msmincho": FontSpec(
        key="msmincho", label="MS 明朝（日系录像机/DV 点阵明体）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\MSMINCHO.TTF"),
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\simsun.ttc"),
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ),
        scripts=("latin", "cjk")),
    "msgothic": FontSpec(
        key="msgothic", label="MS 哥特（索尼/松下日系 OSD 点阵）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\msgothic.ttc"),
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\simhei.ttf"),
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        ),
        scripts=("latin", "cjk")),
    "mingliu": FontSpec(
        key="mingliu", label="细明体 MingLiU（港台录像带繁体点阵）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\mingliub.ttc"),
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\simsun.ttc"),
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ),
        scripts=("latin", "cjk")),
    "consola": FontSpec(
        key="consola", label="Consolas（现代等宽终端）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\consola.ttf"),
        fallbacks=(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ),
        scripts=("latin",)),
    "courier": FontSpec(
        key="courier", label="Courier New（打字机衬线等宽）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\cour.ttf"),
        fallbacks=(
            "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ),
        scripts=("latin",)),
    "gothic": FontSpec(
        key="gothic", label="Century Gothic（老式几何无衬线）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\GOTHIC.TTF"),
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\arialbd.ttf"),
            os.path.expandvars(r"C:\WINDOWS\Fonts\arial.ttf"),
            "/usr/share/fonts/opentype/urw-base35/URWGothic-Demi.otf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ),
        scripts=("latin",)),
    "sony_tall_osd": FontSpec(
        key="sony_tall_osd", label="Sony Handycam 窄高体（摄像机液晶寻像器 OSD）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\msgothic.ttc"),
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\simhei.ttf"),
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        ),
        rom_stretch_x=0.78, rom_stretch_y=1.18,
        scripts=("latin", "cjk")),
    "panasonic_vcr_wide": FontSpec(
        key="panasonic_vcr_wide", label="Panasonic VCR 扁宽点阵（松下录像机磁头屏显）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\simhei.ttf"),
        fallbacks=(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        ),
        rom_stretch_x=1.22, rom_stretch_y=0.92,
        scripts=("latin", "cjk")),
    "karaoke_ld_song": FontSpec(
        key="karaoke_ld_song", label="LD/VCD 卡拉OK 大粗宋（90年代伴唱大字报）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\simsun.ttc"),
        fallbacks=(
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        ),
        rom_stretch_x=1.08, rom_stretch_y=1.05,
        scripts=("latin", "cjk")),
    "karaoke_ld_hei": FontSpec(
        key="karaoke_ld_hei", label="LD/VCD 卡拉OK 综艺粗黑（KTV 双色描边字幕）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\simhei.ttf"),
        fallbacks=(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ),
        rom_stretch_x=1.10, rom_stretch_y=1.02,
        scripts=("latin", "cjk")),
    "cctv_timecode": FontSpec(
        key="cctv_timecode", label="CCTV 监控时间码（BNC 画面分割器等宽数字）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\courbd.ttf"),
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\cour.ttf"),
            os.path.expandvars(r"C:\WINDOWS\Fonts\consolab.ttf"),
            "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Bold.otf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        ),
        rom_stretch_x=0.92, rom_stretch_y=1.10,
        scripts=("latin",)),
    "docu_biaosong": FontSpec(
        key="docu_biaosong", label="80年代纪录片大标宋（片头光学重衬线印字）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\STZHONGS.TTF"),
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\simsun.ttc"),
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        ),
        rom_stretch_x=1.06, rom_stretch_y=1.02,
        stroke_weight=0.28, serif_flare=0.35,
        scripts=("latin", "cjk")),
    "docu_changsong": FontSpec(
        key="docu_changsong", label="80年代电视长宋（解说词窄长体字幕）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\simsun.ttc"),
        fallbacks=(
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        ),
        rom_stretch_x=0.80, rom_stretch_y=1.16,
        stroke_weight=0.10, serif_flare=0.20,
        scripts=("latin", "cjk")),
    "docu_cuhei": FontSpec(
        key="docu_cuhei", label="80年代专题片粗黑（章节主标题大黑体）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\simhei.ttf"),
        fallbacks=(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ),
        rom_stretch_x=1.05, rom_stretch_y=1.00,
        stroke_weight=0.35, serif_flare=0.0,
        scripts=("latin", "cjk")),
    "docu_bianhei": FontSpec(
        key="docu_bianhei", label="广播级字幕机扁黑（人物职务与地名标版）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\simhei.ttf"),
        fallbacks=(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ),
        rom_stretch_x=1.18, rom_stretch_y=0.86,
        stroke_weight=0.15, serif_flare=0.0,
        scripts=("latin", "cjk")),
    "docu_fangsong": FontSpec(
        key="docu_fangsong", label="传统铅印仿宋体（历史文献与卷首语字幕）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\simfang.ttf"),
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\simsun.ttc"),
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ),
        rom_stretch_x=0.92, rom_stretch_y=1.08,
        stroke_weight=-0.18, serif_flare=0.25,
        scripts=("latin", "cjk")),
    "docu_kaishu": FontSpec(
        key="docu_kaishu", label="传统碑刻楷体（纪实人文题跋字幕）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\simkai.ttf"),
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\simsun.ttc"),
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        ),
        rom_stretch_x=1.02, rom_stretch_y=1.04,
        stroke_weight=0.12, serif_flare=0.40,
        scripts=("latin", "cjk")),
    "docu_weibei": FontSpec(
        key="docu_weibei", label="80年代片头魏碑体（金石刀刻重骨片头字）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\STXINWEI.TTF"),
        fallbacks=(
            os.path.expandvars(r"C:\WINDOWS\Fonts\simhei.ttf"),
            os.path.expandvars(r"C:\WINDOWS\Fonts\simsun.ttc"),
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        ),
        rom_stretch_x=1.12, rom_stretch_y=0.96,
        stroke_weight=0.42, serif_flare=0.55,
        scripts=("latin", "cjk")),
    "docu_amber_cg": FontSpec(
        key="docu_amber_cg", label="早期微机汉卡 24×24 点阵宋（硬件字符发生器）",
        path=os.path.expandvars(r"C:\\WINDOWS\\Fonts\\simsun.ttc"),
        fallbacks=(
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ),
        rom_stretch_x=1.00, rom_stretch_y=1.00,
        stroke_weight=0.08, serif_flare=0.0, rom_grid=24,
        scripts=("latin", "cjk")),
}

#: which character class a codepoint belongs to, since the cell size depends on it
CJK_RANGES = ((0x3000, 0x30FF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF))


def is_full_width(ch: str) -> bool:
    """
    Whether this character takes a full cell rather than a half one.

    Determined from the ranges rather than from the font's advance, because the cell is a property of the
    character generator's ROM layout: full-width for CJK and the full-width forms, half for latin. Using
    the font's own advance would work too, but it makes the cell layout depend on which font is loaded,
    and a generator's layout does not change when you swap the ROM.
    """
    c = ord(ch)
    return any(lo <= c <= hi for lo, hi in CJK_RANGES)


#: the height at which the cell is measured.  Large enough that the rounded pixel advance is accurate to
#: better than one per cent, and small enough to be fast.
PROBE_H = 200


def _measure_cell(spec: "FontSpec") -> tuple[int, int, int]:
    """
    (half advance, full advance, ascent) in font units, measured from what the font draws.

    The probe is a run of Latin characters at `PROBE_H` pixels: its pixel width divided by the count gives the
    half-advance in pixels, and the same figure converted back to font units is what the rest of the model
    expects.  A CJK character is then drawn at twice that, which is the half/full convention all three fonts
    follow -- and it is checked rather than assumed, because a font whose CJK advance is not exactly double
    would otherwise be silently mis-laid-out.

    Falls back to the `hmtx` table only if the rasteriser cannot measure, so a font that cannot be drawn is
    still described.
    """
    from PIL import ImageFont
    upm = max(1, spec.units_per_em)
    try:
        face = ImageFont.truetype(spec.path, PROBE_H)
        probes = [("HHHHHHHH", 1), ("婚婚婚婚", 2)]
        widths = {}
        for text, cells in probes:
            try:
                w = float(face.getlength(text))
            except Exception:                                         # noqa: BLE001
                w = float(face.getsize(text)[0])
            per = w / max(1, len(text))
            widths[cells] = per
        half_px = widths.get(1, 0.0)
        full_px = widths.get(2, 0.0)
        if half_px <= 1e-6:
            raise ValueError("no latin width")
        half = int(round(half_px * upm / PROBE_H))
        # the CJK cell should be twice the latin one; where it is not, the larger of the two is used so
        # a wide glyph cannot overflow into its neighbour
        full = int(round(full_px * upm / PROBE_H)) if full_px > 1e-6 else half * 2
        if full < half * 1.5:
            full = half * 2
        try:
            cap_top = -float(face.getbbox("H", anchor="ls")[1])
        except Exception:
            cap_top = float(face.getmetrics()[0]) * 0.80
        ascent = int(round(max(1.0, cap_top) * upm / PROBE_H))
        return half, full, ascent
    except Exception:                                                 # noqa: BLE001
        # the fallback: the smallest advance that occurs more than once, doubled for full width
        if TTFont is not None:
            try:
                f = TTFont(spec.path, fontNumber=0, lazy=True)
                cmap = f.getBestCmap()
                hmtx = f["hmtx"]
                widths: dict[int, int] = {}
                for c in range(0x20, 0x7F):
                    g = cmap.get(c)
                    if g and g in hmtx.metrics:
                        widths[hmtx[g][0]] = widths.get(hmtx[g][0], 0) + 1
                half = min(widths) if widths else upm // 2
                full = half * 2
                asc = int(f["hhea"].ascent)
                f.close()
                return half, full, asc
            except Exception:                                         # noqa: BLE001
                pass
        return upm // 2, upm, int(upm * 0.8)


def load_font(key: str) -> FontSpec:
    """
    Read a font's coverage and cell metrics, tolerating a malformed table or missing `fontTools`.

    `Lyusung` declares an `OS/2` table at format 5 and truncates it, so `fontTools` raises while
    decompiling. A rasteriser only needs `cmap`, `glyf` and `hmtx`, so the font is perfectly usable while
    being unreadable for metadata -- two different questions, and a model that treats the second as fatal
    cannot use a file a user actually has.
    """
    spec = FONTS[key]
    if spec.covered:
        return spec
    if not os.path.exists(spec.path):
        win_fonts = os.environ.get("WINDIR", r"C:\Windows") + r"\Fonts"
        cands = list(spec.fallbacks)
        if "cjk" in spec.scripts:
            cands.extend([
                os.path.join(win_fonts, "simsun.ttc"),
                os.path.join(win_fonts, "simhei.ttf"),
                os.path.join(win_fonts, "msyh.ttc"),
                os.path.join(win_fonts, "msgothic.ttc"),
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            ])
        cands.extend([
            os.path.join(win_fonts, "consola.ttf"),
            os.path.join(win_fonts, "cour.ttf"),
            os.path.join(win_fonts, "lucon.ttf"),
            os.path.join(win_fonts, "arial.ttf"),
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Monaco.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ])
        for cand in cands:
            if os.path.exists(cand):
                spec.path = cand
                break
        else:
            raise FileNotFoundError(spec.path)

    if TTFont is not None:
        f = TTFont(spec.path, fontNumber=0, lazy=True)
        try:
            spec.units_per_em = int(f["head"].unitsPerEm)
            spec.glyph_count = int(f["maxp"].numGlyphs)
            try:
                hhea = f["hhea"]
                spec.descent = int(hhea.descent)
                if not spec.ascent:
                    spec.ascent = int(hhea.ascent)
            except Exception:                                         # noqa: BLE001
                spec.descent = 0
                if not spec.ascent:
                    spec.ascent = int(spec.units_per_em * 0.8)
            cmap = f.getBestCmap()
            if "cjk" not in spec.scripts:
                spec.covered = {cp for cp in cmap.keys() if cp < 0x2E80}
            else:
                spec.covered = set(cmap.keys())
            spec.half_advance, spec.full_advance, spec.ascent = _measure_cell(spec)
        finally:
            f.close()
    else:
        # Graceful Pillow-only fallback when fontTools is not installed
        spec.units_per_em = 1000
        spec.glyph_count = 65535
        spec.descent = -200
        if "cjk" not in spec.scripts:
            spec.covered = set(range(0x20, 0x2E80))
        else:
            spec.covered = set(range(0x20, 0x10000))
        spec.half_advance, spec.full_advance, spec.ascent = _measure_cell(spec)
    return spec


def coverage_report(key: str, text: str) -> tuple[list[str], list[str]]:
    """
    Which characters of `text` this font can draw, and which it cannot.

    Returned rather than assumed so the caller can report a missing glyph instead of drawing a box: a
    subtitle that silently loses its Chinese is worse than one that says it cannot.
    """
    spec = load_font(key)
    ok, missing = [], []
    for ch in text:
        if ch in "\r\n\t":
            continue
        (ok if spec.can_draw(ch) else missing).append(ch)
    return ok, sorted(set(missing))


def font_catalogue() -> list[dict]:
    """Every font, with its coverage resolved, for the UI to offer."""
    out = []
    for key in FONTS:
        try:
            spec = load_font(key)
            d = spec.to_dict()
            d["available"] = True
        except Exception as exc:                                      # noqa: BLE001
            d = FONTS[key].to_dict()
            d["available"] = False
            d["error"] = str(exc)[:80]
        out.append(d)
    return out