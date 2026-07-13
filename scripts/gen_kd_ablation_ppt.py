"""Generate the final KD ablation evidence deck.

1. Final KD choices across all 8 datasets
2. Size-matched Russian pool-source comparison
3. Same-pool gen8k quantity control
4. Five evidence-backed conclusions

Same visual language as gen_kd_ppt.py (clean pastel-panel theme, English).
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

FONT_BODY = "Segoe UI"
FONT_MONO = "Consolas"

C_BG        = RGBColor(0xFF, 0xFF, 0xFF)
C_TITLE     = RGBColor(0x0F, 0x17, 0x2A)
C_BODY      = RGBColor(0x33, 0x41, 0x55)
C_MUTED     = RGBColor(0x64, 0x74, 0x8B)
C_EYEBROW   = RGBColor(0x25, 0x63, 0xEB)
C_RED       = RGBColor(0xDC, 0x26, 0x26)
C_GREEN     = RGBColor(0x05, 0x96, 0x69)
C_AMBER     = RGBColor(0xB4, 0x5A, 0x09)

PANEL_BLUE   = (RGBColor(0xDE, 0xEA, 0xFB), RGBColor(0x1D, 0x4E, 0xD8), RGBColor(0x3B, 0x82, 0xF6))
PANEL_GREEN  = (RGBColor(0xD6, 0xF2, 0xE4), RGBColor(0x04, 0x78, 0x57), RGBColor(0x10, 0xB9, 0x81))
PANEL_SLATE  = (RGBColor(0xEE, 0xF2, 0xF7), RGBColor(0x1E, 0x29, 0x3B), RGBColor(0x64, 0x74, 0x8B))
PANEL_AMBER  = (RGBColor(0xFE, 0xF3, 0xC7), RGBColor(0xB4, 0x5A, 0x09), RGBColor(0xF5, 0x9E, 0x0B))
PANEL_RED    = (RGBColor(0xFD, 0xE4, 0xE4), RGBColor(0xDC, 0x26, 0x26), RGBColor(0xEF, 0x44, 0x44))

C_HEADER_ROW = RGBColor(0xF1, 0xF5, 0xF9)
C_ROW_LINE   = RGBColor(0xE2, 0xE8, 0xF0)
C_WHITE      = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)
LAYOUT = prs.slide_layouts[6]


def add_slide():
    s = prs.slides.add_slide(LAYOUT)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = C_BG
    return s


def rrect(slide, x, y, w, h, fill, line=None, radius=0.06, line_w=1.0):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(x), Inches(y), Inches(w), Inches(h))
    try:
        sh.adjustments[0] = radius
    except Exception:
        pass
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line:
        sh.line.color.rgb = line
        sh.line.width = Pt(line_w)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def rect(slide, x, y, w, h, fill, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line:
        sh.line.color.rgb = line
        sh.line.width = Pt(0.75)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def _spacing(run, spc):
    rPr = run._r.get_or_add_rPr()
    rPr.set('spc', str(spc))


def tx(slide, text, x, y, w, h, size=18, bold=False, color=C_BODY,
       align=PP_ALIGN.LEFT, italic=False, font=FONT_BODY, spacing=None,
       anchor=None, wrap=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    if anchor:
        tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = Pt(2)
    tf.margin_bottom = Pt(2)
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.name = font
    r.font.color.rgb = color
    if spacing:
        _spacing(r, spacing)
    return tb


def eyebrow(slide, text, x=0.55, y=0.42):
    tx(slide, text.upper(), x, y, 11.0, 0.35, size=13, bold=True,
       color=C_EYEBROW, spacing=220, font=FONT_BODY)


def title(slide, text, x=0.55, y=0.74, size=28, w=12.2):
    tx(slide, text, x, y, w, 0.75, size=size, bold=True, color=C_TITLE)


def data_table(slide, x, y, col_w, headers, rows, size=13, header_size=11.5,
               row_h=0.48, highlight_idx=None, highlight_color=None,
               first_col_bold=True, first_col_color=None, font=FONT_BODY,
               left_align_cols=(2,), cell_color_fn=None):
    col_x = [x]
    for w in col_w[:-1]:
        col_x.append(col_x[-1] + w)
    total_w = sum(col_w)
    rrect(slide, x, y, total_w, 0.42, C_HEADER_ROW, radius=0.08)
    for h, cx, w in zip(headers, col_x, col_w):
        tx(slide, h, cx + 0.08, y + 0.04, w - 0.16, 0.34, size=header_size, bold=True,
           color=C_MUTED, align=PP_ALIGN.CENTER, spacing=30)
    yy = y + 0.42
    for i, row in enumerate(rows):
        bg = highlight_color if (highlight_idx == i and highlight_color) else C_WHITE
        rrect(slide, x, yy, total_w, row_h, bg, line=C_ROW_LINE, radius=0.03)
        for j, (cell, cx, w) in enumerate(zip(row, col_x, col_w)):
            bold0 = first_col_bold and j == 0
            col0 = (first_col_color or C_TITLE) if j == 0 else C_BODY
            if cell_color_fn:
                override = cell_color_fn(i, j, cell)
                if override:
                    col0 = override
                    bold0 = True
            al = PP_ALIGN.LEFT if j in left_align_cols else PP_ALIGN.CENTER
            tx(slide, str(cell), cx + 0.1, yy, w - 0.2, row_h, size=size, bold=bold0,
               color=col0, align=al, anchor=MSO_ANCHOR.MIDDLE, font=font)
        yy += row_h
    return yy


# ══════════════════════════════════════════════════════════════
#  SLIDE 1 — Cover
# ══════════════════════════════════════════════════════════════
s = add_slide()
eyebrow(s, "ABSA-QUAD  ·  Final KD Evidence", y=1.6)
tx(s, "What Actually Controls KD Gain?", 0.55, 2.1, 12.2, 1.0,
   size=44, bold=True, color=C_TITLE)
tx(s, "Teacher × Pool × Quality–Coverage Trade-off", 0.55, 3.05, 12.2, 0.65,
   size=26, color=C_EYEBROW, bold=True)
rect(s, 0.6, 3.95, 1.7, 0.05, C_EYEBROW)
tx(s, "Final ablation synthesis · frozen submission · 2026-07-13",
   0.55, 4.15, 12.2, 0.5, size=16, color=C_MUTED, italic=True)


# ══════════════════════════════════════════════════════════════
#  SLIDE 2 — KD experiments across all 8 datasets
# ══════════════════════════════════════════════════════════════
s = add_slide()
eyebrow(s, "KD Experiments")
title(s, "All 8 Datasets, Simplified")

headers = ["DATASET", "TEACHER", "SILVER SOURCE", "BASELINE", "BEST KD", "Δ", "ADOPTED"]
col_w = [1.65, 1.45, 3.5, 1.25, 1.55, 1.45, 1.35]
rows = [
    ("eng_laptop", "DeepSeek", "Real · n=8,578 (Amazon laptop)", "0.3254", "0.4141", "+8.9pp", "Submitted"),
    ("rus_restaurant", "DeepSeek", "Generated 4k · cap 3,000", "0.5267", "0.6162", "+9.0pp", "Submitted"),
    ("ukr_restaurant", "DeepSeek", "Translated · n=6,551 (RU→UK)", "0.5068", "0.5560", "+4.9pp", "Submitted"),
    ("zho_restaurant", "DeepSeek", "Real n=5,000 (dianping); Gen n=4,930 (superseded)", "0.4745", "0.5579", "+8.3pp", "Submitted"),
    ("tat_restaurant", "qwen (local)", "Generated · n=3,986 (native Tatar)", "0.4821", "0.5327", "+5.1pp", "Submitted"),
    ("zho_laptop", "qwen/DeepSeek", "Gen n=4,923; Real n=1,673", "0.4487", "0.4454 / 0.4264", "−0.3~−2.2pp", "Rejected"),
    ("jpn_hotel", "DeepSeek/qwen", "Generated · n=4,756 (no real corpus)", "0.4466", "0.4281", "−1.9pp", "Rejected"),
    ("eng_restaurant", "DeepSeek/qwen", "Gen n=4,974; Real n=5,000 (Yelp)", "0.6554", "0.6013 / 0.4190", "−5.4~−23.6pp", "Rejected"),
]

def _adopted_color(i, j, cell):
    if j == 6:
        return PANEL_GREEN[1] if cell == "Submitted" else C_RED
    return None

bottom = data_table(s, 0.52, 1.62, col_w, headers, rows, highlight_idx=1,
                     highlight_color=PANEL_GREEN[0], row_h=0.52, size=11.5,
                     cell_color_fn=_adopted_color)

tx(s, "Five datasets adopt KD. Source labels describe the selected arm, not a universal ranking: "
      "zho_restaurant prefers real reviews, while rus_restaurant prefers a generated batch.",
   0.55, bottom + 0.18, 12.2, 0.55, size=12, italic=True, color=C_MUTED)


# ══════════════════════════════════════════════════════════════
#  SLIDE 3 — Pool -> Silver -> Training pipeline
# ══════════════════════════════════════════════════════════════
s = add_slide()
eyebrow(s, "Ablation Design")
title(s, "Size-Matched Pool Source Comparison (rus_restaurant)")

headers2 = ["CONTROL / RESULT", "REAL4K", "TRANSLATED4K", "GENERATED4K"]
col_w2 = [4.6, 2.55, 2.55, 2.55]
rows2 = [
    ("Pool size (unlabeled text)", "4,000", "3,466", "4,000"),
    ("DeepSeek-labeled silver", "4,000", "3,466", "4,000"),
    ("Silver used (cap=3000)", "3,000", "3,000", "3,000"),
    ("+ gold (fixed)", "1,240", "1,240", "1,240"),
    ("Final training rows", "4,240", "4,240", "4,240"),
    ("5-seed ensemble cF1", "0.5586", "0.5761", "0.6162"),
]
data_table(s, 0.52, 2.0, col_w2, headers2, rows2, row_h=0.62, size=14.5, header_size=13)


# ══════════════════════════════════════════════════════════════
#  SLIDE 4 — Same-pool quantity control
# ══════════════════════════════════════════════════════════════
s = add_slide()
eyebrow(s, "Quantity Control")
title(s, "Same gen8k Silver: Reducing to 3,000 Does Not Help")

headers3 = ["PAIRED EVIDENCE", "FULL GEN8K (7,988)", "CAP 3,000"]
col_w3 = [2.6, 4.85, 4.85]
rows3 = [
    ("Seed 42", "0.5286", "0.5123"),
    ("Seed 123", "0.5648", "0.5584"),
    ("Paired mean", "0.5467", "0.5353"),
    ("Δ vs full", "—", "−1.14pp (gate failed)"),
]
data_table(s, 0.52, 2.0, col_w3, headers3, rows3, row_h=0.85, size=13, header_size=12)


# ══════════════════════════════════════════════════════════════
#  SLIDE 5 — Final synthesis
# ══════════════════════════════════════════════════════════════
s = add_slide()
eyebrow(s, "Final Synthesis")
title(s, "Five Conclusions Supported by the Full Ablation Matrix")

conclusions = [
    ("1", "Complementarity decides whether KD helps", "Silver must add coverage or priors missing from gold/base.", PANEL_BLUE),
    ("2", "Teacher quality is domain-dependent", "Qwen is competitive in many domains but collapses on eng_restaurant.", PANEL_AMBER),
    ("3", "Pool source has no universal ranking", "zho: real wins; rus: generated wins under size matching.", PANEL_GREEN),
    ("4", "rus gen4k is a batch-quality effect", "Same-pool reduction failed; the 4k advantage is not a generic cap effect.", PANEL_RED),
    ("5", "Best silver balances quality and coverage", "More is not automatically better; no universal cap is supported.", PANEL_SLATE),
]
for i, (num, heading, body, palette) in enumerate(conclusions):
    y = 1.52 + i * 1.05
    rrect(s, 0.62, y, 12.05, 0.85, palette[0], radius=0.05)
    rrect(s, 0.82, y + 0.13, 0.60, 0.58, palette[2], radius=0.10)
    tx(s, num, 0.82, y + 0.13, 0.60, 0.58, size=19, bold=True, color=C_WHITE,
       align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    tx(s, heading, 1.66, y + 0.09, 4.25, 0.62, size=16.5, bold=True,
       color=palette[1], anchor=MSO_ANCHOR.MIDDLE)
    tx(s, body, 6.12, y + 0.10, 6.00, 0.55, size=14.2, color=C_BODY,
       anchor=MSO_ANCHOR.MIDDLE)


out = "/home/g24tka12/ABSA-QUAD/docs/KD_ablation_final_20260713.pptx"
prs.save(out)
print(f"Saved -> {out}  ({len(prs.slides)} slides)")
