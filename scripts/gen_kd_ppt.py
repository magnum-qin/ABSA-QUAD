"""Generate PPT: V23 KD vs Other Data Augmentation — clean pastel-panel theme (English)."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ── Fonts ──────────────────────────────────────────────────────
FONT_BODY = "Segoe UI"
FONT_MONO = "Consolas"

# ── Palette (matching reference design) ────────────────────────
C_BG        = RGBColor(0xFF, 0xFF, 0xFF)
C_TITLE     = RGBColor(0x0F, 0x17, 0x2A)   # near-black title
C_BODY      = RGBColor(0x33, 0x41, 0x55)   # slate body
C_MUTED     = RGBColor(0x64, 0x74, 0x8B)   # muted grey
C_EYEBROW   = RGBColor(0x25, 0x63, 0xEB)   # blue eyebrow
C_CODE      = RGBColor(0x1D, 0x4E, 0xD8)   # code token blue
C_RED       = RGBColor(0xDC, 0x26, 0x26)
C_GREEN     = RGBColor(0x05, 0x96, 0x69)
C_AMBER     = RGBColor(0xB4, 0x5A, 0x09)

# panels: (bg, heading, dot/accent)
PANEL_BLUE   = (RGBColor(0xDE, 0xEA, 0xFB), RGBColor(0x1D, 0x4E, 0xD8), RGBColor(0x3B, 0x82, 0xF6))
PANEL_GREEN  = (RGBColor(0xD6, 0xF2, 0xE4), RGBColor(0x04, 0x78, 0x57), RGBColor(0x10, 0xB9, 0x81))
PANEL_PURPLE = (RGBColor(0xEC, 0xE6, 0xFB), RGBColor(0x6D, 0x28, 0xD9), RGBColor(0x8B, 0x5C, 0xF6))
PANEL_SLATE  = (RGBColor(0xEE, 0xF2, 0xF7), RGBColor(0x1E, 0x29, 0x3B), RGBColor(0x64, 0x74, 0x8B))
PANEL_AMBER  = (RGBColor(0xFE, 0xF3, 0xC7), RGBColor(0xB4, 0x5A, 0x09), RGBColor(0xF5, 0x9E, 0x0B))
PANEL_RED    = (RGBColor(0xFD, 0xE4, 0xE4), RGBColor(0xDC, 0x26, 0x26), RGBColor(0xEF, 0x44, 0x44))

C_HEADER_ROW = RGBColor(0xF1, 0xF5, 0xF9)   # table header grey
C_ROW_LINE   = RGBColor(0xE2, 0xE8, 0xF0)
C_WHITE      = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)
LAYOUT = prs.slide_layouts[6]


# ── Helpers ────────────────────────────────────────────────────

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


def _add_runs(p, text, body_color, size, bold=False, italic=False):
    """Split text on backticks -> code tokens rendered monospace/blue."""
    parts = text.split('`')
    for i, part in enumerate(parts):
        if part == '':
            continue
        r = p.add_run()
        r.text = part
        r.font.size = Pt(size)
        r.font.italic = italic
        if i % 2 == 1:  # inside backticks = code
            r.font.name = FONT_MONO
            r.font.color.rgb = C_CODE
            r.font.bold = True
        else:
            r.font.name = FONT_BODY
            r.font.color.rgb = body_color
            r.font.bold = bold


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


def title(slide, text, x=0.55, y=0.74, size=30, w=12.2):
    tx(slide, text, x, y, w, 0.95, size=size, bold=True, color=C_TITLE)


def badge(slide, text, x, y, w, h, bg, fg=C_WHITE, size=11):
    sh = rrect(slide, x, y, w, h, bg, radius=0.5)
    tf = sh.text_frame
    tf.word_wrap = False
    tf.margin_left = 0; tf.margin_right = 0
    tf.margin_top = 0; tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = True
    r.font.color.rgb = fg; r.font.name = FONT_BODY
    _spacing(r, 80)
    return sh


def bullets(slide, items, x, y, w, accent, size=14, body=C_BODY, gap=7, lh=1.04):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(3))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0; tf.margin_right = 0; tf.margin_top = 0; tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        p.space_before = Pt(0)
        p.line_spacing = lh
        dot = p.add_run(); dot.text = "•  "
        dot.font.size = Pt(size); dot.font.bold = True
        dot.font.color.rgb = accent; dot.font.name = FONT_BODY
        _add_runs(p, item, body, size)
    return tb


def panel_head(slide, x, y, w, text, accent, size=16):
    tx(slide, text, x, y, w, 0.45, size=size, bold=True, color=accent)


# ══════════════════════════════════════════════════════════════
#  SLIDE 1 — Cover
# ══════════════════════════════════════════════════════════════
s = add_slide()
eyebrow(s, "ABSA-QUAD  ·  Group Meeting", y=1.45)
tx(s, "V23 Knowledge Distillation", 0.55, 1.95, 12.2, 1.0,
   size=46, bold=True, color=C_TITLE)
tx(s, "vs. Other Data Augmentation Methods", 0.55, 2.95, 12.2, 0.7,
   size=28, color=C_EYEBROW, bold=True)
rect(s, 0.6, 3.95, 1.7, 0.05, C_EYEBROW)
tx(s, "Comparison & Root-Cause Analysis", 0.55, 4.18, 12.2, 0.5,
   size=18, color=C_MUTED, italic=True)

# dataset chips
chip_labels = ["eng_laptop", "rus", "ukr", "zho_laptop"]
cx = 0.6
for lab in chip_labels:
    w = 0.45 + len(lab) * 0.115
    sh = rrect(s, cx, 5.0, w, 0.5, PANEL_BLUE[0], radius=0.5)
    tf = sh.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_top = 0; tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = lab
    r.font.size = Pt(14); r.font.name = FONT_MONO; r.font.bold = True
    r.font.color.rgb = PANEL_BLUE[1]
    cx += w + 0.25


# ══════════════════════════════════════════════════════════════
#  SLIDE 2 — Background
# ══════════════════════════════════════════════════════════════
s = add_slide()
eyebrow(s, "Background")
title(s, "The eng_laptop Sparsity Problem")

tx(s, "eng_laptop is the hardest subset in the entire project.", 0.55, 1.7, 12.2, 0.45,
   size=18, bold=True, color=C_BODY)

stats = [
    ("4,076",  "gold training samples",         "fewest across all subsets", PANEL_BLUE),
    ("240",    "(entity, attribute) classes",   "avg. ~17 examples per class", PANEL_PURPLE),
    ("0.3035", "V16 baseline dev cF1",          "9.7 pp below SOTA 0.4006", PANEL_SLATE),
]
for i, (num, label, sub, pal) in enumerate(stats):
    x = 0.55 + i * 4.12
    rrect(s, x, 2.3, 3.85, 2.1, pal[0], radius=0.07)
    tx(s, num, x + 0.1, 2.5, 3.65, 0.85, size=42, bold=True, color=pal[1],
       align=PP_ALIGN.CENTER)
    tx(s, label, x + 0.1, 3.38, 3.65, 0.42, size=15, bold=True, color=C_BODY,
       align=PP_ALIGN.CENTER)
    tx(s, sub, x + 0.1, 3.8, 3.65, 0.4, size=13, color=C_MUTED, italic=True,
       align=PP_ALIGN.CENTER)

# takeaway bar
rrect(s, 0.55, 4.85, 12.25, 0.95, PANEL_AMBER[0], radius=0.08)
rect(s, 0.55, 4.85, 0.1, 0.95, PANEL_AMBER[2])
tx(s, "Takeaway", 0.78, 4.98, 3.0, 0.35, size=13, bold=True, color=C_AMBER,
   spacing=120)
tx(s, "The root bottleneck is long-tail class sparsity — not encoder capacity or regularization.",
   0.78, 5.3, 11.8, 0.42, size=16, bold=True, color=C_TITLE)


# ══════════════════════════════════════════════════════════════
#  SLIDE 3 — Three methods overview
# ══════════════════════════════════════════════════════════════
s = add_slide()
eyebrow(s, "Methods")
title(s, "Three Augmentation Approaches at a Glance")

cards = [
    ("Back-Translation (BT)", PANEL_AMBER, None,
     ["Source: original sentence -> de / fr / zh -> English",
      "Labels: copied directly from `gold` annotation",
      "New text: surface variants, same meaning",
      "eng_laptop: +2.19 pp  (V5-repro)"]),
    ("Paraphrase (LLM rewrite)", PANEL_RED, ("FAILED", C_RED),
     ["Source: `DeepSeek-chat` rewrite of existing sentences",
      "Labels: copied directly from `gold` annotation",
      "New text: synonymous rewrites, same semantics",
      "eng_laptop: -2.73 pp  (V18)"]),
    ("V23 Knowledge Distillation", PANEL_GREEN, ("ADOPTED", C_GREEN),
     ["Source: brand-new Amazon laptop reviews (unlabeled)",
      "Labels: `DeepSeek` re-annotates quads from scratch",
      "New text: new scenarios + categories + polarity",
      "eng_laptop: +8.87 pp  (V23)"]),
]
for i, (htxt, pal, bdg, items) in enumerate(cards):
    x = 0.52 + i * 4.16
    rrect(s, x, 1.65, 3.95, 5.35, pal[0], radius=0.05)
    panel_head(s, x + 0.22, 1.85, 3.55, htxt, pal[1], size=15)
    if bdg:
        badge(s, bdg[0], x + 0.22, 2.36, 1.25, 0.34, bdg[1], size=10)
    bullets(s, items, x + 0.22, 2.95, 3.55, pal[2], size=13.5, gap=9)


# ══════════════════════════════════════════════════════════════
#  SLIDE 4 — Concrete example: MOTHERBOARD
# ══════════════════════════════════════════════════════════════
s = add_slide()
eyebrow(s, "Concrete Example  ·  1 gold sample")
title(s, "MOTHERBOARD#OPERATION_PERFORMANCE", size=27)

rows = [
    ("Gold", PANEL_BLUE, None,
     '"I\'ve had mine 10 months and the motherboard has crapped out twice already."',
     "Category: MOTHERBOARD#OPERATION_PERFORMANCE   |   Opinion: NULL"),
    ("BT  x2", PANEL_AMBER, None,
     '"...the motherboard\'s been shitting twice."  [de]      "...has already screwed up twice."  [fr]',
     "Labels copied verbatim — only a few synonyms changed, identical semantic scenario"),
    ("Paraphrase  x2", PANEL_RED, None,
     '"...the motherboard has failed twice already."      "...has died twice within 10 months..."',
     "Labels copied verbatim — fluent rewrite, but zero new information"),
    ("KD Silver  x11", PANEL_GREEN, ("NEW", C_GREEN),
     '"...less than two years later the board is burnt out."   "...will not launch BIOS..."',
     "New text + LLM re-labeling — burnt out / will not launch BIOS: trigger words absent from gold"),
]
for i, (label, pal, bdg, quote, note) in enumerate(rows):
    y = 1.62 + i * 1.36
    rrect(s, 0.52, y, 12.3, 1.24, pal[0], radius=0.07)
    rect(s, 0.52, y, 1.85, 1.24, pal[2])
    tx(s, label, 0.55, y, 1.79, 1.24, size=14, bold=True, color=C_WHITE,
       align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    if bdg:
        badge(s, bdg[0], 11.85, y + 0.12, 0.85, 0.3, bdg[1], size=9)
    tx(s, quote, 2.55, y + 0.12, 10.1, 0.72, size=13, color=C_TITLE, italic=True)
    tx(s, note, 2.55, y + 0.84, 10.1, 0.36, size=11.5, color=pal[1])


# ══════════════════════════════════════════════════════════════
#  SLIDE 5 — MOUSE#USABILITY polarity gap
# ══════════════════════════════════════════════════════════════
s = add_slide()
eyebrow(s, "Concrete Example  ·  Polarity coverage")
title(s, "MOUSE#USABILITY — 1 Positive-Only Gold Sample", size=27)

# left
rrect(s, 0.52, 1.62, 5.9, 5.3, PANEL_RED[0], radius=0.05)
panel_head(s, 0.74, 1.8, 5.5, "BT / Paraphrase — reachable", PANEL_RED[1], size=15)
left = [
    ("Gold", '"the mousepad feels very comfortable to use."', "positive"),
    ("BT", '"the mousepad feels quite comfortable."', "still comfortable-family, positive"),
    ("Para", '"the mousepad is pleasant and easy to use."', "swapped to pleasant, still positive"),
]
for j, (lbl, q, note) in enumerate(left):
    y = 2.4 + j * 1.18
    tx(s, lbl, 0.74, y, 1.2, 0.34, size=13, bold=True, color=PANEL_RED[1])
    tx(s, q, 0.74, y + 0.32, 5.5, 0.46, size=13, color=C_TITLE, italic=True)
    tx(s, note, 0.74, y + 0.74, 5.5, 0.34, size=11.5, color=C_MUTED)
rrect(s, 0.74, 6.05, 5.46, 0.72, C_WHITE, line=PANEL_RED[2], radius=0.1)
tx(s, "Negative polarity = 0 samples.  The model is blind at test time.",
   0.9, 6.18, 5.2, 0.5, size=13, bold=True, color=C_RED)

# right
rrect(s, 6.62, 1.62, 6.2, 5.3, PANEL_GREEN[0], radius=0.05)
panel_head(s, 6.84, 1.8, 5.8, "KD Silver — 17 new samples", PANEL_GREEN[1], size=15)
right = [
    ("unlab_47", '"...mouse pad are comfortable to use and not cramped."', "comfortable + not cramped  ·  positive"),
    ("unlab_599", '"I\'m still struggling with the mouse pad..."', "struggling  ·  NEGATIVE — first seen"),
    ("unlab_1792", '"It\'s not as responsive as I wanted to be."', "not as responsive  ·  NEGATIVE — new trigger"),
]
for j, (lbl, q, note) in enumerate(right):
    y = 2.4 + j * 1.18
    tx(s, lbl, 6.84, y, 2.0, 0.34, size=13, bold=True, color=PANEL_GREEN[1], font=FONT_MONO)
    tx(s, q, 6.84, y + 0.32, 5.8, 0.46, size=13, color=C_TITLE, italic=True)
    tx(s, note, 6.84, y + 0.74, 5.8, 0.34, size=11.5, color=C_MUTED)
rrect(s, 6.84, 6.05, 5.76, 0.72, C_WHITE, line=PANEL_GREEN[2], radius=0.1)
tx(s, "Both polarities covered.  Opinion vocabulary: 1 word -> 17+ words.",
   7.0, 6.18, 5.5, 0.5, size=13, bold=True, color=C_GREEN)


# ══════════════════════════════════════════════════════════════
#  SLIDE 6 — Difference table
# ══════════════════════════════════════════════════════════════
s = add_slide()
eyebrow(s, "Comparison")
title(s, "What Each Method Actually Changes")

headers = ["METHOD", "TEXT SOURCE", "LABEL SOURCE", "NEW EXPR.", "NEW POLARITY", "NEW SCENARIOS"]
col_w = [2.3, 2.2, 2.2, 1.6, 1.7, 2.3]
col_x = [0.52]
for w in col_w[:-1]:
    col_x.append(col_x[-1] + w)

# header
rrect(s, 0.52, 1.7, sum(col_w), 0.52, C_HEADER_ROW, radius=0.08)
for h, x, w in zip(headers, col_x, col_w):
    tx(s, h, x + 0.08, 1.78, w - 0.16, 0.4, size=12, bold=True, color=C_MUTED,
       align=PP_ALIGN.CENTER, spacing=60)

table = [
    ("Back-Translation", "Sentence variants", "Copied from gold", "Minimal", "No", "No", PANEL_AMBER),
    ("Paraphrase", "Rewritten sentences", "Copied from gold", "Few", "No", "No", PANEL_RED),
    ("V23 KD Silver", "New real reviews", "LLM re-annotated", "Extensive", "Yes", "Yes", PANEL_GREEN),
]
for i, row in enumerate(table):
    pal = row[-1]
    cells = row[:-1]
    y = 2.34 + i * 1.0
    bg = pal[0] if i == 2 else C_WHITE
    rrect(s, 0.52, y, sum(col_w), 0.92, bg, line=C_ROW_LINE, radius=0.06)
    for j, (cell, x, w) in enumerate(zip(cells, col_x, col_w)):
        if j == 0:
            col, bld = pal[1], True
        elif cell in ("No", "Minimal", "Few"):
            col, bld = C_RED, False
        elif cell in ("Yes", "Extensive"):
            col, bld = C_GREEN, True
        else:
            col, bld = C_BODY, False
        tx(s, cell, x + 0.08, y, w - 0.16, 0.92, size=14.5, bold=bld, color=col,
           align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

rrect(s, 0.52, 5.55, sum(col_w), 1.05, PANEL_BLUE[0], radius=0.06)
rect(s, 0.52, 5.55, 0.1, 1.05, PANEL_BLUE[2])
tx(s, "Takeaway", 0.74, 5.68, 3.0, 0.34, size=13, bold=True, color=PANEL_BLUE[1], spacing=120)
tx(s, "BT / Paraphrase change surface text only — labels and scenarios stay identical.\n"
      "KD Silver introduces scenarios, trigger words, and polarities the model has never seen.",
   0.74, 5.98, 11.8, 0.6, size=15, bold=True, color=C_TITLE)


# ══════════════════════════════════════════════════════════════
#  SLIDE 7 — Results bar chart
# ══════════════════════════════════════════════════════════════
s = add_slide()
eyebrow(s, "Experiment Results")
title(s, "eng_laptop dev cF1 Progression")

steps = [
    ("V16   deberta-v3-large, raw",      0.3035, C_MUTED,  "baseline", False),
    ("V18   deberta-large + Paraphrase", 0.2762, C_RED,    "-2.73 pp", False),
    ("V19   roberta-large, raw",         0.3062, C_AMBER,  "+0.27 pp", False),
    ("V20   roberta-large + BT + Para",  0.3088, C_AMBER,  "+0.53 pp", False),
    ("V5    deberta-base + 4-pivot BT",  0.3254, RGBColor(0x3B,0x82,0xF6), "+2.19 pp", False),
    ("V23 KD   deberta-base + silver",   0.4141, C_GREEN,  "+8.87 pp", True),
]
bar_x0, bar_y0, bar_h, gap = 3.95, 1.75, 0.66, 0.22
min_v, max_v = 0.25, 0.44
scale = 8.4 / (max_v - min_v)

for i, (label, val, col, delta, best) in enumerate(steps):
    y = bar_y0 + i * (bar_h + gap)
    blen = (val - min_v) * scale
    rrect(s, bar_x0, y, 8.4, bar_h, C_HEADER_ROW, radius=0.25)   # track
    rrect(s, bar_x0, y, max(blen, 0.2), bar_h, col, radius=0.25)  # bar
    tx(s, label, 0.52, y, 3.35, bar_h, size=13, bold=best,
       color=(C_GREEN if best else C_BODY), anchor=MSO_ANCHOR.MIDDLE, wrap=False)
    tx(s, f"{val:.4f}", bar_x0 + blen + 0.12, y, 0.85, bar_h, size=13, bold=True,
       color=col, anchor=MSO_ANCHOR.MIDDLE)
    tx(s, delta, bar_x0 + blen + 1.05, y, 1.4, bar_h, size=13, italic=True,
       color=col, anchor=MSO_ANCHOR.MIDDLE)
    if best:
        badge(s, "BEST", bar_x0 + 0.18, y + 0.16, 0.85, bar_h - 0.32, C_WHITE,
              fg=C_GREEN, size=11)

# SOTA marker
sota_x = bar_x0 + (0.4006 - min_v) * scale
rect(s, sota_x, bar_y0 - 0.12, 0.025, len(steps) * (bar_h + gap) - gap + 0.24, PANEL_AMBER[2])
tx(s, "SOTA 2nd  ·  nchellwig 27B LLM  ·  0.4006", sota_x - 0.2, 6.95, 5.0, 0.4,
   size=12, italic=True, color=C_AMBER)


# ══════════════════════════════════════════════════════════════
#  SLIDE 8 — Why KD works
# ══════════════════════════════════════════════════════════════
s = add_slide()
eyebrow(s, "Root Cause")
title(s, "Why KD Works — Three Mechanisms")

mechs = [
    (PANEL_BLUE, "1   Fills Sparse Category Slots",
     "15+ of 240 categories have only 1-3 gold samples. KD silver independently samples new text "
     "per category: MOUSE#USABILITY grows 1 -> 18, MOTHERBOARD#GENERAL 1 -> 13 — from near-zero to meaningful coverage."),
    (PANEL_AMBER, "2   Injects LLM Polarity Prior",
     'DeepSeek, pre-trained on billions of e-commerce texts, maps "struggling / not responsive / burnt out" '
     "to the correct negative categories. BT/Paraphrase cannot create negative polarity from a single positive gold — KD transfers this prior into the weights."),
    (PANEL_GREEN, "3   No LLM at Inference — Noise Contained",
     "Earlier inference-time fusion failed (LLM outputs are high-variance). KD distills during training only; "
     "silver noise is absorbed by 5-seed ensemble + vote=4. Final inference is pure small-model ensemble — stable, fast, no LLM."),
]
for i, (pal, htxt, body) in enumerate(mechs):
    y = 1.7 + i * 1.74
    rrect(s, 0.52, y, 12.3, 1.6, pal[0], radius=0.06)
    rect(s, 0.52, y, 0.13, 1.6, pal[2])
    tx(s, htxt, 0.82, y + 0.14, 11.8, 0.5, size=20, bold=True, color=pal[1])
    tx(s, body, 0.82, y + 0.64, 11.8, 0.9, size=14.5, color=C_BODY)


# ══════════════════════════════════════════════════════════════
#  SLIDE 9 — KD gain ∝ scarcity
# ══════════════════════════════════════════════════════════════
s = add_slide()
eyebrow(s, "Generalization  ·  4 datasets")
title(s, "KD Gain Scales Inversely with Gold-Data Size", size=27)

data = [
    ("rus",        1766, "+4.86 pp", "Beats PAI SOTA by +1.54 pp",       PANEL_GREEN, True),
    ("ukr",        2967, "+4.92 pp", "Beats PAI SOTA by +1.23 pp",       PANEL_GREEN, True),
    ("eng_laptop", 4076, "+8.87 pp", "Beats nchellwig 2nd by +1.35 pp",  PANEL_GREEN, True),
    ("zho_laptop", 5534, "-0.59 pp", "Silver dilutes sufficient data",   PANEL_RED,   False),
]
for i, (name, gold, delta, verdict, pal, good) in enumerate(data):
    y = 1.65 + i * 1.2
    rrect(s, 0.52, y, 12.3, 1.08, pal[0], radius=0.06)
    # gold bar
    rrect(s, 0.72, y + 0.34, 3.1, 0.42, C_WHITE, radius=0.3)
    rrect(s, 0.72, y + 0.34, gold / 5534 * 3.1, 0.42, PANEL_BLUE[2], radius=0.3)
    tx(s, f"{gold:,} gold", 0.8, y + 0.34, 3.0, 0.42, size=12.5, bold=True,
       color=PANEL_BLUE[1], anchor=MSO_ANCHOR.MIDDLE)
    tx(s, name, 4.05, y, 2.2, 1.08, size=19, bold=True, color=C_TITLE,
       anchor=MSO_ANCHOR.MIDDLE, font=FONT_MONO)
    tx(s, delta, 6.3, y, 2.0, 1.08, size=22, bold=True, color=pal[1],
       align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    tx(s, verdict, 8.45, y, 4.2, 1.08, size=14.5, italic=True, color=pal[1],
       anchor=MSO_ANCHOR.MIDDLE)
    badge(s, "WINS" if good else "HURTS", 11.9, y + 0.37, 0.78, 0.34,
          pal[2], size=9)

rrect(s, 0.52, 6.55, 12.3, 0.62, PANEL_BLUE[0], radius=0.08)
tx(s, "Rule:   gold <= 4k  ->  large positive gain      |      gold >= 5.5k + strong native encoder  ->  KD slightly hurts",
   0.74, 6.65, 11.9, 0.42, size=15, bold=True, color=PANEL_BLUE[1])


# ══════════════════════════════════════════════════════════════
#  SLIDE 10 — Conclusion
# ══════════════════════════════════════════════════════════════
s = add_slide()
eyebrow(s, "Conclusion")
title(s, "KD is a Targeted Fix for Data-Scarce Subsets")

# paper quote
rrect(s, 0.52, 1.62, 12.3, 1.85, PANEL_SLATE[0], radius=0.05)
rect(s, 0.52, 1.62, 0.13, 1.85, PANEL_BLUE[2])
tx(s, "PAPER NARRATIVE", 0.82, 1.74, 11.8, 0.35, size=12, bold=True,
   color=PANEL_BLUE[1], spacing=150)
tx(s,
   '"LLM distillation benefit is inversely proportional to gold-data sufficiency: all three '
   'data-starved subsets (rus, ukr, eng_laptop, <=4k gold) gain +4.9~8.9 pp and surpass prior test '
   'SOTA, while a data-adequate subset with a strong native encoder (zho_laptop, 5.5k gold) is slightly hurt."',
   0.82, 2.12, 11.8, 1.3, size=15, italic=True, color=C_BODY)

takeaways = [
    (PANEL_AMBER, "BT / Paraphrase: surface change only; regularization benefit, but cannot break the class-sparsity ceiling."),
    (PANEL_GREEN, "V23 KD: new text + LLM re-labeling fills long-tail slots and injects polarity prior -> qualitative leap (+8.87 pp)."),
    (PANEL_RED,   "Cost: silver noise exists; needs 5-seed ensemble + vote filtering; harmful on data-sufficient subsets."),
    (PANEL_BLUE,  "440M deberta-v3-base + KD reaches 27B-LLM-team level — 60x smaller, zero LLM at inference."),
]
for i, (pal, text) in enumerate(takeaways):
    y = 3.66 + i * 0.84
    rrect(s, 0.52, y, 12.3, 0.74, pal[0], radius=0.08)
    rect(s, 0.52, y, 0.1, 0.74, pal[2])
    tx(s, text, 0.78, y, 11.9, 0.74, size=14, color=C_BODY, anchor=MSO_ANCHOR.MIDDLE)


out = "/home/g24tka12/ABSA-QUAD/docs/V23_KD_vs_augmentation.pptx"
prs.save(out)
print(f"Saved -> {out}  ({len(prs.slides)} slides)")
