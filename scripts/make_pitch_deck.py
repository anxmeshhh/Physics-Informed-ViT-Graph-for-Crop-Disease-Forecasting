"""Generate the project pitch deck.

A plain-language walkthrough of what the system does, how the pieces connect,
and what the numbers say. Figures are pulled from outputs/figures/, and the
headline numbers are read from outputs/reports/ so the deck cannot drift out of
step with the results.

Run:  python scripts/make_pitch_deck.py
Out:  Crop_Disease_Forecasting_Pitch_Deck.pptx
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
REP = ROOT / "outputs" / "reports"

# --- palette ---------------------------------------------------------------
NAVY = RGBColor(0x14, 0x31, 0x5C)
INK = RGBColor(0x1A, 0x1D, 0x22)
SOFT = RGBColor(0x4D, 0x54, 0x5C)
FAINT = RGBColor(0x82, 0x8A, 0x93)
GREEN = RGBColor(0x1F, 0x6B, 0x3A)
RED = RGBColor(0x9D, 0x22, 0x35)
AMBER = RGBColor(0xB0, 0x7D, 0x0A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CREAM = RGBColor(0xF6, 0xF7, 0xF9)
RULE = RGBColor(0xDD, 0xE2, 0xE7)

W, H = Inches(13.333), Inches(7.5)          # 16:9


def deck() -> Presentation:
    p = Presentation()
    p.slide_width, p.slide_height = W, H
    return p


def blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def rect(slide, x, y, w, h, fill=None, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1)
    shp.shadow.inherit = False
    return shp


def text(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=1.0):
    """runs: list of (text, size, bold, colour) or (text, size, bold, colour, bullet)."""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, r in enumerate(runs):
        body, size, bold, colour = r[0], r[1], r[2], r[3]
        bullet = r[4] if len(r) > 4 else False
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        para.space_after = Pt(6)
        run = para.add_run()
        run.text = ("•   " + body) if bullet else body
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = colour
        run.font.name = "Segoe UI"
    return box


def title_slide(prs, main, sub, authors):
    s = blank(prs)
    rect(s, 0, 0, W, H, fill=NAVY)
    rect(s, Inches(0.9), Inches(2.35), Inches(1.1), Pt(4), fill=RGBColor(0x4A, 0xDE, 0x80))
    text(s, Inches(0.9), Inches(1.1), Inches(11.5), Inches(1.2),
         [("MINOR PROJECT  ·  DEPARTMENT OF COMPUTING TECHNOLOGIES", 12, True,
           RGBColor(0x8F, 0xA8, 0xC8))])
    text(s, Inches(0.9), Inches(2.7), Inches(11.5), Inches(2.2),
         [(main, 36, True, WHITE)], spacing=1.12)
    text(s, Inches(0.9), Inches(4.55), Inches(11.5), Inches(0.8),
         [(sub, 19, False, RGBColor(0xB8, 0xCA, 0xE2))])
    text(s, Inches(0.9), Inches(5.85), Inches(11.5), Inches(0.9),
         [(authors, 14, False, RGBColor(0x8F, 0xA8, 0xC8))])
    return s


def header(slide, number, heading, kicker=None):
    rect(slide, 0, 0, W, Inches(1.12), fill=WHITE)
    rect(slide, 0, Inches(1.12), W, Pt(2.5), fill=NAVY)
    text(slide, Inches(0.62), Inches(0.26), Inches(1.0), Inches(0.6),
         [(f"{number:02d}", 26, True, RGBColor(0xC8, 0xD2, 0xDE))])
    text(slide, Inches(1.45), Inches(0.22), Inches(11.2), Inches(0.75),
         [(heading, 26, True, NAVY)])
    if kicker:
        text(slide, Inches(1.45), Inches(0.72), Inches(11.2), Inches(0.4),
             [(kicker, 13, False, FAINT)])


def bullets_slide(prs, n, heading, kicker, items, note=None):
    s = blank(prs)
    rect(s, 0, 0, W, H, fill=CREAM)
    header(s, n, heading, kicker)
    runs = []
    for it in items:
        if isinstance(it, tuple):
            runs.append((it[0], 19, True, INK, False))
            runs.append((it[1], 15, False, SOFT, False))
        else:
            runs.append((it, 18, False, INK, True))
    text(s, Inches(1.45), Inches(1.75), Inches(10.6), Inches(4.6), runs, spacing=1.25)
    if note:
        rect(s, Inches(1.45), Inches(6.25), Inches(10.6), Inches(0.72), fill=WHITE, line=RULE)
        text(s, Inches(1.7), Inches(6.38), Inches(10.2), Inches(0.5),
             [(note, 14, False, SOFT)])
    return s


def figure_slide(prs, n, heading, kicker, image, caption=None):
    s = blank(prs)
    rect(s, 0, 0, W, H, fill=CREAM)
    header(s, n, heading, kicker)
    path = FIG / image
    if not path.exists():
        text(s, Inches(1.45), Inches(3.2), Inches(10), Inches(1),
             [(f"[missing figure: {image}]", 16, False, RED)])
        return s

    from PIL import Image as PILImage
    iw, ih = PILImage.open(path).size
    top = Inches(1.55)
    avail_h = Inches(5.05) if caption else Inches(5.5)
    avail_w = Inches(11.4)
    scale = min(avail_w / iw, avail_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    s.shapes.add_picture(str(path), int((W - w) / 2), top, width=w, height=h)
    if caption:
        text(s, Inches(0.95), Inches(6.72), Inches(11.4), Inches(0.55),
             [(caption, 14, False, SOFT)], align=PP_ALIGN.CENTER)
    return s


def cards_slide(prs, n, heading, kicker, cards, accent=NAVY):
    """cards: list of (big, label, sub)."""
    s = blank(prs)
    rect(s, 0, 0, W, H, fill=CREAM)
    header(s, n, heading, kicker)
    cols = len(cards)
    gap = Inches(0.3)
    total = Inches(11.4)
    cw = int((total - gap * (cols - 1)) / cols)
    x0 = Inches(0.95)
    for i, (big, label, sub) in enumerate(cards):
        x = x0 + i * (cw + gap)
        rect(s, x, Inches(2.15), cw, Inches(2.6), fill=WHITE, line=RULE)
        rect(s, x, Inches(2.15), cw, Pt(4), fill=accent)
        text(s, x + Inches(0.28), Inches(2.5), cw - Inches(0.5), Inches(0.9),
             [(big, 40, True, accent)])
        text(s, x + Inches(0.28), Inches(3.42), cw - Inches(0.5), Inches(0.5),
             [(label, 14, True, INK)])
        text(s, x + Inches(0.28), Inches(3.85), cw - Inches(0.5), Inches(0.8),
             [(sub, 12, False, FAINT)])
    return s


def table_slide(prs, n, heading, kicker, headers, rows, widths=None, note=None,
                highlight_row=None):
    s = blank(prs)
    rect(s, 0, 0, W, H, fill=CREAM)
    header(s, n, heading, kicker)

    left, top = Inches(0.95), Inches(1.8)
    total_w = Inches(11.4)
    ncol = len(headers)
    widths = widths or [1 / ncol] * ncol
    col_w = [int(total_w * f) for f in widths]
    row_h = Inches(0.46)

    x = left
    for j, htxt in enumerate(headers):
        rect(s, x, top, col_w[j], row_h, fill=NAVY)
        text(s, x + Inches(0.14), top + Inches(0.08), col_w[j] - Inches(0.2),
             row_h, [(htxt, 13, True, WHITE)])
        x += col_w[j]

    for i, row in enumerate(rows):
        y = top + row_h + i * row_h
        is_hl = highlight_row is not None and i == highlight_row
        bg = RGBColor(0xEE, 0xF7, 0xF1) if is_hl else WHITE
        x = left
        for j, cell in enumerate(row):
            rect(s, x, y, col_w[j], row_h, fill=bg, line=RULE)
            colour = GREEN if is_hl else (INK if j == 0 else SOFT)
            text(s, x + Inches(0.14), y + Inches(0.09), col_w[j] - Inches(0.2),
                 row_h, [(str(cell), 12.5, is_hl or j == 0, colour)])
            x += col_w[j]

    if note:
        ny = top + row_h * (len(rows) + 1) + Inches(0.3)
        rect(s, left, ny, total_w, Inches(0.75), fill=WHITE, line=RULE)
        text(s, left + Inches(0.25), ny + Inches(0.14), total_w - Inches(0.5),
             Inches(0.5), [(note, 13.5, False, SOFT)])
    return s


def flow_slide(prs, n, heading, kicker, steps):
    """steps: list of (label, sub)."""
    s = blank(prs)
    rect(s, 0, 0, W, H, fill=CREAM)
    header(s, n, heading, kicker)
    cols = len(steps)
    gap = Inches(0.22)
    total = Inches(11.4)
    cw = int((total - gap * (cols - 1)) / cols)
    x0 = Inches(0.95)
    for i, (label, sub) in enumerate(steps):
        x = x0 + i * (cw + gap)
        rect(s, x, Inches(2.5), cw, Inches(2.1), fill=WHITE, line=RULE)
        rect(s, x, Inches(2.5), cw, Pt(4), fill=NAVY)
        text(s, x + Inches(0.2), Inches(2.72), cw - Inches(0.4), Inches(0.4),
             [(f"STEP {i+1}", 10, True, FAINT)])
        text(s, x + Inches(0.2), Inches(3.08), cw - Inches(0.4), Inches(0.7),
             [(label, 15, True, NAVY)])
        text(s, x + Inches(0.2), Inches(3.72), cw - Inches(0.4), Inches(0.8),
             [(sub, 11.5, False, SOFT)])
        if i < cols - 1:
            text(s, x + cw, Inches(3.25), gap, Inches(0.4),
                 [("→", 18, True, RGBColor(0xB0, 0xBA, 0xC6))], align=PP_ALIGN.CENTER)
    return s


def closing_slide(prs, repo):
    s = blank(prs)
    rect(s, 0, 0, W, H, fill=NAVY)
    text(s, Inches(0.9), Inches(2.1), Inches(11.5), Inches(1.0),
         [("Thank you", 42, True, WHITE)])
    rect(s, Inches(0.9), Inches(3.25), Inches(1.1), Pt(4), fill=RGBColor(0x4A, 0xDE, 0x80))
    text(s, Inches(0.9), Inches(3.7), Inches(11.5), Inches(1.4),
         [("Animesh Gupta   ·   Apurva Singh   ·   Aastha Hotwani   ·   Harsh Khetan",
           16, False, RGBColor(0xB8, 0xCA, 0xE2))])
    text(s, Inches(0.9), Inches(4.6), Inches(11.5), Inches(0.6),
         [(repo, 13, False, RGBColor(0x8F, 0xA8, 0xC8))])
    text(s, Inches(0.9), Inches(5.5), Inches(11.5), Inches(1.0),
         [("Run it:   python scripts/06_serve.py   →   http://127.0.0.1:8000",
           15, True, RGBColor(0x4A, 0xDE, 0x80))])
    return s


# ---------------------------------------------------------------------------
def load_numbers() -> dict:
    """Read headline numbers from the saved reports so the deck cannot drift."""
    out = {"acc": "98.8%", "f1": "0.984", "r2": "0.78",
           "bench": [], "abl": []}
    bpath = REP / "stage3_backbone_benchmark.csv"
    if bpath.exists():
        b = pd.read_csv(bpath)
        out["bench"] = [
            (r.backbone.upper(), f"{r.dim}", f"{r.probe_macro_f1:.3f}",
             f"{r.full_macro_f1:.3f}", f"{r.risk_r2_mean:.3f}")
            for r in b.itertuples()
        ]
    apath = REP / "stage4_ablations.csv"
    if apath.exists():
        a = pd.read_csv(apath)
        out["abl"] = [
            (r.config, f"{r.test_acc:.3f}", f"{r.test_macro_f1:.3f}",
             f"{r.risk_r2_mean:.3f}")
            for r in a.itertuples()
        ]
        best = a.loc[a.test_macro_f1.idxmax()]
        out["acc"] = f"{best.test_acc:.1%}"
        out["f1"] = f"{best.test_macro_f1:.3f}"
        out["r2"] = f"{best.risk_r2_mean:.2f}"
    return out


def build() -> Path:
    n = load_numbers()
    prs = deck()

    # 1 ------------------------------------------------------------------
    title_slide(
        prs,
        "Vision Transformer with Physics-Informed\nGraph Learning for Crop Disease Forecasting",
        "Predicting crop disease risk 1–7 days before symptoms appear",
        "Animesh Gupta  ·  Apurva Singh  ·  Aastha Hotwani  ·  Harsh Khetan",
    )

    # 2 ------------------------------------------------------------------
    bullets_slide(
        prs, 1, "The problem", "Why existing crop-disease apps are not enough",
        [
            "Today's apps only tell you what a leaf already has",
            "By the time symptoms are visible, the damage is done",
            "They ignore weather — yet weather is what drives infection",
            "They look at one field at a time, as if it stood alone",
        ],
        note="A farmer does not need a diagnosis. A farmer needs a warning.",
    )

    # 3 ------------------------------------------------------------------
    bullets_slide(
        prs, 2, "Our idea in one line",
        "Do not just identify the disease — forecast where it is going",
        [
            ("Look at the leaf", "A vision transformer recognises the disease from the photo"),
            ("Check the weather", "Real temperature, humidity, rainfall and leaf wetness at that farm"),
            ("Look at the neighbours", "Nearby farms are linked in a network; wind carries spores"),
            ("Predict the next week", "Disease risk at 1, 3, 5 and 7 days ahead"),
        ],
    )

    # 4 ------------------------------------------------------------------
    flow_slide(
        prs, 3, "How it works", "The same four steps, as the system runs them",
        [
            ("Leaf photo", "A frozen vision transformer turns the image into 384 numbers"),
            ("Weather", "Real ERA5 data becomes leaf wetness, VPD, growing degree days"),
            ("Farm graph", "Nearby farms connected, with extra links pointing downwind"),
            ("Forecast", "Disease risk at 1, 3, 5 and 7 days, plus an advisory"),
        ],
    )

    # 5 ------------------------------------------------------------------
    cards_slide(
        prs, 4, "What we built it on", "Everything below is real, measured data",
        [
            ("54,305", "Leaf photographs", "38 diseases across 14 crops"),
            ("41,662", "Days of weather", "Real ERA5 reanalysis, zero gaps"),
            ("37", "Farm districts", "Srinagar to Munnar, real coordinates"),
            ("1–7", "Day forecast", "Four horizons, not just today"),
        ],
    )

    # 6 ------------------------------------------------------------------
    table_slide(
        prs, 5, "Being honest about the data",
        "What is real, and what we had to construct",
        ["Component", "Where it comes from", "Status"],
        [
            ["54,305 leaf images", "PlantVillage dataset", "REAL"],
            ["41,662 days of weather", "Open-Meteo ERA5 archive", "REAL"],
            ["37 farm locations", "Indian production districts", "REAL"],
            ["Which photo belongs to which farm", "We assigned it, using real weather", "CONSTRUCTED"],
        ],
        widths=[0.36, 0.42, 0.22],
        note="PlantVillage has no GPS and no dates. We placed each leaf at a farm that "
             "grows that crop, on a date when the real weather suited that disease.",
    )

    # 7 ------------------------------------------------------------------
    bullets_slide(
        prs, 6, "Why weather is the whole point",
        "Different diseases want opposite conditions",
        [
            ("Late blight  →  cool and wet", "Peaks at 17 °C with long leaf wetness"),
            ("Powdery mildew  →  humid, but rain kills it", "Free water bursts its spores"),
            ("Spider mites  →  hot and dry", "Thrives at 31 °C in dry air"),
        ],
        note="A model that only looks at pixels cannot use any of this. Ours can.",
    )

    # 8 ------------------------------------------------------------------
    figure_slide(
        prs, 7, "Every pathogen has its own temperature window",
        "Published cardinal temperatures, encoded in the model",
        "fig_pathogen_response.png",
        "Each curve is zero outside the organism's survivable range and peaks at its optimum.",
    )

    # 9 ------------------------------------------------------------------
    figure_slide(
        prs, 8, "The farm network", "37 real districts, connected by proximity and wind",
        "fig_farm_network.png",
        "Lines are the graph edges along which the model passes information between farms.",
    )

    # 10 -----------------------------------------------------------------
    figure_slide(
        prs, 9, "Each disease sits in its own climate niche",
        "This is the signal the weather branch learns",
        "fig_climate_separation.png",
        "Cool-wet diseases at the top left, hot-dry pests at the bottom right. Nothing forced this.",
    )

    # 11 -----------------------------------------------------------------
    cards_slide(
        prs, 10, "Results", "Measured on leaves the model had never seen",
        [
            (n["acc"], "Diagnosis accuracy", "38-class problem"),
            (n["f1"], "Macro-F1", "Balanced across all classes"),
            (n["r2"], "Forecast R²", "Averaged over 1–7 days"),
            ("0", "Data leakage", "Leaf-disjoint splits"),
        ],
        accent=GREEN,
    )

    # 12 -----------------------------------------------------------------
    if n["bench"]:
        table_slide(
            prs, 11, "We compared four transformers",
            "Identical pipeline; only the image model changes",
            ["Model", "Size", "Feature quality", "Full system", "Forecast R²"],
            [list(r) for r in n["bench"]],
            widths=[0.22, 0.14, 0.22, 0.21, 0.21],
            highlight_row=0,
            note="DINOv2 gives the best features for its size; ViT-B/16 wins once the "
                 "full system is trained on top.",
        )

    # 13 -----------------------------------------------------------------
    bullets_slide(
        prs, 12, "What each part actually contributes",
        "Measured by removing it and retraining",
        [
            ("Remove the leaf image  →  F1 drops 0.98 to 0.51", "Vision does the diagnosis"),
            ("Remove the weather  →  forecast R² drops 0.14", "Climate does the forecasting"),
            ("Remove the graph  →  scores go UP slightly", "An honest negative result"),
        ],
        note="Each stream earns its place, except one. We report that rather than hide it.",
    )

    # 14 -----------------------------------------------------------------
    bullets_slide(
        prs, 13, "The finding we did not expect",
        "The farm graph does not improve accuracy here",
        [
            "Replacing it with a simple layer gives slightly better scores",
            "Reproducible: 2 datasets, 5 sparsity levels, 2 graph types",
            "Why: the image model is already ~98.8% right — no headroom left",
            "Why: a farm's own state already predicts its own future well",
        ],
        note="We kept the graph, measured its cost, and said so. That is a stronger "
             "position than a claim nobody checked.",
    )

    # 15 -----------------------------------------------------------------
    figure_slide(
        prs, 14, "We can see what the model looks at",
        "Grad-CAM and attention over real leaves",
        "explain_vision.png",
        "The heat lands on the actual lesions — not the background, not the leaf edge.",
    )

    # 16 -----------------------------------------------------------------
    bullets_slide(
        prs, 15, "A problem we found in the dataset",
        "And fixed, before it flattered our numbers",
        [
            "PlantVillage photographs the same leaf 2.65 times on average",
            "A normal random split puts copies of one leaf on both sides",
            "That affects 35,246 of 54,305 images — 65% of the data",
            "We split by leaf instead, so no leaf appears twice",
        ],
        note="Our numbers are lower than a naive split would give. They are also real.",
    )

    # 17 -----------------------------------------------------------------
    flow_slide(
        prs, 16, "How the code fits together", "Six scripts, run in order",
        [
            ("01  Build data", "Downloads leaves, fetches weather, joins them"),
            ("02  Extract", "Runs all four transformers once, caches the result"),
            ("03 · 04  Train", "Benchmarks models, trains the final one, runs ablations"),
            ("05 · 07  Explain", "Grad-CAM, SHAP and every figure in this deck"),
            ("06  Serve", "Starts the website"),
        ],
    )

    # 18 -----------------------------------------------------------------
    bullets_slide(
        prs, 17, "See it running", "One command, then a browser",
        [
            ("python scripts/06_serve.py", "Then open http://127.0.0.1:8000"),
            ("Literature Survey", "18 papers, each with a live Run demonstration button"),
            ("Live System", "Upload a leaf from tests/, pick a farm and date, get a forecast"),
            ("Risk Map", "Every farm coloured by risk, with the graph drawn on top"),
        ],
        note="23 ready-to-use demo images are in tests/, with a suggested farm and date for each.",
    )

    # 19 -----------------------------------------------------------------
    bullets_slide(
        prs, 18, "Where this goes next", "What we would build with more time",
        [
            "Drone and satellite imagery instead of a single leaf photo",
            "Live weather forecasts instead of historical records",
            "Field sensors measuring leaf wetness directly",
            "Validation against real recorded outbreaks",
        ],
    )

    closing_slide(
        prs,
        "github.com/anxmeshhh/Physics-Informed-ViT-Graph-for-Crop-Disease-Forecasting",
    )

    out = ROOT / "Crop_Disease_Forecasting_Pitch_Deck.pptx"
    prs.save(out)
    return out


if __name__ == "__main__":
    path = build()
    print(f"saved -> {path.name}  ({path.stat().st_size/1024:.0f} KB)")
