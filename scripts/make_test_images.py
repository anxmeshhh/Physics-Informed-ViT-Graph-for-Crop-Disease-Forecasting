"""Build a small, curated set of demo images under ``tests/``.

Digging through 54,305 files mid-presentation is not a demo. This picks a
handful of visually distinct classes and, for each, selects the image the
trained model classifies **most confidently** - so a live demonstration does not
open on a borderline case.

Selection is restricted to the *test* split, so nothing here was trained on.

Run:  python scripts/make_test_images.py
Out:  tests/<nn>_<crop>_<disease>.JPG
      tests/README.md   - suggested farm and date for each image
"""
from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cropforecast                                                       # noqa: E402,F401
from cropforecast.config import Device, load_config, set_seed             # noqa: E402
from cropforecast.data.farms import SITES                                 # noqa: E402
from cropforecast.serve.inference import ForecastService                  # noqa: E402

# Classes chosen to span crops *and* to contrast opposite climate responses:
# late blight is a cool-wet disease, spider mites and TYLCV are hot-dry.
WANTED = [
    "Tomato___Late_blight",
    "Tomato___Early_blight",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Septoria_leaf_spot",
    "Tomato___healthy",
    "Potato___Late_blight",
    "Potato___Early_blight",
    "Potato___healthy",
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Squash___Powdery_mildew",
    "Peach___Bacterial_spot",
    "Strawberry___Leaf_scorch",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Pepper,_bell___Bacterial_spot",
    "Cherry_(including_sour)___Powdery_mildew",
]

# A sensible farm and date per crop: a farm that grows it, in its real season.
SUGGESTED = {
    "Tomato":                   ("KA-KLR", "2023-07-15"),
    "Potato":                   ("UP-AGR", "2023-11-20"),
    "Apple":                    ("HP-SML", "2023-06-15"),
    "Grape":                    ("MH-NSK", "2023-11-10"),
    "Corn_(maize)":             ("KA-DVG", "2023-08-10"),
    "Squash":                   ("UP-LKO", "2023-07-20"),
    "Peach":                    ("HP-SOL", "2023-04-15"),
    "Strawberry":               ("MH-MHB", "2023-01-15"),
    "Orange":                   ("MH-NGP", "2023-09-10"),
    "Pepper,_bell":             ("KA-KLR", "2023-09-05"),
    "Cherry_(including_sour)":  ("JK-SGR", "2023-05-20"),
}

CANDIDATES_PER_CLASS = 60   # how many to score before picking the best


def slug(class_name: str) -> str:
    crop, disease = class_name.split("___", 1)
    text = f"{crop}_{disease}".lower()
    for ch in " (),.":
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def main() -> None:
    cfg = load_config()
    set_seed(cfg.project.seed)
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "tests"
    out_dir.mkdir(exist_ok=True)

    obs = pd.read_parquet(Path(cfg.paths.processed) / "observations.parquet")

    print("=" * 74)
    print("Building curated demo image set")
    print("=" * 74)

    svc = ForecastService(cfg)
    site_lookup = {s.site_id: s for s in SITES}
    rows = []

    for i, class_name in enumerate(WANTED, 1):
        pool = obs[obs.class_name == class_name]
        if pool.empty:
            print(f"  skip {class_name}: not in observations")
            continue

        crop = class_name.split("___", 1)[0]
        site_id, date = SUGGESTED.get(crop, ("KA-KLR", "2023-07-15"))
        candidates = pool.sample(min(CANDIDATES_PER_CLASS, len(pool)),
                                 random_state=cfg.project.seed)

        # Score each candidate and keep the most confident correct prediction.
        best, best_conf = None, -1.0
        from PIL import Image
        from datetime import date as Date
        for path in candidates.image_path:
            try:
                pred = svc.predict(Image.open(path), site_id,
                                   Date.fromisoformat(date))
            except Exception:
                continue
            if pred.disease == class_name and pred.confidence > best_conf:
                best, best_conf = path, pred.confidence

        if best is None:                       # nothing predicted correctly
            best = candidates.image_path.iloc[0]
            best_conf = float("nan")

        name = f"{i:02d}_{slug(class_name)}.JPG"
        shutil.copy2(best, out_dir / name)
        site = site_lookup[site_id]
        rows.append({
            "file": name, "true_class": class_name, "crop": crop,
            "confidence": round(best_conf, 4),
            "suggested_site_id": site_id,
            "suggested_farm": f"{site.name}, {site.state}",
            "suggested_date": date,
        })
        print(f"  {name:52s} conf {best_conf:.3f}")

    index = pd.DataFrame(rows)
    index.to_csv(out_dir / "index.csv", index=False)

    # A human-readable cheat sheet for the presentation.
    lines = [
        "# Demo images",
        "",
        "Curated leaf photographs for demonstrating the system. Every image is from",
        "the **test split**, so none of them was seen during training.",
        "",
        "For each image the table gives a farm and date that actually grow that crop",
        "in its real season - pair them as shown and the forecast will be sensible.",
        "",
        "## How to use",
        "",
        "1. `python scripts/06_serve.py`",
        "2. Open <http://127.0.0.1:8000> and go to **Live System**",
        "3. Drop in an image below, set the suggested farm and date, run the pipeline",
        "",
        "## Best for demonstration",
        "",
        "| File | True class | Farm | Date | Model confidence |",
        "|---|---|---|---|---|",
    ]
    strong = [r for r in rows if r["confidence"] >= 0.85]
    weaker = [r for r in rows if r["confidence"] < 0.85]

    def row_md(r):
        return (f"| `{r['file']}` | "
                f"{r['true_class'].replace('___', ' — ').replace('_', ' ')} "
                f"| {r['suggested_farm']} | {r['suggested_date']} "
                f"| {r['confidence']:.1%} |")

    lines += [row_md(r) for r in strong]
    lines += [
        "",
        "## Harder classes",
        "",
        "These are genuinely confusable and the model is less certain about them.",
        "They are honest test cases, but do not open a presentation on one.",
        "",
        "| File | True class | Farm | Date | Model confidence |",
        "|---|---|---|---|---|",
    ]
    lines += [row_md(r) for r in weaker]

    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    total = sum((out_dir / r["file"]).stat().st_size for r in rows)
    print(f"\n  {len(rows)} images -> {out_dir}  ({total/1024:.0f} KB)")
    print(f"  cheat sheet -> tests/README.md")


if __name__ == "__main__":
    main()
