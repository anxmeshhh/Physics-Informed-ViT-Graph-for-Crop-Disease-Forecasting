"""Stage 10 - Audit the demo: every leaf through every view, checked for agreement.

Run with the server up (python scripts/09_go_live.py), then:

    python scripts/10_audit_demo.py
    python scripts/10_audit_demo.py --url https://<tunnel>.trycloudflare.com

Each demo leaf in tests/index.csv is run through the single-shot and streamed
pipelines, and the answer is cross-checked against everything else that talks
about the same farm on the same day - the district bulletin, the heatmap, the
location view, the "why this result" chain - and against tests/index.csv
itself. A view that disagrees with another is a bug the audience would see, so
every check prints its result and the script exits non-zero if any fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd                                        # noqa: E402

from cropforecast.config import load_config                # noqa: E402

BANDS = ("Low", "Medium", "High")
TOL = 0.002                    # values pass through JSON rounded to 3-4 decimals


class Audit:
    def __init__(self) -> None:
        self.fails: list[str] = []
        self.checks = 0
        self.by_kind: dict[str, int] = defaultdict(int)

    def check(self, ok: bool, kind: str, what: str) -> bool:
        self.checks += 1
        if not ok:
            self.fails.append(f"[{kind}] {what}")
            self.by_kind[kind] += 1
        return ok


def predict(base: str, path: str, row, stream: bool) -> dict:
    with open(TESTS / row.file, "rb") as fh:
        files = {"image": (row.file, fh, "image/jpeg")}
        data = {"site_id": row.suggested_site_id, "date": row.suggested_date}
        r = requests.post(f"{base}{path}", files=files, data=data, timeout=60)
    r.raise_for_status()
    if not stream:
        return r.json()
    msgs = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    err = [m for m in msgs if m["type"] == "error"]
    if err:
        raise RuntimeError(err[0]["detail"])
    result = next(m["result"] for m in msgs if m["type"] == "result")
    result["_steps"] = [m["step"] for m in msgs if m["type"] == "step"]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    cfg = load_config()
    thresholds = json.loads((Path(cfg.paths.processed) / "risk_thresholds.json").read_text())
    index = pd.read_csv(TESTS / "index.csv")
    a = Audit()
    get = lambda p, **kw: requests.get(f"{base}{p}", params=kw, timeout=90)

    health = get("/api/health").json()
    print(f"Auditing {len(index)} demo leaves against {base}  (api_level {health.get('api_level')})\n")
    a.check(health.get("model_trained") is True, "server", "model not loaded")

    samples = {s["file"]: s for s in get("/api/samples").json()["samples"]}
    sites = {s["site_id"]: s for s in get("/api/sites").json()["sites"]}
    bulletins: dict = {}
    heatmaps: dict = {}

    for row in index.itertuples():
        tag = row.file[:28]
        one = predict(base, "/api/predict", row, stream=False)
        live = predict(base, "/api/predict/stream", row, stream=True)
        again = predict(base, "/api/predict", row, stream=False)

        # --- the same leaf, three ways, must give one answer ----------------
        for key in ("disease", "crop", "risk_band", "disease_label"):
            a.check(one[key] == live[key] == again[key], "repeatable",
                    f"{tag}: {key} differs between runs/stream ({one[key]} / {live[key]} / {again[key]})")
        for key in ("confidence",):
            a.check(abs(one[key] - live[key]) < 1e-4 and abs(one[key] - again[key]) < 1e-4,
                    "repeatable", f"{tag}: {key} {one[key]} / {live[key]} / {again[key]}")
        a.check(all(abs(x - y) < 1e-4 for x, y in zip(one["risk"], live["risk"])),
                "repeatable", f"{tag}: risk {one['risk']} vs stream {live['risk']}")
        a.check(len(live["_steps"]) == len(live["trace"]) and
                [s["block"] for s in live["_steps"]] == [s["block"] for s in live["trace"]],
                "stream", f"{tag}: streamed stages differ from the result's trace")

        # --- stating the crop must not change an in-season answer ----------------
        with open(TESTS / row.file, "rb") as fh:
            stated = requests.post(f"{base}/api/predict", timeout=60,
                                   files={"image": (row.file, fh, "image/jpeg")},
                                   data={"site_id": row.suggested_site_id,
                                         "date": row.suggested_date, "crop": row.crop}).json()
        a.check(stated["disease"] == one["disease"], "crop-stated",
                f"{tag}: with crop stated diagnosed {stated['disease']}, inferred {one['disease']}")

        # --- against tests/index.csv ------------------------------------------
        a.check(one["disease"] == row.true_class, "diagnosis",
                f"{tag}: diagnosed {one['disease']}, labelled {row.true_class}")
        a.check(abs(one["confidence"] - row.confidence) < TOL, "index",
                f"{tag}: confidence {one['confidence']:.4f} but index.csv says {row.confidence}")
        a.check(one["site"]["site_id"] == row.suggested_site_id and one["date"] == row.suggested_date,
                "index", f"{tag}: ran at {one['site']['site_id']} {one['date']}")
        s = samples.get(row.file)
        a.check(s is not None and s["site_id"] == row.suggested_site_id and s["date"] == row.suggested_date,
                "samples", f"{tag}: /api/samples disagrees with index.csv")
        a.check(row.suggested_site_id in sites and row.crop in sites[row.suggested_site_id]["crops"],
                "pairing", f"{tag}: {row.suggested_site_id} does not grow {row.crop}")

        # --- label <-> number ---------------------------------------------------
        for h, r, b in zip(one["horizons"], one["risk"], one["risk_band"]):
            lo, hi = thresholds[f"risk_t{h}"]
            want = "High" if r >= hi else "Medium" if r >= lo else "Low"
            a.check(b == want, "band", f"{tag}: +{h}d risk {r:.3f} labelled {b}, cut-offs say {want}")
        worst = max(one["risk_band"], key=BANDS.index)
        word = {"High": "Act now", "Medium": "Watch", "Low": "Normal"}[worst]
        adv = one["advisory"]
        a.check(("Moderate" in adv) == (worst == "Medium") and ("highly favourable" in adv) == (worst == "High"),
                "advisory", f"{tag}: advisory '{adv[:60]}' vs worst band {worst}")

        # --- actions follow the diagnosis and the band ----------------------------
        healthy = one["disease"].endswith("___healthy")
        a.check((len(one["farmer_actions"]) == 0) == healthy, "actions",
                f"{tag}: farmer actions {len(one['farmer_actions'])} for {'healthy' if healthy else 'diseased'} leaf")
        a.check(len(one["officer_actions"]) > 0, "actions", f"{tag}: no officer actions")
        if not healthy and worst == "Low":
            a.check(any("Record" in x for x in one["officer_actions"]), "actions",
                    f"{tag}: confirmed case at Low band without a 'record it' action")

        # --- the why-chain against the bulletin -------------------------------------
        ctx = one.get("context") or {}
        key = (one["crop"], one["date"])
        if key not in bulletins:
            bulletins[key] = {d["site_id"]: d for d in
                              get("/api/bulletin", crop=one["crop"], date=one["date"]).json()["districts"]}
        card = bulletins[key].get(row.suggested_site_id)
        a.check(card is not None, "bulletin", f"{tag}: farm missing from its own bulletin")
        if card and ctx.get("district"):
            a.check(ctx["district"]["alert"] == card["alert"], "why-vs-map",
                    f"{tag}: why-chain says map is {ctx['district']['alert']}, bulletin card says {card['alert']}")
            a.check(abs(ctx["district"]["peak_risk"] - card["peak_risk"]) < TOL, "why-vs-map",
                    f"{tag}: peak {ctx['district']['peak_risk']} vs card {card['peak_risk']}")
            a.check((ctx["district"]["threat"] or None) == ((card.get("threat") or {}).get("name")),
                    "why-vs-map", f"{tag}: threat {ctx['district']['threat']} vs card {(card.get('threat') or {}).get('name')}")
        a.check(ctx.get("in_season") is True, "pairing",
                f"{tag}: {one['date']} is outside the {row.crop} season ({ctx.get('season')})")
        a.check(ctx.get("grown_here") is True, "pairing",
                f"{tag}: {row.suggested_site_id} is not a {row.crop} district")
        td = ctx.get("this_disease")
        a.check((td is None) == healthy, "why", f"{tag}: disease-vs-weather {'missing' if td is None else 'present'}")
        if td:
            a.check(td["disease"] == one["disease"], "why", f"{tag}: why-chain about {td['disease']}")
            a.check(td["peak"] >= 0.5, "pairing",
                    f"{tag}: paired weather suits {td['name']} only {td['peak']:.0%} at its best this week")
            a.check(abs(td["today"] - td["series"][0]) < 1e-6 and td["peak"] == max(td["series"]),
                    "why", f"{tag}: favourability summary inconsistent with its series")

        # --- the heatmap under this farm equals the farm's own reading ----------------
        if key not in heatmaps:
            hm = get("/api/heatmap", crop=one["crop"], date=one["date"])
            heatmaps[key] = hm.json() if hm.ok else None
            a.check(hm.ok, "heatmap", f"{tag}: heatmap {hm.status_code} {hm.text[:80]}")
        hm = heatmaps[key]
        if hm and card:
            anc = {x["name"]: x for x in hm["anchors"]}.get(card["name"])
            a.check(anc is not None and abs(anc["week"] - card["peak_risk"]) < TOL
                    and abs(anc["today"] - card["risk"]) < TOL, "heatmap-vs-map",
                    f"{tag}: heatmap reading {anc} vs card today {card['risk']} week {card['peak_risk']}")

        # --- the location view at the farm's own coordinates --------------------------
        site = sites[row.suggested_site_id]
        nb = get("/api/nearby", lat=site["lat"], lon=site["lon"], crop=one["crop"], date=one["date"])
        if a.check(nb.ok, "location", f"{tag}: /api/nearby {nb.status_code}"):
            nb = nb.json()
            a.check(nb["nearest"]["name"] == site["name"] and nb["nearest"]["distance_km"] < 1,
                    "location", f"{tag}: nearest to the farm itself is {nb['nearest']}")
            if card and nb["note"] is None:
                a.check(abs(nb["local"] - card["risk"]) < 0.02, "location-vs-map",
                        f"{tag}: weather at the farm's coordinates {nb['local']:.3f} vs card {card['risk']:.3f}")

        print(f"  {'ok ' if not any(tag in f for f in a.fails) else 'BAD'} {row.file:52s} "
              f"{one['disease_label']:24s} {one['confidence']:6.1%}  {worst:6s} @ {site['name']} {one['date']}")

    # --- the leaves as a set ---------------------------------------------------------
    ctxs = index.groupby(["crop", "suggested_site_id", "suggested_date"]).size()
    shared = ctxs[ctxs > 1]
    for (crop, sid, d), n in shared.items():
        diseases = index[(index.crop == crop) & (index.suggested_site_id == sid)
                         & (index.suggested_date == d)].true_class.tolist()
        print(f"\n  note: {n} {crop} leaves share {sid} {d}: {', '.join(diseases)}")

    print(f"\n{a.checks} checks, {len(a.fails)} failed")
    for k, n in sorted(a.by_kind.items()):
        print(f"  {k}: {n}")
    for f in a.fails:
        print("  -", f)
    sys.exit(1 if a.fails else 0)


if __name__ == "__main__":
    main()
