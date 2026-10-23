"""Stage 11 - Build frontend_deploy/, the static site to drag onto Netlify Drop.

Run:  python scripts/11_build_frontend.py

Copies the dashboard's static files (src/cropforecast/serve/web) into
frontend_deploy/ at the project root, ready to drag onto
https://app.netlify.com/drop, and writes serve.backend_url from
configs/default.yaml into js/config.js - so with a permanent backend (an ngrok
static domain) the Netlify page needs no ?api= in its link at all. 09_go_live.py
rebuilds it on every launch, so the folder always matches the page being served;
it only needs re-uploading when the page or the backend address changes.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "cropforecast" / "serve" / "web"
OUT = ROOT / "frontend_deploy"
REQUIRED = ["index.html", "css/style.css", "js/app.js", "js/config.js", "_redirects"]


def build(quiet: bool = False) -> Path:
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(WEB, OUT)

    sys.path.insert(0, str(ROOT / "src"))
    from cropforecast.config import load_config
    backend = (load_config().serve.get("backend_url") or "").rstrip("/")
    (OUT / "js" / "config.js").write_text(
        "/* Written by scripts/11_build_frontend.py from serve.backend_url in\n"
        "   configs/default.yaml. A ?api= in the link still wins. */\n"
        f"window.CROPCAST_API = {json.dumps(backend)};\n", encoding="utf-8")
    missing = [f for f in REQUIRED if not (OUT / f).is_file()]
    if missing:
        sys.exit(f"frontend_deploy is incomplete, missing: {', '.join(missing)}")
    if not quiet:
        files = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file())
        size = sum((OUT / f).stat().st_size for f in files) / 1024
        print(f"Built {OUT.name}/  ({len(files)} files, {size:.0f} KB)")
        print(f"  backend built in: {backend or '(none - the link must carry ?api=)'}")
        for f in files:
            print(f"  {f}")
        print("\nDrag the frontend_deploy folder onto https://app.netlify.com/drop "
              "(signed in, so the site is kept).")
    return OUT


if __name__ == "__main__":
    build()
