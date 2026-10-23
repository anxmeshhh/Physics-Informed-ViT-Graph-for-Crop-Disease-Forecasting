"""Stage 11 - Build frontend_deploy/, the static site to drag onto Netlify Drop.

Run:  python scripts/11_build_frontend.py

Copies the dashboard's static files (src/cropforecast/serve/web) into
frontend_deploy/ at the project root, ready to drag onto
https://app.netlify.com/drop. The page finds its backend from the ?api= link
that 09_go_live.py prints, so the folder never needs rebuilding for a new
tunnel - only when the page itself changes. 09_go_live.py rebuilds it on every
launch, so the folder on disk always matches the page being served.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "cropforecast" / "serve" / "web"
OUT = ROOT / "frontend_deploy"
REQUIRED = ["index.html", "css/style.css", "js/app.js", "_redirects"]


def build(quiet: bool = False) -> Path:
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(WEB, OUT)
    missing = [f for f in REQUIRED if not (OUT / f).is_file()]
    if missing:
        sys.exit(f"frontend_deploy is incomplete, missing: {', '.join(missing)}")
    if not quiet:
        files = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file())
        size = sum((OUT / f).stat().st_size for f in files) / 1024
        print(f"Built {OUT.name}/  ({len(files)} files, {size:.0f} KB)")
        for f in files:
            print(f"  {f}")
        print("\nDrag the frontend_deploy folder onto https://app.netlify.com/drop "
              "(signed in, so the site is kept).")
    return OUT


if __name__ == "__main__":
    build()
