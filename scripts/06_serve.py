"""Stage 6 - Launch the dashboard (FastAPI + static HTML/CSS/JS).

Run:  python scripts/06_serve.py
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn                                            # noqa: E402

from cropforecast.config import load_config               # noqa: E402


def main() -> None:
    cfg = load_config()
    host, port = cfg.serve.host, cfg.serve.port
    ckpt = Path(cfg.paths.checkpoints) / "final_model.pt"
    if not ckpt.exists():
        print("WARNING: no trained model found at", ckpt)
        print("         The dashboard will load but /api/predict will return 503.")
        print("         Run scripts/04_train_and_ablate.py first.\n")
    print(f"Dashboard -> http://{host}:{port}")
    print(f"API docs  -> http://{host}:{port}/docs\n")
    uvicorn.run("cropforecast.serve.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
