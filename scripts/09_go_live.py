"""Stage 9 - Go live: backend on this machine, frontend on Netlify.

Starts the API, warms the model up so the first run in front of the panel is
as fast as every later one, opens a Cloudflare quick tunnel to it, and prints
the one link to share.

Run:  python scripts/09_go_live.py
      python scripts/09_go_live.py --frontend https://<site>.netlify.app
      python scripts/09_go_live.py --no-tunnel        # local only

The tunnel needs `cloudflared` (winget install Cloudflare.cloudflared) but no
account. Its URL changes every launch, which is why the share link carries it
as ?api= rather than it being baked into the Netlify build.
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd                                       # noqa: E402
import uvicorn                                            # noqa: E402
from PIL import Image                                     # noqa: E402

from cropforecast.config import PROJECT_ROOT, load_config  # noqa: E402

TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
CLOUDFLARED_DEFAULTS = [r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
                        r"C:\Program Files\cloudflared\cloudflared.exe"]


def get(url: str, timeout: float = 10) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "cropcast-go-live"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return res.status


def wait_for(url: str, seconds: float) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if get(url, timeout=5) == 200:
                return True
        except Exception:                                  # noqa: BLE001
            pass
        time.sleep(1)
    return False


def resolve_via_cloudflare(host: str, seconds: float) -> str | None:
    """Wait until Cloudflare's own DNS publishes the tunnel hostname; return an IP.

    Asking the system resolver too early is worse than waiting: the router
    caches the NXDOMAIN and this machine cannot reach the link for minutes
    afterwards, even once it works for everyone else. DNS-over-HTTPS goes
    straight to Cloudflare and leaves the local cache untouched.
    """
    query = f"https://cloudflare-dns.com/dns-query?name={host}&type=A"
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            req = urllib.request.Request(query, headers={"Accept": "application/dns-json"})
            with urllib.request.urlopen(req, timeout=5) as res:
                for ans in json.load(res).get("Answer", []):
                    if ans.get("type") == 1:
                        return ans["data"]
        except Exception:                                  # noqa: BLE001
            pass
        time.sleep(2)
    return None


def wait_for_pinned(host: str, ip: str, path: str, seconds: float) -> bool:
    """GET https://host/path over a socket opened to ``ip``, bypassing local DNS.

    TLS still verifies against ``host``; only the name lookup is skipped, so the
    check never plants a lookup the router could cache.
    """
    class Pinned(http.client.HTTPSConnection):
        def connect(self):
            sock = socket.create_connection((ip, 443), self.timeout)
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)

    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            conn = Pinned(host, timeout=5, context=ssl.create_default_context())
            conn.request("GET", path, headers={"User-Agent": "cropcast-go-live"})
            if conn.getresponse().status == 200:
                return True
        except Exception:                                  # noqa: BLE001
            pass
        time.sleep(1)
    return False


def warm_up() -> None:
    """Load the model and push one curated sample through the full pipeline.

    Runs in this process against the same service object the API serves from,
    so backbone download, CUDA context and kernel selection all happen now
    rather than during the first live demonstration.
    """
    from cropforecast.serve import api

    t0 = time.perf_counter()
    svc = api.service()
    index = PROJECT_ROOT / "tests" / "index.csv"
    if index.exists():
        row = pd.read_csv(index).iloc[0]
        img = Image.open(PROJECT_ROOT / "tests" / row.file)
        pred = svc.predict(img, row.suggested_site_id,
                           Date.fromisoformat(row.suggested_date))
        print(f"  warm-up run : {row.file} -> {pred.disease} "
              f"({pred.confidence:.1%})")
    print(f"  model ready in {time.perf_counter() - t0:.1f} s")


# The dates the demo opens on: the default bulletin and the "Try an example" case.
DEMO_DATES = [Date(2023, 5, 30), Date(2023, 6, 22)]


def demo_dates() -> list[Date]:
    """The fixed demo dates plus every date in the test-leaf table, since a run
    moves the map to its leaf's date."""
    dates = set(DEMO_DATES)
    index = PROJECT_ROOT / "tests" / "index.csv"
    if index.exists():
        dates |= {Date.fromisoformat(d) for d in pd.read_csv(index).suggested_date.unique()}
    return sorted(dates)


def prefetch_heatmaps() -> None:
    """Fetch the heatmap's grid weather for the demo dates, in the background.

    Each new date is 341 archive lookups the first time and a disk read after
    that, so the heatmap appears instantly when the panel is watching. Fetches
    are spaced out because the free archive allows ~600 lookups a minute.
    """
    from cropforecast.serve import heatmap

    fetched = 0
    for d in demo_dates():
        end = d + timedelta(days=heatmap.DAYS - 1)
        if (heatmap.CACHE_DIR / f"{d}_{end}.parquet").exists():
            continue
        if fetched:
            time.sleep(45)
        try:
            heatmap.grid_weather(d, end)
            fetched += 1
        except Exception as exc:                           # noqa: BLE001
            print(f"  heatmap prefetch for {d} skipped: {exc}")
            return
    print(f"  heatmap weather cached for {len(demo_dates())} demo dates")


def open_tunnel(port: int) -> tuple[subprocess.Popen, str]:
    exe = shutil.which("cloudflared") or next(
        (p for p in CLOUDFLARED_DEFAULTS if Path(p).exists()), None)
    if exe is None:
        sys.exit("cloudflared not found. Install it with:\n"
                 "    winget install Cloudflare.cloudflared\n"
                 "or run with --no-tunnel to serve locally only.")

    proc = subprocess.Popen(
        [exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",
    )
    found: list[str] = []
    ready = threading.Event()

    def pump():
        # cloudflared logs to stderr for its whole life; keep draining it so a
        # full pipe can never stall the tunnel mid-demo.
        for line in proc.stdout:
            if not found:
                m = TUNNEL_URL.search(line)
                if m:
                    found.append(m.group(0))
                    ready.set()

    threading.Thread(target=pump, daemon=True).start()
    if not ready.wait(60):
        proc.terminate()
        sys.exit("cloudflared started but never reported a tunnel URL. "
                 "Check the internet connection and try again.")
    return proc, found[0]


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frontend", default=cfg.serve.get("frontend_url") or "",
                        help="Netlify URL of the deployed frontend")
    parser.add_argument("--no-tunnel", action="store_true",
                        help="serve on this machine only")
    args = parser.parse_args()

    host, port = cfg.serve.host, cfg.serve.port
    local = f"http://{host}:{port}"

    print("Starting backend ...")
    server = uvicorn.Server(uvicorn.Config("cropforecast.serve.api:app",
                                           host=host, port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    if not wait_for(f"{local}/api/health", 60):
        sys.exit(f"Backend did not come up on {local} - is the port already in use?")

    print("Warming up the model ...")
    warm_up()
    threading.Thread(target=prefetch_heatmaps, daemon=True).start()

    tunnel = None
    public = ""
    if not args.no_tunnel:
        print("Opening public tunnel ...")
        t0 = time.time()
        tunnel, public = open_tunnel(port)
        host = urllib.parse.urlparse(public).hostname
        print(f"  tunnel issued    +{time.time() - t0:.0f} s")
        ip = resolve_via_cloudflare(host, 120)
        print(f"  DNS published    +{time.time() - t0:.0f} s" if ip
              else "  warning: Cloudflare has not published the hostname yet")
        if ip and wait_for_pinned(host, ip, "/api/health", 60):
            print(f"  tunnel answering +{time.time() - t0:.0f} s")
        else:
            print("  warning: tunnel not answering yet; it usually does within a minute")

    frontend = args.frontend.rstrip("/")
    bar = "=" * 72
    print(f"\n{bar}\n  LIVE  -  Animesh Gupta  -  Panel Number: 0696\n{bar}")
    print(f"  Local dashboard  : {local}")
    if public:
        print(f"  Public backend   : {public}")
        print(f"  Public dashboard : {public}        (served by this machine)")
        if frontend:
            print(f"\n  SHARE THIS LINK  : {frontend}/?api={public}")
        else:
            print("\n  Netlify link     : <your-site>.netlify.app/?api=" + public)
            print("                     (set serve.frontend_url in configs/default.yaml"
                  " to print it in full)")
    print(f"{bar}\n  Keep this window open during the demo. Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
            if tunnel is not None and tunnel.poll() is not None:
                print("cloudflared exited - the public link is down. Restart this script.")
                tunnel = None
    except KeyboardInterrupt:
        print("\nShutting down ...")
    finally:
        if tunnel is not None:
            tunnel.terminate()
        server.should_exit = True
        time.sleep(1)


if __name__ == "__main__":
    main()
