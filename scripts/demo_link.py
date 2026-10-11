"""Publish a temporary public demo URL to a GitHub Release, and tell the truth about it.

**What this can and cannot do.** It can start a tunnel, publish the URL, and check later
whether that URL still answers. It **cannot keep a link alive**. A `cloudflared` quick
tunnel's hostname is random per process: when the process dies the URL is gone forever,
and a replacement tunnel gets a different URL. Nothing in this repository can run a
24-hour watchdog.

So this script is built around that limit rather than hiding it. The Release it writes
carries the generated-at time, the age, the command to verify liveness, and a plain
statement that a dead link is expected rather than a fault. A link that silently rots
into looking authoritative is worse than no link at all.

Commands:

    python scripts/demo_link.py publish   # start a tunnel and publish/replace the URL
    python scripts/demo_link.py check     # is the published URL still alive?
    python scripts/demo_link.py revive    # check, and republish only if it is dead

`publish` leaves the tunnel and dashboard running as detached background processes so
they outlive the shell that started them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from scripts.demo_tunnel import (  # noqa: E402
    URL_PATTERN,
    find_cloudflared,
    find_tunnel_url,
    spawn,
    terminate,
    wait_for_health,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO = "Arp1tSingh/regimetrader"
TAG = "demo"
TITLE = "Live demo"
PORT = 8599
STATE = REPO_ROOT / "logs" / "demo_link.json"

RELEASE_BODY = """# Live demo

**URL:** {url}

---

### Read this before you click

This link is **temporary** and is expected to go stale.

It is a [cloudflared](https://developers.cloudflare.com/cloudflare-one/) *quick tunnel*
bound to one laptop. The hostname is generated randomly when the process starts, so:

- the link is valid **only while that machine is awake and the tunnel is running**;
- when it dies it is gone for good, and a replacement tunnel gets a **different** URL;
- nothing here runs a 24-hour watchdog, so it can die without anyone noticing.

**A dead link in this section is expected behaviour, not a fault in the project.**

| | |
|---|---|
| Generated | `{generated}` |

### Check whether it is still alive

On the machine that hosts it:

```bash
make demo-link-check     # prints ALIVE or DEAD
make demo-link-revive    # republishes only if it is dead
```

Or just open the URL. If it does not load, it has expired.

### What the demo shows

This is a **negative result**, reported as such. The out-of-sample test found no
demonstrated edge: the strategy's 95% confidence interval on Sharpe crosses zero, and
removing the direction classifier made the system *better*. Full detail in
[`reports/results/RESULTS.md`](https://github.com/{repo}/blob/main/reports/results/RESULTS.md).

Academic prototype. Not investment advice. No real money, no broker connection.
"""


def gh(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["gh", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def body() -> str:
    result = gh("release", "view", TAG, "--repo", REPO, "--json", "body")
    if result.returncode != 0:
        return ""
    try:
        return str(json.loads(result.stdout).get("body") or "")
    except json.JSONDecodeError:
        return ""


def published_url() -> str | None:
    match = URL_PATTERN.search(body())
    return match.group(0) if match else None


def alive(url: str, timeout: float = 25.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/_stcore/health", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def start_tunnel() -> tuple[str, subprocess.Popen, subprocess.Popen]:
    """Start the dashboard and a tunnel as detached processes and return the URL.

    **Both children must not inherit this process's stdout/stderr.** They are left
    running on purpose, and an inherited pipe handle is what a shell waits on — so
    `make demo-link | Select-Object -Last 5` would hang forever waiting for a tunnel that
    is never going to exit. Output goes to `logs/` instead. This is the third time in
    this project that output inheritance has disguised or hidden a working process.
    """
    binary = find_cloudflared()
    if binary is None:
        raise SystemExit("cloudflared not found. Run `make demo-check` for install steps.")

    import sys as _sys

    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    dashboard_log = open(log_dir / "demo_dashboard.log", "ab", buffering=0)  # noqa: SIM115
    dashboard = subprocess.Popen(  # noqa: S603
        [
            _sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/dashboard/app.py",
            "--server.port",
            str(PORT),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=REPO_ROOT,
        stdout=dashboard_log,
        stderr=subprocess.STDOUT,
        **(
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
            if os.name == "nt"
            else {"start_new_session": True}
        ),
    )

    if not wait_for_health(PORT):
        terminate(dashboard, "dashboard")
        raise SystemExit(f"the dashboard never became healthy on port {PORT}")

    tunnel = spawn(
        [binary, "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
        capture_stderr=True,
    )
    url = find_tunnel_url(tunnel)
    if url is None:
        terminate(tunnel, "tunnel")
        terminate(dashboard, "dashboard")
        raise SystemExit("the tunnel did not report a URL in time")
    return url, dashboard, tunnel


def publish() -> int:
    url, _dashboard, _tunnel = start_tunnel()
    generated = datetime.now(timezone.utc)
    body_text = RELEASE_BODY.format(
        url=url, generated=generated.strftime("%Y-%m-%d %H:%M UTC"), repo=REPO
    )

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        json.dumps(
            {"url": url, "generated": generated.isoformat(), "pids": {"port": PORT}}, indent=2
        ),
        encoding="utf-8",
    )

    exists = gh("release", "view", TAG, "--repo", REPO).returncode == 0
    args = (
        ["release", "edit", TAG, "--repo", REPO, "--notes", body_text]
        if exists
        else [
            "release",
            "create",
            TAG,
            "--repo",
            REPO,
            "--title",
            TITLE,
            "--target",
            "main",
            "--notes",
            body_text,
        ]
    )
    result = gh(*args)
    if result.returncode != 0:
        print(result.stderr.strip() or result.stdout.strip(), file=sys.stderr)
        return 1

    print(f"\npublished: {url}")
    print(f"release  : https://github.com/{REPO}/releases/tag/{TAG}")
    print(
        "\nThis URL dies when this laptop sleeps or the tunnel is killed.\n"
        "Run `make demo-link-check` to see the truth, `make demo-link-revive` to fix it."
    )
    return 0


def check() -> int:
    url = published_url()
    if url is None:
        print("DEAD - no URL is published in the release.")
        print("Run `make demo-link-revive` to publish a fresh one.")
        return 2

    stamp = ""
    if STATE.exists():
        try:
            data = json.loads(STATE.read_text(encoding="utf-8"))
            generated = datetime.fromisoformat(str(data["generated"]))
            age = datetime.now(timezone.utc) - generated
            stamp = f"  (published {age.days}d {age.seconds // 3600}h ago)"
        except (json.JSONDecodeError, KeyError, ValueError):
            stamp = ""

    if alive(url):
        print(f"ALIVE  {url}{stamp}")
        return 0
    print(f"DEAD   {url}{stamp}")
    print("Quick tunnels die with their process. Run `make demo-link-revive`.")
    return 2


def revive() -> int:
    url = published_url()
    if url and alive(url):
        print(f"ALIVE  {url} - nothing to do.")
        return 0
    print(f"not alive ({url or 'nothing published'}) - starting a new tunnel...\n")
    return publish()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["publish", "check", "revive"])
    args = parser.parse_args(argv)
    started = time.time()
    code = {"publish": publish, "check": check, "revive": revive}[args.action]()
    print(f"({time.time() - started:.1f}s)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
