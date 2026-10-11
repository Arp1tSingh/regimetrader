"""Run the dashboard behind a temporary public URL, for a live demo.

**Why.** For an in-person viva the local dashboard is enough — the examiner can look at
your screen. The tunnel exists so they can open the app on their own device instead of
you mirroring. It is deliberately **additive**: every failure below falls back to the
plain local dashboard, and then to `reports/results/RESULTS.md`, which needs nothing
running at all.

**The URL is ephemeral.** `cloudflared`'s quick tunnel assigns a random hostname on every
run (`https://<random-words>.trycloudflare.com`). It cannot be written down in advance,
so this script prints it in a boxed block the moment it exists. Read it off the screen.
Do not put one in a document and expect it to work tomorrow.

**Safety.** The tunnel publishes a read-only app for as long as it runs. The dashboard
cannot write to the warehouse — `tests/test_dashboard.py` asserts every query in the data
layer is a `SELECT` — so an examiner poking at it cannot corrupt anything. Kill the
tunnel when the demo is over.

Usage:

    python scripts/demo_tunnel.py --check     # verify prerequisites, launch nothing
    python scripts/demo_tunnel.py             # run the demo
    python scripts/demo_tunnel.py --no-tunnel # just the local dashboard
    python scripts/demo_tunnel.py --port 8502 # if 8501 is taken
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP = "src/dashboard/app.py"

#: cloudflared prints the quick-tunnel URL on stderr, in a banner like
#: `INF |  https://some-words.trycloudflare.com  |`
URL_PATTERN = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

INSTALL_HINT = (
    "cloudflared is not installed. Install it once, before the demo:\n"
    "    winget install cloudflare.cloudflared\n"
    "or download from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/\n"
    "Then re-run with --check to confirm."
)


#: cloudflared installs here on Windows and on Linux/macOS. Checked in addition to PATH
#: because a shell opened *before* the install keeps a stale PATH and would report the
#: tool as missing on the day of the demo. That failure mode is worse than useless.
EXTRA_LOCATIONS = (
    Path(r"C:\Program Files (x86)\cloudflared\cloudflared.exe"),
    Path(r"C:\Program Files\cloudflared\cloudflared.exe"),
    Path.home() / ".cloudflared" / "cloudflared",
    Path("/usr/local/bin/cloudflared"),
    Path("/opt/homebrew/bin/cloudflared"),
)


def find_cloudflared() -> str | None:
    """PATH first, then the standard install locations."""
    found = shutil.which("cloudflared")
    if found:
        return found
    for candidate in EXTRA_LOCATIONS:
        if candidate.exists():
            return str(candidate)
    return None


def streamlit_command(port: int) -> list[str]:
    """The same flags as `make dashboard`, so the two paths behave identically.

    `--server.headless true` is what suppresses Streamlit's first-run "Email:" prompt;
    without it the process hangs forever in a non-interactive shell (DECISIONS R50).
    """
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        APP,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]


def wait_for_health(port: int, timeout: float = 90.0) -> bool:
    """Poll the health endpoint rather than sleeping a fixed amount."""
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/_stcore/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(1.0)
    return False


def spawn(cmd: list[str], capture_stderr: bool = False) -> subprocess.Popen:
    """Start a child in its own process group so teardown kills the whole tree.

    `capture_stderr` is needed for cloudflared: without an explicit pipe the child
    *inherits* this process's handles, so `proc.stderr` is None and there is nothing
    to read the quick-tunnel URL from.
    """
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    if capture_stderr:
        kwargs["stderr"] = subprocess.PIPE
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["text"] = True
        kwargs["bufsize"] = 1
    return subprocess.Popen(cmd, cwd=REPO_ROOT, **kwargs)  # noqa: S603


def terminate(proc: subprocess.Popen | None, label: str) -> None:
    """Stop a child and everything it started."""
    if proc is None or proc.poll() is not None:
        return
    print(f"\nstopping {label}...")
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],  # noqa: S603,S607
                check=False,
                capture_output=True,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def find_tunnel_url(proc: subprocess.Popen, timeout: float = 60.0) -> str | None:
    """cloudflared logs to stderr; scan it for the quick-tunnel hostname."""
    deadline = time.time() + timeout
    if proc.stderr is None:
        return None
    while time.time() < deadline:
        line = proc.stderr.readline()
        if not line:
            if proc.poll() is not None:
                return None
            continue
        found = URL_PATTERN.search(line)
        if found:
            return found.group(0)
    return None


def banner(url: str | None, port: int) -> None:
    width = 74
    print("\n" + "=" * width)
    if url:
        print("  DEMO IS LIVE - give the examiner this URL")
        print("=" * width)
        print(f"\n    {url}\n")
        print("  It is random and changes every run. Read it aloud.")
        print(f"  Local fallback if they cannot reach it: http://localhost:{port}")
    else:
        print("  LOCAL ONLY - no tunnel. Show the dashboard on your screen.")
        print(f"  It is at http://localhost:{port}")
    print("=" * width + "\n")


def preflight(use_tunnel: bool) -> bool:
    """Report what is available without launching anything."""
    print("pre-flight\n" + "-" * 40)
    tunnel = find_cloudflared() if use_tunnel else None
    if use_tunnel:
        if tunnel:
            version = subprocess.run(  # noqa: S603
                [tunnel, "--version"], capture_output=True, text=True, check=False
            ).stdout.strip()
            print(f"  cloudflared : {version}")
        else:
            print("  cloudflared : MISSING")
            print("\n" + INSTALL_HINT)
            return False
    else:
        print("  cloudflared : not needed (--no-tunnel)")

    warehouse = REPO_ROOT / "data" / "warehouse" / "regimetrader.db"
    print(
        f"  warehouse   : {'present' if warehouse.exists() else 'MISSING (screens will be empty)'}"
    )
    results = REPO_ROOT / "reports" / "results" / "summary_test.csv"
    print(f"  results CSV : {'present' if results.exists() else 'MISSING'}")

    from datetime import datetime, timedelta

    if warehouse.exists():
        stamp = datetime.fromtimestamp(warehouse.stat().st_mtime)
        age = (datetime.now() - stamp).days
        stale = age > int(os.environ.get("RT_STALE_DAYS", "3"))
        note = " (STALE - the banner will show)" if stale else ""
        print(f"  last load   : {stamp:%Y-%m-%d %H:%M}, {age} day(s) old{note}")
    del timedelta
    print("-" * 40)
    return True


def main(argv: list[str] | None = None) -> int:
    # Line buffering matters here: when stdout is redirected to a file or a pipe it is
    # block-buffered, so the tunnel URL would sit in the buffer and the demo would look
    # hung. This is the second time output buffering has disguised a working process.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--check", action="store_true", help="pre-flight only, launch nothing")
    parser.add_argument("--no-tunnel", action="store_true", help="local dashboard only")
    args = parser.parse_args(argv)

    use_tunnel = not args.no_tunnel
    os.environ["RT_DEMO_PORT"] = str(args.port)

    if not preflight(use_tunnel):
        return 2
    if args.check:
        print("\npre-flight OK. Re-run without --check to start the demo.\n")
        return 0

    streamlit = tunnel = None
    try:
        print(f"starting dashboard on port {args.port}...")
        streamlit = spawn(streamlit_command(args.port))
        if not wait_for_health(args.port):
            print("the dashboard did not become healthy in time.", file=sys.stderr)
            print(f"  stdout/stderr follows:\n{_drain(streamlit)}", file=sys.stderr)
            return 1
        print("dashboard healthy.\n")

        url = None
        if use_tunnel:
            binary = find_cloudflared()
            if binary is None:
                print("cloudflared vanished since pre-flight; continuing local only.")
            else:
                print("starting tunnel...")
                tunnel = spawn(
                    [binary, "tunnel", "--url", f"http://127.0.0.1:{args.port}", "--no-autoupdate"],
                    capture_stderr=True,
                )
                url = find_tunnel_url(tunnel)
                if url is None:
                    print("tunnel did not report a URL in time; continuing local only.")

        banner(url, args.port)

        print("Ctrl-C to stop.\n")
        while streamlit.poll() is None:
            time.sleep(1)
        return 0
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        terminate(tunnel, "tunnel")
        terminate(streamlit, "dashboard")


def _drain(proc: subprocess.Popen) -> str:
    try:
        out, err = proc.communicate(timeout=2)
        return ((out or b"").decode(errors="replace") + (err or b"").decode(errors="replace"))[
            -2000:
        ]
    except subprocess.TimeoutExpired:
        proc.kill()
        return "(no output captured)"


if __name__ == "__main__":
    raise SystemExit(main())
