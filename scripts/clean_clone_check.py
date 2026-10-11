"""Run the fast test suite against a clean clone, without touching the working tree.

**Why this exists.** Two defects were found by a manual clean-clone check (T10.2), and
both had the same shape: the suite was green in the working directory and broken in a
clone, because the working directory held state the repository does not.

* **R51** - the `.gitignore` pattern `models/` matched `src/models/` at any depth, so the
  whole classifier training module was never committed.
* **R52** - three OLAP CLI tests passed `config/config.smoke.yaml`, whose `paths.db` is a
  gitignored database that exists only after `make smoke` has run. Locally a 16 MB
  `smoke.db` was sitting there from earlier runs; in a clone the tables simply do not
  exist and the CLI exits 1.

Neither is detectable from inside the working directory. The only thing that detects them
is a clone.

**A rejected alternative, recorded so it is not retried.** The first attempt
(`scripts/check_pristine.py`, now deleted) hid the gitignored artifacts in place by
renaming them aside and restored them in a `finally` block. **It destroyed the smoke
artifacts and left a stash behind.** The mechanics were sound in principle and broken in
practice: tests recreated the hidden paths, restore order mattered, and `Path.rename`
will not overwrite on Windows. Anything that deletes real directories and relies on
cleanup is unsafe by construction, because a crash or Ctrl-C loses the data outright.
This script instead does the real thing - it clones - and touches nothing.

Usage:

    python scripts/clean_clone_check.py              # fast suite in a clone
    python scripts/clean_clone_check.py --smoke      # also run `make smoke` there
    python scripts/clean_clone_check.py --keep       # leave the clone for inspection

The clone reuses this repository's virtualenv instead of calling `make setup`, so it
takes about the same time as `make check` rather than an extra 14 minutes of pip. It
still catches missing-source failures, because the clone's own `src/` is what gets
imported.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="also run `make smoke` in the clone")
    parser.add_argument("--keep", action="store_true", help="do not delete the clone")
    parser.add_argument(
        "--branch",
        default=None,
        help="clone this ref instead of the working tree's HEAD (default: HEAD)",
    )
    args = parser.parse_args(argv)

    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if dirty:
        print(
            "WARNING: the working tree has uncommitted changes, so the clone will not\n"
            "         include them. Commit first, or the check may report stale results.\n"
            + dirty[:500]
        )

    workdir = Path(tempfile.mkdtemp(prefix="regimetrader_clone_"))
    clone = workdir / "repo"
    print(f"cloning into {clone}\n")

    code = _run(["git", "clone", "--quiet", str(REPO_ROOT), str(clone)], REPO_ROOT)
    if code != 0:
        shutil.rmtree(workdir, ignore_errors=True)
        return code

    if args.branch:
        _run(["git", "checkout", "--quiet", args.branch], clone)

    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=clone,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    print(f"clone HEAD: {head}\n")

    # Reuse the parent's interpreter: the clone's own `src/` is on sys.path because
    # pytest runs with cwd=clone, so this tests the cloned code, not the parent's.
    python = sys.executable
    failed = False

    if _run([python, "-m", "pytest", "-m", "not slow", "-q"], clone) != 0:
        print("\n`make check` equivalent FAILED in the clean clone.")
        print("A test may be depending on state that exists only in the working tree.")
        failed = True
    else:
        print("\nfast suite passed in a clean clone.")

    if args.smoke and not failed:
        if _run(["make", "smoke"], clone) != 0:
            print("\n`make smoke` FAILED in the clean clone.")
            failed = True
        else:
            print("\n`make smoke` passed in a clean clone.")

    if args.keep:
        print(f"\nclone kept at {clone}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)
        print("\nclone removed")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
