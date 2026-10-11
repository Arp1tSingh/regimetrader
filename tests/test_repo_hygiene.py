"""Repository hygiene: source files must actually be in the repository.

This exists because of a real failure found by the T10.2 clean-clone check. The
`.gitignore` shipped with the project contained the unanchored pattern `models/`, which
git applies at **any** depth, so it silently excluded `src/models/` -- the whole
classifier training module -- from every commit. Ruff also respects `.gitignore` by
default, so those six files were never formatted or linted either.

The result: 502 tests passed locally, `make smoke` passed locally, and a clean clone could
not even import the module. Nothing that ran inside the working directory could detect it,
because the working directory had the files. Only a clone could.

So this test shells out to git on purpose. It is the only check in the suite that looks at
the repository rather than the code, and it is the one that would have caught it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directories whose contents must all be tracked.
SOURCE_DIRS = ("src", "tests", "sql", "scripts", "config", "docs")

#: Paths that are generated, large, or deliberately untracked.
ALLOWED_UNTRACKED = {
    "config/frozen_config.smoke.yaml",
}

#: Patterns that are *intentionally* depth-agnostic, with the reason. `__pycache__` is
#: not anchored because it exists at every depth (`src/etl/__pycache__`, `tests/__pycache__`)
#: and every one of them must be ignored. Anchoring it would leak build artefacts in.
INTENTIONALLY_UNANCHORED = {
    "__pycache__/": "must match src/*/__pycache__ and tests/__pycache__",
}


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"git unavailable in this environment: {result.stderr.strip()}")
    return result.stdout


def _source_files() -> list[Path]:
    files: list[Path] = []
    for directory in SOURCE_DIRS:
        root = REPO_ROOT / directory
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            files.append(path)
    return sorted(files)


def test_every_source_file_is_tracked_by_git() -> None:
    """A file that exists only on disk will be missing from every clone."""
    tracked = set(_git("ls-files").splitlines())
    missing = []
    for path in _source_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in ALLOWED_UNTRACKED or relative in tracked:
            continue
        missing.append(relative)

    assert not missing, (
        "these files exist but are NOT in the repository, so a clone cannot use them:\n  "
        + "\n  ".join(missing)
    )


def test_no_source_file_is_excluded_by_gitignore() -> None:
    """Gitignore patterns must be anchored; `models/` also matches `src/models/`."""
    ignored = set(_git("status", "--porcelain", "--ignored").splitlines())
    offenders = []
    for path in _source_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in ALLOWED_UNTRACKED:
            continue
        if any(
            line[3:].strip().strip('"') == relative for line in ignored if line.startswith("!!")
        ):
            offenders.append(relative)

    assert not offenders, (
        "these source files are matched by .gitignore and will never be committed:\n  "
        + "\n  ".join(offenders)
    )


def test_the_gitignore_directory_patterns_are_anchored() -> None:
    """An unanchored `models/` is the exact bug that hid `src/models/`."""
    lines = [
        line.strip() for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    ]
    unanchored = [
        line
        for line in lines
        if line.endswith("/")
        and not line.startswith("/")
        and not line.startswith("#")
        and line not in INTENTIONALLY_UNANCHORED
    ]
    assert not unanchored, (
        "these .gitignore directory patterns are not anchored to the repo root, so they "
        f"also match at any depth:\n  {', '.join(unanchored)}"
    )


def test_the_tracked_set_has_no_stale_entries() -> None:
    """A file committed and then deleted from disk should not linger as tracked."""
    tracked = {p for p in _git("ls-files").splitlines() if p.endswith(".py")}
    present = {p.relative_to(REPO_ROOT).as_posix() for p in _source_files() if p.suffix == ".py"}
    vanished = sorted(tracked - present)
    assert not vanished, "tracked but absent from the working tree:\n  " + "\n  ".join(vanished)
