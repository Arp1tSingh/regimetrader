# PROGRESS

Update after **every** task: tick the box, add a log line, record key numbers. This file is the memory between sessions. Do not delete history; append.

Legend: `[ ]` todo, `[x]` done and acceptance passed, `[~]` BLOCKED (reason in Blockers).

## Current position

- Phase: 0
- Next task: T0.4
- Last gate passed: none

## Checklist

**Phase 0 Scaffold:** [x] T0.1 [x] T0.2 [x] T0.3 [ ] T0.4 [ ] T0.5
**Phase 1 Warehouse and ETL:** [ ] T1.1 [ ] T1.2 [ ] T1.3 [ ] T1.4 [ ] T1.5 [ ] T1.6 [ ] T1.7 [ ] T1.8
**Phase 2 Indicators and OLAP:** [ ] T2.1 [ ] T2.2
**Phase 3 Regimes:** [ ] T3.1 [ ] T3.2 [ ] T3.3 [ ] T3.4 [ ] T3.5
**Phase 4 Classifiers:** [ ] T4.1 [ ] T4.2 [ ] T4.3 [ ] T4.4 [ ] T4.5
**Phase 5 Strategy and backtest:** [ ] T5.1 [ ] T5.2 [ ] T5.3 [ ] T5.4 [ ] T5.5 [ ] T5.6 [ ] T5.7 [ ] T5.8
**Phase 6 Final evaluation:** [ ] T6.1 [ ] T6.2 [ ] T6.3 [ ] T6.4
**Phase 7 Paper trading:** [ ] T7.1 [ ] T7.2 [ ] T7.3 [ ] T7.4 [ ] T7.5
**Phase 8 Dashboard:** [ ] T8.1 [ ] T8.2
**Phase 9 Stretch:** [ ] T9.1 [ ] T9.2 [ ] T9.3 [ ] T9.4
**Phase 10 Handoff:** [ ] T10.1 [ ] T10.2 [ ] T10.3 [ ] T10.4

## Key numbers (fill as they become known)

| Item | Value |
|---|---|
| Universe file source and date | |
| Instruments loaded / stocks with under 10 years of history | |
| Trading days loaded, first and last date | |
| VIX used in regimes (yes or no) and coverage | |
| Chosen k and why | |
| Regime day counts (BULL / SIDEWAYS / HIGH_VOL / BEAR) | |
| Selected model, `entry_base`, `slots` (from validation) | |
| Classifier accuracy versus majority baseline (per model, validation) | |
| Total trials logged | |
| Test period evaluated? (date, lock file) | |
| Headline test results (strategy vs B1) | |

## Log

| Date | Task | Result and notes |
|---|---|---|
| 2026-10-10 | T0.1 | Repo skeleton created from `07§1`: 24 folders, 13 `__init__.py`, `.gitkeep` files. `git init` on branch `main` (git 2.55.0 present, contrary to the note in the pack). |
| 2026-10-10 | T0.2 | venv at `.venv`, Python **3.14.8** (only interpreter on this machine; `python3` absent). All 39 packages from `requirements.txt` installed cleanly. Versions resolve newer than spec minimums: pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, xgboost 3.4.1, streamlit 1.65.0. Acceptance import check exits 0. |
| 2026-10-10 | T0.3 | `src/common/config.py` with `extends`, override precedence, frozen-config whitelist, `config_hash()`, path resolution, output-folder creation, split-order validation. 19 tests pass. `make check` green. |

## Blockers

| Date | Task | Blocker | Needed from human |
|---|---|---|---|
| | | | |

## Deviations from the spec

Anything you changed or could not do as written. Also add a numbered entry to `docs/spec/DECISIONS.md`.

| Date | Task | Deviation and reason |
|---|---|---|
| 2026-10-10 | T0.1 | The pack states git is not installed on this machine. It is (2.55.0). `git init` performed as T0.1 requires. |
| 2026-10-10 | T0.2 | **Environment, needs a human decision.** Three deviations: (a) no `make` was present, so GNU Make 4.4.1 was installed via `winget install ezwinports.make` (user scope, no admin); (b) no POSIX `python3`, so the interpreter is Python 3.14.8 with `python`; (c) the shipped Makefile is POSIX-only, so it was edited in place to branch on `$(OS)` for the venv path and the `rm -rf` smoke clean. Recipes and POSIX behaviour are unchanged. |
| 2026-10-10 | T0.2 | **pandas 3.0.6 is a major version above the `>=2.2` the spec was written against.** Copy-on-write is now default and several 2.x idioms are removed. Code must be written and reviewed for pandas 3 semantics, not backported pandas 2 behaviour. |
