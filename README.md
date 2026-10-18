# RegimeTrader

Regime-aware stock direction predictor and paper-trading bot, built on a star-schema data
warehouse of Nifty 50 daily history. Data Warehousing and Mining course project.

**Academic prototype. Not investment advice. No real money. No broker integration.**

---

## 1. Read this before looking at any number

**The one-time out-of-sample evaluation found no demonstrated edge.**

On the test period (2024-01-01 to 2026-10-09, 685 trading days) the frozen strategy
scored a Sharpe of **0.4848** with a 95% confidence interval of **[−0.4561, 1.5437]** —
an interval that **crosses zero**. On that same period it:

- **beat** buy-and-hold `^NSEI` on Sharpe (0.4848 vs 0.1576) and CAGR (0.0548 vs 0.0120),
  but **lost on drawdown** (0.1932 vs 0.1559) — and the interval on that difference is
  wide enough to include "no difference";
- **lost** to a plain equal-weight basket (Sharpe 0.7017, CAGR 0.0882);
- **lost** to a 20/50-day moving-average trend rule (Sharpe 0.6117, CAGR 0.0502);
- **lost to its own ablation**: V1, the regime filter with the classifier removed,
  scored Sharpe **0.6738** and CAGR **0.1029** while trading 322 times instead of 1,004.

**Every one of those intervals crosses zero, including the strategy's own.**

The direction classifier showed an accuracy edge over always predicting "up" of
**+0.0001** on validation. It does not add value, and this repository does not claim that
it does.

**Paper trading demonstrates that this system operates. It does not demonstrate that it
makes money.** The daily job exists to prove the pipeline runs, survives a crash and
keeps its books. Paper P&L must never be quoted as a performance result.

Full numbers: [`reports/results/RESULTS.md`](reports/results/RESULTS.md).

---

## 2. Setup

Python 3.10 or newer. On Windows, install [GNU Make](https://www.gnu.org/software/make/)
and put it on `PATH`; every recipe below is otherwise identical on both platforms.

```bash
git clone https://github.com/Arp1tSingh/regimetrader.git
cd regimetrader
make setup
```

`make setup` creates `.venv` and installs `requirements.txt`. For the exact environment
these results were produced in, `requirements.lock` pins all 70 installed packages:

```bash
.venv/Scripts/python.exe -m pip install -r requirements.lock   # Windows
.venv/bin/python -m pip install -r requirements.lock           # Linux / macOS
```

**`make setup` does not reproduce the lock exactly, and that is expected.**
`requirements.txt` states minimum versions (`>=`), so a fresh install resolves newer
transitive dependencies than the ones this project was built against. In the clean-clone
check four of the 70 differed:

| package | `requirements.lock` | fresh `make setup` |
|---|---|---|
| narwhals | 2.26.0 | 2.27.1 |
| platformdirs | 4.12.4 | 4.13.0 |
| soupsieve | 2.10 | 3.0.1 |
| numpy | 2.5.3 | 2.5.4 |

Use `requirements.lock` when you want the environment the reported numbers came from. Use
`make setup` when you want a working environment. The clean-clone check described in §7
runs the fast suite against the newer set; see that section for the result.

The real-data pipeline needs network access (yfinance). **Tests and `make smoke` never
do.**

---

## 3. Commands

Every command below was run in the session that produced this handoff.

| What | Command | Network? |
|---|---|---|
| Lint + fast tests | `make check` | no |
| **Same tests, in a clean clone** | `make clean-clone-check` | no |
| Full test suite | `make test` | no |
| **Whole pipeline on synthetic data** | `make smoke` | no |
| Daily paper job | `make paper` | **yes** — runs an incremental ETL first |
| Dashboard | `make dashboard` | no |
| Real full ETL refresh | `make etl-full` | **yes** |
| Indicators and panel | `make features` | no |
| Refit regimes | `make regime` | no |
| Train classifiers | `make train` | no |
| Tune + validation baselines | `make backtest` | no |
| One-time test evaluation | `make final-test` | no |

`make smoke` is the single command that proves the whole system works end to end: ETL,
indicators, OLAP, regimes, training, tuning, baselines and a paper replay, all on
generated data.

### `make clean-clone-check` — run this after any commit

`make check` can be green in your working directory and fail for anyone who clones the
repository, because your directory holds state the repo does not. That happened **twice**
here: the entire `src/models/` package was silently excluded by an over-broad
`.gitignore` pattern, and three tests passed only because a 16 MB smoke database was
lying around from earlier runs. Neither is visible from inside the working directory.

This target clones the repository to a temp directory and runs the fast suite there.
It **reads, writes and moves nothing** in your working tree, and reuses your existing
virtualenv, so it costs about the same as `make check` (≈33 min) rather than a fresh
`pip install`. Last verified run: **475 passed, 35 deselected, 0 failed** in 32m42s.

Run it before you trust a green `make check`.

`make paper` runs an **incremental** load before trading, so it needs network. To skip
that and run against data already in the warehouse, call the job directly with
`--no-etl`:

```bash
python -m src.paper.daily_job --no-etl
```

### Running the real pipeline

```bash
make etl-full      # ~2,900 index bars, 52 instruments, 2015-01-02 onwards
make features
make regime
make train
make backtest      # tunes on validation ONLY
make final-test    # the one-time test evaluation (guarded)
```

> **Note on re-running.** `make etl-full`, `make features`, `make regime`, `make train`
> and `make backtest` were each run once, when that phase was built. They are **not**
> re-run as part of this handoff, deliberately: re-running `make backtest` would retune
> and append to `trial_log.csv`, and re-running `make train` would overwrite the model
> files that the frozen test result depends on. Reproduce them only on a fresh clone
> where no committed result would be disturbed.

**`make final-test` is the guarded one-time test evaluation.** It has been run and its
lock file (`reports/results/TEST_RUN_LOCK.json`) records `run_count: 2`. It will refuse to
run again. That refusal is the point — see §5.

### The daily job

```bash
make paper
```

It starts from the last committed paper date, processes one trading day at a time in a
single transaction, and is safe to re-run: a date already committed is skipped, not
reprocessed. Scheduling examples are in `deploy/cron.example` and
`deploy/github_actions_daily.yml`. **Nothing is installed** — deployment is your decision.

### The dashboard

```bash
make dashboard
```

Six screens (DECISIONS R55): the five read-only screens — market overview, OLAP
explorer, today's signals, backtest results, model diagnostics — plus the read-only
Bot demo (replay) screen below. It reads the warehouse and the result CSVs and
**recomputes no metric**, so a number on screen cannot drift from the same number in
`RESULTS.md`.

The target passes `--server.headless true`, so **Streamlit will not auto-open your
browser** — it prints the URL to open instead:

```
  Local URL: http://localhost:8501
```

That flag is not cosmetic. Without it, Streamlit's first run blocks on an interactive
"Email:" prompt that never returns in a non-interactive shell (`DECISIONS` R50).

To inspect the screens without a browser:

```bash
.venv/Scripts/python.exe scripts/dashboard_pages.py config/config.yaml
```

To point it at the synthetic warehouse instead:

```bash
set RT_DASH_CONFIG=config/config.smoke.yaml
make dashboard
```

### Demo mode — a temporary public URL for a viva

```bash
winget install cloudflare.cloudflared    # once
make demo-check                          # verify, launches nothing
make demo                                # dashboard + public tunnel
```

`make demo` starts the dashboard and a `cloudflared` quick tunnel, then prints the public
URL in a boxed banner. The examiner can open it on their own device instead of watching
your screen.

**The URL is random and changes on every run.** It cannot be written down in advance.
Read it off the screen. `Ctrl-C` stops both processes.

The tunnel is **additive, never required** — the warehouse is on your machine, so the
local dashboard works with no internet at all. If the link does not load, just show the
dashboard on your screen. It is read-only by construction (every query in
`src/dashboard/data.py` is a `SELECT`, asserted by tests), so an examiner poking at it
cannot alter anything.

`docs/DEMO_SCRIPT.md` has the full 16-minute walkthrough and a fallback table.

### Bot demo (replay) — and its safety limits

The **Bot demo (replay)** screen re-runs the frozen strategy over a few test-period days
on a **temporary copy** of the warehouse, so a visitor can watch the bot decide and then
check that its first-day orders match the recorded `backtest_main` run.

Two standing caveats on what it shows: the 50 names are today's constituents applied
backwards (a static universe — survivorship bias), and prices are split/dividend
adjusted, so share counts are illustrative rather than real tradable quantities.

Because the dashboard is publicly reachable, the screen is built for an anonymous
visitor:

- the warehouse is opened **read-only** (`mode=ro`), so a write raises rather than
  succeeding — there is a test that asserts exactly that;
- every run works on a **temporary copy that is deleted afterwards**, including when it
  fails — there is a test for the failure case too;
- **one run at a time** (a second visitor gets a "busy" message) and each run is
  **time-capped**;
- the start date comes from a **fixed list of five**; there is no text box, so nothing a
  visitor enters can reach a query;
- errors show a **generic message**; the detail goes to the server log;
- results are **cached per start date**, and `paper_live`, `final_test` and everything
  that writes to the warehouse are unreachable from this code path (also tested).

The screen is a **replay of already-scored data. It makes no performance claim.** It shows
no Sharpe, CAGR or drawdown, and a test asserts those names never appear on it.

The one flag this screen lifts is the test-period guard, and only in a call that also sets
`persist=False` and `write_signals=False` (`DECISIONS` R54). It replays days the authorised
run already scored; it never writes a result.

#### Why the fast path is legitimate

The screen recomputes a 267-trading-day tail rather than all 2,898, because a full
recompute costs ~46s against ~8s for the tail. That optimisation is only sound if every
feature is causal, so it is proved rather than asserted:

```bash
python scripts/bot_equivalence.py     # ~3 min; writes reports/results/bot_equivalence.json
```

For day 1 and day 5 it runs a **literal** recompute (every row after the day deleted, full
rebuild) beside the fast path and compares regime label, `prob_up` and target weights.
**Nothing is read from stored `fact_signal`** — both sides recompute through the same
pipeline, and a test corrupts a stored weight to prove the comparison can fail. It then
changes a price **after** the day, keeps that row, and confirms the day's answer does not
move. The dashboard shows the recorded numbers, and a failed proof is displayed rather than
hidden.

#### Publishing the link to GitHub

```bash
make demo-link            # start a tunnel and publish the URL to the `demo` release
make demo-link-check      # prints ALIVE or DEAD for the published URL
make demo-link-revive     # republish, but only if the URL is actually dead
```

The published link lives at
**https://github.com/Arp1tSingh/regimetrader/releases/tag/demo**

**This cannot be kept alive automatically, and the release says so.** A quick-tunnel
hostname is random per process: when the process dies the URL is gone permanently, and a
replacement gets a *different* URL. Nothing here runs a 24-hour watchdog, so if the link
dies while nobody runs a command, it stays dead. `make demo-link-check` exists so you
never have to guess which state it is in.

The release body carries its own generated-at time, the liveness command, and the line
*"A dead link in this section is expected behaviour, not a fault in the project."* That
is deliberate: a link that quietly rots into looking authoritative is the exact failure
this project exists to catch.

---

## 4. Reproducing the results

```bash
make setup && make check && make smoke
```

The full suite is **503 tests (468 fast, 35 slow)**, roughly **59 minutes** end to end;
`make check` runs the 468 fast tests only. `make smoke` takes about **18 minutes**. A
clean-clone verification was performed for this handoff — see §7.

---

## 5. Limitations and honesty

These are the things that would be easiest to hide and that a reader most needs.

### The universe is today's Nifty 50, held fixed

`config/nifty50_symbols.csv` is one static list stamped "as of 2026-10-10", and the
warehouse records no point-in-time index membership. Every symbol is loaded over the
whole period, so the backtests and the paper bot trade today's names — including BSE
Ltd, which entered the Nifty 50 only on 2026-09-30 (replacing Wipro) — as though they
had been constituents since 2015, while names that left the index earlier are absent
entirely. That survivorship bias is a real limitation: the results describe what these
fifty constants would have produced, not what a contemporaneous investor holding the
index of the day could have. The Bot demo (replay) screen carries a one-line version
of this note.

### The test-once protocol, and the `run_count: 2`

The test period was evaluated **once**. That is not a figure of speech: `final_test.py`
holds a lock file, and a second evaluation raises rather than re-deciding the strategy
against a fresh look at the numbers.

**The first run crashed and the rerun was authorised.** `b4_distribution` called
`run_backtest` without `allow_test_period=True`, so the guard fired and raised
`ValueError`. It was an implementation bug, not a data problem. The failure was reported,
and exactly one rerun was authorised by hand, with the reason recorded in
`test_run_log.jsonl`:

> `b4_distribution called run_backtest without allow_test_period=True; crash fix to
> complete B4 and the trial-log/manifest entries.`

No config, threshold, model, seed or metric changed. To show the rerun introduced nothing,
**all six outputs written by the crashed run were checksummed before the rerun and were
bit-identical afterwards**. The lock honestly records `run_count: 2`. A report claiming a
clean single run would misrepresent what happened. (`DECISIONS` R45.)

### 338 trials were logged

`trial_log.csv` holds 342 rows, of which **338 are counted experiments** (4 `calibration`
rows are excluded): 18 `strategy_grid`, 5 `regime_k`, 4 `model_year`, 311 `other`
(including the 300 B4 null simulations). Tuning used train and validation only.

One grid run is preserved separately as `tune_grid_validation_run1.csv` and is still
counted, because a selection-rule revision happened mid-way (R43) and discarding it would
have been the kind of quiet pruning this project exists to prevent.

### Every confidence interval crosses zero

Including the strategy's own `[−0.4561, 1.5437]`. **On this evidence nothing in this
project is distinguishable from luck.** That is the single most important sentence here.

### The classifier adds nothing

| model | accuracy | always-"up" baseline | **edge** | ROC-AUC |
|---|---|---|---|---|
| logreg | 0.506637 | 0.501345 | +0.005292 | 0.5140 |
| tree | 0.498187 | 0.501345 | **−0.003158** | 0.5157 |
| rf | 0.505789 | 0.501345 | +0.004444 | 0.5157 |
| xgb | 0.510760 | 0.501345 | +0.009415 | 0.5201 |

(n = 34,200 per model.) On validation the edge was **+0.0001**. AUCs of 0.514–0.520 are
barely distinguishable from a coin. The per-day top-minus-bottom-quintile return intervals
all contained zero (R35).

### Removing the classifier made the system better

V1 (regime only) beat V2 (regime + classifier) by 0.19 Sharpe and 4.8 points of CAGR,
while trading 322 times instead of 1,004. The classifier was a net negative.

### Costs dominate the result

V2 without costs scores Sharpe 0.8949 and CAGR 0.1115 against 0.4848 and 0.0548 with
them: **costs remove 0.41 Sharpe and 5.7 points of CAGR.** Note the no-cost Sharpe is the
only figure in this project with a confidence interval that nearly excludes zero, and it
is not a number anyone can trade.

### The B4 null is handicapped (R44)

The strongest single statement in the results is that the strategy's Sharpe sits above
**all 200** test-period permutations, whose 90th percentile was −0.2269 and whose best was
0.1198.

**Read it with a large discount.** The null is matched to the real run on capital — a
match measured on validation, where the null's mean weight exposure was 0.7468 against
the real run's 0.7487 — but **not on turnover**. On the test period the null executes
2,478–2,735 round trips against the real run's 1,004, about **2.6×** as many, because a
permuted ranking reshuffles daily while the real model's autocorrelated probabilities give
it stable holdings. The null pays roughly two and a half times the trading costs. **A null
handicapped that way is easier to beat**, so "beat every permutation" overstates the
evidence.

The asymmetry was measured and disclosed rather than corrected. Fixing it *after* seeing
these numbers would have been precisely the post-hoc adjustment this project exists to
prevent.

### The strategy was the worst run of 2026

Per-year returns, derived from the equity the authorised run had already stored
(`yearly_test_all_runs_DERIVED.csv`; every full-period figure was re-derived and tied out
to `summary_test.csv` before anything was written):

| year | V2 | B1 `^NSEI` | B2 | B3 | V1 | V2 no costs |
|---|---|---|---|---|---|---|
| 2024 | +19.39% | +8.47% | +19.65% | +16.99% | +13.56% | +27.41% |
| 2025 | +17.94% | +9.97% | +12.91% | +4.35% | +19.28% | +21.99% |
| 2026 (to 10-09) | **−17.52%** | −13.77% | −6.39% | −6.65% | −3.87% | −13.66% |

The full-period advantage came almost entirely from 2024 and 2025. In 2026 the strategy
was the worst of every run shown. One year of that size moves the full-period Sharpe more
than the whole 2024–2025 advantage — which is a plain statement of how unstable the
difference is.

The strategy also lost to all three baselines **on validation**, where it scored Sharpe
1.8328 against buy-and-hold's 1.9195, equal-weight's 3.4674 and the trend rule's 3.8990.

### An alternative explanation was never tested

The real run carries the *same* regime rules and the *same* capital exposure as its null,
yet returns far more. A classifier with no measurable direction edge should not produce
that gap. The gap is consistent with the real run's **stable, low-churn holdings
capturing whatever momentum exists** while the null's daily reshuffling captures none —
that is, the advantage may come from *holding behaviour* rather than from prediction.

**This was not tested.** It would need a dedicated check — for example re-running the real
configuration with its holdings permuted but its churn held fixed. Recorded as an open
question, not as a finding.

### Known measurement weakness

On `classifier_metrics.csv`, the `year` column reads **2024** for all four test rows, but
those rows cover the **whole** test window (n = 34,200 per model). The label is the first
year of the period, not the period itself. Cosmetic, but it will mislead anyone who reads
the column as a per-year figure.

---

## 6. What is where

| Path | Purpose |
|---|---|
| `AGENTS.md` | Entry point and rules for the agent |
| `docs/spec/` | Specification. **`DECISIONS.md` wins on conflicts**; `TESTS.md` holds the numeric fixtures |
| `docs/build/` | `BUILD_PLAN.md` (ordered tasks) and `PROGRESS.md` (progress log, session memory) |
| `docs/human/` | Report and viva material — for the student, not the agent |
| `docs/DEMO_SCRIPT.md` | A 15–20 minute viva walkthrough |
| `sql/`, `config/` | Schema, OLAP queries, configuration |
| `src/` | `etl`, `features`, `olap`, `regime`, `models`, `trading`, `backtest`, `paper`, `dashboard` |
| `tests/` | 503 tests |
| `reports/results/` | Result tables, `trial_log.csv`, `TEST_RUN_LOCK.json`, `RESULTS.md`, `leakage_checklist.md` |
| `deploy/` | Cron and GitHub Actions **examples** — nothing installed. `OPTION_B_FREE.md` is the prepared free-tier VPS deployment kit (Option B): systemd timer, wrapper, snapshot and backup scripts |
| `requirements.lock` | Exact pinned environment |

---

## 7. What was not done

Listed so nothing here is mistaken for complete:

- **End-to-end test suite** — deferred. The pipeline is covered by `make smoke` and by
  per-module tests, but there is no single test that asserts the full real-data chain
  reproduces end to end.
- **Turnover-matched null** — deferred (R44). The B4 null is matched on capital and not
  on turnover; the fix was deliberately not applied after seeing the results.
- **Association-rule mining, V3/V4 strategies, per-regime models** — Phase 9 stretch, not
  started.
- **Scheduler** — not installed, by decision (R22).
- **Paper trading has never traded.** The first real `paper_live` run started flat at
  2026-10-09, which is the test lock's `data_end_date`, so it correctly declined to
  process it. It will process the first trading day that follows.

---

## 8. Disclaimer

Backtest results are not predictions of future returns. Any live automated trading would
need a review of current SEBI and broker rules, which is out of scope.

This project was built as an academic warehouse and mining exercise. Its result is a
**negative finding, reported as such**: a direction classifier built on 2015–2022 data did
not produce a reliable, cost-adjusted edge on 2024–2026 data, and removing the classifier
*improved* the strategy. A system that does not demonstrably beat simply holding the index,
or holding an equal-weight basket, is not something to deploy.