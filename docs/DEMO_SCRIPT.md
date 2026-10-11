# Demo script — 15 to 20 minutes

A walkthrough for the viva. Every command here has been run and every number quoted has
been produced by that command. Nothing here is aspirational.

**If a command is slow, do not wait for it live — every step has a fallback below.**

---

## Before you start

Have open, in this order:

1. `README.md` — the summary of the result
2. `reports/results/RESULTS.md` — the full report
3. `reports/results/trial_log.csv` — the experiment log
4. `reports/results/TEST_RUN_LOCK.json` — the one-time-evaluation lock

The dashboard should already be running in a second browser tab, and the tunnel banner
should already be on screen (see §0).

---

## 0. Before they walk in — start the demo tunnel (2 min, then leave it running)

**Do this before the viva, never during it.** A tunnel that is first attempted in front
of an examiner is a demo risk.

One-time install, if you have not already:

```bash
winget install cloudflare.cloudflared
```

Verify everything without launching anything:

```bash
make demo-check
```

It prints `pre-flight OK` if cloudflared is present, the warehouse exists and the results
CSVs are there. If it says `cloudflared : MISSING`, install it. Do not carry on until this
passes.

Then start it:

```bash
make demo
```

It launches the dashboard and a temporary public tunnel, and prints this:

```
==========================================================================
  DEMO IS LIVE - give the examiner this URL
==========================================================================

    https://some-random-words.trycloudflare.com

  It is random and changes every run. Read it aloud.
  Local fallback if they cannot reach it: http://localhost:8501
======================================================================
```

**The URL is random and different every single run.** It cannot be written down in
advance and it will not work tomorrow. Read it off the screen and say it out loud. Leave
the terminal visible.

**Tell them early, before you start talking about results:**

> "I have the dashboard on a temporary public link if you would rather look at it on your
> own device than on my screen. If it does not load, no problem, I have it here."

That last sentence matters. It means the tunnel failing is not a failure of the demo.

Leave the command running for the whole viva. Press **Ctrl-C** when you are done to shut
both the tunnel and the dashboard down.

---

## 1. The claim, stated up front (1 min)

**Show:** `README.md` §1.

Say the result first, before showing any chart:

> "This is a negative result, reported as such. The classifier did not add value. On the
> out-of-sample period the strategy scored a Sharpe of 0.4848 with a 95% interval of
> −0.456 to 1.544 — an interval that crosses zero. And removing the classifier entirely
> made the system better."

Do not bury this. If you lead with the equity curve, the examiner's first question will
be "does this beat buy and hold?" and you will have spent the next five minutes
defending rather than explaining.

---

## 2. The one-time evaluation and why it was run once (2 min)

**Show:** `reports/results/TEST_RUN_LOCK.json`

```bash
type reports\results\TEST_RUN_LOCK.json     # Windows
cat  reports/results/TEST_RUN_LOCK.json     # Linux / macOS
```

Point at three fields:

- `run_count: 2` — **say this out loud.** The first run crashed in the B4 step because a
  helper called the backtest without the guard flag, so the guard fired and raised. That
  was an implementation bug, not a data problem. A human authorised exactly one rerun and
  the reason is recorded in the lock.
- `data_end_date: 20261009` — the test period ends here, and this same field is what
  stops the live paper job from writing anything on or before it.
- `frozen_config_hash` — the frozen config has not changed since.

> "To prove the rerun introduced nothing, all six outputs written by the crashed run were
> checksummed before the rerun and were bit-identical afterwards."

**Fallback:** none needed. This is the strongest single thing in the project and it costs
30 seconds.

---

## 3. The results table, with every baseline (3 min)

**Show:** `reports/results/RESULTS.md` §1, or the dashboard **Backtest results** screen.

| run | Sharpe | 95% CI | CAGR | max DD | trades |
|---|---|---|---|---|---|
| **V2 strategy** | **0.4848** | **[−0.4561, 1.5437]** | 0.0548 | 0.1932 | 1,004 |
| B1 `^NSEI` | 0.1576 | [−0.7746, 1.1081] | 0.0120 | 0.1559 | 1 |
| B2 equal weight | 0.7017 | [−0.1866, 1.6856] | 0.0882 | 0.1506 | 1,076 |
| B3 trend `sma_20 > sma_50` | 0.6117 | [−0.2256, 1.4971] | 0.0502 | 0.0975 | 819 |
| V0 classifier only | 0.4481 | [−0.5046, 1.4946] | 0.0596 | 0.2237 | 1,142 |
| **V1 regime only** | **0.6738** | [−0.2337, 1.6914] | **0.1029** | 0.1637 | **322** |
| V2 without costs | 0.8949 | [−0.0516, 1.9391] | 0.1115 | — | 1,000 |

Three things to say:

1. **Every interval crosses zero, including ours.** We are not distinguishable from luck.
2. **V1 beat V2.** V1 is the same system with the classifier deleted. It scored higher
   Sharpe *and* higher CAGR while trading 322 times instead of 1,004.
3. **Costs are the story.** Removing costs lifts Sharpe from 0.485 to 0.895 — 0.41 of
   Sharpe and 5.7 points of CAGR. The edge was never large enough to pay for itself.

**On the dashboard**, point at the red dashed zero line on the Sharpe-with-intervals
chart. Every bar straddles it.

---

## 4. The per-year breakdown, including the bad year (2 min)

**Show:** `reports/results/RESULTS.md`, per-year table. Also on the dashboard.

| year | V2 | B1 | B2 | B3 | V1 | V2 no costs |
|---|---|---|---|---|---|---|
| 2024 | +19.39% | +8.47% | +19.65% | +16.99% | +13.56% | +27.41% |
| 2025 | +17.94% | +9.97% | +12.91% | +4.35% | +19.28% | +21.99% |
| 2026 (to 10-09) | **−17.52%** | −13.77% | −6.39% | −6.65% | −3.87% | −13.66% |

> "The whole-period result comes from 2024 and 2025. In 2026 we were the worst run on the
> page, losing 17.5% where equal-weight lost 6.4%. One year that size moves the full-period
> Sharpe more than the whole 2024–25 advantage. That is how unstable this difference is,
> and it is why the interval crosses zero."

If asked where these came from: derived from the equity the authorised run had already
stored, with all eighteen full-period figures re-derived and tied out to `summary_test.csv`
before anything was written. No rescore, no rerun.

```bash
.venv\Scripts\python.exe scripts\check_results_yearly.py
```

Expected output: `RESULTS.md per-year table matches yearly_test_all_runs_DERIVED.csv exactly.`

---

## 5. The classifier's own numbers (2 min)

**Show:** dashboard **Model diagnostics**.

| model | accuracy | always-"up" | edge | ROC-AUC |
|---|---|---|---|---|
| logreg | 0.506637 | 0.501345 | +0.005292 | 0.5140 |
| tree | 0.498187 | 0.501345 | **−0.003158** | 0.5157 |
| rf | 0.505789 | 0.501345 | +0.004444 | 0.5157 |
| xgb | 0.510760 | 0.501345 | +0.009415 | 0.5201 |

> "The edge is measured against always predicting 'up'. On validation it was +0.0001. AUCs
> of 0.51 to 0.52 are a coin with a slight lean. **This project does not claim the
> classifier adds value**, because the evidence does not support that claim."

Mention the decision tree exclusion: it produces only three distinct probabilities with
76% of rows tied at one value, so any threshold at or below that selects most of the
market and the tie is broken alphabetically. It was excluded from tuning (R31) but kept in
this table — we did not delete the inconvenient model.

---

## 6. The null, and the caveat that weakens it (2 min)

**Show:** `reports/results/summary_test.csv` and the B4 section of `RESULTS.md`.

> "The strongest single statement: our Sharpe sits above all 200 permutation nulls, whose
> 90th percentile was −0.227 and whose best was 0.120."

Then immediately undercut it, which is the point:

> "But the null is handicapped. It is matched to us on capital — measured on validation,
> weight exposure 0.7468 against our 0.7487. It is **not** matched on turnover: it trades
> 2,478 to 2,735 times against our 1,004, about 2.6× as many, because a shuffled ranking
> reshuffles daily while our model's autocorrelated probabilities give it stable holdings.
> So it pays two and a half times the costs. **A null handicapped that way is easier to
> beat**, and saying 'we beat every permutation' without saying this would overstate the
> evidence."

And the alternative explanation:

> "There is another explanation we never tested. Our run has the same regime rules and the
> same exposure as the null, yet returns far more. A classifier with no measurable edge
> should not produce that. It may be our low-churn holdings capturing momentum rather than
> any prediction. We did not test it, so we are not claiming it."

---

## 7. How we know we did not cheat (2 min)

**Show:** `reports/results/leakage_checklist.md`.

14 checks, each verified against the real warehouse, not asserted. Call out:

- features for day *t* use data up to and including the close of *t*; only the label looks
  forward;
- scalers, winsorizers and K-Means fitted on training data only, applied frozen forward;
- chronological splits, never shuffled;
- the paper job may not write signals on or before the lock's `data_end_date` — this one
  was **caught in practice**: the first live run wrote a signal for 2026-10-09, it was
  rolled back, and there is now a regression test for it.

Then the trial log:

```bash
.venv\Scripts\python.exe -c "import pandas as pd; t=pd.read_csv('reports/results/trial_log.csv'); print(len(t),'rows;',(t['kind']!='calibration').sum(),'counted')"
```

Expected: `342 rows; 338 counted`. Say: 18 grid points, 5 k-selections, 4 model-years, and
300 null simulations. Tuning used train and validation only.

---

## 8. The live system and the warehouse (1–2 min)

**Show:** dashboard **Market overview** and **OLAP explorer**.

- Market overview: index coloured by regime bands, India VIX, the stale-data banner.
- OLAP explorer: the sector-by-quarter heat map, 15 sectors across 48 quarters of mean
  daily return, with year/quarter/regime filters and a sector→stock drill-down.

Say:

> "The dashboard is read-only. Every number it shows is read from a file an authorised run
> already wrote — it recomputes no metric, so it cannot disagree with the report. There
> are tests that fail if a chart drops a baseline or trims the date range."

**If asked "is it trading?":**

> "No. Paper trading demonstrates the system operates, not that it makes money. The first
> live run started flat at 2026-10-09, which is exactly the test lock's end date, so it
> correctly refused to process it and exited cleanly. Nothing has traded yet, and paper
> P&L would never be performance evidence."

---

## 9. Close on the negative result (30 sec)

> "To summarise: the warehouse and the pipeline work end to end, the one-time evaluation
> protocol held, the dashboard and paper job run — but the strategy did not demonstrate an
> edge, the classifier added nothing, and removing it improved the system. The honest
> output of this project is that finding, with the evidence and the caveats attached."

---

## Fallbacks if something is slow or fails

**Never let a live command stall the demo.** Every step has a file you can open instead.

| Problem | Do this instead |
|---|---|
| `make demo-check` says cloudflared MISSING | Install it **now**, not later: `winget install cloudflare.cloudflared`. The script also looks in the standard install folder, so a stale PATH is not the cause. |
| The tunnel URL will not load for them | Say the sentence you promised: "no problem, I have it here." Switch to the local dashboard. Nobody marks you down for this. |
| The tunnel did not print a URL | It prints `LOCAL ONLY` and carries on without it. Just use the local dashboard. Do not retry live. |
| Port 8501 is taken | `make demo` accepts `--port`, or run `.venv/Scripts/python.exe scripts/demo_tunnel.py --port 8502`. |
| No internet in the exam room | The tunnel never connects. The local dashboard works regardless, because the warehouse is on your machine. This is why the tunnel is additive and never required. |
| The published release link is dead | `make demo-link-revive`. It only republishes if the link is actually dead, so it is safe to run at any time. **Do not assume the link is live** — `make demo-link-check` tells you the truth in about 4 seconds. |
| Laptop goes to sleep mid-demo | Turn sleep off in Settings → System → Power before you start. |
| `make smoke` takes too long | **Do not run it live.** It takes **about 18 minutes**. Open `reports_smoke/results/` and the committed log lines in `docs/build/PROGRESS.md`. Say it passes end to end on synthetic data. |
| Dashboard will not start | Show `reports/results/RESULTS.md`. Every chart on the dashboard reads the same CSVs. |
| A query is slow on the OLAP screen | Set the year filter to 2024. The heat map only needs one year. |
| Asked to run the full test suite | **Do not.** It takes ~59 minutes. Say: "503 tests, 468 fast, 35 slow, about an hour. I have the log." Then show `docs/build/PROGRESS.md`. |
| Asked to re-run the backtest | **Do not.** `make final-test` will refuse — that is the guard working. Show `TEST_RUN_LOCK.json` and explain the protocol. |
| Anything at all fails on the projector | Fall back to the numbers in §3–§6. The result does not depend on anything running. |

## Timing

| Step | Topic | Minutes |
|---|---|---|
| 0 | Start the tunnel and leave it running | 2 (before they arrive) |
| 1 | The claim up front | 1 |
| 2 | One-time evaluation, `run_count: 2` | 2 |
| 3 | Results table with baselines | 3 |
| 4 | Per-year, including 2026 | 2 |
| 5 | Classifier numbers | 2 |
| 6 | Null and its caveat | 2 |
| 7 | Leakage checks and trial count | 2 |
| 8 | Dashboard, live system | 1.5 |
| 9 | Close | 0.5 |
| | **Total** | **~16 min** |

Step 0 is not part of the 16 minutes — do it beforehand.