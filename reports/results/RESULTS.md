# RESULTS — RegimeTrader

**Not investment advice. No real money. No broker integration.**

Test period: **2024-01-01 to 2026-10-09** (685 trading days). Frozen strategy: **rf**,
`entry_base` **0.504415**, `slots` **10**, selected on the 2023 validation year by the
activity filter (R39) and the B4 null filter (R40).

---

## 1. Headline results

All figures exactly as produced. Confidence intervals are moving-block bootstrap of
daily returns, block 10 days, 1,000 resamples, seed 42.

| run | Sharpe | 95% CI | CAGR | max drawdown | round trips |
|---|---|---|---|---|---|
| **V2 strategy (frozen)** | **0.4848** | **[−0.4561, 1.5437]** | **0.0548** | **0.1932** | 1,004 |
| B1 buy and hold `^NSEI` | 0.1576 | [−0.7746, 1.1081] | 0.0120 | 0.1559 | 1 |
| B2 equal weight, monthly | 0.7017 | [−0.1866, 1.6856] | 0.0882 | 0.1506 | 1,076 |
| B3 trend `sma_20 > sma_50` | 0.6117 | [−0.2256, 1.4971] | 0.0502 | 0.0975 | 819 |
| V0 classifier only, no regime rules | 0.4481 | [−0.5046, 1.4946] | 0.0596 | 0.2237 | 1,142 |
| **V1 regime filter only, no classifier** | **0.6738** | [−0.2337, 1.6914] | **0.1029** | 0.1637 | 322 |
| V2 without costs and slippage | 0.8949 | [−0.0516, 1.9391] | 0.1115 | 0.1566 | 1,000 |

### Against buy and hold, stated plainly

- **Sharpe: the strategy won.** 0.4848 against 0.1576.
- **CAGR: the strategy won.** 0.0548 against 0.0120.
- **Drawdown: the strategy lost.** 0.1932 against 0.1559 — it was *worse* than simply
  holding the index.

### Against the other baselines

- **B2 (equal weight) beat the strategy on all three measures**: Sharpe 0.7017 vs 0.4848,
  CAGR 0.0882 vs 0.0548, drawdown 0.1506 vs 0.1932.
- **B3 (a plain SMA trend rule, no machine learning) beat the strategy on Sharpe**
  (0.6117 vs 0.4848) **and on drawdown** (0.0975 vs 0.1932), while the strategy edged it
  on CAGR (0.0548 vs 0.0502).

### Year by year (the frozen strategy)

| year | days | return | Sharpe | max drawdown |
|---|---|---|---|---|
| 2024 | 246 | +19.39% | 1.3319 | 0.1371 |
| 2025 | 249 | +17.94% | 1.7356 | 0.0467 |
| 2026 (to 10-09) | 190 | **−17.52%** | −1.7241 | 0.1932 |

**Per-year returns for every run, derived from stored equity curves.**

`yearly_test.csv` contained only the strategy, so the baseline years below were
**derived** from the daily equity the authorised run had already persisted in
`fact_portfolio_daily`. Nothing was rescored and nothing was rerun. Before writing
the table, every run's **full-period** CAGR, Sharpe and max drawdown were re-derived from
the same stored equity and compared with `summary_test.csv`: all eighteen figures agreed
to within 1e-16. Source file: `yearly_test_all_runs_DERIVED.csv`.

| year | V2 strategy | B1 `^NSEI` | B2 equal weight | B3 trend | V1 regime only | V2 no costs |
|---|---|---|---|---|---|---|
| 2024 | **+19.39%** | +8.47% | +19.65% | +16.99% | +13.56% | +27.41% |
| 2025 | **+17.94%** | +9.97% | +12.91% | +4.35% | +19.28% | +21.99% |
| 2026 (to 10-09) | **−17.52%** | −13.77% | −6.39% | −6.65% | −3.87% | −13.66% |

The per-year picture sharpens the headline rather than softening it. The strategy's
full-period win over buy-and-hold came almost entirely from 2024 and 2025. **In 2026 it
was the worst of every run shown**, losing 17.5% where B2 lost 6.4% and buy-and-hold
lost 13.8%. One year of that size moves the full-period Sharpe more than the whole
2024-2025 advantage, which is a plain statement of how unstable the difference is.

## 2. What must be said about these numbers

### Every confidence interval crosses zero

All seven intervals — **including the strategy's own** [−0.4561, 1.5437] — contain zero.
On this evidence nothing here is statistically distinguishable from luck. The apparent
Sharpe advantage over buy-and-hold is not established.

### The classifier does not add value, and the ablation shows it

**V1 — the regime filter with the classifier removed — beat V2 by 0.19 Sharpe and 4.8
points of CAGR** (0.6738 / 0.1029 against 0.4848 / 0.0548), while trading 322 times
instead of 1,004. Everything the classifier contributed was a net negative.

This is consistent with the classifier's own scores, which show no edge over always
predicting up:

| model | accuracy | majority baseline | edge | ROC-AUC |
|---|---|---|---|---|
| logreg | 0.5066 | 0.5013 | +0.0053 | 0.5140 |
| tree | 0.4982 | 0.5013 | −0.0032 | 0.5157 |
| **rf (the frozen model)** | **0.5058** | **0.5013** | **+0.0044** | **0.5157** |
| xgb | 0.5108 | 0.5013 | +0.0094 | 0.5201 |

All four test-period ROC-AUC values sit between 0.514 and 0.520, marginally above the
0.5 coin flip. On the validation year the best model's edge was **+0.0001** and the
per-day top-quintile minus bottom-quintile return spreads had intervals that **all
contained zero**. **There is no evidence in this project that the classifier adds
value, and this report does not claim any.**

### Costs consume most of the result

V2 without costs scores **0.8949** Sharpe against **0.4848** with them. At the configured
0.30% round trip, transaction costs remove roughly **0.41 Sharpe and 5.7 points of
CAGR** (0.1115 → 0.0548). The strategy's edge is far too small to survive realistic
friction.

### The B4 null caveat (DECISIONS R44)

The permutation null beat-or-matched is the strongest single statement here: the
strategy's Sharpe of 0.4848 sits above **all 200** test-period permutations, whose 90th
percentile was **−0.2272** and whose best was **0.1198**.

**This must be read with a large discount.** B4 is matched to the real run on capital —
that match was measured on validation, where the null's mean weight exposure was 0.7468
against the real run's 0.7487 (R44) — but **not on turnover**. On the test period the
null executes 2,478–2,735 round trips against the real run's 1,004, about **2.6×** as
many, because a permuted ranking reshuffles daily while the real model's autocorrelated
probabilities give it stable holdings. The null therefore pays roughly two and a half
times the trading costs. A null handicapped that way is easier to beat, so "beat every
permutation" overstates the evidence.

The same asymmetry was measured on validation (R44) and was **disclosed rather than
corrected**, because changing the null after seeing its numbers would have been a
post-hoc adjustment.

### The untested alternative explanation

The strategy beats its own null by a wide margin while its classifier shows no measurable
edge. The gap is consistent with the strategy's **stable, low-churn holdings** capturing
whatever momentum exists, rather than with any predictive power. **This was never
tested.** Distinguishing the two would require, for example, running the real
configuration with its holdings permuted but its churn held fixed. Until that is done,
the honest position is that the result is unexplained.

---

## 3. Protocol, honesty and accounting

### The run crashed once and was completed under run_count 2

The first invocation of `final_test.py` wrote the lock with `run_count: 1`, scored the
strategy, the ablations, the baselines and the classifiers, and then **crashed** in the
B4 step: `b4_distribution` called `run_backtest` without `allow_test_period=True`, so
the test-period guard fired and raised `ValueError`. It was my bug, not a data problem.

I stopped and reported rather than re-running. The fix was one keyword parameter
defaulting to `False`, plumbed from `final_test.py` — no config, threshold, model, seed
or metric changed. A single rerun was authorised with the reason
`"b4_distribution called run_backtest without allow_test_period=True; crash fix to
complete B4 and the trial-log/manifest entries."`

**The lock records `run_count: 2`**, and the reason is stored in
`reports/results/test_run_log.jsonl`. **All six pre-existing outputs were checksummed
before the rerun and were bit-identical afterwards**, which confirms the rerun neither
changed nor introduced a number. The scores in this report are the ones the rerun
produced.

### Trial count

**338 counted trials** (342 rows less 4 `calibration` rows, which the spec excludes):
5 `regime_k` + 4 `model_year` + 18 `strategy_grid` + **300 B4 null simulations**
(validation) + 3 validation baselines + 1 selection-rule revision + **7 final-test runs**.

Run 1 of the tuning grid is preserved unaltered in
`reports/results/tune_grid_validation_run1.csv` and remains counted.

### Leakage checklist

`reports/results/leakage_checklist.md` — 14 checks, each verified against the real
warehouse rather than asserted from the code. Regimes fitted through 2022-12-31;
`dim_model.trained_through` is 2022-12-31 for all 16 model rows with the embargo
assertion enforced in the trainer; fills equal the same day's open adjusted by slippage
(never the previous close); zero persisted test-run rows before 2024-01-01; the frozen
config's SHA-256 is unchanged from the pre-run commit.

### What was never done

The test period was not consulted before the freeze. No config, threshold, model or
metric was changed in response to any number in this report. No `paper_live` run was
performed and Phase 7 has not started.

---

## 4. Conclusion

The regime filter is the part of this system with a plausible mechanism, and on the test
period it outperformed the full system on its own. The classifier did not earn its
place: removing it improved both Sharpe and return while trading two-thirds as often.
The strategy beat buy-and-hold on Sharpe and return and lost to it on drawdown; it lost
to plain equal-weight and to a simple moving-average trend rule. Every confidence
interval crosses zero. After costs, and measured against a null that turns out to be
handicapped by its own turnover, **this project does not demonstrate a reliable edge.**