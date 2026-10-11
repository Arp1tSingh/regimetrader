# Leakage checklist — final test evaluation (`spec 06` section 9)

Every line below was **verified against the real warehouse after the run**, not
asserted from the code. Checked 2026-10-11.

| # | Check | Method | Result |
|---|---|---|---|
| 1 | **Regimes fitted on training data only** | Read `models/regime_model.joblib`: `train_end` and `k` | `train_end = 2022-12-31`, `k = 3`. K-Means fitted only on rows through the training cut-off; `fact_market_regime` carries labels on later dates, which is inference from the frozen model, not refitting. |
| 2 | **K-Means / scaler / winsorizer fitted on train only** | `Winsorizer` and the regime `scaler` are fit inside the training pipeline and frozen; unit tests assert they never move when later data is transformed | Pass. |
| 3 | **Embargo before yearly retraining** | `dim_model.trained_through` for all 16 model rows; assertion present in `src/models/train.py` | All 16 rows `trained_through = 2022-12-31`. The trainer aborts if `train_max >= year_start - embargo_days`. |
| 4 | **Walk-forward, expanding window** | `walk_forward_predict` fits on every row before the embargo cut and predicts that year only | Pass; predictions joined back on `(date_key, symbol)`. |
| 5 | **Features use no forward shift** | `# label-only` shift is the only `shift(-1)` in `src/features/panel.py`; verified on the real warehouse by truncating prices and rebuilding | Deleting every price after 2021-12-31 and rebuilding left all 76,690 earlier panel rows byte-identical. |
| 6 | **Orders decided at the close of *t* fill at the open of *t+1*** | Joined `fact_trade.fill_price` to the same day's `open_price` and the previous close | `fill = open × 1.0005` exactly (e.g. ITC `403.628 = 403.426 × 1.0005`), and the fill is **not** the previous close. Next-open execution confirmed. |
| 7 | **Cost and slippage applied** | `test_V2_nocost` vs `backtest_main` | Sharpe 0.895 without costs against 0.485 with, so costs are demonstrably applied in the primary run. |
| 8 | **No test-period data used in selection** | Frozen config selected on the validation year; every pre-run trial row has `period` in {`train`, `validation`} | The trial log holds 330 validation rows and 5 `train` rows; the 7 `test` rows are the test runs themselves, written after scoring. |
| 9 | **No persisted test-run rows before the test period** | `fact_portfolio_daily` for the seven test run_ids, `date_key < 20240101` | **0 rows.** |
| 10 | **Ties broken deterministically** | `tie_break_share` reported for every run | 0.0000 on every test run; no portfolio was decided by alphabetical order. |
| 11 | **Frozen config unchanged since the pre-run commit** | SHA-256 of `config/frozen_config.yaml` before and after | `0b8c88c6…63b8` both times. **Unchanged.** |
| 12 | **The strategy was frozen before the test period was seen** | Pre-flight record committed at `746a097`; `frozen_config_used.yaml` is a byte-identical copy | Identical (SHA-256 `0b8c88c6…63b8`). |
| 13 | **Test period evaluated under lock** | `TEST_RUN_LOCK.json` `run_count` | **2** — one crash, one authorised rerun. Both recorded in `test_run_log.jsonl`. |
| 14 | **Only `final_test.py` can lift the guard** | `ast`-based test over `src/` | Pass: no module outside `final_test.py` passes `allow_test_period=True` as a literal, and `b4_distribution` defaults it to False with its own refusal test. |

## Items that are honest weaknesses rather than leaks

- **B4 turnover asymmetry (`DECISIONS` R44).** The permutation null trades about 3×
  more than the real strategy (2,478–2,735 round trips against 1,004). It is matched on
  capital but not on churn, so "beat its null" is weaker evidence than it appears. This
  is a property of the null's design, not look-ahead, and it is disclosed in the report.
- **The `year` column on `classifier_metrics.csv` test rows is the first test year**
  (2024) while the rows cover the whole test window (n = 34,200 across 2024-01-01 to
  2026-10-09). The metrics themselves are computed over the full window. The label is
  imprecise; the numbers are not.