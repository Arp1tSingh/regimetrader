# RegimeTrader

Regime-aware stock direction predictor and paper-trading bot, built on a star-schema data warehouse of Nifty 50 daily history.
Data Warehousing and Mining course project. **Academic prototype. Not investment advice. No real money.**

## Read this before looking at any number

**The one-time out-of-sample evaluation found no demonstrated edge.** On the test
period (2024-01-01 to 2026-10-09) the frozen strategy scored a Sharpe of 0.485 with a
95% interval of **[−0.456, 1.544]** — an interval that crosses zero. It beat
buy-and-hold on Sharpe and return and lost to it on drawdown, and it lost to a plain
equal-weight portfolio and to a simple moving-average trend rule. The full numbers are
in `reports/results/RESULTS.md`.

**Paper trading demonstrates that this system operates. It does not demonstrate that it
makes money.** The paper job exists to prove the pipeline runs daily, survives a crash
and keeps its books — not to accumulate evidence of performance. Paper P&L must never be
quoted as a performance result, and nothing in this repository should be read as a
recommendation to trade this system. The classifier in particular showed an accuracy
edge of **+0.0001** over simply always predicting "up".

> The full README (setup, commands, structure, reproduction) is written in task T10.1.

## How to build this with a coding agent

1. Put this whole folder in your project directory and open it with your agent (Claude Code or similar).
2. `config/nifty50_symbols.csv` is **not** included (the current constituent list has to come from NSE). Either place it there yourself (format: `config/nifty50_symbols.FORMAT_EXAMPLE.csv`) or let the agent try to download it in task T1.1; it will ask you if that fails.
3. Start the agent with:

   > Read AGENTS.md and build the project by following docs/build/BUILD_PLAN.md. Update docs/build/PROGRESS.md after every task.

   `CLAUDE.md` simply imports `AGENTS.md`; other agents read `AGENTS.md` directly.

## What is where

| Path | Purpose |
|---|---|
| `AGENTS.md` | Entry point and rules for the agent |
| `docs/spec/` | The specification (`DECISIONS.md` wins on conflicts; `TESTS.md` holds the numeric test fixtures) |
| `docs/build/BUILD_PLAN.md`, `PROGRESS.md` | Ordered tasks with acceptance checks; the agent's progress log |
| `docs/human/` | Problem statement, timeline and team, report and viva preparation (for you, not the agent) |
| `sql/`, `config/` | Ready-made schema, OLAP queries and configuration |
| `Makefile`, `pyproject.toml`, `requirements.txt` | Ready-made tooling |

## Disclaimer

Backtest results are not predictions of future returns. Any live automated trading would need a review of current SEBI and broker rules, which is out of scope.

This project was built as an academic warehouse and mining exercise. Its result is a
**negative finding, reported as such**: a direction classifier built on 2015-2022 data
did not produce a reliable, cost-adjusted edge on 2024-2026 data, and removing the
classifier *improved* the strategy. A system that demonstrably does not beat simply
holding the index, or holding an equal-weight basket, is not something to deploy.

## Paper trading

`python -m src.paper.daily_job` runs the daily operation. Scheduling examples live in
`deploy/cron.example` and `deploy/github_actions_daily.yml`; **nothing is installed** —
deployment is your decision.

The first real run started flat at the latest trading date (2026-10-09) and correctly
declined to process it: that date is the test lock's `data_end_date`, and the job is not
permitted to write signals on or before it. It will process the first trading day that
follows it.
