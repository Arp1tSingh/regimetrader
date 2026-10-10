# RegimeTrader

Regime-aware stock direction predictor and paper-trading bot, built on a star-schema data warehouse of Nifty 50 daily history.
Data Warehousing and Mining course project. **Academic prototype. Not investment advice. No real money.**

> This README is a stub. The coding agent expands it in task T10.1 (setup, commands, structure, how to reproduce).

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
