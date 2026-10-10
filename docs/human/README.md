# docs/human: for the student, not the agent

`01`, `09` and `10` are the original blueprint files, unchanged. The coding agent does not need them.
The agent-facing specification is in `docs/spec/`, which is a rewrite of blueprint files `02` to `08` (plus `DECISIONS.md` and `TESTS.md`).

## Where blueprint section references now point

The originals refer to sections like "`04`, section 5". Use this map when following those references.

| Reference in `10` (practice material) | Now found in |
|---|---|
| `03`, section 2 (star schema) | `docs/spec/03` section 2 and `sql/schema.sql` |
| `03`, section 4 (ETL) | `docs/spec/03` section 4 |
| `03`, section 6 (OLAP, OLTP versus OLAP) | `docs/spec/03` section 6 and `sql/olap/*.sql` |
| `04`, section 2 (preprocessing) | `docs/spec/04` section 3 (section 2 is now the feature catalogue) |
| `04`, section 5 (K-Means numerical) | `WORKED_EXAMPLES.md` section 1; method in `docs/spec/04` section 5 |
| `04`, section 6 (compare classifiers) | `docs/spec/04` section 6 |
| `04`, section 7 (confusion matrix) | `WORKED_EXAMPLES.md` section 2; method in `docs/spec/04` section 7 |
| `04`, section 8 (Apriori) | `WORKED_EXAMPLES.md` section 3; method in `docs/spec/04` section 8 |
| `05`, `06` (strategy, backtest) | Same file numbers in `docs/spec/` |
| `07` (project structure) | Was missing from the blueprint; rewritten as `docs/spec/07_project_structure.md` |

Everything the agent decides differently from the blueprint is listed in `docs/spec/DECISIONS.md`. Read it before the viva: it is the honest list of what was clarified or added.
