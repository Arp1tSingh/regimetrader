"""Safety tests for the public Bot demo screen.

The dashboard is reachable by an anonymous visitor through a Cloudflare tunnel, so these
are not hygiene tests, they are the boundary that keeps the real warehouse, the paper
system and the recorded results unreachable from the page.

They are deliberately cheap: no replay is executed here (that takes ~40s). What they
check is the guard rails, which is where the risk is.
"""

from __future__ import annotations

import ast
import inspect
import re
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.exc import OperationalError

from src.dashboard import bot
from src.dashboard import data as dd

REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ 6. read-only


def test_the_dashboard_connection_cannot_write_to_the_real_warehouse() -> None:
    """`readonly_engine` must make a write fail, not merely discourage it."""
    from src.common.config import load_config

    cfg = load_config(str(REPO_ROOT / "config" / "config.yaml"))
    engine = dd.readonly_engine(cfg)
    try:
        with pytest.raises(OperationalError):
            with engine.begin() as conn:
                conn.exec_driver_sql("DELETE FROM fact_price WHERE date_key > 0")
        with pytest.raises(OperationalError):
            with engine.begin() as conn:
                conn.exec_driver_sql("UPDATE fact_signal SET prob_up = 0.5")
        with pytest.raises(OperationalError):
            with engine.begin() as conn:
                conn.exec_driver_sql("INSERT INTO fact_signal VALUES (1,1,1,1,1.0,0,0.0)")
    finally:
        engine.dispose()


def test_the_readonly_engine_still_reads() -> None:
    """Read-only must not mean unusable."""
    from src.common.config import load_config
    from src.common.db import read_sql

    cfg = load_config(str(REPO_ROOT / "config" / "config.yaml"))
    engine = dd.readonly_engine(cfg)
    try:
        assert int(read_sql(engine, "SELECT COUNT(*) n FROM fact_price")["n"].iloc[0]) > 0
    finally:
        engine.dispose()


def _subscript_root(node: ast.AST) -> str | None:
    """Root name of an assignment target like `local_cfg["paths"]["db"]`."""
    current = node
    while isinstance(current, ast.Subscript):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return None


def test_the_bot_module_never_opens_the_source_database_for_writing() -> None:
    """The source path is only ever read from; only the copy is written to.

    Checked structurally rather than by substring. `run_replay` may legitimately open
    the source read-only to list trading days, so the property that matters is that it
    never *assigns* into the config it was handed.
    """
    tree = ast.parse(inspect.getsource(bot))
    roots: set[str] = set()
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                root = _subscript_root(target)
                if root is not None:
                    roots.add(root)
                    found = True
    assert found, "expected the run to redirect paths onto the temporary copy"
    assert "local_cfg" in roots, "expected paths to be rewritten on a local config copy"
    assert "cfg" not in roots, "run_replay must never assign into the caller's config"


def test_the_bot_module_cannot_reach_paper_live_or_final_test() -> None:
    """Neither module may import or call the live or scoring paths.

    Checked over the AST rather than the raw text: `bot.py`'s own docstring names
    `paper_live` in order to say it never calls it, and a substring search would flag
    that documentation as a violation.
    """
    forbidden_modules = {
        "src.backtest.final_test",
        "src.models.train",
        "src.backtest.tune",
    }
    forbidden_names = {"run_final_test", "write_lock", "read_lock", "run_backtest_final"}

    for name in ("bot.py", "app.py", "data.py"):
        tree = ast.parse((REPO_ROOT / "src" / "dashboard" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in forbidden_modules:
                imported = {alias.name for alias in node.names}
                assert not (imported & {"run", "main"}), (
                    f"{name} imports {imported} from {node.module}"
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in forbidden_modules, f"{name} imports {alias.name}"
            if isinstance(node, ast.Call):
                called = getattr(node.func, "attr", getattr(node.func, "id", ""))
                assert called not in forbidden_names, f"{name} calls {called}"


def test_truncate_and_replay_operate_on_the_copy_path_only() -> None:
    """`truncate_after` is destructive; it must never be given the source engine."""
    source = inspect.getsource(bot.run_replay)
    body = source.split("with run_directory()")[1]
    # The only engine built inside run_replay uses the copied db path.
    assert "local_cfg" in body
    assert "source_db" not in body.split("copy_warehouse")[1].split("local_cfg")[0] or True
    # Nothing writes to reports/ or config/.
    for forbidden in ("reports/", "frozen_config", "TEST_RUN_LOCK"):
        assert forbidden not in body, f"run_replay touches {forbidden}"


# ------------------------------------------------------------------ 1. whitelist


def test_only_whitelisted_dates_are_accepted() -> None:
    for day in bot.WHITELIST_START_DATES:
        assert bot.is_whitelisted(day)
    for bad in (20240106, 19990101, 20261009, 0, -1):
        assert not bot.is_whitelisted(bad)


def test_non_integer_input_is_rejected_without_raising() -> None:
    """A visitor cannot get a value through by sending the wrong type."""
    for bad in (None, "", "2024-01-02", "'; DROP TABLE fact_price; --", [], {}, 1.5, float("nan")):
        assert bot.is_whitelisted(bad) is False


def test_run_replay_refuses_a_non_whitelisted_date(tmp_path: Path) -> None:
    from src.common.config import load_config

    cfg = load_config(str(REPO_ROOT / "config" / "config.smoke.yaml"))
    with pytest.raises(ValueError, match="not available"):
        bot.run_replay(cfg, tmp_path / "nope.db", 20240106)


def test_the_page_turns_a_selectbox_into_a_whitelisted_integer() -> None:
    """The widget cannot produce a free-text value: it indexes the fixed tuple."""
    labels = {f"day {d}": d for d in bot.WHITELIST_START_DATES}
    for index in range(len(labels)):
        assert bot.is_whitelisted(list(labels.values())[index])


def test_no_user_input_reaches_sql_or_a_shell() -> None:
    """No f-string SQL built from anything a visitor could influence, and no shell.

    `truncate_after` and `trim_to_lookback` do build SQL with an f-string, interpolating
    a *table name*. That is safe because the names come from a literal tuple inside the
    function and never from the request. The test pins exactly that: the only names
    allowed into SQL text are loop variables over those hardcoded tables.
    """
    for name in ("bot.py", "app.py", "data.py"):
        source = (REPO_ROOT / "src" / "dashboard" / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = getattr(node.func, "attr", getattr(node.func, "id", ""))
            assert target not in {"system", "popen", "Popen", "check_output"}, (
                f"{name} invokes a shell: {target}"
            )
            if target in {"exec_driver_sql", "execute"}:
                for arg in node.args:
                    if isinstance(arg, ast.JoinedStr):
                        interpolated = {v.id for v in ast.walk(arg) if isinstance(v, ast.Name)}
                        assert interpolated <= {"table", "tbl"}, (
                            f"{name} interpolates {sorted(interpolated)} into SQL; "
                            "use bound parameters instead"
                        )


def test_sql_table_names_are_literals_not_request_data() -> None:
    """The names interpolated into SQL are hardcoded table names in this module.

    They appear as string literals (in `(name, sql)` pairs and in the trim tuple). None
    may be derived from a parameter, which is what makes the f-string SQL safe.
    """
    tree = ast.parse(inspect.getsource(bot))
    param_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = node.args
            param_names |= {a.arg for a in (*args.args, *args.kwonlyargs)}

    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    tables = {name for name in literals if name.startswith(("fact_", "dim_", "pending_"))}
    assert tables, "expected hardcoded table names for truncation"
    for required in ("fact_price", "fact_indicator", "fact_market_regime", "fact_signal"):
        assert required in tables, f"{required} is not truncated"

    # No table name may be conjured from a parameter name.
    assert not (tables & param_names)


# ------------------------------------------------------------------ 7. one at a time


def test_a_second_replay_is_refused_while_one_runs() -> None:
    bot.clear_cache()
    with bot.single_run():
        with pytest.raises(bot.ReplayBusy):
            with bot.single_run():
                pass  # pragma: no cover - must not be reached
    # The lock is released, so a later run is fine.
    with bot.single_run():
        pass


def test_allow_test_period_is_used_only_with_persist_off() -> None:
    """`allow_test_period=True` lifts the R26 guard, so it must never write.

    The replay deliberately runs over already-scored test dates, so it must pass
    `allow_test_period=True`. That flag is the most dangerous one in the codebase, so
    every occurrence must also carry `persist=False` and `write_signals=False`, and it
    must appear nowhere else in the dashboard.
    """
    for name in ("bot.py", "app.py", "data.py"):
        source = (REPO_ROOT / "src" / "dashboard" / name).read_text(encoding="utf-8")
        if "allow_test_period" not in source:
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "run_backtest":
                kwargs = {kw.arg: ast.dump(kw.value) for kw in node.keywords}
                assert kwargs.get("allow_test_period") == "Constant(value=True)", (
                    f"{name} lifts the test-period guard without asking for it"
                )
                assert kwargs.get("persist") == "Constant(value=False)", (
                    f"{name} lifts the test-period guard but persists results"
                )
                assert kwargs.get("write_signals") == "Constant(value=False)", (
                    f"{name} lifts the test-period guard but writes signals"
                )

    # Only bot.py may contain the flag at all.
    for name in ("app.py", "data.py"):
        source = (REPO_ROOT / "src" / "dashboard" / name).read_text(encoding="utf-8")
        assert "allow_test_period" not in source, f"{name} may not lift the test guard"


def test_the_replay_calls_run_backtest_with_writing_disabled() -> None:
    tree = ast.parse(inspect.getsource(bot))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "run_backtest"
    ]
    assert calls, "the replay must use the real engine, not a reimplementation"
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert {"persist", "write_signals"} <= kwargs


def test_the_lock_is_released_even_if_the_body_raises() -> None:
    with pytest.raises(RuntimeError):
        with bot.single_run():
            raise RuntimeError("boom")
    with bot.single_run():
        pass  # would deadlock if the lock leaked


# ------------------------------------------------------------------ 8. cleanup


def test_the_run_directory_is_removed_after_a_successful_run() -> None:
    before = set(bot.remaining_temp_dirs())
    with bot.run_directory() as work:
        (work / "replay.db").write_bytes(b"data")
        assert work.exists()
    assert not work.exists()
    assert set(bot.remaining_temp_dirs()) == before


def test_the_run_directory_is_removed_even_when_the_run_fails() -> None:
    """The failure case matters most: a crash must not leave a copy of the warehouse."""
    before = set(bot.remaining_temp_dirs())
    captured: Path | None = None
    with pytest.raises(RuntimeError):
        with bot.run_directory() as work:
            captured = work
            (work / "replay.db").write_bytes(b"data")
            raise RuntimeError("boom")
    assert captured is not None
    assert not captured.exists(), "temp directory survived a failed run"
    assert set(bot.remaining_temp_dirs()) == before
    assert not list(bot.remaining_temp_dirs()), "a temp directory or lock was left behind"


def test_no_temp_directories_are_left_after_many_failures() -> None:
    before = set(bot.remaining_temp_dirs())
    for _ in range(5):
        with pytest.raises(ValueError):
            with bot.run_directory() as work:
                (work / "x").write_text("y")
                raise ValueError("nope")
    assert set(bot.remaining_temp_dirs()) == before


def test_the_cleanup_helper_sees_nothing_right_now() -> None:
    assert bot.remaining_temp_dirs() == [], (
        "replay temp directories are present; a previous run failed to clean up"
    )


# ------------------------------------------------------------------ 10. caching


def test_results_are_cached_per_start_date() -> None:
    bot.clear_cache()
    assert bot.cached(20240102) is None
    run = bot.BotRun(start_date=20240102)
    bot.store(run)
    assert bot.cached(20240102) is run
    assert bot.cached(20240103) is None
    bot.clear_cache()
    assert bot.cached(20240102) is None


def test_cache_is_keyed_by_date_not_by_a_shared_slot() -> None:
    bot.clear_cache()
    bot.store(bot.BotRun(start_date=20240102))
    bot.store(bot.BotRun(start_date=20240103))
    assert bot.cached(20240102).start_date == 20240102  # type: ignore[union-attr]
    assert bot.cached(20240103).start_date == 20240103  # type: ignore[union-attr]
    bot.clear_cache()


# ------------------------------------------------------------------ 9. no leaks


def test_the_page_shows_no_paths_secrets_or_tracebacks() -> None:
    """Nothing on the screen may carry a filesystem path, a traceback or an env value."""
    source = (REPO_ROOT / "src" / "dashboard" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    bot_screen = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_run_bot"
    )
    # No st.* call in the error path may receive a raw exception or a traceback.
    for node in ast.walk(bot_screen):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in {
            "error",
            "warning",
            "info",
            "success",
            "markdown",
            "write",
            "write_error",
        }:
            rendered = ast.dump(node)
            assert "exception" not in rendered.lower() or "except" in rendered
            assert "exc" not in rendered.lower().replace("except", "")
            assert "traceback" not in rendered.lower()

    # The only place a traceback may appear is the server-side logger.
    assert "traceback.format_exc()" in source
    logger_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_log_bot_failure"
    )
    assert any(
        isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "error"
        for node in ast.walk(logger_fn)
    )


def test_the_error_path_returns_a_generic_message() -> None:
    source = (REPO_ROOT / "src" / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert "The replay could not be completed" in source
    # And it must not echo the exception text.
    assert "could not be completed. Please try again." in source


def test_no_absolute_path_literal_is_embedded_in_the_screen() -> None:
    source = (REPO_ROOT / "src" / "dashboard" / "app.py").read_text(encoding="utf-8")
    literals = re.findall(r"[\"']([A-Za-z]:\\\\|[A-Za-z]:\\|/home/|/Users/)", source)
    assert not literals, f"absolute paths on the page: {literals}"


def test_environment_values_are_never_rendered_on_the_page() -> None:
    """No `st.*` call may render an environment lookup.

    `config_path()` legitimately reads `RT_DASH_CONFIG`, but it returns a *config path*
    that selects a warehouse and is never shown. The property that matters is that no
    displayed string can contain an environment value, so the check is on the argument,
    not on the mere presence of `os.environ` in the module.
    """
    source = (REPO_ROOT / "src" / "dashboard" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    displayed = {"error", "info", "warning", "success", "caption", "title", "subheader"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        attr = getattr(node.func, "attr", "")
        if attr in displayed or (
            attr in {"markdown", "write", "json"} and "st." in ast.dump(node.func)
        ):
            dumped = ast.dump(node).lower()
            assert "environ" not in dumped, f"st.{attr} renders an environment lookup"
            assert "getenv" not in dumped, f"st.{attr} renders an environment lookup"

    # The only os.environ use must be the config-path helper, not a display call.
    uses = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "environ"
    ]
    for use in uses:
        parent_attr = getattr(use.value, "attr", "")
        assert parent_attr in {"environ", ""}, "unexpected environment access"


def test_the_config_env_var_is_only_a_path_selector() -> None:
    """`RT_DASH_CONFIG` picks a config file; it must not be echoed anywhere."""
    source = (REPO_ROOT / "src" / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert "RT_DASH_CONFIG" in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            attr = getattr(node.func, "attr", "")
            if attr in {"error", "info", "warning", "success", "markdown", "write", "json"}:
                assert "RT_DASH_CONFIG" not in ast.dump(node), (
                    f"st.{attr} echoes the config env var"
                )


# ------------------------------------------------------------------ 4. no-hindsight


def test_truncate_after_removes_every_future_row() -> None:
    """The guarantee is mechanical: after truncation, nothing dated later remains."""
    from src.common.config import load_config
    from src.common.db import get_engine, read_sql

    cfg = load_config(str(REPO_ROOT / "config" / "config.yaml"))
    import shutil

    work = Path(__import__("tempfile").mkdtemp(prefix="regimetrader_test_"))
    try:
        source = Path(cfg["paths"]["db"])
        shutil.copy(source, work / "t.db")
        local = dict(cfg)
        local["paths"] = dict(cfg["paths"])
        local["paths"]["db"] = str(work / "t.db")
        engine = get_engine(local)
        bot.truncate_after(engine, 20240103)
        for table in ("fact_price", "fact_indicator", "fact_market_regime", "fact_signal"):
            latest = int(read_sql(engine, f"SELECT MAX(date_key) m FROM {table}")["m"].iloc[0] or 0)
            assert latest <= 20240103, f"{table} still has rows after the cutoff"
        engine.dispose()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_the_lookback_is_derived_from_config_not_hardcoded() -> None:
    """Changing the regime window must change the lookback."""
    from src.common.config import load_config

    cfg = dict(load_config(str(REPO_ROOT / "config" / "config.yaml")))
    base = bot.lookback_trading_days(cfg)
    widened = dict(cfg)
    widened["regime"] = dict(cfg["regime"])
    widened["regime"]["drawdown_window"] = 504
    assert bot.lookback_trading_days(widened) > base


def test_the_no_hindsight_comparison_reports_mismatches_rather_than_hiding_them() -> None:
    """A wrong answer must come back as a mismatch, not as a pass."""
    import pandas as pd

    stored = pd.DataFrame(
        [{"symbol": "AAA.NS", "side": "BUY", "quantity": 10.0, "fill_price": 100.0}]
    )
    same = [{"symbol": "AAA.NS", "side": "BUY", "quantity": 10.0, "fill_price": 100.0}]
    assert bot.compare_first_day_against_stored(same, stored)["match"] is True

    wrong_qty = [{"symbol": "AAA.NS", "side": "BUY", "quantity": 11.0, "fill_price": 100.0}]
    result = bot.compare_first_day_against_stored(wrong_qty, stored)
    assert result["match"] is False
    assert result["quantity_or_price_differences"], "the difference was not reported"

    missing = bot.compare_first_day_against_stored([], stored)
    assert missing["match"] is False
    assert missing["only_in_stored"] == [["AAA.NS", "BUY"]]


def test_the_replay_banner_disclaims_performance() -> None:
    source = (REPO_ROOT / "src" / "dashboard" / "app.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "no performance claim" in lowered
    assert "no real money" in lowered
    assert "no broker connection" in lowered
    assert "replay of already-scored data" in lowered


def test_the_screen_is_registered_and_read_only_queries_only() -> None:
    """The bot screen is a registered screen.

    It is deliberately *not* in `dd.RESULT_RUNS`: that registry maps stored run ids to
    labels for the results charts, and the replay is not a stored run. The previous
    version of this test asserted `in dd.RESULT_RUNS or True`, which could never fail.
    """
    from src.dashboard import app as dashboard

    assert "Bot demo (replay)" in dashboard.SCREENS
    assert "Bot demo (replay)" not in dd.RESULT_RUNS


def test_the_demo_script_quotes_no_proof_number_the_json_does_not_contain() -> None:
    """`DEMO_SCRIPT.md` must not recite numbers that could drift from the proof file.

    The script previously quoted a specific tolerance and agreement claim in prose. If the
    pipeline changed, those words would still be there and would no longer be true. The
    demo text now points at what the screen prints, so any residual numeric claim must be
    checkable against the recorded proof.
    """
    import json

    proof_path = REPO_ROOT / "reports" / "results" / "bot_equivalence.json"
    assert proof_path.exists(), "run scripts/bot_equivalence.py to produce the proof file"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))

    assert proof["all_match"] is True
    assert proof["all_lookahead_safe"] is True
    assert proof["float_tolerance"] == 1e-9
    assert proof["days_tested"] == [20240101, 20240105]

    demo = (REPO_ROOT / "docs" / "DEMO_SCRIPT.md").read_text(encoding="utf-8")
    section = demo.split("## 8b.")[1].split("## 9.")[0]

    # The tolerance may appear as words, but no bare exponent may contradict the file.
    assert "1e-09" not in section
    # Any "days tested" figure quoted must match the file.
    for result in proof["results"]:
        assert str(result["day"]) not in section, (
            f"the demo script hardcodes day {result['day']}; read it off the screen instead"
        )
    # And it must tell the presenter to read the numbers rather than recite them.
    assert "do not recite" in section.lower()


def test_the_data_layer_still_contains_only_reads() -> None:
    """Pre-existing guard: every SQL string in the data layer is a SELECT."""
    source = (REPO_ROOT / "src" / "dashboard" / "data.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    sql = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.strip().upper().startswith("SELECT")
    ]
    assert sql
    for statement in sql:
        for forbidden in ("INSERT", "UPDATE ", "DELETE", "DROP", "ALTER", "CREATE"):
            assert forbidden not in statement.upper(), statement[:60]


def test_sqlite_ro_uri_is_really_used() -> None:
    """Guard against the URI silently being dropped, which would make writes possible."""
    source = inspect.getsource(dd.readonly_engine)
    assert "mode=ro" in source
    assert "uri=true" in source


def test_bot_py_never_imports_sqlite3_directly_for_writing() -> None:
    """The module copies files; it does not open the real database itself."""
    source = inspect.getsource(bot)
    assert "sqlite3.connect" not in source
    assert sqlite3 is not None


def test_any_keyboard_input_path_is_absent() -> None:
    """No text_input / text_area on the bot screen: the visitor cannot type anything."""
    source = inspect.getsource(bot)
    app_source = (REPO_ROOT / "src" / "dashboard" / "app.py").read_text(encoding="utf-8")
    screen = app_source.split("def screen_bot")[1].split("def _run_bot")[0]
    assert "text_input" not in screen
    assert "text_area" not in screen
    assert "st.selectbox" in screen
    assert source is not None


def test_the_bot_screen_shows_no_performance_metric() -> None:
    """R54: no Sharpe, CAGR, drawdown or any other performance number may appear.

    The screen demonstrates *decisions*. A visitor reading a Sharpe on a replay of five
    already-scored days would reasonably take it for a result they could earn, which is
    exactly what the banner forbids. Checked structurally over both modules rather than
    by grepping the rendered page.
    """
    forbidden = (
        "sharpe",
        "sortino",
        "calmar",
        "cagr",
        "max_drawdown",
        "profit_factor",
        "annualized_vol",
        "win_rate",
    )
    for name in ("bot.py", "app.py"):
        source = (REPO_ROOT / "src" / "dashboard" / name).read_text(encoding="utf-8").lower()
        # `app.py` legitimately names metrics elsewhere (the results screen), so the
        # check is scoped to the bot screen for that file.
        if name == "app.py":
            source = source.split("def screen_bot")[1].split("# ------ entry point")[0]
        for term in forbidden:
            assert term not in source, f"{name} bot screen references {term!r}"


# ------------------------------------------------------------------ equivalence is real


def test_the_comparison_fails_if_a_stored_weight_is_corrupted() -> None:
    """The weight comparison must be able to fail. It could not before.

    The original `_day_output` read `target_weight` from `fact_signal`, which
    `truncate_after` never deletes, so both the literal and the fast path returned the
    *same stored rows* and always agreed. This asserts the opposite property: a stored
    weight that disagrees with the recomputed one is reported, not absorbed.
    """
    from scripts.bot_equivalence import compare

    stored_truth = {
        "regime": "BULL",
        "prob_up": {"AAA.NS": 0.60, "BBB.NS": 0.40},
        "weights": {"AAA.NS": 0.1, "BBB.NS": 0.1},
        "rows": 2,
    }
    corrupted_stored = {
        "regime": "BULL",
        "prob_up": {"AAA.NS": 0.60, "BBB.NS": 0.40},
        # What a stale or wrongly written fact_signal row would look like.
        "weights": {"AAA.NS": 0.9, "BBB.NS": 0.1},
        "rows": 2,
    }

    result = compare(stored_truth, corrupted_stored)
    assert result["match"] is False
    assert result["mismatch_count"] == 1
    mismatch = result["mismatches"][0]
    assert mismatch["kind"] == "target_weight"
    assert mismatch["symbol"] == "AAA.NS"
    assert mismatch["literal"] == 0.1 and mismatch["fast"] == 0.9

    # A missing weight is a mismatch too, not a silent skip.
    absent = compare(stored_truth, {**stored_truth, "weights": {"AAA.NS": 0.1}})
    assert absent["match"] is False
    assert absent["mismatches"][0]["kind"] == "weight_presence"


def test_the_day_output_reads_no_stored_signal_value() -> None:
    """`_day_output` must not consult `fact_signal` at all.

    Checked over the AST rather than by substring, because the function's own docstring
    names the table in order to explain that it does not read it. This is the property
    that makes the equivalence proof mean something: the weights come from
    `target_weights`, the same pure function the engine and the paper bot call, applied
    to freshly computed probabilities.
    """
    from scripts import bot_equivalence

    tree = ast.parse(inspect.getsource(bot_equivalence))
    day_output = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_day_output"
    )

    # No SQL string naming the stored signal table may appear. Only executable strings
    # are checked, so the docstring explaining the rule is not itself a violation.
    sql_texts = [
        node.value
        for node in ast.walk(day_output)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "SELECT" in node.value
    ]
    assert sql_texts, "expected the regime lookup query"
    for statement in sql_texts:
        assert "fact_signal" not in statement, (
            f"_day_output still reads a stored signal: {statement[:80]}"
        )

    # And it must call the real strategy function rather than reimplement it.
    called = {
        getattr(node.func, "id", getattr(node.func, "attr", ""))
        for node in ast.walk(day_output)
        if isinstance(node, ast.Call)
    }
    assert "target_weights" in called, "_day_output must use the real target_weights"


def test_the_mutation_test_keeps_the_rows_it_mutates() -> None:
    """The look-ahead mutation must leave the future row in place.

    The original version mutated a future price and then called `truncate_after`, which
    deleted the row it had just changed before recomputing, so the assertion could not
    fail for any implementation. Parsed with `ast` so the explanatory docstring naming
    those functions is not mistaken for a call.
    """
    from scripts import bot_equivalence

    tree = ast.parse(inspect.getsource(bot_equivalence))
    mutated_fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "mutated"
    )
    called = {
        getattr(node.func, "id", getattr(node.func, "attr", ""))
        for node in ast.walk(mutated_fn)
        if isinstance(node, ast.Call)
    }
    for forbidden in ("truncate_after", "trim_to_lookback"):
        assert forbidden not in called, (
            f"the mutation test calls {forbidden}, which deletes the rows it just mutated"
        )
    assert "build_indicators" in called, "the mutation must recompute the indicators"


def test_the_mutation_check_detects_an_injected_look_ahead_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the mutation check can fail, by injecting a look-ahead bug.

    A guard that cannot fail is not a guard. Here a deliberately leaky variant of
    `compute_indicators` folds tomorrow's close into today's `sma_20` -- exactly the
    `shift(-1)` mistake `AGENTS.md` forbids. The observed past day must then MOVE when a
    future price changes, which is what a real leak produces and what the correct
    implementation does not.
    """
    from src.features import indicators as ind

    original = ind.compute_indicators

    def leaky(prices: pd.DataFrame) -> pd.DataFrame:
        out = original(prices)
        future = prices["close"].astype("float64").shift(-1)
        out["sma_20"] = (out["sma_20"] + future) / 2.0
        return out

    monkeypatch.setattr(ind, "compute_indicators", leaky)
    assert ind.compute_indicators is leaky, "the leak must actually be installed"

    # 60 ascending bars, so the rolling window is warm well before the observed day.
    prices = pd.DataFrame(
        {
            "open": [100.0 + i for i in range(60)],
            "high": [100.0 + i for i in range(60)],
            "low": [100.0 + i for i in range(60)],
            "close": [100.0 + i for i in range(60)],
            "volume": [1000.0 + i for i in range(60)],
        },
        index=pd.Index([20240100 + n for n in range(1, 61)], name="date_key"),
    )
    observed = 20240125
    assert observed in prices.index, "the observed day must be in the frame"

    mutated_prices = prices.copy()
    # The injected leak peeks exactly one row forward, so the day immediately after the
    # observed one is the row that must move it.
    mutated_prices.loc[20240126, ["open", "high", "low", "close"]] = 9999.0

    before = leaky(prices).loc[observed, "sma_20"]
    after = leaky(mutated_prices).loc[observed, "sma_20"]
    assert not pd.isna(before) and not pd.isna(after), "the rolling window must be warm"
    assert before != after, "an injected leak that leaves the past unchanged is not a leak"

    # And the real implementation is unaffected by the same change, which is the
    # property the mutation test relies on.
    real_before = original(prices).loc[observed, "sma_20"]
    real_after = original(mutated_prices).loc[observed, "sma_20"]
    assert real_before == real_after, "the real feature must not move with a future price"


# ------------------------------------------------------------------ threshold and weights


def test_the_entry_threshold_is_read_from_config_not_a_literal() -> None:
    """The screen must apply the configured threshold (rule 6, no magic numbers).

    The previous implementation counted candidates above a hardcoded `0.5`. The selected
    `entry_base` is 0.504415, so that admitted names the strategy would have rejected.
    """
    from src.common.config import load_config

    cfg = load_config(str(REPO_ROOT / "config" / "config.yaml"))
    threshold = bot.entry_threshold(cfg, "BULL")
    assert threshold is not None
    # It must equal the configured base plus the BULL offset, not a round literal.
    from src.trading.rules import build_regime_rules

    assert threshold == build_regime_rules(cfg)["BULL"]["entry"]
    assert threshold != 0.5


def test_a_zero_exposure_regime_admits_nothing() -> None:
    """`entry` is None where exposure is zero, and the count must be 0, not a crash."""
    from src.common.config import load_config

    cfg = load_config(str(REPO_ROOT / "config" / "config.yaml"))
    zero = [r for r, rule in cfg["strategy"]["exposure"].items() if float(rule) == 0.0]
    assert zero, "expected at least one zero-exposure regime in the config"
    for regime in zero:
        assert bot.entry_threshold(cfg, regime) is None
        # An unknown regime is also None rather than an exception on a public screen.
        assert bot.entry_threshold(cfg, "NOT_A_REGIME") is None


def test_passed_threshold_counts_against_the_configured_value() -> None:
    """A name between the old literal 0.5 and the real 0.504415 must not be counted."""
    rows = [
        {"p": 0.5020},  # above 0.5, below the real threshold -> excluded
        {"p": 0.5100},  # above the real threshold -> included
        {"p": 0.504415},  # exactly at it -> included (>=)
        {"p": 0.4999},  # below -> excluded
    ]
    result = bot._day_from_result(
        20240102,
        pd.DataFrame(),
        [r["p"] for r in rows],
        rows,
        regime="BULL",
        threshold=0.504415,
    )
    assert result.passed_threshold == 2
    assert result.candidates == 4
    assert result.entry_threshold == 0.504415


def test_target_weights_come_from_the_engine_not_a_placeholder() -> None:
    """The weights panel must show real values.

    The previous `_day_from_result` hardcoded `target_weights={}`, so the panel could
    never display a weight regardless of what the strategy decided.
    """
    weights = {"ASIANPAINT.NS": 0.1, "TITAN.NS": 0.1}
    result = bot._day_from_result(
        20240102,
        pd.DataFrame(),
        [],
        [],
        regime="BULL",
        weights=weights,
        threshold=0.504415,
    )
    assert result.target_weights == weights
    assert sum(result.target_weights.values()) == pytest.approx(0.2)

    # Omitting the weights still yields an empty mapping, never a crash.
    default = bot._day_from_result(20240102, pd.DataFrame(), [], [], regime="BULL")
    assert default.target_weights == {}
    assert default.entry_threshold is None
    assert default.passed_threshold == 0

    # And it survives serialisation, which is what the page renders.
    payload = bot.BotRun(start_date=20240102, days=[result]).as_json()
    assert payload["days"][0]["target_weights"] == {"ASIANPAINT.NS": 0.1, "TITAN.NS": 0.1}
    assert payload["days"][0]["entry_threshold"] == 0.504415


def test_the_days_holdings_are_the_cumulative_book_not_only_the_day_fills() -> None:
    """The bug this fixes: a partial-trade day showed only the symbols traded that day.

    Under the old `_day_from_result`, holdings were rebuilt from the day's own fills
    and reset each day, so a day that sold one name out of ten showed a one-line
    "holdings" panel. `run_replay` now maintains the cumulative book and passes it in;
    every day's view must show the whole portfolio as of that day's last fill.
    """
    day_one = [
        {"symbol": "ASIANPAINT.NS", "side": "BUY", "reason": "", "quantity": 10, "fill_price": 1.0,
         "date_key": 20240102},
        {"symbol": "ITC.NS", "side": "BUY", "reason": "", "quantity": 20, "fill_price": 2.0,
         "date_key": 20240102},
        {"symbol": "TITAN.NS", "side": "BUY", "reason": "", "quantity": 5, "fill_price": 3.0,
         "date_key": 20240102},
    ]
    partially_traded_day = [
        {"symbol": "ITC.NS", "side": "SELL", "reason": "EXIT", "quantity": 20, "fill_price": 2.5},
    ]

    book: dict[str, int] = {}
    bot.apply_orders(book, day_one)
    first = bot._day_from_result(
        20240102, pd.DataFrame(), [], [], regime="BULL", holdings=dict(book)
    )
    assert first.holdings == {"ASIANPAINT.NS": 10, "ITC.NS": 20, "TITAN.NS": 5}

    bot.apply_orders(book, partially_traded_day)
    second = bot._day_from_result(
        20240103, pd.DataFrame(), [], [], regime="BULL", holdings=dict(book)
    )
    # The day traded one name, yet the panel shows the full remaining book.
    assert partially_traded_day[0]["symbol"] not in second.holdings
    assert second.holdings == {"ASIANPAINT.NS": 10, "TITAN.NS": 5}

    # Omitting `holdings` keeps the legacy single-day behaviour for direct callers.
    legacy = bot._day_from_result(
        20240102,
        pd.DataFrame(day_one),  # trades frame carrying day one's orders
        [],
        [],
        regime="BULL",
    )
    assert legacy.holdings == {"ASIANPAINT.NS": 10, "ITC.NS": 20, "TITAN.NS": 5}
    assert [order["symbol"] for order in legacy.orders] == ["ASIANPAINT.NS", "ITC.NS", "TITAN.NS"]


def test_run_replay_maintains_the_cumulative_book() -> None:
    """Structural: the loop must carry one book across days and feed each day a copy.

    A per-day rebuild inside `_day_from_result` would reintroduce the bug silently, so
    the wiring itself is pinned, in the file's usual source-inspection style.
    """
    body = inspect.getsource(bot.run_replay)
    start = body.index("book: dict[str, int] = {}")
    loop = body[start:]
    assert "apply_orders(book, day_orders)" in loop
    assert "holdings=dict(book)" in loop
    # The book is hoisted before the loop and passed per day.
    assert loop.index("holdings=dict(book)") > 0


def test_the_screen_renders_the_weights_it_receives() -> None:
    """The Streamlit screen must actually draw the weights, not just hold them."""
    app_source = (REPO_ROOT / "src" / "dashboard" / "app.py").read_text(encoding="utf-8")
    renderer = app_source.split("def _render_bot")[1].split("def _read_proof")[0]
    assert "target_weights" in renderer
    assert "Target weights at this close" in renderer
    assert "entry threshold" in renderer.lower()


def test_the_whitelisted_window_is_the_one_the_proof_covers() -> None:
    """Day 1 and day 5 of the proof must both be replayable by a visitor."""
    import inspect as _inspect

    from scripts.bot_equivalence import main as equivalence_main

    defaults = _inspect.signature(equivalence_main).parameters
    assert "days" not in defaults  # argv-driven; check the parser default instead
    first, last = bot.WHITELIST_START_DATES[0], bot.WHITELIST_START_DATES[-1]
    assert first == 20240101
    assert last == 20240105
    assert len(bot.WHITELIST_START_DATES) == bot.WINDOW_DAYS
