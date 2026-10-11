"""Alerts for the daily job (`spec 08` section 6).

Every alert goes to `logs/alerts.log` and stderr. Email is sent only when
`SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD` and `ALERT_TO` are all set in the
environment; nothing is ever sent silently, and a missing SMTP configuration is a
configuration fact, not an error.
"""

from __future__ import annotations

import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Mapping

#: Levels, most severe first.
LEVELS = ("ERROR", "WARNING", "INFO")


def alert(
    cfg: Mapping[str, Any], level: str, message: str, *, log_path: Path | None = None
) -> None:
    """Record one alert to `logs/alerts.log` and stderr, and email if configured.

    Args:
        cfg: Loaded config; `paths.logs` gives the directory.
        level: One of `LEVELS`.
        message: Human-readable text.
        log_path: Override for the log file, used by tests.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown alert level {level!r}; expected one of {LEVELS}")
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).isoformat()
    line = f"{stamp} [{level}] {message}"
    print(line, file=sys.stderr)

    directory = Path(log_path.parent) if log_path is not None else Path(cfg["paths"]["logs"])
    directory.mkdir(parents=True, exist_ok=True)
    target = log_path if log_path is not None else directory / "alerts.log"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")

    if level == "INFO":
        return
    _email(cfg, level, message)


def _email(cfg: Mapping[str, Any], level: str, message: str) -> bool:
    """Send an email when SMTP is fully configured. Returns whether it was sent."""
    required = ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "ALERT_TO")
    if not all(os.environ.get(name) for name in required):
        return False

    mail = EmailMessage()
    mail["Subject"] = f"[RegimeTrader] {level}"
    mail["From"] = os.environ["SMTP_USER"]
    mail["To"] = os.environ["ALERT_TO"]
    mail.set_content(f"{level}\n\n{message}\n")
    try:
        with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", "25"))) as smtp:
            smtp.starttls()
            smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            smtp.send_message(mail)
    except (OSError, smtplib.SMTPException) as exc:  # pragma: no cover - environment dependent
        print(f"alert email failed: {exc}", file=sys.stderr)
        return False
    return True


def stale_days(cfg: Mapping[str, Any], last_data_key: int | None, today_key: int) -> int:
    """Calendar days between the latest market bar and `today_key`.

    Args:
        cfg: Loaded config.
        last_data_key: Latest `date_key` in `fact_price`, or None.
        today_key: Today's `date_key`.

    Returns:
        The gap in calendar days, or 0 when there is no data yet.
    """
    if not last_data_key:
        return 0
    from datetime import datetime

    start = datetime.strptime(str(int(last_data_key)), "%Y%m%d")
    end = datetime.strptime(str(int(today_key)), "%Y%m%d")
    return (end - start).days


def is_stale(cfg: Mapping[str, Any], last_data_key: int | None, today_key: int) -> bool:
    """True when the market data is older than `data.stale_after_days`."""
    return stale_days(cfg, last_data_key, today_key) > int(cfg["data"]["stale_after_days"])
