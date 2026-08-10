import time
from datetime import datetime

import individual_api.production_runtime_fix as runtime
from individual_api.production_runtime_fix import (
    _current_info_url,
    _fetch_previous_day_safe_fast,
)


def test_current_race_history_lookup_uses_non_result_page_only():
    url, start = _current_info_url(
        "36", "odawara", datetime(2026, 8, 3), 3, 4, "odds"
    )
    assert start.strftime("%Y-%m-%d") == "2026-08-01"
    assert "pageType=odds" in url
    assert "pageType=result" not in url
    assert "pageType=showResult" not in url


def test_first_day_never_fetches_history_or_current_result_page():
    result = _fetch_previous_day_safe_fast(
        "小田原", "2026-08-01", 1, 4, ["テスト選手"]
    )
    assert result["status"] == "FIRST_DAY_SKIPPED"
    assert result["resolved_day_no"] == 1
    assert result["diagnostics"]["current_result_page_used"] is False
    assert result["diagnostics"]["resolver"] == "fail_open_v3"


def test_global_history_timeout_fails_open_without_timeout_status(monkeypatch):
    monkeypatch.setattr(runtime, "_HISTORY_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(runtime, "_HISTORY_CIRCUIT_SECONDS", 60.0)
    monkeypatch.setattr(runtime, "_HISTORY_CIRCUIT_UNTIL", 0.0)

    def slow_history(*_args, **_kwargs):
        time.sleep(0.5)
        return {"status": "OK", "riders": {"x": {}}}

    monkeypatch.setattr(runtime, "_fetch_previous_day_safe_fast", slow_history)
    started = time.monotonic()
    result = runtime._bounded_previous_day(
        "小田原", "2026-08-03", 3, 4, ["テスト選手"]
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.25
    assert result["status"] == "PREVIOUS_DAY_NOT_FOUND"
    assert result["diagnostics"]["stage"] == "global_timeout_fallback"
    assert result["diagnostics"]["fallback"] == "continue_pr31_without_previous_day"
    assert "TIMEOUT" not in result["status"]


def test_open_history_circuit_skips_repeated_network_attempt(monkeypatch):
    monkeypatch.setattr(runtime, "_HISTORY_CIRCUIT_UNTIL", time.monotonic() + 60.0)
    called = False

    def should_not_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("history fetch should be skipped while circuit is open")

    monkeypatch.setattr(runtime, "_fetch_previous_day_safe_fast", should_not_run)
    result = runtime._bounded_previous_day(
        "小田原", "2026-08-03", 3, 4, ["テスト選手"]
    )

    assert called is False
    assert result["status"] == "PREVIOUS_DAY_NOT_FOUND"
    assert result["diagnostics"]["stage"] == "circuit_open"
