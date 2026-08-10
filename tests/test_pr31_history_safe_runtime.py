from datetime import datetime

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
    assert result["diagnostics"]["resolver"] == "safe_odds_parallel_v2"
