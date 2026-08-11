import time

import individual_api.history_prefetch as hp
from individual_api import pr31_runtime


def _clear_pending() -> None:
    with hp._LOCK:
        for future in hp._FUTURES.values():
            future.cancel()
        hp._FUTURES.clear()


def test_prefetch_reuses_same_fetch_result(monkeypatch) -> None:
    _clear_pending()
    calls = []

    def fake_fetch(venue, race_date, day_no, race_no, rider_names):
        calls.append((venue, race_date, day_no, race_no, tuple(rider_names)))
        time.sleep(0.02)
        return {
            "status": "PREVIOUS_DAY_NOT_FOUND",
            "source": "KDreams",
            "resolved_day_no": 3,
            "riders": {},
        }

    monkeypatch.setattr(hp, "_BASE_FETCH", fake_fetch)
    monkeypatch.setattr(pr31_runtime, "fetch_previous_day", hp._fetch_with_prefetch)
    assert hp.prefetch_previous_day("小田原", "2026-08-03", 3, 4, ["A"]) is True
    result = hp._fetch_with_prefetch("小田原", "2026-08-03", 3, 4, ["A"])
    assert result["status"] == "PREVIOUS_DAY_NOT_FOUND"
    assert len(calls) == 1
    _clear_pending()


def test_first_day_never_prefetches(monkeypatch) -> None:
    _clear_pending()
    monkeypatch.setattr(pr31_runtime, "fetch_previous_day", hp._fetch_with_prefetch)
    assert hp.prefetch_previous_day("小田原", "2026-08-01", 1, 4, ["A"]) is False
    with hp._LOCK:
        assert hp._FUTURES == {}
