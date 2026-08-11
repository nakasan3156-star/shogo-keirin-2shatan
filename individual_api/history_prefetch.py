"""Overlap optional previous-day KDreams I/O with fixed PDF parsing.

This changes only execution timing. The exact same production previous-day
fetch function is called with the same arguments, and PR31 consumes its result
unchanged.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable

from . import pr31_runtime

_BASE_FETCH: Callable[..., dict[str, Any]] | None = None
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pr31-history-prefetch")
_LOCK = threading.Lock()
_FUTURES: dict[tuple[Any, ...], Future] = {}
_INSTALLED = False
_MAX_PENDING = 16
_WAIT_SECONDS = 9.0


def _key(
    venue: str | None,
    race_date: str | None,
    day_no: int,
    race_no: int,
    rider_names: list[str],
) -> tuple[Any, ...]:
    return venue, race_date, int(day_no), int(race_no), tuple(rider_names)


def prefetch_previous_day(
    venue: str | None,
    race_date: str | None,
    day_no: int,
    race_no: int,
    rider_names: list[str],
) -> bool:
    """Start the already-installed production fetch without waiting for it."""
    if _BASE_FETCH is None or day_no == 1:
        return False
    key = _key(venue, race_date, day_no, race_no, rider_names)
    with _LOCK:
        existing = _FUTURES.get(key)
        if existing is not None and not existing.cancelled():
            return True
        while len(_FUTURES) >= _MAX_PENDING:
            oldest = next(iter(_FUTURES))
            old_future = _FUTURES.pop(oldest)
            if not old_future.done():
                old_future.cancel()
        _FUTURES[key] = _EXECUTOR.submit(
            _BASE_FETCH,
            venue,
            race_date,
            day_no,
            race_no,
            list(rider_names),
        )
    return True


def _fetch_with_prefetch(
    venue: str | None,
    race_date: str | None,
    day_no: int,
    race_no: int,
    rider_names: list[str],
) -> dict[str, Any]:
    assert _BASE_FETCH is not None
    key = _key(venue, race_date, day_no, race_no, rider_names)
    with _LOCK:
        future = _FUTURES.pop(key, None)
    if future is None:
        return _BASE_FETCH(venue, race_date, day_no, race_no, rider_names)
    try:
        return future.result(timeout=_WAIT_SECONDS)
    except FutureTimeout:
        future.cancel()
        return _BASE_FETCH(venue, race_date, day_no, race_no, rider_names)
    except Exception:
        return _BASE_FETCH(venue, race_date, day_no, race_no, rider_names)


def install_history_prefetch() -> None:
    global _BASE_FETCH, _INSTALLED
    if _INSTALLED:
        return
    _BASE_FETCH = pr31_runtime.fetch_previous_day
    pr31_runtime.fetch_previous_day = _fetch_with_prefetch
    _INSTALLED = True
