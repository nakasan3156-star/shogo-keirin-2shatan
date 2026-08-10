"""Production guards for fixed KEIRIN.JP three-PDF input.

PR31 prediction rules, thresholds, probability models, EV rules and betting
conditions are never changed here.  The current race result page is never
requested.  Same-meeting history uses a current KDreams odds/race-info page
only to identify the previous start, then reads completed previous-day results.
"""
from __future__ import annotations

import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout, as_completed
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from . import keirin_line_runtime_fix as line_runtime
from . import keirin_pdf_adapter as pdf_adapter
from . import keirin_real_pdf_adapter as real_adapter
from . import pr31_runtime
from . import previous_day_kdreams as previous_day

_ORIGINAL_EXTRACT_TEXT = pdf_adapter._extract_text
_ORIGINAL_PARSE_LINES = line_runtime.parse_lines_from_pdfs
_ORIGINAL_PARSE_BASIC = real_adapter.parse_basic_real
_ORIGINAL_PARSE_HS = real_adapter.parse_hs_real
_INSTALLED = False
_HISTORY_TIMEOUT_SECONDS = 8.0
_HISTORY_CIRCUIT_SECONDS = 120.0
_HISTORY_CIRCUIT_UNTIL = 0.0
_HISTORY_CIRCUIT_LOCK = threading.Lock()
_HISTORY_RESOLVER = "fail_open_v3"


@lru_cache(maxsize=24)
def _cached_extract_text(path_value: str, label: str) -> str:
    return _ORIGINAL_EXTRACT_TEXT(Path(path_value), label)


def _extract_text_cached(path: str | Path, label: str) -> str:
    return _cached_extract_text(str(Path(path).resolve()), str(label))


def _role_counts(text: str) -> tuple[int, int]:
    basic = 0
    hs = 0
    try:
        rows = real_adapter._profile_rows(text)
    except Exception:
        rows = []
    for _bike, _name, _prefecture, _style, values in rows:
        if real_adapter._BASIC_VALUES.match(values):
            basic += 1
        if real_adapter._HS_VALUES.match(values):
            hs += 1
    return basic, hs


def _normalized_header(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))


def _parse_basic_fixed(text: str, path: str | Path | None = None) -> list[dict[str, Any]]:
    name = Path(path).name if path is not None else ""
    if "オッズ" in name:
        raise real_adapter.PdfInputError("FIXED_ROLE_NOT_BASIC", "2車単オッズPDFです")
    basic_count, hs_count = _role_counts(text)
    header = _normalized_header(text)
    if hs_count >= 3 and hs_count > basic_count:
        raise real_adapter.PdfInputError("FIXED_ROLE_NOT_BASIC", "H/S着度数PDFです")
    if basic_count >= 3 or "競走得点" in header:
        return _ORIGINAL_PARSE_BASIC(text, path)
    return _ORIGINAL_PARSE_BASIC(text, path)


def _parse_hs_fixed(
    text: str, bikes: list[int], path: str | Path | None = None
) -> dict[int, dict[str, int | float]]:
    name = Path(path).name if path is not None else ""
    if "オッズ" in name:
        raise real_adapter.PdfInputError("FIXED_ROLE_NOT_HS", "2車単オッズPDFです")
    basic_count, hs_count = _role_counts(text)
    header = _normalized_header(text)
    if basic_count >= 3 and basic_count > hs_count:
        raise real_adapter.PdfInputError("FIXED_ROLE_NOT_HS", "基本情報PDFです")
    if hs_count >= 3 or ("着外" in header and "1着" in header and "2着" in header and "3着" in header):
        return _ORIGINAL_PARSE_HS(text, bikes, path)
    return _ORIGINAL_PARSE_HS(text, bikes, path)


def _fast_consensus_lines(
    paths: Iterable[str | Path], bikes: list[int]
) -> tuple[list[list[int]], str, str]:
    """Return early only when two different official PDFs agree by coordinates."""
    unique_paths: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path not in unique_paths:
            unique_paths.append(path)

    confirmations: dict[tuple[tuple[int, ...], ...], list[Any]] = {}
    for path in unique_paths:
        candidates = line_runtime._coordinate_candidates(path, bikes)
        if not candidates:
            continue
        top = candidates[0]
        matched = confirmations.setdefault(top.lines, [])
        matched.append(top)
        if len({candidate.source for candidate in matched}) >= 2:
            parsed = [list(line) for line in top.lines]
            if line_runtime._valid(parsed, bikes):
                return parsed, top.source, f"{top.method}:two_pdf_consensus"

    return _ORIGINAL_PARSE_LINES(unique_paths, bikes)


def _current_info_url(
    code: str,
    slug: str,
    current: datetime,
    day_no: int,
    race_no: int,
    page_type: str = "odds",
) -> tuple[str, datetime]:
    """Build a non-result URL for the race being predicted."""
    start = current - timedelta(days=day_no - 1)
    rid = f"{code}{start.strftime('%Y%m%d')}{day_no:02d}{race_no:04d}"
    return f"https://keirin.kdreams.jp/{slug}/racedetail/{rid}/?pageType={page_type}", start


def _find_current_info(
    code: str,
    slug: str,
    current: datetime,
    day_no: int,
    race_no: int,
    rider_names: list[str],
) -> tuple[int, str, str, datetime | None]:
    """Resolve current day from non-result KDreams pages and retain the HTML."""
    candidates = [day_no] if day_no >= 2 else list(range(1, 7))

    def fetch_candidate(candidate: int) -> tuple[int, str, str, datetime]:
        url, start = _current_info_url(code, slug, current, candidate, race_no, "odds")
        return candidate, url, previous_day._fetch_html(url), start

    if len(candidates) == 1:
        candidate, url, html, start = fetch_candidate(candidates[0])
        if previous_day._current_page_matches(html, current, rider_names):
            return candidate, url, html, start
        return 0, "", "", None

    pool = ThreadPoolExecutor(max_workers=len(candidates), thread_name_prefix="pr31-day-resolve")
    futures = [pool.submit(fetch_candidate, candidate) for candidate in candidates]
    try:
        for future in as_completed(futures):
            try:
                candidate, url, html, start = future.result()
            except Exception:
                continue
            if previous_day._current_page_matches(html, current, rider_names):
                for pending in futures:
                    if pending is not future:
                        pending.cancel()
                return candidate, url, html, start
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return 0, "", "", None


def _fetch_previous_day_safe_fast(
    venue: str | None,
    race_date: str | None,
    day_no: int,
    race_no: int,
    rider_names: list[str],
) -> dict[str, Any]:
    started = time.monotonic()
    diagnostics: dict[str, Any] = {
        "resolver": _HISTORY_RESOLVER,
        "current_result_page_used": False,
    }

    if day_no == 1:
        return {
            "status": "FIRST_DAY_SKIPPED",
            "source": "KDreams",
            "resolved_day_no": 1,
            "riders": {},
            "diagnostics": diagnostics,
        }
    if not venue or venue not in previous_day.VENUES or not race_date or race_no <= 0:
        return {
            "status": "IDENTITY_UNAVAILABLE",
            "source": "KDreams",
            "resolved_day_no": day_no if day_no >= 1 else 3,
            "riders": {},
            "diagnostics": diagnostics,
        }
    try:
        current = datetime.strptime(race_date, "%Y-%m-%d")
    except ValueError:
        return {
            "status": "DATE_INVALID",
            "source": "KDreams",
            "resolved_day_no": day_no if day_no >= 1 else 3,
            "riders": {},
            "diagnostics": diagnostics,
        }

    code, slug = previous_day.VENUES[venue]
    resolved_day, current_url, current_html, start = _find_current_info(
        code, slug, current, day_no, race_no, rider_names
    )
    diagnostics["resolve_ms"] = round((time.monotonic() - started) * 1000)
    if not resolved_day or start is None:
        diagnostics["stage"] = "current_info_not_found"
        diagnostics["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        return {
            "status": "PREVIOUS_DAY_NOT_FOUND",
            "source": "KDreams",
            "resolved_day_no": day_no if day_no >= 2 else 3,
            "previous_date": (current - timedelta(days=1)).strftime("%Y-%m-%d"),
            "riders": {},
            "diagnostics": diagnostics,
        }
    if resolved_day == 1:
        diagnostics["stage"] = "first_day"
        diagnostics["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        return {
            "status": "FIRST_DAY_SKIPPED",
            "source": "KDreams",
            "resolved_day_no": 1,
            "current_info_url": current_url,
            "riders": {},
            "diagnostics": diagnostics,
        }

    summary = previous_day._previous_summary(current_html, rider_names)
    if not summary:
        yoso_url, _ = _current_info_url(code, slug, current, resolved_day, race_no, "yoso")
        yoso_html = previous_day._fetch_html(yoso_url)
        summary = previous_day._previous_summary(yoso_html, rider_names)
        if summary:
            current_url = yoso_url
    diagnostics["summary_count"] = len(summary)
    if not summary:
        diagnostics["stage"] = "previous_summary_not_found"
        diagnostics["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        return {
            "status": "PREVIOUS_DAY_NOT_FOUND",
            "source": "KDreams",
            "resolved_day_no": resolved_day,
            "current_info_url": current_url,
            "previous_date": (current - timedelta(days=1)).strftime("%Y-%m-%d"),
            "riders": {},
            "diagnostics": diagnostics,
        }

    previous_day_no = resolved_day - 1
    urls: dict[int, str] = {}
    for item in summary.values():
        prev_race = int(item["previous_race_no"])
        rid = f"{code}{start.strftime('%Y%m%d')}{previous_day_no:02d}{prev_race:04d}"
        urls[prev_race] = f"https://keirin.kdreams.jp/{slug}/racedetail/{rid}/?pageType=result"

    detail_by_name: dict[str, dict[str, Any]] = {}
    if urls:
        with ThreadPoolExecutor(max_workers=min(9, len(urls)), thread_name_prefix="pr31-prev-results") as pool:
            futures = {pool.submit(previous_day._fetch_html, url): (race, url) for race, url in urls.items()}
            for future in as_completed(futures):
                race, url = futures[future]
                try:
                    html = future.result()
                except Exception:
                    html = ""
                target_names = [
                    name for name, item in summary.items()
                    if int(item["previous_race_no"]) == race
                ]
                parsed = previous_day._result_detail(html, target_names)
                line_map = previous_day._lineup_map(html)
                winner = previous_day._winner_car(html)
                winner_line = line_map.get(winner) if winner is not None else None
                for name, item in parsed.items():
                    item["previous_detail_url"] = url
                    item["previous_line_no"] = line_map.get(item.get("car_no"))
                    item["previous_winner_car"] = winner
                    item["previous_winner_line_no"] = winner_line
                    detail_by_name[name] = item

    riders: dict[str, dict[str, Any]] = {}
    for name, base in summary.items():
        detail = detail_by_name.get(name, {})
        item = {**base, **detail}
        item["comment"] = str(detail.get("comment") or base.get("short_review") or "")
        item.update(previous_day._labels(item))
        riders[name] = item

    diagnostics["detail_race_count"] = len(urls)
    diagnostics["detail_rider_count"] = len(detail_by_name)
    diagnostics["stage"] = "ok" if riders else "no_riders"
    diagnostics["elapsed_ms"] = round((time.monotonic() - started) * 1000)
    return {
        "status": "OK" if riders else "PREVIOUS_DAY_NOT_FOUND",
        "source": "KDreams",
        "resolved_day_no": resolved_day,
        "current_info_url": current_url,
        "previous_date": (current - timedelta(days=1)).strftime("%Y-%m-%d"),
        "riders": riders,
        "diagnostics": diagnostics,
    }


def _history_fallback(day_no: int, stage: str, error_type: str | None = None) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "resolver": _HISTORY_RESOLVER,
        "current_result_page_used": False,
        "stage": stage,
        "fallback": "continue_pr31_without_previous_day",
    }
    if error_type:
        diagnostics["error_type"] = error_type
    return {
        "status": "PREVIOUS_DAY_NOT_FOUND",
        "source": "KDreams",
        "resolved_day_no": day_no if day_no >= 1 else 3,
        "riders": {},
        "diagnostics": diagnostics,
    }


def _history_circuit_open() -> bool:
    with _HISTORY_CIRCUIT_LOCK:
        return time.monotonic() < _HISTORY_CIRCUIT_UNTIL


def _open_history_circuit() -> None:
    global _HISTORY_CIRCUIT_UNTIL
    with _HISTORY_CIRCUIT_LOCK:
        _HISTORY_CIRCUIT_UNTIL = max(
            _HISTORY_CIRCUIT_UNTIL,
            time.monotonic() + _HISTORY_CIRCUIT_SECONDS,
        )


def _close_history_circuit() -> None:
    global _HISTORY_CIRCUIT_UNTIL
    with _HISTORY_CIRCUIT_LOCK:
        _HISTORY_CIRCUIT_UNTIL = 0.0


def _bounded_previous_day(
    venue: str | None,
    race_date: str | None,
    day_no: int,
    race_no: int,
    rider_names: list[str],
) -> dict[str, Any]:
    """Fail open: optional KDreams I/O can never stop or expose a timeout as prediction failure."""
    if day_no == 1:
        return _fetch_previous_day_safe_fast(venue, race_date, day_no, race_no, rider_names)
    if _history_circuit_open():
        return _history_fallback(day_no, "circuit_open")

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pr31-history-bound")
    future = executor.submit(
        _fetch_previous_day_safe_fast,
        venue,
        race_date,
        day_no,
        race_no,
        rider_names,
    )
    try:
        result = future.result(timeout=_HISTORY_TIMEOUT_SECONDS)
        if result.get("status") == "OK":
            _close_history_circuit()
        return result
    except FutureTimeout:
        future.cancel()
        _open_history_circuit()
        return _history_fallback(day_no, "global_timeout_fallback")
    except Exception as exc:
        _open_history_circuit()
        return _history_fallback(day_no, "exception_fallback", type(exc).__name__)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def install_production_runtime_fix() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    pdf_adapter._extract_text = _extract_text_cached
    real_adapter._extract_text = _extract_text_cached
    line_runtime._extract_text = _extract_text_cached
    pr31_runtime._extract_text = _extract_text_cached

    real_adapter.parse_basic_real = _parse_basic_fixed
    real_adapter.parse_hs_real = _parse_hs_fixed
    line_runtime.parse_lines_from_pdfs = _fast_consensus_lines

    pr31_runtime.fetch_previous_day = _bounded_previous_day
    _INSTALLED = True
