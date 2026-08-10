"""Production performance guards for fixed KEIRIN.JP three-PDF input.

No PR31 prediction rule, threshold, probability model, EV rule, or betting
condition is changed. Production accepts the same three KEIRIN.JP PDF roles:
basic rider info, H/S finish counts, and exacta odds.
"""
from __future__ import annotations

import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout, as_completed
from datetime import datetime
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
_ORIGINAL_FETCH_PREVIOUS_DAY = pr31_runtime.fetch_previous_day
_ORIGINAL_PARSE_BASIC = real_adapter.parse_basic_real
_ORIGINAL_PARSE_HS = real_adapter.parse_hs_real
_INSTALLED = False


@lru_cache(maxsize=24)
def _cached_extract_text(path_value: str, label: str) -> str:
    return _ORIGINAL_EXTRACT_TEXT(Path(path_value), label)


def _extract_text_cached(path: str | Path, label: str) -> str:
    return _cached_extract_text(str(Path(path).resolve()), str(label))


def _role_counts(text: str) -> tuple[int, int]:
    """Classify KEIRIN.JP race-info PDFs from already extracted text only."""
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


def _resolve_day_parallel(
    venue: str | None,
    race_date: str | None,
    race_no: int,
    rider_names: list[str],
) -> int:
    """Resolve 最終日/unknown day number without six sequential KDreams waits."""
    if not venue or venue not in previous_day.VENUES or not race_date or race_no <= 0:
        return 0
    try:
        current = datetime.strptime(race_date, "%Y-%m-%d")
    except ValueError:
        return 0

    code, slug = previous_day.VENUES[venue]

    def fetch_candidate(candidate: int) -> tuple[int, str, str]:
        url, _start = previous_day._current_race_url(code, slug, current, candidate, race_no)
        return candidate, url, previous_day._fetch_html(url)

    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="pr31-day-resolve") as pool:
        futures = [pool.submit(fetch_candidate, candidate) for candidate in range(1, 7)]
        for future in as_completed(futures):
            try:
                candidate, _url, html = future.result()
            except Exception:
                continue
            if previous_day._current_page_matches(html, current, rider_names):
                for pending in futures:
                    if pending is not future:
                        pending.cancel()
                return candidate
    return 0


def _bounded_previous_day(
    venue: str | None,
    race_date: str | None,
    day_no: int,
    race_no: int,
    rider_names: list[str],
) -> dict[str, Any]:
    """Optional KDreams enrichment must never block the main PR31 prediction."""
    if day_no == 0:
        resolved = _resolve_day_parallel(venue, race_date, race_no, rider_names)
        if resolved >= 1:
            day_no = resolved

    if day_no == 1:
        return {
            "status": "FIRST_DAY_SKIPPED",
            "source": "KDreams",
            "resolved_day_no": 1,
            "riders": {},
        }

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pr31-prevday")
    future = executor.submit(
        _ORIGINAL_FETCH_PREVIOUS_DAY,
        venue,
        race_date,
        day_no,
        race_no,
        rider_names,
    )
    try:
        return future.result(timeout=8.0)
    except FutureTimeout:
        future.cancel()
        resolved = day_no if day_no >= 1 else 3
        return {
            "status": "PREVIOUS_DAY_TIMEOUT",
            "source": "KDreams",
            "resolved_day_no": resolved,
            "riders": {},
        }
    except Exception:
        resolved = day_no if day_no >= 1 else 3
        return {
            "status": "PREVIOUS_DAY_NOT_FOUND",
            "source": "KDreams",
            "resolved_day_no": resolved,
            "riders": {},
        }
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
