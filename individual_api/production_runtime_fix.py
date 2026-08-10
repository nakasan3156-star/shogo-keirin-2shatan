"""Production-only performance guards for the PR31 runtime.

No prediction rule, threshold, probability model, or betting condition is changed.
This module only removes repeated PDF text extraction, short-circuits line parsing
when two official PDFs independently agree, and bounds best-effort KDreams I/O.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from . import keirin_line_runtime_fix as line_runtime
from . import keirin_pdf_adapter as pdf_adapter
from . import keirin_real_pdf_adapter as real_adapter
from . import pr31_runtime

_ORIGINAL_EXTRACT_TEXT = pdf_adapter._extract_text
_ORIGINAL_PARSE_LINES = line_runtime.parse_lines_from_pdfs
_ORIGINAL_FETCH_PREVIOUS_DAY = pr31_runtime.fetch_previous_day
_INSTALLED = False


@lru_cache(maxsize=24)
def _cached_extract_text(path_value: str, label: str) -> str:
    return _ORIGINAL_EXTRACT_TEXT(Path(path_value), label)


def _extract_text_cached(path: str | Path, label: str) -> str:
    return _cached_extract_text(str(Path(path).resolve()), str(label))


def _fast_consensus_lines(
    paths: Iterable[str | Path], bikes: list[int]
) -> tuple[list[list[int]], str, str]:
    """Return early only when two different official PDFs agree by coordinates.

    If that strict condition is not met, fall back to the existing full parser.
    """
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


def _bounded_previous_day(
    venue: str | None,
    race_date: str | None,
    day_no: int,
    race_no: int,
    rider_names: list[str],
) -> dict[str, Any]:
    """Keep optional KDreams enrichment from ever blocking the main prediction."""
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

    # All these modules imported the same extractor by value, so patch each alias.
    pdf_adapter._extract_text = _extract_text_cached
    real_adapter._extract_text = _extract_text_cached
    line_runtime._extract_text = _extract_text_cached
    pr31_runtime._extract_text = _extract_text_cached

    # normalize_real_bundle imports parse_lines_from_pdfs dynamically at call time.
    line_runtime.parse_lines_from_pdfs = _fast_consensus_lines

    # predict_pr31 resolves this module global at call time.
    pr31_runtime.fetch_previous_day = _bounded_previous_day
    _INSTALLED = True
