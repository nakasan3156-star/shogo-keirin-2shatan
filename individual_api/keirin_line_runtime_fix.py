"""Adaptive KEIRIN.JP line-order parsing for browser-saved PDFs.

The lineup is present in every one of the three official PDFs, but browser and
printer versions change the text coordinates and whitespace.  This module
therefore scans every page of every selected PDF and combines three independent
parsers: pdfplumber word coordinates, ``pdftotext -layout`` columns, and a
character-coordinate fallback.
"""
from __future__ import annotations

import logging
import math
import re
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .keirin_pdf_adapter import PdfInputError, _extract_text


@dataclass(frozen=True)
class _Candidate:
    lines: tuple[tuple[int, ...], ...]
    source: str
    method: str
    weight: int


def _normal(value: str) -> str:
    return unicodedata.normalize("NFKC", value or "").replace("\u00a0", " ")


def _valid(lines: list[list[int]], bikes: list[int]) -> bool:
    if not 5 <= len(bikes) <= 9 or bikes != list(range(1, len(bikes) + 1)):
        return False
    if not lines or any(not line for line in lines):
        return False
    flat = [bike for line in lines for bike in line]
    return flat and sorted(flat) == bikes and len(flat) == len(set(flat))


def _gap_threshold(gaps: list[float]) -> float | None:
    """Split the PDF-specific gap distribution into within/between-line gaps."""
    positive = [float(gap) for gap in gaps if gap > 0 and math.isfinite(gap)]
    if len(positive) < 2:
        return None
    low, high = min(positive), max(positive)
    if low <= 0 or high / low < 1.28:
        return None

    left, right = low, high
    for _ in range(20):
        threshold = (left + right) / 2.0
        lower = [gap for gap in positive if gap <= threshold]
        upper = [gap for gap in positive if gap > threshold]
        if not lower or not upper:
            return None
        new_left = statistics.fmean(lower)
        new_right = statistics.fmean(upper)
        if abs(new_left - left) + abs(new_right - right) < 1e-6:
            left, right = new_left, new_right
            break
        left, right = new_left, new_right

    if right / max(left, 1e-9) < 1.28:
        return None
    return (left + right) / 2.0


def _groups_from_positions(found: list[int], xs: list[float]) -> list[list[int]]:
    if not found or len(found) != len(xs):
        return []
    if len(found) == 1:
        return [[found[0]]]
    gaps = [xs[index] - xs[index - 1] for index in range(1, len(xs))]
    if any(gap <= 0 for gap in gaps):
        return []
    threshold = _gap_threshold(gaps)
    if threshold is None:
        return [found.copy()]
    parsed: list[list[int]] = [[found[0]]]
    for bike, gap in zip(found[1:], gaps):
        if gap > threshold:
            parsed.append([])
        parsed[-1].append(bike)
    return parsed


def _cluster_by_y(words: list[dict], tolerance: float) -> list[list[dict]]:
    ordered = sorted(words, key=lambda word: (float(word["top"]), float(word["x0"])))
    clusters: list[list[dict]] = []
    for word in ordered:
        if not clusters:
            clusters.append([word])
            continue
        mean_top = statistics.fmean(float(item["top"]) for item in clusters[-1])
        if abs(float(word["top"]) - mean_top) <= tolerance:
            clusters[-1].append(word)
        else:
            clusters.append([word])
    return clusters


def _coordinate_candidates(path: Path, bikes: list[int]) -> list[_Candidate]:
    candidates: list[tuple[float, _Candidate]] = []
    try:
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        import pdfplumber

        with pdfplumber.open(path) as document:
            for page_number, page in enumerate(document.pages):
                words = page.extract_words(x_tolerance=2, y_tolerance=2)
                heights = [float(word.get("height", 0.0)) for word in words if float(word.get("height", 0.0)) > 0]
                tolerance = max(2.0, min(7.0, (statistics.median(heights) if heights else 10.0) * 0.58))
                labels = [word for word in words if "並び予想" in _normal(str(word.get("text", "")))]
                digit_words = [
                    word for word in words
                    if re.fullmatch(r"[1-9]", _normal(str(word.get("text", ""))))
                    and int(_normal(str(word["text"]))) in bikes
                ]
                for row in _cluster_by_y(digit_words, tolerance):
                    row.sort(key=lambda word: float(word["x0"]))
                    found = [int(_normal(str(word["text"]))) for word in row]
                    if len(found) != len(bikes) or sorted(found) != bikes:
                        continue
                    xs = [float(word["x0"]) for word in row]
                    parsed = _groups_from_positions(found, xs)
                    if not _valid(parsed, bikes):
                        continue
                    row_top = statistics.fmean(float(word["top"]) for word in row)
                    nearest_label = min((abs(row_top - float(label["top"])) for label in labels), default=float("inf"))
                    score = 100.0
                    if nearest_label < 180:
                        score += 90.0 - min(nearest_label, 90.0)
                    if row_top < float(page.height) * 0.48:
                        score += 15.0
                    if len(parsed) > 1:
                        score += 15.0
                    candidate = _Candidate(
                        tuple(tuple(line) for line in parsed),
                        path.name,
                        f"pdfplumber_coordinates:p{page_number + 1}",
                        300,
                    )
                    candidates.append((score, candidate))
    except Exception:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _score, candidate in candidates]


def _layout_candidates(path: Path, bikes: list[int]) -> list[_Candidate]:
    try:
        text = _normal(_extract_text(path, path.name))
    except Exception:
        return []
    rows = text.splitlines()
    labels = [index for index, row in enumerate(rows) if "並び予想" in row]
    scored: list[tuple[float, _Candidate]] = []
    for row_index, raw_line in enumerate(rows):
        tokens = list(re.finditer(r"(?<!\d)([1-9])(?!\d)", raw_line))
        found = [int(token.group(1)) for token in tokens]
        if len(found) != len(bikes) or sorted(found) != bikes:
            continue
        parsed = _groups_from_positions(found, [float(token.start()) for token in tokens])
        if not _valid(parsed, bikes):
            continue
        nearest = min((abs(row_index - label) for label in labels), default=999)
        score = 100.0 + (80.0 - min(nearest, 80) if nearest <= 80 else 0.0)
        if len(parsed) > 1:
            score += 10.0
        scored.append((score, _Candidate(
            tuple(tuple(line) for line in parsed),
            path.name,
            f"pdftotext_layout:l{row_index + 1}",
            200,
        )))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _score, candidate in scored]


def _character_candidates(path: Path, bikes: list[int]) -> list[_Candidate]:
    candidates: list[tuple[float, _Candidate]] = []
    try:
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        import pdfplumber

        with pdfplumber.open(path) as document:
            for page_number, page in enumerate(document.pages):
                chars = [
                    char for char in page.chars
                    if re.fullmatch(r"[1-9]", _normal(str(char.get("text", ""))))
                    and int(_normal(str(char["text"]))) in bikes
                ]
                heights = [float(char.get("height", 0.0)) for char in chars if float(char.get("height", 0.0)) > 0]
                tolerance = max(1.5, min(6.0, (statistics.median(heights) if heights else 9.0) * 0.55))
                page_text = _normal(page.extract_text() or "")
                for row in _cluster_by_y(chars, tolerance):
                    row.sort(key=lambda char: float(char["x0"]))
                    found = [int(_normal(str(char["text"]))) for char in row]
                    if len(found) != len(bikes) or sorted(found) != bikes:
                        continue
                    parsed = _groups_from_positions(found, [float(char["x0"]) for char in row])
                    if not _valid(parsed, bikes):
                        continue
                    score = 100.0 + (60.0 if "並び予想" in page_text else 0.0)
                    if len(parsed) > 1:
                        score += 10.0
                    candidates.append((score, _Candidate(
                        tuple(tuple(line) for line in parsed),
                        path.name,
                        f"pdfplumber_characters:p{page_number + 1}",
                        100,
                    )))
    except Exception:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _score, candidate in candidates]


def _coordinate_parse(path: Path, bikes: list[int]) -> list[list[int]] | None:
    candidates = _coordinate_candidates(Path(path), bikes)
    return [list(line) for line in candidates[0].lines] if candidates else None


def _text_parse(path: Path, bikes: list[int]) -> list[list[int]] | None:
    candidates = _layout_candidates(Path(path), bikes)
    return [list(line) for line in candidates[0].lines] if candidates else None


def _candidate_pdfs(path: Path) -> list[Path]:
    candidates = [path]
    try:
        siblings = sorted(
            sibling for sibling in path.parent.iterdir()
            if sibling.is_file() and sibling.suffix.lower() == ".pdf"
        )
    except OSError:
        siblings = []
    for sibling in siblings:
        if sibling not in candidates:
            candidates.append(sibling)
    return candidates


def parse_lines_from_pdfs(
    paths: Iterable[str | Path], bikes: list[int]
) -> tuple[list[list[int]], str, str]:
    unique_paths: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path not in unique_paths:
            unique_paths.append(path)

    attempted: list[str] = []
    all_candidates: list[_Candidate] = []
    for path in unique_paths:
        attempted.append(path.name)
        all_candidates.extend(_coordinate_candidates(path, bikes))
        all_candidates.extend(_layout_candidates(path, bikes))
        all_candidates.extend(_character_candidates(path, bikes))

    if not all_candidates:
        raise PdfInputError(
            "LINE_PARSE_FAILED_ALL_SOURCES",
            "並び予想を3PDFすべてから読み取れません。",
            attempted,
        )

    totals: dict[tuple[tuple[int, ...], ...], int] = {}
    sources: dict[tuple[tuple[int, ...], ...], set[str]] = {}
    best_candidate: dict[tuple[tuple[int, ...], ...], _Candidate] = {}
    for candidate in all_candidates:
        totals[candidate.lines] = totals.get(candidate.lines, 0) + candidate.weight
        sources.setdefault(candidate.lines, set()).add(candidate.source)
        current = best_candidate.get(candidate.lines)
        if current is None or candidate.weight > current.weight:
            best_candidate[candidate.lines] = candidate
    ranked = sorted(totals, key=lambda lines: (len(sources[lines]), totals[lines]), reverse=True)
    winner = ranked[0]
    if len(ranked) > 1:
        first_key = (len(sources[winner]), totals[winner])
        second_key = (len(sources[ranked[1]]), totals[ranked[1]])
        if first_key == second_key:
            raise PdfInputError(
                "LINE_PARSE_AMBIGUOUS",
                "並び予想の候補が一致しないため安全停止しました。",
                attempted,
            )
    selected = best_candidate[winner]
    parsed = [list(line) for line in winner]
    if not _valid(parsed, bikes):
        raise PdfInputError("LINE_VALIDATION_FAILED", "並び予想の検証に失敗しました。", attempted)
    return parsed, selected.source, selected.method


def parse_lines_resilient(pdf: str | Path, bikes: list[int]) -> list[list[int]]:
    attempted: list[str] = []
    for path in _candidate_pdfs(Path(pdf)):
        attempted.append(path.name)
        parsed = _coordinate_parse(path, bikes) or _text_parse(path, bikes)
        if parsed and _valid(parsed, bikes):
            return parsed
    raise PdfInputError(
        "LINE_PARSE_FAILED_ALL_SOURCES",
        "並び予想を3PDFすべてから読み取れません。",
        attempted,
    )


def install_line_parser_fix() -> None:
    """Keep legacy adapters on the same strict parser during migration."""
    from . import keirin_pdf_adapter
    from . import keirin_real_pdf_adapter
    from .production_runtime_fix import install_production_runtime_fix

    keirin_pdf_adapter._parse_lines = parse_lines_resilient
    keirin_real_pdf_adapter._parse_lines = parse_lines_resilient
    install_production_runtime_fix()
