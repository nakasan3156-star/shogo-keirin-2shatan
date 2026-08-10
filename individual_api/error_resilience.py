"""KEIRIN.JP fixed-input resilience guards.

This module does not change PR31 features, probabilities, calibration, EV, or
purchase rules.  It only adds a final lineup parser for browser-saved official
KEIRIN.JP PDFs when the normal multi-parser cannot resolve the lineup.
"""
from __future__ import annotations

import logging
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Iterable

from . import keirin_line_runtime_fix as line_runtime
from .keirin_pdf_adapter import PdfInputError

_BASE_PARSE_LINES = None
_INSTALLED = False


def _norm(value: str) -> str:
    return unicodedata.normalize("NFKC", value or "").replace("\u00a0", " ")


def _group_variants(found: list[int], xs: list[float]) -> list[tuple[list[list[int]], float]]:
    """Generate safe grouping candidates from small but real horizontal gaps.

    The normal parser deliberately requires a strong 1.28 gap ratio.  Some
    browser PDF renderers compress whitespace enough that the correct line gaps
    are only around 1.1x.  This fallback is used only after the normal parser
    fails, and only accepts complete permutations of the race bike numbers.
    """
    if len(found) < 2 or len(found) != len(xs):
        return []
    gaps = [float(xs[i] - xs[i - 1]) for i in range(1, len(xs))]
    if any(gap <= 0 for gap in gaps):
        return []
    unique = sorted(set(round(gap, 4) for gap in gaps))
    if len(unique) < 2:
        return []

    variants: dict[tuple[tuple[int, ...], ...], float] = {}
    for low, high in zip(unique, unique[1:]):
        threshold = (low + high) / 2.0
        within = [gap for gap in gaps if gap <= threshold]
        between = [gap for gap in gaps if gap > threshold]
        if not within or not between:
            continue
        separation = min(between) / max(within)
        if separation < 1.08:
            continue
        groups: list[list[int]] = [[found[0]]]
        for bike, gap in zip(found[1:], gaps):
            if gap > threshold:
                groups.append([])
            groups[-1].append(bike)
        if len(groups) < 2:
            continue
        key = tuple(tuple(group) for group in groups)
        variants[key] = max(variants.get(key, 0.0), separation)
    return [([list(group) for group in key], score) for key, score in variants.items()]


def _row_candidates_from_words(path: Path, bikes: list[int]) -> list[tuple[float, list[list[int]], str]]:
    candidates: list[tuple[float, list[list[int]], str]] = []
    try:
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        import pdfplumber

        with pdfplumber.open(path) as document:
            for page_no, page in enumerate(document.pages, start=1):
                words = page.extract_words(x_tolerance=1.5, y_tolerance=2)
                labels = [
                    word for word in words
                    if "並び予想" in _norm(str(word.get("text", "")))
                ]
                digits = [
                    word for word in words
                    if re.fullmatch(r"[1-9]", _norm(str(word.get("text", ""))))
                    and int(_norm(str(word["text"]))) in bikes
                ]
                heights = [float(w.get("height", 0.0)) for w in digits if float(w.get("height", 0.0)) > 0]
                tolerance = max(2.0, min(8.0, (statistics.median(heights) if heights else 10.0) * 0.70))
                for row in line_runtime._cluster_by_y(digits, tolerance):
                    row.sort(key=lambda word: float(word["x0"]))
                    found = [int(_norm(str(word["text"]))) for word in row]
                    if len(found) != len(bikes) or sorted(found) != bikes:
                        continue
                    xs = [float(word["x0"]) for word in row]
                    row_top = statistics.fmean(float(word["top"]) for word in row)
                    label_distance = min(
                        (abs(row_top - float(label["top"])) for label in labels),
                        default=999.0,
                    )
                    for groups, separation in _group_variants(found, xs):
                        if not line_runtime._valid(groups, bikes):
                            continue
                        score = separation * 100.0
                        if label_distance <= 180:
                            score += 120.0 - min(label_distance, 120.0)
                        if "オッズ" in path.name:
                            score += 100.0
                        candidates.append((score, groups, f"resilient_words:p{page_no}"))
    except Exception:
        return []
    return candidates


def _row_candidates_from_layout(path: Path, bikes: list[int]) -> list[tuple[float, list[list[int]], str]]:
    try:
        text = _norm(line_runtime._extract_text(path, path.name))
    except Exception:
        return []
    rows = text.splitlines()
    labels = [index for index, row in enumerate(rows) if "並び予想" in row]
    candidates: list[tuple[float, list[list[int]], str]] = []
    for row_no, raw in enumerate(rows):
        tokens = list(re.finditer(r"(?<!\d)([1-9])(?!\d)", raw))
        found = [int(token.group(1)) for token in tokens]
        if len(found) != len(bikes) or sorted(found) != bikes:
            continue
        xs = [float(token.start()) for token in tokens]
        distance = min((abs(row_no - label) for label in labels), default=999)
        for groups, separation in _group_variants(found, xs):
            if not line_runtime._valid(groups, bikes):
                continue
            score = separation * 80.0
            if distance <= 80:
                score += 100.0 - min(float(distance), 100.0)
            if "オッズ" in path.name:
                score += 100.0
            candidates.append((score, groups, f"resilient_layout:l{row_no + 1}"))
    return candidates


def _fallback_lines(paths: Iterable[str | Path], bikes: list[int]) -> tuple[list[list[int]], str, str]:
    ordered = [Path(raw) for raw in paths]
    ordered.sort(key=lambda path: ("オッズ" not in path.name, path.name))
    candidates: list[tuple[float, list[list[int]], str, Path]] = []
    for path in ordered:
        for score, groups, method in _row_candidates_from_words(path, bikes):
            candidates.append((score, groups, method, path))
        for score, groups, method in _row_candidates_from_layout(path, bikes):
            candidates.append((score, groups, method, path))

    if not candidates:
        raise PdfInputError(
            "LINE_PARSE_FAILED_ALL_SOURCES",
            "並び予想を読み取れませんでした。3PDFを選び直してください。",
            [path.name for path in ordered],
        )

    totals: dict[tuple[tuple[int, ...], ...], float] = {}
    sources: dict[tuple[tuple[int, ...], ...], set[str]] = {}
    best: dict[tuple[tuple[int, ...], ...], tuple[float, str, Path]] = {}
    for score, groups, method, path in candidates:
        key = tuple(tuple(group) for group in groups)
        totals[key] = totals.get(key, 0.0) + score
        sources.setdefault(key, set()).add(path.name)
        current = best.get(key)
        if current is None or score > current[0]:
            best[key] = (score, method, path)

    def rank(key: tuple[tuple[int, ...], ...]) -> tuple[int, int, float]:
        odds_support = int(any("オッズ" in source for source in sources[key]))
        return odds_support, len(sources[key]), totals[key]

    winner = max(totals, key=rank)
    groups = [list(group) for group in winner]
    if not line_runtime._valid(groups, bikes):
        raise PdfInputError("LINE_VALIDATION_FAILED", "並び予想の検証に失敗しました。")
    _score, method, path = best[winner]
    return groups, path.name, method


def _resilient_parse_lines(
    paths: Iterable[str | Path], bikes: list[int]
) -> tuple[list[list[int]], str, str]:
    path_list = list(paths)
    try:
        return _BASE_PARSE_LINES(path_list, bikes)
    except PdfInputError as exc:
        if exc.code not in {
            "LINE_PARSE_FAILED_ALL_SOURCES",
            "LINE_PARSE_AMBIGUOUS",
            "LINE_VALIDATION_FAILED",
        }:
            raise
        return _fallback_lines(path_list, bikes)


def install_error_resilience() -> None:
    global _BASE_PARSE_LINES, _INSTALLED
    if _INSTALLED:
        return
    _BASE_PARSE_LINES = line_runtime.parse_lines_from_pdfs
    line_runtime.parse_lines_from_pdfs = _resilient_parse_lines
    _INSTALLED = True
