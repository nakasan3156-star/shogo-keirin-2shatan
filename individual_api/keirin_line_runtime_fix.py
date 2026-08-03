"""Harden KEIRIN.JP line-order parsing across browser/PDF layout variants."""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

from .keirin_pdf_adapter import PdfInputError, _extract_text


def _valid(lines: list[list[int]], bikes: list[int]) -> bool:
    flat = [bike for line in lines for bike in line]
    return bool(lines) and sorted(flat) == sorted(bikes) and len(flat) == len(set(flat))


def _groups_from_positions(found: list[int], xs: list[float]) -> list[list[int]]:
    if not found:
        return []
    if len(found) == 1:
        return [[found[0]]]
    gaps = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
    positive = sorted(gap for gap in gaps if gap > 0)
    if not positive:
        return [found]
    if len(set(round(gap, 1) for gap in positive)) >= 2:
        ordered = sorted(positive)
        jumps = [ordered[i] - ordered[i - 1] for i in range(1, len(ordered))]
        split_at = max(range(len(jumps)), key=jumps.__getitem__) + 1
        low = ordered[split_at - 1]
        high = ordered[split_at]
        threshold = (low + high) / 2.0
    else:
        threshold = positive[0] * 1.5
    parsed: list[list[int]] = [[found[0]]]
    for bike, gap in zip(found[1:], gaps):
        if gap > threshold:
            parsed.append([])
        parsed[-1].append(bike)
    return parsed


def _coordinate_parse(path: Path, bikes: list[int]) -> list[list[int]] | None:
    try:
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        import pdfplumber
        with pdfplumber.open(path) as document:
            for page in document.pages[:3]:
                words = page.extract_words(x_tolerance=2, y_tolerance=2)
                labels = [w for w in words if "並び予想" in unicodedata.normalize("NFKC", w.get("text", ""))]
                for label in labels:
                    label_top = float(label["top"])
                    for low, high in ((8, 48), (0, 65), (-5, 80)):
                        row = [
                            w for w in words
                            if label_top + low <= float(w["top"]) <= label_top + high
                            and re.fullmatch(r"[1-9]", unicodedata.normalize("NFKC", w.get("text", "")))
                        ]
                        if not row:
                            continue
                        # Cluster by actual y-distance instead of fixed buckets.
                        row.sort(key=lambda w: (float(w["top"]), float(w["x0"])))
                        y_groups: list[list[dict]] = []
                        for word in row:
                            if not y_groups:
                                y_groups.append([word])
                                continue
                            mean_top = sum(float(item["top"]) for item in y_groups[-1]) / len(y_groups[-1])
                            if abs(float(word["top"]) - mean_top) <= 7:
                                y_groups[-1].append(word)
                            else:
                                y_groups.append([word])
                        for words_on_row in y_groups:
                            words_on_row.sort(key=lambda w: float(w["x0"]))
                            found = [int(unicodedata.normalize("NFKC", w["text"])) for w in words_on_row]
                            if sorted(found) != sorted(bikes) or len(found) != len(bikes):
                                continue
                            xs = [float(w["x0"]) for w in words_on_row]
                            parsed = _groups_from_positions(found, xs)
                            if _valid(parsed, bikes):
                                return parsed
    except Exception:
        return None
    return None


def _text_parse(path: Path, bikes: list[int]) -> list[list[int]] | None:
    text = unicodedata.normalize("NFKC", _extract_text(path, path.name)).replace("\u00a0", " ")
    marker = text.find("並び予想")
    windows = [text[marker:marker + 1800]] if marker >= 0 else []
    windows.append(text[:2200])
    for window in windows:
        for raw_line in window.splitlines():
            tokens = list(re.finditer(r"(?<!\d)([1-9])(?!\d)", raw_line))
            found = [int(token.group(1)) for token in tokens]
            if sorted(found) != sorted(bikes) or len(found) != len(bikes):
                continue
            xs = [float(token.start()) for token in tokens]
            parsed = _groups_from_positions(found, xs)
            if _valid(parsed, bikes):
                return parsed
    return None


def _candidate_pdfs(path: Path) -> list[Path]:
    """The lineup is printed on all three KEIRIN.JP PDFs; retry every sibling PDF."""
    candidates = [path]
    try:
        siblings = sorted(path.parent.glob("*.pdf"))
    except OSError:
        siblings = []
    for sibling in siblings:
        if sibling not in candidates:
            candidates.append(sibling)
    return candidates


def parse_lines_resilient(pdf: str | Path, bikes: list[int]) -> list[list[int]]:
    path = Path(pdf)
    attempted: list[str] = []
    for candidate in _candidate_pdfs(path):
        attempted.append(candidate.name)
        parsed = _coordinate_parse(candidate, bikes) or _text_parse(candidate, bikes)
        if parsed and _valid(parsed, bikes):
            return parsed
    raise PdfInputError(
        "LINE_PARSE_FAILED_ALL_SOURCES",
        "並び予想を3PDFすべてから読み取れません。",
        attempted,
    )


def install_line_parser_fix() -> None:
    from . import keirin_pdf_adapter
    from . import keirin_real_pdf_adapter
    keirin_pdf_adapter._parse_lines = parse_lines_resilient
    keirin_real_pdf_adapter._parse_lines = parse_lines_resilient
