"""Try KEIRIN.JP line formation from every selected same-race PDF."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .keirin_pdf_adapter import PdfInputError
from .keirin_line_runtime_fix import parse_lines_resilient


def parse_lines_from_selected_pdfs(paths: Iterable[str | Path], bikes: list[int]) -> tuple[list[list[int]], str]:
    errors: list[str] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path in seen:
            continue
        seen.add(path)
        try:
            lines = parse_lines_resilient(path, bikes)
            return lines, path.name
        except PdfInputError as exc:
            errors.append(f"{path.name}:{exc.code}")
    raise PdfInputError(
        "LINE_PARSE_FAILED_ALL_SOURCES",
        "並び予想を3PDFすべてから読み取れません。",
        errors,
    )
