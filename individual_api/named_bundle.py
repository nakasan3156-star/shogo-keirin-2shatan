"""Fast fixed-role KEIRIN.JP three-PDF normalization.

The web UI already knows the three PDF roles. Re-running role discovery across
all PDFs adds work and extra failure points, so this path parses each fixed role
exactly once. PR31 prediction logic is not changed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import keirin_real_pdf_adapter as real
from .keirin_pdf_adapter import PdfInputError


def _parse_fixed_role_lines(
    basic_path: Path,
    hs_path: Path,
    odds_path: Path,
    bikes: list[int],
) -> tuple[list[list[int]], str, str]:
    """Use the official odds PDF lineup first; retain the full resilient fallback."""
    from . import keirin_line_runtime_fix as line_runtime

    coordinate = line_runtime._coordinate_candidates(odds_path, bikes)
    if coordinate:
        top = coordinate[0]
        parsed = [list(line) for line in top.lines]
        if line_runtime._valid(parsed, bikes):
            return parsed, odds_path.name, f"fixed_odds_fast_v2:{top.method}"

    return line_runtime.parse_lines_from_pdfs(
        [odds_path, basic_path, hs_path], bikes
    )


def normalize_named_bundle(
    basic_pdf: str | Path,
    hs_pdf: str | Path,
    odds_pdf: str | Path,
    ex_image: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    basic_path = Path(basic_pdf)
    hs_path = Path(hs_pdf)
    odds_path = Path(odds_pdf)

    basic_text = real._extract_text(basic_path, basic_path.name)
    basic_identity = real._identity(basic_text, basic_path.name)
    riders = real.parse_basic_real(basic_text, basic_path)
    bikes = [int(rider["bike"]) for rider in riders]

    hs_text = real._extract_text(hs_path, hs_path.name)
    odds_text = real._extract_text(odds_path, odds_path.name)
    basic_doc = {"path": basic_path, "text": basic_text, "identity": basic_identity}
    hs_doc = {
        "path": hs_path,
        "text": hs_text,
        "identity": real._identity(hs_text, hs_path.name),
    }
    odds_doc = {
        "path": odds_path,
        "text": odds_text,
        "identity": real._identity(odds_text, odds_path.name),
    }
    if not real._same_race((basic_doc, hs_doc, odds_doc)):
        raise PdfInputError(
            "REAL_NAMED_RACE_MISMATCH",
            "3種類のPDFが同じレースではありません。選び直してください。",
            ["同じレースの3PDF"],
        )

    hs_rows = real.parse_hs_real(hs_text, bikes, hs_path)
    odds = real._parse_keirin_jp_odds_pdf(odds_path, odds_text, bikes)

    expected_odds = len(bikes) * (len(bikes) - 1)
    actual_odds = sum(
        1
        for first in range(len(bikes))
        for second in range(len(bikes))
        if first != second and odds[first][second] is not None
    )
    if actual_odds != expected_odds:
        raise PdfInputError(
            "REAL_NAMED_ODDS_INCOMPLETE",
            "2車単オッズPDFを最後まで読み取れませんでした。選び直してください。",
            ["2車単オッズPDF"],
        )

    for rider in riders:
        rider.update(hs_rows[int(rider["bike"])])

    lines, line_source, line_method = _parse_fixed_role_lines(
        basic_path, hs_path, odds_path, bikes
    )
    identity = basic_doc["identity"]
    race_number = int(identity["race"])

    payload = {
        "grade": real._grade(basic_text),
        "source_files": {
            "racecard_pdf": basic_path.name,
            "hs_pdf": hs_path.name,
            "odds_pdf": odds_path.name,
        },
        "riders": riders,
        "lines": lines,
        "odds": odds,
        "conditions": {},
    }
    audit = {
        "race": identity,
        "selection_method": "real_named_parse",
        "selected": {
            "basic": basic_path.name,
            "hs": hs_path.name,
            "odds": odds_path.name,
        },
        "ignored": [],
        "rider_count": len(riders),
        "odds_count": actual_odds,
        "lines": lines,
        "line_source": line_source,
        "line_method": line_method,
        "pre_race_status": {
            "basic_pdf": real._pre_race_status(basic_text, race_number, "basic_pdf"),
            "hs_pdf": real._pre_race_status(hs_text, race_number, "hs_pdf"),
            "odds_pdf": real._keirin_jp_odds_status(odds_text, race_number),
        },
        "actual_pdf_verified_layout": True,
        "result_data_used": False,
        "web_data_used": False,
        "missing_optional": ["ex_image"] if ex_image is None else [],
    }
    return payload, audit
