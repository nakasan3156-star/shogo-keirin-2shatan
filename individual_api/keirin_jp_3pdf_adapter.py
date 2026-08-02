"""KEIRIN.JP公式3PDFを正規化する競輪2車単アダプター。

必須:
1. 基本情報・並び予想PDF
2. 着度数・H・S回数PDF
3. 2車単オッズPDF

EX画像は任意。
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

try:
    from .keirin_jp_pdf_adapter import (
        _keirin_jp_odds_status,
        _parse_keirin_jp_odds_pdf,
    )
    from .keirin_pdf_adapter import (
        PREFECTURE_TO_REGION,
        PdfInputError,
        _check_pdf,
        _extract_text,
        _grade,
        _identity,
        _normalize_prefecture,
        _parse_lines,
        _pre_race_status,
    )
except ImportError:
    from keirin_jp_pdf_adapter import (
        _keirin_jp_odds_status,
        _parse_keirin_jp_odds_pdf,
    )
    from keirin_pdf_adapter import (
        PREFECTURE_TO_REGION,
        PdfInputError,
        _check_pdf,
        _extract_text,
        _grade,
        _identity,
        _normalize_prefecture,
        _parse_lines,
        _pre_race_status,
    )


REQUIRED_UPLOADS = ("basic_pdf", "hs_pdf", "odds_pdf")
SOURCE_LABELS = {
    "basic_pdf": "KEIRIN.JP 基本情報・並び予想PDF",
    "hs_pdf": "KEIRIN.JP 着度数・H・S回数PDF",
    "odds_pdf": "KEIRIN.JP 2車単オッズPDF",
}
_CLASS_LINE = re.compile(r"^([^/]+)/([ASL]\d)([ASL]\d)/(逃|追|両)$")
_SCORE_LINE = re.compile(
    r"^([0-9]{2,3}\.[0-9]{1,2})\s+"
    r"(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+"
    r"(\d{1,2})\s+(\d{1,2})$"
)
_HS_LINE = re.compile(
    r"^(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})\s+"
    r"(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})$"
)
_BIKE_NAME = re.compile(r"^([1-9])(?:\s+(.+))?$")


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\u00a0", " ")


def _source_guard(text: str, label: str) -> None:
    if "COPYRIGHT JKA" not in text and "開催発売案内" not in text:
        raise PdfInputError(
            "SOURCE_MISMATCH",
            f"{label}はKEIRIN.JP公式PDFではありません",
            [label],
        )


def _table_lines(text: str, start_marker: str) -> list[str]:
    normalized = _norm(text)
    start = normalized.find(start_marker)
    if start < 0:
        start = 0
    end = normalized.find("誘導", start)
    if end < 0:
        end = len(normalized)
    return [
        re.sub(r"\s+", " ", line).strip()
        for line in normalized[start:end].splitlines()
        if line.strip()
    ]


def _bike_and_name(lines: list[str], class_index: int) -> tuple[int, str]:
    """級班行の直前から車番と氏名を復元する。

    KEIRIN.JPは「1 氏名」と同じ行の場合と、
    「1」「氏名」が別行の場合がある。9車では枠番がさらに前にあるが、
    氏名に最も近い一桁数字を車番として採用する。
    """
    previous = lines[class_index - 1] if class_index >= 1 else ""
    direct = _BIKE_NAME.fullmatch(previous)
    if direct and direct.group(2):
        return int(direct.group(1)), re.sub(r"\s+", "", direct.group(2))

    name = re.sub(r"\s+", "", previous)
    if not name or name.isdigit():
        raise PdfInputError("RIDER_NAME_PARSE_FAILED", "選手名を取得できません")

    for index in range(class_index - 2, max(-1, class_index - 5), -1):
        candidate = lines[index]
        match = _BIKE_NAME.fullmatch(candidate)
        if match:
            if match.group(2):
                return int(match.group(1)), re.sub(r"\s+", "", match.group(2))
            return int(match.group(1)), name
    raise PdfInputError("BIKE_PARSE_FAILED", f"{name}の車番を取得できません")


def parse_basic_text(text: str) -> list[dict[str, Any]]:
    lines = _table_lines(text, "競走得点")
    riders: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        class_match = _CLASS_LINE.fullmatch(line)
        if not class_match or index + 1 >= len(lines):
            continue
        score_match = _SCORE_LINE.fullmatch(lines[index + 1])
        if not score_match:
            continue
        bike, name = _bike_and_name(lines, index)
        raw_prefecture = re.sub(r"\s+", "", class_match.group(1))
        prefecture = _normalize_prefecture(raw_prefecture)
        riders.append(
            {
                "bike": bike,
                "name": name,
                "region": PREFECTURE_TO_REGION[prefecture]
                if prefecture
                else "未取得",
                "prefecture_raw": raw_prefecture or "未取得",
                "style": class_match.group(4),
                "score": float(score_match.group(1)),
                "escape": int(score_match.group(2)),
                "makuri": int(score_match.group(3)),
                "sashi": int(score_match.group(4)),
                "mark": int(score_match.group(5)),
                "B": int(score_match.group(6)),
            }
        )

    unique = {int(rider["bike"]): rider for rider in riders}
    riders = [unique[bike] for bike in sorted(unique)]
    if len(riders) not in {5, 6, 7, 8, 9}:
        raise PdfInputError(
            "BASIC_PARSE_FAILED",
            f"基本情報の選手取得に失敗しました（{len(riders)}人）",
        )
    expected = list(range(1, len(riders) + 1))
    if [int(rider["bike"]) for rider in riders] != expected:
        raise PdfInputError("BIKE_SEQUENCE_ERROR", "基本情報の車番が連番ではありません")
    return riders


def parse_hs_text(text: str, bikes: list[int]) -> dict[int, dict[str, int | float]]:
    lines = _table_lines(text, "着\n外")
    rows: dict[int, dict[str, int | float]] = {}
    for index, line in enumerate(lines):
        class_match = _CLASS_LINE.fullmatch(line)
        if not class_match or index + 1 >= len(lines):
            continue
        hs_match = _HS_LINE.fullmatch(lines[index + 1])
        if not hs_match:
            continue
        bike, _ = _bike_and_name(lines, index)
        first, second, third, out, h_count, s_count = map(int, hs_match.groups())
        total = first + second + third + out
        rows[bike] = {
            "first": first,
            "second": second,
            "third": third,
            "out": out,
            "H": h_count,
            "S": s_count,
            "win_rate": 100.0 * first / total if total else 0.0,
            "quinella_rate": 100.0 * (first + second) / total if total else 0.0,
        }
    if set(rows) != set(bikes):
        raise PdfInputError(
            "HS_PARSE_FAILED",
            f"H・S回数は{len(bikes)}人必要ですが{len(rows)}人取得しました",
        )
    return rows


def _validate_identities(
    texts: dict[str, str], paths: dict[str, Path]
) -> dict[str, str | int | None]:
    identities = {
        key: _identity(text, paths[key].name) for key, text in texts.items()
    }
    if any(
        identity["venue"] is None or identity["race"] is None
        for identity in identities.values()
    ):
        raise PdfInputError(
            "RACE_ID_NOT_FOUND", "3PDFの開催場・レース番号を確認できません"
        )
    venue_races = {
        (identity["venue"], identity["race"])
        for identity in identities.values()
    }
    if len(venue_races) != 1:
        raise PdfInputError("RACE_MISMATCH", "3PDFが同じレースではありません")
    dates = {
        identity["date"]
        for identity in identities.values()
        if identity["date"] is not None
    }
    if len(dates) > 1:
        raise PdfInputError("RACE_MISMATCH", "3PDFの日付が一致しません")
    venue, race = next(iter(venue_races))
    return {"venue": venue, "race": race, "date": next(iter(dates), None)}


def normalize_three_pdfs(
    basic_pdf: str | Path,
    hs_pdf: str | Path,
    odds_pdf: str | Path,
    ex_image: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "basic_pdf": _check_pdf(basic_pdf, "basic_pdf"),
        "hs_pdf": _check_pdf(hs_pdf, "hs_pdf"),
        "odds_pdf": _check_pdf(odds_pdf, "odds_pdf"),
    }
    texts = {
        key: _norm(_extract_text(path, key)) for key, path in paths.items()
    }
    for key, text in texts.items():
        _source_guard(text, SOURCE_LABELS[key])

    if "競走得点" not in texts["basic_pdf"]:
        raise PdfInputError(
            "SOURCE_ROLE_MISMATCH",
            "①は基本情報の競走得点表を保存してください",
            [SOURCE_LABELS["basic_pdf"]],
        )
    if not re.search(r"着\s*外", texts["hs_pdf"]) or not re.search(
        r"H[・·\s]*S", texts["hs_pdf"]
    ):
        raise PdfInputError(
            "SOURCE_ROLE_MISMATCH",
            "②は着度数・H・S回数を保存してください",
            [SOURCE_LABELS["hs_pdf"]],
        )
    if "2車単オッズ" not in texts["odds_pdf"]:
        raise PdfInputError(
            "SOURCE_ROLE_MISMATCH",
            "③は2車単オッズを保存してください",
            [SOURCE_LABELS["odds_pdf"]],
        )

    identity = _validate_identities(texts, paths)
    race_number = int(identity["race"])
    status = {
        "basic_pdf": _pre_race_status(
            texts["basic_pdf"], race_number, "basic_pdf"
        ),
        "hs_pdf": _pre_race_status(texts["hs_pdf"], race_number, "hs_pdf"),
        "odds_pdf": _keirin_jp_odds_status(texts["odds_pdf"], race_number),
    }

    riders = parse_basic_text(texts["basic_pdf"])
    bikes = [int(rider["bike"]) for rider in riders]
    hs_rows = parse_hs_text(texts["hs_pdf"], bikes)
    for rider in riders:
        rider.update(hs_rows[int(rider["bike"])])

    # 並び予想はKEIRIN.JPの2車単オッズPDFに掲載される。
    lines = _parse_lines(paths["odds_pdf"], bikes)
    odds = _parse_keirin_jp_odds_pdf(
        paths["odds_pdf"], texts["odds_pdf"], bikes
    )

    source_files = {
        "racecard_pdf": paths["basic_pdf"].name,
        "basic_pdf": paths["basic_pdf"].name,
        "hs_pdf": paths["hs_pdf"].name,
        "odds_pdf": paths["odds_pdf"].name,
    }
    missing_optional: list[str] = []
    if ex_image is None:
        missing_optional.append("ex_image")
    else:
        image_path = Path(ex_image)
        try:
            if image_path.stat().st_size <= 0:
                raise OSError
        except OSError:
            missing_optional.append("ex_image")
        else:
            source_files["ex_image"] = image_path.name

    payload = {
        "grade": _grade(texts["basic_pdf"]),
        "source_files": source_files,
        "riders": riders,
        "lines": lines,
        "odds": odds,
        "conditions": {},
    }
    audit = {
        "race": identity,
        "input_source": "KEIRIN.JP_ONLY",
        "required_pdf_count": 3,
        "required_inputs": list(REQUIRED_UPLOADS),
        "rider_count": len(riders),
        "odds_count": len(bikes) * (len(bikes) - 1),
        "lines": lines,
        "pre_race_status": status,
        "feature_usage": {
            "basic_pdf": [
                "score",
                "escape",
                "makuri",
                "sashi",
                "mark",
                "B",
            ],
            "hs_pdf": [
                "first",
                "second",
                "third",
                "out",
                "H",
                "S",
                "win_rate",
            ],
            "odds_pdf": ["lines", "all_ordered_pair_odds"],
        },
        "result_data_used": False,
        "web_data_used": False,
        "missing_optional": missing_optional
        + ["wind_mps", "temperature_c", "bank_type"],
    }
    return payload, audit
