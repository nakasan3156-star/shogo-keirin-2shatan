"""KEIRIN.JP公式3PDFを正規化する競輪2車単アダプター。

2枚のレース情報PDFは順不同。内容から基本情報とH・S回数を自動判定する。
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

try:
    from .keirin_jp_pdf_adapter import _keirin_jp_odds_status, _parse_keirin_jp_odds_pdf
    from .keirin_jp_6pdf_adapter import _source_guard, parse_hs_text
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
    from keirin_jp_pdf_adapter import _keirin_jp_odds_status, _parse_keirin_jp_odds_pdf
    from keirin_jp_6pdf_adapter import _source_guard, parse_hs_text
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

REQUIRED_UPLOADS = ("race_info_pdf_1", "race_info_pdf_2", "odds_pdf")
_VALID_RIDER_COUNTS = {5, 6, 7, 8, 9}
_NAME_SUFFIX = re.compile(r"(?:追加|補充|欠場|再乗)$")


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\u00a0", " ")


def _clean_name(value: str) -> str:
    value = re.sub(r"\s+", "", value)
    value = value.replace("(追加)", "").replace("(補充)", "")
    return _NAME_SUFFIX.sub("", value)


def _rider_record(
    bike: int,
    name: str,
    raw_prefecture: str,
    style: str,
    values: tuple[str, str, str, str, str, str],
) -> dict[str, Any]:
    prefecture_raw = re.sub(r"\s+", "", raw_prefecture)
    prefecture = _normalize_prefecture(prefecture_raw)
    score, escape, makuri, sashi, mark, back = values
    return {
        "bike": bike,
        "name": _clean_name(name),
        "region": PREFECTURE_TO_REGION.get(prefecture, "未取得") if prefecture else "未取得",
        "prefecture_raw": prefecture_raw or "未取得",
        "style": style,
        "score": float(score),
        "escape": int(escape),
        "makuri": int(makuri),
        "sashi": int(sashi),
        "mark": int(mark),
        "B": int(back),
    }


def _parse_basic_linewise(text: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in _norm(text).splitlines()]
    profile_re = re.compile(r"^([^/]+)/((?:[ASL]\d){1,2})/(逃|追|両)$")
    stats_re = re.compile(
        r"^([0-9]{2,3}\.[0-9]{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+"
        r"(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})(?:\s|$)"
    )
    riders: list[dict[str, Any]] = []

    for index, line in enumerate(lines):
        profile = profile_re.match(re.sub(r"\s+", "", line))
        if not profile:
            continue

        previous = index - 1
        while previous >= 0 and not lines[previous]:
            previous -= 1
        if previous < 0:
            continue

        bike: int | None = None
        name = ""
        inline = re.match(r"^(?:[1-6]\s+)?([1-9])\s+(.+)$", lines[previous])
        if inline:
            bike = int(inline.group(1))
            name = inline.group(2)
        else:
            name = lines[previous]
            bike_line = previous - 1
            while bike_line >= 0 and not lines[bike_line]:
                bike_line -= 1
            if bike_line >= 0:
                bike_match = re.match(r"^(?:[1-6]\s+)?([1-9])$", lines[bike_line])
                if bike_match:
                    bike = int(bike_match.group(1))
        if bike is None or not _clean_name(name):
            continue

        following = index + 1
        while following < len(lines) and not lines[following]:
            following += 1
        if following >= len(lines):
            continue
        # PDFによって数値列が折り返されるため、最大3行を連結して判定する。
        stats_text = " ".join(lines[following : following + 3])
        stats = stats_re.match(stats_text)
        if not stats:
            continue

        riders.append(
            _rider_record(
                bike,
                name,
                profile.group(1),
                profile.group(3),
                tuple(stats.groups()),
            )
        )
    return riders


def _parse_basic_blockwise(text: str) -> list[dict[str, Any]]:
    """行順が崩れたPDF向けの第2解析経路。"""
    normalized = _norm(text)
    start_candidates = [
        value for value in (normalized.find("競走得点"), normalized.find("枠\n番\n車\n番"))
        if value >= 0
    ]
    start = min(start_candidates) if start_candidates else 0
    end_candidates = [
        value for value in (
            normalized.find("誘導", start + 1),
            normalized.find("COPYRIGHT JKA", start + 1),
        )
        if value >= 0
    ]
    body = normalized[start : min(end_candidates) if end_candidates else len(normalized)]
    pattern = re.compile(
        r"(?ms)(?:^|\n)\s*(?:[1-6]\s+)?([1-9])(?:[ \t]+([^\n]+))?\s*\n"
        r"(?:\s*([^\n]+?)\s*\n)?\s*([^/\n]+)/(?:[ASL]\d){1,2}/(逃|追|両)\s*\n"
        r"\s*([0-9]{2,3}\.[0-9]{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+"
        r"(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})"
    )
    riders: list[dict[str, Any]] = []
    for match in pattern.finditer(body):
        riders.append(
            _rider_record(
                int(match.group(1)),
                match.group(2) or match.group(3) or "",
                match.group(4),
                match.group(5),
                tuple(match.groups()[5:11]),
            )
        )
    return riders


def parse_basic_text(text: str) -> list[dict[str, Any]]:
    """基本情報を2経路で解析し、車番単位で統合する。"""
    combined = _parse_basic_linewise(text) + _parse_basic_blockwise(text)
    unique = {
        int(rider["bike"]): rider
        for rider in combined
        if rider.get("name") and 1 <= int(rider["bike"]) <= 9
    }
    riders = [unique[bike] for bike in sorted(unique)]
    if len(riders) not in _VALID_RIDER_COUNTS:
        raise PdfInputError(
            "BASIC_PARSE_FAILED",
            f"競走得点が載った基本情報PDFを確認できません（取得{len(riders)}人）",
        )
    expected = list(range(1, len(riders) + 1))
    if [int(rider["bike"]) for rider in riders] != expected:
        raise PdfInputError("BIKE_SEQUENCE_ERROR", "基本情報の車番が連番ではありません")
    return riders


def _detect_race_info_roles(
    text_1: str,
    text_2: str,
) -> tuple[int, list[dict[str, Any]], dict[int, dict[str, int | float]]]:
    """2枚の順番を無視して基本情報とH・S回数を判定する。"""
    basic_candidates: dict[int, list[dict[str, Any]]] = {}
    for index, text in enumerate((text_1, text_2)):
        try:
            basic_candidates[index] = parse_basic_text(text)
        except PdfInputError:
            pass

    if not basic_candidates:
        raise PdfInputError(
            "BASIC_PDF_NOT_FOUND",
            "2枚のレース情報PDFに、競走得点が載った基本情報PDFがありません",
        )
    if len(basic_candidates) > 1:
        raise PdfInputError(
            "DUPLICATE_BASIC_PDF",
            "基本情報PDFを2枚選んでいます。もう1枚は着度数・H・S回数PDFにしてください",
        )

    basic_index, riders = next(iter(basic_candidates.items()))
    hs_index = 1 - basic_index
    bikes = [int(rider["bike"]) for rider in riders]
    try:
        hs = parse_hs_text((text_1, text_2)[hs_index], bikes)
    except PdfInputError as exc:
        raise PdfInputError(
            "HS_PDF_NOT_FOUND",
            "もう1枚に着度数・H・S回数の表がありません。2枚は順不同で選べます",
        ) from exc
    return basic_index, riders, hs


def _validate_identity(texts: dict[str, str], paths: dict[str, Path]) -> dict[str, Any]:
    identities = {key: _identity(text, paths[key].name) for key, text in texts.items()}
    if any(value["venue"] is None or value["race"] is None for value in identities.values()):
        raise PdfInputError("RACE_ID_NOT_FOUND", "3PDFの開催場・レース番号を確認できません")
    venue_races = {(value["venue"], value["race"]) for value in identities.values()}
    if len(venue_races) != 1:
        raise PdfInputError("RACE_MISMATCH", "3PDFが同じレースではありません")
    dates = {value["date"] for value in identities.values() if value["date"] is not None}
    if len(dates) > 1:
        raise PdfInputError("RACE_MISMATCH", "3PDFの日付が一致しません")
    venue, race = next(iter(venue_races))
    return {"venue": venue, "race": race, "date": next(iter(dates), None)}


def normalize_three_pdfs(
    race_info_pdf_1: str | Path,
    race_info_pdf_2: str | Path,
    odds_pdf: str | Path,
    ex_image: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "race_info_pdf_1": _check_pdf(race_info_pdf_1, "race_info_pdf_1"),
        "race_info_pdf_2": _check_pdf(race_info_pdf_2, "race_info_pdf_2"),
        "odds_pdf": _check_pdf(odds_pdf, "odds_pdf"),
    }
    texts = {key: _norm(_extract_text(path, key)) for key, path in paths.items()}
    for key, text in texts.items():
        _source_guard(text, key)

    identity = _validate_identity(texts, paths)
    basic_index, riders, hs = _detect_race_info_roles(
        texts["race_info_pdf_1"], texts["race_info_pdf_2"]
    )
    basic_key = f"race_info_pdf_{basic_index + 1}"
    hs_key = f"race_info_pdf_{2 - basic_index}"
    basic_path, hs_path = paths[basic_key], paths[hs_key]
    basic_text, hs_text = texts[basic_key], texts[hs_key]

    bikes = [int(rider["bike"]) for rider in riders]
    for rider in riders:
        rider.update(hs[int(rider["bike"])])

    lines = _parse_lines(basic_path, bikes)
    odds = _parse_keirin_jp_odds_pdf(paths["odds_pdf"], texts["odds_pdf"], bikes)
    race_number = int(identity["race"])
    payload = {
        "grade": _grade(basic_text),
        "source_files": {
            "racecard_pdf": basic_path.name,
            "hs_pdf": hs_path.name,
            "odds_pdf": paths["odds_pdf"].name,
        },
        "riders": riders,
        "lines": lines,
        "odds": odds,
        "conditions": {},
    }
    audit = {
        "race": identity,
        "input_source": "KEIRIN.JP_ONLY",
        "required_pdf_count": 3,
        "input_order_ignored": True,
        "detected_roles": {
            "basic_pdf": basic_path.name,
            "hs_pdf": hs_path.name,
            "odds_pdf": paths["odds_pdf"].name,
        },
        "rider_count": len(riders),
        "odds_count": len(bikes) * (len(bikes) - 1),
        "lines": lines,
        "pre_race_status": {
            "basic_pdf": _pre_race_status(basic_text, race_number, "basic_pdf"),
            "hs_pdf": _pre_race_status(hs_text, race_number, "hs_pdf"),
            "odds_pdf": _keirin_jp_odds_status(texts["odds_pdf"], race_number),
        },
        "result_data_used": False,
        "web_data_used": False,
        "missing_optional": ["ex_image"] if ex_image is None else [],
    }
    return payload, audit
