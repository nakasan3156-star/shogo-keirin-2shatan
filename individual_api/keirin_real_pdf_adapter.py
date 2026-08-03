"""実際のKEIRIN.JP PDFレイアウトを解析して予測入力を作る。

見出し文字では判定しない。全選手・全H/S・全2車単を実際に読めたPDFだけ採用する。
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

try:
    from .keirin_jp_pdf_adapter import _keirin_jp_odds_status, _parse_keirin_jp_odds_pdf
    from .keirin_pdf_adapter import (
        PREFECTURE_TO_REGION,
        PdfInputError,
        _extract_text,
        _grade,
        _identity,
        _normalize_prefecture,
        _parse_lines,
        _pre_race_status,
    )
except ImportError:
    from keirin_jp_pdf_adapter import _keirin_jp_odds_status, _parse_keirin_jp_odds_pdf
    from keirin_pdf_adapter import (
        PREFECTURE_TO_REGION,
        PdfInputError,
        _extract_text,
        _grade,
        _identity,
        _normalize_prefecture,
        _parse_lines,
        _pre_race_status,
    )

_VALID_COUNTS = {5, 6, 7, 8, 9}
_NAME_SUFFIX = re.compile(r"(?:追加|補充|欠場|再乗)$")
_PROFILE_LINE = re.compile(
    r"^([^/]+)/((?:[ASL]\d){1,2})/(逃|追|両)(?:\s+(.*))?$"
)
_BASIC_VALUES = re.compile(
    r"^([0-9]{2,3}\.[0-9]{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+"
    r"(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})(?:\s|$)"
)
_HS_VALUES = re.compile(
    r"^(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})\s+"
    r"(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})(?:\s|$)"
)


def _norm(text: str) -> str:
    return (
        unicodedata.normalize("NFKC", text)
        .replace("\u00a0", " ")
        .translate(str.maketrans({"⻑": "長", "⻘": "青"}))
    )


def _clean_name(value: str) -> str:
    compact = re.sub(r"\s+", "", value)
    compact = compact.replace("(追加)", "").replace("(補充)", "")
    return _NAME_SUFFIX.sub("", compact)


def _previous_bike_name(lines: list[str], index: int) -> tuple[int | None, str]:
    previous = index - 1
    while previous >= 0 and not lines[previous]:
        previous -= 1
    if previous < 0:
        return None, ""

    inline = re.match(r"^(?:[1-6]\s+)?([1-9])\s+(.+)$", lines[previous])
    if inline:
        return int(inline.group(1)), inline.group(2)

    name = lines[previous]
    bike_line = previous - 1
    while bike_line >= 0 and not lines[bike_line]:
        bike_line -= 1
    if bike_line >= 0:
        bike_match = re.match(r"^(?:[1-6]\s+)?([1-9])$", lines[bike_line])
        if bike_match:
            return int(bike_match.group(1)), name
    return None, ""


def _profile_rows(text: str) -> list[tuple[int, str, str, str, str]]:
    """車番・氏名・府県・脚質・数値列を同一行/次行の両方から取得する。"""
    lines = [line.strip() for line in _norm(text).splitlines()]
    rows: list[tuple[int, str, str, str, str]] = []
    for index, line in enumerate(lines):
        profile = _PROFILE_LINE.match(re.sub(r"\s+", " ", line))
        if not profile:
            continue
        bike, name = _previous_bike_name(lines, index)
        if bike is None or not _clean_name(name):
            continue
        values = (profile.group(4) or "").strip()
        if not values:
            following = index + 1
            while following < len(lines) and not lines[following]:
                following += 1
            values = " ".join(lines[following : following + 3])
        rows.append((bike, _clean_name(name), profile.group(1), profile.group(3), values))
    return rows


def _coordinate_profile_rows(path: Path, value_kind: str) -> list[tuple[int, str, str, str, list[str]]]:
    """Read rider table rows by columns when pdftotext breaks narrow 9-rider rows."""
    try:
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        import pdfplumber
    except Exception:
        return []

    rows: dict[int, tuple[int, str, str, str, list[str]]] = {}
    try:
        with pdfplumber.open(path) as document:
            for page in document.pages:
                words = page.extract_words(x_tolerance=1.5, y_tolerance=2)
                anchors = []
                if value_kind == "basic":
                    anchors = [
                        word for word in words
                        if re.fullmatch(r"[0-9]{2,3}\.[0-9]{1,2}", _norm(str(word.get("text", ""))))
                    ]
                else:
                    # H/S rows have six integer columns to the right of the rider profile.
                    top_values: dict[float, list[dict]] = {}
                    for word in words:
                        value = _norm(str(word.get("text", "")))
                        if float(word.get("x0", 0.0)) > float(page.width) * 0.40 and re.fullmatch(r"\d{1,3}", value):
                            key = round(float(word["top"]), 1)
                            top_values.setdefault(key, []).append(word)
                    anchors = [
                        min(group, key=lambda word: float(word["x0"]))
                        for group in top_values.values()
                        if len(group) >= 6
                    ]

                for anchor in anchors:
                    row_top = float(anchor["top"])
                    right_numbers = [
                        word for word in words
                        if abs(float(word["top"]) - row_top) <= 2.5
                        and float(word["x0"]) >= float(page.width) * 0.40
                        and re.fullmatch(
                            r"[0-9]{2,3}\.[0-9]{1,2}|\d{1,3}",
                            _norm(str(word.get("text", ""))),
                        )
                    ]
                    right_numbers.sort(key=lambda word: float(word["x0"]))
                    values = [_norm(str(word["text"])) for word in right_numbers]
                    if value_kind == "basic":
                        if len(values) != 6 or not re.fullmatch(r"[0-9]{2,3}\.[0-9]{1,2}", values[0]):
                            continue
                    elif len(values) != 6 or any(not re.fullmatch(r"\d{1,3}", value) for value in values):
                        continue

                    bike_words = [
                        word for word in words
                        if abs(float(word["top"]) - row_top) <= 2.5
                        and float(word["x0"]) < float(page.width) * 0.20
                        and re.fullmatch(r"[1-9]", _norm(str(word.get("text", ""))))
                    ]
                    if not bike_words:
                        continue
                    # The frame number is left of the bike number when both exist.
                    bike = int(_norm(str(max(bike_words, key=lambda word: float(word["x0"]))["text"])))

                    name_words = [
                        word for word in words
                        if float(page.width) * 0.11 <= float(word["x0"]) < float(page.width) * 0.43
                        and row_top - 13.0 <= float(word["top"]) <= row_top - 2.0
                        and not re.fullmatch(r"\d+", _norm(str(word.get("text", ""))))
                    ]
                    name_words.sort(key=lambda word: (float(word["top"]), float(word["x0"])))
                    if not name_words:
                        continue
                    name_top = min(float(word["top"]) for word in name_words)
                    name = _clean_name("".join(
                        _norm(str(word["text"])) for word in name_words
                        if abs(float(word["top"]) - name_top) <= 2.5
                    ))

                    profile_words = [
                        word for word in words
                        if float(page.width) * 0.11 <= float(word["x0"]) < float(page.width) * 0.43
                        and row_top + 2.0 <= float(word["top"]) <= row_top + 19.0
                    ]
                    profile_words.sort(key=lambda word: (float(word["top"]), float(word["x0"])))
                    profile = re.sub(r"\s+", "", "".join(_norm(str(word["text"])) for word in profile_words))
                    profile_match = re.search(r"([^/]+)/((?:[ASL]\d){1,2})/(逃|追|両)", profile)
                    if not name or not profile_match:
                        continue
                    rows[bike] = (bike, name, profile_match.group(1), profile_match.group(3), values)
    except Exception:
        return []
    return [rows[bike] for bike in sorted(rows)]


def parse_basic_real(text: str, path: str | Path | None = None) -> list[dict[str, Any]]:
    riders: dict[int, dict[str, Any]] = {}
    for bike, name, raw_prefecture, style, values in _profile_rows(text):
        match = _BASIC_VALUES.match(values)
        if not match:
            continue
        prefecture_raw = re.sub(r"\s+", "", raw_prefecture)
        prefecture = _normalize_prefecture(prefecture_raw)
        score, escape, makuri, sashi, mark, back = match.groups()
        riders[bike] = {
            "bike": bike,
            "name": name,
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
    if path is not None:
        for bike, name, raw_prefecture, style, values in _coordinate_profile_rows(Path(path), "basic"):
            score, escape, makuri, sashi, mark, back = values
            prefecture_raw = re.sub(r"\s+", "", raw_prefecture)
            prefecture = _normalize_prefecture(prefecture_raw)
            riders[bike] = {
                "bike": bike,
                "name": name,
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
    result = [riders[bike] for bike in sorted(riders)]
    if len(result) not in _VALID_COUNTS:
        raise PdfInputError(
            "REAL_BASIC_PARSE_FAILED",
            f"基本情報PDFから全選手を取得できません（取得{len(result)}人）",
        )
    if [item["bike"] for item in result] != list(range(1, len(result) + 1)):
        raise PdfInputError("REAL_BASIC_SEQUENCE_ERROR", "基本情報の車番が連番ではありません")
    return result


def parse_hs_real(
    text: str, bikes: list[int], path: str | Path | None = None
) -> dict[int, dict[str, int | float]]:
    rows: dict[int, dict[str, int | float]] = {}
    for bike, _name, _prefecture, _style, values in _profile_rows(text):
        match = _HS_VALUES.match(values)
        if not match:
            continue
        first, second, third, out, h_count, s_count = map(int, match.groups())
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
    if path is not None:
        for bike, _name, _prefecture, _style, values in _coordinate_profile_rows(Path(path), "hs"):
            first, second, third, out, h_count, s_count = map(int, values)
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
            "REAL_HS_PARSE_FAILED",
            f"H・S PDFは{len(bikes)}人必要ですが{len(rows)}人しか取得できません",
        )
    return rows


def _same_race(documents: tuple[dict[str, Any], ...]) -> bool:
    venue_races = {
        (doc["identity"].get("venue"), doc["identity"].get("race"))
        for doc in documents
        if doc["identity"].get("venue") is not None
        and doc["identity"].get("race") is not None
    }
    if len(venue_races) != 1:
        return False
    dates = {
        doc["identity"].get("date")
        for doc in documents
        if doc["identity"].get("date") is not None
    }
    return len(dates) <= 1


def normalize_real_bundle(
    paths: list[str | Path],
    ex_image: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        text = _extract_text(path, path.name)
        documents.append(
            {
                "path": path,
                "text": text,
                "identity": _identity(text, path.name),
            }
        )

    basic_candidates: list[dict[str, Any]] = []
    for document in documents:
        try:
            riders = parse_basic_real(document["text"], document["path"])
        except PdfInputError:
            continue
        basic_candidates.append({"document": document, "riders": riders})

    if not basic_candidates:
        raise PdfInputError(
            "REAL_BASIC_NOT_FOUND",
            "追加したPDFに、全選手の競走得点・逃・捲・差・マ・Bを読める基本情報PDFがありません。",
            ["基本情報PDF"],
        )

    best: dict[str, Any] | None = None
    saw_hs = False
    saw_odds = False
    for basic_candidate in basic_candidates:
        basic = basic_candidate["document"]
        riders = basic_candidate["riders"]
        bikes = [int(rider["bike"]) for rider in riders]
        expected_odds = len(bikes) * (len(bikes) - 1)

        for hs_doc in documents:
            if hs_doc["path"] == basic["path"] or not _same_race((basic, hs_doc)):
                continue
            try:
                hs_rows = parse_hs_real(hs_doc["text"], bikes, hs_doc["path"])
            except PdfInputError:
                continue
            saw_hs = True

            for odds_doc in documents:
                if odds_doc["path"] in {basic["path"], hs_doc["path"]}:
                    continue
                if not _same_race((basic, hs_doc, odds_doc)):
                    continue
                try:
                    odds = _parse_keirin_jp_odds_pdf(
                        odds_doc["path"], odds_doc["text"], bikes
                    )
                except (PdfInputError, TypeError, ValueError):
                    continue
                actual_odds = sum(
                    1
                    for first in range(len(bikes))
                    for second in range(len(bikes))
                    if first != second and odds[first][second] is not None
                )
                if actual_odds != expected_odds:
                    continue
                saw_odds = True
                score = len(riders) * 1000 + actual_odds
                if best is None or score > best["score"]:
                    best = {
                        "basic": basic,
                        "hs": hs_doc,
                        "odds": odds_doc,
                        "riders": riders,
                        "hs_rows": hs_rows,
                        "odds_matrix": odds,
                        "score": score,
                    }

    if best is None:
        missing: list[str] = []
        if not saw_hs:
            missing.append("着度数・H・S回数PDF")
        if not saw_odds:
            missing.append("2車単オッズPDF")
        message = (
            "追加したPDFから必要データを最後まで読み取れません。"
            + ("不足: " + "・".join(missing) if missing else "3PDFが同じレースか確認してください。")
        )
        raise PdfInputError("REAL_BUNDLE_INCOMPLETE", message, missing)

    riders = best["riders"]
    for rider in riders:
        rider.update(best["hs_rows"][int(rider["bike"])])
    bikes = [int(rider["bike"]) for rider in riders]
    basic_path = best["basic"]["path"]
    hs_path = best["hs"]["path"]
    odds_path = best["odds"]["path"]
    basic_text = best["basic"]["text"]
    identity = best["basic"]["identity"]
    race_number = int(identity["race"])
    try:
        from .keirin_line_runtime_fix import parse_lines_from_pdfs
    except ImportError:
        from keirin_line_runtime_fix import parse_lines_from_pdfs

    lines, line_source, line_method = parse_lines_from_pdfs(
        [basic_path, hs_path, odds_path], bikes
    )

    payload = {
        "grade": _grade(basic_text),
        "source_files": {
            "racecard_pdf": basic_path.name,
            "hs_pdf": hs_path.name,
            "odds_pdf": odds_path.name,
        },
        "riders": riders,
        "lines": lines,
        "odds": best["odds_matrix"],
        "conditions": {},
    }
    selected = {basic_path, hs_path, odds_path}
    audit = {
        "race": identity,
        "selection_method": "real_full_parse",
        "selected": {
            "basic": basic_path.name,
            "hs": hs_path.name,
            "odds": odds_path.name,
        },
        "ignored": [doc["path"].name for doc in documents if doc["path"] not in selected],
        "rider_count": len(riders),
        "odds_count": len(bikes) * (len(bikes) - 1),
        "lines": lines,
        "line_source": line_source,
        "line_method": line_method,
        "pre_race_status": {
            "basic_pdf": _pre_race_status(basic_text, race_number, "basic_pdf"),
            "hs_pdf": _pre_race_status(best["hs"]["text"], race_number, "hs_pdf"),
            "odds_pdf": _keirin_jp_odds_status(best["odds"]["text"], race_number),
        },
        "actual_pdf_verified_layout": True,
        "result_data_used": False,
        "web_data_used": False,
        "missing_optional": ["ex_image"] if ex_image is None else [],
    }
    return payload, audit
