"""KEIRIN.JP公式3PDFを正規化する競輪2車単アダプター。"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

try:
    from .keirin_jp_pdf_adapter import _keirin_jp_odds_status, _parse_keirin_jp_odds_pdf
    from .keirin_jp_6pdf_adapter import _source_guard, parse_hs_text
    from .keirin_pdf_adapter import (
        PREFECTURE_TO_REGION, PdfInputError, _check_pdf, _extract_text, _grade,
        _identity, _normalize_prefecture, _parse_lines, _pre_race_status,
    )
except ImportError:
    from keirin_jp_pdf_adapter import _keirin_jp_odds_status, _parse_keirin_jp_odds_pdf
    from keirin_jp_6pdf_adapter import _source_guard, parse_hs_text
    from keirin_pdf_adapter import (
        PREFECTURE_TO_REGION, PdfInputError, _check_pdf, _extract_text, _grade,
        _identity, _normalize_prefecture, _parse_lines, _pre_race_status,
    )

REQUIRED_UPLOADS = ("basic_pdf", "hs_pdf", "odds_pdf")


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\u00a0", " ")


def parse_basic_text(text: str) -> list[dict[str, Any]]:
    """KEIRIN.JP基本情報を行単位で読む。名前が車番と同じ行でも別行でも対応。"""
    lines = [line.strip() for line in _norm(text).splitlines()]
    riders: list[dict[str, Any]] = []
    profile_re = re.compile(r"^([^/]+)/([ASL]\d)([ASL]\d)/(逃|追|両)$")
    stats_re = re.compile(r"^([0-9]{2,3}\.[0-9]{1,2})\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)$")

    for i, line in enumerate(lines):
        profile = profile_re.match(re.sub(r"\s+", "", line))
        if not profile:
            continue
        # profile直前の非空行は「車番+氏名」または氏名だけ。
        j = i - 1
        while j >= 0 and not lines[j]:
            j -= 1
        if j < 0:
            continue
        name_line = lines[j]
        bike: int | None = None
        name = ""
        inline = re.match(r"^(?:[1-6]\s+)?([1-9])\s+(.+)$", name_line)
        if inline:
            bike = int(inline.group(1)); name = inline.group(2)
        else:
            name = name_line
            k = j - 1
            while k >= 0 and not lines[k]:
                k -= 1
            if k >= 0:
                bike_match = re.match(r"^(?:[1-6]\s+)?([1-9])$", lines[k])
                if bike_match:
                    bike = int(bike_match.group(1))
        if bike is None:
            continue
        name = re.sub(r"\s+|追加$", "", name)

        n = i + 1
        while n < len(lines) and not lines[n]:
            n += 1
        if n >= len(lines):
            continue
        stats = stats_re.match(lines[n])
        if not stats:
            continue
        raw_prefecture = re.sub(r"\s+", "", profile.group(1))
        prefecture = _normalize_prefecture(raw_prefecture)
        riders.append({
            "bike": bike,
            "name": name,
            "region": PREFECTURE_TO_REGION[prefecture] if prefecture else "未取得",
            "prefecture_raw": raw_prefecture or "未取得",
            "style": profile.group(4),
            "score": float(stats.group(1)),
            "escape": int(stats.group(2)),
            "makuri": int(stats.group(3)),
            "sashi": int(stats.group(4)),
            "mark": int(stats.group(5)),
            "B": int(stats.group(6)),
        })

    unique = {int(r["bike"]): r for r in riders}
    riders = [unique[b] for b in sorted(unique)]
    if len(riders) not in {5, 6, 7, 8, 9}:
        raise PdfInputError("BASIC_PARSE_FAILED", f"基本情報の選手取得に失敗しました（{len(riders)}人）")
    if [r["bike"] for r in riders] != list(range(1, len(riders) + 1)):
        raise PdfInputError("BIKE_SEQUENCE_ERROR", "基本情報の車番が連番ではありません")
    return riders


def _validate_identity(texts: dict[str, str], paths: dict[str, Path]) -> dict[str, Any]:
    ids = {key: _identity(text, paths[key].name) for key, text in texts.items()}
    pairs = {(v["venue"], v["race"]) for v in ids.values() if v["venue"] is not None and v["race"] is not None}
    if len(pairs) != 1:
        raise PdfInputError("RACE_MISMATCH", "3PDFが同じレースではありません")
    dates = {v["date"] for v in ids.values() if v["date"] is not None}
    if len(dates) > 1:
        raise PdfInputError("RACE_MISMATCH", "3PDFの日付が一致しません")
    venue, race = next(iter(pairs))
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
    texts = {key: _norm(_extract_text(path, key)) for key, path in paths.items()}
    for key, text in texts.items():
        _source_guard(text, key)
    identity = _validate_identity(texts, paths)
    race_number = int(identity["race"])
    riders = parse_basic_text(texts["basic_pdf"])
    bikes = [int(r["bike"]) for r in riders]
    hs = parse_hs_text(texts["hs_pdf"], bikes)
    for rider in riders:
        rider.update(hs[int(rider["bike"])])
    lines = _parse_lines(paths["basic_pdf"], bikes)
    odds = _parse_keirin_jp_odds_pdf(paths["odds_pdf"], texts["odds_pdf"], bikes)
    payload = {
        "grade": _grade(texts["basic_pdf"]),
        "source_files": {
            "racecard_pdf": paths["basic_pdf"].name,
            "hs_pdf": paths["hs_pdf"].name,
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
        "rider_count": len(riders),
        "odds_count": len(bikes) * (len(bikes) - 1),
        "lines": lines,
        "pre_race_status": {
            "basic_pdf": _pre_race_status(texts["basic_pdf"], race_number, "basic_pdf"),
            "hs_pdf": _pre_race_status(texts["hs_pdf"], race_number, "hs_pdf"),
            "odds_pdf": _keirin_jp_odds_status(texts["odds_pdf"], race_number),
        },
        "result_data_used": False,
        "web_data_used": False,
        "missing_optional": ["ex_image"] if ex_image is None else [],
    }
    return payload, audit
