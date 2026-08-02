"""KEIRIN.JP公式6PDFを正規化する競輪2車単アダプター。

必須: 基本情報・直近成績・対戦成績・当場成績・着度数/H/S回数・2車単オッズ。
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

try:
    from .keirin_jp_pdf_adapter import _keirin_jp_odds_status, _parse_keirin_jp_odds_pdf
    from .keirin_pdf_adapter import (
        PREFECTURE_TO_REGION, PdfInputError, _check_pdf, _extract_text, _grade,
        _identity, _normalize_prefecture, _parse_lines, _pre_race_status,
    )
except ImportError:
    from keirin_jp_pdf_adapter import _keirin_jp_odds_status, _parse_keirin_jp_odds_pdf
    from keirin_pdf_adapter import (
        PREFECTURE_TO_REGION, PdfInputError, _check_pdf, _extract_text, _grade,
        _identity, _normalize_prefecture, _parse_lines, _pre_race_status,
    )

REQUIRED_UPLOADS = (
    "basic_pdf", "recent_pdf", "matchup_pdf", "track_pdf", "hs_pdf", "odds_pdf",
)
SOURCE_LABELS = {
    "basic_pdf": "KEIRIN.JP 基本情報PDF",
    "recent_pdf": "KEIRIN.JP 直近成績PDF",
    "matchup_pdf": "KEIRIN.JP 対戦成績PDF",
    "track_pdf": "KEIRIN.JP 当場成績PDF",
    "hs_pdf": "KEIRIN.JP 着度数・H・S回数PDF",
    "odds_pdf": "KEIRIN.JP 2車単オッズPDF",
}
_PAIR_RECORD = re.compile(r"(?<!\d)(\d{1,2})\s*-\s*(\d{1,2})(?!\d)")


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\u00a0", " ")


def _source_guard(text: str, label: str) -> None:
    if "COPYRIGHT JKA" not in text and "開催発売案内" not in text:
        raise PdfInputError("SOURCE_MISMATCH", f"{label}はKEIRIN.JP公式PDFではありません", [label])


def _table_body(text: str, markers: tuple[str, ...]) -> str:
    starts = [text.find(marker) for marker in markers if text.find(marker) >= 0]
    start = min(starts) if starts else 0
    ends = [text.find(marker, start + 1) for marker in ("誘導", "COPYRIGHT JKA", "出走表\n直近4ヶ月")]
    ends = [value for value in ends if value >= 0]
    return text[start:(min(ends) if ends else len(text))]


def parse_basic_text(text: str) -> list[dict[str, Any]]:
    """基本情報から得点、決まり手、B、脚質、氏名、地区を読む。"""
    body = _table_body(_norm(text), ("競走得点", "枠\n番\n車\n番"))
    pattern = re.compile(
        r"(?ms)(?:^|\n)\s*(?:[1-6]\s+)?([1-9])(?:[ \t]+([^\n]+))?\s*\n"
        r"(?:\s*([^\n]+?)\s*\n)?\s*([^/\n]+)/(?:[ASL]\d){1,2}/(逃|追|両)\s*\n"
        r"\s*([0-9]{2,3}\.[0-9]{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+"
        r"(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})"
    )
    riders: list[dict[str, Any]] = []
    for m in pattern.finditer(body):
        raw_name = re.sub(r"\s+", "", m.group(2) or m.group(3) or "")
        raw_prefecture = re.sub(r"\s+", "", m.group(4))
        prefecture = _normalize_prefecture(raw_prefecture)
        riders.append({
            "bike": int(m.group(1)), "name": raw_name,
            "region": PREFECTURE_TO_REGION[prefecture] if prefecture else "未取得",
            "prefecture_raw": raw_prefecture or "未取得", "style": m.group(5),
            "score": float(m.group(6)), "escape": int(m.group(7)),
            "makuri": int(m.group(8)), "sashi": int(m.group(9)),
            "mark": int(m.group(10)), "B": int(m.group(11)),
        })
    riders.sort(key=lambda item: item["bike"])
    if len(riders) not in {5, 6, 7, 8, 9}:
        raise PdfInputError("BASIC_PARSE_FAILED", f"基本情報の選手取得に失敗しました（{len(riders)}人）")
    if [item["bike"] for item in riders] != list(range(1, len(riders) + 1)):
        raise PdfInputError("BIKE_SEQUENCE_ERROR", "基本情報の車番が連番ではありません")
    return riders


def parse_hs_text(text: str, bikes: list[int]) -> dict[int, dict[str, int | float]]:
    """着度数・H・S回数を車番別に読む。"""
    body = _table_body(_norm(text), ("1\n着", "着\n外\nH S", "枠\n番\n車\n番"))
    pattern = re.compile(
        r"(?ms)(?:^|\n)\s*(?:[1-6]\s+)?([1-9])(?:[ \t]+([^\n]+))?\s*\n"
        r"(?:\s*([^\n]+?)\s*\n)?\s*[^/\n]+/(?:[ASL]\d){1,2}/(?:逃|追|両)\s*\n"
        r"\s*(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})"
    )
    rows: dict[int, dict[str, int | float]] = {}
    for m in pattern.finditer(body):
        bike = int(m.group(1)); first, second, third, out, h_count, s_count = map(int, m.groups()[3:9])
        total = first + second + third + out
        rows[bike] = {
            "first": first, "second": second, "third": third, "out": out,
            "H": h_count, "S": s_count,
            "win_rate": 100.0 * first / total if total else 0.0,
            "quinella_rate": 100.0 * (first + second) / total if total else 0.0,
        }
    if set(rows) != set(bikes):
        raise PdfInputError("HS_PARSE_FAILED", f"H・S回数は{len(bikes)}人必要ですが{len(rows)}人取得しました")
    return rows


def _name_pattern(name: str) -> re.Pattern[str]:
    return re.compile(r"\s*".join(re.escape(char) for char in re.sub(r"\s+", "", name)))


def _best_name_block(text: str, name: str, all_names: list[str]) -> str:
    own = _name_pattern(name); candidates: list[str] = []
    for match in own.finditer(text):
        start, end = match.start(), min(len(text), match.start() + 1100)
        for other in all_names:
            if other == name:
                continue
            found = _name_pattern(other).search(text, match.end())
            if found and found.start() < end:
                end = found.start()
        candidates.append(text[start:end])
    return max(candidates, key=lambda block: (len(_PAIR_RECORD.findall(block)), len(block)), default="")


def _form_feature(text: str, riders: list[dict[str, Any]]) -> dict[int, float]:
    names = [str(item["name"]) for item in riders]; result: dict[int, float] = {}
    for rider in riders:
        block = _best_name_block(text, str(rider["name"]), names)
        if not block:
            result[int(rider["bike"])] = 0.0; continue
        b_count = len(re.findall(r"(?<![A-Z])B(?![A-Z])", block))
        starts = max(1, len(re.findall(r"(?<!\d)\d{1,2}/\d{1,2}(?!\d)", block)))
        adverse = len(re.findall(r"欠場|失格|落車|棄権|病気", block))
        positive = len(re.findall(r"(?:^|\s)[123](?:\s|$)", block))
        raw = 0.75 * b_count / starts + 0.08 * min(positive, 6) - 0.70 * adverse
        result[int(rider["bike"])] = max(-3.0, min(3.0, raw))
    return result


def _matchup_feature(text: str, riders: list[dict[str, Any]]) -> dict[int, float]:
    names = [str(item["name"]) for item in riders]; result: dict[int, float] = {}
    for rider in riders:
        records = [(int(a), int(b)) for a, b in _PAIR_RECORD.findall(_best_name_block(text, str(rider["name"]), names))]
        wins, losses = sum(a for a, _ in records), sum(b for _, b in records); total = wins + losses
        result[int(rider["bike"])] = (wins - losses) / total if total else 0.0
    return result


def _validate_identities(texts: dict[str, str], paths: dict[str, Path]) -> dict[str, str | int | None]:
    identities = {key: _identity(text, paths[key].name) for key, text in texts.items()}
    venue_races = {(item["venue"], item["race"]) for item in identities.values() if item["venue"] is not None and item["race"] is not None}
    if len(venue_races) != 1 or any(item["venue"] is None or item["race"] is None for item in identities.values()):
        raise PdfInputError("RACE_ID_NOT_FOUND", "6PDFの開催場・レース番号を確認できません")
    dates = {item["date"] for item in identities.values() if item["date"] is not None}
    if len(dates) > 1:
        raise PdfInputError("RACE_MISMATCH", "6PDFの日付が一致しません")
    venue, race = next(iter(venue_races))
    return {"venue": venue, "race": race, "date": next(iter(dates), None)}


def normalize_six_pdfs(
    basic_pdf: str | Path, recent_pdf: str | Path, matchup_pdf: str | Path,
    track_pdf: str | Path, hs_pdf: str | Path, odds_pdf: str | Path,
    ex_image: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "basic_pdf": _check_pdf(basic_pdf, "basic_pdf"),
        "recent_pdf": _check_pdf(recent_pdf, "recent_pdf"),
        "matchup_pdf": _check_pdf(matchup_pdf, "matchup_pdf"),
        "track_pdf": _check_pdf(track_pdf, "track_pdf"),
        "hs_pdf": _check_pdf(hs_pdf, "hs_pdf"),
        "odds_pdf": _check_pdf(odds_pdf, "odds_pdf"),
    }
    texts = {key: _norm(_extract_text(path, key)) for key, path in paths.items()}
    for key, text in texts.items():
        _source_guard(text, SOURCE_LABELS[key])
    identity = _validate_identities(texts, paths); race_number = int(identity["race"])
    status = {
        key: (_keirin_jp_odds_status(text, race_number) if key == "odds_pdf" else _pre_race_status(text, race_number, key))
        for key, text in texts.items()
    }
    riders = parse_basic_text(texts["basic_pdf"]); bikes = [int(item["bike"]) for item in riders]
    hs = parse_hs_text(texts["hs_pdf"], bikes)
    recent = _form_feature(texts["recent_pdf"], riders)
    track = _form_feature(texts["track_pdf"], riders)
    matchup = _matchup_feature(texts["matchup_pdf"], riders)
    for rider in riders:
        bike = int(rider["bike"]); rider.update(hs[bike])
        rider["recent_form"], rider["track_form"], rider["matchup_score"] = recent[bike], track[bike], matchup[bike]
        rider["score"] = float(rider["score"]) + 0.55 * recent[bike] + 0.30 * track[bike] + 0.25 * matchup[bike]
        rider["win_rate"] = max(0.0, min(100.0, float(rider["win_rate"]) + 1.50 * recent[bike] + 0.80 * track[bike] + matchup[bike]))
    lines = _parse_lines(paths["basic_pdf"], bikes)
    odds = _parse_keirin_jp_odds_pdf(paths["odds_pdf"], texts["odds_pdf"], bikes)
    source_files = {
        "racecard_pdf": paths["basic_pdf"].name, "hs_pdf": paths["hs_pdf"].name,
        "odds_pdf": paths["odds_pdf"].name, **{key: path.name for key, path in paths.items()},
    }
    missing_optional: list[str] = []
    if ex_image is None or not Path(ex_image).exists() or Path(ex_image).stat().st_size <= 0:
        missing_optional.append("ex_image")
    else:
        source_files["ex_image"] = Path(ex_image).name
    payload = {
        "grade": _grade(texts["basic_pdf"]), "source_files": source_files,
        "riders": riders, "lines": lines, "odds": odds, "conditions": {},
    }
    audit = {
        "race": identity, "input_source": "KEIRIN.JP_ONLY", "required_pdf_count": 6,
        "required_inputs": list(REQUIRED_UPLOADS), "rider_count": len(riders),
        "odds_count": len(bikes) * (len(bikes) - 1), "lines": lines,
        "pre_race_status": status,
        "feature_usage": {
            "basic_pdf": ["score", "escape", "makuri", "sashi", "mark", "B", "lines"],
            "recent_pdf": "recent_form", "matchup_pdf": "matchup_score",
            "track_pdf": "track_form", "hs_pdf": ["first", "second", "third", "out", "H", "S", "win_rate"],
            "odds_pdf": "all_ordered_pair_odds",
        },
        "rider_features": [{
            "bike": int(r["bike"]), "name": r["name"], "recent_form": float(r["recent_form"]),
            "matchup_score": float(r["matchup_score"]), "track_form": float(r["track_form"]),
        } for r in riders],
        "result_data_used": False, "web_data_used": False,
        "missing_optional": missing_optional + ["wind_mps", "temperature_c", "bank_type"],
    }
    return payload, audit
