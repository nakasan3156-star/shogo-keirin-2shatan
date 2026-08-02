"""KEIRIN.JP公式5PDFを正規化する競輪2車単アダプター。

必須入力:
1. 基本情報・並び予想
2. 直近成績①（短期表示）
3. 直近成績②（詳細表示）
4. 着度数・H・S回数
5. 2車単オッズ
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    from .keirin_jp_6pdf_adapter import (
        _form_feature,
        _name_pattern,
        _norm,
        _source_guard,
        parse_basic_text,
        parse_hs_text,
    )
    from .keirin_jp_pdf_adapter import (
        _keirin_jp_odds_status,
        _parse_keirin_jp_odds_pdf,
    )
    from .keirin_pdf_adapter import (
        PdfInputError,
        _check_pdf,
        _extract_text,
        _grade,
        _identity,
        _parse_lines,
        _pre_race_status,
    )
except ImportError:
    from keirin_jp_6pdf_adapter import (
        _form_feature,
        _name_pattern,
        _norm,
        _source_guard,
        parse_basic_text,
        parse_hs_text,
    )
    from keirin_jp_pdf_adapter import (
        _keirin_jp_odds_status,
        _parse_keirin_jp_odds_pdf,
    )
    from keirin_pdf_adapter import (
        PdfInputError,
        _check_pdf,
        _extract_text,
        _grade,
        _identity,
        _parse_lines,
        _pre_race_status,
    )


REQUIRED_UPLOADS = (
    "basic_pdf",
    "recent_short_pdf",
    "recent_detail_pdf",
    "hs_pdf",
    "odds_pdf",
)

SOURCE_LABELS = {
    "basic_pdf": "KEIRIN.JP 基本情報・並び予想PDF",
    "recent_short_pdf": "KEIRIN.JP 直近成績①PDF",
    "recent_detail_pdf": "KEIRIN.JP 直近成績②PDF",
    "hs_pdf": "KEIRIN.JP 着度数・H・S回数PDF",
    "odds_pdf": "KEIRIN.JP 2車単オッズPDF",
}


def _require_role(text: str, key: str) -> None:
    if key == "basic_pdf" and "競走得点" not in text:
        raise PdfInputError(
            "SOURCE_ROLE_MISMATCH",
            "基本情報・並び予想PDFには競走得点表が必要です",
            [SOURCE_LABELS[key]],
        )
    if key in {"recent_short_pdf", "recent_detail_pdf"} and not (
        "今回成績" in text and "前回成績" in text
    ):
        raise PdfInputError(
            "SOURCE_ROLE_MISMATCH",
            f"{SOURCE_LABELS[key]}は直近成績画面から保存してください",
            [SOURCE_LABELS[key]],
        )
    if key == "hs_pdf" and not (
        re.search(r"着\s*外", text) and re.search(r"H\s*S", text)
    ):
        raise PdfInputError(
            "SOURCE_ROLE_MISMATCH",
            "着度数・H・S回数PDFの表を確認できません",
            [SOURCE_LABELS[key]],
        )
    if key == "odds_pdf" and "2車単オッズ" not in text:
        raise PdfInputError(
            "SOURCE_ROLE_MISMATCH",
            "2車単オッズPDFを選択してください",
            [SOURCE_LABELS[key]],
        )


def _validate_identities(
    texts: dict[str, str],
    paths: dict[str, Path],
) -> dict[str, str | int | None]:
    identities = {
        key: _identity(text, paths[key].name)
        for key, text in texts.items()
    }
    if any(
        item["venue"] is None or item["race"] is None
        for item in identities.values()
    ):
        raise PdfInputError(
            "RACE_ID_NOT_FOUND",
            "5PDFの開催場・レース番号をすべて確認できません",
        )
    venue_races = {
        (item["venue"], item["race"])
        for item in identities.values()
    }
    if len(venue_races) != 1:
        raise PdfInputError("RACE_MISMATCH", "5PDFが同じレースではありません")
    dates = {
        item["date"] for item in identities.values()
        if item["date"] is not None
    }
    if len(dates) > 1:
        raise PdfInputError("RACE_MISMATCH", "5PDFの日付が一致しません")
    venue, race = next(iter(venue_races))
    return {"venue": venue, "race": race, "date": next(iter(dates), None)}


def _require_all_riders(
    text: str,
    riders: list[dict[str, Any]],
    label: str,
) -> None:
    missing = [
        str(rider["name"])
        for rider in riders
        if not _name_pattern(str(rider["name"])).search(text)
    ]
    if missing:
        raise PdfInputError(
            "RECENT_RIDER_MISSING",
            f"{label}で全選手を確認できません: {', '.join(missing[:4])}",
            [label],
        )


def normalize_five_pdfs(
    basic_pdf: str | Path,
    recent_short_pdf: str | Path,
    recent_detail_pdf: str | Path,
    hs_pdf: str | Path,
    odds_pdf: str | Path,
    ex_image: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "basic_pdf": _check_pdf(basic_pdf, "basic_pdf"),
        "recent_short_pdf": _check_pdf(recent_short_pdf, "recent_short_pdf"),
        "recent_detail_pdf": _check_pdf(recent_detail_pdf, "recent_detail_pdf"),
        "hs_pdf": _check_pdf(hs_pdf, "hs_pdf"),
        "odds_pdf": _check_pdf(odds_pdf, "odds_pdf"),
    }
    texts = {
        key: _norm(_extract_text(path, key))
        for key, path in paths.items()
    }
    for key, text in texts.items():
        _source_guard(text, SOURCE_LABELS[key])
        _require_role(text, key)

    identity = _validate_identities(texts, paths)
    race_number = int(identity["race"])
    pre_race_status = {
        key: (
            _keirin_jp_odds_status(text, race_number)
            if key == "odds_pdf"
            else _pre_race_status(text, race_number, key)
        )
        for key, text in texts.items()
    }

    riders = parse_basic_text(texts["basic_pdf"])
    bikes = [int(rider["bike"]) for rider in riders]
    _require_all_riders(
        texts["recent_short_pdf"], riders, SOURCE_LABELS["recent_short_pdf"]
    )
    _require_all_riders(
        texts["recent_detail_pdf"], riders, SOURCE_LABELS["recent_detail_pdf"]
    )

    hs = parse_hs_text(texts["hs_pdf"], bikes)
    recent_short = _form_feature(texts["recent_short_pdf"], riders)
    recent_detail = _form_feature(texts["recent_detail_pdf"], riders)

    for rider in riders:
        bike = int(rider["bike"])
        short_value = float(recent_short[bike])
        detail_value = float(recent_detail[bike])
        recent_value = 0.5 * short_value + 0.5 * detail_value
        rider.update(hs[bike])
        rider["recent_short_form"] = short_value
        rider["recent_detail_form"] = detail_value
        rider["recent_form"] = recent_value
        # 2つの直近成績を同率でまとめ、既存能力モデルの入力へ反映する。
        rider["score"] = float(rider["score"]) + 0.60 * recent_value
        rider["win_rate"] = max(
            0.0,
            min(100.0, float(rider["win_rate"]) + 1.50 * recent_value),
        )

    lines = _parse_lines(paths["basic_pdf"], bikes)
    odds = _parse_keirin_jp_odds_pdf(
        paths["odds_pdf"], texts["odds_pdf"], bikes
    )

    source_files = {
        "racecard_pdf": paths["basic_pdf"].name,
        "hs_pdf": paths["hs_pdf"].name,
        "odds_pdf": paths["odds_pdf"].name,
        **{key: path.name for key, path in paths.items()},
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
        "required_pdf_count": 5,
        "required_inputs": list(REQUIRED_UPLOADS),
        "rider_count": len(riders),
        "odds_count": len(bikes) * (len(bikes) - 1),
        "lines": lines,
        "pre_race_status": pre_race_status,
        "feature_usage": {
            "basic_pdf": [
                "score", "escape", "makuri", "sashi", "mark", "B", "lines"
            ],
            "recent_short_pdf": "recent_short_form",
            "recent_detail_pdf": "recent_detail_form",
            "recent_combination": "equal_average",
            "hs_pdf": [
                "first", "second", "third", "out", "H", "S", "win_rate"
            ],
            "odds_pdf": "all_ordered_pair_odds",
        },
        "rider_features": [
            {
                "bike": int(rider["bike"]),
                "name": rider["name"],
                "recent_short_form": float(rider["recent_short_form"]),
                "recent_detail_form": float(rider["recent_detail_form"]),
                "recent_form": float(rider["recent_form"]),
            }
            for rider in riders
        ],
        "result_data_used": False,
        "web_data_used": False,
        "missing_optional": missing_optional
        + ["wind_mps", "temperature_c", "bank_type"],
    }
    return payload, audit
