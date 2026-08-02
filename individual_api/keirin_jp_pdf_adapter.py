"""KEIRIN.JPの2車単オッズPDFを使う3PDF正規化アダプター。"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

try:
    from .keirin_pdf_adapter import (
        PdfInputError,
        _check_pdf,
        _extract_text,
        _grade,
        _identity,
        _parse_hs_counts,
        _parse_lines,
        _parse_riders,
        _pre_race_status,
    )
except ImportError:  # 直接スクリプトとして実行する場合
    from keirin_pdf_adapter import (
        PdfInputError,
        _check_pdf,
        _extract_text,
        _grade,
        _identity,
        _parse_hs_counts,
        _parse_lines,
        _parse_riders,
        _pre_race_status,
    )


_HEADER_PATTERN = re.compile(r"([1-9])番車")
_INTEGER_PATTERN = re.compile(r"[1-9]")
_ODDS_PATTERN = re.compile(r"\d{1,5}(?:,\d{3})*\.\d+")


def _word_text(word: dict[str, Any]) -> str:
    return unicodedata.normalize("NFKC", str(word.get("text", ""))).strip()


def _center_x(word: dict[str, Any]) -> float:
    return (float(word["x0"]) + float(word["x1"])) / 2.0


def _cluster_by_top(
    words: Iterable[dict[str, Any]],
    tolerance: float = 4.0,
) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
        top = float(word["top"])
        if not clusters:
            clusters.append([word])
            continue
        cluster_top = sum(float(item["top"]) for item in clusters[-1]) / len(clusters[-1])
        if abs(top - cluster_top) <= tolerance:
            clusters[-1].append(word)
        else:
            clusters.append([word])
    return clusters


def _find_headers(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    headers: list[dict[str, Any]] = []
    for word in words:
        match = _HEADER_PATTERN.fullmatch(_word_text(word))
        if match:
            headers.append({**word, "first_bike": int(match.group(1))})

    # PDFの文字分割で「1」と「番車」に分かれる場合も許容する。
    for suffix in [word for word in words if _word_text(word) == "番車"]:
        candidates = [
            word
            for word in words
            if _INTEGER_PATTERN.fullmatch(_word_text(word))
            and float(word["x1"]) <= float(suffix["x0"]) + 3
            and 0 <= float(suffix["x0"]) - float(word["x1"]) <= 24
            and abs(float(word["top"]) - float(suffix["top"])) <= 4
        ]
        if not candidates:
            continue
        number_word = max(candidates, key=lambda item: float(item["x1"]))
        first_bike = int(_word_text(number_word))
        if any(
            item["first_bike"] == first_bike
            and abs(float(item["top"]) - float(suffix["top"])) <= 4
            for item in headers
        ):
            continue
        headers.append(
            {
                "text": f"{first_bike}番車",
                "x0": float(number_word["x0"]),
                "x1": float(suffix["x1"]),
                "top": min(float(number_word["top"]), float(suffix["top"])),
                "bottom": max(float(number_word["bottom"]), float(suffix["bottom"])),
                "first_bike": first_bike,
            }
        )
    return headers


def _store_pair(
    pairs: dict[tuple[int, int], float],
    first: int,
    second: int,
    odds: float,
) -> None:
    pair = (first, second)
    previous = pairs.get(pair)
    if previous is not None and abs(previous - odds) > 1e-9:
        raise PdfInputError(
            "ODDS_DUPLICATE_CONFLICT",
            f"2車単{first}-{second}のオッズが重複し、値が一致しません",
        )
    pairs[pair] = odds


def _parse_section(
    words: list[dict[str, Any]],
    page_width: float,
    active: dict[int, int],
    y_min: float,
    y_max: float,
    bikes: list[int],
    pairs: dict[tuple[int, int], float],
) -> None:
    if not active or y_max <= y_min:
        return
    bike_set = set(bikes)
    slot_width = page_width / 3.0
    for slot, first in active.items():
        x_min, x_max = slot * slot_width, (slot + 1) * slot_width
        section_words = [
            word
            for word in words
            if y_min <= float(word["top"]) < y_max
            and x_min <= _center_x(word) < x_max
        ]
        for row in _cluster_by_top(section_words, tolerance=3.5):
            integers = [
                word
                for word in row
                if _INTEGER_PATTERN.fullmatch(_word_text(word))
                and int(_word_text(word)) in bike_set
            ]
            odds_words = [
                word for word in row if _ODDS_PATTERN.fullmatch(_word_text(word))
            ]
            for odds_word in odds_words:
                left_numbers = [
                    word
                    for word in integers
                    if float(word["x1"]) <= float(odds_word["x0"]) + 3
                ]
                if not left_numbers:
                    continue
                second_word = max(left_numbers, key=lambda item: float(item["x1"]))
                second = int(_word_text(second_word))
                if second == first:
                    continue
                odds = float(_word_text(odds_word).replace(",", ""))
                _store_pair(pairs, first, second, odds)


def _parse_keirin_jp_odds_pages(
    pages: Iterable[tuple[float, float, list[dict[str, Any]]]],
    bikes: list[int],
) -> list[list[float | None]]:
    pairs: dict[tuple[int, int], float] = {}
    active: dict[int, int] = {}
    saw_header = False

    for page_width, page_height, raw_words in pages:
        words = [
            {**word, "text": _word_text(word)}
            for word in raw_words
            if _word_text(word)
        ]
        header_rows = _cluster_by_top(_find_headers(words), tolerance=5.0)
        cursor = 0.0
        for header_row in header_rows:
            header_top = min(float(item["top"]) for item in header_row)
            _parse_section(
                words,
                page_width,
                active,
                cursor,
                max(cursor, header_top - 3),
                bikes,
                pairs,
            )
            next_active: dict[int, int] = {}
            for header in header_row:
                slot = min(2, max(0, int(_center_x(header) / (page_width / 3.0))))
                first = int(header["first_bike"])
                if slot in next_active and next_active[slot] != first:
                    raise PdfInputError(
                        "ODDS_HEADER_CONFLICT",
                        "KEIRIN.JPオッズ表の列見出しを一意に読めません",
                    )
                next_active[slot] = first
            active = next_active
            saw_header = saw_header or bool(active)
            cursor = max(float(item.get("bottom", item["top"])) for item in header_row) + 2

        _parse_section(
            words,
            page_width,
            active,
            cursor,
            page_height,
            bikes,
            pairs,
        )

    if not saw_header:
        raise PdfInputError(
            "KEIRIN_JP_ODDS_HEADER_NOT_FOUND",
            "KEIRIN.JPの『○番車 全選択』見出しを取得できません",
        )

    expected = {(first, second) for first in bikes for second in bikes if first != second}
    missing = sorted(expected - set(pairs))
    extra = sorted(set(pairs) - expected)
    if missing or extra:
        missing_text = ", ".join(f"{a}-{b}" for a, b in missing[:8])
        raise PdfInputError(
            "ODDS_PARSE_FAILED",
            f"2車単は{len(expected)}通り必要ですが{len(pairs)}通り取得。"
            f"欠損例={missing_text or 'なし'}",
        )

    return [
        [None if first == second else pairs[(first, second)] for second in bikes]
        for first in bikes
    ]


def _parse_keirin_jp_odds_pdf(
    odds_pdf: Path,
    odds_text: str,
    bikes: list[int],
) -> list[list[float | None]]:
    normalized = unicodedata.normalize("NFKC", odds_text)
    if "2車単オッズ" not in normalized:
        raise PdfInputError(
            "ODDS_SOURCE_MISMATCH",
            "KEIRIN.JPの2車単オッズPDFを選択してください",
        )
    if "COPYRIGHT JKA" not in normalized and "開催発売案内" not in normalized:
        raise PdfInputError(
            "ODDS_SOURCE_MISMATCH",
            "3番目はKEIRIN.JP公式の2車単オッズPDFが必要です",
        )

    try:
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        import pdfplumber

        pages: list[tuple[float, float, list[dict[str, Any]]]] = []
        with pdfplumber.open(odds_pdf) as document:
            for page in document.pages:
                pages.append(
                    (
                        float(page.width),
                        float(page.height),
                        page.extract_words(
                            x_tolerance=2,
                            y_tolerance=3,
                            keep_blank_chars=False,
                        ),
                    )
                )
        return _parse_keirin_jp_odds_pages(pages, bikes)
    except PdfInputError:
        raise
    except Exception as exc:
        raise PdfInputError(
            "ODDS_PARSE_FAILED",
            "KEIRIN.JPの2車単オッズ表を解析できません",
        ) from exc


def _keirin_jp_odds_status(text: str, race_number: int) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")
    if re.search(fr"(?:^|\s){race_number}\s*R\s*は締め切りました", normalized):
        raise PdfInputError(
            "POST_RACE_SOURCE",
            "odds_pdfは締切後の資料です。発売中のKEIRIN.JPオッズPDFを使用してください",
        )
    current = re.search(r"\d{1,2}:\d{2}\s*現在\s*(?:オッズ更新)?", normalized)
    if current:
        return re.sub(r"\s+", " ", current.group(0)).strip()
    return _pre_race_status(normalized, race_number, "odds_pdf")


def normalize_pdfs(
    racecard_pdf: str | Path,
    hs_pdf: str | Path,
    odds_pdf: str | Path,
    ex_image: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """出走表はnetkeirin、H/Sと2車単オッズはKEIRIN.JPから正規化する。"""
    paths = {
        "racecard_pdf": _check_pdf(racecard_pdf, "racecard_pdf"),
        "hs_pdf": _check_pdf(hs_pdf, "hs_pdf"),
        "odds_pdf": _check_pdf(odds_pdf, "odds_pdf"),
    }
    racecard_text = unicodedata.normalize(
        "NFKC", _extract_text(paths["racecard_pdf"], "racecard_pdf")
    ).replace("\u00a0", " ")
    hs_text = unicodedata.normalize(
        "NFKC", _extract_text(paths["hs_pdf"], "hs_pdf")
    ).replace("\u00a0", " ")
    odds_text = unicodedata.normalize(
        "NFKC", _extract_text(paths["odds_pdf"], "odds_pdf")
    ).replace("\u00a0", " ")

    identities = {
        "racecard_pdf": _identity(racecard_text, paths["racecard_pdf"].name),
        "hs_pdf": _identity(hs_text, paths["hs_pdf"].name),
        "odds_pdf": _identity(odds_text, paths["odds_pdf"].name),
    }
    venue_races = {
        (item["venue"], item["race"])
        for item in identities.values()
        if item["venue"] is not None and item["race"] is not None
    }
    if len(venue_races) != 1 or any(
        item["venue"] is None or item["race"] is None
        for item in identities.values()
    ):
        raise PdfInputError(
            "RACE_ID_NOT_FOUND",
            "3PDFの開催場・レース番号を確認できません",
        )
    dates = {item["date"] for item in identities.values() if item["date"] is not None}
    if len(dates) > 1:
        raise PdfInputError("RACE_MISMATCH", "3PDFの日付が一致しません")

    venue, race_value = next(iter(venue_races))
    identity = {
        "venue": venue,
        "race": race_value,
        "date": next(iter(dates), None),
    }
    race_number = int(identity["race"])
    source_status = {
        "racecard_pdf": _pre_race_status(
            racecard_text,
            race_number,
            "racecard_pdf",
        ),
        "odds_pdf": _keirin_jp_odds_status(odds_text, race_number),
    }

    stat_bikes = [
        int(match.group(2))
        for match in re.finditer(
            r"^\s*([1-6])\s*([1-9])\s+[0-9]+\.[0-9]+\s+(?:逃|追|両)",
            racecard_text,
            re.MULTILINE,
        )
    ]
    hs_rows = _parse_hs_counts(paths["hs_pdf"], stat_bikes)
    riders = _parse_riders(racecard_text, hs_rows)
    bikes = [int(rider["bike"]) for rider in riders]
    lines = _parse_lines(paths["hs_pdf"], bikes)
    odds = _parse_keirin_jp_odds_pdf(paths["odds_pdf"], odds_text, bikes)

    source_files = {key: path.name for key, path in paths.items()}
    missing_optional: list[str] = []
    if ex_image is not None:
        image_path = Path(ex_image)
        try:
            if image_path.stat().st_size <= 0:
                raise OSError
        except OSError:
            missing_optional.append("ex_image")
        else:
            source_files["ex_image"] = image_path.name
    else:
        missing_optional.append("ex_image")

    payload = {
        "grade": _grade(racecard_text),
        "source_files": source_files,
        "riders": riders,
        "lines": lines,
        "odds": odds,
        "conditions": {},
    }
    audit = {
        "race": identity,
        "rider_count": len(riders),
        "odds_count": len(bikes) * (len(bikes) - 1),
        "odds_source": "KEIRIN.JP",
        "odds_parser": "coordinate_grouped_2shatan",
        "lines": lines,
        "missing_optional": missing_optional
        + ["wind_mps", "temperature_c", "bank_type"],
        "result_data_used": False,
        "web_data_used": False,
        "pre_race_status": source_status,
        "compatibility_warnings": [
            f"{rider['bike']}番: 氏名または地区を取得できず数値項目のみ使用"
            for rider in riders
            if rider["name"].endswith("氏名未取得）")
            or rider["region"] == "未取得"
        ],
    }
    return payload, audit
