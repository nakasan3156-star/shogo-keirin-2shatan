"""KEIRIN.JPオッズPDFを発売中・締切後の両方で安定解析する実運用パッチ。"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any


def _clean_word_text(word: dict[str, Any]) -> str:
    """NFKC後にPDF由来の不可視制御文字を除去する。"""
    value = unicodedata.normalize("NFKC", str(word.get("text", "")))
    value = "".join(
        character
        for character in value
        if not unicodedata.category(character).startswith("C")
    )
    return value.strip()


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("C"))
    return re.sub(r"\s+", "", text)


def _matrix_from_closed_tables(
    tables: list[list[list[object | None]]],
    bikes: list[int],
) -> list[list[float | None]]:
    """締切後PDFの3列組 [1着, 2着, オッズ] から全組み合わせを復元する。"""
    bike_set = set(bikes)
    pairs: dict[tuple[int, int], float] = {}

    for table in tables:
        if not table:
            continue
        width = max((len(row) for row in table if row), default=0)
        if width < 3:
            continue
        groups = width // 3
        active_first: list[int | None] = [None] * groups
        for raw_row in table:
            row = list(raw_row or []) + [None] * (width - len(raw_row or []))
            for group in range(groups):
                first_text = _cell_text(row[group * 3])
                second_text = _cell_text(row[group * 3 + 1])
                odds_text = _cell_text(row[group * 3 + 2]).replace(",", "")

                if first_text.isdigit() and int(first_text) in bike_set:
                    active_first[group] = int(first_text)
                first = active_first[group]
                if first is None or not second_text.isdigit():
                    continue
                second = int(second_text)
                if second not in bike_set or second == first:
                    continue
                if not re.fullmatch(r"\d+(?:\.\d+)?", odds_text):
                    continue
                odds = float(odds_text)
                if odds <= 0:
                    continue
                pair = (first, second)
                previous = pairs.get(pair)
                if previous is not None and abs(previous - odds) > 1e-9:
                    continue
                pairs[pair] = odds

    expected = {(first, second) for first in bikes for second in bikes if first != second}
    if set(pairs) != expected:
        missing = sorted(expected - set(pairs))
        raise ValueError(
            f"closed odds incomplete: {len(pairs)}/{len(expected)}, "
            f"missing={missing[:8]}"
        )
    return [
        [None if first == second else pairs[(first, second)] for second in bikes]
        for first in bikes
    ]


def _parse_closed_odds_pdf(path: Path, bikes: list[int]) -> list[list[float | None]]:
    import pdfplumber

    tables: list[list[list[object | None]]] = []
    with pdfplumber.open(path) as document:
        for page in document.pages:
            tables.extend(page.extract_tables() or [])
    return _matrix_from_closed_tables(tables, bikes)


def install_odds_parser_fix() -> None:
    """制御文字、締切後表、締切後ステータスを既存パーサーへ適用する。"""
    try:
        from . import keirin_jp_pdf_adapter as adapter
        from .keirin_pdf_adapter import PdfInputError
    except ImportError:
        import keirin_jp_pdf_adapter as adapter
        from keirin_pdf_adapter import PdfInputError

    if getattr(adapter, "_CLOSED_ODDS_FIX_INSTALLED", False):
        return

    adapter._word_text = _clean_word_text
    adapter._HEADER_PATTERN = re.compile(r"([1-9])番車(?:.*全選択)?")

    original_parser = adapter._parse_keirin_jp_odds_pdf
    original_status = adapter._keirin_jp_odds_status

    def parse_with_fallback(
        odds_pdf: Path,
        odds_text: str,
        bikes: list[int],
    ) -> list[list[float | None]]:
        try:
            return original_parser(odds_pdf, odds_text, bikes)
        except PdfInputError as original_error:
            try:
                return _parse_closed_odds_pdf(Path(odds_pdf), bikes)
            except Exception:
                raise original_error

    def allow_closed_status(text: str, race_number: int) -> str:
        normalized = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")
        closed = re.search(
            fr"(?:^|\s){race_number}\s*R\s*は締め切りました[。.]?",
            normalized,
        )
        if closed:
            time_match = re.search(r"\d{1,2}:\d{2}\s*現在", normalized)
            return (
                "締切後オッズ "
                + (re.sub(r"\s+", " ", time_match.group(0)).strip() if time_match else "時刻未取得")
            )
        return original_status(text, race_number)

    adapter._parse_keirin_jp_odds_pdf = parse_with_fallback
    adapter._keirin_jp_odds_status = allow_closed_status
    adapter._CONTROL_CHAR_FIX_INSTALLED = True
    adapter._CLOSED_ODDS_FIX_INSTALLED = True
