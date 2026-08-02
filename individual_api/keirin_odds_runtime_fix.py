"""KEIRIN.JPオッズPDFの制御文字混入を吸収する実運用パッチ。"""
from __future__ import annotations

import re
import unicodedata
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


def install_odds_parser_fix() -> None:
    """既存オッズパーサーへ安全な見出し正規化を1回だけ適用する。"""
    try:
        from . import keirin_jp_pdf_adapter as adapter
    except ImportError:
        import keirin_jp_pdf_adapter as adapter

    if getattr(adapter, "_CONTROL_CHAR_FIX_INSTALLED", False):
        return

    adapter._word_text = _clean_word_text
    adapter._HEADER_PATTERN = re.compile(r"([1-9])番車(?:.*全選択)?")
    adapter._CONTROL_CHAR_FIX_INSTALLED = True
