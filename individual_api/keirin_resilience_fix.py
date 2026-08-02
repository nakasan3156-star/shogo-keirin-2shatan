"""実PDF運用で欠損しても安全に計算継続できる項目を吸収する。"""
from __future__ import annotations

from typing import Any


def install_resilience_fix() -> None:
    """並び欠損を単騎扱いへ落とし、予測停止を防ぐ。"""
    try:
        from . import keirin_real_pdf_adapter as adapter
    except ImportError:
        import keirin_real_pdf_adapter as adapter

    if getattr(adapter, "_RESILIENCE_FIX_INSTALLED", False):
        return

    original_parse_lines = adapter._parse_lines

    def safe_parse_lines(path: Any, bikes: list[int]) -> list[list[int]]:
        try:
            lines = original_parse_lines(path, bikes)
            flat = [int(bike) for line in lines for bike in line]
            if lines and sorted(flat) == sorted(int(bike) for bike in bikes):
                return [[int(bike) for bike in line] for line in lines]
        except Exception:
            pass
        # KEIRIN.JP基本情報PDFに並びが載らない場合は、全員単騎として
        # ライン補正だけ無効化し、個人能力・H/S・オッズで計算継続する。
        return [[int(bike)] for bike in bikes]

    adapter._parse_lines = safe_parse_lines
    adapter._RESILIENCE_FIX_INSTALLED = True
