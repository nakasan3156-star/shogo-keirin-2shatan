"""誤った3PDF実装を安全停止する一時アダプター。

本来の入力はKEIRIN.JPのレース情報5PDFと2車単オッズPDFの合計6PDF。
正しい6PDF版が復元・検証されるまで買い目を返さない。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .keirin_dual_strategy_api import VERSION
except ImportError:  # 直接スクリプトとして実行する場合
    from keirin_dual_strategy_api import VERSION


EXPECTED_INPUTS = [
    "KEIRIN.JP 基本情報PDF",
    "KEIRIN.JP 直近成績PDF",
    "KEIRIN.JP 対戦成績PDF",
    "KEIRIN.JP 当場成績PDF",
    "KEIRIN.JP 着度数・H・S回数PDF",
    "KEIRIN.JP 2車単オッズPDF",
]


def predict_from_files(
    racecard_pdf: str | Path,
    hs_pdf: str | Path,
    odds_pdf: str | Path,
    ex_image: str | Path | None = None,
    lambda_value: float = 0.50,
) -> dict[str, Any]:
    """正しい6PDF版が完成するまで常にNO_BETで安全停止する。"""
    return {
        "version": VERSION,
        "status": "SYSTEM_SUSPENDED",
        "purchase_status": "NO_BET",
        "strategies": {
            "shogo": {"purchase_status": "SUSPENDED", "candidates": []},
            "residual": {"purchase_status": "SUSPENDED", "candidates": []},
        },
        "error": {
            "code": "WRONG_INPUT_SPEC_SUSPENDED",
            "message": (
                "3PDF版は本来のしょーご式ではないため停止しました。"
                "KEIRIN.JPのレース情報5PDFと2車単オッズPDFを使う6PDF版へ修正中です。"
            ),
            "missing": EXPECTED_INPUTS,
        },
        "audit": {
            "bet_output_disabled": True,
            "reason": "wrong_input_spec",
            "expected_source": "KEIRIN.JP only",
            "expected_pdf_count": 6,
        },
    }
