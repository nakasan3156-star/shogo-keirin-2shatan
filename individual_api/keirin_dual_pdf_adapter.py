"""固定3PDFを正規化し、しょーご式5点と残差式3点を同時実行する。"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

try:
    from .keirin_dual_strategy_api import VERSION, predict
    from .keirin_pdf_adapter import (
        PdfInputError,
        _extract_text,
        _input_error,
        normalize_pdfs,
    )
except ImportError:  # 直接スクリプトとして実行する場合
    from keirin_dual_strategy_api import VERSION, predict
    from keirin_pdf_adapter import (
        PdfInputError,
        _extract_text,
        _input_error,
        normalize_pdfs,
    )


def _race_type(racecard_text: str) -> str:
    normalized = unicodedata.normalize("NFKC", racecard_text)
    women_markers = (
        r"ガールズ(?:ケイリン)?",
        r"女子競輪",
        r"L級(?:1|2)?班?",
    )
    return "WOMEN" if any(re.search(pattern, normalized) for pattern in women_markers) else "MEN"


def predict_from_files(
    racecard_pdf: str | Path,
    hs_pdf: str | Path,
    odds_pdf: str | Path,
    ex_image: str | Path | None = None,
    lambda_value: float = 0.50,
) -> dict[str, Any]:
    """同じ3PDFから2方式を分離計算し、例外を外へ出さず返す。"""
    try:
        payload, audit = normalize_pdfs(racecard_pdf, hs_pdf, odds_pdf, ex_image)
        racecard_path = Path(racecard_pdf)
        racecard_text = _extract_text(racecard_path, "racecard_pdf")
        payload["race_type"] = _race_type(racecard_text)
        payload["lambda_value"] = float(lambda_value)
        audit["race_type"] = payload["race_type"]
        audit["girls_excluded"] = True
        result = predict(payload)
        result["pdf_audit"] = audit
        return result
    except PdfInputError as exc:
        result = _input_error(exc)
        result["version"] = VERSION
        return result
    except (TypeError, ValueError) as exc:
        return {
            "version": VERSION,
            "status": "INPUT_ERROR",
            "purchase_status": "NO_BET",
            "error": {
                "code": "INVALID_DUAL_STRATEGY_INPUT",
                "message": str(exc),
                "missing": [],
            },
        }
    except Exception:
        return {
            "version": VERSION,
            "status": "PROCESSING_ERROR",
            "purchase_status": "NO_BET",
            "error": {
                "code": "UNEXPECTED_DUAL_PDF_PROCESSING_ERROR",
                "message": "PDF処理を安全停止しました",
                "missing": [],
            },
        }
