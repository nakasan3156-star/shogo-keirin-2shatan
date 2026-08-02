"""KEIRIN.JP公式6PDFから、しょーご式5点と市場残差3点を返す。"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

try:
    from .keirin_dual_strategy_api import VERSION, predict
    from .keirin_jp_6pdf_adapter import normalize_six_pdfs
    from .keirin_pdf_adapter import PdfInputError, _extract_text, _input_error
except ImportError:
    from keirin_dual_strategy_api import VERSION, predict
    from keirin_jp_6pdf_adapter import normalize_six_pdfs
    from keirin_pdf_adapter import PdfInputError, _extract_text, _input_error


def _race_type(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "WOMEN" if re.search(r"ガールズ|女子競輪|L級", normalized) else "MEN"


def predict_from_files(
    basic_pdf: str | Path,
    recent_pdf: str | Path,
    matchup_pdf: str | Path,
    track_pdf: str | Path,
    hs_pdf: str | Path,
    odds_pdf: str | Path,
    ex_image: str | Path | None = None,
    lambda_value: float = 0.50,
) -> dict[str, Any]:
    """同じレースの公式6PDFを照合し、2方式を完全分離して計算する。"""
    try:
        payload, audit = normalize_six_pdfs(
            basic_pdf, recent_pdf, matchup_pdf, track_pdf, hs_pdf, odds_pdf, ex_image
        )
        payload["race_type"] = _race_type(_extract_text(Path(basic_pdf), "basic_pdf"))
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
            "version": VERSION, "status": "INPUT_ERROR", "purchase_status": "NO_BET",
            "error": {"code": "INVALID_SIX_PDF_INPUT", "message": str(exc), "missing": []},
        }
    except Exception:
        return {
            "version": VERSION, "status": "PROCESSING_ERROR", "purchase_status": "NO_BET",
            "error": {
                "code": "UNEXPECTED_SIX_PDF_PROCESSING_ERROR",
                "message": "6PDF処理を安全停止しました", "missing": [],
            },
        }
