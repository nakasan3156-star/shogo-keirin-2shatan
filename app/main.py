from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from individual_api.keirin_ac_strategy_api import VERSION, predict
from individual_api.keirin_line_runtime_fix import install_line_parser_fix
from individual_api.keirin_odds_runtime_fix import install_odds_parser_fix
from individual_api.keirin_pdf_adapter import PdfInputError, _input_error

install_odds_parser_fix()

from individual_api.keirin_real_pdf_adapter import normalize_real_bundle
from .bundle_ui import INDEX_HTML

install_line_parser_fix()

app = FastAPI(title="章悟式∞競輪OS A/C統合API", version=VERSION)


def _upload_path(root: Path, upload: UploadFile, label: str) -> Path:
    """Preserve identity-bearing filenames while stripping client directories."""
    original = Path(upload.filename or f"{label}.pdf").name.replace("\x00", "")
    return root / f"{label}__{original}"


def _check_pin(pin: str) -> None:
    required = os.getenv("SHOGO_ACCESS_PIN", "").strip()
    if required and pin != required:
        raise HTTPException(403, "専用PINが違います。")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": VERSION,
        "model_status": "keirin_jp_resilient_pdf_parse",
        "upload_mode": "multiple_pdfs_auto_detect",
        "required_roles": ["出走表・基本情報", "着度数・H・S回数", "2車単オッズ"],
        "selection_method": "real_full_parse_with_safe_fallbacks",
        "closed_odds": "allowed",
        "missing_lines": "singleton_fallback",
        "strategies": {"shogo": 5, "residual": 3},
        "legacy_health_compatibility_only": True,
        "active_model_status": "A_and_C_frozen",
        "active_upload_mode": "keirin_jp_three_pdfs_auto_detect",
        "active_selection_method": "real_full_parse_strict_same_race",
        "active_missing_lines": "safe_stop_after_resilient_retry",
        "active_line_parser": "adaptive_coordinate_and_text",
        "active_strategies": {
            "a": "purchase_filter_max_3",
            "c": "individual_line_scenario_mc100k_then_ev_purchase",
        },
        "c_simulations": 100000,
        "c_seed": 3156,
        "c_ev_formula": "probability_times_odds",
        "c_ev_purchase": "enabled_3_to_5_or_no_bet",
        "residual_b": "removed_from_prediction_path",
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return INDEX_HTML


async def _run_bundle(files: list[UploadFile]) -> JSONResponse:
    if len(files) != 3:
        raise HTTPException(400, "競輪.jpのPDFを3枚ちょうど追加してください。")

    with tempfile.TemporaryDirectory(prefix="keirin-ac-pdf-") as tmp:
        root = Path(tmp)
        saved: list[Path] = []
        hashes: set[bytes] = set()
        for index, upload in enumerate(files, start=1):
            original = Path(upload.filename or f"file_{index}.pdf").name.replace("\x00", "")
            data = await upload.read()
            if not data or not data.startswith(b"%PDF"):
                raise HTTPException(400, f"{original}は有効なPDFではありません。")
            if data in hashes:
                raise HTTPException(400, "同じPDFが重複しています。3種類のPDFを追加してください。")
            hashes.add(data)
            path = root / f"{index:02d}__{original}"
            path.write_bytes(data)
            saved.append(path)

        try:
            payload, pdf_audit = normalize_real_bundle(saved, None)
            payload["race_type"] = "MEN"
            result = predict(payload)
            result["pdf_audit"] = pdf_audit
        except PdfInputError as exc:
            result = _input_error(exc)
            result["version"] = VERSION
        except Exception as exc:
            result = {
                "version": VERSION,
                "status": "PROCESSING_ERROR",
                "purchase_status": "NO_BET",
                "error": {
                    "code": "SAFE_PROCESSING_STOP",
                    "message": "PDF解析を安全停止しました。3PDFが同じレースか確認してください。",
                    "missing": [],
                    "detail": type(exc).__name__,
                },
            }
        return JSONResponse(
            status_code=200 if result.get("status") == "OK" else 422,
            content=result,
        )


@app.post("/analyze-bundle")
async def analyze_bundle(
    files: list[UploadFile] = File(...),
    pin: str = Form(default=""),
) -> JSONResponse:
    _check_pin(pin)
    return await _run_bundle(files)


@app.post("/analyze")
async def analyze_legacy(
    basic_pdf: UploadFile = File(...),
    hs_pdf: UploadFile = File(...),
    odds_pdf: UploadFile = File(...),
    pin: str = Form(default=""),
) -> JSONResponse:
    _check_pin(pin)
    return await _run_bundle([basic_pdf, hs_pdf, odds_pdf])
