from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from individual_api.keirin_line_runtime_fix import install_line_parser_fix
from individual_api.keirin_odds_runtime_fix import install_odds_parser_fix
from individual_api.keirin_pdf_adapter import PdfInputError, _input_error
from individual_api.pr31_runtime import VERSION, predict_pr31

install_odds_parser_fix()

from individual_api.keirin_real_pdf_adapter import normalize_real_bundle
from .bundle_ui import INDEX_HTML

install_line_parser_fix()

app = FastAPI(title="章悟式∞競輪OS PR31 API", version=VERSION)
MAX_PDF_BYTES = 25 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


def _check_pin(pin: str) -> None:
    required = os.getenv("SHOGO_ACCESS_PIN", "").strip()
    if required and pin != required:
        raise HTTPException(403, "専用PINが違います。")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": VERSION,
        "engine": "PR31_FROZEN_ONLY",
        "a_strategy": "removed",
        "c_strategy": "removed",
        "required_roles": ["出走表・基本情報", "着度数・H・S回数", "2車単オッズ"],
        "upload_mode": "keirin_jp_three_pdfs_auto_detect",
        "selection_method": "PR31_probability_then_conditional_exacta_then_calibration_then_EV",
        "previous_day": "day2_or_later_only_KDreams_best_effort",
        "first_day": "no_previous_day_adjustment",
        "purchase_points": "3_to_5_or_no_bet",
        "missing_previous_day": "continue_without_fabrication",
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return INDEX_HTML


async def _run_bundle(files: list[UploadFile]) -> JSONResponse:
    if len(files) != 3:
        raise HTTPException(400, "競輪.jpのPDFを3枚ちょうど追加してください。")

    with tempfile.TemporaryDirectory(prefix="keirin-pr31-pdf-") as tmp:
        root = Path(tmp)
        saved: list[Path] = []
        hashes: set[bytes] = set()
        for index, upload in enumerate(files, start=1):
            original = Path(upload.filename or f"file_{index}.pdf").name.replace("\x00", "")
            path = root / f"{index:02d}__{original}"
            digest = hashlib.sha256()
            total = 0
            first_chunk = True
            with path.open("wb") as handle:
                while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                    if first_chunk and not chunk.startswith(b"%PDF"):
                        raise HTTPException(400, f"{original}は有効なPDFではありません。")
                    first_chunk = False
                    total += len(chunk)
                    if total > MAX_PDF_BYTES:
                        raise HTTPException(413, f"{original}は25MB以下にしてください。")
                    digest.update(chunk)
                    handle.write(chunk)
            if total == 0:
                raise HTTPException(400, f"{original}は空のPDFです。")
            fingerprint = digest.digest()
            if fingerprint in hashes:
                raise HTTPException(400, "同じPDFが重複しています。3種類のPDFを追加してください。")
            hashes.add(fingerprint)
            saved.append(path)

        try:
            payload, pdf_audit = normalize_real_bundle(saved, None)
            payload["race_type"] = "MEN"
            basic_name = pdf_audit["selected"]["basic"]
            basic_path = next(path for path in saved if path.name == basic_name)
            result = predict_pr31(payload, pdf_audit, basic_path)
            result["pdf_audit"] = pdf_audit
        except PdfInputError as exc:
            result = _input_error(exc)
            result["version"] = VERSION
        except RuntimeError as exc:
            code = str(exc)
            result = {
                "version": VERSION,
                "status": "PROCESSING_ERROR",
                "purchase_status": "NO_BET",
                "error": {
                    "code": code if code.startswith("PR31_") else "SAFE_PROCESSING_STOP",
                    "message": "PR31 Frozenモデルを安全に実行できませんでした。",
                    "missing": [],
                },
            }
        except Exception:
            result = {
                "version": VERSION,
                "status": "PROCESSING_ERROR",
                "purchase_status": "NO_BET",
                "error": {
                    "code": "SAFE_PROCESSING_STOP",
                    "message": "PDF解析またはPR31計算を安全停止しました。3PDFが同じレースか確認してください。",
                    "missing": [],
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
