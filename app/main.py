from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from individual_api.keirin_dual_strategy_api import VERSION, predict
from individual_api.keirin_odds_runtime_fix import install_odds_parser_fix
from individual_api.keirin_pdf_adapter import PdfInputError, _input_error

install_odds_parser_fix()

from individual_api.keirin_real_pdf_adapter import normalize_real_bundle
from .bundle_ui import INDEX_HTML

app = FastAPI(title="章悟式∞競輪OS 実PDF検証API", version=VERSION)


def _check_pin(pin: str) -> None:
    required = os.getenv("SHOGO_ACCESS_PIN", "").strip()
    if required and pin != required:
        raise HTTPException(403, "専用PINが違います。")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_status": "keirin_jp_real_pdf_full_parse",
        "upload_mode": "multiple_pdfs_auto_detect",
        "required_roles": ["基本情報", "着度数・H・S回数", "2車単オッズ"],
        "selection_method": "real_full_parse",
        "extra_pdfs": "ignored",
        "strategies": {"shogo": 5, "residual": 3},
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return INDEX_HTML


async def _save_pdf(upload: UploadFile, path: Path, label: str) -> None:
    data = await upload.read()
    if not data or not data.startswith(b"%PDF"):
        raise HTTPException(400, f"{label}は有効なPDFではありません。")
    path.write_bytes(data)


async def _save_optional_image(upload: UploadFile | None, root: Path) -> Path | None:
    if upload is None or not upload.filename:
        return None
    data = await upload.read()
    if not data:
        return None
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(400, "EX画像はPNG・JPG・WebPにしてください。")
    path = root / f"ex{suffix}"
    path.write_bytes(data)
    return path


async def _run_bundle(
    files: list[UploadFile],
    ex_image: UploadFile | None,
    lambda_value: float,
) -> JSONResponse:
    if not 0 <= lambda_value <= 1:
        raise HTTPException(400, "λは0以上1以下にしてください。")
    if len(files) < 3:
        raise HTTPException(400, "PDFを3枚以上追加してください。")
    if len(files) > 20:
        raise HTTPException(400, "PDFは20枚以内にしてください。")

    with tempfile.TemporaryDirectory(prefix="keirin-real-pdf-") as tmp:
        root = Path(tmp)
        saved: list[Path] = []
        for index, upload in enumerate(files, start=1):
            original = Path(upload.filename or f"file_{index}.pdf").name.replace("\x00", "")
            path = root / f"{index:02d}__{original}"
            await _save_pdf(upload, path, original)
            saved.append(path)
        ex_path = await _save_optional_image(ex_image, root)
        try:
            payload, pdf_audit = normalize_real_bundle(saved, ex_path)
            payload["race_type"] = "MEN"
            payload["lambda_value"] = float(lambda_value)
            result = predict(payload)
            result["pdf_audit"] = pdf_audit
        except PdfInputError as exc:
            result = _input_error(exc)
            result["version"] = VERSION
        return JSONResponse(
            status_code=200 if result.get("status") == "OK" else 422,
            content=result,
        )


@app.post("/analyze-bundle")
async def analyze_bundle(
    files: list[UploadFile] = File(...),
    ex_image: UploadFile | None = File(default=None),
    lambda_value: float = Form(default=0.50),
    pin: str = Form(default=""),
) -> JSONResponse:
    _check_pin(pin)
    return await _run_bundle(files, ex_image, lambda_value)


@app.post("/analyze")
async def analyze_legacy(
    basic_pdf: UploadFile = File(...),
    hs_pdf: UploadFile = File(...),
    odds_pdf: UploadFile = File(...),
    ex_image: UploadFile | None = File(default=None),
    lambda_value: float = Form(default=0.50),
    pin: str = Form(default=""),
) -> JSONResponse:
    _check_pin(pin)
    return await _run_bundle(
        [basic_pdf, hs_pdf, odds_pdf], ex_image, lambda_value
    )
