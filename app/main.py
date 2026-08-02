from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from individual_api.keirin_dual_pdf_adapter import predict_from_files
from individual_api.keirin_dual_strategy_api import VERSION

app = FastAPI(title="章悟式∞競輪OS 3PDF API", version=VERSION)


def _check_pin(pin: str) -> None:
    required = os.getenv("SHOGO_ACCESS_PIN", "").strip()
    if required and pin != required:
        raise HTTPException(403, "専用PINが違います。")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_status": "keirin_jp_3pdf_dual_strategy",
        "required_documents": [
            "KEIRIN.JP 基本情報・並び予想PDF",
            "KEIRIN.JP 着度数・H・S回数PDF",
            "KEIRIN.JP 2車単オッズPDF",
        ],
        "strategies": {"shogo": 5, "residual": 3},
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return INDEX_HTML


async def _save_pdf(upload: UploadFile, path: Path, label: str) -> None:
    data = await upload.read()
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, f"{label}はPDFではありません。")
    path.write_bytes(data)


@app.post("/analyze")
async def analyze(
    basic_pdf: UploadFile = File(...),
    hs_pdf: UploadFile = File(...),
    odds_pdf: UploadFile = File(...),
    ex_image: UploadFile | None = File(default=None),
    lambda_value: float = Form(default=0.50),
    pin: str = Form(default=""),
) -> JSONResponse:
    _check_pin(pin)
    if not 0 <= lambda_value <= 1:
        raise HTTPException(400, "λは0以上1以下にしてください。")
    with tempfile.TemporaryDirectory(prefix="keirin-3pdf-") as tmp:
        root = Path(tmp)
        basic = root / "basic.pdf"
        hs = root / "hs.pdf"
        odds = root / "odds.pdf"
        await _save_pdf(basic_pdf, basic, "基本情報")
        await _save_pdf(hs_pdf, hs, "着度数・H・S回数")
        await _save_pdf(odds_pdf, odds, "2車単オッズ")
        ex_path = None
        if ex_image is not None and ex_image.filename:
            data = await ex_image.read()
            if data:
                suffix = Path(ex_image.filename).suffix.lower()
                if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                    raise HTTPException(400, "EX画像の形式が違います。")
                ex_path = root / f"ex{suffix}"
                ex_path.write_bytes(data)
        result = predict_from_files(
            basic, hs, odds, ex_path, lambda_value=lambda_value
        )
        return JSONResponse(
            status_code=200 if result.get("status") == "OK" else 422,
            content=result,
        )


INDEX_HTML = """<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>章悟式∞競輪OS</title><style>body{font-family:sans-serif;background:#f3f5f7;margin:0}.w{max-width:760px;margin:auto;padding:16px}.c{background:white;padding:20px;border-radius:18px;margin:14px 0}label{display:block;font-weight:700;margin-top:16px}input{width:100%;box-sizing:border-box;padding:12px;margin-top:6px}button{width:100%;padding:16px;margin-top:20px;background:#087546;color:white;border:0;border-radius:12px;font-size:18px;font-weight:700}.err{color:#a00}pre{white-space:pre-wrap}</style></head><body><main class='w'><h1>章悟式∞競輪OS</h1><section class='c'><p><b>必要なのはKEIRIN.JPの3PDFだけ。</b></p><form id='f'><label>① 基本情報・並び予想PDF</label><input type='file' name='basic_pdf' required><label>② 着度数・H・S回数PDF</label><input type='file' name='hs_pdf' required><label>③ 2車単オッズPDF</label><input type='file' name='odds_pdf' required><label>EX画像（任意）</label><input type='file' name='ex_image'><label>残差λ</label><input type='number' name='lambda_value' min='0' max='1' step='0.05' value='0.50'><label>専用PIN</label><input type='password' name='pin'><button>しょーご式5点＋残差3点を計算</button></form><p id='m'></p><p id='e' class='err'></p></section><section id='r' class='c' hidden><pre id='o'></pre></section></main><script>const f=document.querySelector('#f'),m=document.querySelector('#m'),e=document.querySelector('#e'),r=document.querySelector('#r'),o=document.querySelector('#o');f.onsubmit=async x=>{x.preventDefault();e.textContent='';m.textContent='計算中…';r.hidden=true;try{const q=await fetch('/analyze',{method:'POST',body:new FormData(f)}),d=await q.json();if(!q.ok)throw Error(d.error?.message||'計算失敗');const a=d.strategies?.shogo?.candidates||[],b=d.strategies?.residual?.candidates||[];o.textContent='しょーご式 5点\n'+a.map(v=>v.pair.join('-')+'  EV '+v.conservative_ev.toFixed(2)).join('\n')+'\n\n残差 3点\n'+b.map(v=>v.pair.join('-')+'  EV '+v.conservative_ev.toFixed(2)).join('\n');r.hidden=false;m.textContent='完了';}catch(z){e.textContent=z.message;m.textContent='';}};</script></body></html>"""
