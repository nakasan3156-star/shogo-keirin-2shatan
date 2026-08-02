from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from individual_api.keirin_dual_pdf_adapter import predict_from_files
from individual_api.keirin_dual_strategy_api import VERSION

from .backtest import evaluate_predictions
from .parser import parse_result_pdf


app = FastAPI(
    title="章悟式∞競輪OS しょーご式5点＋市場残差3点 API",
    version=VERSION,
    description=(
        "同じ3PDFから能力予測を作り、しょーご式5点と"
        "市場残差システム3点を別々に返します。女子競輪は対象外です。"
    ),
)


class BacktestRequest(BaseModel):
    races: list[dict[str, Any]]
    thresholds: list[float] = Field(default_factory=lambda: [1.0, 1.05, 1.10, 1.15, 1.20])
    stake_per_pair: int = 100


def _check_pin(pin: str) -> None:
    required = os.getenv("SHOGO_ACCESS_PIN", "").strip()
    if required and pin != required:
        raise HTTPException(403, "専用PINが違います。")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_version": VERSION,
        "model_status": "dual_strategy",
        "bet_type": "2車単",
        "race_type": "MEN_ONLY",
        "required_documents": [
            "netkeirin出走表PDF",
            "KEIRIN.JP H・S回数PDF",
            "netkeirin 2車単オッズPDF",
        ],
        "optional_inputs": ["EXデータスクショ"],
        "strategies": {
            "shogo": {"candidate_count": 5, "simulations": 100_000},
            "residual": {"candidate_count": 3, "simulations": 100_000},
        },
        "result_data_used": False,
        "web_data_used": False,
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return INDEX_HTML


async def _save_upload(upload: UploadFile, destination: Path, label: str) -> None:
    payload = await upload.read()
    if not payload:
        raise HTTPException(400, f"{label}が空です。")
    if not payload.startswith(b"%PDF"):
        raise HTTPException(400, f"{label}はPDFではありません。")
    destination.write_bytes(payload)


def _upload_path(root: Path, upload: UploadFile, prefix: str) -> Path:
    """開催情報を含む元ファイル名を安全な一時パス上で保持する。"""
    original = Path(upload.filename or f"{prefix}.pdf").name.replace("\x00", "")
    if not original:
        original = f"{prefix}.pdf"
    return root / f"{prefix}__{original}"


@app.post("/analyze")
async def analyze(
    racecard_pdf: UploadFile = File(...),
    hs_pdf: UploadFile = File(...),
    odds_pdf: UploadFile = File(...),
    ex_image: UploadFile | None = File(default=None),
    lambda_value: float = Form(default=0.50),
    pin: str = Form(default=""),
) -> JSONResponse:
    _check_pin(pin)
    if not 0.0 <= lambda_value <= 1.0:
        raise HTTPException(400, "λは0以上1以下にしてください。")
    try:
        with tempfile.TemporaryDirectory(prefix="shogo-keirin-dual-") as tmp:
            root = Path(tmp)
            racecard_path = _upload_path(root, racecard_pdf, "racecard")
            hs_path = _upload_path(root, hs_pdf, "hs")
            odds_path = _upload_path(root, odds_pdf, "odds")
            await _save_upload(racecard_pdf, racecard_path, "出走表PDF")
            await _save_upload(hs_pdf, hs_path, "H・S回数PDF")
            await _save_upload(odds_pdf, odds_path, "2車単オッズPDF")

            ex_path: Path | None = None
            if ex_image is not None and ex_image.filename:
                payload = await ex_image.read()
                if payload:
                    suffix = Path(ex_image.filename).suffix.lower()
                    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                        raise HTTPException(400, "EXデータはPNG・JPG・WebP画像を選択してください。")
                    ex_path = root / f"ex{suffix}"
                    ex_path.write_bytes(payload)

            result = predict_from_files(
                racecard_path,
                hs_path,
                odds_path,
                ex_path,
                lambda_value=lambda_value,
            )
            status_code = 200 if result.get("status") == "OK" else 422
            return JSONResponse(status_code=status_code, content=result)
    except HTTPException:
        raise
    except Exception:
        return JSONResponse(
            status_code=500,
            content={
                "version": VERSION,
                "status": "PROCESSING_ERROR",
                "purchase_status": "NO_BET",
                "error": {
                    "code": "UNEXPECTED_SERVER_ERROR",
                    "message": "APIを安全停止しました。入力ファイルを確認してください。",
                    "missing": [],
                },
            },
        )


@app.post("/backtest")
def backtest(request: BacktestRequest) -> dict[str, Any]:
    try:
        return evaluate_predictions(request.races, request.thresholds, request.stake_per_pair)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/parse-result")
async def parse_result(result_pdf: UploadFile = File(...), pin: str = Form(default="")) -> dict[str, Any]:
    _check_pin(pin)
    result_bytes = await result_pdf.read()
    if not result_bytes.startswith(b"%PDF"):
        raise HTTPException(400, "結果はPDFファイルである必要があります。")
    try:
        with tempfile.TemporaryDirectory(prefix="shogo-keirin-result-") as tmp:
            path = Path(tmp) / "result.pdf"
            path.write_bytes(result_bytes)
            return parse_result_pdf(path)
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise HTTPException(422, str(exc)) from exc


INDEX_HTML = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#075f3b"><title>章悟式∞競輪OS</title>
<style>
body{margin:0;background:#f3f5f7;color:#17212b;font-family:system-ui,-apple-system,sans-serif}.wrap{max-width:860px;margin:auto;padding:22px}
.card{background:white;border-radius:22px;padding:22px;margin:18px 0;box-shadow:0 8px 30px #17212b14}h1{font-size:30px;margin:18px 0}h2{font-size:21px}
label{display:block;font-weight:750;margin:17px 0 7px}input{width:100%;box-sizing:border-box;padding:13px;border:1px solid #aeb8c2;border-radius:10px;background:white;font:inherit}
button{width:100%;border:0;border-radius:13px;padding:17px;background:#087546;color:white;font-size:18px;font-weight:800;margin-top:20px}.muted{color:#62707d;font-size:14px}
.error{color:#a51616;white-space:pre-wrap}.pill{display:inline-block;padding:5px 9px;border-radius:999px;background:#e7f6ee;color:#075f3b;font-weight:700;margin:3px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.metric{background:#f3f8f5;border-radius:12px;padding:12px}.metric b{display:block;margin-top:4px}
.strategy-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.strategy{margin:0}.strategy h2{margin-top:0}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px 6px;border-bottom:1px solid #e4e8ec;text-align:right}th:first-child,td:first-child{text-align:left}
pre{white-space:pre-wrap;word-break:break-word;background:#111827;color:#d1fae5;padding:14px;border-radius:12px;max-height:420px;overflow:auto}
@media(max-width:680px){.wrap{padding:16px}.card{padding:18px}.grid,.strategy-grid{grid-template-columns:1fr}h1{font-size:28px}}
</style></head><body><main class="wrap"><h1>章悟式∞競輪OS</h1>
<section class="card"><p>同じ3PDFから、<b>しょーご式5点</b>と<b>市場残差システム3点</b>を完全に分けて計算します。女子競輪は対象外です。</p>
<form id="form">
<label>① netkeirin 出走表PDF</label><input type="file" name="racecard_pdf" accept="application/pdf" required>
<label>② KEIRIN.JP H・S回数PDF</label><input type="file" name="hs_pdf" accept="application/pdf" required>
<label>③ netkeirin 2車単オッズPDF</label><input type="file" name="odds_pdf" accept="application/pdf" required>
<label>EXデータスクショ（任意）</label><input type="file" name="ex_image" accept="image/png,image/jpeg,image/webp">
<label>残差λ（0＝市場のみ、1＝能力のみ）</label><input type="number" name="lambda_value" min="0" max="1" step="0.05" value="0.50">
<label>専用PIN</label><input type="password" name="pin" autocomplete="current-password">
<button type="submit">2方式を計算する</button></form><p id="status" class="muted"></p><p id="error" class="error"></p></section>
<section id="result" hidden><div class="card"><h2 id="race"></h2><div id="lines"></div>
<div class="grid" style="margin-top:14px"><div class="metric">主導権候補<b id="control"></b></div><div class="metric">本命ライン<b id="mainline"></b></div>
<div class="metric">最有力展開<b id="scenario"></b></div><div class="metric">共通買い目<b id="common"></b></div></div></div>
<div class="strategy-grid">
<div class="card strategy"><h2>しょーご式（5点）</h2><table><thead><tr><th>2車単</th><th>確率</th><th>オッズ</th><th>保守EV</th></tr></thead><tbody id="shogo"></tbody></table><p class="muted">個人能力→役割→ライン→展開→10万回の能力確率を使い、保守EV上位5点。</p></div>
<div class="card strategy"><h2>市場残差（3点）</h2><table><thead><tr><th>2車単</th><th>確率</th><th>オッズ</th><th>保守EV</th></tr></thead><tbody id="residual"></tbody></table><p class="muted">市場確率 ×（能力確率÷市場確率）^λ を再正規化し、10万回後の保守EV上位3点。</p></div>
</div>
<details class="card"><summary>全JSON</summary><pre id="json"></pre></details></section></main>
<script>
const f=document.querySelector('#form'),s=document.querySelector('#status'),e=document.querySelector('#error'),r=document.querySelector('#result');
const pct=v=>`${(Number(v)*100).toFixed(1)}%`;
const rows=p=>(p||[]).map(v=>`<tr><td>${v.pair.join('-')}</td><td>${pct(v.probability)}</td><td>${v.odds.toFixed(1)}</td><td><b>${v.conservative_ev.toFixed(2)}</b></td></tr>`).join('')||'<tr><td colspan="4">候補なし</td></tr>';
f.addEventListener('submit',async x=>{x.preventDefault();e.textContent='';r.hidden=true;s.textContent='3PDF解析と2方式の10万回計算中…';
try{const q=await fetch('/analyze',{method:'POST',body:new FormData(f)}),d=await q.json();if(!q.ok)throw Error(d.error?.message||d.detail||'計算に失敗しました');
const a=d.pdf_audit||{},id=a.race||{};document.querySelector('#race').textContent=`${id.venue||''} ${id.race||''}R・${d.version}`;
document.querySelector('#lines').innerHTML=(a.lines||[]).map(v=>`<span class="pill">${v.join('-')}</span>`).join('');
document.querySelector('#control').textContent=`${d.predicted_control}番（${pct(d.control_confidence)}）`;
const mainLine=Array.isArray(d.predicted_main_line_bikes)?d.predicted_main_line_bikes:(a.lines||[]).find(v=>v.includes(Number(d.predicted_main_line)))||[d.predicted_main_line];
document.querySelector('#mainline').textContent=mainLine.filter(v=>v!==null&&v!==undefined).join('-');
document.querySelector('#scenario').textContent=d.predicted_scenario||'未取得';
document.querySelector('#common').textContent=(d.common_candidates||[]).map(v=>v.join('-')).join('、')||'なし';
document.querySelector('#shogo').innerHTML=rows(d.strategies?.shogo?.candidates);
document.querySelector('#residual').innerHTML=rows(d.strategies?.residual?.candidates);
document.querySelector('#json').textContent=JSON.stringify(d,null,2);r.hidden=false;s.textContent=`完了：能力${Number(d.ability_simulations).toLocaleString()}回＋残差${Number(d.residual_simulations).toLocaleString()}回`;
}catch(z){e.textContent=z.message;s.textContent='';}});
</script></body></html>"""
