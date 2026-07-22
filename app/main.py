from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .backtest import evaluate_predictions
from .engine import MODEL_VERSION, predict
from .parser import (
    DOCUMENT_BASIC,
    DOCUMENT_ODDS,
    classify_pdf,
    document_identity,
    parse_entry_pdf,
    parse_ex_text,
    parse_odds_pdf,
    parse_result_pdf,
)


app = FastAPI(
    title="章悟式∞競輪OS 2車単PDF API",
    version=MODEL_VERSION,
    description="基本情報PDF・2車単オッズPDFと任意のEX文字データから、展開・モンテカルロ・EVを返す研究API。",
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
        "model_version": MODEL_VERSION,
        "model_status": "research_unvalidated",
        "bet_type": "2車単",
        "field_sizes": [5, 6, 7, 8, 9],
        "required_documents": ["基本情報PDF", "2車単オッズPDF"],
        "optional_inputs": ["EX文字データ"],
        "pdf_classification": "labelled_fields_with_dedicated_validation",
        "monte_carlo_default": 100_000,
        "seed_default": 3156,
        "pdftotext_available": shutil.which("pdftotext") is not None,
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return INDEX_HTML


@app.post("/analyze")
async def analyze(
    basic_pdf: UploadFile = File(...),
    odds_pdf: UploadFile = File(...),
    ex_text: str = Form(default=""),
    pin: str = Form(default=""),
    monte_carlo_runs: int = Form(default=100_000),
    seed: int = Form(default=3156),
) -> dict[str, Any]:
    _check_pin(pin)
    # The form already assigns an explicit role to each upload. Older versions
    # ignored those roles and guessed from a few headings, so valid netkeirin
    # print layouts could be rejected as "unknown".
    uploads = [basic_pdf, odds_pdf]
    expected_types = [DOCUMENT_BASIC, DOCUMENT_ODDS]
    try:
        with tempfile.TemporaryDirectory(prefix="shogo-keirin-pdf-") as tmp:
            root = Path(tmp)
            paths: list[Path] = []
            for index, upload in enumerate(uploads):
                payload = await upload.read()
                if not payload.startswith(b"%PDF"):
                    raise HTTPException(400, f"{upload.filename or index + 1} はPDFではありません。")
                safe_name = Path(upload.filename or f"document_{index}.pdf").name
                path = root / f"{index}_{safe_name}"
                path.write_bytes(payload)
                paths.append(path)

            classified = dict(zip(expected_types, paths, strict=True))
            audit: list[dict[str, str]] = []
            for expected, upload, path in zip(expected_types, uploads, paths, strict=True):
                audit.append({
                    "filename": upload.filename or path.name,
                    "input_field": expected,
                    "content_hint": classify_pdf(path),
                    "validation": "dedicated_parser",
                })

            try:
                race = parse_entry_pdf(classified[DOCUMENT_BASIC])
            except ValueError as exc:
                raise ValueError(f"基本情報PDFを読めません: {exc}") from exc
            identities = {kind: document_identity(path) for kind, path in classified.items()}
            for kind, identity in identities.items():
                if identity["venue"] != race.venue or identity["race_number"] != race.race_number:
                    raise ValueError(
                        f"別レースのPDFが混ざっています: {kind}={identity['venue']}{identity['race_number']}R、"
                        f"基本={race.venue}{race.race_number}R"
                    )

            numbers = [rider.number for rider in race.riders]
            try:
                odds = parse_odds_pdf(classified[DOCUMENT_ODDS], numbers)
            except ValueError as exc:
                raise ValueError(f"2車単オッズPDFを読めません: {exc}") from exc
            ex_data, ex_warnings = parse_ex_text(ex_text, numbers)
            output = predict(
                race,
                odds,
                ex_data=ex_data,
                ex_warnings=ex_warnings,
                runs=monte_carlo_runs,
                seed=seed,
                history=None,
            )
            output["document_audit"] = audit
            output["input_policy"] = {
                "external_web_fetch": False,
                "screenshots": False,
                "optional_ex_text_used": bool(ex_data),
                "unreadable_values": "未取得",
                "results_or_payouts_used_for_prediction": False,
            }
            return output
    except HTTPException:
        raise
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise HTTPException(422, str(exc)) from exc


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
body{margin:0;background:#f3f5f7;color:#17212b;font-family:system-ui,-apple-system,sans-serif}.wrap{max-width:760px;margin:auto;padding:22px}
.card{background:white;border-radius:22px;padding:22px;margin:18px 0;box-shadow:0 8px 30px #17212b14}h1{font-size:30px;margin:18px 0}h2{font-size:21px}
label{display:block;font-weight:750;margin:17px 0 7px}input,textarea{width:100%;box-sizing:border-box;padding:13px;border:1px solid #aeb8c2;border-radius:10px;background:white;font:inherit}textarea{min-height:180px;resize:vertical}
button{width:100%;border:0;border-radius:13px;padding:17px;background:#087546;color:white;font-size:18px;font-weight:800;margin-top:20px}.muted{color:#62707d;font-size:14px}
.error{color:#a51616;white-space:pre-wrap}.pill{display:inline-block;padding:5px 9px;border-radius:999px;background:#e7f6ee;color:#075f3b;font-weight:700;margin:3px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px 6px;border-bottom:1px solid #e4e8ec;text-align:right}th:first-child,td:first-child{text-align:left}
pre{white-space:pre-wrap;word-break:break-word;background:#111827;color:#d1fae5;padding:14px;border-radius:12px;max-height:420px;overflow:auto}
</style></head><body><main class="wrap"><h1>章悟式∞競輪OS</h1>
<section class="card"><p>netkeirinの基本情報PDFと2車単オッズPDFの2点で計算します。EX文字データは任意で、未入力・一部欠損でも停止しません。5〜9車対応。</p>
<form id="form">
<label>基本情報PDF</label><input type="file" name="basic_pdf" accept="application/pdf" required>
<label>2車単オッズPDF</label><input type="file" name="odds_pdf" accept="application/pdf" required>
<label>EX文字データ（任意）</label><textarea name="ex_text" placeholder="枠,車番,選手名,かまし成功率,つっぱり成功率,ちぎり率,ちぎられ率&#10;1,1,島村匠,-,-,-,4%(1/21)"></textarea>
<label>専用PIN</label><input type="password" name="pin" autocomplete="current-password">
<input type="hidden" name="monte_carlo_runs" value="100000"><input type="hidden" name="seed" value="3156">
<button type="submit">10万回で計算する</button></form><p id="status" class="muted"></p><p id="error" class="error"></p></section>
<section id="result" hidden><div class="card"><h2 id="race"></h2><div id="lines"></div></div>
<div class="card"><h2>EV 1.00以上</h2><table><thead><tr><th>2車単</th><th>確率</th><th>オッズ</th><th>EV</th></tr></thead><tbody id="ev"></tbody></table></div>
<details class="card"><summary>全JSON</summary><pre id="json"></pre></details></section></main>
<script>
const f=document.querySelector('#form'),s=document.querySelector('#status'),e=document.querySelector('#error'),r=document.querySelector('#result');
f.addEventListener('submit',async x=>{x.preventDefault();e.textContent='';r.hidden=true;s.textContent='PDF解析と10万回計算中…';
try{const q=await fetch('/analyze',{method:'POST',body:new FormData(f)}),d=await q.json();if(!q.ok)throw Error(d.detail||'計算に失敗しました');
document.querySelector('#race').textContent=`${d.race.venue} ${d.race.race_number}R・${d.race.entrant_count}車`;
document.querySelector('#lines').innerHTML=d.race.lines.map(v=>`<span class="pill">${v.join('-')}</span>`).join('');
const p=d.all_combinations.filter(v=>v.ev>=1).sort((a,b)=>b.ev-a.ev);document.querySelector('#ev').innerHTML=p.map(v=>`<tr><td>${v.first}-${v.second}</td><td>${(v.estimated_probability*100).toFixed(2)}%</td><td>${v.odds.toFixed(1)}</td><td><b>${v.ev.toFixed(2)}</b></td></tr>`).join('')||'<tr><td colspan="4">該当なし</td></tr>';
document.querySelector('#json').textContent=JSON.stringify(d,null,2);r.hidden=false;s.textContent=`完了：${d.calculation.monte_carlo_runs.toLocaleString()}回、確率合計 ${d.calculation.probability_sum_all_ordered_pairs}`;
}catch(z){e.textContent=z.message;s.textContent='';}});
</script></body></html>"""
