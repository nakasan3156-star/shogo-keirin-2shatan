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
            "KEIRIN.JP レース情報PDF×2（基本情報と着度数・H・S回数、順不同）",
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
        race_info_1 = root / "race_info_1.pdf"
        race_info_2 = root / "race_info_2.pdf"
        odds = root / "odds.pdf"
        await _save_pdf(basic_pdf, race_info_1, "レース情報PDF①")
        await _save_pdf(hs_pdf, race_info_2, "レース情報PDF②")
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
            race_info_1, race_info_2, odds, ex_path, lambda_value=lambda_value
        )
        return JSONResponse(
            status_code=200 if result.get("status") == "OK" else 422,
            content=result,
        )


INDEX_HTML = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<meta name="theme-color" content="#07643f">
<title>章悟式∞競輪OS</title>
<style>
:root{font-family:system-ui,-apple-system,"Noto Sans JP",sans-serif;color:#17212b;background:#f3f5f7}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0}.wrap{max-width:760px;margin:auto;padding:14px}.card{background:#fff;border-radius:20px;padding:18px;margin:14px 0;box-shadow:0 8px 24px #17212b12}h1{font-size:26px;margin:12px 2px 16px}h2{font-size:20px;margin:0 0 12px}.lead{margin:0 0 8px;line-height:1.6}.notice{margin:10px 0 4px;padding:12px;border-radius:12px;background:#edf8f2;color:#075f3b;font-weight:750;line-height:1.55}.field-title{font-weight:800;font-size:17px;margin:18px 0 8px}.file-box{display:flex;align-items:center;gap:12px;width:100%;min-height:72px;padding:12px;border:2px solid #b8c2cb;border-radius:16px;background:#fff;cursor:pointer;-webkit-tap-highlight-color:transparent}.file-box:active{background:#f2f7f4}.file-box.selected{border-color:#087d49;background:#f2faf6}.file-button{flex:0 0 auto;padding:12px 14px;border:1px solid #6f7a83;border-radius:9px;background:#f8f9fa;font-size:17px;font-weight:700}.file-name{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:16px;color:#394550}.file-name.empty{color:#7b8791}.native-file{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}.optional{font-size:13px;color:#6a7680;font-weight:600}.settings{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:18px}.settings label{font-weight:800;font-size:15px}.settings input{width:100%;height:52px;margin-top:7px;padding:10px 12px;border:1px solid #aeb8c2;border-radius:12px;font-size:18px}.submit{width:100%;min-height:68px;margin-top:22px;border:0;border-radius:16px;background:#087d49;color:#fff;font-size:19px;font-weight:900}.submit:disabled{opacity:.55}.status{min-height:24px;margin:16px 2px 0;font-weight:700}.error{color:#b11919;white-space:pre-wrap;font-weight:700}.result-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.result-card{background:#fff;border-radius:18px;padding:16px}.result-card h2{font-size:19px}.pick{display:grid;grid-template-columns:70px 1fr 76px;align-items:center;gap:8px;padding:11px 0;border-bottom:1px solid #e6eaed}.pick:last-child{border-bottom:0}.pair{font-size:21px;font-weight:900}.prob{font-size:14px;color:#52606a}.ev{text-align:right;font-weight:900}.empty-result{color:#727e87;padding:12px 0}@media(max-width:600px){.wrap{padding:10px}.card{padding:16px}.file-box{min-height:78px}.file-button{font-size:16px}.settings{grid-template-columns:1fr}.result-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<main class="wrap">
<h1>章悟式∞競輪OS</h1>
<section class="card">
<p class="lead"><b>必要なのはKEIRIN.JPの3PDFだけ。</b></p>
<div class="notice">レース情報2枚は、<b>基本情報</b>と<b>着度数・H・S回数</b>。<br>①②の順番はどっちでもOK。自動で判別します。</div>
<form id="form">
<div class="field-title">① レース情報PDF</div>
<label class="file-box" for="basic_pdf"><span class="file-button">ファイルを選択</span><span class="file-name empty" data-name="basic_pdf">選択されていません</span></label>
<input class="native-file" id="basic_pdf" type="file" name="basic_pdf" accept="application/pdf" required>

<div class="field-title">② レース情報PDF</div>
<label class="file-box" for="hs_pdf"><span class="file-button">ファイルを選択</span><span class="file-name empty" data-name="hs_pdf">選択されていません</span></label>
<input class="native-file" id="hs_pdf" type="file" name="hs_pdf" accept="application/pdf" required>

<div class="field-title">③ 2車単オッズPDF</div>
<label class="file-box" for="odds_pdf"><span class="file-button">ファイルを選択</span><span class="file-name empty" data-name="odds_pdf">選択されていません</span></label>
<input class="native-file" id="odds_pdf" type="file" name="odds_pdf" accept="application/pdf" required>

<div class="field-title">EX画像 <span class="optional">任意</span></div>
<label class="file-box" for="ex_image"><span class="file-button">画像を選択</span><span class="file-name empty" data-name="ex_image">選択されていません</span></label>
<input class="native-file" id="ex_image" type="file" name="ex_image" accept="image/png,image/jpeg,image/webp">

<div class="settings">
<label>残差λ<input type="number" name="lambda_value" min="0" max="1" step="0.05" value="0.50"></label>
<label>専用PIN<input type="password" name="pin" autocomplete="current-password"></label>
</div>
<button class="submit" id="submit" type="submit">しょーご式5点＋残差3点を計算</button>
</form>
<p id="status" class="status"></p>
<p id="error" class="error"></p>
</section>
<section id="result" class="result-grid" hidden>
<div class="result-card"><h2>しょーご式（5点）</h2><div id="shogo"></div></div>
<div class="result-card"><h2>市場残差（3点）</h2><div id="residual"></div></div>
</section>
</main>
<script>
const form=document.querySelector('#form');
const statusEl=document.querySelector('#status');
const errorEl=document.querySelector('#error');
const resultEl=document.querySelector('#result');
const submitEl=document.querySelector('#submit');
const shogoEl=document.querySelector('#shogo');
const residualEl=document.querySelector('#residual');
for(const input of document.querySelectorAll('.native-file')){
  input.addEventListener('change',()=>{
    const target=document.querySelector(`[data-name="${input.name}"]`);
    const box=target.closest('.file-box');
    const file=input.files&&input.files[0];
    target.textContent=file?file.name:'選択されていません';
    target.classList.toggle('empty',!file);
    box.classList.toggle('selected',!!file);
    errorEl.textContent='';
  });
}
function renderRows(items){
  if(!items||!items.length)return '<div class="empty-result">候補なし</div>';
  return items.map(v=>`<div class="pick"><div class="pair">${v.pair.join('-')}</div><div class="prob">確率 ${(Number(v.probability)*100).toFixed(1)}%<br>オッズ ${Number(v.odds).toFixed(1)}</div><div class="ev">EV<br>${Number(v.conservative_ev).toFixed(2)}</div></div>`).join('');
}
function clearResult(){
  resultEl.hidden=true;
  shogoEl.innerHTML='';
  residualEl.innerHTML='';
}
form.addEventListener('submit',async event=>{
  event.preventDefault();
  errorEl.textContent='';
  clearResult();
  submitEl.disabled=true;
  statusEl.textContent='3PDFを解析して計算中…';
  try{
    const response=await fetch('/analyze',{method:'POST',body:new FormData(form)});
    const data=await response.json();
    if(!response.ok)throw new Error(data.error?.message||data.detail||'計算に失敗しました');
    shogoEl.innerHTML=renderRows(data.strategies?.shogo?.candidates);
    residualEl.innerHTML=renderRows(data.strategies?.residual?.candidates);
    resultEl.hidden=false;
    statusEl.textContent='計算完了';
    resultEl.scrollIntoView({behavior:'smooth',block:'start'});
  }catch(error){
    clearResult();
    errorEl.textContent=error.message;
    statusEl.textContent='';
  }finally{
    submitEl.disabled=false;
  }
});
</script>
</body>
</html>"""
