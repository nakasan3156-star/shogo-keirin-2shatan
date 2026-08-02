"""複数のKEIRIN.JP PDFから必要な3種類を自動選別する。"""
from __future__ import annotations

import itertools
import re
import unicodedata
from pathlib import Path
from typing import Any

try:
    from .keirin_pdf_adapter import PdfInputError, _extract_text, _identity
except ImportError:
    from keirin_pdf_adapter import PdfInputError, _extract_text, _identity


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\u00a0", " ")


def role_scores(text: str) -> dict[str, int]:
    """PDF本文だけで基本情報・H/S・2車単オッズを判定する。"""
    normalized = _norm(text)
    compact = re.sub(r"\s+", "", normalized)

    basic = 0
    if "競走得点" in compact:
        basic += 8
    if all(token in compact for token in ("逃", "捲", "差", "マ")):
        basic += 3
    if "バック回数" in compact or "競走得点・バック回数" in compact:
        basic += 2

    hs = 0
    if "着外" in compact:
        hs += 4
    if all(token in compact for token in ("1着", "2着", "3着")):
        hs += 4
    if re.search(r"H[·・.]?S", compact):
        hs += 4

    odds = 0
    if "2車単オッズ" in compact:
        odds += 10
    if "番車全選択" in compact:
        odds += 3
    if "オッズ更新" in compact or "現在オッズ" in compact:
        odds += 2

    return {"basic": basic, "hs": hs, "odds": odds}


def _same_race(items: tuple[dict[str, Any], ...]) -> bool:
    venue_races = {
        (item["identity"].get("venue"), item["identity"].get("race"))
        for item in items
        if item["identity"].get("venue") is not None
        and item["identity"].get("race") is not None
    }
    if len(venue_races) > 1:
        return False
    dates = {
        item["identity"].get("date")
        for item in items
        if item["identity"].get("date") is not None
    }
    return len(dates) <= 1


def select_bundle_roles(paths: list[str | Path]) -> tuple[dict[str, Path], dict[str, Any]]:
    """余分なPDFを無視し、同一レースの基本・H/S・オッズを1枚ずつ選ぶ。"""
    documents: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        text = _extract_text(path, path.name)
        scores = role_scores(text)
        documents.append(
            {
                "path": path,
                "filename": path.name,
                "identity": _identity(text, path.name),
                "scores": scores,
            }
        )

    candidates = {
        "basic": [doc for doc in documents if doc["scores"]["basic"] >= 8],
        "hs": [doc for doc in documents if doc["scores"]["hs"] >= 8],
        "odds": [doc for doc in documents if doc["scores"]["odds"] >= 10],
    }
    missing: list[str] = []
    if not candidates["basic"]:
        missing.append("基本情報PDF")
    if not candidates["hs"]:
        missing.append("着度数・H・S回数PDF")
    if not candidates["odds"]:
        missing.append("2車単オッズPDF")
    if missing:
        raise PdfInputError(
            "BUNDLE_REQUIRED_PDF_MISSING",
            "選択したPDFの中に「" + "・".join(missing) + "」がありません。"
            "KEIRIN.JPで必要なタブをPDF保存し、まとめて追加してください。",
            missing,
        )

    best: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    best_score = -1
    for basic, hs, odds in itertools.product(
        candidates["basic"], candidates["hs"], candidates["odds"]
    ):
        combo = (basic, hs, odds)
        if len({item["path"] for item in combo}) != 3 or not _same_race(combo):
            continue
        score = (
            basic["scores"]["basic"]
            + hs["scores"]["hs"]
            + odds["scores"]["odds"]
        )
        identities = [item["identity"] for item in combo]
        if all(item.get("venue") is not None and item.get("race") is not None for item in identities):
            score += 50
        if score > best_score:
            best = combo
            best_score = score

    if best is None:
        raise PdfInputError(
            "BUNDLE_RACE_MISMATCH",
            "基本情報・H/S・2車単オッズが同じレースではありません。",
        )

    basic, hs, odds = best
    selected_paths = {
        "basic": basic["path"],
        "hs": hs["path"],
        "odds": odds["path"],
    }
    selected_set = set(selected_paths.values())
    audit = {
        "selected": {role: path.name for role, path in selected_paths.items()},
        "ignored": [doc["filename"] for doc in documents if doc["path"] not in selected_set],
        "uploaded_pdf_count": len(documents),
        "selection_score": best_score,
    }
    return selected_paths, audit
