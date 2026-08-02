"""複数のKEIRIN.JP PDFから必要な3種類を実解析で自動選別する。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .keirin_jp_3pdf_adapter import parse_basic_text
    from .keirin_jp_6pdf_adapter import parse_hs_text
    from .keirin_jp_pdf_adapter import _parse_keirin_jp_odds_pdf
    from .keirin_pdf_adapter import PdfInputError, _extract_text, _identity
except ImportError:
    from keirin_jp_3pdf_adapter import parse_basic_text
    from keirin_jp_6pdf_adapter import parse_hs_text
    from keirin_jp_pdf_adapter import _parse_keirin_jp_odds_pdf
    from keirin_pdf_adapter import PdfInputError, _extract_text, _identity


def _same_race(items: tuple[dict[str, Any], ...]) -> bool:
    """開催場・レース番号・日付が矛盾しない組み合わせだけを許可する。"""
    venue_races = {
        (item["identity"].get("venue"), item["identity"].get("race"))
        for item in items
        if item["identity"].get("venue") is not None
        and item["identity"].get("race") is not None
    }
    if len(venue_races) != 1:
        return False
    dates = {
        item["identity"].get("date")
        for item in items
        if item["identity"].get("date") is not None
    }
    return len(dates) <= 1


def _load_documents(paths: list[str | Path]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        text = _extract_text(path, path.name)
        documents.append(
            {
                "path": path,
                "filename": path.name,
                "text": text,
                "identity": _identity(text, path.name),
            }
        )
    return documents


def _parse_basic_candidates(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """文字の有無ではなく、全選手を実際に取得できたPDFだけを基本情報にする。"""
    candidates: list[dict[str, Any]] = []
    for document in documents:
        try:
            riders = parse_basic_text(document["text"])
        except (PdfInputError, TypeError, ValueError):
            continue
        candidates.append({"document": document, "riders": riders})
    return candidates


def select_bundle_roles(
    paths: list[str | Path],
) -> tuple[dict[str, Path], dict[str, Any]]:
    """余分なPDFを無視し、実解析に成功する同一レース3PDFだけを選ぶ。

    共通メニューに「競走得点」「H・S回数」という文字が載るため、単語判定は
    一切使わない。基本情報は全選手、H/Sは全車、オッズは全組み合わせを実際に
    読めた場合だけ採用する。
    """
    documents = _load_documents(paths)
    basic_candidates = _parse_basic_candidates(documents)
    if not basic_candidates:
        raise PdfInputError(
            "BUNDLE_BASIC_PARSE_FAILED",
            "追加したPDFの中に、全選手の競走得点・決まり手を読み取れる基本情報PDFがありません。",
            ["基本情報PDF"],
        )

    best: dict[str, Any] | None = None
    saw_hs = False
    saw_odds = False

    for basic_candidate in basic_candidates:
        basic_document = basic_candidate["document"]
        riders = basic_candidate["riders"]
        bikes = [int(rider["bike"]) for rider in riders]

        hs_candidates: list[dict[str, Any]] = []
        odds_candidates: list[dict[str, Any]] = []

        for document in documents:
            if document["path"] == basic_document["path"]:
                continue
            if not _same_race((basic_document, document)):
                continue

            try:
                hs_rows = parse_hs_text(document["text"], bikes)
            except (PdfInputError, TypeError, ValueError):
                pass
            else:
                if set(hs_rows) == set(bikes):
                    hs_candidates.append({"document": document, "rows": hs_rows})
                    saw_hs = True

            try:
                odds = _parse_keirin_jp_odds_pdf(
                    document["path"], document["text"], bikes
                )
            except (PdfInputError, TypeError, ValueError):
                pass
            else:
                expected = len(bikes) * (len(bikes) - 1)
                actual = sum(
                    1
                    for first in range(len(bikes))
                    for second in range(len(bikes))
                    if first != second and odds[first][second] is not None
                )
                if actual == expected:
                    odds_candidates.append(
                        {"document": document, "odds": odds, "count": actual}
                    )
                    saw_odds = True

        for hs_candidate in hs_candidates:
            for odds_candidate in odds_candidates:
                combo = (
                    basic_document,
                    hs_candidate["document"],
                    odds_candidate["document"],
                )
                if len({item["path"] for item in combo}) != 3:
                    continue
                if not _same_race(combo):
                    continue
                identity_complete = all(
                    item["identity"].get("venue") is not None
                    and item["identity"].get("race") is not None
                    for item in combo
                )
                score = (
                    1000
                    + len(riders) * 100
                    + len(hs_candidate["rows"]) * 10
                    + odds_candidate["count"]
                    + (500 if identity_complete else 0)
                )
                if best is None or score > best["score"]:
                    best = {
                        "basic": basic_document,
                        "hs": hs_candidate["document"],
                        "odds": odds_candidate["document"],
                        "score": score,
                        "rider_count": len(riders),
                        "odds_count": odds_candidate["count"],
                    }

    if best is None:
        missing: list[str] = []
        if not saw_hs:
            missing.append("着度数・H・S回数PDF")
        if not saw_odds:
            missing.append("2車単オッズPDF")
        if missing:
            raise PdfInputError(
                "BUNDLE_PARSED_ROLE_MISSING",
                "追加したPDFから「" + "・".join(missing) + "」を実データとして読み取れませんでした。",
                missing,
            )
        raise PdfInputError(
            "BUNDLE_RACE_MISMATCH",
            "読み取れた基本情報・H/S・2車単オッズが同じレースではありません。",
        )

    selected_paths = {
        "basic": best["basic"]["path"],
        "hs": best["hs"]["path"],
        "odds": best["odds"]["path"],
    }
    selected_set = set(selected_paths.values())
    audit = {
        "selection_method": "full_content_parse",
        "selected": {role: path.name for role, path in selected_paths.items()},
        "ignored": [
            document["filename"]
            for document in documents
            if document["path"] not in selected_set
        ],
        "uploaded_pdf_count": len(documents),
        "rider_count": best["rider_count"],
        "odds_count": best["odds_count"],
        "selection_score": best["score"],
    }
    return selected_paths, audit
