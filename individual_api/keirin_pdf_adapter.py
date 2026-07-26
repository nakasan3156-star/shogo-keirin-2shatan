"""章悟式∞競輪OS: 固定3PDFを正規化して個人評価型APIを実行する。"""

from __future__ import annotations

import re
import subprocess
import logging
from pathlib import Path
from typing import Any

from keirin_individual_api import VERSION, predict


MAX_FILE_BYTES = 50_000_000
REQUIRED_UPLOADS = ("racecard_pdf", "hs_pdf", "odds_pdf")

PREFECTURE_TO_REGION = {
    "北海道": "北日本", "青森": "北日本", "岩手": "北日本", "宮城": "北日本",
    "秋田": "北日本", "山形": "北日本", "福島": "北日本",
    "茨城": "関東", "栃木": "関東", "群馬": "関東", "埼玉": "関東",
    "東京": "関東", "新潟": "関東", "長野": "関東", "山梨": "関東",
    "千葉": "南関東", "神奈川": "南関東", "静岡": "南関東",
    "富山": "中部", "石川": "中部", "岐阜": "中部", "愛知": "中部", "三重": "中部",
    "福井": "近畿", "滋賀": "近畿", "京都": "近畿", "大阪": "近畿", "兵庫": "近畿", "奈良": "近畿", "和歌山": "近畿",
    "鳥取": "中国", "島根": "中国", "岡山": "中国", "広島": "中国", "山口": "中国",
    "徳島": "四国", "香川": "四国", "愛媛": "四国", "高知": "四国",
    "福岡": "九州", "佐賀": "九州", "長崎": "九州", "熊本": "九州",
    "大分": "九州", "宮崎": "九州", "鹿児島": "九州", "沖縄": "九州",
}


class PdfInputError(ValueError):
    def __init__(self, code: str, message: str, missing: list[str] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.missing = missing or []


def _input_error(exc: PdfInputError) -> dict[str, Any]:
    return {
        "version": VERSION,
        "status": "INPUT_ERROR",
        "purchase_status": "NO_BET",
        "error": {"code": exc.code, "message": exc.message, "missing": exc.missing},
    }


def _check_pdf(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    try:
        size = resolved.stat().st_size
    except OSError:
        raise PdfInputError("MISSING_PDF", f"{label}が見つかりません", [label])
    if size == 0:
        raise PdfInputError("EMPTY_PDF", f"{label}が空です", [label])
    if size > MAX_FILE_BYTES:
        raise PdfInputError("PDF_TOO_LARGE", f"{label}が50MBを超えています")
    try:
        with resolved.open("rb") as stream:
            magic = stream.read(5)
    except OSError:
        raise PdfInputError("UNREADABLE_PDF", f"{label}を読み取れません", [label])
    if magic != b"%PDF-":
        raise PdfInputError("INVALID_PDF", f"{label}はPDFではありません", [label])
    return resolved


def _extract_text(path: Path, label: str) -> str:
    try:
        completed = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        raise PdfInputError("PDF_ENGINE_MISSING", "pdftotextが利用できません")
    except subprocess.TimeoutExpired:
        raise PdfInputError("PDF_TIMEOUT", f"{label}の解析が時間切れになりました")
    if completed.returncode != 0 or not completed.stdout.strip():
        raise PdfInputError("PDF_PARSE_FAILED", f"{label}を解析できません", [label])
    return completed.stdout


def _identity(text: str) -> dict[str, str | int | None]:
    target = re.search(r"(?m)^\s*(\d{1,2})R(?:\s*$|\s+[Ａ-ＺA-Z])", text)
    venue = re.search(r"(?m)^\s*([^\s]+)\s+(?:初日|\d{4}/\d{2}/\d{2})", text)
    date = re.search(r"(\d{4})[年/](\d{1,2})[月/](\d{1,2})日?", text)
    if date:
        date_value = f"{int(date.group(1)):04d}-{int(date.group(2)):02d}-{int(date.group(3)):02d}"
    else:
        date_value = None
    return {
        "venue": venue.group(1) if venue else None,
        "date": date_value,
        "race": int(target.group(1)) if target else None,
    }


def _grade(text: str) -> str:
    match = re.search(r"(?m)^\s*(FI|FII|GIII|GI)\s+", text)
    if not match:
        raise PdfInputError("GRADE_NOT_FOUND", "出走表からグレードを取得できません")
    return {"FI": "F1", "FII": "F2", "GIII": "G3", "GI": "G1"}[match.group(1)]


def _pre_race_status(text: str, race_number: int, label: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    ended = False
    for index, line in enumerate(lines):
        if line != f"{race_number}R":
            continue
        for following in lines[index + 1:index + 6]:
            if re.fullmatch(r"(?:締切\d+分前|投票受付中|発売中)", following):
                return following
            if following == "終了":
                ended = True
    if ended:
        raise PdfInputError("POST_RACE_SOURCE", f"{label}はレース終了後の資料です")
    raise PdfInputError("PRE_RACE_STATUS_NOT_FOUND", f"{label}のレース前状態を確認できません")


def _parse_riders(racecard_text: str, hs_text: str) -> list[dict[str, Any]]:
    stat_pattern = re.compile(
        r"^\s*([1-9]{2})\s+([0-9]+\.[0-9]+)\s+(逃|追|両)"
        r"\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
        r"\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+)%",
        re.MULTILINE,
    )
    stats = list(stat_pattern.finditer(racecard_text))
    name_pattern = re.compile(
        r"^\s{5,}([^\d\s][^\n]{1,20})\n\s+([^\s]+)\s+\d+歳\s*$",
        re.MULTILINE,
    )
    names = list(name_pattern.finditer(racecard_text))
    if len(stats) not in {5, 6, 7, 8, 9} or len(names) != len(stats):
        raise PdfInputError("RIDER_PARSE_FAILED", "出走表の全選手を正しく読み取れません")

    hs_pattern = re.compile(
        r"^\s*([1-9])\s+[^\n]+\n\s*[^\n]+/(?:逃|追|両)"
        r"\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$",
        re.MULTILINE,
    )
    hs_rows = {
        int(match.group(1)): {"H": int(match.group(6)), "S_hs": int(match.group(7))}
        for match in hs_pattern.finditer(hs_text)
    }
    bikes = [int(match.group(1)[-1]) for match in stats]
    if set(hs_rows) != set(bikes):
        raise PdfInputError("HS_PARSE_FAILED", "H・S表の全選手を正しく読み取れません")

    riders: list[dict[str, Any]] = []
    for stat, name in zip(stats, names):
        bike = int(stat.group(1)[-1])
        prefecture = name.group(2).replace(" ", "")
        if prefecture not in PREFECTURE_TO_REGION:
            raise PdfInputError("REGION_PARSE_FAILED", f"{bike}番の府県を地区へ変換できません")
        riders.append({
            "bike": bike,
            "name": name.group(1).replace(" ", "").replace("追加", ""),
            "region": PREFECTURE_TO_REGION[prefecture],
            "score": float(stat.group(2)),
            "B": int(stat.group(5)),
            "escape": int(stat.group(6)),
            "makuri": int(stat.group(7)),
            "sashi": int(stat.group(8)),
            "mark": int(stat.group(9)),
            "win_rate": float(stat.group(14)),
            "H": hs_rows[bike]["H"],
        })
    riders.sort(key=lambda rider: rider["bike"])
    if [rider["bike"] for rider in riders] != list(range(1, len(riders) + 1)):
        raise PdfInputError("BIKE_SEQUENCE_ERROR", "車番が連番ではありません")
    return riders


def _parse_lines(hs_pdf: Path, bikes: list[int]) -> list[list[int]]:
    try:
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        import pdfplumber
        with pdfplumber.open(hs_pdf) as document:
            for page in document.pages[:2]:
                words = page.extract_words()
                labels = [word for word in words if "並び予想" in word["text"]]
                for label in labels:
                    y = float(label["top"])
                    candidates = [
                        word for word in words
                        if y + 12 <= float(word["top"]) <= y + 38
                        and re.fullmatch(r"[1-9]", word["text"])
                    ]
                    candidates.sort(key=lambda word: float(word["x0"]))
                    found = [int(word["text"]) for word in candidates]
                    if sorted(found) != sorted(bikes):
                        continue
                    lines: list[list[int]] = [[found[0]]]
                    previous_x = float(candidates[0]["x0"])
                    for word, bike in zip(candidates[1:], found[1:]):
                        x = float(word["x0"])
                        if x - previous_x > 42:
                            lines.append([])
                        lines[-1].append(bike)
                        previous_x = x
                    if sorted(bike for line in lines for bike in line) == sorted(bikes):
                        return lines
    except Exception as exc:
        raise PdfInputError("LINE_PARSE_FAILED", "並び予想を読み取れません") from exc
    raise PdfInputError("LINE_PARSE_FAILED", "並び予想を読み取れません")


def _parse_odds(odds_text: str, bikes: list[int]) -> list[list[float | None]]:
    total = len(bikes) * (len(bikes) - 1)
    marker = re.search(rf"1[～~〜\-]{total}人気", odds_text)
    end = odds_text.find("※結果・成績・オッズ")
    if not marker or end < marker.start():
        raise PdfInputError("ODDS_TABLE_NOT_FOUND", "2車単人気順表を取得できません")
    lines = odds_text[marker.start():end].splitlines()
    parsed: list[tuple[int, int, int, float]] = []
    for index, line in enumerate(lines):
        match = re.match(r"^\s*(\d{1,2})\s+([1-9])\s+([0-9]+(?:\.[0-9]+)?)\s*$", line)
        if not match:
            continue
        rank, first, price = int(match.group(1)), int(match.group(2)), float(match.group(3))
        if rank != len(parsed) + 1:
            continue
        second = None
        for following in lines[index + 1:index + 9]:
            candidate = re.match(r"^\s*([1-9])\s*$", following)
            if candidate:
                second = int(candidate.group(1))
                break
        if second is not None:
            parsed.append((rank, first, second, price))
    if len(parsed) != total:
        raise PdfInputError(
            "ODDS_PARSE_FAILED",
            f"2車単は{total}通り必要ですが{len(parsed)}通りしか読めません",
        )
    pairs = {(first, second): price for _, first, second, price in parsed}
    expected = {(first, second) for first in bikes for second in bikes if first != second}
    if set(pairs) != expected:
        raise PdfInputError("ODDS_PAIR_MISMATCH", "2車単の組み合わせに欠損または重複があります")
    return [
        [None if first == second else pairs[(first, second)] for second in bikes]
        for first in bikes
    ]


def normalize_pdfs(
    racecard_pdf: str | Path,
    hs_pdf: str | Path,
    odds_pdf: str | Path,
    ex_image: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "racecard_pdf": _check_pdf(racecard_pdf, "racecard_pdf"),
        "hs_pdf": _check_pdf(hs_pdf, "hs_pdf"),
        "odds_pdf": _check_pdf(odds_pdf, "odds_pdf"),
    }
    racecard_text = _extract_text(paths["racecard_pdf"], "racecard_pdf")
    hs_text = _extract_text(paths["hs_pdf"], "hs_pdf")
    odds_text = _extract_text(paths["odds_pdf"], "odds_pdf")

    identities = {
        "racecard_pdf": _identity(racecard_text),
        "hs_pdf": _identity(hs_text),
        "odds_pdf": _identity(odds_text),
    }
    identity_values = {
        (item["venue"], item["date"], item["race"]) for item in identities.values()
    }
    if None in {value for identity in identities.values() for value in identity.values()}:
        raise PdfInputError("RACE_ID_NOT_FOUND", "3PDFの開催・日付・レース番号を確認できません")
    if len(identity_values) != 1:
        raise PdfInputError("RACE_MISMATCH", "3PDFが同一レースではありません")
    identity = next(iter(identities.values()))
    race_number = int(identity["race"])
    source_status = {
        "racecard_pdf": _pre_race_status(racecard_text, race_number, "racecard_pdf"),
        "odds_pdf": _pre_race_status(odds_text, race_number, "odds_pdf"),
    }

    riders = _parse_riders(racecard_text, hs_text)
    bikes = [rider["bike"] for rider in riders]
    lines = _parse_lines(paths["hs_pdf"], bikes)
    odds = _parse_odds(odds_text, bikes)

    source_files = {key: path.name for key, path in paths.items()}
    missing_optional = []
    if ex_image is not None:
        image_path = Path(ex_image)
        try:
            if image_path.stat().st_size <= 0:
                raise OSError
        except OSError:
            missing_optional.append("ex_image")
        else:
            source_files["ex_image"] = image_path.name
    else:
        missing_optional.append("ex_image")

    payload = {
        "grade": _grade(racecard_text),
        "source_files": source_files,
        "riders": riders,
        "lines": lines,
        "odds": odds,
        "conditions": {},
    }
    audit = {
        "race": identity,
        "rider_count": len(riders),
        "odds_count": len(bikes) * (len(bikes) - 1),
        "lines": lines,
        "missing_optional": missing_optional + ["wind_mps", "temperature_c", "bank_type"],
        "result_data_used": False,
        "web_data_used": False,
        "pre_race_status": source_status,
    }
    return payload, audit


def predict_from_files(
    racecard_pdf: str | Path,
    hs_pdf: str | Path,
    odds_pdf: str | Path,
    ex_image: str | Path | None = None,
) -> dict[str, Any]:
    """固定3PDFを直接受け取り、例外を外へ出さず予測結果を返す。"""
    try:
        payload, audit = normalize_pdfs(racecard_pdf, hs_pdf, odds_pdf, ex_image)
        result = predict(payload)
        result["pdf_audit"] = audit
        return result
    except PdfInputError as exc:
        return _input_error(exc)
    except Exception:
        return {
            "version": VERSION,
            "status": "PROCESSING_ERROR",
            "purchase_status": "NO_BET",
            "error": {
                "code": "UNEXPECTED_PDF_PROCESSING_ERROR",
                "message": "PDF処理を安全停止しました",
                "missing": [],
            },
        }
