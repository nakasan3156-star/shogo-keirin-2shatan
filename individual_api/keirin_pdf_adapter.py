"""章悟式∞競輪OS: 固定3PDFを正規化して個人評価型APIを実行する。"""

from __future__ import annotations

import re
import subprocess
import logging
import unicodedata
from pathlib import Path
from typing import Any

try:
    from .keirin_individual_api import VERSION, predict
except ImportError:  # 直接スクリプトとして実行する場合
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

PREFECTURE_ALIASES = {
    # netkeirinの狭い府県欄では府県名が途中で省略されることがある。
    "和歌": "和歌山",
    "神奈": "神奈川",
    "鹿児": "鹿児島",
}


def _normalize_prefecture(value: str) -> str | None:
    """Normalize full, suffixed and uniquely abbreviated prefecture labels."""
    normalized = re.sub(r"[\s・･.．]+", "", value or "")
    normalized = PREFECTURE_ALIASES.get(normalized, normalized)
    if normalized in PREFECTURE_TO_REGION:
        return normalized
    if normalized.endswith(("都", "府", "県")):
        without_suffix = normalized[:-1]
        without_suffix = PREFECTURE_ALIASES.get(without_suffix, without_suffix)
        if without_suffix in PREFECTURE_TO_REGION:
            return without_suffix
    candidates = [
        prefecture
        for prefecture in PREFECTURE_TO_REGION
        if len(normalized) >= 2 and prefecture.startswith(normalized)
    ]
    return candidates[0] if len(candidates) == 1 else None


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



def _identity(text: str, filename: str = "") -> dict[str, str | int | None]:
    """Read race identity across common full/half-width and date formats."""
    normalized_text = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")
    normalized_filename = unicodedata.normalize("NFKC", filename)
    combined = f"{normalized_text}\n{normalized_filename}"
    target = re.search(
        r"(?m)^\s*(\d{1,2})\s*R(?:\s*$|\s+[A-Z])",
        normalized_text,
    ) or re.search(r"(?:^|\s)(\d{1,2})\s*R(?:\s|_|$)", normalized_filename)
    venue = re.search(
        r"(?m)^\s*([^\s]+)\s+(?:初日|最終日|\d+日目|\d{4}[/.-]\d{1,2}[/.-]\d{1,2})",
        normalized_text,
    ) or re.search(r"(?:^|[/_ -])([^\s/_-]+?)競輪(?:場)?(?:\s|$)", normalized_filename)
    date = re.search(
        r"(20\d{2})\s*(?:年|[/.-])\s*(\d{1,2})\s*(?:月|[/.-])\s*(\d{1,2})\s*日?",
        combined,
    )
    date_value = (
        f"{int(date.group(1)):04d}-{int(date.group(2)):02d}-{int(date.group(3)):02d}"
        if date else None
    )
    return {
        "venue": venue.group(1).removesuffix("競輪場").removesuffix("競輪") if venue else None,
        "date": date_value,
        "race": int(target.group(1)) if target else None,
    }


def _grade(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    match = re.search(r"(?m)^\s*(GIII|GII|GI|FII|FI)(?:\s|$)", normalized)
    if not match:
        # Grade affects purchase eligibility, not the numerical prediction.
        # Unknown grades therefore calculate safely and remain NO_BET.
        return "UNKNOWN"
    return {
        "FI": "F1", "FII": "F2", "GI": "G1", "GII": "G2", "GIII": "G3",
    }[match.group(1)]


def _pre_race_status(text: str, race_number: int, label: str) -> str:
    lines = [
        re.sub(r"\s+", "", unicodedata.normalize("NFKC", line))
        for line in text.splitlines()
    ]
    ended = False
    for index, line in enumerate(lines):
        if line != f"{race_number}R":
            continue
        for following in lines[index + 1:index + 13]:
            if re.fullmatch(r"(?:締切\d+分前|投票受付中|投票中|発売中)", following):
                return following
            if following in {"終了", "確定"}:
                ended = True
    if ended:
        raise PdfInputError("POST_RACE_SOURCE", f"{label}はレース終了後の資料です")
    # Some mobile/browser print layouts omit the status badge. Absence alone
    # must not stop a valid pre-race PDF; explicit 終了/確定 still stops it.
    return "未取得（終了表示なし）"

def _parse_hs_counts(hs_pdf: Path, bikes: list[int]) -> dict[int, dict[str, int]]:
    """KEIRIN.JP表を座標で読み、車番ごとのH・S回数を取得する。

    9車立てでは選手名や府県が複数行に分割され、pdftotextの行構造が
    崩れる場合がある。列位置は維持されるため、H/S見出しと同じ列に
    ある数値を車番行へ対応付ける。
    """
    try:
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        import pdfplumber
        rows: dict[int, dict[str, int]] = {}
        with pdfplumber.open(hs_pdf) as document:
            for page in document.pages[:2]:
                words = page.extract_words()
                headers = {
                    word["text"]: word
                    for word in words
                    if word["text"] in {"H", "S"}
                }
                if set(headers) != {"H", "S"}:
                    continue
                h_x = (float(headers["H"]["x0"]) + float(headers["H"]["x1"])) / 2
                s_x = (float(headers["S"]["x0"]) + float(headers["S"]["x1"])) / 2
                header_top = min(float(headers["H"]["top"]), float(headers["S"]["top"]))
                bike_words = [
                    word for word in words
                    if header_top + 12 < float(word["top"])
                    and 50 <= float(word["x0"]) <= 72
                    and re.fullmatch(r"[1-9]", word["text"])
                ]
                for bike_word in bike_words:
                    bike = int(bike_word["text"])
                    y = float(bike_word["top"])

                    def column_value(column_x: float) -> int | None:
                        matches = [
                            word for word in words
                            if re.fullmatch(r"\d{1,2}", word["text"])
                            and abs(float(word["top"]) - y) <= 3
                            and abs(
                                (float(word["x0"]) + float(word["x1"])) / 2
                                - column_x
                            ) <= 12
                        ]
                        return int(matches[0]["text"]) if len(matches) == 1 else None

                    h_value = column_value(h_x)
                    s_value = column_value(s_x)
                    if h_value is not None and s_value is not None:
                        rows[bike] = {"H": h_value, "S_hs": s_value}
                if set(rows) == set(bikes):
                    return rows
    except Exception as exc:
        raise PdfInputError("HS_PARSE_FAILED", "H・S表の全選手を正しく読み取れません") from exc
    raise PdfInputError("HS_PARSE_FAILED", "H・S表の全選手を正しく読み取れません")


RIDER_STAT_PATTERN = re.compile(
    # Depending on how the browser prints netkeirin, the coloured frame-number
    # cell can disappear from the PDF text layer.  The frame is not used by the
    # model, so accept both "11 100.00 ..." and "1 100.00 ..." layouts while
    # still requiring the complete numeric performance row.
    r"^\s*(?:([1-6])\s*)?([1-9])\s+([0-9]+\.[0-9]+)\s+(逃|追|両)"
    r"\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    r"\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s*[%％]?",
    re.MULTILINE,
)


def _rider_stat_rows(racecard_text: str) -> list[re.Match[str]]:
    text = unicodedata.normalize("NFKC", racecard_text).replace("\u00a0", " ")
    matches = list(RIDER_STAT_PATTERN.finditer(text))

    # A repeated table header/page fragment may duplicate a row.  Keep exactly
    # one verified row per bike; conflicting duplicate rows remain an error.
    rows_by_bike: dict[int, re.Match[str]] = {}
    for match in matches:
        bike = int(match.group(2))
        previous = rows_by_bike.get(bike)
        if previous is not None and previous.groups()[2:] != match.groups()[2:]:
            raise PdfInputError(
                "RIDER_PARSE_FAILED",
                f"出走表の{bike}番車の成績行が重複しています",
            )
        rows_by_bike.setdefault(bike, match)
    return sorted(rows_by_bike.values(), key=lambda match: int(match.group(2)))



def _parse_riders(
    racecard_text: str,
    hs_rows: dict[int, dict[str, int]],
) -> list[dict[str, Any]]:
    text = unicodedata.normalize("NFKC", racecard_text).replace("\u00a0", " ")
    stats = _rider_stat_rows(text)
    if len(stats) not in {5, 6, 7, 8, 9}:
        raise PdfInputError(
            "RIDER_PARSE_FAILED",
            f"出走表の成績行を正しく読み取れません（取得={len(stats)}人）",
        )

    profile_pattern = re.compile(
        r"(?m)^\s{2,}([^\d\s%.]{2,16})\s*$\n"
        r"\s*([^\s]+)\s+(\d{2})歳\s*$"
    )
    profiles = [
        {
            "start": match.start(),
            "name": match.group(1),
            "prefecture_raw": match.group(2),
        }
        for match in profile_pattern.finditer(text)
    ]
    used_profiles: set[int] = set()
    bikes = [int(match.group(2)) for match in stats]
    if set(hs_rows) != set(bikes):
        raise PdfInputError("HS_PARSE_FAILED", "H・S表の全選手を正しく読み取れません")

    riders: list[dict[str, Any]] = []
    for index, stat in enumerate(stats):
        bike = int(stat.group(2))
        block_end = stats[index + 1].start() if index + 1 < len(stats) else len(text)
        candidates = [
            profile for profile in profiles
            if int(profile["start"]) not in used_profiles
            and stat.end() <= int(profile["start"]) < block_end
        ]
        if not candidates:
            candidates = [
                profile for profile in profiles
                if int(profile["start"]) not in used_profiles
                and stat.start() - 500 <= int(profile["start"]) <= block_end + 1200
            ]
        profile = min(
            candidates,
            key=lambda item: abs(int(item["start"]) - stat.end()),
            default=None,
        )
        if profile:
            used_profiles.add(int(profile["start"]))
            name = str(profile["name"]).replace(" ", "").replace("追加", "")
            raw_prefecture = str(profile["prefecture_raw"])
        else:
            # Name and region are display/line-bonus fields only. Preserve all
            # verified numeric rows instead of aborting on a page-break variant.
            name = f"{bike}番車（氏名未取得）"
            raw_prefecture = ""

        prefecture = _normalize_prefecture(raw_prefecture)
        riders.append({
            "bike": bike,
            "name": name,
            "region": PREFECTURE_TO_REGION[prefecture] if prefecture else "未取得",
            "prefecture_raw": raw_prefecture or "未取得",
            "score": float(stat.group(3)),
            "B": int(stat.group(6)),
            "escape": int(stat.group(7)),
            "makuri": int(stat.group(8)),
            "sashi": int(stat.group(9)),
            "mark": int(stat.group(10)),
            "win_rate": float(stat.group(15)),
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
                        if y + 10 <= float(word["top"]) <= y + 42
                        and re.fullmatch(r"[1-9]", unicodedata.normalize("NFKC", word["text"]))
                    ]
                    candidates.sort(key=lambda word: float(word["x0"]))
                    found = [int(unicodedata.normalize("NFKC", word["text"])) for word in candidates]
                    if sorted(found) != sorted(bikes):
                        continue
                    parsed: list[list[int]] = [[found[0]]]
                    previous_x = float(candidates[0]["x0"])
                    for word, bike in zip(candidates[1:], found[1:]):
                        x = float(word["x0"])
                        if x - previous_x > 42:
                            parsed.append([])
                        parsed[-1].append(bike)
                        previous_x = x
                    if sorted(bike for line in parsed for bike in line) == sorted(bikes):
                        return parsed
    except Exception:
        pass

    # Text-layout fallback for browsers that flatten PDF word coordinates.
    text = unicodedata.normalize("NFKC", _extract_text(hs_pdf, "hs_pdf"))
    marker = text.find("並び予想")
    window = text[marker:marker + 1200] if marker >= 0 else text[:1200]
    for raw_line in window.splitlines():
        found = [int(value) for value in re.findall(r"(?<!\d)[1-9](?!\d)", raw_line)]
        if sorted(found) != sorted(bikes):
            continue
        parsed = []
        for segment in re.split(r"\s{8,}", raw_line.strip()):
            members = [int(value) for value in re.findall(r"(?<!\d)[1-9](?!\d)", segment)]
            if members:
                parsed.append(members)
        if sorted(bike for line in parsed for bike in line) == sorted(bikes):
            return parsed
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
    racecard_text = unicodedata.normalize(
        "NFKC", _extract_text(paths["racecard_pdf"], "racecard_pdf")
    ).replace("\u00a0", " ")
    hs_text = unicodedata.normalize(
        "NFKC", _extract_text(paths["hs_pdf"], "hs_pdf")
    ).replace("\u00a0", " ")
    odds_text = unicodedata.normalize(
        "NFKC", _extract_text(paths["odds_pdf"], "odds_pdf")
    ).replace("\u00a0", " ")

    identities = {
        "racecard_pdf": _identity(racecard_text, paths["racecard_pdf"].name),
        "hs_pdf": _identity(hs_text, paths["hs_pdf"].name),
        "odds_pdf": _identity(odds_text, paths["odds_pdf"].name),
    }
    venue_races = {
        (item["venue"], item["race"]) for item in identities.values()
        if item["venue"] is not None and item["race"] is not None
    }
    if len(venue_races) != 1 or any(
        item["venue"] is None or item["race"] is None
        for item in identities.values()
    ):
        raise PdfInputError("RACE_ID_NOT_FOUND", "3PDFの開催場・レース番号を確認できません")
    dates = {item["date"] for item in identities.values() if item["date"] is not None}
    if len(dates) > 1:
        raise PdfInputError("RACE_MISMATCH", "3PDFの日付が一致しません")
    venue, race_value = next(iter(venue_races))
    identity = {
        "venue": venue,
        "race": race_value,
        "date": next(iter(dates), None),
    }
    race_number = int(identity["race"])
    source_status = {
        "racecard_pdf": _pre_race_status(racecard_text, race_number, "racecard_pdf"),
        "odds_pdf": _pre_race_status(odds_text, race_number, "odds_pdf"),
    }

    stat_bikes = [int(match.group(2)) for match in _rider_stat_rows(racecard_text)]
    hs_rows = _parse_hs_counts(paths["hs_pdf"], stat_bikes)
    riders = _parse_riders(racecard_text, hs_rows)
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
        "compatibility_warnings": [
            f"{rider['bike']}番: 氏名または地区を取得できず数値項目のみ使用"
            for rider in riders
            if rider["name"].endswith("氏名未取得）") or rider["region"] == "未取得"
        ],
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
