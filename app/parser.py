from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unicodedata
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pdfplumber


@dataclass(frozen=True)
class Rider:
    frame: int
    number: int
    name: str
    prefecture: str
    age: int
    term: int | None
    grade: str | None
    race_score: float
    style: str
    start_count: int
    back_count: int
    escape_count: int
    makuri_count: int
    sashi_count: int
    mark_count: int
    first_count: int
    second_count: int
    third_count: int
    outside_count: int
    win_rate: float
    top2_rate: float
    top3_rate: float
    gear: float
    comment: str

    @property
    def races(self) -> int:
        return self.first_count + self.second_count + self.third_count + self.outside_count

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["races"] = self.races
        return value


@dataclass(frozen=True)
class RaceInput:
    race_id: str
    venue: str
    race_number: int
    race_class: str
    distance_m: int | None
    riders: tuple[Rider, ...]
    lines: tuple[tuple[int, ...], ...]
    line_warnings: tuple[str, ...]


DOCUMENT_BASIC = "basic"
DOCUMENT_RECENT = "recent"
DOCUMENT_ODDS = "odds"
DOCUMENT_UNKNOWN = "unknown"


PREFECTURE_REGION = {
    "北海道": "北日本", "青森": "北日本", "岩手": "北日本", "宮城": "北日本", "秋田": "北日本", "山形": "北日本", "福島": "北日本",
    "茨城": "関東", "栃木": "関東", "群馬": "関東", "埼玉": "関東", "東京": "関東", "新潟": "関東", "長野": "関東", "山梨": "関東",
    "千葉": "南関東", "神奈川": "南関東", "静岡": "南関東",
    "愛知": "中部", "岐阜": "中部", "三重": "中部", "富山": "中部", "石川": "中部",
    "福井": "近畿", "滋賀": "近畿", "京都": "近畿", "大阪": "近畿", "兵庫": "近畿", "奈良": "近畿", "和歌山": "近畿",
    "鳥取": "中国", "島根": "中国", "岡山": "中国", "広島": "中国", "山口": "中国",
    "徳島": "四国", "香川": "四国", "愛媛": "四国", "高知": "四国",
    "福岡": "九州", "佐賀": "九州", "長崎": "九州", "熊本": "九州", "大分": "九州", "宮崎": "九州", "鹿児島": "九州", "沖縄": "九州",
}


ROW_RE = re.compile(
    r"(?m)^\s*([1-6])([1-9])\s+([0-9]+\.[0-9]+)\s+([逃追両])\s+"
    + r"\s+".join([r"(\d+)"] * 10)
    + r"\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)\s+([^\n]*)$"
)


def pdftotext(path: Path) -> str:
    if shutil.which("pdftotext") is None:
        raise ValueError("pdftotext が見つかりません。Popplerが必要です。")
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def classify_pdf(path: Path) -> str:
    """Classify by PDF contents, never by the user-visible filename."""
    text = pdftotext(path)
    compact = re.sub(r"\s+", "", text)
    if "2車単" in compact and "人気選択組み合わせオッズ" in compact:
        return DOCUMENT_ODDS
    if "今節" in compact and "直近1" in compact and "直近2" in compact:
        return DOCUMENT_RECENT
    if "選手名" in compact and "競走" in compact and "得点" in compact and "勝率" in compact:
        return DOCUMENT_BASIC
    return DOCUMENT_UNKNOWN


def document_identity(path: Path) -> dict[str, Any]:
    text = pdftotext(path)
    race_id, venue, race_number, race_class, distance = _metadata(text, path.name)
    return {
        "race_id": race_id,
        "venue": venue,
        "race_number": race_number,
        "race_class": race_class,
        "distance_m": distance,
    }


def _metadata(text: str, source_name: str) -> tuple[str, str, int, str, int | None]:
    venue_match = re.search(r"(函館|青森|いわき平|弥彦|前橋|取手|宇都宮|大宮|西武園|京王閣|立川|松戸|川崎|平塚|小田原|伊東|静岡|名古屋|岐阜|大垣|豊橋|富山|松阪|四日市|福井|奈良|向日町|和歌山|岸和田|玉野|広島|防府|高松|小松島|高知|松山|小倉|久留米|武雄|佐世保|別府|熊本)\s+.*?\n(\d{1,2})R", text, re.S)
    if not venue_match:
        venue_match = re.search(r"(函館|青森|いわき平|弥彦|前橋|取手|宇都宮|大宮|西武園|京王閣|立川|松戸|川崎|平塚|小田原|伊東|静岡|名古屋|岐阜|大垣|豊橋|富山|松阪|四日市|福井|奈良|向日町|和歌山|岸和田|玉野|広島|防府|高松|小松島|高知|松山|小倉|久留米|武雄|佐世保|別府|熊本).*?(\d{1,2})R", source_name)
    if not venue_match:
        raise ValueError("競輪場またはレース番号を取得できません。")
    venue, race_number_text = venue_match.group(1), venue_match.group(2)
    class_match = re.search(r"([ＡＳＬ][級班]\s*(?:チ)?(?:一般|特選|選抜|予選|準決勝|決勝|特\s*選|決\s*勝))", text)
    race_class = re.sub(r"\s+", "", class_match.group(1)) if class_match else "未取得"
    distance_match = re.search(r"(\d{3,4})m\s+\d+周", text)
    distance = int(distance_match.group(1)) if distance_match else None
    date_match = re.search(r"(20\d{2})年(\d{2})月(\d{2})日", source_name)
    if date_match:
        date = "".join(date_match.groups())
    else:
        short_date = re.search(r"(\d{1,2})/(\d{1,2})", text)
        date = f"unknown-{short_date.group(1)}{short_date.group(2)}" if short_date else "unknown"
    race_id = f"{date}_{venue}_{int(race_number_text)}R"
    return race_id, venue, int(race_number_text), race_class, distance


def parse_entry_pdf(path: Path) -> RaceInput:
    text = pdftotext(path)
    matches = list(ROW_RE.finditer(text))
    if len(matches) < 5 or len(matches) > 9:
        raise ValueError(f"出走選手を正しく取得できませんでした（取得={len(matches)}人）。")

    # netkeirin changes the text order of the profile area depending on the
    # print layout, page break and browser.  Do not assume that the profile is
    # always between one statistics row and the next one.  Collect every
    # profile first and use the nearest unused profile as a fallback.
    prefectures = "|".join(sorted(PREFECTURE_REGION, key=len, reverse=True))
    profile_patterns = (
        re.compile(
            rf"(?m)^\s*([^\s\d%.]{{2,16}})\s*$\n\s*({prefectures})\s+(\d{{2}})歳\s*$"
            rf"(?:\n\s*(\d{{1,3}})期\s+([^\s]+)\s*$)?"
        ),
        re.compile(
            rf"(?m)^\s*([^\s\d%.]{{2,16}})\s+({prefectures})\s+(\d{{2}})歳"
            rf"(?:\s+(\d{{1,3}})期\s+([^\s]+))?\s*$"
        ),
    )
    profiles: list[dict[str, Any]] = []
    seen_profiles: set[tuple[int, str]] = set()
    for pattern in profile_patterns:
        for profile in pattern.finditer(text):
            key = (profile.start(), profile.group(1))
            if key in seen_profiles:
                continue
            seen_profiles.add(key)
            profiles.append(
                {
                    "start": profile.start(),
                    "name": profile.group(1),
                    "prefecture": profile.group(2),
                    "age": int(profile.group(3)),
                    "term": int(profile.group(4)) if profile.group(4) else None,
                    "grade": profile.group(5) if profile.group(5) else None,
                }
            )
    profiles.sort(key=lambda item: int(item["start"]))

    riders: list[Rider] = []
    profile_warnings: list[str] = []
    used_profile_positions: set[int] = set()
    for index, match in enumerate(matches):
        block_end = matches[index + 1].start() if index + 1 < len(matches) else match.end() + 800
        block = text[match.end():block_end]
        name_match = re.search(
            r"(?m)^\s*([^\s\d%.]{2,10})\s*$\n\s*([^\s\d]+)\s+(\d+)歳\s*$",
            block,
        )
        profile: dict[str, Any] | None = None
        if name_match:
            term_match = re.search(r"(?m)^\s*(\d+)期\s+([^\s]+)\s*$", block)
            profile = {
                "start": match.end() + name_match.start(),
                "name": name_match.group(1),
                "prefecture": name_match.group(2),
                "age": int(name_match.group(3)),
                "term": int(term_match.group(1)) if term_match else None,
                "grade": term_match.group(2) if term_match else None,
            }
        else:
            # A page break can place the name before the statistics row, or
            # after a repeated table header.  The nearest unused profile is a
            # safer association than aborting the whole race.
            candidates = [
                item
                for item in profiles
                if int(item["start"]) not in used_profile_positions
                and match.start() - 1200 <= int(item["start"]) <= block_end
            ]
            if candidates:
                profile = min(candidates, key=lambda item: abs(int(item["start"]) - match.end()))

        number = int(match.group(2))
        if profile:
            used_profile_positions.add(int(profile["start"]))
        else:
            # Name/prefecture/age do not enter the current numerical ability
            # model.  Preserve the valid statistics row and continue with an
            # explicit missing-data marker instead of turning a harmless PDF
            # layout variation into a fatal error.
            profile = {
                "start": -number,
                "name": f"{number}番車（氏名未取得）",
                "prefecture": "未取得",
                "age": 0,
                "term": None,
                "grade": None,
            }
            profile_warnings.append(
                f"{number}番車: 選手プロフィールを取得できず、成績数値のみで計算しました。"
            )
        values = match.groups()
        counts = [int(value) for value in values[4:14]]
        riders.append(
            Rider(
                frame=int(values[0]),
                number=int(values[1]),
                name=str(profile["name"]),
                prefecture=str(profile["prefecture"]),
                age=int(profile["age"]),
                term=profile["term"],
                grade=profile["grade"],
                race_score=float(values[2]),
                style=values[3],
                start_count=counts[0],
                back_count=counts[1],
                escape_count=counts[2],
                makuri_count=counts[3],
                sashi_count=counts[4],
                mark_count=counts[5],
                first_count=counts[6],
                second_count=counts[7],
                third_count=counts[8],
                outside_count=counts[9],
                win_rate=float(values[14]) / 100.0,
                top2_rate=float(values[15]) / 100.0,
                top3_rate=float(values[16]) / 100.0,
                gear=float(values[17]),
                comment=values[18].strip(),
            )
        )

    numbers = [rider.number for rider in riders]
    if len(numbers) != len(set(numbers)):
        raise ValueError("車番が重複しています。PDFの読み取りに失敗しました。")
    lines, warnings = infer_lines(riders)
    warnings = profile_warnings + warnings
    race_id, venue, race_number, race_class, distance = _metadata(text, path.name)
    return RaceInput(
        race_id=race_id,
        venue=venue,
        race_number=race_number,
        race_class=race_class,
        distance_m=distance,
        riders=tuple(sorted(riders, key=lambda rider: rider.number)),
        lines=tuple(tuple(line) for line in lines),
        line_warnings=tuple(warnings),
    )


def _normalized(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value or ""))


def _finish_value(token: str) -> int | None:
    normalized = _normalized(token)
    if re.fullmatch(r"[1-9]", normalized):
        return int(normalized)
    return None


def _recent_score(finishes: list[int]) -> float | None:
    if not finishes:
        return None
    weights = [0.86 ** index for index in range(len(finishes))]
    values = [(10.0 - finish) / 9.0 for finish in finishes]
    return round(100.0 * sum(weight * value for weight, value in zip(weights, values, strict=True)) / sum(weights), 3)


def parse_recent_pdf(path: Path, race: RaceInput) -> dict[str, Any]:
    """Parse the netkeirin '直近成績' print view using PDF word coordinates.

    The view has the same filename as the basic racecard.  Rider results sit above
    each rider name, so coordinate bands are more reliable than line-oriented OCR.
    """
    text = pdftotext(path)
    compact = re.sub(r"\s+", "", text)
    if not ("今節" in compact and "直近1" in compact and "直近2" in compact):
        raise ValueError("直近成績PDFとして認識できません。『直近成績』タブを開いて保存してください。")

    rider_data: dict[int, dict[str, Any]] = {}
    warnings: list[str] = []
    target_names = {_normalized(rider.name): rider for rider in race.riders}

    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(x_tolerance=1, y_tolerance=2, keep_blank_chars=False)
            normalized_words = [(_normalized(word.get("text", "")), word) for word in words]
            for normalized_name, rider in target_names.items():
                if rider.number in rider_data:
                    continue
                name_word = next(
                    (
                        word
                        for value, word in normalized_words
                        if value == normalized_name and 170.0 <= float(word["x0"]) <= 280.0
                    ),
                    None,
                )
                if not name_word:
                    continue
                name_top = float(name_word["top"])
                result_words = [
                    word
                    for _, word in normalized_words
                    if float(word["x0"]) >= 285.0
                    and name_top - 90.0 <= float(word["top"]) <= name_top - 18.0
                ]
                result_words.sort(key=lambda word: (float(word["x0"]), float(word["top"])))
                raw_tokens = [_normalized(word.get("text", "")) for word in result_words]
                finishes = [value for token in raw_tokens if (value := _finish_value(token)) is not None]
                statuses = [token for token in raw_tokens if token in {"棄", "失", "欠", "落", "故", "再"}]
                rider_data[rider.number] = {
                    "name": rider.name,
                    "page": page_number,
                    "finishes": finishes,
                    "non_finish_statuses": statuses,
                    "recent_form_score": _recent_score(finishes),
                    "audit_tokens": raw_tokens,
                }

    for rider in race.riders:
        item = rider_data.get(rider.number)
        if not item:
            warnings.append(f"{rider.number}番 {rider.name}: 直近成績欄を取得できませんでした。")
            continue
        if not item["finishes"] and not item["non_finish_statuses"]:
            warnings.append(f"{rider.number}番 {rider.name}: 着順を取得できませんでした。")

        finishes = item["finishes"]
        latest = finishes[0] if finishes else None
        aggression = (rider.back_count + rider.start_count + 2.0) / (rider.races + 8.0)
        if latest is not None and latest >= 4 and rider.style in {"逃", "両"}:
            # A deterministic proxy only: an aggressive rider's recent loss may
            # contain more performance than the finishing position suggests.
            proxy = 25.0 + 40.0 * aggression + 25.0 * rider.top3_rate
        else:
            proxy = 0.0
        item["lost_strong_proxy"] = {
            "score": round(min(100.0, proxy), 3),
            "basis": "直近着順×基本情報のS/B・脚質による代理点（映像不利は未判定）",
            "verified_video_interference": False,
        }

    identity = document_identity(path)
    status = "complete" if len(rider_data) == len(race.riders) and not warnings else "partial"
    return {
        "status": status,
        "source": "uploaded_netkeirin_recent_pdf",
        "target_race_id": identity["race_id"],
        "venue": identity["venue"],
        "race_number": identity["race_number"],
        "riders": rider_data,
        "warnings": warnings,
        "claim": "PDF内の着順と基本情報による代理評価。接触・牽制・コース取り等は未取得。",
    }


def _name_target(comment: str, rider: Rider, candidates: list[Rider]) -> Rider | None:
    possible: list[tuple[int, Rider]] = []
    for candidate in candidates:
        if candidate.number == rider.number:
            continue
        for length in range(min(4, len(candidate.name)), 1, -1):
            prefix = candidate.name[:length]
            if prefix in comment:
                possible.append((length, candidate))
                break
    return max(possible, default=(0, None), key=lambda item: item[0])[1]


def infer_lines(riders: list[Rider]) -> tuple[list[list[int]], list[str]]:
    by_number = {rider.number: rider for rider in riders}
    parent: dict[int, int] = {}
    warnings: list[str] = []

    for rider in riders:
        target = _name_target(rider.comment, rider, riders)
        if target:
            parent[rider.number] = target.number

    def root(number: int) -> int:
        seen: set[int] = set()
        while number in parent and number not in seen:
            seen.add(number)
            number = parent[number]
        return number

    def chains() -> dict[int, list[int]]:
        grouped: dict[int, list[tuple[int, int]]] = {}
        for number in by_number:
            current, depth, seen = number, 0, set()
            while current in parent and current not in seen:
                seen.add(current)
                current = parent[current]
                depth += 1
            grouped.setdefault(current, []).append((depth, number))
        return {key: [n for _, n in sorted(value)] for key, value in grouped.items()}

    region_words = tuple(sorted(set(PREFECTURE_REGION.values()), key=len, reverse=True))
    region_aliases = {"南関東": ("南関東", "南関")}
    for rider in riders:
        if rider.number in parent:
            continue
        requested = next(
            (
                region
                for region in region_words
                if any(
                    f"{alias}勢" in rider.comment
                    or (alias in rider.comment and re.search(r"[23２３]", rider.comment))
                    for alias in region_aliases.get(region, (region,))
                )
            ),
            None,
        )
        if not requested:
            continue
        groups = chains()
        scored: list[tuple[int, int, list[int]]] = []
        for group_root, members in groups.items():
            if rider.number in members:
                continue
            matches = sum(PREFECTURE_REGION.get(by_number[number].prefecture) == requested for number in members)
            if matches:
                scored.append((matches, len(members), members))
        if scored:
            members = max(scored, key=lambda item: (item[0], item[1]))[2]
            parent[rider.number] = members[-1]
        else:
            warnings.append(f"{rider.number}番 {rider.name}: 『{requested}勢』の接続先を特定できません。")

    groups = list(chains().values())
    groups.sort(key=lambda line: line[0])
    for rider in riders:
        if rider.style == "追" and len(next(line for line in groups if rider.number in line)) == 1 and "単騎" not in rider.comment:
            warnings.append(f"{rider.number}番 {rider.name}: ライン接続を特定できず単騎扱いです。")
    return groups, warnings


ODDS_ROW_RE = re.compile(
    r"(?m)^\s*(\d{1,3})\s+([1-9])\s+([0-9]+(?:\.[0-9]+)?)\s*$\n"
    r"\s*([^\d\n][^\n]*)\n\s*([1-9])\s*$\n\s*([^\d\n][^\n]*)"
)


def parse_odds_pdf(path: Path, rider_numbers: list[int]) -> dict[tuple[int, int], float]:
    text = pdftotext(path)
    start_marker = "人気選択組み合わせ オッズ"
    if start_marker not in text:
        raise ValueError("2車単の人気順オッズ表を取得できません。")
    block = text[text.index(start_marker):]
    end = block.find("※結果・成績・オッズ")
    if end >= 0:
        block = block[:end]
    odds: dict[tuple[int, int], float] = {}
    for match in ODDS_ROW_RE.finditer(block):
        first, second = int(match.group(2)), int(match.group(5))
        odds[(first, second)] = float(match.group(3))
    expected = {(first, second) for first in rider_numbers for second in rider_numbers if first != second}
    missing = expected - set(odds)
    extra = set(odds) - expected
    if missing or extra:
        raise ValueError(
            f"2車単オッズが完全ではありません（取得={len(odds)}、期待={len(expected)}、不足={len(missing)}、余分={len(extra)}）。"
        )
    return odds


def _ocr_column(image_path: Path, box: tuple[int, int, int, int]) -> list[dict[str, float | int | None]]:
    try:
        from PIL import Image, ImageEnhance
    except ImportError as exc:
        raise ValueError("EX画像の解析にはPillowが必要です。") from exc
    if shutil.which("tesseract") is None:
        raise ValueError("EX画像の解析に必要なtesseractが見つかりません。")
    image = Image.open(image_path).convert("L")
    cropped = image.crop(box)
    cropped = cropped.resize((cropped.width * 4, cropped.height * 4))
    cropped = ImageEnhance.Contrast(cropped).enhance(2.0)
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        cropped.save(tmp.name)
        result = subprocess.run(
            ["tesseract", tmp.name, "stdout", "-l", "eng", "--psm", "6"],
            check=True,
            capture_output=True,
            text=True,
        )
    rows: list[dict[str, float | int | None]] = []
    pending_rate: float | None = None
    for line in result.stdout.splitlines():
        rate_match = re.search(r"(\d{1,3})%", line)
        if rate_match:
            pending_rate = float(rate_match.group(1)) / 100.0
        sample_match = re.search(r"\((\d+)\s*/\s*(\d+)\)", line)
        if not sample_match:
            continue
        success, total = int(sample_match.group(1)), int(sample_match.group(2))
        rate = pending_rate if total > 0 else None
        if total > 0 and rate is None:
            rate = success / total
        rows.append({"rate": rate, "success": success, "total": total})
        pending_rate = None
    return rows


def parse_ex_image(path: Path, rider_numbers: list[int]) -> tuple[dict[int, dict[str, Any]], list[str]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("EX画像の解析にはPillowが必要です。") from exc
    width, height = Image.open(path).size
    # WINTICKETのEXデータ全画面スクリーンショット。座標は画面比率で指定する。
    x_ranges = [(0.275, 0.449), (0.449, 0.624), (0.624, 0.791), (0.791, 0.959)]
    y0, y1 = int(height * 0.397), int(height * 0.775)
    columns = [
        _ocr_column(path, (int(width * left), y0, int(width * right), y1))
        for left, right in x_ranges
    ]
    warnings: list[str] = []
    if any(len(column) != len(rider_numbers) for column in columns):
        counts = [len(column) for column in columns]
        raise ValueError(f"EX画像の行数を確定できません（列ごとの取得行数={counts}）。")
    names = ["kamashi", "tsuppari", "chigiri", "chigirareru"]
    result: dict[int, dict[str, Any]] = {}
    for row_index, number in enumerate(sorted(rider_numbers)):
        result[number] = {name: columns[column_index][row_index] for column_index, name in enumerate(names)}
    warnings.append("EX画像はOCR取得です。抽出した分子・分母をJSONで確認してください。")
    return result, warnings


def parse_ex_data(path: Path, rider_numbers: list[int]) -> tuple[dict[int, dict[str, Any]], list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return parse_ex_image(path, rider_numbers)
    raise ValueError("現在のEXデータはPNG/JPG/WebP画像に対応しています。")


EX_FIELD_ALIASES = {
    "kamashi": {"かまし", "かまし成功率", "kamashi"},
    "tsuppari": {"つっぱり", "つっぱり成功率", "tsuppari"},
    "chigiri": {"ちぎり", "ちぎり率", "chigiri"},
    "chigirareru": {"ちぎられ", "ちぎられ率", "chigirareru"},
}


def _ex_cell(value: str) -> dict[str, float | int | None] | None:
    normalized = unicodedata.normalize("NFKC", value).strip()
    if normalized in {"", "-", "--", "—", "未取得", "なし", "null", "None"}:
        return None
    rate_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", normalized)
    sample_match = re.search(r"\(?\s*(\d+)\s*/\s*(\d+)\s*\)?", normalized)
    if not rate_match and not sample_match:
        return None
    success = int(sample_match.group(1)) if sample_match else None
    total = int(sample_match.group(2)) if sample_match else None
    if total == 0:
        rate = None
    elif rate_match:
        rate = float(rate_match.group(1)) / 100.0
    elif success is not None and total:
        rate = success / total
    else:
        rate = None
    return {"rate": rate, "success": success, "total": total}


def _ex_columns(line: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", line).strip()
    if not normalized:
        return []
    if re.search(r"[,\t、]", normalized):
        return [part.strip() for part in re.split(r"\s*[,\t、]\s*", normalized)]
    return re.split(r"\s+", normalized)


def parse_ex_text(text: str, rider_numbers: list[int]) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """Parse optional pasted EX data without making prediction completion depend on it.

    Header order, extra columns and missing values may vary. Invalid rows are
    ignored with warnings; they never make the required two-PDF pipeline fail.
    """
    warnings: list[str] = []
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return {}, warnings

    header_index = next(
        (index for index, line in enumerate(lines) if "車番" in unicodedata.normalize("NFKC", line)),
        None,
    )
    header = _ex_columns(lines[header_index]) if header_index is not None else []
    normalized_header = [re.sub(r"\s+", "", value).lower() for value in header]

    number_index: int | None = None
    field_indices: dict[str, int] = {}
    if header:
        for index, value in enumerate(normalized_header):
            if value in {"車番", "車", "番号", "選手番号"}:
                number_index = index
            for field, aliases in EX_FIELD_ALIASES.items():
                if value in aliases:
                    field_indices[field] = index

    data_lines = lines[(header_index + 1) if header_index is not None else 0:]
    output: dict[int, dict[str, Any]] = {}
    valid_numbers = set(rider_numbers)
    fields = ["kamashi", "tsuppari", "chigiri", "chigirareru"]
    for row_number, line in enumerate(data_lines, start=1):
        columns = _ex_columns(line)
        if not columns:
            continue
        if number_index is None:
            # Supported headerless layouts:
            # frame,number,name,4 metrics / number,name,4 metrics / number,4 metrics
            if len(columns) >= 7 and columns[0].isdigit() and columns[1].isdigit():
                row_number_index, inferred = 1, {field: 3 + i for i, field in enumerate(fields)}
            elif len(columns) >= 6 and columns[0].isdigit():
                row_number_index, inferred = 0, {field: 2 + i for i, field in enumerate(fields)}
            elif len(columns) >= 5 and columns[0].isdigit():
                row_number_index, inferred = 0, {field: 1 + i for i, field in enumerate(fields)}
            else:
                warnings.append(f"EX {row_number}行目: 車番と項目を特定できず無視しました。")
                continue
            indices = inferred
        else:
            row_number_index, indices = number_index, field_indices
        if row_number_index >= len(columns) or not columns[row_number_index].isdigit():
            warnings.append(f"EX {row_number}行目: 車番を読めず無視しました。")
            continue
        number = int(columns[row_number_index])
        if number not in valid_numbers:
            warnings.append(f"EX {row_number}行目: 出走表にない車番{number}を無視しました。")
            continue
        item: dict[str, Any] = {}
        for field, index in indices.items():
            if index < len(columns):
                parsed = _ex_cell(columns[index])
                if parsed is not None:
                    item[field] = parsed
        if item:
            output[number] = item

    if not output:
        warnings.append("EX文字データから有効な数値を取得できなかったため、EXなしで計算しました。")
    else:
        missing = sorted(valid_numbers - set(output))
        if missing:
            warnings.append(f"EX未取得の車番: {', '.join(map(str, missing))}。取得分だけ使用しました。")
    return output, warnings


def parse_result_pdf(path: Path) -> dict[str, Any]:
    text = pdftotext(path)
    first_match = re.search(r"1着\s+[1-6]\s+([1-9])", text)
    second_match = re.search(r"2着\s+[1-6]\s+([1-9])", text)
    payout_match = re.search(r"２車単\s*([1-9])\s*>\s*([1-9])\s*([0-9,]+)円", text)
    if not first_match or not second_match or not payout_match:
        raise ValueError("結果PDFから1着・2着・2車単払戻を取得できません。")
    first, second = int(first_match.group(1)), int(second_match.group(1))
    payout_first, payout_second = int(payout_match.group(1)), int(payout_match.group(2))
    if (first, second) != (payout_first, payout_second):
        raise ValueError("着順と2車単払戻の組み合わせが一致しません。")
    payout_yen = int(payout_match.group(3).replace(",", ""))
    return {
        "actual_first": first,
        "actual_second": second,
        "actual_payout_yen_per_100": payout_yen,
        "actual_payout_odds": payout_yen / 100.0,
    }
