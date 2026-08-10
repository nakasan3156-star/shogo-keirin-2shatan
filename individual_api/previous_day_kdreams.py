"""Kドリームスから同一開催の直前日だけを読み、PR31前日特徴を作る。

現在レースの結果表は解析しない。現在レース出走表に載る「前回出走レースの成績」から
直前日のレース番号を特定し、その前日レース詳細だけを取得する。
初日は外部取得しない。取得失敗は本体予測を止めず、未取得として返す。
"""
from __future__ import annotations

import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

import requests
from bs4 import BeautifulSoup

VENUES = {
    "函館": ("11", "hakodate"), "青森": ("12", "aomori"), "いわき平": ("13", "iwakitaira"),
    "弥彦": ("21", "yahiko"), "前橋": ("22", "maebashi"), "取手": ("23", "toride"),
    "宇都宮": ("24", "utsunomiya"), "大宮": ("25", "omiya"), "西武園": ("26", "seibuen"),
    "京王閣": ("27", "keiokaku"), "立川": ("28", "tachikawa"), "松戸": ("31", "matsudo"),
    "千葉": ("32", "chiba"), "川崎": ("34", "kawasaki"), "平塚": ("35", "hiratsuka"),
    "小田原": ("36", "odawara"), "伊東": ("37", "ito"), "静岡": ("38", "shizuoka"),
    "名古屋": ("42", "nagoya"), "岐阜": ("43", "gifu"), "大垣": ("44", "ogaki"),
    "豊橋": ("45", "toyohashi"), "富山": ("46", "toyama"), "松阪": ("47", "matsusaka"),
    "四日市": ("48", "yokkaichi"), "福井": ("51", "fukui"), "奈良": ("53", "nara"),
    "向日町": ("54", "mukomachi"), "和歌山": ("55", "wakayama"), "岸和田": ("56", "kishiwada"),
    "玉野": ("61", "tamano"), "広島": ("62", "hiroshima"), "防府": ("63", "hofu"),
    "高松": ("71", "takamatsu"), "小松島": ("73", "komatsushima"), "高知": ("74", "kochi"),
    "松山": ("75", "matsuyama"), "小倉": ("81", "kokura"), "久留米": ("83", "kurume"),
    "武雄": ("84", "takeo"), "佐世保": ("85", "sasebo"), "別府": ("86", "beppu"),
    "熊本": ("87", "kumamoto"),
}

# PR #31の定義を一字一句同じ条件に固定する。
PR31_PATTERNS = {
    "A": r"先行争|叩き合|踏み合|叩き叩かれ",
    "B": r"先行|逃げ|突張|カマシ逃げ|ペース駆け|正攻法逃げ",
    "C": r"番手飛|競り|競勝|番手奪",
    "D": r"前不発|目標が不発|目標共倒れ|不発ライン|前が不発",
    "E": r"後方置かれ|最後方|後方|後手ライン",
    "F": r"牽制|進路|詰まり|阻ま|張られ|捌かれ",
    "G": r"位置取|追上|斬り込|脚使",
    "H": r"再仕掛|立て直|仕掛け直",
    "I": r"脚余|仕掛け遅|余し|届かず",
}

# 2025H1→H2→2026-01/02で再現した追加表示用ラベル。PR31へ二重加点はしない。
VALIDATED_PATTERNS = {
    "bandte_fight_4plus": r"番手飛|競り|競勝|番手奪|競負|競負け",
    "blocked_4plus": r"牽制|進路|詰まり|阻ま|張られ|捌かれ|包ま|コース無|コースなく",
}


def _norm(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value or ""))


def detect_day_no(text: str) -> int:
    """初日/数字日をPDFヘッダから返す。最終日は0としてKドリームス側で実日数を解決する。"""
    t = unicodedata.normalize("NFKC", text or "")
    head = "\n".join(t.splitlines()[:40])
    if re.search(r"(?:^|\n).{0,60}?初日(?:\s|$)", head):
        return 1
    m = re.search(r"(?:^|\n).{0,60}?([2-6])日目(?:\s|$)", head)
    if m:
        return int(m.group(1))
    if "最終日" in head:
        return 0
    # ヘッダ崩れ時だけ全体を見る。数字日を優先する。
    m = re.search(r"([2-6])日目", t)
    if m:
        return int(m.group(1))
    if "最終日" in t[:3000]:
        return 0
    return 1


def _fetch_html(url: str) -> str:
    try:
        response = requests.get(
            url,
            timeout=(1.5, 2.5),
            headers={"User-Agent": "Mozilla/5.0 ShogoKeirinOS/PR31"},
        )
    except requests.RequestException:
        return ""
    return response.text if response.status_code == 200 else ""


def _previous_summary(html: str, rider_names: list[str]) -> dict[str, dict[str, Any]]:
    """現在レース出走表の『前回出走レースの成績』だけを解析する。現在結果表は触らない。"""
    if not html:
        return {}
    soup = BeautifulSoup(html, "lxml")
    wanted = {_norm(name): name for name in rider_names}
    found: dict[str, dict[str, Any]] = {}
    for table in soup.find_all("table"):
        table_text = _norm(table.get_text(" ", strip=True))
        if "前回出走レースの成績" not in table_text and "走り評" not in table_text:
            continue
        for tr in table.find_all("tr"):
            row = _norm(tr.get_text(" ", strip=True))
            match_name = next((key for key in wanted if key and key in row), None)
            if not match_name:
                continue
            # 例: 望月湧世 ... 初日 11R 8 叩き捲られ 詳細
            m = re.search(
                r"(初日|[2-6]日目|最終日)(\d{1,2})R(落|失|棄|故|[1-9])([^0-9]{0,80}?)(?:詳細|$)",
                row,
            )
            if not m:
                continue
            finish_raw = m.group(3)
            review = re.sub(r"詳細$", "", m.group(4)).strip("・- ")
            found[wanted[match_name]] = {
                "previous_day_label": m.group(1),
                "previous_race_no": int(m.group(2)),
                "finish": int(finish_raw) if finish_raw.isdigit() else None,
                "accident_finish": "" if finish_raw.isdigit() else finish_raw,
                "short_review": review,
            }
    return found


def _result_detail(html: str, rider_names: list[str]) -> dict[str, dict[str, Any]]:
    """完了済み前日レースのresult_table相当から着順・S/B・上がり・勝敗因だけを読む。"""
    if not html:
        return {}
    soup = BeautifulSoup(html, "lxml")
    wanted = {_norm(name): name for name in rider_names}
    found: dict[str, dict[str, Any]] = {}
    tables = soup.select("table.result_table") or soup.find_all("table")
    for table in tables:
        compact = _norm(table.get_text(" ", strip=True))
        if "勝敗因" not in compact or "着順" not in compact:
            continue
        header_row = None
        headers: list[str] = []
        for tr in table.find_all("tr"):
            cells = [_norm(x.get_text(" ", strip=True)) for x in tr.find_all(["th", "td"], recursive=False)]
            if "着順" in cells and any("選手名" in c for c in cells):
                header_row = tr
                headers = cells
                break
        if not headers:
            # Kドリームス標準 result_table の固定列。
            headers = ["予想", "着順", "車番", "選手名", "着差", "上り", "決まり手", "S/B", "勝敗因"]

        def idx(*names: str) -> int | None:
            for i, h in enumerate(headers):
                if any(name in h for name in names):
                    return i
            return None

        finish_i = idx("着順")
        car_i = idx("車番")
        name_i = idx("選手名")
        lap_i = idx("上り")
        sb_i = idx("S/B", "S／B")
        reason_i = idx("勝敗因")
        required = [finish_i, car_i, name_i, lap_i, reason_i]
        if any(i is None for i in required):
            continue
        for tr in table.find_all("tr"):
            if tr is header_row:
                continue
            cells = [_norm(x.get_text(" ", strip=True)) for x in tr.find_all("td", recursive=False)]
            if len(cells) <= max(int(i) for i in required if i is not None):
                continue
            cell_name = cells[int(name_i)]
            match_name = next((key for key in wanted if key and key in cell_name), None)
            if not match_name:
                continue
            finish_raw = cells[int(finish_i)]
            car_raw = cells[int(car_i)]
            lap_raw = cells[int(lap_i)]
            sb = cells[int(sb_i)] if sb_i is not None and int(sb_i) < len(cells) else ""
            try:
                lap = float(lap_raw)
            except (TypeError, ValueError):
                lap = None
            found[wanted[match_name]] = {
                "finish": int(finish_raw) if finish_raw.isdigit() else None,
                "car_no": int(car_raw) if car_raw.isdigit() else None,
                "actual_start": int("S" in sb.upper()),
                "actual_back": int("B" in sb.upper()),
                "final_lap_time": lap,
                "comment": cells[int(reason_i)],
            }
    return found


def _labels(item: dict[str, Any]) -> dict[str, Any]:
    finish = item.get("finish")
    comment = str(item.get("comment") or item.get("short_review") or "")
    back = int(item.get("actual_back") or 0)
    pr31 = {k: int(bool(re.search(pat, comment))) for k, pat in PR31_PATTERNS.items()}
    # PR31のBだけは「長く踏んでB取得かつ3着内」に限定。
    pr31["B"] = int(bool(pr31["B"] and finish is not None and int(finish) <= 3 and back == 1))
    lost4 = finish is not None and int(finish) >= 4
    validated = {
        key: bool(lost4 and re.search(pat, comment))
        for key, pat in VALIDATED_PATTERNS.items()
    }
    # 前日ラインを確定できない場合は創作しない。
    validated["back_4plus_otherline_win"] = False
    return {"pr31": pr31, "validated": validated}


def _current_race_url(code: str, slug: str, current: datetime, day_no: int, race_no: int) -> tuple[str, datetime]:
    start = current - timedelta(days=day_no - 1)
    rid = f"{code}{start.strftime('%Y%m%d')}{day_no:02d}{race_no:04d}"
    return f"https://keirin.kdreams.jp/{slug}/racedetail/{rid}/?pageType=showResult", start


def fetch_previous_day(
    venue: str | None,
    race_date: str | None,
    day_no: int,
    race_no: int,
    rider_names: list[str],
) -> dict[str, Any]:
    if day_no == 1:
        return {"status": "FIRST_DAY_SKIPPED", "source": "KDreams", "resolved_day_no": 1, "riders": {}}
    if not venue or venue not in VENUES or not race_date or race_no <= 0:
        return {"status": "IDENTITY_UNAVAILABLE", "source": "KDreams", "resolved_day_no": max(day_no, 1), "riders": {}}
    try:
        current = datetime.strptime(race_date, "%Y-%m-%d")
    except ValueError:
        return {"status": "DATE_INVALID", "source": "KDreams", "resolved_day_no": max(day_no, 1), "riders": {}}

    code, slug = VENUES[venue]
    candidate_days = [day_no] if day_no >= 2 else [2, 3, 4, 5, 6]
    best: tuple[int, str, datetime, dict[str, dict[str, Any]]] | None = None
    for candidate in candidate_days:
        current_url, start = _current_race_url(code, slug, current, candidate, race_no)
        html = _fetch_html(current_url)
        summary = _previous_summary(html, rider_names)
        score = len(summary)
        if score and (best is None or score > len(best[3])):
            best = (candidate, current_url, start, summary)
        if score >= max(1, len(rider_names) - 1):
            break

    if best is None:
        return {
            "status": "PREVIOUS_DAY_NOT_FOUND",
            "source": "KDreams",
            "resolved_day_no": max(day_no, 2 if day_no == 0 else day_no),
            "previous_date": (current - timedelta(days=1)).strftime("%Y-%m-%d"),
            "riders": {},
        }

    resolved_day, current_url, start, summary = best
    previous_day_no = resolved_day - 1
    previous_urls: dict[int, str] = {}
    for item in summary.values():
        prev_race = int(item["previous_race_no"])
        rid = f"{code}{start.strftime('%Y%m%d')}{previous_day_no:02d}{prev_race:04d}"
        previous_urls[prev_race] = f"https://keirin.kdreams.jp/{slug}/racedetail/{rid}/?pageType=showResult"

    detail_by_name: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(previous_urls)))) as pool:
        futures = {
            pool.submit(_fetch_html, url): (race, url)
            for race, url in previous_urls.items()
        }
        for future in as_completed(futures):
            race, url = futures[future]
            html = future.result()
            target_names = [name for name, item in summary.items() if int(item["previous_race_no"]) == race]
            parsed = _result_detail(html, target_names)
            for name, item in parsed.items():
                item["previous_detail_url"] = url
                detail_by_name[name] = item

    riders: dict[str, dict[str, Any]] = {}
    for name, base in summary.items():
        detail = detail_by_name.get(name, {})
        item = {**base, **detail}
        # 詳細の勝敗因が取れない場合のみ走り評をPR31コメントに使う。
        item["comment"] = str(detail.get("comment") or base.get("short_review") or "")
        item.update(_labels(item))
        riders[name] = item

    return {
        "status": "OK" if riders else "PREVIOUS_DAY_NOT_FOUND",
        "source": "KDreams",
        "resolved_day_no": resolved_day,
        "current_race_url": current_url,
        "previous_date": (current - timedelta(days=1)).strftime("%Y-%m-%d"),
        "riders": riders,
    }
