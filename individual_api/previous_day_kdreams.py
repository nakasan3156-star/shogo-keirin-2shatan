"""Kドリームスの前日結果からPR31前日特徴と実測済み展開不利ラベルを作る。

初日は外部取得しない。取得失敗は本体予測を止めず、前日特徴を未取得として返す。
"""
from __future__ import annotations

import re
import unicodedata
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

PR31_PATTERNS = {
    "A": r"先行争|叩き合|踏み合|叩き叩かれ",
    "B": r"先行|逃げ|突張|突っ張|カマシ|ペース駆け|正攻法逃げ",
    "C": r"番手飛|競り|競勝|番手奪|競負|競負け",
    "D": r"前不発|目標が不発|目標共倒れ|不発ライン|前が不発|目標不発",
    "E": r"後方置かれ|最後方|後方|後手ライン|後手",
    "F": r"牽制|進路|詰まり|阻ま|張られ|捌かれ|包ま|コース無|コースなく",
    "G": r"位置取|追上|追い上|斬り込|脚使|踏まされ",
    "H": r"再仕掛|立て直|仕掛け直",
    "I": r"脚余|仕掛け遅|余し|届かず|届かない|届かぬ",
}


def _norm(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value or ""))


def detect_day_no(text: str) -> int:
    t = unicodedata.normalize("NFKC", text or "")
    if "初日" in t:
        return 1
    m = re.search(r"(\d+)日目", t)
    if m:
        return int(m.group(1))
    if "最終日" in t:
        return 3
    return 1


def _header_index(headers: list[str], needles: tuple[str, ...]) -> int | None:
    for i, h in enumerate(headers):
        if any(n in h for n in needles):
            return i
    return None


def _result_rows(html: str, rider_names: list[str]) -> dict[str, dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    wanted = {_norm(name): name for name in rider_names}
    found: dict[str, dict[str, Any]] = {}
    for table in soup.find_all("table"):
        headers = [_norm(x.get_text(" ", strip=True)) for x in table.find_all("th")]
        if not headers or not any("着" in h or "順位" in h for h in headers):
            continue
        name_i = _header_index(headers, ("選手名", "選手"))
        finish_i = _header_index(headers, ("着順", "順位", "着"))
        back_i = _header_index(headers, ("バック", "B"))
        comment_i = _header_index(headers, ("コメント", "短評"))
        for tr in table.find_all("tr"):
            cells = [_norm(x.get_text(" ", strip=True)) for x in tr.find_all(["td", "th"])]
            if not cells:
                continue
            joined = "".join(cells)
            match_key = next((key for key in wanted if key and key in joined), None)
            if not match_key:
                continue
            finish = None
            if finish_i is not None and finish_i < len(cells):
                m = re.search(r"[1-9]", cells[finish_i])
                finish = int(m.group()) if m else None
            if finish is None:
                for cell in cells[:3]:
                    if re.fullmatch(r"[1-9]", cell):
                        finish = int(cell)
                        break
            actual_back = 0
            if back_i is not None and back_i < len(cells):
                actual_back = int(bool(re.search(r"1|B|バック", cells[back_i])))
            comment = cells[comment_i] if comment_i is not None and comment_i < len(cells) else ""
            found[wanted[match_key]] = {
                "finish": finish,
                "actual_back": actual_back,
                "comment": comment,
            }
    # コメントが別表の場合は選手名周辺の本文も補助的に使う。
    plain = _norm(soup.get_text(" ", strip=True))
    for original in rider_names:
        item = found.setdefault(original, {"finish": None, "actual_back": 0, "comment": ""})
        if not item["comment"]:
            key = _norm(original)
            pos = plain.find(key)
            if pos >= 0:
                item["comment"] = plain[max(0, pos - 120): pos + len(key) + 280]
    return found


def _labels(item: dict[str, Any]) -> dict[str, Any]:
    finish = item.get("finish")
    comment = str(item.get("comment") or "")
    back = int(item.get("actual_back") or 0)
    lost4 = finish is not None and int(finish) >= 4
    pr31 = {k: int(bool(lost4 and re.search(pat, comment))) for k, pat in PR31_PATTERNS.items()}
    # PR31のBだけは元仕様どおり「長く踏んでB取得かつ3着内」。
    pr31["B"] = int(bool(finish is not None and int(finish) <= 3 and back == 1 and re.search(PR31_PATTERNS["B"], comment)))
    validated = {
        "bandte_fight_4plus": bool(lost4 and re.search(PR31_PATTERNS["C"], comment)),
        "blocked_4plus": bool(lost4 and re.search(PR31_PATTERNS["F"], comment)),
        # 別線判定には前日ラインが必要。Kドリームス表から確定できない時はFalse固定で創作しない。
        "back_4plus_otherline_win": False,
    }
    return {"pr31": pr31, "validated": validated}


def fetch_previous_day(venue: str | None, race_date: str | None, day_no: int, rider_names: list[str]) -> dict[str, Any]:
    if day_no <= 1:
        return {"status": "FIRST_DAY_SKIPPED", "source": "KDreams", "riders": {}}
    if not venue or venue not in VENUES or not race_date:
        return {"status": "IDENTITY_UNAVAILABLE", "source": "KDreams", "riders": {}}
    try:
        current = datetime.strptime(race_date, "%Y-%m-%d")
    except ValueError:
        return {"status": "DATE_INVALID", "source": "KDreams", "riders": {}}
    previous = current - timedelta(days=1)
    code, slug = VENUES[venue]
    date_token = previous.strftime("%Y%m%d")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 ShogoKeirinOS/PR31"})
    merged: dict[str, dict[str, Any]] = {}
    urls: list[str] = []
    # 同一場同日に通常は1開催。event sequenceはサイト側の内部番号なので1〜3だけ安全に探索。
    for seq in range(1, 4):
        for race_no in range(1, 13):
            race_id = f"{code}{date_token}{seq:02d}{race_no:04d}"
            url = f"https://keirin.kdreams.jp/{slug}/racedetail/{race_id}/?pageType=result"
            try:
                response = session.get(url, timeout=2.5)
            except requests.RequestException:
                continue
            if response.status_code != 200 or previous.strftime("%Y") not in response.text:
                continue
            rows = _result_rows(response.text, rider_names)
            matched = {name: row for name, row in rows.items() if row.get("finish") is not None or row.get("comment")}
            if not matched:
                continue
            urls.append(url)
            merged.update(matched)
            if len(merged) >= len(rider_names):
                break
        if len(merged) >= len(rider_names):
            break
    out = {}
    for name in rider_names:
        item = merged.get(name)
        if not item:
            continue
        out[name] = {**item, **_labels(item)}
    return {
        "status": "OK" if out else "PREVIOUS_DAY_NOT_FOUND",
        "source": "KDreams",
        "previous_date": previous.strftime("%Y-%m-%d"),
        "urls_checked_with_match": urls,
        "riders": out,
    }
