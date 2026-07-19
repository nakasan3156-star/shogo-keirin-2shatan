from __future__ import annotations

import hashlib
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from lxml import html
from pypdf import PdfReader


NETKEIRIN_HOST = "keirin.netkeiba.com"
RECENT_URL = "https://keirin.netkeiba.com/race/entry/results.html"
DETAIL_URL = "https://keirin.netkeiba.com/db/result/"
RACE_ID_RE = re.compile(r"(?<!\d)(\d{12})(?!\d)")


class HistoryFetchError(RuntimeError):
    pass


class CachedHttpClient:
    """Private research fetcher with host allow-list, cache and gentle pacing."""

    def __init__(self, cache_dir: Path | None = None, min_interval_seconds: float = 0.5):
        configured = os.getenv("SHOGO_KEIRIN_CACHE_DIR")
        self.cache_dir = cache_dir or Path(configured or "/tmp/shogo-keirin-history-cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._last_request = 0.0
        self.requests = 0
        self.cache_hits = 0

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != NETKEIRIN_HOST:
            raise HistoryFetchError("許可されていない取得先です。")

    def get(self, url: str) -> bytes:
        self._validate_url(url)
        cache_path = self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.html"
        if cache_path.exists() and cache_path.stat().st_size > 1000:
            self.cache_hits += 1
            return cache_path.read_bytes()

        wait = self.min_interval_seconds - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ShogoKeirinResearch/0.2; private-use)",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "ja,en;q=0.7",
            },
        )
        try:
            with urlopen(request, timeout=20) as response:
                body = response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise HistoryFetchError(f"公開ページを取得できませんでした: {exc}") from exc
        self._last_request = time.monotonic()
        self.requests += 1
        if len(body) < 1000:
            raise HistoryFetchError("取得ページが短すぎるため採用しませんでした。")
        cache_path.write_bytes(body)
        return body


def extract_race_id_from_pdf(path: Path, race_number: int) -> str:
    """Read link annotations only. Never guesses a race id from the filename."""
    candidates: list[str] = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        for annotation_ref in page.get("/Annots", []):
            annotation = annotation_ref.get_object()
            uri = annotation.get("/A", {}).get("/URI")
            if not uri:
                continue
            parsed = urlparse(str(uri))
            if parsed.hostname != NETKEIRIN_HOST:
                continue
            values = parse_qs(parsed.query).get("race_id", [])
            for value in values:
                if RACE_ID_RE.fullmatch(value) and value.endswith(f"{race_number:02d}"):
                    candidates.append(value)
    if not candidates:
        raise HistoryFetchError("出走表PDFから対象レースIDを取得できませんでした。")
    return Counter(candidates).most_common(1)[0][0]


def _clean(value: str) -> str:
    return " ".join(value.split())


def parse_recent_page(body: bytes) -> list[dict[str, Any]]:
    document = html.fromstring(body)
    riders: list[dict[str, Any]] = []
    for row in document.xpath("//tr[starts-with(@id,'player-')]"):
        row_id = row.get("id", "")
        number_match = re.fullmatch(r"player-(\d+)", row_id)
        player_ids = row.xpath(".//*[starts-with(@id,'name_') or starts-with(@id,'names_')]/@id")
        if not number_match or not player_ids:
            continue
        id_match = re.search(r"names?_(\d+)_(\d{12})", player_ids[0])
        if not id_match:
            continue
        links: list[dict[str, str]] = []
        seen: set[str] = set()
        for href in row.xpath(".//a[contains(@href,'/db/result/')]/@href"):
            query = parse_qs(urlparse(href).query)
            race_values = query.get("race_id", [])
            if not race_values or not RACE_ID_RE.fullmatch(race_values[0]):
                continue
            past_race_id = race_values[0]
            if past_race_id in seen:
                continue
            seen.add(past_race_id)
            links.append({"race_id": past_race_id, "url": href})
        name = _clean("".join(row.xpath(".//*[contains(@class,'PlayerName')]//a[1]/text()")))
        riders.append(
            {
                "number": int(number_match.group(1)),
                "player_id": id_match.group(1),
                "name": name,
                "recent_links": links,
            }
        )
    if not riders:
        raise HistoryFetchError("直近成績ページから選手を取得できませんでした。")
    return riders


def _parse_lines(document: Any) -> list[list[int]]:
    box = document.xpath("//*[contains(concat(' ',normalize-space(@class),' '),' DeployBox ')][1]")
    if not box:
        return []
    lines: list[list[int]] = []
    current: list[int] = []
    for item in box[0].xpath("./*[contains(@class,'DeployInBox')]"):
        if item.xpath(".//*[contains(@class,'WakuSeparat')]"):
            if current:
                lines.append(current)
                current = []
            continue
        numbers = item.xpath(".//*[contains(@class,'Shaban_Num')]/text()")
        if numbers and numbers[0].strip().isdigit():
            current.append(int(numbers[0].strip()))
    if current:
        lines.append(current)
    return lines


def parse_result_page(body: bytes, race_id: str) -> dict[str, Any]:
    document = html.fromstring(body)
    lines = _parse_lines(document)
    rows: list[dict[str, Any]] = []
    for row in document.xpath("//table[contains(@class,'ResultRefund')]//tr[contains(@class,'PlayerList')]"):
        cells = row.xpath("./td")
        if len(cells) < 8:
            continue
        finish_text = _clean(cells[0].text_content())
        finish_match = re.match(r"(\d+)着", finish_text)
        number_text = _clean(cells[2].text_content())
        profile_links = row.xpath(".//a[contains(@href,'/db/profile/')]/@href")
        player_id = None
        if profile_links:
            player_id = parse_qs(urlparse(profile_links[0]).query).get("id", [None])[0]
        if player_id is None:
            image_ids = row.xpath(".//img[starts-with(@id,'player_img_')]/@id")
            if image_ids:
                player_id = image_ids[0].replace("player_img_", "")
        name = _clean("".join(row.xpath(".//*[contains(@class,'PlayerName')]//a[1]/text()")))
        agari_text = _clean(cells[5].text_content())
        rows.append(
            {
                "finish": int(finish_match.group(1)) if finish_match else None,
                "finish_status": finish_text,
                "number": int(number_text) if number_text.isdigit() else None,
                "player_id": player_id,
                "name": name,
                "gap": _clean(cells[4].text_content()),
                "agari": float(agari_text) if re.fullmatch(r"\d+(?:\.\d+)?", agari_text) else None,
                "decisive": _clean(cells[6].text_content()),
                "sb": _clean(cells[7].text_content()),
            }
        )
    if not rows:
        raise HistoryFetchError(f"過去レース{race_id}の結果を取得できませんでした。")
    title = _clean(document.xpath("string(//title)"))
    return {"race_id": race_id, "title": title, "lines": lines, "results": rows}


def _role_for_number(lines: list[list[int]], number: int) -> tuple[int | None, int]:
    for line in lines:
        if number in line:
            return line.index(number), len(line)
    return None, 1


def score_lost_strong(player_id: str, races: list[dict[str, Any]]) -> dict[str, Any]:
    """Structured-data proxy. It does not claim video-only interference."""
    evidence: list[dict[str, Any]] = []
    weighted_sum = 0.0
    weight_sum = 0.0
    for recency, race in enumerate(races):
        target = next((row for row in race["results"] if row.get("player_id") == player_id), None)
        if not target or target.get("finish") is None or target.get("number") is None:
            continue
        finish = int(target["finish"])
        number = int(target["number"])
        position, line_size = _role_for_number(race["lines"], number)
        numeric_agari = sorted(row["agari"] for row in race["results"] if row.get("agari") is not None)
        agari_rank = numeric_agari.index(target["agari"]) + 1 if target.get("agari") in numeric_agari else None
        points = 0
        reasons: list[str] = []
        if finish > 2 and agari_rank is not None and agari_rank <= 2:
            points += 30
            reasons.append(f"着順{finish}着に対して上がり{agari_rank}位")
        if finish > 2 and position == 0 and "B" in target.get("sb", ""):
            points += 25
            reasons.append("ライン先頭でBを取りながら着外")
        if position == 0 and finish > 2:
            same_line = next((line for line in race["lines"] if number in line), [number])
            teammates = [row for row in race["results"] if row.get("number") in same_line[1:]]
            if any(row.get("finish") is not None and row["finish"] <= 2 for row in teammates):
                points += 20
                reasons.append("先頭が着外でも同ライン後位が2着以内")
        if line_size == 1 and finish <= 4:
            points += 12
            reasons.append("単騎で4着以内")
        elif line_size == 2 and finish <= 3:
            points += 8
            reasons.append("2車ラインで3着以内")
        points = min(points, 100)
        weight = 1.0 / (1.0 + recency * 0.35)
        weighted_sum += points * weight
        weight_sum += weight
        evidence.append(
            {
                "race_id": race["race_id"],
                "finish": finish,
                "role": "solo" if position is None or line_size == 1 else ("leader" if position == 0 else f"line_position_{position + 1}"),
                "line_size": line_size,
                "agari": target.get("agari"),
                "agari_rank": agari_rank,
                "sb": target.get("sb"),
                "proxy_points": points,
                "reasons": reasons,
            }
        )
    score = weighted_sum / weight_sum if weight_sum else 0.0
    return {
        "score": round(score, 3),
        "sample_races": len(evidence),
        "method": "structured_result_proxy_v0.2",
        "video_interference_included": False,
        "evidence": evidence,
    }


def collect_history(
    entry_pdf: Path,
    race_number: int,
    expected_numbers: list[int],
    races_per_rider: int = 3,
    client: CachedHttpClient | None = None,
) -> dict[str, Any]:
    if races_per_rider < 1 or races_per_rider > 5:
        raise ValueError("history_races_per_rider は1〜5です。")
    client = client or CachedHttpClient()
    warnings: list[str] = []
    target_race_id = extract_race_id_from_pdf(entry_pdf, race_number)
    recent_url = f"{RECENT_URL}?{urlencode({'race_id': target_race_id})}"
    recent = parse_recent_page(client.get(recent_url))
    by_number = {item["number"]: item for item in recent}
    missing_numbers = sorted(set(expected_numbers) - set(by_number))
    if missing_numbers:
        warnings.append(f"直近成績ページにない車番: {missing_numbers}")

    selected_by_number: dict[int, list[str]] = {}
    unique_race_ids: list[str] = []
    for number in expected_numbers:
        item = by_number.get(number)
        selected = [] if not item else [link["race_id"] for link in item["recent_links"] if link["race_id"] != target_race_id][:races_per_rider]
        selected_by_number[number] = selected
        for race_id in selected:
            if race_id not in unique_race_ids:
                unique_race_ids.append(race_id)

    parsed_races: dict[str, dict[str, Any]] = {}
    for race_id in unique_race_ids:
        try:
            url = f"{DETAIL_URL}?{urlencode({'race_id': race_id})}"
            parsed_races[race_id] = parse_result_page(client.get(url), race_id)
        except HistoryFetchError as exc:
            warnings.append(str(exc))

    riders: dict[int, dict[str, Any]] = {}
    for number in expected_numbers:
        item = by_number.get(number)
        player_id = item["player_id"] if item else None
        histories = [parsed_races[race_id] for race_id in selected_by_number[number] if race_id in parsed_races]
        proxy = score_lost_strong(player_id, histories) if player_id else {
            "score": 0.0, "sample_races": 0, "method": "structured_result_proxy_v0.2",
            "video_interference_included": False, "evidence": [],
        }
        riders[number] = {
            "player_id": player_id,
            "name": item.get("name") if item else None,
            "requested_races": len(selected_by_number[number]),
            "fetched_races": len(histories),
            "race_ids": [race["race_id"] for race in histories],
            "lost_strong_proxy": proxy,
        }

    requested = sum(len(values) for values in selected_by_number.values())
    fetched = sum(item["fetched_races"] for item in riders.values())
    if fetched == requested and not missing_numbers:
        status = "complete"
    elif fetched:
        status = "partial"
    else:
        status = "failed"
    return {
        "status": status,
        "source": "netkeirin_public_recent_results",
        "target_race_id": target_race_id,
        "requested_rider_races": requested,
        "fetched_rider_races": fetched,
        "unique_result_pages": len(parsed_races),
        "network_requests": client.requests,
        "cache_hits": client.cache_hits,
        "races_per_rider": races_per_rider,
        "riders": riders,
        "warnings": warnings,
        "usage_scope": "private_personal_research_only",
    }
