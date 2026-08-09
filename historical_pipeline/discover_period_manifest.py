#!/usr/bin/env python3
"""Discover and freeze the eligible race manifest for an arbitrary date range.

Eligibility is intentionally identical to the 2025 collection: women,
KEIRIN ADVANCE/PIST6, two-or-more scratches, races without a normal line, and
races without a final result are excluded.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from lxml import html

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ShogoKeirinResearch/1.0)"}
DETAIL_RE = re.compile(
    r"https://keirin\.kdreams\.jp/(?P<slug>[^/]+)/racedetail/"
    r"(?P<race_id>(?P<venue_code>\d{2})(?P<start_date>\d{8})"
    r"(?P<day_no>\d{2})00(?P<race_no>\d{2}))/"
)


def clean(value: str | None) -> str:
    return re.sub(r"\s+", "", value or "")


def norm(value: str | None) -> str:
    return unicodedata.normalize("NFKC", clean(value))


def digits(value: str | None) -> str:
    found = re.search(r"\d+", norm(value))
    return found.group() if found else ""


def node_text(node) -> str:
    return clean("".join(node.itertext())) if node is not None else ""


def get(url: str, minimum: int = 8_000, attempts: int = 7) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=HEADERS, timeout=(15, 90))
            response.raise_for_status()
            if len(response.content) < minimum:
                raise ValueError(f"short response: {len(response.content)}")
            return response.content
        except Exception as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(min(30, 1.7**attempt) + random.random())
    raise RuntimeError(f"failed: {url}: {error}")


def all_dates(start: date, end: date) -> list[date]:
    out = []
    current = start
    while current <= end:
        out.append(current)
        current += timedelta(days=1)
    return out


def discover_day(day: date) -> tuple[list[dict[str, str]], str | None]:
    key = day.strftime("%Y%m%d")
    url = f"https://keirin.kdreams.jp/raceresult/{day:%Y/%m/%d}/"
    try:
        tree = html.fromstring(get(url))
        found = {}
        for href in tree.xpath('//a[contains(@href,"/racedetail/")]/@href'):
            match = DETAIL_RE.search(href)
            if not match:
                continue
            row = match.groupdict()
            row["race_date"] = key
            row["detail_url"] = match.group(0)
            found[row["race_id"]] = row
        return list(found.values()), None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def parse_lines(race_node):
    groups: list[list[str]] = []
    roles: dict[str, str] = {}
    current: list[str] = []
    containers = race_node.xpath('.//dl[contains(@class,"line_position")]//div[contains(@class,"line_position_inner")]')
    if not containers:
        containers = race_node.xpath('.//div[contains(@class,"line_position")]')
    if not containers:
        return groups, roles
    for icon in containers[0].xpath('./span[contains(@class,"icon_p")]'):
        if "space" in (icon.get("class") or "").split():
            if current:
                groups.append(current)
                current = []
            continue
        parts = [clean("".join(x.itertext())) for x in icon.xpath("./span")]
        number = next((p for p in parts if re.fullmatch(r"[1-9]", p)), "")
        if not number:
            continue
        current.append(number)
        role = "".join(p for p in parts if p not in {number, "←"})
        if role:
            roles[number] = role
    if current:
        groups.append(current)
    return groups, roles


def race_id_from(node) -> str:
    links = node.xpath('.//p[contains(@class,"race")]//a/@href')
    match = re.search(r"/(\d{16})/", links[0]) if links else None
    return match.group(1) if match else ""


def payout_block(table, bet_header: str, row_kind: str):
    rows = table.xpath("./tr")
    if len(rows) < 2:
        return "", "", ""
    target_index = None
    bet_seen = -1
    for cell in rows[0].xpath("./th|./td"):
        value = norm(node_text(cell))
        if value in {"2枠連", "2車連", "3連勝"}:
            bet_seen += 1
            if value == bet_header:
                target_index = bet_seen
                break
    if target_index is None:
        return "", "", ""
    row = rows[0] if row_kind == "複" else rows[1]
    cells = row.xpath("./td")
    value_index = target_index * 2 + 1
    if value_index >= len(cells):
        return "", "", ""
    value_cell = cells[value_index]
    pair = node_text(value_cell.xpath(".//dt")[0]) if value_cell.xpath(".//dt") else ""
    dd = value_cell.xpath(".//dd")
    raw_pay = node_text(dd[0]) if dd else ""
    payout = re.sub(r"\D", "", norm(raw_pay.split("(")[0]))
    popularity = ""
    match = re.search(r"\((\d+)\)", norm(raw_pay))
    if match:
        popularity = match.group(1)
    return pair, payout, popularity


def parse_results(root):
    result_ids: set[str] = set()
    payouts: dict[str, dict[str, str]] = {}
    for race in root.xpath('//ul[contains(@class,"result_list")]/li'):
        rid = race_id_from(race)
        if not rid:
            continue
        order = race.xpath('.//table[contains(@class,"order_table")]')
        if order:
            for head, cell in zip(order[0].xpath("./tr[1]/th"), order[0].xpath("./tr[2]/td")):
                if norm(node_text(head)) == "1着" and digits(node_text(cell)):
                    result_ids.add(rid)
        values: dict[str, str] = {}
        refund = race.xpath('.//table[contains(@class,"refund_table")]')
        if refund:
            for prefix, header, kind in (
                ("two_car_quinella", "2車連", "複"),
                ("two_car_exacta", "2車連", "単"),
                ("trio", "3連勝", "複"),
                ("trifecta", "3連勝", "単"),
            ):
                pair, pay, pop = payout_block(refund[0], header, kind)
                values[f"{prefix}_combination"] = pair
                values[f"{prefix}_payout_yen"] = pay
                values[f"{prefix}_popularity"] = pop
        payouts[rid] = values
    return result_ids, payouts


def collect_event(event: dict[str, str]):
    event_id = event["event_day_id"]
    base = f"https://keirin.kdreams.jp/{event['slug']}"
    try:
        card = html.fromstring(get(f"{base}/racecard/{event_id}/"))
        result = html.fromstring(get(f"{base}/raceresult/{event_id}/"))
        return event, card, result, None
    except Exception as exc:
        return event, None, None, f"{type(exc).__name__}: {exc}"


def parse_event(event: dict[str, str], card, result):
    base = f"https://keirin.kdreams.jp/{event['slug']}"
    event_id = event["event_day_id"]
    result_ids, payouts = parse_results(result)
    title = card.xpath("//title/text()")
    venue_match = re.search(r"(.+?)競輪", clean(title[0]) if title else "")
    venue_name = venue_match.group(1) if venue_match else event["slug"]
    grade_nodes = card.xpath('//p[contains(@class,"raceinfo_contents-title")]//span[contains(@class,"icon_grade")]')
    event_nodes = card.xpath('//p[contains(@class,"raceinfo_contents-title")]//span[contains(@class,"text")]')
    grade = norm(node_text(grade_nodes[0])) if grade_nodes else ""
    event_name = node_text(event_nodes[0]) if event_nodes else ""
    expected = 9 if grade in {"GP", "G1", "G2", "G3"} else 7
    rows = []
    exclusions = []
    for race in card.xpath('//ul[contains(@class,"racecard_list")]/li'):
        rid = race_id_from(race)
        if not rid:
            continue
        number_node = race.xpath('.//p[contains(@class,"race")]//span[contains(@class,"num")]')
        class_node = race.xpath('.//p[contains(@class,"race")]//span[contains(@class,"name")]')
        race_no = digits(node_text(number_node[0])) if number_node else rid[-2:]
        race_class = node_text(class_node[0]) if class_node else ""
        groups, roles = parse_lines(race)
        actual = 0
        for tr in race.xpath('.//div[contains(@class,"racecard_table")]//tr[td[contains(@class,"rider")]]'):
            rider = tr.xpath('./td[contains(@class,"rider")]')
            num = tr.xpath('./td[contains(@class,"num")]')
            if rider and num and digits(node_text(num[0])) and node_text(rider[0]):
                actual += 1
        scratches = max(0, expected - actual)
        combined = norm(f"{event_name}{race_class}")
        reasons = []
        if re.search(r"L級|ガールズ|女子", combined, re.I): reasons.append("women")
        if re.search(r"ADVANCE|アドバンス", combined, re.I): reasons.append("keirin_advance")
        if re.search(r"PIST6|250競走", combined, re.I): reasons.append("pist6")
        if scratches >= 2: reasons.append("two_or_more_scratches")
        if not any(len(group) >= 2 for group in groups): reasons.append("no_line_formation")
        if actual == 0: reasons.append("no_entrants")
        if rid not in result_ids: reasons.append("no_result")
        row = {
            "race_id": rid, "race_date": event["race_date"], "start_date": event["start_date"],
            "day_no": event["day_no"], "venue_code": event["venue_code"], "venue_name": venue_name,
            "event_grade": grade, "event_name": event_name, "race_no": race_no,
            "race_class": race_class, "detail_url": f"{base}/racedetail/{rid}/",
            "racecard_url": f"{base}/racecard/{event_id}/", "result_url": f"{base}/raceresult/{event_id}/",
            "scheduled_starters": str(expected), "actual_starters": str(actual),
            "scratch_count": str(scratches), "lineup": "|".join("-".join(x) for x in groups),
            "lineup_tactics": ";".join(f"{car}:{roles[car]}" for car in sorted(roles, key=int)),
            "line_count": str(len(groups)), "line_sizes": "-".join(str(len(x)) for x in groups),
            **payouts.get(rid, {}), "eligible": "1" if not reasons else "0",
            "exclusion_reason": ";".join(reasons),
        }
        rows.append(row)
        if reasons:
            exclusions.append({"race_id": rid, "race_date": event["race_date"], "reason": ";".join(reasons)})
    return rows, exclusions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    discovered: list[dict[str, str]] = []
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(12, args.workers)) as pool:
        futures = {pool.submit(discover_day, d): d for d in all_dates(start, end)}
        for future in as_completed(futures):
            rows, error = future.result()
            discovered.extend(rows)
            if error: errors[f"date:{futures[future].isoformat()}"] = error
    if errors:
        raise SystemExit(json.dumps(errors, ensure_ascii=False))

    events = {}
    for row in discovered:
        event_id = row["race_id"][:-2]
        events[event_id] = {**{k: row[k] for k in ["race_date", "start_date", "day_no", "venue_code", "slug"]}, "event_day_id": event_id}
    parsed: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(collect_event, event) for event in events.values()]
        for idx, future in enumerate(as_completed(futures), 1):
            event, card, result, error = future.result()
            if error:
                errors[f"event:{event['event_day_id']}"] = error
            else:
                rows, excluded = parse_event(event, card, result)
                parsed.extend(rows); exclusions.extend(excluded)
            if idx % 50 == 0 or idx == len(events):
                print(f"events {idx}/{len(events)} errors={len(errors)}", flush=True)
    if errors:
        Path(args.audit).write_text(json.dumps({"errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"manifest discovery incomplete: {len(errors)} errors")

    eligible = sorted((x for x in parsed if x["eligible"] == "1"), key=lambda x: (x["race_date"], x["race_id"]))
    if not eligible:
        raise SystemExit("no eligible races")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(eligible[0]))
        writer.writeheader(); writer.writerows(eligible)
    reason_counts: dict[str, int] = {}
    for row in exclusions:
        for reason in row["reason"].split(";"):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    audit = {
        "start": args.start, "end": args.end, "days": len(all_dates(start, end)),
        "discovered_race_links": len(discovered), "event_days": len(events),
        "parsed_races": len(parsed), "eligible_races": len(eligible),
        "excluded_races": len(parsed) - len(eligible), "duplicate_eligible_ids": len(eligible) - len({x['race_id'] for x in eligible}),
        "exclusion_reason_counts_nonexclusive": reason_counts, "errors": {},
    }
    Path(args.audit).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
