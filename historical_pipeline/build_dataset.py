#!/usr/bin/env python3
"""Build leakage-separated CSV/SQLite tables from the 12 monthly archives."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import re
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from bs4 import BeautifulSoup


FEATURE_FIELDS = [
    "race_id", "race_date", "venue_code", "venue_name", "race_no", "race_class",
    "scheduled_starters", "actual_starters", "line_count", "car_no", "player_id",
    "name", "rank", "style", "line_no", "line_position", "line_size",
    "score", "s_count", "h_count", "b_count", "escape", "makuri", "sashi", "mark",
    "finish_1", "finish_2", "finish_3", "finish_out", "win_rate", "top2_rate", "top3_rate",
    "gear_ratio", "previous_order", "previous_lap", "previous_back", "previous_standing",
    "previous_factor", "previous_accident", "recent5_avg_order", "recent5_back_rate",
    "recent5_standing_rate", "recent5_avg_lap", "lose_strong_prerace_score",
]
RESULT_FIELDS = [
    "race_id", "car_no", "player_id", "finish_order", "winning_move", "actual_start",
    "actual_back", "final_lap_time", "margin", "has_accident", "accident_name",
    "weather", "wind_speed", "distance_m", "result_comment",
]


def num(value, default=""):
    text = str(value if value is not None else "").strip().replace(",", "")
    if not text or text.lower() in {"nan", "none"}:
        return default
    try:
        value = float(text)
        return int(value) if value.is_integer() else value
    except ValueError:
        return default


def line_map(lineup: str) -> dict[int, tuple[int, int, int]]:
    result = {}
    for line_no, group in enumerate((lineup or "").split("|"), 1):
        cars = [int(x) for x in group.split("-") if x.isdigit()]
        for pos, car in enumerate(cars, 1):
            result[car] = (line_no, pos, len(cars))
    return result


def parse_kdreams(body: bytes) -> tuple[dict[int, dict], list[dict], dict[int, str]]:
    soup = BeautifulSoup(body, "lxml")
    metrics: dict[int, dict] = {}
    comments: dict[int, str] = {}
    table = soup.select_one("table.racecard_table")
    if table:
        for tr in table.select("tr[class*='n']"):
            cells = tr.find_all(["td", "th"], recursive=False)
            if len(cells) < 23:
                continue
            car = num(cells[4].get_text(" ", strip=True))
            if not isinstance(car, int):
                continue
            rider_text = cells[5].get_text(" ", strip=True)
            home = cells[5].select_one(".home")
            name = rider_text.replace(home.get_text(" ", strip=True), "").strip() if home else rider_text
            vals = [c.get_text(" ", strip=True) for c in cells]
            metrics[car] = {
                "name": name, "rank": vals[6], "style": vals[7], "gear_ratio": num(vals[8]),
                "score": num(vals[9]), "s_count": num(vals[10], 0), "b_count": num(vals[11], 0),
                "escape": num(vals[12], 0), "makuri": num(vals[13], 0),
                "sashi": num(vals[14], 0), "mark": num(vals[15], 0),
                "finish_1": num(vals[16], 0), "finish_2": num(vals[17], 0),
                "finish_3": num(vals[18], 0), "finish_out": num(vals[19], 0),
                "win_rate": num(vals[20]), "top2_rate": num(vals[21]), "top3_rate": num(vals[22]),
            }

    # 2車単 table: columns are first place and rows are second place.
    odds: list[dict] = []
    container = soup.select_one("#JS_ODDSCONTENTS_2shatan")
    odds_table = container.select_one("table.odds_table") if container else None
    if odds_table:
        rows = odds_table.find_all("tr", recursive=False)
        headers = [num(x.get_text(strip=True)) for x in rows[0].find_all("th") if str(num(x.get_text(strip=True))).isdigit()]
        for tr in rows[2:]:
            second_node = tr.find("th")
            second = num(second_node.get_text(strip=True)) if second_node else ""
            values = tr.find_all("td", recursive=False)
            if not isinstance(second, int):
                continue
            for first, td in zip(headers, values):
                odd = num(td.get_text(strip=True))
                if isinstance(first, int) and first != second and odd != "":
                    odds.append({"first_car": first, "second_car": second,
                                 "combination": f"{first}-{second}", "exacta_odds": odd})

    result_table = soup.select_one("table.result_table")
    if result_table:
        for tr in result_table.find_all("tr")[1:]:
            cells = tr.find_all("td", recursive=False)
            if len(cells) < 9:
                continue
            car = num(cells[2].get_text(strip=True))
            if isinstance(car, int):
                comments[car] = cells[8].get_text(" ", strip=True)
    return metrics, odds, comments


def recent_rows(record: dict) -> list[dict]:
    seen, rows = set(), []
    sources = []
    sources.extend(record.get("currentCupResults") or [])
    sources.extend(record.get("previousCupResults") or [])
    for cup in record.get("latestCupResults") or []:
        sources.extend(cup.get("raceResults") or [])
    for row in sources:
        key = (row.get("raceId"), row.get("playerId"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def prior_features(record: dict) -> dict:
    rows = recent_rows(record)
    rows.sort(key=lambda x: str(x.get("raceId", ""))[4:12], reverse=True)
    previous = rows[0] if rows else {}
    five = rows[:5]
    orders = [num(x.get("order")) for x in five if num(x.get("order")) != ""]
    laps = [num(x.get("finalHalfRecord")) for x in five if num(x.get("finalHalfRecord")) not in {"", 0}]
    order = num(previous.get("order"))
    lap = num(previous.get("finalHalfRecord"))
    back = int(bool(previous.get("back"))) if previous else ""
    standing = int(bool(previous.get("standing"))) if previous else ""
    # Defined entirely from the previous race. It is a pre-race feature, never a current result label.
    lose_strong = 0.0
    if isinstance(order, (int, float)) and order >= 4:
        lose_strong += 1.0
        lose_strong += 1.0 if back else 0.0
        lose_strong += 0.5 if standing else 0.0
        if laps and isinstance(lap, (int, float)) and lap <= sorted(laps)[len(laps) // 2]:
            lose_strong += 0.5
    return {
        "previous_order": order, "previous_lap": lap, "previous_back": back,
        "previous_standing": standing, "previous_factor": previous.get("factor", ""),
        "previous_accident": int(bool(previous.get("hasAccident"))) if previous else "",
        "recent5_avg_order": sum(orders) / len(orders) if orders else "",
        "recent5_back_rate": sum(bool(x.get("back")) for x in five) / len(five) if five else "",
        "recent5_standing_rate": sum(bool(x.get("standing")) for x in five) / len(five) if five else "",
        "recent5_avg_lap": sum(laps) / len(laps) if laps else "",
        "lose_strong_prerace_score": lose_strong,
    }


def parse_one(args):
    row, raw_root = args
    rid, month = row["race_id"], row["race_date"][:6]
    root = Path(raw_root) / month
    with gzip.open(root / "kdreams" / f"{rid}.gz", "rb") as fh:
        metrics, odds, comments = parse_kdreams(fh.read())
    with gzip.open(root / "winticket_records" / f"{rid}.gz", "rt", encoding="utf-8") as fh:
        pre = json.load(fh)
    with gzip.open(root / "winticket_results" / f"{rid}.gz", "rt", encoding="utf-8") as fh:
        official = json.load(fh)

    entries = {str(x.get("playerId")): x for x in pre.get("entries") or []}
    records = {str(x.get("playerId")): x for x in pre.get("records") or []}
    lm = line_map(row.get("lineup", ""))
    features = []
    for player_id, entry in entries.items():
        car = int(entry["number"])
        record = records.get(player_id, {})
        base = metrics.get(car, {})
        line_no, line_pos, line_size = lm.get(car, (0, 0, 1))
        out = {key: row.get(key, "") for key in [
            "race_id", "race_date", "venue_code", "venue_name", "race_no", "race_class",
            "scheduled_starters", "actual_starters", "line_count"]}
        out.update({"car_no": car, "player_id": player_id, "line_no": line_no,
                    "line_position": line_pos, "line_size": line_size})
        out.update(base)
        out["name"] = base.get("name", "")
        out["h_count"] = num(record.get("home"), 0)
        # Prefer Winticket scalar values where present.
        aliases = {"score": "racePoint", "s_count": "standing", "b_count": "back",
                   "escape": "frontRunner", "makuri": "deepCloser", "sashi": "stalker",
                   "mark": "marker", "finish_1": "first", "finish_2": "second",
                   "finish_3": "third", "finish_out": "others", "gear_ratio": "gearRatio",
                   "win_rate": "firstRate", "top2_rate": "secondRate", "top3_rate": "thirdRate"}
        for target, source in aliases.items():
            if record.get(source) is not None:
                out[target] = num(record.get(source), 0)
        out.update(prior_features(record))
        features.append(out)

    race_meta = official.get("race") or {}
    official_entries = {str(x.get("playerId")): x for x in official.get("entries") or []}
    results = []
    for rr in official.get("results") or []:
        pid = str(rr.get("playerId", ""))
        entry = official_entries.get(pid, {})
        car = entry.get("number", "")
        results.append({
            "race_id": rid, "car_no": car, "player_id": pid, "finish_order": rr.get("order", ""),
            "winning_move": rr.get("factor", ""), "actual_start": int(bool(rr.get("standing"))),
            "actual_back": int(bool(rr.get("back"))), "final_lap_time": rr.get("finalHalfRecord", ""),
            "margin": rr.get("margin", ""), "has_accident": int(bool(rr.get("hasAccident"))),
            "accident_name": rr.get("accidentName", ""), "weather": race_meta.get("weather", ""),
            "wind_speed": race_meta.get("windSpeed", ""), "distance_m": race_meta.get("distance", ""),
            "result_comment": comments.get(int(car), "") if str(car).isdigit() else "",
        })
    for odd in odds:
        odd["race_id"] = rid
    race_out = dict(row)
    race_out.update({"distance_m": race_meta.get("distance", ""), "weather": race_meta.get("weather", ""),
                     "wind_speed": race_meta.get("windSpeed", ""), "official_result_rows": len(results),
                     "exacta_odds_rows": len(odds)})
    return race_out, features, odds, results


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="historical_pipeline/data/eligible_manifest_2025.csv.gz")
    ap.add_argument("--raw-root", default="raw")
    ap.add_argument("--output-dir", default="dataset")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    with gzip.open(args.manifest, "rt", encoding="utf-8", newline="") as fh:
        manifest = list(csv.DictReader(fh))
    races, features, odds, results, errors = [], [], [], [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(parse_one, (row, args.raw_root)): row["race_id"] for row in manifest}
        for idx, future in enumerate(as_completed(futures), 1):
            rid = futures[future]
            try:
                race, frows, orows, rrows = future.result()
                races.append(race); features.extend(frows); odds.extend(orows); results.extend(rrows)
            except Exception as exc:
                errors[rid] = f"{type(exc).__name__}: {exc}"
            if idx % 1000 == 0:
                print(f"parsed {idx}/{len(manifest)} errors={len(errors)}", flush=True)
    if errors:
        Path("dataset_errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2))
        raise SystemExit(f"dataset incomplete: {len(errors)} parse errors")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    race_fields = list(manifest[0]) + ["distance_m", "weather", "wind_speed", "official_result_rows", "exacta_odds_rows"]
    write_csv(out / "races_2025.csv", race_fields, sorted(races, key=lambda x: (x["race_date"], x["race_id"])))
    write_csv(out / "rider_features_2025.csv", FEATURE_FIELDS, sorted(features, key=lambda x: (x["race_date"], x["race_id"], int(x["car_no"]))))
    write_csv(out / "exacta_odds_2025.csv", ["race_id", "first_car", "second_car", "combination", "exacta_odds"], odds)
    write_csv(out / "official_results_2025.csv", RESULT_FIELDS, results)

    db = out / "keirin_backtest_2025.sqlite"
    conn = sqlite3.connect(db)
    for table, path in [("races", out / "races_2025.csv"), ("rider_features", out / "rider_features_2025.csv"),
                        ("exacta_odds", out / "exacta_odds_2025.csv"), ("official_results", out / "official_results_2025.csv")]:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh); fields = next(reader)
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            conn.execute(f'CREATE TABLE "{table}" ({",".join(f"\"{x}\" TEXT" for x in fields)})')
            sql = f'INSERT INTO "{table}" VALUES ({",".join("?" for _ in fields)})'
            batch = []
            for row in reader:
                batch.append(row)
                if len(batch) >= 20000:
                    conn.executemany(sql, batch); batch.clear()
            if batch: conn.executemany(sql, batch)
    conn.executescript("""
      CREATE INDEX idx_races_id ON races(race_id);
      CREATE INDEX idx_features_id ON rider_features(race_id,car_no);
      CREATE INDEX idx_odds_id ON exacta_odds(race_id,first_car,second_car);
      CREATE INDEX idx_results_id ON official_results(race_id,car_no);
    """)
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    conn.commit(); conn.close()
    audit = {"races": len(races), "rider_features": len(features), "exacta_odds": len(odds),
             "official_results": len(results), "parse_errors": 0, "sqlite_integrity": integrity,
             "feature_scope": "pre-race only", "label_scope": "official result only"}
    (out / "dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
