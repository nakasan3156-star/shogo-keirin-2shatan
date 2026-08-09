#!/usr/bin/env python3
"""Collect one month of keirin data and package it as a resumable release asset."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import random
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


KD_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/html,application/xhtml+xml"}
WT_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
WT_RECORDS = "https://api.winticket.jp/v1/keirin/cups/{cup}/schedules/{day}/races/{race}?fields=entries%2Crecords"
WT_RESULTS = "https://api.winticket.jp/v1/keirin/cups/{cup}/schedules/{day}/races/{race}?fields=race%2Cresults%2Centries%2Cschedule"


def get(url: str, headers: dict[str, str], validator, retries: int = 8) -> bytes:
    error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=(15, 90))
            response.raise_for_status()
            body = response.content
            validator(body)
            return body
        except Exception as exc:
            error = exc
            time.sleep(min(30, 1.7 ** attempt) + random.random())
    raise RuntimeError(f"failed after {retries} attempts: {url}: {error}")


def valid_kd(body: bytes) -> None:
    if len(body) < 50_000 or b"racedetail" not in body.lower():
        raise ValueError(f"unexpected Kdreams payload: {len(body)}")


def valid_json(required: tuple[str, ...]):
    def check(body: bytes) -> None:
        obj = json.loads(body)
        if any(not obj.get(key) for key in required):
            raise ValueError(f"missing required keys: {required}")
    return check


def collect(row: dict[str, str]) -> tuple[str, dict[str, bytes] | None, str | None]:
    rid = row["race_id"]
    cup = f"{row['start_date']}{int(row['venue_code']):02d}"
    day, race = int(row["day_no"]), int(row["race_no"])
    try:
        kd = get(row["detail_url"], KD_HEADERS, valid_kd)
        wr = get(WT_RECORDS.format(cup=cup, day=day, race=race), WT_HEADERS,
                 valid_json(("entries", "records")))
        wo = get(WT_RESULTS.format(cup=cup, day=day, race=race), WT_HEADERS,
                 valid_json(("race", "entries", "results")))
        return rid, {"kdreams": kd, "winticket_records": wr, "winticket_results": wo}, None
    except Exception as exc:
        return rid, None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True, help="YYYYMM")
    ap.add_argument("--manifest", default="historical_pipeline/data/eligible_manifest_2025.csv.gz")
    ap.add_argument("--output-dir", default="dist")
    ap.add_argument("--asset-prefix", default="keirin_2025")
    ap.add_argument("--workers", type=int, default=80)
    args = ap.parse_args()

    with gzip.open(args.manifest, "rt", encoding="utf-8", newline="") as fh:
        rows = [row for row in csv.DictReader(fh) if row["race_date"].startswith(args.month)]
    if not rows:
        raise SystemExit(f"no manifest rows for {args.month}")

    payloads: dict[str, dict[str, bytes]] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(collect, row) for row in rows]
        for idx, future in enumerate(as_completed(futures), 1):
            rid, payload, error = future.result()
            if error:
                errors[rid] = error
            else:
                payloads[rid] = payload or {}
            if idx % 100 == 0 or idx == len(rows):
                print(f"{args.month}: {idx}/{len(rows)} errors={len(errors)}", flush=True)

    # One complete retry pass; no incomplete archive is published.
    if errors:
        retry_rows = [row for row in rows if row["race_id"] in errors]
        errors = {}
        with ThreadPoolExecutor(max_workers=max(8, args.workers // 2)) as pool:
            futures = [pool.submit(collect, row) for row in retry_rows]
            for future in as_completed(futures):
                rid, payload, error = future.result()
                if error:
                    errors[rid] = error
                else:
                    payloads[rid] = payload or {}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "month": args.month, "expected_races": len(rows), "complete_races": len(payloads),
        "failed_races": len(errors), "errors": errors,
    }
    (out_dir / f"audit_{args.month}.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors or len(payloads) != len(rows):
        raise SystemExit(f"month incomplete: {len(payloads)}/{len(rows)}")

    archive = out_dir / f"{args.asset_prefix}_{args.month}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rid in sorted(payloads):
            for source, body in payloads[rid].items():
                zf.writestr(f"{source}/{rid}.gz", gzip.compress(body, compresslevel=6))
        zf.writestr("audit.json", json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps({"archive": str(archive), **audit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
