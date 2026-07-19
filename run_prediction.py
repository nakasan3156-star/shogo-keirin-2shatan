from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.engine import predict
from app.parser import parse_entry_pdf, parse_odds_pdf, parse_recent_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="章悟式∞競輪OS 2車単PDF版")
    parser.add_argument("basic_pdf", type=Path)
    parser.add_argument("recent_pdf", type=Path)
    parser.add_argument("odds_pdf", type=Path)
    parser.add_argument("--runs", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=3156)
    args = parser.parse_args()

    race = parse_entry_pdf(args.basic_pdf)
    recent = parse_recent_pdf(args.recent_pdf, race)
    numbers = [rider.number for rider in race.riders]
    odds = parse_odds_pdf(args.odds_pdf, numbers)
    output = predict(race, odds, runs=args.runs, seed=args.seed, history=recent)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
