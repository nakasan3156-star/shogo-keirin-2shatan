#!/usr/bin/env python3
"""PR #31と同じ学習分割・特徴量・校正を使い、本番API用Frozen bundleを書き出す。"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from historical_pipeline.chatgpt_baseline_backtest import (
    COMPONENT_END, EVENT_FEATURES, PAIR_END, SEED, VALID_END,
    add_component_predictions, add_prerace_features, attach_market,
    calibrate_pair_prob, category_effects, fit_pair_model, predict_pairs,
    race_event_table, reliability_tables, scenario_pair_rows, venue_tables,
)


def build(dataset_dir: Path) -> dict:
    races = pd.read_csv(dataset_dir / "races_2025.csv", dtype={"race_id": str, "venue_code": str})
    riders = pd.read_csv(dataset_dir / "rider_features_2025.csv", dtype={"race_id": str, "venue_code": str, "player_id": str})
    results = pd.read_csv(dataset_dir / "official_results_2025.csv", dtype={"race_id": str, "player_id": str})
    odds = pd.read_csv(dataset_dir / "exacta_odds_2025.csv", dtype={"race_id": str})
    for d in (races, riders, results, odds):
        for c in ("car_no", "first_car", "second_car", "finish_order", "actual_back", "actual_start", "exacta_odds", "race_date"):
            if c in d:
                d[c] = pd.to_numeric(d[c], errors="coerce")

    df = add_prerace_features(riders, races, results)
    lab = results[["race_id", "car_no", "finish_order", "actual_back", "actual_start", "winning_move", "final_lap_time", "result_comment"]]
    df = df.merge(lab, on=["race_id", "car_no"], how="inner")
    df = df[df.finish_order.gt(0)].copy()
    df["y_win"] = df.finish_order.eq(1).astype(int)
    df["y_top2"] = df.finish_order.le(2).astype(int)
    df["y_top3"] = df.finish_order.le(3).astype(int)
    df["y_back"] = df.actual_back.fillna(0).astype(int)

    component = df[df.race_date <= COMPONENT_END].copy()
    pair_train = df[(df.race_date > COMPONENT_END) & (df.race_date <= PAIR_END)].copy()
    valid = df[(df.race_date > PAIR_END) & (df.race_date <= VALID_END)].copy()

    cats, _ = category_effects(component)
    component_models, used = add_component_predictions([component, pair_train, valid], component, cats)

    events = race_event_table(df)
    event_train = events[events.race_date <= COMPONENT_END]
    event_model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED)
    event_model.fit(event_train[EVENT_FEATURES].fillna(0), event_train.early_battle_label)

    battle_maps = {}
    for name, frame in (("pair", pair_train), ("valid", valid)):
        e = events[events.race_id.isin(frame.race_id.unique())].copy()
        e["p_battle"] = event_model.predict_proba(e[EVENT_FEATURES].fillna(0))[:, 1]
        battle_maps[name] = dict(zip(e.race_id, e.p_battle))

    venue, venue_base = venue_tables(component)
    pair_model = fit_pair_model(scenario_pair_rows(pair_train, battle_maps["pair"], venue), results)
    valid_pairs = predict_pairs(scenario_pair_rows(valid, battle_maps["valid"], venue), pair_model)
    valid_market = attach_market(valid_pairs, odds, races, results)
    shadow = valid_market.copy()
    iso = calibrate_pair_prob(valid_market, shadow)
    _, _, joint_rel = reliability_tables(valid_market)

    return {
        "bundle_version": "pr31-frozen-1",
        "source": "PR #31 frozen; 2025 only",
        "seed": SEED,
        "component_models": component_models,
        "event_model": event_model,
        "pair_model": pair_model,
        "isotonic": iso,
        "category_effects": cats,
        "used_prior_categories": used,
        "venue": venue,
        "venue_base": venue_base,
        "joint_reliability": joint_rel,
        # PR #32はPR31を変更せず、この条件で2026-01..02を完全未使用評価した。
        "purchase_rule": {"max_points": 5, "min_ev": 2.0, "min_prob": 0.03, "confidence_max": 1.0},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="dataset")
    ap.add_argument("--output", default="pr31_frozen.joblib")
    args = ap.parse_args()
    bundle = build(Path(args.dataset_dir))
    joblib.dump(bundle, args.output, compress=3)
    print({"output": args.output, "version": bundle["bundle_version"], "used_prior_categories": bundle["used_prior_categories"]})


if __name__ == "__main__":
    main()
