#!/usr/bin/env python3
"""Apply the merged PR #31 baseline unchanged to untouched Jan-Feb 2026.

All fitting and calibration data come from the frozen 2025 time splits.  The
2026 labels are joined only after the pre-race predictions and purchase rule
have been fixed.  No rule search is performed in this program.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import chatgpt_baseline_backtest as baseline

FROZEN_PR = 31
FROZEN_MERGE_SHA = "bf76468b51e7435094e05af924c18102053ed277"
FROZEN_RULE = {"max_points": 5, "min_ev": 2.0, "min_prob": 0.03, "confidence_max": 1.0}
EXTERNAL_START = 20260101
EXTERNAL_END = 20260228


def load_period(root: Path, label: str):
    races = pd.read_csv(root / f"races_{label}.csv", dtype={"race_id": str, "venue_code": str})
    riders = pd.read_csv(root / f"rider_features_{label}.csv", dtype={"race_id": str, "venue_code": str})
    results = pd.read_csv(root / f"official_results_{label}.csv", dtype={"race_id": str})
    odds = pd.read_csv(root / f"exacta_odds_{label}.csv", dtype={"race_id": str})
    for frame in [races, riders, results, odds]:
        for col in ["car_no", "first_car", "second_car", "finish_order", "actual_back", "actual_start", "exacta_odds", "race_date"]:
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return races, riders, results, odds


def corrected_path_metrics(bets: pd.DataFrame):
    if bets.empty:
        return {"max_losing_streak": 0, "max_drawdown_yen": 0.0}
    b = bets.sort_values(["race_date", "race_id", "ev"], ascending=[True, True, False]).copy()
    race_profit = b.groupby(["race_date", "race_id"], sort=True).apply(
        lambda x: float(x["return"].sum() - x["stake"].sum()), include_groups=False
    )
    streak = maximum = 0
    for loss in race_profit.lt(0):
        streak = streak + 1 if loss else 0
        maximum = max(maximum, streak)
    equity = race_profit.cumsum().to_numpy()
    peak = np.maximum.accumulate(np.r_[0.0, equity])[1:]
    return {
        "max_losing_streak": int(maximum),
        "max_drawdown_yen": float(np.max(peak - equity)) if len(equity) else 0.0,
    }


def robustness(bets: pd.DataFrame):
    stake = float(bets["stake"].sum()) if not bets.empty else 0.0
    if not stake:
        return {"odds_20pct_worse_roi": 0.0, "largest_hit_removed_roi": 0.0}
    returns = pd.to_numeric(bets["return"], errors="coerce").fillna(0.0)
    winning = bets.loc[returns.gt(0)].copy()
    largest = float(winning["return"].max()) if len(winning) else 0.0
    threshold_rows = []
    for threshold in [10_000, 20_000, 50_000]:
        kept_return = float(returns.where(returns.lt(threshold), 0).sum())
        threshold_rows.append({
            "excluded_payout_yen_gte": threshold,
            "excluded_hits": int((returns >= threshold).sum()),
            "roi": kept_return / stake,
        })
    return {
        "odds_20pct_worse_roi": float((returns * 0.8).sum() / stake),
        "largest_hit_payout_yen": largest,
        "largest_hit_removed_roi": float((returns.sum() - largest) / stake),
        "payout_threshold_exclusions": threshold_rows,
    }


def segment_metrics(bets: pd.DataFrame, total_races: int):
    if bets.empty:
        return {
            "target_races": int(total_races), "purchase_races": 0, "purchase_rate": 0.0,
            "bets": 0, "avg_points": 0.0, "hits": 0, "bet_hit_rate": 0.0,
            "race_hit_rate": 0.0, "stake_yen": 0.0, "return_yen": 0.0, "roi": 0.0,
            **corrected_path_metrics(bets), **robustness(bets),
        }
    stake = float(bets.stake.sum())
    ret = float(bets["return"].sum())
    purchase_races = int(bets.race_id.nunique())
    return {
        "target_races": int(total_races), "purchase_races": purchase_races,
        "purchase_rate": purchase_races / total_races if total_races else 0.0,
        "bets": int(len(bets)), "avg_points": len(bets) / purchase_races,
        "hits": int(bets.is_hit.sum()), "bet_hit_rate": float(bets.is_hit.mean()),
        "race_hit_rate": float(bets.groupby("race_id").is_hit.max().mean()),
        "stake_yen": stake, "return_yen": ret, "profit_yen": ret - stake,
        "roi": ret / stake, **corrected_path_metrics(bets), **robustness(bets),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--external-dir", required=True)
    ap.add_argument("--output-dir", default="frozen_2026")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    races25, riders25, results25, odds25 = load_period(Path(args.train_dir), "2025")
    races26, riders26, results26, odds26 = load_period(Path(args.external_dir), "2026_01_02")
    races26 = races26[races26.race_date.between(EXTERNAL_START, EXTERNAL_END)].copy()
    external_ids = set(races26.race_id)
    riders26 = riders26[riders26.race_id.isin(external_ids)].copy()
    results26 = results26[results26.race_id.isin(external_ids)].copy()
    odds26 = odds26[odds26.race_id.isin(external_ids)].copy()

    races = pd.concat([races25, races26], ignore_index=True, sort=False)
    riders = pd.concat([riders25, riders26], ignore_index=True, sort=False)
    results = pd.concat([results25, results26], ignore_index=True, sort=False)
    df = baseline.add_prerace_features(riders, races, results)
    labels = results[["race_id", "car_no", "finish_order", "actual_back", "actual_start", "winning_move", "final_lap_time", "result_comment"]]
    df = df.merge(labels, on=["race_id", "car_no"], how="inner")
    df = df[df.finish_order.gt(0)].copy()
    df["y_win"] = df.finish_order.eq(1).astype(int)
    df["y_top2"] = df.finish_order.le(2).astype(int)
    df["y_top3"] = df.finish_order.le(3).astype(int)
    df["y_back"] = df.actual_back.fillna(0).astype(int)

    component = df[df.race_date.le(baseline.COMPONENT_END)].copy()
    pair_train = df[df.race_date.gt(baseline.COMPONENT_END) & df.race_date.le(baseline.PAIR_END)].copy()
    valid = df[df.race_date.gt(baseline.PAIR_END) & df.race_date.le(baseline.VALID_END)].copy()
    external = df[df.race_date.between(EXTERNAL_START, EXTERNAL_END)].copy()
    categories, _ = baseline.category_effects(component)
    _, used_categories = baseline.add_component_predictions(
        [component, pair_train, valid, external], component, categories
    )

    events = baseline.race_event_table(df)
    event_train = events[events.race_date.le(baseline.COMPONENT_END)]
    event_model = LogisticRegression(
        max_iter=500, class_weight="balanced", random_state=baseline.SEED
    ).fit(event_train[baseline.EVENT_FEATURES].fillna(0), event_train.early_battle_label)
    battle_maps = {}
    battle_metrics = []
    for name, frame in [("pair_train", pair_train), ("validation", valid), ("external_2026_01_02", external)]:
        event_frame = events[events.race_id.isin(frame.race_id.unique())].copy()
        event_frame["p_battle"] = event_model.predict_proba(event_frame[baseline.EVENT_FEATURES].fillna(0))[:, 1]
        battle_maps[name] = dict(zip(event_frame.race_id, event_frame.p_battle))
        battle_metrics.append({"split": name, **baseline.metric(event_frame.early_battle_label, event_frame.p_battle)})

    venue, _ = baseline.venue_tables(component)
    scenario_train = baseline.scenario_pair_rows(pair_train, battle_maps["pair_train"], venue)
    pair_model = baseline.fit_pair_model(scenario_train, results25)
    valid_pairs = baseline.predict_pairs(
        baseline.scenario_pair_rows(valid, battle_maps["validation"], venue), pair_model
    )
    external_pairs = baseline.predict_pairs(
        baseline.scenario_pair_rows(external, battle_maps["external_2026_01_02"], venue), pair_model
    )
    valid_market = baseline.attach_market(valid_pairs, odds25, races25, results25)
    external_market = baseline.attach_market(external_pairs, odds26, races26, results26)
    baseline.calibrate_pair_prob(valid_market, external_market)
    probability_reliability, odds_reliability, joint_reliability = baseline.reliability_tables(valid_market)
    external_market = baseline.apply_reliability(external_market, joint_reliability)

    # The only purchase call: PR #31's already-locked thresholds, with no 2026 search.
    _, bets = baseline.portfolio(external_market, **FROZEN_RULE)
    target_races = int(external.race_id.nunique())
    overall = segment_metrics(bets, target_races)
    monthly = []
    for month, month_races in races26.assign(month=races26.race_date.astype(str).str[:6]).groupby("month"):
        ids = set(month_races.race_id)
        monthly.append({"month": month, **segment_metrics(bets[bets.race_id.isin(ids)].copy(), len(ids))})

    topk = []
    ranked = external_market.sort_values(["race_id", "calibrated_probability"], ascending=[True, False])
    for k in [1, 2, 3, 4, 5]:
        topk.append({"k": k, "hit_rate": float(ranked.groupby("race_id").head(k).groupby("race_id").is_hit.max().mean())})

    report = {
        "execution": "Python actual run",
        "evaluation": "completely unused 2026-01-01..2026-02-28",
        "frozen_source": {"pull_request": FROZEN_PR, "merge_sha": FROZEN_MERGE_SHA},
        "logic_or_threshold_changes": False,
        "rule_search_on_2026": False,
        "frozen_rule": FROZEN_RULE,
        "training_splits": {
            "component": "2025-01-01..2025-06-30", "pair": "2025-07-01..2025-08-31",
            "calibration_and_reliability": "2025-09-01..2025-10-31",
        },
        "external_dataset": {
            "races": int(races26.race_id.nunique()), "rider_rows": int(len(riders26)),
            "odds_rows": int(len(odds26)), "result_rows": int(len(results26)),
            "period_min": str(int(races26.race_date.min())), "period_max": str(int(races26.race_date.max())),
        },
        "used_prior_categories_frozen_from_2025": used_categories,
        "overall": overall, "monthly": monthly, "topk_exacta": topk,
        "battle_metrics": battle_metrics,
        "high_payout_exclusion_definition": "Primary sensitivity removes the single largest hit; threshold sensitivities are also reported.",
        "notes": [
            "Probabilities were completed before joining odds for purchase EV.",
            "No 2026 result was used for fitting, calibration, reliability correction, rule selection, or threshold adjustment.",
            "The 20% deterioration sensitivity multiplies every realized winning payout by 0.8 while stake is unchanged.",
        ],
    }
    (out / "frozen_2026_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    bets.to_csv(out / "selected_bets_2026_01_02.csv", index=False, encoding="utf-8-sig")
    external_market.to_csv(out / "pair_predictions_2026_01_02.csv", index=False, encoding="utf-8-sig")
    external[["race_id", "race_date", "venue_name", "race_no", "car_no", "p_back", "p_win", "p_top2", "p_top3", "finish_order"]].to_csv(
        out / "rider_predictions_2026_01_02.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(monthly).to_csv(out / "monthly_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(battle_metrics).to_csv(out / "battle_metrics.csv", index=False, encoding="utf-8-sig")
    probability_reliability.to_csv(out / "frozen_probability_reliability.csv", index=False, encoding="utf-8-sig")
    odds_reliability.to_csv(out / "frozen_odds_reliability.csv", index=False, encoding="utf-8-sig")
    joint_reliability.to_csv(out / "frozen_joint_reliability.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
