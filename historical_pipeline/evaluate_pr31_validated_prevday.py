#!/usr/bin/env python3
"""本番PR31 Frozenそのもの vs +「B取得4着以下で別線1着」をA/B評価する。

PR31側のcomponent/event/pair/isotonic/reliability/venueは本番Frozen bundleをそのまま使用し、
PLUS_J側では一切再学習・再校正しない。追加するのは2025H1で固定した+2.262pp相当の
logit bonusだけ。2025年11-12月と2026年1-2月から補正量・閾値を調整しない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import chatgpt_baseline_backtest as baseline

RULE = {"max_points": 5, "min_ev": 2.0, "min_prob": 0.03, "confidence_max": 1.0, "min_points": 3}
EXTERNAL_START = 20260101
EXTERNAL_END = 20260228
J_DISCOVERY_TOP2_RESIDUAL_PP = 2.262
J_LOGIT_BONUS = J_DISCOVERY_TOP2_RESIDUAL_PP / 100.0


def load_period(root: Path, label: str):
    races = pd.read_csv(root / f"races_{label}.csv", dtype={"race_id": str, "venue_code": str})
    riders = pd.read_csv(root / f"rider_features_{label}.csv", dtype={"race_id": str, "venue_code": str, "player_id": str})
    results = pd.read_csv(root / f"official_results_{label}.csv", dtype={"race_id": str, "player_id": str})
    odds = pd.read_csv(root / f"exacta_odds_{label}.csv", dtype={"race_id": str})
    for d in (races, riders, results, odds):
        for c in ("car_no", "first_car", "second_car", "finish_order", "actual_back", "actual_start", "exacta_odds", "race_date", "line_no"):
            if c in d:
                d[c] = pd.to_numeric(d[c], errors="coerce")
    return races, riders, results, odds


def prior_j_map(riders: pd.DataFrame, races: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """同一開催の直前走だけから B取得4着以下＋別線1着 を作る。"""
    pre = riders[["race_id", "player_id", "car_no", "line_no"]].copy()
    race_cols = races[["race_id", "race_date", "start_date", "day_no", "venue_code"]].copy()
    h = pre.merge(race_cols, on="race_id", how="left")
    h = h.merge(results[["race_id", "car_no", "finish_order", "actual_back"]], on=["race_id", "car_no"], how="left")

    winners = results.loc[pd.to_numeric(results.finish_order, errors="coerce").eq(1), ["race_id", "car_no"]].copy()
    winners = winners.sort_values(["race_id", "car_no"]).drop_duplicates("race_id")
    winner_line = winners.merge(pre[["race_id", "car_no", "line_no"]], on=["race_id", "car_no"], how="left")
    winner_line = winner_line[["race_id", "line_no"]].rename(columns={"line_no": "winner_line_no"})
    h = h.merge(winner_line, on="race_id", how="left")

    h = h.sort_values(["player_id", "race_date", "race_id"]).copy()
    for c in ("start_date", "venue_code", "race_date", "finish_order", "actual_back", "line_no", "winner_line_no"):
        h[f"prev_{c}"] = h.groupby("player_id", sort=False)[c].shift(1)
    same = (
        h.day_no.gt(1)
        & h.start_date.eq(h.prev_start_date)
        & h.venue_code.eq(h.prev_venue_code)
        & h.race_date.gt(h.prev_race_date)
    )
    known = h.prev_line_no.notna() & h.prev_winner_line_no.notna()
    h["prior_J"] = (
        same
        & pd.to_numeric(h.prev_actual_back, errors="coerce").eq(1)
        & pd.to_numeric(h.prev_finish_order, errors="coerce").ge(4)
        & known
        & h.prev_line_no.ne(h.prev_winner_line_no)
    ).astype(int)
    return h[["race_id", "car_no", "prior_J"]]


def prepare_df(riders, races, results):
    df = baseline.add_prerace_features(riders, races, results)
    df = df.merge(prior_j_map(riders, races, results), on=["race_id", "car_no"], how="left")
    df["prior_J"] = pd.to_numeric(df.prior_J, errors="coerce").fillna(0).astype(int)
    lab = results[["race_id", "car_no", "finish_order", "actual_back", "actual_start", "winning_move", "final_lap_time", "result_comment"]]
    df = df.merge(lab, on=["race_id", "car_no"], how="inner")
    df = df[df.finish_order.gt(0)].copy()
    df["y_win"] = df.finish_order.eq(1).astype(int)
    df["y_top2"] = df.finish_order.le(2).astype(int)
    df["y_top3"] = df.finish_order.le(3).astype(int)
    df["y_back"] = df.actual_back.fillna(0).astype(int)
    return df


def apply_frozen_component_predictions(frames, bundle, add_j: bool):
    cats = bundle["category_effects"]
    used = list(bundle["used_prior_categories"])
    for y, raw in (("y_back", "p_back_raw"), ("y_win", "p_win_raw"), ("y_top2", "p_top2_raw"), ("y_top3", "p_top3_raw")):
        model = bundle["component_models"][y]
        for f in frames:
            f[raw] = model.predict_proba(f[baseline.FEATURES].fillna(-99))[:, 1]
    for f in frames:
        bonus = np.zeros(len(f))
        for k in used:
            lift = float(cats.loc[cats.category.eq(k), "y_top2_lift_pp"].iloc[0]) / 100.0
            bonus += f[f"prior_{k}"].to_numpy() * np.clip(lift, -0.08, 0.08)
        if add_j:
            bonus += f.prior_J.to_numpy() * J_LOGIT_BONUS
        for raw in ("p_win_raw", "p_top2_raw", "p_top3_raw"):
            f[raw] = baseline.sigmoid(baseline.safe_logit(f[raw]) + bonus)
        f["p_back_candidate"] = np.where(f.is_self_power.eq(1), f.p_back_raw, 0)
        baseline.normalize_by_race(f, "p_back_candidate", "p_back", 1)
        baseline.normalize_by_race(f, "p_win_raw", "p_win", 1)
        baseline.normalize_by_race(f, "p_top2_raw", "p_top2", 2)
        baseline.normalize_by_race(f, "p_top3_raw", "p_top3", 3)


def battle_map(frame: pd.DataFrame, bundle) -> dict:
    events = baseline.race_event_table(frame)
    if events.empty:
        return {}
    events["p_battle"] = bundle["event_model"].predict_proba(events[baseline.EVENT_FEATURES].fillna(0))[:, 1]
    return dict(zip(events.race_id, events.p_battle))


def apply_frozen_market_stack(raw_market: pd.DataFrame, bundle) -> pd.DataFrame:
    d = raw_market.copy()
    iso = bundle["isotonic"]
    d["cal_raw"] = iso.predict(d.pair_probability)
    d["calibrated_probability"] = d.cal_raw / d.groupby("race_id").cal_raw.transform("sum")
    return baseline.apply_reliability(d, bundle["joint_reliability"])


def market_for(frame, battles, bundle, odds, races, results):
    if frame.empty:
        return pd.DataFrame()
    scenarios = baseline.scenario_pair_rows(frame, battles, bundle["venue"])
    pairs = baseline.predict_pairs(scenarios, bundle["pair_model"])
    raw = baseline.attach_market(pairs, odds, races, results)
    return apply_frozen_market_stack(raw, bundle)


def production_bets(market: pd.DataFrame) -> pd.DataFrame:
    if market.empty:
        return market.copy()
    q = market[
        (market.ev >= RULE["min_ev"])
        & (market.purchase_probability >= RULE["min_prob"])
        & (market.race_entropy <= RULE["confidence_max"])
    ].copy()
    q = q.sort_values(["race_id", "ev", "calibrated_probability"], ascending=[True, False, False])
    q = q[q.groupby("race_id").cumcount() < RULE["max_points"]]
    counts = q.groupby("race_id").race_id.transform("size")
    q = q[counts >= RULE["min_points"]].copy()
    if q.empty:
        q["stake"] = pd.Series(dtype=float)
        q["return"] = pd.Series(dtype=float)
        return q
    q = q.sort_values(["race_date", "race_id", "ev"], ascending=[True, True, False])
    q["stake"] = 100.0
    payout = pd.to_numeric(q.two_car_exacta_payout_yen, errors="coerce").fillna(0.0)
    q["return"] = np.where(q.is_hit.eq(1), payout, 0.0)
    return q


def metrics(bets: pd.DataFrame, target_races: int) -> dict:
    if bets.empty:
        return {"target_races": int(target_races), "purchase_races": 0, "purchase_rate": 0.0, "bets": 0, "avg_points": 0.0, "hits": 0, "race_hit_rate": 0.0, "stake_yen": 0.0, "return_yen": 0.0, "roi": 0.0, "max_losing_streak": 0, "max_drawdown_yen": 0.0, "odds_20pct_worse_roi": 0.0, "largest_hit_removed_roi": 0.0}
    purchase_races = int(bets.race_id.nunique())
    stake = float(bets.stake.sum())
    ret = float(bets["return"].sum())
    race_profit = bets.groupby(["race_date", "race_id"], sort=True).apply(lambda x: float(x["return"].sum() - x.stake.sum()), include_groups=False)
    streak = mx = 0
    for loss in race_profit.lt(0):
        streak = streak + 1 if loss else 0
        mx = max(mx, streak)
    equity = race_profit.cumsum().to_numpy()
    peak = np.maximum.accumulate(np.r_[0.0, equity])[1:]
    returns = pd.to_numeric(bets["return"], errors="coerce").fillna(0.0)
    largest = float(returns.max()) if len(returns) else 0.0
    return {
        "target_races": int(target_races), "purchase_races": purchase_races,
        "purchase_rate": purchase_races / target_races if target_races else 0.0,
        "bets": int(len(bets)), "avg_points": len(bets) / purchase_races,
        "hits": int(bets.is_hit.sum()), "race_hit_rate": float(bets.groupby("race_id").is_hit.max().mean()),
        "stake_yen": stake, "return_yen": ret, "roi": ret / stake if stake else 0.0,
        "max_losing_streak": int(mx), "max_drawdown_yen": float(np.max(peak - equity)) if len(equity) else 0.0,
        "odds_20pct_worse_roi": float((returns * 0.8).sum() / stake),
        "largest_hit_removed_roi": float((ret - largest) / stake),
    }


def report_for(name, full_frame, market):
    bets = production_bets(market)
    day2_ids = set(full_frame.loc[full_frame.day_no.gt(1), "race_id"])
    return {
        "name": name,
        "all": metrics(bets, int(full_frame.race_id.nunique())),
        "day2plus": metrics(bets[bets.race_id.isin(day2_ids)].copy(), len(day2_ids)),
    }, bets


def promotion_gate(base25, plus25, base26, plus26):
    pairs = ((base25["all"], plus25["all"]), (base26["all"], plus26["all"]))
    no_large_roi_drop = all(p["roi"] >= b["roi"] * 0.95 for b, p in pairs)
    no_large_dd_worse = all(p["max_drawdown_yen"] <= b["max_drawdown_yen"] * 1.10 + 100 for b, p in pairs)
    no_streak_worse = all(p["max_losing_streak"] <= b["max_losing_streak"] + 2 for b, p in pairs)
    robust_not_worse = all(p["largest_hit_removed_roi"] >= b["largest_hit_removed_roi"] * 0.95 for b, p in pairs)
    base_stake = base25["all"]["stake_yen"] + base26["all"]["stake_yen"]
    plus_stake = plus25["all"]["stake_yen"] + plus26["all"]["stake_yen"]
    base_ret = base25["all"]["return_yen"] + base26["all"]["return_yen"]
    plus_ret = plus25["all"]["return_yen"] + plus26["all"]["return_yen"]
    base_combined = base_ret / base_stake if base_stake else 0.0
    plus_combined = plus_ret / plus_stake if plus_stake else 0.0
    passed = bool(no_large_roi_drop and no_large_dd_worse and no_streak_worse and robust_not_worse and plus_combined > base_combined)
    return {
        "passed": passed,
        "rules_fixed_before_external": True,
        "pr31_models_retrained_for_plus_j": False,
        "pr31_calibration_retrained_for_plus_j": False,
        "no_large_roi_drop": no_large_roi_drop,
        "no_large_dd_worse": no_large_dd_worse,
        "no_streak_worse": no_streak_worse,
        "largest_hit_removed_not_worse": robust_not_worse,
        "base_combined_roi": base_combined,
        "plus_combined_roi": plus_combined,
        "decision": "PROMOTE_TO_PROBABILITY" if passed else "DISPLAY_ONLY_KEEP_PR31_FROZEN",
    }


def combine_with_affected(base_market, affected_market, affected_ids):
    if not affected_ids:
        return base_market.copy()
    return pd.concat([base_market[~base_market.race_id.isin(affected_ids)], affected_market], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--external-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", default="validated_prevday_eval")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.model)
    if bundle.get("bundle_version") != "pr31-frozen-1":
        raise RuntimeError("PR31_FROZEN_MODEL_VERSION_MISMATCH")

    races25, riders25, results25, odds25 = load_period(Path(args.train_dir), "2025")
    races26, riders26, results26, odds26 = load_period(Path(args.external_dir), "2026_01_02")
    races26 = races26[races26.race_date.between(EXTERNAL_START, EXTERNAL_END)].copy()
    ids26 = set(races26.race_id)
    riders26 = riders26[riders26.race_id.isin(ids26)].copy()
    results26 = results26[results26.race_id.isin(ids26)].copy()
    odds26 = odds26[odds26.race_id.isin(ids26)].copy()

    races_all = pd.concat([races25, races26], ignore_index=True, sort=False)
    riders_all = pd.concat([riders25, riders26], ignore_index=True, sort=False)
    results_all = pd.concat([results25, results26], ignore_index=True, sort=False)
    df = prepare_df(riders_all, races_all, results_all)
    test0 = df[df.race_date.ge(baseline.TEST_START) & df.race_date.le(20251231)].copy()
    ext0 = df[df.race_date.between(EXTERNAL_START, EXTERNAL_END)].copy()

    test_b, ext_b = test0.copy(), ext0.copy()
    apply_frozen_component_predictions([test_b, ext_b], bundle, add_j=False)
    battles25 = battle_map(test_b, bundle)
    battles26 = battle_map(ext_b, bundle)
    base_market25 = market_for(test_b, battles25, bundle, odds25, races25, results25)
    base_market26 = market_for(ext_b, battles26, bundle, odds26, races26, results26)

    affected25 = set(test0.loc[test0.prior_J.eq(1), "race_id"])
    affected26 = set(ext0.loc[ext0.prior_J.eq(1), "race_id"])
    test_j = test0[test0.race_id.isin(affected25)].copy()
    ext_j = ext0[ext0.race_id.isin(affected26)].copy()
    apply_frozen_component_predictions([test_j, ext_j], bundle, add_j=True)
    plus_aff25 = market_for(test_j, {k: v for k, v in battles25.items() if k in affected25}, bundle, odds25, races25, results25) if affected25 else pd.DataFrame()
    plus_aff26 = market_for(ext_j, {k: v for k, v in battles26.items() if k in affected26}, bundle, odds26, races26, results26) if affected26 else pd.DataFrame()
    plus_market25 = combine_with_affected(base_market25, plus_aff25, affected25)
    plus_market26 = combine_with_affected(base_market26, plus_aff26, affected26)

    base25, base_bets25 = report_for("PR31_FROZEN", test0, base_market25)
    plus25, plus_bets25 = report_for("PR31_PLUS_J", test0, plus_market25)
    base26, base_bets26 = report_for("PR31_FROZEN", ext0, base_market26)
    plus26, plus_bets26 = report_for("PR31_PLUS_J", ext0, plus_market26)
    gate = promotion_gate(base25, plus25, base26, plus26)

    report = {
        "execution": "Python actual run",
        "model_source": "production pr31-runtime-v1/pr31_frozen.joblib",
        "variant": "B取得4着以下で別線1着 only; PR31 Frozen stack unchanged",
        "bonus_source": {"period": "2025H1 discovery", "top2_residual_pp": J_DISCOVERY_TOP2_RESIDUAL_PP, "logit_bonus": J_LOGIT_BONUS},
        "rule": RULE,
        "used_prior_categories_base": list(bundle["used_prior_categories"]),
        "affected_races": {"2025NovDec": len(affected25), "2026JanFeb": len(affected26)},
        "base": {"2025_test": base25, "2026_external": base26},
        "plus_j": {"2025_test": plus25, "2026_external": plus26},
        "promotion_gate": gate,
        "external_tuning": False,
        "frozen_stack": {"component_models_shared": True, "event_model_shared": True, "pair_model_shared": True, "isotonic_shared": True, "reliability_shared": True, "venue_shared": True},
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for model_name, y25, y26 in (("base", base25, base26), ("plus_j", plus25, plus26)):
        for period, r in (("2025_test", y25), ("2026_external", y26)):
            rows.append({"model": model_name, "period": period, "scope": "all", **r["all"]})
            rows.append({"model": model_name, "period": period, "scope": "day2plus", **r["day2plus"]})
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    base_bets25.to_csv(out / "base_bets_2025_test.csv", index=False)
    plus_bets25.to_csv(out / "plus_j_bets_2025_test.csv", index=False)
    base_bets26.to_csv(out / "base_bets_2026_external.csv", index=False)
    plus_bets26.to_csv(out / "plus_j_bets_2026_external.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
