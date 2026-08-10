#!/usr/bin/env python3
"""PR31固定 vs +「B取得4着以下で別線1着」前日補正をA/B評価する。

重要:
- PR31のcomponent/event/pair/isotonic/reliabilityはBASEで1回だけ学習・固定する。
- PLUS_J側ではそれらを一切再学習・再校正しない。
- 追加するのは2025H1で固定済みの +2.262pp 相当logit bonusだけ。
- 2025年11-12月と2026年1-2月では閾値・補正量を調整しない。
- 運用は3〜5点、3点未満は見送り。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

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
    """各現在走に、同一開催直前走の B4+ & 別線勝利ラベルを付与する。"""
    pre = riders[["race_id", "player_id", "car_no", "line_no"]].copy()
    race_cols = races[["race_id", "race_date", "start_date", "day_no", "venue_code"]].copy()
    h = pre.merge(race_cols, on="race_id", how="left")
    fact = results[["race_id", "car_no", "finish_order", "actual_back"]].copy()
    h = h.merge(fact, on=["race_id", "car_no"], how="left")

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
    known_lines = h.prev_line_no.notna() & h.prev_winner_line_no.notna()
    h["prior_J"] = (
        same
        & pd.to_numeric(h.prev_actual_back, errors="coerce").eq(1)
        & pd.to_numeric(h.prev_finish_order, errors="coerce").ge(4)
        & known_lines
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


def apply_frozen_component_predictions(frames, component_models, cats, used_base, add_j: bool):
    for y, raw in (("y_back", "p_back_raw"), ("y_win", "p_win_raw"), ("y_top2", "p_top2_raw"), ("y_top3", "p_top3_raw")):
        model = component_models[y]
        for f in frames:
            f[raw] = model.predict_proba(f[baseline.FEATURES].fillna(-99))[:, 1]

    for f in frames:
        bonus = np.zeros(len(f))
        for k in used_base:
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


def production_bets(market: pd.DataFrame) -> pd.DataFrame:
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
        return {
            "target_races": int(target_races), "purchase_races": 0, "purchase_rate": 0.0,
            "bets": 0, "avg_points": 0.0, "hits": 0, "race_hit_rate": 0.0,
            "stake_yen": 0.0, "return_yen": 0.0, "roi": 0.0,
            "max_losing_streak": 0, "max_drawdown_yen": 0.0,
            "odds_20pct_worse_roi": 0.0, "largest_hit_removed_roi": 0.0,
        }
    purchase_races = int(bets.race_id.nunique())
    stake = float(bets.stake.sum())
    ret = float(bets["return"].sum())
    race_profit = bets.groupby(["race_date", "race_id"], sort=True).apply(
        lambda x: float(x["return"].sum() - x.stake.sum()), include_groups=False
    )
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


def frozen_calibrate_and_reliability(valid_market: pd.DataFrame):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=.95).fit(
        valid_market.pair_probability, valid_market.is_hit
    )
    valid_market = valid_market.copy()
    valid_market["cal_raw"] = iso.predict(valid_market.pair_probability)
    valid_market["calibrated_probability"] = (
        valid_market.cal_raw / valid_market.groupby("race_id").cal_raw.transform("sum")
    )
    _, _, joint = baseline.reliability_tables(valid_market)
    return iso, joint


def apply_frozen_market_stack(market: pd.DataFrame, iso, joint) -> pd.DataFrame:
    d = market.copy()
    d["cal_raw"] = iso.predict(d.pair_probability)
    d["calibrated_probability"] = d.cal_raw / d.groupby("race_id").cal_raw.transform("sum")
    return baseline.apply_reliability(d, joint)


def market_for(frame, battle_map, venue, pair_model, odds, races, results, iso, joint):
    scenarios = baseline.scenario_pair_rows(frame, battle_map, venue)
    pairs = baseline.predict_pairs(scenarios, pair_model)
    raw_market = baseline.attach_market(pairs, odds, races, results)
    return apply_frozen_market_stack(raw_market, iso, joint)


def report_for(name, test_frame, ext_frame, test_market, ext_market):
    bets25 = production_bets(test_market)
    bets26 = production_bets(ext_market)
    day2_ids25 = set(test_frame.loc[test_frame.day_no.gt(1), "race_id"])
    day2_ids26 = set(ext_frame.loc[ext_frame.day_no.gt(1), "race_id"])
    return {
        "name": name,
        "2025_test": metrics(bets25, int(test_frame.race_id.nunique())),
        "2025_test_day2plus": metrics(bets25[bets25.race_id.isin(day2_ids25)].copy(), len(day2_ids25)),
        "2026_external": metrics(bets26, int(ext_frame.race_id.nunique())),
        "2026_external_day2plus": metrics(bets26[bets26.race_id.isin(day2_ids26)].copy(), len(day2_ids26)),
    }, bets25, bets26


def promotion_gate(base: dict, plus: dict) -> dict:
    periods = ["2025_test", "2026_external"]
    no_large_roi_drop = all(plus[p]["roi"] >= base[p]["roi"] * 0.95 for p in periods)
    no_large_dd_worse = all(plus[p]["max_drawdown_yen"] <= base[p]["max_drawdown_yen"] * 1.10 + 100 for p in periods)
    no_streak_worse = all(plus[p]["max_losing_streak"] <= base[p]["max_losing_streak"] + 2 for p in periods)
    robust_not_worse = all(plus[p]["largest_hit_removed_roi"] >= base[p]["largest_hit_removed_roi"] * 0.95 for p in periods)
    base_stake = sum(base[p]["stake_yen"] for p in periods)
    plus_stake = sum(plus[p]["stake_yen"] for p in periods)
    base_combined = sum(base[p]["return_yen"] for p in periods) / base_stake if base_stake else 0.0
    plus_combined = sum(plus[p]["return_yen"] for p in periods) / plus_stake if plus_stake else 0.0
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--external-dir", required=True)
    ap.add_argument("--output-dir", default="validated_prevday_eval")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    component0 = df[df.race_date.le(baseline.COMPONENT_END)].copy()
    pair0 = df[df.race_date.gt(baseline.COMPONENT_END) & df.race_date.le(baseline.PAIR_END)].copy()
    valid0 = df[df.race_date.gt(baseline.PAIR_END) & df.race_date.le(baseline.VALID_END)].copy()
    test0 = df[df.race_date.ge(baseline.TEST_START) & df.race_date.le(20251231)].copy()
    ext0 = df[df.race_date.between(EXTERNAL_START, EXTERNAL_END)].copy()

    # PR31 frozen component stack: one fit only.
    component_b, pair_b, valid_b, test_b, ext_b = [x.copy() for x in (component0, pair0, valid0, test0, ext0)]
    cats, _ = baseline.category_effects(component_b)
    component_models, used_base = baseline.add_component_predictions(
        [component_b, pair_b, valid_b, test_b, ext_b], component_b, cats
    )

    # PLUS_J uses exactly the same trained component models and base A-I effects.
    test_j, ext_j = test0.copy(), ext0.copy()
    apply_frozen_component_predictions([test_j, ext_j], component_models, cats, used_base, add_j=True)

    events = baseline.race_event_table(df)
    event_train = events[events.race_date.le(baseline.COMPONENT_END)]
    event_model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=baseline.SEED).fit(
        event_train[baseline.EVENT_FEATURES].fillna(0), event_train.early_battle_label
    )
    venue, _ = baseline.venue_tables(component_b)

    # PR31 frozen pair model: fit once on BASE pair period only.
    battle_maps = {}
    for key, frame in (("pair", pair_b), ("valid", valid_b), ("test", test_b), ("external", ext_b)):
        e = events[events.race_id.isin(frame.race_id.unique())].copy()
        e["p_battle"] = event_model.predict_proba(e[baseline.EVENT_FEATURES].fillna(0))[:, 1]
        battle_maps[key] = dict(zip(e.race_id, e.p_battle))

    pair_model = baseline.fit_pair_model(
        baseline.scenario_pair_rows(pair_b, battle_maps["pair"], venue), results25
    )

    # PR31 frozen calibration/reliability: fit once on BASE validation only.
    valid_pairs = baseline.predict_pairs(
        baseline.scenario_pair_rows(valid_b, battle_maps["valid"], venue), pair_model
    )
    valid_market_raw = baseline.attach_market(valid_pairs, odds25, races25, results25)
    iso, joint = frozen_calibrate_and_reliability(valid_market_raw)

    base_test_market = market_for(test_b, battle_maps["test"], venue, pair_model, odds25, races25, results25, iso, joint)
    base_ext_market = market_for(ext_b, battle_maps["external"], venue, pair_model, odds26, races26, results26, iso, joint)
    plus_test_market = market_for(test_j, battle_maps["test"], venue, pair_model, odds25, races25, results25, iso, joint)
    plus_ext_market = market_for(ext_j, battle_maps["external"], venue, pair_model, odds26, races26, results26, iso, joint)

    base_report, base25, base26 = report_for("PR31_FROZEN", test_b, ext_b, base_test_market, base_ext_market)
    plus_report, plus25, plus26 = report_for("PR31_PLUS_J", test_j, ext_j, plus_test_market, plus_ext_market)
    gate = promotion_gate(base_report, plus_report)

    j_counts = {}
    for name, frame in (
        ("2025H1", component0),
        ("2025JulOct", pd.concat([pair0, valid0], ignore_index=True)),
        ("2025NovDec", test0),
        ("2026JanFeb", ext0),
    ):
        m = frame.prior_J.eq(1)
        j_counts[name] = {"rows": int(m.sum()), "races": int(frame.loc[m, "race_id"].nunique())}

    report = {
        "execution": "Python actual run",
        "variant": "B取得4着以下で別線1着 only; PR31 frozen stack unchanged",
        "bonus_source": {
            "period": "2025H1 discovery",
            "top2_residual_pp": J_DISCOVERY_TOP2_RESIDUAL_PP,
            "logit_bonus": J_LOGIT_BONUS,
        },
        "rule": RULE,
        "used_prior_categories_base": used_base,
        "j_counts": j_counts,
        "base": base_report,
        "plus_j": plus_report,
        "promotion_gate": gate,
        "external_tuning": False,
        "frozen_stack": {
            "component_models_shared": True,
            "event_model_shared": True,
            "pair_model_shared": True,
            "isotonic_shared": True,
            "reliability_shared": True,
        },
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for model_name, model_report in (("base", base_report), ("plus_j", plus_report)):
        for p in ("2025_test", "2025_test_day2plus", "2026_external", "2026_external_day2plus"):
            rows.append({"model": model_name, "period": p, **model_report[p]})
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    base25.to_csv(out / "base_bets_2025_test.csv", index=False)
    plus25.to_csv(out / "plus_j_bets_2025_test.csv", index=False)
    base26.to_csv(out / "base_bets_2026_external.csv", index=False)
    plus26.to_csv(out / "plus_j_bets_2026_external.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
