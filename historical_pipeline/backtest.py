#!/usr/bin/env python3
"""Leakage-safe numerical keirin model and locked 2025 holdout backtest."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


TRAIN_END = 20250831
VALID_END = 20251031
TEST_START = 20251101
EV_GRID = [1.05, 1.10, 1.15, 1.20, 1.30, 1.40]
RANDOM_STATE = 20250809


BASE_FEATURES = [
    "score", "s_count", "h_count", "b_count", "escape", "makuri", "sashi", "mark",
    "finish_1", "finish_2", "finish_3", "finish_out", "win_rate", "top2_rate", "top3_rate",
    "gear_ratio", "scheduled_starters", "actual_starters", "line_count",
]
LINE_FEATURES = [
    "line_position", "line_size", "score_rel", "b_rel", "h_rel", "escape_rel", "makuri_rel",
    "line_score_max", "line_b_sum", "leader_count", "lead_conflict_index",
]
PRIOR_FEATURES = [
    "previous_order", "previous_lap", "previous_back", "previous_standing", "previous_accident",
    "recent5_avg_order", "recent5_back_rate", "recent5_standing_rate", "recent5_avg_lap",
    "lose_strong_prerace_score", "previous_lap_rel",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25, 25)))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric = set(BASE_FEATURES + LINE_FEATURES + PRIOR_FEATURES + ["car_no", "finish_order"])
    for col in numeric & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    by_race = df.groupby("race_id")
    for source, target in [("score", "score_rel"), ("b_count", "b_rel"), ("h_count", "h_rel"),
                           ("escape", "escape_rel"), ("makuri", "makuri_rel")]:
        df[target] = df[source] - by_race[source].transform("mean")
    df["line_score_max"] = df.groupby(["race_id", "line_no"])["score"].transform("max")
    df["line_b_sum"] = df.groupby(["race_id", "line_no"])["b_count"].transform("sum")
    leaders = (df["line_position"] == 1).astype(int)
    df["leader_count"] = leaders.groupby(df["race_id"]).transform("sum")
    leader_b = df["b_count"].where(df["line_position"] == 1, 0)
    top = leader_b.groupby(df["race_id"]).transform("max")
    second = leader_b.where(leader_b < top, 0).groupby(df["race_id"]).transform("max")
    df["lead_conflict_index"] = second / (top + 1.0)
    df["previous_lap_rel"] = df["previous_lap"] - df["recent5_avg_lap"]
    return df


def fit_model(train: pd.DataFrame, features: list[str], label: str):
    model = HistGradientBoostingClassifier(
        learning_rate=0.055, max_iter=220, max_leaf_nodes=31, min_samples_leaf=35,
        l2_regularization=1.2, random_state=RANDOM_STATE,
    )
    model.fit(train[features].fillna(-99), train[label].astype(int))
    return model


def metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "rows": int(len(y)), "positive_rate": float(np.mean(y)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, p)),
    }


def normalize_race_prob(df: pd.DataFrame, raw_col: str, out_col: str, total: float) -> None:
    denom = df.groupby("race_id")[raw_col].transform("sum").replace(0, np.nan)
    df[out_col] = np.minimum(1.0, df[raw_col] * total / denom).fillna(total / df.groupby("race_id")[raw_col].transform("count"))


def venue_adjust(train: pd.DataFrame, target: pd.DataFrame, pred_col: str, label: str) -> np.ndarray:
    residual = train[label] - train[pred_col]
    stats = train.assign(_res=residual).groupby("venue_code")["_res"].agg(["sum", "count"])
    adjustment = (stats["sum"] / (stats["count"] + 250)).to_dict()
    base = np.clip(target[pred_col].to_numpy(), 1e-5, 1 - 1e-5)
    delta = target["venue_code"].map(adjustment).fillna(0).to_numpy()
    return sigmoid(np.log(base / (1 - base)) + delta)


def build_pairs(riders: pd.DataFrame, odds: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    first = riders[["race_id", "car_no", "line_no", "line_position", "score", "p_win", "p_top2", "p_back"]].copy()
    second = first.copy()
    first.columns = ["race_id"] + [f"first_{x}" for x in first.columns[1:]]
    second.columns = ["race_id"] + [f"second_{x}" for x in second.columns[1:]]
    pairs = first.merge(second, on="race_id")
    pairs = pairs[pairs["first_car_no"] != pairs["second_car_no"]].copy()
    denom = pairs.groupby(["race_id", "first_car_no"])["second_p_top2"].transform("sum")
    pairs["base_pair_prob"] = pairs["first_p_win"] * pairs["second_p_top2"] / denom.replace(0, np.nan)
    pairs["same_line"] = (pairs["first_line_no"] == pairs["second_line_no"]).astype(int)
    pairs["line_order"] = (pairs["same_line"].eq(1) &
                           (pairs["second_line_position"] == pairs["first_line_position"] + 1)).astype(int)
    pairs["score_diff"] = pairs["first_score"] - pairs["second_score"]
    odds2 = odds.rename(columns={"first_car": "first_car_no", "second_car": "second_car_no"})
    pairs = pairs.merge(odds2[["race_id", "first_car_no", "second_car_no", "exacta_odds"]],
                        on=["race_id", "first_car_no", "second_car_no"], how="inner")
    pairs = pairs.merge(races[["race_id", "race_date", "venue_name", "race_no",
                               "two_car_exacta_combination", "two_car_exacta_payout_yen"]], on="race_id")
    pairs["combination"] = pairs["first_car_no"].astype(int).astype(str) + "-" + pairs["second_car_no"].astype(int).astype(str)
    pairs["is_hit"] = (pairs["combination"] == pairs["two_car_exacta_combination"]).astype(int)
    return pairs


PAIR_FEATURES = ["log_base", "same_line", "line_order", "score_diff", "first_p_back",
                 "second_p_back", "first_line_position", "second_line_position"]


def calibrate_pairs(train: pd.DataFrame, target: pd.DataFrame) -> tuple[LogisticRegression, np.ndarray]:
    for frame in (train, target):
        frame["log_base"] = np.log(np.clip(frame["base_pair_prob"], 1e-8, 1))
    model = LogisticRegression(C=0.4, max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(train[PAIR_FEATURES].fillna(0), train["is_hit"])
    raw = model.predict_proba(target[PAIR_FEATURES].fillna(0))[:, 1]
    denom = pd.Series(raw, index=target.index).groupby(target["race_id"]).transform("sum").to_numpy()
    return model, raw / np.where(denom > 0, denom, 1)


def portfolio(pairs: pd.DataFrame, threshold: float, odds_factor: float = 1.0) -> dict:
    work = pairs.copy()
    work["ev"] = work["pair_probability"] * work["exacta_odds"]
    selected = []
    for _, group in work.sort_values(["race_id", "ev"], ascending=[True, False]).groupby("race_id"):
        picks = group[group["ev"] >= threshold].head(5)
        if len(picks) >= 3:
            selected.append(picks)
    if not selected:
        return {"threshold": threshold, "races": 0, "bets": 0, "roi": 0, "hit_rate": 0,
                "max_losing_streak": 0, "max_drawdown_yen": 0, "profit_yen": 0}
    bet = pd.concat(selected).sort_values(["race_date", "race_id", "ev"], ascending=[True, True, False])
    bet["stake"] = 100
    payout = pd.to_numeric(bet["two_car_exacta_payout_yen"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
    bet["return"] = np.where(bet["is_hit"].eq(1), payout * odds_factor, 0)
    race_profit = bet.groupby(["race_date", "race_id"], sort=True).apply(
        lambda x: float(x["return"].sum() - x["stake"].sum()), include_groups=False)
    equity = race_profit.cumsum(); peak = equity.cummax(); drawdown = peak - equity
    losses = (race_profit < 0).astype(int).to_list(); max_streak = streak = 0
    for loss in losses:
        streak = streak + 1 if loss else 0; max_streak = max(max_streak, streak)
    stake = float(bet["stake"].sum()); returned = float(bet["return"].sum())
    return {
        "threshold": threshold, "races": int(bet["race_id"].nunique()), "bets": int(len(bet)),
        "hits": int(bet["is_hit"].sum()), "hit_rate": float(bet["is_hit"].mean()),
        "stake_yen": stake, "return_yen": returned, "profit_yen": returned - stake,
        "roi": returned / stake if stake else 0, "max_losing_streak": int(max_streak),
        "max_drawdown_yen": float(drawdown.max()) if len(drawdown) else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset-dir", default="dataset"); ap.add_argument("--output-dir", default="backtest")
    args = ap.parse_args(); src = Path(args.dataset_dir); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    riders = pd.read_csv(src / "rider_features_2025.csv", dtype={"race_id": str, "venue_code": str})
    results = pd.read_csv(src / "official_results_2025.csv", dtype={"race_id": str})
    odds = pd.read_csv(src / "exacta_odds_2025.csv", dtype={"race_id": str})
    races = pd.read_csv(src / "races_2025.csv", dtype={"race_id": str, "venue_code": str})
    for col in ["first_car", "second_car", "exacta_odds"]: odds[col] = pd.to_numeric(odds[col], errors="coerce")
    results["finish_order"] = pd.to_numeric(results["finish_order"], errors="coerce")
    results = results[results["finish_order"] > 0].copy()
    results["car_no"] = pd.to_numeric(results["car_no"], errors="coerce")
    riders["car_no"] = pd.to_numeric(riders["car_no"], errors="coerce")
    df = riders.merge(results[["race_id", "car_no", "finish_order", "actual_back", "actual_start",
                                "winning_move", "final_lap_time", "result_comment"]], on=["race_id", "car_no"], how="inner")
    df["race_date"] = pd.to_numeric(df["race_date"], errors="coerce")
    df = add_features(df)
    df["y_win"] = (df["finish_order"] == 1).astype(int); df["y_top2"] = (df["finish_order"] <= 2).astype(int)
    df["y_top3"] = (df["finish_order"] <= 3).astype(int); df["y_back"] = pd.to_numeric(df["actual_back"], errors="coerce").fillna(0).astype(int)
    train = df[df["race_date"] <= TRAIN_END].copy(); valid = df[(df["race_date"] > TRAIN_END) & (df["race_date"] <= VALID_END)].copy(); test = df[df["race_date"] >= TEST_START].copy()
    full_features = BASE_FEATURES + LINE_FEATURES + PRIOR_FEATURES
    stage_metrics = []
    for name, feats in [("能力数値", BASE_FEATURES), ("ライン・相対差", BASE_FEATURES + LINE_FEATURES),
                        ("負けて強し・上がり", full_features)]:
        model = fit_model(train, feats, "y_win"); pred = model.predict_proba(test[feats].fillna(-99))[:, 1]
        stage_metrics.append({"stage": name, **metrics(test["y_win"], pred)})

    models = {}
    for label, out_name in [("y_win", "p_win_raw"), ("y_top2", "p_top2_raw"), ("y_top3", "p_top3_raw"), ("y_back", "p_back")]:
        model = fit_model(train, full_features, label); models[label] = model
        for frame in (train, valid, test): frame[out_name] = model.predict_proba(frame[full_features].fillna(-99))[:, 1]
    for frame in (train, valid, test):
        normalize_race_prob(frame, "p_win_raw", "p_win", 1.0); normalize_race_prob(frame, "p_top2_raw", "p_top2", 2.0); normalize_race_prob(frame, "p_top3_raw", "p_top3", 3.0)
    test["p_win_venue"] = venue_adjust(train, test, "p_win", "y_win")
    normalize_race_prob(test, "p_win_venue", "p_win_venue_norm", 1.0)
    stage_metrics.append({"stage": "場別補正", **metrics(test["y_win"], test["p_win_venue_norm"])})
    for label, pred, title in [("y_back", "p_back", "バック取得確率"), ("y_win", "p_win", "1着率"),
                               ("y_top2", "p_top2", "2連対率"), ("y_top3", "p_top3", "3連対率")]:
        stage_metrics.append({"stage": title, **metrics(test[label], test[pred])})

    # Race-level development labels and probabilities.
    race_events = df.groupby("race_id").agg(
        race_date=("race_date", "first"), lead_conflict_index=("lead_conflict_index", "first"),
        max_b=("b_count", "max"), mean_b=("b_count", "mean"), line_count=("line_count", "first"),
        actual_back_car=("car_no", lambda x: 0),
    ).reset_index()
    event_labels = df.groupby("race_id").apply(lambda g: pd.Series({
        "lead_battle": int(any(re.search("先行争|叩き合|踏み合", str(x)) for x in g["result_comment"].fillna(""))),
        "bandte_benefit": int(any((g["finish_order"] == 1) & (g["line_position"] > 1) & g["winning_move"].isin(["差", "マ"]))),
        "cross_line_makuri": int(any((g["finish_order"] == 1) & (g["winning_move"] == "捲"))),
        "development_benefit": int(any((g["finish_order"] <= 2) & (g["score_rel"] < -2.0))),
    }), include_groups=False).reset_index()
    race_events = race_events.merge(event_labels, on="race_id")
    event_features = ["lead_conflict_index", "max_b", "mean_b", "line_count"]
    for label, title in [("lead_battle", "先行争い発生率"), ("bandte_benefit", "番手恩恵"),
                         ("cross_line_makuri", "別線捲り成功率"), ("development_benefit", "展開恩恵型")]:
        tr = race_events[race_events["race_date"] <= TRAIN_END]; te = race_events[race_events["race_date"] >= TEST_START]
        if tr[label].nunique() < 2 or te[label].nunique() < 2:
            stage_metrics.append({"stage": title, "rows": int(len(te)), "positive_rate": float(te[label].mean()), "auc": None, "note": "label sparse"})
        else:
            m = LogisticRegression(max_iter=300, class_weight="balanced", random_state=RANDOM_STATE).fit(tr[event_features].fillna(0), tr[label])
            stage_metrics.append({"stage": title, **metrics(te[label], m.predict_proba(te[event_features].fillna(0))[:, 1])})

    # Pair probabilities and market backtest. Pair calibration uses train only.
    all_riders = pd.concat([train, valid, test], ignore_index=True)
    pairs = build_pairs(all_riders, odds, races)
    pair_train = pairs[pairs["race_date"] <= TRAIN_END].copy(); pair_valid = pairs[(pairs["race_date"] > TRAIN_END) & (pairs["race_date"] <= VALID_END)].copy(); pair_test = pairs[pairs["race_date"] >= TEST_START].copy()
    pair_model, pair_valid["pair_probability"] = calibrate_pairs(pair_train, pair_valid)
    for frame in (pair_train, pair_test): frame["log_base"] = np.log(np.clip(frame["base_pair_prob"], 1e-8, 1))
    raw = pair_model.predict_proba(pair_test[PAIR_FEATURES].fillna(0))[:, 1]
    pair_test["pair_probability"] = raw / pd.Series(raw, index=pair_test.index).groupby(pair_test["race_id"]).transform("sum").to_numpy()
    valid_runs = [portfolio(pair_valid, threshold) for threshold in EV_GRID]
    eligible = [x for x in valid_runs if x["bets"] >= 500]
    locked = max(eligible or valid_runs, key=lambda x: (x["roi"], x["bets"]))["threshold"]
    test_run = portfolio(pair_test, locked)
    tolerance = {f"odds_minus_{x}%": portfolio(pair_test, locked, 1 - x / 100)["roi"] for x in [5, 10, 15, 20]}
    top1 = pair_test.sort_values(["race_id", "pair_probability"], ascending=[True, False]).groupby("race_id").head(1)
    top3 = pair_test.sort_values(["race_id", "pair_probability"], ascending=[True, False]).groupby("race_id").head(3)
    stage_metrics.append({"stage": "展開条件付き2車単確率", "rows": int(pair_test["race_id"].nunique()),
                          "top1_hit_rate": float(top1["is_hit"].mean()), "top3_hit_rate": float(top3.groupby("race_id")["is_hit"].max().mean()),
                          "multiclass_logloss": float(-np.log(np.clip(pair_test.loc[pair_test["is_hit"].eq(1), "pair_probability"], 1e-9, 1)).mean())})
    stage_metrics.append({"stage": "EV・実回収率", **test_run, **tolerance})

    report = {
        "execution": "Python actual run", "random_seed": RANDOM_STATE,
        "split": {"train": "2025-01-01..2025-08-31", "validation": "2025-09-01..2025-10-31", "locked_test": "2025-11-01..2025-12-31"},
        "no_player_identity_features": True, "validation_threshold_grid": EV_GRID,
        "locked_ev_threshold": locked, "stage_metrics": stage_metrics,
        "validation_portfolios": valid_runs, "locked_test_portfolio": test_run,
        "odds_deterioration_roi": tolerance,
        "definitions": {
            "lead_battle": "result comment contains 先行争/叩き合/踏み合; sparse labels are explicitly reported",
            "bandte_benefit": "winner is line position 2+ and winning move is 差 or マ",
            "cross_line_makuri": "winner move is 捲; exacta pair line relation is modeled separately",
            "development_benefit": "top-two finisher whose pre-race score was 2+ below race mean",
            "lose_strong": "previous finish 4+ plus previous B/S and relative prior lap; current result is never used",
        },
    }
    (out / "backtest_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(stage_metrics).to_csv(out / "stage_metrics.csv", index=False, encoding="utf-8-sig")
    pair_test[["race_id", "race_date", "venue_name", "race_no", "combination", "pair_probability", "exacta_odds", "is_hit"]].to_csv(out / "test_pair_predictions.csv", index=False, encoding="utf-8-sig")
    test[["race_id", "race_date", "venue_code", "car_no", "line_no", "line_position", "p_win", "p_top2", "p_top3", "p_back", "finish_order"]].to_csv(out / "test_rider_predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
