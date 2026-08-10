"""PR #31 Frozenを唯一の予測エンジンとして実行する本番ランタイム。"""
from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from historical_pipeline.chatgpt_baseline_backtest import (
    EVENT_FEATURES, FEATURES, ODDS_BINS, PAIR_FEATURES, PROB_BINS,
    normalize_by_race, predict_pairs, safe_logit, scenario_pair_rows, sigmoid,
)
from individual_api.keirin_pdf_adapter import _extract_text
from individual_api.previous_day_kdreams import VENUES, detect_day_no, fetch_previous_day

VERSION = "3.0.0-pr31-frozen"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "pr31_frozen.joblib"
RANK_LEVEL = {"A3": 1, "A2": 2, "A1": 3, "S2": 4, "S1": 5, "SS": 6}
STYLE_CODE = {"逃": 3, "両": 2, "追": 1}


def _load_bundle() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise RuntimeError("PR31_FROZEN_MODEL_MISSING")
    bundle = joblib.load(MODEL_PATH)
    if bundle.get("bundle_version") != "pr31-frozen-1":
        raise RuntimeError("PR31_FROZEN_MODEL_VERSION_MISMATCH")
    return bundle


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value or ""))


def _rank_from_text(text: str, name: str) -> str | None:
    compact = _compact(text)
    target = re.escape(_compact(name))
    m = re.search(target + r"[^/]{0,40}/(SS|S1|S2|A1|A2|A3|L1)/(?:逃|追|両)", compact)
    return m.group(1) if m else None


def _distance_from_text(text: str) -> float:
    normalized = unicodedata.normalize("NFKC", text or "").replace(",", "")
    values = [int(x) for x in re.findall(r"(?<!\d)(1[056]25|2[024]25|2000)m", normalized)]
    return float(values[0]) if values else 2000.0


def _wind_from_text(text: str) -> float:
    normalized = unicodedata.normalize("NFKC", text or "")
    m = re.search(r"風速\s*([0-9]+(?:\.[0-9]+)?)\s*m", normalized)
    return float(m.group(1)) if m else 0.0


def _identity(audit: dict[str, Any], text: str) -> tuple[str | None, str | None, int, str]:
    race = audit.get("race", {}) if isinstance(audit, dict) else {}
    venue = race.get("venue")
    date = race.get("date")
    race_no = int(race.get("race") or 0)
    venue_code = VENUES[venue][0] if venue in VENUES else "00"
    race_id = f"runtime-{date or 'unknown'}-{venue_code}-{race_no}"
    return venue, date, race_no, race_id


def _prior_for(name: str, prior: dict[str, Any]) -> dict[str, float]:
    row = prior.get("riders", {}).get(name, {}) if isinstance(prior, dict) else {}
    flags = row.get("pr31", {}) if isinstance(row, dict) else {}
    out = {f"prior_{k}": float(flags.get(k, 0)) for k in "ABCDEFGHI"}
    out["has_previous_day"] = float(bool(row))
    out["prior_finish"] = float(row.get("finish") or 0) if row else 0.0
    out["prior_back"] = float(row.get("actual_back") or 0) if row else 0.0
    out["prior_start"] = float(row.get("actual_start") or 0) if row else 0.0
    out["prior_lap_raw"] = float(row.get("final_lap_time") or 0) if row else 0.0
    return out


def _feature_frame(
    payload: dict[str, Any],
    audit: dict[str, Any],
    basic_text: str,
    prior: dict[str, Any],
    day_no: int,
) -> pd.DataFrame:
    riders = payload["riders"]
    lines = [[int(x) for x in line] for line in payload["lines"]]
    line_lookup: dict[int, tuple[int, int, int]] = {}
    for line_no, line in enumerate(lines, start=1):
        for pos, bike in enumerate(line, start=1):
            line_lookup[bike] = (line_no, pos, len(line))
    venue, date, race_no, race_id = _identity(audit, basic_text)
    date_num = int(str(date).replace("-", "")) if date else 0
    distance = _distance_from_text(basic_text)
    wind = _wind_from_text(basic_text)
    n = len(riders)
    rows = []
    for r in riders:
        bike = int(r["bike"])
        line_no, pos, size = line_lookup[bike]
        total = float(r["first"] + r["second"] + r["third"] + r["out"])
        rank = _rank_from_text(basic_text, str(r["name"]))
        row = {
            "race_id": race_id, "race_date": date_num, "venue_code": VENUES.get(venue, ("00", ""))[0],
            "venue_name": venue or "未取得", "race_no": race_no, "race_class": payload.get("grade", "UNKNOWN"),
            "scheduled_starters": n, "actual_starters": n, "line_count": len(lines),
            "car_no": bike, "player_id": "", "name": r["name"], "rank": rank or "",
            "style": r.get("style", "追"), "line_no": line_no, "line_position": pos, "line_size": size,
            "score": float(r["score"]), "s_count": float(r.get("S", 0)), "h_count": float(r["H"]),
            "b_count": float(r["B"]), "escape": float(r["escape"]), "makuri": float(r["makuri"]),
            "sashi": float(r["sashi"]), "mark": float(r["mark"]), "finish_1": float(r["first"]),
            "finish_2": float(r["second"]), "finish_3": float(r["third"]), "finish_out": float(r["out"]),
            "win_rate": float(r["win_rate"]), "top2_rate": float(r["quinella_rate"]),
            "top3_rate": 100.0 * float(r["first"] + r["second"] + r["third"]) / total if total else 0.0,
            "day_no": day_no, "distance_m": distance, "wind_speed": wind,
            "rank_level": float(RANK_LEVEL.get(rank or "", 0)), "style_code": float(STYLE_CODE.get(r.get("style", "追"), 0)),
        }
        row.update(_prior_for(str(r["name"]), prior))
        rows.append(row)
    f = pd.DataFrame(rows)

    # PR31学習時と同じ: 現在レースに出る各選手の前日上がり平均との差。
    lap_raw = pd.to_numeric(f.pop("prior_lap_raw"), errors="coerce").replace(0, np.nan)
    lap_mean = float(lap_raw.mean()) if lap_raw.notna().any() else math.nan
    f["prior_lap_rel"] = (lap_raw - lap_mean).fillna(0.0) if math.isfinite(lap_mean) else 0.0

    g = f.groupby("race_id", sort=False)
    for s, t in (("score", "score_rel"), ("b_count", "b_rel"), ("h_count", "h_rel"), ("escape", "escape_rel"), ("makuri", "makuri_rel"), ("sashi", "sashi_rel"), ("mark", "mark_rel")):
        f[t] = f[s] - g[s].transform("mean")
    for s, t in (("score", "score_rank"), ("b_count", "b_rank"), ("h_count", "h_rank"), ("escape", "escape_rank"), ("makuri", "makuri_rank")):
        f[t] = g[s].rank(method="min", ascending=False)

    def top_gap(col: str) -> float:
        z = np.sort(f[col].fillna(0).to_numpy())[::-1]
        return float(z[0] - z[1]) if len(z) > 1 else float(z[0] if len(z) else 0)

    f["b_top_gap"] = top_gap("b_count"); f["h_top_gap"] = top_gap("h_count"); f["score_top_gap"] = top_gap("score")
    line = f.groupby(["race_id", "line_no"], sort=False)
    f["line_score"] = line.score.transform("sum"); f["line_b"] = line.b_count.transform("sum"); f["line_h"] = line.h_count.transform("sum")
    for s, t in (("line_score", "line_score_rel"), ("line_b", "line_b_rel"), ("line_h", "line_h_rel")):
        f[t] = f[s] - g[s].transform("mean")
    f["is_leader"] = f.line_position.eq(1).astype(int); f["is_bandte"] = f.line_position.eq(2).astype(int); f["is_third"] = f.line_position.eq(3).astype(int)
    f["is_single"] = f.line_size.eq(1).astype(int)
    f["is_self_power"] = (f.line_position.eq(1) | f.b_count.gt(0) | (f.escape + f.makuri).gt(0)).astype(int)
    f["two_line"] = int(len(lines) == 2); f["three_line"] = int(len(lines) == 3); f["fragmented"] = int(len(lines) >= 4)
    leaders = f[f.is_leader.eq(1)]

    def leader_gap(col: str) -> float:
        z = np.sort(leaders[col].fillna(0).to_numpy())[::-1]
        return float(z[0] - z[1]) if len(z) > 1 else float(z[0] if len(z) else 0)

    f["leader_b_gap"] = leader_gap("b_count"); f["leader_h_gap"] = leader_gap("h_count")
    f["escape_leader_count"] = int((leaders.escape > 0).sum())
    for c in FEATURES:
        if c not in f:
            f[c] = 0.0
    return f


def _apply_component_models(f: pd.DataFrame, bundle: dict[str, Any]) -> None:
    for y, raw in (("y_back", "p_back_raw"), ("y_win", "p_win_raw"), ("y_top2", "p_top2_raw"), ("y_top3", "p_top3_raw")):
        f[raw] = bundle["component_models"][y].predict_proba(f[FEATURES].fillna(-99))[:, 1]
    cats = bundle["category_effects"]
    bonus = np.zeros(len(f))
    for k in bundle["used_prior_categories"]:
        lift = float(cats.loc[cats.category.eq(k), "y_top2_lift_pp"].iloc[0]) / 100.0
        bonus += f[f"prior_{k}"].to_numpy() * np.clip(lift, -0.08, 0.08)
    for raw in ("p_win_raw", "p_top2_raw", "p_top3_raw"):
        f[raw] = sigmoid(safe_logit(f[raw]) + bonus)
    f["p_back_candidate"] = np.where(f.is_self_power.eq(1), f.p_back_raw, 0)
    normalize_by_race(f, "p_back_candidate", "p_back", 1)
    normalize_by_race(f, "p_win_raw", "p_win", 1); normalize_by_race(f, "p_top2_raw", "p_top2", 2); normalize_by_race(f, "p_top3_raw", "p_top3", 3)


def _battle_probability(f: pd.DataFrame, bundle: dict[str, Any]) -> float:
    leaders = f[f.is_leader.eq(1)]
    row = pd.DataFrame([{
        "line_count": float(f.line_count.iloc[0]), "b_top_gap": float(f.b_top_gap.iloc[0]), "h_top_gap": float(f.h_top_gap.iloc[0]),
        "leader_b_gap": float(f.leader_b_gap.iloc[0]), "leader_h_gap": float(f.leader_h_gap.iloc[0]),
        "escape_leader_count": float(f.escape_leader_count.iloc[0]), "max_leader_b": float(leaders.b_count.max() if len(leaders) else 0),
        "max_leader_h": float(leaders.h_count.max() if len(leaders) else 0),
    }])
    return float(bundle["event_model"].predict_proba(row[EVENT_FEATURES].fillna(0))[:, 1][0])


def _market(pairs: pd.DataFrame, payload: dict[str, Any], bundle: dict[str, Any]) -> pd.DataFrame:
    bikes = [int(r["bike"]) for r in payload["riders"]]
    index = {b: i for i, b in enumerate(bikes)}
    p = pairs.copy()
    raw = bundle["isotonic"].predict(p.pair_probability)
    p["cal_raw"] = raw
    p["calibrated_probability"] = p.cal_raw / p.cal_raw.sum()
    p["exacta_odds"] = [float(payload["odds"][index[int(a)]][index[int(b)]]) for a, b in zip(p.first_car, p.second_car)]
    p["prob_band"] = pd.cut(p.calibrated_probability, PROB_BINS, right=False, include_lowest=True)
    p["odds_band"] = pd.cut(p.exacta_odds, ODDS_BINS, right=False, include_lowest=True)
    joint = bundle["joint_reliability"]
    mp = {f"{pb}|{ob}": float(v) for pb, ob, v in zip(joint.prob_band.astype(str), joint.odds_band.astype(str), joint.reliability_factor)}
    key = p.prob_band.astype(str) + "|" + p.odds_band.astype(str)
    p["odds_reliability"] = key.map(mp).fillna(0.50)
    p["purchase_probability"] = p.calibrated_probability * p.odds_reliability
    p["ev"] = p.purchase_probability * p.exacta_odds
    entropy = float(-(p.calibrated_probability * np.log(np.clip(p.calibrated_probability, 1e-12, 1))).sum() / math.log(len(p)))
    p["race_entropy"] = entropy
    return p


def predict_pr31(payload: dict[str, Any], audit: dict[str, Any], basic_path: Path) -> dict[str, Any]:
    bundle = _load_bundle()
    basic_text = _extract_text(basic_path, basic_path.name)
    venue, race_date, race_no, race_id = _identity(audit, basic_text)
    raw_day_no = detect_day_no(basic_text)
    rider_names = [str(r["name"]) for r in payload["riders"]]
    prior = fetch_previous_day(venue, race_date, raw_day_no, race_no, rider_names)
    resolved_day = int(prior.get("resolved_day_no") or 0)
    day_no = resolved_day if resolved_day >= 1 else (raw_day_no if raw_day_no >= 1 else 3)
    f = _feature_frame(payload, audit, basic_text, prior, day_no)
    _apply_component_models(f, bundle)
    battle = _battle_probability(f, bundle)
    scenarios = scenario_pair_rows(f, {race_id: battle}, bundle["venue"])
    pairs = predict_pairs(scenarios, bundle["pair_model"])
    market = _market(pairs, payload, bundle)
    rule = bundle["purchase_rule"]
    q = market[(market.ev >= float(rule["min_ev"])) & (market.purchase_probability >= float(rule["min_prob"])) & (market.race_entropy <= float(rule["confidence_max"]))].copy()
    q = q.sort_values(["ev", "calibrated_probability"], ascending=False).head(int(rule["max_points"]))
    # 運用ルール: 3点未満なら見送り。PR31の確率・EV計算自体には触れない。
    purchase_status = "BET" if len(q) >= 3 else "NO_BET"
    selected = q if purchase_status == "BET" else q.iloc[0:0]

    label_names = {
        "bandte_fight_4plus": "競り・番手飛ばされ等で4着以下",
        "blocked_4plus": "牽制・進路・詰まり等で4着以下",
        "back_4plus_otherline_win": "B取得4着以下で別線1着",
    }
    prior_riders = []
    for name, item in prior.get("riders", {}).items():
        validated = item.get("validated", {})
        hits = [label_names.get(k, k) for k, v in validated.items() if v]
        prior_riders.append({
            "name": name,
            "finish": item.get("finish"),
            "actual_start": item.get("actual_start"),
            "actual_back": item.get("actual_back"),
            "final_lap_time": item.get("final_lap_time"),
            "comment": item.get("comment", ""),
            "short_review": item.get("short_review", ""),
            "validated_patterns": hits,
        })
    rider_rows = f.sort_values("car_no")[["car_no", "name", "p_back", "p_win", "p_top2", "p_top3"]].to_dict("records")
    pair_rows = market.sort_values(["ev", "calibrated_probability"], ascending=False).head(15)[["first_car", "second_car", "calibrated_probability", "purchase_probability", "exacta_odds", "ev"]].to_dict("records")
    selections = selected[["first_car", "second_car", "calibrated_probability", "purchase_probability", "exacta_odds", "ev"]].to_dict("records")
    return {
        "version": VERSION,
        "status": "OK",
        "engine": "PR31_FROZEN_ONLY",
        "a_strategy": "REMOVED",
        "c_strategy": "REMOVED",
        "purchase_status": purchase_status,
        "purchase_rule": {**rule, "min_points": 3},
        "race": {"venue": venue, "date": race_date, "day_no": day_no, "battle_probability": battle},
        "riders": rider_rows,
        "pair_ranking": pair_rows,
        "selections": selections,
        "previous_day": {
            "status": prior.get("status"),
            "source": prior.get("source"),
            "previous_date": prior.get("previous_date"),
            "current_race_url": prior.get("current_race_url"),
            "validated_definition": [
                "競り・番手飛ばされ等で4着以下",
                "牽制・進路・詰まり等で4着以下",
                "B取得4着以下で別線1着（前日ライン確定時のみ）",
            ],
            "matched_riders": prior_riders,
            "note": "初日は判定しない。2日目以降は同一開催の直前日だけ。取得不能時は創作せず前日特徴なしでPR31を計算。",
        },
    }
