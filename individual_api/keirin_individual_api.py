"""章悟式∞競輪OS 個人評価型 2車単API（暫定固定版）。

入力PDF/OCRを正規化した辞書を受け取り、結果データを一切参照せず、
個人能力→役割適性→ライン補正→10万回シミュレーション→EVの順で処理する。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


VERSION = "1.1.4-pdf-resilient"
N_SIMULATIONS = 100_000
REQUIRED_SOURCE_KEYS = {"racecard_pdf", "hs_pdf", "odds_pdf"}
REQUIRED_RIDER_FIELDS = {
    "bike", "name", "region", "score", "win_rate", "escape", "makuri",
    "sashi", "mark", "H", "B",
}


@dataclass(frozen=True)
class FixedConfig:
    min_control_confidence: float = 0.38
    min_scenario_confidence: float = 0.35
    min_odds: float = 8.0
    max_odds: float = 30.0
    min_probability: float = 0.01
    min_conservative_ev: float = 1.10
    max_candidates: int = 2
    confidence_z: float = 1.645
    effective_sample_size: int = 1_000
    purchase_grades: tuple[str, ...] = ("F1", "G3")


CONFIG = FixedConfig()


def _z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = float(values.std())
    return np.zeros_like(values) if sd < 1e-12 else (values - values.mean()) / sd


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ex = np.exp(values - values.max())
    return ex / ex.sum()


def _wilson_lower(p: np.ndarray) -> np.ndarray:
    n, zc = float(CONFIG.effective_sample_size), CONFIG.confidence_z
    denominator = 1.0 + zc**2 / n
    return (
        p + zc**2/(2*n)
        - zc*np.sqrt(p*(1-p)/n + zc**2/(4*n*n))
    ) / denominator


def _error(code: str, message: str, missing: list[str] | None = None) -> dict[str, Any]:
    return {
        "version": VERSION,
        "status": "INPUT_ERROR",
        "purchase_status": "NO_BET",
        "error": {"code": code, "message": message, "missing": missing or []},
    }


def _validate(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return _error("INVALID_PAYLOAD", "JSONルートはオブジェクトにしてください")
    source_files = payload.get("source_files")
    if not isinstance(source_files, dict):
        return _error("INVALID_SOURCE_FILES", "source_filesはオブジェクトにしてください")
    missing_sources = sorted(
        key for key in REQUIRED_SOURCE_KEYS
        if not isinstance(source_files.get(key), str) or not source_files[key].strip()
    )
    if missing_sources:
        return _error("MISSING_SOURCE", "固定入力3点が揃っていません", missing_sources)
    if payload.get("grade") not in {"F1", "G3", "G1"}:
        return _error("INVALID_GRADE", "gradeはF1/G3/G1のいずれかです")
    riders = payload.get("riders")
    if not isinstance(riders, list) or len(riders) < 5:
        return _error("INVALID_RIDERS", "ridersは5人以上必要です")
    missing = []
    for i, rider in enumerate(riders):
        if not isinstance(rider, dict):
            return _error("INVALID_RIDER", f"riders[{i}]はオブジェクトにしてください")
        for field in sorted(REQUIRED_RIDER_FIELDS - set(rider)):
            missing.append(f"riders[{i}].{field}")
    if missing:
        return _error("MISSING_RIDER_DATA", "読取不能項目は推測せず停止します", missing)
    bikes = []
    numeric_fields = REQUIRED_RIDER_FIELDS - {"name", "region"}
    for i, rider in enumerate(riders):
        if not isinstance(rider["name"], str) or not rider["name"].strip():
            return _error("INVALID_RIDER_NAME", f"riders[{i}].nameが不正です")
        if not isinstance(rider["region"], str) or not rider["region"].strip():
            return _error("INVALID_RIDER_REGION", f"riders[{i}].regionが不正です")
        for field in numeric_fields:
            try:
                value = float(rider[field])
            except (TypeError, ValueError):
                return _error("INVALID_RIDER_VALUE", f"riders[{i}].{field}が数値ではありません")
            if not np.isfinite(value):
                return _error("INVALID_RIDER_VALUE", f"riders[{i}].{field}が有限値ではありません")
        try:
            bike_value = float(rider["bike"])
            bike = int(bike_value)
        except (TypeError, ValueError, OverflowError):
            return _error("INVALID_BIKE", f"riders[{i}].bikeが不正です")
        if bike_value != bike or bike <= 0:
            return _error("INVALID_BIKE", f"riders[{i}].bikeは正の整数にしてください")
        bikes.append(bike)
    if len(set(bikes)) != len(bikes):
        return _error("DUPLICATE_BIKE", "車番が重複しています")
    if bikes != sorted(bikes):
        return _error("INVALID_RIDER_ORDER", "ridersとoddsの行・列は車番昇順にしてください")
    lines = payload.get("lines")
    if not isinstance(lines, list) or not lines or any(not isinstance(line, list) or not line for line in lines):
        return _error("INVALID_LINES", "linesは空でないライン配列にしてください")
    try:
        flat = [int(b) for line in lines for b in line]
    except (TypeError, ValueError, OverflowError):
        return _error("INVALID_LINES", "linesの車番が不正です")
    if sorted(flat) != sorted(bikes):
        return _error("INVALID_LINES", "全選手が並びに一度ずつ必要です")
    odds = payload.get("odds")
    if (
        not isinstance(odds, list)
        or len(odds) != len(bikes)
        or any(not isinstance(row, list) or len(row) != len(bikes) for row in odds)
    ):
        return _error("INVALID_ODDS", "2車単オッズ行列の寸法が不正です")
    for i, row in enumerate(odds):
        for j, value in enumerate(row):
            if i == j:
                continue
            try:
                valid = value is not None and np.isfinite(float(value)) and float(value) > 0
            except (TypeError, ValueError):
                valid = False
            if not valid:
                return _error("MISSING_ODDS", "読取不能な2車単オッズは推測せず停止します", [f"odds[{i}][{j}]"])
    conditions = payload.get("conditions", {})
    if not isinstance(conditions, dict):
        return _error("INVALID_CONDITIONS", "conditionsはオブジェクトにしてください")
    if conditions.get("bank_type") is not None and not isinstance(conditions["bank_type"], str):
        return _error("INVALID_CONDITIONS", "conditions.bank_typeが不正です")
    for field in ("wind_mps", "temperature_c"):
        if conditions.get(field) is not None:
            try:
                valid = np.isfinite(float(conditions[field]))
            except (TypeError, ValueError):
                valid = False
            if not valid:
                return _error("INVALID_CONDITIONS", f"conditions.{field}が不正です")
    return None


def _predict_strict(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic prediction for normalized three-file input."""
    validation_error = _validate(payload)
    if validation_error:
        return validation_error

    riders = sorted(payload["riders"], key=lambda r: int(r["bike"]))
    bikes = [int(r["bike"]) for r in riders]
    n = len(riders)
    idx = {bike: i for i, bike in enumerate(bikes)}
    lines = {int(line[0]): [int(b) for b in line] for line in payload["lines"]}

    def arr(field: str) -> np.ndarray:
        return np.asarray([float(r[field]) for r in riders])

    score, wr = arr("score"), arr("win_rate")
    esc, mak, dif, mark = arr("escape"), arr("makuri"), arr("sashi"), arr("mark")
    h_count, b_count = arr("H"), arr("B")

    ability = (0.48*_z(score) + 0.24*_z(wr) + 0.08*_z(esc) + 0.08*_z(mak)
               + 0.07*_z(dif) + 0.05*_z(mark))
    block = 0.45*_z(mark) + 0.30*_z(dif) + 0.15*_z(score) + 0.10*_z(wr)
    leader = 0.38*ability + 0.22*_z(h_count) + 0.18*_z(b_count) + 0.12*_z(esc) + 0.10*_z(mak)
    follower = 0.40*ability + 0.24*_z(dif) + 0.20*_z(mark) + 0.16*block
    singleton = 0.48*ability + 0.32*_z(mak) + 0.12*_z(dif) + 0.08*_z(h_count)
    tail = 0.52*ability + 0.28*_z(mark) + 0.20*_z(dif)

    role_base = ability.copy()
    roles: dict[int, str] = {}
    for root, members in lines.items():
        if len(members) == 1:
            role_base[idx[root]], roles[root] = singleton[idx[root]], "単騎"
        else:
            role_base[idx[root]], roles[root] = leader[idx[root]], "先頭"
            role_base[idx[members[1]]], roles[members[1]] = follower[idx[members[1]]], "番手"
            for bike in members[2:]:
                role_base[idx[bike]], roles[bike] = tail[idx[bike]], "3番手以降"

    roots = list(lines)
    wind = payload.get("conditions", {}).get("wind_mps")
    wind = None if wind is None else float(wind)
    bank_type = payload.get("conditions", {}).get("bank_type", "未取得")
    control_scores, line_scores = [], []
    for root in roots:
        members, ri = lines[root], idx[root]
        member_regions = {riders[idx[b]]["region"] for b in members}
        same_region = float(
            len(members) > 1
            and "未取得" not in member_regions
            and len(member_regions) == 1
        )
        bank = 0.0
        if bank_type == "335_indoor":
            bank = 0.10
        elif wind is not None:
            bank = -0.05 * max(0.0, wind - 2.0)
        control = 0.68*leader[ri] + 0.12*(len(members)-1) + 0.08*same_region + bank
        support = np.mean([role_base[idx[b]] for b in members[1:]]) if len(members) > 1 else -0.3
        control_scores.append(control)
        line_scores.append(control + 0.35*support + 0.12*(len(members)-1))
    control_p = _softmax(np.asarray(control_scores)/0.90)
    predicted_control = roots[int(np.argmax(control_p))]
    predicted_main_line = roots[int(np.argmax(line_scores))]
    predicted_main_line_bikes = list(lines[predicted_main_line])

    scenario_names = ["先行押し切り", "番手差し", "別線捲り", "ライン崩壊"]
    scenario_p = np.zeros((len(roots), 4))
    rival_for = []
    for k, root in enumerate(roots):
        members, ri = lines[root], idx[root]
        fi = idx[members[1]] if len(members) > 1 else None
        rivals = [r for r in roots if r != root]
        rival = max(rivals, key=lambda r: singleton[idx[r]]) if rivals else root
        rival_for.append(rival)
        cohesion = 0.20*(len(members)-1)
        values = np.array([
            0.82*leader[ri] + cohesion,
            -1.4 if fi is None else 0.82*follower[fi] + cohesion,
            0.90*singleton[idx[rival]] - 0.18*control_scores[k],
            -0.15*cohesion + 0.20*np.std(role_base) + (0.10 if len(members) == 1 else 0.0),
        ])
        if bank_type == "335_indoor":
            values[2] += 0.08
        if bank_type != "335_indoor" and wind is not None and wind >= 3:
            values += np.array([-0.12, 0.08, 0.0, 0.05])
        scenario_p[k] = _softmax(values/0.85)

    seed_payload = {k: payload[k] for k in ["grade", "riders", "lines"]}
    seed_payload["conditions"] = payload.get("conditions", {})
    seed = int(hashlib.sha256(json.dumps(seed_payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    controls = rng.choice(len(roots), N_SIMULATIONS, p=control_p)
    scenarios = np.empty(N_SIMULATIONS, dtype=np.int8)
    for k in range(len(roots)):
        mask = controls == k
        scenarios[mask] = rng.choice(4, int(mask.sum()), p=scenario_p[k])

    win_util = np.broadcast_to(0.78*role_base, (N_SIMULATIONS, n)).copy()
    for k, root in enumerate(roots):
        members, ri, rri = lines[root], idx[root], idx[rival_for[k]]
        fi = idx[members[1]] if len(members) > 1 else None
        for scenario in range(4):
            mask = (controls == k) & (scenarios == scenario)
            if scenario == 0:
                win_util[mask, ri] += 1.75
                if fi is not None: win_util[mask, fi] += 0.35
            elif scenario == 1:
                if fi is not None: win_util[mask, fi] += 1.80
                win_util[mask, ri] += 0.45
            elif scenario == 2:
                win_util[mask, rri] += 1.75 + 0.18*_z(mak)[rri]
            else:
                for bike in members: win_util[mask, idx[bike]] -= 0.25
                win_util[mask] += 0.16*_z(mak+dif)
    first = np.argmax(win_util - np.log(-np.log(rng.random((N_SIMULATIONS, n)))), axis=1)

    second_util = np.broadcast_to(0.72*role_base + 0.12*_z(mark+dif), (N_SIMULATIONS, n)).copy()
    for k, root in enumerate(roots):
        members, mask = lines[root], controls == k
        if len(members) > 1:
            ri, fi = idx[root], idx[members[1]]
            second_util[mask & (first == ri), fi] += 1.35 + 0.15*block[fi]
            second_util[mask & (first == fi), ri] += 0.95
        for bike in members[2:]: second_util[mask, idx[bike]] += 0.42
        rival = rival_for[k]
        if len(lines[rival]) > 1:
            rri, rfi = idx[rival], idx[lines[rival][1]]
            second_util[mask & (first == rri), rfi] += 1.15 + 0.12*block[rfi]
    second_util[np.arange(N_SIMULATIONS), first] = -1e9
    second = np.argmax(second_util - np.log(-np.log(rng.random((N_SIMULATIONS, n)))), axis=1)

    counts = np.zeros((n, n), dtype=np.int64)
    np.add.at(counts, (first, second), 1)
    pair_p = counts / N_SIMULATIONS
    scenario_counts = np.bincount(scenarios, minlength=4) / N_SIMULATIONS
    predicted_scenario = scenario_names[int(np.argmax(scenario_counts))]

    odds = np.asarray([[np.nan if v is None else float(v) for v in row] for row in payload["odds"]])
    p_low, ev = _wilson_lower(pair_p), pair_p*odds
    ev_low = p_low*odds
    gate = float(control_p.max()) >= CONFIG.min_control_confidence and float(scenario_counts.max()) >= CONFIG.min_scenario_confidence
    eligible = (np.isfinite(ev_low) & (ev_low >= CONFIG.min_conservative_ev)
                & (pair_p >= CONFIG.min_probability) & (odds >= CONFIG.min_odds)
                & (odds <= CONFIG.max_odds) & gate)
    np.fill_diagonal(eligible, False)
    order = np.argsort(np.where(eligible, ev_low, -np.inf), axis=None)[::-1]
    candidates = []
    for flat in order[:min(CONFIG.max_candidates, int(eligible.sum()))]:
        i, j = np.unravel_index(flat, pair_p.shape)
        candidates.append({"pair": [bikes[i], bikes[j]], "probability": float(pair_p[i,j]),
                           "odds": float(odds[i,j]), "ev": float(ev[i,j]),
                           "conservative_ev": float(ev_low[i,j])})

    individual = []
    for i, rider in enumerate(riders):
        individual.append({"bike": bikes[i], "name": rider["name"], "role": roles[bikes[i]],
                           "ability": float(ability[i]), "leader": float(leader[i]),
                           "follower": float(follower[i]), "singleton": float(singleton[i]),
                           "tail": float(tail[i]), "assigned_role_score": float(role_base[i]),
                           "first_rate": float(np.mean(first == i)), "second_rate": float(np.mean(second == i))})
    pairs = [{"pair": [bikes[i], bikes[j]], "probability": float(pair_p[i,j])}
             for i in range(n) for j in range(n) if i != j]
    pairs.sort(key=lambda item: item["probability"], reverse=True)

    grade_enabled = payload["grade"] in CONFIG.purchase_grades
    return {
        "version": VERSION, "status": "OK", "seed": seed, "simulations": N_SIMULATIONS,
        "purchase_status": "CANDIDATES" if grade_enabled and candidates else "NO_BET",
        "grade_enabled": grade_enabled, "input_complete": True,
        "predicted_control": predicted_control, "control_confidence": float(control_p.max()),
        "predicted_main_line": predicted_main_line,
        "predicted_main_line_bikes": predicted_main_line_bikes,
        "predicted_scenario": predicted_scenario,
        "scenario_probabilities": {scenario_names[i]: float(scenario_counts[i]) for i in range(4)},
        "individual_scores": individual, "pair_probabilities": pairs,
        "candidates": candidates if grade_enabled else [],
        "audit_candidates": candidates, "config": asdict(CONFIG),
        "missing_optional": (
            ([] if payload["source_files"].get("ex_image") else ["ex_image"])
            + [k for k in ["wind_mps", "temperature_c", "bank_type"]
               if payload.get("conditions", {}).get(k) is None]
        ),
    }


def predict(payload: Any) -> dict[str, Any]:
    """例外を外へ出さず、正常結果またはNO_BETエラーを必ず返す。"""
    try:
        return _predict_strict(payload)
    except Exception:
        return {
            "version": VERSION,
            "status": "PROCESSING_ERROR",
            "purchase_status": "NO_BET",
            "error": {
                "code": "UNEXPECTED_PROCESSING_ERROR",
                "message": "予測処理を安全停止しました。入力を確認してください",
                "missing": [],
            },
        }


def predict_json(text: str) -> str:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        payload = None
        result = _error("INVALID_JSON", "JSONを読み取れません")
    else:
        result = predict(payload)
    return json.dumps(result, ensure_ascii=False, sort_keys=True)
