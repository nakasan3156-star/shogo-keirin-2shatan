"""A方式とC方式 Ver.1.0 Frozenを完全分離して返す統合API。"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

VERSION = "2.0.0-ac-frozen"
N_SIMULATIONS = 100_000


def _error(code: str, message: str, missing: list[str] | None = None) -> dict[str, Any]:
    return {
        "version": VERSION,
        "status": "INPUT_ERROR",
        "purchase_status": "NO_BET",
        "error": {"code": code, "message": message, "missing": missing or []},
    }


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / (hi - lo)


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(values, dtype=float) / temperature
    scaled -= scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _validate(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return _error("INVALID_PAYLOAD", "JSONルートはオブジェクトにしてください")
    if str(payload.get("race_type", "MEN")).upper() != "MEN":
        return _error("WOMEN_EXCLUDED", "女子競輪は対象外です")
    riders = payload.get("riders")
    if not isinstance(riders, list) or len(riders) < 5:
        return _error("INVALID_RIDERS", "選手データを5人以上取得できません")
    lines = payload.get("lines")
    if not isinstance(lines, list) or not lines:
        return _error("LINES_NOT_FOUND", "ライン構成を取得できません")
    bikes = sorted(int(r["bike"]) for r in riders)
    try:
        flat = [int(b) for line in lines for b in line]
    except (TypeError, ValueError):
        return _error("INVALID_LINES", "ライン構成が不正です")
    if sorted(flat) != bikes or len(flat) != len(set(flat)):
        return _error("INVALID_LINES", "全選手がライン構成に一度ずつ必要です")
    if len(lines) == len(bikes) and all(len(line) == 1 for line in lines):
        return _error("LINES_NOT_FOUND", "ライン未取得のため安全停止しました")
    odds = payload.get("odds")
    n = len(bikes)
    if not isinstance(odds, list) or len(odds) != n or any(not isinstance(row, list) or len(row) != n for row in odds):
        return _error("INVALID_ODDS", "2車単オッズ行列が不正です")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                valid = odds[i][j] is not None and float(odds[i][j]) > 0
            except (TypeError, ValueError):
                valid = False
            if not valid:
                return _error("MISSING_ODDS", "2車単オッズ42通りを取得できません", [f"odds[{i}][{j}]"])
    return None


def _a_strategy(riders: list[dict[str, Any]], lines: list[list[int]], odds: np.ndarray) -> dict[str, Any]:
    bikes = [int(r["bike"]) for r in riders]
    index = {bike: i for i, bike in enumerate(bikes)}
    ordered_odds: list[tuple[float, int, int]] = []
    for i, first in enumerate(bikes):
        for j, second in enumerate(bikes):
            if i != j:
                ordered_odds.append((float(odds[i, j]), first, second))
    ordered_odds.sort(key=lambda item: (item[0], item[1], item[2]))
    popularity = {(first, second): rank for rank, (_, first, second) in enumerate(ordered_odds, start=1)}

    candidates: list[dict[str, Any]] = []
    for line in lines:
        for left in range(len(line)):
            for right in range(left + 1, len(line)):
                first, second = int(line[left]), int(line[right])
                value = float(odds[index[first], index[second]])
                rank = popularity[(first, second)]
                if 5 <= rank <= 10 and 10.0 <= value <= 20.0:
                    candidates.append({
                        "pair": [first, second],
                        "odds": value,
                        "popularity_rank": rank,
                        "line": [int(b) for b in line],
                        "strategy": "A",
                    })
    candidates.sort(key=lambda item: (item["popularity_rank"], item["odds"], item["pair"]))
    candidates = candidates[:3]
    return {
        "name": "A方式",
        "selection_rule": "同ライン順目・人気5〜10位・10〜20倍・最大3点",
        "candidate_count": len(candidates),
        "purchase_status": "BET" if candidates else "NO_BET",
        "candidates": candidates,
    }


def _c_strategy(payload: dict[str, Any], riders: list[dict[str, Any]], lines: list[list[int]], odds: np.ndarray) -> dict[str, Any]:
    bikes = [int(r["bike"]) for r in riders]
    index = {bike: i for i, bike in enumerate(bikes)}
    n = len(riders)

    def arr(field: str) -> np.ndarray:
        return np.asarray([float(r.get(field, 0.0)) for r in riders], dtype=float)

    score = _minmax(arr("score"))
    win = _minmax(arr("win_rate"))
    quinella = _minmax(arr("quinella_rate"))
    first = arr("first")
    second = arr("second")
    third = arr("third")
    out = arr("out")
    starts = first + second + third + out
    trio_raw = np.divide(first + second + third, starts, out=np.zeros_like(starts), where=starts > 0)
    trio = _minmax(trio_raw)

    escape_raw, makuri_raw, sashi_raw, mark_raw = arr("escape"), arr("makuri"), arr("sashi"), arr("mark")
    decision_total = escape_raw + makuri_raw + sashi_raw + mark_raw
    escape_rate = np.divide(escape_raw, decision_total, out=np.zeros_like(decision_total), where=decision_total > 0)
    makuri_rate = np.divide(makuri_raw, decision_total, out=np.zeros_like(decision_total), where=decision_total > 0)
    sashi_rate = np.divide(sashi_raw, decision_total, out=np.zeros_like(decision_total), where=decision_total > 0)
    mark_rate = np.divide(mark_raw, decision_total, out=np.zeros_like(decision_total), where=decision_total > 0)
    form = np.maximum.reduce([escape_rate, makuri_rate, sashi_rate, mark_rate])

    ability = 0.40 * score + 0.20 * win + 0.15 * quinella + 0.10 * trio + 0.15 * form
    b_norm, h_norm = _minmax(arr("B")), _minmax(arr("H"))
    lead = 0.40 * b_norm + 0.30 * h_norm + 0.20 * escape_rate + 0.10 * ability
    makuri = 0.50 * makuri_rate + 0.20 * b_norm + 0.20 * ability + 0.10 * quinella
    sashi = 0.50 * sashi_rate + 0.25 * ability + 0.15 * quinella + 0.10 * trio
    mark = 0.50 * mark_rate + 0.25 * quinella + 0.15 * ability + 0.10 * trio

    line_scores = []
    for line in lines:
        head = index[int(line[0])]
        member_ability = float(np.mean([ability[index[int(b)]] for b in line]))
        support = (len(line) - 1) / 2.0
        line_scores.append(0.55 * lead[head] + 0.20 * ability[head] + 0.15 * member_ability + 0.10 * support)
    control_p = _softmax(np.asarray(line_scores), temperature=0.20)

    seed_source = {
        "grade": payload.get("grade"),
        "riders": riders,
        "lines": lines,
        "conditions": payload.get("conditions", {}),
    }
    seed = int(hashlib.sha256(json.dumps(seed_source, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    controls = rng.choice(len(lines), size=N_SIMULATIONS, p=control_p)
    final_score = np.broadcast_to(ability, (N_SIMULATIONS, n)).copy()

    top_two = np.argsort(lead)[-2:]
    collapse_probability = 0.0
    collapse_lines: set[int] = set()
    if len(top_two) == 2 and abs(float(lead[top_two[1]] - lead[top_two[0]])) <= 0.10:
        collapse_probability = min(0.50, 0.15 + 0.35 * (1.0 - abs(float(lead[top_two[1]] - lead[top_two[0]]))))
        for line_idx, line in enumerate(lines):
            if int(line[0]) in {bikes[int(top_two[0])], bikes[int(top_two[1])]}:
                collapse_lines.add(line_idx)

    for line_idx, line in enumerate(lines):
        mask = controls == line_idx
        if not np.any(mask):
            continue
        rows = np.where(mask)[0]
        head = index[int(line[0])]
        final_score[rows, head] += 0.30
        if len(line) > 1:
            second_idx = index[int(line[1])]
            final_score[rows, second_idx] += 0.18
            p_sashi = _clip(0.05 + 0.55 * sashi[second_idx] + 0.20 * ability[second_idx] - 0.25 * ability[head], 0.03, 0.65)
            success = rng.random(len(rows)) < p_sashi
            s_rows = rows[success]
            final_score[s_rows, second_idx] += 0.40
            final_score[s_rows, head] -= 0.15

        p_remain = _clip(0.15 + 0.45 * lead[head] + 0.20 * ability[head] + 0.20 * np.mean([ability[index[int(b)]] for b in line]), 0.05, 0.85)
        remain = rng.random(len(rows)) < p_remain
        final_score[rows[remain], head] += 0.10

        rivals = [int(other[0]) for k, other in enumerate(lines) if k != line_idx]
        if rivals:
            rival_bike = max(rivals, key=lambda b: float(makuri[index[b]]))
            rival = index[rival_bike]
            p_makuri = _clip(0.08 + 0.50 * makuri[rival] + 0.20 * ability[rival] - 0.30 * lead[head], 0.03, 0.75)
            success = rng.random(len(rows)) < p_makuri
            m_rows = rows[success]
            final_score[m_rows, rival] += 0.45
            rival_line = next(other for other in lines if int(other[0]) == rival_bike)
            if len(rival_line) > 1:
                follower = index[int(rival_line[1])]
                follow_success = rng.random(len(m_rows)) >= _clip(0.35 - 0.25 * mark[follower] - 0.15 * ability[follower], 0.03, 0.35)
                final_score[m_rows[follow_success], follower] += 0.22
                final_score[m_rows[~follow_success], follower] -= 0.40

        if line_idx in collapse_lines and collapse_probability > 0:
            collapsed = rng.random(len(rows)) < collapse_probability
            c_rows = rows[collapsed]
            for bike in line:
                final_score[c_rows, index[int(bike)]] -= 0.30
            singles = [int(other[0]) for other in lines if len(other) == 1]
            for bike in singles:
                final_score[c_rows, index[bike]] += 0.15

        for bike in line[1:]:
            rider_idx = index[int(bike)]
            p_fail = _clip(0.35 - 0.25 * mark[rider_idx] - 0.15 * ability[rider_idx], 0.03, 0.35)
            failed = rng.random(len(rows)) < p_fail
            final_score[rows[failed], rider_idx] -= 0.40

    final_score += rng.normal(0.0, 0.22, size=final_score.shape)
    order = np.argsort(final_score, axis=1)[:, ::-1]
    first_idx, second_idx = order[:, 0], order[:, 1]
    counts = np.zeros((n, n), dtype=np.int64)
    np.add.at(counts, (first_idx, second_idx), 1)
    probability = counts / float(N_SIMULATIONS)

    all_pairs: list[dict[str, Any]] = []
    for i, first_bike in enumerate(bikes):
        for j, second_bike in enumerate(bikes):
            if i == j:
                continue
            p = float(probability[i, j])
            odd = float(odds[i, j])
            all_pairs.append({
                "pair": [first_bike, second_bike],
                "probability": p,
                "odds": odd,
                "ev": p * odd,
                "strategy": "C",
            })
    all_pairs.sort(key=lambda item: (-item["probability"], item["pair"]))
    candidates = [{**item, "rank": rank} for rank, item in enumerate(all_pairs[:6], start=1)]

    rider_scores = []
    for i, rider in enumerate(riders):
        rider_scores.append({
            "bike": bikes[i],
            "name": rider.get("name"),
            "ability": float(ability[i]),
            "lead_index": float(lead[i]),
            "makuri_index": float(makuri[i]),
            "sashi_index": float(sashi[i]),
            "mark_index": float(mark[i]),
        })
    return {
        "name": "C方式 Ver.1.0 Frozen",
        "selection_rule": "個人能力→展開分岐→固定シード10万回MC→予測確率上位6点",
        "candidate_count": 6,
        "purchase_status": "SIX_PICKS",
        "simulations": N_SIMULATIONS,
        "seed": seed,
        "control_probabilities": [
            {"line": [int(b) for b in line], "probability": float(control_p[k])}
            for k, line in enumerate(lines)
        ],
        "rider_scores": rider_scores,
        "candidates": candidates,
        "all_pair_probabilities": all_pairs,
    }


def predict(payload: Any) -> dict[str, Any]:
    validation = _validate(payload)
    if validation:
        return validation
    riders = sorted(payload["riders"], key=lambda r: int(r["bike"]))
    lines = [[int(b) for b in line] for line in payload["lines"]]
    odds = np.asarray(payload["odds"], dtype=float)
    a_result = _a_strategy(riders, lines, odds)
    c_result = _c_strategy(payload, riders, lines, odds)
    a_pairs = {tuple(item["pair"]) for item in a_result["candidates"]}
    c_pairs = {tuple(item["pair"]) for item in c_result["candidates"]}
    return {
        "version": VERSION,
        "status": "OK",
        "race_type": "MEN",
        "purchase_status": "A_AND_C_READY",
        "strategies": {"a": a_result, "c": c_result},
        "candidates": c_result["candidates"],
        "common_candidates": [list(pair) for pair in sorted(a_pairs & c_pairs)],
        "audit": {
            "engines_separated": True,
            "residual_b_removed": True,
            "result_data_used": False,
            "odds_used_in_a_filter": True,
            "odds_used_in_c_probability": False,
            "c_simulations": N_SIMULATIONS,
        },
    }
