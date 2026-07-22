from __future__ import annotations

import math
from typing import Any

import numpy as np

from .parser import RaceInput, Rider


MODEL_VERSION = "1.1.0-two-pdf-optional-ex"
SCENARIO_NAMES = ("leader_hold", "bante_sashi", "makuri_line", "line_break_cross", "solo_or_detached")


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - float(values.mean())) / std


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exp = np.exp(np.clip(shifted, -30.0, 30.0))
    return exp / float(exp.sum())


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 0.02, 0.98)
    return np.log(clipped / (1.0 - clipped))


def _smoothed_rates(riders: list[Rider]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    win, top2, top3, tactic = [], [], [], []
    for rider in riders:
        total = max(0, rider.races)
        win.append((rider.first_count + 1.0) / (total + 7.0))
        top2.append((rider.first_count + rider.second_count + 2.0) / (total + 7.0))
        top3.append((rider.first_count + rider.second_count + rider.third_count + 3.0) / (total + 7.0))
        relevant = rider.escape_count + rider.makuri_count if rider.style in {"逃", "両"} else rider.sashi_count + rider.mark_count
        tactic.append((relevant + 1.0) / (total + 4.0))
    return tuple(np.asarray(values, dtype=float) for values in (win, top2, top3, tactic))


def _posterior_rate(item: dict[str, Any] | None, default: float = 0.5) -> float:
    if not item or not item.get("total"):
        return default
    return (float(item["success"]) + 2.0) / (float(item["total"]) + 4.0)


def _lost_scores(riders: list[Rider], history: dict[str, Any] | None) -> np.ndarray:
    if not history or history.get("status") == "failed":
        return np.zeros(len(riders), dtype=float)
    by_number = history.get("riders", {})
    return np.asarray(
        [float(by_number.get(rider.number, {}).get("lost_strong_proxy", {}).get("score", 0.0)) for rider in riders],
        dtype=float,
    )


def _recent_scores(riders: list[Rider], history: dict[str, Any] | None) -> np.ndarray:
    if not history or history.get("status") == "failed":
        return np.zeros(len(riders), dtype=float)
    by_number = history.get("riders", {})
    return np.asarray(
        [float(by_number.get(rider.number, {}).get("recent_form_score") or 0.0) for rider in riders],
        dtype=float,
    )


def _base_strengths(
    race: RaceInput,
    ex_data: dict[int, dict[str, Any]],
    history: dict[str, Any] | None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    riders = list(race.riders)
    points = np.asarray([rider.race_score for rider in riders], dtype=float)
    win, top2, top3, tactic = _smoothed_rates(riders)
    components: dict[str, np.ndarray] = {
        "race_score": 0.52 * _zscore(points),
        "win_form": 0.17 * _zscore(_logit(win)),
        "top2_form": 0.13 * _zscore(_logit(top2)),
        "top3_form": 0.04 * _zscore(_logit(top3)),
        "tactical_result": 0.06 * _zscore(_logit(tactic)),
    }
    lost = _lost_scores(riders, history)
    recent = _recent_scores(riders, history)
    if np.any(recent):
        components["uploaded_recent_form"] = 0.12 * _zscore(recent)
    if np.any(lost):
        # Small coefficient: this is a structured proxy, not verified video interference.
        components["lost_strong_proxy"] = 0.08 * _zscore(lost)

    ex_values = []
    for rider in riders:
        item = ex_data.get(rider.number, {})
        if rider.style in {"逃", "両"}:
            value = 0.5 * _posterior_rate(item.get("kamashi")) + 0.5 * _posterior_rate(item.get("tsuppari"))
        else:
            value = 1.0 - _posterior_rate(item.get("chigirareru"), default=0.15)
        ex_values.append(value)
    if ex_data:
        components["ex_tactics"] = 0.05 * _zscore(np.asarray(ex_values, dtype=float))

    base = np.sum(np.vstack(list(components.values())), axis=0)
    details = [
        {name: round(float(values[index]), 6) for name, values in components.items()}
        for index in range(len(riders))
    ]
    return base, details


def _line_indices(race: RaceInput) -> list[np.ndarray]:
    index_by_number = {rider.number: index for index, rider in enumerate(race.riders)}
    lines = [np.asarray([index_by_number[number] for number in line], dtype=int) for line in race.lines]
    covered = sorted(index for line in lines for index in line.tolist())
    if covered != list(range(len(race.riders))):
        raise ValueError("ラインが全選手を一度ずつ含んでいません。")
    return lines


def _line_parameters(
    race: RaceInput,
    base: np.ndarray,
    ex_data: dict[int, dict[str, Any]],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[np.ndarray]]:
    riders = list(race.riders)
    lines = _line_indices(race)
    line_strengths: list[float] = []
    same_line_probabilities: list[float] = []
    first_probabilities: list[np.ndarray] = []

    for indices in lines:
        leader = riders[int(indices[0])]
        followers = indices[1:]
        leader_activity = (leader.back_count + 1.0) / (leader.races + 4.0)
        support = float(np.mean(base[followers])) if len(followers) else 0.0
        line_strengths.append(float(base[indices[0]] + 0.28 * support + 0.10 * math.log1p(len(followers)) + 0.08 * leader_activity))

        role_utility = base[indices].copy()
        role_utility[0] += 0.10 + 0.10 * leader_activity
        for position, rider_index in enumerate(indices[1:], start=1):
            rider = riders[int(rider_index)]
            follow_rate = (rider.sashi_count + rider.mark_count + 1.0) / (rider.races + 4.0)
            role_utility[position] += 0.08 * follow_rate - 0.025 * max(0, position - 1)
        first_probabilities.append(_softmax(role_utility / 0.72))

        if len(indices) == 1:
            same_line_probabilities.append(0.0)
            continue
        detached_values = [
            _posterior_rate(ex_data.get(riders[int(index)].number, {}).get("chigirareru"), default=0.15)
            for index in indices[1:]
        ]
        leader_chigiri = _posterior_rate(ex_data.get(leader.number, {}).get("chigiri"), default=0.10)
        same = (
            0.43
            + 0.07 * (len(indices) - 2)
            + 0.06 * math.tanh(support)
            - 0.12 * leader_chigiri
            - 0.18 * float(np.mean(detached_values))
        )
        same_line_probabilities.append(float(np.clip(same, 0.18, 0.75)))

    return lines, _softmax(np.asarray(line_strengths, dtype=float) / 0.68), np.asarray(same_line_probabilities), first_probabilities


def _scenario_name(line: np.ndarray, first_index: int, second_index: int, same_line: bool) -> str:
    if len(line) == 1:
        return "solo_or_detached"
    if not same_line:
        return "line_break_cross"
    if first_index == int(line[0]):
        return "leader_hold"
    if second_index == int(line[0]):
        return "bante_sashi"
    return "makuri_line"


def simulate_pairs(
    race: RaceInput,
    ex_data: dict[int, dict[str, Any]],
    history: dict[str, Any] | None,
    runs: int,
    seed: int,
) -> tuple[dict[tuple[int, int], int], np.ndarray, list[dict[str, float]], dict[str, int], dict[str, Any]]:
    if runs < 10_000 or runs > 2_000_000:
        raise ValueError("monte_carlo_runs は10,000〜2,000,000回です。")
    riders = list(race.riders)
    base, component_details = _base_strengths(race, ex_data, history)
    lines, line_probabilities, same_probabilities, first_probabilities = _line_parameters(race, base, ex_data)
    rng = np.random.default_rng(seed)
    counts = np.zeros((len(riders), len(riders)), dtype=np.int64)
    scenario_counts = {name: 0 for name in SCENARIO_NAMES}
    all_indices = np.arange(len(riders), dtype=int)
    chunk_size = 100_000

    for start in range(0, runs, chunk_size):
        size = min(chunk_size, runs - start)
        winning_lines = rng.choice(len(lines), size=size, p=line_probabilities)
        for line_index, line in enumerate(lines):
            positions = np.flatnonzero(winning_lines == line_index)
            if not len(positions):
                continue
            chosen_first = rng.choice(line, size=len(positions), p=first_probabilities[line_index])
            choose_same = np.zeros(len(positions), dtype=bool)
            if len(line) > 1:
                choose_same = rng.random(len(positions)) < same_probabilities[line_index]

            chosen_second = np.empty(len(positions), dtype=int)
            for first_index in line:
                first_mask = chosen_first == first_index
                if not np.any(first_mask):
                    continue
                same_mask = first_mask & choose_same
                if np.any(same_mask):
                    candidates = line[line != first_index]
                    probabilities = _softmax(base[candidates] / 0.78)
                    chosen_second[same_mask] = rng.choice(candidates, size=int(np.sum(same_mask)), p=probabilities)
                cross_mask = first_mask & ~choose_same
                if np.any(cross_mask):
                    candidates = all_indices[~np.isin(all_indices, line)]
                    if not len(candidates):
                        candidates = all_indices[all_indices != first_index]
                    probabilities = _softmax(base[candidates] / 0.82)
                    chosen_second[cross_mask] = rng.choice(candidates, size=int(np.sum(cross_mask)), p=probabilities)

            np.add.at(counts, (chosen_first, chosen_second), 1)
            for first_index, second_index, same in zip(chosen_first, chosen_second, choose_same, strict=True):
                scenario_counts[_scenario_name(line, int(first_index), int(second_index), bool(same))] += 1

    pair_counts = {
        (first.number, second.number): int(counts[first_index, second_index])
        for first_index, first in enumerate(riders)
        for second_index, second in enumerate(riders)
        if first_index != second_index
    }
    parameters = {
        "line_win_probabilities": [
            {"line": [riders[int(index)].number for index in line], "probability": round(float(line_probabilities[i]), 8)}
            for i, line in enumerate(lines)
        ],
        "same_line_top2_probabilities": [
            {"line": [riders[int(index)].number for index in line], "probability": round(float(same_probabilities[i]), 8)}
            for i, line in enumerate(lines)
        ],
    }
    return pair_counts, base, component_details, scenario_counts, parameters


def predict(
    race: RaceInput,
    odds: dict[tuple[int, int], float],
    ex_data: dict[int, dict[str, Any]] | None = None,
    ex_warnings: list[str] | None = None,
    runs: int = 100_000,
    seed: int = 3156,
    history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ex_data = ex_data or {}
    ex_warnings = ex_warnings or []
    pair_counts, base, component_details, scenario_counts, scenario_parameters = simulate_pairs(
        race, ex_data, history, runs, seed
    )
    expected_pairs = {(a.number, b.number) for a in race.riders for b in race.riders if a.number != b.number}
    if set(odds) != expected_pairs:
        raise ValueError("出走選手と2車単オッズの組み合わせが一致しません。")
    inverse_sum = sum(1.0 / price for price in odds.values())
    combinations: list[dict[str, Any]] = []
    for pair, count in pair_counts.items():
        probability = count / runs
        price = odds[pair]
        simulation_se = math.sqrt(max(probability * (1.0 - probability), 0.0) / runs)
        market_probability = (1.0 / price) / inverse_sum
        combinations.append(
            {
                "first": pair[0],
                "second": pair[1],
                "estimated_probability": round(probability, 8),
                "simulation_se": round(simulation_se, 8),
                "fair_odds": round(1.0 / probability, 3) if probability > 0 else None,
                "odds": price,
                "ev": round(probability * price, 4),
                "market_probability_normalized": round(market_probability, 8),
                "probability_edge": round(probability - market_probability, 8),
                "research_candidate": probability * price >= 1.0,
            }
        )
    combinations.sort(key=lambda item: (-item["ev"], -item["estimated_probability"], item["first"], item["second"]))

    history_riders = history.get("riders", {}) if history else {}
    riders_output = []
    for index, rider in enumerate(race.riders):
        first_probability = sum(item["estimated_probability"] for item in combinations if item["first"] == rider.number)
        second_probability = sum(item["estimated_probability"] for item in combinations if item["second"] == rider.number)
        riders_output.append(
            {
                **rider.to_dict(),
                "line": next(list(line) for line in race.lines if rider.number in line),
                "model_components": component_details[index],
                "latent_strength": round(float(base[index]), 6),
                "estimated_first_probability": round(first_probability, 8),
                "estimated_second_probability": round(second_probability, 8),
                "lost_strong_proxy": history_riders.get(rider.number, {}).get("lost_strong_proxy"),
            }
        )

    all_warnings = list(race.line_warnings) + list(ex_warnings)
    if history and history.get("status") != "complete":
        all_warnings.append(f"互換用の直近成績取得状態は{history.get('status')}です。")
    all_warnings.append("研究版です。時系列アウト・オブ・サンプル検証前は実投資判定に使用しません。")

    scenario_distribution = {
        name: {"count": count, "probability": round(count / runs, 8)}
        for name, count in scenario_counts.items()
    }
    return {
        "schema_version": "3.0",
        "model_version": MODEL_VERSION,
        "model_status": "research_unvalidated",
        "race": {
            "race_id": race.race_id,
            "source_race_id": history.get("target_race_id") if history else None,
            "venue": race.venue,
            "race_number": race.race_number,
            "race_class": race.race_class,
            "distance_m": race.distance_m,
            "entrant_count": len(race.riders),
            "lines": [list(line) for line in race.lines],
        },
        "calculation": {
            "bet_type": "2車単",
            "monte_carlo_runs": runs,
            "seed": seed,
            "odds_used_for_probability_estimation": False,
            "probability_sum_all_ordered_pairs": round(sum(item["estimated_probability"] for item in combinations), 8),
            "market_inverse_odds_sum": round(inverse_sum, 6),
            "ex_data_used": bool(ex_data),
            "history_used": bool(history and history.get("status") in {"complete", "partial"}),
            "scenario_model": "two_stage_line_then_order_v1.0_pdf",
        },
        "scenario_distribution": scenario_distribution,
        "scenario_parameters": scenario_parameters,
        "history": history,
        "investment_gate": {
            "live_betting_approved": False,
            "reason": "確率校正と過去レースの時系列分離検証が未完了",
            "raw_ev_threshold_for_research": 1.0,
        },
        "riders": riders_output,
        "all_combinations": combinations,
        "top10_by_ev": combinations[:10],
        "warnings": all_warnings,
        "ex_data": ex_data,
    }
