"""章悟式5点と市場残差3点を同じ能力予測から分岐して返すAPI。"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

try:
    from .keirin_individual_api import N_SIMULATIONS, predict as predict_ability
except ImportError:  # 直接スクリプトとして実行する場合
    from keirin_individual_api import N_SIMULATIONS, predict as predict_ability


VERSION = "1.2.0-dual-strategy"
RESIDUAL_SIMULATIONS = 100_000
DEFAULT_LAMBDA = 0.50
CONFIDENCE_Z = 1.645
EPS = 1e-12


def _wilson_lower(p: np.ndarray, n: int) -> np.ndarray:
    n_float = float(n)
    denominator = 1.0 + CONFIDENCE_Z**2 / n_float
    return (
        p
        + CONFIDENCE_Z**2 / (2 * n_float)
        - CONFIDENCE_Z
        * np.sqrt(
            p * (1.0 - p) / n_float
            + CONFIDENCE_Z**2 / (4 * n_float**2)
        )
    ) / denominator


def _stable_seed(payload: dict[str, Any], namespace: str) -> int:
    encoded = json.dumps(
        {"namespace": namespace, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int(hashlib.sha256(encoded).hexdigest()[:16], 16) % (2**32)


def _market_probability(odds: np.ndarray) -> np.ndarray:
    inverse = np.zeros_like(odds, dtype=float)
    valid = np.isfinite(odds) & (odds > 0)
    inverse[valid] = 1.0 / odds[valid]
    np.fill_diagonal(inverse, 0.0)
    total = float(inverse.sum())
    if total <= 0:
        raise ValueError("有効な2車単オッズがありません")
    return inverse / total


def _residual_probability(
    market: np.ndarray,
    ability: np.ndarray,
    lambda_value: float,
) -> np.ndarray:
    # market * (ability / market) ** lambda
    log_probability = (
        (1.0 - lambda_value) * np.log(np.clip(market, EPS, None))
        + lambda_value * np.log(np.clip(ability, EPS, None))
    )
    np.fill_diagonal(log_probability, -np.inf)
    finite = np.isfinite(log_probability)
    maximum = float(np.max(log_probability[finite]))
    probability = np.zeros_like(log_probability, dtype=float)
    probability[finite] = np.exp(log_probability[finite] - maximum)
    probability /= probability.sum()
    return probability


def _matrix_from_pairs(
    pairs: list[dict[str, Any]],
    bikes: list[int],
) -> np.ndarray:
    index = {bike: position for position, bike in enumerate(bikes)}
    matrix = np.zeros((len(bikes), len(bikes)), dtype=float)
    for item in pairs:
        first, second = (int(item["pair"][0]), int(item["pair"][1]))
        matrix[index[first], index[second]] = float(item["probability"])
    np.fill_diagonal(matrix, 0.0)
    total = float(matrix.sum())
    if total <= 0:
        raise ValueError("能力確率を取得できません")
    return matrix / total


def _rank_candidates(
    probability: np.ndarray,
    odds: np.ndarray,
    bikes: list[int],
    count: int,
    strategy: str,
    extra_matrices: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    lower = _wilson_lower(probability, N_SIMULATIONS)
    conservative_ev = lower * odds
    ev = probability * odds
    np.fill_diagonal(conservative_ev, -np.inf)
    order = np.argsort(conservative_ev, axis=None)[::-1]
    candidates: list[dict[str, Any]] = []
    for flat in order:
        if len(candidates) >= count:
            break
        i, j = np.unravel_index(int(flat), probability.shape)
        if i == j or not np.isfinite(conservative_ev[i, j]):
            continue
        item: dict[str, Any] = {
            "pair": [bikes[i], bikes[j]],
            "probability": float(probability[i, j]),
            "odds": float(odds[i, j]),
            "ev": float(ev[i, j]),
            "conservative_ev": float(conservative_ev[i, j]),
            "rank": len(candidates) + 1,
            "strategy": strategy,
        }
        if extra_matrices:
            for name, matrix in extra_matrices.items():
                item[name] = float(matrix[i, j])
        candidates.append(item)
    return candidates


def _error(code: str, message: str) -> dict[str, Any]:
    return {
        "version": VERSION,
        "status": "INPUT_ERROR",
        "purchase_status": "NO_BET",
        "error": {"code": code, "message": message, "missing": []},
    }


def predict(payload: Any) -> dict[str, Any]:
    """同じレース前データから、しょーご式5点と残差式3点を別計算する。"""
    if not isinstance(payload, dict):
        return _error("INVALID_PAYLOAD", "JSONルートはオブジェクトにしてください")

    race_type = str(payload.get("race_type", "MEN")).upper()
    if race_type != "MEN":
        return _error("WOMEN_EXCLUDED", "女子競輪は対象外です")

    try:
        lambda_value = float(payload.get("lambda_value", DEFAULT_LAMBDA))
    except (TypeError, ValueError):
        return _error("INVALID_LAMBDA", "lambda_valueは0以上1以下の数値にしてください")
    if not 0.0 <= lambda_value <= 1.0:
        return _error("INVALID_LAMBDA", "lambda_valueは0以上1以下にしてください")

    ability_result = predict_ability(payload)
    if ability_result.get("status") != "OK":
        result = dict(ability_result)
        result["version"] = VERSION
        return result

    riders = sorted(payload["riders"], key=lambda rider: int(rider["bike"]))
    bikes = [int(rider["bike"]) for rider in riders]
    odds = np.asarray(
        [
            [np.nan if value is None else float(value) for value in row]
            for row in payload["odds"]
        ],
        dtype=float,
    )
    ability = _matrix_from_pairs(ability_result["pair_probabilities"], bikes)

    shogo_candidates = _rank_candidates(
        ability,
        odds,
        bikes,
        count=5,
        strategy="SHOGO_TOP5",
    )

    market = _market_probability(odds)
    residual_theoretical = _residual_probability(market, ability, lambda_value)
    residual_seed_payload = {
        "grade": payload.get("grade"),
        "riders": payload.get("riders"),
        "lines": payload.get("lines"),
        "conditions": payload.get("conditions", {}),
        "odds": payload.get("odds"),
        "lambda_value": lambda_value,
    }
    residual_seed = _stable_seed(residual_seed_payload, "residual")
    rng = np.random.default_rng(residual_seed)
    flat_counts = rng.multinomial(
        RESIDUAL_SIMULATIONS,
        residual_theoretical.ravel(),
    )
    residual_mc = flat_counts.reshape(residual_theoretical.shape) / RESIDUAL_SIMULATIONS
    np.fill_diagonal(residual_mc, 0.0)

    residual_candidates = _rank_candidates(
        residual_mc,
        odds,
        bikes,
        count=3,
        strategy="RESIDUAL_TOP3",
        extra_matrices={
            "market_probability": market,
            "ability_probability": ability,
            "residual_probability": residual_theoretical,
        },
    )

    shogo_pairs = {tuple(item["pair"]) for item in shogo_candidates}
    residual_pairs = {tuple(item["pair"]) for item in residual_candidates}
    common_pairs = sorted(shogo_pairs & residual_pairs)

    all_pairs: list[dict[str, Any]] = []
    residual_lower = _wilson_lower(residual_mc, RESIDUAL_SIMULATIONS)
    for i, first in enumerate(bikes):
        for j, second in enumerate(bikes):
            if i == j:
                continue
            all_pairs.append(
                {
                    "pair": [first, second],
                    "ability_probability": float(ability[i, j]),
                    "market_probability": float(market[i, j]),
                    "residual_probability": float(residual_theoretical[i, j]),
                    "residual_mc_probability": float(residual_mc[i, j]),
                    "odds": float(odds[i, j]),
                    "shogo_ev": float(ability[i, j] * odds[i, j]),
                    "residual_ev": float(residual_mc[i, j] * odds[i, j]),
                    "residual_conservative_ev": float(
                        residual_lower[i, j] * odds[i, j]
                    ),
                }
            )
    all_pairs.sort(
        key=lambda item: item["residual_conservative_ev"],
        reverse=True,
    )

    result = dict(ability_result)
    result.update(
        {
            "version": VERSION,
            "race_type": "MEN",
            "purchase_status": "DUAL_CANDIDATES",
            "simulations": N_SIMULATIONS,
            "ability_simulations": N_SIMULATIONS,
            "residual_simulations": RESIDUAL_SIMULATIONS,
            "lambda_value": lambda_value,
            "residual_formula": "market * (ability / market) ** lambda",
            "residual_seed": residual_seed,
            "strategies": {
                "shogo": {
                    "name": "しょーご式",
                    "selection_rule": "能力確率から算出した保守EV上位5点",
                    "candidate_count": 5,
                    "purchase_status": "FIVE_PICKS",
                    "candidates": shogo_candidates,
                },
                "residual": {
                    "name": "市場残差システム",
                    "selection_rule": "残差補正後の保守EV上位3点",
                    "candidate_count": 3,
                    "purchase_status": "THREE_PICKS",
                    "formula": "市場確率 ×（能力確率÷市場確率）^λ",
                    "lambda_value": lambda_value,
                    "candidates": residual_candidates,
                },
            },
            # 旧画面・既存利用者向け。主系統はしょーご式5点。
            "candidates": shogo_candidates,
            "common_candidates": [list(pair) for pair in common_pairs],
            "dual_pair_probabilities": all_pairs,
            "audit": {
                "girls_excluded": True,
                "strategies_separated": True,
                "ability_probability_sum": float(ability.sum()),
                "market_probability_sum": float(market.sum()),
                "residual_probability_sum": float(residual_theoretical.sum()),
                "residual_mc_probability_sum": float(residual_mc.sum()),
            },
        }
    )
    return result
