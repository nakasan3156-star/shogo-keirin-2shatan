from __future__ import annotations

from copy import deepcopy

from keirin_dual_strategy_api import predict


PAYLOAD = {
    "race_type": "MEN",
    "grade": "F2",
    "lambda_value": 0.50,
    "source_files": {
        "racecard_pdf": "racecard.pdf",
        "hs_pdf": "hs.pdf",
        "odds_pdf": "odds.pdf",
    },
    "riders": [
        {"bike": 1, "name": "選手1", "region": "近畿", "score": 92.4, "win_rate": 24.0, "escape": 5, "makuri": 3, "sashi": 1, "mark": 0, "H": 4, "B": 8},
        {"bike": 2, "name": "選手2", "region": "近畿", "score": 90.8, "win_rate": 18.0, "escape": 0, "makuri": 1, "sashi": 4, "mark": 5, "H": 1, "B": 0},
        {"bike": 3, "name": "選手3", "region": "近畿", "score": 88.9, "win_rate": 12.0, "escape": 0, "makuri": 0, "sashi": 3, "mark": 6, "H": 0, "B": 0},
        {"bike": 4, "name": "選手4", "region": "関東", "score": 91.5, "win_rate": 21.0, "escape": 3, "makuri": 5, "sashi": 1, "mark": 0, "H": 3, "B": 6},
        {"bike": 5, "name": "選手5", "region": "関東", "score": 89.7, "win_rate": 15.0, "escape": 0, "makuri": 1, "sashi": 5, "mark": 4, "H": 0, "B": 0},
        {"bike": 6, "name": "選手6", "region": "九州", "score": 87.2, "win_rate": 10.0, "escape": 1, "makuri": 4, "sashi": 1, "mark": 1, "H": 1, "B": 2},
        {"bike": 7, "name": "選手7", "region": "中部", "score": 86.5, "win_rate": 8.0, "escape": 0, "makuri": 2, "sashi": 3, "mark": 2, "H": 1, "B": 1},
    ],
    "lines": [[1, 2, 3], [4, 5], [6], [7]],
    "odds": [
        [None, 5.2, 18.4, 13.1, 24.5, 31.2, 42.0],
        [7.8, None, 21.0, 19.5, 28.1, 40.0, 48.0],
        [26.0, 18.0, None, 35.0, 44.0, 52.0, 61.0],
        [11.2, 20.0, 33.0, None, 6.4, 25.0, 32.0],
        [23.0, 29.0, 41.0, 8.8, None, 38.0, 46.0],
        [28.0, 36.0, 49.0, 21.0, 34.0, None, 27.0],
        [39.0, 47.0, 58.0, 30.0, 43.0, 26.0, None],
    ],
    "conditions": {"wind_mps": 2.0, "temperature_c": 28.0, "bank_type": "400_outdoor"},
}


def test_dual_strategies_return_fixed_counts_and_are_deterministic() -> None:
    first = predict(deepcopy(PAYLOAD))
    second = predict(deepcopy(PAYLOAD))
    assert first == second
    assert first["status"] == "OK"
    assert first["purchase_status"] == "DUAL_CANDIDATES"
    assert len(first["strategies"]["shogo"]["candidates"]) == 5
    assert len(first["strategies"]["residual"]["candidates"]) == 3
    assert len(first["dual_pair_probabilities"]) == 42
    assert first["audit"]["strategies_separated"] is True
    assert abs(first["audit"]["ability_probability_sum"] - 1.0) < 1e-12
    assert abs(first["audit"]["market_probability_sum"] - 1.0) < 1e-12
    assert abs(first["audit"]["residual_probability_sum"] - 1.0) < 1e-12
    assert abs(first["audit"]["residual_mc_probability_sum"] - 1.0) < 1e-12


def test_lambda_endpoints_match_market_and_ability() -> None:
    market_payload = deepcopy(PAYLOAD)
    market_payload["lambda_value"] = 0.0
    market_result = predict(market_payload)
    for item in market_result["dual_pair_probabilities"]:
        assert abs(item["residual_probability"] - item["market_probability"]) < 1e-12

    ability_payload = deepcopy(PAYLOAD)
    ability_payload["lambda_value"] = 1.0
    ability_result = predict(ability_payload)
    for item in ability_result["dual_pair_probabilities"]:
        assert abs(item["residual_probability"] - item["ability_probability"]) < 1e-12


def test_women_are_rejected() -> None:
    payload = deepcopy(PAYLOAD)
    payload["race_type"] = "WOMEN"
    result = predict(payload)
    assert result["status"] == "INPUT_ERROR"
    assert result["purchase_status"] == "NO_BET"
    assert result["error"]["code"] == "WOMEN_EXCLUDED"
