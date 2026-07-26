import copy
import math

from keirin_individual_api import N_SIMULATIONS, predict, predict_json


def sample_payload(grade="F1"):
    riders = []
    for bike in range(1, 8):
        riders.append({
            "bike": bike, "name": f"選手{bike}", "region": "関東" if bike <= 3 else "近畿",
            "score": 95 + bike, "win_rate": 5 + bike*2, "escape": bike % 3,
            "makuri": (bike+1) % 4, "sashi": (bike+2) % 4, "mark": bike % 2,
            "H": (8-bike) % 5, "B": (9-bike) % 6,
        })
    odds = []
    for i in range(7):
        row = []
        for j in range(7):
            row.append(None if i == j else float(8 + i*3 + j))
        odds.append(row)
    return {
        "grade": grade,
        "source_files": {
            "racecard_pdf": "racecard.pdf",
            "hs_pdf": "hs.pdf",
            "odds_pdf": "odds.pdf",
            "ex_image": "ex.png",
        },
        "riders": riders, "lines": [[1, 2, 3], [4, 5], [6, 7]], "odds": odds,
        "conditions": {"bank_type": "400_outdoor", "wind_mps": 2.0, "temperature_c": 24.0},
    }


def run_tests():
    payload = sample_payload()
    first = predict(payload)
    second = predict(copy.deepcopy(payload))
    assert first == second
    assert first["status"] == "OK"
    assert first["simulations"] == N_SIMULATIONS
    assert math.isclose(sum(p["probability"] for p in first["pair_probabilities"]), 1.0)
    assert len(first["candidates"]) <= 2

    changed_odds = sample_payload()
    changed_odds["odds"][0][1] = 29.9
    odds_result = predict(changed_odds)
    assert first["seed"] == odds_result["seed"]
    assert first["pair_probabilities"] == odds_result["pair_probabilities"]

    without_ex = sample_payload()
    del without_ex["source_files"]["ex_image"]
    without_ex_result = predict(without_ex)
    assert without_ex_result["status"] == "OK"
    assert "ex_image" in without_ex_result["missing_optional"]

    without_hs_pdf = sample_payload()
    del without_hs_pdf["source_files"]["hs_pdf"]
    without_hs_result = predict(without_hs_pdf)
    assert without_hs_result["purchase_status"] == "NO_BET"
    assert without_hs_result["error"]["code"] == "MISSING_SOURCE"

    g1 = predict(sample_payload("G1"))
    assert g1["purchase_status"] == "NO_BET"
    assert g1["candidates"] == []
    assert g1["audit_candidates"] is not None

    broken = sample_payload()
    del broken["riders"][0]["B"]
    error = predict(broken)
    assert error["status"] == "INPUT_ERROR"
    assert error["purchase_status"] == "NO_BET"
    assert "riders[0].B" in error["error"]["missing"]

    missing_odds = sample_payload()
    missing_odds["odds"][0][1] = None
    odds_error = predict(missing_odds)
    assert odds_error["status"] == "INPUT_ERROR"
    assert odds_error["purchase_status"] == "NO_BET"

    unordered = sample_payload()
    unordered["riders"][0], unordered["riders"][1] = unordered["riders"][1], unordered["riders"][0]
    order_error = predict(unordered)
    assert order_error["error"]["code"] == "INVALID_RIDER_ORDER"

    malformed_payloads = [
        None,
        [],
        {"source_files": None},
        {"source_files": [], "grade": "F1"},
        {**sample_payload(), "riders": [None, None, None, None, None]},
        {**sample_payload(), "lines": [1, 2, 3]},
        {**sample_payload(), "odds": [None] * 7},
        {**sample_payload(), "conditions": "強風"},
    ]
    for malformed in malformed_payloads:
        malformed_result = predict(malformed)
        assert malformed_result["purchase_status"] == "NO_BET"
        assert malformed_result["status"] in {"INPUT_ERROR", "PROCESSING_ERROR"}

    invalid_json = predict_json("{broken")
    assert '"purchase_status": "NO_BET"' in invalid_json
    assert '"code": "INVALID_JSON"' in invalid_json
    print("17 tests passed")


if __name__ == "__main__":
    run_tests()
