from __future__ import annotations

from copy import deepcopy

from keirin_ac_strategy_api import (
    C_MAX_PURCHASE_CANDIDATES,
    C_MIN_EV,
    C_MIN_ODDS,
    C_MIN_PROBABILITY,
    C_MIN_PURCHASE_CANDIDATES,
    C_SEED,
    C_TOP6_HIT_RATE,
    C_TOP6_HITS,
    C_VALIDATION_RACES,
    N_SIMULATIONS,
    predict,
)
from test_keirin_dual_strategy_api import PAYLOAD


def test_c_ev_is_applied_only_after_probability_and_can_purchase() -> None:
    result = predict(deepcopy(PAYLOAD))
    c = result["strategies"]["c"]

    assert c["simulations"] == N_SIMULATIONS == 100_000
    assert c["seed"] == C_SEED == 3156
    assert c["purchase_status"] == "BET"
    assert C_MIN_PURCHASE_CANDIDATES <= len(c["purchase_candidates"]) <= C_MAX_PURCHASE_CANDIDATES
    assert c["validation_scope"] == "RANKING_VALIDATED_EV_FORMULA_FIXED"
    assert c["ev_formula"] == "probability * odds"
    assert c["ev_formula_fixed"] is True
    assert c["ev_backtest_validated"] is False
    assert c["validation_races"] == C_VALIDATION_RACES == 876
    assert c["validated_top6_hits"] == C_TOP6_HITS == 486
    assert c["validated_top6_hit_rate"] == C_TOP6_HIT_RATE
    assert len(c["candidates"]) == 6
    assert all(item["ev"] == item["probability"] * item["odds"] for item in c["candidates"])
    assert all(item["probability"] >= C_MIN_PROBABILITY for item in c["purchase_candidates"])
    assert all(C_MIN_ODDS <= item["odds"] <= 30.0 for item in c["purchase_candidates"])
    assert all(item["ev"] >= C_MIN_EV for item in c["purchase_candidates"])
    assert result["audit"]["c_ev_used_for_purchase"] is True
    assert result["audit"]["c_odds_applied_after_probability"] is True


def test_top_level_candidates_include_separated_a_and_c_purchase_candidates() -> None:
    result = predict(deepcopy(PAYLOAD))
    by_strategy = result["purchase_candidates_by_strategy"]
    assert by_strategy["a"] == result["strategies"]["a"]["candidates"]
    assert by_strategy["c"] == result["strategies"]["c"]["purchase_candidates"]
    assert result["candidates"] == by_strategy["a"] + by_strategy["c"]
    assert result["purchase_status"] == "A_AND_C_BET"
