from __future__ import annotations

from copy import deepcopy

from keirin_ac_strategy_api import (
    C_TOP6_HIT_RATE,
    C_TOP6_HITS,
    C_VALIDATION_RACES,
    N_SIMULATIONS,
    predict,
)
from test_keirin_dual_strategy_api import PAYLOAD


def test_c_ranking_is_never_exposed_as_purchase_or_ev() -> None:
    result = predict(deepcopy(PAYLOAD))
    c = result["strategies"]["c"]

    assert c["simulations"] == N_SIMULATIONS == 100_000
    assert c["purchase_status"] == "REFERENCE_ONLY"
    assert c["purchase_candidates"] == []
    assert c["validation_scope"] == "RANKING_ONLY"
    assert c["ev_validated"] is False
    assert c["validation_races"] == C_VALIDATION_RACES == 876
    assert c["validated_top6_hits"] == C_TOP6_HITS == 486
    assert c["validated_top6_hit_rate"] == C_TOP6_HIT_RATE
    assert len(c["candidates"]) == 6
    assert all("ev" not in item for item in c["candidates"])
    assert result["audit"]["c_ev_used_for_purchase"] is False


def test_top_level_candidates_are_a_purchase_candidates_only() -> None:
    result = predict(deepcopy(PAYLOAD))
    assert result["candidates"] == result["strategies"]["a"]["candidates"]
    assert result["purchase_status"] in {"A_BET", "NO_BET"}
