from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from individual_api.keirin_ac_strategy_api import N_SIMULATIONS, predict
from individual_api.keirin_real_pdf_adapter import normalize_real_bundle


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "keirin_jp" / "2026-08-03_odawara_12r"
BASIC = FIXTURE_ROOT / "レース情報｜KEIRIN(73).PDF"
HS = FIXTURE_ROOT / "レース情報｜KEIRIN(74).PDF"
ODDS = FIXTURE_ROOT / "オッズ｜KEIRIN(12).PDF"
EXPECTED_LINES = [[1, 9, 4], [7, 2, 8], [3, 5], [6]]
EXPECTED_NAMES = [
    "郡司浩平",
    "三谷将太",
    "伊藤旭",
    "佐々木龍",
    "柏野智典",
    "渡邉壘",
    "福永大智",
    "柴崎俊光",
    "松谷秀幸",
]


def _assert_complete_payload(payload: dict, audit: dict) -> None:
    assert audit["race"] == {"venue": "小田原", "date": "2026-08-03", "race": 12}
    assert audit["selected"] == {
        "basic": BASIC.name,
        "hs": HS.name,
        "odds": ODDS.name,
    }
    assert audit["rider_count"] == 9
    assert audit["odds_count"] == 72
    assert audit["lines"] == EXPECTED_LINES
    assert [rider["bike"] for rider in payload["riders"]] == list(range(1, 10))
    assert [rider["name"] for rider in payload["riders"]] == EXPECTED_NAMES
    assert [(rider["H"], rider["S"]) for rider in payload["riders"]] == [
        (4, 8),
        (0, 1),
        (2, 3),
        (0, 2),
        (0, 1),
        (9, 4),
        (3, 12),
        (0, 4),
        (0, 4),
    ]
    assert sum(
        value is not None
        for first, row in enumerate(payload["odds"])
        for second, value in enumerate(row)
        if first != second
    ) == 72


def test_ss_grade_real_three_pdf_normalization_and_both_strategies() -> None:
    payload, audit = normalize_real_bundle([HS, ODDS, BASIC])
    _assert_complete_payload(payload, audit)

    payload["race_type"] = "MEN"
    result = predict(payload)
    assert result["status"] == "OK"
    assert result["strategies"]["a"]["name"] == "A方式"
    assert result["strategies"]["a"]["candidates"][0]["pair"] == [1, 4]
    assert result["strategies"]["c"]["simulations"] == N_SIMULATIONS == 100_000
    assert len(result["strategies"]["c"]["candidates"]) == 6


def test_ss_grade_real_three_pdf_fastapi_returns_200_with_a_and_c() -> None:
    with TestClient(app) as client:
        handles = [ODDS.open("rb"), BASIC.open("rb"), HS.open("rb")]
        try:
            response = client.post(
                "/analyze-bundle",
                files=[
                    ("files", (path.name, handle, "application/pdf"))
                    for path, handle in zip((ODDS, BASIC, HS), handles)
                ],
            )
        finally:
            for handle in handles:
                handle.close()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["purchase_status"] == "A_AND_C_READY"
    assert body["pdf_audit"]["rider_count"] == 9
    assert body["pdf_audit"]["odds_count"] == 72
    assert body["pdf_audit"]["lines"] == EXPECTED_LINES
    assert body["strategies"]["a"]["candidates"][0]["pair"] == [1, 4]
    assert body["strategies"]["c"]["simulations"] == 100_000
    assert len(body["strategies"]["c"]["candidates"]) == 6
