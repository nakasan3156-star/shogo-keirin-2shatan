from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from individual_api.keirin_ac_strategy_api import N_SIMULATIONS, predict
from individual_api.keirin_real_pdf_adapter import normalize_real_bundle


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "keirin_jp" / "2026-08-03_odawara_4r"
BASIC = FIXTURE_ROOT / "レース情報｜KEIRIN(71).PDF"
HS = FIXTURE_ROOT / "レース情報｜KEIRIN(72).PDF"
ODDS = FIXTURE_ROOT / "オッズ｜KEIRIN(11).PDF"
EXPECTED_LINES = [[1, 5], [2], [3, 9, 7], [6], [8, 4]]
EXPECTED_NAMES = [
    "梅崎隆介",
    "星野洋輝",
    "中井俊亮",
    "内藤高裕",
    "山口敦也",
    "坂本周作",
    "横関裕樹",
    "小池千啓",
    "川口公太朗",
]


def _assert_complete_payload(payload: dict, audit: dict) -> None:
    assert audit["race"] == {"venue": "小田原", "date": "2026-08-03", "race": 4}
    assert audit["rider_count"] == 9
    assert audit["odds_count"] == 72
    assert audit["lines"] == EXPECTED_LINES
    assert audit["line_method"].startswith("pdfplumber_coordinates")
    assert len(payload["riders"]) == 9
    assert [rider["bike"] for rider in payload["riders"]] == list(range(1, 10))
    assert [rider["name"] for rider in payload["riders"]] == EXPECTED_NAMES
    assert [(rider["H"], rider["S"]) for rider in payload["riders"]] == [
        (5, 3), (0, 5), (0, 6), (0, 3), (0, 2), (4, 0), (0, 10), (5, 2), (0, 2)
    ]
    assert len(payload["odds"]) == 9
    assert all(len(row) == 9 for row in payload["odds"])
    assert sum(
        value is not None
        for first, row in enumerate(payload["odds"])
        for second, value in enumerate(row)
        if first != second
    ) == 72


def test_real_three_pdf_normalization_and_both_strategies() -> None:
    # Deliberately use arbitrary upload order; role detection must use content.
    payload, audit = normalize_real_bundle([ODDS, HS, BASIC])
    _assert_complete_payload(payload, audit)

    payload["race_type"] = "MEN"
    result = predict(payload)
    assert result["status"] == "OK"
    assert "a" in result["strategies"]
    assert "c" in result["strategies"]
    assert result["strategies"]["a"]["name"] == "A方式"
    assert result["strategies"]["c"]["simulations"] == N_SIMULATIONS == 100_000
    assert len(result["strategies"]["c"]["candidates"]) == 6


def test_real_three_pdf_fastapi_returns_200_and_ui_binds_a_c_results() -> None:
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
        assert body["pdf_audit"]["lines"] == EXPECTED_LINES
        assert body["pdf_audit"]["rider_count"] == 9
        assert body["pdf_audit"]["odds_count"] == 72
        assert body["strategies"]["c"]["simulations"] == 100_000

        html = client.get("/")
        assert html.status_code == 200
        assert 'id="aResult"' in html.text
        assert 'id="cResult"' in html.text
        assert "data.strategies?.a?.candidates" in html.text
        assert "data.strategies?.c?.candidates" in html.text
