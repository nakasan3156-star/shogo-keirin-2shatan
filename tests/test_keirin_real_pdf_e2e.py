from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from individual_api.keirin_real_pdf_adapter import normalize_real_bundle
from individual_api.pr31_runtime import predict_pr31


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "keirin_jp" / "2026-08-03_odawara_4r"
BASIC = FIXTURE_ROOT / "レース情報｜KEIRIN(71).PDF"
HS = FIXTURE_ROOT / "レース情報｜KEIRIN(72).PDF"
ODDS = FIXTURE_ROOT / "オッズ｜KEIRIN(11).PDF"
EXPECTED_LINES = [[1, 5], [2], [3, 9, 7], [6], [8, 4]]
EXPECTED_NAMES = [
    "梅崎隆介", "星野洋輝", "中井俊亮", "内藤高裕", "山口敦也",
    "坂本周作", "横関裕樹", "小池千啓", "川口公太朗",
]


def _no_network_previous(*_args, **_kwargs) -> dict:
    return {
        "status": "PREVIOUS_DAY_NOT_FOUND",
        "source": "KDreams",
        "resolved_day_no": 3,
        "previous_date": "2026-08-02",
        "riders": {},
    }


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


def test_real_three_pdf_normalization_and_pr31(monkeypatch) -> None:
    monkeypatch.setattr("individual_api.pr31_runtime.fetch_previous_day", _no_network_previous)
    payload, audit = normalize_real_bundle([ODDS, HS, BASIC])
    _assert_complete_payload(payload, audit)
    payload["race_type"] = "MEN"
    result = predict_pr31(payload, audit, BASIC)
    assert result["status"] == "OK"
    assert result["engine"] == "PR31_FROZEN_ONLY"
    assert result["a_strategy"] == "REMOVED"
    assert result["c_strategy"] == "REMOVED"
    assert result["race"]["day_no"] == 3
    assert len(result["riders"]) == 9
    assert len(result["pair_ranking"]) > 0
    assert all("ev" in item for item in result["pair_ranking"])
    assert result["previous_day"]["status"] == "PREVIOUS_DAY_NOT_FOUND"


def test_real_three_pdf_fastapi_returns_200_and_ui_binds_pr31_results(monkeypatch) -> None:
    monkeypatch.setattr("individual_api.pr31_runtime.fetch_previous_day", _no_network_previous)
    with TestClient(app) as client:
        basic_handle, hs_handle, odds_handle = BASIC.open("rb"), HS.open("rb"), ODDS.open("rb")
        try:
            response = client.post(
                "/analyze",
                files={
                    "basic_pdf": (BASIC.name, basic_handle, "application/pdf"),
                    "hs_pdf": (HS.name, hs_handle, "application/pdf"),
                    "odds_pdf": (ODDS.name, odds_handle, "application/pdf"),
                },
            )
        finally:
            basic_handle.close()
            hs_handle.close()
            odds_handle.close()

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "OK"
        assert body["engine"] == "PR31_FROZEN_ONLY"
        assert body["a_strategy"] == "REMOVED"
        assert body["c_strategy"] == "REMOVED"
        assert body["purchase_status"] in {"BET", "NO_BET"}
        assert body["race"]["day_no"] == 3
        assert body["pdf_audit"]["lines"] == EXPECTED_LINES
        assert body["pdf_audit"]["rider_count"] == 9
        assert body["pdf_audit"]["odds_count"] == 72
        assert body["previous_day"]["status"] == "PREVIOUS_DAY_NOT_FOUND"
        assert len(body["pair_ranking"]) > 0
        assert all("ev" in item for item in body["pair_ranking"])

        html = client.get("/")
        assert html.status_code == 200
        assert "PR31" in html.text
        assert "A方式・C方式：撤廃" not in html.text
        assert "① 基本情報PDF" in html.text
        assert "② H/S・着度数PDF" in html.text
        assert "③ 2車単オッズPDF" in html.text
        assert "fetch('/analyze'" in html.text
        assert "負けて強し／展開不利" in html.text
        assert "data.selections" in html.text
        assert "data.previous_day" in html.text
