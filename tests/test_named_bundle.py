from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app, health
from individual_api.named_bundle import normalize_named_bundle

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "keirin_jp" / "2026-08-03_odawara_4r"
BASIC = FIXTURE_ROOT / "レース情報｜KEIRIN(71).PDF"
HS = FIXTURE_ROOT / "レース情報｜KEIRIN(72).PDF"
ODDS = FIXTURE_ROOT / "オッズ｜KEIRIN(11).PDF"
EXPECTED_LINES = [[1, 5], [2], [3, 9, 7], [6], [8, 4]]


def test_named_bundle_parses_each_fixed_role_directly() -> None:
    payload, audit = normalize_named_bundle(BASIC, HS, ODDS)
    assert audit["selection_method"] == "real_named_parse"
    assert "history_schedule" not in audit
    assert audit["web_data_used"] is False
    assert audit["rider_count"] == 9
    assert audit["odds_count"] == 72
    assert audit["lines"] == EXPECTED_LINES
    assert audit["line_source"].endswith(ODDS.name)
    assert audit["line_method"].startswith("fixed_odds_fast_v2:")
    assert len(payload["riders"]) == 9


def test_health_declares_pdf_only_runtime() -> None:
    result = health()
    assert result["engine"] == "PR31_FROZEN_ONLY"
    assert result["prediction_inputs"] == "KEIRIN_JP_3PDF_ONLY"
    assert result["previous_day_features"] == "removed"
    assert result["external_web_lookup"] == "disabled"


def test_named_fastapi_path_returns_pr31_without_previous_day() -> None:
    with TestClient(app) as client:
        with BASIC.open("rb") as basic, HS.open("rb") as hs, ODDS.open("rb") as odds:
            response = client.post(
                "/analyze",
                files={
                    "basic_pdf": (BASIC.name, basic, "application/pdf"),
                    "hs_pdf": (HS.name, hs, "application/pdf"),
                    "odds_pdf": (ODDS.name, odds, "application/pdf"),
                },
            )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OK"
    assert body["engine"] == "PR31_FROZEN_ONLY"
    assert "previous_day" not in body
    assert body["pdf_audit"]["selection_method"] == "real_named_parse"
    assert "history_schedule" not in body["pdf_audit"]
    assert body["pdf_audit"]["web_data_used"] is False
    assert body["pdf_audit"]["rider_count"] == 9
    assert body["pdf_audit"]["odds_count"] == 72
    assert body["pdf_audit"]["lines"] == EXPECTED_LINES
    assert body["pdf_audit"]["line_method"].startswith("fixed_odds_fast_v2:")
