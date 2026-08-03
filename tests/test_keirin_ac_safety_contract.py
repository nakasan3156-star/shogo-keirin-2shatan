from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as app_main
from app.main import app
from individual_api.keirin_ac_strategy_api import predict
from individual_api.keirin_real_pdf_adapter import normalize_real_bundle


FIXTURES = Path(__file__).parent / "fixtures" / "keirin_jp"
RACE_12R = FIXTURES / "2026-08-03_odawara_12r"
RACE_4R = FIXTURES / "2026-08-03_odawara_4r"


def _synthetic_payload(n: int = 7) -> dict:
    riders = []
    for bike in range(1, n + 1):
        riders.append({
            "bike": bike,
            "name": f"選手{bike}",
            "score": 85.0 + bike,
            "win_rate": 5.0 + bike,
            "quinella_rate": 15.0 + bike,
            "first": bike,
            "second": bike + 1,
            "third": bike + 2,
            "out": bike + 6,
            "escape": bike % 3,
            "makuri": (bike + 1) % 4,
            "sashi": (bike + 2) % 5,
            "mark": (bike + 3) % 4,
            "H": bike % 4,
            "B": (bike + 1) % 5,
        })
    lines = [[1, 2, 3], [4, 5], list(range(6, n + 1))]
    odds = [
        [None if first == second else 8.0 + first * n + second for second in range(n)]
        for first in range(n)
    ]
    return {
        "race_type": "MEN",
        "grade": "F2",
        "riders": riders,
        "lines": lines,
        "odds": odds,
        "conditions": {},
    }


def _assert_safe_stop(result: dict) -> None:
    assert result["status"] in {"INPUT_ERROR", "PROCESSING_ERROR"}
    assert result["purchase_status"] == "NO_BET"
    assert result["error"]["message"]
    assert "detail" not in result["error"]


def test_direct_api_never_raises_for_malformed_inputs() -> None:
    base = _synthetic_payload()
    malformed = [None, [], {}, {"riders": []}]

    missing = deepcopy(base)
    del missing["riders"][0]["B"]
    malformed.append(missing)

    not_finite = deepcopy(base)
    not_finite["riders"][0]["score"] = float("nan")
    malformed.append(not_finite)

    bad_lines = deepcopy(base)
    bad_lines["lines"] = [[1, 2], [2, 3], [4, 5, 6, 7]]
    malformed.append(bad_lines)

    placeholder_odds = deepcopy(base)
    placeholder_odds["odds"][0][1] = 9999.9
    malformed.append(placeholder_odds)

    for payload in malformed:
        _assert_safe_stop(predict(payload))


def test_seven_rider_result_is_finite_complete_and_repeatable() -> None:
    payload = _synthetic_payload(7)
    first = predict(deepcopy(payload))
    second = predict(deepcopy(payload))

    assert first == second
    assert first["status"] == "OK"
    pairs = first["strategies"]["c"]["all_pair_probabilities"]
    assert len(pairs) == 42
    assert np.isclose(sum(item["probability"] for item in pairs), 1.0)
    assert all(np.isfinite(item["probability"]) and np.isfinite(item["ev"]) for item in pairs)
    assert first["strategies"]["c"]["seed"] == 3156


@pytest.mark.parametrize("race_dir", [RACE_12R, RACE_4R])
def test_real_nine_rider_pdfs_remain_repeatable(race_dir: Path) -> None:
    payload, _ = normalize_real_bundle(sorted(race_dir.glob("*.PDF")), None)
    payload["race_type"] = "MEN"

    first = predict(deepcopy(payload))
    second = predict(deepcopy(payload))

    assert first == second
    assert first["status"] == "OK"
    pairs = first["strategies"]["c"]["all_pair_probabilities"]
    assert len(pairs) == 72
    assert np.isclose(sum(item["probability"] for item in pairs), 1.0)


def test_bundle_rejects_wrong_count_invalid_duplicate_and_oversize(monkeypatch) -> None:
    with TestClient(app) as client:
        two = [("files", (f"{i}.pdf", b"%PDF-distinct-" + bytes([i]), "application/pdf")) for i in range(2)]
        assert client.post("/analyze-bundle", files=two).status_code == 400

        invalid = [("files", (f"{i}.pdf", b"not-a-pdf-" + bytes([i]), "application/pdf")) for i in range(3)]
        assert client.post("/analyze-bundle", files=invalid).status_code == 400

        duplicate = [("files", (f"{i}.pdf", b"%PDF-same", "application/pdf")) for i in range(3)]
        duplicate_response = client.post("/analyze-bundle", files=duplicate)
        assert duplicate_response.status_code == 400
        assert "重複" in duplicate_response.json()["detail"]

        monkeypatch.setattr(app_main, "MAX_PDF_BYTES", 16)
        oversize = [("files", (f"{i}.pdf", b"%PDF-" + bytes([65 + i]) * 20, "application/pdf")) for i in range(3)]
        assert client.post("/analyze-bundle", files=oversize).status_code == 413


def test_unexpected_parser_failure_is_sanitized(monkeypatch) -> None:
    def fail_safely(*_args, **_kwargs):
        raise RuntimeError("internal-secret-must-not-leak")

    monkeypatch.setattr(app_main, "normalize_real_bundle", fail_safely)
    files = [("files", (f"{i}.pdf", b"%PDF-distinct-" + bytes([i]), "application/pdf")) for i in range(3)]
    with TestClient(app) as client:
        response = client.post("/analyze-bundle", files=files)

    assert response.status_code == 422
    body = response.json()
    _assert_safe_stop(body)
    assert "RuntimeError" not in response.text
    assert "internal-secret" not in response.text


def test_mixed_race_bundle_returns_safe_stop_instead_of_server_error() -> None:
    basic = RACE_4R / "レース情報｜KEIRIN(71).PDF"
    hs = RACE_4R / "レース情報｜KEIRIN(72).PDF"
    odds = RACE_12R / "オッズ｜KEIRIN(12).PDF"
    handles = [path.open("rb") for path in (basic, hs, odds)]
    try:
        files = [("files", (path.name, handle, "application/pdf")) for path, handle in zip((basic, hs, odds), handles)]
        with TestClient(app) as client:
            response = client.post("/analyze-bundle", files=files)
    finally:
        for handle in handles:
            handle.close()

    assert response.status_code == 422
    _assert_safe_stop(response.json())
