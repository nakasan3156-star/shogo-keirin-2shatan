from individual_api.keirin_individual_api import _validate


def _payload(grade: str) -> dict:
    riders = []
    for bike in range(1, 6):
        riders.append(
            {
                "bike": bike,
                "name": f"選手{bike}",
                "region": "関東",
                "score": 90.0 + bike,
                "win_rate": 10.0 + bike,
                "escape": bike,
                "makuri": bike,
                "sashi": bike,
                "mark": bike,
                "H": bike,
                "B": bike,
            }
        )
    odds = [
        [None if first == second else 10.0 for second in range(1, 6)]
        for first in range(1, 6)
    ]
    return {
        "grade": grade,
        "source_files": {
            "racecard_pdf": "racecard.pdf",
            "hs_pdf": "hs.pdf",
            "odds_pdf": "odds.pdf",
        },
        "riders": riders,
        "lines": [[1, 2], [3, 4], [5]],
        "odds": odds,
        "conditions": {},
    }


def test_f2_is_valid_for_calculation():
    assert _validate(_payload("F2")) is None


def test_g2_and_unknown_grade_are_nonfatal():
    assert _validate(_payload("G2")) is None
    assert _validate(_payload("UNKNOWN")) is None


def test_unrecognized_grade_still_stops_safely():
    result = _validate(_payload("INVALID"))
    assert result is not None
    assert result["status"] == "INPUT_ERROR"
    assert result["error"]["code"] == "INVALID_GRADE"
