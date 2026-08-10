from individual_api.previous_day_kdreams import _labels, detect_day_no, fetch_previous_day


def test_first_day_never_fetches_previous_results() -> None:
    result = fetch_previous_day("小田原", "2026-08-03", 1, ["選手A"])
    assert result["status"] == "FIRST_DAY_SKIPPED"
    assert result["riders"] == {}


def test_day_detection_prefers_explicit_day_header() -> None:
    assert detect_day_no("小田原 初日 2026/08/03\n4R") == 1
    assert detect_day_no("小田原 2日目 2026/08/04\n4R") == 2
    assert detect_day_no("小田原 3日目 2026/08/05\n4R") == 3
    assert detect_day_no("小田原 最終日 2026/08/06\n4R") >= 2


def test_validated_bandte_and_blocked_labels_are_loss_only() -> None:
    bandte = _labels({"finish": 5, "actual_back": 0, "comment": "競りで番手飛ばされ苦しい"})
    assert bandte["validated"]["bandte_fight_4plus"] is True
    assert bandte["validated"]["blocked_4plus"] is False

    blocked = _labels({"finish": 4, "actual_back": 0, "comment": "直線で進路詰まりコースなく"})
    assert blocked["validated"]["blocked_4plus"] is True

    placed = _labels({"finish": 2, "actual_back": 0, "comment": "直線で進路詰まり"})
    assert placed["validated"]["blocked_4plus"] is False


def test_other_line_win_is_not_fabricated_without_previous_line_data() -> None:
    result = _labels({"finish": 5, "actual_back": 1, "comment": "先行して4角まで粘る"})
    assert result["validated"]["back_4plus_otherline_win"] is False
