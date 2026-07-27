import pytest

from individual_api.keirin_pdf_adapter import (
    PdfInputError,
    _grade,
    _identity,
    _normalize_prefecture,
    _pre_race_status,
)


def test_abbreviated_prefectures_are_normalized():
    assert _normalize_prefecture("神奈") == "神奈川"
    assert _normalize_prefecture("鹿児") == "鹿児島"
    assert _normalize_prefecture("和歌") == "和歌山"


def test_prefecture_suffix_is_accepted():
    assert _normalize_prefecture("大阪府") == "大阪"
    assert _normalize_prefecture("神奈川県") == "神奈川"


def test_unknown_prefecture_remains_missing_instead_of_being_guessed():
    assert _normalize_prefecture("不明表記") is None


def test_fullwidth_grade_is_normalized_and_missing_grade_is_nonfatal():
    assert _grade("  ＧⅢ 開設記念") == "G3"
    assert _grade("グレード表記なし") == "UNKNOWN"


def test_hyphenated_date_and_spaced_race_number_are_accepted():
    assert _identity("豊橋 2026-07-27 3日目\n10 R S級", "") == {
        "venue": "豊橋",
        "date": "2026-07-27",
        "race": 10,
    }


def test_missing_status_badge_is_nonfatal_but_explicit_end_stops():
    assert _pre_race_status("10R\nS級予選", 10, "racecard") == "未取得（終了表示なし）"
    with pytest.raises(PdfInputError):
        _pre_race_status("10R\nS級予選\n終了", 10, "racecard")
