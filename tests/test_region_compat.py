import pytest

from individual_api.keirin_pdf_adapter import (
    PdfInputError,
    _grade,
    _identity,
    _normalize_prefecture,
    _pre_race_status,
    _rider_stat_rows,
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


def test_rider_rows_accept_missing_frame_cells_and_missing_percent_glyph():
    text = """
1 101.25 逃 10 8 4 2 1 0 5 3 1 11 25.0
2  2 99.50 追 12 0 0 0 4 3 4 3 2 13 18.2%
3 98.00 両 9 3 1 2 2 1 3 2 1 10 18.7％
4  4 97.75 追 8 0 0 0 3 2 2 3 1 12 11.1 %
5 96.30 逃 11 6 3 1 0 0 4 2 2 14 18.1%
"""
    rows = _rider_stat_rows(text)
    assert [int(row.group(2)) for row in rows] == [1, 2, 3, 4, 5]
    assert [float(row.group(3)) for row in rows] == [
        101.25, 99.50, 98.00, 97.75, 96.30,
    ]


def test_identical_duplicate_rider_row_is_ignored():
    row = "1 101.25 逃 10 8 4 2 1 0 5 3 1 11 25.0%"
    rows = _rider_stat_rows(f"{row}\n{row}\n")
    assert len(rows) == 1
