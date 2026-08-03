from pathlib import Path

from individual_api.keirin_line_runtime_fix import _groups_from_positions, _valid


def test_adaptive_grouping_for_nine_rider_komagire():
    found = [1, 5, 2, 3, 9, 7, 6, 8, 4]
    xs = [178.4, 206.4, 262.4, 318.5, 346.5, 374.5, 430.5, 486.5, 514.5]
    parsed = _groups_from_positions(found, xs)
    assert parsed == [[1, 5], [2], [3, 9, 7], [6], [8, 4]]
    assert _valid(parsed, list(range(1, 10)))


def test_no_duplicate_or_missing_bikes_allowed():
    assert not _valid([[1, 2], [2, 3]], [1, 2, 3])
