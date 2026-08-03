from individual_api.keirin_line_runtime_fix import _groups_from_positions


def test_grouping_survives_scaled_spacing():
    found = [1, 5, 2, 3, 9, 7, 6, 8, 4]
    xs = [100, 113.5, 141, 168, 181.5, 195, 222, 249, 262.5]
    assert _groups_from_positions(found, xs) == [[1, 5], [2], [3, 9, 7], [6], [8, 4]]
