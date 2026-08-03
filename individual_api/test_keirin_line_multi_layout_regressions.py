from __future__ import annotations

import pytest

from individual_api.keirin_line_runtime_fix import _groups_from_positions, _valid


def positions_for(lines: list[list[int]], *, intra: float, inter: float, offset: float = 0.0, jitter: float = 0.0):
    found: list[int] = []
    xs: list[float] = []
    x = offset
    k = 0
    for line_index, line in enumerate(lines):
        for member_index, bike in enumerate(line):
            found.append(bike)
            xs.append(x + (jitter if k % 2 else -jitter))
            k += 1
            if member_index < len(line) - 1:
                x += intra
        if line_index < len(lines) - 1:
            x += inter
    return found, xs


CASES = [
    # 小田原 2026-08-03 4R・9車・コマ切れ
    [[1, 5], [2], [3, 9, 7], [6], [8, 4]],
    # 武雄 2026-08-01 7R・7車・四分戦
    [[2, 5], [1, 6], [7], [4, 3]],
    # 武雄 2026-08-01 6R・7車・四分戦
    [[6, 4], [1, 3], [5, 2], [7]],
    # 高知 2026-08-02 5R・7車・三分戦
    [[7, 1], [3, 5], [4, 2, 6]],
    # 京王閣 2026-08-01 6R・7車・三分戦
    [[4, 2], [3, 1], [5, 6, 7]],
    # 和歌山 2026-07-30 2R・7車・二分戦
    [[1, 5, 7], [2, 4, 3, 6]],
    # 9車三分戦
    [[1, 4, 7], [2, 5, 8], [3, 6, 9]],
    # 9車四分戦（2車・3車ラインと単騎を含む）
    [[1, 5], [2, 6, 9], [3], [4, 7, 8]],
]


@pytest.mark.parametrize("expected", CASES)
@pytest.mark.parametrize(
    "intra,inter,offset,jitter",
    [
        (28.0, 56.0, 0.0, 0.0),
        (14.0, 31.0, 90.0, 0.4),
        (36.0, 80.0, 12.0, 0.8),
        (9.0, 22.0, 250.0, 0.2),
    ],
)
def test_real_formations_survive_scale_offset_and_small_jitter(
    expected: list[list[int]], intra: float, inter: float, offset: float, jitter: float
) -> None:
    found, xs = positions_for(expected, intra=intra, inter=inter, offset=offset, jitter=jitter)
    actual = _groups_from_positions(found, xs)
    assert actual == expected
    assert _valid(actual, sorted(found))


def test_duplicate_bike_is_rejected() -> None:
    assert not _valid([[1, 2], [2, 3]], [1, 2, 3])


def test_missing_bike_is_rejected() -> None:
    assert not _valid([[1, 2], [4]], [1, 2, 3, 4])


def test_preserves_order_inside_each_line() -> None:
    expected = [[7, 1], [3, 5], [4, 2, 6]]
    found, xs = positions_for(expected, intra=21.0, inter=47.0)
    assert _groups_from_positions(found, xs) == expected
