"""KEIRIN.JP 2車単オッズ表のページまたぎ回帰テスト。"""

from __future__ import annotations

import pytest

from keirin_jp_pdf_adapter import (
    PdfInputError,
    _keirin_jp_odds_status,
    _parse_keirin_jp_odds_pages,
)


def _word(
    text: str | int,
    x0: float,
    x1: float,
    top: float,
) -> dict[str, float | str]:
    return {
        "text": str(text),
        "x0": x0,
        "x1": x1,
        "top": top,
        "bottom": top + 10,
    }


def _headers(first_bikes: list[int], top: float) -> list[dict[str, float | str]]:
    return [
        _word(f"{first}番車", slot * 300 + 80, slot * 300 + 130, top)
        for slot, first in enumerate(first_bikes)
    ]


def _rows(
    first_bikes: list[int],
    top: float,
    bikes: list[int],
) -> list[dict[str, float | str]]:
    words: list[dict[str, float | str]] = []
    for slot, first in enumerate(first_bikes):
        row = 0
        for second in bikes:
            if second == first:
                continue
            y = top + row * 24
            base = slot * 300
            words.extend(
                [
                    _word(second, base + 110, base + 120, y),
                    _word(f"{first * 100 + second}.1", base + 180, base + 235, y),
                ]
            )
            row += 1
    return words


def test_keirin_jp_seven_bike_table_across_two_pages() -> None:
    bikes = list(range(1, 8))
    page1 = (
        _headers([1, 2, 3], 100)
        + _rows([1, 2, 3], 140, bikes)
        + _headers([4, 5, 6], 500)
    )
    page2 = (
        _rows([4, 5, 6], 40, bikes)
        + _headers([7], 300)
        + _rows([7], 340, bikes)
    )

    matrix = _parse_keirin_jp_odds_pages(
        [(900, 700, page1), (900, 700, page2)],
        bikes,
    )

    assert sum(value is not None for row in matrix for value in row) == 42
    for i, first in enumerate(bikes):
        for j, second in enumerate(bikes):
            if first == second:
                assert matrix[i][j] is None
            else:
                assert matrix[i][j] == float(f"{first * 100 + second}.1")


def test_closed_keirin_jp_odds_is_rejected() -> None:
    with pytest.raises(PdfInputError) as caught:
        _keirin_jp_odds_status(
            "9Rは締め切りました。\n2車単オッズ\nCOPYRIGHT JKA.",
            9,
        )
    assert caught.value.code == "POST_RACE_SOURCE"


def test_live_keirin_jp_odds_status_is_accepted() -> None:
    assert (
        _keirin_jp_odds_status(
            "7R A級準決\n23:16 現在 オッズ更新\n2車単オッズ",
            7,
        )
        == "23:16 現在 オッズ更新"
    )
