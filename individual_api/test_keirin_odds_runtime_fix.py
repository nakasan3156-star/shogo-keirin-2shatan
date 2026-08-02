from keirin_odds_runtime_fix import (
    _matrix_from_closed_tables,
    install_odds_parser_fix,
)


def _word(text, x0, x1, top, bottom):
    return {"text": text, "x0": x0, "x1": x1, "top": top, "bottom": bottom}


def test_control_character_headers_parse_all_pairs():
    install_odds_parser_fix()
    from keirin_jp_pdf_adapter import _parse_keirin_jp_odds_pages

    words = [
        _word("1番車\x01全選択", 40, 150, 20, 32),
        _word("2番車\x01全選択", 240, 350, 20, 32),
        _word("3番車\x01全選択", 440, 550, 20, 32),
        _word("2", 60, 70, 55, 65),
        _word("12.3", 115, 145, 55, 65),
        _word("3", 60, 70, 80, 90),
        _word("23.4", 115, 145, 80, 90),
        _word("1", 260, 270, 55, 65),
        _word("34.5", 315, 345, 55, 65),
        _word("3", 260, 270, 80, 90),
        _word("45.6", 315, 345, 80, 90),
        _word("1", 460, 470, 55, 65),
        _word("56.7", 515, 545, 55, 65),
        _word("2", 460, 470, 80, 90),
        _word("67.8", 515, 545, 80, 90),
    ]
    matrix = _parse_keirin_jp_odds_pages([(600.0, 120.0, words)], [1, 2, 3])
    assert matrix == [
        [None, 12.3, 23.4],
        [34.5, None, 45.6],
        [56.7, 67.8, None],
    ]


def test_closed_odds_table_parses_all_pairs():
    # KEIRIN.JP締切後PDFの実表と同じ、3列組×3ブロック形式。
    table = [
        ["1", "2", "23.9", "2", "1", "22.0", "3", "1", "7.9"],
        [None, "3", "21.9", None, "3", "21.3", None, "2", "9.6"],
        ["4", "1", "41.8", "5", "1", "112.2", "6", "1", "344.4"],
        [None, "2", "15.8", None, "2", "114.1", None, "2", "105.8"],
        ["7", "1", "22.1", None, None, None, None, None, None],
        [None, "2", "106.4", None, None, None, None, None, None],
    ]
    # 3車版へ縮約して完全性を検査。
    small = [
        ["1", "2", "12.3", "2", "1", "34.5", "3", "1", "56.7"],
        [None, "3", "23.4", None, "3", "45.6", None, "2", "67.8"],
    ]
    matrix = _matrix_from_closed_tables([table, small], [1, 2, 3])
    assert matrix == [
        [None, 12.3, 23.4],
        [34.5, None, 45.6],
        [56.7, 67.8, None],
    ]


def test_closed_status_is_allowed():
    install_odds_parser_fix()
    from keirin_jp_pdf_adapter import _keirin_jp_odds_status

    status = _keirin_jp_odds_status("5Rは締め切りました。\n22:18 現在", 5)
    assert status == "締切後オッズ 22:18 現在"
