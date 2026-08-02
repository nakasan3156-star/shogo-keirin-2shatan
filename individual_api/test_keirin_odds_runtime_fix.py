from keirin_odds_runtime_fix import install_odds_parser_fix


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
