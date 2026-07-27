from individual_api.keirin_pdf_adapter import _normalize_prefecture


def test_abbreviated_prefectures_are_normalized():
    assert _normalize_prefecture("神奈") == "神奈川"
    assert _normalize_prefecture("鹿児") == "鹿児島"
    assert _normalize_prefecture("和歌") == "和歌山"


def test_prefecture_suffix_is_accepted():
    assert _normalize_prefecture("大阪府") == "大阪"
    assert _normalize_prefecture("神奈川県") == "神奈川"


def test_unknown_prefecture_remains_missing_instead_of_being_guessed():
    assert _normalize_prefecture("不明表記") is None
