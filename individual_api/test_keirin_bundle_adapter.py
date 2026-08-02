from pathlib import Path

import pytest

import keirin_bundle_adapter as bundle
from keirin_pdf_adapter import PdfInputError


REAL_BASIC = "REAL_BASIC 競走得点 逃 捲 差 マ Ｂ"
REAL_HS = "REAL_HS 1着 2着 3着 着外 H S"
REAL_ODDS = "REAL_ODDS 2車単オッズ"
# KEIRIN.JPの別タブにも共通メニューとしてこの文字が載る。
DECOY = "基本情報 直近成績 対戦成績 当場成績 着度数 H・S回数 競走得点・バック回数"


def _riders():
    return [
        {"bike": bike, "name": f"選手{bike}"}
        for bike in range(1, 8)
    ]


def _odds_matrix():
    return [
        [None if first == second else 10.0 + first + second for second in range(7)]
        for first in range(7)
    ]


def _install_real_parsers(monkeypatch):
    def parse_basic(text):
        if "REAL_BASIC" not in text:
            raise PdfInputError("BASIC_PARSE_FAILED", "not basic")
        return _riders()

    def parse_hs(text, bikes):
        if "REAL_HS" not in text:
            raise PdfInputError("HS_PARSE_FAILED", "not hs")
        return {bike: {"H": 0, "S": 0} for bike in bikes}

    def parse_odds(path, text, bikes):
        if "REAL_ODDS" not in text:
            raise PdfInputError("ODDS_PARSE_FAILED", "not odds")
        return _odds_matrix()

    monkeypatch.setattr(bundle, "parse_basic_text", parse_basic)
    monkeypatch.setattr(bundle, "parse_hs_text", parse_hs)
    monkeypatch.setattr(bundle, "_parse_keirin_jp_odds_pdf", parse_odds)


def test_bundle_uses_full_parse_and_ignores_keyword_decoy(monkeypatch, tmp_path: Path):
    texts = {
        "decoy.pdf": DECOY,
        "odds.pdf": REAL_ODDS,
        "hs.pdf": REAL_HS,
        "basic.pdf": REAL_BASIC,
    }
    paths = []
    for name in texts:
        path = tmp_path / name
        path.write_bytes(b"dummy")
        paths.append(path)

    monkeypatch.setattr(bundle, "_extract_text", lambda path, label: texts[path.name])
    monkeypatch.setattr(
        bundle,
        "_identity",
        lambda text, filename: {"venue": "京王閣", "race": 6, "date": "2026/08/01"},
    )
    _install_real_parsers(monkeypatch)

    selected, audit = bundle.select_bundle_roles(paths)
    assert selected["basic"].name == "basic.pdf"
    assert selected["hs"].name == "hs.pdf"
    assert selected["odds"].name == "odds.pdf"
    assert audit["ignored"] == ["decoy.pdf"]
    assert audit["selection_method"] == "full_content_parse"
    assert audit["rider_count"] == 7
    assert audit["odds_count"] == 42


def test_bundle_rejects_keyword_only_basic(monkeypatch, tmp_path: Path):
    texts = {
        "decoy.pdf": DECOY,
        "hs.pdf": REAL_HS,
        "odds.pdf": REAL_ODDS,
    }
    paths = []
    for name in texts:
        path = tmp_path / name
        path.write_bytes(b"dummy")
        paths.append(path)

    monkeypatch.setattr(bundle, "_extract_text", lambda path, label: texts[path.name])
    monkeypatch.setattr(
        bundle,
        "_identity",
        lambda text, filename: {"venue": "京王閣", "race": 6, "date": None},
    )
    _install_real_parsers(monkeypatch)

    with pytest.raises(PdfInputError) as exc:
        bundle.select_bundle_roles(paths)
    assert exc.value.code == "BUNDLE_BASIC_PARSE_FAILED"


def test_bundle_rejects_mixed_races(monkeypatch, tmp_path: Path):
    texts = {
        "basic.pdf": REAL_BASIC,
        "hs.pdf": REAL_HS,
        "odds.pdf": REAL_ODDS,
    }
    paths = []
    for name in texts:
        path = tmp_path / name
        path.write_bytes(b"dummy")
        paths.append(path)

    monkeypatch.setattr(bundle, "_extract_text", lambda path, label: texts[path.name])
    monkeypatch.setattr(
        bundle,
        "_identity",
        lambda text, filename: {
            "venue": "京王閣",
            "race": 7 if filename == "odds.pdf" else 6,
            "date": "2026/08/01",
        },
    )
    _install_real_parsers(monkeypatch)

    with pytest.raises(PdfInputError) as exc:
        bundle.select_bundle_roles(paths)
    assert exc.value.code in {"BUNDLE_PARSED_ROLE_MISSING", "BUNDLE_RACE_MISMATCH"}
