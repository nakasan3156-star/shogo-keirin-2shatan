from pathlib import Path

import pytest

import keirin_bundle_adapter as bundle
from keirin_pdf_adapter import PdfInputError


BASIC = "競走得点 逃 捲 差 マ Ｂ 競走得点・バック回数"
HS = "1 着 2 着 3 着 着 外 H・S"
ODDS = "２車単オッズ 1番車 全選択 21:10 現在 オッズ更新"
RECENT = "今回成績 前回成績 もっと見る"


def test_role_scores_are_distinct():
    assert bundle.role_scores(BASIC)["basic"] >= 8
    assert bundle.role_scores(HS)["hs"] >= 8
    assert bundle.role_scores(ODDS)["odds"] >= 10
    assert max(bundle.role_scores(RECENT).values()) == 0


def test_bundle_ignores_extra_and_order(monkeypatch, tmp_path: Path):
    texts = {
        "recent.pdf": RECENT,
        "odds.pdf": ODDS,
        "hs.pdf": HS,
        "basic.pdf": BASIC,
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
    selected, audit = bundle.select_bundle_roles(paths)
    assert selected["basic"].name == "basic.pdf"
    assert selected["hs"].name == "hs.pdf"
    assert selected["odds"].name == "odds.pdf"
    assert audit["ignored"] == ["recent.pdf"]


def test_bundle_rejects_missing_basic(monkeypatch, tmp_path: Path):
    texts = {"hs.pdf": HS, "odds.pdf": ODDS, "recent.pdf": RECENT}
    paths = []
    for name in texts:
        path = tmp_path / name
        path.write_bytes(b"dummy")
        paths.append(path)
    monkeypatch.setattr(bundle, "_extract_text", lambda path, label: texts[path.name])
    monkeypatch.setattr(bundle, "_identity", lambda text, filename: {"venue": "京王閣", "race": 6, "date": None})
    with pytest.raises(PdfInputError) as exc:
        bundle.select_bundle_roles(paths)
    assert exc.value.code == "BUNDLE_REQUIRED_PDF_MISSING"
