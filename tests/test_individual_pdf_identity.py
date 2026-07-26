from pathlib import Path

from fastapi import UploadFile

from app.main import _upload_path
from individual_api.keirin_pdf_adapter import _identity


def test_netkeirin_identity_uses_year_from_original_filename():
    text = """
7/26(日)
豊橋 初日
9R
FII サンプル杯
Ａ級 特予選
"""
    filename = (
        "豊橋競輪 サンプル杯 FII 2026年07月26日 9R 特予選 "
        "出走表 _ 競輪レース情報 - netkeirin（ネットケイリン）.PDF"
    )
    assert _identity(text, filename) == {
        "venue": "豊橋",
        "date": "2026-07-26",
        "race": 9,
    }


def test_keirin_jp_identity_comes_from_pdf_text_with_generic_filename():
    text = """
サンプル杯
豊橋 2026/07/26 (初日)
9R Ａ級特予選 2025m (5周) 先固
"""
    assert _identity(text, "レース情報｜KEIRIN.PDF") == {
        "venue": "豊橋",
        "date": "2026-07-26",
        "race": 9,
    }


def test_final_day_and_filename_fallback_are_supported():
    text = """
7/26(日)
豊橋 最終日
12R
"""
    filename = "豊橋競輪 FI 2026年07月26日 12R 決勝 オッズ.PDF"
    assert _identity(text, filename) == {
        "venue": "豊橋",
        "date": "2026-07-26",
        "race": 12,
    }


def test_upload_path_keeps_original_filename_and_removes_directories(tmp_path):
    upload = UploadFile(
        filename="../豊橋競輪 FII 2026年07月26日 9R 出走表.PDF",
        file=None,
    )
    path = _upload_path(tmp_path, upload, "racecard")
    assert path.parent == Path(tmp_path)
    assert path.name == "racecard__豊橋競輪 FII 2026年07月26日 9R 出走表.PDF"
