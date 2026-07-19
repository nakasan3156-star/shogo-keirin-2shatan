from pathlib import Path

from app.history import extract_race_id_from_pdf, score_lost_strong


WORKSPACE = Path(__file__).resolve().parents[3]
ENTRY = WORKSPACE / "upload" / "奈良競輪 前検日コメならウィンチケット杯 FII 2026年07月18日 9R 決　勝 出走表 _ 競輪レース情報 - netkeirin（ネットケイリン）.PDF"


def test_extract_target_race_id_from_real_pdf_annotations():
    assert extract_race_id_from_pdf(ENTRY, 9) == "202607185309"


def test_lost_strong_proxy_uses_only_structured_evidence():
    race = {
        "race_id": "202601010101",
        "lines": [[1, 2], [3]],
        "results": [
            {"player_id": "p1", "number": 1, "finish": 4, "agari": 10.0, "sb": "B"},
            {"player_id": "p2", "number": 2, "finish": 2, "agari": 10.2, "sb": ""},
            {"player_id": "p3", "number": 3, "finish": 1, "agari": 9.9, "sb": ""},
        ],
    }
    output = score_lost_strong("p1", [race])
    assert output["score"] == 75.0
    assert output["video_interference_included"] is False
    assert output["evidence"][0]["reasons"] == [
        "着順4着に対して上がり2位",
        "ライン先頭でBを取りながら着外",
        "先頭が着外でも同ライン後位が2着以内",
    ]
