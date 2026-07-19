from pathlib import Path

from app.engine import predict
from app.parser import (
    DOCUMENT_BASIC,
    DOCUMENT_ODDS,
    DOCUMENT_RECENT,
    RaceInput,
    Rider,
    classify_pdf,
    parse_entry_pdf,
    parse_odds_pdf,
    parse_recent_pdf,
)


WORKSPACE = Path(__file__).resolve().parents[3]
UPLOAD = WORKSPACE / "upload"
YAHIKO_ENTRY = UPLOAD / "弥彦競輪 ＪＰＦ杯・伊藤克信賞 FII 2026年07月19日 9R チャレンジ決勝 出走表 _ 競輪レース情報 - netkeirin（ネットケイリン）.PDF"
YAHIKO_ODDS = UPLOAD / "弥彦競輪 ＪＰＦ杯・伊藤克信賞 FII 2026年07月19日 9R チャレンジ決勝 オッズ _ 競輪レース情報 - netkeirin（ネットケイリン）.PDF"
KOCHI_ENTRY = UPLOAD / "高知競輪 サマーナイトフェスティバル GII 2026年07月19日 1R 一　般 出走表 _ 競輪レース情報 - netkeirin（ネットケイリン）.PDF"
KOCHI_ODDS = UPLOAD / "高知競輪 サマーナイトフェスティバル GII 2026年07月19日 1R 一　般 オッズ _ 競輪レース情報 - netkeirin（ネットケイリン）.PDF"
SHIZUOKA_RECENT = UPLOAD / "静岡競輪 ＪＣ×ＨＰＣＪＣ・Ｋドリ杯 FI 2026年07月19日 10R 準決勝 出走表 _ 競輪レース情報 - netkeirin（ネットケイリン）.PDF"
KOCHI3_ENTRY = UPLOAD / "高知競輪 サマーナイトフェスティバル GII 2026年07月19日 3R 一　般 出走表 _ 競輪レース情報 - netkeirin（ネットケイリン）.PDF"
KOCHI3_RECENT = UPLOAD / "高知競輪 サマーナイトフェスティバル GII 2026年07月19日 3R 一　般 出走表 _ 競輪レース情報 - netkeirin（ネットケイリン）(1).PDF"
KOCHI3_ODDS = UPLOAD / "高知競輪 サマーナイトフェスティバル GII 2026年07月19日 3R 一　般 オッズ _ 競輪レース情報 - netkeirin（ネットケイリン）.PDF"


def test_content_classification():
    assert classify_pdf(YAHIKO_ENTRY) == DOCUMENT_BASIC
    assert classify_pdf(YAHIKO_ODDS) == DOCUMENT_ODDS
    assert classify_pdf(SHIZUOKA_RECENT) == DOCUMENT_RECENT


def test_real_seven_and_nine_rider_fields():
    cases = [
        (YAHIKO_ENTRY, YAHIKO_ODDS, 7, 42, ((1, 7), (3, 4), (5, 2), (6,))),
        (KOCHI_ENTRY, KOCHI_ODDS, 9, 72, ((1, 7), (2, 6), (4, 9), (8, 3, 5))),
    ]
    for entry, odds_path, field_size, combinations, expected_lines in cases:
        race = parse_entry_pdf(entry)
        odds = parse_odds_pdf(odds_path, [rider.number for rider in race.riders])
        assert len(race.riders) == field_size
        assert len(odds) == combinations
        assert race.lines == expected_lines


def test_recent_pdf_extracts_all_seven_riders():
    names = ["小川三士郎", "菅原大也", "纐纈洸翔", "黒沢征治", "滝本幸正", "新村穣", "中井俊亮"]
    riders = tuple(
        Rider(
            frame=min(number, 6), number=number, name=name, prefecture="未取得", age=30,
            term=None, grade=None, race_score=100.0, style="逃" if number < 7 else "追",
            start_count=2, back_count=5, escape_count=1, makuri_count=1,
            sashi_count=0, mark_count=0, first_count=3, second_count=3,
            third_count=3, outside_count=11, win_rate=.15, top2_rate=.30,
            top3_rate=.45, gear=3.92, comment="",
        )
        for number, name in enumerate(names, start=1)
    )
    race = RaceInput(
        race_id="20260719_静岡_10R", venue="静岡", race_number=10,
        race_class="準決勝", distance_m=None, riders=riders,
        lines=tuple((number,) for number in range(1, 8)), line_warnings=(),
    )
    recent = parse_recent_pdf(SHIZUOKA_RECENT, race)
    assert recent["status"] == "complete"
    assert len(recent["riders"]) == 7
    assert all(recent["riders"][number]["finishes"] for number in range(1, 8))


def test_real_monte_carlo_is_complete_and_reproducible():
    for entry, odds_path, expected in ((YAHIKO_ENTRY, YAHIKO_ODDS, 42), (KOCHI_ENTRY, KOCHI_ODDS, 72)):
        race = parse_entry_pdf(entry)
        odds = parse_odds_pdf(odds_path, [rider.number for rider in race.riders])
        history = {
            "status": "complete",
            "target_race_id": race.race_id,
            "riders": {
                rider.number: {
                    "recent_form_score": 50.0 + rider.number,
                    "lost_strong_proxy": {"score": 10.0 if rider.style in {"逃", "両"} else 0.0},
                }
                for rider in race.riders
            },
        }
        first = predict(race, odds, runs=100_000, seed=3156, history=history)
        second = predict(race, odds, runs=100_000, seed=3156, history=history)
        assert first["all_combinations"] == second["all_combinations"]
        assert len(first["all_combinations"]) == expected
        assert first["calculation"]["probability_sum_all_ordered_pairs"] == 1.0
        assert first["calculation"]["odds_used_for_probability_estimation"] is False


def test_same_race_three_pdf_pipeline_is_complete_and_reproducible():
    assert classify_pdf(KOCHI3_ENTRY) == DOCUMENT_BASIC
    assert classify_pdf(KOCHI3_RECENT) == DOCUMENT_RECENT
    assert classify_pdf(KOCHI3_ODDS) == DOCUMENT_ODDS

    race = parse_entry_pdf(KOCHI3_ENTRY)
    history = parse_recent_pdf(KOCHI3_RECENT, race)
    odds = parse_odds_pdf(KOCHI3_ODDS, [rider.number for rider in race.riders])

    assert race.race_id == "20260719_高知_3R"
    assert len(race.riders) == 9
    assert race.lines == ((2,), (3, 6), (4,), (5, 8), (9, 1, 7))
    assert history["status"] == "complete"
    assert len(history["riders"]) == 9
    assert all(history["riders"][number]["finishes"] for number in range(1, 10))
    assert len(odds) == 72

    first = predict(race, odds, runs=100_000, seed=3156, history=history)
    second = predict(race, odds, runs=100_000, seed=3156, history=history)
    assert first["all_combinations"] == second["all_combinations"]
    assert len(first["all_combinations"]) == 72
    assert first["calculation"]["monte_carlo_runs"] == 100_000
    assert first["calculation"]["probability_sum_all_ordered_pairs"] == 1.0
    assert all(item["odds"] is not None for item in first["all_combinations"])
