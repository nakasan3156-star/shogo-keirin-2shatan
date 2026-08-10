from individual_api.previous_day_kdreams import (
    _labels,
    _lineup_map,
    _previous_summary,
    _result_detail,
    _winner_car,
    detect_day_no,
    fetch_previous_day,
)


def test_first_day_never_fetches_previous_results() -> None:
    result = fetch_previous_day("小田原", "2026-08-03", 1, 4, ["選手A"])
    assert result["status"] == "FIRST_DAY_SKIPPED"
    assert result["resolved_day_no"] == 1
    assert result["riders"] == {}


def test_day_detection_prefers_explicit_day_header_and_defers_final_resolution() -> None:
    assert detect_day_no("小田原 初日 2026/08/03\n4R") == 1
    assert detect_day_no("小田原 2日目 2026/08/04\n4R") == 2
    assert detect_day_no("小田原 3日目 2026/08/05\n4R") == 3
    assert detect_day_no("小田原 最終日 2026/08/06\n4R") == 0


def test_current_race_previous_summary_reads_only_previous_section() -> None:
    html = """
    <html><body>
      <table>
        <tr><th>選手名</th><th>選手コメント</th><th>前回出走レースの成績</th><th>走り評</th></tr>
        <tr><td>望月 湧世</td><td>自力。</td><td>初日 11R 8</td><td><a>叩き捲られ詳細</a></td></tr>
        <tr><td>高木 和仁</td><td>坂本君。</td><td>初日 11R 6</td><td><a>３番手離れ詳細</a></td></tr>
      </table>
      <table class="result_table"><tr><th>着順</th><th>選手名</th></tr><tr><td>1</td><td>別の選手</td></tr></table>
    </body></html>
    """
    rows = _previous_summary(html, ["望月 湧世", "高木 和仁"])
    assert rows["望月 湧世"]["previous_race_no"] == 11
    assert rows["望月 湧世"]["finish"] == 8
    assert rows["望月 湧世"]["short_review"] == "叩き捲られ"
    assert rows["高木 和仁"]["finish"] == 6


def test_previous_result_detail_reads_finish_sb_lap_and_reason() -> None:
    html = """
    <table class="result_table">
      <tr><th>予想</th><th>着順</th><th>車番</th><th>選手名</th><th>着差</th><th>上り</th><th>決まり手</th><th>S／B</th><th>勝敗因</th></tr>
      <tr><td>◎</td><td>1</td><td>1</td><td>伊藤 旭</td><td></td><td>9.3</td><td>差</td><td></td><td>東矢を交す</td></tr>
      <tr><td>○</td><td>2</td><td>9</td><td>東矢 圭吾</td><td>1/8</td><td>9.4</td><td>捲</td><td>B</td><td>ロング捲り</td></tr>
      <tr><td></td><td>8</td><td>8</td><td>望月 湧世</td><td>6車身</td><td>10.3</td><td></td><td>S</td><td>叩き捲られ</td></tr>
    </table>
    """
    rows = _result_detail(html, ["東矢 圭吾", "望月 湧世"])
    assert rows["東矢 圭吾"] == {
        "finish": 2,
        "car_no": 9,
        "actual_start": 0,
        "actual_back": 1,
        "final_lap_time": 9.4,
        "comment": "ロング捲り",
    }
    assert rows["望月 湧世"]["actual_start"] == 1
    assert rows["望月 湧世"]["actual_back"] == 0


def test_kdreams_lineup_is_split_by_non_follow_roles() -> None:
    html = """
    <html><body>
      <div>並び予想 ← 3先行 2追込 7追込 1押え先 4追込 5追込 6自在</div>
      <div>オッズ</div>
    </body></html>
    """
    assert _lineup_map(html) == {3: 1, 2: 1, 7: 1, 1: 2, 4: 2, 5: 2, 6: 3}


def test_winner_car_reads_completed_previous_result_only() -> None:
    html = """
    <table class="result_table">
      <tr><th>着順</th><th>車番</th><th>選手名</th></tr>
      <tr><td>1</td><td>4</td><td>勝者</td></tr>
      <tr><td>2</td><td>2</td><td>二着</td></tr>
    </table>
    """
    assert _winner_car(html) == 4


def test_validated_bandte_and_blocked_labels_are_loss_only() -> None:
    bandte = _labels({"finish": 5, "actual_back": 0, "comment": "競りで番手飛ばされ苦しい"})
    assert bandte["pr31"]["C"] == 1
    assert bandte["validated"]["bandte_fight_4plus"] is True
    assert bandte["validated"]["blocked_4plus"] is False

    blocked = _labels({"finish": 4, "actual_back": 0, "comment": "直線で進路詰まりコースなく"})
    assert blocked["pr31"]["F"] == 1
    assert blocked["validated"]["blocked_4plus"] is True

    placed = _labels({"finish": 2, "actual_back": 0, "comment": "直線で進路詰まり"})
    assert placed["pr31"]["F"] == 1
    assert placed["validated"]["blocked_4plus"] is False


def test_pr31_b_keeps_original_back_and_top3_gate() -> None:
    good = _labels({"finish": 3, "actual_back": 1, "comment": "先行して粘る"})
    bad_finish = _labels({"finish": 4, "actual_back": 1, "comment": "先行して粘る"})
    bad_back = _labels({"finish": 2, "actual_back": 0, "comment": "先行して粘る"})
    assert good["pr31"]["B"] == 1
    assert bad_finish["pr31"]["B"] == 0
    assert bad_back["pr31"]["B"] == 0


def test_other_line_win_requires_back_loss_and_confirmed_different_line() -> None:
    hit = _labels({
        "finish": 5,
        "actual_back": 1,
        "comment": "先行して粘る",
        "previous_line_no": 1,
        "previous_winner_line_no": 2,
    })
    same_line = _labels({
        "finish": 5,
        "actual_back": 1,
        "previous_line_no": 1,
        "previous_winner_line_no": 1,
    })
    missing = _labels({"finish": 5, "actual_back": 1})
    not_back = _labels({
        "finish": 5,
        "actual_back": 0,
        "previous_line_no": 1,
        "previous_winner_line_no": 2,
    })
    assert hit["validated"]["back_4plus_otherline_win"] is True
    assert same_line["validated"]["back_4plus_otherline_win"] is False
    assert missing["validated"]["back_4plus_otherline_win"] is False
    assert not_back["validated"]["back_4plus_otherline_win"] is False
