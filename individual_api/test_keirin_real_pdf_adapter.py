from keirin_real_pdf_adapter import parse_basic_real, parse_hs_real
from individual_api.keirin_line_runtime_fix import _groups_from_positions, _valid


BASIC_TEXT = '''
   1 片山  智晴
     岡 山/S2A1/追                             93.20        0        0    1    3     0
   2 伊藤  大彦
     徳 島/A1A1/追                             86.82        0        1    3    1     0
   3 浅見  隼追加
     福 岡/A1A1/逃                             85.12        5        5    1    0     6
   4 中武三四郎
     大 阪/A1A2/逃                             84.37        8        1    0    0     15
   5 仲松  勝太
     沖 縄/A2A2/追                             84.19        0        0    5    0     0
   6 山原  利秀
     高 知/A3A2/追                             75.38        0        0    3    2     0
   7 柏野  健吾
     岡 山/A3A2/逃                             73.09        14       3    0    0     19
'''

HS_TEXT = '''
   1 片山  智晴
     岡 山/S2A1/追                             1       3              3       18         1       0
   2 伊藤  大彦
     徳 島/A1A1/追                             2       3              9       9          1       1
   3 浅見  隼追加
     福 岡/A1A1/逃                             6       5              0       13         8       0
   4 中武三四郎
     大 阪/A1A2/逃                             3       6              5       13         14      3
   5 仲松  勝太
     沖 縄/A2A2/追                             1       4              5       11         0       5
   6 山原  利秀
     高 知/A3A2/追                             2       3              3       10         0       0
   7 柏野  健吾
     岡 山/A3A2/逃                             6       11             0       15         19      0
'''


def test_actual_keirin_jp_layout_reads_seven_riders():
    riders = parse_basic_real(BASIC_TEXT)
    assert len(riders) == 7
    assert [rider["bike"] for rider in riders] == list(range(1, 8))
    assert riders[0]["name"] == "片山智晴"
    assert riders[2]["name"] == "浅見隼"
    assert riders[3]["B"] == 15
    assert riders[6]["escape"] == 14


def test_actual_keirin_jp_layout_reads_all_hs_rows():
    rows = parse_hs_real(HS_TEXT, list(range(1, 8)))
    assert set(rows) == set(range(1, 8))
    assert rows[1]["first"] == 1
    assert rows[4]["H"] == 14
    assert rows[5]["S"] == 5
    assert rows[7]["second"] == 11


def _positions(lines, intra=28.0, inter=56.0, offset=0.0, jitter=0.0):
    found, xs = [], []
    x = offset
    count = 0
    for line_index, line in enumerate(lines):
        for member_index, bike in enumerate(line):
            found.append(bike)
            xs.append(x + (jitter if count % 2 else -jitter))
            count += 1
            if member_index < len(line) - 1:
                x += intra
        if line_index < len(lines) - 1:
            x += inter
    return found, xs


def test_multiple_real_keirin_line_formations_and_layout_scales():
    formations = [
        [[1, 5], [2], [3, 9, 7], [6], [8, 4]],  # 小田原・9車・コマ切れ
        [[2, 5], [1, 6], [7], [4, 3]],          # 武雄・四分戦
        [[6, 4], [1, 3], [5, 2], [7]],          # 武雄・四分戦
        [[7, 1], [3, 5], [4, 2, 6]],            # 高知・三分戦
        [[4, 2], [3, 1], [5, 6, 7]],            # 京王閣・三分戦
        [[1, 5, 7], [2, 4, 3, 6]],              # 和歌山・二分戦
    ]
    layouts = [
        (28.0, 56.0, 0.0, 0.0),
        (14.0, 31.0, 90.0, 0.4),
        (36.0, 80.0, 12.0, 0.8),
        (9.0, 22.0, 250.0, 0.2),
    ]
    for expected in formations:
        for intra, inter, offset, jitter in layouts:
            found, xs = _positions(expected, intra, inter, offset, jitter)
            actual = _groups_from_positions(found, xs)
            assert actual == expected
            assert _valid(actual, sorted(found))


def test_line_validation_rejects_duplicate_and_missing_bikes():
    assert not _valid([[1, 2], [2, 3]], [1, 2, 3])
    assert not _valid([[1, 2], [4]], [1, 2, 3, 4])
