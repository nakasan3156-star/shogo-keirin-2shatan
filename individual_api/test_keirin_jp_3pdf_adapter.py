from keirin_jp_3pdf_adapter import _detect_race_info_roles, parse_basic_text


BASIC_7 = '''
枠
番
車
番
選手名
府県/級班前現/脚質
競走得点 逃 捲 差 マ Ｂ
調
子
1 是永 幸寛
福 岡/A2A1/追
84.57 0 0 1 1 0
2 奥出 良
新 潟/A2A2/逃
84.07 3 5 0 0 9
3 葛西雄太郎
愛 媛/A2A2/両
81.69 3 0 3 3 7
4
上吹越俊一
鹿児島/A1A1/両
85.27 3 4 0 0 6
5
榊原 洋
岡 山/A2A2/追
82.16 0 0 2 1 0
6
別所 英幸
福 岡/A3A2/追
74.40 0 1 4 1 1
7
浦川 尊明
茨 城/A1A1/追
90.42 0 1 8 3 0
誘導 日浦 崇道
'''

HS_7 = '''
枠
番
車
番
選手名
府県/級班前現/脚質
1
着
2
着
3
着
着
外
H S
1
是永 幸寛
福 岡/A2A1/追
1 1 6 13 0 3
2
奥出 良
新 潟/A2A2/逃
6 2 2 17 12 0
3 葛西雄太郎
愛 媛/A2A2/両
1 8 3 21 10 6
4 上吹越俊一
鹿児島/A1A1/両
6 1 3 8 4 0
5 榊原 洋
岡 山/A2A2/追
1 2 4 24 0 14
6 別所 英幸
福 岡/A3A2/追
5 1 1 9 0 4
7
浦川 尊明
茨 城/A1A1/追
6 6 7 10 0 2
誘導 日浦 崇道
'''

BASIC_9 = '''
枠
番
車
番
選手名
府県/級班前現/脚質 競走得点 逃 捲 差 マ Ｂ
1 1
伊藤 旭
熊 本/S1S1/両
106.50 0 3 4 1 2
2 2
高久保雄介
京 都/S1S2/両
103.20 1 4 2 0 4
3 3 杉森 輝大
茨 城/S1S1/追
108.10 0 1 6 3 0
4
4 高木 和仁
福 岡/S2S2/追
95.30 0 0 2 1 0
5 鈴木 陸来
静 岡/S2S2/逃
99.40 7 5 1 0 7
5
6 松田 治之
大 阪/S1S1/追
94.10 0 0 1 1 0
7
武田 亮
東 京/S2S2/逃
101.20 8 3 0 0 11
6
8
望月 湧世
静 岡/S2S2/逃
97.80 10 2 0 0 15
9
東矢 圭吾
熊 本/S1S1/逃
104.60 4 4 1 0 4
誘導 山田 幸司
'''


def test_basic_parser_handles_real_seven_rider_layout() -> None:
    riders = parse_basic_text(BASIC_7)
    assert len(riders) == 7
    assert [r["bike"] for r in riders] == list(range(1, 8))
    assert riders[0]["name"] == "是永幸寛"
    assert riders[3]["name"] == "上吹越俊一"
    assert riders[6]["score"] == 90.42


def test_basic_parser_handles_nine_rider_frame_and_bike_layout() -> None:
    riders = parse_basic_text(BASIC_9)
    assert len(riders) == 9
    assert [r["bike"] for r in riders] == list(range(1, 10))
    assert riders[0]["name"] == "伊藤旭"
    assert riders[7]["B"] == 15
    assert riders[8]["style"] == "逃"


def test_race_info_pdfs_are_auto_detected_in_either_order() -> None:
    basic_index, riders, hs = _detect_race_info_roles(BASIC_7, HS_7)
    assert basic_index == 0
    assert len(riders) == 7
    assert hs[2]["H"] == 12

    reversed_index, reversed_riders, reversed_hs = _detect_race_info_roles(HS_7, BASIC_7)
    assert reversed_index == 1
    assert reversed_riders == riders
    assert reversed_hs == hs
