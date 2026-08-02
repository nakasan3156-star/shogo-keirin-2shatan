from keirin_jp_3pdf_adapter import parse_basic_text


def test_basic_parser_handles_keirin_jp_mobile_layout():
    text = '''
枠
番
車
番
選手名
府県/級班前現/脚質 競走得点 逃 捲 差 マ Ｂ
1
宇佐見優介
福 島/A2A1/追
94.66 0 0 4 3 0
2
鈴木 庸之
新 潟/A1A1/両
93.08 0 1 4 0 0
3 大高 彰馬
福 島/A1A1/逃
90.66 1 4 1 0 2
4 土田 栄二
茨 城/A1A1/逃
89.95 11 0 1 0 15
5 松坂 侑亮
神奈川/A1A1/逃
88.62 6 8 1 0 14
6 原岡泰志郎
千 葉/A2A2/両
88.04 0 0 5 1 0
7
望月 紀男
静 岡/A2A2/追
86.29 0 0 2 1 0
誘導
'''
    riders = parse_basic_text(text)
    assert len(riders) == 7
    assert [r["bike"] for r in riders] == list(range(1, 8))
    assert riders[0]["name"] == "宇佐見優介"
    assert riders[2]["name"] == "大高彰馬"
    assert riders[4]["B"] == 14
