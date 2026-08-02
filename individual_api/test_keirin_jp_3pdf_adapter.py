from __future__ import annotations

import unittest

from keirin_dual_strategy_api import predict
from keirin_jp_3pdf_adapter import parse_basic_text, parse_hs_text


BASIC = '''出走表
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
COPYRIGHT JKA. ALL RIGHTS RESERVED.'''

HS = '''出走表
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
9 6 6 15 0 0
2
奥出 良
新 潟/A2A2/逃
5 3 4 18 0 2
3 葛西雄太郎
愛 媛/A2A2/両
1 3 3 14 6 3
4 上吹越俊一
鹿児島/A1A1/両
1 0 0 8 8 2
5 榊原 洋
岡 山/A2A2/追
0 2 7 24 0 0
6 別所 英幸
福 岡/A3A2/追
0 0 4 12 0 0
7
浦川 尊明
茨 城/A1A1/追
0 1 5 21 0 0
誘導 日浦 崇道
COPYRIGHT JKA. ALL RIGHTS RESERVED.'''


class ThreePdfParserTest(unittest.TestCase):
    def test_basic_parser_handles_same_and_split_lines(self) -> None:
        riders = parse_basic_text(BASIC)
        self.assertEqual(len(riders), 7)
        self.assertEqual([r["bike"] for r in riders], list(range(1, 8)))
        self.assertEqual(riders[0]["name"], "是永幸寛")
        self.assertEqual(riders[3]["name"], "上吹越俊一")
        self.assertEqual(riders[6]["score"], 90.42)

    def test_hs_parser_handles_same_and_split_lines(self) -> None:
        rows = parse_hs_text(HS, list(range(1, 8)))
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[3]["H"], 6)
        self.assertEqual(rows[3]["S"], 3)
        self.assertEqual(rows[7]["first"], 0)

    def test_dual_output_stays_fixed(self) -> None:
        riders = parse_basic_text(BASIC)
        hs = parse_hs_text(HS, list(range(1, 8)))
        for rider in riders:
            rider.update(hs[rider["bike"]])
        odds = [
            [None if i == j else float(5 + i * 7 + j) for j in range(7)]
            for i in range(7)
        ]
        payload = {
            "grade": "F2",
            "race_type": "MEN",
            "lambda_value": 0.5,
            "source_files": {
                "racecard_pdf": "basic.pdf",
                "hs_pdf": "hs.pdf",
                "odds_pdf": "odds.pdf",
            },
            "riders": riders,
            "lines": [[4, 2], [3, 1], [5, 6, 7]],
            "odds": odds,
            "conditions": {},
        }
        first = predict(payload)
        second = predict(payload)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "OK")
        self.assertEqual(len(first["strategies"]["shogo"]["candidates"]), 5)
        self.assertEqual(len(first["strategies"]["residual"]["candidates"]), 3)
        self.assertEqual(len(first["dual_pair_probabilities"]), 42)


if __name__ == "__main__":
    unittest.main()
