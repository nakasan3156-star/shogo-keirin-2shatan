from __future__ import annotations

import unittest

from keirin_dual_strategy_api import predict
from keirin_jp_5pdf_adapter import _require_role
from keirin_jp_6pdf_adapter import parse_basic_text, parse_hs_text
from keirin_pdf_adapter import PdfInputError


BASIC = '''枠
番
車
番
選手名
府県/級班前現/脚質 競走得点 逃 捲 差 マ Ｂ
調
子
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
7 望月 紀男
静 岡/A2A2/追
86.29 0 0 2 1 0
誘導'''

HS = '''枠
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
宇佐見優介
福 島/A2A1/追
1 6 3 2 0 5
2
鈴木 庸之
新 潟/A1A1/両
2 3 1 6 0 3
3 大高 彰馬
福 島/A1A1/逃
2 4 1 8 1 1
4 土田 栄二
茨 城/A1A1/逃
6 6 4 5 15 8
5 松坂 侑亮
神奈川/A1A1/逃
11 4 1 11 13 9
6 原岡泰志郎
千 葉/A2A2/両
4 2 7 11 0 9
7 望月 紀男
静 岡/A2A2/追
2 1 4 11 0 0
誘導'''


class FivePdfTest(unittest.TestCase):
    def test_required_roles(self) -> None:
        _require_role("競走得点 逃 捲 差 マ B", "basic_pdf")
        _require_role("今回成績 前回成績", "recent_short_pdf")
        _require_role("今回成績 前回成績", "recent_detail_pdf")
        _require_role("1着 2着 3着 着外 H S", "hs_pdf")
        _require_role("2車単オッズ", "odds_pdf")
        with self.assertRaises(PdfInputError):
            _require_role("対戦成績", "recent_short_pdf")

    def test_basic_hs_and_dual_fixed_counts(self) -> None:
        riders = parse_basic_text(BASIC)
        hs = parse_hs_text(HS, list(range(1, 8)))
        for rider in riders:
            rider.update(hs[rider["bike"]])
            rider["recent_short_form"] = 0.0
            rider["recent_detail_form"] = 0.0
            rider["recent_form"] = 0.0

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

    def test_women_rejected(self) -> None:
        result = predict({"race_type": "WOMEN"})
        self.assertEqual(result["purchase_status"], "NO_BET")
        self.assertEqual(result["error"]["code"], "WOMEN_EXCLUDED")


if __name__ == "__main__":
    unittest.main()
