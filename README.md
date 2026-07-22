# 章悟式∞競輪OS Ver.1.1 PDF API版

netkeirinから保存した同一レースの2種類のPDFを使う、5〜9車対応の2車単研究APIです。

## 入力

1. 「基本情報」タブの出走表PDF
2. 2車単の人気順オッズPDF

EXデータを文字に変換できる場合は任意で追加できます。EXが空、部分的、列順違い、欠損行ありでも計算は継続します。`-` と `(0/0)` は0%ではなく未取得として扱います。

各PDFは指定された入力欄専用の解析器で検査します。別レースのPDF混入、2車単の組合せ不足、車番不一致は計算前に停止します。Web取得、スクリーンショット、結果・払戻は予測入力に使いません。

## 計算

- 競走得点、勝率・連対率、脚質、S/B、決まり手、ライン
- 任意EXのかまし、つっぱり、ちぎり、ちぎられ（取得できた選手だけ）
- ライン勝敗→同ライン／別線→1、2着順の二段階モンテカルロ
- 標準100,000回、seed 3156
- `EV = 2車単推定確率 × 現在オッズ`

直近成績PDFを使わないため、映像上の不利や「負けて強し」を見たことにはしません。

## 起動

```bash
docker compose up --build
```

ブラウザで `http://localhost:8001/`、API仕様は `http://localhost:8001/docs` を開きます。

## API

`POST /analyze` のmultipart項目:

- `basic_pdf`
- `odds_pdf`
- `ex_text`（任意）
- `pin`（環境変数 `SHOGO_ACCESS_PIN` を設定した場合）
- `monte_carlo_runs`（標準100000）
- `seed`（標準3156）

主な監査項目:

- `document_audit`
- `calculation.probability_sum_all_ordered_pairs`
- `calculation.odds_used_for_probability_estimation = false`
- `input_policy.results_or_payouts_used_for_prediction = false`

## 現在の状態

研究版です。5〜9車・同一レース2PDF・確率合計1.0・同一seed再現性を実ファイルで検査しますが、回収率の未使用期間検証が終わるまで実投資承認は常にfalseです。
