# 章悟式∞競輪OS Ver.1.0 PDF API版

netkeirinから保存した同一レースの3種類のPDFだけを使う、7車・9車対応の2車単研究APIです。

## 入力

1. 「基本情報」タブの出走表PDF
2. 「直近成績」タブの出走表PDF
3. 2車単の人気順オッズPDF

ファイル名は信用せず、PDF本文の固定語で種類を判定します。別レースのPDF混入、組合せ不足、車番不一致は計算前に停止します。Web取得、スクリーンショット、結果・払戻は予測入力に使いません。

## 計算

- 競走得点、勝率・連対率、脚質、S/B、決まり手、ライン
- 今節と直近1〜3開催の着順
- 直近成績とS/B・脚質による「負けて強し代理点」
- ライン勝敗→同ライン／別線→1、2着順の二段階モンテカルロ
- 標準100,000回、seed 3156
- `EV = 2車単推定確率 × 現在オッズ`

負けて強しはPDF数値だけによる代理評価です。映像上の接触、牽制、コース取りを見たことにはしません。

## 起動

```bash
docker compose up --build
```

ブラウザで `http://localhost:8001/`、API仕様は `http://localhost:8001/docs` を開きます。

## API

`POST /analyze` のmultipart項目:

- `basic_pdf`
- `recent_pdf`
- `odds_pdf`
- `pin`（環境変数 `SHOGO_ACCESS_PIN` を設定した場合）
- `monte_carlo_runs`（標準100000）
- `seed`（標準3156）

主な監査項目:

- `document_audit`
- `calculation.probability_sum_all_ordered_pairs`
- `calculation.odds_used_for_probability_estimation = false`
- `input_policy.results_or_payouts_used_for_prediction = false`

## 現在の状態

研究版です。7車42通り・9車72通り・同一レース3PDFの一括処理・確率合計1.0・同一seed再現性を実ファイルで検査しますが、回収率の未使用期間検証が終わるまで実投資承認は常にfalseです。
