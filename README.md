# 章悟式∞競輪OS KEIRIN.JP 5PDF API

同じレースのKEIRIN.JP公式5PDFを照合し、男子競輪の2車単を2方式で独立計算します。

## 必須入力

1. 基本情報・並び予想PDF
2. 直近成績①PDF
3. 直近成績②PDF
4. 着度数・H・S回数PDF
5. 2車単オッズPDF

EXデータ画像は任意です。5PDFは開催場・日付・レース番号を照合し、別レース混在、欠損、締切後オッズ、女子競輪は `NO_BET` で停止します。

## 共通能力計算

基本情報の競走得点・決まり手・B・脚質・並び、着度数・H・S、2つの直近成績を使います。直近成績①と②は同率でまとめ、選手能力・役割・ライン・展開を作った後、固定シード10万回で全2車単の能力確率を計算します。

## しょーご式：固定5点

能力確率から算出した保守EV上位5点を返します。

## 市場残差：固定3点

KEIRIN.JPの全2車単オッズを市場確率へ変換し、次式で能力確率と混合します。

```text
市場確率 ×（能力確率 ÷ 市場確率）^ λ
```

正規化後、固定シード10万回で再計算し、保守EV上位3点を返します。

- λ=0: 市場確率のみ
- λ=0.5: 初期値
- λ=1: 能力確率のみ

2方式は混ぜません。共通買い目は表示だけです。

## 固定条件

- 入力元は全てKEIRIN.JP
- 男子のみ
- 券種は2車単
- しょーご式5点／残差3点
- 同じ入力とλなら同じ結果
- 結果・払戻・レース後データ・Web自動取得は予測に使わない
- 全組み合わせのオッズが揃わなければ停止

## 起動

```bash
docker compose -f docker-compose.individual.yml up --build
```

- Render/FastAPI: `POST /analyze`
- 単体サーバー: `POST /predict-files`
- 稼働確認: `GET /health`

主な出力は `strategies.shogo.candidates`、`strategies.residual.candidates`、`common_candidates`、`dual_pair_probabilities`、`pdf_audit` です。
