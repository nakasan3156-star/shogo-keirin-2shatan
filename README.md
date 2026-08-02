# 章悟式∞競輪OS KEIRIN.JP 6PDF API

同じレースのKEIRIN.JP公式6PDFを照合し、男子競輪の2車単を2方式で独立計算します。

## 必須入力

1. 基本情報PDF
2. 直近成績PDF
3. 対戦成績PDF
4. 当場成績PDF
5. 着度数・H・S回数PDF
6. 2車単オッズPDF

EXデータ画像は任意です。6PDFは開催場・日付・レース番号を照合し、別レース混在、欠損、締切後オッズ、女子競輪は `NO_BET` で停止します。

## しょーご式：固定5点

基本情報の競走得点・決まり手・B、着度数・H・S、直近成績、対戦成績、当場成績、ライン構成から選手能力・役割・展開を作り、固定シード10万回で2車単能力確率を計算します。能力確率から算出した保守EV上位5点を返します。

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
