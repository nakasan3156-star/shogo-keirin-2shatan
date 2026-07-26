# 章悟式∞競輪OS 個人評価型PDF API

選手一人ずつの能力を先に評価し、役割適性、ライン補正、展開シナリオ、10万回シミュレーションの順で2車単確率とEVを算出するAPIです。

## 固定入力

1. netkeirin出走表PDF
2. KEIRIN.JP「着度数・H・S回数」PDF
3. netkeirin 2車単オッズPDF

EXデータ画像は任意です。Web検索やレース結果は予測に使用しません。

3PDFの開催場、日付、レース番号を照合し、選手、H/B、並び、全2車単オッズが揃ってから計算します。空ファイル、破損PDF、別レース混在、欠損があれば `INPUT_ERROR / NO_BET` で安全停止します。

## 固定購入条件

- FⅠ・GⅢのみ
- GⅠは `NO_BET`
- 8.0～30.0倍
- Wilson 90%下限による保守EV 1.10以上
- 推定確率1%以上
- 主導権確信度38%以上
- 展開確信度35%以上
- 最大2点

## 起動

```bash
docker compose -f docker-compose.individual.yml up --build
```

- 稼働確認: `GET http://localhost:8787/health`
- PDF予測: `POST http://localhost:8787/predict-files`

```bash
curl -X POST http://localhost:8787/predict-files \
  -F "racecard_pdf=@racecard.pdf" \
  -F "hs_pdf=@hs.pdf" \
  -F "odds_pdf=@odds.pdf"
```

## 出力

- 主導権候補と確信度
- 展開シナリオ確率
- 各選手の1着率・2着率
- 全2車単確率
- EV・保守EV
- 最大2点の購入候補
- PDF照合監査

## 検証

- 正規化エンジン17テスト合格
- HTTP multipart実PDFテスト合格
- 青森2026年7月23日10Rで、7選手、3ライン、全42通り、10万回処理を通し確認
- 同じPDFから同じ結果を返すことを確認
- 空PDF、破損PDF、別レース混在を `NO_BET` で停止することを確認

現行版: `1.1.2-individual-frozen`
