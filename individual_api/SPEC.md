# 章悟式∞競輪OS Ver.1.1 個人評価型 2車単API（暫定固定）

## 固定入力

1. netkeirin出走表PDF
2. KEIRIN.JP「着度数・H・S回数」PDF
3. netkeirin 2車単オッズPDF

netkeirin EXデータスクリーンショットは任意。添付がなくても処理を続行し、`missing_optional` に記録する。並びはnetkeirin出走表から取得し、並び自体が読めない場合は推測せず `INPUT_ERROR / NO_BET` とする。

Web検索・結果データは本番予測に使用しない。読み取れない必須値は推測せず `INPUT_ERROR / NO_BET` とする。

## 固定処理順

1. 選手一人ずつ基礎能力を計算
2. 先頭・番手・3番手以降・単騎の役割別能力を計算
3. 相手内で標準化
4. 最後にライン長・同地区・バンク条件を補正
5. B主導権と4展開を確率化
6. 固定シードで10万回シミュレーション
7. 全2車単確率を確定
8. 確率確定後にのみオッズを使用してEVを計算

同一入力からは同じシード・同じ確率・同じ候補を返す。

## 購入候補条件

- FⅠ・GⅢのみ
- GⅠは `NO_BET`
- 8.0～30.0倍
- Wilson 90%下限による保守EV 1.10以上
- 推定確率1%以上
- B確信度38%以上
- 展開確信度35%以上
- 最大2点

APIは実際の投票を行わず候補だけを返す。

## 起動と呼び出し

必要環境はPython 3.10以上とNumPy。

```bash
python keirin_api_server.py
```

- `GET /health`: 稼働確認
- `POST /predict`: 正規化済みJSONから予測

最小入力形:

```json
{
  "grade": "F1",
  "source_files": {
    "racecard_pdf": "racecard.pdf",
    "hs_pdf": "hs.pdf",
    "odds_pdf": "odds.pdf"
  },
  "riders": [
    {
      "bike": 1,
      "name": "選手名",
      "region": "関東",
      "score": 101.2,
      "win_rate": 18.0,
      "escape": 3,
      "makuri": 5,
      "sashi": 1,
      "mark": 0,
      "H": 8,
      "B": 12
    }
  ],
  "lines": [[1, 2, 3], [4, 5], [6, 7]],
  "odds": [[null, 12.4], [8.9, null]],
  "conditions": {
    "bank_type": "400_outdoor",
    "wind_mps": 2.0,
    "temperature_c": 24.0
  }
}
```

例では構造を短く示すため選手・オッズを省略している。実入力では全出走選手と全2車単を入れる。

## ブラインド検証

- 30レース
- 候補19点
- 的中3点
- 的中オッズ: 8.9倍、12.8倍、18.9倍
- 投資換算: 1,900円
- 払戻換算: 4,060円
- 回収率: 213.7%
- 最大払戻依存率: 46.6%

グレード別ではFⅠ 361.7%、GⅢ 270.0%、GⅠ 0%。この結果に基づき、GⅠ購入を停止する。

30レースは採用可否を最終確定するには少ないため、この版は係数を変えずに追加ブラインド検証する暫定固定版とする。

## API境界

`keirin_individual_api.predict(payload)` は正規化済み入力を受け取る計算エンジン。PDF/OCRアダプターは3ファイルを同じ正規形へ変換し、必須値の読取可否を記録する。

`keirin_pdf_adapter.predict_from_files(racecard_pdf, hs_pdf, odds_pdf)` は固定3PDFを直接受け取る。開催場・日付・レース番号が3PDFで一致すること、全選手、H、並び、全2車単が揃うことを確認してから計算エンジンを実行する。

HTTPでは `POST /predict-files` に `multipart/form-data` で次のフィールドを送る。

- `racecard_pdf`: netkeirin出走表PDF
- `hs_pdf`: KEIRIN.JPの着度数・H・S回数PDF
- `odds_pdf`: netkeirin 2車単オッズPDF
- `ex_image`: 任意

```bash
curl -X POST http://127.0.0.1:8787/predict-files \
  -F "racecard_pdf=@racecard.pdf" \
  -F "hs_pdf=@hs.pdf" \
  -F "odds_pdf=@odds.pdf"
```

必要環境はPython 3.10以上、NumPy、pdfplumber、Popplerの `pdftotext`。

空ファイル、破損PDF、別レース混在、選手欠損、H欠損、並び読取失敗、2車単欠損のいずれかがあれば、予測や購入候補を出さず `INPUT_ERROR / NO_BET` を返す。

青森2026年7月23日10Rの実PDF3点で、開催照合、7選手、並び `5-1 / 2-4-6 / 3-7`、全42通りの2車単、10万回シミュレーションまで通し確認済み。同じ3PDFで同じ出力になることも確認済み。

`riders` は車番昇順とし、`odds` の行・列も同じ車番順とする。対角以外に欠損・非数・0以下のオッズがあれば、候補を出さず `INPUT_ERROR / NO_BET` とする。

主な出力:

- 個人・役割別能力値
- 各選手1着率・2着率
- B主導権候補と確信度
- 展開確率
- 全2車単確率
- 参考EV
- 購入候補または `NO_BET`
- 欠損・監査情報
