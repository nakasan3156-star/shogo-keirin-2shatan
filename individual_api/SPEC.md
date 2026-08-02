# 章悟式∞競輪OS 2方式2車単API

## 固定入力

1. netkeirin 出走表PDF
2. KEIRIN.JP「着度数・H・S回数」PDF
3. KEIRIN.JP 2車単オッズPDF
4. EXデータスクリーンショットは任意

Web検索・結果データ・払戻・レース後情報は予測に使用しない。開催場、日付、レース番号、全選手、H/B、並び、全2車単オッズのいずれかが欠ければ `INPUT_ERROR / NO_BET` とする。

KEIRIN.JPオッズPDFは「○番車 全選択」の表を座標で解析し、ページをまたいでも1着車ごとの列を維持する。7車は42通り、9車は72通りがすべて揃わない限り計算しない。`○Rは締め切りました` を含む締切後PDFは `POST_RACE_SOURCE / NO_BET` で拒否する。

## 共通の能力予測

1. 選手一人ずつ基礎能力を計算
2. 先頭・番手・3番手以降・単騎の役割別能力を計算
3. ライン長・同地区・バンク条件を補正
4. 主導権と展開シナリオを確率化
5. 固定シードで10万回シミュレーション
6. 全2車単の能力確率を確定
7. 確率確定後にのみKEIRIN.JPオッズを結合

同一入力と同一λからは同じ結果を返す。

## しょーご式

- 能力確率を使用
- Wilson下限から保守EVを算出
- 保守EV上位5点を固定出力

## 市場残差システム

```text
市場確率 ×（能力確率 ÷ 市場確率）^ λ
```

- 市場確率はKEIRIN.JPの全2車単オッズの逆数を正規化
- λ初期値は0.50
- λ=0は市場のみ、λ=1は能力のみ
- 混合確率を正規化後、固定シードで10万回再計算
- 保守EV上位3点を固定出力

2方式は別々に計算し、共通買い目は表示だけに使用する。

## 対象外

- 女子競輪
- ガールズケイリン
- L級

対象外は `WOMEN_EXCLUDED / NO_BET` を返す。

## API

- `GET /health`: 稼働確認
- `POST /predict`: 正規化済みJSON
- `POST /predict-files`: 3PDFから予測
- Render版は `POST /analyze`

`predict-files` のフィールド:

- `racecard_pdf`: netkeirin出走表PDF
- `hs_pdf`: KEIRIN.JP H・S回数PDF
- `odds_pdf`: KEIRIN.JP 2車単オッズPDF
- `ex_image`: 任意

必要環境はPython 3.10以上、NumPy、pdfplumber、Popplerの `pdftotext`。
