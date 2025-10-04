# MARC-ja データ生成 & 学習パイプライン

このプロジェクトでは、MARC-ja（日本語レビュー分類）データセットを対象に

1. データ分割（学習用サブセットの作成）
2. LLM を用いたレビュー生成（データ拡張）
3. 拡張データを使った BERT モデルのファインチューニング

を順番に実施できるスクリプト群を提供します。以下では仮想環境 `.venv` を前提としたターミナルでの実行例を紹介します。

## 0. 初期セットアップ

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1        # PowerShell の場合
# source .venv/bin/activate          # bash/zsh の場合

python -m pip install --upgrade pip
python -m pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

必要に応じて Hugging Face にログイン (`huggingface-cli login`)、`hf_transfer` や `bitsandbytes` なども導入してください。

---

## 1. データ分割（学習サブセットの作成）

`MARC-ja/make_sampling_data.py` は JGLUE/MARC-ja の学習データから一部をサンプリングし、JSONL として保存します。設定値は `MARC-ja/config.yaml` の `data.train_subset_*` にも記載されています。

```powershell
(.venv) PS C:\Users\rikut\OneDrive\ドキュメント\Research\sentiment_analysis>
python MARC-ja\make_sampling_data.py `
    --config MARC-ja\config.yaml `
    --ratio 0.10 `
    --seed 42 `
    --output data\marc_ja_train_subset.jsonl `
    --overwrite
```

- `--ratio` で元データから取り出す割合（0〜1]）を指定します。
- 生成したファイルを学習で使うには、`MARC-ja/config.yaml` の `data.train_subset_enabled` を `true` に更新してください。

---

## 2. 分割済みデータを使ってレビューを生成（データ拡張）

`MARC-ja/augment.py` は `MARC-ja/augmentation.yaml` の設定を読み込み、指定した LLM でレビューを生成します。生成した JSONL は `data/` 配下に保存されます。

```powershell
(.venv) PS C:\Users\rikut\OneDrive\ドキュメント\Research\sentiment_analysis>
python MARC-ja\augment.py `
    --config MARC-ja\augmentation.yaml `
    --prompt zero_shot
```

- `--prompt` を複数回指定すると対象プロンプトを絞り込めます（例: `--prompt zero_shot --prompt ja_to_en_translation`）。
- 出力先は `augmentation.yaml` の `output_path_template` / `output_path` に従います。
- Hugging Face モデルで量子化を使う際は、`augmentation.yaml` のコメントを参考に GPU が対応している dtype を選択してください。

---

## 3. 生成データを混ぜて BERT をファインチューニング

`MARC-ja/train.py` は `MARC-ja/config.yaml` の設定を元に BERT ベースの分類器を学習します。

1. `MARC-ja/config.yaml` で以下を必要に応じて更新します。
   - `data.train_subset_enabled: true` … 手順 1 で作成したサブセットを使用
   - `data.augmented_mix.enabled: true` … 手順 2 の生成データを混在させる場合
   - `data.augmented_mix.augmented_paths` … 生成された JSONL のパスに合わせて編集
2. 学習結果の保存先 (`training.output_dir`) やハイパーパラメータも必要に応じて調整します。

実行例:

```powershell
(.venv) PS C:\Users\rikut\OneDrive\ドキュメント\Research\sentiment_analysis>
python MARC-ja\train.py --config MARC-ja\config.yaml
```

- 学習中のログは標準出力に表示され、既定では `outputs/bert-marcja` にチェックポイントが保存されます。
- `training.report_to` が `wandb` になっている場合は、Weights & Biases へも自動でログが送られます。

---

## 参考: ベンチマークスクリプト

ローカル GPU 上で UnsLoTH モデルの推論スループットを測定する場合は `benchmarks/unsloth_benchmark.py` を利用できます。

```powershell
(.venv) PS>
python benchmarks\unsloth_benchmark.py \
    --families gemma3 qwen3 \
    --batch-sizes 1 2 4 8 \
    --output-json results\gemma_qwen3_benchmark.json
```

`--model` で任意の Hugging Face モデル ID を追加したり、`--prompts-file` で独自プロンプトを使うことも可能です。

---

## トラブルシューティング

- 4bit 量子化で NaN が発生する場合は、`quantization_compute_dtype` を `float16` に変更する、または一度フル精度で動作確認してください。
- FlashAttention 2 が原因で CUDA エラーが出る場合は、`--disable-flash-attn` オプションを付けて再実行できます。
- 追加の生成タスクやモデルを扱う際は、`augmentation.yaml` / `config.yaml` のコメントに従ってパスやハイパーパラメータを調整してください。

---

## ライセンスと利用上の注意

- JGLUE/MARC-ja データセット、Hugging Face 上の各種モデルのライセンスに従ってご利用ください。
- 生成したデータを再配布する場合も、元データ・モデルの利用規約を必ず確認してください。
