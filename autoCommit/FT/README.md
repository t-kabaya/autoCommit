# Fine-tuning Gemma 2B on CommitPackFT

このディレクトリには、CommitPackFTデータセットを使用してGemma 2Bモデルをファインチューニングするためのコードが含まれています。

## セットアップ

### 1. 依存関係のインストール

```bash
cd FT
pip install -r requirements.txt
```

### 2. Hugging Faceへのログイン

Gemmaモデルにアクセスするには、Hugging Faceのアカウントが必要です：

```bash
huggingface-cli login
```

Gemmaモデルのライセンスに同意してください：
https://huggingface.co/google/gemma-2-2b

## 使用方法

### データセットの確認

まず、データセットを確認します：

```bash
python prepare_dataset.py --num_samples 5
```

データセットのパターンを分析：

```bash
python prepare_dataset.py --analyze --subset_size 1000
```

小さなサブセットを作成：

```bash
python prepare_dataset.py --create_subset --subset_size 1000
```

### ファインチューニングの実行

#### 基本的な実行（1000サンプルでクイックテスト）

```bash
python finetune_gemma.py \
    --num_samples 1000 \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

#### フルトレーニング（10000サンプル以上）

```bash
python finetune_gemma.py \
    --num_samples 10000 \
    --num_epochs 5 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --output_dir ../models/gemma-2b-commitpack-full
```

#### カスタム設定

```bash
python finetune_gemma.py \
    --model_name google/gemma-2-2b \
    --output_dir ../models/my-model \
    --num_samples 5000 \
    --num_epochs 3 \
    --batch_size 2 \
    --learning_rate 1e-4
```

## 設定

`config.yaml`ファイルで設定をカスタマイズできます：

- **model.name**: 使用するベースモデル
- **training.num_samples**: トレーニングに使用するサンプル数
- **training.num_epochs**: エポック数
- **training.batch_size**: バッチサイズ
- **lora.r**: LoRAランク（高いほど高品質だがメモリ使用量が増加）

## 出力

ファインチューニング済みモデルは`../models/`ディレクトリに保存されます：

- `../models/gemma-2b-commitpack-ft/`: デフォルトの出力ディレクトリ
- モデルの重み（adapter_model.bin）
- トークナイザー設定
- トレーニングログ

## トレーニングの監視

TensorBoardでトレーニングを監視できます：

```bash
tensorboard --logdir ../models/gemma-2b-commitpack-ft/runs
```

## メモリ要件

- **最小**: 16GB VRAM（8-bit量子化、バッチサイズ2）
- **推奨**: 24GB VRAM（バッチサイズ4）
- **最適**: 40GB+ VRAM（バッチサイズ8+）

メモリが不足する場合：
- `--batch_size`を減らす（2または1）
- `gradient_accumulation_steps`を増やしてバッチサイズを補完
- より小さいLoRAランクを使用（r=8）

## ファインチューニング後のモデル使用

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ベースモデルをロード
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

# LoRAアダプタをロード
model = PeftModel.from_pretrained(base_model, "../models/gemma-2b-commitpack-ft")

# 推論
prompt = "### Instruction:\\nGenerate a commit message..."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

## トラブルシューティング

### CUDA out of memory

- バッチサイズを減らす
- `gradient_accumulation_steps`を増やす
- より少ないサンプルでトレーニング

### モデルのダウンロードエラー

- Hugging Faceにログインしているか確認
- Gemmaのライセンスに同意しているか確認

### データセットのロードエラー

- インターネット接続を確認
- キャッシュをクリア: `rm -rf ~/.cache/huggingface/datasets`
