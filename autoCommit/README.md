# gac - Git Auto Commit

ローカルLLMでgitコミットメッセージを自動生成するCLIツール。

**完全オフライン。外部API不要。プライバシー保護。**

## 特徴

- 🤖 ローカルLLM (llama.cpp + GGUF)
- 🎯 Conventional Commits形式
- 📝 diff/履歴を分析
- ⚡ 軽量・高速
- 🔒 完全プライベート

## インストール

```bash
# uvのインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# gacのインストール
cd autoCommit
uv venv
source .venv/bin/activate  # または: .venv/bin/activate (Windowsは .venv\Scripts\activate)
uv pip install -e .

# セットアップ（llama.cpp + Gemma 3 1B モデルをダウンロード）
gac install
```

## 使い方

```bash
# 基本
gac commit

# pushも同時に
gac commit --push

# 複数候補から選択
gac commit --interactive

# メッセージ生成のみ
gac commit --dry-run
```

**注意**: デフォルトで`git add .`と自動コミットが実行されます。

## 設定

`~/.gac/config.toml`

```toml
model = "~/.gac/models/gemma-3-1b-it-Q4_K_M.gguf"
llama_cli = "~/.gac/bin/llama-cli"
temperature = 0.2
max_tokens = 64
num_candidates = 3
```

## その他のコマンド

```bash
gac config   # 設定確認
gac install  # 再インストール
gac version  # バージョン確認
```

## モデル情報

- **デフォルト**: Gemma 3 1B IT (Q4_K_M)
- **サイズ**: ~700MB
- **速度**: 高速（M1/M2/M3/M4で最適化）

### カスタムモデル

```bash
# config.tomlを編集
model = "/path/to/your/model.gguf"
```

推奨モデル:
- Gemma 3 (1B, 2B) - 軽量
- Qwen2.5-Coder (1.5B, 3B) - コード特化
- Phi-3 Mini - 超小型

## トラブルシューティング

### インストール失敗

```bash
rm -rf ~/.gac
gac install
```

### 生成が遅い

`~/.gac/config.toml`を編集:
```toml
max_tokens = 32
temperature = 0.1
```

## 開発

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

## ライセンス

MIT

## 謝辞

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Gemma 3](https://huggingface.co/google/gemma-3-1b-it)
- [Typer](https://typer.tiangolo.com/)
