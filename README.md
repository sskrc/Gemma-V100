# Gemma 4 V100 (12B & E4B — Tesla V100 `sm_70` 最適化版)

Tesla V100-PCIE-16GB (`sm_70`, Volta) 向けに `ggml-org/llama.cpp` (`32bdc3a`) へ専用の CUDA カーネル融合・量子化バッファ再利用・GPU Greedy Argmax 最適化を適用し、MTP (Multi-Token Prediction) 投機的デコーディング・Vision (`mmproj`)・ルーターモードを有効化した構成です。

- **パッチ単体ファイル**: `v100-gemma4.patch`（上流 `ggml-org/llama.cpp` に対する全変更差分）
- **ビルドコマンド**:
  ```bash
  cmake -B llama.cpp/build -S llama.cpp \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
  cmake --build llama.cpp/build --target llama-server llama-bench -j $(nproc)
  ```
- **起動方法**: `./serve.sh {start|stop|restart|unload|status|key}` で `0.0.0.0:8100` に llama.cpp の router モードを起動します(`models.ini`: `gemma4-12b`, `gemma4-e4b`、MTP 投機デコード + V100 CUDA カーネル融合最適化済み)。

- **実測速度 (Tesla V100 16GB)**:
  - **`gemma4-e4b`**: 生成 **184.48〜192.75 tok/s** (従来 154.87〜155.59 tok/s、**+23.9%**)、Prompt処理 **388.05 tok/s** (**+43.8%**)
  - **`gemma4-12b`**: 生成 **145.90〜156.32 tok/s** (従来 124.90〜129.36 tok/s、**+20.8%**)、Prompt処理 **364.38 tok/s** (**+15.8%**)
- **APIキー**: llama.cpp 標準の `--api-key-file` を使用。キーは `Gemma4/.api_key` (`.api-key` シンボリックリンク対応)。
  `Authorization: Bearer <key>`。無し・不正は `401`。
- **起動直後はモデル未ロード**(router プロセスのみ、VRAM 4 MiB)。最初のリクエストで該当モデルをロードします。
- **オフロード**: `./serve.sh unload` または `POST /models/unload` `{"model":"gemma4-12b"}`(ロードは `POST /models/load`、状態は `GET /models`)。

```bash
KEY=$(./serve.sh key)
curl http://HOST:8100/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-e4b","messages":[{"role":"user","content":"こんにちは"}],"max_tokens":100}'
```
OpenAI 互換: `GET /v1/models`, `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`。`"stream":true` でストリーミング。
- **モデル情報 (`GET /v1/models`)**: 各モデルに `meta.n_ctx`(1 スロットあたりの実効コンテキスト長)・`meta.n_ctx_train`・`architecture.input_modalities` が入ります。上流の router mode は読み込み中のモデルにしか `meta` を付けないため、未ロードのモデルでも preset の `ctx-size ÷ parallel` を同じキーで返すようにパッチしています(`tools/server/server-models.cpp`)。`context_length` は上流どおり `n_ctx_train` です。
- **Thinking**: 既定でオン。`"reasoning_effort":"none"` か `"chat_template_kwargs":{"enable_thinking":false}` でオフ。low/medium/high などの段階は効きません(オン/オフのみ)。
