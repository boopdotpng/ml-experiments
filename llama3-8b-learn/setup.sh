#!/usr/bin/env bash
# download the real Llama 3 8B Instruct BF16 checkpoint (~16GB) into this dir
set -euo pipefail
cd "$(dirname "$0")"

../.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
  repo_id="unsloth/llama-3-8b-Instruct",
  revision="f3710969eb766fb49d4d1ed3aeabcb03390772bd",
  local_dir=".",
  allow_patterns=[
    "*.safetensors", "model.safetensors.index.json",
    "config.json", "generation_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "LICENSE",
  ],
  max_workers=4,
)
PY

echo "checkpoint ready in $(pwd)"
