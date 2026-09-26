#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

../.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
  repo_id="allenai/OLMoE-1B-7B-0125-Instruct",
  local_dir=".",
  allow_patterns=[
    "*.safetensors", "model.safetensors.index.json",
    "config.json", "generation_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "README.md", "LICENSE*",
  ],
  max_workers=4,
)
PY

echo "checkpoint ready in $(pwd)"
