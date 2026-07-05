#!/usr/bin/env bash
# Downloads the Llama 3.2 1B Instruct checkpoint (unsloth mirror, no auth needed)
# into llama3-tinygrad/, then links/copies it into hip-llama3/checkpoints/.
#
# The add-transformer and sort-transformer models are trained from scratch in
# seconds/minutes -- nothing to download for those, just run their main.py.
set -euo pipefail
cd "$(dirname "$0")"

REPO="unsloth/Llama-3.2-1B-Instruct"
BASE="https://huggingface.co/$REPO/resolve/main"
DEST="llama3-tinygrad"
FILES=(
  model.safetensors
  config.json
  generation_config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  chat_template.jinja
)

for f in "${FILES[@]}"; do
  if [ -f "$DEST/$f" ]; then
    echo "have $DEST/$f, skipping"
  else
    echo "downloading $f"
    curl -L --fail --progress-bar -o "$DEST/$f.tmp" "$BASE/$f"
    mv "$DEST/$f.tmp" "$DEST/$f"
  fi
done

# hip-llama3 expects the same checkpoint; hardlink (fall back to copy)
CKPT="hip-llama3/checkpoints/unsloth-llama-3.2-1b-instruct"
mkdir -p "$CKPT"
for f in "${FILES[@]}"; do
  [ -f "$CKPT/$f" ] || ln "$DEST/$f" "$CKPT/$f" 2>/dev/null || cp "$DEST/$f" "$CKPT/$f"
done

echo
echo "done. next steps:"
echo "  llama3-tinygrad: cd llama3-tinygrad && python main.py"
echo "  hip-llama3:      cd hip-llama3 && python scripts/export_llama.py  # writes models/*.hllm"
