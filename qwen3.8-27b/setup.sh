#!/usr/bin/env bash
# fake Qwen3.8 27B checkpoint: real configs + index, and each shard is a sparse
# file holding the real safetensors header with zeroed tensor data. shapes/dtypes
# load fine but it takes ~1MB on disk instead of ~55GB.
set -euo pipefail
cd "$(dirname "$0")"

../.venv/bin/python - <<'PY'
import json, struct, urllib.request
from huggingface_hub import hf_hub_download, hf_hub_url

repo_id, revision = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
for name in ["config.json", "generation_config.json", "model.safetensors.index.json",
             "preprocessor_config.json", "video_preprocessor_config.json"]:
  hf_hub_download(repo_id, name, revision=revision, local_dir=".")

def fetch(url: str, start: int, end: int) -> tuple[bytes, int]:
  req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
  with urllib.request.urlopen(req) as r:
    return r.read(), int(r.headers["Content-Range"].rsplit("/", 1)[1])

shards = sorted(set(json.load(open("model.safetensors.index.json"))["weight_map"].values()))
for name in shards:
  url = hf_hub_url(repo_id, name, revision=revision)
  head, total = fetch(url, 0, 7)
  n = struct.unpack("<Q", head)[0]
  header, _ = fetch(url, 8, 8 + n - 1)
  with open(name, "wb") as f:
    f.write(head + header)
    f.truncate(total)
  print(f"{name}: {n} byte header, {total / 1e9:.2f} GB sparse")
PY

echo "fake checkpoint ready in $(pwd)"
