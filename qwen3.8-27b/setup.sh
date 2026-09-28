#!/usr/bin/env bash
# fake Qwen3.8 27B checkpoint: real configs + index, and each shard is a sparse
# file holding the real safetensors header with zeroed tensor data. shapes/dtypes
# load fine but it takes ~1MB on disk instead of ~55GB.
set -euo pipefail
cd "$(dirname "$0")"

../.venv/bin/python - <<'PY'
import json, os, struct, time, urllib.request
from huggingface_hub import hf_hub_download, hf_hub_url

repo_id, revision = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
for name in ["config.json", "generation_config.json",
             "preprocessor_config.json", "video_preprocessor_config.json"]:
  hf_hub_download(repo_id, name, revision=revision, local_dir=".")
hf_hub_download(repo_id, "model.safetensors.index.json", revision=revision, local_dir="weights")

def fetch(url: str, start: int, end: int, tries: int = 5) -> tuple[bytes, int]:
  # without a timeout a stalled connection hangs forever
  for i in range(tries):
    try:
      req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
      with urllib.request.urlopen(req, timeout=30) as r:
        return r.read(), int(r.headers["Content-Range"].rsplit("/", 1)[1])
    except OSError as e:
      if i == tries - 1: raise
      print(f"  retry {i+1}/{tries-1}: {e}")
      time.sleep(2 ** i)

shards = sorted(set(json.load(open("weights/model.safetensors.index.json"))["weight_map"].values()))
for name in shards:
  path = os.path.join("weights", name)
  if os.path.exists(path):
    print(f"{name}: already exists, skipping")
    continue
  url = hf_hub_url(repo_id, name, revision=revision)
  head, total = fetch(url, 0, 7)
  n = struct.unpack("<Q", head)[0]
  header, _ = fetch(url, 8, 8 + n - 1)
  # write to a temp name so an interrupted run doesn't leave a half-written shard behind
  with open(path + ".tmp", "wb") as f:
    f.write(head + header)
    f.truncate(total)
  os.rename(path + ".tmp", path)
  print(f"{name}: {n} byte header, {total / 1e9:.2f} GB sparse")
PY

echo "fake checkpoint ready in $(pwd)/weights"
