# HIP Llama 3.2 1B, Readable Edition

This repo is a deliberately simple HIP/C++ inference runner for
`unsloth/Llama-3.2-1B-Instruct`.

It is meant for reading kernels and understanding the pieces, not for winning
tokens/sec. The model weights are stored as bf16. Activations, logits, and the
KV cache are float32 so the code stays compact and easy to inspect.

For the kernels, start with [`docs/kernels.md`](docs/kernels.md), or open the
interactive [`docs/visualization/index.html`](docs/visualization/index.html) —
a self-contained page that animates how the KV-cache and FlashAttention decode
kernels map onto GPU threads.

## What Is Built

- `scripts/export_llama.py` converts the Hugging Face safetensors checkpoint to
  one dependency-free `.hllm` file.
- `src/hip_llama.hip` contains the HIP kernels and C++ host runner.
- `scripts/run_prompt.py` tokenizes an instruct prompt, runs the C++ binary, and
  decodes the returned token ids.
- The runner supports three modes:
  - `naive`: recompute the whole sequence and materialize attention scores.
  - `kv`: decode token by token while caching each layer's K/V tensors.
  - `flash`: same KV-cache decode path, but attention uses online softmax.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

If CMake cannot infer your GPU architecture, pass it explicitly, for example:

```bash
cmake -S . -B build -DCMAKE_HIP_ARCHITECTURES=native
```

## Export Weights

The checkpoint has already been downloaded into:

```text
checkpoints/unsloth-llama-3.2-1b-instruct
```

Export it with:

```bash
../.venv/bin/python scripts/export_llama.py \
  --checkpoint checkpoints/unsloth-llama-3.2-1b-instruct \
  --out models/llama3.2-1b-instruct.hllm
```

The exported file is just:

1. a small packed header,
2. a tensor table of fixed-width names plus offsets,
3. raw bf16 tensor payloads.

That keeps `src/hip_llama.hip` independent of Python, JSON, and safetensors.

## Run

Use the Python helper when you want text:

```bash
../.venv/bin/python scripts/run_prompt.py \
  --prompt "Explain KV cache in one short paragraph." \
  --mode kv \
  --steps 32
```

Or call the binary directly with token ids:

```bash
build/hip_llama \
  --model models/llama3.2-1b-instruct.hllm \
  --tokens 128000,9906 \
  --steps 8 \
  --mode flash
```

The binary prints `ALL_TOKENS`, which the Python helper decodes.

## Important Limitations

- Greedy sampling only.
- No tensor cores/MFMA usage; matvec is a plain one-block-per-row reduction.
- No fused RMSNorm/matmul/activation kernels.
- No paged attention.
- KV cache is float32, not bf16.
- `naive` mode is intentionally very slow for anything beyond small prompts.

Those are all good future experiments, but they would make the first reading
pass much less friendly.

