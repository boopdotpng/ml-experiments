# CUDA Llama 3.2 1B — readable, fast, and library-free

This is a handwritten CUDA port of `../hip-llama3` for an RTX 5090. It keeps
one fast inference path, uses no cuBLAS/CUTLASS/quantization, and leaves every
GPU kernel in [`src/cuda_llama.cu`](src/cuda_llama.cu).

The implementation is deliberately specialized for Llama 3.2 1B:

- BF16 weights, activations, logits, and KV cache.
- FP32 GEMM accumulators, RMS reductions, attention scores, online-softmax
  state, and value accumulators.
- Greedy batch-one generation.
- `head_dim=64` and dimensions divisible by 64.
- Correct Llama 3 `rotate_half` layout and scaled RoPE frequencies.
- One FlashAttention path; there is no materialized attention matrix or slow
  fallback.

Start with the [kernel guide](docs/kernels.md), then open the
[interactive optimization site](docs/index.html).

## Build

CUDA 13.2 and CMake 3.24+ are expected. The default target is Blackwell SM 120.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The build prints ptxas resource use. The measured build has no register spills.
To confirm tensor-core code generation:

```bash
cuobjdump --dump-sass build/cuda_llama | grep HMMA
```

You should see `HMMA.16816.F32.BF16`: BF16 inputs and FP32 accumulation.

## Model

The runner reuses the simple `.hllm` export from the HIP project:

```bash
../.venv/bin/python ../hip-llama3/scripts/export_llama.py \
  --checkpoint ../hip-llama3/checkpoints/unsloth-llama-3.2-1b-instruct \
  --out ../hip-llama3/models/llama3.2-1b-instruct.hllm
```

No model parser or inference library is linked into the executable.

## Run and benchmark

Run deterministic synthetic input directly:

```bash
./build/cuda_llama --prompt-length 512 --steps 129 --warmup 1 --runs 5
```

Or tokenize and decode text:

```bash
../.venv/bin/python scripts/run_prompt.py \
  --prompt "Explain FlashAttention in two sentences." --steps 48
```

Reproduce the benchmark table:

```bash
../.venv/bin/python scripts/benchmark.py
```

`steps=129` means the prompt produces the first sampled token and the timed
decode graph performs another 128 token iterations. The runner intentionally
does not stop a benchmark at EOS; that keeps the timed work fixed.

## RTX 5090 results

Measured on the local RTX 5090 (170 SMs, 32 GB), CUDA 13.2, driver 580.173.02.
Each number is the median of five runs after one warmup. CUDA events exclude
weight loading, allocation, and CUDA Graph construction.

| Prompt | Prefill tok/s | Decode tok/s | Decode context range |
|------:|--------------:|-------------:|---------------------:|
| 128   | 11,847 | 446 | 128–255 |
| 512   | 19,785 | 374 | 512–639 |
| 1024  | 20,630 | 308 | 1024–1151 |

Decode falls with context length because attention must read more cached K/V
and do more exponentials each token. The 2.3 GiB model-weight stream remains
the fixed cost.

## Correctness check

For the deterministic 16-token test prompt, Hugging Face BF16 greedy decoding
and this runner both produced:

```text
49220, 73367, 41427, 73367
```

This caught and fixed a subtle issue inherited from the educational HIP code:
Llama pairs RoPE channels across the two half-heads `(0,32), (1,33), ...`, not
as adjacent values `(0,1), (2,3), ...`.

## What is fused

- Q, K, and V share one prefill launch and one decode launch.
- Decode RMSNorm writes one FP32 scale; projections apply normalization while
  loading the input instead of materializing a normalized vector.
- Decode gate + up projections share input loads and apply SiLU immediately.
- Output/down projections add the residual in their epilogue.
- Decode argmax records the next token on device.
- A captured CUDA Graph replays the whole decode token without CPU launch or
  token-copy round trips.

There is intentionally no sampling, batching, paged cache, quantization, or
generic-model abstraction. Those would obscure the path being studied.

