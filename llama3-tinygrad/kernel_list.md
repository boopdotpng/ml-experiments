## kernels required to run llama3-tinygrad fast

Model in `main.py` is Llama 3.2 1B:

- `emb_dim=2048`, `n_layers=16`, `n_heads=32`, `n_kv_heads=8`, `head_dim=64`, `mlp_size=8192`
- vocab = `128256`
- max context = `8192`
- target path here is BS=1 inference only
- no padding tokens, no padding masks, no packed batches, no multi-user batching
- checkpoint weights are all `bfloat16`
- KV cache is explicitly `bfloat16`
- token ids are integer tensors
- RoPE `COS`/`SIN` tables are tinygrad default float, normally `float32`
- all BF16 reductions/matmuls in tinygrad accumulate in FP32 automatically
- attention score matmul explicitly uses `dtype=dtypes.float32`, which keeps the score output FP32 for softmax

I ran:

```sh
BEAM=2 DEBUG=2 /Users/boop/code/ml/.venv/bin/python main.py --prompt hi --max-new-tokens 1
```

For decode, tinygrad launches:

- 1 host-to-device copy of the next token id, 4 bytes
- 216 METAL kernels inside the decode JIT

So the current tinygrad path is effectively **217 launches per generated token**, or **216 compute/JIT kernels** if the 4-byte token upload is excluded.

This is not the kernel list to copy for Blackhole. It is tinygrad's current BEAM=2 decomposition. The useful porting boundary is source-level dataflow: fuse elementwise/reduction prep into matmul and attention kernels, avoid DRAM writes for intermediates, and treat KV cache traffic as the main unavoidable decode bandwidth.

### dtype stability requirements

These are not optional if the port is meant to be numerically stable:

- Matmul/reduction accumulators must be FP32.
  - tinygrad does this automatically for BF16 through `sum_acc_dtype`: BF16 inputs are cast to FP32 before `sum`, then cast back to BF16 when no explicit output dtype is requested.
  - This covers Q/K/V/O projections, MLP projections, vocab projection, RMSNorm mean, and softmax sums.
- Attention scores and softmax should stay FP32.
  - `scores = q.matmul(k.transpose(-2, -1), dtype=dtypes.float32)` does not just request FP32 accumulation; accumulation was already FP32. It keeps the score tensor/output FP32 so scale, mask, exp, sum, reciprocal, and probability math run FP32.
- RoPE angle/table computation and application should be FP32.
  - Positions reach thousands; BF16 phase error becomes large enough to corrupt the rotation.
  - Current code gets FP32 from default-float RoPE tables and type promotion: BF16 Q/K times FP32 cos/sin produces FP32 rotation math.
- RMSNorm mean-of-squares and `rsqrt` should use FP32.
  - It is a 2048-wide reduction plus a small epsilon.
- Storage can remain BF16.
  - Weights, activations when written out, KV cache, and optionally stored cos/sin values do not need FP32 storage.

One current model quirk: because attention scores/probabilities are FP32, `attn @ v` produces FP32 and the residual stream can become FP32 from layer 0 onward. HF Llama usually stores the residual stream in BF16. For a Blackhole port, choose intentionally: BF16 residual storage for bandwidth/parity, FP32 on-chip residual for accuracy while it stays local.

### one-time setup / weight load

Not part of steady-state decode:

1. load 146 BF16 tensors from `model.safetensors`
2. copy weights to device
3. allocate per-layer KV cache lazily on first attention call:
   - shape `(2, 1, 8, 8192, 64)`
   - dtype `bfloat16`
   - about 16 MiB per layer
4. build RoPE tables:
   - `COS`, `SIN`: `(8192, 64)`
   - normally `float32`

### inference shape assumptions

Use these assumptions for the Blackhole kernel plan:

- `B=1` always
- decode is `S=1`
- prefill is `S=prompt_length`, still `B=1`
- no pad tokens
- no padding mask
- no dynamic batch compaction
- no beam search sampling; generation is greedy argmax in this code
- causal mask is only relevant for prefill when `S > 1`

### full model order

1. embedding lookup
2. repeat 16 transformer blocks
3. final RMSNorm
4. tied vocab projection with `embed_tokens.weight.T`
5. argmax over vocab

### embedding

Inputs:

- token ids: `(B, S)` integer
- embedding table: `(128256, 2048)` BF16

Output:

- hidden state `x`: `(B, S, 2048)`

For decode `B=1, S=1`, this is one row fetch: 2048 BF16 values.

### per block order

Each of the 16 blocks does:

1. input RMSNorm
2. attention
3. residual add
4. post-attention RMSNorm
5. MLP
6. residual add

### RMSNorm

Used twice per block plus once at the end.

Input/output shape for BS=1:

- prefill: `(1, S, 2048)`
- decode: `(1, 1, 2048)`

Weights:

- `(2048)` BF16

Math:

1. square each element
2. reduce sum/mean over last dim 2048
3. add `eps=1e-5`
4. reciprocal sqrt
5. multiply original `x`
6. multiply RMS weight

Dtype notes:

- tinygrad upcasts the mean-of-squares reduction to FP32 automatically
- for hand kernels, accumulate RMS sum in FP32 and do `rsqrt` in FP32
- output may be converted to BF16 for storage, but if immediately feeding a matmul, keep the normalized value in registers/SRAM

Good fusion targets:

- fuse input RMSNorm directly into Q/K/V projection reads
- fuse post-attention RMSNorm directly into gate/up projection reads
- fuse final RMSNorm directly into vocab projection reads

### attention

Input:

- normalized `x`: `(1, S, 2048)`

Projection weights:

- `q_proj`: `(2048, 2048)` BF16
- `k_proj`: `(512, 2048)` BF16
- `v_proj`: `(512, 2048)` BF16
- `o_proj`: `(2048, 2048)` BF16

Attention order:

1. Q projection: `x @ q_proj.T`, reshape to `(1, 32, S, 64)`
2. K projection: `x @ k_proj.T`, reshape to `(1, 8, S, 64)`
3. V projection: `x @ v_proj.T`, reshape to `(1, 8, S, 64)`
4. RoPE on Q
5. RoPE on K
6. cast K and V to BF16 and write KV cache
7. read K history `(1, 8, T, 64)`, where `T=start_pos+S`
8. read V history `(1, 8, T, 64)`
9. logical GQA repeat: 8 KV heads serve 32 Q heads, group size 4
10. scores `q @ k.T`, shape `(1, 32, S, T)`, FP32 accumulation and FP32 output
11. scale by `1/sqrt(64) = 0.125`
12. causal mask only when `S != 1`; decode has no mask
13. softmax over `T`
14. weighted value `attn @ v`, shape `(1, 32, S, 64)`
15. reshape to `(1, S, 2048)`
16. output projection `attn_out @ o_proj.T`
17. residual add

Minimum conceptual kernels:

1. fused RMSNorm + QKV projection
2. fused RoPE(Q,K) + K/V BF16 cache write
3. score + scale
4. softmax
5. value matmul
6. output projection + residual add

Better decode design:

- combine score, online softmax, and value accumulation into one streaming attention kernel per layer
- stream K/V from DRAM once
- keep only per-head running max, denominator, and weighted V accumulator
- keep those online-softmax state values and value accumulators FP32
- never materialize scores `(32, T)` or probabilities `(32, T)` in DRAM
- implement GQA by mapping `kv_head = q_head // 4`

### RoPE

Input shapes:

- Q: `(1, 32, S, 64)`
- K: `(1, 8, S, 64)`
- COS/SIN slice: `(S, 64)`

Math:

1. split head dim into halves of 32
2. `rotated = concat(-x2, x1)`
3. `out = x * cos + rotated * sin`

Dtypes:

- Q/K projection output from BF16 weights
- COS/SIN normally FP32
- rotation math should be FP32
- K is cast to BF16 for cache storage
- Q can be consumed as FP32 by score computation without a DRAM write

### KV cache

Per layer:

- K cache: `(1, 8, 8192, 64)` BF16
- V cache: `(1, 8, 8192, 64)` BF16
- source stores these together as `(2, 1, 8, 8192, 64)`

Decode per token per layer:

- write new K: `8*64` BF16 = 1 KiB
- write new V: `8*64` BF16 = 1 KiB
- read K history: `8*T*64` BF16
- read V history: `8*T*64` BF16

Across 16 layers at `T=8192`, K+V reads are about:

```text
16 * 2 * 8 * 8192 * 64 * 2 bytes = 256 MiB/token
```

That is the main unavoidable decode bandwidth unless the cache is compressed, paged differently, or kept on-chip for short contexts.

### softmax

Input:

- scores `(1, 32, S, T)`

Math:

1. reduce max over `T`
2. subtract max
3. exp
4. reduce sum over `T`
5. reciprocal
6. multiply

Dtype:

- scores are FP32 from explicit score matmul
- keep max, exp, sum, reciprocal, probabilities, and online-softmax state in FP32
- avoid storing probabilities

### MLP

Input:

- post-attention RMSNorm output `(1, S, 2048)`

Weights:

- `gate_proj`: `(8192, 2048)` BF16
- `up_proj`: `(8192, 2048)` BF16
- `down_proj`: `(2048, 8192)` BF16

Order:

1. gate projection: `(1, S, 2048) @ (2048, 8192) -> (1, S, 8192)`, FP32 accumulation
2. up projection: `(1, S, 2048) @ (2048, 8192) -> (1, S, 8192)`, FP32 accumulation
3. SiLU on gate: `gate * sigmoid(gate)`
4. elementwise multiply: `hidden = silu(gate) * up`
5. down projection: `(1, S, 8192) @ (8192, 2048) -> (1, S, 2048)`, FP32 accumulation
6. residual add

Good fusion targets:

- fuse post-attention RMSNorm into gate/up matmul input loads
- compute gate and up together if the kernel can stream both weight matrices
- fuse SiLU and `* up` before down projection
- for decode, keep the 8192-wide hidden in SRAM if possible; it is only 16 KiB in BF16 for one token
- fuse down projection output with residual add

### final norm + vocab projection + argmax

Final RMSNorm:

- same RMSNorm over `(1, S, 2048)`

Vocab projection:

- `x @ embed_tokens.weight.T`
- weight shape `(128256, 2048)` BF16
- output logits `(1, S, 128256)`
- FP32 accumulation is required

Sampling:

- only `logits[:, -1, :]`
- flatten
- argmax

For greedy decode, do not store logits to DRAM. Stream vocab tiles and maintain `(best_value, best_index)`.

### BEAM=2 tinygrad decode pattern

For `prompt="hi"` with BOS, the decode JIT ran at `start_pos=2`, so `T=3` for that replay. Tinygrad keeps `start_pos` symbolic, so later decode tokens reuse the JIT with larger `T`.

Observed structure:

- 1 token upload
- 216 METAL kernels inside decode JIT
- initial embedding kernel
- repeated block pattern for 16 layers
- final norm/projection/argmax kernels

The repeated source-level block pattern is:

1. RMSNorm / prep
2. QKV projection work
3. RoPE and KV write/read prep
4. score matmul
5. softmax reductions
6. value matmul
7. output projection
8. residual / prep for MLP
9. gate/up projections
10. SiLU and multiply
11. down projection
12. residual / next-layer prep

### can this be one megakernel?

Not one literal full-model kernel if that means all 16 layers, all weights, and KV resident on-chip. The weights are about 2.3 GiB and KV cache grows with context, so DRAM traffic is fundamental.

But yes, a decode megakernel/program per token is the right direction if it means:

- keep the 2048-wide hidden vector on chip across operations
- stream weights from DRAM for each projection
- fuse RMSNorm into matmul input handling
- fuse RoPE into Q/K production
- write only BF16 K/V cache, not intermediate Q/K/V tensors
- stream K/V cache once for attention with online softmax
- fuse output projection residual
- fuse MLP activation into down projection input
- stream vocab projection and argmax without writing logits

Suggested practical split for Blackhole:

1. one persistent decode program that loops over layers
2. inside each layer, separate optimized matmul/attention phases because weight tiles and KV cache streaming dominate
3. no DRAM writes for intermediate hidden, scores, probabilities, gate, up, or logits
4. only persistent DRAM writes during decode:
   - updated K/V cache
   - final next-token id

For prefill, use a different path. The causal `S x T` attention shape makes prefill a batched attention/matmul problem, while decode is a streaming vector problem.
