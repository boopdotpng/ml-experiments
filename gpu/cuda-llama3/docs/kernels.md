# Kernel guide

This guide follows the one fast path in `src/cuda_llama.cu`. There is no hidden
GEMM library: every launch is in the source.

## Precision contract

Storage is BF16: weights, residual stream, Q/K/V, MLP intermediates, logits,
and KV cache. Values round to BF16 at model-operation boundaries, like normal
BF16 inference.

The reductions that are sensitive to range or accumulated error are FP32:

- tensor-core GEMM accumulator fragments;
- RMS sum of squares and reciprocal scale;
- Q.K dot products;
- online-softmax `m` and `l`;
- attention value accumulators;
- decode matrix-vector dot products.

No integer or reduced-width format is used.

## Prefill tensor-core GEMM

The projection is `Y[M,N] = X[M,K] * W[N,K]^T`. The on-disk row-major weight
layout is also a column-major view of `W^T[K,N]`, so no repack is needed.

WMMA computes a 16×16×16 unit:

```text
A: 16×16 BF16 row-major
B: 16×16 BF16 column-major
C: 16×16 FP32 accumulator
```

For prompts over 128 tokens, eight warps form a 64×128 block tile. A warp owns
16 output columns and four 16-row accumulators. At each K step it loads one B
weight fragment and reuses it across four A fragments. This cuts weight reads
by four compared with independent 16-row tiles.

Short prompts use a 16×64 tile. The larger tile has better reuse but creates too
few blocks to fill 170 SMs; the small tile trades reuse for parallelism.

The accumulator is stored to shared FP32 memory once. Threads then perform the
visible epilogue: optional FP32 residual add followed by BF16 rounding. Q/K/V
and gate/up are logical concatenations, so each group needs one launch without
physically concatenating weights.

## Decode matrix-vector projection

One token cannot fill a tensor-core matrix tile. More importantly, decode must
read roughly 2.3 GiB of weights for every token, so it is bandwidth-bound.

Each warp produces four output rows. A lane loads one input value, reuses it
against four coalesced weight rows, keeps four FP32 sums, then uses shuffle
reductions. This gives memory-level parallelism without computing unused tile
rows.

There are three readable fused forms:

1. Q/K/V applies the RMS scale and norm weight while reading `x`, then writes K
   and V directly into the current cache position.
2. Gate/up shares normalized input loads, reduces both projections, and applies
   SiLU before writing one BF16 MLP vector.
3. O/down adds the residual in FP32 immediately after each row reduction.

## FlashAttention: the numerical idea

Normal attention needs:

```text
score_s = q·k_s / sqrt(d)
p_s     = exp(score_s) / sum_j exp(score_j)
out     = sum_s p_s v_s
```

Materializing all scores costs quadratic memory in prefill. Online softmax
streams them while maintaining a maximum `m`, scaled denominator `l`, and
scaled value accumulator `a`:

```text
new_m = max(m, score)
old   = exp(m     - new_m)
new   = exp(score - new_m)
a     = a*old + v*new
l     = l*old + new
m     = new_m
```

Subtracting a maximum prevents overflow. When a later score becomes the new
maximum, multiplying old state by `exp(old_m-new_m)` changes its reference
scale without revisiting or storing earlier scores. The final result is `a/l`.

### Prefill mapping

One warp owns one causal `(token, query_head)` row. Its lanes cooperate on the
64-value Q.K reduction; each lane keeps two value channels in FP32 registers.
Many token/head rows exist, so the GPU has abundant parallel work.

### Decode mapping and stripe merge

Decode has only 32 query heads. Assigning one warp per head leaves most of a
5090 idle and serializes the entire history. The fast kernel assigns one
eight-warp block per head. Warp `r` handles positions:

```text
r, r+8, r+16, ...
```

Each stripe produces a valid `(m_r, l_r, a_r)` state. The states combine using
the same rescaling identity:

```text
M = max_r(m_r)
L = sum_r l_r * exp(m_r - M)
A = sum_r a_r * exp(m_r - M)
out = A / L
```

This keeps exact softmax semantics (up to floating-point association), exposes
eight-way context parallelism, and still never writes a score array. It raised
512-context decode from about 238 to 374 tok/s in this implementation.

This is a simplified FlashAttention kernel: it specializes head dimension 64,
batch one, causal attention, and Llama's 4:1 grouped-query mapping. Those fixed
facts are why the source stays compact.

## KV cache and GQA

The BF16 cache is:

```text
[layer, position, kv_head, channel]
```

Llama 3.2 1B has 32 query heads and 8 KV heads. Query head `h` uses
`kv_head = h/4`. Decode projections write the new K and V directly into the
layer's current position. Prefill copies its contiguous K/V result into the
same layout once.

## RoPE correction

Llama's `rotate_half` pairs channel `i` with `i+head_dim/2`. For a 64-value
head the pairs are `(0,32), (1,33), ...`. The HIP teaching implementation used
adjacent pairs; the CUDA port corrects this and implements Llama 3.2's 32×
frequency scaling/interpolation.

## CUDA Graph decode

The current token, absolute position, generation step, and output tokens live
in device memory. Decode kernels read those pointers, and device argmax writes
the next token and advances state. Therefore the whole token step can be
captured once and replayed with `cudaGraphLaunch`—no per-kernel CPU dispatch and
no token copy to the CPU in the timed loop.

## Reading order

In `src/cuda_llama.cu`, read:

1. `tensorcore_gemm_kernel`
2. `qkv_decode_kernel`
3. `flash_prefill_kernel`
4. `flash_decode_kernel`
5. `gate_up_decode_kernel`
6. `Runner::prefill`, then `Runner::decode_body`

