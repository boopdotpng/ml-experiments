# Kernel Explainer

> **Interactive version:** [`docs/visualization/index.html`](visualization/index.html)
> is a self-contained Three.js page that animates the decode attention path —
> the GPU thread mapping, the KV cache layout, and flash vs two-pass online
> softmax. Open it in a browser (no build needed) alongside this document.

HIP is AMD's CUDA-like programming layer. The syntax here should feel familiar
if you have seen CUDA: `blockIdx`, `threadIdx`, `__global__` kernels, device
memory allocation, and explicit host launches are the same basic model.

## Llama Block

Each transformer layer does this:

```text
x
  -> RMSNorm
  -> Q/K/V projections
  -> RoPE on Q and K
  -> causal attention
  -> output projection
  -> residual add
  -> RMSNorm
  -> gate/up/down MLP with SiLU
  -> residual add
```

The weights are bf16, but each kernel converts bf16 words to float32 before
accumulating. The helper is:

```cpp
__device__ float bf16_to_float(uint16_t bits) {
  return __uint_as_float(uint32_t(bits) << 16);
}
```

That works because bf16 is literally the upper 16 bits of an IEEE float32.

## Embedding Gather

The first model operation is not a matrix multiply. It is a gather from the
token embedding table:

```text
x[position, channel] = embedding[token_id, channel]
```

In `embedding_kernel`, each HIP block owns one token position:

```cpp
int pos = blockIdx.x;
int token = tokens[pos];
```

The threads in that block copy the embedding row into the activation buffer:

```cpp
for (int i = threadIdx.x; i < dim; i += blockDim.x) {
  x[pos * dim + i] = bf16_to_float(emb[token * dim + i]);
}
```

For Llama 3.2 1B, `dim = 2048`, so a 256-thread block gives each thread about
eight channels to copy. The embedding table is shaped:

```text
model.embed_tokens.weight[vocab_size, hidden_size]
```

and this repo stores it as bf16. The kernel converts each bf16 value to float32
as it writes `x`, because the later educational kernels accumulate in float32.

This gather is also why the final logits projection can reuse the same tensor
when the model has tied embeddings:

```text
logits[token_id] = dot(final_hidden, embedding[token_id])
```

The checkpoint here has `tie_word_embeddings = true`, so there is no separate
`lm_head.weight`.

## Matrix-Vector Projection

`matvec_bf16_kernel` computes one output row per block:

```text
y[row] = dot(W[row, :], x[:])
```

The 256 threads in the block stride through the input columns, accumulate
partial sums, then reduce through shared memory. This is simple and readable.
A fast version would use tiling, vectorized loads, and likely MFMA/tensor-core
style instructions.

In this model, the same kernel handles:

- Q/K/V projections,
- attention output projection,
- MLP gate/up/down projections,
- final logits projection.

## RMSNorm

`rmsnorm_kernel` computes:

```text
scale = rsqrt(mean(x*x) + eps)
out[i] = x[i] * scale * weight[i]
```

One block handles one token position. Threads reduce the sum of squares through
shared memory, then loop over the hidden dimension to write normalized values.

## RoPE

`rope_kernel` rotates pairs of channels in each attention head:

```text
[a, b] -> [a*cos - b*sin, a*sin + b*cos]
```

The angle depends on absolute token position and channel pair. This is why the
decode path passes `pos` into the kernel before writing K into the cache.

The indexing is the part worth staring at:

```cpp
int pair = idx % half;
int h = (idx / half) % heads;
int pos = idx / (half * heads);
int abs_pos = pos + pos_offset;
int base = (pos * heads + h) * head_dim + pair * 2;
```

The Q and K tensors are laid out as:

```text
[position, head, channel]
```

RoPE treats adjacent channels as 2D vectors. If `head_dim = 64`, there are
`half = 32` channel pairs per head. Thread `idx` maps to exactly one
`(position, head, pair)` and rotates two floats in place.

The frequency for each pair is:

```cpp
inv_freq = exp(-log(theta) * (2 * pair / head_dim))
angle = abs_pos * inv_freq
```

Lower channel pairs rotate faster; higher pairs rotate slower. The important
operational detail is that RoPE must be applied before K is stored in the cache.
Cached K values are position-specific. Reusing unrotated K and trying to rotate
later would change the attention math you are trying to preserve.

In full-sequence `naive` mode, `pos_offset` is zero and `pos` comes from the
batch sequence index. In decode mode, there is only one token in the working
buffer, so local `pos` is zero and `pos_offset` carries the absolute generation
position.

## Naive Attention

`naive` mode is the easiest version to read:

1. `attention_scores_kernel` materializes every causal score:
   `scores[t, head, s] = dot(q[t, head], k[s, kv_head]) / sqrt(head_dim)`.
2. `softmax_rows_kernel` normalizes each row over previous positions.
3. `attention_apply_kernel` multiplies probabilities by V.

This makes the attention matrix real in memory. For sequence length `T`, that
matrix is `T * n_heads * T` floats. It is conceptually clean and increasingly
annoying as `T` grows.

## KV Cache

During autoregressive generation, token `t + 1` can attend to old K/V values
from tokens `0..t`, but those old K/V values do not change. The KV cache stores
them per layer:

```text
K_cache[layer, position, kv_head, head_dim]
V_cache[layer, position, kv_head, head_dim]
```

`kv` mode runs one token at a time. For each layer it:

1. computes this token's Q/K/V,
2. applies RoPE to Q and K using the absolute position,
3. writes K/V into the cache,
4. attends Q over cached K/V from positions `0..pos`.

This avoids recomputing Q/K/V for old tokens on every generated step. It does
not remove the need to look back over the previous context; it removes the
repeated transformer work for those previous tokens.

The host-side decode loop is `decode_one(token, pos, flash)`. For each layer it
does:

```text
RMSNorm current hidden
Q = q_proj(hidden)
K = k_proj(hidden)
V = v_proj(hidden)
apply RoPE to Q and K at absolute position pos
write K/V into cache[layer, pos]
attention(Q, cache K/V from 0..pos)
output projection + residual
MLP + residual
```

The cache tensors are allocated as flat float arrays:

```text
kc_[n_layers * max_seq * n_kv_heads * head_dim]
vc_[n_layers * max_seq * n_kv_heads * head_dim]
```

The logical indexing is:

```text
cache[layer, position, kv_head, channel]
```

and the flattened offset is:

```cpp
int kv_dim = n_kv_heads * head_dim;
int off = (layer * max_seq + pos) * kv_dim + i;
```

That is exactly what `store_kv_cache_kernel` writes. Notice that only K and V go
into the cache. Q is only needed for the current token.

Llama 3.2 1B uses grouped-query attention:

```text
n_heads = 32
n_kv_heads = 8
```

So four query heads share one KV head. The kernels map query head `h` to KV head
`kvh` like this:

```cpp
int group = n_heads / n_kv_heads;
int kvh = h / group;
```

The simple `kv_attention_kernel` computes one output element per thread:

```text
out[query_head, channel]
```

For that one output element, it loops over every previous source position twice:

1. first pass: compute all Q dot K scores and find the max,
2. second pass: exponentiate, sum the denominator, and accumulate weighted V.

That two-pass softmax is clear but wasteful. It recomputes Q dot K in the second
pass, and each output channel repeats the same score work. A serious kernel
would cooperate across threads so a head computes scores once and reuses them
for many V channels.

## FlashAttention-Style Decode

FlashAttention is about doing attention without materializing the whole score
matrix. The full algorithm tiles Q/K/V through fast memory and maintains an
online softmax. This repo's `flash` mode keeps only the online-softmax idea for
single-token decode:

```text
m = -inf
l = 0
acc = 0

for each source position:
  score = dot(q, k) / sqrt(head_dim)
  next_m = max(m, score)
  acc = acc * exp(m - next_m) + v * exp(score - next_m)
  l = l * exp(m - next_m) + exp(score - next_m)
  m = next_m

out = acc / l
```

The important bit is that the softmax denominator and weighted V sum are updated
streamingly. A production FlashAttention kernel does this cooperatively across a
block, with tiles in shared memory and much better reuse. Here, the kernel is
slow but makes the numerical trick visible.

The normal softmax attention formula is:

```text
score_s = dot(q, k_s) / sqrt(head_dim)
p_s = exp(score_s) / sum_j exp(score_j)
out = sum_s p_s * v_s
```

Naively, you first materialize all `score_s`, then normalize, then multiply by
V. Online softmax lets you stream one source position at a time while keeping
three running values:

```text
m   = max score seen so far
l   = softmax denominator, rescaled to m
acc = weighted V sum, rescaled to m
```

Each new score updates those values:

```cpp
float next_m = fmaxf(m, score);
float old_scale = expf(m - next_m);
float new_scale = expf(score - next_m);
acc = acc * old_scale + new_scale * val[d];
l = l * old_scale + new_scale;
m = next_m;
```

Why the rescaling? If a later score is larger than the previous max, all earlier
exponentials must be interpreted relative to the new max. Multiplying by
`exp(old_m - new_m)` moves the old denominator and accumulator into the new
scale without storing the old scores.

At the end:

```cpp
out[idx] = acc / l;
```

This is FlashAttention's numerical heart, but not its performance heart. Real
FlashAttention also tiles Q/K/V, keeps tiles in shared memory or registers,
uses cooperative reductions, and writes far less intermediate data to global
memory. This toy decode kernel still has each output channel independently
looping over the cache and recomputing the Q dot K score. That is slow,
but it makes the online-softmax mechanism very visible.

One useful way to compare the two decode kernels:

```text
kv_attention_kernel:
  pass 1 over cache: find max score
  pass 2 over cache: compute denominator and weighted V

flash_decode_attention_kernel:
  pass 1 over cache: update max, denominator, and weighted V together
```

So the flash-style kernel uses one streaming pass and does not need a scores
buffer. The KV cache is still present in both versions; FlashAttention changes
how attention reads that cache, not whether the cache exists.

## Are KV Cache And FlashAttention Connected?

They are complementary:

- KV cache is a stateful inference trick across generated tokens.
- FlashAttention is a memory-efficient attention algorithm inside one forward
  computation.

Modern LLM runtimes use both. For prefill, FlashAttention helps with the full
prompt attention. For decode, KV cache is essential, and FlashAttention-style or
fused attention kernels can still help with long contexts.
