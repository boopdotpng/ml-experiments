# Kernel visualization

An interactive companion to [`docs/kernels.md`](../kernels.md), focused on the
decode-time attention path: the **KV cache** and the **FlashAttention-style**
kernel, and how these kernels actually map onto GPU threads.

The physical model here is AMD HIP, but the execution model (grid → blocks →
threads, `blockIdx`/`threadIdx`, `__global__` launches) is identical to CUDA.

## Open it

It is a single self-contained page — just open the file, no build or server:

```bash
xdg-open docs/visualization/index.html   # or double-click it
```

`three.min.js` is vendored next to it, so it works fully offline. (If your
browser blocks local files for some reason, serve the folder with
`python -m http.server` and visit the printed URL.)

## The four views

Deep-linkable via the URL hash (e.g. `index.html#flash`).

- **GPU threads** (`#gpu`) — one decode-attention launch is
  `<<<div_up(2048,256)=8 blocks, 256 threads>>>`. Each thread owns one output
  element; `idx → (head, channel)` via `d = idx%64`, `h = idx/64`. Because
  `head_dim=64` and a block is 256 threads, each block holds exactly 4 query
  heads, and since GQA `group=4`, all 4 share **one KV head = blockIdx.x**.
- **KV cache** (`#cache`) — the `cache[layer, position, kv_head, channel]`
  layout, one f32 slice appended per decode step, and the 32→8 grouped-query
  sharing that keeps it at `512` numbers per position.
- **Flash vs KV** (`#flash`) — steps through both decode kernels over the same
  scores. `kv_attention_kernel` does two passes (find max, then recompute +
  accumulate); `flash_decode_attention_kernel` does one streaming pass with the
  online-softmax rescale `exp(m_old - m_new)`. Watch `m`, `l`, `acc` update.
- **Decode pipeline** (`#pipeline`) — every `__global__` launch in one
  `decode_one` step, per layer, color-coded by what it touches.

All numbers are the real Llama 3.2 1B config: `dim=2048`, `n_heads=32`,
`n_kv_heads=8`, `head_dim=64`, `n_layers=16`.
