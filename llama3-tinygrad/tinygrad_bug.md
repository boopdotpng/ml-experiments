Both questions answered now. Here's the full picture:

## The viz crash: not your fault, it's a tinygrad bug

Your `Variable("start_pos", ...).bind(...)` pattern is correct. The crash is in the viz profiler's event pairing:

1. **`timeline_layout` pairs kernels with metadata by timestamp order** (`viz/serve.py:232-240`): it tracks the most recent `"exec"` point event and uses its `var_vals` to evaluate symbolic FLOPS estimates for each kernel that follows.
2. **Metal graphs fabricate per-kernel timestamps** (`runtime/graph/metal.py:101-102`): Metal doesn't report per-kernel GPU times, so the batched entries get evenly-spaced synthetic timestamps spread across the *entire* GPU execution window (that's why every entry in your trace showed dur=92.4µs).
3. **The CPU races ahead of the GPU.** While the GPU is still inside the batched graph, your un-JITted sampling ops (the `argmax` kernel `r_2_32_4_501`, allocs) already dispatched — and their `exec` events, with **empty `var_vals`**, land timestamp-wise in the middle of the graph window. Symbolic kernels like `E_(start_pos+1)_8_4` then get paired with the wrong exec event, and `sym_infer(ops, {})` blows up with `_f() missing 1 required positional argument: 'start_pos'`. I replicated this against your saved trace: 23,222 mispaired events.

I patched your local clone (`tinygrad/viz/serve.py:239-245`) to skip the FLOPS/bandwidth annotation when the paired `var_vals` can't evaluate the estimate, instead of crashing — those numbers would have been wrong anyway since the pairing itself is bogus. Verified viz now boots and serves your exact trace (`*** ready in 657ms`, HTTP 200). The change is uncommitted in the gitignored clone; this is genuinely worth an upstream issue/PR since anyone profiling a symbolic-shaped JIT on Metal hits it.

## Can you pass a raw int as `start_pos`? — No

I checked `TinyJit.__call__`/`_prepare_jit_inputs` (`engine/jit.py:229-247`): the JIT only extracts **Tensors** and **bound Variable UOps** from the args. A plain Python int is invisible to it — at capture it gets baked into kernel shapes/indices as a constant, and every replay silently reuses the captured value with no error. So bound Variables are *the* mechanism for this, not a workaround.

## Your code vs. tinygrad's reference llama

Your dispatch is structurally identical to `extra/models/llama.py:221-225`:

```python
# reference — same shape check, same Variable bounds, same int fallback for prefill
if tokens.shape[0:2] == (1,1) and self.forward_jit is not None and start_pos != 0:
  return self.forward_jit(tokens, Variable("start_pos", 1, self.max_context-1).bind(start_pos), ...)
return self.forward(tokens, start_pos, ...)
```

So no, you're not doing anything wrong. Two optional refinements from the reference:

1. **KV cache: use `.assign()` instead of manual uop plumbing.** The reference does `self.cache_kv[:, :, start_pos:start_pos+seqlen].assign(...).realize()` (`llama.py:90`) where you hand-build the dependency with `Tensor(self.k_cache.uop.after(...store(k.uop)))` (`main.py:111-112`). Yours works, but `.assign()` is the supported idiom and expresses the same store+ordering.

2. **Move sampling inside the JIT.** The reference's `forward` samples on-device and returns the token, so only one int crosses per step. Your `logits.argmax().numpy()` runs as separate un-JITted kernels each token — slightly slower, and incidentally it's exactly those dispatches whose exec events were stealing the viz pairing. Fine to keep while you're still building (you noted greedy-for-determinism), but worth folding in later.