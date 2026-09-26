# Llama forward + backward in HIP

A complete, readable training pass for the **Llama 3.2 1B architecture**, with an
interactive forward/backward viewer. The small preset runs the same operators with
smaller dimensions. Both train from random initialization using next-token cross
entropy and SGD. All model math and parameter updates run in HIP; Python builds
the graph, generates launches, manages buffers, and supplies integer token data.

**Start with [the interactive 1B viewer](generated/1b/index.html)** or
[the small-model viewer](generated/tiny/index.html). These are self-contained HTML
files: open them in a browser. Click an operation for its forward and backward
HIP, use “Dependency graph” for branches, or “Backward execution list” for the
actual launch order. The arrow controls walk backward from the loss.

## Run

From this folder, with Python, NumPy, and `hipcc` available:

```bash
# Emit HIP source, graph, operation list, and viewer; no GPU allocation.
python train.py emit --preset 1b

# Compile HIP and train the small model. Uses the system HIP compiler.
python train.py train --preset tiny --steps 10

# Actual 1.236B parameter training; ~9.26 GiB of GPU tensor storage at B=1,T=8.
python train.py train --preset 1b --seq 8 --steps 1 --lr 0.001

# Explicit GPU target/device if needed on a system with several GPUs:
python train.py train --arch gfx1201 --device 0
```

An isolated `.venv` has been created locally. Use `.venv/bin/python` here to use
the installed dependencies. To set up another machine:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Without `--tokens`, the runner repeats a short synthetic token pattern. To use
your own tokenized batch, pass `--tokens batch.npy --batch B --seq T`. The array
must contain integers with shape `[B, T+1]`, in `[0, vocab_size)`. Each row is a
separate sequence; the runner uses `row[:-1]` as inputs and `row[1:]` as targets.
Every row/token contributes to the mean loss. Tokenization, padding masks,
ignore-label handling, checkpoint loading/saving, and a dataset loader are not
implemented. The default loop intentionally reuses the batch to demonstrate
learning; callers can use `Runner.set_batch` between iterations.

## Architecture and scope

“Llama 3 1B” here means **Llama 3.2 1B**, using the dimensions recorded in the
existing local Llama 3.2 checkpoint configuration. Model identifier:
[`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B).

| Property | Small test preset | 1B preset |
| --- | ---: | ---: |
| Transformer layers | 2 | 16 |
| Model width | 64 | 2048 |
| SwiGLU hidden width | 256 | 8192 |
| Query / KV heads | 4 / 1 | 32 / 8 |
| Head dimension | 16 | 64 |
| Vocabulary | 128 | 128256 |
| Trainable parameters, tied embedding counted once | 127296 | 1235814400 |
| Default batch / sequence | 2 / 8 | 1 / 8 |
| Forward / backward launches | 39 / 63 | 277 / 455 |

Both presets use pre-RMSNorm (`eps=1e-5`), bias-free Q/K/V/O projections,
causal grouped-query attention, scaled Llama-3 RoPE (`theta=500000`, factor 32,
original context 8192, frequency factors 1 and 4), two residual additions per
layer, SiLU-gated MLPs, final RMSNorm, and tied token embedding/output weights.
RoPE uses the Hugging Face half-split layout. Its scalar frequency/angle math is
computed in double precision to avoid accumulated phase error at long positions;
stored tensors and gradients are float32.

This is an educational implementation, with explicit backward operations and
all forward tensors retained. Dense projections use explicit FP32 dot-product
loops, with one GPU thread computing one output element in a 16×16 thread block;
attention materializes `[B,H,T,T]` scores and probabilities. It has no MFMA,
FlashAttention, BF16, activation checkpointing, Adam states, or distributed
training. Memory grows quadratically with sequence length. The memory estimate
includes values, gradients, saved inverse RMS values, and token arrays; runtime
and compiler overhead are additional. The runner checks available VRAM before
allocating. The configured maximum position is 131072, but this does not imply
that a 131072-token training sequence is practical here.

## What “generated” means

[`model.py`](model.py) creates a typed tensor graph. [`codegen.py`](codegen.py)
lowers each forward operation and its explicit vector-Jacobian product into
ordinary HIP launches. Dimensions are hardcoded as named integer constants
inside each kernel. Backward visits the graph in reverse topological
order. The derivative rules are explicitly implemented, rather than delegated to
PyTorch or a symbolic differentiation package.

The emitter produces these files for each preset:

| File | Contents |
| --- | --- |
| `generated/1b/forward.hip` | Actual ordered forward launches with tensor IDs |
| `generated/1b/backward.hip` | Actual ordered backward launches, starting with `dLoss=1` |
| `generated/1b/kernels.hip` | Plain HIP functions with fixed 1B dimensions; each forward is beside its backward |
| `generated/1b/training.hip` | Compilable translation unit plus SGD and isolated test dispatch |
| `generated/1b/graph.json` | Shapes, parameter flags, saved buffers, kernels, and launch dimensions |
| `generated/1b/backward-operations.md` | Every backward launch and the gradients it writes |
| `generated/1b/graph.dot` | Forward and backward dependency graphs, including parameter leaves |
| `generated/1b/index.html` | Standalone interactive viewer generated from the same manifest |

`hipcc` compiles these ordinary functions into a shared library. Layers with
identical shapes call the same function. Thus **launch count is not unique kernel
count**. Launch files use `v[id]` for
forward values and `g[id]` for gradients; names and shapes are in `graph.json`.
**Read [`src/kernels.hip`](src/kernels.hip)**: it contains the emitted 1B functions,
including the literal dimensions, and is identical to `generated/1b/kernels.hip`.
There are no C++ templates, transpose-mode flags, or generic GEMM helpers.
For example, `linear_backward_input_hidden` loops over output features, while
`linear_backward_weight_hidden` loops over tokens. Each formula is written out
directly beside its forward operation.

The math bodies live in [`src/kernel_bodies.hip.in`](src/kernel_bodies.hip.in).
[`kernel_codegen.py`](kernel_codegen.py) copies these bodies into ordinary
functions and inserts the fixed dimensions. This is only needed to produce the
separate small validation cases and regenerate the 1B source. You do not need to
read the emitter to understand the kernels. Changes to the math bodies should
be followed by `python train.py emit --preset 1b`; this refreshes the source file
and viewer. Emitting a small test preset does not change `src/kernels.hip`.
Libraries are cached by source, flags, and compiler
version; generated source is kept, while binaries and temporary builds are ignored.

HIP's `__global__` marks a GPU entry point, and `<<<grid, block>>>` chooses how
many threads to launch. Those are HIP syntax. C++ `template<...>` is optional:
the previous implementation used it to fix dimensions and select matrix
transpose modes at compile time. Named hardcoded constants keep the dimensions
known to the compiler without making the reader follow template instantiations.
The simple dot-product loops prioritize readability over the throughput of a
tiled matrix multiplication.

To inspect the compiler's AMD assembly as well, for this machine:

```bash
hipcc -O2 -std=c++17 --offload-arch=gfx1201 --cuda-device-only -S \
  generated/tiny/training.hip -o build/tiny-gfx1201.s
```

## How to read backward

Read the [walkthrough](docs/backward.md), then start with `linear_backward_input`
and `linear_backward_weight` in the viewer. `dX` always means `∂loss/∂X`.
Backward computes a vector-Jacobian product; it does **not** reconstruct a
previous activation or build an enormous Jacobian matrix.

At the start of each backward pass, all gradient buffers are zeroed and `dLoss`
is seeded with 1. Each kernel adds with `+=`. The shared embedding receives its
output-projection gradient first and its gather gradient last. Only the embedding
scatter requires atomics; other kernels assign a unique thread/block to each
destination element and explicitly reduce all contributions. Kernel launches
run in order on the default stream. All updates happen after backward completes.

## Validation

The tests execute the generated HIP kernels on the GPU. PyTorch is only the
independent CPU oracle used by the tests; training itself does not import it.

```bash
# Install the optional CPU oracle in the local environment.
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

.venv/bin/python tests/validate.py --arch gfx1201
.venv/bin/python tests/full_smoke.py --arch gfx1201
```

The numerical suite checks:

- Every primitive's forward and every input/weight derivative with independent
  random upstream gradients, including adding onto preexisting gradient buffers.
- Full-model values and gradients of every differentiable tensor against an
  independently assembled float64 PyTorch model.
- Central finite differences in a direction in every parameter tensor, compared
  to both the PyTorch gradient and the HIP gradient.
- Batch boundaries, repeated token IDs, tied embeddings, grouped KV reductions,
  causal masking, softmax row-gradient sums, RMSNorm near zero, large CE logits,
  non-tile-aligned dimensions, sequence length 1, and long-position RoPE.
- Gradient reset between iterations, the SGD update equation, and decreasing loss
  over two consecutive updates. Invalid token shapes/types/ranges are rejected.

The full-size test executes all 16 layers at the actual 1B dimensions, checks
every parameter gradient for finite, nonzero values, and checks that an SGD
update lowers loss. Full-size testing is a smoke test; the detailed numerical
comparisons use the smaller configurations. Reports are saved in
[`generated/validation.json`](generated/validation.json) and
[`generated/full-smoke.json`](generated/full-smoke.json).
