# Tinygrad Mixture-of-Depths-and-Experts LM

An interpretable 6.85M-parameter character language model trained on Tiny
Shakespeare. It implements the **staged MoDE** design from Raposo et al.,
*Mixture-of-Depths: Dynamically allocating compute in transformer-based language
models*.

## Architecture

```text
tokens [B,128]
  -> 8 decoder blocks, width 128, 8 attention heads
  -> blocks 1,3,5,7: full attention + top-2 MoE
  -> blocks 2,4,6,8: MoD top-k gather + attention + top-2 MoE + scatter
  -> tied character-language-model head (~65 characters)
```

Each MoE has eight packed SwiGLU experts with hidden size 256. Router IDs index
the leading expert axis directly on the GPU, following the repository's OLMoE
inference pattern. There is no Python expert loop or CPU routing decision.

Each MoD router scores all positions and gathers a fixed top-k capacity before
both attention and MoE. Unselected tokens take the identity residual around the
entire block. The selected updates are scattered into the full residual stream.

Training uses:

- fixed 50% MoD capacity and top-k routing;
- a Switch-style MoE aggregate load-balancing loss (`0.01`);
- a MoE router z-loss (`0.001`);
- the paper's MoD top-k membership predictor BCE (`0.01`);
- interleaved full blocks, as recommended by the paper.

The bounded `sigmoid(router_logit)` update gate is a small-model stabilization;
the paper's equation writes the router weight directly. Ranking is unchanged.

## Training

Run from this directory using the repository virtual environment:

```sh
../.venv/bin/python main.py train
```

The Tiny Shakespeare corpus downloads automatically to the ignored `data/`
directory. Checkpoints and tokenizer metadata go to `checkpoints/`; the default
run updates that checkpoint every 500 steps.

The training step is wrapped in `TinyJit` using the current tinygrad API:
`@Context(TRAINING=1)`. Batch-window sampling, MoD top-k, token gather/scatter,
MoE top-k, packed expert indexing, loss, backward, and AdamW all live inside the
captured Metal graph. The first call executes normally, the second captures, and
subsequent calls replay.

Training tensors have fixed shapes, so a symbolic variable is neither necessary
nor helpful there. The Llama inference code uses a bound `Variable` for
`start_pos` because the scalar changes while decode tensor shapes remain fixed.
Here MoD capacity itself depends on sequence length, so generation grows eagerly
until it reaches the fixed 128-character rolling window, then reuses one
fixed-shape `TinyJit`.

Useful overrides:

```sh
STEPS=5000 BS=16 LR=3e-4 MOD_CAPACITY=0.5 ../.venv/bin/python main.py train
DATA=/path/to/any.txt ../.venv/bin/python main.py train
```

## Compute budget

At batch 16 and 3,000 steps, the default configuration presents 6.14M training
tokens and is estimated at:

```text
448M forward FLOPs per 128-token sequence
21.5G training FLOPs per step (forward + approximately 2x backward)
64.6T training FLOPs for the run
```

This is about **73.7%** of the same eight-layer top-2 MoE with every block full,
or a 26.3% saving. Half the tokens skip half the blocks: linear projection and
expert work therefore falls to 75%, not 50%. Attention's quadratic matrix work
falls more aggressively. Setting routed capacity to 12.5%, as in the paper's
large-model optimum, brings the interleaved design much closer to half-compute.

The trainer calculates and prints this estimate for the actual configuration.
It intentionally omits normalization, activations, routing sorts, memory
movement, and optimizer operations; measured milliseconds/step remain the
hardware truth.

## Sampling and routing interpretation

```sh
../.venv/bin/python main.py sample $'ROMEO:\n'
../.venv/bin/python inspect_routing.py \
  $'ROMEO:\nBut, soft! what light through yonder window breaks?'
```

The exporter prints every character's realized depth and expert path, summarizes
expert utilization and MoD threshold agreement per layer, and writes the full
machine-readable trace to `routing_trace.json`. Set `ROUTE_MODE=threshold` to
inspect the causal predictor path rather than fixed-capacity training top-k.
