# Forward and backward, next to each other

Suppose the scalar training loss is `L`. In this project, `dX` is shorthand for
`∂L/∂X`. Its shape is the same as `X`: each entry tells you how sensitive the loss
is to that entry of `X`. An intermediate activation gradient tells earlier
operations what to do; a parameter gradient is eventually used by the optimizer.

For `y = f(x)`, backward receives `dy` from downstream and computes
`dx += J_f(x)ᵀ dy`. This is the chain rule written as a vector-Jacobian product.
The kernels compute that product directly, without constructing `J_f`.

## A linear layer is the best starting point

We store weights as `[output_features, input_features]`:

```text
Forward:                       Backward:

X [M,K] ─┐                     dY [M,N] ─┬─ × W      → dX [M,K]
         ├─ X Wᵀ → Y [M,N]              └─ transpose × X → dW [N,K]
W [N,K] ─┘

Y  = X Wᵀ                      dX += dY W
                               dW += dYᵀ X
```

Here `M = batch × sequence`. Forward needs one GEMM; backward needs two.
`dW[o,i] = Σm dY[m,o] × X[m,i]`: every token contributes to the weight gradient.
The matrices used by backward are the values from forward, **before** any update.

For the simplest scalar example, `x=2`, `w=3`, `y=xw=6`, and an upstream gradient
`dy=0.5` give `dx=dy*w=1.5` and `dw=dy*x=1`. The scalar example becomes the two
matrix products above when many features and tokens share the weights.

In the hardcoded HIP implementation, these products are explicit loops. For the
hidden-width projection, `tokens=8`, `input_dim=2048`, and `output_dim=2048`:

```cpp
// One thread computes dx[token, in].
float sum = 0.0f;
for (int out = 0; out < output_dim; ++out) {
  sum += dy[token * output_dim + out] * w[out * input_dim + in];
}
dx[token * input_dim + in] += sum;

// One thread computes dw[out, in].
float sum = 0.0f;
for (int token = 0; token < tokens; ++token) {
  sum += dy[token * output_dim + out] * x[token * input_dim + in];
}
dw[out * input_dim + in] += sum;
```

These are two separate kernels. A transpose in the equation changes which
indices a loop reads; no matrix needs to be physically transposed. The actual
functions are `linear_backward_input_hidden` and `linear_backward_weight_hidden`
in [`src/kernels.hip`](../src/kernels.hip).

## Complete local derivative map

`⊙` means elementwise multiplication. `r`, `s`, and `p` below are forward values.
The summation axes matter: attention softmax sums over keys, normalization takes
a mean over features, and parameter gradients sum over batch/token rows.

| Forward operation | Backward operation(s) | Forward values needed |
| --- | --- | --- |
| `x = E[ids]` | `dE[ids] += dx` (scatter-add) | IDs |
| `y = X Wᵀ` | `dX += dY W`; `dW += dYᵀ X` | X, W |
| `r = (mean(x²)+eps)^(-1/2)`; `y=w⊙x*r` | `dx += r*w⊙dy − x*r³*mean(dy⊙w⊙x)`; `dw += Σrows dy⊙x*r` | x, w, r |
| `y = R(θ)x` | `dx += R(θ)ᵀdy = R(-θ)dy` | Positions and frequencies |
| `S = Q Kᵀ/√D`, with causal mask | `dQ += dS K/√D`; `dK += dSᵀ Q/√D` | Q, K; reconstruct mask |
| `P = softmax(S)` | `dS += P⊙(dP − Σkeys P⊙dP)` | P |
| `O = P V` | `dP += dO Vᵀ`; `dV += Pᵀ dO` | P, V |
| `y = a+b` | `da += dy`; `db += dy` | None |
| `y = gate⊙sigmoid(gate)⊙up` | `dgate += dy⊙up⊙(s+gate⊙s⊙(1-s))`; `dup += dy⊙gate⊙s` | gate, up; recompute s |
| `loss[t] = logsumexp(z[t]) − z[t,target[t]]` | `dz[t] += dloss[t]*(softmax(z[t])−onehot(target[t]))` | Logits, targets |
| `L = mean(loss)` | Seed `dL=1`; `dloss[t] += 1/M` | M |

For GQA, each K or V head is shared by a group of query heads. The table's `dK`
and `dV` formulas must be **summed over those query heads**, not averaged. The
kernels perform this reduction without physically expanding K/V. Future keys
are excluded in the score and value primitives, so their gradients are zero.

## One transformer block, in reverse

The exact generated order reverses independent sibling operations as well; the
branch groupings below describe dependencies rather than a unique schedule.

| Forward, top to bottom | Backward, bottom to top |
| --- | --- |
| `a = RMSNorm(x)` | Combine the Q/K/V gradients into `da`; RMSNorm yields its scale gradient and an input contribution; add the attention skip contribution to `dx` |
| `Q=aWqᵀ`, `K=aWkᵀ`, `V=aWvᵀ` | Each projection computes a weight gradient and adds to the shared `da` |
| Rotate Q and K | Rotate dQ and dK by the transposed rotations |
| Scores, causal mask | Differentiate QKᵀ and sum shared-head dK |
| Softmax | Use the probability-weighted row reduction |
| Mix V | Compute dP and shared-head dV |
| Output projection | Compute dWo and dContext |
| `h = x + attention_output` | Fork `dh`: send it to `x` via the skip and to attention via the output projection |
| `a2 = RMSNorm(h)` | Compute the norm scale gradient and add its input gradient to `dh` |
| Gate and up projections | Compute dWgate/dWup and sum their contributions into `da2` |
| SwiGLU | Product rule yields dGate and dUp |
| Down projection | Compute dWdown and dSwiGLU |
| `y = h + down` | Fork incoming `dy`: send it directly to h and through the down projection |

The full model begins backward with mean loss, cross entropy, the tied output
projection, and final RMSNorm. It then traverses layers 15 through 0 (for 1B),
ending with embedding scatter-add.

## Why gradients accumulate

For `y=x+f(x)`, the input influences the result twice:

```text
Forward                                Backward

        ┌───────────────┐                      ┌── dy (skip) ───────┐
x ──────┤               + ─── y       dy ────┤                    + ─── dx
        └── f(x) ───────┘                      └── J_f(x)ᵀ dy ─────┘
```

`dx = dy + J_f(x)ᵀdy`. Overwriting instead of adding would lose one path.
The same idea explains why Q/K/V projection gradients add at their shared input,
why GQA gradients sum into shared heads, and why the token embedding receives
both lookup and output-projection gradients. In this implementation every
backward destination uses `+=`, with atomics for colliding embedding IDs.

At iteration boundaries, gradients are cleared. Accumulation **within a backward
pass** is required by the chain rule; accumulating **across batches/iterations**
is a separate training feature and is not enabled by this runner.

## The optimizer is a separate step

Backward computes `dW`. The demonstration optimizer then runs `W -= lr*dW`
once per unique parameter tensor. It cannot update a weight halfway through
backward, because earlier operations still need the old weight to calculate
their gradients. All weights remain fixed until every derivative is complete.
