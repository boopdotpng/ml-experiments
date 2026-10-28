# interp: tiny transformer interpretability tutorial

Small tinygrad transformers trained on toy tasks, built for a beginner tutorial (slides). The framing throughout: every component is "what can this write to the residual stream?"

- `readme.md`: the task definitions and headline results. Keep it in sync when a task or result changes.
- `tutorial.md`: the teaching outline. **The user writes the tutorial text themselves.** Only add or adjust headings there when asked; don't write prose into it.

## Tasks

Each task needs one more piece of the transformer than the one before (details and numbers in `readme.md`):

0. Hidden digit on a model with no layers and no position embedding (embed → unembed). It stays at chance, because the last token is always a letter and nothing moves between positions. Position embeddings are deliberately left out of task 0: with nothing reading other positions, they'd only add a constant bias. They're introduced later, once attention can compare positions.
1. Hidden digit (`qwdk7mzpa → 7`): one head, one layer, no position embedding. Reaches 100%.
2. Dictionary lookup (`a2f5c9d3 f → 5`): two layers, one head each.
3. Two-key sum mod 10 (`a2f5c9d3 fc → 4`): needs an MLP. Tops out around 98%; the user accepted that, so don't spend time chasing 100%.
4. Two-key difference mod 10 (`a2f5c9d3 fc → 6`): needs two heads per layer, or a third layer. One head per layer is seed-dependent: it either fetches only one value and stalls around 42%, or (4 of 9 seeds) learns a lopsided 1/3–2/3 blend of both values that the MLP untangles, reaching 89–97%. Two heads reach 100% on every seed tried (6 of 6).

## Files

Each `task<N>.py` trains its model (plus the shapes that fail, for comparison) and runs its own analysis, writing `plots/task<N>/` and `data/task<N>/`. Tasks 2–4 also save `checkpoints/task<N>.safetensors` (gitignored). Run them from this directory. Shared code lives in `lib/`:

- `lib/model.py`: `Model(mlp, seq_len, pos, causal, n_layers, n_heads)`. No layernorm and no biases. The MLP sits only after the last layer. There's no output projection: each head's V writes `emb_dim` numbers straight into the residual stream, and the heads' outputs are summed.
- `lib/dataset.py`: `hidden_digit_dataset()` (tasks 0 and 1), `lookup_dataset()` (task 2) and `two_key_dataset(op="sum"|"diff")` (tasks 3 and 4). Vocabulary: letters a–z are tokens 0–25, digits 0–9 are tokens 26–35.
- `lib/analysis.py`: shared by every task. Training loop (`train`), plot styling, CSV/PNG output (`Out`), the forward pass in numpy with every component's write to the residual stream kept separate (`forward`), the exact path expansion of the stream (`residual_paths`), the Q/K part split (`qk_parts`), and the full two-key analysis (`two_key_analysis`, used by tasks 3 and 4).
- `infer.py <checkpoint> <input>...`: works out the model shape from a task 2–4 checkpoint and prints the answers.

Tasks 0–4 are reproducible: rerunning a script regenerates byte-identical CSVs and PNGs. Keep it that way (seed before building each model, fixed numpy seeds in analysis).

## Plots and data

The user builds their own charts for the slides and doesn't embed matplotlib images. Every task script must therefore write **both**:

- `plots/task<N>/<name>.png`: a quick visual check.
- `data/task<N>/<name>.csv`: the raw numbers behind each plot, with a header row and one row per bar, point or cell. Add a `kind` column (letter/digit) wherever it helps color the chart.

Task 0 outputs (`data/task0/`):

| file | contents | what it shows |
|---|---|---|
| `loss.csv`, `loss_reference.csv` | training loss per step; ln 36 and ln 10 | the loss falls from ln 36 (even over all tokens) to ln 10 (even over the digits) and stays there |
| `prediction.csv` | average predicted probability per output token, with random weights and after training | after training: letters ≈ 0, every digit ≈ 0.1 |
| `table.csv` | P(digit given the last letter), 26×10 | no letter favors any digit |
| `moved_tokens.csv` | how far each token embedding moved from its starting value | only the letter embeddings moved; the digits never sit at the last position, so they get zero gradient and stay at their random start |
| `embedding.csv`, `unembedding.csv` | the trained weights: one row per token, 32 columns each | the whole model. the score for output token t given last letter q is unembedding[t] · embedding[q] |

Task 1 outputs (`data/task1/`):

| file | contents | what it shows |
|---|---|---|
| `loss.csv`, `loss_reference.csv` | training loss per step; ln 36 and ln 10 | the loss drops to about 0, with a short pause at ln 10 |
| `attention.csv` | attention from the last position over each position, for 8 test sequences | almost all of it lands on the digit |
| `qk.csv`, `qk_table.csv` | attention score before softmax for each key token, given a query letter (mean/min/max; full 26×36 table) | digits score high and letters score low, whatever the query letter |
| `ov.csv` | P(output digit given the head copies digit d), 10×10 | the identity: the head writes the digit it reads |
| `ov_copying.csv`, `ov_eigenvalues.csv` | copying score Σλ / Σ\|λ\| of the OV circuit (`copying_score` in lib/analysis.py), centered: trained, digit block only (uncentered), uncentered, and random weights (one seed, plus mean / sd over 20); every eigenvalue | the same finding without picking the digits by hand: +0.86 trained, +1.00 on the digits, −0.01 ± 0.10 random. Uncentered it's only +0.32: one big negative eigenvalue along "every token equally", which softmax ignores |
| `embedding.csv`, `unembedding.csv`, `w_q.csv`, `w_k.csv`, `w_v.csv` | the trained weights | QK = (E Wqᵀ)(E Wkᵀ)ᵀ / √32, OV = E Wvᵀ Uᵀ |
| `qk_space.csv` | every query and key vector, projected onto the average query direction and the biggest other key direction | every letter asks the same question (query cosine 0.996); digit keys point along it, letter keys away |
| `qk_rank.csv` | singular values of the 26×36 QK table | one direction explains 100.0%: the head asks one question, "is this a digit?" |
| `walkthrough.csv`, `walkthrough_logits.csv` | one test sequence step by step: score, attention, V per position; then the logits from the last letter's embedding vs the head's write | the head's write carries the whole answer; the embedding votes ≈ 0 |
| `ov_full.csv` | OV logits for all 36 source tokens | |
| residual-stream files | see below | |

Residual-stream files, written by every task from 1 to 4 (`analysis.residual_plots` and `analysis.residual_map`):

| file | contents |
|---|---|
| `residual_logits.csv` | direct logit attribution at the last position: what each component (token, position, each head, MLP) alone adds to the correct answer's logit, minus the average digit logit |
| `residual_example.csv` | one test sequence: each component's logit for every digit, and their sum (= the model's logits) |
| `write_size.csv` | length of each component's write at each position, averaged over the test set |
| `head_copies.csv`, `head_copies_test_average.csv` | for each head and position: the most-attended position and its token, the attention on it, and the cosine between the head's actual write and V of that one position (1.0 = a clean copy) |
| `residual_map.csv` | the stream of one example as an exact sum of paths ("token t from position j, through heads L1H1→L2H1, weight = product of attentions"), per stage and position; the MLP stays one opaque term. The paths sum back to the real stream (asserted) |

Task 2 outputs (`data/task2/`), plus the residual-stream files:

| file | contents | what it shows |
|---|---|---|
| `loss.csv`, `accuracy.csv` | loss per step and test accuracy for 5 shapes (long format, `model` column) | only 2 layers with position embeddings get past ~38% (100%) |
| `attention_layer1.csv` | average layer 1 attention, 9×9 | each value position looks one back, at its key (0.91–0.98) |
| `qk_layer1_split.csv` | layer 1 score split into the part from the key's position embedding and from the key's token | "one back" comes from position; the token part is nearly flat |
| `attention_layer2.csv` | layer 2 attention from the query position, 8 examples | it lands on the value after the matching key (0.965 average) |
| `qk_layer2_parts.csv`, `qk_layer2_margin.csv` | layer 2 score advantage of the matching value, split by (query part × key part); per-sequence margin | 13.6 of 14.0 comes from (query's own token) × (key letter that L1H1 wrote into the value position); the match wins in 100% of test sequences |
| `ov_layer2.csv` | P(output digit given layer 2 copies digit d), measured on the real stream | identity (0.993) |
| `ablation.csv` | test accuracy with one piece knocked out | |

Tasks 3 and 4 outputs (`data/task3/`, `data/task4/`; `analysis.two_key_analysis`), plus the residual-stream files. Positions are k1 v1 … k4 v4 q1 q2, a = value at q1, b = value at q2:

| file | contents | what it shows |
|---|---|---|
| `loss.csv`, `accuracy.csv` | per shape, as in task 2 | task 3: no MLP stalls at 27–30%; task 4: see above |
| `attention_layer1.csv`, `attention_layer2.csv` | layer 1 average per head; layer 2 from the last position, 8 examples per head | |
| `attention_split.csv` | each layer 2 head's attention on the value at q1, the value at q2, elsewhere | task 3: one head 0.47/0.47; task 4: L2H2 → a (0.99), L2H1 → b (0.94) |
| `qk_layer2_parts.csv` | score advantage split by (query part × key part), per head and target | q2's value is found by q2's own token; q1's value by the copy of q1's letter that L1H1 made at the last position |
| `probes.csv` | linear-probe accuracy for a, b and the answer at the last position, after each layer and the MLP | the answer is only linearly readable after the MLP. task 3: a and b ≈ 55% after layer 2 (an unordered blend); task 4: 100% each |
| `accuracy_grid.csv` | accuracy for every (a, b) | |
| `neurons.csv`, `neuron_summary.csv` | average MLP activation per (a, b) per neuron; r² explained by the answer, a, b, raw a±b | neurons respond to sets of a and b values, not to the answer (r² ≈ 0.05); the MLP is a distributed lookup table, not one neuron per answer |
| `ablation.csv` | knockouts, including single heads when a layer has two | task 4: L1H2 is nearly spare (95.1% without it) |

`data/task4/one_head/` (seed 0, stalls at 41.7%: layer 2 takes only b) and `data/task4/one_head_lucky/` (seed 1, 96.9%: a 0.30/0.66 blend) have the same files for the one-head-per-layer model.

For plot styling, follow the dataviz skill if it's available. The colors used so far: surface `#fcfcfb`, ink `#0b0b0b`, muted `#898781`, grid `#e1e0d9`, series blue `#2a78d6`.

## tinygrad notes (local checkout at ../tinygrad)

- Training mode is `Context(TRAINING=1)`; `Tensor.train()` no longer exists. The optimizer raises an error unless TRAINING is set. TRAINING also switches dropout and BatchNorm.
- Use `loss.realize(*opt.schedule_step())` inside a `@TinyJit` step. Sample the batch inside the step with `Tensor.randint`. Return tensors from jitted functions, never `.item()`, because the jit replays and returns whatever object it captured.
- Every parameter passed to the optimizer needs a gradient. Don't create modules that aren't used, such as an MLP that's switched off.
- The device is Metal by default. `DEV=CPU` selects the CPU (`CPU=1` is deprecated), which is about 8× slower here.
- A training step takes about 1–3 ms; that's kernel launch overhead, not compute. A full run takes 10–40 s.

## Conventions

- `Tensor.manual_seed(0)` at the top of every training script, for reproducible runs.
- Two-space indentation and compact tinygrad style, matching the existing files.
- Put throwaway experiments in `/tmp/<name>/`, not in this directory. Report the numbers to the user and only add a script here if they ask.
- Training runs show sudden phase changes: accuracy sits on a plateau (about 37–40%) for a random number of steps, then jumps. A single run that hasn't jumped yet doesn't mean the model can't learn the task. Try more steps or another seed before concluding that.
