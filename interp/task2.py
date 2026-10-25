# task 2: dictionary lookup (a2f5c9d3f -> 5) on two layers with one head each, plus position embeddings.
# layer 1 looks one position back and writes each key into the value token after it;
# layer 2 finds the value token marked with the query key and copies its digit.
# also trains the models that fail, for comparison: one layer (with 1 or 2 heads), and two layers without position information
import os
import numpy as np
from tinygrad import Tensor, nn
from lib.dataset import lookup_dataset, VOCAB, N_LETTERS, decode
from lib.model import Model, head_dim
from lib.analysis import *
Tensor.manual_seed(0) # reproducible training runs

STEPS, LR = 4000, 1e-3
FINAL = "2 layers, 1 head each"
SHAPES = {"1 layer, 1 head": dict(n_layers=1, n_heads=1), "1 layer, 2 heads": dict(n_layers=1, n_heads=2),
          FINAL: dict(n_layers=2, n_heads=1),
          "2 layers, no position embedding": dict(n_layers=2, n_heads=1, pos=False),
          "2 layers, no position embedding, no mask": dict(n_layers=2, n_heads=1, pos=False, causal=False)}
POSITIONS = ["k1", "v1", "k2", "v2", "k3", "v3", "k4", "v4", "query"]
xtrain, xtest, ytrain, ytest = lookup_dataset()

if __name__ == "__main__":
  out = Out("task2")
  runs, models = {}, {}
  for name, shape in SHAPES.items():
    Tensor.manual_seed(0)
    models[name] = Model(seq_len=9, **shape)
    runs[name] = train(models[name], xtrain, ytrain, xtest, ytest, STEPS, LR, name)
  model = models[FINAL]
  os.makedirs("checkpoints", exist_ok=True)
  nn.state.safe_save(nn.state.get_state_dict(model), "checkpoints/task2.safetensors")

  # 1. loss and accuracy for every shape: only two layers with position information get past ~38%
  loss_plot(out, runs, "dictionary lookup: training loss by model shape", FINAL)
  accuracy_plot(out, runs, "dictionary lookup: test accuracy by model shape", FINAL)

  X, Y = xtest.numpy(), ytest.numpy()
  r = forward(model, X)
  assert np.allclose(sum(r["parts"].values()), r["resid"][-1], atol=1e-4)
  pred = r["logits"].argmax(-1)

  # 2. layer 1 attention, averaged over the test set: every value position looks one back, at its key
  avg1 = r["pattern"][0][:, 0].mean(0)
  fig, ax = plt.subplots(figsize=(5.5, 4.5))
  heatmap(ax, avg1, POSITIONS, POSITIONS, annotate=True)
  ax.set(title="layer 1: average attention", xlabel="position attended to (key)", ylabel="position attending (query)")
  out.png(fig, "attention_layer1")
  out.csv("attention_layer1", ["query_position", "key_position", "attention"], [(POSITIONS[i], POSITIONS[j], float(avg1[i, j])) for i in range(9) for j in range(9)])

  # 3. where "one back" comes from. split each layer 1 score q·k into the part from the key's position embedding and the part
  # from the key's token: the position part has the one-back pattern, the token part is nearly flat. so layer 1 picks by place, not content
  P, attn1 = model.pos_emb.weight.numpy(), model.layers[0].attn
  q1 = r["resid"][0] @ attn1.q_proj.weight.numpy().T                              # (B, 9, head_dim)
  from_pos = (q1 @ (P @ attn1.k_proj.weight.numpy().T).T).mean(0) / head_dim ** 0.5
  from_tok = np.einsum("bid,bjd->bij", q1, model.emb.weight.numpy()[X] @ attn1.k_proj.weight.numpy().T).mean(0) / head_dim ** 0.5
  upper = np.triu_indices(9, 1)
  from_pos[upper] = from_tok[upper] = np.nan                                      # the mask hides later positions
  lim = np.nanmax(np.abs(np.stack([from_pos, from_tok])))
  fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
  for ax, M, name in zip(axes, (from_pos, from_tok), ("key's position", "key's token")):
    im = heatmap(ax, M, POSITIONS, POSITIONS, cmap=diverging, vmin=-lim, vmax=lim)
    ax.set(title=f"score from the {name}", xlabel="key position", ylabel="query position")
  fig.colorbar(im, ax=axes, shrink=0.8, label="layer 1 score before softmax (test average)")
  out.png(fig, "qk_layer1_split")
  out.csv("qk_layer1_split", ["query_position", "key_position", "score_from_key_position", "score_from_key_token"],
          [(POSITIONS[i], POSITIONS[j], float(from_pos[i, j]), float(from_tok[i, j])) for i in range(9) for j in range(i+1)])

  # 4. layer 2 attention from the query position, on a few test sequences: it lands on the value after the query key
  n_show = 8
  pattern2 = r["pattern"][1][:, 0, -1]                      # (B, 9)
  fig, ax = plt.subplots(figsize=(6, 4))
  ax.imshow(pattern2[:n_show], cmap=sequential, vmin=0, vmax=1, aspect="auto")
  for i in range(n_show):
    for c in range(9): ax.text(c, i, VOCAB[X[i, c]], ha="center", va="center", color=SURFACE if pattern2[i, c] > 0.5 else INK, fontsize=11)
  ax.set_xticks(range(9), POSITIONS)
  ax.set_yticks(range(n_show), [decode(x) for x in X[:n_show]], fontfamily="monospace")
  ax.set(title="layer 2: attention from the query position", xlabel="position attended to")
  ax.spines[:].set_visible(False)
  out.png(fig, "attention_layer2")
  out.csv("attention_layer2", ["example", "sequence", "position", "token", "kind", "attention"],
          [(i, decode(X[i]), POSITIONS[c], VOCAB[X[i, c]], kind(X[i, c]), float(pattern2[i, c])) for i in range(n_show) for c in range(9)])
  target = np.array([2*list(x[0:8:2]).index(x[-1]) + 1 for x in X])     # position of the answer's value
  on_target = pattern2[np.arange(len(X)), target].mean()

  # 5. layer 2 QK: why does the query position pick the right value? split query and key into the parts of the stream they're built from
  # and measure how much each pair favors the matching value over the other three. nearly all of it is one pair: the query's own token
  # against the key letter that L1H1 copied into the value position. layer 2 matches on something layer 1 wrote
  is_value = np.zeros((len(X), 9), dtype=bool); is_value[:, 1:8:2] = True
  names, M = qk_parts(model, r, 1, 0, -1, target, is_value)
  qk_parts_plot(out, names, M, "qk_layer2_parts", "layer 2: what makes the query pick the matching value?\n(score of the matching value minus the other values)")
  s2 = r["scores"][1][:, 0, -1]
  margin = s2[np.arange(len(X)), target] - np.where(is_value & (np.arange(9) != target[:, None]), s2, -np.inf).max(-1)
  out.csv("qk_layer2_margin", ["example", "sequence", "matching_value_score", "best_other_value_score", "margin"],
          [(i, decode(X[i]), float(s2[i, target[i]]), float(s2[i, target[i]] - margin[i]), float(margin[i])) for i in range(len(X))])

  # 6. layer 2 OV: what the head writes when it copies the value token holding digit d. measured on the real stream at value positions
  U = model.unembed.weight.numpy()
  ov = np.zeros((10, 10))
  for d in range(10):
    vp = np.nonzero(X[:, 1:8:2] == N_LETTERS + d)
    ov[d] = softmax(r["values"][1][vp[0], 0, 2*vp[1]+1] @ U.T).mean(0)[N_LETTERS:]
  fig, ax = plt.subplots(figsize=(5, 4.5))
  im = heatmap(ax, ov, list(VOCAB[N_LETTERS:]), list(VOCAB[N_LETTERS:]))
  fig.colorbar(im, ax=ax, shrink=0.8, label="probability")
  ax.set(title="layer 2 OV: P(output digit | head copies digit)", xlabel="output digit", ylabel="digit attended to")
  out.png(fig, "ov_layer2")
  out.csv("ov_layer2", ["source_digit"] + [f"p_digit_{d}" for d in VOCAB[N_LETTERS:]], [[VOCAB[N_LETTERS+i]] + [float(p) for p in row] for i, row in enumerate(ov)])

  # 7. residual stream: what each component adds to the correct answer's logit at the query position
  residual_plots(out, model, r, X, Y, POSITIONS, "dictionary lookup")
  residual_map(out, model, r, X, POSITIONS, "dictionary lookup")

  # 8. ablations: knock out one piece of the trained model and measure the test accuracy again
  ablations = {"full model": r, "layer 1 head removed": forward(model, X, drop_attn=(0,)),
               "layer 2 head removed": forward(model, X, drop_attn=(1,)), "position embedding removed": forward(model, X, pos=False)}
  ablation_plot(out, {n: (a["logits"].argmax(-1) == Y).mean() * 100 for n, a in ablations.items()}, "dictionary lookup: knock out one piece")

  print(f"test accuracy {runs[FINAL][1]:.2f}%; layer 2 attention on the right value {on_target:.3f}")
  print(f"layer 2 QK: matching value wins by {np.median(margin):.1f} (median), in {(margin > 0).mean()*100:.2f}% of test sequences; token x L1H1 term {M[0, 2]:.1f} of {M.sum():.1f}")
  print(f"layer 2 OV diagonal {np.diag(ov).mean():.3f}")
