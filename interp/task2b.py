# task 2b: dictionary lookup where the query key is sometimes missing (a2f5c9d3x -> -). a fork of task 2 to show softmax+1.
# when the key is there, layer 2 attends to its value as in task 2. when it isn't, softmax+1 could attend to nothing, but it doesn't
# (at most 0.02 on none, 5 of 5 seeds): the "-" has to come from somewhere, and attending to letters (often k1) gives the head
# a "not a digit" write that votes for it. attending to nothing writes nothing. so a sink appears whenever it's useful.
# trains the task 2 model twice from the same seed, once with each softmax, and compares where layer 2 looks
import numpy as np
from tinygrad import Tensor
from lib.dataset import lookup_missing_dataset, VOCAB, N_LETTERS, NONE, decode
from lib.model import Model
from lib.analysis import *
Tensor.manual_seed(0) # reproducible training runs

STEPS, LR = 6000, 1e-3
FINAL = "softmax+1"
SHAPES = {"softmax": dict(), "softmax+1": dict(softmax1=True)}
POSITIONS = ["k1", "v1", "k2", "v2", "k3", "v3", "k4", "v4", "query"]
KEYS = POSITIONS + ["none"]   # "none": what softmax+1 leaves unspent (always 0 for normal softmax)
xtrain, xtest, ytrain, ytest = lookup_missing_dataset()

# attention with a "none" column: 1 minus the row sum
def with_none(p:np.ndarray) -> np.ndarray: return np.concatenate([p, 1 - p.sum(-1, keepdims=True)], -1)

if __name__ == "__main__":
  out = Out("task2b")
  runs, models = {}, {}
  for name, shape in SHAPES.items():
    Tensor.manual_seed(0)
    models[name] = Model(seq_len=9, vocab_size=NONE+1, **shape)
    runs[name] = train(models[name], xtrain, ytrain, xtest, ytest, STEPS, LR, name)

  # 1. loss and accuracy: both get there; the split shows the missing-key rows aren't the hard part
  loss_plot(out, runs, "lookup with missing keys: training loss", FINAL)
  accuracy_plot(out, runs, "lookup with missing keys: test accuracy", FINAL)
  X, Y = xtest.numpy(), ytest.numpy()
  missing = Y == NONE
  rs = {name: forward(m, X) for name, m in models.items()}
  rows = []
  for name, r in rs.items():
    correct = r["logits"].argmax(-1) == Y
    rows += [(name, case, int(sel.sum()), float(correct[sel].mean() * 100)) for case, sel in (("key present", ~missing), ("key missing", missing))]
  out.csv("accuracy_split", ["model", "case", "examples", "test_accuracy"], rows)

  # 2. layer 2 attention from the query position, averaged separately over present and missing keys.
  # present: both put it on the value after the key. missing: both look at letters (softmax+1 mostly at k1), not at none
  cases = [(name, case, sel) for name in rs for case, sel in (("key present", ~missing), ("key missing", missing))]
  avg2 = np.stack([with_none(rs[name]["pattern"][1][sel, 0, -1]).mean(0) for name, case, sel in cases])
  fig, ax = plt.subplots(figsize=(7.5, 3))
  heatmap(ax, avg2, KEYS, [f"{name}, {case}" for name, case, _ in cases], annotate=True)
  ax.axvline(len(POSITIONS) - 0.5, color=SURFACE, linewidth=3)
  ax.set(title="layer 2: average attention from the query position", xlabel="position attended to")
  out.png(fig, "attention_layer2_average")
  out.csv("attention_layer2_average", ["model", "case", "key_position", "attention"],
          [(name, case, KEYS[j], float(avg2[i, j])) for i, (name, case, _) in enumerate(cases) for j in range(len(KEYS))])

  # 3. the same on single test sequences: 4 with the key present, 4 with it missing
  show = np.concatenate([np.nonzero(~missing)[0][:4], np.nonzero(missing)[0][:4]])
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
  rows = []
  for ax, (name, r) in zip(axes, rs.items()):
    p = with_none(r["pattern"][1][show, 0, -1])
    ax.imshow(p, cmap=sequential, vmin=0, vmax=1, aspect="auto")
    for i, e in enumerate(show):
      for c in range(len(KEYS)):
        ax.text(c, i, VOCAB[X[e, c]] if c < len(POSITIONS) else "∅", ha="center", va="center", color=SURFACE if p[i, c] > 0.5 else INK, fontsize=11)
      rows += [(name, int(e), decode(X[e]), decode([Y[e]]), KEYS[c], VOCAB[X[e, c]] if c < len(POSITIONS) else "", float(p[i, c])) for c in range(len(KEYS))]
    ax.axvline(len(POSITIONS) - 0.5, color=SURFACE, linewidth=3)
    ax.set_xticks(range(len(KEYS)), KEYS)
    ax.set_yticks(range(len(show)), [f"{decode(X[e])} → {decode([Y[e]])}" for e in show], fontfamily="monospace")
    ax.set(title=f"{name}: layer 2 attention from the query", xlabel="position attended to")
    ax.spines[:].set_visible(False)
  out.png(fig, "attention_layer2")
  out.csv("attention_layer2", ["model", "example", "sequence", "answer", "position", "token", "attention"], rows)

  # 4. layer 1 attention, averaged over the test set. the "one back" pattern is the same; softmax+1 also lets k1, which has
  # nothing before it, attend to nothing instead of to itself
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
  rows = []
  for ax, (name, r) in zip(axes, rs.items()):
    avg1 = with_none(r["pattern"][0][:, 0].mean(0))
    avg1[np.triu_indices(len(POSITIONS), 1)] = np.nan                            # the mask hides later positions
    heatmap(ax, avg1, KEYS, POSITIONS, annotate=True)
    ax.axvline(len(POSITIONS) - 0.5, color=SURFACE, linewidth=3)
    ax.set(title=f"{name}: layer 1 average attention", xlabel="position attended to (key)", ylabel="position attending (query)")
    rows += [(name, POSITIONS[i], KEYS[j], float(avg1[i, j])) for i in range(len(POSITIONS)) for j in range(len(KEYS)) if not np.isnan(avg1[i, j])]
  out.png(fig, "attention_layer1")
  out.csv("attention_layer1", ["model", "query_position", "key_position", "attention"], rows)

  # 5. who votes for "-" on the missing-key rows: each component's logit for "-" minus the average digit logit, at the query position
  comps = list(rs[FINAL]["parts"])
  dash = {}
  for name, r in rs.items():
    U = models[name].unembed.weight.numpy()
    logits = {c: r["parts"][c][missing, -1] @ U.T for c in comps}
    dash[name] = {c: l[:, NONE] - l[:, N_LETTERS:NONE].mean(-1) for c, l in logits.items()}
  fig, ax = plt.subplots(figsize=(7, 3.5))
  w = 0.38
  for k, (name, color) in enumerate(zip(rs, SERIES)):
    vals = [dash[name][c].mean() for c in comps]
    ax.bar(np.arange(len(comps)) + (k - 0.5) * w, vals, w, color=color, label=name)
  ax.axhline(0, color=MUTED, linewidth=1)
  ax.set_xticks(range(len(comps)), comps)
  ax.set(title='key missing: what each component adds to the "-" logit', ylabel='"-" logit minus average digit logit')
  ax.grid(axis="y", color=GRID, linewidth=0.5)
  ax.legend(frameon=False)
  out.png(fig, "dash_logits")
  out.csv("dash_logits", ["model", "component", "dash_logit_vs_digits"], [(name, c, float(dash[name][c].mean())) for name in rs for c in comps])

  for name, r in rs.items():
    none = 1 - r["pattern"][1][:, 0, -1].sum(-1)
    print(f"{name}: {runs[name][1]:.2f}%; layer 2 attention on none at the query: present {none[~missing].mean():.3f}, missing {none[missing].mean():.3f}")
