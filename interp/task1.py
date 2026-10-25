# task 1: one layer, one head, no MLP, no position embedding, on the hidden digit task (qwdk7mzpa -> 7).
# the head finds the digit by its content (QK) and copies it to the last position (OV).
# no position embedding: "which token is the digit" doesn't depend on where it is
import os, math, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tinygrad import Tensor, nn, TinyJit, Context
from dataset import hidden_digit_dataset, VOCAB, N_LETTERS, decode
from model import Model, head_dim
from analysis import Out, forward, residual_plots, residual_map, SERIES
Tensor.manual_seed(0) # reproducible training runs

STEPS = 1000
xtrain, xtest, ytrain, ytest = hidden_digit_dataset()
model = Model(n_layers=1, n_heads=1, pos=False)
opt = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-3)

@TinyJit
def test_acc() -> Tensor: return (model(xtest).argmax(axis=-1) == ytest).mean() * 100

@TinyJit
@Context(TRAINING=1)
def train_step() -> Tensor:
  opt.zero_grad()
  samples = Tensor.randint(512, high=xtrain.shape[0])
  loss = model(xtrain[samples]).sparse_categorical_crossentropy(ytrain[samples]).backward()
  return loss.realize(*opt.schedule_step())

# plot styling
SURFACE, INK, MUTED, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "font.size": 10})
sequential = LinearSegmentedColormap.from_list("blue", ["#f0f6fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

# raw numbers behind every plot go to data/task1/*.csv, so slides can draw their own charts
def write_csv(name:str, header:list[str], rows) -> None:
  with open(f"data/task1/{name}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows([[f"{v:.6g}" if isinstance(v, float) else v for v in row] for row in rows])

def kind(t:int) -> str: return "letter" if t < N_LETTERS else "digit"
def softmax(x:np.ndarray) -> np.ndarray:
  e = np.exp(x - x.max(-1, keepdims=True))
  return e / e.sum(-1, keepdims=True)

if __name__ == "__main__":
  losses = []
  for i in range(STEPS):
    losses.append(train_step().item())
    if i % 500 == 499: print(f"loss and val acc at step {i+1}: {losses[-1]:.2f}, {test_acc().item():.2f}%")
  os.makedirs("plots/task1", exist_ok=True)
  os.makedirs("data/task1", exist_ok=True)

  # the whole model in numpy. with no position embedding, every quantity depends only on tokens
  attn = model.layers[0].attn
  E, U = model.emb.weight.numpy(), model.unembed.weight.numpy()                         # (36, 32) each
  Wq, Wk, Wv = attn.q_proj.weight.numpy(), attn.k_proj.weight.numpy(), attn.v_proj.weight.numpy()
  qk = (E @ Wq.T) @ (E @ Wk.T).T / head_dim ** 0.5    # (36 query tokens, 36 key tokens): attention score before softmax
  ov = (E @ Wv.T) @ U.T                               # (36 source tokens, 36 output tokens): logits a token writes when attended to

  # 1. loss: from ln 36 (random) to near 0. compare with task 0, which got stuck at ln 10
  fig, ax = plt.subplots(figsize=(7, 3.5))
  ax.plot(losses, color=BLUE, linewidth=2)
  for y, label in ((math.log(36), "even over all 36 tokens: ln 36 = 3.58"), (math.log(10), "even over the 10 digits: ln 10 = 2.30")):
    ax.axhline(y, color=MUTED, linewidth=1, linestyle="--")
    ax.text(STEPS, y, "  " + label, color=MUTED, va="center", ha="left")
  ax.set(title="1 layer, 1 head: training loss on the hidden digit task", xlabel="step", ylabel="loss")
  ax.grid(axis="y", color=GRID, linewidth=0.5)
  fig.savefig("plots/task1/loss.png", dpi=150, bbox_inches="tight")
  write_csv("loss", ["step", "loss"], [(i+1, l) for i, l in enumerate(losses)])
  write_csv("loss_reference", ["label", "loss"], [("even over all 36 tokens (ln 36)", math.log(36)), ("even over the 10 digits (ln 10)", math.log(10))])

  # 2. where the last position looks, on a few test sequences: all of it on the digit
  X = xtest.numpy()
  pattern = softmax(qk[X[:, -1:], X])           # (n_test, 9): last position's attention over every position
  n_show = 8
  fig, ax = plt.subplots(figsize=(6, 4))
  ax.imshow(pattern[:n_show], cmap=sequential, vmin=0, vmax=1, aspect="auto")
  for r in range(n_show):
    for c in range(X.shape[1]):
      ax.text(c, r, VOCAB[X[r, c]], ha="center", va="center", color=SURFACE if pattern[r, c] > 0.5 else INK, fontsize=11)
  ax.set_xticks(range(X.shape[1]))
  ax.set_yticks(range(n_show), [decode(x) for x in X[:n_show]], fontfamily="monospace")
  ax.set(title="attention from the last position", xlabel="position attended to")
  ax.spines[:].set_visible(False)
  fig.savefig("plots/task1/attention.png", dpi=150, bbox_inches="tight")
  write_csv("attention", ["example", "sequence", "position", "token", "kind", "attention"],
            [(r, decode(X[r]), c, VOCAB[X[r, c]], kind(X[r, c]), float(pattern[r, c])) for r in range(n_show) for c in range(X.shape[1])])
  on_digit = pattern[X >= N_LETTERS].mean()

  # 3. QK: which tokens does a letter at the last position look for? every query letter scores the digits highest
  scores = qk[:N_LETTERS]                             # (26 query letters, 36 key tokens)
  fig, ax = plt.subplots(figsize=(9, 3))
  ax.bar(range(len(VOCAB)), scores.mean(0), color=BLUE, width=0.7)
  ax.set_xticks(range(len(VOCAB)), list(VOCAB), fontsize=8)
  ax.axvline(N_LETTERS - 0.5, color=MUTED, linewidth=1, linestyle=":")
  ax.axhline(0, color=MUTED, linewidth=1)
  ax.grid(axis="y", color=GRID, linewidth=0.5)
  ax.set(title="QK: attention score for each key token (average over query letters)", xlabel="key token", ylabel="score before softmax")
  fig.savefig("plots/task1/qk.png", dpi=150, bbox_inches="tight")
  write_csv("qk", ["key_token", "kind", "score_mean", "score_min", "score_max"],
            [(t, kind(i), float(scores[:, i].mean()), float(scores[:, i].min()), float(scores[:, i].max())) for i, t in enumerate(VOCAB)])
  write_csv("qk_table", ["query_letter"] + [f"key_{t}" for t in VOCAB], [[VOCAB[q]] + [float(s) for s in scores[q]] for q in range(N_LETTERS)])

  # 4. OV: what does a digit write when the head copies it? its own identity. P(output | source digit) is the identity matrix
  copy = softmax(ov[N_LETTERS:])[:, N_LETTERS:]       # (10 source digits, 10 output digits), softmax over all 36 outputs
  fig, ax = plt.subplots(figsize=(5, 4.5))
  im = ax.imshow(copy, cmap=sequential, vmin=0, vmax=1)
  ax.set_xticks(range(10), list(VOCAB[N_LETTERS:]))
  ax.set_yticks(range(10), list(VOCAB[N_LETTERS:]))
  ax.set(title="OV: P(output digit | head copies digit)", xlabel="output digit", ylabel="digit attended to")
  ax.spines[:].set_visible(False)
  fig.colorbar(im, ax=ax, shrink=0.8, label="probability")
  fig.savefig("plots/task1/ov.png", dpi=150, bbox_inches="tight")
  write_csv("ov", ["source_digit"] + [f"p_digit_{d}" for d in VOCAB[N_LETTERS:]],
            [[VOCAB[N_LETTERS+i]] + [float(p) for p in row] for i, row in enumerate(copy)])

  # 5. the trained weights themselves, for anyone who wants to redo the QK and OV products
  for name, W in (("embedding", E), ("unembedding", U)):
    write_csv(name, ["token", "kind"] + [f"d{k}" for k in range(W.shape[1])],
              [[t, kind(i)] + [float(v) for v in W[i]] for i, t in enumerate(VOCAB)])
  for name, W in (("w_q", Wq), ("w_k", Wk), ("w_v", Wv)):
    write_csv(name, ["row"] + [f"d{k}" for k in range(W.shape[1])], [[r] + [float(v) for v in W[r]] for r in range(W.shape[0])])

  # 6. Q: what is the last position looking for? Q·K is a dot product in a 32-number head space, so put every query and key in one picture.
  # the x axis is the direction of the average query, so a key's score is (its x) times (the query's length) / √32.
  # every letter asks nearly the same question (the queries sit together), digit keys point along it, letter keys point away
  Q, K = E @ Wq.T, E @ Wk.T                          # (36, 32): query and key vector of every token
  qdir = Q[:N_LETTERS].mean(0); qdir /= np.linalg.norm(qdir)
  rest = K - np.outer(K @ qdir, qdir)                 # what's left of the keys after removing the query direction
  ydir = np.linalg.svd(rest, full_matrices=False)[2][0]
  fig, ax = plt.subplots(figsize=(7, 5))
  for M, pick, color, marker, label in ((K, range(N_LETTERS), MUTED, "o", "key of a letter"), (K, range(N_LETTERS, 36), BLUE, "o", "key of a digit"),
                                          (Q, range(N_LETTERS), SERIES[1], "^", "query of a letter (at the last position)")):
    pts = np.array([[M[t] @ qdir, M[t] @ ydir] for t in pick])
    ax.scatter(pts[:, 0], pts[:, 1], color=color, marker=marker, s=40, label=label, edgecolor=SURFACE, linewidth=1)
    for (x, y), t in zip(pts, pick): ax.annotate(VOCAB[t], (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7, color=MUTED)
  ax.axvline(0, color=GRID, linewidth=1); ax.axhline(0, color=GRID, linewidth=1)
  ax.annotate("", xy=(np.linalg.norm(Q[:N_LETTERS].mean(0)), 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=SERIES[1], lw=2))
  ax.set(title="Q and K in the head's space", xlabel="along the average query  (score ∝ how far right a key sits)", ylabel="biggest other direction of the keys")
  ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
  fig.savefig("plots/task1/qk_space.png", dpi=150, bbox_inches="tight")
  write_csv("qk_space", ["token", "kind", "role", "along_query", "other_direction"],
            [(VOCAB[t], kind(t), role, float(M[t] @ qdir), float(M[t] @ ydir)) for role, M in (("key", K), ("query", Q)) for t in range(36) if role == "key" or t < N_LETTERS])
  qn = Q[:N_LETTERS] / np.linalg.norm(Q[:N_LETTERS], axis=1, keepdims=True)
  query_cos = (qn @ qn.T)[~np.eye(N_LETTERS, dtype=bool)].mean()

  # 7. how many questions does the head ask? the singular values of the 26×36 QK table. if one dominates, the table is
  # (one number per query letter) × (one number per key token): a single question, "how digit-like is this key?"
  sv = np.linalg.svd(scores, compute_uv=False)
  share = sv**2 / (sv**2).sum()
  fig, ax = plt.subplots(figsize=(6, 3))
  ax.bar(range(1, 11), share[:10], color=[BLUE] + [MUTED]*9, width=0.7)
  ax.set_xticks(range(1, 11))
  ax.set(title="QK table: how much each independent 'question' explains", xlabel="singular direction", ylabel="share of the table", ylim=(0, 1))
  ax.grid(axis="y", color=GRID, linewidth=0.5)
  fig.savefig("plots/task1/qk_rank.png", dpi=150, bbox_inches="tight")
  write_csv("qk_rank", ["direction", "singular_value", "share"], [(i+1, float(v), float(sh)) for i, (v, sh) in enumerate(zip(sv, share))])

  # 8. one forward pass, step by step, at the last position of one test sequence:
  # Q of the last letter meets K of every position -> score -> softmax -> attention; V of every position, weighted by attention, is the write;
  # the stream at the end is (embedding of the last letter) + (the write), and the unembedding turns it into logits
  ex = 0
  x = X[ex]
  q_last = Q[x[-1]]
  score = K[x] @ q_last / head_dim ** 0.5
  att = softmax(score)
  V = E[x] @ Wv.T                                     # (9, 32): what each position would write
  write = att @ V
  from_emb, from_head = E[x[-1]] @ U.T, write @ U.T
  pos_names = [f"p{i+1}" for i in range(len(x) - 1)] + ["last"]
  fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1.6]))
  labels = [f"{p}\n{VOCAB[t]}" for p, t in zip(pos_names, x)]
  axes[0].bar(range(len(x)), score, color=[BLUE if t >= N_LETTERS else MUTED for t in x], width=0.7)
  axes[0].set_xticks(range(len(x)), labels, fontsize=8)
  axes[0].set(title=f"1. Q·K/√32: query '{VOCAB[x[-1]]}' vs each key", ylabel="score")
  axes[1].bar(range(len(x)), att, color=[BLUE if t >= N_LETTERS else MUTED for t in x], width=0.7)
  axes[1].set_xticks(range(len(x)), labels, fontsize=8)
  axes[1].set(title="2. softmax: attention", ylim=(0, 1))
  digits = list(range(N_LETTERS, 36))
  w = 0.27
  for j, (vals, color, label) in enumerate(((from_emb, MUTED, f"embedding of '{VOCAB[x[-1]]}'"), (from_head, BLUE, "head's write (V)"), (from_emb + from_head, SERIES[1], "sum = output"))):
    axes[2].bar(np.arange(10) + (j-1)*w, vals[digits] - vals[digits].mean(), width=w - 0.02, color=color, label=label)
  axes[2].set_xticks(range(10), list(VOCAB[N_LETTERS:]))
  axes[2].axhline(0, color=MUTED, linewidth=1)
  axes[2].set(title="3. the stream's vote for each digit", xlabel="output digit", ylabel="logit minus digit average")
  axes[2].legend(frameon=False, fontsize=8)
  for ax in axes: ax.grid(axis="y", color=GRID, linewidth=0.5)
  fig.suptitle(f"one forward pass: {decode(x)} → {VOCAB[ytest.numpy()[ex]]}", fontweight="bold")
  fig.tight_layout()
  fig.savefig("plots/task1/walkthrough.png", dpi=150, bbox_inches="tight")
  write_csv("walkthrough", ["position", "token", "kind", "score", "attention", "v_length", "weighted_v_length", "v_reads_as"],
            [(pos_names[i], VOCAB[t], kind(t), float(score[i]), float(att[i]), float(np.linalg.norm(V[i])), float(att[i]*np.linalg.norm(V[i])), VOCAB[int((V[i] @ U.T).argmax())])
             for i, t in enumerate(x)])
  write_csv("walkthrough_logits", ["output_token", "kind", "from_embedding", "from_head", "total"],
            [(VOCAB[t], kind(t), float(from_emb[t]), float(from_head[t]), float(from_emb[t] + from_head[t])) for t in range(36)])
  write_csv("ov_full", ["source_token", "kind"] + [f"logit_{t}" for t in VOCAB], [[VOCAB[i], kind(i)] + [float(v) for v in ov[i]] for i in range(36)])

  # 9. the residual stream: each component's vote, how much each writes per position, what the head copies, and the exact path map
  out = Out("task1")
  r = forward(model, X)
  residual_plots(out, model, r, X, ytest.numpy(), pos_names, "hidden digit")
  residual_map(out, model, r, X, pos_names, "hidden digit")

  print(f"test accuracy {test_acc().item():.2f}%, average attention on the digit {on_digit:.3f}")
  print(f"QK score, digits {scores[:, N_LETTERS:].mean():.2f} vs letters {scores[:, :N_LETTERS].mean():.2f}; OV diagonal {np.diag(copy).mean():.3f}")
  print(f"queries of different letters: average cosine {query_cos:.3f}; first QK direction explains {share[0]*100:.1f}% of the table")
