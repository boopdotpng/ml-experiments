# shared pieces for task0.py-task4.py: training loop, plot styling, csv output,
# and the model's forward pass in numpy so every intermediate (attention, residual stream, mlp neurons) can be read off
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tinygrad import Tensor, nn, TinyJit, Context
from .dataset import VOCAB, N_LETTERS
from .model import head_dim

# train with Adam on random batches of 512, return the loss per step and the final test accuracy
def train(model, xtrain:Tensor, ytrain:Tensor, xtest:Tensor, ytest:Tensor, steps:int, lr:float, name:str="") -> tuple[list[float], float]:
  opt = nn.optim.Adam(nn.state.get_parameters(model), lr=lr)
  @TinyJit
  def test_acc() -> Tensor: return (model(xtest).argmax(axis=-1) == ytest).mean() * 100
  @TinyJit
  @Context(TRAINING=1)
  def train_step() -> Tensor:
    opt.zero_grad()
    samples = Tensor.randint(512, high=xtrain.shape[0])
    loss = model(xtrain[samples]).sparse_categorical_crossentropy(ytrain[samples]).backward()
    return loss.realize(*opt.schedule_step())
  losses = []
  for i in range(steps):
    losses.append(train_step().item())
    if i % 1000 == 999: print(f"{name}: loss and val acc at step {i+1}: {losses[-1]:.2f}, {test_acc().item():.2f}%")
  return losses, test_acc().item()

# plot styling
SURFACE, INK, MUTED, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # categorical, always in this order
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "font.size": 10})
sequential = LinearSegmentedColormap.from_list("blue", ["#f0f6fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
diverging = LinearSegmentedColormap.from_list("blue_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f09a99", "#b52f2f"])

class Out:
  def __init__(self, task:str):
    self.task = task
    os.makedirs(f"plots/{task}", exist_ok=True)
    os.makedirs(f"data/{task}", exist_ok=True)
  # raw numbers behind every plot go to data/<task>/*.csv, so slides can draw their own charts
  def csv(self, name:str, header:list[str], rows) -> None:
    with open(f"data/{self.task}/{name}.csv", "w", newline="") as f:
      w = csv.writer(f)
      w.writerow(header)
      w.writerows([[f"{v:.6g}" if isinstance(v, (float, np.floating)) else v for v in row] for row in rows])
  def png(self, fig, name:str) -> None:
    fig.savefig(f"plots/{self.task}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def kind(t:int) -> str: return "letter" if t < N_LETTERS else "digit"
def softmax(x:np.ndarray) -> np.ndarray:
  e = np.exp(x - x.max(-1, keepdims=True))
  return e / e.sum(-1, keepdims=True)

# the whole forward pass in numpy, keeping every write to the residual stream separate.
# r["parts"] maps a component name ("token", "position", "L1H1", ..., "MLP") to its write, (B, T, emb_dim); they sum to the final stream.
# r["pattern"][l] / r["scores"][l]: attention after / before softmax in layer l, (B, heads, T, T). r["values"][l]: V per head, (B, heads, T, emb_dim).
# ablations: drop_attn lists layers (0, 1, ...) or single heads ("L1H2") whose attention write is zeroed, pos=False leaves out the position embedding, mlp=False skips the mlp
def forward(model, X:np.ndarray, drop_attn:tuple=(), pos:bool=True, mlp:bool=True) -> dict:
  B, T = X.shape
  x = model.emb.weight.numpy()[X]
  r = {"parts": {"token": x}, "resid": [], "scores": [], "pattern": [], "values": []}
  if pos and model.pos_emb is not None:
    r["parts"]["position"] = np.broadcast_to(model.pos_emb.weight.numpy()[:T], x.shape)
    x = x + r["parts"]["position"]
  for i, layer in enumerate(model.layers):
    r["resid"].append(x)
    a, H = layer.attn, layer.attn.n_heads
    q = (x @ a.q_proj.weight.numpy().T).reshape(B, T, H, head_dim).transpose(0, 2, 1, 3)
    k = (x @ a.k_proj.weight.numpy().T).reshape(B, T, H, head_dim).transpose(0, 2, 1, 3)
    v = (x @ a.v_proj.weight.numpy().T).reshape(B, T, H, -1).transpose(0, 2, 1, 3)
    scores = q @ k.transpose(0, 1, 3, 2) / head_dim ** 0.5
    if a.causal: scores = scores + np.triu(np.full((T, T), -np.inf), 1)
    pattern = softmax(scores)
    writes = pattern @ v                                                          # (B, H, T, emb_dim)
    for h in range(H): r["parts"][f"L{i+1}H{h+1}"] = writes[:, h] * (i not in drop_attn and f"L{i+1}H{h+1}" not in drop_attn)
    x = x + sum(r["parts"][f"L{i+1}H{h+1}"] for h in range(H))
    r["scores"].append(scores); r["pattern"].append(pattern); r["values"].append(v)
    if layer.mlp is not None:
      r["resid"].append(x)
      neurons = np.maximum(x @ layer.mlp.up.weight.numpy().T, 0)
      r["neurons"] = neurons[:, -1]                                               # (B, mlp_size) at the last position
      r["parts"]["MLP"] = neurons @ layer.mlp.down.weight.numpy().T * mlp
      x = x + r["parts"]["MLP"]
  r["resid"].append(x)
  r["logits"] = x[:, -1] @ model.unembed.weight.numpy().T
  return r

def head_names(r:dict) -> list[str]: return [n for n in r["parts"] if n.startswith("L")]

# direct logit attribution at the last position: what each component's write alone adds to the logit of the correct answer,
# measured against the average digit logit (a write that raises every digit equally doesn't help pick one)
def logit_attribution(model, r:dict, y:np.ndarray) -> dict:
  U = model.unembed.weight.numpy()
  res = {}
  for name, w in r["parts"].items():
    logits = w[:, -1] @ U.T                                                       # (B, 36)
    res[name] = logits[np.arange(len(y)), y] - logits[:, N_LETTERS:].mean(-1)
  return res

# what each head writes at each position, tracked back to where it came from. a head's write is
# sum_j attention_j * V(x_j); if the head attends to one position, that's a copy of V of that one position.
# "explained" is the cosine similarity between the actual write and V of the most attended position alone (1.0 = a clean copy)
def copy_table(r:dict, X:np.ndarray, example:int=None) -> list[tuple]:
  rows = []
  for l, (pattern, v) in enumerate(zip(r["pattern"], r["values"])):
    for h in range(pattern.shape[1]):
      w = r["parts"][f"L{l+1}H{h+1}"]
      src = pattern[:, h].argmax(-1)                                              # (B, T)
      top_v = np.take_along_axis(v[:, h], src[..., None], axis=1)                  # (B, T, emb_dim)
      cos = (w * top_v).sum(-1) / (np.linalg.norm(w, axis=-1) * np.linalg.norm(top_v, axis=-1) + 1e-9)
      weight = np.take_along_axis(pattern[:, h], src[..., None], axis=-1)[..., 0]
      for t in range(X.shape[1]):
        if example is None: rows.append((f"L{l+1}H{h+1}", t, float(weight[:, t].mean()), float(cos[:, t].mean())))
        else: rows.append((f"L{l+1}H{h+1}", t, VOCAB[X[example, t]], int(src[example, t]), VOCAB[X[example, src[example, t]]],
                           float(weight[example, t]), float(cos[example, t])))
  return rows

# controlled inputs for the two-key tasks: n random sequences "k1 v1 k2 v2 k3 v3 k4 v4 q1 q2" where the value at q1 is a and at q2 is b
def two_key_inputs(a:int, b:int, n:int, rng:np.random.Generator, p:int=4) -> np.ndarray:
  keys = rng.random((n, N_LETTERS)).argsort(axis=1)[:, :p]
  values = rng.integers(0, 10, (n, p))
  qi = rng.random((n, p)).argsort(axis=1)[:, :2]
  values[np.arange(n), qi[:, 0]], values[np.arange(n), qi[:, 1]] = a, b
  X = np.empty((n, 2*p + 2), dtype=np.int32)
  X[:, 0:2*p:2], X[:, 1:2*p:2] = keys, values + N_LETTERS
  X[:, -2:] = np.take_along_axis(keys, qi, axis=1)
  return X

def loss_plot(out:Out, runs:dict, title:str, final:str) -> None:
  import math
  fig, ax = plt.subplots(figsize=(7.5, 3.8))
  steps = max(len(l) for l, _ in runs.values())
  for (name, (losses, acc)), color in zip(runs.items(), SERIES):
    smooth = np.convolve(losses, np.ones(25)/25, mode="valid")    # 25-step running mean, raw values go to the csv
    ax.plot(np.arange(len(smooth)) + 25, smooth, color=color, linewidth=2.5 if name == final else 1.5, label=f"{name}: {acc:.1f}%")
  for y, label in ((math.log(36), "ln 36"), (math.log(10), "ln 10")):
    ax.axhline(y, color=MUTED, linewidth=1, linestyle="--")
    ax.text(steps, y, "  " + label, color=MUTED, va="center", ha="left")
  ax.set(title=title, xlabel="step", ylabel="loss (25-step mean)", ylim=(0, 3.8))
  ax.grid(axis="y", color=GRID, linewidth=0.5)
  ax.legend(frameon=False, title="model: test accuracy", alignment="left", loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
  out.png(fig, "loss")
  out.csv("loss", ["model", "step", "loss"], [(name, i+1, l) for name, (losses, _) in runs.items() for i, l in enumerate(losses)])
  out.csv("loss_reference", ["label", "loss"], [("even over all 36 tokens (ln 36)", math.log(36)), ("even over the 10 digits (ln 10)", math.log(10))])

def accuracy_plot(out:Out, runs:dict, title:str, final:str) -> None:
  names = list(runs)
  accs = [runs[n][1] for n in names]
  fig, ax = plt.subplots(figsize=(7, 0.55*len(names) + 1))
  ax.barh(range(len(names)), accs, color=[BLUE if n == final else MUTED for n in names], height=0.6)
  for i, a in enumerate(accs): ax.text(a + 1, i, f"{a:.1f}%", va="center", color=INK)
  ax.set_yticks(range(len(names)), names)
  ax.invert_yaxis()
  ax.set(title=title, xlabel="test accuracy (%)", xlim=(0, 110))
  ax.axvline(10, color=MUTED, linewidth=1, linestyle=":")
  ax.text(10, len(names) - 0.4, " chance", color=MUTED, va="bottom")
  ax.grid(axis="x", color=GRID, linewidth=0.5)
  out.png(fig, "accuracy")
  out.csv("accuracy", ["model", "test_accuracy", "is_final_model"], [(n, runs[n][1], n == final) for n in names])

def heatmap(ax, M:np.ndarray, xt:list, yt:list, cmap=sequential, vmin=0, vmax=1, annotate:bool=False, fmt:str="{:.2f}"):
  im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
  ax.set_xticks(range(len(xt)), xt)
  ax.set_yticks(range(len(yt)), yt)
  ax.spines[:].set_visible(False)
  if annotate:
    for r in range(M.shape[0]):
      for c in range(M.shape[1]):
        if np.isnan(M[r, c]): continue
        red, green, blue, _ = cmap((M[r, c]-vmin)/(vmax-vmin))
        ax.text(c, r, fmt.format(M[r, c]), ha="center", va="center", fontsize=7, color=INK if 0.3*red + 0.59*green + 0.11*blue > 0.55 else SURFACE)
  return im

# everything about the residual stream for one trained model:
#  residual_logits:   each component's average push toward the correct answer at the last position (direct logit attribution)
#  residual_example:  one worked example, each component's logit for every digit, and their sum (the model's actual logits)
#  write_size:        how big each component's write is at each position, averaged over the test set
#  head_copies:       for one example, which position and token each head copies at every position, and how cleanly
def residual_plots(out:Out, model, r:dict, X:np.ndarray, Y:np.ndarray, positions:list[str], title:str, example:int=0) -> None:
  from .dataset import decode
  names = list(r["parts"])
  dla = logit_attribution(model, r, Y)
  fig, ax = plt.subplots(figsize=(6.5, 0.5*len(names) + 1))
  means = [dla[n].mean() for n in names]
  ax.barh(range(len(names)), means, xerr=[dla[n].std() for n in names], color=BLUE, height=0.6, error_kw=dict(ecolor=MUTED, lw=1))
  ax.set_yticks(range(len(names)), names)
  ax.invert_yaxis()
  ax.axvline(0, color=MUTED, linewidth=1)
  ax.grid(axis="x", color=GRID, linewidth=0.5)
  ax.set(title=f"{title}: who pushes the right answer?", xlabel="logit of the correct digit minus the average digit logit\n(mean ± std over the test set)")
  out.png(fig, "residual_logits")
  out.csv("residual_logits", ["component", "mean", "std", "min", "max"], [(n, dla[n].mean(), dla[n].std(), dla[n].min(), dla[n].max()) for n in names])

  U = model.unembed.weight.numpy()
  rows = names + ["sum = model output"]
  L = np.stack([r["parts"][n][example, -1] @ U.T for n in names])[:, N_LETTERS:]
  L = np.concatenate([L, L.sum(0, keepdims=True)])
  L = L - L.mean(1, keepdims=True)
  fig, ax = plt.subplots(figsize=(7, 0.45*len(rows) + 1.2))
  lim = np.abs(L).max()
  im = heatmap(ax, L, list(VOCAB[N_LETTERS:]), rows, cmap=diverging, vmin=-lim, vmax=lim, annotate=True, fmt="{:.1f}")
  ax.axhline(len(names) - 0.5, color=INK, linewidth=1)
  fig.colorbar(im, ax=ax, shrink=0.8, label="logit minus row average")
  ax.set(title=f"{decode(X[example])} → {VOCAB[Y[example]]}: each component's vote", xlabel="output digit")
  out.png(fig, "residual_example")
  out.csv("residual_example", ["sequence", "answer", "component"] + [f"logit_{d}" for d in VOCAB[N_LETTERS:]],
          [[decode(X[example]), VOCAB[Y[example]], n] + [float(v) for v in row] for n, row in zip(rows, L)])

  N = np.stack([np.linalg.norm(r["parts"][n], axis=-1).mean(0) for n in names])
  fig, ax = plt.subplots(figsize=(7, 0.45*len(names) + 1.2))
  im = heatmap(ax, N, positions, names, vmin=0, vmax=N.max(), annotate=True, fmt="{:.1f}")
  fig.colorbar(im, ax=ax, shrink=0.8, label="length of the write")
  ax.set(title=f"{title}: how much each component writes, by position", xlabel="position")
  out.png(fig, "write_size")
  out.csv("write_size", ["component", "position", "mean_norm"], [(n, positions[t], float(N[i, t])) for i, n in enumerate(names) for t in range(len(positions))])

  heads = head_names(r)
  rows = copy_table(r, X, example)
  W = np.array([row[5] for row in rows]).reshape(len(heads), len(positions))
  fig, ax = plt.subplots(figsize=(1.0*len(positions) + 1.5, 0.75*len(heads) + 1.2))
  heatmap(ax, W, [f"{p}\n{VOCAB[X[example, t]]}" for t, p in enumerate(positions)], heads)
  for row, (h, t, tok, src, src_tok, w, cos) in zip(range(len(rows)), rows):
    ax.text(t, heads.index(h), f"←{src_tok}\n{positions[src]}", ha="center", va="center", fontsize=8, color=SURFACE if w > 0.55 else INK)
  ax.set(title=f"{decode(X[example])}: what each head copies, at each position\n(color = attention on the copied position)")
  out.png(fig, "head_copies")
  out.csv("head_copies", ["head", "position", "token", "copied_from_position", "copied_token", "attention", "write_vs_pure_copy_cosine"],
          [(h, positions[t], tok, positions[src], src_tok, w, cos) for h, t, tok, src, src_tok, w, cos in rows])
  out.csv("head_copies_test_average", ["head", "position", "attention_on_top_position", "write_vs_pure_copy_cosine"],
          [(h, positions[t], w, cos) for h, t, w, cos in copy_table(r, X)])

def ablation_plot(out:Out, accs:dict, title:str) -> None:
  names = list(accs)
  fig, ax = plt.subplots(figsize=(6.5, 0.5*len(names) + 1))
  ax.barh(range(len(names)), [accs[n] for n in names], color=[BLUE] + [MUTED]*(len(names)-1), height=0.6)
  for i, n in enumerate(names): ax.text(accs[n] + 1, i, f"{accs[n]:.1f}%", va="center")
  ax.set_yticks(range(len(names)), names)
  ax.invert_yaxis()
  ax.set(title=title, xlabel="test accuracy (%)", xlim=(0, 110))
  ax.grid(axis="x", color=GRID, linewidth=0.5)
  out.png(fig, "ablation")
  out.csv("ablation", ["model", "test_accuracy"], [(n, accs[n]) for n in names])

# the residual stream of one example as an exact sum of paths. with no layernorm and the attention patterns fixed, every head is linear:
# its write at position i is sum_j attention_ij * V(stream at j). expanding that recursively, the stream at every position is a sum of
# "the token (or position embedding) at position j, carried through heads L1H1 -> L2H1 -> ..., scaled by the product of attention weights".
# the MLP isn't linear, so it stays one opaque term. returns, per stage, per position, a list of (kind, source position, chain, weight, vector)
def residual_paths(model, r:dict, X:np.ndarray, example:int) -> list[list[list[tuple]]]:
  T = X.shape[1]
  E = model.emb.weight.numpy()
  paths = [[("token", i, (), 1.0, E[X[example, i]])] for i in range(T)]
  if "position" in r["parts"]:
    for i in range(T): paths[i].append(("position", i, (), 1.0, r["parts"]["position"][example, i]))
  stages = [[list(p) for p in paths]]
  for l, layer in enumerate(model.layers):
    H, Wv = layer.attn.n_heads, layer.attn.v_proj.weight.numpy()
    new = [list(p) for p in paths]
    for h in range(H):
      Vh, A = Wv[h*Wv.shape[1]:(h+1)*Wv.shape[1]], r["pattern"][l][example, h]
      for i in range(T):
        for j in range(i+1):
          new[i] += [(k, src, chain + (f"L{l+1}H{h+1}",), w * A[i, j], A[i, j] * (vec @ Vh.T)) for k, src, chain, w, vec in paths[j]]
    paths = new
    if layer.mlp is not None:
      stages.append([list(p) for p in paths])
      for i in range(T): paths[i].append(("MLP", i, ("MLP",), 1.0, r["parts"]["MLP"][example, i]))
    stages.append([list(p) for p in paths])
  for i in range(T): assert np.allclose(sum(p[4] for p in paths[i]), r["resid"][-1][example, i], atol=1e-3)
  return stages

def path_label(kind:str, src:int, chain:tuple, X:np.ndarray, example:int, positions:list[str]) -> str:
  if kind == "MLP": return "MLP output"
  what = VOCAB[X[example, src]] if kind == "token" else f"pos({positions[src]})"
  return what + (f" ←{positions[src]} {'→'.join(chain)}" if chain else " (own)")

# one picture of the whole forward pass: rows are stages (embedding, after each layer), columns are positions, and each cell lists
# the biggest paths in the stream there. the csv has every path with weight >= 0.01
def residual_map(out:Out, model, r:dict, X:np.ndarray, positions:list[str], title:str, example:int=0, top:int=4) -> None:
  from .dataset import decode
  stages = residual_paths(model, r, X, example)
  names = ["embedding"] + [f"after layer {l+1}" for l in range(len(model.layers))]
  if "MLP" in r["parts"]: names.append("after MLP")
  T = X.shape[1]
  fig, ax = plt.subplots(figsize=(2.25*T, 1.1*len(stages) + 1.2))
  ax.set(xlim=(-0.5, T-0.5), ylim=(len(stages)-0.5, -0.5))
  ax.set_xticks(range(T), [f"{p}\n{VOCAB[X[example, t]]}" for t, p in enumerate(positions)])
  ax.set_yticks(range(len(stages)), names)
  ax.spines[:].set_visible(False)
  ax.tick_params(length=0)
  rows = []
  for s, stage in enumerate(stages):
    ax.axhline(s + 0.5, color=GRID, linewidth=1)
    for i, cell in enumerate(stage):
      size = [np.linalg.norm(p[4]) for p in cell]
      total = np.linalg.norm(sum(p[4] for p in cell))
      order = np.argsort(size)[::-1]
      lines = [f"{size[k]:4.1f}  {path_label(*cell[k][:3], X, example, positions)}" for k in order[:top] if size[k] > 0.1 * max(size)]
      ax.text(i - 0.47, s, "\n".join(lines), va="center", ha="left", fontsize=7, fontfamily="monospace",
              color=INK if s == len(stages) - 1 or i == T - 1 else MUTED)
      rows += [(names[s], positions[i], VOCAB[X[example, i]], cell[k][0], positions[cell[k][1]], VOCAB[X[example, cell[k][1]]],
                "→".join(cell[k][2]) or "direct", cell[k][3], size[k], total) for k in order if cell[k][3] >= 0.01]
  ax.set_title(f"{title}, {decode(X[example])}: what's in the residual stream\n(length of each path's contribution, biggest first)")
  out.png(fig, "residual_map")
  out.csv("residual_map", ["stage", "position", "token", "source_kind", "source_position", "source_token", "via", "weight", "length", "stream_length"], rows)

# which parts of the stream make a head pick the right position? split the head's query (at query_pos) and keys into the
# components written before that layer, score every (query part, key part) pair, and measure how much more each pair gives the
# target position than the average other candidate. the parts sum to the full advantage
def qk_parts(model, r:dict, layer:int, head:int, query_pos:int, target:np.ndarray, candidates:np.ndarray) -> tuple[list[str], np.ndarray]:
  a = model.layers[layer].attn
  Wq = a.q_proj.weight.numpy()[head*head_dim:(head+1)*head_dim]
  Wk = a.k_proj.weight.numpy()[head*head_dim:(head+1)*head_dim]
  names = [n for n in r["parts"] if n in ("token", "position") or (n.startswith("L") and int(n[1]) <= layer)]
  B = len(target)
  M = np.zeros((len(names), len(names)))
  for i, nq in enumerate(names):
    q = r["parts"][nq][:, query_pos] @ Wq.T
    for j, nk in enumerate(names):
      s = np.einsum("bd,btd->bt", q, r["parts"][nk] @ Wk.T) / head_dim ** 0.5
      t = s[np.arange(B), target]
      others = np.where(candidates & (np.arange(s.shape[1]) != target[:, None]), s, np.nan)
      M[i, j] = (t - np.nanmean(others, -1)).mean()
  return names, M

def qk_parts_plot(out:Out, names:list[str], M:np.ndarray, name:str, title:str) -> None:
  fig, ax = plt.subplots(figsize=(1.1*len(names) + 2.5, 0.8*len(names) + 1.5))
  lim = np.abs(M).max()
  im = heatmap(ax, M, names, names, cmap=diverging, vmin=-lim, vmax=lim, annotate=True, fmt="{:.1f}")
  fig.colorbar(im, ax=ax, shrink=0.8, label="score advantage of the target")
  ax.set(title=title, xlabel="key built from", ylabel="query built from")
  out.png(fig, name)
  out.csv(name, ["query_part"] + [f"key_{n}" for n in names], [[n] + [float(v) for v in row] for n, row in zip(names, M)])

# everything for the two-key tasks (3: sum, 4: difference), on a trained model with an mlp after the last layer
TWO_KEY_POSITIONS = ["k1", "v1", "k2", "v2", "k3", "v3", "k4", "v4", "q1", "q2"]
def two_key_analysis(out:Out, model, op:str, X:np.ndarray, Y:np.ndarray, title:str) -> dict:
  from .dataset import decode
  P, T = TWO_KEY_POSITIONS, X.shape[1]
  combine = (lambda a, b: (a + b) % 10) if op == "sum" else (lambda a, b: (a - b) % 10)
  sym = "+" if op == "sum" else "−"
  r = forward(model, X)
  assert np.allclose(sum(r["parts"].values()), r["resid"][-1], atol=1e-3)
  B, n_layers = len(X), len(model.layers)
  first = 2*(X[:, 0:8:2] == X[:, 8:9]).argmax(1) + 1          # position of the value at q1, and at q2
  second = 2*(X[:, 0:8:2] == X[:, 9:10]).argmax(1) + 1
  is_value = np.zeros((B, T), dtype=bool); is_value[:, 1:8:2] = True
  stats = {}

  # 1. attention in every layer but the last, averaged over the test set, one panel per head
  for l in range(n_layers - 1):
    H = model.layers[l].attn.n_heads
    fig, axes = plt.subplots(1, H, figsize=(5.5*H, 4.8), squeeze=False)
    rows = []
    for h, ax in enumerate(axes[0]):
      avg = r["pattern"][l][:, h].mean(0)
      heatmap(ax, avg, P, P, annotate=True)
      ax.set(title=f"L{l+1}H{h+1}: average attention", xlabel="position attended to (key)", ylabel="position attending (query)")
      rows += [(f"L{l+1}H{h+1}", P[i], P[j], float(avg[i, j])) for i in range(T) for j in range(T)]
    out.png(fig, f"attention_layer{l+1}")
    out.csv(f"attention_layer{l+1}", ["head", "query_position", "key_position", "attention"], rows)

  # 2. last layer: attention from q2 on a few test sequences, per head, and on average: how much lands on each queried value
  L, H = n_layers - 1, model.layers[-1].attn.n_heads
  n_show = 8
  fig, axes = plt.subplots(1, H, figsize=(6*H, 4), squeeze=False)
  rows = []
  for h, ax in enumerate(axes[0]):
    pat = r["pattern"][L][:, h, -1]
    ax.imshow(pat[:n_show], cmap=sequential, vmin=0, vmax=1, aspect="auto")
    for i in range(n_show):
      for c in range(T): ax.text(c, i, VOCAB[X[i, c]], ha="center", va="center", color=SURFACE if pat[i, c] > 0.5 else INK, fontsize=11)
    ax.set_xticks(range(T), P)
    ax.set_yticks(range(n_show), [f"{decode(x)} → {VOCAB[y]}" for x, y in zip(X[:n_show], Y[:n_show])], fontfamily="monospace")
    ax.set(title=f"L{L+1}H{h+1}: attention from the last position", xlabel="position attended to")
    ax.spines[:].set_visible(False)
    rows += [(f"L{L+1}H{h+1}", i, decode(X[i]), P[c], VOCAB[X[i, c]], kind(X[i, c]), float(pat[i, c])) for i in range(n_show) for c in range(T)]
  out.png(fig, f"attention_layer{L+1}")
  out.csv(f"attention_layer{L+1}", ["head", "example", "sequence", "position", "token", "kind", "attention"], rows)

  split = {}
  for h in range(H):
    pat = r["pattern"][L][:, h, -1]
    on1, on2 = pat[np.arange(B), first], pat[np.arange(B), second]
    split[f"L{L+1}H{h+1}"] = (on1.mean(), on2.mean(), 1 - on1.mean() - on2.mean())
  fig, ax = plt.subplots(figsize=(6, 0.8*H + 1.4))
  for i, (name, parts) in enumerate(split.items()):
    left = 0
    for j, (v, color, label) in enumerate(zip(parts, (SERIES[0], SERIES[1], GRID), ("value at q1", "value at q2", "everything else"))):
      ax.barh(i, v, left=left, color=color, height=0.6, edgecolor=SURFACE, linewidth=2, label=label if i == 0 else None)
      if v > 0.06: ax.text(left + v/2, i, f"{v:.2f}", ha="center", va="center", color=SURFACE if j < 2 else INK)
      left += v
  ax.set_yticks(range(H), list(split))
  ax.invert_yaxis()
  ax.set(xlim=(0, 1), title=f"L{L+1}: where the last position looks (test average)", xlabel="attention")
  ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3)
  out.png(fig, "attention_split")
  out.csv("attention_split", ["head", "on_value_at_q1", "on_value_at_q2", "elsewhere"], [(n, *v) for n, v in split.items()])
  stats["split"] = split

  # 3. last layer QK: which parts of the stream make each head find the value at q1, and the value at q2
  fig, axes = plt.subplots(H, 2, figsize=(11, 4.2*H), squeeze=False)
  rows = []
  for h in range(H):
    for c, (target, tname) in enumerate(((first, "value at q1"), (second, "value at q2"))):
      names, M = qk_parts(model, r, L, h, -1, target, is_value)
      lim = max(np.abs(M).max(), 1e-6)
      heatmap(axes[h, c], M, names, names, cmap=diverging, vmin=-lim, vmax=lim, annotate=True, fmt="{:.1f}")
      axes[h, c].set(title=f"L{L+1}H{h+1} finding the {tname}", xlabel="key built from", ylabel="query built from")
      rows += [(f"L{L+1}H{h+1}", tname, nq, nk, float(M[i, j])) for i, nq in enumerate(names) for j, nk in enumerate(names)]
  fig.suptitle("score advantage of the target value over the other values, split by stream part", fontweight="bold")
  fig.tight_layout()
  out.png(fig, f"qk_layer{L+1}_parts")
  out.csv(f"qk_layer{L+1}_parts", ["head", "target", "query_part", "key_part", "score_advantage"], rows)

  # 4. the residual stream: who votes for the answer, how much each component writes, what each head copies, the full path map
  residual_plots(out, model, r, X, Y, P, title)
  residual_map(out, model, r, X, P, title)

  # 5. every pair of values (a, b), 100 random sequences each: accuracy, what the stream says before vs after the mlp, the mlp's neurons
  rng = np.random.default_rng(0)
  n = 100
  G = np.concatenate([two_key_inputs(a, b, n, rng) for a in range(10) for b in range(10)])
  rg = forward(model, G)
  answer = np.array([combine(a, b) for a in range(10) for b in range(10)]).repeat(n)
  acc = (rg["logits"].argmax(-1) - N_LETTERS == answer).reshape(10, 10, n).mean(-1) * 100
  fig, ax = plt.subplots(figsize=(5.5, 4.8))
  im = heatmap(ax, acc, list(range(10)), list(range(10)), vmin=0, vmax=100, annotate=True, fmt="{:.0f}")
  fig.colorbar(im, ax=ax, shrink=0.8, label="accuracy (%)")
  ax.set(title=f"accuracy for every pair of values\n(answer = (a {sym} b) mod 10)", xlabel="b (value at q2)", ylabel="a (value at q1)")
  out.png(fig, "accuracy_grid")
  out.csv("accuracy_grid", ["a", "b", "answer", "accuracy"], [(a, b, combine(a, b), float(acc[a, b])) for a in range(10) for b in range(10)])

  # linear probes: can a straight line through the stream at the last position read off a, b, or the answer? fit on half the
  # pairs-grid sequences (ridge regression onto one-hot labels), score on the other half. after layer 1 the values haven't arrived;
  # after the last attention layer a and b are there but the answer isn't linearly readable; after the MLP it is
  stages = {f"after layer {l+1}": rg["resid"][l+1][:, -1] for l in range(n_layers)}
  stages["after MLP"] = rg["resid"][-1][:, -1]
  A, Bv = np.arange(10).repeat(10*n), np.tile(np.arange(10).repeat(n), 10)
  targets = {"a (value at q1)": A, "b (value at q2)": Bv, f"answer (a {sym} b)": answer}
  idx = rng.permutation(len(G)); tr, te = idx[:len(G)//2], idx[len(G)//2:]
  probe = {}
  for sname, Z in stages.items():
    Z1 = np.concatenate([Z, np.ones((len(Z), 1))], 1)
    for tname, lab in targets.items():
      W = np.linalg.solve(Z1[tr].T @ Z1[tr] + 1e-2*np.eye(Z1.shape[1]), Z1[tr].T @ np.eye(10)[lab[tr]])
      probe[(sname, tname)] = ((Z1[te] @ W).argmax(-1) == lab[te]).mean() * 100
  fig, ax = plt.subplots(figsize=(8, 3.8))
  w = 0.8 / len(targets)
  for j, (tname, color) in enumerate(zip(targets, SERIES)):
    vals = [probe[(sname, tname)] for sname in stages]
    ax.bar(np.arange(len(stages)) + (j - (len(targets)-1)/2)*w, vals, width=w - 0.03, color=color, label=tname)
    for i, v in enumerate(vals): ax.text(i + (j - (len(targets)-1)/2)*w, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK)
  ax.axhline(10, color=MUTED, linewidth=1, linestyle=":")
  ax.set_xticks(range(len(stages)), list(stages))
  ax.set(ylim=(0, 112), ylabel="linear probe accuracy (%)", title="what can be read off the last position with a straight line?")
  ax.grid(axis="y", color=GRID, linewidth=0.5)
  ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
  out.png(fig, "probes")
  out.csv("probes", ["stage", "target", "probe_accuracy"], [(sname, tname, float(v)) for (sname, tname), v in probe.items()])
  stats["probes"] = probe

  # mlp neurons: mean activation for every (a, b). r2 says how much of a neuron's variation over the grid is explained by
  # knowing only the answer, only a, only b, or the plain a+b / a-b without mod 10
  act = rg["neurons"].reshape(10, 10, n, -1).mean(2)              # (10, 10, mlp_size)
  aa, bb = np.meshgrid(range(10), range(10), indexing="ij")
  def r2(v, groups):
    total = ((v - v.mean())**2).sum()
    return 1 - sum(((v[groups == g] - v[groups == g].mean())**2).sum() for g in np.unique(groups)) / total if total > 0 else np.nan
  raw = aa + bb if op == "sum" else aa - bb
  summary = []
  for k in range(act.shape[-1]):
    v = act[..., k]
    summary.append((k, float(v.mean()), float(v.std()), r2(v, combine(aa, bb)), r2(v, aa), r2(v, bb), r2(v, raw)))
  out.csv("neuron_summary", ["neuron", "mean_activation", "std_over_pairs", "r2_answer", "r2_a", "r2_b", f"r2_a{'+' if op == 'sum' else '-'}b_no_mod"], summary)
  out.csv("neurons", ["neuron", "a", "b", "answer", "mean_activation"], [(k, a, b, combine(a, b), float(act[a, b, k])) for k in range(act.shape[-1]) for a in range(10) for b in range(10)])
  top = sorted(summary, key=lambda s: -s[2])[:8]
  fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
  for ax, s in zip(axes.flat, top):
    k = s[0]
    heatmap(ax, act[..., k], list(range(10)), list(range(10)), vmin=0, vmax=act[..., k].max())
    ax.set_title(f"neuron {k}\nr² answer {s[3]:.2f}, raw a{sym}b {s[6]:.2f}", fontsize=9)
    ax.set_xlabel("b"); ax.set_ylabel("a")
  fig.suptitle(f"MLP neurons with the most variation: average activation for each (a, b)", fontweight="bold")
  fig.tight_layout()
  out.png(fig, "neurons")
  stats["neurons_top"] = top

  # 6. ablations
  ab_runs = {"full model": r}
  for l in range(n_layers):
    ab_runs[f"layer {l+1} attention removed"] = forward(model, X, drop_attn=(l,))
    if model.layers[l].attn.n_heads > 1:
      for h in range(model.layers[l].attn.n_heads): ab_runs[f"only L{l+1}H{h+1} removed"] = forward(model, X, drop_attn=(f"L{l+1}H{h+1}",))
  ab_runs["MLP removed"] = forward(model, X, mlp=False)
  ab_runs["position embedding removed"] = forward(model, X, pos=False)
  stats["ablation"] = {k: (v["logits"].argmax(-1) == Y).mean() * 100 for k, v in ab_runs.items()}
  ablation_plot(out, stats["ablation"], f"{title}: knock out one piece")
  return stats
