# task 0 with tied embeddings: the same zero-layer model as task0.py, but one table does both jobs, so
# logits for last letter q = E @ E[q] (the unembedding is the embedding, transposed). same data, seed, steps and optimizer.
# side experiment, not part of the tutorial outline. untied, the letters' rows can point one way in the embedding and
# the other in the unembedding; tied, the last letter's score is E[q] . E[q] = |E[q]|^2, always positive, so the model
# has to make every digit score even higher than that
import os, math, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tinygrad import Tensor, nn, TinyJit, Context
from dataset import hidden_digit_dataset, VOCAB, N_LETTERS
Tensor.manual_seed(0) # reproducible training runs

STEPS = 2000
xtrain, xtest, ytrain, ytest = hidden_digit_dataset()

class TiedModel:
  def __init__(self): self.emb = nn.Embedding(len(VOCAB), 32)
  def __call__(self, x: Tensor) -> Tensor: return self.emb(x)[:, -1] @ self.emb.weight.T   # (B, 36)

model = TiedModel()
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

SURFACE, INK, MUTED, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "font.size": 10})

def write_csv(name:str, header:list[str], rows) -> None:
  with open(f"data/task0_tied/{name}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows([[f"{v:.6g}" if isinstance(v, float) else v for v in row] for row in rows])

def cos(W): W = W / np.linalg.norm(W, axis=1, keepdims=True); return W @ W.T
def block(C, r, c, diag=True): m = C[r][:, c]; return m[~np.eye(len(m), dtype=bool)].mean() if not diag else m.mean()

if __name__ == "__main__":
  os.makedirs("plots/task0_tied", exist_ok=True)
  os.makedirs("data/task0_tied", exist_ok=True)
  losses = []
  for i in range(STEPS):
    losses.append(train_step().item())
    if i % 500 == 499: print(f"loss and val acc at step {i+1}: {losses[-1]:.2f}, {test_acc().item():.2f}%")
  E = model.emb.weight.numpy()
  L, D = slice(0, N_LETTERS), slice(N_LETTERS, None)
  logits = E[L] @ E.T                                            # (26 last letters, 36 output tokens)
  probs = np.exp(logits - logits.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
  untied = [float(r["loss"]) for r in csv.DictReader(open("data/task0/loss.csv"))] if os.path.exists("data/task0/loss.csv") else None

  # 1. loss next to the untied model's
  fig, ax = plt.subplots(figsize=(7, 3.5))
  ax.plot(losses, color=ORANGE, linewidth=2, label="tied")
  if untied: ax.plot(untied, color=BLUE, linewidth=2, label="untied (task0.py)")
  for y in (math.log(36), math.log(10)): ax.axhline(y, color=MUTED, linewidth=1, linestyle="--")
  ax.set(title="0 layers, tied vs untied: training loss", xlabel="step", ylabel="loss"); ax.legend(frameon=False)
  fig.savefig("plots/task0_tied/loss.png", dpi=150, bbox_inches="tight")
  write_csv("loss", ["step", "loss"], [(i+1, l) for i, l in enumerate(losses)])

  # 2. the logits for every last letter: the diagonal (its own score) is |E[q]|^2
  fig, ax = plt.subplots(figsize=(10, 6))
  v = np.abs(logits).max()
  im = ax.imshow(logits, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
  ax.set_xticks(range(36), list(VOCAB), fontsize=7); ax.set_yticks(range(N_LETTERS), list(VOCAB[:N_LETTERS]), fontsize=7)
  ax.set(title="tied: logits E @ E[last letter]", xlabel="output token", ylabel="last letter"); fig.colorbar(im, ax=ax)
  fig.savefig("plots/task0_tied/logits.png", dpi=150, bbox_inches="tight")
  write_csv("logits", ["last_letter"] + list(VOCAB), [[VOCAB[i]] + [float(x) for x in row] for i, row in enumerate(logits)])

  # 3. cosine similarity of the one table's rows
  C = cos(E)
  fig, ax = plt.subplots(figsize=(6, 5.5))
  im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
  ax.set_xticks(range(36), list(VOCAB), fontsize=6); ax.set_yticks(range(36), list(VOCAB), fontsize=6)
  ax.set(title="tied: cos between rows of the one table"); fig.colorbar(im, ax=ax, shrink=.8)
  fig.savefig("plots/task0_tied/cosine.png", dpi=150, bbox_inches="tight")
  write_csv("embedding", ["token", "kind"] + [f"d{k}" for k in range(32)],
            [[t, "letter" if i < N_LETTERS else "digit"] + [float(x) for x in E[i]] for i, t in enumerate(VOCAB)])

  own = logits[np.arange(N_LETTERS), np.arange(N_LETTERS)]
  other = (logits[:, L].sum(1) - own) / (N_LETTERS - 1)
  print(f"test accuracy {test_acc().item():.2f}%, final loss {np.mean(losses[-100:]):.3f} (ln 10 = {math.log(10):.3f})")
  print(f"P(any digit) {probs[:, D].sum(1).mean():.3f}, P(own letter) {probs[np.arange(N_LETTERS), np.arange(N_LETTERS)].mean():.4f}, P(other letters) {probs[:, L].sum(1).mean() - probs[np.arange(N_LETTERS), np.arange(N_LETTERS)].mean():.4f}")
  print(f"logits: own letter {own.mean():.2f}, other letters {other.mean():.2f}, digits {logits[:, D].mean():.2f} (spread across letters {logits[:, D].std(0).mean():.3f})")
  print(f"cos: letter-letter {block(C, L, L, False):+.2f}, digit-digit {block(C, D, D, False):+.2f}, letter-digit {block(C, L, D):+.2f}")
  n = np.linalg.norm(E, axis=1)
  print(f"norms: letters {n[L].mean():.2f}, digits {n[D].mean():.2f}")
