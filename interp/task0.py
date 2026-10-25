# task 0: no layers and no position embedding, on the hidden digit task. logits = unembed(token embedding) at the last position.
# nothing moves between positions and the last token is always a letter, so it can't see the digit and stays at chance.
# (position embeddings come later, once attention can compare positions; here they'd only add a constant bias)
import os, math, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tinygrad import Tensor, nn, TinyJit, Context
from dataset import hidden_digit_dataset, VOCAB, N_LETTERS
from model import Model
Tensor.manual_seed(0) # reproducible training runs

STEPS = 2000
xtrain, xtest, ytrain, ytest = hidden_digit_dataset()
model = Model(n_layers=0, pos=False)
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

# average predicted probability of each of the 36 output tokens over the test set
def mean_prediction() -> np.ndarray: return model(xtest).softmax(-1).mean(0).numpy()

# plot styling
SURFACE, INK, MUTED, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold", "font.size": 10})
sequential = LinearSegmentedColormap.from_list("blue", ["#f0f6fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

# raw numbers behind every plot go to data/task0/*.csv, so slides can draw their own charts
def write_csv(name:str, header:list[str], rows) -> None:
  with open(f"data/task0/{name}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows([[f"{v:.6g}" if isinstance(v, float) else v for v in row] for row in rows])

def token_axis(ax):
  ax.set_xticks(range(len(VOCAB)), list(VOCAB), fontsize=8)
  ax.axvline(N_LETTERS - 0.5, color=MUTED, linewidth=1, linestyle=":")
  ax.grid(axis="y", color=GRID, linewidth=0.5)

if __name__ == "__main__":
  emb_init = model.emb.weight.numpy().copy()
  pred_init = mean_prediction()

  losses = []
  for i in range(STEPS):
    losses.append(train_step().item())
    if i % 500 == 499: print(f"loss and val acc at step {i+1}: {losses[-1]:.2f}, {test_acc().item():.2f}%")
  pred_trained = mean_prediction()
  os.makedirs("plots/task0", exist_ok=True)
  os.makedirs("data/task0", exist_ok=True)

  # 1. loss: starts at ln 36 (random weights spread over all 36 tokens), drops to ln 10 (even over the 10 digits) and stays
  fig, ax = plt.subplots(figsize=(7, 3.5))
  ax.plot(losses, color=BLUE, linewidth=2)
  for y, label in ((math.log(36), "even over all 36 tokens: ln 36 = 3.58"), (math.log(10), "even over the 10 digits: ln 10 = 2.30")):
    ax.axhline(y, color=MUTED, linewidth=1, linestyle="--")
    ax.text(STEPS, y, "  " + label, color=MUTED, va="center", ha="left")
  ax.set(title="0 layers: training loss on the hidden digit task", xlabel="step", ylabel="loss")
  ax.grid(axis="y", color=GRID, linewidth=0.5)
  fig.savefig("plots/task0/loss.png", dpi=150, bbox_inches="tight")
  write_csv("loss", ["step", "loss"], [(i+1, l) for i, l in enumerate(losses)])
  write_csv("loss_reference", ["label", "loss"], [("even over all 36 tokens (ln 36)", math.log(36)), ("even over the 10 digits (ln 10)", math.log(10))])

  # 2. what it predicts: before training, a bit of everything; after, every digit evenly and no letters
  fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharey=True)
  for ax, pred, title in ((axes[0], pred_init, "random weights: average prediction"), (axes[1], pred_trained, "trained: average prediction")):
    ax.bar(range(len(VOCAB)), pred, color=BLUE, width=0.7)
    ax.axhline(0.1, color=MUTED, linewidth=1, linestyle="--")
    ax.set(title=title, ylabel="probability")
    token_axis(ax)
  axes[1].set_xlabel("output token")
  fig.tight_layout()
  fig.savefig("plots/task0/prediction.png", dpi=150, bbox_inches="tight")
  write_csv("prediction", ["token", "kind", "probability_random_weights", "probability_trained"],
            [(t, "letter" if i < N_LETTERS else "digit", float(pred_init[i]), float(pred_trained[i])) for i, t in enumerate(VOCAB)])

  # 3. the same, split by the last letter: no letter favours any digit
  E, U = model.emb.weight.numpy(), model.unembed.weight.numpy()
  logits = E[:N_LETTERS] @ U.T                                  # (26 last letters, 36 output tokens)
  probs = np.exp(logits - logits.max(1, keepdims=True))
  probs = (probs / probs.sum(1, keepdims=True))[:, N_LETTERS:]   # keep the 10 digit columns
  fig, ax = plt.subplots(figsize=(5, 8))
  im = ax.imshow(probs, cmap=sequential, vmin=0, vmax=0.2, aspect="auto")
  ax.set_xticks(range(10), list(VOCAB[N_LETTERS:]))
  ax.set_yticks(range(N_LETTERS), list(VOCAB[:N_LETTERS]))
  ax.set(title="P(digit | last letter): flat at 0.1", xlabel="output digit", ylabel="last letter")
  ax.spines[:].set_visible(False)
  fig.colorbar(im, ax=ax, shrink=0.6, label="probability")
  fig.savefig("plots/task0/table.png", dpi=150, bbox_inches="tight")
  write_csv("table", ["last_letter"] + [f"p_digit_{d}" for d in VOCAB[N_LETTERS:]],
            [[VOCAB[i]] + [float(p) for p in row] for i, row in enumerate(probs)])

  # 4. what training changed: only what the last position reads, i.e. the letters.
  #    digits only ever sit at positions 0-7 (never the last), never reach the output, get no gradient and stay at their random init
  fig, ax = plt.subplots(figsize=(9, 3))
  ax.bar(range(len(VOCAB)), np.linalg.norm(E - emb_init, axis=1), color=BLUE, width=0.7)
  ax.set(title="how far each token embedding moved during training", xlabel="token", ylabel="distance moved from init")
  token_axis(ax)
  fig.savefig("plots/task0/moved.png", dpi=150, bbox_inches="tight")
  # rounded to 4 decimals: the untouched rows differ from init by ~1e-7 of float noise, i.e. zero
  write_csv("moved_tokens", ["token", "kind", "distance_moved"],
            [(t, "letter" if i < N_LETTERS else "digit", round(float(d), 4)) for i, (t, d) in enumerate(zip(VOCAB, np.linalg.norm(E - emb_init, axis=1)))])

  # 5. the trained weights themselves: the whole model is these two (36, 32) tables. logits for last letter q = U @ E[q]
  for name, W in (("embedding", E), ("unembedding", U)):
    write_csv(name, ["token", "kind"] + [f"d{k}" for k in range(W.shape[1])],
              [[t, "letter" if i < N_LETTERS else "digit"] + [float(v) for v in W[i]] for i, t in enumerate(VOCAB)])

  print(f"average prediction: letters total {pred_trained[:N_LETTERS].sum():.4f}, digits {np.round(pred_trained[N_LETTERS:], 3)}")
  print("token embedding moved, letters:", np.round(np.linalg.norm(E - emb_init, axis=1)[:N_LETTERS].mean(), 3), "digits:", np.round(np.linalg.norm(E - emb_init, axis=1)[N_LETTERS:], 3))
