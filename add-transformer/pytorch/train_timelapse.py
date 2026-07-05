"""Train fresh, snapshot q/k/v_proj every SNAP_EVERY steps, render a GIF timelapse.

Non-invasive: reuses Model/dataset/loss from main.py, does NOT overwrite model.pt.
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import main as M
from main import Model, loss_fn, get_batch, get_device, evaluate
from dataset import dataset, get_dataset_config

SNAP_EVERY = 50
STEPS = 1400          # dynamics finish ~900 steps; run a bit past to show settling
PROJS = ["q_proj", "k_proj", "v_proj"]

torch.manual_seed(0)
config = get_dataset_config()
device = get_device()
trainx, trainy, testx, testy = dataset(n_train=M.N_TRAIN, n_test=M.N_TEST)
trainx, trainy = trainx.to(device), trainy.to(device)
testx, testy = testx.to(device), testy.to(device)
model = Model(config).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=M.LR)

frames = []  # each: dict(step, loss, grid=[[ (32,32) per proj ] per layer ])

@torch.no_grad()
def snapshot(step, loss):
  grid = []
  for layer in model.layers:
    row = [getattr(layer.attn, p).detach().cpu().clone() for p in PROJS]
    grid.append(row)
  frames.append({"step": step, "loss": loss, "grid": grid})

snapshot(0, float("nan"))
for step in range(1, STEPS + 1):
  x, y = get_batch(trainx, trainy)
  opt.zero_grad(set_to_none=True)
  loss = loss_fn(model(x), y, config)
  loss.backward()
  opt.step()
  if step % SNAP_EVERY == 0:
    snapshot(step, loss.item())
    acc = evaluate(model, testx, testy, config, n=512)
    print(f"step {step:5d}  loss {loss.item():.4f}  acc {acc:6.2%}  (snapshot {len(frames)})")

print(f"\ncollected {len(frames)} frames; rendering GIF...")

# global symmetric color scale so growth-from-noise is visible and comparable
absmax = max(w.abs().max().item() for f in frames for row in f["grid"] for w in row)
n_layers = len(frames[0]["grid"])

fig, axes = plt.subplots(n_layers, len(PROJS), figsize=(15, 9))
ims = [[None] * len(PROJS) for _ in range(n_layers)]
for li in range(n_layers):
  for pi, pname in enumerate(PROJS):
    ax = axes[li, pi]
    ims[li][pi] = ax.imshow(frames[0]["grid"][li][pi], cmap="RdBu_r",
                            vmin=-absmax, vmax=absmax, animated=True)
    ax.set_title(f"layer {li} {pname}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(ims[0][0], ax=axes, fraction=0.025, pad=0.02)
suptitle = fig.suptitle("", fontsize=13)

def update(fi):
  f = frames[fi]
  for li in range(n_layers):
    for pi in range(len(PROJS)):
      ims[li][pi].set_data(f["grid"][li][pi])
  ls = "init" if fi == 0 else f"loss {f['loss']:.3f}"
  suptitle.set_text(f"q/k/v_proj weights  |  step {f['step']}  ({ls})")
  return [im for row in ims for im in row] + [suptitle]

anim = FuncAnimation(fig, update, frames=len(frames), interval=200, blit=False)
anim.save("qkv_timelapse.gif", writer=PillowWriter(fps=5), dpi=150)
print("saved -> qkv_timelapse.gif")
