"""Train fresh, snapshot ATTENTION on a fixed probe example every SNAP_EVERY steps,
render a GIF so you can watch head specialization (esp. head 3's -1 diagonal) emerge.

Non-invasive: reuses Model/dataset/loss from main.py, does NOT overwrite model.pt.
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import main as M
from main import Model, loss_fn, get_batch, get_device, evaluate, n_heads, head_dim
from dataset import dataset, get_dataset_config, encode, format_example

SNAP_EVERY = 50
STEPS = 1400

torch.manual_seed(0)
config = get_dataset_config()
device = get_device()
trainx, trainy, testx, testy = dataset(n_train=M.N_TRAIN, n_test=M.N_TEST)
trainx, trainy = trainx.to(device), trainy.to(device)
testx, testy = testx.to(device), testy.to(device)
model = Model(config).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=M.LR)

# fixed probe example (teacher-forced 31-token input)
full = format_example(3412, 5879)
x_str = full[:-1]
labels = list(x_str)
probe = torch.tensor([encode(x_str, config)], dtype=torch.long, device=device)
S = probe.shape[1]

@torch.no_grad()
def attn_snapshot():
  was_training = model.training
  model.eval()
  maps = []
  x = model.tok_emb[probe] + model.pos_emb[:S]
  for layer in model.layers:
    h = layer.norm1(x)
    A = layer.attn
    q = (h @ A.q_proj).reshape(1, S, n_heads, head_dim).transpose(1, 2)
    k = (h @ A.k_proj).reshape(1, S, n_heads, head_dim).transpose(1, 2)
    v = (h @ A.v_proj).reshape(1, S, n_heads, head_dim).transpose(1, 2)
    qk = q @ k.transpose(-2, -1)
    mask = torch.ones(S, S, dtype=torch.bool, device=h.device).tril()
    attn = (qk.masked_fill(~mask, -float("inf")) / head_dim ** 0.5).softmax(dim=-1)
    maps.append(attn[0].cpu().clone())  # (4, S, S)
    out = (attn @ v).transpose(1, 2).reshape(1, S, n_heads * head_dim) @ A.o_proj
    x = x + out
    x = x + layer.mlp(layer.norm2(x))
  if was_training:
    model.train()
  return torch.stack(maps)  # (n_layers, 4, S, S)

frames = []  # dict(step, loss, acc, maps)
frames.append({"step": 0, "loss": float("nan"), "acc": 0.0, "maps": attn_snapshot()})
for step in range(1, STEPS + 1):
  x, y = get_batch(trainx, trainy)
  opt.zero_grad(set_to_none=True)
  loss = loss_fn(model(x), y, config)
  loss.backward()
  opt.step()
  if step % SNAP_EVERY == 0:
    acc = evaluate(model, testx, testy, config, n=512)
    frames.append({"step": step, "loss": loss.item(), "acc": acc, "maps": attn_snapshot()})
    print(f"step {step:5d}  loss {loss.item():.4f}  acc {acc:6.2%}  (frame {len(frames)})")

print(f"\ncollected {len(frames)} frames; rendering GIF...")
n_layers = frames[0]["maps"].shape[0]

fig, axes = plt.subplots(n_layers, n_heads, figsize=(24, 12))
ims = [[None] * n_heads for _ in range(n_layers)]
for li in range(n_layers):
  for hh in range(n_heads):
    ax = axes[li, hh]
    ims[li][hh] = ax.imshow(frames[0]["maps"][li, hh], cmap="viridis",
                            vmin=0, vmax=1, animated=True)
    ax.set_title(f"layer {li} head {hh}", fontsize=10)
    ax.set_xticks(range(S)); ax.set_xticklabels(labels, fontsize=6)
    ax.set_yticks(range(S)); ax.set_yticklabels(labels, fontsize=6)
suptitle = fig.suptitle("", fontsize=14)

def update(fi):
  f = frames[fi]
  for li in range(n_layers):
    for hh in range(n_heads):
      ims[li][hh].set_data(f["maps"][li, hh])
  ls = "init" if fi == 0 else f"loss {f['loss']:.3f}"
  suptitle.set_text(f"attention on '3412+5879=...'  |  step {f['step']}  {ls}  acc {f['acc']:.0%}"
                    f"   (row=query, col=key)")
  return [im for row in ims for im in row] + [suptitle]

anim = FuncAnimation(fig, update, frames=len(frames), interval=200, blit=False)
anim.save("attn_timelapse.gif", writer=PillowWriter(fps=4), dpi=150)
print("saved -> attn_timelapse.gif")
