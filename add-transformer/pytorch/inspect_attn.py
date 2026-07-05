"""Pull real attention patterns out of the trained model for one example."""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import main as M
from main import load_model, n_heads, head_dim
from dataset import get_dataset_config, encode, format_example

config = get_dataset_config()
model = load_model()
model.eval()
device = next(model.parameters()).device

a, b = 3412, 5879
full = format_example(a, b)          # 32-char string
x_str = full[:-1]                    # 31 tokens (teacher-forced input)
labels = list(x_str)
ids = torch.tensor([encode(x_str, config)], dtype=torch.long, device=device)  # (1, 31)
S = ids.shape[1]

# Manually run the forward, capturing per-layer attention weights.
attn_maps = []  # list of (n_heads, S, S) per layer

@torch.no_grad()
def run():
  x = model.tok_emb[ids] + model.pos_emb[:S]   # (1, S, 32)
  for layer in model.layers:
    h = layer.norm1(x)                          # RMSNorm
    A = layer.attn
    B_, Sl, _ = h.shape
    q = (h @ A.q_proj).reshape(B_, Sl, n_heads, head_dim).transpose(1, 2)
    k = (h @ A.k_proj).reshape(B_, Sl, n_heads, head_dim).transpose(1, 2)
    v = (h @ A.v_proj).reshape(B_, Sl, n_heads, head_dim).transpose(1, 2)
    qk = q @ k.transpose(-2, -1)
    mask = torch.ones(Sl, Sl, dtype=torch.bool, device=h.device).tril()
    qk = qk.masked_fill(~mask, -float("inf"))
    attn = (qk / head_dim ** 0.5).softmax(dim=-1)   # (1, 4, S, S)
    attn_maps.append(attn[0].clone())
    out = (attn @ v).transpose(1, 2).reshape(B_, Sl, n_heads * head_dim) @ A.o_proj
    x = x + out
    x = x + layer.mlp(layer.norm2(x))

run()

fig, axes = plt.subplots(len(model.layers), n_heads, figsize=(22, 11))
for L, amap in enumerate(attn_maps):
  for hh in range(n_heads):
    ax = axes[L, hh]
    ax.imshow(amap[hh].cpu(), cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_title(f"layer {L} head {hh}", fontsize=10)
    ax.set_xticks(range(S)); ax.set_xticklabels(labels, fontsize=5)
    ax.set_yticks(range(S)); ax.set_yticklabels(labels, fontsize=5)
    if hh == 0:
      ax.set_ylabel("query (row)", fontsize=8)
    if L == len(model.layers) - 1:
      ax.set_xlabel("key (col)", fontsize=8)

fig.suptitle(f"attention for '{x_str}'   (row=query attends to col=key)", fontsize=13)
plt.tight_layout()
plt.savefig("attn.png", dpi=220)
print("saved -> attn.png")

# Text summary: for each generated position, which key does each head focus on most?
print(f"\nexample: {full}")
print("positions:", " ".join(f"{i}:{c}" for i, c in enumerate(labels)))
print("\nfor selected query positions, top-attended key per head (layer 1):")
amap = attn_maps[1]
for qi in range(config.prompt_len - 1, S):   # generated region
  tops = []
  for hh in range(n_heads):
    j = int(amap[hh, qi].argmax())
    tops.append(f"h{hh}->{j}:{labels[j]}({amap[hh,qi,j]:.2f})")
  print(f"  q{qi:2d}:{labels[qi]}  " + "  ".join(tops))
