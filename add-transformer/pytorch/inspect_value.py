"""Interp of the VALUE path: v = x@v_proj, attn@v, and per-head o_proj contribution."""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import load_model, n_heads, head_dim
from dataset import get_dataset_config, encode, format_example

config = get_dataset_config()
model = load_model()
model.eval()
device = next(model.parameters()).device

a, b = 3412, 5879
full = format_example(a, b)
x_str = full[:-1]
labels = list(x_str)
ids = torch.tensor([encode(x_str, config)], dtype=torch.long, device=device)
S = ids.shape[1]
LAYER = 1  # inspect the layer nearest the output

cap = {}

@torch.no_grad()
def run():
  x = model.tok_emb[ids] + model.pos_emb[:S]
  for li, layer in enumerate(model.layers):
    h = layer.norm1(x)
    A = layer.attn
    q = (h @ A.q_proj).reshape(1, S, n_heads, head_dim).transpose(1, 2)
    k = (h @ A.k_proj).reshape(1, S, n_heads, head_dim).transpose(1, 2)
    v = (h @ A.v_proj).reshape(1, S, n_heads, head_dim).transpose(1, 2)  # (1,4,S,8)
    qk = q @ k.transpose(-2, -1)
    mask = torch.ones(S, S, dtype=torch.bool, device=h.device).tril()
    attn = (qk.masked_fill(~mask, -float("inf")) / head_dim ** 0.5).softmax(dim=-1)
    a_heads = attn @ v                                  # (1,4,S,8) per-head attended value
    merged = a_heads.transpose(1, 2).reshape(1, S, n_heads * head_dim)
    out = merged @ A.o_proj                             # (1,S,32) full attn output
    if li == LAYER:
      # per-head contribution to the residual: keep one head's 8 dims, zero rest, then o_proj
      contrib = []
      for hh in range(n_heads):
        m = torch.zeros_like(merged)
        m[:, :, hh * head_dim:(hh + 1) * head_dim] = merged[:, :, hh * head_dim:(hh + 1) * head_dim]
        contrib.append((m @ A.o_proj)[0])              # (S,32)
      cap["v"] = v[0]            # (4,S,8)
      cap["av"] = a_heads[0]     # (4,S,8)
      cap["contrib"] = torch.stack(contrib)  # (4,S,32)
    x = x + out
    x = x + layer.mlp(layer.norm2(x))

run()

# ---- figure: 3 rows x 4 heads --------------------------------------------
fig, axes = plt.subplots(3, n_heads, figsize=(22, 14))
for hh in range(n_heads):
  # row 0: value vectors v (S x 8)
  ax = axes[0, hh]
  ax.imshow(cap["v"][hh].cpu(), cmap="RdBu_r", aspect="auto",
            vmin=-cap["v"].abs().max(), vmax=cap["v"].abs().max())
  ax.set_title(f"head {hh}: V = x@v_proj  (S x 8)", fontsize=9)
  ax.set_yticks(range(S)); ax.set_yticklabels(labels, fontsize=5)
  ax.set_xlabel("head_dim", fontsize=7)

  # row 1: attended value attn@v (S x 8)
  ax = axes[1, hh]
  ax.imshow(cap["av"][hh].cpu(), cmap="RdBu_r", aspect="auto",
            vmin=-cap["av"].abs().max(), vmax=cap["av"].abs().max())
  ax.set_title(f"head {hh}: attn@V  (S x 8)", fontsize=9)
  ax.set_yticks(range(S)); ax.set_yticklabels(labels, fontsize=5)
  ax.set_xlabel("head_dim", fontsize=7)

  # row 2: per-head contribution to residual after o_proj (S x 32)
  ax = axes[2, hh]
  c = cap["contrib"][hh].cpu()
  ax.imshow(c, cmap="RdBu_r", aspect="auto", vmin=-c.abs().max(), vmax=c.abs().max())
  ax.set_title(f"head {hh}: contribution to residual (S x 32)", fontsize=9)
  ax.set_yticks(range(S)); ax.set_yticklabels(labels, fontsize=5)
  ax.set_xlabel("residual dim (32)", fontsize=7)

axes[0, 0].set_ylabel("query position", fontsize=9)
axes[1, 0].set_ylabel("query position", fontsize=9)
axes[2, 0].set_ylabel("query position", fontsize=9)
fig.suptitle(f"layer {LAYER} value path for '{x_str}'  (v_proj -> attn@v -> o_proj)", fontsize=13)
plt.tight_layout()
plt.savefig("value_path.png", dpi=220)
print("saved -> value_path.png")

# ---- text: per-head contribution magnitude at each position --------------
norms = cap["contrib"].norm(dim=-1)  # (4, S)
print(f"\nexample: {full}")
print("\nper-head residual-contribution L2 norm at each position (layer", LAYER, "):")
print("pos/tok   " + "  ".join(f"h{hh}" for hh in range(n_heads)))
for i in range(S):
  print(f"  {i:2d}:{labels[i]}    " + "  ".join(f"{norms[hh,i]:.2f}" for hh in range(n_heads)))

# Confirm carry-feeder: head 3 output at each W should match its value at W-1.
print("\nhead 3 check: cos(attn@V at W_pos , V at (W_pos - 1)) -- expect ~1 if copying prev token")
import torch.nn.functional as F
for wpos in [10, 14, 18, 22]:
  out_w = cap["av"][3, wpos]
  v_prev = cap["v"][3, wpos - 1]
  cos = F.cosine_similarity(out_w, v_prev, dim=0).item()
  print(f"  W at pos {wpos}: cos(attn@V[{wpos}], V[{wpos-1}:{labels[wpos-1]}]) = {cos:+.3f}")
