"""Inspect the learned positional embeddings of the trained model."""
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import load_model
from dataset import get_dataset_config

config = get_dataset_config()
model = load_model()
pos = model.pos_emb.detach().cpu()  # (32, 32)
print("pos_emb shape:", tuple(pos.shape))

# Token string for the canonical example, to label each position.
labels = list("3412+5879=W1C1W9C0W2C1W9C0A09291")  # 32 tokens
assert len(labels) == pos.shape[0]

# Per-position L2 norm
norms = pos.norm(dim=-1)
print("\nper-position L2 norm:")
for i, (c, n) in enumerate(zip(labels, norms.tolist())):
  print(f"  pos {i:2d} [{c}]  |v|={n:.3f}")

# Cosine-similarity matrix between positions
unit = F.normalize(pos, dim=-1)
cos = unit @ unit.T  # (32, 32)

# Adjacent-position cosine similarity (is there a smooth ramp?)
adj = torch.diagonal(cos, offset=1)
print("\nadjacent cos sim (pos i vs i+1): mean=%.3f min=%.3f max=%.3f" %
      (adj.mean(), adj.min(), adj.max()))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

im0 = axes[0].imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
axes[0].set_title("cosine similarity between position embeddings")
axes[0].set_xticks(range(32)); axes[0].set_xticklabels(labels, fontsize=7)
axes[0].set_yticks(range(32)); axes[0].set_yticklabels(labels, fontsize=7)
axes[0].set_xlabel("position"); axes[0].set_ylabel("position")
fig.colorbar(im0, ax=axes[0], fraction=0.046)

# PCA to 2D for a scatter of the 32 points
pos_c = pos - pos.mean(0, keepdim=True)
U, S, Vh = torch.linalg.svd(pos_c, full_matrices=False)
proj = pos_c @ Vh[:2].T  # (32, 2)
var = (S[:2] ** 2 / (S ** 2).sum()).tolist()
axes[1].plot(proj[:, 0], proj[:, 1], "-", color="lightgray", zorder=1)
sc = axes[1].scatter(proj[:, 0], proj[:, 1], c=range(32), cmap="viridis", zorder=2)
for i, (x, y) in enumerate(proj.tolist()):
  axes[1].annotate(f"{i}:{labels[i]}", (x, y), fontsize=7)
axes[1].set_title(f"PCA of position embeddings (var {var[0]:.0%}, {var[1]:.0%})")
fig.colorbar(sc, ax=axes[1], fraction=0.046, label="position index")

plt.tight_layout()
plt.savefig("pos_emb.png", dpi=220)
print("\nsaved -> pos_emb.png")
