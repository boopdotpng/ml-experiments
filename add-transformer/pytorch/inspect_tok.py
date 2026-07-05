"""Inspect the learned TOKEN embeddings of the trained model."""
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import load_model
from dataset import get_dataset_config

config = get_dataset_config()
chars = config.chars  # ['0'..'9','+','=','W','C','A']
model = load_model()
tok = model.tok_emb.detach().cpu()  # (15, 32)
print("tok_emb shape:", tuple(tok.shape), " chars:", chars)

# Per-token norm
print("\nper-token L2 norm:")
for c, n in zip(chars, tok.norm(dim=-1).tolist()):
  print(f"  [{c}]  |v|={n:.3f}")

unit = F.normalize(tok, dim=-1)
cos = unit @ unit.T  # (15, 15)

# --- Digit-only structure -------------------------------------------------
digit_idx = list(range(10))
D = tok[digit_idx]                       # (10, 32)
Dc = D - D.mean(0, keepdim=True)
U, S, Vh = torch.linalg.svd(Dc, full_matrices=False)
proj = Dc @ Vh[:2].T                      # (10, 2)
var = (S[:2] ** 2 / (S ** 2).sum()).tolist()
pc1 = proj[:, 0]

vals = torch.arange(10, dtype=torch.float)
# correlation of PC1 with digit magnitude (number-line test)
def corr(a, b):
  a = a - a.mean(); b = b - b.mean()
  return (a @ b / (a.norm() * b.norm())).item()
print(f"\ndigit PCA: PC1 var={var[0]:.0%} PC2 var={var[1]:.0%}")
print(f"corr(PC1, digit value)        = {corr(pc1, vals):+.3f}  (|~1| => linear number line)")
print(f"corr(PC1, sin(2pi*d/10))      = {corr(pc1, torch.sin(2*torch.pi*vals/10)):+.3f}")
print(f"corr(PC2, cos(2pi*d/10))      = {corr(proj[:,1], torch.cos(2*torch.pi*vals/10)):+.3f}")

# mean cosine sim of digit pairs as a function of |i-j| (number-line smoothness)
print("\nmean digit-digit cos sim by value gap |i-j|:")
for gap in range(1, 10):
  pairs = [cos[i, j].item() for i in range(10) for j in range(10) if j - i == gap]
  print(f"  gap {gap}: {sum(pairs)/len(pairs):+.3f}")

# --- Plots ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
im = axes[0].imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
axes[0].set_xticks(range(15)); axes[0].set_xticklabels(chars)
axes[0].set_yticks(range(15)); axes[0].set_yticklabels(chars)
axes[0].set_title("token embedding cosine similarity")
fig.colorbar(im, ax=axes[0], fraction=0.046)

axes[1].plot(proj[:, 0], proj[:, 1], "-", color="lightgray", zorder=1)
sc = axes[1].scatter(proj[:, 0], proj[:, 1], c=range(10), cmap="viridis", s=120, zorder=2)
for d, (x, y) in enumerate(proj.tolist()):
  axes[1].annotate(str(d), (x, y), fontsize=13, weight="bold")
axes[1].set_title(f"digit embeddings, PCA (var {var[0]:.0%}, {var[1]:.0%})\nline=value order 0->9")
fig.colorbar(sc, ax=axes[1], fraction=0.046, label="digit value")
plt.tight_layout()
plt.savefig("tok_emb.png", dpi=220)
print("\nsaved -> tok_emb.png")
