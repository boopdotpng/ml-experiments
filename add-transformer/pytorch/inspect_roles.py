"""Do same-ROLE positions cluster? (all W slots, all C slots, etc.)"""
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import load_model

model = load_model()
pos = model.pos_emb.detach().cpu()  # (32, 32)
unit = F.normalize(pos, dim=-1)
cos = unit @ unit.T

# Role of each position in the fixed template.
# AAAA+BBBB=  W?C?W?C?W?C?W?C?  A  ?????
roles = (["opA"] * 4 + ["+"] + ["opB"] * 4 + ["="]
         + ["W", "Wd", "C", "Cd"] * 4
         + ["A"] + ["ans"] * 5)
assert len(roles) == 32

def avg_cos(role_a, role_b):
  ia = [i for i, r in enumerate(roles) if r == role_a]
  ib = [i for i, r in enumerate(roles) if r == role_b]
  vals = [cos[i, j].item() for i in ia for j in ib if i != j]
  return sum(vals) / len(vals) if vals else float("nan")

print("mean within-role cosine similarity (vs ~0.0 baseline for random):")
for r in ["opA", "opB", "W", "Wd", "C", "Cd", "ans"]:
  print(f"  {r:4s} (n={roles.count(r)})  within={avg_cos(r, r):+.3f}")

print("\nW-slot vs C-slot cross:", f"{avg_cos('W', 'C'):+.3f}")
print("opA vs opB cross:       ", f"{avg_cos('opA', 'opB'):+.3f}")

# Visualize: reorder the cos matrix so same-role positions are adjacent.
order = sorted(range(32), key=lambda i: (roles[i], i))
reordered = cos[order][:, order]
lbls = [f"{i}:{roles[i]}" for i in order]

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(reordered, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(32)); ax.set_xticklabels(lbls, fontsize=6, rotation=90)
ax.set_yticks(range(32)); ax.set_yticklabels(lbls, fontsize=6)
ax.set_title("cos sim, positions grouped by ROLE\n(bright blocks on diagonal = role clustering)")
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("pos_roles.png", dpi=220)
print("\nsaved -> pos_roles.png")
