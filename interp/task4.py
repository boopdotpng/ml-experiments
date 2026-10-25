# task 4: two-key difference mod 10 (a2f5c9d3fc -> (5-9)%10 = 6) on two layers with two heads each, plus an MLP after layer 2.
# unlike the sum, order matters: a 50/50 blend of both values (what one head does in task 3) loses which was which.
# with two heads, each fetches one value. also trains the shapes around it for comparison, and runs the same analysis on
# the one-head model twice: seed 0, which grabs only the value at q2 and stalls around 42% (plots/task4/one_head), and seed 1, which
# finds a lopsided blend (about 1/3 a, 2/3 b) that the MLP can untangle, reaching about 97% (plots/task4/one_head_lucky).
# over 9 seeds, the one-head model took the lucky route 4 times (89-97%); two heads reached 100% every time
import os
import numpy as np
from tinygrad import Tensor, nn
from lib.dataset import two_key_dataset
from lib.model import Model
from lib.analysis import *
Tensor.manual_seed(0) # reproducible training runs

STEPS, LR = 6000, 3e-3
FINAL, ONE_HEAD, LUCKY = "2 layers, 2 heads each, plus MLP", "2 layers, 1 head each, plus MLP", "2 layers, 1 head each, plus MLP (seed 1)"
SEEDS = {LUCKY: 1}   # every other model trains from seed 0
SHAPES = {ONE_HEAD: dict(n_layers=2, n_heads=1, mlp=True), LUCKY: dict(n_layers=2, n_heads=1, mlp=True),
          "2 layers, 2 heads each, no MLP": dict(n_layers=2, n_heads=2),
          FINAL: dict(n_layers=2, n_heads=2, mlp=True), "3 layers, 1 head each, plus MLP": dict(n_layers=3, n_heads=1, mlp=True)}
xtrain, xtest, ytrain, ytest = two_key_dataset(op="diff")

if __name__ == "__main__":
  out = Out("task4")
  runs, models = {}, {}
  for name, shape in SHAPES.items():
    Tensor.manual_seed(SEEDS.get(name, 0))
    models[name] = Model(seq_len=10, **shape)
    runs[name] = train(models[name], xtrain, ytrain, xtest, ytest, STEPS, LR, name)
  model = models[FINAL]
  os.makedirs("checkpoints", exist_ok=True)
  nn.state.safe_save(nn.state.get_state_dict(model), "checkpoints/task4.safetensors")
  nn.state.safe_save(nn.state.get_state_dict(models[ONE_HEAD]), "checkpoints/task4_one_head.safetensors")

  loss_plot(out, runs, "two-key difference: training loss by model shape", FINAL)
  accuracy_plot(out, runs, "two-key difference: test accuracy by model shape", FINAL)
  X, Y = xtest.numpy(), ytest.numpy()
  s = two_key_analysis(out, model, "diff", X, Y, "two-key difference")
  s1 = two_key_analysis(Out("task4/one_head"), models[ONE_HEAD], "diff", X, Y, "two-key difference, one head per layer")
  s2 = two_key_analysis(Out("task4/one_head_lucky"), models[LUCKY], "diff", X, Y, "two-key difference, one head per layer (seed 1)")

  for name, st in ((FINAL, s), (ONE_HEAD, s1), (LUCKY, s2)):
    print(f"--- {name}: test accuracy {runs[name][1]:.2f}%")
    print("last layer attention (value at q1, value at q2, elsewhere):", {k: tuple(round(float(x), 3) for x in v) for k, v in st["split"].items()})
    print("linear probes:", {f"{k[0]} / {k[1]}": round(float(v), 1) for k, v in st["probes"].items()})
    print("ablations:", {k: round(float(v), 1) for k, v in st["ablation"].items()})
