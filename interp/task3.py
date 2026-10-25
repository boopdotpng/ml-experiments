# task 3: two-key sum mod 10 (a2f5c9d3fc -> (5+9)%10 = 4) on two layers with one head each, plus an MLP after layer 2.
# attention can carry both values to the last position, but it can only copy tokens that already exist; the answer is a new digit,
# so something has to compute it. the MLP does. one head is enough in layer 2: a 50/50 blend of the two values loses which came first,
# and a sum doesn't care. also trains the attention-only models that fail, for comparison
import os
import numpy as np
from tinygrad import Tensor, nn
from lib.dataset import two_key_dataset
from lib.model import Model
from lib.analysis import *
Tensor.manual_seed(0) # reproducible training runs

STEPS, LR = 6000, 3e-3
FINAL = "2 layers, 1 head each, plus MLP"
SHAPES = {"2 layers, 1 head each": dict(n_layers=2, n_heads=1), "2 layers, 2 heads each": dict(n_layers=2, n_heads=2),
          FINAL: dict(n_layers=2, n_heads=1, mlp=True)}
xtrain, xtest, ytrain, ytest = two_key_dataset(op="sum")

if __name__ == "__main__":
  out = Out("task3")
  runs, models = {}, {}
  for name, shape in SHAPES.items():
    Tensor.manual_seed(0)
    models[name] = Model(seq_len=10, **shape)
    runs[name] = train(models[name], xtrain, ytrain, xtest, ytest, STEPS, LR, name)
  model = models[FINAL]
  os.makedirs("checkpoints", exist_ok=True)
  nn.state.safe_save(nn.state.get_state_dict(model), "checkpoints/task3.safetensors")

  loss_plot(out, runs, "two-key sum: training loss by model shape", FINAL)
  accuracy_plot(out, runs, "two-key sum: test accuracy by model shape", FINAL)
  s = two_key_analysis(out, model, "sum", xtest.numpy(), ytest.numpy(), "two-key sum")

  print(f"test accuracy {runs[FINAL][1]:.2f}%")
  print("last layer attention (value at q1, value at q2, elsewhere):", {k: tuple(round(float(x), 3) for x in v) for k, v in s["split"].items()})
  print("linear probes:", {f"{k[0]} / {k[1]}": round(float(v), 1) for k, v in s["probes"].items()})
  print("ablations:", {k: round(float(v), 1) for k, v in s["ablation"].items()})
  for k, mean, std, r_ans, r_a, r_b, r_raw in s["neurons_top"]: print(f"neuron {k}: std {std:.2f}, r2 answer {r_ans:.2f}, a {r_a:.2f}, b {r_b:.2f}, a+b {r_raw:.2f}")
