"""Train and sample the tinygrad Tiny Shakespeare MoDE language model.

Examples:
  ../.venv/bin/python main.py train
  ../.venv/bin/python main.py sample "ROMEO:\n"

Configuration follows the other tinygrad experiments in this repository and is
set with environment variables, e.g. STEPS=10 BS=2 DIM=64.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from tinygrad import Context, Device, GlobalCounters, Tensor, TinyJit, nn
from tinygrad.helpers import getenv
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_save

from data import CharTokenizer, get_batch, load_corpus
from model import Model, ModelConfig, estimate_training_flops, parameter_counts


ROOT = Path(__file__).parent
DEFAULT_OUT = ROOT / "checkpoints"


def config_from_env(vocab_size: int) -> ModelConfig:
  return ModelConfig(
    vocab_size=vocab_size,
    max_seq_len=getenv("SEQ_LEN", 128),
    dim=getenv("DIM", 128),
    n_layers=getenv("LAYERS", 8),
    n_heads=getenv("HEADS", 8),
    n_experts=getenv("EXPERTS", 8),
    experts_per_token=getenv("TOPK_EXPERTS", 2),
    expert_hidden=getenv("EXPERT_HIDDEN", 256),
    mod_capacity=getenv("MOD_CAPACITY", 0.5),
  )


def human_flops(value: float) -> str:
  for unit, scale in (("EFLOPs", 1e18), ("PFLOPs", 1e15), ("TFLOPs", 1e12), ("GFLOPs", 1e9), ("MFLOPs", 1e6)):
    if value >= scale:
      return f"{value / scale:.3g} {unit}"
  return f"{value:.0f} FLOPs"


def lm_loss(logits: Tensor, targets: Tensor) -> Tensor:
  return logits.sparse_categorical_crossentropy(targets)


def total_loss(logits: Tensor, targets: Tensor, mod: Tensor, balance: Tensor, z_loss: Tensor):
  lm = lm_loss(logits, targets)
  total = (
    lm
    + getenv("MOE_BALANCE_WEIGHT", 0.01) * balance
    + getenv("MOE_Z_WEIGHT", 0.001) * z_loss
    + getenv("MOD_PREDICT_WEIGHT", 0.01) * mod
  )
  return total, lm


def save_artifacts(model: Model, tokenizer: CharTokenizer, out_dir: Path, metadata: dict | None = None):
  out_dir.mkdir(parents=True, exist_ok=True)
  safe_save(get_state_dict(model), str(out_dir / "model.safetensors"))
  tokenizer.save(out_dir / "vocab.json")
  payload = {"model": asdict(model.config), "metadata": metadata or {}}
  (out_dir / "config.json").write_text(json.dumps(payload, indent=2) + "\n")


def load_artifacts(out_dir: Path = DEFAULT_OUT):
  config_path, vocab_path, weights_path = (
    out_dir / "config.json", out_dir / "vocab.json", out_dir / "model.safetensors"
  )
  for path in (config_path, vocab_path, weights_path):
    if not path.exists():
      raise FileNotFoundError(f"missing {path}; train the model first")
  tokenizer = CharTokenizer.load(vocab_path)
  config = ModelConfig(**json.loads(config_path.read_text())["model"])
  model = Model(config)
  load_state_dict(model, safe_load(str(weights_path)), verbose=False)
  Tensor.realize(*get_parameters(model))
  return model, tokenizer


def print_model_budget(model: Model, batch_size: int, steps: int):
  total, active = parameter_counts(model)
  estimate = estimate_training_flops(model.config, batch_size, model.config.max_seq_len, steps)
  print(
    f"model total_params={total:,} approximate_active_params/token={active:,} "
    f"device={Device.DEFAULT}"
  )
  print(
    f"compute capacity={int(estimate['capacity_tokens'])}/{model.config.max_seq_len} "
    f"forward/sequence={human_flops(estimate['forward_per_sequence'])} "
    f"train/step={human_flops(estimate['train_per_step'])} "
    f"run={human_flops(estimate['train_run'])} tokens={int(estimate['tokens_seen']):,}"
  )
  print(
    f"MoD fraction of the same all-full top-{model.config.experts_per_token} MoE: "
    f"{estimate['fraction_of_full_moe']:.1%} "
    f"({1.0 - estimate['fraction_of_full_moe']:.1%} estimated saving)"
  )


def route_metrics(model: Model, tokens: Tensor) -> str:
  with Context(TRAINING=0):
    _, _, _, _, traces = model(tokens, route_mode="topk", collect_routes=True)
  expert_counts = np.zeros(model.config.n_experts, dtype=np.int64)
  agreements, threshold_rates = [], []
  for trace in traces:
    experts = trace["experts"].numpy()
    expert_counts += np.bincount(experts.reshape(-1), minlength=model.config.n_experts)
    if "mod_logits" in trace:
      logits = trace["mod_logits"].numpy()
      selected_positions = trace["positions"].numpy()
      target = np.zeros_like(logits, dtype=np.bool_)
      for batch in range(target.shape[0]):
        target[batch, selected_positions[batch]] = True
      predicted = logits > 0
      agreements.append((predicted == target).mean())
      threshold_rates.append(predicted.mean())
  usage = expert_counts / max(1, expert_counts.sum())
  return (
    f"expert_usage[min={usage.min():.1%} max={usage.max():.1%}] "
    f"mod_threshold[active={np.mean(threshold_rates):.1%} agreement={np.mean(agreements):.1%}]"
  )


def train():
  Tensor.manual_seed(getenv("SEED", 0))
  data_path = getenv("DATA", "") or None
  train_data, val_data, tokenizer, corpus_path = load_corpus(data_path)
  # Concrete buffers are required as TinyJit inputs.  Random window selection
  # happens on device inside each captured step.
  train_data, val_data = train_data.contiguous().realize(), val_data.contiguous().realize()

  config = config_from_env(tokenizer.vocab_size)
  steps, batch_size = getenv("STEPS", 3000), getenv("BS", 16)
  eval_every, eval_batches = getenv("EVAL_EVERY", 100), getenv("EVAL_BATCHES", 10)
  route_every = getenv("ROUTE_EVERY", 500)
  save_every = getenv("SAVE_EVERY", 500)
  out_dir = Path(getenv("OUT", str(DEFAULT_OUT)))

  model = Model(config)
  optimizer = AdamW(get_parameters(model), lr=getenv("LR", 3e-4), weight_decay=getenv("WEIGHT_DECAY", 0.01))
  # Realizing parameters and optimizer state before capture prevents one-time
  # initialization kernels from becoming part of the replayable training graph.
  Tensor.realize(*get_parameters(model), *get_parameters(optimizer))
  print(f"corpus={corpus_path} chars={train_data.shape[0] + val_data.shape[0]:,} vocab={tokenizer.vocab_size}")
  print_model_budget(model, batch_size, steps)

  @TinyJit
  @Context(TRAINING=1)
  def train_step(corpus: Tensor):
    x, y = get_batch(corpus, batch_size, config.max_seq_len)
    optimizer.zero_grad()
    logits, mod, balance, z_loss, _ = model(x, route_mode="topk")
    loss, language = total_loss(logits, y, mod, balance, z_loss)
    loss.backward()
    # schedule_step exposes optimizer mutations to the captured graph.  Returning
    # every metric keeps their reductions live without host synchronization.
    loss.realize(language, mod, balance, z_loss, *optimizer.schedule_step())
    return loss, language, mod, balance, z_loss

  @TinyJit
  @Context(TRAINING=0)
  def eval_step(corpus: Tensor):
    x, y = get_batch(corpus, batch_size, config.max_seq_len)
    logits, mod, balance, z_loss, _ = model(x, route_mode="topk")
    loss, language = total_loss(logits, y, mod, balance, z_loss)
    return loss.realize(), language.realize(), mod.realize(), balance.realize(), z_loss.realize()

  started, last_log = time.perf_counter(), time.perf_counter()
  last_step = 0
  for step in range(steps):
    loss, language, mod, balance, z_loss = train_step(train_data)
    if step % eval_every == 0 or step == steps - 1:
      vals = np.array([[float(metric.item()) for metric in eval_step(val_data)] for _ in range(eval_batches)])
      now = time.perf_counter()
      elapsed_steps = max(1, step - last_step)
      print(
        f"step {step:5d} train={loss.item():.4f} lm={language.item():.4f} "
        f"val={vals[:, 0].mean():.4f} ppl={math.exp(min(20, vals[:, 1].mean())):.2f} "
        f"mod={mod.item():.3f} balance={balance.item():.3f} z={z_loss.item():.3f} "
        f"{(now - last_log) * 1000 / elapsed_steps:.1f}ms/step"
      )
      last_log, last_step = now, step
    if route_every and (step % route_every == 0 or step == steps - 1):
      sample_x, _ = get_batch(val_data, min(batch_size, 4), config.max_seq_len)
      print("route", route_metrics(model, sample_x))
    if save_every and step > 0 and step % save_every == 0:
      save_artifacts(model, tokenizer, out_dir, {
        "step": step,
        "batch_size": batch_size,
        "elapsed_seconds": time.perf_counter() - started,
        "corpus": str(corpus_path),
      })
      print(f"checkpoint step={step} -> {out_dir}")

  metadata = {
    "steps": steps,
    "batch_size": batch_size,
    "elapsed_seconds": time.perf_counter() - started,
    "corpus": str(corpus_path),
  }
  save_artifacts(model, tokenizer, out_dir, metadata)
  print(f"saved -> {out_dir}")


def sample(prompt: str, max_new_tokens: int = 300):
  out_dir = Path(getenv("OUT", str(DEFAULT_OUT)))
  model, tokenizer = load_artifacts(out_dir)
  ids = tokenizer.encode(prompt)
  if not ids:
    raise ValueError("prompt must contain at least one character")
  temperature, top_k = getenv("TEMPERATURE", 0.8), getenv("SAMPLE_TOPK", 40)

  # Training has one fixed shape and one TinyJit.  Generation grows until the
  # rolling context is full, then this fixed-shape JIT replays.  A symbolic
  # sequence length is deliberately not used: MoD top-k capacity changes the
  # actual tensor shape, whereas Llama's symbolic start_pos keeps shapes fixed.
  @TinyJit
  @Context(TRAINING=0)
  def steady_sample_step(tokens: Tensor):
    return model(tokens, route_mode="threshold")[0][:, -1].realize()

  print(prompt, end="", flush=True)
  rng = np.random.default_rng(getenv("SEED", 0))
  for _ in range(max_new_tokens):
    context = ids[-model.config.max_seq_len:]
    tokens = Tensor([context]).contiguous().realize()
    with Context(TRAINING=0):
      logits = steady_sample_step(tokens) if len(context) == model.config.max_seq_len else model(tokens, route_mode="threshold")[0][:, -1].realize()
    values = logits.numpy()[0] / max(temperature, 1e-5)
    keep = min(top_k, values.shape[0])
    cutoff = np.partition(values, -keep)[-keep]
    values[values < cutoff] = -np.inf
    probs = np.exp(values - np.max(values))
    probs /= probs.sum()
    next_id = int(rng.choice(len(probs), p=probs))
    ids.append(next_id)
    print(tokenizer.decode([next_id]), end="", flush=True)
  print()


def usage():
  print("usage: ../.venv/bin/python main.py [train|sample [PROMPT]]")
  print("configuration: STEPS BS LR SEQ_LEN DIM LAYERS HEADS EXPERTS TOPK_EXPERTS EXPERT_HIDDEN MOD_CAPACITY")


if __name__ == "__main__":
  command = sys.argv[1] if len(sys.argv) > 1 else "train"
  if command == "train":
    train()
  elif command == "sample":
    sample(sys.argv[2] if len(sys.argv) > 2 else "ROMEO:\n", getenv("MAX_NEW_TOKENS", 300))
  else:
    usage()
    raise SystemExit(2)
