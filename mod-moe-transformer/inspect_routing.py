"""Export and print per-token Mixture-of-Depths / expert routing traces."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from tinygrad import Context, Tensor
from tinygrad.helpers import getenv

from main import DEFAULT_OUT, load_artifacts


def routing_trace(model, tokenizer, prompt: str, route_mode: str):
  prompt = prompt[-model.config.max_seq_len:]
  token_ids = tokenizer.encode(prompt)
  tokens = Tensor([token_ids]).contiguous().realize()
  with Context(TRAINING=0):
    _, _, _, _, traces = model(tokens, route_mode=route_mode, collect_routes=True)

  depth = np.zeros(len(token_ids), dtype=np.int64)
  token_experts: list[list[dict]] = [[] for _ in token_ids]
  layers = []
  for layer_index, trace in enumerate(traces):
    routed = "mod_logits" in trace
    positions = trace["positions"].numpy()[0].astype(int)
    experts = trace["experts"].numpy()[0].astype(int)
    active = trace["mod_active"].numpy()[0].astype(bool) if routed else np.ones(len(positions), dtype=bool)
    active_positions = positions[active]
    depth[active_positions] += 1

    for compact_index, position in enumerate(positions):
      if active[compact_index]:
        token_experts[position].append({
          "layer": layer_index + 1,
          "experts": experts[compact_index].tolist(),
        })

    all_experts = experts[active].reshape(-1)
    usage = np.bincount(all_experts, minlength=model.config.n_experts)
    layer = {
      "layer": layer_index + 1,
      "kind": "mod" if routed else "full",
      "active_positions": active_positions.tolist(),
      "expert_ids": experts[active].tolist(),
      "expert_counts": usage.tolist(),
    }
    if routed:
      logits = trace["mod_logits"].numpy()[0]
      selected = np.zeros(len(token_ids), dtype=bool)
      selected[active_positions] = True
      threshold = logits > 0
      layer.update({
        "router_logits": logits.tolist(),
        "threshold_active_fraction": float(threshold.mean()),
        "threshold_topk_agreement": float((threshold == selected).mean()),
      })
    layers.append(layer)

  return {
    "prompt": prompt,
    "route_mode": route_mode,
    "capacity": model.config.mod_capacity,
    "tokens": [
      {
        "position": i,
        "character": char,
        "token_id": token_ids[i],
        "depth": int(depth[i]),
        "expert_routes": token_experts[i],
      }
      for i, char in enumerate(prompt)
    ],
    "layers": layers,
  }


def print_report(trace: dict):
  print(f"prompt={trace['prompt']!r}")
  print(f"routing={trace['route_mode']} mod_capacity={trace['capacity']:.1%}")
  for layer in trace["layers"]:
    counts = layer["expert_counts"]
    usage = np.array(counts) / max(1, sum(counts))
    suffix = ""
    if layer["kind"] == "mod":
      suffix = (
        f" threshold_active={layer['threshold_active_fraction']:.1%}"
        f" agreement={layer['threshold_topk_agreement']:.1%}"
      )
    print(
      f"L{layer['layer']:02d} {layer['kind']:4s} tokens={len(layer['active_positions']):3d} "
      f"experts[min={usage.min():.1%} max={usage.max():.1%} counts={counts}]{suffix}"
    )

  print("\nposition  char  depth  expert path")
  for token in trace["tokens"]:
    path = " ".join(
      f"L{route['layer']}:{','.join(map(str, route['experts']))}"
      for route in token["expert_routes"]
    )
    print(f"{token['position']:8d}  {token['character']!r:4s}  {token['depth']:5d}  {path}")


def main():
  out_dir = Path(getenv("OUT", str(DEFAULT_OUT)))
  model, tokenizer = load_artifacts(out_dir)
  prompt = sys.argv[1] if len(sys.argv) > 1 else "ROMEO:\nBut, soft! what light through yonder window breaks?"
  route_mode = getenv("ROUTE_MODE", "topk")
  trace = routing_trace(model, tokenizer, prompt, route_mode)
  print_report(trace)
  output = Path(getenv("TRACE_OUT", "routing_trace.json"))
  output.write_text(json.dumps(trace, indent=2) + "\n")
  print(f"\nwrote {output}")


if __name__ == "__main__":
  main()
