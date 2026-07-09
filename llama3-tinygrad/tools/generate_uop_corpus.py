#!/usr/bin/env python3
"""Generate a hardware-free, pre-codegen UOp corpus for steady-state Llama decode.

This intentionally never realizes a Tensor.  Empty buffers stand in for model
weights, RoPE tables, the token input, and the already-populated KV caches.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
DEFAULT_TINYGRAD = REPO.parent / "tiny"
os.environ["DEV"] = "CPU"  # storage tag only; this graph is captured before backend lowering.

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tinygrad", type=Path, default=Path(os.environ.get("TINYGRAD_ROOT", DEFAULT_TINYGRAD)))
parser.add_argument("--out", type=Path, default=REPO / "corpus" / "decode")
parser.add_argument("--start-pos", type=int, default=4, help="bound value used for the symbolic decode position")
args = parser.parse_args()

if not (args.tinygrad / "tinygrad" / "__init__.py").is_file():
  raise SystemExit(f"tinygrad checkout not found at {args.tinygrad}; pass --tinygrad or set TINYGRAD_ROOT")
sys.path.insert(0, str(args.tinygrad))
sys.path.insert(0, str(REPO))

import main as llama  # noqa: E402
from tinygrad import Tensor, Variable, dtypes  # noqa: E402
from tinygrad.callify import transform_to_call  # noqa: E402
from tinygrad.device import Device  # noqa: E402
from tinygrad.helpers import Context  # noqa: E402
from tinygrad.nn.state import get_state_dict, load_state_dict  # noqa: E402
from tinygrad.schedule import create_schedule  # noqa: E402
from tinygrad.schedule.rangeify import get_kernel_graph  # noqa: E402
from tinygrad.uop.ops import Ops, ProgramInfo, UOp  # noqa: E402


@dataclass(frozen=True)
class Launch:
  name: str
  intent: str
  layer: int | None
  deps: tuple[int, ...]
  anchor_ids: tuple[int, ...]


def git_revision(path: Path) -> str:
  try:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    return "unknown"


def dtype_name(dtype) -> str:
  return repr(dtype).replace("dtypes.", "")


def format_dim(dim) -> str:
  return dim.render() if isinstance(dim, UOp) else str(dim)


def format_shape(uop: UOp) -> str:
  try:
    shape = uop.shape
  except (AttributeError, RuntimeError, ValueError):
    return "?"
  if not shape: return "()"
  return "(" + ", ".join(format_dim(x) for x in shape) + ("," if len(shape) == 1 else "") + ")"


def format_arg(uop: UOp) -> str:
  if uop.op is Ops.REDUCE:
    return f"combine={uop.arg[0].name} axes={uop.arg[1]}"
  if uop.arg is None: return ""
  ret = repr(uop.arg).replace("\n", " ")
  return ret if len(ret) <= 140 else ret[:137] + "..."


def op_histogram(nodes: list[UOp]) -> str:
  counts = Counter(x.op.name for x in nodes)
  return "\n".join(f"  {count:5d}  {name}" for name, count in counts.most_common())


def dump_nodes(nodes: list[UOp], global_ids: dict[UOp, int], boundaries: dict[UOp, int] | None = None) -> str:
  local_ids = {u: i for i, u in enumerate(nodes)}
  lines = []
  for u in nodes:
    srcs = []
    for src in u.src:
      if src in local_ids: srcs.append(f"n{local_ids[src]:04d}")
      elif boundaries is not None and src in boundaries: srcs.append(f"anchor:{boundaries[src]:03d}")
      elif src in global_ids: srcs.append(f"g{global_ids[src]:04d}")
      else: srcs.append("external")
    arg = format_arg(u)
    lines.append(f"n{local_ids[u]:04d}  g{global_ids[u]:04d}  {u.op.name:<14} "
                 f"shape={format_shape(u):<28} dtype={dtype_name(u.dtype):<14}"
                 f"{(' arg=' + arg) if arg else ''}{(' <- ' + ', '.join(srcs)) if srcs else ''}")
  return "\n".join(lines)


def anchor_slice(roots: tuple[UOp, ...], anchors: dict[UOp, int], topo: list[UOp]) -> tuple[list[UOp], tuple[int, ...]]:
  included: set[UOp] = set()
  deps: set[int] = set()
  stack = list(roots)
  root_set = set(roots)
  while stack:
    u = stack.pop()
    if u in included: continue
    if u in anchors and u not in root_set:
      deps.add(anchors[u])
      continue
    included.add(u)
    stack.extend(u.src)
  return [u for u in topo if u in included], tuple(sorted(deps))


def build_launches(n_layers: int, anchors: list[UOp]) -> list[Launch]:
  # These are current tinygrad's launch boundaries. Some are split reductions and
  # some fuse several pristine REDUCE nodes; anchor_ids point into raw_graph.txt.
  launches = [
    Launch("embedding_partial", "split embedding lookup: 501 vocabulary rows per partial", None, (), (0, 1)),
    Launch("embedding_finish", "finish the 256-way split embedding reduction", None, (0,), (1,)),
  ]
  for layer in range(n_layers):
    prev = 1 if layer == 0 else 2 + 14 * (layer - 1) + 13
    k = len(launches)
    a = 2 + 14 * layer
    launches += [
      Launch("input_rmsnorm", "FP32 mean-square, epsilon, sqrt, reciprocal", layer, (prev,), (a,)),
      Launch("k_projection", "RMS-normalized hidden @ K weight; FP32 accumulation, BF16 store", layer, (k,), (a+2,)),
      Launch("q_projection", "RMS-normalized hidden @ Q weight; FP32 accumulation, BF16 store", layer, (k,), (a+1,)),
      Launch("v_projection_k_rope_kv_store", "fused V projection, K RoPE, and K/V cache update", layer, (k, k+1), (a+2, a+3, a+4)),
      Launch("q_rope", "apply rotary embedding to Q at symbolic start_pos", layer, (k+2,), (a+1,)),
      Launch("attention_scores", "Q @ K-history transpose and scale by 0.125", layer, (k+3, k+4), (a+5,)),
      Launch("softmax_max", "maximum attention score over history", layer, (k+5,), (a+6,)),
      Launch("softmax_sum", "sum exp2(score-max) over history", layer, (k+5, k+6), (a+7,)),
      Launch("softmax_normalize", "materialize exp2(score-max) / sum", layer, (k+5, k+6, k+7), (a+6, a+7)),
      Launch("attention_values", "softmax probabilities @ V-history", layer, (k+3, k+8), (a+8,)),
      Launch("output_projection_residual", "attention output projection plus residual", layer, (prev, k+9), (a+9,)),
      Launch("post_attention_rmsnorm", "FP32 mean-square, epsilon, sqrt, reciprocal", layer, (k+10,), (a+10,)),
      Launch("gate_up_silu", "fused gate/up projections and SiLU(gate)*up", layer, (k+11,), (a+11, a+12)),
      Launch("down_projection_residual", "MLP down projection plus residual", layer, (k+10, k+12), (a+13,)),
    ]
  final_a = 2 + 14 * n_layers
  prev = 2 + 14 * (n_layers - 1) + 13
  k = len(launches)
  launches += [
    Launch("final_rmsnorm", "final FP32 RMSNorm reduction", None, (prev,), (final_a,)),
    Launch("vocab_projection", "hidden @ tied embedding transpose", None, (k,), (final_a+1,)),
    Launch("argmax_partial_max", "first 501-way chunked max over vocabulary", None, (k+1,), (final_a+2,)),
    Launch("argmax_candidates", "select candidate indices and reduce each 501-wide chunk", None, (k+1, k+2), (final_a+2, final_a+3)),
    Launch("argmax_finish", "final max over 256 candidate indices", None, (k+3,), (final_a+4,)),
  ]
  if max(a for x in launches for a in x.anchor_ids) >= len(anchors):
    raise RuntimeError(f"launch annotation expects more than {len(anchors)} semantic anchors")
  return launches


def scheduled_summary(call: UOp) -> tuple[str, str, int]:
  ast = call.src[0]
  if ast.op is not Ops.SINK or ast.arg.applied_opts != ():
    raise RuntimeError("launch witness must be an unoptimized SINK")
  info = ProgramInfo.from_sink(ast)
  params = {u.arg.slot: u for u in ast.toposort() if u.op is Ops.PARAM and u.arg.slot >= 0}
  outputs = ", ".join(f"{params[i].max_numel()}x{dtype_name(params[i].dtype.base)}" for i in info.outs)
  reductions = []
  for u in ast.toposort():
    if u.op is Ops.REDUCE:
      ranges = "x".join(format_dim(r.src[0].arg if r.src[0].op is Ops.CONST else r.src[0]) for r in u.src[1:]) or "scalar"
      reductions.append(f"{u.arg[0].name}[{ranges}]")
  return outputs or "in-place", ",".join(reductions) or "none", len(ast.toposort())


def main() -> None:
  with Context(ALLOW_DEVICE_USAGE=0):
    # Replace every data source with an unallocated BUFFER identity.  In particular,
    # do not call safe_load, Tensor.realize, TinyJit, renderer code, or a runtime.
    llama.COS = Tensor.empty(llama.max_seq_len, llama.head_dim, dtype=dtypes.float32)
    llama.SIN = Tensor.empty(llama.max_seq_len, llama.head_dim, dtype=dtypes.float32)
    model = llama.Model()
    state = get_state_dict(model)
    load_state_dict(model, {name: Tensor.empty(*tensor.shape, dtype=dtypes.bfloat16)
                            for name, tensor in state.items()}, verbose=False, realize=False)
    for layer in model.layers:
      layer.self_attn.kv_cache = Tensor.empty(2, 1, llama.n_kv_heads, llama.max_seq_len,
                                               llama.head_dim, dtype=dtypes.bfloat16)
    token = Tensor.empty(1, 1, dtype=dtypes.int32)
    start_pos = Variable("start_pos", 1, llama.max_seq_len-1).bind(args.start_pos)
    output = model.forward(token, start_pos)

    raw_root = UOp.sink(output.uop)
    raw_topo = list(raw_root.toposort())
    global_ids = {u: i for i, u in enumerate(raw_topo)}
    anchor_nodes = [u for u in raw_topo if u.op in (Ops.REDUCE, Ops.STORE)]
    anchor_ids = {u: i for i, u in enumerate(anchor_nodes)}

    # This is used only to witness launch partition/order.  It stops at SINK ASTs:
    # no Kernel/full_rewrite, renderer, BEAM, linearizer, compilation, or execution.
    call, _ = transform_to_call(raw_root)
    linear = create_schedule(get_kernel_graph(call.src[0]))

    if hardware_devices := Device._opened_devices - {"PYTHON"}:
      raise RuntimeError(f"corpus generation unexpectedly opened hardware devices: {sorted(hardware_devices)}")

  launches = build_launches(llama.n_layers, anchor_nodes)
  if len(linear.src) != len(launches):
    raise RuntimeError(f"tinygrad scheduled {len(linear.src)} launches, annotations describe {len(launches)}; update the corpus generator")

  out = args.out.resolve()
  if out.exists(): shutil.rmtree(out)
  kernels = out / "kernels"
  kernels.mkdir(parents=True)
  revision = git_revision(args.tinygrad)

  common = (f"tinygrad_commit: {revision}\nmodel: Llama-3.2-1B-Instruct decode, B=1 S=1 start_pos=Variable[1,8191] bound to {args.start_pos}\n"
            "capture: lazy Tensor UOps; CPU is a storage tag only; no backend lowering; no hardware device opened; no Tensor realized\n")

  graph_lines = [
    "LLAMA DECODE PRISTINE TENSOR UOP GRAPH",
    "========================================",
    common.rstrip(),
    "",
    "This is the source-of-truth lowering graph, captured before transform_to_call,",
    "create_schedule, run_rangeify, Kernel/full_rewrite, renderer optimization, BEAM,",
    "linearization, compilation, or execution. Launch boundaries do not yet exist here.",
    "",
    f"nodes: {len(raw_topo)}",
    f"semantic anchors (REDUCE or explicit STORE): {len(anchor_nodes)}",
    "",
    "OP HISTOGRAM",
    op_histogram(raw_topo),
    "",
    "TOPOLOGICAL GRAPH",
    "Each row is: local/global id, op, logical shape, dtype, arg, sources.",
    dump_nodes(raw_topo, global_ids),
    "",
  ]
  (out / "raw_graph.txt").write_text("\n".join(graph_lines))

  manifest = [
    "LLAMA DECODE LAUNCH MANIFEST",
    "============================",
    common.rstrip(),
    "",
    "The scheduler creates launch boundaries during rangeification, so no object can",
    "simultaneously be both a pristine Tensor graph and an actual per-launch AST.",
    "Each kernel file therefore contains (1) the actual launch role/order witnessed",
    "at the unoptimized SINK boundary and (2) a slice of the pristine semantic graph.",
    "The rangeified SINK itself is summarized, never dumped as lowering input.",
    "Host-to-device token upload, weight loading, RoPE construction, KV allocation,",
    "prefill, renderer/codegen passes, and TinyJit replay are intentionally excluded.",
    "",
    f"compute launches: {len(launches)}",
    f"raw graph nodes: {len(raw_topo)}",
    "",
    "id   layer  role                              deps                 outputs                 reductions         semantic_anchors",
  ]

  for i, (launch, scheduled) in enumerate(zip(launches, linear.src)):
    roots = tuple(anchor_nodes[a] for a in launch.anchor_ids)
    slice_nodes, semantic_deps = anchor_slice(roots, anchor_ids, raw_topo)
    outputs, reductions, scheduled_nodes = scheduled_summary(scheduled)
    layer = "--" if launch.layer is None else f"{launch.layer:02d}"
    deps = ",".join(f"K{x:03d}" for x in launch.deps) or "--"
    anchors = ",".join(f"A{x:03d}/g{global_ids[anchor_nodes[x]]:04d}" for x in launch.anchor_ids)
    filename = f"{i:03d}_{('layer_%02d_' % launch.layer) if launch.layer is not None else ''}{launch.name}.txt"
    manifest.append(f"K{i:03d}  {layer:>5}  {launch.name:<34} {deps:<20} {outputs:<23} {reductions:<18} {anchors}")

    text = [
      f"K{i:03d} — {launch.name}",
      "=" * (7 + len(launch.name)),
      common.rstrip(),
      f"layer: {layer}",
      f"depends_on: {deps}",
      f"intent: {launch.intent}",
      f"semantic_anchors: {anchors}",
      "",
      "LAUNCH-BOUNDARY WITNESS (summary only; already rangeified, not lowering input)",
      f"scheduled_sink_nodes: {scheduled_nodes}",
      f"output_buffers: {outputs}",
      f"range_reductions: {reductions}",
      "applied_optimizations: none (KernelInfo.applied_opts == ())",
      "",
      "PRISTINE SEMANTIC SLICE",
      "This slice is from raw_graph.txt before transform_to_call. It stops at earlier",
      "REDUCE/STORE anchors. Fused/split launches may share or aggregate anchors; the",
      "intent line above is the authoritative annotation for the launch boundary.",
      f"slice_nodes: {len(slice_nodes)}",
      f"earlier_semantic_anchors: {','.join('A%03d' % x for x in semantic_deps) or '--'}",
      "",
      "OP HISTOGRAM",
      op_histogram(slice_nodes),
      "",
      "TOPOLOGICAL SLICE",
      dump_nodes(slice_nodes, global_ids, anchor_ids),
      "",
    ]
    (kernels / filename).write_text("\n".join(text))

  (out / "manifest.txt").write_text("\n".join(manifest) + "\n")
  (out / "README.txt").write_text(
    "CORPUS LAYOUT\n=============\n\n"
    "raw_graph.txt\n  The only graph that should feed a TTIR matcher/lowerer. It is the complete\n"
    "  lazy Tensor UOp DAG before scheduling and all kernel/codegen optimization.\n\n"
    "manifest.txt\n  Actual compute-launch order plus concise semantic names, dependencies, output\n"
    "  storage, reduction witnesses, and links to pristine graph anchors.\n\n"
    "kernels/*.txt\n  One annotated file per scheduled compute launch. Each has a concise post-rangeify\n"
    "  boundary witness and a pre-transform semantic slice. Do not lower the witness.\n\n"
    "Regenerate from the repository root with:\n"
    "  $(realpath ../.venv)/bin/python tools/generate_uop_corpus.py\n\n"
    "The generator sets DEV=CPU as a storage tag, enters ALLOW_DEVICE_USAGE=0, uses only Tensor.empty\n"
    "buffers, and never calls realize, compile, a renderer, TinyJit, or a runtime.\n")
  print(f"wrote {len(launches)} launches and {len(raw_topo)} raw UOps to {out}")


if __name__ == "__main__":
  main()
