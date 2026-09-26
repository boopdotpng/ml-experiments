"""Tinygrad staged Mixture-of-Depths-and-Experts decoder.

The routed blocks follow Raposo et al. (2024): a fixed-capacity top-k router
chooses tokens for the complete attention + MLP block while all other tokens
take the identity residual.  Here the MLP is a packed top-2 sparse MoE.

All routing decisions stay as tinygrad tensors.  MoD gathers/scatters token
activations, while MoE indexes packed expert weights, just like OLMoE inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from tinygrad import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
  vocab_size: int
  max_seq_len: int = 128
  dim: int = 128
  n_layers: int = 8
  n_heads: int = 8
  n_experts: int = 8
  experts_per_token: int = 2
  expert_hidden: int = 256
  mod_capacity: float = 0.5

  def __post_init__(self):
    if self.dim % self.n_heads != 0:
      raise ValueError("dim must be divisible by n_heads")
    if self.experts_per_token > self.n_experts:
      raise ValueError("experts_per_token cannot exceed n_experts")
    if self.n_layers < 2:
      raise ValueError("interleaved MoD needs at least two layers")
    if not 0.0 < self.mod_capacity <= 1.0:
      raise ValueError("mod_capacity must be in (0, 1]")

  @property
  def head_dim(self) -> int:
    return self.dim // self.n_heads

  def is_mod_layer(self, layer: int) -> bool:
    # One-indexed layers 2, 4, 6, 8, ... are routed.  Every routed block is
    # preceded by a full block, matching the paper's strongest configuration.
    return layer % 2 == 1


def gather_tokens(x: Tensor, indices: Tensor) -> Tensor:
  """x[B,S,D] + indices[B,C] -> selected[B,C,D], entirely on device."""
  B, _, D = x.shape
  C = indices.shape[1]
  expanded = indices.reshape(B, C, 1).expand(B, C, D)
  return x.gather(1, expanded)


def scatter_token_updates(x: Tensor, indices: Tensor, updates: Tensor) -> Tensor:
  """Add unique selected-token updates back into a full [B,S,D] residual."""
  B, _, D = x.shape
  C = indices.shape[1]
  expanded = indices.reshape(B, C, 1).expand(B, C, D)
  return x + x.zeros_like().scatter(1, expanded, updates)


class RotaryEmbedding:
  def __init__(self, config: ModelConfig):
    half = config.head_dim // 2
    if config.head_dim % 2:
      raise ValueError("RoPE requires an even head dimension")
    inv_freq = 1.0 / (10000.0 ** (Tensor.arange(0, config.head_dim, 2) / config.head_dim))
    angles = Tensor.arange(config.max_seq_len).reshape(config.max_seq_len, 1) * inv_freq.reshape(1, half)
    # Static lookup tables are not parameters.
    self.cos = angles.cos().contiguous().is_param_(False)
    self.sin = angles.sin().contiguous().is_param_(False)

  def _lookup(self, table: Tensor, positions: Tensor) -> Tensor:
    B, S = positions.shape
    half = table.shape[1]
    source = table.reshape(1, table.shape[0], half).expand(B, table.shape[0], half)
    indices = positions.reshape(B, S, 1).expand(B, S, half)
    return source.gather(1, indices).reshape(B, 1, S, half)

  def __call__(self, x: Tensor, positions: Tensor) -> Tensor:
    B, H, S, D = x.shape
    pairs = x.reshape(B, H, S, D // 2, 2)
    cos, sin = self._lookup(self.cos, positions), self._lookup(self.sin, positions)
    even, odd = pairs[..., 0], pairs[..., 1]
    return Tensor.stack(even * cos - odd * sin, even * sin + odd * cos, dim=-1).reshape(B, H, S, D)


class Attention:
  def __init__(self, config: ModelConfig, rope: RotaryEmbedding):
    self.config, self.rope = config, rope
    self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
    self.k_proj = nn.Linear(config.dim, config.dim, bias=False)
    self.v_proj = nn.Linear(config.dim, config.dim, bias=False)
    self.o_proj = nn.Linear(config.dim, config.dim, bias=False)

  def __call__(self, x: Tensor, positions: Tensor, active: Tensor | None = None) -> Tensor:
    B, S, _ = x.shape
    H, Dh = self.config.n_heads, self.config.head_dim
    q = self.q_proj(x).reshape(B, S, H, Dh).transpose(1, 2)
    k = self.k_proj(x).reshape(B, S, H, Dh).transpose(1, 2)
    v = self.v_proj(x).reshape(B, S, H, Dh).transpose(1, 2)
    q, k = self.rope(q, positions), self.rope(k, positions)

    # Comparing original positions remains correct after MoD compacts a sparse
    # token set.  Threshold routing may leave inactive padding slots; inactive
    # queries get a harmless self edge to avoid an all-masked softmax row.
    q_pos, k_pos = positions.reshape(B, 1, S, 1), positions.reshape(B, 1, 1, S)
    allowed = k_pos <= q_pos
    if active is not None:
      active_q = active.reshape(B, 1, S, 1)
      active_k = active.reshape(B, 1, 1, S)
      eye = Tensor.arange(S).reshape(1, 1, S, 1) == Tensor.arange(S).reshape(1, 1, 1, S)
      allowed = (allowed & active_k) | ((~active_q) & eye)

    scores = (q @ k.transpose(-2, -1)) * (Dh ** -0.5)
    probs = scores.masked_fill(~allowed, -float("inf")).softmax(axis=-1)
    out = probs @ v
    out = out.transpose(1, 2).reshape(B, S, self.config.dim)
    return self.o_proj(out)


class PackedSparseMoE:
  """Top-k SwiGLU experts whose weights are packed along the expert axis."""

  def __init__(self, config: ModelConfig):
    self.config = config
    E, D, F = config.n_experts, config.dim, config.expert_hidden
    self.router = nn.Linear(D, E, bias=False)

    # Weight layout is chosen so selected expert ids can index the leading axis,
    # then use ordinary batched matmuls for every token and route slot.
    in_bound, hidden_bound = 1.0 / math.sqrt(D), 1.0 / math.sqrt(F)
    self.gate_proj = Tensor.uniform(E, D, F, low=-in_bound, high=in_bound)
    self.up_proj = Tensor.uniform(E, D, F, low=-in_bound, high=in_bound)
    self.down_proj = Tensor.uniform(E, F, D, low=-hidden_bound, high=hidden_bound)

  def __call__(self, x: Tensor, collect_routes: bool = False):
    leading, D = x.shape[:-1], x.shape[-1]
    T, E, K = math.prod(leading), self.config.n_experts, self.config.experts_per_token
    flat = x.reshape(T, D)
    logits = self.router(flat)
    all_probs = logits.softmax(axis=-1)
    selected_probs, selected = all_probs.topk(K, dim=-1)
    selected_probs = selected_probs / selected_probs.sum(axis=-1, keepdim=True)

    # GPU-resident expert dispatch: [T,K] indexes packed [E,D,F]/[E,F,D]
    # weights.  There is no Python loop or CPU-side expert choice.
    gate_w, up_w, down_w = self.gate_proj[selected], self.up_proj[selected], self.down_proj[selected]
    expanded = flat.reshape(T, 1, 1, D)
    gate = (expanded @ gate_w).squeeze(-2).silu()
    up = (expanded @ up_w).squeeze(-2)
    hidden = gate * up
    expert_out = (hidden.unsqueeze(-2) @ down_w).squeeze(-2)
    out = (expert_out * selected_probs.reshape(T, K, 1)).sum(axis=1).reshape(*leading, D)

    # Switch-style aggregate balancing.  It intentionally does not maximize
    # per-token entropy: stable expert specialization is desirable.
    mean_probs = all_probs.mean(axis=0)
    assignment_fraction = selected.one_hot(E).float().mean(axis=(0, 1)).detach()
    balance_loss = E * (mean_probs * assignment_fraction).sum()
    z_loss = logits.logsumexp(axis=-1).square().mean()
    trace = selected.reshape(*leading, K) if collect_routes else None
    return out, balance_loss, z_loss, trace


class TransformerBlock:
  def __init__(self, config: ModelConfig, rope: RotaryEmbedding, routed: bool):
    self.config, self.routed = config, routed
    self.attn_norm = nn.RMSNorm(config.dim)
    self.attn = Attention(config, rope)
    self.moe_norm = nn.RMSNorm(config.dim)
    self.moe = PackedSparseMoE(config)
    self.mod_router = nn.Linear(config.dim, 1, bias=False) if routed else None

  def _compute_block(self, x: Tensor, positions: Tensor, active: Tensor | None, collect_routes: bool):
    h = x + self.attn(self.attn_norm(x), positions, active)
    moe_delta, balance, z_loss, experts = self.moe(self.moe_norm(h), collect_routes)
    return h + moe_delta, balance, z_loss, experts

  def __call__(self, x: Tensor, positions: Tensor, route_mode: str = "topk", collect_routes: bool = False):
    B, S, _ = x.shape
    zero = x.sum() * 0.0
    if not self.routed:
      out, balance, z_loss, experts = self._compute_block(x, positions, None, collect_routes)
      trace = {"positions": positions, "experts": experts} if collect_routes else None
      return out, zero, balance, z_loss, trace

    assert self.mod_router is not None
    router_logits = self.mod_router(self.attn_norm(x)).squeeze(-1)
    capacity = max(1, round(S * self.config.mod_capacity))
    _, selected = router_logits.topk(capacity, dim=1)
    # topk orders by router score; restoring sequence order makes compacted
    # causal attention equivalent to attention over the selected subsequence.
    selected = selected.sort(dim=1)[0]
    selected_x = gather_tokens(x, selected)
    selected_positions = positions.gather(1, selected)
    selected_logits = router_logits.gather(1, selected)

    if route_mode == "topk":
      active = selected_logits.ones_like().cast("bool")
    elif route_mode == "threshold":
      # Static-shape causal predictor path.  Negative-score slots remain padded
      # inside the fixed maximum capacity and are masked out of attention and
      # the residual update, so routing stays entirely on device.
      active = selected_logits > 0
    else:
      raise ValueError(f"unknown route_mode {route_mode!r}; expected 'topk' or 'threshold'")

    processed, balance, z_loss, experts = self._compute_block(selected_x, selected_positions, active, collect_routes)
    # A sigmoid gate is a bounded small-model stabilization of the paper's
    # router-weighted block delta; top-k ranking itself is unchanged.
    gate = selected_logits.sigmoid() * active
    updates = (processed - selected_x) * gate.reshape(B, capacity, 1)
    out = scatter_token_updates(x, selected, updates)

    # The paper's sampling auxiliary objective: make top-k membership
    # predictable from a per-token zero threshold for causal generation.
    topk_target = selected.one_hot(S).sum(axis=1).float().detach()
    mod_predict_loss = router_logits.binary_crossentropy_logits(topk_target)
    trace = None
    if collect_routes:
      trace = {
        "positions": selected_positions,
        "experts": experts,
        "mod_logits": router_logits,
        "mod_active": active,
      }
    return out, mod_predict_loss, balance, z_loss, trace


class Model:
  def __init__(self, config: ModelConfig):
    self.config = config
    self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
    self.rope = RotaryEmbedding(config)
    self.layers = [
      TransformerBlock(config, self.rope, routed=config.is_mod_layer(i))
      for i in range(config.n_layers)
    ]
    self.norm = nn.RMSNorm(config.dim)

  def __call__(self, tokens: Tensor, route_mode: str = "topk", collect_routes: bool = False):
    B, S = tokens.shape
    if S > self.config.max_seq_len:
      raise ValueError(f"sequence length {S} exceeds max_seq_len={self.config.max_seq_len}")
    x = self.tok_emb(tokens)
    positions = Tensor.arange(S).reshape(1, S).expand(B, S)
    zero = x.sum() * 0.0
    mod_loss, balance_loss, z_loss = zero, zero, zero
    traces = []
    for layer in self.layers:
      x, layer_mod, layer_balance, layer_z, trace = layer(x, positions, route_mode, collect_routes)
      mod_loss, balance_loss, z_loss = mod_loss + layer_mod, balance_loss + layer_balance, z_loss + layer_z
      if collect_routes: traces.append(trace)

    n_mod = sum(layer.routed for layer in self.layers)
    mod_loss = mod_loss / max(1, n_mod)
    balance_loss = balance_loss / len(self.layers)
    z_loss = z_loss / len(self.layers)
    x = self.norm(x)
    # Weight tying keeps this genuinely tiny and gives token embeddings a direct
    # path to the language-model objective.
    logits = x @ self.tok_emb.weight.transpose()
    return logits, mod_loss, balance_loss, z_loss, traces


def parameter_counts(model: Model) -> tuple[int, int]:
  """Return total and approximate active parameters for one token."""
  total = sum(p.numel() for p in nn.state.get_parameters(model))
  cfg = model.config
  expert_per_layer = 3 * cfg.dim * cfg.expert_hidden
  inactive_experts = cfg.n_layers * (cfg.n_experts - cfg.experts_per_token) * expert_per_layer
  return total, total - inactive_experts


def estimate_training_flops(config: ModelConfig, batch_size: int, seq_len: int, steps: int) -> dict[str, float]:
  """Approximate matmul FLOPs (multiply and add count as two operations).

  Backward is estimated as twice the forward pass.  Norms, activations, sorting,
  gathers, scatters, softmax, optimizer updates, and auxiliary losses are omitted,
  so this is a transparent architecture comparison rather than hardware telemetry.
  """
  D, E, K, F, V = config.dim, config.n_experts, config.experts_per_token, config.expert_hidden, config.vocab_size

  def layer_flops(tokens: int, attention_tokens: int) -> float:
    projections = 8 * tokens * D * D                 # q, k, v, o
    attention = 4 * attention_tokens * attention_tokens * D  # qk^T and av
    expert_router = 2 * tokens * D * E
    active_experts = 6 * tokens * K * D * F          # gate, up, down
    return projections + attention + expert_router + active_experts

  full = layer_flops(seq_len, seq_len)
  capacity = max(1, round(seq_len * config.mod_capacity))
  routed = layer_flops(capacity, capacity) + 2 * seq_len * D  # MoD router scores all tokens
  n_mod = sum(config.is_mod_layer(i) for i in range(config.n_layers))
  n_full = config.n_layers - n_mod
  lm_head = 2 * seq_len * D * V
  forward_per_sequence = n_full * full + n_mod * routed + lm_head
  dense_mod_baseline = config.n_layers * full + lm_head
  forward_per_step = batch_size * forward_per_sequence
  train_per_step = 3 * forward_per_step
  return {
    "capacity_tokens": float(capacity),
    "forward_per_sequence": forward_per_sequence,
    "forward_per_step": forward_per_step,
    "train_per_step": train_per_step,
    "train_run": train_per_step * steps,
    "full_moe_forward_per_sequence": dense_mod_baseline,
    "fraction_of_full_moe": forward_per_sequence / dense_mod_baseline,
    "tokens_seen": float(batch_size * seq_len * steps),
  }
