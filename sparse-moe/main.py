from pathlib import Path
import argparse, json, time
from tinygrad import Tensor, TinyJit, Variable, GlobalCounters, Device, dtypes
from tinygrad.nn.state import safe_load, load_state_dict
from tokenizers import Tokenizer
from tinygrad import nn

ROOT = Path(__file__).parent

# model
emb_dim = 2048
n_layers = 16
vocab_size = 50304
max_seq_len = 4096

# attention: ordinary MHA
n_heads = 16
head_dim = 128                 # 2048 // 16

mlp_size = 1024
n_experts = 64
n_experts_per_tok = 8

norm_eps = 1e-5
rope_theta = 10000.0

# architecture behavior
attention_bias = False
attention_dropout = 0.0
norm_topk_prob = False
tie_word_embeddings = False

# generation
eos_token_id = 50279
pad_token_id = 1

def rope_cache():
  inv_freq = 1.0 / (
    rope_theta ** (Tensor.arange(0, head_dim, 2) / head_dim)
  )
  angles = Tensor.arange(max_seq_len).unsqueeze(1) * inv_freq.unsqueeze(0)
  angles = angles.repeat(1,2)

  cos = angles.cos().contiguous().realize()
  sin = angles.sin().contiguous().realize()
  return cos, sin

COS, SIN = rope_cache()

def apply_rope(x: Tensor, start_pos:int) -> Tensor:
  # x is (B, heads, S, head_dim)
  S = x.shape[2] # seq_len 

  cos = COS[start_pos:start_pos+S].reshape(1, 1, S, head_dim).cast(x.dtype)
  sin= SIN[start_pos:start_pos+S].reshape(1, 1, S, head_dim).cast(x.dtype)

  x1 = x[..., :head_dim // 2]
  x2 = x[..., head_dim // 2:]

  rotate_half = (-x2).cat(x1, dim=-1)

  return x * cos + rotate_half * sin

class Attention:
  def __init__(self):
    self.q_proj = nn.Linear(emb_dim, n_heads * head_dim, bias=False)
    self.k_proj = nn.Linear(emb_dim, n_heads * head_dim, bias=False)
    self.v_proj = nn.Linear(emb_dim, n_heads * head_dim, bias=False)

    self.o_proj = nn.Linear(n_heads * head_dim, emb_dim, bias=False)

    # norms
    self.q_norm = nn.RMSNorm(emb_dim, eps=norm_eps)
    self.k_norm = nn.RMSNorm(emb_dim, eps=norm_eps)

    # kv cache: (2 for K/V, B=1, heads, max_seq_len, head_dim)
    self.kv_cache = None
  def __call__(self, x: Tensor, start_pos:int=0) -> Tensor:
    B, S, _ = x.shape
    if B != 1:
      raise NotImplementedError(f"Attention KV cache only supports batch size one, got B={B}")
    if start_pos + S > max_seq_len:
      raise ValueError(f"attention position {start_pos + S} exceeds max_seq_len={max_seq_len}")

    q = self.q_norm(self.q_proj(x)).reshape(B, S, n_heads, head_dim).transpose(1,2)
    k = self.k_norm(self.k_proj(x)).reshape(B, S, n_heads, head_dim).transpose(1,2)
    v = self.v_proj(x).reshape(B, S, n_heads, head_dim).transpose(1,2)

    q = apply_rope(q, start_pos)
    k = apply_rope(k, start_pos)

    if self.kv_cache is None:
      self.kv_cache = Tensor.zeros(
        2, 1, n_heads, max_seq_len, head_dim,
        dtype=dtypes.bfloat16,
        device=k.device,
      ).realize()

    # Store rotated K and unrotated V. The explicit store dependency keeps the
    # mutation ordered correctly when the one-token path is wrapped by TinyJit.
    kv = Tensor.stack(k.cast(dtypes.bfloat16), v.cast(dtypes.bfloat16))
    cache_slice = self.kv_cache[:, :B, :, start_pos:start_pos+S, :]
    kv_cache = Tensor(self.kv_cache.uop.after(cache_slice.uop.store(kv.uop)))

    # Read every cached key/value through the end of this input chunk.
    T = start_pos + S
    k = kv_cache[0, :B, :, :T, :] # (B, heads, T, head_dim)
    v = kv_cache[1, :B, :, :T, :] # (B, heads, T, head_dim)

    # Accumulate attention logits and run softmax in FP32, matching OLMoE.
    scores = q.matmul(k.transpose(-2, -1), dtype=dtypes.float32)
    scores = scores * (head_dim ** -0.5)

    if S != 1:
      causal_mask = Tensor.full(
        (1, 1, S, T),
        float("-inf"),
        buffer=False
      ).triu(start_pos + 1)

      scores = scores + causal_mask

    attn = scores.softmax(axis=-1).cast(q.dtype)

    out = (attn @ v).transpose(1,2)
    out = out.reshape(B, S, n_heads * head_dim)

    return self.o_proj(out)

class SparseMoE:
  def __init__(self):
    # Router: one score for each of the 64 experts.
    self.gate = nn.Linear(emb_dim, n_experts, bias=False)

    # All expert SwiGLU weights are packed along axis 0 so GPU top-k indices can
    # select experts without converting those indices to Python integers.
    # The checkpoint converter will stack experts.0 ... experts.63 into these.
    self.gate_proj = Tensor.empty(
      n_experts, mlp_size, emb_dim, dtype=dtypes.bfloat16
    ) # (64, 1024, 2048)
    self.up_proj = Tensor.empty(
      n_experts, mlp_size, emb_dim, dtype=dtypes.bfloat16
    ) # (64, 1024, 2048)
    self.down_proj = Tensor.empty(
      n_experts, emb_dim, mlp_size, dtype=dtypes.bfloat16
    ) # (64, 2048, 1024)

  def __call__(self, x: Tensor) -> Tensor:
    # Batch size one for now. Prefill temporarily applies the decode path to
    # each token independently; grouped expert dispatch can replace this later.
    B, S, D = x.shape
    if B != 1:
      raise NotImplementedError(f"SparseMoE only supports batch size one, got x.shape={x.shape}")

    if S == 1:
      return self._one_token(x)

    outputs = [self._one_token(x[:, i:i+1, :]) for i in range(S)]
    return outputs[0].cat(*outputs[1:], dim=1) # (1, S, 2048)

  def _one_token(self, x: Tensor) -> Tensor:
    # x: (1, 1, 2048)
    B, S, D = x.shape
    assert B == 1 and S == 1 and D == emb_dim

    # Softmax across all 64 experts, then retain the best eight. OLMoE has
    # norm_topk_prob=False, so do not renormalize these eight probabilities.
    router_probs = self.gate(x).float().softmax(-1).reshape(n_experts) # (64,)
    probs, selected = router_probs.topk(n_experts_per_tok)             # both (8,)
    probs = probs.cast(x.dtype)

    # selected projection weights:
    #   gate/up: (8, 1024, 2048) -> (8, 2048, 1024)
    #   down:    (8, 2048, 1024) -> (8, 1024, 2048)
    selected_gate = self.gate_proj[selected].permute(0, 2, 1)
    selected_up = self.up_proj[selected].permute(0, 2, 1)
    selected_down = self.down_proj[selected].permute(0, 2, 1)

    # x broadcasts from (1, 1, 2048) across the eight selected experts.
    gate = x.dot(selected_gate).silu() # (8, 1, 1024)
    up = x.dot(selected_up)            # (8, 1, 1024)
    hidden = gate * up                 # (8, 1, 1024)
    expert_outputs = hidden.dot(selected_down) # (8, 1, 2048)

    # Weight the eight expert results and return to residual-stream shape.
    output = (expert_outputs * probs.reshape(n_experts_per_tok, 1, 1)).sum(axis=0)
    return output.reshape(B, S, D) # (1, 1, 2048)

class Block:
  def __init__(self):
    self.input_layernorm = nn.RMSNorm(emb_dim, eps=norm_eps)
    self.self_attn = Attention()

    self.post_attention_layernorm = nn.RMSNorm(emb_dim, eps=norm_eps)
    self.mlp = SparseMoE()
  def __call__(self, x: Tensor, start_pos:int=0) -> Tensor:
    x = x + self.self_attn(self.input_layernorm(x), start_pos)
    x = x + self.mlp(self.post_attention_layernorm(x))
    return x

class Model:
  def __init__(self):
    self.embed_tokens = nn.Embedding(vocab_size, emb_dim)
    self.layers = [Block() for _ in range(n_layers)]
    self.norm = nn.RMSNorm(emb_dim, eps=norm_eps)

    # OLMoE keeps a separate output projection; it is not tied to embeddings.
    self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)

  def forward(self, tokens: Tensor, start_pos:int=0) -> Tensor:
    x = self.embed_tokens(tokens) # (B, S) -> (B, S, 2048)
    for layer in self.layers:
      x = layer(x, start_pos)
    x = self.norm(x)
    return self.lm_head(x) # (B, S, 50304)

  def __call__(self, tokens: Tensor, start_pos:int=0) -> Tensor:
    return self.forward(tokens, start_pos)

def load_weights(print_weights:bool=True) -> dict[str, Tensor]:
  shards = sorted(ROOT.glob("model-*.safetensors"))
  if not shards:
    raise FileNotFoundError("No safetensor shards found. Run ./download.sh first.")

  # safe_load creates lazy, disk-backed tensors; this does not eagerly copy the
  # entire 13.8 GB checkpoint to RAM/GPU.
  weights: dict[str, Tensor] = {}
  for shard in shards:
    if print_weights: print(f"\n# {shard.name}")
    for name, tensor in safe_load(shard).items():
      if name in weights:
        raise ValueError(f"duplicate tensor name across shards: {name}")
      weights[name] = tensor
      if print_weights and ".mlp.experts." not in name:
        print(f"{name:<72} shape={str(tensor.shape):<20} dtype={tensor.dtype}")

  if print_weights: print(f"\nfound {len(weights):,} tensors in {len(shards)} shards")
  return weights


def convert_from_huggingface_for_this_model(weights: dict[str, Tensor]) -> dict[str, Tensor]:
  converted: dict[str, Tensor] = {}
  packed_experts: dict[str, dict[int, Tensor]] = {}

  for name, tensor in weights.items():
    # Packing directly on DISK produces a multi-view STACK that has no codegen
    # renderer. Move each lazy view to the execution device first, matching
    # tinygrad's built-in Hugging Face conversion pattern.
    tensor = tensor.to(Device.DEFAULT)

    if ".mlp.experts." in name:
      # model.layers.3.mlp.experts.17.up_proj.weight
      parts = name.split(".")
      layer = int(parts[2])
      expert = int(parts[5])
      projection = parts[6] # gate_proj, up_proj, or down_proj
      target = f"layers.{layer}.mlp.{projection}"
      packed_experts.setdefault(target, {})[expert] = tensor
      continue

    # Our object hierarchy otherwise matches Hugging Face after removing the
    # outer `model.` prefix. lm_head.weight already has no such prefix.
    target = name[len("model."):] if name.startswith("model.") else name
    converted[target] = tensor

  for target, experts in packed_experts.items():
    expected = set(range(n_experts))
    actual = set(experts)
    if actual != expected:
      missing = sorted(expected - actual)
      extra = sorted(actual - expected)
      raise ValueError(f"bad expert set for {target}: missing={missing}, extra={extra}")

    # Add an expert axis: 64 x (projection weight shape). This remains lazy
    # until load_state_dict moves and realizes the packed tensor on the device.
    converted[target] = Tensor.stack(
      *[experts[expert] for expert in range(n_experts)], dim=0
    )

  return converted


def load_eos_ids() -> set[int]:
  with open(ROOT / "generation_config.json") as f:
    eos = json.load(f)["eos_token_id"]
  return set(eos if isinstance(eos, list) else [eos])


def generate(model, tokenizer: Tokenizer, prompt: str, raw_prompt: bool = False):
  if raw_prompt:
    model_prompt = prompt
  else:
    with open(ROOT / "tokenizer_config.json") as f:
      bos_token = json.load(f)["bos_token"]
    model_prompt = f"{bos_token}<|user|>\n{prompt}\n<|assistant|>\n"
  prompt_ids = tokenizer.encode(model_prompt, add_special_tokens=False).ids
  if not prompt_ids:
    raise ValueError("prompt encoded to zero tokens")
  if len(prompt_ids) >= max_seq_len:
    raise ValueError(f"prompt length must be less than max_seq_len={max_seq_len}")
  max_new_tokens = max_seq_len - len(prompt_ids)

  # Argmax is part of the compiled decode graph but not part of your Model.
  def greedy_forward(tokens: Tensor, start_pos: int) -> Tensor:
    logits = model.forward(tokens, start_pos)
    return logits[:, -1, :].flatten().argmax()

  decode_jit = TinyJit(greedy_forward)
  eos_ids = load_eos_ids()
  print(f"prompt: {prompt!r}")
  print("generated: ", end="", flush=True)

  # Temporary prefill: feed the real prompt one token at a time so attention
  # and MoE always take the same S=1 path used by decode. Predictions before
  # the final prompt token are deliberately ignored; their cache writes remain.
  GlobalCounters.reset()
  st = time.perf_counter()
  next_id = None
  for pos, prompt_id in enumerate(prompt_ids):
    token = Tensor([[prompt_id]]).contiguous()
    if pos == 0:
      next_id = greedy_forward(token, 0).item()
    else:
      sp = Variable("start_pos", 1, max_seq_len - 1).bind(pos)
      next_id = decode_jit(token, sp).item()
  assert next_id is not None
  pt = time.perf_counter()
  prefill_mem = GlobalCounters.global_mem

  generated_ids: list[int] = []
  decode_times: list[float] = []
  GlobalCounters.reset()
  gt = time.perf_counter()
  for i in range(max_new_tokens):
    generated_ids.append(next_id)
    print(tokenizer.decode([next_id], skip_special_tokens=True), end="", flush=True)
    if next_id in eos_ids or i == max_new_tokens - 1:
      break

    start_pos = len(prompt_ids) + i
    sp = Variable("start_pos", 1, max_seq_len - 1).bind(start_pos)
    ds = time.perf_counter()
    next_id = decode_jit(Tensor([[next_id]]).contiguous(), sp).item()
    decode_times.append(time.perf_counter() - ds)

  gen_time = time.perf_counter() - gt
  print()
  print(f"prefill: {len(prompt_ids)/(pt-st):.2f} tok/s, {prefill_mem/(pt-st)/1e9:.2f} GB/s")
  print(f"gen: {len(generated_ids)/gen_time if generated_ids else 0:.2f} tok/s, "
        f"{GlobalCounters.global_mem/gen_time/1e9:.2f} GB/s, "
        f"{GlobalCounters.global_mem//1_000_000}/{GlobalCounters.mem_used//1_000_000} MB")
  if len(decode_times) > 2:
    steady = sum(decode_times[2:])
    print(f"gen steady-state: {len(decode_times)-2} tok in {steady*1e3:.2f} ms, "
          f"{(len(decode_times)-2)/steady:.2f} tok/s")
  return generated_ids


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--prompt", default="The capital of France is")
  parser.add_argument("--raw-prompt", action="store_true", help="skip the OLMoE-Instruct chat template")
  parser.add_argument("--list-weights", action="store_true", help="print checkpoint tensors and stop")
  args = parser.parse_args()

  weights = load_weights(print_weights=args.list_weights)
  if args.list_weights:
    return
  if "Model" not in globals():
    print("\nModel is not implemented yet. Add it above; the generation runner is ready.")
    return

  tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer.json"))
  model = Model()
  converted = convert_from_huggingface_for_this_model(weights)
  del weights
  load_state_dict(model, converted, strict=True, consume=True, verbose=True)
  print("loaded OLMoE weights")
  generate(model, tokenizer, args.prompt, args.raw_prompt)


if __name__ == "__main__":
  main()
