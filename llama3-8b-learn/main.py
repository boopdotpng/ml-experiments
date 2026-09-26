from __future__ import annotations

import argparse, json, time
from pathlib import Path

from tinygrad import Tensor, nn, dtypes, TinyJit, Variable, GlobalCounters
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from jinja2.sandbox import ImmutableSandboxedEnvironment

MODEL_DIR = Path(__file__).resolve().parent
# BF16 HF checkpoint, pinned download revision:
# unsloth/llama-3-8b-Instruct @ f3710969eb766fb49d4d1ed3aeabcb03390772bd
# Original Llama 3 8B Instruct.
emb_dim = 4096
n_layers = 32
n_heads = 32
n_kv_heads = 8
head_dim = 128
mlp_size = 14336
vocab_size = 128256
norm_eps = 1e-5
rope_theta = 500000.0
max_seq_len = 8192
weight_dtype = dtypes.bfloat16


class Model:
  def __init__(self):
    self.embed_tokens = nn.Embedding(vocab_size, emb_dim)
    self.layers = [Block() for _ in range(n_layers)]
    self.norm = RMSNorm(emb_dim)
    self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)

  def __call__(self, tokens: Tensor, start_pos: int = 0) -> Tensor:
    raise NotImplementedError


class Block:
  def __init__(self):
    self.input_layernorm = RMSNorm(emb_dim)
    self.self_attn = SelfAttention()
    self.post_attention_layernorm = RMSNorm(emb_dim)
    self.mlp = MLP()

  def __call__(self, x: Tensor, start_pos: int = 0) -> Tensor:
    raise NotImplementedError


class RMSNorm:
  def __init__(self, dim: int, eps: float = norm_eps):
    self.eps = eps
    self.weight = Tensor.ones(dim)

  def __call__(self, x: Tensor) -> Tensor:
    raise NotImplementedError


class MLP:
  def __init__(self):
    self.gate_proj = nn.Linear(emb_dim, mlp_size, bias=False)
    self.up_proj = nn.Linear(emb_dim, mlp_size, bias=False)
    self.down_proj = nn.Linear(mlp_size, emb_dim, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    raise NotImplementedError


class SelfAttention:
  def __init__(self):
    self.q_proj = nn.Linear(emb_dim, n_heads * head_dim, bias=False)
    self.k_proj = nn.Linear(emb_dim, n_kv_heads * head_dim, bias=False)
    self.v_proj = nn.Linear(emb_dim, n_kv_heads * head_dim, bias=False)
    self.o_proj = nn.Linear(n_heads * head_dim, emb_dim, bias=False)
    self.kv_cache = None

  def __call__(self, x: Tensor, start_pos: int = 0) -> Tensor:
    raise NotImplementedError


def checkpoint_files() -> list[Path]:
  index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
  paths = [MODEL_DIR / name for name in sorted(set(index["weight_map"].values()))]
  for path in paths:
    if not path.is_file():
      raise FileNotFoundError(f"Missing checkpoint shard: {path}")
  return paths


def read_weights() -> dict[str, Tensor]:
  weights = {}
  for path in checkpoint_files():
    for name, tensor in safe_load(path).items():
      weights[name.removeprefix("model.")] = tensor
  return weights


def list_weights():
  for path in checkpoint_files():
    print(f"\n{path.name}")
    for name, tensor in safe_load(path).items():
      print(f"  {name}  {tensor.shape}  {tensor.dtype}")
  print("\nLoader removes the leading 'model.' prefix; all other names stay unchanged.")


def load_weights(model: Model):
  weights = read_weights()
  # Check both directions; tinygrad strict=True only checks model-side keys.
  names = get_state_dict(model).keys()
  missing, unused = names - weights.keys(), weights.keys() - names
  if missing or unused:
    raise ValueError(f"Checkpoint mismatch: {len(missing)} missing tensors, {len(unused)} unused tensors")
  load_state_dict(model, weights, strict=True, consume=True)


def format_prompt(prompt: str, system: str | None = None) -> str:
  config = json.loads((MODEL_DIR / "tokenizer_config.json").read_text())
  env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
  def raise_exception(message):
    raise ValueError(message)
  env.globals["raise_exception"] = raise_exception
  messages = [] if system is None else [{"role": "system", "content": system}]
  messages.append({"role": "user", "content": prompt})
  return env.from_string(config["chat_template"]).render(
    messages=messages, add_generation_prompt=True, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>")


def load_eos_ids(tokenizer: Tokenizer) -> set[int]:
  eos = json.loads((MODEL_DIR / "generation_config.json").read_text())["eos_token_id"]
  ids = set(eos if isinstance(eos, list) else [eos])
  for token in ("<|end_of_text|>", "<|eot_id|>"):
    if (token_id := tokenizer.token_to_id(token)) is not None:
      ids.add(token_id)
  return ids


def prepare_prompt(tokenizer: Tokenizer, prompt: str, max_new_tokens: int,
                   raw: bool = False, system: str | None = None) -> list[int]:
  if raw and system is not None:
    raise ValueError("--system requires chat formatting (omit --raw)")
  text = prompt if raw else format_prompt(prompt, system)
  # The chat template includes BOS. Raw mode uses the tokenizer's BOS processor.
  ids = tokenizer.encode(text, add_special_tokens=raw).ids
  if not ids:
    raise ValueError("Prompt must contain at least one token")
  if max_new_tokens < 0:
    raise ValueError("max_new_tokens must be nonnegative")
  if len(ids) + max_new_tokens > max_seq_len:
    raise ValueError(f"prompt + generation length exceeds max_seq_len={max_seq_len}")
  return ids


def generate(model: Model, tokenizer: Tokenizer, prompt_ids: list[int], max_new_tokens: int):
  raise NotImplementedError


def main():
  parser = argparse.ArgumentParser(description="Llama 3 8B Instruct tinygrad learning scaffold")
  parser.add_argument("--prompt", default="hello, how are you")
  parser.add_argument("--system", default=None)
  parser.add_argument("--max-new-tokens", type=int, default=128)
  parser.add_argument("--raw", action="store_true", help="Skip chat formatting; still prepend BOS")
  parser.add_argument("--inspect-prompt", action="store_true", help="Show formatted prompt and IDs without loading the model")
  parser.add_argument("--generate", action="store_true", help="Load weights and run your generation implementation")
  args = parser.parse_args()

  if not args.generate and not args.inspect_prompt:
    list_weights()
    return

  tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
  prompt_ids = prepare_prompt(tokenizer, args.prompt, args.max_new_tokens, raw=args.raw, system=args.system)
  if args.inspect_prompt:
    print(tokenizer.decode(prompt_ids, skip_special_tokens=False))
    print(f"{len(prompt_ids)} tokens: {prompt_ids}")
    print(f"stop IDs: {sorted(load_eos_ids(tokenizer))}")
    return
  if args.max_new_tokens == 0:
    return
  model = Model()
  load_weights(model)
  generate(model, tokenizer, prompt_ids, args.max_new_tokens)


if __name__ == "__main__":
  main()
