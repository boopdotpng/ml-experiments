from tinygrad import Tensor,  nn, dtypes, TinyJit, Variable
from tinygrad.nn.state import safe_load, load_state_dict
from tokenizers import Tokenizer
import argparse, json
import math

# Llama 3.2 1B config
emb_dim = 2048
n_layers = 16
n_heads = 32
n_kv_heads = 8
head_dim = 64
mlp_size = 8192

vocab_size = 128256
norm_eps = 1e-5

# your inference limit
# original context, not scaled
# rope tables only generated up until here
max_seq_len = 8192

# official Llama 3.2 1B RoPE params
rope_theta = 500000.0
rope_factor = 32.0
rope_low_freq_factor = 1.0
rope_high_freq_factor = 4.0
rope_original_max_position_embeddings = 8192

# rope table cache 
def rope_table():
  # normal rope frequency setup
  inv_freq = 1.0 / (rope_theta ** (Tensor.arange(0, head_dim, 2) / head_dim))
  # tokens needed for one full rope rotation 
  wavelen = 2 * math.pi / inv_freq
  # how many full rotations inside the original 8192-pretrain? 
  # this is what the llama3 rule is based on 
  cycles = rope_original_max_position_embeddings / wavelen
  smooth = ((cycles - rope_low_freq_factor) / (rope_high_freq_factor - rope_low_freq_factor)).clip(0.0, 1.0)
  # lerp scaling
  inv_freq = inv_freq * ((1.0 / rope_factor) + smooth * (1.0 - (1.0 / rope_factor)))
  angles = Tensor.arange(max_seq_len).unsqueeze(1) * inv_freq.unsqueeze(0)
  angles = angles.repeat(1,2)
  return angles.cos().contiguous(), angles.sin().contiguous()

COS, SIN = rope_table() # both (8192, 64)

def apply_rope(x: Tensor, pos: int):
  S = x.shape[2]
  cos = COS[pos:pos+S].reshape(1, 1, S, head_dim)
  sin = SIN[pos:pos+S].reshape(1, 1, S, head_dim)

  x1 = x[..., :head_dim//2]
  x2 = x[..., head_dim//2:]

  rotated = (-x2).cat(x1, dim=-1)
  return x * cos + rotated * sin

class RMSNorm:
  def __init__(self, dim:int, eps=norm_eps):
    self.eps = eps
    self.weight = Tensor.ones(dim) 

  def __call__(self, x:Tensor) -> Tensor:
    # square, avg across last dim, add eps
    # div by sqrt above ^
    # multiply eltwise by weights
    x = x / ((x**2).mean(axis=-1, keepdim=True)+self.eps).sqrt()
    return x * self.weight

class MLP:
  def __init__(self):
    self.gate_proj = nn.Linear(emb_dim, mlp_size, bias=False)
    self.up_proj = nn.Linear(emb_dim, mlp_size, bias=False)
    self.down_proj = nn.Linear(mlp_size, emb_dim, bias=False)
  def __call__(self, x:Tensor) -> Tensor:
    gate = self.gate_proj(x) # what features should be allowed through
    up = self.up_proj(x) # candidate features / expanded representation
    hidden = gate.silu() * up # apply soft conditional filter 
    return self.down_proj(hidden) # residual update, back to original dims

class Attention:
  def __init__(self):
    self.q_proj = nn.Linear(emb_dim, head_dim * n_heads, bias=False)
    self.k_proj = nn.Linear(emb_dim, head_dim * n_kv_heads, bias=False)
    self.v_proj = nn.Linear(emb_dim, head_dim * n_kv_heads, bias=False)
    self.o_proj = nn.Linear(n_heads * head_dim, emb_dim, bias=False)

    # kv cache
    # BS=1 inference for now
    self.k_cache = None
    self.v_cache = None
  def __call__(self, x:Tensor, start_pos:int=0) -> Tensor:
    # batch, seq_len, ...
    B, S, _ = x.shape
    q = self.q_proj(x).reshape(B, S, n_heads, head_dim).transpose(1,2) # (B, n_heads, S, head_dim)
    k = self.k_proj(x).reshape(B, S, n_kv_heads, head_dim).transpose(1,2) # (B, n_kv_heads, S, head_dim)
    v = self.v_proj(x).reshape(B, S, n_kv_heads, head_dim).transpose(1,2) # (B, n_kv_heads, S, head_dim)

    # apply rope
    # cos and sin are 8192, 64 (head_dim)  
    # pick the positions in this forward pass
    q = apply_rope(q, start_pos) # (B, n_heads, S, head_dim)
    k = apply_rope(k, start_pos) # (B, n_kv_heads, S, head_dim)

    if self.k_cache is None:
      self.k_cache = Tensor.zeros(1, n_kv_heads, max_seq_len, head_dim, dtype=k.dtype, device=k.device).realize()
      self.v_cache = Tensor.zeros(1, n_kv_heads, max_seq_len, head_dim, dtype=v.dtype, device=v.device).realize()

    # store k and v 
    k_cache = Tensor(self.k_cache.uop.after(self.k_cache[:B, :, start_pos:start_pos+S, :].uop.store(k.uop)))
    v_cache = Tensor(self.v_cache.uop.after(self.v_cache[:B, :, start_pos:start_pos+S, :].uop.store(v.uop)))

    k = k_cache[:B, :, :start_pos+S, :] # (B, n_kv_heads, T, head_dim), T = start_pos + S
    v = v_cache[:B, :, :start_pos+S, :] # (B, n_kv_heads, T, head_dim)

    # gqa 
    k = k.repeat_interleave(n_heads // n_kv_heads, 1) # (B, n_heads, T, head_dim)
    v = v.repeat_interleave(n_heads // n_kv_heads, 1) # (B, n_heads, T, head_dim)

    scores = q.matmul(k.transpose(-2, -1), dtype=dtypes.float32) # (B, n_heads, S, T)
    scores = scores * (head_dim ** -0.5)

    # causal mask
    T = start_pos + S # all available history for this prompt 
    # when doing one-token-at-a-time decode, you don't need a mask. future tokens are not known
    if S != 1:
      causal_mask = Tensor.full((1,1,S,T), float("-inf"), buffer=False).triu(start_pos+1)
      scores = scores + causal_mask
    
    attn = scores.softmax(-1)

    out = attn @ v # (B, n_heads, S, head_dim)
    out = out.transpose(1,2).reshape(B, S, n_heads * head_dim) # (B, S, emb_dim)
    return self.o_proj(out)


class Block:
  def __init__(self):
    self.mlp = MLP()
    self.post_attention_layernorm = RMSNorm(emb_dim, eps=norm_eps)
    self.input_layernorm = RMSNorm(emb_dim, eps=norm_eps)
    self.self_attn = Attention()

  def __call__(self, x:Tensor, start_pos:int=0) -> Tensor:
    # order:  
    # rmsnorm 
    # attention
    # residual add
    # rmsnorm
    # mlp
    # residual add
    x = x + self.self_attn(self.input_layernorm(x), start_pos)
    x = x + self.mlp(self.post_attention_layernorm(x))
    return x 

class Model:
  def __init__(self):
    self.decode_jit = TinyJit(self.forward)
    self.embed_tokens = nn.Embedding(vocab_size, emb_dim)
    self.layers = [Block() for _ in range(n_layers)]
    self.norm = RMSNorm(emb_dim, eps=norm_eps)

  def forward(self, x: Tensor, start_pos:int):
    x = self.embed_tokens(x)
    for layer in self.layers: x = layer(x, start_pos)
    x = self.norm(x) 
    return x @ self.embed_tokens.weight.T

  def __call__(self, x:Tensor, start_pos:int=0) -> Tensor:
    if x.shape[1] == 1 and start_pos != 0:
      return self.decode_jit(x.contiguous(), start_pos)
    return self.forward(x, start_pos)

def convert_from_huggingface_for_this_model(weights:dict[str, Tensor]) -> dict[str, Tensor]:
  converted = {}
  for k,v in weights.items():
    if k.startswith("model."): k = k[len("model."):]
    if k == "lm_head.weight": continue
    converted[k] = v
  return converted

def load_eos_ids(path="./generation_config.json") -> set[int]:
  with open(path) as f:
    eos = json.load(f)["eos_token_id"]
  return set(eos if isinstance(eos, list) else [eos])

def sample_next_token(logits:Tensor) -> int:
  # Greedy for now. This keeps the loop deterministic while the model is being built.
  return int(logits.argmax().numpy())

def generate(model:Model, tokenizer:Tokenizer, prompt:str, max_new_tokens:int|None, add_bos:bool=True):
  prompt_ids = tokenizer.encode(prompt, add_special_tokens=add_bos).ids
  if max_new_tokens is None:
    max_new_tokens = max_seq_len - len(prompt_ids)
  if len(prompt_ids) + max_new_tokens > max_seq_len:
    raise ValueError(f"prompt + generation length exceeds max_seq_len={max_seq_len}")

  eos_ids = load_eos_ids()
  print(f"prompt: {prompt!r}")
  print(f"prompt ids: {prompt_ids}")
  print("generated: ", end="", flush=True)

  logits = model(Tensor([prompt_ids]), start_pos=0)
  next_id = sample_next_token(logits[0, -1])

  generated_ids = []
  for i in range(max_new_tokens):
    generated_ids.append(next_id)
    print(tokenizer.decode([next_id], skip_special_tokens=True), end="", flush=True)
    if next_id in eos_ids:
      break

    start_pos = len(prompt_ids) + i
    sp = Variable("start_pos", 1, max_seq_len-1).bind(start_pos)
    logits = model(Tensor([[next_id]]), start_pos=sp)
    next_id = sample_next_token(logits[0, -1])

  print()
  return generated_ids

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--prompt", default="hello, how are you")
  parser.add_argument("--max-new-tokens", type=int, default=None)
  parser.add_argument("--no-bos", action="store_true")
  args = parser.parse_args()

  tokenizer = Tokenizer.from_file("./tokenizer.json")
  weights = convert_from_huggingface_for_this_model(safe_load("./model.safetensors"))
  model = Model()
  load_state_dict(model, weights, strict=False)

  generate(model, tokenizer, args.prompt, args.max_new_tokens, add_bos=not args.no_bos)

if __name__ == "__main__":
  main()
