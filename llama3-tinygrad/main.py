from tinygrad import Tensor, GlobalCounters, helpers, Context, nn
from tinygrad.nn.state import safe_load, load_state_dict

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

class MLP:
  def __init__(self):
    self.gate_proj = nn.Linear(emb_dim, mlp_size, bias=False)
    self.up_proj = nn.Linear(emb_dim, mlp_size, bias=False)
    self.down_proj = nn.Linear(mlp_size, emb_dim, bias=False)
  def __call__(self, x:Tensor) -> Tensor:
    pass

class Attention:
  def __init__(self):
    self.q_proj = nn.Linear(emb_dim, head_dim * n_heads, bias=False)
    self.k_proj = nn.Linear(emb_dim, head_dim * n_kv_heads, bias=False)
    self.v_proj = nn.Linear(emb_dim, head_dim * n_kv_heads, bias=False)

    self.o_proj = nn.Linear(n_heads * head_dim, emb_dim, bias=False)
    pass
  def __call__(self, x:Tensor) -> Tensor:
    pass

class Block:
  def __init__(self):
    self.mlp = MLP()
    self.post_attention_layernorm = nn.RMSNorm(emb_dim, eps=norm_eps)
    self.input_layernorm = nn.RMSNorm(emb_dim, eps=norm_eps)
    self.self_attn = Attention()

  def __call__(self, x:Tensor) -> Tensor:
    pass

class Model:
  def __init__(self):
    self.embed_tokens = nn.Embedding(vocab_size, emb_dim)
    self.layers = [Block() for _ in range(n_layers)]
    self.norm = nn.RMSNorm(emb_dim, eps=norm_eps)
  def __call__(self, x:Tensor) -> Tensor:
    pass

def convert_from_huggingface_for_this_model(weights:dict[str, Tensor]) -> dict[str, Tensor]:
  converted = {}
  for k,v in weights.items():
    if k.startswith("model."): k = k[len("model."):]
    if k == "lm_head.weight": continue
    converted[k] = v
  return converted

def main():
  weights = convert_from_huggingface_for_this_model(safe_load("./model.safetensors"))
  model = Model()
  load_state_dict(model, weights)

  for k,v in weights.items(): print(k, v.shape)

if __name__ == "__main__":
  main()
