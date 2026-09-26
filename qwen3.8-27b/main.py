from tinygrad import Tensor, nn, GlobalCounters
from tinygrad.nn.state import safe_load, load_state_dict, get_state_dict
import os

# Qwen3.5 27B text config (config.json -> text_config)
emb_dim = 5120
n_layers = 64
mlp_size = 17408

vocab_size = 248320
norm_eps = 1e-6

# full attention layers (every 4th layer: 3, 7, 11, ..., 63)
full_attention_interval = 4
n_heads = 24
n_kv_heads = 4
head_dim = 256
attn_output_gate = True # sigmoid gate on attention output

# linear attention layers (gated deltanet, the other 3 of every 4)
linear_n_k_heads = 16
linear_n_v_heads = 48
linear_k_head_dim = 128
linear_v_head_dim = 128
linear_conv_kernel = 4
linear_output_gate = "swish" # silu(z) gate on the normed output

# multi token prediction head
mtp_n_layers = 1

# your inference limit
# rope tables only generated up until here
max_seq_len = 8192
max_position_embeddings = 262144

# rope params, only the first 25% of each head dim gets rotated
rope_theta = 10000000.0
partial_rotary_factor = 0.25
rotary_dim = int(head_dim * partial_rotary_factor) # 64
# mrope splits the 32 frequency pairs across (t, h, w) positions, interleaved.
# for text only t == h == w, so it's plain rope over rotary_dim
mrope_section = [11, 11, 10]
mrope_interleaved = True

# special tokens (generation_config.json)
bos_token_id = 248044
eos_token_ids = [248046, 248044]
pad_token_id = 248044

# default sampling
temperature = 1.0
top_k = 20
top_p = 0.95

class LanguageModel:
  def __init__(self) -> None:
    self.embed_tokens = nn.Embedding(vocab_size=vocab_size, embed_size=emb_dim)
    self.layers = [Block(i) for i in range(n_layers)]
  def __call__(self, x: Tensor) -> Tensor:
    # x.shape = (BS, 1)
    x = self.embed_tokens(x)
    pass

class Model:
  def __init__(self) -> None:
    self.language_model = LanguageModel()
  def __call__(self, x: Tensor) -> Tensor:
    pass

class Mlp:
  def __init__(self) -> None:
    self.gate_proj = nn.Linear(emb_dim, mlp_size, bias=False)
    self.up_proj = nn.Linear(emb_dim, mlp_size, bias=False)
    self.down_proj = nn.Linear(mlp_size, emb_dim, bias=False)
  def __call__(self, x: Tensor) -> Tensor:
    # x.shape = (1, 1, 5120)
    gate = self.gate_proj(x).silu()
    up = self.up_proj(x)
    return self.down_proj(gate*up)

class Block:
  def __init__(self, layer_no:int) -> None:
    self.mlp = Mlp()
    pass
  def __call__(self, x: Tensor) -> Tensor:
    pass

class SelfAttention:
  def __init__(self) -> None:
    pass
  def __call__(self, x: Tensor) -> Tensor:
    pass

class DeltaNetAttention:
  def __init__(self) -> None:
    pass
  def __call__(self, x: Tensor) -> Tensor:
    pass

def main():
  # temp, print all tensors in model
  sd: dict[str, Tensor]= {}
  for f in sorted(os.listdir(".")):
    if f.endswith(".safetensors"): sd.update(safe_load(f))

  # remove all vision tensors
  # and remove "model" prefix
  sd_new = {}
  for k,v in sd.items():
    if k.startswith(("model.visual.", "mtp.")): continue
    if k.startswith("model."): k = k[len("model."):]
    sd_new[k] = v

  model = Model()

  # print missing safetensors
  model_keys = get_state_dict(model)
  for k in sorted(sd_new.keys() - model_keys.keys()): print("not in model:", k, sd_new[k].shape)
  for k in sorted(model_keys.keys() - sd_new.keys()): print("not in checkpoint:", k, model_keys[k].shape)
if __name__ == "__main__":
  main()
