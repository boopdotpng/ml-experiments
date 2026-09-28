from tinygrad import Tensor, dtypes, nn, GlobalCounters
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

# COS, SIN, precomputed rope tables
def rope_table():
  # tinyshape: run
  inv_freq = 1.0 / (rope_theta ** (Tensor.arange(0, rotary_dim, 2) / rotary_dim))
  angles = Tensor.arange(max_seq_len).float().unsqueeze(1) * inv_freq.unsqueeze(0)
  angles = angles.repeat(1,2)
  return angles.cos().contiguous(), angles.sin().contiguous()

COS, SIN = rope_table()

def apply_rope(x: Tensor, pos:int):
  # x.shape = (2, 3, n_heads, head_dim)
  S = x.shape[1] # sequence length
  x_rot = x[..., :64] # rope only on first 64 head_dim
  x1 = x_rot[..., :32] # first 32 pairs
  x2 = x_rot[..., 32:] # second 32 pairs # hf stores 0..32 and 32..64, not next to each other

  # broadcast cos and sin to match q and k (both b,s,h,d)
  cos = COS[pos:pos+S, :rotary_dim//2].reshape(1, S, 1, rotary_dim//2)
  sin = SIN[pos:pos+S, :rotary_dim//2].reshape(1, S, 1, rotary_dim//2)

  out1 = x1*cos - x2*sin
  out2 = x2*cos + x1*sin

  return out1.cat(out2, x[..., rotary_dim:], dim=-1).cast(x.dtype)

class QwenRMSNorm:
  def __init__(self, dim:int, eps:float = norm_eps):
    self.weight = Tensor.zeros(dim) # stored as offset from 1
    self.eps = eps
  def __call__(self, x:Tensor) -> Tensor:
    xf = x.float()
    norm = xf * (xf.square().mean(axis=-1, keepdim=True) + self.eps).rsqrt()
    return ((1 + self.weight.float()) * norm).cast(x.dtype)

class LanguageModel:
  def __init__(self) -> None:
    self.embed_tokens = nn.Embedding(vocab_size=vocab_size, embed_size=emb_dim)
    self.layers = [Block(i) for i in range(n_layers)]
    self.norm = QwenRMSNorm(emb_dim)
  def __call__(self, x: Tensor, pos:int) -> Tensor:
    # x.shape = (B, 3)
    x = self.embed_tokens(x)
    for layer in self.layers: x = layer(x, pos)
    x = self.norm(x)
    return x

class Model:
  def __init__(self) -> None:
    self.language_model = LanguageModel()
    self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False) # not tied to embed_tokens
  def __call__(self, x: Tensor, pos:int) -> Tensor:
    # x.shape = (B, 3)
    x = self.language_model(x, pos)
    return self.lm_head(x[:, -1, :]).argmax(-1)

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
    self.use_full_attn = (layer_no+1) % full_attention_interval == 0 # full_attn?
    self.mlp = Mlp()
    self.input_layernorm = QwenRMSNorm(emb_dim)
    self.post_attention_layernorm = QwenRMSNorm(emb_dim)
    if (layer_no+1) % full_attention_interval == 0:
      self.self_attn = SelfAttn()
    else:
      self.linear_attn = LinearAttn()

  def __call__(self, x: Tensor, pos:int) -> Tensor:
    h = self.input_layernorm(x)
    h = self.self_attn(h, pos) if self.use_full_attn else self.linear_attn(h)
    x = x + h
    return x + self.mlp(self.post_attention_layernorm(x))

class SelfAttn:
  def __init__(self) -> None:
    self.k_norm = QwenRMSNorm(head_dim)
    self.q_norm = QwenRMSNorm(head_dim)

    self.q_proj = nn.Linear(emb_dim, n_heads * head_dim * 2, bias=False) # query + output gate
    self.k_proj = nn.Linear(emb_dim, n_kv_heads * head_dim, bias=False)
    self.v_proj = nn.Linear(emb_dim, n_kv_heads * head_dim, bias=False)
    self.o_proj = nn.Linear(n_heads * head_dim, emb_dim, bias=False)

    # todo! stop hardcoding batch size
    self.kv_cache = None

  def __call__(self, x: Tensor, pos:int) -> Tensor:
    # x.shape = (1, 3, emb_dim)
    B, S, _ = x.shape # batch, seq_len
    # llama3 attn with annoying extra steps
    q_and_gate = self.q_proj(x).reshape(B, S, n_heads, head_dim*2)
    # split the gate
    q = q_and_gate[..., :head_dim]
    q_gate = q_and_gate[..., head_dim:]

    k = self.k_proj(x).reshape(B, S, n_kv_heads, head_dim)
    v = self.v_proj(x).reshape(B, S, n_kv_heads, head_dim)

    # q and k norms
    q = self.q_norm(q)
    k = self.k_norm(k)

    # rope on q and k
    q = apply_rope(q, pos)
    k = apply_rope(k, pos)

    # write kv cache
    #             2, B  seq
    if self.kv_cache is None:
      self.kv_cache = Tensor.empty(2, B, max_seq_len, n_kv_heads, head_dim, dtype=k.dtype).realize()
    self.kv_cache[:, :, pos:pos+S].assign(Tensor.stack(k,v))

    # read back from kv cache
    T = pos + S # full range of sequence including kv cache
    k = self.kv_cache[0, :, :T]
    v = self.kv_cache[1, :, :T]

    # GQA, 6 q heads per kv head
    k = k.repeat_interleave(n_heads//n_kv_heads, 2)
    v = v.repeat_interleave(n_heads//n_kv_heads, 2)

    # need last two dims, q (s,d) and k (d,t)
    q = q.transpose(2, 1)
    k = k.transpose(2, 1)
    v = v.transpose(2,1)

    scores = q @ k.transpose(3, 2)
    scores = scores / (head_dim ** 0.5)

    # mask
    if S != 1:
      mask = Tensor.full((S,T), float("-inf")).triu(pos + 1)
      scores = scores + mask

    # attn softmax and values
    out = scores.float().softmax(-1).cast(q.dtype)
    out = (out @ v).transpose(1,2) * q_gate.sigmoid() # qwen gate

    # output projection
    return self.o_proj(out.reshape(B, S, n_heads * head_dim))

lin_key_dim = linear_n_k_heads * linear_k_head_dim # 2048
lin_value_dim = linear_n_v_heads * linear_v_head_dim # 6144
lin_conv_dim = 2 * lin_key_dim + lin_value_dim # 10240, q + k + v

class LinearAttn:
  def __init__(self) -> None:
    self.A_log = Tensor.empty(linear_n_v_heads).float()
    # depthwise causal conv: each of the 10240 qkv channels gets its own 4-tap filter
    self.conv1d = nn.Conv1d(lin_conv_dim, lin_conv_dim, linear_conv_kernel, groups=lin_conv_dim, bias=False)
    self.dt_bias = Tensor.empty(linear_n_v_heads).float()
    self.in_proj_a = nn.Linear(emb_dim, linear_n_v_heads, bias=False)
    self.in_proj_b = nn.Linear(emb_dim, linear_n_v_heads, bias=False)
    self.in_proj_qkv = nn.Linear(emb_dim, lin_conv_dim, bias=False)
    self.in_proj_z = nn.Linear(emb_dim, lin_value_dim, bias=False)
    self.norm = nn.RMSNorm(linear_v_head_dim)
    self.out_proj = nn.Linear(lin_value_dim, emb_dim, bias=False)

    self.conv_state = None

  def __call__(self, x: Tensor) -> Tensor:
    # x.shape = (1, 1, 5120)
    B, S, _ = x.shape
    qkv = self.in_proj_qkv(x) # qkv from regular attention, same thing
    # on float casts: anything used in recurrence needs to be float due to accumulation
    a = self.in_proj_a(x).float() # per head, how much to forget
    b = self.in_proj_b(x).float() # per head, how hard to write
    z = self.in_proj_z(x).reshape(B, S, linear_n_v_heads, linear_v_head_dim).float() # output gate

    if self.conv_state is None:
      # 3 previous tokens
      self.conv_state = Tensor.zeros(B, linear_conv_kernel-1, lin_conv_dim, dtype=qkv.dtype).realize()

    # t-3 t-2 t-1 and t (current token). 3 token lookback
    window = self.conv_state.cat(qkv, dim=1)

    # update conv state. remove the oldest token
    self.conv_state.assign(window[:, 1:])

    # batch, channels, length
    out = self.conv1d(window.transpose(2,1)).transpose(1,2).silu()

    # split qkv
    q, k, v = out.split([lin_key_dim, lin_key_dim, lin_value_dim], dim=-1)
    q = q.reshape(B, 1, 16, 128).float()
    k = k.reshape(B, 1, 16, 128).float()
    v = v.reshape(B, 1, 48, 128).float()

    # l2 norm q and k
    q = q / ((q.square().sum(-1, keepdim=True) + 1e-6).sqrt())
    k = k / ((k.square().sum(-1, keepdim=True) + 1e-6).sqrt())

    q = q / (linear_k_head_dim ** 0.5)

    # repeat!
    q = q.repeat_interleave(3, 2)
    k = k.repeat_interleave(3, 2)



def main():
  # temp, print all tensors in model
  sd: dict[str, Tensor]= {}
  for f in sorted(os.listdir("weights")):
    if f.endswith(".safetensors"): sd.update(safe_load(os.path.join("weights", f)))
  assert sd, "no .safetensors shards in weights/, run ./setup.sh"

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

  model(Tensor.ones(1,1), pos=0)
if __name__ == "__main__":
  main()
