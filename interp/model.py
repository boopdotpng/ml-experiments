from tinygrad import Tensor, nn
from dataset import VOCAB

vocab_size = len(VOCAB)
emb_dim, n_layers, n_heads, head_dim, mlp_size = 32, 2, 1, 32, 64

class Attention:
  def __init__(self, causal=True, n_heads=n_heads):
    self.causal, self.n_heads = causal, n_heads
    self.q_proj = nn.Linear(emb_dim, head_dim*n_heads, bias=False)
    self.k_proj = nn.Linear(emb_dim, head_dim*n_heads, bias=False)
    # no output projection: each head's V writes straight to the residual stream and the heads' writes are summed.
    # same thing as concat + output projection, minus a matrix that only ever acts multiplied with V
    self.v_proj = nn.Linear(emb_dim, emb_dim*n_heads, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    # x.shape = (B, T, emb_dim), T = 9 for task 1, 10 for tasks 3 and 4
    B, T, H = x.shape[0], x.shape[1], self.n_heads
    q = self.q_proj(x).reshape(B, T, H, head_dim).transpose(1, 2)   # (B, H, T, head_dim)
    k = self.k_proj(x).reshape(B, T, H, head_dim).transpose(1, 2)
    v = self.v_proj(x).reshape(B, T, H, emb_dim).transpose(1, 2)    # (B, H, T, emb_dim)

    # causal mask: need to mimic normal model training
    scores = q @ k.transpose(-1, -2) / head_dim ** 0.5              # (B, H, T, T)
    if self.causal: scores = scores + Tensor.full((T,T), float("-inf")).triu(1)
    scores = scores.softmax(-1)

    return (scores @ v).sum(1)                                      # sum over heads -> (B, T, emb_dim)

class Mlp:
  def __init__(self):
    self.up = nn.Linear(emb_dim, mlp_size, bias=False)
    self.down = nn.Linear(mlp_size, emb_dim, bias=False)
  def __call__(self, x: Tensor) -> Tensor: return self.down(self.up(x).relu())

class Block:
  def __init__(self, mlp=False, causal=True, n_heads=n_heads):
    self.attn = Attention(causal, n_heads)
    self.mlp = Mlp() if mlp else None # optimizer unhappy
  def __call__(self, x: Tensor) -> Tensor:
    x = x + self.attn(x)
    if self.mlp : x = x + self.mlp(x)
    return x

class Model:
  # pos=False / causal=False are for the tutorial: without both, the model can't tell token order at all
  def __init__(self, mlp=False, seq_len=9, pos=True, causal=True, n_layers=n_layers, n_heads=n_heads):
    self.emb = nn.Embedding(vocab_size, emb_dim) # 36, 32
    self.pos_emb = nn.Embedding(seq_len, emb_dim) if pos else None # 4 pairs plus one (task 1) or two (task 2) query keys
    # mlp only after the last layer: it transforms the digit once the lookup is done
    self.layers = [Block(mlp=mlp and i == n_layers-1, causal=causal, n_heads=n_heads) for i in range(n_layers)]

    # dont tie embeddings
    self.unembed = nn.Linear(emb_dim, vocab_size, bias=False)
  def __call__(self, x: Tensor) -> Tensor:
    x = self.emb(x)
    if self.pos_emb is not None: x = x + self.pos_emb(Tensor.arange(x.shape[1]))
    for l in self.layers: x = l(x)
    return self.unembed(x[:, -1])
