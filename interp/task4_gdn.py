# task 4 model (2 layers x 2 heads, MLP after layer 2, pos emb) with every softmax attention swapped for a gated deltanet
# (the LinearAttn from qwen3.8-27b, at this size). usage: python task4_gdn.py <full|noconv|bare|full_nopos> <seed> <steps>
import sys, math
from tinygrad import Tensor, nn
from model import Mlp, emb_dim, head_dim, vocab_size
from dataset import two_key_dataset
from analysis import train

# both take q, k (B, H, T, dk), v (B, H, T, dv), beta and log decay (B, H, T); return (B, H, T, dv). q and k already normalized
def recurrent(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, logg:Tensor, _=0) -> Tensor:
  # one token at a time, as in qwen3.8-27b. reference for chunked()
  B, H, T, dk = q.shape
  g, state, outs = logg.exp(), Tensor.zeros(B, H, dk, v.shape[-1]), []
  for t in range(T):
    state = state * g[:, :, t].reshape(B, H, 1, 1)
    kv_mem = (state * k[:, :, t].unsqueeze(-1)).sum(-2)
    delta = (v[:, :, t] - kv_mem) * beta[:, :, t].unsqueeze(-1)
    state = state + k[:, :, t].unsqueeze(-1) * delta.unsqueeze(-2)
    outs.append((state * q[:, :, t].unsqueeze(-1)).sum(-2))
  return Tensor.stack(*outs, dim=2)

def chunked(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, logg:Tensor, C:int=16) -> Tensor:
  # same result, C tokens at a time (the WY form from the gated deltanet paper). inside a chunk everything is matmuls;
  # only the state is carried between chunks. here T = 10 fits in one chunk, so there's no loop at all
  B, H, T, dk = q.shape
  if (P := -T % C): q, k, v, beta, logg = [t.pad(((0,0), (0,0), (0,P)) + ((0,0),)*(t.ndim-3)) for t in (q, k, v, beta, logg)]
  n = (T + P) // C   # padding is at the end with k = v = beta = 0 and no decay, so it can't touch earlier tokens
  q, k, v = [t.reshape(B, H, n, C, t.shape[-1]) for t in (q, k, v)]
  beta, G = beta.reshape(B, H, n, C), logg.reshape(B, H, n, C).cumsum(-1)    # G: total log decay since chunk start
  lower, eye = Tensor.ones(C, C).tril().bool(), Tensor.eye(C)
  decay = lower.where(G.unsqueeze(-1) - G.unsqueeze(-2), -math.inf).exp()   # decay[i, j] = g_(j+1) ... g_i for j <= i, else 0
  kb = k * beta.unsqueeze(-1)
  # every write in the chunk depends on the earlier ones through (I + L), L = strictly lower part of beta_i k_i.k_j decay.
  # L is nilpotent, so (I + L)^-1 = (I + N)(I + N^2)(I + N^4)... with N = -L: log2(C) matmuls instead of a token loop
  N = -(kb @ k.transpose(-1, -2) * decay) * (lower.float() - eye)
  Tinv, Np = eye + N, N
  for _ in range(math.ceil(math.log2(C)) - 1): Np = Np @ Np; Tinv = Tinv @ (eye + Np)
  w = Tinv @ (kb * G.exp().unsqueeze(-1))   # how much each token's write is cancelled by the incoming state
  u = Tinv @ (v * beta.unsqueeze(-1))       # each token's write, if the chunk started from an empty state
  qk = q @ k.transpose(-1, -2) * decay      # causal, decayed attention inside the chunk
  state, outs = Tensor.zeros(B, H, dk, v.shape[-1]), []
  for i in range(n):
    v_new = u[:, :, i] - w[:, :, i] @ state
    outs.append((q[:, :, i] * G[:, :, i].exp().unsqueeze(-1)) @ state + qk[:, :, i] @ v_new)
    if i < n - 1: state = state * G[:, :, i, -1].exp().reshape(B, H, 1, 1) + \
      (k[:, :, i] * (G[:, :, i, -1:] - G[:, :, i]).exp().unsqueeze(-1)).transpose(-1, -2) @ v_new
  return Tensor.stack(*outs, dim=2).reshape(B, H, n*C, v.shape[-1])[:, :, :T]

class GDN:
  def __init__(self, n_heads=2, conv=True, gates=True, chunk=16):
    self.H, self.conv_on, self.gates, self.chunk = n_heads, conv, gates, chunk
    self.qkv = nn.Linear(emb_dim, (2*head_dim + emb_dim)*n_heads, bias=False)
    C = (2*head_dim + emb_dim)*n_heads
    if conv: self.conv1d = nn.Conv1d(C, C, 4, groups=C, padding=3, bias=False)
    self.in_a = nn.Linear(emb_dim, n_heads, bias=False)
    self.in_b = nn.Linear(emb_dim, n_heads, bias=False)
    self.A_log = Tensor.uniform(n_heads, low=1, high=16).log()
    dt = (Tensor.rand(n_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).exp()
    self.dt_bias = (dt.exp() - 1).log()  # inverse softplus, so softplus(dt_bias) = dt
    if gates:
      self.in_z = nn.Linear(emb_dim, emb_dim*n_heads, bias=False)
      self.norm = nn.RMSNorm(emb_dim)
  def __call__(self, x:Tensor) -> Tensor:
    B, T, H = x.shape[0], x.shape[1], self.H
    qkv = self.qkv(x)
    if self.conv_on: qkv = self.conv1d(qkv.transpose(1, 2))[:, :, :T].transpose(1, 2).silu()
    q, k, v = qkv.split([head_dim*H, head_dim*H, emb_dim*H], dim=-1)
    q, k, v = q.reshape(B, T, H, head_dim), k.reshape(B, T, H, head_dim), v.reshape(B, T, H, emb_dim)
    q = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt() * head_dim ** -0.5
    k = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    beta = self.in_b(x).sigmoid()                                               # (B, T, H)
    logg = -self.A_log.exp() * (self.in_a(x) + self.dt_bias).softplus()         # (B, T, H), log of the decay g
    o = (chunked if self.chunk else recurrent)(*(t.transpose(1, 2) for t in (q, k, v, beta, logg)), self.chunk)
    o = o.transpose(1, 2)                                                       # (B, T, H, emb_dim)
    if self.gates: o = self.norm(o) * self.in_z(x).reshape(B, T, H, emb_dim).silu()
    return o.sum(2)                                                             # sum heads, like model.py (no out proj)

class Block:
  def __init__(self, mlp, **kw): self.attn, self.mlp = GDN(**kw), (Mlp() if mlp else None)
  def __call__(self, x):
    x = x + self.attn(x)
    return x + self.mlp(x) if self.mlp else x

class Model:
  def __init__(self, pos=True, n_layers=2, **kw):
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.pos_emb = nn.Embedding(10, emb_dim) if pos else None
    self.layers = [Block(mlp=i == n_layers-1, **kw) for i in range(n_layers)]
    self.unembed = nn.Linear(emb_dim, vocab_size, bias=False)
  def __call__(self, x):
    x = self.emb(x)
    if self.pos_emb is not None: x = x + self.pos_emb(Tensor.arange(x.shape[1]))
    for l in self.layers: x = l(x)
    return self.unembed(x[:, -1])

def check():
  # chunked must match recurrent, forward and gradients, with one chunk (C=16) and several chunks plus padding (C=4)
  Tensor.manual_seed(0)
  B, H, T = 3, 2, 10
  def inputs():
    q, k = [(t := Tensor.randn(B, H, T, head_dim)) / (t.square().sum(-1, keepdim=True)).sqrt() for _ in range(2)]
    return q, k, Tensor.randn(B, H, T, emb_dim), Tensor.rand(B, H, T), -Tensor.rand(B, H, T)
  base = [t.realize() for t in inputs()]
  for C in (16, 4, 3):
    outs = []
    for f in (recurrent, chunked):
      xs = [Tensor(t.numpy()) for t in base]
      o = f(*xs, C)
      grads = (o * Tensor.arange(o.numel()).reshape(o.shape).sin()).sum().gradient(*xs)
      outs.append([o.numpy()] + [g.numpy() for g in grads])
    err = max(abs(a - b).max() for a, b in zip(*outs))
    print(f"chunk {C}: max abs difference to recurrent (output and all grads) {err:.2e}")
    assert err < 1e-4

if __name__ == "__main__":
  if sys.argv[1] == "check": check(); sys.exit()
  variant, seed, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
  kw = {"full": {}, "noconv": dict(conv=False), "bare": dict(conv=False, gates=False), "full_nopos": dict(pos=False)}[variant]
  Tensor.manual_seed(seed)
  xtrain, xtest, ytrain, ytest = two_key_dataset(op="diff")
  m = Model(**kw)
  _, acc = train(m, xtrain, ytrain, xtest, ytest, steps, 3e-3, f"{variant} s{seed}")
  print(f"RESULT {variant} seed {seed}: {acc:.2f}%")
