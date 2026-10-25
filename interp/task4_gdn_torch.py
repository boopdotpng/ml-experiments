# pytorch port of task4_gdn.py, for running on a cuda gpu. same model, init, data and training loop.
# usage: python task4_gdn_torch.py <full|noconv|bare|full_nopos> <seed> <steps>
import sys, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

vocab_size, emb_dim, head_dim, mlp_size, N_LETTERS = 36, 32, 32, 64, 26
dev = "cuda" if torch.cuda.is_available() else "cpu"

# same as dataset.two_key_dataset(op="diff"), in numpy so this file doesn't need tinygrad
def two_key_dataset(n_train=50000, n_test=1000, op="diff", p=4, seed=0):
  rng = np.random.default_rng(seed)
  n = n_train + n_test
  keys = rng.random((n, N_LETTERS)).argsort(axis=1)[:, :p]
  values = rng.integers(0, 10, (n, p))
  qi = rng.random((n, p)).argsort(axis=1)[:, :2]
  X = np.empty((n, 2*p + 2), dtype=np.int64)
  X[:, 0:2*p:2] = keys
  X[:, 1:2*p:2] = values + N_LETTERS
  X[:, -2:] = np.take_along_axis(keys, qi, axis=1)
  v = np.take_along_axis(values, qi, axis=1)
  y = ((v[:, 0] + v[:, 1] if op == "sum" else v[:, 0] - v[:, 1]) % 10 + N_LETTERS).astype(np.int64)
  t = lambda a: torch.tensor(a, device=dev)
  return t(X[:n_train]), t(X[n_train:]), t(y[:n_train]), t(y[n_train:])

# tinygrad inits: Linear and Conv1d are uniform(+-1/sqrt(fan_in)), Embedding is glorot uniform
def linear(i, o):
  l = nn.Linear(i, o, bias=False)
  nn.init.uniform_(l.weight, -1/math.sqrt(i), 1/math.sqrt(i))
  return l
def embedding(n, d):
  e = nn.Embedding(n, d)
  nn.init.xavier_uniform_(e.weight)
  return e

class GDN(nn.Module):
  def __init__(self, n_heads=2, conv=True, gates=True):
    super().__init__()
    self.H, self.conv_on, self.gates = n_heads, conv, gates
    C = (2*head_dim + emb_dim)*n_heads
    self.qkv = linear(emb_dim, C)
    if conv:
      self.conv1d = nn.Conv1d(C, C, 4, groups=C, padding=3, bias=False)
      nn.init.uniform_(self.conv1d.weight, -1/math.sqrt(C*4), 1/math.sqrt(C*4))   # tinygrad's fan_in ignores groups
    self.in_a, self.in_b = linear(emb_dim, n_heads), linear(emb_dim, n_heads)
    self.A_log = nn.Parameter(torch.empty(n_heads).uniform_(1, 16).log())
    dt = (torch.rand(n_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).exp()
    self.dt_bias = nn.Parameter((dt.exp() - 1).log())   # inverse softplus, so softplus(dt_bias) = dt
    if gates:
      self.in_z = linear(emb_dim, emb_dim*n_heads)
      self.norm = nn.RMSNorm(emb_dim, eps=1e-6)
  def forward(self, x):
    B, T, H = x.shape[0], x.shape[1], self.H
    qkv = self.qkv(x)
    if self.conv_on: qkv = F.silu(self.conv1d(qkv.transpose(1, 2))[:, :, :T].transpose(1, 2))
    q, k, v = qkv.split([head_dim*H, head_dim*H, emb_dim*H], dim=-1)
    q, k, v = q.reshape(B, T, H, head_dim), k.reshape(B, T, H, head_dim), v.reshape(B, T, H, emb_dim)
    q = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt() * head_dim ** -0.5
    k = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    beta = self.in_b(x).sigmoid()                                               # (B, T, H)
    g = (-self.A_log.exp() * F.softplus(self.in_a(x) + self.dt_bias)).exp()    # (B, T, H)
    state, outs = x.new_zeros(B, H, head_dim, emb_dim), []
    for t in range(T):
      state = state * g[:, t].reshape(B, H, 1, 1)
      kv_mem = (state * k[:, t].unsqueeze(-1)).sum(-2)
      delta = (v[:, t] - kv_mem) * beta[:, t].unsqueeze(-1)
      state = state + k[:, t].unsqueeze(-1) * delta.unsqueeze(-2)
      outs.append((state * q[:, t].unsqueeze(-1)).sum(-2))
    o = torch.stack(outs, dim=1)                                                # (B, T, H, emb_dim)
    if self.gates: o = self.norm(o) * F.silu(self.in_z(x).reshape(B, T, H, emb_dim))
    return o.sum(2)                                                             # sum heads, like model.py (no out proj)

class Mlp(nn.Module):
  def __init__(self):
    super().__init__()
    self.up, self.down = linear(emb_dim, mlp_size), linear(mlp_size, emb_dim)
  def forward(self, x): return self.down(F.relu(self.up(x)))

class Block(nn.Module):
  def __init__(self, mlp, **kw):
    super().__init__()
    self.attn, self.mlp = GDN(**kw), (Mlp() if mlp else None)
  def forward(self, x):
    x = x + self.attn(x)
    return x + self.mlp(x) if self.mlp is not None else x

class Model(nn.Module):
  def __init__(self, pos=True, n_layers=2, **kw):
    super().__init__()
    self.emb = embedding(vocab_size, emb_dim)
    self.pos_emb = embedding(10, emb_dim) if pos else None
    self.layers = nn.ModuleList([Block(mlp=i == n_layers-1, **kw) for i in range(n_layers)])
    self.unembed = linear(emb_dim, vocab_size)
  def forward(self, x):
    x = self.emb(x)
    if self.pos_emb is not None: x = x + self.pos_emb.weight[:x.shape[1]]
    for l in self.layers: x = l(x)
    return self.unembed(x[:, -1])

def train(model, xtrain, ytrain, xtest, ytest, steps, lr, name=""):
  opt = torch.optim.Adam(model.parameters(), lr=lr, fused=dev == "cuda", capturable=dev == "cuda")
  def train_step():
    samples = torch.randint(0, xtrain.shape[0], (512,), device=dev)
    loss = F.cross_entropy(model(xtrain[samples]), ytrain[samples])
    opt.zero_grad(set_to_none=False)
    loss.backward()
    opt.step()
    return loss.detach()
  @torch.no_grad()
  def test_acc(): return (model(xtest).argmax(-1) == ytest).float().mean().item() * 100
  if dev == "cuda":   # capture one step as a cuda graph: the model is tiny, so kernel launches are the whole cost
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
      for _ in range(3): train_step()   # warmup also allocates adam state
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): static_loss = train_step()
    def step(): graph.replay(); return static_loss
  else: step = train_step
  losses = []
  for i in range(steps):
    losses.append(step().clone())
    if i % 1000 == 999: print(f"{name}: loss and val acc at step {i+1}: {losses[-1].item():.2f}, {test_acc():.2f}%")
  return [l.item() for l in losses], test_acc()

if __name__ == "__main__":
  variant, seed, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
  kw = {"full": {}, "noconv": dict(conv=False), "bare": dict(conv=False, gates=False), "full_nopos": dict(pos=False)}[variant]
  torch.manual_seed(seed)
  xtrain, xtest, ytrain, ytest = two_key_dataset()
  m = Model(**kw).to(dev)
  t0 = time.perf_counter()
  _, acc = train(m, xtrain, ytrain, xtest, ytest, steps, 3e-3, f"{variant} s{seed}")
  print(f"RESULT {variant} seed {seed}: {acc:.2f}% in {time.perf_counter()-t0:.1f}s on {dev}")
