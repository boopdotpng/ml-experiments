import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import dataset, get_dataset_config, encode, decode, format_example

WEIGHTS = "model.pt"

n_layers = 2  # one routing, one carry
n_heads = 4
emb_dim = 32
head_dim = 8  # emb_dim / n_heads
mlp_hidden = 128


def init_weight(*shape):
  return nn.Parameter(torch.randn(*shape) * 0.02)


class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.up = init_weight(emb_dim, mlp_hidden)  # 32, 128
    self.down = init_weight(mlp_hidden, emb_dim)  # 128, 32

  def forward(self, x: torch.Tensor):
    # x.shape = (B, S, 32)
    x = x @ self.up  # B, S, 128
    x = x.relu()
    x = x @ self.down  # B, S, 32
    return x


class SelfAttention(nn.Module):
  def __init__(self):
    super().__init__()
    self.q_proj = init_weight(emb_dim, n_heads * head_dim)
    self.k_proj = init_weight(emb_dim, n_heads * head_dim)
    self.v_proj = init_weight(emb_dim, n_heads * head_dim)
    self.o_proj = init_weight(n_heads * head_dim, emb_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, S, _ = x.shape  # batch, sequence length

    # reshape into 4 heads
    q = (x @ self.q_proj).reshape(B, S, n_heads, head_dim).transpose(1, 2)  # B, 4, S, 8
    k = (x @ self.k_proj).reshape(B, S, n_heads, head_dim).transpose(1, 2)  # B, 4, S, 8
    v = (x @ self.v_proj).reshape(B, S, n_heads, head_dim).transpose(1, 2)  # B, 4, S, 8

    # parallel computation for 4 heads each (S, 8) @ (8, S)
    qk = q @ k.transpose(-2, -1)  # only transpose last two dims # (B, 4, S, S)
    mask = torch.ones(S, S, device=x.device, dtype=torch.bool).tril()
    qk = qk.masked_fill(~mask, -float("inf"))
    attn = (qk / head_dim ** 0.5).softmax(dim=-1)  # (B, 4, S, S) attention weights
    a = attn @ v  # (B, 4, S, 8)

    # merge heads back
    # (B, 4, S, 8) -> (B, S, 4, 8) -> (B, S, 32)
    a = a.transpose(1, 2).reshape(B, S, n_heads * head_dim)
    return a @ self.o_proj  # (B, S, 32)


class RMSNorm(nn.Module):
  def __init__(self, dim=emb_dim, eps=1e-5):
    super().__init__()
    self.g = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    rms = (x.square().mean(dim=-1, keepdim=True) + self.eps).sqrt()
    return x / rms * self.g


class Block(nn.Module):
  def __init__(self):
    super().__init__()
    self.attn = SelfAttention()
    self.mlp = MLP()

    # RMSNorm
    self.norm1 = RMSNorm()
    self.norm2 = RMSNorm()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # norm into attn, never save norm values directly
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x


class Model(nn.Module):
  def __init__(self, config=None):
    super().__init__()
    self.config = config or get_dataset_config()
    self.tok_emb = init_weight(self.config.vocab_size, emb_dim)
    self.pos_emb = init_weight(self.config.seq_len, emb_dim)

    self.layers = nn.ModuleList([Block() for _ in range(n_layers)])

    self.norm_f = RMSNorm()
    self.lm_head = init_weight(emb_dim, self.config.vocab_size)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    _, S = x.shape
    x = self.tok_emb[x] + self.pos_emb[:S]
    for layer in self.layers:
      x = layer(x)
    x = self.norm_f(x)
    return x @ self.lm_head


BATCH_SIZE = 256
LR = 1e-3
STEPS = 10000
EVAL_EVERY = 200
TARGET_ACC = 1.0  # deterministic task: 100% exact-match = solved
EVAL_N = 512
N_TRAIN = 20000
N_TEST = 2000


def get_device():
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_fn(logits: torch.Tensor, y: torch.Tensor, config) -> torch.Tensor:
  # Supervise only the generated scratchpad plus final answer.
  logits = logits[:, config.answer_start:]
  y = y[:, config.answer_start:]
  return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def get_batch(X: torch.Tensor, Y: torch.Tensor):
  idx = torch.randint(X.shape[0], (BATCH_SIZE,), device=X.device)
  return X[idx], Y[idx]


@torch.no_grad()
def evaluate(model, X: torch.Tensor, Y: torch.Tensor, config, n: int = 512, final_only=True) -> float:
  # Generate the whole target autoregressively from the prompt and require an
  # exact final-answer match. The cot model still has to generate its scratchpad
  # to reach those final tokens, but harmless formatting drift is not counted as
  # failure unless final_only is disabled.
  was_training = model.training
  model.eval()
  X, Y = X[:n], Y[:n]
  seq = X[:, :config.prompt_len]
  for _ in range(config.target_len):
    nxt = model(seq)[:, -1].argmax(dim=-1, keepdim=True)  # greedy next token (n, 1)
    seq = torch.cat((seq, nxt), dim=1)
  pred = seq[:, config.prompt_len:]
  true = Y[:, config.answer_start:]
  if final_only:
    pred = pred[:, -config.answer_len:]
    true = Y[:, -config.answer_len:]
  acc = (pred == true).all(dim=1).float().mean().item()
  if was_training:
    model.train()
  return acc


@torch.no_grad()
def predict(model, a, b):
  # Greedy autoregressive generation from the prompt.
  was_training = model.training
  model.eval()
  config = model.config
  device = next(model.parameters()).device
  prompt = format_example(a, b)[:config.prompt_len]
  seq = torch.tensor([encode(prompt, config)], dtype=torch.long, device=device)
  for _ in range(config.target_len):
    nxt = model(seq)[:, -1].argmax(dim=-1, keepdim=True)
    seq = torch.cat((seq, nxt), dim=1)
  raw = decode(seq[0, config.prompt_len:].detach().cpu().tolist(), config)
  marker = config.final_marker
  final = raw.rsplit(marker, 1)[-1] if marker in raw else ""
  pred = int(final) if final.isdigit() else None
  if was_training:
    model.train()
  return pred, raw


def train():
  torch.manual_seed(0)
  config = get_dataset_config()
  device = get_device()
  trainx, trainy, testx, testy = dataset(n_train=N_TRAIN, n_test=N_TEST)
  trainx, trainy = trainx.to(device), trainy.to(device)
  testx, testy = testx.to(device), testy.to(device)
  model = Model(config).to(device)
  opt = torch.optim.AdamW(model.parameters(), lr=LR)

  step = 0
  for step in range(STEPS):
    x, y = get_batch(trainx, trainy)
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y, config)
    loss.backward()
    opt.step()
    if step % EVAL_EVERY == 0:
      acc = evaluate(model, testx, testy, config, n=min(EVAL_N, testx.shape[0]))
      print(f"step {step:5d}  loss {loss.item():.4f}  final_acc {acc:6.2%}")
      if acc >= TARGET_ACC:  # confirm on full test set
        full = evaluate(model, testx, testy, config, n=testx.shape[0])
        if full >= 0.999:
          print(
            f"*** solved: {full:.2%} on full {testx.shape[0]} test examples "
            f"- early stopping at step {step}"
          )
          break

  final = evaluate(model, testx, testy, config, n=testx.shape[0])
  print(f"final final_acc {final:.2%} on {testx.shape[0]} test examples")

  torch.save(model.state_dict(), WEIGHTS)
  print(f"saved weights -> {WEIGHTS}")
  return model


def load_model():
  config = get_dataset_config()
  device = get_device()
  model = Model(config).to(device)
  model.load_state_dict(torch.load(WEIGHTS, map_location=device))
  return model


def parse_expr(expr):
  a, b = (int(t) for t in expr.split("+", 1))
  return a, b


if __name__ == "__main__":
  if len(sys.argv) == 1 or sys.argv[1] == "train":
    train()
  elif sys.argv[1] == "infer":
    expr = sys.argv[2] if len(sys.argv) > 2 else "7314+2890"
    a, b = parse_expr(expr)
    pred, raw = predict(load_model(), a, b)
    if pred is None:
      print(f"{a} + {b} = ?   [model emitted non-numeric tokens '{raw}']")
    else:
      ok = "ok" if pred == a + b else f"wrong, true {a + b}"
      print(f"{a} + {b} = {pred}   [model tokens '{raw}']   {ok}")
  else:
    print("usage: python3 main.py [train|infer [A+B]]")
    raise SystemExit(2)
