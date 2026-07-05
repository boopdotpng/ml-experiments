import sys
from tinygrad import Tensor, TinyJit
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_save, safe_load
from dataset import dataset, get_dataset_config, encode, decode, format_example

WEIGHTS = "model.safetensors"

n_layers = 2 # one routing, one carry 
n_heads = 4 # 
emb_dim = 32 
head_dim = 8 # emb_dim / n_heads
mlp_hidden = 128 

class MLP:
  def __init__(self):
    self.up = Tensor.normal(emb_dim, mlp_hidden, std=0.02) # 32, 128
    self.down = Tensor.normal(mlp_hidden, emb_dim, std=0.02) # 128, 32

  def __call__(self, x: Tensor):
    # x.shape = (B, S, 32)
    x = x @ self.up # B, S, 128
    x = x.relu()
    x = x @ self.down # B, S, 32
    return x

class SelfAttention:
  def __init__(self):
    self.q_proj = Tensor.normal(emb_dim, n_heads*head_dim, std=0.02)
    self.k_proj = Tensor.normal(emb_dim, n_heads*head_dim, std=0.02)
    self.v_proj = Tensor.normal(emb_dim, n_heads*head_dim, std=0.02)
    self.o_proj = Tensor.normal(n_heads*head_dim, emb_dim, std=0.02)

  def __call__(self, x: Tensor) -> Tensor:
    B, S, _  = x.shape # batch, sequence length 

    # reshape into 4 heads
    q = (x @ self.q_proj).reshape(B, S, n_heads, head_dim).transpose(1,2) # B, 4, S, 8
    k = (x @ self.k_proj).reshape(B, S, n_heads, head_dim).transpose(1,2) # B, 4, S, 8
    v = (x @ self.v_proj).reshape(B, S, n_heads, head_dim).transpose(1,2) # B, 4, S, 8

    # parallel computation for 4 heads each (S, 8) @ (8, S)
    qk = q @ k.transpose(-2, -1) # only transpose last two dims # (B, 4, S, S) 
    mask = Tensor.ones(S, S).tril()
    qk = mask.where(qk, -float("inf"))
    attn = (qk / head_dim ** 0.5).softmax(axis=-1)  # (B, 4, S, S) attention weights
    a = attn @ v                                     # (B, 4, S, 8)

    # merge heads back 
    # (B, 4, S, 8) -> (B, S, 4, 8) -> (B, S, 32)
    a = a.transpose(1,2).reshape(B, S, n_heads * head_dim) 
    return a @ self.o_proj # (B, S, 32)

class RMSNorm:
  def __init__(self, dim=emb_dim, eps=1e-5):
    self.g = Tensor.ones(dim)
    self.eps = eps
  def __call__(self, x: Tensor) -> Tensor:
    rms = (x.square().mean(axis=-1, keepdim=True) + self.eps).sqrt()
    return x / rms * self.g

class Block:
  def __init__(self):
    self.attn = SelfAttention()
    self.mlp = MLP()

    # RMSNorm
    self.norm1 = RMSNorm()
    self.norm2 = RMSNorm()

  def __call__(self, x: Tensor) -> Tensor:
    # norm into attn, never save norm values directly
    x = x + self.attn(self.norm1(x)) 
    x = x + self.mlp(self.norm2(x))
    return x

class Model:
  def __init__(self, config=None):
    self.config = config or get_dataset_config()
    self.tok_emb = Tensor.normal(self.config.vocab_size, emb_dim, std=0.02)
    self.pos_emb = Tensor.normal(self.config.seq_len, emb_dim, std=0.02)

    self.layers = [Block() for _ in range(n_layers)]

    self.norm_f = RMSNorm()
    self.lm_head = Tensor.normal(emb_dim, self.config.vocab_size, std=0.02)
  def __call__(self, x: Tensor) -> Tensor:
    B, S, = x.shape
    x = self.tok_emb[x] + self.pos_emb[:S]
    x = Tensor.sequential(x, self.layers)
    x = self.norm_f(x)
    return x @ self.lm_head

BATCH_SIZE   = 256
LR           = 1e-3
STEPS        = 10000
EVAL_EVERY   = 200
TARGET_ACC   = 1.0                     # deterministic task: 100% exact-match = solved
EVAL_N       = 512
N_TRAIN      = 20000
N_TEST       = 2000

def loss_fn(logits: Tensor, y: Tensor, config) -> Tensor:
  # Supervise only the generated scratchpad plus final answer.
  return logits[:, config.answer_start:].sparse_categorical_crossentropy(y[:, config.answer_start:])

def get_batch(X: Tensor, Y: Tensor):
  idx = Tensor.randint(BATCH_SIZE, high=X.shape[0])
  return X[idx].realize(), Y[idx].realize()

def make_eval_forward(model):
  # Autoregressive eval changes sequence length every token, so each shape needs
  # its own JIT. Reusing one TinyJit across lengths will fail or miscompile.
  jitted = {}
  def forward(x: Tensor) -> Tensor:
    key = x.shape
    if key not in jitted:
      @TinyJit
      @Tensor.train(False)
      def _forward(x: Tensor) -> Tensor:
        return model(x).realize()
      jitted[key] = _forward
    return jitted[key](x)
  return forward

def evaluate(model, X: Tensor, Y: Tensor, config, n: int = 512, final_only=True, forward=None) -> float:
  # Generate the whole target autoregressively from the prompt and require an
  # exact final-answer match. The cot model still has to generate its scratchpad
  # to reach those final tokens, but harmless formatting drift is not counted as
  # failure unless final_only is disabled.
  forward = forward or model
  X, Y = X[:n], Y[:n]
  seq = X[:, :config.prompt_len]
  for _ in range(config.target_len):
    nxt = forward(seq)[:, -1].argmax(axis=-1, keepdim=True)  # greedy next token  (n, 1)
    seq = seq.cat(nxt, dim=1)
  pred = seq[:, config.prompt_len:]
  true = Y[:, config.answer_start:]
  if final_only:
    pred = pred[:, -config.answer_len:]
    true = Y[:, -config.answer_len:]
  return (pred == true).min(axis=1).mean().item()

def predict(model, a, b):
  # Greedy autoregressive generation from the prompt.
  config = model.config
  prompt = format_example(a, b)[:config.prompt_len]
  seq = Tensor([encode(prompt, config)])
  for _ in range(config.target_len):
    nxt = model(seq)[:, -1].argmax(axis=-1, keepdim=True)
    seq = seq.cat(nxt, dim=1)
  raw = decode(seq[0, config.prompt_len:].tolist(), config)
  marker = config.final_marker
  final = raw.rsplit(marker, 1)[-1] if marker in raw else ""
  pred = int(final) if final.isdigit() else None
  return pred, raw

def train():
  Tensor.manual_seed(0)
  config = get_dataset_config()
  trainx, trainy, testx, testy = dataset(n_train=N_TRAIN, n_test=N_TEST)
  model = Model(config)
  opt = AdamW(get_parameters(model), lr=LR)
  Tensor.realize(*get_parameters(model), *get_parameters(opt))
  eval_forward = make_eval_forward(model)

  @TinyJit
  @Tensor.train()
  def train_step(x: Tensor, y: Tensor) -> Tensor:
    opt.zero_grad()
    loss = loss_fn(model(x), y, config)
    loss.backward()
    opt.step()
    return loss.realize()

  step = 0
  for step in range(STEPS):
    x, y = get_batch(trainx, trainy)
    loss = train_step(x, y)
    if step % EVAL_EVERY == 0:
      acc = evaluate(model, testx, testy, config, n=min(EVAL_N, testx.shape[0]), forward=eval_forward)
      print(f"step {step:5d}  loss {loss.item():.4f}  final_acc {acc:6.2%}")
      if acc >= TARGET_ACC:                                    # confirm on full test set
        full = evaluate(model, testx, testy, config, n=testx.shape[0], forward=eval_forward)
        if full >= 0.999:
          print(f"*** solved: {full:.2%} on full {testx.shape[0]} test examples "
                f"- early stopping at step {step}")
          break

  final = evaluate(model, testx, testy, config, n=testx.shape[0], forward=eval_forward)
  print(f"final final_acc {final:.2%} on {testx.shape[0]} test examples")

  safe_save(get_state_dict(model), WEIGHTS)
  print(f"saved weights -> {WEIGHTS}")
  return model

def load_model():
  config = get_dataset_config()
  model = Model(config)
  load_state_dict(model, safe_load(WEIGHTS))
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
