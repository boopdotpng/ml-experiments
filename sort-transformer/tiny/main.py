import time
from tinygrad import Tensor, TinyJit, Variable, nn, Context, Device, GlobalCounters
from tinygrad.helpers import getenv
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save
from dataset import BOS_ID, EOS_ID, PAD_ID, SEP_ID, decode, dataset, encode

WEIGHTS = "model.safetensors"

n_heads = 4
emb_dim = 128
n_layers = 4 
mlp_hidden = 256
head_dim = emb_dim // n_heads # 32 
n_kv_heads = 2 # GQA switches this to 2 heads, since kv heads are shared
assert n_heads % n_kv_heads == 0
n_rep = n_heads // n_kv_heads
vocab_size = 15 # 0-9 " " pad, bos, eos, sep
max_seq_len = 128 # max model capability , dataset only goes to 48

batch = 256
LR = 1e-3
STEPS = 1000
EVAL_EVERY = 50
EVAL_BS = 512

class MLP:
  def __init__(self):
    self.up_proj = nn.Linear(emb_dim, mlp_hidden, bias=False)
    self.gate_proj = nn.Linear(emb_dim, mlp_hidden, bias=False)
    self.down_proj = nn.Linear(mlp_hidden, emb_dim, bias=False)
  def __call__(self, x:Tensor) -> Tensor:
    return self.down_proj(self.gate_proj(x).swish() * self.up_proj(x))

# RoPE tables. angles depend ONLY on absolute position + pair index, never on the
# data, so compute them once for every position the model supports and slice.
def _rope_tables() -> tuple[Tensor, Tensor]:
  # (16,) how fast each pair rotates as position increases
  freq = 1.0 / (10000.0 ** (Tensor.arange(0, head_dim, 2) / head_dim))
  # angles[position, pair] = position * freq[pair]  ->  (max_seq_len, 16)
  angles = Tensor.arange(max_seq_len).reshape(max_seq_len, 1) * freq.reshape(1, head_dim//2)
  return angles.cos().contiguous(), angles.sin().contiguous()
ROPE_COS, ROPE_SIN = _rope_tables()

def apply_rope(t: Tensor, start_pos) -> Tensor:
  # t: (B, H, S, head_dim). token i in this chunk sits at ABSOLUTE position start_pos+i.
  # start_pos is an int, or a bound Variable inside the jitted decode -- slicing
  # with a symbolic bound works because S stays concrete: (start_pos+S)-start_pos = S
  B, H, S, _ = t.shape
  # (B, H, S, 16, 2): ..., pair, component. [(x0,x1), (x2,x3), ...]
  pairs = t.reshape(B, H, S, head_dim//2, 2)
  cos = ROPE_COS[start_pos:start_pos+S].reshape(1, 1, S, head_dim//2)
  sin = ROPE_SIN[start_pos:start_pos+S].reshape(1, 1, S, head_dim//2)
  even, odd = pairs[..., 0], pairs[..., 1]
  return Tensor.stack(
    even * cos - odd * sin,
    even * sin + odd * cos,
    dim=-1,
  ).reshape(B, H, S, head_dim)

# True where attending is FORBIDDEN: query row i may not see key col j > i.
# scores are the 5-D grouped GQA shape (B, n_kv_heads, n_rep, Q, K), so masks
# carry singleton axes to broadcast there. this helper is the one place that
# knows that shape -- pad masks must match it (see Model.__call__).
def causal_mask(S: int) -> Tensor:
  return (Tensor.ones(S,S).tril() == 0).reshape(1, 1, 1, S, S)

class Attention:
  def __init__(self):
    self.q_proj = nn.Linear(emb_dim, n_heads*head_dim, bias=False) # 128, 128
    # GQA, less heads
    self.k_proj = nn.Linear(emb_dim, n_kv_heads*head_dim, bias=False) # 128, 64
    self.v_proj = nn.Linear(emb_dim, n_kv_heads*head_dim, bias=False) # 128, 64

    # projects back to the opposite of q_proj
    self.o_proj = nn.Linear(n_heads*head_dim, emb_dim, bias=False) # 128, 128

    # KV cache, allocated lazily on first cached call (B isn't known until then).
    # full max_seq_len up front: generation only ever writes into it, never reallocates.
    # k and v share one buffer (dim 0: 0=k, 1=v) so each step is ONE assign/kernel, not two
    self.cache_kv: Tensor | None = None  # (2, B, n_kv_heads, max_seq_len, head_dim)

  def __call__(self, x:Tensor, mask: Tensor | None, start_pos=None) -> Tensor:
    # start_pos=None: training/eval. full sequence, no cache.
    # start_pos=int (or bound Variable when jitted): inference. write this chunk's
    #   k,v into the cache at [start_pos:start_pos+S], attend over cache[:start_pos+S].
    B, S, _ = x.shape
    # x.shape = batch, 48, 128 # 48 is the seq len for all examples (padded)
    # at inference S is the chunk: whole prompt on prefill, 1 on decode steps

    # B, S, 128  ->  B, S, 4, 32 -> B, 4, S, 32
    q = self.q_proj(x).reshape(B, S, n_heads, head_dim).transpose(2, 1)

    # GQA: these get split into two heads instead of 4 (n_kv_heads)

    # B, S, 128  ->  B, S, 2, 32 -> B, 2, S, 32
    k = self.k_proj(x).reshape(B, S, n_kv_heads, head_dim).transpose(2,1)

    # B, S, 128  ->  B, S, 2, 32 -> B, 2, S, 32
    v = self.v_proj(x).reshape(B, S, n_kv_heads, head_dim).transpose(2,1)

    # RoPE here.
    # rotate BEFORE caching: a token's rotation depends only on its own absolute
    # position, which never changes -- so post-RoPE k is safe to store forever.
    # NOT `start_pos or 0`: a bound Variable is a graph node, asking it for
    # truthiness is meaningless. only `is None` is a safe python-side question.
    pos = 0 if start_pos is None else start_pos
    q = apply_rope(q, pos)
    k = apply_rope(k, pos)

    if start_pos is not None:
      if self.cache_kv is None:
        # .contiguous() = give the zeros a real buffer, .realize() = allocate it NOW
        self.cache_kv = Tensor.zeros(2, B, n_kv_heads, max_seq_len, head_dim).contiguous().realize()
      # copy the S new rows in (prefill: S rows at once, decode: 1 row)
      # stack(k, v) -> (2, B, 2, S, 32), lands in both halves with one kernel launch
      self.cache_kv[:, :, :, start_pos:start_pos+S].assign(Tensor.stack(k, v)).realize()
      # attend over everything so far. note: cached k,v are the 2-head GQA
      # tensors -- half the memory of caching post-repeat_interleave.
      k = self.cache_kv[0, :, :, :start_pos+S]
      v = self.cache_kv[1, :, :, :start_pos+S]

    # gqa, the no-copy way. repeat_interleave would inflate k,v back to 4 heads
    # and re-read every cached byte n_rep times. instead, expose the sharing:

    # group q heads by the kv head they share -- q heads [0,1] use kv head 0,
    # [2,3] use kv head 1. same pairing repeat_interleave's h0 h0 h1 h1 gave us.
    # B, 4, S, 32  ->  B, 2, 2, S, 32   (B, n_kv_heads, n_rep, S, head_dim)
    q = q.reshape(B, n_kv_heads, n_rep, S, head_dim)

    # k,v stay compact. the singleton group axis broadcasts in the matmul, so
    # both q heads in a group read the SAME kv memory -- no duplication anywhere
    # B, 2, K, 32  ->  B, 2, 1, K, 32   (K = S here, or start_pos+S from cache)
    k = k.unsqueeze(2)
    v = v.unsqueeze(2)

    # the actual attention calculation, now grouped
    # training: (B, 2, 2, S, S). decode: (B, 2, 2, 1, start_pos+1) -- one row
    scores = q @ k.transpose(-2, -1)
    scores = scores * (head_dim ** -0.5)
    if mask is not None: scores = scores.masked_fill(mask, -float("inf"))
    probs = scores.softmax(axis=-1)

    ctx = probs @ v # (B, n_kv_heads, n_rep, S, head_dim)

    # merge (n_kv_heads, n_rep) back into n_heads, then back to output proj
    # B, 2, 2, S, 32  ->  B, 4, S, 32  ->  B, S, 128
    ctx = ctx.reshape(B, n_heads, S, head_dim).transpose(2, 1).reshape(B, S, n_heads * head_dim)

    return self.o_proj(ctx)

class Block:
  def __init__(self):
    self.attn_norm = nn.RMSNorm(emb_dim)
    self.attn = Attention()
    self.mlp_norm = nn.RMSNorm(emb_dim)
    self.mlp = MLP()
  def __call__(self, x:Tensor, mask: Tensor | None, start_pos: int | None = None) -> Tensor:
    x = x + self.attn(self.attn_norm(x), mask, start_pos)
    x = x + self.mlp(self.mlp_norm(x))
    return x

class Model:
  def __init__(self):
    self.tok_emb = nn.Embedding(vocab_size, emb_dim)
    self.layers = [Block() for _ in range(n_layers)]
    self.norm = nn.RMSNorm(emb_dim)
    self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False) # shape tbd

    # jit for the steady-state decode step ONLY. rules it satisfies: input is
    # always (1, 1), start_pos varies but enters as a bound Variable (symbolic,
    # patched into the tape at replay), weights/cache are mutated in place.
    self.generate_jit = TinyJit(self.generate_step)

  def __call__(self, x:Tensor) -> Tensor:
    # training/eval path: full sequence, no cache.
    B, S = x.shape

    # attention mask
    # during training, all samples are right-padded

    # future mask (no cheating)
    future_mask = causal_mask(S)

    # padding token mask. pad is a property of the KEY column, so it sits in the
    # last axis and broadcasts over every query row / head / group
    pad_mask = (x == PAD_ID).reshape(B, 1, 1, 1, S) # 11 is the pad token

    mask = future_mask | pad_mask

    x = self.tok_emb(x)
    for layer in self.layers: x = layer(x, mask)
    x = self.norm(x)
    return self.lm_head(x)

  def generate(self, x:Tensor, start_pos: int) -> Tensor:
    # dispatch: decode steps are all shape (1,1) so they can share one jitted tape,
    # with start_pos bound to a symbolic Variable. prefill has a different prompt
    # length every call -- jitting it would just violate the fixed-shape rule.
    if x.shape == (1, 1) and start_pos > 0:
      return self.generate_jit(x, Variable("start_pos", 1, max_seq_len-1).bind(start_pos))
    return self.generate_step(x, start_pos)

  def generate_step(self, x:Tensor, start_pos) -> Tensor:
    # inference path: x is a CHUNK of new tokens, the cache holds the past.
    # prefill: x = whole prompt, start_pos=0. decode: x = one token, start_pos=N.
    B, S = x.shape

    # future mask (no cheating) -- only needed on prefill. a decode step's single
    # query sits at the last position and may see the whole cache, so no mask.
    # no pad mask at all: live generation has no pad tokens.
    mask = None if S == 1 else causal_mask(S)

    x = self.tok_emb(x)
    for layer in self.layers: x = layer(x, mask, start_pos)
    x = self.norm(x)
    return self.lm_head(x[:, -1]) # only the last position's logits matter here

def loss_fn(logits: Tensor, y: Tensor) -> Tensor:
  mask = y != -1
  safe_y = mask.where(y, 0)
  loss = logits.sparse_categorical_crossentropy(safe_y, reduction="none")
  return (loss * mask).sum() / mask.sum()

def prompt_tokens(lst: list[int]) -> list[int]:
  return [BOS_ID] + encode(" ".join(map(str, lst))) + [SEP_ID]

def predict(model: Model, lst: list[int], max_new: int | None = None) -> tuple[str, list[int]]:
  ids = prompt_tokens(lst)
  max_new = max_new if max_new is not None else 2 * len(lst) + 2
  out: list[int] = []

  # prefill: the whole prompt at once. populates the kv cache and hands us the
  # logits for what comes after the prompt. runs ONCE.
  logits = model.generate(Tensor([ids]), start_pos=0)

  for _ in range(max_new):
    nxt = int(logits.argmax(axis=-1).item())
    ids.append(nxt)
    if nxt == EOS_ID: break
    out.append(nxt)
    if len(ids) >= max_seq_len: break # cache is full
    # decode: ONLY the new token goes in. its k,v get appended to the cache,
    # its q attends over everything cached so far.
    logits = model.generate(Tensor([[nxt]]), start_pos=len(ids)-1)
  return decode(out), ids

def run_inference_report(model: Model):
  examples = [
    ("in_dist", [5, 2, 8, 2, 1]),
    ("long_len20", [9, 0, 8, 1, 7, 2, 6, 3, 5, 4, 9, 1, 8, 2, 7, 3, 6, 4, 5, 0]),
  ]
  for name, values in examples:
    pred, _ = predict(model, values)
    expected = " ".join(map(str, sorted(values)))
    print(f"{name}: input={values}")
    print(f"{name}: pred={pred!r}")
    print(f"{name}: expected={expected!r}")
    print(f"{name}: exact={pred == expected}")

def main():
  Tensor.manual_seed(0)
  xtrain, ytrain, xtest, ytest = dataset()
  model = Model()
  opt = Adam(get_parameters(model), lr=getenv("LR", LR))

  steps = getenv("STEPS", STEPS)
  bs = getenv("BS", batch)
  eval_every = getenv("EVAL_EVERY", EVAL_EVERY)
  eval_bs = min(getenv("EVAL_BS", EVAL_BS), xtest.shape[0])
  target_token_acc = getenv("TARGET_TOKEN_ACC", 1.0)

  @TinyJit
  @Context(TRAINING=1)
  def train_step(X: Tensor, Y: Tensor) -> Tensor:
    opt.zero_grad()
    idx = Tensor.randint(bs, high=X.shape[0])
    loss = loss_fn(model(X[idx]), Y[idx])
    loss.backward()
    return loss.realize(*opt.schedule_step())

  if getenv("BENCH", 0):
    warmup = getenv("WARMUP", 5)
    bench_steps = getenv("BENCH_STEPS", 20)
    for _ in range(warmup): train_step(xtrain, ytrain)
    Device[Device.DEFAULT].synchronize()
    GlobalCounters.reset()
    start = time.perf_counter()
    for _ in range(bench_steps): train_step(xtrain, ytrain)
    Device[Device.DEFAULT].synchronize()
    elapsed = time.perf_counter() - start
    gpu_time = GlobalCounters.time_sum_s
    print(
      f"bench steps={bench_steps} warmup={warmup} batch={bs} device={Device.DEFAULT} "
      f"wall_total={elapsed:.4f}s wall_step={elapsed / bench_steps * 1000:.2f}ms "
      f"gpu_total={gpu_time:.4f}s gpu_step={gpu_time / bench_steps * 1000:.2f}ms "
      f"kernels/step={GlobalCounters.kernel_count / bench_steps:.1f} "
      f"gops/step={GlobalCounters.global_ops / bench_steps / 1e9:.2f} "
      f"gb/step={GlobalCounters.global_mem / bench_steps / 1e9:.2f}"
    )
    return

  @TinyJit
  @Context(TRAINING=0)
  def eval_step(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    logits = model(x)
    loss = loss_fn(logits, y)
    pred = logits.argmax(axis=-1)
    mask = y != -1
    correct = ((pred == y) * mask).sum()
    total = mask.sum()
    acc = correct / total
    wrong = total - correct
    return loss.realize(), acc.realize(), wrong.realize(), total.realize()

  xeval, yeval = xtest[:eval_bs].realize(), ytest[:eval_bs].realize()
  for step in range(steps):
    loss = train_step(xtrain, ytrain)

    if step % eval_every == 0 or step == steps - 1:
      test_loss, test_acc, wrong, total = eval_step(xeval, yeval)
      test_acc_item = test_acc.item()
      wrong_item = wrong.item()
      print(
        f"step {step:5d}  loss {loss.item():.4f}  "
        f"test_loss {test_loss.item():.4f}  token_acc {test_acc_item * 100:6.2f}%  "
        f"wrong {wrong_item:.0f}/{total.item():.0f}"
      )
      if test_acc_item >= target_token_acc:
        print(f"solved: token_acc {test_acc_item * 100:.2f}% at step {step}")
        safe_save(get_state_dict(model), getenv("WEIGHTS", WEIGHTS))
        print(f"saved weights -> {getenv('WEIGHTS', WEIGHTS)}")
        break

  run_inference_report(model)


if __name__ == "__main__":
  main()
