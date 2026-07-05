"""CoT 4-digit addition dataset as next-token prediction.

Example:
  "3412+5879=W1C1W9C0W2C1W9C0A09291"

The scratchpad runs from ones to thousands, then emits the regular forward
zero-padded sum. Each step is "W{digit}C{carry}".

x/y are the standard shifted next-token pair. No loss masking is applied here;
use CONFIG.answer_start to train only the generated target after the prompt.
"""

from dataclasses import dataclass
import random

import torch

N_DIGITS = 4
ANSWER_LEN = N_DIGITS + 1
MAX_OPERAND = 10 ** N_DIGITS - 1
PROMPT_LEN = 2 * N_DIGITS + 2
FINAL_MARKER = "A"  # one-symbol token for "answer:"
# Word-level tokens: each scratchpad word is a single symbol.
#   W = write, C = carry, A = answer
CHARS = list("0123456789") + ["+", "=", "W", "C", "A"]


@dataclass(frozen=True)
class DatasetConfig:
  name: str
  chars: list[str]
  n_digits: int
  answer_len: int
  prompt_len: int
  seq_len: int
  final_marker: str

  @property
  def ctx_len(self):
    return self.seq_len - 1

  @property
  def answer_start(self):
    return self.prompt_len - 1

  @property
  def target_len(self):
    return self.seq_len - self.prompt_len

  @property
  def vocab_size(self):
    return len(self.chars)

  @property
  def stoi(self):
    return {c: i for i, c in enumerate(self.chars)}

  @property
  def itos(self):
    return {i: c for c, i in self.stoi.items()}


COT_STEP_LEN = len("W0C0")  # one symbol per word, no spaces/newline
CONFIG = DatasetConfig(
  name="cot",
  chars=CHARS,
  n_digits=N_DIGITS,
  answer_len=ANSWER_LEN,
  prompt_len=PROMPT_LEN,
  seq_len=PROMPT_LEN + N_DIGITS * COT_STEP_LEN + len(FINAL_MARKER) + ANSWER_LEN,
  final_marker=FINAL_MARKER,
)

stoi = CONFIG.stoi
itos = CONFIG.itos
VOCAB_SIZE = CONFIG.vocab_size
SEQ_LEN = CONFIG.seq_len
CTX_LEN = CONFIG.ctx_len


def get_dataset_config():
  return CONFIG


def encode(s, config=CONFIG):
  return [config.stoi[c] for c in s]


def decode(ids, config=CONFIG):
  return "".join(config.itos[int(i)] for i in ids)


def format_example(a, b):
  carry = 0
  steps = []
  aa = f"{a:0{N_DIGITS}d}"
  bb = f"{b:0{N_DIGITS}d}"
  for i in range(N_DIGITS - 1, -1, -1):
    da, db = int(aa[i]), int(bb[i])
    total = da + db + carry
    write = total % 10
    carry = total // 10
    steps.append(f"W{write}C{carry}")  # was "write {write} carry {carry}\n"
  answer = f"{a + b:0{ANSWER_LEN}d}"  # zero-padded, fixed width kept
  return f"{aa}+{bb}={''.join(steps)}{FINAL_MARKER}{answer}"


def make_xy(a, b):
  seq = encode(format_example(a, b))
  if len(seq) != CONFIG.seq_len:
    raise AssertionError(f"CoT example length {len(seq)} != {CONFIG.seq_len}")
  return seq[:-1], seq[1:]


def dataset(n_train=20000, n_test=2000, seed=0, max_operand=MAX_OPERAND):
  """Disjoint train/test 4-digit addition. Returns trainx, trainy, testx, testy."""
  rng = random.Random(seed)
  if n_train + n_test > (max_operand + 1) ** 2:
    raise ValueError("not enough unique (a,b) pairs for requested size")

  seen, pairs = set(), []
  while len(pairs) < n_train + n_test:
    ab = (rng.randint(0, max_operand), rng.randint(0, max_operand))
    if ab not in seen:
      seen.add(ab)
      pairs.append(ab)
  rng.shuffle(pairs)

  def build(split):
    xy = [make_xy(a, b) for a, b in split]
    return (
      torch.tensor([p[0] for p in xy], dtype=torch.long),
      torch.tensor([p[1] for p in xy], dtype=torch.long),
    )

  trainx, trainy = build(pairs[:n_train])
  testx, testy = build(pairs[n_train:])
  return trainx, trainy, testx, testy


if __name__ == "__main__":
  trainx, trainy, testx, testy = dataset(n_train=8, n_test=4)
  print(f"cot: vocab={CONFIG.vocab_size} seq_len={CONFIG.seq_len} ctx={CONFIG.ctx_len}")
  print(f"trainx {trainx.shape}  testx {testx.shape}")
  x, y = make_xy(3412, 5879)
  print("x:", decode(x))
  print("y:", decode(y))
  print("target:", decode(y[CONFIG.answer_start:]))
