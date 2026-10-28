from tinygrad import Tensor
import numpy as np
import string

# vocab: letters a-z are tokens 0-25, digits 0-9 are tokens 26-35.
# task 2b adds one more output token, "-" (36): "the query key isn't in the sequence". it never appears in an input
VOCAB = string.ascii_lowercase + string.digits
N_LETTERS = 26
NONE = len(VOCAB)

def encode(s:str) -> list[int]: return [(VOCAB + "-").index(c) for c in s]
def decode(toks) -> str: return ''.join((VOCAB + "-")[int(t)] for t in toks)

# X: (n, seq_len) random letters with exactly one digit somewhere before the last position, e.g. "qwdk7mzpa"
# y: (n,) token of that digit. the last position is always a letter, so the answer has to be fetched from elsewhere
def hidden_digit_dataset(n_train:int=10000, n_test:int=1000, seq_len:int=9, seed:int=0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  rng = np.random.default_rng(seed)
  n = n_train + n_test

  X = rng.integers(0, N_LETTERS, (n, seq_len)).astype(np.int32)
  pos = rng.integers(0, seq_len - 1, n)                      # where the digit hides
  y = (rng.integers(0, 10, n) + N_LETTERS).astype(np.int32)
  X[np.arange(n), pos] = y

  return Tensor(X[:n_train]), Tensor(X[n_train:]), Tensor(y[:n_train]), Tensor(y[n_train:])

# X: (n, 2*pairs+1) tokens like "a2f5c9d3" + query key -> "a2f5c9d3f"
# y: (n,) token of the value stored at the query key
def lookup_dataset(n_train:int=10000, n_test:int=1000, pairs_per_sequence:int=4, seed:int=0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  rng = np.random.default_rng(seed)
  n, p = n_train + n_test, pairs_per_sequence

  keys = rng.random((n, N_LETTERS)).argsort(axis=1)[:, :p]   # p unique letters per row
  values = rng.integers(0, 10, (n, p))
  qi = rng.integers(0, p, n)                                 # which pair gets queried

  X = np.empty((n, 2*p + 1), dtype=np.int32)
  X[:, 0:2*p:2] = keys
  X[:, 1:2*p:2] = values + N_LETTERS
  X[:, -1] = keys[np.arange(n), qi]
  y = (values[np.arange(n), qi] + N_LETTERS).astype(np.int32)

  return Tensor(X[:n_train]), Tensor(X[n_train:]), Tensor(y[:n_train]), Tensor(y[n_train:])

# lookup_dataset, but in a fraction `missing` of the rows the query key is a letter that isn't one of the keys, e.g. "a2f5c9d3x".
# y: the value at the query key, or NONE ("-") when it's missing
def lookup_missing_dataset(n_train:int=10000, n_test:int=1000, pairs_per_sequence:int=4, missing:float=0.25, seed:int=0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  rng = np.random.default_rng(seed)
  n, p = n_train + n_test, pairs_per_sequence

  letters = rng.random((n, N_LETTERS)).argsort(axis=1)      # first p are the keys, the rest are free for a missing query
  keys = letters[:, :p]
  values = rng.integers(0, 10, (n, p))
  qi = rng.integers(0, p, n)
  absent = rng.random(n) < missing

  X = np.empty((n, 2*p + 1), dtype=np.int32)
  X[:, 0:2*p:2] = keys
  X[:, 1:2*p:2] = values + N_LETTERS
  X[:, -1] = np.where(absent, letters[np.arange(n), p + rng.integers(0, N_LETTERS - p, n)], keys[np.arange(n), qi])
  y = np.where(absent, NONE, values[np.arange(n), qi] + N_LETTERS).astype(np.int32)

  return Tensor(X[:n_train]), Tensor(X[n_train:]), Tensor(y[:n_train]), Tensor(y[n_train:])

# X: (n, 2*pairs+2) tokens like "a2f5c9d3" + two different query keys -> "a2f5c9d3fc"
# y: (n,) token of (value at first key + value at second key) mod 10, e.g. (5+9)%10 -> 4
#    or with op="diff", (value at first key - value at second key) mod 10, e.g. (5-9)%10 -> 6
def two_key_dataset(n_train:int=50000, n_test:int=1000, op:str="sum", pairs_per_sequence:int=4, seed:int=0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  rng = np.random.default_rng(seed)
  n, p = n_train + n_test, pairs_per_sequence

  keys = rng.random((n, N_LETTERS)).argsort(axis=1)[:, :p]   # p unique letters per row
  values = rng.integers(0, 10, (n, p))
  qi = rng.random((n, p)).argsort(axis=1)[:, :2]             # two different pairs get queried

  X = np.empty((n, 2*p + 2), dtype=np.int32)
  X[:, 0:2*p:2] = keys
  X[:, 1:2*p:2] = values + N_LETTERS
  X[:, -2:] = np.take_along_axis(keys, qi, axis=1)
  v = np.take_along_axis(values, qi, axis=1)
  y = ((v[:, 0] + v[:, 1] if op == "sum" else v[:, 0] - v[:, 1]) % 10 + N_LETTERS).astype(np.int32)

  return Tensor(X[:n_train]), Tensor(X[n_train:]), Tensor(y[:n_train]), Tensor(y[n_train:])
