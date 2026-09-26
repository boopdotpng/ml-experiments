"""Character-level text data for the tiny MoDE language model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import urllib.request

import numpy as np
from tinygrad import Tensor


TINY_SHAKESPEARE_URL = (
  "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
  "data/tinyshakespeare/input.txt"
)


@dataclass(frozen=True)
class CharTokenizer:
  chars: tuple[str, ...]

  @classmethod
  def from_text(cls, text: str):
    return cls(tuple(sorted(set(text))))

  @property
  def vocab_size(self) -> int:
    return len(self.chars)

  @property
  def stoi(self) -> dict[str, int]:
    return {char: i for i, char in enumerate(self.chars)}

  def encode(self, text: str) -> list[int]:
    table = self.stoi
    missing = sorted(set(text) - set(table))
    if missing:
      raise ValueError(f"text contains characters outside the vocabulary: {missing!r}")
    return [table[char] for char in text]

  def decode(self, ids) -> str:
    return "".join(self.chars[int(i)] for i in ids)

  def save(self, path: Path):
    path.write_text(json.dumps({"chars": self.chars}, indent=2) + "\n")

  @classmethod
  def load(cls, path: Path):
    return cls(tuple(json.loads(path.read_text())["chars"]))


def ensure_tiny_shakespeare(path: Path) -> Path:
  if path.exists():
    return path
  path.parent.mkdir(parents=True, exist_ok=True)
  print(f"downloading Tiny Shakespeare -> {path}")
  urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
  return path


def load_corpus(path: str | Path | None = None):
  corpus_path = Path(path) if path else Path(__file__).parent / "data" / "tinyshakespeare.txt"
  if path is None:
    ensure_tiny_shakespeare(corpus_path)
  if not corpus_path.exists():
    raise FileNotFoundError(corpus_path)
  text = corpus_path.read_text()
  if len(text) < 1000:
    raise ValueError(f"corpus is too small ({len(text)} characters): {corpus_path}")
  tokenizer = CharTokenizer.from_text(text)
  encoded = np.fromiter((tokenizer.stoi[c] for c in text), dtype=np.int32, count=len(text))
  split = int(0.9 * len(encoded))
  train, val = Tensor(encoded[:split]), Tensor(encoded[split:])
  return train, val, tokenizer, corpus_path


def get_batch(data: Tensor, batch_size: int, seq_len: int):
  if data.shape[0] <= seq_len:
    raise ValueError(f"data has {data.shape[0]} tokens but seq_len is {seq_len}")
  starts = Tensor.randint(batch_size, high=data.shape[0] - seq_len - 1)
  offsets = Tensor.arange(seq_len + 1).reshape(1, seq_len + 1)
  indices = starts.reshape(batch_size, 1) + offsets
  windows = data.gather(0, indices.flatten()).reshape(batch_size, seq_len + 1)
  return windows[:, :-1], windows[:, 1:]
