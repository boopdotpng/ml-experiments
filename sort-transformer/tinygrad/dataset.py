"""Variable-length list sorting as next-token prediction (tinygrad).

Token sequence:  <bos> 5 _ 2 _ 8 _ 2 _ 1 <sep> 1 _ 2 _ 2 _ 5 _ 8 <eos> <pad>*
                 \------- input list -------/      \---- sorted output ----/

<sep> is the (only) boundary: it tells the model the input is done and the
sorted output begins -- required because lists are variable length. At
inference, prefill runs everything up to & including <sep>, then decode emits
the sorted tokens until <eos>.

x = seq[:-1], y = seq[1:].  y is set to -1 (scce's default ignore_index)
everywhere we must NOT supervise: the input region and all <pad> targets.
The attention pad mask is derivable from (x == pad_id), so the model only
needs pad_id from the config -- no separate mask tensor is returned.
"""

from dataclasses import dataclass
import random
from tinygrad import Tensor

CHARS = list("0123456789") + [" "]
SPECIALS = ["<pad>", "<bos>", "<eos>", "<sep>"]
VOCAB = CHARS + SPECIALS
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for c, i in STOI.items()}
PAD_ID, BOS_ID, EOS_ID, SEP_ID = (STOI[t] for t in SPECIALS)
IGNORE = -1  # matches Tensor.sparse_categorical_crossentropy default ignore_index


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "sort"
    min_len: int = 2
    max_len: int = 12
    max_val: int = 9
    vocab_size: int = len(VOCAB)
    pad_id: int = PAD_ID
    bos_id: int = BOS_ID
    eos_id: int = EOS_ID
    sep_id: int = SEP_ID
    ignore_id: int = IGNORE

    @property
    def seq_len(self):  # longest example: <bos> src <sep> dst <eos>, both sides max_len single digits
        return 4 * self.max_len + 1

    @property
    def ctx_len(self):
        return self.seq_len - 1


CONFIG = DatasetConfig()


def get_dataset_config():
    return CONFIG


def encode(s):
    return [STOI[c] for c in s]


def decode(ids):
    return "".join(ITOS[int(i)] for i in ids if int(i) in ITOS and int(i) not in
                   (PAD_ID, BOS_ID, EOS_ID))


def format_example(lst):
    """List -> token-id sequence: <bos> input <sep> sorted <eos>."""
    src = encode(" ".join(map(str, lst)))
    dst = encode(" ".join(map(str, sorted(lst))))
    return [BOS_ID] + src + [SEP_ID] + dst + [EOS_ID]


def make_xy(lst, config=CONFIG):
    """List -> padded (x, y). y[j] = -1 wherever loss must be skipped
    (input region + pad targets); supervised on sorted digits + <eos>."""
    ids = format_example(lst)
    ans = ids.index(SEP_ID) + 1  # index in `ids` of the first supervised target token
    ids += [PAD_ID] * (config.seq_len - len(ids))
    x = ids[:-1]
    y = [t if (j + 1) >= ans and t != PAD_ID else IGNORE for j, t in enumerate(ids[1:])]
    return x, y


def dataset(n_train=20000, n_test=2000, seed=0, min_len=2, max_len=12, max_val=9):
    """Disjoint train/test variable-length sort. Returns trainx, trainy, testx, testy
    as int tinygrad tensors padded to config.seq_len. Loss masking lives in y (== -1)."""
    cfg = DatasetConfig(min_len=min_len, max_len=max_len, max_val=max_val)
    rng = random.Random(seed)

    seen, lists = set(), []
    while len(lists) < n_train + n_test:
        lst = tuple(rng.randint(0, max_val) for _ in range(rng.randint(min_len, max_len)))
        if lst not in seen:
            seen.add(lst)
            lists.append(lst)
    rng.shuffle(lists)

    def build(split):
        xy = [make_xy(l, cfg) for l in split]
        return Tensor([p[0] for p in xy]), Tensor([p[1] for p in xy])

    trainx, trainy = build(lists[:n_train])
    testx, testy = build(lists[n_train:])
    return trainx, trainy, testx, testy


if __name__ == "__main__":
    cfg = DatasetConfig(max_len=6)
    trainx, trainy, testx, testy = dataset(n_train=8, n_test=4, max_len=6)
    print(f"sort: vocab={cfg.vocab_size} seq_len={cfg.seq_len} "
          f"pad={PAD_ID} bos={BOS_ID} eos={EOS_ID} sep={SEP_ID}")
    print(f"trainx {trainx.shape}  testx {testx.shape}")
    x, y = make_xy([5, 2, 8, 2, 1], cfg)
    print("x:     ", decode(x))
    print("y(sup):", decode([t for t in y if t != IGNORE]))  # should read the sorted answer
