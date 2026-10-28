"""Pretraining data: download + tokenize the project mix, and a loader for it.

    python data.py --subset          # ~50M tokens, ~100MB on disk; for testing shapes/train steps locally
    python data.py                   # full 20B tokens, ~40GB on disk; tokenize on a CPU box, not the GPU
    python data.py --tokens 2e9      # any other budget, same mix (debug-scale runs)
    python data.py --sources code wiki   # only some sources (e.g. while gated access is pending)

Sources (fractions of the token budget; the plan leaves ~5% unassigned, so these are
normalized to sum to 1):
    web   60%  Nemotron-CC (v1)                quality=high, actual (organic, no synthetic rephrases);
                                               hosted ungated on Common Crawl's S3, not on HF
    pdfs  15%  HuggingFaceFW/finepdfs          eng_Latn
    code  10%  common-pile/stackv2_edu_filtered  Stack-Edu with file contents, int_score >= 3;
                                               python 30, javascript 15, typescript 10, c 15,
                                               cpp 15, rust 10, shell 5 (% of code)
    math   5%  nvidia/Nemotron-CC-Math-v1      4plus
    wiki   5%  HuggingFaceFW/finewiki          enwiki

The plan names Nemotron-CC-v2.1, but that HF repo is gated (access pending); v1 is the same
pipeline and the one the Nemotron-CC paper's DCLM / FineWeb-Edu comparison is about. Its files
are sorted by crawl (2013 first), so they're taken in a fixed shuffled order to spread across
crawls. Nemotron-CC-Math is gated on HF too: have HF_TOKEN set (or `hf auth login`). A gated
source without access is skipped with a warning and everything else is still written. HuggingFaceTB/stack-edu itself only ships blob ids (contents live on Software
Heritage S3), so code comes from Common Pile's copy, which has the text but keeps only openly
licensed repos and everything with int_score >= 2; we apply Stack-Edu's >= 3 cutoff here, and
strip leading license-header comments.

Parquet / json.gz / jsonl.zstd files are streamed over HTTP (range reads on the hub), tokenized, and
written as llm.c-style shards: a 256 x int32 header (magic, version, ntok) followed by uint16
tokens, with <|endoftext|> in front of every document. Nothing raw is kept on disk.

The tokenizer is the GPT-2 BPE (tiktoken, vocab 50257) as a stand-in until the project's own
32,768-vocab BPE exists; it goes in get_encoder(). meta.json records which one wrote the shards,
and the script refuses to add to a directory written with a different tokenizer or source.

    data/web/val.bin                  held-out split (--val-tokens per source, whole docs)
    data/web/train_00000.bin          one shard per input file
    data/code/python/val.bin          code is split per language; its val sizes sum to
    data/code/python/train_00000.bin  --val-tokens, in the same proportions as the mix

Reruns skip shards that already exist, so an interrupted download just picks up where it
left off. The token budget is fixed when a source is first written; changing --tokens
later means deleting data/ (or use a different --out).

Needs: pip install pyarrow tiktoken huggingface_hub numpy zstandard
"""

import argparse
import gzip
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC, VERSION, HEADER = 20240520, 1, 256  # llm.c / modded-nanogpt shard format


# ---------------------------------------------------------------- tokenizer

class GPT2:
    name, vocab_size, eot = "gpt2", 50257, 50256

    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")

    def encode_batch(self, texts, threads):
        return self.enc.encode_ordinary_batch(texts, num_threads=threads)


def get_encoder(name):
    # the project's own BPE goes here: anything with name, vocab_size (< 65536), eot, encode_batch
    if name == "gpt2":
        return GPT2()
    raise ValueError(f"unknown tokenizer {name!r}")

# ---------------------------------------------------------------- cleaning

LICENSE = re.compile(r"licen[cs]e|copyright|\(c\)|spdx|all rights reserved|warrant|redistribut", re.I)
# leading comment block, per comment syntax. Hash languages keep their shebang.
HEADER_C = re.compile(r"\A(?:\s*(?:/\*.*?\*/|//[^\n]*))+\s*", re.S)  # c, cpp, js, ts, rust
HEADER_HASH = re.compile(r"\A(#![^\n]*\n)?((?:\s*(?:#[^\n]*|\"\"\".*?\"\"\"|'''.*?'''))+\s*)", re.S)


def clean_c(doc):
    m = HEADER_C.match(doc)
    return doc[m.end():] if m and LICENSE.search(m.group(0)) else doc


def clean_hash(doc):
    m = HEADER_HASH.match(doc)
    return (m.group(1) or "") + doc[m.end():] if m and LICENSE.search(m.group(2)) else doc


# ---------------------------------------------------------------- sources

@dataclass
class Source:
    repo: str  # HF dataset repo, or "cc:<path>" for a dataset hosted on data.commoncrawl.org
    files: list  # glob patterns inside the repo; sorted (or shuffled with a fixed seed), so shard order is stable
    frac: float  # share of the token budget (before normalizing)
    column: str = "text"
    keep: object = None  # doc dict -> bool, for json.gz sources
    clean: object = None  # text -> text
    shuffle: bool = False  # take files in a fixed random order instead of sorted


# common-pile/stackv2_edu_filtered: 95 json.gz files, each one language, in this order (checked
# by reading the first records of every file). Docs are also filtered by language, so a file
# that straddles two languages only contributes its own.
STACK = "common-pile/stackv2_edu_filtered"
STACK_FILES = {"Python": range(73, 85), "JavaScript": range(50, 65), "TypeScript": range(91, 95),
               "C": range(18, 20), "C++": range(20, 28), "Rust": range(86, 87), "Shell": range(87, 89)}
CODE = {"python": ("Python", 0.30, clean_hash), "javascript": ("JavaScript", 0.15, clean_c),
        "typescript": ("TypeScript", 0.10, clean_c), "c": ("C", 0.15, clean_c), "cpp": ("C++", 0.15, clean_c),
        "rust": ("Rust", 0.10, clean_c), "shell": ("Shell", 0.05, clean_hash)}


def stack_edu(lang):
    return lambda d: d["int_score"] >= 3 and d["metadata"]["language"] == lang


SOURCES = {
    "web": Source("cc:contrib/Nemotron/Nemotron-CC", ["data-jsonl/quality=high/kind=actual/kind2=actual/*.jsonl.zstd"],
                  0.60, shuffle=True),
    "pdfs": Source("HuggingFaceFW/finepdfs", ["data/eng_Latn/train/*.parquet"], 0.15),
    **{f"code/{name}": Source(STACK, [f"stack-edu-{i:04d}.json.gz" for i in STACK_FILES[lang]], 0.10 * w,
                              keep=stack_edu(lang), clean=fn) for name, (lang, w, fn) in CODE.items()},
    "math": Source("nvidia/Nemotron-CC-Math-v1", ["4plus/*.parquet"], 0.05),
    "wiki": Source("HuggingFaceFW/finewiki", ["data/enwiki/*.parquet"], 0.05),
}
TOTAL_FRAC = sum(src.frac for src in SOURCES.values())
GROUPS = sorted({k.split("/")[0] for k in SOURCES})


def group_frac(name):
    g = name.split("/")[0]
    return sum(v.frac for k, v in SOURCES.items() if k.split("/")[0] == g)


# ---------------------------------------------------------------- shard io

def write_shard(path: Path, chunks):
    """Write an iterable of uint16 arrays as one shard, atomically (tmp + rename)."""
    tmp = path.with_suffix(".tmp")
    n = 0
    with open(tmp, "wb") as f:
        f.write(np.zeros(HEADER, np.int32).tobytes())
        for c in chunks:
            f.write(c.astype(np.uint16, copy=False).tobytes())
            n += len(c)
        f.seek(0)
        f.write(np.array([MAGIC, VERSION, n] + [0] * (HEADER - 3), np.int32).tobytes())
    tmp.rename(path)
    return n


def read_shard(path) -> np.memmap:
    header = np.fromfile(path, np.int32, HEADER)
    assert header[0] == MAGIC and header[1] == VERSION, f"bad shard header in {path}"
    return np.memmap(path, np.uint16, "r", offset=HEADER * 4, shape=(int(header[2]),))


def shard_len(path) -> int:
    return int(np.fromfile(path, np.int32, 3)[2])


# ---------------------------------------------------------------- download + tokenize

CC = "https://data.commoncrawl.org/"


def list_files(fs, src: Source):
    if src.repo.startswith("cc:"):
        import fnmatch
        import requests

        root = src.repo[3:]
        r = requests.get(f"{CC}{root}/data-jsonl.paths.gz", timeout=60)
        r.raise_for_status()
        paths = gzip.decompress(r.content).decode().split()
        files = sorted(CC + p for p in paths if any(fnmatch.fnmatch(p, f"{root}/{pat}") for pat in src.files))
    else:
        files = sorted(f for pat in src.files for f in fs.glob(f"datasets/{src.repo}/{pat}"))
    assert files, f"no files matching {src.files} in {src.repo}"
    return [files[i] for i in np.random.default_rng(0).permutation(len(files))] if src.shuffle else files


def open_file(fs, path):
    if path.startswith(CC):
        import requests

        r = requests.get(path, stream=True, timeout=60)
        r.raise_for_status()
        return r.raw
    return fs.open(path, block_size=16 << 20)


def iter_docs(fs, path, src: Source, batch_rows=1024):
    """Yield lists of document texts from one parquet, json.gz or jsonl.zstd file."""
    with open_file(fs, path) as fh:
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(fh).iter_batches(batch_size=batch_rows, columns=[src.column]):
                yield batch.column(0).to_pylist()
            return
        if path.endswith(".zstd"):
            import io
            import zstandard

            lines = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        else:
            lines = gzip.open(fh, "rt")
        texts = []
        for line in lines:
            d = json.loads(line)
            if src.keep is None or src.keep(d):
                texts.append(d[src.column])
            if len(texts) == batch_rows:
                yield texts
                texts = []
        if texts:
            yield texts


def tokenize_file(fs, enc, path, src: Source, threads):
    """Yield uint16 token arrays (EOT + doc, EOT + doc, ...) for one input file."""
    for texts in iter_docs(fs, path, src):
        if src.clean:
            texts = [src.clean(t) for t in texts if t]
        toks = enc.encode_batch([t for t in texts if t and not t.isspace()], threads)
        flat = []
        for t in toks:
            flat.append(enc.eot)
            flat.extend(t)
        yield np.array(flat, np.uint16)


def take(chunks, budget, eot):
    """Pass chunks through until `budget` tokens, then stop at the next document start."""
    n = 0
    for c in chunks:
        if n + len(c) >= budget:
            starts = np.flatnonzero(c == eot)
            starts = starts[starts >= budget - n]
            if len(starts):
                yield c[: starts[0]]
                return
        n += len(c)
        yield c


def split_val(chunks, val_path: Path, val_budget: int, eot):
    """Pass-through generator that diverts the first `val_budget` tokens (whole docs) to val_path."""
    val, n = [], 0
    for c in chunks:
        if val is not None:
            if n + len(c) < val_budget:
                val.append(c)
                n += len(c)
                continue
            # first document start at/after the budget: val gets whole docs, train the rest
            starts = np.flatnonzero(c == eot)
            starts = starts[starts >= val_budget - n]
            cut = int(starts[0]) if len(starts) else len(c)
            val.append(c[:cut])
            write_shard(val_path, val)
            val, c = None, c[cut:]
        if len(c):
            yield c
    if val is not None:
        write_shard(val_path, val)


def prepare_source(name, enc, out: Path, budget: int, val_budget: int, threads: int):
    from huggingface_hub import HfFileSystem
    from huggingface_hub.errors import GatedRepoError

    src = SOURCES[name]
    fs = HfFileSystem()
    files = list_files(fs, src)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)

    have = sum(shard_len(p) for p in d.glob("train_*.bin"))
    print(f"[{name}] {src.repo}: {len(files)} files, target {budget/1e6:,.1f}M train tokens"
          + (f", have {have/1e6:,.1f}M" if have else ""))

    for i, path in enumerate(files):
        if have >= budget:
            break
        shard = d / f"train_{i:05d}.bin"
        if shard.exists():
            continue
        t0 = time.time()
        for attempt in range(3):
            try:
                chunks = tokenize_file(fs, enc, path, src, threads)
                if i == 0:  # carve the held-out val split off the front of the first file
                    chunks = split_val(chunks, d / "val.bin", val_budget, enc.eot)
                n = write_shard(shard, take(chunks, budget - have, enc.eot))
                break
            except GatedRepoError:
                raise
            except (OSError, ConnectionError, TimeoutError) as e:
                if attempt == 2:
                    raise
                print(f"[{name}]   {e!r}, retrying {path}")
        have += n
        dt = time.time() - t0
        print(f"[{name}]   {shard.name}: {n/1e6:,.1f}M tok in {dt:.0f}s ({n/dt/1e6:.1f}M tok/s), "
              f"total {have/1e6:,.1f}M / {budget/1e6:,.1f}M")
    else:
        if have < budget:
            print(f"[{name}] warning: ran out of files at {have/1e6:,.0f}M tokens")
    return have


# ---------------------------------------------------------------- loader

class TokenLoader:
    """One pass over every train shard under `root`, in (B, T) batches of non-overlapping
    T+1-token windows, globally shuffled across sources. Since sources were written in
    proportion to the mix, every batch is mixed in that proportion on average.

        loader = TokenLoader("data", B=8, T=4096, device="mps")
        for x, y in loader: ...           # len(loader) = steps per epoch
        for x, y in loader.val("code", n_batches=20): ...   # or "web", "pdfs", "code/rust"

    For DDP pass rank/world; each rank gets a disjoint slice of the same permutation.
    """

    def __init__(self, root="data", B=8, T=4096, device="cpu", seed=0, rank=0, world=1):
        import torch

        self.torch, self.B, self.T, self.device, self.root = torch, B, T, device, Path(root)
        self.shards = [read_shard(p) for p in sorted(self.root.rglob("train_*.bin"))]
        assert self.shards, f"no train shards under {root}/, run data.py first"
        counts = np.array([(len(s) - 1) // T for s in self.shards])
        self.offsets = np.concatenate([[0], np.cumsum(counts)])
        order = np.random.default_rng(seed).permutation(int(self.offsets[-1]))
        self.order = order[rank::world]
        self.tokens = int(self.offsets[-1]) * T

    def __len__(self):
        return len(self.order) // self.B

    def _window(self, g):
        s = int(np.searchsorted(self.offsets, g, side="right")) - 1
        i = (g - int(self.offsets[s])) * self.T
        return self.shards[s][i : i + self.T + 1]

    def _batch(self, windows):
        buf = self.torch.from_numpy(np.stack(windows).astype(np.int64))
        if str(self.device).startswith("cuda"):
            buf = buf.pin_memory()
        buf = buf.to(self.device, non_blocking=True)
        return buf[:, :-1], buf[:, 1:]

    def __iter__(self):
        for b in range(len(self)):
            yield self._batch([self._window(g) for g in self.order[b * self.B : (b + 1) * self.B]])

    def val(self, source, n_batches=None):
        """Deterministic val batches for a source ("web", "code", "code/python", ...). With
        n_batches, windows are spread evenly over the whole split (so "code" covers every language)."""
        vals = [read_shard(p) for p in sorted((self.root / source).rglob("val.bin"))]
        assert vals, f"no val.bin under {self.root / source}"
        windows = [(v, i * self.T) for v in vals for i in range((len(v) - 1) // self.T)]
        if n_batches is not None and n_batches * self.B < len(windows):
            windows = [windows[i] for i in np.linspace(0, len(windows) - 1, n_batches * self.B).astype(int)]
        for b in range(len(windows) // self.B):
            yield self._batch([v[i : i + self.T + 1] for v, i in windows[b * self.B : (b + 1) * self.B]])

    @property
    def sources(self):
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and any(p.rglob("val.bin")))


# ---------------------------------------------------------------- cli

def main():
    from huggingface_hub.errors import GatedRepoError

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", action="store_true", help="small ~50M-token version for local testing")
    ap.add_argument("--tokens", type=float, help="total train tokens (default 20e9, or 50e6 with --subset)")
    ap.add_argument("--val-tokens", type=float, help="held-out tokens per source (default 5e6, or 1e6 with --subset)")
    ap.add_argument("--sources", nargs="+", choices=GROUPS, default=GROUPS)
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--out", type=Path, help="output dir (default data/ or data-subset/ next to this file)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()

    total = args.tokens or (50e6 if args.subset else 20e9)
    val = args.val_tokens or (1e6 if args.subset else 5e6)
    out = args.out or Path(__file__).parent / ("data-subset" if args.subset else "data")
    out.mkdir(parents=True, exist_ok=True)
    enc = get_encoder(args.tokenizer)
    names = [k for k in SOURCES if k.split("/")[0] in args.sources]

    # never mix shards from different tokenizers or source definitions in one directory
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta and meta.get("tokenizer") != enc.name:
        raise SystemExit(f"{out}/ was written with tokenizer {meta.get('tokenizer')!r}; delete it or use --out")
    spec = {k: [SOURCES[k].repo, SOURCES[k].files] for k in names}
    stale = [k for k, v in meta.get("sources", {}).items() if k in spec and v != spec[k]]
    stale += [k for k in meta.get("sources", {}) if k not in SOURCES]
    if stale:
        raise SystemExit(f"{out}/ has shards from an older source definition ({', '.join(stale)}); delete it or use --out")

    t0 = time.time()
    counts, skipped = dict(meta.get("train_tokens", {})), []
    for name in names:
        frac = SOURCES[name].frac
        try:
            counts[name] = prepare_source(name, enc, out, int(total * frac / TOTAL_FRAC),
                                          int(val * frac / group_frac(name)), args.threads)
        except GatedRepoError:
            print(f"[{name}] skipped: no access to gated repo {SOURCES[name].repo} "
                  f"(request it at huggingface.co/datasets/{SOURCES[name].repo})")
            skipped.append(name)

    sources = {**meta.get("sources", {}), **{k: spec[k] for k in counts if k in spec}}
    meta = {"tokenizer": enc.name, "vocab_size": enc.vocab_size, "eot": enc.eot, "train_tokens": counts,
            "val_tokens_per_source": int(val), "sources": sources}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"done in {(time.time()-t0)/60:.1f} min: "
          + ", ".join(f"{k} {v/1e6:,.1f}M" for k, v in counts.items())
          + f" -> {out}/" + (f"; skipped {', '.join(skipped)}" if skipped else ""))


if __name__ == "__main__":
    main()
