# Task: Variable-Length List Sorting with a Llama3-Style Decoder-Only Transformer

## Goal

Train a small decoder-only transformer to sort a variable-length list of integers
as token-level next-token prediction. Unlike the fixed-width addition task, lists
have **different lengths per example**, which forces real `<pad>` tokens and
pad-aware masking — the whole point of this exercise.

The secondary goal is architectural: rebuild the model with the Llama3 ingredients
(RoPE, SwiGLU, GQA, RMSNorm, KV-cache inference) instead of the learned-positional /
ReLU-MLP / vanilla-MHA stack from the addition task. The sorting task is the
regression harness: after each architectural swap, the model should still train to
~100% exact-match.

## Problem Framing

Each example is a variable-length token sequence:

```text
<bos> 5 _ 2 _ 8 _ 2 _ 1 <sep> 1 _ 2 _ 2 _ 5 _ 8 <eos> <pad>*
      \------ input list -----/      \---- sorted output ----/
```

(`_` is the space token.)

- **Input:** `<bos>` then the space-separated input list.
- **Boundary:** a single `<sep>` token marks "input done, sorted output begins".
  This replaces a spelled-out `sort:`/`=` prompt — variable-length lists only need
  a boundary marker, not literal prompt text.
- **Target:** the sorted list, ascending, space-separated, terminated by `<eos>`.

List length is sampled per example (`L ~ uniform(min_len=2, max_len=12)`, values
`0..max_val=9`), so sequences are ragged and are padded to `seq_len = 4*max_len + 1`.

## Why this task

Every Llama3 component earns its place here:

- **Padding + pad-aware mask** — ragged list lengths make batching impossible
  without `<pad>`. Cannot be skipped.
- **RoPE** — position-sensitive task with a clean generalization story: train on
  lists `L <= 12`, evaluate on `L = 20`, and measure length extrapolation.
- **GQA** — drop in `n_kv_heads < n_heads` once attention works; confirm accuracy
  holds.
- **KV cache** — autoregressive generation of the sorted output at inference;
  left-pad the batch so every row decodes in lockstep.
- **SwiGLU + RMSNorm** — straight architectural swaps validated by "does the toy
  still hit ~100%."

## Tokenization

Token-level vocabulary, **15 tokens** (keep numbers single-digit at first so the
vocab is tiny; generalize to multi-digit later):

```text
ids 0..9  : '0' '1' '2' '3' '4' '5' '6' '7' '8' '9'
id  10    : ' '  (space)
id  11    : <pad>
id  12    : <bos>
id  13    : <eos>
id  14    : <sep>
```

Notes:
- `<pad>` is a real token id; it must never receive loss and must never be
  attended to as a key.
- `<sep>` is the single input/output boundary (there is no literal `sort:`/`=`).
- `<eos>` terminates generation; `<bos>` gives a clean position 0.

## Padding & Masking (the core of this task)

`<pad>` shows up in three independent places — get all three right:

1. **Attention mask** — a query must not attend to `<pad>` keys, or softmax mixes
   garbage into context. AND the pad-key mask together with the causal mask.
2. **Loss mask** — do not supervise `<pad>` target positions, and (as before) only
   supervise tokens after the `<sep>` boundary. In the code this lives directly in
   `y`: positions to skip are set to `-1` (matching
   `sparse_categorical_crossentropy`'s default `ignore_index`), so no separate loss
   mask tensor is needed.
3. **Position handling under RoPE** — positions come from token index:
   - **Right-pad** for training: real tokens get positions `0..L-1`, pads sit
     after — simplest.
   - **Left-pad** for batched generation: so every row's "next token" lands in the
     same column. Pads then occupy positions `0..k-1` and would shift real tokens,
     so apply a per-row position offset (positions start at the first real token)
     and mask the pad keys. This interaction between padding, RoPE, and the KV
     cache is the subtle part worth getting right.

Note: training uses **right-padding**, and the supervised boundary is the `<sep>`
token (loss starts at the first sorted digit, i.e. the token after `<sep>`).

## Dataset (target API)

Mirror the addition repo's shape so the training loop ports over with minimal
changes:

`dataset(n_train=20000, n_test=2000, seed=0, min_len=2, max_len=12, max_val=9)`
returns `trainx, trainy, testx, testy` as int tinygrad tensors, padded to
`seq_len = 4*max_len + 1`. Train/test lists are disjoint (dedup via a `seen` set).

- `x = seq[:-1]`, `y = seq[1:]`.
- Loss masking lives **in `y`**: `y[j] = -1` wherever we must not supervise (the
  input region and every `<pad>` target); supervised on the sorted digits + `<eos>`.
- No mask tensor is returned. The **attention pad mask is derived** from
  `(x == pad_id)`, so the model only needs `pad_id` from the config.

Helpers (implemented):

- `format_example(lst)` -> token-id list `<bos> src <sep> dst <eos>`
- `encode(s)` / `decode(ids)` (decode skips `<pad>`/`<bos>`/`<eos>`)
- `make_xy(lst)` -> padded `(x, y)` with `y` already loss-masked to `-1`
- `get_dataset_config()` -> `DatasetConfig`: `vocab_size`, `pad_id`, `bos_id`,
  `eos_id`, `sep_id`, `ignore_id`, `min_len`/`max_len`/`max_val`, and derived
  `seq_len` / `ctx_len` properties

## Model configuration

Starting hyperparameters (deliberately over-capacity for the toy so bugs, not
capacity, explain any failure):

```text
emb_dim (d_model)  = 128
n_layers           = 4
n_heads            = 4
head_dim           = 32     # emb_dim // n_heads
n_kv_heads         = 4      # GQA step drops this to 2
mlp_hidden         = 256    # SwiGLU: ~(2/3)*4*d_model
vocab              = 15     # from dataset config, not hardcoded elsewhere
max_seq_len        = 128    # must cover the L=20 extrapolation eval (ctx 80),
                            #   NOT just the train ctx_len of 48
```

Note the two distinct length numbers: the **dataset** pads to `ctx_len = 48`
(L<=12), while the **model** sizes RoPE/mask/KV-cache to `max_seq_len = 128` so the
out-of-distribution `L = 20` eval (ctx 80) fits.

## Suggested build order

Keep the sorting task as the regression test after every step:

1. **Baseline port** — reproduce the addition model on sorting *with right-padding +
   pad-aware attention/loss masks*. Confirm ~100% exact-match. This isolates the
   padding work from the architecture work.
2. **RoPE** — delete learned `pos_emb`, apply rotary embeddings to q/k. Re-confirm.
3. **SwiGLU MLP** — swap the ReLU 2-matrix MLP for `down(silu(gate(x)) * up(x))`.
4. **GQA** — generalize attention to `n_kv_heads < n_heads` with `repeat_kv`.
5. **KV cache + left-pad generation** — incremental decode with a `start_pos` /
   position offset; this is where padding, RoPE, and the cache all interact.
6. **Length extrapolation eval** — train on `max_len=12`, evaluate at `L=20`, report
   the accuracy gap as the RoPE payoff.

## Success criteria

- Exact-match accuracy ~100% on in-distribution test lists (`L <= 12`).
- Generation works batched with left-padding and a KV cache.
- A measured (even if imperfect) length-extrapolation number at `L = 20`.
