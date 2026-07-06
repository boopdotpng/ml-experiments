# Task: CoT 4-Digit Addition with a Tiny Decoder-Only Transformer

## Goal

Train a small decoder-only transformer to solve 4-digit addition as character-level
next-token prediction. The only dataset format kept in this repo is the CoT
scratchpad format, so the model generates intermediate carry/write steps before
the final answer.

## Problem Framing

Each example is one fixed-width string:

```text
3412+5879=write 1 carry 1
write 9 carry 0
write 2 carry 1
write 9 carry 0
answer: 09291
```

The prompt is always `AAAA+BBBB=`. The generated target is the scratchpad plus
`answer: ` and the zero-padded forward sum.

## Tokenization

Character-level over digits and the scratchpad characters:

```text
0 1 2 3 4 5 6 7 8 9 + = space : newline w r i t e c a y n s
```

Key constants live in `dataset.py`:

- `N_DIGITS = 4`
- `ANSWER_LEN = 5`
- `PROMPT_LEN = 10`
- `SEQ_LEN = 87`
- `CTX_LEN = 86`

## Dataset

`dataset(n_train=20000, n_test=2000, seed=0, max_operand=9999)` returns
`trainx, trainy, testx, testy` as tinygrad tensors.

`x = seq[:-1]` and `y = seq[1:]`. The loss is masked to
`CONFIG.answer_start`, so training only supervises tokens generated after the
prompt.

Helpers:

- `format_example(a, b)`
- `encode(s)` / `decode(ids)`
- `make_xy(a, b)`
