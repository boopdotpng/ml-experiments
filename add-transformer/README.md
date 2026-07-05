# add-transformer

Tiny decoder-only transformer trained to do 4-digit addition as character-level
next-token prediction, using a chain-of-thought scratchpad (per-digit
`write X carry Y` steps before the final answer). See [task.md](tinygrad/task.md)
for the full problem framing.

- `tinygrad/` — original tinygrad implementation. `python main.py` trains and
  saves `model.safetensors`.
- `pytorch/` — PyTorch port (faster to train on NVIDIA), plus a pile of
  interpretability scripts: `inspect_attn.py`, `inspect_pos.py`, `inspect_tok.py`,
  `inspect_value.py`, `inspect_roles.py`, and timelapse renderers
  (`train_attn_timelapse.py`, `train_timelapse.py`). The PNGs/GIFs checked in
  here are their output.

No downloads needed — the model trains from scratch in well under a minute.
