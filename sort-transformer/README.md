# sort-transformer

Small Llama3-style decoder-only transformer trained to sort variable-length
lists of integers as a sequence-to-sequence task. See
[task.md](tinygrad/task.md) for the problem framing and tokenization.

- `tinygrad/` — tinygrad implementation. `python main.py` trains and saves
  `model.safetensors`.

No downloads needed — trains from scratch quickly.
