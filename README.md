# ml-experiments

Small ML experiments, mostly written against [tinygrad](https://github.com/tinygrad/tinygrad)
(pinned as a submodule), with PyTorch ports where training speed on NVIDIA mattered.

## Projects

| Project | What it is |
|---|---|
| [add-transformer](add-transformer/) | Tiny decoder-only transformer that does 4-digit addition with a CoT scratchpad. tinygrad + pytorch versions. |
| [sort-transformer](sort-transformer/) | Llama3-style decoder-only transformer that sorts variable-length integer lists. tinygrad version. |
| [rectified-flow](rectified-flow/) | Rectified flow image generation on CIFAR-10 (UNet + FiLM, CFG in the class-conditional version). tinygrad + pytorch versions. |
| [llama3-tinygrad](llama3-tinygrad/) | Llama 3.2 1B Instruct inference in tinygrad, loading HF safetensors directly. |
| [hip-llama3](hip-llama3/) | Readable HIP/C++ inference runner for Llama 3.2 1B (bf16 weights, hand-written kernels). |

Authorship: `add-transformer`, `sort-transformer`, and `llama3-tinygrad` are
written by hand; `rectified-flow` and `hip-llama3` are AI-written.

## Setup

```sh
git clone --recursive git@github.com:boopdotpng/ml-experiments.git
cd ml-experiments
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt   # installs the tinygrad submodule editable
./download.sh                     # fetches Llama 3.2 1B weights + tokenizer (~2.4 GB)
```

`download.sh` only fetches the Llama checkpoint (shared between `llama3-tinygrad`
and `hip-llama3`). The toy transformers train from scratch in seconds/minutes —
just run their `main.py`.
