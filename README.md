# ml-experiments

Small ML experiments, mostly written against [tinygrad](https://github.com/tinygrad/tinygrad),
with PyTorch ports where training speed on NVIDIA mattered.

## Projects

| Project | What it is | Author |
|---|---|---|
| [add-transformer](add-transformer/) | Tiny decoder-only transformer that does 4-digit addition with a CoT scratchpad. tinygrad + pytorch versions. | human |
| [sort-transformer](sort-transformer/) | Llama3-style decoder-only transformer that sorts variable-length integer lists. tinygrad version. | human |
| [rectified-flow](rectified-flow/) | Rectified flow image generation on CIFAR-10 (UNet + FiLM, CFG in the class-conditional version). tinygrad + pytorch versions. | AI |
| [llama3-tinygrad](llama3-tinygrad/) | Llama 3.2 1B Instruct inference in tinygrad, loading HF safetensors directly. | human |
| [hip-llama3](hip-llama3/) | Readable HIP/C++ inference runner for Llama 3.2 1B (bf16 weights, hand-written kernels). | AI |

## Next

Roughly in order, each building on the last:

- [ ] KV cache + attention optimizations (paged/flash attention, GQA tricks)
- [ ] Inference optimizations (quantization, speculative decoding, batching)
- [ ] Training optimizations (mixed precision, fused kernels, distributed data/tensor parallel)
- [ ] Mixture-of-experts inference (routing, expert parallelism)
- [ ] RL fine-tuning (PPO, GRPO, and friends)
- [ ] End goal: train a "frontier" LLM from start to finish as a series of experiments —
      data pipeline, pretraining, SFT, RL, eval

## Setup

```sh
git clone git@github.com:boopdotpng/ml-experiments.git
cd ml-experiments
git clone https://github.com/tinygrad/tinygrad.git   # local clone, gitignored
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e ./tinygrad
./download.sh   # fetches Llama 3.2 1B weights + tokenizer (~2.4 GB)
```

`download.sh` only fetches the Llama checkpoint (shared between `llama3-tinygrad`
and `hip-llama3`). The toy transformers train from scratch in seconds/minutes —
just run their `main.py`.
