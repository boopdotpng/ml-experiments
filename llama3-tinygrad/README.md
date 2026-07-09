# llama3-tinygrad

Llama 3.2 1B Instruct inference in tinygrad, loading the HuggingFace
safetensors checkpoint directly (with a small key-remap from the HF naming).

Run `../download.sh` first to fetch `model.safetensors` plus the tokenizer and
config files from `unsloth/Llama-3.2-1B-Instruct`, then:

```sh
python main.py
```

## Pre-optimization UOp corpus

[`corpus/decode/raw_graph.txt`](corpus/decode/raw_graph.txt) is a hardware-free
capture of the complete steady-state decode Tensor UOp DAG. It is taken before
`transform_to_call`, scheduling, rangeification, kernel optimization, rendering,
or compilation, and is the source-of-truth graph for a TTIR lowerer.

[`corpus/decode/manifest.txt`](corpus/decode/manifest.txt) and the files under
`corpus/decode/kernels/` annotate the scheduler's 231 current compute-launch
boundaries without treating the rangeified kernel ASTs as lowering input. This
distinction matters: tinygrad does not create launch boundaries until
`run_rangeify`, after the pristine Tensor graph has ceased to exist as separate
kernels.

Regenerate without opening or executing on any hardware device:

```sh
$(realpath ../.venv)/bin/python tools/generate_uop_corpus.py
```
