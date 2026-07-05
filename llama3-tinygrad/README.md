# llama3-tinygrad

Llama 3.2 1B Instruct inference in tinygrad, loading the HuggingFace
safetensors checkpoint directly (with a small key-remap from the HF naming).

Run `../download.sh` first to fetch `model.safetensors` plus the tokenizer and
config files from `unsloth/Llama-3.2-1B-Instruct`, then:

```sh
python main.py
```
