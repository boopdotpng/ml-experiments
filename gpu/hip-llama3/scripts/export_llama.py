#!/usr/bin/env python3
"""Export a Hugging Face Llama checkpoint to the tiny .hllm format.

The C++ runner intentionally avoids JSON and safetensors dependencies. This
script is the bridge: it reads config.json + model.safetensors, verifies the
tensor names Llama uses, and writes all weights as raw bf16 words.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch
from safetensors.torch import load_file


MAGIC = b"HLLAMA3\0"
VERSION = 1
NAME_BYTES = 96


def bf16_words(tensor: torch.Tensor) -> bytes:
    t = tensor.detach().cpu().contiguous()
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t.view(torch.uint16).numpy().tobytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/unsloth-llama-3.2-1b-instruct")
    parser.add_argument("--out", default="models/llama3.2-1b-instruct.hllm")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with (ckpt / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    tensors = load_file(ckpt / "model.safetensors")
    n_layers = int(cfg["num_hidden_layers"])
    dim = int(cfg["hidden_size"])
    hidden_dim = int(cfg["intermediate_size"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg.get("num_key_value_heads", n_heads))
    head_dim = int(cfg.get("head_dim", dim // n_heads))
    vocab_size = int(cfg["vocab_size"])
    max_seq_len = int(cfg.get("max_position_embeddings", 131072))
    rope_theta = float(cfg.get("rope_theta", 500000.0))
    rms_norm_eps = float(cfg.get("rms_norm_eps", 1e-5))
    bos_id = int(cfg.get("bos_token_id", 128000))
    eos = cfg.get("eos_token_id", 128009)
    eos_id = int(eos[0] if isinstance(eos, list) else eos)
    pad_id = int(cfg.get("pad_token_id", eos_id) or eos_id)

    names: list[str] = ["model.embed_tokens.weight"]
    for layer in range(n_layers):
        p = f"model.layers.{layer}"
        names += [
            f"{p}.input_layernorm.weight",
            f"{p}.self_attn.q_proj.weight",
            f"{p}.self_attn.k_proj.weight",
            f"{p}.self_attn.v_proj.weight",
            f"{p}.self_attn.o_proj.weight",
            f"{p}.post_attention_layernorm.weight",
            f"{p}.mlp.gate_proj.weight",
            f"{p}.mlp.up_proj.weight",
            f"{p}.mlp.down_proj.weight",
        ]
    names.append("model.norm.weight")

    has_lm_head = "lm_head.weight" in tensors
    if has_lm_head:
        names.append("lm_head.weight")

    missing = [name for name in names if name not in tensors]
    if missing:
        raise SystemExit("Missing tensors:\n" + "\n".join(missing))

    header = struct.pack(
        "<8sIIiiiiiiiiffiii",
        MAGIC,
        VERSION,
        len(names),
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        vocab_size,
        max_seq_len,
        rope_theta,
        rms_norm_eps,
        bos_id,
        eos_id,
        pad_id,
    )

    table_bytes = len(names) * (NAME_BYTES + 8 + 8)
    offset = len(header) + table_bytes
    entries: list[tuple[str, int, int]] = []
    payloads: list[bytes] = []

    for name in names:
        data = bf16_words(tensors[name])
        count = tensors[name].numel()
        entries.append((name, offset, count))
        payloads.append(data)
        offset += len(data)
        print(f"{name:56s} {tuple(tensors[name].shape)!s:20s} {tensors[name].dtype}")

    with out.open("wb") as f:
        f.write(header)
        for name, off, count in entries:
            encoded = name.encode("utf-8")
            if len(encoded) >= NAME_BYTES:
                raise SystemExit(f"Tensor name too long: {name}")
            f.write(encoded + b"\0" * (NAME_BYTES - len(encoded)))
            f.write(struct.pack("<QQ", off, count))
        for payload in payloads:
            f.write(payload)

    tied = "yes" if not has_lm_head else "no"
    print(f"\nWrote {out} ({out.stat().st_size / 1024**3:.2f} GiB)")
    print(f"lm_head tied to embeddings: {tied}")


if __name__ == "__main__":
    main()
