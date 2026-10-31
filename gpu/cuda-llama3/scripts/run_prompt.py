#!/usr/bin/env python3
"""Tokenize a chat prompt, run the CUDA binary, and decode its generated ids."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--checkpoint", type=Path, default=ROOT.parent / "hip-llama3/checkpoints/unsloth-llama-3.2-1b-instruct")
    parser.add_argument("--model", type=Path, default=ROOT.parent / "hip-llama3/models/llama3.2-1b-instruct.hllm")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/cuda_llama")
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    if args.raw:
        ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    else:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
        )
        # Transformers 4 returned a list here; Transformers 5 returns a
        # BatchEncoding by default. Keep the helper compatible with both.
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded

    command = [
        str(args.binary), "--model", str(args.model),
        "--tokens", ",".join(map(str, ids)),
        "--steps", str(args.steps), "--max-seq", str(len(ids) + args.steps),
        "--warmup", "0", "--runs", "1",
    ]
    process = subprocess.run(command, check=True, text=True, capture_output=True)
    print(process.stderr, end="")
    match = re.search(r"^GENERATED_TOKENS:\s*(.*)$", process.stdout, re.MULTILINE)
    if not match:
        raise SystemExit("runner did not return generated tokens")
    generated = [int(value) for value in match.group(1).split(",") if value]
    if tokenizer.eos_token_id in generated:
        generated = generated[: generated.index(tokenizer.eos_token_id) + 1]
    print(tokenizer.decode(generated, skip_special_tokens=False))


if __name__ == "__main__":
    main()
