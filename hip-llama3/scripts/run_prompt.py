#!/usr/bin/env python3
"""Tokenize a prompt, run the HIP binary, and decode generated token ids."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/unsloth-llama-3.2-1b-instruct")
    parser.add_argument("--model", default="models/llama3.2-1b-instruct.hllm")
    parser.add_argument("--binary", default="build/hip_llama")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mode", choices=["naive", "kv", "flash"], default="kv")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--max-seq", type=int, default=512)
    parser.add_argument("--raw", action="store_true", help="Do not apply the instruct chat template")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    if args.raw:
        ids = tok.encode(args.prompt, add_special_tokens=True)
    else:
        messages = [{"role": "user", "content": args.prompt}]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True)

    cmd = [
        str(Path(args.binary)),
        "--model",
        args.model,
        "--tokens",
        ",".join(map(str, ids)),
        "--steps",
        str(args.steps),
        "--mode",
        args.mode,
        "--max-seq",
        str(args.max_seq),
    ]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    print(proc.stderr, end="")
    print(proc.stdout, end="")

    match = re.search(r"ALL_TOKENS:\s*([0-9, -]+)", proc.stdout)
    if not match:
        raise SystemExit("Runner did not print ALL_TOKENS")
    out_ids = [int(x) for x in match.group(1).replace(",", " ").split()]
    print("\nDECODED:")
    print(tok.decode(out_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
