#!/usr/bin/env python3
"""Reproduce the CUDA-event throughput table printed in the README."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def metric(name: str, text: str) -> float:
    match = re.search(rf"^{name}:\s*([0-9.]+)$", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing {name}")
    return float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="128,512,1024")
    parser.add_argument("--decode", type=int, default=128)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/cuda_llama")
    parser.add_argument("--model", type=Path, default=ROOT.parent / "hip-llama3/models/llama3.2-1b-instruct.hllm")
    args = parser.parse_args()

    print("| prompt | prefill tok/s | decode tok/s |")
    print("|------:|--------------:|-------------:|")
    for length in map(int, args.lengths.split(",")):
        command = [
            str(args.binary), "--model", str(args.model),
            "--prompt-length", str(length), "--steps", str(args.decode + 1),
            "--max-seq", str(length + args.decode + 1),
            "--warmup", "1", "--runs", str(args.runs),
        ]
        process = subprocess.run(command, check=True, text=True, capture_output=True)
        prefill = metric("PREFILL_TOKENS_PER_SECOND", process.stdout)
        decode = metric("DECODE_TOKENS_PER_SECOND", process.stdout)
        print(f"| {length} | {prefill:,.0f} | {decode:,.0f} |")


if __name__ == "__main__":
    main()

