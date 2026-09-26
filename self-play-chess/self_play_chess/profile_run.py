"""Inference throughput/profile: DEV=METAL BEAM=2 PROFILE=1 python -m self_play_chess.profile_run"""
import argparse
import json
import time
from pathlib import Path
from importlib.metadata import version
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.helpers import profile_marker
from tinygrad.helpers import BEAM
from .encoding import encode
from .environment import ChessEnv
from .inference import Inference
from .model import ChessTransformer, ModelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    Tensor.manual_seed(42)
    model = ChessTransformer(ModelConfig(width=32, layers=1, policy_width=8))
    inference = Inference(model, jit=not args.eager)
    envs = [ChessEnv() for _ in range(args.batch)]
    print(f"Warming {Device.DEFAULT}: batch={args.batch}, jit={not args.eager}", flush=True)
    start = time.perf_counter()
    for _ in range(3):
        inference(encode(envs))
    warmup = time.perf_counter()-start
    times = []
    profile_marker("measure-start")
    for _ in range(args.repeats):
        observation = encode(envs)
        begin = time.perf_counter()
        inference(observation)
        Device.default.synchronize()
        times.append(time.perf_counter()-begin)
    profile_marker("measure-end")
    result = dict(device=Device.DEFAULT, batch=args.batch, jit=not args.eager, beam=BEAM.value, tinygrad=version("tinygrad"),
                          warmup_seconds=warmup, median_ms=float(np.median(times)*1000),
                          positions_per_second=float(args.batch/np.median(times)))
    print(json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as file:
            file.write(json.dumps(result)+"\n")


if __name__ == "__main__":
    main()
