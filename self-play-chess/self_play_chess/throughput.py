"""Disposable end-to-end rollout and optimizer benchmarks, never training run weights."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from tinygrad import Device
from tinygrad.helpers import BEAM
from .encoding import Observation, encode
from .environment import ChessEnv
from .training import LiveState, Trainer, TrainingConfig
from .model import ModelConfig
from dataclasses import asdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel-games", type=int, default=8)
    parser.add_argument("--plies", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--policy-width", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        trainer = Trainer(LiveState(), Path(directory), TrainingConfig(parallel_games=args.parallel_games,
                          batch_size=args.batch_size, max_plies=args.plies, move_delay=0),
                          ModelConfig(width=args.width, layers=args.layers, heads=args.heads, policy_width=args.policy_width))
        print("Warming rollout graphs…", flush=True)
        trainer.play_parallel(args.parallel_games, learn=False)
        before = time.perf_counter()
        trainer.play_parallel(args.parallel_games, learn=False)
        elapsed = time.perf_counter()-before
        rows = [json.loads(line) for line in (Path(directory)/"metrics.jsonl").read_text().splitlines()][-args.parallel_games:]
        plies = sum(row["plies"] for row in rows)
        # Synthetic outcome labels ONLY benchmark optimizer throughput on disposable weights.
        observation = encode([ChessEnv()])
        arrays = {name: getattr(observation, name).numpy() for name in Observation.__dataclass_fields__}
        actions = np.flatnonzero(arrays["legal"][0])
        samples = [(arrays, int(actions[i % len(actions)]), float(i % 3 - 1), i % 3) for i in range(args.batch_size)]
        print(f"Warm rollout: {plies/elapsed:.1f} moves/sec. Testing backward and Adam…", flush=True)
        trainer.update_samples("A", samples)
        times = []
        for _ in range(3):
            before = time.perf_counter()
            trainer.update_samples("A", samples)
            times.append(time.perf_counter()-before)
        result = dict(device=Device.DEFAULT, arch=getattr(Device.default, "arch", "apple-metal"), beam=BEAM.value,
                      parallel_games=args.parallel_games, rollout_seconds=elapsed, legal_moves=plies,
                      legal_moves_per_second=plies/elapsed, training_batch=args.batch_size,
                      model_config=asdict(trainer.model_config), parameters=trainer.live.snapshot()["parameters"],
                      optimizer_step_ms=float(np.median(times)*1000), training_samples_per_second=float(args.batch_size/np.median(times)),
                      note="Warm rollouts include encoding, legal rules, sampling, SVG/JSON/checkpoint work. Updates measured separately on disposable synthetic targets; not claimed as sustained learning throughput.")
        print(json.dumps(result), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("a") as file:
                file.write(json.dumps(result)+"\n")


if __name__ == "__main__":
    main()
