#!/usr/bin/env python3
"""Generate, inspect, or train a Llama-3.2-shaped model in HIP."""
import argparse
from pathlib import Path
import time

import numpy as np

from model import Config, Graph
from codegen import ROOT, generate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['emit', 'train'])
    p.add_argument('--preset', choices=['tiny', '1b'], default='tiny')
    p.add_argument('--batch', type=int)
    p.add_argument('--seq', type=int)
    p.add_argument('--position-offset', type=int, default=0)
    p.add_argument('--out', type=Path)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--arch', help='HIP GPU target, e.g. gfx1201')
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--lr', type=float, default=.05)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--tokens', type=Path, help='.npy integer array [batch, seq+1]; labels are shifted automatically')
    args = p.parse_args()
    overrides = {k: getattr(args, k) for k in ('batch', 'seq', 'position_offset') if getattr(args, k) is not None}
    c = Config.preset(args.preset, **overrides)
    graph = Graph(c)
    out = args.out or ROOT / 'generated' / args.preset
    manifest = generate(graph, out)
    print(f'{c.name}: {manifest["parameter_count"]:,} parameters; {graph.memory_bytes/2**30:.3f} GiB FP32 tensors')
    print(f'{manifest["forward_launches"]} forward / {manifest["backward_launches"]} backward kernel launches')
    print(f'Viewer: {(out / "index.html").resolve()}')
    if args.action == 'emit':
        return
    if args.steps < 1 or args.lr <= 0 or not np.isfinite(args.lr):
        p.error('--steps and --lr must be positive and finite')
    sequence = (np.load(args.tokens, allow_pickle=False) if args.tokens else
                np.tile(np.arange(c.seq+1) % min(7, c.vocab), (c.batch, 1)).astype(np.int32))
    from runtime import Runner
    with Runner(graph, out, device=args.device, arch=args.arch, seed=args.seed) as runner:
        print('GPU:', runner.device_name, flush=True)
        runner.set_batch(sequence)
        losses = []
        for step in range(args.steps):
            start = time.perf_counter()
            loss = runner.forward()
            if not np.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            runner.backward()
            runner.step(args.lr)
            losses.append(loss)
            print(f'step {step:3d}  loss {loss:.7f}  elapsed {time.perf_counter()-start:.3f}s', flush=True)
        final = runner.forward()
        print(f'After {args.steps} updates: {losses[0]:.7f} -> {final:.7f}', flush=True)


if __name__ == '__main__':
    main()
