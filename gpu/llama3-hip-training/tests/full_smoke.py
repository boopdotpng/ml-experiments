#!/usr/bin/env python3
"""Exercise the actual 1B dimensions, all parameter gradients, and one update."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import Config, Graph
from runtime import Runner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arch',default=None)
    p.add_argument('--device',type=int,default=0)
    args=p.parse_args()
    graph=Graph(Config.preset('1b'))
    started=time.perf_counter()
    print(f'Full preset: {sum(t.size for t in graph.parameters):,} parameters; {graph.memory_bytes/2**30:.3f} GiB tensors',flush=True)
    with Runner(graph,ROOT/'generated'/'1b',device=args.device,arch=args.arch) as runner:
        runner.set_batch((np.arange(9)%3)[None,:].astype(np.int32))
        before=runner.forward()
        print(f'Forward loss: {before:.7f}',flush=True)
        runner.backward()
        print('Backward completed. Checking every parameter gradient for finite values.',flush=True)
        gradients=[]
        for t in graph.parameters:
            grad=runner.read(t,True)
            if not np.isfinite(grad).all(): raise AssertionError(f'Non-finite gradient: {t.name}')
            maximum=float(np.abs(grad).max())
            if maximum==0: raise AssertionError(f'Entire parameter gradient is zero: {t.name}')
            gradients.append(dict(name=t.name,elements=t.size,max_abs=maximum))
            del grad
        runner.step(.001)
        after=runner.forward()
        if not np.isfinite(before) or not np.isfinite(after) or not after<before:
            raise AssertionError(f'Full-model step did not lower loss: {before} -> {after}')
        report=dict(status='passed',timestamp=datetime.now(timezone.utc).isoformat(),device=runner.device_name,
                    parameter_count=sum(t.size for t in graph.parameters),allocated_bytes=graph.memory_bytes,
                    loss_before=before,loss_after=after,learning_rate=.001,
                    duration_seconds=time.perf_counter()-started,gradients=gradients)
    out=ROOT/'generated'/'full-smoke.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS: full 1B forward, backward, {len(gradients)} finite parameter tensors, SGD. Loss {before:.7f} -> {after:.7f}. Report: {out}',flush=True)


if __name__=='__main__': main()
