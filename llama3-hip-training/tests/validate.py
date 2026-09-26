#!/usr/bin/env python3
"""Real GPU checks: isolated VJPs, full-model gradients, finite differences, SGD."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import Config, Graph
from runtime import Runner
from reference import operation, model

torch.set_num_threads(1)
checks=[]


def compare(name, actual, expected, atol=3e-6, rtol=4e-4):
    expected=np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(f'{name}: shape {actual.shape} != {expected.shape}')
    np.testing.assert_allclose(actual,expected,atol=atol,rtol=rtol,err_msg=name)
    finite=np.isfinite(expected)
    error=float(np.max(np.abs(actual[finite]-expected[finite]),initial=0))
    checks.append(dict(name=name,max_absolute_error=error))


def tokens(c):
    # Repeated IDs within and across batches exercise the embedding atomicAdd.
    return np.tile(np.arange(c.seq+1)%min(c.vocab,3),(c.batch,1)).astype(np.int32)


def isolated(runner,c):
    rng=np.random.default_rng(400)
    graph=runner.graph
    sequence=tokens(c)
    runner.set_batch(sequence)
    ids=torch.tensor(sequence[:,:-1].copy(),dtype=torch.long)
    targets=torch.tensor(sequence[:,1:].copy(),dtype=torch.long)
    seen=set()
    for index,op in enumerate(graph.ops):
        key=(op.kind,op.out.shape,tuple(t.shape for t in op.inputs))
        if key in seen: continue
        seen.add(key)
        xs=[]
        for j,t in enumerate(op.inputs):
            a=rng.normal(0,.7,t.shape).astype(np.float32)
            if op.kind=='rmsnorm' and j==0:
                a[0]=0  # epsilon dominates this row; dx is still nonzero.
                if len(a)>1: a[1]*=1e-4
            if op.kind=='swiglu' and j==0: a*=5
            if op.kind=='softmax':
                a[...,np.triu_indices(c.seq,1)[0],np.triu_indices(c.seq,1)[1]]=-np.inf
            if op.kind=='cross_entropy':
                a*=50
                a[:,0]+=1000  # stable CE under large logits and tiny probabilities.
            runner.upload(runner.v[t.id],a)
            xs.append(torch.tensor(a,dtype=torch.float64,requires_grad=True))
        expected=operation(op,xs,c,ids,targets)
        runner.check(runner.lib.single_forward(index,runner.v,runner.g,runner.ids,runner.targets))
        compare(f'{c.name}/isolated/{op.out.name}/forward',runner.read(op.out),expected.detach().numpy(),atol=1e-5)
        upstream=rng.normal(0,.4,op.out.shape).astype(np.float32)
        # Masked scores are constants; don't multiply -inf outputs in a scalar loss.
        expected.backward(torch.tensor(upstream,dtype=torch.float64))
        runner.zero_grad()
        runner.upload(runner.g[op.out.id],upstream)
        bases=[]
        for t in op.inputs:
            baseline=rng.normal(0,.02,t.shape).astype(np.float32)
            runner.upload(runner.g[t.id],baseline)
            bases.append(baseline)
        runner.check(runner.lib.single_backward(index,runner.v,runner.g,runner.ids,runner.targets))
        for t,ref,base in zip(op.inputs,xs,bases):
            compare(f'{c.name}/isolated/{op.out.name}/d({t.name})',runner.read(t,True),ref.grad.numpy()+base,atol=8e-6)
        if op.kind=='softmax':
            ds=runner.read(op.inputs[0],True)-bases[0]
            compare(f'{c.name}/softmax/row-gradient-sum',ds.sum(-1),np.zeros(ds.shape[:-1]),atol=2e-6)
    return len(seen)


def end_to_end(runner,c,initial):
    graph=runner.graph
    for t in graph.parameters: runner.upload(runner.v[t.id],initial[t.name])
    sequence=tokens(c)
    runner.set_batch(sequence)
    ids=torch.tensor(sequence[:,:-1].copy(),dtype=torch.long)
    targets=torch.tensor(sequence[:,1:].copy(),dtype=torch.long)
    params={name:torch.tensor(a,dtype=torch.float64,requires_grad=True) for name,a in initial.items()}
    loss,saved=model(params,c,ids,targets)
    loss.backward()
    before=runner.forward()
    runner.backward()
    for t in graph.tensors:
        if not t.grad: continue
        ref=params[t.name] if t.parameter else saved[t.name]
        compare(f'{c.name}/model/{t.name}/value',runner.read(t),ref.detach().numpy(),atol=5e-6)
        compare(f'{c.name}/model/{t.name}/gradient',runner.read(t,True),ref.grad.numpy(),atol=2e-6)
    # Calling backward twice must reset the previous iteration's accumulators.
    first={t.name:runner.read(t,True) for t in graph.parameters}
    runner.backward()
    for t in graph.parameters:
        compare(f'{c.name}/reset/{t.name}',runner.read(t,True),first[t.name],atol=1e-6)
    # Independent central differences for a direction in EVERY parameter tensor.
    # Float64 oracle prevents FP32 cancellation from hiding the small Q/K gradients.
    rng=np.random.default_rng(31)
    for name,param in params.items():
        direction=torch.tensor(rng.normal(size=param.shape),dtype=torch.float64)
        direction/=torch.linalg.vector_norm(direction)
        analytic=float((param.grad*direction).sum())
        original=param.detach().clone()
        eps=1e-5
        with torch.no_grad(): param.copy_(original+eps*direction)
        plus=float(model(params,c,ids,targets)[0].detach()[0])
        with torch.no_grad(): param.copy_(original-eps*direction)
        minus=float(model(params,c,ids,targets)[0].detach()[0])
        with torch.no_grad(): param.copy_(original)
        finite=(plus-minus)/(2*eps)
        compare(f'{c.name}/finite-difference/{name}',np.array([analytic]),np.array([finite]),atol=2e-8,rtol=1e-3)
        gpu_direction=float((first[name].astype(np.float64)*direction.numpy()).sum())
        compare(f'{c.name}/HIP-vs-finite-difference/{name}',np.array([gpu_direction]),np.array([finite]),atol=2e-7,rtol=2e-3)
    lr=.02
    runner.step(lr)
    for t in graph.parameters:
        compare(f'{c.name}/SGD/{t.name}',runner.read(t),initial[t.name]-lr*first[t.name],atol=1e-7)
    after=runner.forward()
    if not after<before: raise AssertionError(f'SGD failed to lower loss: {before} -> {after}')
    # A subsequent fresh forward/backward/update must also use the updated weights.
    runner.backward()
    runner.step(lr)
    final=runner.forward()
    if not final<after: raise AssertionError(f'Second SGD step failed: {after} -> {final}')
    return dict(before=before,after_one_step=after,after_two_steps=final,parameters_checked=len(params))


def input_validation(runner,c):
    for bad in (np.zeros((c.batch,c.seq),np.int32),np.full((c.batch,c.seq+1),-1,np.int32),
                np.full((c.batch,c.seq+1),c.vocab,np.int32),np.zeros((c.batch,c.seq+1),np.float32)):
        try: runner.set_batch(bad)
        except ValueError: pass
        else: raise AssertionError('Invalid tokens were accepted')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arch',default=None)
    p.add_argument('--device',type=int,default=0)
    args=p.parse_args()
    cases=[Config(),
           Config(name='tails',batch=2,seq=5,dim=24,hidden=37,layers=2,heads=6,kv_heads=2,vocab=43,position_offset=19),
           Config(name='single',batch=1,seq=1,dim=16,hidden=19,layers=1,heads=2,kv_heads=2,vocab=23),
           Config(name='scaled_rope',batch=1,seq=3,dim=64,hidden=67,layers=1,heads=1,kv_heads=1,vocab=71,position_offset=8192)]
    started=time.perf_counter()
    summaries=[]
    for c in cases:
        print(f'Checking {c.name}: {asdict(c)}',flush=True)
        with Runner(Graph(c),ROOT/'build'/'validation'/c.name,device=args.device,arch=args.arch) as runner:
            initial={t.name:runner.read(t) for t in runner.graph.parameters}
            count=isolated(runner,c)
            result=end_to_end(runner,c,initial)
            input_validation(runner,c)
            summaries.append(dict(config=asdict(c),isolated_operations=count,**result))
            print(f'  PASS: {count} isolated shapes; {result["parameters_checked"]} parameter tensors; '
                  f'loss {result["before"]:.6f} -> {result["after_two_steps"]:.6f}',flush=True)
            device_name=runner.device_name
    report=dict(status='passed',device=device_name,torch_version=torch.__version__,
                duration_seconds=time.perf_counter()-started,comparisons=len(checks),cases=summaries,checks=checks)
    output=ROOT/'generated'/'validation.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS: {len(checks)} numerical comparisons on {device_name}. Report: {output}',flush=True)


if __name__=='__main__': main()
