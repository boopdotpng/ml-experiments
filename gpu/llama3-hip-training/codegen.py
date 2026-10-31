"""Lower forward ops and their vector-Jacobian products into actual HIP launches."""
import json
import re
from pathlib import Path
from kernel_codegen import dimension_names, emit_kernels, name_kernel

ROOT = Path(__file__).resolve().parent


def v(t):
    return f"v[{t.id}]"


def g(t):
    return f"g[{t.id}]"


def launch(kernel, dims, args, count=None, rows=None, matrix=None):
    if matrix:
        grid, block = f"dim3({(matrix[1]+15)//16}, {(matrix[0]+15)//16})", "dim3(16, 16)"
    else:
        grid, block = str(rows if rows is not None else (count + 255) // 256), "256"
    return dict(primitive=kernel, dims=dims, dimensions=dict(zip(dimension_names(kernel), dims)),
                args=args, grid=grid, block=block)


def resolve_launch(entry, config):
    entry['kernel'] = name_kernel(entry['primitive'], entry['dims'], config)
    entry['code'] = (f"  {entry['kernel']}<<<{entry['grid']}, {entry['block']}>>>({', '.join(entry['args'])});\n"
                     "  CHECK(hipGetLastError());")
    return entry


def lower(op, c):
    """One forward op -> forward launches and backward launches. += is mandatory."""
    o, xs = op.out, op.inputs
    n, m = o.size, c.batch * c.seq
    x = xs[0]
    if op.kind == "embedding":
        dims = (m, c.dim)
        return [launch("embedding_forward", dims, [v(x), "ids", v(o)], count=n)], [
            launch("embedding_backward", dims, [g(o), "ids", g(x)], count=n)]
    if op.kind == "linear":
        w = xs[1]
        outdim, indim = w.shape
        dims = (m, outdim, indim)
        return [launch("linear_forward", dims, [v(x), v(w), v(o)], matrix=(m, outdim))], [
            launch("linear_backward_input", dims, [g(o), v(w), g(x)], matrix=(m, indim)),
            launch("linear_backward_weight", dims, [g(o), v(x), g(w)], matrix=(outdim, indim))]
    if op.kind == "rmsnorm":
        w, r, d = xs[1], op.saved[0], o.shape[-1]
        dims = (m, d)
        return [launch("rmsnorm_forward", dims, [v(x), v(w), v(o), v(r)], rows=m)], [
            launch("rmsnorm_backward_input", dims, [v(x), v(w), v(r), g(o), g(x)], rows=m),
            launch("rmsnorm_backward_weight", dims, [v(x), v(r), g(o), g(w)], count=d)]
    if op.kind == "rope":
        dims = (c.batch, c.seq, o.shape[-1] // c.head_dim, c.head_dim)
        return [launch("rope_forward", dims, [v(x), v(o)], count=n//2)], [
            launch("rope_backward", dims, [g(o), g(x)], count=n//2)]
    if op.kind in ("scores", "values"):
        y = xs[1]
        dims = (c.batch, c.seq, c.heads, c.kv_heads, c.head_dim)
        prefix = "attention_" + op.kind
        suffixes = ("q", "k") if op.kind == "scores" else ("p", "v")
        return [launch(prefix+"_forward", dims, [v(x), v(y), v(o)], count=n)], [
            launch(prefix+"_backward_"+suffixes[0], dims, [g(o), v(y), g(x)], count=x.size),
            launch(prefix+"_backward_"+suffixes[1], dims, [g(o), v(x), g(y)], count=y.size)]
    if op.kind == "softmax":
        dims = (n//o.shape[-1], o.shape[-1])
        return [launch("softmax_forward", dims, [v(x), v(o)], rows=dims[0])], [
            launch("softmax_backward", dims, [v(o), g(o), g(x)], rows=dims[0])]
    if op.kind in ("add", "swiglu"):
        y = xs[1]
        args = [g(o), g(x), g(y)] if op.kind == "add" else [v(x), v(y), g(o), g(x), g(y)]
        return [launch(op.kind+"_forward", (n,), [v(x), v(y), v(o)], count=n)], [
            launch(op.kind+"_backward", (n,), args, count=n)]
    if op.kind == "cross_entropy":
        dims = (m, c.vocab)
        return [launch("cross_entropy_forward", dims, [v(x), "targets", v(o)], rows=m)], [
            launch("cross_entropy_backward", dims, [v(x), "targets", g(o), g(x)], rows=m)]
    if op.kind == "mean":
        return [launch("mean_forward", (m,), [v(x), v(o)], rows=1)], [
            launch("mean_backward", (m,), [g(o), g(x)], count=m)]
    raise ValueError(op.kind)


def generate(graph, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for i, op in enumerate(graph.ops):
        fwd, bwd = lower(op, graph.config)
        for entry in fwd + bwd:
            resolve_launch(entry, graph.config)
        for entry in bwd:
            entry['gradients'] = list(dict.fromkeys(int(i) for i in re.findall(r'g\[(\d+)\]', entry['code']) if int(i) != op.out.id))
        records.append(dict(index=i, name=op.out.name, kind=op.kind, out=op.out.id,
                            shape=op.out.shape, inputs=[t.id for t in op.inputs],
                            saved=[t.id for t in op.saved], forward=fwd, backward=bwd))
    sources = {}
    for phase, ordered in (("forward", records), ("backward", records[::-1])):
        lines = [f'// GENERATED {phase.upper()} execution order. Do not edit; run train.py emit.',
                 '// v[] = saved forward values; g[] = accumulated gradients.',
                 f'extern "C" int run_{phase}(float** v, float** g, const int* ids, const int* targets) {{']
        if phase == "backward":
            lines.extend(['  // The runner zeros all g[] before backward. Seed dLoss/dLoss = 1.',
                          '  float seed = 1.0f;',
                          f'  CHECK(hipMemcpy(g[{graph.loss.id}], &seed, sizeof(float), hipMemcpyHostToDevice));'])
        for record in ordered:
            lines.append(f'  // {record["name"]}: {record["kind"]} -> {record["shape"]}')
            for entry in record[phase]:
                entry["file"] = phase + ".hip"
                entry["line"] = len(lines) + 1
                lines.extend(entry["code"].splitlines())
        lines.extend(['  return hipDeviceSynchronize();', '}'])
        sources[phase] = '\n'.join(lines) + '\n'
        (directory / (phase + '.hip')).write_text(sources[phase])
    updates = [resolve_launch(launch('sgd_update', (t.size,), [v(t),g(t),'lr'], count=t.size), graph.config)
               for t in graph.parameters]
    entries = [e for r in records for phase in ('forward', 'backward') for e in r[phase]] + updates
    kernels = emit_kernels(entries, graph.config)
    (directory / 'kernels.hip').write_text(kernels)
    if graph.config.name == '1b':
        # Keep the file already open in VS Code as the readable fixed-size model.
        (ROOT / 'src/kernels.hip').write_text(kernels)
    update = ['extern "C" int run_sgd(float** v, float** g, float lr) {']
    for t, entry in zip(graph.parameters, updates):
        update += [f'  // {t.name}', entry['code']]
    update += ['  return hipDeviceSynchronize();', '}']
    (directory / 'training.hip').write_text('#include "kernels.hip"\n'
        '#define CHECK(expr) do { hipError_t e=(hipError_t)(expr); if(e!=hipSuccess) return int(e); } while(0)\n'
        '#include "forward.hip"\n#include "backward.hip"\n' + '\n'.join(update) + '\n')
    # Isolated dispatch uses the exact same emitted launches as the training path.
    with (directory / 'training.hip').open('a') as f:
        for phase in ('forward', 'backward'):
            f.write(f'extern "C" int single_{phase}(int op, float** v, float** g, const int* ids, const int* targets) {{\n  switch(op) {{\n')
            for r in records:
                f.write(f'  case {r["index"]}: {{\n')
                f.write('\n'.join(e['code'] for e in r[phase]))
                f.write('\n    break;\n  }\n')
            f.write('  default: return int(hipErrorInvalidValue);\n  }\n  return hipDeviceSynchronize();\n}\n')
    manifest = graph.manifest() | dict(operations=records,
        forward_launches=sum(len(r['forward']) for r in records),
        backward_launches=sum(len(r['backward']) for r in records))
    (directory / 'graph.json').write_text(json.dumps(manifest, indent=2) + '\n')
    lines = ['# Generated backward execution order', '',
             'Each gradient is accumulated with `+=`. The seed is `dLoss = 1`.', '',
             '| Step | Forward operation | Backward kernel | Gradients written |',
             '| --- | --- | --- | --- |']
    step = 0
    for r in records[::-1]:
        for entry in r['backward']:
            step += 1
            lines.append(f'| {step} | {r["name"]} | `{entry["kernel"]}` | ' +
                         ', '.join('d' + graph.tensors[i].name for i in entry['gradients']) + ' |')
    (directory / 'backward-operations.md').write_text('\n'.join(lines) + '\n')
    # DOT includes both traversals, branch edges, and shared parameter leaves.
    dot = ['digraph Training {', 'rankdir=LR;', 'node [shape=box, fontname="sans-serif"];']
    for t in graph.tensors:
        if not t.grad:
            continue
        dot += [f'v{t.id} [label="{t.name}\\n{t.shape}"];',
                f'g{t.id} [label="d {t.name}", color="#d88832"];']
    for op in graph.ops:
        for x in op.inputs:
            dot += [f'v{x.id} -> v{op.out.id};', f'g{op.out.id} -> g{x.id} [color="#d88832"];']
    dot += [f'v{graph.loss.id} -> g{graph.loss.id} [label="seed 1"];', '}']
    (directory / 'graph.dot').write_text('\n'.join(dot) + '\n')
    from visualize import write_viewer
    write_viewer(manifest, directory, kernels)
    return manifest
