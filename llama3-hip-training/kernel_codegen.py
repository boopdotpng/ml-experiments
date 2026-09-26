"""Emit ordinary HIP functions with literal dimensions, without C++ templates.

The .hip.in file contains the math. This small Python emitter puts fixed, named
constants inside each function. The resulting src/kernels.hip is the 1B version
to read; each validation configuration gets its own separate generated file.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def dimension_names(primitive):
    if primitive.startswith('embedding_'): return ('tokens', 'dim')
    if primitive.startswith('linear_'): return ('tokens', 'output_dim', 'input_dim')
    if primitive.startswith('rmsnorm_'): return ('rows', 'dim')
    if primitive.startswith('rope_'): return ('batch', 'seq_len', 'heads', 'head_dim')
    if primitive.startswith('attention_'): return ('batch', 'seq_len', 'q_heads', 'kv_heads', 'head_dim')
    if primitive.startswith('softmax_'): return ('rows', 'cols')
    if primitive.startswith('cross_entropy_'): return ('rows', 'vocab')
    if primitive.startswith('mean_'): return ('tokens',)
    return ('elements',)


def name_kernel(primitive, dims, config):
    """Use architecture roles in names, so readers don't have to decode M/N/K."""
    if primitive.startswith('linear_'):
        _, out_dim, in_dim = dims
        if out_dim == in_dim == config.dim: role = 'hidden'
        elif out_dim == config.kv_heads * config.head_dim and in_dim == config.dim: role = 'kv'
        elif out_dim == config.hidden and in_dim == config.dim: role = 'mlp_up'
        elif out_dim == config.dim and in_dim == config.hidden: role = 'mlp_down'
        elif out_dim == config.vocab and in_dim == config.dim: role = 'lm_head'
        else: raise ValueError(f'Unexpected linear dimensions: {dims}')
        return primitive + '_' + role
    if primitive.startswith('rope_'):
        return primitive + ('_q' if dims[2] == config.heads else '_k')
    if primitive == 'sgd_update': return primitive + '_' + str(dims[0])
    return primitive


def function_source(source, name):
    pattern = r'__(?:global|device)__ (?:void|float|double) ' + re.escape(name) + r'\('
    found = re.search(pattern, source)
    if not found:
        raise ValueError(f'Missing HIP function {name}')
    brace = source.index('{', found.end())
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[found.start():end]


def emit_kernels(entries, config):
    fragments = (ROOT / 'src/kernel_bodies.hip.in').read_text()
    header, rest = fragments.split('// BEGIN KERNEL BODIES\n')
    _, runtime = rest.split('// BEGIN RUNTIME\n')
    header = header.replace('@HEAD_DIM@', str(config.head_dim)).replace('@POSITION_OFFSET@', str(config.position_offset))
    emitted = [f'// GENERATED ordinary HIP for {config.name}: all dimensions below are fixed literals.',
               '// The 1B version is also saved as src/kernels.hip for reading in VS Code.',
               '// No templates, transpose flags, or generic GEMM helper.',
               header]
    seen = {}
    for e in entries:
        name = e['kernel']
        if name in seen:
            if seen[name] != e['dimensions']:
                raise ValueError(f'Conflicting hardcoded shapes for {name}')
            continue
        seen[name] = e['dimensions']
        source = function_source(fragments, e['primitive'])
        source = source.replace(e['primitive'] + '(', name + '(', 1)
        declarations = '\n'.join(f'  const int {key} = {value};' for key,value in e['dimensions'].items())
        explanation = {
            'linear_forward': '// y[token, out] = sum_in x[token, in] * w[out, in].',
            'linear_backward_input': '// dx[token, in] += sum_out dy[token, out] * w[out, in].',
            'linear_backward_weight': '// dw[out, in] += sum_token dy[token, out] * x[token, in].',
        }.get(e['primitive'], '')
        source = source.replace('{', '{\n' + declarations + '\n' + ('  ' + explanation + '\n' if explanation else ''), 1)
        emitted.extend([f'// {name}: ' + ', '.join(f'{k}={v}' for k,v in e['dimensions'].items()), source, ''])
    emitted.append(runtime)
    return '\n'.join(emitted)
