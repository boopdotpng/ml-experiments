"""Compile emitted HIP and execute it through ctypes. No torch dependency."""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np

from codegen import generate


def compile_hip(directory, arch=None):
    directory = Path(directory).resolve()
    hipcc = os.environ.get('HIPCC', 'hipcc')
    flags = ['-O2', '-std=c++17', '-shared', '-fPIC', '-lamdhip64']
    if arch:
        flags += [f'--offload-arch={arch}']
    digest = hashlib.sha256()
    for filename in ('training.hip', 'kernels.hip', 'forward.hip', 'backward.hip'):
        digest.update((directory / filename).read_bytes())
    digest.update(repr((hipcc, flags)).encode())
    # Include toolchain version: replacing hipcc must not reuse an old binary.
    env = {k: val for k, val in os.environ.items() if not k.startswith('BASH_FUNC_')}
    digest.update(subprocess.check_output([hipcc, '--version'], env=env, stderr=subprocess.STDOUT))
    lib = directory / ('training-' + digest.hexdigest()[:16] + '.so')
    if not lib.exists():
        tmp = lib.with_suffix('.tmp.so')
        try:
            subprocess.run([hipcc, *flags, str(directory / 'training.hip'), '-o', str(tmp)],
                           check=True, env=env)
            tmp.replace(lib)
        finally:
            tmp.unlink(missing_ok=True)
    return lib


class Runner:
    def __init__(self, graph, directory, device=0, arch=None, seed=7):
        self.graph = graph
        self.manifest = generate(graph, directory)
        self.lib = ct.CDLL(str(compile_hip(directory, arch)))
        p, z = ct.c_void_p, ct.c_size_t
        signatures = dict(gpu_init=[ct.c_int], gpu_memory=[ct.POINTER(z),ct.POINTER(z)],
            gpu_alloc=[ct.POINTER(p),z], gpu_free=[p], gpu_zero=[p,z],
            gpu_upload=[p,p,z], gpu_download=[p,p,z],
            run_forward=[ct.POINTER(p),ct.POINTER(p),p,p],
            run_backward=[ct.POINTER(p),ct.POINTER(p),p,p],
            run_sgd=[ct.POINTER(p),ct.POINTER(p),ct.c_float],
            single_forward=[ct.c_int,ct.POINTER(p),ct.POINTER(p),p,p],
            single_backward=[ct.c_int,ct.POINTER(p),ct.POINTER(p),p,p])
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, ct.c_int
        self.lib.gpu_error.argtypes, self.lib.gpu_error.restype = [ct.c_int], ct.c_char_p
        self.lib.gpu_name.argtypes, self.lib.gpu_name.restype = [], ct.c_char_p
        self.check(self.lib.gpu_init(device))
        available, total = z(), z()
        self.check(self.lib.gpu_memory(ct.byref(available), ct.byref(total)))
        if graph.memory_bytes > available.value * 0.92:
            raise MemoryError(f'Needs {graph.memory_bytes/2**30:.2f} GiB of tensors; '
                              f'GPU has {available.value/2**30:.2f} GiB free. Reduce --seq/--batch.')
        self.device_name = self.lib.gpu_name().decode()
        self.allocations = []
        self.v = (p * len(graph.tensors))()
        self.g = (p * len(graph.tensors))()
        try:
            rng = np.random.default_rng(seed)
            for t in graph.tensors:
                self.v[t.id] = self.alloc(t.size * 4)
                if t.grad:
                    self.g[t.id] = self.alloc(t.size * 4)
                if t.parameter:
                    a = np.ones(t.shape, np.float32) if len(t.shape) == 1 else rng.standard_normal(t.shape, dtype=np.float32) * .02
                    self.upload(self.v[t.id], a)
            count = graph.config.batch * graph.config.seq
            self.ids, self.targets = self.alloc(count*4), self.alloc(count*4)
            self.data_ready, self.forward_ready, self.backward_ready = False, False, False
        except BaseException:
            self.close()
            raise

    def check(self, code):
        if code:
            raise RuntimeError(self.lib.gpu_error(code).decode())

    def alloc(self, nbytes):
        ptr = ct.c_void_p()
        self.check(self.lib.gpu_alloc(ct.byref(ptr), nbytes))
        self.allocations.append(ptr)
        return ptr.value

    def upload(self, ptr, array):
        array = np.ascontiguousarray(array)
        self.check(self.lib.gpu_upload(ptr, array.ctypes.data, array.nbytes))

    def read(self, tensor, gradient=False):
        array = np.empty(tensor.shape, dtype=np.float32)
        ptr = self.g[tensor.id] if gradient else self.v[tensor.id]
        if not ptr:
            raise ValueError(f'{tensor.name} has no gradient')
        self.check(self.lib.gpu_download(array.ctypes.data, ptr, array.nbytes))
        return array

    def set_batch(self, sequence):
        c = self.graph.config
        sequence = np.asarray(sequence)
        if sequence.shape != (c.batch, c.seq + 1):
            raise ValueError(f'Expected integer token array {(c.batch, c.seq+1)}, got {sequence.shape}')
        if not np.issubdtype(sequence.dtype, np.integer) or (sequence < 0).any() or (sequence >= c.vocab).any():
            raise ValueError(f'Tokens must be integers in [0, {c.vocab})')
        self.upload(self.ids, sequence[:, :-1].astype(np.int32))
        self.upload(self.targets, sequence[:, 1:].astype(np.int32))
        self.data_ready, self.forward_ready, self.backward_ready = True, False, False

    def zero_grad(self):
        for t in self.graph.tensors:
            if t.grad:
                self.check(self.lib.gpu_zero(self.g[t.id], t.size * 4))

    def forward(self):
        if not self.data_ready:
            raise RuntimeError('Call set_batch before forward')
        self.check(self.lib.run_forward(self.v, self.g, self.ids, self.targets))
        self.forward_ready, self.backward_ready = True, False
        return float(self.read(self.graph.loss)[0])

    def backward(self):
        if not self.forward_ready:
            raise RuntimeError('Call forward before backward')
        # A new training iteration starts fresh. Within the traversal, += sums paths.
        self.zero_grad()
        self.check(self.lib.run_backward(self.v, self.g, self.ids, self.targets))
        self.backward_ready = True

    def step(self, lr):
        if not self.backward_ready:
            raise RuntimeError('Call backward before step')
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError('Learning rate must be finite and positive')
        self.check(self.lib.run_sgd(self.v, self.g, lr))
        self.forward_ready, self.backward_ready = False, False

    def close(self):
        for ptr in reversed(self.allocations):
            self.lib.gpu_free(ptr)
        self.allocations.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
