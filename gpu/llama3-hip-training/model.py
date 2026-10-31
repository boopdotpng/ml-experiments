"""A small, inspectable graph IR. No framework autograd is used for HIP training."""
from dataclasses import asdict, dataclass, field
from math import prod


@dataclass(frozen=True)
class Config:
    name: str = "tiny"
    batch: int = 2
    seq: int = 8
    dim: int = 64
    hidden: int = 256
    layers: int = 2
    heads: int = 4
    kv_heads: int = 1
    vocab: int = 128
    position_offset: int = 0

    @property
    def head_dim(self):
        return self.dim // self.heads

    @classmethod
    def preset(cls, name, **kwargs):
        base = dict(name=name)
        if name == "1b":
            base.update(batch=1, seq=8, dim=2048, hidden=8192, layers=16,
                        heads=32, kv_heads=8, vocab=128256)
        elif name != "tiny":
            raise ValueError(f"Unknown preset: {name}")
        return cls(**(base | kwargs))

    def validate(self):
        for name in ("batch", "seq", "dim", "hidden", "layers", "heads", "kv_heads", "vocab"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.dim % self.heads or self.heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("dim must divide into heads, heads into KV groups, and head_dim must be even")
        if self.position_offset < 0 or self.position_offset + self.seq > 131072:
            raise ValueError("positions must be in [0, 131072)")


@dataclass
class Tensor:
    id: int
    name: str
    shape: tuple
    parameter: bool = False
    grad: bool = True

    @property
    def size(self):
        return prod(self.shape)


@dataclass
class Op:
    kind: str
    out: Tensor
    inputs: list
    saved: list = field(default_factory=list)


class Graph:
    def __init__(self, config):
        config.validate()
        self.config = config
        self.tensors, self.ops = [], []
        c = config
        m, d, f, k = c.batch * c.seq, c.dim, c.hidden, c.kv_heads * c.head_dim
        emb = self.tensor("embedding.weight", (c.vocab, d), parameter=True)
        x = self.op("embedding", "embedding", [emb], (m, d))
        for layer in range(c.layers):
            prefix = f"layer{layer}."

            def linear(name, a, width):
                w = self.tensor(prefix + name + ".weight", (width, a.shape[-1]), parameter=True)
                return self.op("linear", prefix + name, [a, w], (m, width))

            def norm(name, a):
                w = self.tensor(prefix + name + ".weight", (d,), parameter=True)
                return self.op("rmsnorm", prefix + name, [a, w], a.shape)

            a = norm("attention_norm", x)
            q, key, v = linear("q", a, d), linear("k", a, k), linear("v", a, k)
            q = self.op("rope", prefix + "q_rope", [q], q.shape)
            key = self.op("rope", prefix + "k_rope", [key], key.shape)
            scores = self.op("scores", prefix + "scores", [q, key], (c.batch, c.heads, c.seq, c.seq))
            p = self.op("softmax", prefix + "probabilities", [scores], scores.shape)
            context = self.op("values", prefix + "context", [p, v], (m, d))
            attn = linear("o", context, d)
            residual = self.op("add", prefix + "attention_residual", [x, attn], x.shape)
            a = norm("ffn_norm", residual)
            gate, up = linear("gate", a, f), linear("up", a, f)
            act = self.op("swiglu", prefix + "swiglu", [gate, up], (m, f))
            down = linear("down", act, d)
            x = self.op("add", prefix + "ffn_residual", [residual, down], x.shape)
        norm_w = self.tensor("final_norm.weight", (d,), parameter=True)
        x = self.op("rmsnorm", "final_norm", [x, norm_w], x.shape)
        # SAME tensor, not a copied parameter. Backward sums both uses into its grad.
        self.logits = self.op("linear", "lm_head", [x, emb], (m, c.vocab))
        losses = self.op("cross_entropy", "token_losses", [self.logits], (m,))
        self.loss = self.op("mean", "loss", [losses], (1,))

    def tensor(self, name, shape, parameter=False, grad=True):
        if prod(shape) >= 2**31:
            raise ValueError(f"{name} exceeds this educational backend's 32-bit index range")
        t = Tensor(len(self.tensors), name, tuple(shape), parameter, grad)
        self.tensors.append(t)
        return t

    def op(self, kind, name, inputs, shape):
        out = self.tensor(name, shape)
        saved = [self.tensor(name + ".inv_rms", (shape[0],), grad=False)] if kind == "rmsnorm" else []
        self.ops.append(Op(kind, out, inputs, saved))
        return out

    @property
    def parameters(self):
        return [t for t in self.tensors if t.parameter]

    @property
    def memory_bytes(self):
        return sum(t.size * 4 * (1 + int(t.grad)) for t in self.tensors) + self.config.batch * self.config.seq * 8

    def manifest(self):
        return dict(config=asdict(self.config), parameter_count=sum(t.size for t in self.parameters),
                    allocated_bytes=self.memory_bytes, tensors=[asdict(t) for t in self.tensors])
