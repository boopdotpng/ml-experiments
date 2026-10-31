"""Independent float64 PyTorch oracle; never used by the HIP training runner."""
import math
import torch
import torch.nn.functional as F


def rmsnorm(x, w):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * w


def rope(x, c):
    b, t, d = c.batch, c.seq, c.head_dim
    a = x.reshape(b, t, -1, d)
    freq = 1.0 / 500000.0 ** (torch.arange(0, d, 2, dtype=x.dtype) / d)
    wavelength = 2 * math.pi / freq
    smooth = (8192.0 / wavelength - 1) / 3
    middle = (1-smooth) * freq / 32 + smooth * freq
    freq = torch.where(wavelength > 8192, freq/32, torch.where(wavelength < 2048, freq, middle))
    angles = torch.arange(c.position_offset, c.position_offset+t, dtype=x.dtype)[:, None] * freq
    cos, sin = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]
    left, right = a.chunk(2, dim=-1)
    return torch.cat((left*cos-right*sin, right*cos+left*sin), dim=-1).reshape_as(x)


def scores(q, k, c):
    q = q.reshape(c.batch,c.seq,c.heads,c.head_dim).transpose(1,2)
    k = k.reshape(c.batch,c.seq,c.kv_heads,c.head_dim).transpose(1,2)
    k = torch.repeat_interleave(k, c.heads//c.kv_heads, dim=1)
    s = (q @ k.transpose(-1,-2)) / math.sqrt(c.head_dim)
    mask = torch.ones(c.seq,c.seq,dtype=torch.bool).triu(1)
    return s.masked_fill(mask, -torch.inf)


def values(p, v, c):
    v = v.reshape(c.batch,c.seq,c.kv_heads,c.head_dim).transpose(1,2)
    v = torch.repeat_interleave(v, c.heads//c.kv_heads, dim=1)
    # HIP's PV primitive explicitly excludes future keys in its forward definition.
    return (p.tril() @ v).transpose(1,2).reshape(c.batch*c.seq,c.dim)


def operation(op, xs, c, ids, targets):
    x = xs[0]
    if op.kind == 'embedding': return F.embedding(ids, x).reshape(op.out.shape)
    if op.kind == 'linear': return F.linear(x, xs[1])
    if op.kind == 'rmsnorm': return rmsnorm(x,xs[1])
    if op.kind == 'rope': return rope(x,c)
    if op.kind == 'scores': return scores(x,xs[1],c)
    if op.kind == 'softmax': return x.softmax(dim=-1)
    if op.kind == 'values': return values(x,xs[1],c)
    if op.kind == 'add': return x+xs[1]
    if op.kind == 'swiglu': return F.silu(x)*xs[1]
    if op.kind == 'cross_entropy': return F.cross_entropy(x,targets.reshape(-1),reduction='none')
    if op.kind == 'mean': return x.mean().reshape(1)
    raise ValueError(op.kind)


def model(parameters, c, ids, targets):
    """Full model assembled independently using high-level torch operations."""
    saved = {}

    def keep(name,x):
        x.retain_grad()
        saved[name] = x
        return x

    def norm(name,x):
        return keep(name, rmsnorm(x,parameters[name+'.weight']))

    def linear(name,x):
        return keep(name, F.linear(x,parameters[name+'.weight']))

    x=keep('embedding',F.embedding(ids,parameters['embedding.weight']).reshape(c.batch*c.seq,c.dim))
    for i in range(c.layers):
        p=f'layer{i}.'
        a=norm(p+'attention_norm',x)
        q,k,v=linear(p+'q',a),linear(p+'k',a),linear(p+'v',a)
        q,k=keep(p+'q_rope',rope(q,c)),keep(p+'k_rope',rope(k,c))
        s=keep(p+'scores',scores(q,k,c))
        probs=keep(p+'probabilities',s.softmax(-1))
        context=keep(p+'context',values(probs,v,c))
        o=linear(p+'o',context)
        residual=keep(p+'attention_residual',x+o)
        a=norm(p+'ffn_norm',residual)
        gate,up=linear(p+'gate',a),linear(p+'up',a)
        act=keep(p+'swiglu',F.silu(gate)*up)
        down=linear(p+'down',act)
        x=keep(p+'ffn_residual',residual+down)
    x=norm('final_norm',x)
    logits=keep('lm_head',F.linear(x,parameters['embedding.weight']))
    losses=keep('token_losses',F.cross_entropy(logits,targets.reshape(-1),reduction='none'))
    loss=keep('loss',losses.mean().reshape(1))
    return loss,saved
