import argparse
import math
import os
import time
from typing import Iterable

os.environ.setdefault("DEV", "AMD")

import tinygrad.nn as nn
from PIL import Image
from tinygrad import Tensor, TinyJit
from tinygrad.helpers import GlobalCounters
from tinygrad.nn.datasets import cifar
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_save


_timestep_freqs_cache: dict[tuple[str, int], Tensor] = {}


def timestep_embedding(t: Tensor, dim: int = 256) -> Tensor:
  half = dim // 2
  key = (str(t.device), dim)
  freqs = _timestep_freqs_cache.get(key)
  if freqs is None:
    freqs = Tensor.exp(-math.log(10000) * Tensor.arange(0, half) / half).realize()
    _timestep_freqs_cache[key] = freqs
  args = t.reshape(-1, 1) * freqs.reshape(1, -1)
  return Tensor.sin(args).cat(Tensor.cos(args), dim=1)


class ResBlock:
  def __init__(self, in_ch: int, out_ch: int, t_dim: int, groups: int = 8, conditioning: str = "film"):
    self.norm1 = nn.GroupNorm(groups, in_ch)
    self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
    self.norm2 = nn.GroupNorm(groups, out_ch)
    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    self.conditioning = conditioning
    self.t_proj = nn.Linear(t_dim, out_ch * 2 if conditioning == "film" else out_ch)
    self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

  def __call__(self, x: Tensor, t_emb: Tensor) -> Tensor:
    h = self.conv1(Tensor.silu(self.norm1(x)))
    if self.conditioning == "film":
      scale_shift = self.t_proj(t_emb).reshape(x.shape[0], 2, -1, 1, 1)
      h = self.norm2(h) * (1 + scale_shift[:, 0]) + scale_shift[:, 1]
      h = self.conv2(Tensor.silu(h))
    else:
      h = h + self.t_proj(t_emb).reshape(x.shape[0], -1, 1, 1)
      h = self.conv2(Tensor.silu(self.norm2(h)))
    return h + (self.skip(x) if self.skip is not None else x)


class Downsample:
  def __init__(self, ch: int):
    self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

  def __call__(self, x: Tensor) -> Tensor:
    return self.conv(x)


class Upsample:
  def __init__(self, in_ch: int, out_ch: int):
    self.tconv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)

  def __call__(self, x: Tensor) -> Tensor:
    return self.tconv(x)


class Model:
  def __init__(self, base_ch: int = 64, t_dim: int = 256, conditioning: str = "film"):
    self.t_dim = t_dim
    self.t_mlp = [nn.Linear(t_dim, t_dim * 4), Tensor.silu, nn.Linear(t_dim * 4, t_dim)]

    self.e1 = ResBlock(3, base_ch, t_dim, conditioning=conditioning)
    self.d1 = Downsample(base_ch)
    self.e2 = ResBlock(base_ch, base_ch * 2, t_dim, conditioning=conditioning)
    self.d2 = Downsample(base_ch * 2)
    self.e3 = ResBlock(base_ch * 2, base_ch * 4, t_dim, conditioning=conditioning)
    self.d3 = Downsample(base_ch * 4)
    self.e4 = ResBlock(base_ch * 4, base_ch * 8, t_dim, conditioning=conditioning)
    self.d4 = Downsample(base_ch * 8)

    self.mid1 = ResBlock(base_ch * 8, base_ch * 16, t_dim, conditioning=conditioning)
    self.mid2 = ResBlock(base_ch * 16, base_ch * 16, t_dim, conditioning=conditioning)

    self.u4 = Upsample(base_ch * 16, base_ch * 8)
    self.de4 = ResBlock(base_ch * 16, base_ch * 8, t_dim, conditioning=conditioning)
    self.u3 = Upsample(base_ch * 8, base_ch * 4)
    self.de3 = ResBlock(base_ch * 8, base_ch * 4, t_dim, conditioning=conditioning)
    self.u2 = Upsample(base_ch * 4, base_ch * 2)
    self.de2 = ResBlock(base_ch * 4, base_ch * 2, t_dim, conditioning=conditioning)
    self.u1 = Upsample(base_ch * 2, base_ch)
    self.de1 = ResBlock(base_ch * 2, base_ch, t_dim, conditioning=conditioning)
    self.out = nn.Conv2d(base_ch, 3, 1)

  def __call__(self, x: Tensor, t: Tensor) -> Tensor:
    t_emb = Tensor.silu(timestep_embedding(t, self.t_dim).sequential(self.t_mlp))

    s1 = self.e1(x, t_emb)
    x = self.d1(s1)
    s2 = self.e2(x, t_emb)
    x = self.d2(s2)
    s3 = self.e3(x, t_emb)
    x = self.d3(s3)
    s4 = self.e4(x, t_emb)
    x = self.d4(s4)

    x = self.mid2(self.mid1(x, t_emb), t_emb)

    x = self.de4(self.u4(x).cat(s4, dim=1), t_emb)
    x = self.de3(self.u3(x).cat(s3, dim=1), t_emb)
    x = self.de2(self.u2(x).cat(s2, dim=1), t_emb)
    x = self.de1(self.u1(x).cat(s1, dim=1), t_emb)
    return self.out(x)


def load_cifar() -> Tensor:
  train_x, _, test_x, _ = cifar()
  print(f"cifar train={train_x.shape} test={test_x.shape}")
  return (train_x / 127.5 - 1.0).realize()


def make_trainer(model: Model, train_x: Tensor, batch: int, lr: float):
  optim = Adam(get_parameters(model), lr=lr)

  @TinyJit
  @Tensor.train()
  def train_step():
    noise = Tensor.randn(batch, 3, 32, 32)
    imgs = train_x[Tensor.randint(batch, high=train_x.shape[0])]
    t = Tensor.rand(batch)
    t_img = t.reshape(batch, 1, 1, 1)
    xt = (1 - t_img) * noise + t_img * imgs
    loss = (model(xt, t) - (imgs - noise)).square().mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss

  return train_step


def split_state(state: dict[str, Tensor], use_ema: bool) -> dict[str, Tensor]:
  prefix = "ema." if use_ema and any(k.startswith("ema.") for k in state) else "model."
  if any(k.startswith(prefix) for k in state):
    return {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
  return {k: v for k, v in state.items() if not k.startswith("ema.") and not k.startswith("model.")}


def get_prefixed_state(state: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
  return {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}


def save_checkpoint(model: Model, path: str, ema_params: list[Tensor] | None = None) -> None:
  state = {f"model.{k}": v for k, v in get_state_dict(model).items()}
  if ema_params is not None:
    for key, value in zip(get_state_dict(model).keys(), ema_params):
      state[f"ema.{key}"] = value.realize()
  safe_save(state, path)


def update_ema(ema_params: list[Tensor], params: Iterable[Tensor], decay: float) -> None:
  for ema, param in zip(ema_params, params):
    ema.assign(ema * decay + param.detach() * (1 - decay)).realize()


def sample(model: Model, n_steps: int = 80, bs: int = 16, method: str = "heun", eps: float = 1e-3) -> Tensor:
  Tensor.training = False
  x = Tensor.randn(bs, 3, 32, 32).realize()
  dt = (1.0 - eps) / n_steps

  for i in range(n_steps):
    ti = eps + i * dt
    t = Tensor.full((bs,), ti)
    v = model(x, t)
    x_euler = x + dt * v
    if method == "heun" and i < n_steps - 1:
      v_next = model(x_euler, Tensor.full((bs,), ti + dt))
      x = (x + dt * 0.5 * (v + v_next)).realize()
    else:
      x = x_euler.realize()
  return x


def save_grid(samples: Tensor, path: str, cols: int = 4) -> None:
  samples = ((samples + 1) * 127.5).clip(0, 255).cast("uint8")
  bs = samples.shape[0]
  rows = math.ceil(bs / cols)
  if rows * cols != bs:
    pad = Tensor.zeros(rows * cols - bs, 3, 32, 32).cast("uint8")
    samples = samples.cat(pad, dim=0)
  grid = samples.reshape(rows, cols, 3, 32, 32).permute(0, 3, 1, 4, 2).reshape(rows * 32, cols * 32, 3)
  Image.fromarray(grid.numpy()).save(path)
  print(f"saved {path}")


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("--steps", type=int, default=5000)
  p.add_argument("--batch", type=int, default=256)
  p.add_argument("--base-ch", type=int, default=64)
  p.add_argument("--t-dim", type=int, default=256)
  p.add_argument("--conditioning", choices=("film", "add"), default="film")
  p.add_argument("--lr", type=float, default=1e-4)
  p.add_argument("--ema", type=float, default=0.999)
  p.add_argument("--ema-every", type=int, default=10)
  p.add_argument("--reset-ema", action="store_true")
  p.add_argument("--no-ema", action="store_true")
  p.add_argument("--resume", action="store_true")
  p.add_argument("--sample-only", "--sample", action="store_true")
  p.add_argument("--ckpt", default="model.safetensors")
  p.add_argument("--out", default="samples.png")
  p.add_argument("--sample-steps", type=int, default=80)
  p.add_argument("--sample-bs", type=int, default=16)
  p.add_argument("--sampler", choices=("euler", "heun"), default="heun")
  p.add_argument("--log-every", type=int, default=10)
  p.add_argument("--save-every", type=int, default=1000)
  p.add_argument("--seed", type=int, default=1337)
  return p.parse_args()


def main() -> None:
  args = parse_args()
  Tensor.manual_seed(args.seed)
  print(f"device={Tensor.randn(1).device} base_ch={args.base_ch} t_dim={args.t_dim} conditioning={args.conditioning} batch={args.batch}")

  model = Model(base_ch=args.base_ch, t_dim=args.t_dim, conditioning=args.conditioning)
  params = get_parameters(model)
  ema_params = [p.detach().realize() for p in params] if not args.no_ema else None

  if args.resume or args.sample_only:
    state = safe_load(args.ckpt)
    load_state_dict(model, split_state(state, use_ema=args.sample_only and not args.no_ema), verbose=False)
    if args.resume and ema_params is not None and any(k.startswith("ema.") for k in state) and not args.reset_ema:
      ema_state = get_prefixed_state(state, "ema.")
      for key, ema in zip(get_state_dict(model).keys(), ema_params):
        if key in ema_state:
          ema.assign(ema_state[key]).realize()
    elif args.resume and ema_params is not None:
      for ema, param in zip(ema_params, params):
        ema.assign(param.detach()).realize()
    print(f"loaded {args.ckpt}")

  if not args.sample_only:
    train_x = load_cifar()
    train_step = make_trainer(model, train_x, args.batch, args.lr)
    loss_ema, start, last_log = None, time.perf_counter(), time.perf_counter()
    for step in range(1, args.steps + 1):
      loss = train_step()
      loss_value = loss.item()
      loss_ema = loss_value if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_value
      if ema_params is not None and step % args.ema_every == 0:
        update_ema(ema_params, params, args.ema ** args.ema_every)
      if step % args.log_every == 0 or step == 1:
        elapsed = time.perf_counter() - start
        now = time.perf_counter()
        window = step if step == 1 else args.log_every
        print(f"step {step:6d}/{args.steps} loss={loss_value:.4f} ema={loss_ema:.4f} avg={elapsed / step:.3f}s/step win={(now - last_log) / window:.3f}s/step mem={GlobalCounters.mem_used / 1e9:.2f}GB")
        last_log = now
      if args.save_every and step % args.save_every == 0:
        save_checkpoint(model, args.ckpt, ema_params)
    save_checkpoint(model, args.ckpt, ema_params)
    print(f"saved {args.ckpt}")

  if args.sample_bs > 0:
    if ema_params is not None and not args.sample_only:
      for param, ema in zip(params, ema_params):
        param.assign(ema).realize()
      print("sampling with EMA weights")
    samples = sample(model, n_steps=args.sample_steps, bs=args.sample_bs, method=args.sampler)
    save_grid(samples, args.out, cols=int(math.sqrt(args.sample_bs)))


if __name__ == "__main__":
  main()
