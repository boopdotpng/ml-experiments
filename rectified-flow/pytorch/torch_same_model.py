import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def valid_groups(ch: int, groups: int) -> int:
  return math.gcd(ch, groups)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
  half = dim // 2
  freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
  args = t[:, None] * freqs[None]
  return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
  def __init__(self, in_ch: int, out_ch: int, t_dim: int, groups: int = 8, conditioning: str = "film"):
    super().__init__()
    self.conditioning = conditioning
    self.norm1 = nn.GroupNorm(valid_groups(in_ch, groups), in_ch)
    self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
    self.norm2 = nn.GroupNorm(valid_groups(out_ch, groups), out_ch)
    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    self.t_proj = nn.Linear(t_dim, out_ch * 2 if conditioning == "film" else out_ch)
    self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

  def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
    h = self.conv1(F.silu(self.norm1(x)))
    if self.conditioning == "film":
      scale, shift = self.t_proj(t_emb).view(x.shape[0], 2, -1, 1, 1).unbind(1)
      h = self.norm2(h) * (1 + scale) + shift
      h = self.conv2(F.silu(h))
    else:
      h = h + self.t_proj(t_emb).view(x.shape[0], -1, 1, 1)
      h = self.conv2(F.silu(self.norm2(h)))
    return h + self.skip(x)


class Downsample(nn.Module):
  def __init__(self, ch: int):
    super().__init__()
    self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.conv(x)


class Upsample(nn.Module):
  def __init__(self, in_ch: int, out_ch: int):
    super().__init__()
    self.tconv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.tconv(x)


class Model(nn.Module):
  def __init__(self, base_ch: int = 64, t_dim: int = 256, conditioning: str = "film"):
    super().__init__()
    self.t_dim = t_dim
    self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim * 4), nn.SiLU(), nn.Linear(t_dim * 4, t_dim))

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

  def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_emb = F.silu(self.t_mlp(timestep_embedding(t, self.t_dim)))

    s1 = self.e1(x, t_emb)
    x = self.d1(s1)
    s2 = self.e2(x, t_emb)
    x = self.d2(s2)
    s3 = self.e3(x, t_emb)
    x = self.d3(s3)
    s4 = self.e4(x, t_emb)
    x = self.d4(s4)

    x = self.mid2(self.mid1(x, t_emb), t_emb)

    x = self.de4(torch.cat([self.u4(x), s4], dim=1), t_emb)
    x = self.de3(torch.cat([self.u3(x), s3], dim=1), t_emb)
    x = self.de2(torch.cat([self.u2(x), s2], dim=1), t_emb)
    x = self.de1(torch.cat([self.u1(x), s1], dim=1), t_emb)
    return self.out(x)


def make_loader(batch: int, workers: int) -> DataLoader:
  tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
  return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)


def cycle(loader):
  while True:
    for x, _ in loader:
      yield x


def make_ema(model: nn.Module) -> dict[str, torch.Tensor]:
  return {k: v.detach().clone() for k, v in model.state_dict().items() if torch.is_floating_point(v)}


@torch.no_grad()
def update_ema(model: nn.Module, ema: dict[str, torch.Tensor], decay: float) -> None:
  state = model.state_dict()
  for k, v in ema.items():
    v.lerp_(state[k].detach(), 1 - decay)


def load_ema(model: nn.Module, ema: dict[str, torch.Tensor]) -> None:
  state = model.state_dict()
  state.update(ema)
  model.load_state_dict(state)


@torch.no_grad()
def sample(model: nn.Module, bs: int, steps: int, method: str, device: torch.device) -> torch.Tensor:
  model.eval()
  x = torch.randn(bs, 3, 32, 32, device=device)
  eps, dt = 1e-3, (1 - 1e-3) / steps
  for i in range(steps):
    ti = eps + i * dt
    t = torch.full((bs,), ti, device=device)
    v = model(x, t)
    x_euler = x + dt * v
    if method == "heun" and i < steps - 1:
      v_next = model(x_euler, torch.full((bs,), ti + dt, device=device))
      x = x + dt * 0.5 * (v + v_next)
    else:
      x = x_euler
  return x


def save_grid(samples: torch.Tensor, path: str, cols: int) -> None:
  samples = ((samples.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu()
  rows = math.ceil(samples.shape[0] / cols)
  canvas = torch.zeros(rows * 32, cols * 32, 3, dtype=torch.uint8)
  for i, img in enumerate(samples):
    r, c = divmod(i, cols)
    canvas[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = img
  Image.fromarray(canvas.numpy()).save(path)
  print(f"saved {path}", flush=True)


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("--steps", type=int, default=1000)
  p.add_argument("--batch", type=int, default=128)
  p.add_argument("--base-ch", type=int, default=32)
  p.add_argument("--t-dim", type=int, default=128)
  p.add_argument("--conditioning", choices=("film", "add"), default="film")
  p.add_argument("--lr", type=float, default=1e-4)
  p.add_argument("--workers", type=int, default=8)
  p.add_argument("--compile", action="store_true")
  p.add_argument("--amp", action="store_true")
  p.add_argument("--ema", type=float, default=0.999)
  p.add_argument("--ema-every", type=int, default=10)
  p.add_argument("--no-ema", action="store_true")
  p.add_argument("--ckpt", default="torch_same_model.pt")
  p.add_argument("--resume", action="store_true")
  p.add_argument("--sample-only", action="store_true")
  p.add_argument("--out", default="torch_same_samples.png")
  p.add_argument("--sample-bs", type=int, default=16)
  p.add_argument("--sample-steps", type=int, default=100)
  p.add_argument("--sampler", choices=("euler", "heun"), default="heun")
  p.add_argument("--log-every", type=int, default=50)
  p.add_argument("--save-every", type=int, default=0)
  p.add_argument("--seed", type=int, default=1337)
  return p.parse_args()


def main() -> None:
  args = parse_args()
  torch.manual_seed(args.seed)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"device={device} name={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} base_ch={args.base_ch} batch={args.batch} amp={args.amp} compile={args.compile}", flush=True)

  model = Model(args.base_ch, args.t_dim, args.conditioning).to(device)
  opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
  ema = None if args.no_ema else make_ema(model)
  if args.resume or args.sample_only:
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "opt" in ckpt and args.resume:
      opt.load_state_dict(ckpt["opt"])
    if ema is not None:
      ema = ckpt.get("ema", make_ema(model))
      if args.sample_only:
        load_ema(model, ema)
    print(f"loaded {args.ckpt}", flush=True)

  run_model = torch.compile(model, mode="max-autotune") if args.compile else model
  scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

  if not args.sample_only:
    batches = cycle(make_loader(args.batch, args.workers))
    start = last = time.perf_counter()
    loss_ema = None
    for step in range(1, args.steps + 1):
      imgs = next(batches).to(device, non_blocking=True)
      noise = torch.randn_like(imgs)
      t = torch.rand(args.batch, device=device)
      xt = torch.lerp(noise, imgs, t[:, None, None, None])
      target = imgs - noise
      opt.zero_grad(set_to_none=True)
      with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
        loss = F.mse_loss(run_model(xt, t), target)
      scaler.scale(loss).backward()
      scaler.step(opt)
      scaler.update()
      if ema is not None and step % args.ema_every == 0:
        update_ema(model, ema, args.ema ** args.ema_every)

      loss_value = float(loss.detach())
      loss_ema = loss_value if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_value
      if step == 1 or step % args.log_every == 0:
        torch.cuda.synchronize() if device.type == "cuda" else None
        now = time.perf_counter()
        window = step if step == 1 else args.log_every
        mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
        print(f"step {step:6d}/{args.steps} loss={loss_value:.4f} ema={loss_ema:.4f} avg={(now - start) / step:.4f}s/step win={(now - last) / window:.4f}s/step mem={mem:.2f}GB", flush=True)
        last = now
      if args.save_every and step % args.save_every == 0:
        torch.save({"model": model.state_dict(), "ema": ema, "opt": opt.state_dict(), "args": vars(args)}, args.ckpt)

    torch.save({"model": model.state_dict(), "ema": ema, "opt": opt.state_dict(), "args": vars(args)}, args.ckpt)
    print(f"saved {args.ckpt}", flush=True)

  if args.sample_bs:
    samples = sample(model, args.sample_bs, args.sample_steps, args.sampler, device)
    save_grid(samples, args.out, cols=int(math.sqrt(args.sample_bs)))


if __name__ == "__main__":
  main()
