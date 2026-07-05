import argparse
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def valid_groups(ch: int, groups: int = 8) -> int:
  return math.gcd(ch, groups)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
  half = dim // 2
  freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
  args = t[:, None] * freqs[None]
  return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
  def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
    super().__init__()
    self.norm1 = nn.GroupNorm(valid_groups(in_ch), in_ch)
    self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
    self.norm2 = nn.GroupNorm(valid_groups(out_ch), out_ch)
    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    self.emb = nn.Linear(emb_dim, out_ch * 2)
    self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

  def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    h = self.conv1(F.silu(self.norm1(x)))
    scale, shift = self.emb(emb).view(x.shape[0], 2, -1, 1, 1).unbind(1)
    h = self.norm2(h) * (1 + scale) + shift
    h = self.conv2(F.silu(h))
    return h + self.skip(x)


class AttentionBlock(nn.Module):
  def __init__(self, ch: int, heads: int = 4):
    super().__init__()
    self.norm = nn.GroupNorm(valid_groups(ch), ch)
    self.qkv = nn.Conv2d(ch, ch * 3, 1)
    self.proj = nn.Conv2d(ch, ch, 1)
    self.heads = heads

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
    q = q.view(b, self.heads, c // self.heads, h * w).transpose(2, 3)
    k = k.view(b, self.heads, c // self.heads, h * w).transpose(2, 3)
    v = v.view(b, self.heads, c // self.heads, h * w).transpose(2, 3)
    y = F.scaled_dot_product_attention(q, k, v)
    y = y.transpose(2, 3).contiguous().view(b, c, h, w)
    return x + self.proj(y)


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


class ClassRF(nn.Module):
  def __init__(self, base_ch: int = 96, emb_dim: int = 384, class_drop: float = 0.1):
    super().__init__()
    self.emb_dim, self.class_drop = emb_dim, class_drop
    self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim * 4), nn.SiLU(), nn.Linear(emb_dim * 4, emb_dim))
    self.class_emb = nn.Embedding(11, emb_dim)

    self.e1 = ResBlock(3, base_ch, emb_dim)
    self.d1 = Downsample(base_ch)
    self.e2 = ResBlock(base_ch, base_ch * 2, emb_dim)
    self.a2 = AttentionBlock(base_ch * 2)
    self.d2 = Downsample(base_ch * 2)
    self.e3 = ResBlock(base_ch * 2, base_ch * 4, emb_dim)
    self.a3 = AttentionBlock(base_ch * 4)
    self.d3 = Downsample(base_ch * 4)
    self.e4 = ResBlock(base_ch * 4, base_ch * 8, emb_dim)
    self.a4 = AttentionBlock(base_ch * 8)
    self.d4 = Downsample(base_ch * 8)

    self.mid1 = ResBlock(base_ch * 8, base_ch * 16, emb_dim)
    self.mida = AttentionBlock(base_ch * 16, heads=8)
    self.mid2 = ResBlock(base_ch * 16, base_ch * 16, emb_dim)

    self.u4 = Upsample(base_ch * 16, base_ch * 8)
    self.de4 = ResBlock(base_ch * 16, base_ch * 8, emb_dim)
    self.ua4 = AttentionBlock(base_ch * 8)
    self.u3 = Upsample(base_ch * 8, base_ch * 4)
    self.de3 = ResBlock(base_ch * 8, base_ch * 4, emb_dim)
    self.ua3 = AttentionBlock(base_ch * 4)
    self.u2 = Upsample(base_ch * 4, base_ch * 2)
    self.de2 = ResBlock(base_ch * 4, base_ch * 2, emb_dim)
    self.ua2 = AttentionBlock(base_ch * 2)
    self.u1 = Upsample(base_ch * 2, base_ch)
    self.de1 = ResBlock(base_ch * 2, base_ch, emb_dim)
    self.out = nn.Conv2d(base_ch, 3, 1)

  def embed(self, t: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
    emb = self.time_mlp(timestep_embedding(t, self.emb_dim))
    if y is None:
      y = torch.full((t.shape[0],), 10, device=t.device, dtype=torch.long)
    elif self.training and self.class_drop > 0:
      y = torch.where(torch.rand_like(y.float()) < self.class_drop, torch.full_like(y, 10), y)
    return F.silu(emb + self.class_emb(y))

  def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
    emb = self.embed(t, y)
    s1 = self.e1(x, emb)
    x = self.d1(s1)
    s2 = self.a2(self.e2(x, emb))
    x = self.d2(s2)
    s3 = self.a3(self.e3(x, emb))
    x = self.d3(s3)
    s4 = self.a4(self.e4(x, emb))
    x = self.d4(s4)
    x = self.mid2(self.mida(self.mid1(x, emb)), emb)
    x = self.ua4(self.de4(torch.cat([self.u4(x), s4], dim=1), emb))
    x = self.ua3(self.de3(torch.cat([self.u3(x), s3], dim=1), emb))
    x = self.ua2(self.de2(torch.cat([self.u2(x), s2], dim=1), emb))
    x = self.de1(torch.cat([self.u1(x), s1], dim=1), emb)
    return self.out(x)


def make_loader(batch: int, workers: int) -> DataLoader:
  tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
  return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)


def cycle(loader):
  while True:
    for batch in loader:
      yield batch


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
def sample(model: ClassRF, labels: torch.Tensor, steps: int, method: str, guidance: float, device: torch.device) -> torch.Tensor:
  model.eval()
  x = torch.randn(labels.shape[0], 3, 32, 32, device=device)
  eps, dt = 1e-3, (1 - 1e-3) / steps
  for i in range(steps):
    ti = eps + i * dt
    t = torch.full((labels.shape[0],), ti, device=device)
    vc = model(x, t, labels)
    if guidance != 1:
      vu = model(x, t, None)
      vc = vu + guidance * (vc - vu)
    x_euler = x + dt * vc
    if method == "heun" and i < steps - 1:
      tn = torch.full((labels.shape[0],), ti + dt, device=device)
      vn = model(x_euler, tn, labels)
      if guidance != 1:
        vun = model(x_euler, tn, None)
        vn = vun + guidance * (vn - vun)
      x = x + dt * 0.5 * (vc + vn)
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


def save_checkpoint(path: str, model: nn.Module, ema: dict[str, torch.Tensor], opt: torch.optim.Optimizer, args: argparse.Namespace, step: int, keep_milestones: bool = False) -> None:
  payload = {"model": model.state_dict(), "ema": ema, "opt": opt.state_dict(), "args": vars(args), "step": step}
  torch.save(payload, path)
  if keep_milestones:
    ckpt_path = Path(path)
    milestone = ckpt_path.with_name(f"{ckpt_path.stem}_step{step:06d}{ckpt_path.suffix}")
    shutil.copy2(path, milestone)
    print(f"saved {milestone}", flush=True)


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("--steps", type=int, default=50000)
  p.add_argument("--batch", type=int, default=512)
  p.add_argument("--base-ch", type=int, default=96)
  p.add_argument("--emb-dim", type=int, default=384)
  p.add_argument("--lr", type=float, default=1e-4)
  p.add_argument("--workers", type=int, default=8)
  p.add_argument("--amp", action="store_true")
  p.add_argument("--compile", action="store_true")
  p.add_argument("--ema", type=float, default=0.999)
  p.add_argument("--ema-every", type=int, default=10)
  p.add_argument("--class-drop", type=float, default=0.1)
  p.add_argument("--ckpt", default="torch_cifar_rf.pt")
  p.add_argument("--resume", action="store_true")
  p.add_argument("--sample-only", action="store_true")
  p.add_argument("--raw", action="store_true")
  p.add_argument("--out", default="torch_cifar_rf.png")
  p.add_argument("--sample-steps", type=int, default=100)
  p.add_argument("--sampler", choices=("euler", "heun"), default="heun")
  p.add_argument("--guidance", type=float, default=1.5)
  p.add_argument("--sample-bs", type=int, default=16)
  p.add_argument("--class-id", type=int, default=-1, help="-1 makes a balanced grid, 0-9 repeats one CIFAR class")
  p.add_argument("--log-every", type=int, default=500)
  p.add_argument("--save-every", type=int, default=5000)
  p.add_argument("--keep-milestones", action="store_true")
  p.add_argument("--seed", type=int, default=1337)
  return p.parse_args()


def main() -> None:
  args = parse_args()
  torch.manual_seed(args.seed)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"device={device} name={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} base={args.base_ch} batch={args.batch} amp={args.amp} compile={args.compile}", flush=True)

  model = ClassRF(args.base_ch, args.emb_dim, args.class_drop).to(device)
  opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
  ema = make_ema(model)
  if args.resume or args.sample_only:
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    if args.resume and "opt" in ckpt:
      opt.load_state_dict(ckpt["opt"])
    if "ema" in ckpt and not args.raw:
      ema = ckpt["ema"]
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
      imgs, labels = next(batches)
      imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
      noise = torch.randn_like(imgs)
      t = torch.rand(args.batch, device=device)
      xt = torch.lerp(noise, imgs, t[:, None, None, None])
      opt.zero_grad(set_to_none=True)
      with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
        loss = F.mse_loss(run_model(xt, t, labels), imgs - noise)
      scaler.scale(loss).backward()
      scaler.step(opt)
      scaler.update()
      if step % args.ema_every == 0:
        update_ema(model, ema, args.ema ** args.ema_every)

      loss_value = float(loss.detach())
      loss_ema = loss_value if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_value
      if step == 1 or step % args.log_every == 0:
        if device.type == "cuda":
          torch.cuda.synchronize()
        now = time.perf_counter()
        window = step if step == 1 else args.log_every
        mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
        print(f"step {step:6d}/{args.steps} loss={loss_value:.4f} ema={loss_ema:.4f} avg={(now - start) / step:.4f}s/step win={(now - last) / window:.4f}s/step mem={mem:.2f}GB", flush=True)
        last = now
      if args.save_every and step % args.save_every == 0:
        save_checkpoint(args.ckpt, model, ema, opt, args, step, args.keep_milestones)
    save_checkpoint(args.ckpt, model, ema, opt, args, args.steps, args.keep_milestones)
    print(f"saved {args.ckpt}", flush=True)

  if args.class_id >= 0:
    labels = torch.full((args.sample_bs,), args.class_id, device=device, dtype=torch.long)
  else:
    labels = torch.arange(args.sample_bs, device=device) % 10
  samples = sample(model, labels, args.sample_steps, args.sampler, args.guidance, device)
  save_grid(samples, args.out, cols=int(math.sqrt(args.sample_bs)))


if __name__ == "__main__":
  main()
