"""One-shot handwritten text-line diffusion (DiffBrush, Dai et al., ICCV 2025).

The network code in this folder is vendored from https://github.com/dailenson/DiffBrush
(MIT, see LICENSE-DiffBrush). This module wraps it for CPU inference:
weights are fetched on first use into the ductus cache, never shipped in the repo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LETTERS = " _!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
LINE_H = 64          # model works on 64 px high lines
LINE_W = 1024        # fixed canvas width the model was trained on
CKPT_GDRIVE_ID = "1EWzBmLtnQ42cTf3k_CYQ-nF3RXCb35I6"   # pretrained DiffBrush (IAM)
VAE_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # only the `vae/` folder is used


def cache_dir() -> Path:
    d = Path(os.environ.get("DUCTUS_CACHE", Path.home() / ".cache" / "ductus"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_weights(verbose: bool = True) -> tuple[Path, Path]:
    """Download the DiffBrush checkpoint and the SD-1.5 VAE if they are missing."""
    d = cache_dir()
    ckpt = d / "DiffBrush-ckpt.pt"
    if not ckpt.exists():
        import gdown  # small dependency, only needed once
        gdown.download(id=CKPT_GDRIVE_ID, output=str(ckpt), quiet=not verbose)
    vae_dir = d / "sd15"
    if not (vae_dir / "vae" / "diffusion_pytorch_model.safetensors").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(VAE_REPO, allow_patterns=["vae/config.json", "vae/diffusion_pytorch_model.safetensors"],
                          local_dir=str(vae_dir))
    return ckpt, vae_dir


def supported(ch: str) -> bool:
    return ch in LETTERS


@dataclass
class DiffBrushEngine:
    steps: int = 25        # DDIM steps; 25 is close to 50 in quality and twice as fast
    eta: float = 0.0       # >0 adds sampling noise (messier, more varied)
    threads: int = 0       # 0 = all cores

    def __post_init__(self):
        import torch
        from ... import speed
        if self.threads:
            torch.set_num_threads(self.threads)
        else:
            speed.setup()
        self.device = speed.device()
        from .unet import UNetModel
        from diffusers import AutoencoderKL
        ckpt, vae_dir = fetch_weights()
        self.torch = torch
        self.unet = UNetModel(in_channels=4, model_channels=512, out_channels=4, num_res_blocks=1,
                              attention_resolutions=(1, 1), channel_mult=(1, 1), num_heads=4,
                              context_dim=512, nb_classes=496)
        self.unet.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.unet.eval().to(self.device)
        self.vae = AutoencoderKL.from_pretrained(str(vae_dir), subfolder="vae").eval().to(self.device)
        g = np.load(Path(__file__).with_name("glyphs.npz"))
        self.glyphs = torch.stack([torch.from_numpy(g[str(ord(c))].astype(np.float32)) for c in LETTERS])
        self.index = {c: i for i, c in enumerate(LETTERS)}
        # linear noise schedule used in training
        beta = torch.linspace(1e-4, 0.02, 1000)
        self.alpha_hat = torch.cumprod(1. - beta, dim=0)
        self.beta = beta

    # -- inputs -----------------------------------------------------------------
    def style_tensor(self, style_img: np.ndarray):
        """Grayscale line image (ink dark on white, any size) -> [1,1,64,W<=1024] in 0..1."""
        from PIL import Image
        im = Image.fromarray(style_img.astype(np.uint8))
        w = max(8, int(round(im.width * LINE_H / im.height)))
        im = im.resize((min(w, LINE_W), LINE_H), Image.BICUBIC)
        a = np.asarray(im, dtype=np.float32) / 255.0
        return self.torch.from_numpy(a)[None, None]

    def content_tensor(self, text: str):
        ids = [self.index[c] for c in text]
        return (1.0 - self.glyphs[ids])[None]

    # -- sampling ---------------------------------------------------------------
    def generate(self, texts: list[str], style_img: np.ndarray, seed: int = 0) -> list[np.ndarray]:
        """Generate one 64x1024 grayscale line per text (all in one batch when lengths allow).
        Texts in one call are padded to the longest with spaces; returns uint8 arrays."""
        torch = self.torch
        for t in texts:
            bad = [c for c in t if c not in self.index]
            if bad:
                raise ValueError(f"unsupported characters for diffusion engine: {bad!r}")
        n = len(texts)
        L = max(len(t) for t in texts)
        dev = self.device
        content = torch.cat([self.content_tensor(t.ljust(L)) for t in texts]).to(dev)
        style = self.style_tensor(style_img).repeat(n, 1, 1, 1).to(dev)
        gen = torch.Generator().manual_seed(int(seed))     # noise drawn on the CPU: same seed, same page
        x = torch.randn((n, 4, LINE_H // 8, LINE_W // 8), generator=gen).to(dev)
        times = list(reversed(torch.linspace(-1, 999, steps=self.steps + 1).int().tolist()))
        with torch.inference_mode():
            for t, t_next in zip(times[:-1], times[1:]):
                tt = torch.full((n,), t, dtype=torch.long, device=dev)
                eps = self.unet(x, tt, style, content)
                ah = self.alpha_hat[t]
                x0 = (x - (1 - ah).sqrt() * eps) / ah.sqrt()
                if t_next < 0:
                    x = x0
                    break
                ahn = self.alpha_hat[t_next]
                sigma = self.eta * ((self.beta[t] * (1 - ahn) / (1 - ah)).sqrt())
                c = (1 - ahn - sigma ** 2).sqrt()
                x = x0 * ahn.sqrt() + c * eps + sigma * torch.randn(x.shape, generator=gen).to(dev)
            img = self.vae.decode(x / 0.18215).sample
        img = ((img / 2 + 0.5).clamp(0, 1).mean(1) * 255).round().byte().cpu().numpy()
        return [img[i] for i in range(n)]
