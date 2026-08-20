from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DACConfig:
    d_model: int = 256
    sample_rate: int = 48000
    hop_length: int = 240
    hidden: int = 256
    harmonics: int = 32
    min_hz: float = 40.0
    max_hz: float = 12000.0


class NeuralDAC(nn.Module):
    """
    Hidden frames -> interpretable synthesis controls -> normalized float PCM.

    This is a software signal renderer. It is not a direct neural-stimulation driver.
    """
    def __init__(self, cfg: DACConfig):
        super().__init__()
        self.cfg = cfg
        self.control = nn.Sequential(
            nn.RMSNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, 3 + cfg.harmonics),
        )

    def forward(self, frames, initial_phase: Optional[torch.Tensor] = None):
        B, T, _ = frames.shape
        c = self.control(frames)

        pitch_u = torch.sigmoid(c[..., 0])
        amp = torch.sigmoid(c[..., 1])
        noise_amp = 0.05 * torch.sigmoid(c[..., 2])
        harmonic_amp = torch.softmax(c[..., 3:], dim=-1)

        ratio = torch.tensor(
            self.cfg.max_hz / self.cfg.min_hz,
            device=frames.device,
            dtype=frames.dtype,
        )
        f0 = self.cfg.min_hz * torch.pow(ratio, pitch_u)

        n = T * self.cfg.hop_length

        def up(x):
            return F.interpolate(
                x.transpose(1, 2),
                size=n,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        f = up(f0[..., None]).squeeze(-1)
        a = up(amp[..., None]).squeeze(-1)
        noise = up(noise_amp[..., None]).squeeze(-1)
        h = up(harmonic_amp)

        phase = torch.cumsum(2 * math.pi * f / self.cfg.sample_rate, dim=1)
        if initial_phase is not None:
            phase = phase + initial_phase[:, None]

        k = torch.arange(
            1,
            self.cfg.harmonics + 1,
            device=frames.device,
            dtype=frames.dtype,
        )
        waves = torch.sin(phase[..., None] * k)
        alias = (f[..., None] * k < self.cfg.sample_rate / 2).to(frames.dtype)
        pcm = torch.tanh(a * (waves * h * alias).sum(-1) + torch.randn_like(f) * noise)

        return pcm, {
            "f0_hz": f0,
            "amplitude": amp,
            "harmonic_amplitude": harmonic_amp,
            "noise_amplitude": noise_amp,
            "terminal_phase": phase[:, -1],
        }

    @staticmethod
    def pcm16(pcm):
        return (pcm.clamp(-1, 1) * 32767).round().to(torch.int16)
