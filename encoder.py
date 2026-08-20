from typing import Optional
import torch
import torch.nn as nn

from .model import (
    RVIConfig,
    Block,
    RelationalMemory,
)


class RVIMultiMatrixEncoder(nn.Module):
    """
    Bidirectional encoder using the same multi-matrix / RVI / CTMS machinery.

    Returns:
      sequence [B,T,D]
      pooled   [B,D]
      memory   RelationalMemory | None
      diagnostics (optional)
    """
    def __init__(self, cfg: RVIConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        memory: Optional[RelationalMemory] = None,
        return_diagnostics=False,
    ):
        x = self.tok(input_ids)
        field = None
        current_memory = memory
        diagnostics = []

        for block in self.blocks:
            x, field, current_memory, diag, aux = block(
                x,
                previous_field=field,
                memory=current_memory,
                causal=False,
            )
            diagnostics.append(diag)

        x = self.norm(x)

        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            m = attention_mask.to(x.dtype).unsqueeze(-1)
            pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

        if return_diagnostics:
            return x, pooled, current_memory, diagnostics
        return x, pooled, current_memory


RVIEncoder = RVIMultiMatrixEncoder
