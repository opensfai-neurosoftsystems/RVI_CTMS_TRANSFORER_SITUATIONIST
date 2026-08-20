from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RVIConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.0

    # RVI
    rvi_enabled: bool = True
    rvi_rank: int = 32
    rvi_alpha: float = 0.35      # interference contribution to attention logits
    rvi_beta: float = 0.15       # persistent relational-state feedback
    ctms_recurse: int = 2        # recursive refinement passes

    def validate(self):
        assert self.d_model % self.n_heads == 0
        assert self.rvi_rank > 0
        assert self.ctms_recurse >= 1


@dataclass
class RelationalState:
    """
    Explicit second-order state.
    phase: [B,H,T,R]
    amp:   [B,H,T,R]
    field: [B,H,T,T]
    """
    phase: torch.Tensor
    amp: torch.Tensor
    field: torch.Tensor


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x):
        # x: [B,H,T,D]
        T = x.size(-2)
        cos = self.cos[:T][None, None, :, :]
        sin = self.sin[:T][None, None, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.stack((y1, y2), dim=-1).flatten(-2)


class RVIRelationalKernel(nn.Module):
    """
    Converts hidden states into amplitude/phase channels, then computes
    a phase-sensitive pairwise interference field:

        I_ij = sum_r a_i,r a_j,r cos(phi_i,r - phi_j,r - tau_r)

    This is an explicit, testable extension beyond ordinary dot-product attention.
    """
    def __init__(self, cfg: RVIConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.r = cfg.rvi_rank
        self.amp_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.rvi_rank, bias=False)
        self.phase_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.rvi_rank, bias=False)
        self.delay = nn.Parameter(torch.zeros(cfg.n_heads, cfg.rvi_rank))
        self.band_weight = nn.Parameter(torch.ones(cfg.n_heads, cfg.rvi_rank) / cfg.rvi_rank)
        self.state_gate = nn.Parameter(torch.tensor(cfg.rvi_beta))

    def forward(
        self,
        x: torch.Tensor,
        previous: Optional[RelationalState] = None
    ) -> RelationalState:
        B, T, _ = x.shape
        amp = F.softplus(self.amp_proj(x)).view(B, T, self.h, self.r).transpose(1, 2)
        phase = math.pi * torch.tanh(
            self.phase_proj(x).view(B, T, self.h, self.r).transpose(1, 2)
        )

        # [B,H,T,1,R] - [B,H,1,T,R] - [1,H,1,1,R]
        dphi = (
            phase.unsqueeze(3)
            - phase.unsqueeze(2)
            - self.delay[None, :, None, None, :]
        )
        pair_amp = amp.unsqueeze(3) * amp.unsqueeze(2)
        w = self.band_weight[None, :, None, None, :]
        field = (pair_amp * torch.cos(dphi) * w).sum(dim=-1)

        # Normalize each query row to keep scale controlled.
        field = field / (field.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)

        if previous is not None and previous.field.shape == field.shape:
            gate = torch.sigmoid(self.state_gate)
            field = (1.0 - gate) * field + gate * previous.field

        return RelationalState(phase=phase, amp=amp, field=field)


class CTMSRefiner(nn.Module):
    """
    A compact differentiable analogue of:
    SEED -> BLOOM -> FORK -> RECURSE -> REDCHECK -> FLATTEN.

    It recursively refines the relational field while enforcing causal masking.
    """
    def __init__(self, cfg: RVIConfig):
        super().__init__()
        self.steps = cfg.ctms_recurse
        self.mix = nn.Parameter(torch.zeros(self.steps, cfg.n_heads))
        self.contradiction_gate = nn.Parameter(torch.zeros(self.steps, cfg.n_heads))

    def forward(self, field: torch.Tensor, causal_mask: torch.Tensor):
        # field: [B,H,T,T]
        x = field
        for s in range(self.steps):
            # RECURSE: relation-of-relations composition.
            probs = torch.softmax(x.masked_fill(~causal_mask, float("-inf")), dim=-1)
            composed = probs @ probs

            # REDCHECK: penalize unstable disagreement between direct and composed relations.
            disagreement = torch.tanh(x - composed)
            mix = torch.sigmoid(self.mix[s])[None, :, None, None]
            red = torch.sigmoid(self.contradiction_gate[s])[None, :, None, None]
            x = (1.0 - mix) * x + mix * composed - red * disagreement

            # Keep invalid future edges inert.
            x = x.masked_fill(~causal_mask, 0.0)
        return x


class RVISelfAttention(nn.Module):
    def __init__(self, cfg: RVIConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(self.dh, cfg.max_seq_len)
        self.rvi = RVIRelationalKernel(cfg) if cfg.rvi_enabled else None
        self.ctms = CTMSRefiner(cfg) if cfg.rvi_enabled else None

    def forward(
        self,
        x: torch.Tensor,
        previous_state: Optional[RelationalState] = None
    ) -> Tuple[torch.Tensor, Optional[RelationalState], Dict[str, torch.Tensor]]:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def heads(t):
            return t.view(B, T, self.h, self.dh).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        q, k = self.rope(q), self.rope(k)

        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)

        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        causal4 = causal[None, None, :, :]

        state = None
        diagnostics = {}

        if self.rvi is not None:
            state = self.rvi(x, previous_state)
            refined = self.ctms(state.field, causal4)
            logits = logits + self.cfg.rvi_alpha * refined
            state = RelationalState(state.phase, state.amp, refined)

            diagnostics = {
                "rvi_field_mean": refined.mean().detach(),
                "rvi_field_std": refined.std().detach(),
                "phase_lock": torch.cos(
                    state.phase.unsqueeze(3) - state.phase.unsqueeze(2)
                ).mean().detach(),
            }

        logits = logits.masked_fill(~causal4, float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = F.dropout(attn, p=self.cfg.dropout, training=self.training)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y), state, diagnostics


class MLP(nn.Module):
    def __init__(self, cfg: RVIConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg: RVIConfig):
        super().__init__()
        self.n1 = nn.RMSNorm(cfg.d_model)
        self.attn = RVISelfAttention(cfg)
        self.n2 = nn.RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, previous_state=None):
        a, state, diag = self.attn(self.n1(x), previous_state)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x, state, diag


class RVITransformer(nn.Module):
    def __init__(self, cfg: RVIConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight

    def forward(self, input_ids, targets=None, return_diagnostics=False):
        B, T = input_ids.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} > max_seq_len {self.cfg.max_seq_len}")

        x = self.tok(input_ids)
        states: List[Optional[RelationalState]] = []
        diagnostics = []
        relational_state: Optional[RelationalState] = None

        # The interference field is explicitly persistent across depth.
        # Each layer receives the previous layer's relational geometry,
        # updates it, and passes the revised state forward.
        for block in self.blocks:
            x, relational_state, diag = block(x, relational_state)
            states.append(relational_state)
            diagnostics.append(diag)

        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        if return_diagnostics:
            return logits, loss, diagnostics
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, temperature=0.8, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            x = input_ids[:, -self.cfg.max_seq_len:]
            logits, _ = self(x)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, nxt], dim=1)
        return input_ids
