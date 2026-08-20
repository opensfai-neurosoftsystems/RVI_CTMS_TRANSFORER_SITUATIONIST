from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import json
import random

import torch

from .tokenizer import ByteTokenizer
from .model import RVIConfig


@dataclass
class CorpusRecord:
    record_id: str
    text: str
    matrices: List[str]
    operator: Optional[str]
    metadata: Dict


def load_jsonl(path) -> List[CorpusRecord]:
    records = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        records.append(CorpusRecord(
            record_id=str(obj.get("id", f"line-{line_no}")),
            text=str(obj["text"]),
            matrices=list(obj.get("matrices", [])),
            operator=obj.get("operator"),
            metadata=dict(obj.get("metadata", {})),
        ))
    return records


def sample_annotated_batch(
    records: List[CorpusRecord],
    cfg: RVIConfig,
    batch_size: int,
    seq_len: int,
    device,
    tokenizer=None,
):
    tok = tokenizer or ByteTokenizer()
    eligible = []
    encoded = []

    for rec in records:
        ids = tok.encode(rec.text)
        if len(ids) >= seq_len + 1:
            eligible.append(rec)
            encoded.append(ids)

    if not eligible:
        raise ValueError("no annotated records are long enough for seq_len")

    xs, ys, matrix_labels, operator_labels = [], [], [], []
    matrix_lookup = {name: i for i, name in enumerate(cfg.matrix_names)}
    operator_lookup = {name: i for i, name in enumerate(cfg.operator_names)}

    for _ in range(batch_size):
        j = random.randrange(len(eligible))
        rec = eligible[j]
        ids = encoded[j]
        start = random.randrange(0, len(ids) - seq_len)

        x = torch.tensor(ids[start:start + seq_len], dtype=torch.long)
        y = torch.tensor(ids[start + 1:start + seq_len + 1], dtype=torch.long)
        xs.append(x)
        ys.append(y)

        ml = torch.zeros(cfg.n_matrices, dtype=torch.float32)
        for name in rec.matrices:
            if name in matrix_lookup:
                ml[matrix_lookup[name]] = 1.0
        matrix_labels.append(ml)

        op = operator_lookup.get(rec.operator, 0)
        operator_labels.append(op)

    return (
        torch.stack(xs).to(device),
        torch.stack(ys).to(device),
        torch.stack(matrix_labels).to(device),
        torch.tensor(operator_labels, dtype=torch.long, device=device),
    )


def plain_text_tensor(path, tokenizer=None):
    tok = tokenizer or ByteTokenizer()
    raw = Path(path).read_text(encoding="utf-8")
    return torch.tensor(tok.encode(raw), dtype=torch.long)


def sample_plain_batch(data, batch_size, seq_len, device):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix]).to(device)
    return x, y
