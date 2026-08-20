"""
Build an external multi-matrix meaning index from JSONL records.

This uses the transformer's compressed matrix memory as the vector address
for every matrix. It is a prototype indexer, not a replacement for a trained
retrieval encoder.
"""

import argparse
import json
from pathlib import Path

import torch

from rvi_multimatrix import (
    RVIMultiMatrixTransformer,
    RVIConfig,
    ByteTokenizer,
    MultiMatrixMeaningIndex,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", default="meaning_index.pt")
    ap.add_argument("--max-bytes", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = RVIConfig(**ckpt["config"])
    model = RVIMultiMatrixTransformer(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = ByteTokenizer()
    index = MultiMatrixMeaningIndex(cfg.matrix_names, cfg.matrix_dim)

    with torch.no_grad():
        for n, line in enumerate(Path(args.jsonl).read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            obj = json.loads(line)
            ids = tok.encode(str(obj["text"]))[:args.max_bytes]
            if not ids:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=args.device)
            mem = model.encode_matrix_memory(x)
            summary = mem.matrix_summary[0].cpu()

            vectors = {
                name: summary[i]
                for i, name in enumerate(cfg.matrix_names)
            }
            index.add(
                str(obj.get("id", f"record-{n}")),
                vectors,
                metadata=dict(obj.get("metadata", {})),
            )

    index.save(args.out)
    print(f"saved {len(index)} records to {args.out}")


if __name__ == "__main__":
    main()
