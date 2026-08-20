from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import torch
import torch.nn.functional as F


@dataclass
class IndexHit:
    record_id: str
    score: float
    per_matrix: Dict[str, float]
    metadata: Dict[str, Any]


class MultiMatrixMeaningIndex:
    """
    External multi-matrix address space.

    Each record can have an independent vector for each matrix:
      semantic, temporal, spatial, causal, embodied, media, lineage,
      contradiction, operator, provenance, ...

    Retrieval can weight matrices differently per query.
    """

    def __init__(self, matrix_names: List[str], matrix_dim: int):
        self.matrix_names = list(matrix_names)
        self.matrix_dim = int(matrix_dim)
        self.ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.vectors: Dict[str, List[torch.Tensor]] = {
            name: [] for name in self.matrix_names
        }

    def add(
        self,
        record_id: str,
        vectors: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.ids.append(str(record_id))
        self.metadata.append(dict(metadata or {}))

        for name in self.matrix_names:
            if name not in vectors:
                v = torch.zeros(self.matrix_dim)
            else:
                v = vectors[name].detach().float().flatten().cpu()
                if v.numel() != self.matrix_dim:
                    raise ValueError(
                        f"{name}: expected vector dim {self.matrix_dim}, got {v.numel()}"
                    )
            self.vectors[name].append(v)

    def __len__(self):
        return len(self.ids)

    def search(
        self,
        query_vectors: Dict[str, torch.Tensor],
        matrix_weights: Optional[Dict[str, float]] = None,
        top_k: int = 5,
        contradiction_flip: bool = False,
    ) -> List[IndexHit]:
        if not self.ids:
            return []

        weights = matrix_weights or {k: 1.0 for k in query_vectors}
        total = torch.zeros(len(self.ids))
        per_matrix_scores: Dict[str, torch.Tensor] = {}

        for name, q in query_vectors.items():
            if name not in self.vectors or not self.vectors[name]:
                continue

            bank = torch.stack(self.vectors[name], dim=0)  # [N,D]
            q = q.detach().float().flatten().cpu()
            if q.numel() != self.matrix_dim:
                raise ValueError(
                    f"{name}: expected query dim {self.matrix_dim}, got {q.numel()}"
                )

            s = F.cosine_similarity(bank, q[None, :], dim=-1)

            # Optional contradiction retrieval:
            # low cosine in the contradiction matrix is promoted.
            if contradiction_flip and name == "contradiction":
                s = -s

            per_matrix_scores[name] = s
            total += float(weights.get(name, 1.0)) * s

        norm = sum(abs(float(v)) for v in weights.values()) or 1.0
        total = total / norm

        vals, idx = torch.topk(total, k=min(top_k, len(self.ids)))
        hits = []
        for score, i in zip(vals.tolist(), idx.tolist()):
            hits.append(IndexHit(
                record_id=self.ids[i],
                score=float(score),
                per_matrix={
                    name: float(scores[i])
                    for name, scores in per_matrix_scores.items()
                },
                metadata=self.metadata[i],
            ))
        return hits

    def save(self, path):
        payload = {
            "matrix_names": self.matrix_names,
            "matrix_dim": self.matrix_dim,
            "ids": self.ids,
            "metadata": self.metadata,
            "vectors": {
                k: torch.stack(v, dim=0) if v else torch.empty(0, self.matrix_dim)
                for k, v in self.vectors.items()
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu")
        obj = cls(payload["matrix_names"], payload["matrix_dim"])
        obj.ids = list(payload["ids"])
        obj.metadata = list(payload["metadata"])
        obj.vectors = {
            k: [row.clone() for row in tensor]
            for k, tensor in payload["vectors"].items()
        }
        return obj
