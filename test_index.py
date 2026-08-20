import tempfile
import torch

from rvi_multimatrix import MultiMatrixMeaningIndex


idx = MultiMatrixMeaningIndex(
    ["semantic", "temporal", "contradiction"],
    matrix_dim=4,
)

idx.add(
    "a",
    {
        "semantic": torch.tensor([1., 0., 0., 0.]),
        "temporal": torch.tensor([0., 1., 0., 0.]),
        "contradiction": torch.tensor([1., 0., 0., 0.]),
    },
    {"name": "A"},
)
idx.add(
    "b",
    {
        "semantic": torch.tensor([0., 1., 0., 0.]),
        "temporal": torch.tensor([1., 0., 0., 0.]),
        "contradiction": torch.tensor([-1., 0., 0., 0.]),
    },
    {"name": "B"},
)

hits = idx.search(
    {"semantic": torch.tensor([1., 0., 0., 0.])},
    top_k=1,
)
assert hits[0].record_id == "a"

hits2 = idx.search(
    {"contradiction": torch.tensor([1., 0., 0., 0.])},
    top_k=1,
    contradiction_flip=True,
)
assert hits2[0].record_id == "b"

with tempfile.NamedTemporaryFile(suffix=".pt") as f:
    idx.save(f.name)
    loaded = MultiMatrixMeaningIndex.load(f.name)
    assert len(loaded) == 2

print("INDEX PASS")
