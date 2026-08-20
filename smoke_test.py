import torch
from rvi_transformer import RVITransformer, RVIConfig

cfg = RVIConfig(
    d_model=64, n_heads=4, n_layers=2, d_ff=128,
    max_seq_len=64, rvi_rank=8, ctms_recurse=2
)
m = RVITransformer(cfg)
x = torch.randint(0, 256, (2, 32))
y = torch.randint(0, 256, (2, 32))
logits, loss, diagnostics = m(x, y, return_diagnostics=True)
assert logits.shape == (2, 32, 256)
assert torch.isfinite(loss)
assert loss.item() > 0
loss.backward()
print("smoke test passed", float(loss), diagnostics[-1])
