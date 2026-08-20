import torch
from rvi_multimatrix import RVIConfig, RVIMultiMatrixTransformer, ByteTokenizer

cfg = RVIConfig(
    d_model=128,
    n_heads=4,
    n_layers=3,
    d_ff=512,
    max_seq_len=128,
    rvi_rank=16,
    matrix_dim=32,
)

model = RVIMultiMatrixTransformer(cfg)
tok = ByteTokenizer()

text = "Vague ideas must be confronted with clear images."
x = torch.tensor([tok.encode(text)], dtype=torch.long)

logits, loss, diagnostics, memory = model(
    x,
    x,
    return_diagnostics=True,
    return_memory=True,
)

print("logits:", tuple(logits.shape))
print("memory:", tuple(memory.matrix_summary.shape))
print("matrix names:", cfg.matrix_names)
print("last diagnostics:")
for k, v in diagnostics[-1].items():
    if torch.is_tensor(v) and v.numel() == 1:
        print(f"  {k}: {float(v):.5f}")
