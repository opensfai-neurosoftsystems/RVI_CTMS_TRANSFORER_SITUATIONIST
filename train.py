import argparse, json
from pathlib import Path
import torch
from rvi_transformer import RVITransformer, RVIConfig
from rvi_transformer.tokenizer import ByteTokenizer


def make_batch(data, batch_size, seq_len, device):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--config", type=str, default="config.json")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", type=str, default="checkpoint.pt")
    args = ap.parse_args()

    cfg = RVIConfig(**json.loads(Path(args.config).read_text()))
    tok = ByteTokenizer()
    raw = Path(args.text).read_text(encoding="utf-8")
    data = torch.tensor(tok.encode(raw), dtype=torch.long)

    if len(data) < args.seq_len + 2:
        raise ValueError("training text is shorter than seq_len")

    model = RVITransformer(cfg).to(args.device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = make_batch(data, args.batch_size, args.seq_len, args.device)
        _, loss, diag = model(x, y, return_diagnostics=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % 50 == 0:
            d = diag[-1] if diag and diag[-1] else {}
            extra = " ".join(f"{k}={float(v):.4f}" for k, v in d.items())
            print(f"step={step} loss={loss.item():.4f} {extra}")

    state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save({"config": cfg.__dict__, "model": state}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
