"""
Minimal A/B harness:
A = ordinary transformer (RVI disabled)
B = RVI+CTMS transformer

Train each with the same data/steps/seed and compare held-out loss.
"""
import copy, json, argparse, torch
from pathlib import Path
from rvi_transformer import RVITransformer, RVIConfig
from rvi_transformer.tokenizer import ByteTokenizer


def batch(data, bs, sl, device):
    ix = torch.randint(0, len(data)-sl-1, (bs,))
    x = torch.stack([data[i:i+sl] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+sl+1] for i in ix]).to(device)
    return x, y


def run(cfg, train_data, val_data, steps, bs, sl, device, seed):
    torch.manual_seed(seed)
    model = RVITransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(steps):
        x, y = batch(train_data, bs, sl, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    vals = []
    with torch.no_grad():
        for _ in range(20):
            x, y = batch(val_data, bs, sl, device)
            _, loss = model(x, y)
            vals.append(loss.item())
    return sum(vals)/len(vals), sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = ByteTokenizer()
    data = torch.tensor(tok.encode(Path(args.text).read_text(encoding="utf-8")), dtype=torch.long)
    cut = int(len(data)*0.9)
    tr, va = data[:cut], data[cut:]

    base_dict = json.loads(Path(args.config).read_text())
    a_dict = copy.deepcopy(base_dict); a_dict["rvi_enabled"] = False
    b_dict = copy.deepcopy(base_dict); b_dict["rvi_enabled"] = True

    a_loss, a_n = run(RVIConfig(**a_dict), tr, va, args.steps, args.batch_size, args.seq_len, args.device, args.seed)
    b_loss, b_n = run(RVIConfig(**b_dict), tr, va, args.steps, args.batch_size, args.seq_len, args.device, args.seed)

    print(f"BASELINE val_loss={a_loss:.4f} params={a_n:,}")
    print(f"RVI_CTMS val_loss={b_loss:.4f} params={b_n:,}")
    print(f"delta={b_loss-a_loss:+.4f} (negative favors RVI_CTMS)")


if __name__ == "__main__":
    main()
