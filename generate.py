import argparse, torch
from rvi_transformer import RVITransformer, RVIConfig
from rvi_transformer.tokenizer import ByteTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoint.pt")
    ap.add_argument("--prompt", default="Recursive vector interference")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg = RVIConfig(**ckpt["config"])
    model = RVITransformer(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    tok = ByteTokenizer()
    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=args.device)
    out = model.generate(ids, max_new_tokens=args.tokens, temperature=args.temperature)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
