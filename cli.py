from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import torch

from .version import __version__
from .model import (
    RVIConfig,
    RVIMultiMatrixTransformer,
    RelationalMemory,
    DEFAULT_MATRICES,
    CTMS_OPERATORS,
)
from .tokenizer import ByteTokenizer
from .meaning_index import MultiMatrixMeaningIndex
from .dac import NeuralDAC, DACConfig
from .corpus import (
    load_jsonl,
    sample_annotated_batch,
    plain_text_tensor,
    sample_plain_batch,
)


EXIT_OK = 0
EXIT_USAGE = 2
EXIT_DATA = 3
EXIT_MODEL = 4
EXIT_RUNTIME = 5
EXIT_SAFETY = 6

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.json"
DEFAULT_KERNEL = PROJECT_ROOT / "kernel" / "CTMS.md"
DEFAULT_MANUAL = PROJECT_ROOT / "docs" / "CLI_MANUAL.md"


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def load_config(path: str | Path) -> RVIConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    cfg = RVIConfig(**data)
    cfg.validate()
    return cfg


def config_dict(cfg: RVIConfig) -> Dict:
    return dict(cfg.__dict__)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def device_from_arg(name: str):
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return name


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path: str | Path, device: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    ckpt = torch.load(p, map_location=device)
    if "config" not in ckpt or "model" not in ckpt:
        raise ValueError("checkpoint must contain 'config' and 'model'")
    cfg = RVIConfig(**ckpt["config"])
    model = RVIMultiMatrixTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def inspect_forward(model, input_ids, memory=None):
    """
    Run the model block-by-block so the CLI can inspect the final router,
    operator probabilities, field and relational memory without changing
    the public training API.
    """
    x = model.tok(input_ids)
    field = None
    current_memory = memory
    layer_data = []
    for layer_id, block in enumerate(model.blocks):
        x, field, current_memory, diag, aux = block(
            x,
            previous_field=field,
            memory=current_memory,
            causal=True,
        )
        layer_data.append({
            "layer": layer_id,
            "diag": diag,
            "aux": aux,
            "field": field,
            "memory": current_memory,
        })
    x = model.norm(x)
    logits = model.lm_head(x)
    return logits, x, current_memory, layer_data


def tensor_stats(t: Optional[torch.Tensor]):
    if t is None:
        return None
    x = t.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "mean": float(x.mean()),
        "std": float(x.std()) if x.numel() > 1 else 0.0,
        "min": float(x.min()),
        "max": float(x.max()),
        "norm": float(x.norm()),
    }


def print_json(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


# ------------------------------------------------------------------
# Workspace
# ------------------------------------------------------------------

def cmd_init(args):
    root = Path(args.path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for d in ["corpus/raw", "corpus/annotated", "runs", "checkpoints", "indices",
              "exports", "traces", "audio", "configs", "reports"]:
        (root / d).mkdir(parents=True, exist_ok=True)

    cfg_dest = root / "configs" / "config.json"
    if not cfg_dest.exists() or args.force:
        shutil.copy2(DEFAULT_CONFIG, cfg_dest)

    schema = root / "corpus" / "annotated" / "example.jsonl"
    if not schema.exists() or args.force:
        schema.write_text(
            json.dumps({
                "id": "TD-001:000001",
                "text": "The same event can have more than one relational address. " * 8,
                "matrices": ["semantic", "temporal", "contradiction"],
                "operator": "FORK",
                "metadata": {
                    "document_id": "TD-001",
                    "source": "replace-with-provenance",
                    "page": 1
                }
            }, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )

    manifest = root / "workspace.json"
    manifest.write_text(json.dumps({
        "format": "rvi-ctms-workspace",
        "version": 1,
        "created_by": f"rvi-multimatrix {__version__}",
        "kernel_sha256": sha256(DEFAULT_KERNEL) if DEFAULT_KERNEL.exists() else None,
    }, indent=2), encoding="utf-8")

    print(root)
    return EXIT_OK


# ------------------------------------------------------------------
# Doctor / environment
# ------------------------------------------------------------------

def cmd_doctor(args):
    device = device_from_arg(args.device)
    report = {
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device_requested": args.device,
        "device_resolved": device,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "kernel_exists": DEFAULT_KERNEL.exists(),
        "kernel_sha256": sha256(DEFAULT_KERNEL) if DEFAULT_KERNEL.exists() else None,
        "config_exists": Path(args.config).exists(),
        "manual_exists": DEFAULT_MANUAL.exists(),
    }
    ok = True

    try:
        cfg = load_config(args.config)
        report["config_valid"] = True
        report["config"] = {
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "matrices": cfg.matrix_names,
            "operators": cfg.operator_names,
        }
        if args.smoke:
            tiny = copy.deepcopy(cfg)
            tiny.d_model = 64
            tiny.n_heads = 4
            tiny.n_layers = 2
            tiny.d_ff = 128
            tiny.max_seq_len = 64
            tiny.rvi_rank = min(8, tiny.rvi_rank)
            tiny.matrix_dim = min(16, tiny.matrix_dim)
            tiny.matrix_top_k = min(2, tiny.n_matrices)
            model = RVIMultiMatrixTransformer(tiny).to(device)
            x = torch.randint(0, tiny.vocab_size, (1, 16), device=device)
            y = torch.randint(0, tiny.vocab_size, (1, 16), device=device)
            _, loss, diag, mem = model(
                x, y, return_diagnostics=True, return_memory=True
            )
            report["smoke"] = {
                "loss": float(loss.detach().cpu()),
                "memory_shape": list(mem.matrix_summary.shape),
                "diagnostics": {
                    k: float(v.detach().cpu())
                    for k, v in diag[-1].items()
                    if torch.is_tensor(v) and v.numel() == 1
                }
            }
    except Exception as exc:
        ok = False
        report["config_valid"] = False
        report["error"] = repr(exc)

    if args.json:
        print_json(report)
    else:
        for k, v in report.items():
            print(f"{k}: {v}")
        print("STATUS:", "PASS" if ok else "FAIL")

    return EXIT_OK if ok else EXIT_RUNTIME


# ------------------------------------------------------------------
# Kernel
# ------------------------------------------------------------------

def cmd_kernel_show(args):
    p = Path(args.path)
    if not p.exists():
        eprint(f"kernel not found: {p}")
        return EXIT_DATA
    print(p.read_text(encoding="utf-8"))
    return EXIT_OK


def cmd_kernel_hash(args):
    p = Path(args.path)
    if not p.exists():
        eprint(f"kernel not found: {p}")
        return EXIT_DATA
    print(sha256(p))
    return EXIT_OK


def cmd_kernel_validate(args):
    p = Path(args.path)
    if not p.exists():
        eprint(f"kernel not found: {p}")
        return EXIT_DATA

    txt = p.read_text(encoding="utf-8")
    required = [
        "thought-tree", "traverse", "flatten-tree",
        "♢", "◇", "◆", "↺", "⇝", "⊢", "⊨", "⊝", "⋈", "⋔",
        "↑", "⍟", "∞", "§", "⊥", "⊤"
    ]
    missing = [x for x in required if x not in txt]

    warnings = []
    if "chain-of-thought" in txt.lower():
        warnings.append(
            "CTMS.md contains historical language requesting chain-of-thought logging. "
            "The executable CLI does not expose private reasoning; it emits only structured "
            "operator/state traces."
        )

    result = {
        "path": str(p),
        "sha256": sha256(p),
        "required_markers": len(required),
        "missing": missing,
        "warnings": warnings,
        "valid": not missing,
    }
    if args.json:
        print_json(result)
    else:
        print("kernel:", p)
        print("sha256:", result["sha256"])
        print("valid:", result["valid"])
        if missing:
            print("missing:", ", ".join(missing))
        for w in warnings:
            print("warning:", w)
    return EXIT_OK if not missing else EXIT_DATA


def cmd_kernel_operators(args):
    rows = [
        ("♢", "NEXT", "advance to the next candidate"),
        ("◇", "PAUSE", "hold the current state"),
        ("◆", "RESUME", "continue a paused traversal"),
        ("↺", "ABANDON/RESET", "terminate or reset a branch"),
        ("⇝", "REDIRECT", "change route without discarding the question"),
        ("⊢", "AXIOM", "declare a local premise for the run"),
        ("⊨", "VALIDATED", "accepted under the active contract"),
        ("⊍", "EXPLORE", "expand candidates without commitment"),
        ("⊝", "DISCARD", "reject a candidate with a recorded reason"),
        ("⋈", "JOIN", "recombine branches while preserving provenance"),
        ("⋔", "SPLIT", "create explicit alternate branches"),
        ("↑", "TRANSCEND", "change abstraction level"),
        ("⍟", "METAMORPHOSE", "change representation"),
        ("∞", "RECURSE", "feed transformed state back into traversal"),
        ("§", "SELF-REFERENCE", "inspect circular/self-referential structure"),
        ("⊥", "INCOMPLETE", "cannot close under current evidence/contract"),
        ("⊤", "COMPLETE", "closed under the current contract"),
    ]
    if args.json:
        print_json([{"glyph": a, "name": b, "meaning": c} for a, b, c in rows])
    else:
        for a, b, c in rows:
            print(f"{a:2s}  {b:16s}  {c}")
    return EXIT_OK


def cmd_kernel_manual(args):
    p = DEFAULT_MANUAL
    if not p.exists():
        eprint(f"manual not found: {p}")
        return EXIT_DATA
    if args.path_only:
        print(p)
    else:
        print(p.read_text(encoding="utf-8"))
    return EXIT_OK


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

def cmd_config_show(args):
    cfg = load_config(args.config)
    print_json(config_dict(cfg))
    return EXIT_OK


def cmd_config_validate(args):
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        eprint(exc)
        return EXIT_DATA
    print(f"valid: {args.config}")
    print(f"matrices={cfg.n_matrices} operators={cfg.n_operators}")
    return EXIT_OK


def cmd_config_init(args):
    dest = Path(args.out)
    if dest.exists() and not args.force:
        eprint(f"exists: {dest}; use --force")
        return EXIT_DATA
    shutil.copy2(DEFAULT_CONFIG, dest)
    print(dest)
    return EXIT_OK


def cmd_config_diff(args):
    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    keys = sorted(set(a) | set(b))
    out = []
    for k in keys:
        if a.get(k) != b.get(k):
            out.append({"key": k, "a": a.get(k), "b": b.get(k)})
    print_json(out)
    return EXIT_OK


# ------------------------------------------------------------------
# Corpus
# ------------------------------------------------------------------

def corpus_iter(path):
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            yield i, json.loads(line)


def cmd_corpus_validate(args):
    cfg = load_config(args.config)
    errors = []
    warnings = []
    seen = set()
    count = 0

    matrix_set = set(cfg.matrix_names)
    operator_set = set(cfg.operator_names)

    for line_no, obj in corpus_iter(args.jsonl):
        count += 1
        rid = str(obj.get("id", ""))
        text = obj.get("text")
        matrices = obj.get("matrices", [])
        op = obj.get("operator")

        if not rid:
            errors.append(f"line {line_no}: missing id")
        elif rid in seen:
            errors.append(f"line {line_no}: duplicate id {rid}")
        seen.add(rid)

        if not isinstance(text, str) or not text:
            errors.append(f"line {line_no}: text must be non-empty string")

        unknown_m = sorted(set(matrices) - matrix_set)
        if unknown_m:
            errors.append(f"line {line_no}: unknown matrices {unknown_m}")

        if op is not None and op not in operator_set:
            errors.append(f"line {line_no}: unknown operator {op}")

        md = obj.get("metadata", {})
        if not isinstance(md, dict):
            errors.append(f"line {line_no}: metadata must be object")
        elif "document_id" not in md:
            warnings.append(f"line {line_no}: metadata.document_id missing")

    report = {
        "records": count,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }
    if args.json:
        print_json(report)
    else:
        print(f"records: {count}")
        print(f"errors: {len(errors)}")
        print(f"warnings: {len(warnings)}")
        for x in errors[:50]:
            print("ERROR", x)
        for x in warnings[:50]:
            print("WARN ", x)
    return EXIT_OK if not errors else EXIT_DATA


def cmd_corpus_stats(args):
    cfg = load_config(args.config)
    tok = ByteTokenizer()
    matrix_counts = {m: 0 for m in cfg.matrix_names}
    operator_counts = {o: 0 for o in cfg.operator_names}
    lengths = []
    documents = set()
    records = 0

    for _, obj in corpus_iter(args.jsonl):
        records += 1
        txt = str(obj.get("text", ""))
        lengths.append(len(tok.encode(txt)))
        for m in obj.get("matrices", []):
            matrix_counts[m] = matrix_counts.get(m, 0) + 1
        op = obj.get("operator")
        if op:
            operator_counts[op] = operator_counts.get(op, 0) + 1
        md = obj.get("metadata", {})
        if isinstance(md, dict) and md.get("document_id"):
            documents.add(str(md["document_id"]))

    result = {
        "records": records,
        "documents": len(documents),
        "bytes_total": sum(lengths),
        "bytes_mean": statistics.mean(lengths) if lengths else 0,
        "bytes_median": statistics.median(lengths) if lengths else 0,
        "bytes_min": min(lengths) if lengths else 0,
        "bytes_max": max(lengths) if lengths else 0,
        "matrix_counts": matrix_counts,
        "operator_counts": operator_counts,
    }
    print_json(result)
    return EXIT_OK


def cmd_corpus_split(args):
    rng = random.Random(args.seed)
    rows = [obj for _, obj in corpus_iter(args.jsonl)]
    rng.shuffle(rows)

    n = len(rows)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    parts = {
        "train": rows[:n_train],
        "val": rows[n_train:n_train + n_val],
        "test": rows[n_train + n_val:],
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, vals in parts.items():
        p = out / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for obj in vals:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"{name}: {len(vals)} -> {p}")
    return EXIT_OK


def cmd_corpus_hash(args):
    print(sha256(args.path))
    return EXIT_OK


# ------------------------------------------------------------------
# Model / checkpoint
# ------------------------------------------------------------------

def cmd_model_info(args):
    cfg = load_config(args.config)
    model = RVIMultiMatrixTransformer(cfg)
    result = {
        "version": __version__,
        "params": count_params(model),
        "config": config_dict(cfg),
        "parameter_bytes_fp32": count_params(model) * 4,
        "parameter_bytes_bf16": count_params(model) * 2,
    }
    print_json(result)
    return EXIT_OK


def cmd_model_smoke(args):
    device = device_from_arg(args.device)
    cfg = load_config(args.config)
    model = RVIMultiMatrixTransformer(cfg).to(device)
    T = min(args.seq_len, cfg.max_seq_len)
    x = torch.randint(0, cfg.vocab_size, (args.batch_size, T), device=device)
    y = torch.randint(0, cfg.vocab_size, (args.batch_size, T), device=device)

    logits, loss, diag, mem = model(
        x, y, return_diagnostics=True, return_memory=True
    )
    loss.backward()
    result = {
        "logits": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "memory": list(mem.matrix_summary.shape) if mem else None,
        "last_diagnostics": {
            k: float(v.detach().cpu())
            for k, v in diag[-1].items()
            if torch.is_tensor(v) and v.numel() == 1
        }
    }
    print_json(result)
    return EXIT_OK


def cmd_checkpoint_info(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = RVIConfig(**ckpt["config"])
    state = ckpt["model"]
    result = {
        "path": str(Path(args.checkpoint).resolve()),
        "sha256": sha256(args.checkpoint),
        "config": config_dict(cfg),
        "tensor_count": len(state),
        "parameter_count": int(sum(v.numel() for v in state.values())),
    }
    print_json(result)
    return EXIT_OK


def cmd_checkpoint_diff(args):
    a = torch.load(args.a, map_location="cpu")
    b = torch.load(args.b, map_location="cpu")
    sa, sb = a["model"], b["model"]
    keys = sorted(set(sa) | set(sb))
    rows = []
    for k in keys:
        if k not in sa or k not in sb:
            rows.append({"key": k, "status": "missing", "a": k in sa, "b": k in sb})
            continue
        if sa[k].shape != sb[k].shape:
            rows.append({
                "key": k,
                "status": "shape",
                "a": list(sa[k].shape),
                "b": list(sb[k].shape),
            })
            continue
        delta = (sa[k].float() - sb[k].float())
        rows.append({
            "key": k,
            "status": "ok",
            "l2": float(delta.norm()),
            "max_abs": float(delta.abs().max()),
        })
    if args.only_changed:
        rows = [r for r in rows if r.get("status") != "ok" or r.get("l2", 0) != 0]
    print_json(rows)
    return EXIT_OK


# ------------------------------------------------------------------
# Train / generate
# ------------------------------------------------------------------

def cmd_train(args):
    device = device_from_arg(args.device)
    cfg = load_config(args.config)
    tok = ByteTokenizer()

    records = None
    data = None
    if args.jsonl:
        records = load_jsonl(args.jsonl)
    elif args.text:
        data = plain_text_tensor(args.text, tok)
        if len(data) < args.seq_len + 2:
            raise ValueError("training text is shorter than seq_len")
    else:
        raise ValueError("use --text or --jsonl")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    model = RVIMultiMatrixTransformer(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "metrics.jsonl"

    model.train()
    started = time.time()

    for step in range(1, args.steps + 1):
        if records is not None:
            x, y, matrix_labels, operator_labels = sample_annotated_batch(
                records, cfg, args.batch_size, args.seq_len, device, tok
            )
            _, loss, diag = model(
                x, y,
                matrix_labels=matrix_labels,
                operator_labels=operator_labels,
                return_diagnostics=True,
            )
        else:
            x, y = sample_plain_batch(
                data, args.batch_size, args.seq_len, device
            )
            _, loss, diag = model(x, y, return_diagnostics=True)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            d = diag[-1] if diag else {}
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad.detach().cpu()) if torch.is_tensor(grad) else float(grad),
                "elapsed_s": time.time() - started,
            }
            for k, v in d.items():
                if torch.is_tensor(v) and v.numel() == 1:
                    row[k] = float(v.detach().cpu())
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print_json(row)

        if args.save_every and step % args.save_every == 0:
            bare = model._orig_mod if hasattr(model, "_orig_mod") else model
            p = run_dir / f"step_{step:08d}.pt"
            torch.save({"config": bare.cfg.__dict__, "model": bare.state_dict()}, p)

    bare = model._orig_mod if hasattr(model, "_orig_mod") else model
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": bare.cfg.__dict__, "model": bare.state_dict()}, out)
    print("saved", out)
    return EXIT_OK


def cmd_generate(args):
    device = device_from_arg(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    tok = ByteTokenizer()
    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        ids,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tok.decode(out[0].tolist()))
    return EXIT_OK


# ------------------------------------------------------------------
# Matrix/operator inspection
# ------------------------------------------------------------------

def text_ids(text, device):
    tok = ByteTokenizer()
    ids = tok.encode(text)
    if not ids:
        raise ValueError("text encoded to zero bytes")
    return torch.tensor([ids], dtype=torch.long, device=device)


def cmd_matrix_list(args):
    cfg = load_config(args.config)
    for i, m in enumerate(cfg.matrix_names):
        print(f"{i:02d} {m}")
    return EXIT_OK


def cmd_matrix_route(args):
    device = device_from_arg(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    x = text_ids(args.text, device)
    _, _, _, layers = inspect_forward(model, x)
    aux = layers[-1]["aux"]
    router = aux.get("matrix_router")
    if router is None:
        eprint("multi-matrix router unavailable")
        return EXIT_MODEL
    probs = router.mean(dim=1)[0].detach().cpu()
    rows = sorted(
        [{"matrix": name, "probability": float(probs[i])}
         for i, name in enumerate(cfg.matrix_names)],
        key=lambda r: r["probability"],
        reverse=True,
    )
    print_json(rows)
    return EXIT_OK


def cmd_matrix_coupling(args):
    device = device_from_arg(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    layer = model.blocks[args.layer]
    rvi = layer.attn.cross_rvi
    if rvi is None:
        eprint("cross-matrix RVI unavailable in this checkpoint")
        return EXIT_MODEL

    c = torch.softmax(rvi.matrix_mix, dim=-1).detach().cpu()
    if args.head is not None:
        heads = [args.head]
    else:
        heads = list(range(c.size(0)))

    out = {}
    for h in heads:
        matrix = []
        for i, src in enumerate(cfg.matrix_names):
            row = {"from": src}
            for j, dst in enumerate(cfg.matrix_names):
                row[dst] = float(c[h, i, j])
            matrix.append(row)
        out[f"head_{h}"] = matrix
    print_json(out)
    return EXIT_OK


def cmd_operator_list(args):
    cfg = load_config(args.config)
    for i, op in enumerate(cfg.operator_names):
        print(f"{i:02d} {op}")
    return EXIT_OK


def cmd_operator_predict(args):
    device = device_from_arg(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    x = text_ids(args.text, device)
    _, _, _, layers = inspect_forward(model, x)
    logits = layers[-1]["aux"].get("operator_logits")
    if logits is None:
        eprint("operator router unavailable")
        return EXIT_MODEL
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu()
    rows = sorted(
        [{"operator": op, "probability": float(probs[i])}
         for i, op in enumerate(cfg.operator_names)],
        key=lambda r: r["probability"],
        reverse=True,
    )
    print_json(rows)
    return EXIT_OK


# ------------------------------------------------------------------
# Memory / traces
# ------------------------------------------------------------------

def cmd_memory_encode(args):
    device = device_from_arg(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    x = text_ids(args.text, device)
    mem = model.encode_matrix_memory(x)

    out = {
        "matrices": cfg.matrix_names,
        "stats": {
            name: tensor_stats(mem.matrix_summary[:, i, :])
            for i, name in enumerate(cfg.matrix_names)
        }
    }
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "matrix_names": cfg.matrix_names,
            "matrix_summary": mem.matrix_summary.detach().cpu(),
        }, p)
        out["saved"] = str(p)
    print_json(out)
    return EXIT_OK


def cmd_trace_inspect(args):
    device = device_from_arg(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    x = text_ids(args.text, device)
    _, _, mem, layers = inspect_forward(model, x)

    trace = []
    for item in layers:
        row = {"layer": item["layer"]}
        for k, v in item["diag"].items():
            if torch.is_tensor(v) and v.numel() == 1:
                row[k] = float(v.detach().cpu())
        if item["field"] is not None:
            row["field"] = tensor_stats(item["field"])
        if item["memory"] is not None:
            row["memory"] = tensor_stats(item["memory"].matrix_summary)
        trace.append(row)

    result = {
        "kind": "structured-model-trace",
        "note": (
            "This is an observable operator/state trace. It is not private "
            "chain-of-thought."
        ),
        "text_bytes": int(x.numel()),
        "layers": trace,
    }
    if args.out:
        Path(args.out).write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    print_json(result)
    return EXIT_OK


# ------------------------------------------------------------------
# Index
# ------------------------------------------------------------------

def record_vectors(model, cfg, text: str, device: str):
    x = text_ids(text, device)
    mem = model.encode_matrix_memory(x)
    summary = mem.matrix_summary[0].detach().cpu()
    return {name: summary[i] for i, name in enumerate(cfg.matrix_names)}


def cmd_index_build(args):
    device = device_from_arg(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    idx = MultiMatrixMeaningIndex(cfg.matrix_names, cfg.matrix_dim)

    for n, obj in corpus_iter(args.jsonl):
        text = str(obj.get("text", ""))
        if not text:
            continue
        vecs = record_vectors(model, cfg, text[:args.max_chars], device)
        idx.add(
            str(obj.get("id", f"line-{n}")),
            vecs,
            metadata=dict(obj.get("metadata", {})),
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    idx.save(args.out)
    print(f"saved {len(idx)} records -> {args.out}")
    return EXIT_OK


def parse_weights(spec: Optional[str]) -> Dict[str, float]:
    if not spec:
        return {}
    out = {}
    for pair in spec.split(","):
        name, value = pair.split("=", 1)
        out[name.strip()] = float(value)
    return out


def cmd_index_query(args):
    device = device_from_arg(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    idx = MultiMatrixMeaningIndex.load(args.index)
    vecs = record_vectors(model, cfg, args.text, device)

    weights = parse_weights(args.weights)
    if weights:
        q = {k: vecs[k] for k in weights if k in vecs}
    else:
        q = vecs

    hits = idx.search(
        q,
        matrix_weights=weights or None,
        top_k=args.top_k,
        contradiction_flip=args.contradiction_flip,
    )
    print_json([
        {
            "record_id": h.record_id,
            "score": h.score,
            "per_matrix": h.per_matrix,
            "metadata": h.metadata,
        }
        for h in hits
    ])
    return EXIT_OK


def cmd_index_info(args):
    idx = MultiMatrixMeaningIndex.load(args.index)
    print_json({
        "records": len(idx),
        "matrix_names": idx.matrix_names,
        "matrix_dim": idx.matrix_dim,
    })
    return EXIT_OK


# ------------------------------------------------------------------
# DAC
# ------------------------------------------------------------------

def write_wav(path: Path, pcm: torch.Tensor, sample_rate: int):
    samples = (pcm.detach().clamp(-1, 1).cpu() * 32767).round().to(torch.int16)
    if samples.ndim == 2:
        samples = samples[0]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.numpy().tobytes())


def cmd_dac_render(args):
    device = device_from_arg(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    x = text_ids(args.text, device)

    _, hidden, _, _ = inspect_forward(model, x)
    dac_cfg = DACConfig(
        d_model=cfg.d_model,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        harmonics=args.harmonics,
    )
    dac = NeuralDAC(dac_cfg).to(device)
    pcm, controls = dac(hidden)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out, pcm, args.sample_rate)

    print_json({
        "wav": str(out),
        "sample_rate": args.sample_rate,
        "samples": int(pcm.shape[-1]),
        "duration_s": float(pcm.shape[-1] / args.sample_rate),
        "f0": tensor_stats(controls["f0_hz"]),
        "amplitude": tensor_stats(controls["amplitude"]),
        "boundary": (
            "Sensory/audio renderer only; this command does not drive direct "
            "neural stimulation hardware."
        ),
    })
    return EXIT_OK


# ------------------------------------------------------------------
# Ablation / benchmark
# ------------------------------------------------------------------

def cfg_profile(base: Dict, name: str):
    d = copy.deepcopy(base)
    if name == "baseline":
        d.update(rvi_enabled=False, multimatrix_enabled=False, ctms_enabled=False)
    elif name == "rvi":
        d.update(rvi_enabled=True, multimatrix_enabled=False, ctms_enabled=True)
    elif name == "matrix":
        d.update(rvi_enabled=False, multimatrix_enabled=True, ctms_enabled=False)
    elif name == "full":
        d.update(rvi_enabled=True, multimatrix_enabled=True, ctms_enabled=True)
    elif name == "full_no_ctms":
        d.update(rvi_enabled=True, multimatrix_enabled=True, ctms_enabled=False)
    else:
        raise ValueError(name)
    return RVIConfig(**d)


def train_eval_profile(cfg, tr, va, args, device):
    torch.manual_seed(args.seed)
    model = RVIMultiMatrixTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    start = time.time()
    for _ in range(args.steps):
        x, y = sample_plain_batch(tr, args.batch_size, args.seq_len, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    train_s = time.time() - start
    vals = []
    model.eval()
    with torch.no_grad():
        for _ in range(args.eval_batches):
            x, y = sample_plain_batch(va, args.batch_size, args.seq_len, device)
            _, loss = model(x, y)
            vals.append(float(loss.detach().cpu()))
    return {
        "val_loss": statistics.mean(vals),
        "params": count_params(model),
        "train_s": train_s,
    }


def cmd_ablate(args):
    device = device_from_arg(args.device)
    tok = ByteTokenizer()
    data = torch.tensor(
        tok.encode(Path(args.text).read_text(encoding="utf-8")),
        dtype=torch.long
    )
    cut = int(len(data) * args.train_fraction)
    tr, va = data[:cut], data[cut:]
    base = json.loads(Path(args.config).read_text(encoding="utf-8"))

    profiles = args.profiles.split(",")
    results = {}
    for name in profiles:
        name = name.strip()
        cfg = cfg_profile(base, name)
        results[name] = train_eval_profile(cfg, tr, va, args, device)
        print_json({name: results[name]})

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return EXIT_OK


def cmd_benchmark(args):
    device = device_from_arg(args.device)
    cfg = load_config(args.config)
    model = RVIMultiMatrixTransformer(cfg).to(device)
    model.eval()

    T = min(args.seq_len, cfg.max_seq_len)
    x = torch.randint(0, cfg.vocab_size, (args.batch_size, T), device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(args.iters):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

    tokens = args.iters * args.batch_size * T
    result = {
        "device": device,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "seq_len": T,
        "elapsed_s": elapsed,
        "tokens_per_s": tokens / elapsed,
        "ms_per_iter": 1000 * elapsed / args.iters,
        "params": count_params(model),
    }
    if device == "cuda":
        result["cuda_max_memory_bytes"] = torch.cuda.max_memory_allocated()
    print_json(result)
    return EXIT_OK


# ------------------------------------------------------------------
# Manual search
# ------------------------------------------------------------------

def cmd_manual_search(args):
    p = DEFAULT_MANUAL
    text = p.read_text(encoding="utf-8")
    q = args.query.lower()
    lines = text.splitlines()
    hits = []
    for i, line in enumerate(lines):
        if q in line.lower():
            start = max(0, i - args.context)
            end = min(len(lines), i + args.context + 1)
            hits.append({
                "line": i + 1,
                "context": lines[start:end],
            })
            if len(hits) >= args.limit:
                break
    print_json(hits)
    return EXIT_OK


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="ctms",
        description=(
            "CTMS / RVI Multi-Matrix Transformer research CLI. "
            "The canonical CTMS.md remains separate from trainable RVI/CTMS analogues."
        ),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # init
    s = sub.add_parser("init", help="create a reproducible CTMS/RVI workspace")
    s.add_argument("path")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_init)

    # doctor
    s = sub.add_parser("doctor", help="check runtime, kernel, config and optional smoke test")
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    s.add_argument("--device", default="auto")
    s.add_argument("--smoke", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_doctor)

    # kernel
    k = sub.add_parser("kernel", help="inspect and validate the canonical CTMS kernel")
    ks = k.add_subparsers(dest="kernel_command", required=True)

    s = ks.add_parser("show")
    s.add_argument("--path", default=str(DEFAULT_KERNEL))
    s.set_defaults(func=cmd_kernel_show)

    s = ks.add_parser("hash")
    s.add_argument("--path", default=str(DEFAULT_KERNEL))
    s.set_defaults(func=cmd_kernel_hash)

    s = ks.add_parser("validate")
    s.add_argument("--path", default=str(DEFAULT_KERNEL))
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_kernel_validate)

    s = ks.add_parser("operators")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_kernel_operators)

    s = ks.add_parser("manual")
    s.add_argument("--path-only", action="store_true")
    s.set_defaults(func=cmd_kernel_manual)

    # config
    c = sub.add_parser("config", help="configuration operations")
    cs = c.add_subparsers(dest="config_command", required=True)

    s = cs.add_parser("show")
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    s.set_defaults(func=cmd_config_show)

    s = cs.add_parser("validate")
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    s.set_defaults(func=cmd_config_validate)

    s = cs.add_parser("init")
    s.add_argument("--out", default="config.json")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_config_init)

    s = cs.add_parser("diff")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(func=cmd_config_diff)

    # corpus
    c = sub.add_parser("corpus", help="annotated training corpus tools")
    cs = c.add_subparsers(dest="corpus_command", required=True)

    s = cs.add_parser("validate")
    s.add_argument("jsonl")
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_corpus_validate)

    s = cs.add_parser("stats")
    s.add_argument("jsonl")
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    s.set_defaults(func=cmd_corpus_stats)

    s = cs.add_parser("split")
    s.add_argument("jsonl")
    s.add_argument("--out", default="corpus/splits")
    s.add_argument("--train", type=float, default=0.8)
    s.add_argument("--val", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=1337)
    s.set_defaults(func=cmd_corpus_split)

    s = cs.add_parser("hash")
    s.add_argument("path")
    s.set_defaults(func=cmd_corpus_hash)

    # model
    m = sub.add_parser("model", help="model construction and smoke tests")
    ms = m.add_subparsers(dest="model_command", required=True)

    s = ms.add_parser("info")
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    s.set_defaults(func=cmd_model_info)

    s = ms.add_parser("smoke")
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    s.add_argument("--device", default="auto")
    s.add_argument("--batch-size", type=int, default=1)
    s.add_argument("--seq-len", type=int, default=32)
    s.set_defaults(func=cmd_model_smoke)

    # checkpoint
    c = sub.add_parser("checkpoint", help="checkpoint inspection and comparison")
    cs = c.add_subparsers(dest="checkpoint_command", required=True)

    s = cs.add_parser("info")
    s.add_argument("checkpoint")
    s.set_defaults(func=cmd_checkpoint_info)

    s = cs.add_parser("diff")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--only-changed", action="store_true")
    s.set_defaults(func=cmd_checkpoint_diff)

    # train
    s = sub.add_parser("train", help="train on plain UTF-8 text or annotated JSONL")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--jsonl")
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    s.add_argument("--steps", type=int, default=1000)
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--seq-len", type=int, default=256)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--weight-decay", type=float, default=0.1)
    s.add_argument("--grad-clip", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=1337)
    s.add_argument("--device", default="auto")
    s.add_argument("--compile", action="store_true")
    s.add_argument("--log-every", type=int, default=50)
    s.add_argument("--save-every", type=int, default=0)
    s.add_argument("--run-dir", default="runs/current")
    s.add_argument("--out", default="checkpoints/checkpoint_v04.pt")
    s.set_defaults(func=cmd_train)

    # generate
    s = sub.add_parser("generate", help="autoregressive generation")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--prompt", default="The relation changes when")
    s.add_argument("--tokens", type=int, default=200)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=50)
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_generate)

    # matrix
    m = sub.add_parser("matrix", help="multi-matrix inspection")
    ms = m.add_subparsers(dest="matrix_command", required=True)

    s = ms.add_parser("list")
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    s.set_defaults(func=cmd_matrix_list)

    s = ms.add_parser("route")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--text", required=True)
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_matrix_route)

    s = ms.add_parser("coupling")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--layer", type=int, default=0)
    s.add_argument("--head", type=int)
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_matrix_coupling)

    # operator
    o = sub.add_parser("operator", help="CTMS differentiable operator-router inspection")
    osub = o.add_subparsers(dest="operator_command", required=True)

    s = osub.add_parser("list")
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    s.set_defaults(func=cmd_operator_list)

    s = osub.add_parser("predict")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--text", required=True)
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_operator_predict)

    # memory
    m = sub.add_parser("memory", help="compressed multi-matrix relational memory")
    ms = m.add_subparsers(dest="memory_command", required=True)
    s = ms.add_parser("encode")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--text", required=True)
    s.add_argument("--out")
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_memory_encode)

    # trace
    t = sub.add_parser("trace", help="observable model state traces")
    ts = t.add_subparsers(dest="trace_command", required=True)
    s = ts.add_parser("inspect")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--text", required=True)
    s.add_argument("--out")
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_trace_inspect)

    # index
    i = sub.add_parser("index", help="external multi-matrix meaning index")
    isub = i.add_subparsers(dest="index_command", required=True)

    s = isub.add_parser("build")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--jsonl", required=True)
    s.add_argument("--out", default="indices/meaning_index.pt")
    s.add_argument("--max-chars", type=int, default=512)
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_index_build)

    s = isub.add_parser("query")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--index", required=True)
    s.add_argument("--text", required=True)
    s.add_argument("--weights", help="semantic=1,temporal=.5,contradiction=2")
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--contradiction-flip", action="store_true")
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_index_query)

    s = isub.add_parser("info")
    s.add_argument("index")
    s.set_defaults(func=cmd_index_info)

    # dac
    d = sub.add_parser("dac", help="software PCM/sensory renderer")
    ds = d.add_subparsers(dest="dac_command", required=True)
    s = ds.add_parser("render")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--text", required=True)
    s.add_argument("--out", default="audio/rvi.wav")
    s.add_argument("--sample-rate", type=int, default=48000)
    s.add_argument("--hop-length", type=int, default=240)
    s.add_argument("--harmonics", type=int, default=32)
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_dac_render)

    # ablate
    s = sub.add_parser("ablate", help="structural A/B ablation suite")
    s.add_argument("--text", required=True)
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    s.add_argument("--profiles", default="baseline,rvi,matrix,full_no_ctms,full")
    s.add_argument("--steps", type=int, default=200)
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--seq-len", type=int, default=128)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--seed", type=int, default=1337)
    s.add_argument("--eval-batches", type=int, default=10)
    s.add_argument("--train-fraction", type=float, default=0.9)
    s.add_argument("--device", default="auto")
    s.add_argument("--out")
    s.set_defaults(func=cmd_ablate)

    # benchmark
    s = sub.add_parser("benchmark", help="forward throughput benchmark")
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    s.add_argument("--device", default="auto")
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--seq-len", type=int, default=256)
    s.add_argument("--warmup", type=int, default=3)
    s.add_argument("--iters", type=int, default=20)
    s.set_defaults(func=cmd_benchmark)

    # manual
    m = sub.add_parser("manual", help="read/search the complete CLI manual")
    ms = m.add_subparsers(dest="manual_command", required=True)
    s = ms.add_parser("show")
    s.set_defaults(func=lambda args: cmd_kernel_manual(argparse.Namespace(path_only=False)))
    s = ms.add_parser("path")
    s.set_defaults(func=lambda args: cmd_kernel_manual(argparse.Namespace(path_only=True)))
    s = ms.add_parser("search")
    s.add_argument("query")
    s.add_argument("--context", type=int, default=3)
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_manual_search)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        eprint("interrupted")
        return 130
    except FileNotFoundError as exc:
        eprint(exc)
        return EXIT_DATA
    except (ValueError, AssertionError, json.JSONDecodeError) as exc:
        eprint(f"data/config error: {exc}")
        return EXIT_DATA
    except RuntimeError as exc:
        eprint(f"runtime error: {exc}")
        return EXIT_RUNTIME
    except Exception as exc:
        if os.environ.get("CTMS_DEBUG") == "1":
            raise
        eprint(f"error: {exc}")
        eprint("set CTMS_DEBUG=1 for a traceback")
        return EXIT_RUNTIME
