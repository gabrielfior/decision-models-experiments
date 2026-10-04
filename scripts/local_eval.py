"""Evaluate a decider (run dir, optionally an exported one) on the dev split locally (MPS/CPU/CUDA).

usage: uv run python scripts/local_eval.py --run <dir> [--torso <id or exported torso dir>] [--n 2000] [--device mps]
                                           [--quant int8|int4g128] [--refit-temps]
Prints dev accuracy, Brier, selection with the run's saved per-type temperatures (or refit ones).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import prepare as P  # noqa: E402


def fake_quant_(model, scheme: str):
    """Weight-only fake quantisation in place: int8 per-output-channel symmetric, or int4 with groups of 128
    along the input dim (asymmetric, per group). Values are quantised then dequantised, so the forward pass
    measures exactly what an integer kernel would compute (up to its accumulation precision)."""
    import torch
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)):
            continue
        w = mod.weight.data
        w32 = w.float()
        if scheme == "int8":
            s = w32.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
            q = torch.clamp(torch.round(w32 / s), -127, 127) * s
        elif scheme.startswith("int4g"):
            g = int(scheme[5:])
            out, inn = w32.shape
            pad = (-inn) % g
            wp = torch.nn.functional.pad(w32, (0, pad)).reshape(out, -1, g)
            lo, hi = wp.amin(dim=2, keepdim=True), wp.amax(dim=2, keepdim=True)
            s = ((hi - lo) / 15.0).clamp(min=1e-8)
            q = (torch.clamp(torch.round((wp - lo) / s), 0, 15) * s + lo).reshape(out, -1)[:, :inn]
        else:
            raise ValueError(scheme)
        mod.weight.data.copy_(q.to(w.dtype))
        n += w.numel()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--torso", default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--refit-temps", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import torch
    import serve
    device = a.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    cfg = json.loads((a.run / "config.json").read_text())
    torso = a.torso or cfg["torso"]
    t0 = time.time()
    d = serve.Decider(torso, a.run, None, device, 1)
    nq = fake_quant_(d.model, a.quant) if a.quant else 0
    params = sum(p.numel() for p in d.model.parameters())
    rows = P.load_split("dev")[: a.n]
    t1 = time.time()
    dev_a = P.predict_rows(d.logits, d.tok, rows, order="identity", device=device, markers=d.markers, batch_size=a.batch)
    temps = P.fit_temperature_by_type(dev_a) if a.refit_temps else d.temps
    s = P.score_preds(dev_a, temps)
    out = {"tag": a.tag, "torso": torso, "run": str(a.run), "quant": a.quant, "params_M": round(params / 1e6), "quantised_M": round(nq / 1e6),
           "n": len(rows), "acc": round(s["accuracy"], 4), "brier": round(s["brier"], 4), "selection": round(s["selection"], 4),
           "ece": round(s["ece"], 4), "ms_per_row": round((time.time() - t1) * 1000 / len(rows)), "load_s": round(t1 - t0)}
    print(json.dumps(out))
    np.save(a.run / f"dev_logits{('-' + a.tag) if a.tag else ''}.npy", np.array([p["logits"] for p in dev_a], dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
