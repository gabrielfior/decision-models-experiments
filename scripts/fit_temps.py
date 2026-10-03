"""fit_temps.py — refit a run's per-type temperatures on dev (or heldout) and write them into config.json.

train.py fits one temperature per question type on dev right after training. A soup (scripts/soup.py),
an adapter re-exported from another machine, or a run whose head was swapped carries temperatures
fitted for a DIFFERENT set of logits, so its probabilities (and hence Brier) are off even when the
argmax is right. This script loads the run exactly as serve.py does (serve.Decider, tta=1), runs the
split through it, refits T per type with prepare.fit_temperature_by_type, and rewrites config.json:
    "temps"      the new per-type temperatures (what serve.py reads)
    "temps_prev" the ones that were there before
    "temps_fit"  {"split", "n_rows", "per_k"} provenance
With --per-k it also fits one T per (type, n_options) bucket that has >= --min-rows rows (others fall
back to the type's T) and stores them under "temps_per_k" as {"choice:4": T, ...}; serve.py ignores
that key today, so --per-k is a report until serving reads it (temp_for() below is the lookup to use).

Usage (needs the torso; GPU if available, else CPU and slow):
    uv run python scripts/fit_temps.py --run runs/<dir> [--split dev|heldout] [--per-k] [--dry-run]
The pure helpers (merge_temps, fit_temperature_by_k, temp_for) are tested in tests/test_soup.py; nothing
torch-heavy runs at import time.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prepare as P  # noqa: E402  (numpy only at import time)


def bucket_key(qtype: str, n_options: int) -> str:
    return f"{qtype}:{int(n_options)}"


def temp_for(row: dict, temps: dict[str, float], per_k: dict[str, float] | None) -> float:
    """The temperature to apply to a row: its (type, n_options) bucket if fitted, else its type's, else 1."""
    if per_k:
        k = bucket_key(row["qtype"], row["n_options"])
        if k in per_k:
            return float(per_k[k])
    return float(temps.get(row["qtype"], 1.0))


def fit_temperature_by_k(preds: list[dict], by_type: dict[str, float], min_rows: int = 50) -> dict[str, float]:
    """One T per (type, n_options) bucket with at least min_rows predictions; smaller buckets get the type's T."""
    counts = Counter(bucket_key(p["qtype"], p["n_options"]) for p in preds)
    out = {}
    for key in sorted(counts):
        qtype = key.split(":")[0]
        if counts[key] >= min_rows:
            sub = [p for p in preds if bucket_key(p["qtype"], p["n_options"]) == key]
            out[key] = P.fit_temperature_by_type(sub)[qtype]
        else:
            out[key] = float(by_type.get(qtype, 1.0))
    return out


def merge_temps(cfg: dict, temps: dict[str, float], per_k: dict[str, float] | None = None, *,
                split: str, n_rows: int) -> dict:
    """A copy of the run config with the new temperatures in "temps", the old ones kept under
    "temps_prev" (likewise "temps_per_k" / "temps_per_k_prev"), and the fit's provenance in "temps_fit"."""
    out = dict(cfg)
    if "temps" in cfg:
        out["temps_prev"] = cfg["temps"]
    out["temps"] = dict(temps)
    if per_k is not None:
        if "temps_per_k" in cfg:
            out["temps_per_k_prev"] = cfg["temps_per_k"]
        out["temps_per_k"] = dict(per_k)
    out["temps_fit"] = {"split": split, "n_rows": int(n_rows), "per_k": per_k is not None}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="refit a run's temperatures on a split and write them to its config.json")
    ap.add_argument("--run", required=True, help="run dir with config.json, head.pt and (optionally) lora/")
    ap.add_argument("--split", default="dev", choices=("dev", "heldout", "train"))
    ap.add_argument("--torso", default=None, help="override config.json's torso")
    ap.add_argument("--per-k", action="store_true", help="also fit one T per (type, n_options) bucket -> temps_per_k")
    ap.add_argument("--min-rows", type=int, default=50, help="smallest bucket that gets its own T with --per-k")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N rows of the split (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="print the temperatures but do not write config.json")
    a = ap.parse_args(argv)

    import torch
    import serve
    run_dir = Path(a.run)
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    torso = a.torso or cfg.get("torso") or "Qwen/Qwen3.5-0.8B-Base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = P.load_split(a.split)
    if a.limit:
        rows = rows[: a.limit]
    print(f"run {run_dir}  torso {torso}  split {a.split} ({len(rows)} rows)  device {device}", flush=True)

    dec = serve.Decider(torso, run_dir, None, device, tta=1)
    preds = P.predict_rows(dec.logits, dec.tok, rows, order="identity", device=device, batch_size=a.batch_size, markers=dec.markers)
    temps = P.fit_temperature_by_type(preds)
    per_k = fit_temperature_by_k(preds, temps, a.min_rows) if a.per_k else None

    before = P.score_preds(preds, cfg.get("temps", {}))
    after = P.score_preds(preds, temps)
    print(json.dumps({"temps_prev": cfg.get("temps"), "temps": temps, "temps_per_k": per_k,
                      "brier_before": before.get("brier"), "brier_after": after.get("brier"),
                      "selection_before": before.get("selection"), "selection_after": after.get("selection")},
                     indent=2, default=str), flush=True)
    if a.dry_run:
        return
    new_cfg = merge_temps(cfg, temps, per_k, split=a.split, n_rows=len(rows))
    cfg_path.write_text(json.dumps(new_cfg, indent=2, default=str))
    print(f"wrote {cfg_path}", flush=True)


if __name__ == "__main__":
    main()
