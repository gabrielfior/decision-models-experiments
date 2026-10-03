"""soup.py — average N LoRA runs (adapter + head) into one run dir ("model soup").

Each input run dir is what train.py writes: lora/adapter_model.safetensors + lora/adapter_config.json
(peft), head.pt (the head's state dict) and config.json. Runs that differ only in seed / data order
land in one loss basin, so the uniform (or weighted) mean of their adapter tensors and head weights is
usually a better single model than any member — at the inference cost of ONE model, unlike an ensemble.

The adapter configs must agree on r, lora_alpha and target_modules (the tensors would not even line
up otherwise). Every tensor is averaged on the CPU; non-floating tensors (counters) are copied from
the first run. The output dir gets lora/ (adapter_config.json copied, averaged safetensors), head.pt
and config.json copied from the first run with "note": "soup of [...]" and the member list + weights
under "soup". serve.py --run <out> loads it like any other run; refit temperatures afterwards with
    uv run python scripts/fit_temps.py --run <out>
Usage:
    uv run python scripts/soup.py --out runs/soup-abc runs/a runs/b runs/c [--weights 1,1,2]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

MUST_MATCH = ("r", "lora_alpha", "target_modules")


def check_adapter_configs(cfgs: list[dict]) -> None:
    """Raise ValueError naming the first key in MUST_MATCH on which the peft adapter configs disagree."""
    for key in MUST_MATCH:
        vals = [c.get(key) for c in cfgs]
        norm = [sorted(v) if isinstance(v, (list, tuple, set)) else v for v in vals]
        if any(v != norm[0] for v in norm[1:]):
            raise ValueError(f"adapter configs differ on {key}: {vals}")


def normalise_weights(n: int, weights: list[float] | None) -> list[float]:
    if weights is None:
        return [1.0 / n] * n
    if len(weights) != n or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(f"weights must be {n} non-negative numbers with a positive sum, got {weights}")
    s = float(sum(weights))
    return [float(w) / s for w in weights]


def average_tensors(dicts: list[dict[str, torch.Tensor]], weights: list[float] | None = None) -> dict[str, torch.Tensor]:
    """Weighted mean of each tensor across the dicts; keys and shapes must match exactly.
    Averaging is done in float32 and cast back to the first dict's dtype; non-float tensors come from the first dict."""
    if not dicts:
        raise ValueError("nothing to average")
    w = normalise_weights(len(dicts), weights)
    keys = set(dicts[0])
    for i, d in enumerate(dicts[1:], 1):
        if set(d) != keys:
            raise ValueError(f"tensor keys differ between member 0 and member {i}: "
                             f"{sorted(keys ^ set(d))[:5]}")
    out = {}
    for k in dicts[0]:
        first = dicts[0][k]
        if not first.is_floating_point():
            out[k] = first.clone()
            continue
        acc = torch.zeros(first.shape, dtype=torch.float32)
        for wi, d in zip(w, dicts):
            if d[k].shape != first.shape:
                raise ValueError(f"shape mismatch for {k}: {tuple(d[k].shape)} vs {tuple(first.shape)}")
            acc += wi * d[k].detach().to("cpu", torch.float32)
        out[k] = acc.to(first.dtype).contiguous()
    return out


def soup(run_dirs: list[Path], out_dir: Path, weights: list[float] | None = None) -> Path:
    """Average the runs into out_dir and return it. See the module docstring for the layout."""
    runs = [Path(r) for r in run_dirs]
    if not runs:
        raise ValueError("need at least one run dir")
    w = normalise_weights(len(runs), weights)
    out_dir = Path(out_dir)
    has_lora = [(r / "lora" / "adapter_model.safetensors").exists() for r in runs]
    if any(has_lora) and not all(has_lora):
        raise ValueError("every run must have lora/adapter_model.safetensors, or none of them")

    out_dir.mkdir(parents=True, exist_ok=True)
    if all(has_lora):
        cfgs = [json.loads((r / "lora" / "adapter_config.json").read_text()) for r in runs]
        check_adapter_configs(cfgs)
        adapters = [load_file(str(r / "lora" / "adapter_model.safetensors"), device="cpu") for r in runs]
        (out_dir / "lora").mkdir(exist_ok=True)
        shutil.copyfile(runs[0] / "lora" / "adapter_config.json", out_dir / "lora" / "adapter_config.json")
        save_file(average_tensors(adapters, w), str(out_dir / "lora" / "adapter_model.safetensors"), metadata={"format": "pt"})

    heads = [torch.load(r / "head.pt", map_location="cpu") for r in runs]
    torch.save(average_tensors(heads, w), out_dir / "head.pt")

    cfg = json.loads((runs[0] / "config.json").read_text())
    names = [str(r) for r in runs]
    cfg["note"] = f"soup of {names}"
    cfg["soup"] = {"runs": names, "weights": w}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="run dirs to average (lora/, head.pt, config.json)")
    ap.add_argument("--out", required=True, help="output run dir")
    ap.add_argument("--weights", default=None, help="comma-separated weights, one per run (default uniform); normalised to sum 1")
    a = ap.parse_args(argv)
    weights = [float(x) for x in a.weights.split(",")] if a.weights else None
    out = soup([Path(r) for r in a.runs], Path(a.out), weights)
    print(json.dumps({"out": str(out), **json.loads((out / "config.json").read_text())["soup"]}, indent=2))


if __name__ == "__main__":
    main()
