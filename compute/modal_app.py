"""Modal driver for the GPU steps, capped at MODAL_BUDGET_USD of compute this month.

Why Modal first: the account is already authenticated and serverless billing means an
idle pod cannot leak money, which the plan calls the biggest cost risk. The A10G has
24 GB like an RTX 3090 at ~$1.10/hr; the first ~$5 covers the feature cache and the
LoRA baseline. Everything after that moves to RunPod (see runpod/).

Usage (from the repo root, after `uv sync --group cloud`):
    uv run python compute/modal_budget.py            # prints month-to-date compute spend
    uv run modal run compute/modal_app.py::cache     # hidden-state cache for TAP_LAYERS
    uv run modal run compute/modal_app.py::train_lora --note "baseline: Kev recipe, 1 epoch"
    uv run modal volume get decider-data cache/ data/cache/   # pull the cache for local Tier A

The volume `decider-data` holds /vol/hf (model downloads), /vol/data (raw rows + splits),
/vol/cache (hidden states) and /vol/runs (adapters, heads, reports).
"""
from __future__ import annotations

import os
import subprocess

import modal

APP = "decider-autoresearch"
GPU = os.environ.get("DECIDER_GPU", "A10G")
VOL = modal.Volume.from_name("decider-data", create_if_missing=True)
V = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "transformers==5.17.0", "peft==0.21.0", "accelerate>=1.15.0", "datasets>=3.0",
        "huggingface-hub>=0.30", "safetensors>=0.4", "numpy>=2.0",
        "flash-linear-attention==0.5.2",   # fused GatedDeltaNet kernel; the PyTorch fallback OOMs a 24 GB card in training
    )
    .env({"HF_HOME": f"{V}/hf", "DECIDER_DATA": f"{V}/data", "DECIDER_CACHE": f"{V}/cache",
          "DECIDER_RUNS": f"{V}/runs", "DECIDER_RESULTS": f"{V}/runs/results.tsv", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("prepare", "head", "train")
    .add_local_dir("data/splits", remote_path="/root/data/splits")   # the committed row ids
)


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()

app = modal.App(APP, image=image)


def _guard():
    """Refuse to start GPU work once this month's Modal compute exceeds the cap."""
    cap = float(os.environ.get("MODAL_BUDGET_USD", "20"))   # raised from 5 on 2026-10-03 evening at the user's request
    spent = float(subprocess.check_output(["python", "compute/modal_budget.py", "--value"]).decode())
    if spent >= cap:
        raise SystemExit(f"Modal compute this month ${spent:.2f} >= cap ${cap:.2f}; move to RunPod")
    print(f"Modal compute this month ${spent:.2f} of ${cap:.2f} cap")


@app.function(gpu=GPU, volumes={V: VOL}, timeout=2 * 3600)
def _cache(torso: str, commit: str, batch_size: int = 16):
    os.environ["DECIDER_COMMIT"] = commit
    import prepare as P
    P.cache_features(torso, batch_size=batch_size)
    VOL.commit()


@app.function(gpu=GPU, volumes={V: VOL}, timeout=3 * 3600)
def _train(argv: list[str], commit: str):
    os.environ["DECIDER_COMMIT"] = commit
    # One results file per run: concurrent containers committing the same volume file clobber each
    # other (last writer wins), which lost two rows on 2026-10-03. Merge locally from runs/*/results.tsv.
    out = argv[argv.index("--out") + 1]
    os.makedirs(out, exist_ok=True)
    os.environ["DECIDER_RESULTS"] = f"{out}/results.tsv"
    import train
    train.main(argv)
    VOL.commit()


@app.function(volumes={V: VOL}, timeout=3600)
def _data():
    import prepare as P
    P.build_splits()
    VOL.commit()


@app.local_entrypoint()
def data():
    _data.remote()


@app.local_entrypoint()
def cache(torso: str = "Qwen/Qwen3.5-0.8B-Base", batch_size: int = 16):
    _guard()
    _cache.remote(torso, _commit(), batch_size)


@app.local_entrypoint()
def train_lora(note: str, seed: int = 0, max_rows: int = 0, extra: str = ""):
    """extra: further train.py flags as one string, e.g. "--rank 64 --depth-heads"."""
    _guard()
    import time
    argv = ["--note", note, "--seed", str(seed), "--out", f"{V}/runs/{note[:40].replace(' ', '_')}-s{seed}-{time.strftime('%m%d%H%M')}", *extra.split()]
    if max_rows:
        argv += ["--max-rows", str(max_rows)]
    _train.remote(argv, _commit())
