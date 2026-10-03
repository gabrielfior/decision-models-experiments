#!/usr/bin/env bash
# First-boot setup on the pod, run over SSH after `runpod/pod.py sync`. Idempotent.
# /workspace is the pod volume: it survives stop/start, not terminate.
set -euo pipefail
cd /workspace/decider
export HF_HOME=/workspace/hf DECIDER_DATA=/workspace/decider/data DECIDER_CACHE=/workspace/decider/data/cache DECIDER_RUNS=/workspace/decider/runs
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# Pins match the image's torch 2.8.0 / CUDA 12.8; fla is the fused GatedDeltaNet kernel (pure Triton, no compile).
uv sync --group gpu && uv pip install "flash-linear-attention==0.5.2"
mkdir -p "$HF_HOME" data/raw data/cache runs
uv run python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), f'{torch.cuda.mem_get_info()[1]/2**30:.1f} GB')"
uv run python -c "import fla; print('fla ok')" || echo "fla import failed; the slow fallback kernel will be used"
echo "setup done; stop the pod when the batch is finished: it bills while idle"
