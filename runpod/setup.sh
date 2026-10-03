#!/usr/bin/env bash
# First-boot setup on a RunPod pod. Idempotent: safe to re-run after a restart.
# The network volume at /workspace survives pod restarts; the container disk does not.
set -euo pipefail
cd /workspace
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if [ ! -d decider ]; then git clone --recurse-submodules "${REPO_URL:?set REPO_URL in the template env}" decider; fi
cd decider && git pull --ff-only && git submodule update --init
# Pins match the image's torch 2.8.0 / CUDA 12.8; the kernels group is optional speed for GatedDeltaNet.
uv sync --group gpu --group kernels || uv sync --group gpu
mkdir -p "$HF_HOME" data/raw data/cache runs
uv run python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "setup done; stop the pod when you are finished: it bills while idle"
