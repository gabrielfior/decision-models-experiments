#!/usr/bin/env bash
# Batch 2 on the pod: the plan's scaling check (Qwen3.5-2B, same head and recipe) and an epoch probe on 0.8B.
set -uo pipefail
cd /workspace/decider
export HF_HOME=/workspace/hf DECIDER_DATA=/workspace/decider/data DECIDER_CACHE=/workspace/decider/data/cache DECIDER_RUNS=/workspace/decider/runs
export DECIDER_RESULTS=/workspace/decider/results.tsv DECIDER_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo pod)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/logs
run() { local name=$1; shift; echo "=== $(date -u +%FT%TZ) start $name"; uv run python train.py --out "runs/$name" "$@" > "runs/logs/$name.log" 2>&1; echo "=== $(date -u +%FT%TZ) exit $? $name"; tail -2 "runs/logs/$name.log"; }
run tierB-qwen2b-ln16    --seed 0 --torso Qwen/Qwen3.5-2B-Base --note "scaling check: Qwen3.5-2B + LoRA r16, LN16 head, 1 epoch"
run tierB-ln16-3ep       --seed 0 --epochs 3 --note "LN16 head + LoRA r16, 3 epochs (epoch probe), dropout active"
echo "=== batch done $(date -u +%FT%TZ)"
