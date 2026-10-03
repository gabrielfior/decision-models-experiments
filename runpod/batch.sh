#!/usr/bin/env bash
# Tier B batch on the pod: Kev-recipe variants back to back, then stop the pod so it does not idle.
# Run detached:  nohup bash runpod/batch.sh > runs/logs/batch.log 2>&1 &
# Each run ~20-25 min on the 3090 (8k rows, 1 epoch). Results append to results.tsv on the pod; pull with runpod/pod.py.
set -uo pipefail
cd /workspace/decider
export HF_HOME=/workspace/hf DECIDER_DATA=/workspace/decider/data DECIDER_CACHE=/workspace/decider/data/cache DECIDER_RUNS=/workspace/decider/runs
export DECIDER_RESULTS=/workspace/decider/results.tsv DECIDER_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo pod)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/logs
run() {  # run <name> <train.py args...>
  local name=$1; shift
  echo "=== $(date -u +%FT%TZ) start $name"
  uv run python train.py --out "runs/$name" "$@" > "runs/logs/$name.log" 2>&1
  echo "=== $(date -u +%FT%TZ) exit $? $name"; tail -3 "runs/logs/$name.log"
}
run tierB-baseline-s1   --seed 1 --note "baseline replicate seed 1: Kev recipe r16, pointer head"
run tierB-rank64        --seed 0 --rank 64 --note "LoRA rank 64 (alpha 128), Kev recipe otherwise"
run tierB-depth-heads   --seed 0 --depth-heads --note "laddered depth heads at taps 12/16/20/24, summed loss (Needle)"
run tierB-top-half      --seed 0 --top-half-only --note "LoRA on the top 12 layers only, r16"
echo "=== batch done $(date -u +%FT%TZ)"
# Stop billing. runpodctl ships in runpod/* images and knows this pod's id.
command -v runpodctl >/dev/null && runpodctl stop pod "${RUNPOD_POD_ID:-}" || echo "could not self-stop; stop the pod from the laptop: uv run python runpod/pod.py stop"
