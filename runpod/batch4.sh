#!/usr/bin/env bash
# Session-3 batch: new torsos (same recipe: LN head at 67% depth + LoRA r16, 2 epochs, 8k rows), each scored on
# JevBench public plain and with two-order averaging. usage: batch4.sh  (runs the list at the bottom)
set -uo pipefail
cd /workspace/decider
export HF_HOME=/workspace/hf DECIDER_DATA=/workspace/decider/data DECIDER_CACHE=/workspace/decider/data/cache DECIDER_RUNS=/workspace/decider/runs
export DECIDER_RESULTS=/workspace/decider/results.tsv DECIDER_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo pod)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/logs
run() { local name=$1 torso=$2; shift 2; echo "=== $(date -u +%FT%TZ) start $name ($torso)"; uv run python train.py --out "runs/$name" --torso "$torso" "$@" > "runs/logs/$name.log" 2>&1; echo "=== $(date -u +%FT%TZ) exit $? $name"; grep -E '"dev_acc"|"heldout_acc"|"order_sens"|"train_s"' "runs/logs/$name.log" | tr -d '\n'; echo; }
jev() { local name=$1 torso=$2; for tta in 1 2; do local nm=$name; [ $tta = 2 ] && nm=$name-tta2; scripts/jevbench.sh "$nm" --torso "$torso" --run "runs/$name" --tta $tta > "runs/logs/jev-$nm.log" 2>&1; echo "--- jevbench $nm: $(grep -oE "'accuracy': [0-9.]+|'n_correct': [0-9]+|'brier_mean': [0-9.]+" "runs/logs/jev-$nm.log" | tr '\n' ' ')"; done; }
T3=openbmb/MiniCPM5-1B-Base; T4=Qwen/Qwen3.5-2B
run tierB-minicpm1b-s0 $T3 --seed 0 --epochs 2 --note "MiniCPM5-1B-Base torso: LN16 head + LoRA r16, 2 epochs, 8k"; jev tierB-minicpm1b-s0 $T3
run tierB-q2b-instruct-s0 $T4 --seed 0 --epochs 2 --note "Qwen3.5-2B (instruct) torso: LN16 head + LoRA r16, 2 epochs, 8k"; jev tierB-q2b-instruct-s0 $T4
echo "=== batch4 done $(date -u +%FT%TZ)"
