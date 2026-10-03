#!/usr/bin/env bash
# Goal 2 batch on the pod: Qwen3.5-2B variants, each scored on JevBench public plain and with two-order averaging.
set -uo pipefail
cd /workspace/decider
export HF_HOME=/workspace/hf DECIDER_DATA=/workspace/decider/data DECIDER_CACHE=/workspace/decider/data/cache DECIDER_RUNS=/workspace/decider/runs
export DECIDER_RESULTS=/workspace/decider/results.tsv DECIDER_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo pod)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/logs
T=Qwen/Qwen3.5-2B-Base
run() { local name=$1; shift; echo "=== $(date -u +%FT%TZ) start $name"; uv run python train.py --out "runs/$name" --torso $T "$@" > "runs/logs/$name.log" 2>&1; echo "=== $(date -u +%FT%TZ) exit $? $name"; grep -E '"dev_acc"|"heldout_acc"|"order_sens"' "runs/logs/$name.log" | tr -d '\n'; echo; }
jev() { local name=$1; scripts/jevbench.sh "$name" --torso $T --run "runs/$name" > "runs/logs/jev-$name.log" 2>&1; echo "--- jevbench $name: $(grep -oE "'accuracy': [0-9.]+|'n_correct': [0-9]+|'brier_mean': [0-9.]+" "runs/logs/jev-$name.log" | tr '\n' ' ')";
         scripts/jevbench.sh "$name-tta" --torso $T --run "runs/$name" --tta > "runs/logs/jev-$name-tta.log" 2>&1; echo "--- jevbench $name TTA: $(grep -oE "'accuracy': [0-9.]+|'n_correct': [0-9]+|'brier_mean': [0-9.]+" "runs/logs/jev-$name-tta.log" | tr '\n' ' ')"; }
run tierB-2b-ln16-2ep  --seed 0 --epochs 2 --note "2B: LN16 head + LoRA r16, 2 epochs";                                              jev tierB-2b-ln16-2ep
run tierB-2b-xopt-1ep  --seed 0 --head-cfg '{"name":"cross_option","residual_pointer":true}' --note "2B: cross-option+residual head + LoRA r16, 1 epoch"; jev tierB-2b-xopt-1ep
run tierB-2b-ln16-s1   --seed 1 --note "2B: LN16 head + LoRA r16, 1 epoch, seed 1";                                                   jev tierB-2b-ln16-s1
echo "=== batch done $(date -u +%FT%TZ)"
