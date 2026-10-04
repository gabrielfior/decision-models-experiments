#!/usr/bin/env bash
# After batch5: the small Qwen torso on the 9.9k-row recipe (size/accuracy curve for the publishable model), then stop.
set -uo pipefail
cd /workspace/decider
export HF_HOME=/workspace/hf DECIDER_DATA=/workspace/decider/data DECIDER_CACHE=/workspace/decider/data/cache DECIDER_RUNS=/workspace/decider/runs
export DECIDER_RESULTS=/workspace/decider/results.tsv DECIDER_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo pod)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { local name=$1 torso=$2; shift 2; echo "=== $(date -u +%FT%TZ) start $name ($torso)"; uv run python train.py --out "runs/$name" --torso "$torso" "$@" > "runs/logs/$name.log" 2>&1; echo "=== $(date -u +%FT%TZ) exit $? $name"; grep -E '"dev_acc"|"dev_brier"|"heldout_acc"|"order_sens"|"train_s"' "runs/logs/$name.log" | tr -d '\n'; echo; }
jev() { local name=$1 torso=$2; for tta in 1 2; do local nm=$name; [ $tta = 2 ] && nm=$name-tta2; scripts/jevbench.sh "$nm" --torso "$torso" --run "runs/$name" --tta $tta > "runs/logs/jev-$nm.log" 2>&1; echo "--- jevbench $nm: $(grep -oE "'n_correct': [0-9]+|'brier_mean': [0-9.]+" "runs/logs/jev-$nm.log" | head -2 | tr '\n' ' ')"; done; }
Q=Qwen/Qwen3.5-0.8B-Base
run tierB-08b-9k9-s0 $Q --seed 0 --epochs 2 --train-splits train,heldout --calib-split dev --note "0.8B data lever: LN16 head + LoRA r16, 2 epochs, 9.9k rows (train+heldout), temps on dev"; jev tierB-08b-9k9-s0 $Q
echo "=== batch6 done $(date -u +%FT%TZ)"
