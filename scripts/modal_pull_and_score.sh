#!/usr/bin/env bash
# Pull a finished Modal run dir to runs/modal/ and score it on JevBench public locally (MPS), plain and TTA2.
# usage: scripts/modal_pull_and_score.sh <volume run dir name> <score-name> <torso>
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=$1; NAME=$2; TORSO=$3
mkdir -p runs/modal
uv run modal volume get decider-data "runs/$RUN" runs/modal/ --force >/dev/null 2>&1 || { echo "pull failed $RUN"; exit 1; }
export PYTORCH_ENABLE_MPS_FALLBACK=1
for tta in 1 2; do
  nm=$NAME; [ $tta = 2 ] && nm=$NAME-tta2
  scripts/jevbench.sh "$nm" --torso "$TORSO" --run "runs/modal/$RUN" --tta $tta > "runs/jevbench/$nm.log" 2>&1
  echo "--- $nm: $(grep -oE "'accuracy': [0-9.]+|'n_correct': [0-9]+|'brier_mean': [0-9.]+" "runs/jevbench/$nm.log" | tr '\n' ' ')"
done
