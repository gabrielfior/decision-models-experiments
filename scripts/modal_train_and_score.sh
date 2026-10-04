#!/usr/bin/env bash
# Train on Modal (with retry), find the run dir on the volume by its note slug, then score it plain and TTA2.
# usage: scripts/modal_train_and_score.sh <log> <score-name> <seed> "<note>" "<extra train.py flags>" [torso]
LOG=$1; NAME=$2; SEED=$3; NOTE=$4; EXTRA=$5; TORSO=${6:-Qwen/Qwen3.5-2B-Base}
export MODAL_BUDGET_USD=${MODAL_BUDGET_USD:-29.4}
scripts/modal_retry.sh "$LOG" 2 compute/modal_app.py::train_lora --seed "$SEED" --note "$NOTE" --extra "$EXTRA" || { echo "train failed" >> "$LOG"; exit 1; }
SLUG=$(python3 -c "import sys; print(sys.argv[1][:40].replace(' ','_').replace('/','-'))" "$NOTE")
RUN=$(uv run modal volume ls decider-data runs/ 2>/dev/null | grep -F "$SLUG" | grep -E -- "-s${SEED}-" | grep -oE "runs/[^ │]*" | sed 's|^runs/||' | sort | tail -1)
echo "run=$RUN" >> "$LOG"
[ -n "$RUN" ] || { echo "run dir not found" >> "$LOG"; exit 1; }
scripts/modal_retry.sh "$LOG" 2 compute/modal_app.py::score --name "$NAME" --runs "$RUN" --torso "$TORSO" --tta 1
scripts/modal_retry.sh "$LOG" 2 compute/modal_app.py::score --name "$NAME-tta2" --runs "$RUN" --torso "$TORSO" --tta 2
echo "done $NAME" >> "$LOG"
