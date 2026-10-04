#!/usr/bin/env bash
# Run the stock JevBench harness (public 231) against a decider served by serve.py.
# Usage: scripts/jevbench.sh <run-name> [serve.py args...]
#   scripts/jevbench.sh baseline-tierA --head runs/tierA/head.pt
#   scripts/jevbench.sh baseline-tierB --run runs/<dir>
# Output lands in runs/jevbench/<run-name>/ (outside the jevbench/ submodule, as the harness requires).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}           # on Modal the container has no uv: PY=python
NAME=${1:?run name}; shift
OUT=runs/jevbench/$NAME; mkdir -p "$OUT"
TASKS=$OUT/tasks.jsonl
cat jevbench/datasets/public/original.jsonl jevbench/datasets/public/easy.jsonl jevbench/datasets/public/hard.jsonl > "$TASKS"
echo "tasks: $(wc -l < "$TASKS")"

$PY serve.py --port 8811 --name "$NAME" "$@" > "$OUT/serve.log" 2>&1 &
SERVE=$!
trap 'kill $SERVE 2>/dev/null || true' EXIT
for i in $(seq 1 600); do grep -q "serving" "$OUT/serve.log" 2>/dev/null && break; sleep 1; done
grep -q "serving" "$OUT/serve.log" || { echo "server did not start"; tail -20 "$OUT/serve.log"; exit 1; }

export PYTHONPATH=jevbench
T0=$(date +%s)
$PY -m jevbench.cli run --tasks "$TASKS" --adapter typesafe --endpoint http://127.0.0.1:8811 \
  --key-env '' --model "$NAME" --results "$OUT/results.jsonl" --raw-dir "$OUT/raw" --ledger "$OUT/ledger.jsonl" \
  --cost-basis no_billable_account_public_endpoint --reserve-usd 0 --manifest "$OUT/manifest.json"
$PY -m jevbench.cli summarize --tasks "$TASKS" --results "$OUT/results.jsonl" --public-export "$OUT/summary.json"
echo "wall $(( $(date +%s) - T0 ))s"
python3 - "$OUT/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
keys = ["accuracy", "macro_accuracy", "n_correct", "n_scorable", "brier_mean", "ece", "ordinal_mae", "schema_validity", "paraphrase_consistency"]
print({k: s.get(k) for k in keys})
print("latency", s.get("latency"))
print("splits", {k: (v.get("accuracy") if isinstance(v, dict) else v) for k, v in (s.get("splits") or {}).items()})
PY
