#!/usr/bin/env bash
# Print the RunPod account balance. Needs RUNPOD_API_KEY in the environment.
set -euo pipefail
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is not set}"
curl -s "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
  -H 'content-type: application/json' \
  --data '{"query":"query { myself { clientBalance spendLimit currentSpendPerHr } }"}'
echo
