cd /workspace/decider
while ! grep -q "batch6 done" runs/logs/batch6.out 2>/dev/null; do sleep 30; done
bash runpod/batch7.sh > runs/logs/batch7.out 2>&1
