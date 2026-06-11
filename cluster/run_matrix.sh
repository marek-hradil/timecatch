#!/bin/bash
# Submits all 4 experiments × 5 datasets (skipping corrupt-detect-mtl-aqa, kept from a prior run),
# polls until done, then consolidates every results.csv into a flat folder for easy rsync-back.
set -uo pipefail   # not -e: keep going if a single submit fails
cd "$(dirname "$0")/.."
source cluster/env.sh

RESULTS_DIR="$REMOTE_DIR/results-qwen2-5-vl-7b"

# Rename the prior mtl-aqa+corrupt-detect output dir for naming consistency.
if [ -d outputs/corrupt_detect_2026-05-17_23-11-03 ]; then
  mv outputs/corrupt_detect_2026-05-17_23-11-03 outputs/corrupt_detect_mtl-aqa_2026-05-17_23-11-03
  echo "renamed prior corrupt_detect dir → corrupt_detect_mtl-aqa"
fi

declare -a yamls=(
  corrupt-detect-clevrer corrupt-detect-craft corrupt-detect-craft-long corrupt-detect-drive-lm
  corrupt-localize-clevrer corrupt-localize-craft corrupt-localize-craft-long corrupt-localize-drive-lm corrupt-localize-mtl-aqa
  swap-detect-clevrer swap-detect-craft swap-detect-craft-long swap-detect-drive-lm swap-detect-mtl-aqa
  swap-localize-clevrer swap-localize-craft swap-localize-craft-long swap-localize-drive-lm swap-localize-mtl-aqa
)

declare -A jobs
echo "=== submitting 19 jobs ==="
for name in "${yamls[@]}"; do
  yaml="config/experiments/${name}.yaml"
  jobid=$(bash cluster/submit.sh "$yaml" 2>&1)
  jobs[$name]=$jobid
  echo "  $name → $jobid"
done

echo ""
echo "=== polling every 30s until all complete ==="
prev=""
start=$(date +%s)
while true; do
  remaining=0
  running=0
  for name in "${yamls[@]}"; do
    jid=${jobs[$name]}
    state=$(squeue -u $USER -j "$jid" -h -o %T 2>/dev/null || true)
    if [ -n "$state" ]; then
      remaining=$((remaining + 1))
      [ "$state" = "RUNNING" ] && running=$((running + 1))
    fi
  done
  elapsed=$(( $(date +%s) - start ))
  status="$(date +%H:%M:%S) elapsed=${elapsed}s remaining=${remaining}/${#yamls[@]} running=${running}"
  if [ "$status" != "$prev" ]; then
    echo "$status"
    prev="$status"
  fi
  [ $remaining -eq 0 ] && break
  sleep 30
done

echo ""
echo "=== final sacct ==="
for name in "${yamls[@]}"; do
  jid=${jobs[$name]}
  result=$(sacct -j "$jid" -X --format=State,ExitCode,Elapsed --noheader 2>/dev/null | head -1 | tr -s ' ')
  echo "  $name [$jid] $result"
done

echo ""
echo "=== collecting results into $RESULTS_DIR ==="
mkdir -p "$RESULTS_DIR"
for script in corrupt_detect corrupt_localize swap_detect swap_localize; do
  for ds in clevrer craft craft-long drive-lm mtl-aqa; do
    latest=$(ls -td outputs/${script}_${ds}_* 2>/dev/null | head -1)
    if [ -n "$latest" ] && [ -f "$latest/results.csv" ]; then
      cp "$latest/results.csv" "$RESULTS_DIR/${script}_${ds}.csv"
      rows=$(($(wc -l < "$latest/results.csv") - 1))
      echo "  ${script}_${ds}.csv ($rows rows) ← $latest"
    else
      echo "  ${script}_${ds}.csv MISSING (no matching dir)"
    fi
  done
done

echo ""
echo "=== $RESULTS_DIR contents ==="
ls -la "$RESULTS_DIR"
echo ""
echo "DONE."
echo "To pull results to laptop:"
echo "  rsync -avz transfer1.bsc.es:$RESULTS_DIR/ ./results/qwen2-5-vl-7b/"
