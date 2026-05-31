#!/bin/bash
# Submit all experiments for ONE model across all 4 tasks × 4 datasets,
# poll until done, then collect results into results-<MODEL>/.
# Usage (on the cluster login node):
#   bash cluster/run_matrix_model.sh qwen3-vl-2b
#   bash cluster/run_matrix_model.sh qwen3-vl-4b
set -uo pipefail
cd "$(dirname "$0")/.."
source cluster/env.sh

MODEL="${1:-}"
if [ -z "$MODEL" ]; then
  echo "Usage: $0 <model>   (e.g. qwen3-vl-2b | qwen3-vl-4b)" >&2
  exit 1
fi

RESULTS_DIR="$REMOTE_DIR/results-${MODEL}"

declare -a tasks=(corrupt-detect corrupt-localize swap-detect swap-localize)
declare -a datasets=(clevrer craft drive-lm mtl-aqa)

declare -a yamls=()
for task in "${tasks[@]}"; do
  for ds in "${datasets[@]}"; do
    yamls+=("config/experiments/${task}-${ds}-${MODEL}.yaml")
  done
done

echo "=== submitting ${#yamls[@]} jobs for model=${MODEL} ==="
declare -A jobs
for yaml in "${yamls[@]}"; do
  if [ ! -f "$yaml" ]; then
    echo "  SKIP (missing): $yaml" >&2
    continue
  fi
  name=$(basename "$yaml" .yaml)
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
  for name in "${!jobs[@]}"; do
    jid=${jobs[$name]}
    state=$(squeue -u $USER -j "$jid" -h -o %T 2>/dev/null || true)
    if [ -n "$state" ]; then
      remaining=$((remaining + 1))
      [ "$state" = "RUNNING" ] && running=$((running + 1))
    fi
  done
  elapsed=$(( $(date +%s) - start ))
  status="$(date +%H:%M:%S) elapsed=${elapsed}s remaining=${remaining}/${#jobs[@]} running=${running}"
  if [ "$status" != "$prev" ]; then
    echo "$status"
    prev="$status"
  fi
  [ $remaining -eq 0 ] && break
  sleep 30
done

echo ""
echo "=== final sacct ==="
for name in "${!jobs[@]}"; do
  jid=${jobs[$name]}
  result=$(sacct -j "$jid" -X --format=State,ExitCode,Elapsed --noheader 2>/dev/null | head -1 | tr -s ' ')
  echo "  $name [$jid] $result"
done

echo ""
echo "=== collecting results into $RESULTS_DIR ==="
mkdir -p "$RESULTS_DIR"
for task in corrupt_detect corrupt_localize swap_detect swap_localize; do
  for ds in clevrer craft drive-lm mtl-aqa; do
    latest=$(ls -td outputs/${task}_${ds}_${MODEL}_* 2>/dev/null | head -1)
    if [ -n "$latest" ] && [ -f "$latest/results.csv" ]; then
      cp "$latest/results.csv" "$RESULTS_DIR/${task}_${ds}.csv"
      rows=$(($(wc -l < "$latest/results.csv") - 1))
      echo "  ${task}_${ds}.csv ($rows rows) ← $latest"
    else
      echo "  ${task}_${ds}.csv MISSING"
    fi
  done
done

echo ""
echo "=== $RESULTS_DIR contents ==="
ls -la "$RESULTS_DIR"
echo ""
echo "DONE."
echo "To pull results to laptop:"
echo "  rsync -avz transfer1.bsc.es:$RESULTS_DIR/ ./results-${MODEL}/"
