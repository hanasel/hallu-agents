#!/usr/bin/env bash
# Runs the verification pilot for summarization AND qa, each to its own file,
# each resumable across quota windows. Data2txt already done separately.
# Safe to Ctrl-C and re-run: --resume skips completed uids, cache replays.

set -u
cd "$(dirname "$0")/.."          # repo root

# --- config ---
N=100
MAX_CLAIMS=25
MAX_ATTEMPTS=200
SLEEP_BETWEEN=90
LOG=outputs/run_summ_qa.log

# Ordered list of tasks to run. Each gets its own output file.
TASKS="summarization qa"

if [ -z "${TOGETHER_API_KEY:-}" ]; then
  echo "TOGETHER_API_KEY not set. Run: export TOGETHER_API_KEY=..." >&2
  exit 1
fi

echo "=== run started $(date) ===" | tee -a "$LOG"

for TASK in $TASKS; do
  OUT="outputs/rgt_verify_together_${TASK}.jsonl"
  echo "" | tee -a "$LOG"
  echo "########## TASK: $TASK -> $OUT ##########" | tee -a "$LOG"

  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "--- $TASK attempt $attempt at $(date) ---" | tee -a "$LOG"

    python scripts/rgt_verify_pilot.py \
      --task "$TASK" \
      --n "$N" \
      --max-claims "$MAX_CLAIMS" \
      --diagnose-full \
      --out "$OUT" 2>&1 | tee -a "$LOG"

    done_count=$(wc -l < "$OUT" 2>/dev/null || echo 0)
    echo "--- $TASK: scored $done_count / $N ---" | tee -a "$LOG"

    if [ "$done_count" -ge "$N" ]; then
      echo "=== $TASK COMPLETE at $(date): $done_count responses ===" | tee -a "$LOG"
      break
    fi

    echo "--- $TASK incomplete (likely quota); sleeping ${SLEEP_BETWEEN}s ---" | tee -a "$LOG"
    sleep "$SLEEP_BETWEEN"
  done
done

echo "" | tee -a "$LOG"
echo "=== ALL TASKS DONE at $(date) ===" | tee -a "$LOG"