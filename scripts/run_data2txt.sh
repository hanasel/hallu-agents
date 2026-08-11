#!/usr/bin/env bash
# Unattended, resumable Together verification run.
# Re-invokes the cap-25 command until all 100 responses are scored.
# Safe to Ctrl-C and restart: --resume skips completed uids, cache replays.

set -u
cd "$(dirname "$0")/.."          # repo root, regardless of where it's called from

# --- config ---
TASK=data2txt
N=100
OUT=outputs/rgt_verify_together_data2txt.jsonl
LOG=outputs/run_${TASK}.log
MAX_ATTEMPTS=200                 # generous; loop exits early when the run completes
SLEEP_BETWEEN=90                 # seconds to wait after an API-error stop before retrying

# --- key check: fail loudly NOW, not 3 hours in ---
if [ -z "${TOGETHER_API_KEY:-}" ]; then
  echo "TOGETHER_API_KEY not set. Run: export TOGETHER_API_KEY=..." >&2
  exit 1
fi

echo "=== run started $(date) ===" | tee -a "$LOG"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "--- attempt $attempt at $(date) ---" | tee -a "$LOG"

  python scripts/rgt_verify_pilot.py \
    --task "$TASK" \
    --n "$N" \
    --max-claims 25 \
    --diagnose-full \
    --out "$OUT" 2>&1 | tee -a "$LOG"

  # How many distinct responses have we scored so far?
  done_count=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  echo "--- scored $done_count / $N so far ---" | tee -a "$LOG"

  if [ "$done_count" -ge "$N" ]; then
    echo "=== COMPLETE at $(date): $done_count responses ===" | tee -a "$LOG"
    break
  fi

  echo "--- incomplete (likely quota); sleeping ${SLEEP_BETWEEN}s ---" | tee -a "$LOG"
  sleep "$SLEEP_BETWEEN"
done