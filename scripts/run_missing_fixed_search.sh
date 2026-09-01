#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/results/gpoptval4_missing_logs"
OUTPUT_DIR="${ROOT_DIR}/results/gpoptval4"
PATGEN_BIN="${PATGEN_BIN:-patgen}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

run_dataset() {
  local LANG="$1"
  LOG_NAME="${LANG//\//_}"

  local LANG_DIR="${OUTPUT_DIR}/${LANG}"
  local HISTORY="${LANG_DIR}/gpoptval4_history.csv"
  local FINAL_PATTERNS="${LANG_DIR}/gpoptval4_final.pat"
  if [[ -s "${FINAL_PATTERNS}" && -f "${HISTORY}" ]] && [[ "$(wc -l < "${HISTORY}")" -ge 154 ]]; then
    echo "Skipping complete fixed-parameter run for ${LANG}"
    return
  fi

  rm -rf "${LANG_DIR}"
  echo "Starting fixed-parameter run for ${LANG} at $(date '+%Y-%m-%d %H:%M:%S')"
  (
    cd "${ROOT_DIR}"
    uv run python -m scripts.optimize_validation \
      --lang "${LANG}" \
      --output-dir "${OUTPUT_DIR}" \
      --iterations 30 \
      --batch-size 5 \
      --objective f17_trie \
      --good-weight 3 \
      --max-bad-weight 30 \
      --max-threshold 1 \
      --ucb-kappa 2.5 \
      --trie-weight 0.0005 \
      --patgen "${PATGEN_BIN}" \
      --export-final-patterns
  ) > "${LOG_DIR}/${LOG_NAME}.log" 2>&1
  echo "Finished fixed-parameter run for ${LANG} at $(date '+%Y-%m-%d %H:%M:%S')"
}

run_queue() {
  for LANG in "$@"; do
    run_dataset "${LANG}"
  done
}

run_queue \
  "de/wiktionary" \
  "it/wiktionary" \
  "pl/wiktionary" \
  "pt/wiktionary" &

run_queue \
  "el/wiktionary" \
  "nl/wiktionary" &

wait

echo "All missing fixed-parameter runs finished at $(date '+%Y-%m-%d %H:%M:%S')"
