#!/usr/bin/env bash
# Run the final 17-dataset GPopt8 matrix, two datasets at a time.
# Each dataset uses five PATGEN workers; two concurrent runs occupy all 12 CPUs
# with ten PATGEN workers and two GP coordinator processes.

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PATGEN_BIN="${PATGEN_BIN:-/home/dev/patgen-10x}"
OUTPUT_DIR="${OUTPUT_DIR:-results/gpopt8}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
LOG_DIR="${OUTPUT_DIR}/_logs"
STATUS_FILE="${LOG_DIR}/run-status.tsv"
mkdir -p "${LOG_DIR}"

DATASETS=(
  cssk/cshyphen
  cs/cshyphen_cstenten
  cs/cshyphen_ujc
  cs/wiktionary
  de/wiktionary
  de/wortliste
  el/wiktionary
  es/wiktionary
  is/hyphenation-is
  it/wiktionary
  nl/wiktionary
  pl/wiktionary
  pt/wiktionary
  ru/wiktionary
  th/orchid
  tr/wiktionary
  uk/wiktionary
)

printf 'dataset\tstarted_utc\tfinished_utc\texit\n' > "${STATUS_FILE}"

run_dataset() {
  local dataset="$1"
  local safe_name="${dataset//\//_}"
  local history="${OUTPUT_DIR}/${dataset}/final_history.csv"
  local patterns="${OUTPUT_DIR}/${dataset}/final_patterns.pat"
  local log="${LOG_DIR}/${safe_name}.log"
  local started finished rc rows

  rows=0
  if [[ -f "${history}" ]]; then
    rows=$(($(wc -l < "${history}") - 1))
  fi
  if [[ "${rows}" -eq 153 && -s "${patterns}" ]]; then
    started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s\t%s\t%s\t%s\n' "${dataset}" "${started}" "${started}" "skipped-complete" >> "${STATUS_FILE}"
    echo "[${started}] SKIP complete ${dataset}"
    return 0
  fi

  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[${started}] START ${dataset}"
  PATGEN_OPT_WEIGHT_SPACE=legacy uv run python -u -m scripts.per_level_search \
    --lang "${dataset}" \
    --patgen "${PATGEN_BIN}" \
    --output-dir "${OUTPUT_DIR}" \
    --iterations 30 \
    --batch-size 5 \
    --seed 42 \
    --ucb-kappa 2.5 \
    --min-threshold 1 \
    --max-threshold 5 \
    --objective f17_trie \
    --trie-weight 0.0005 \
    --export-final-patterns \
    2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\n' "${dataset}" "${started}" "${finished}" "${rc}" >> "${STATUS_FILE}"
  echo "[${finished}] END ${dataset} exit=${rc}"
  return "${rc}"
}

running=0
for dataset in "${DATASETS[@]}"; do
  run_dataset "${dataset}" &
  running=$((running + 1))
  if [[ "${running}" -ge "${MAX_PARALLEL}" ]]; then
    wait -n
    running=$((running - 1))
  fi
done
wait

finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[${finished}] GPopt8 QUEUE COMPLETE"
