#!/usr/bin/env bash
# Rerun the HPO ablation (Random Search and TPE) in the full per-level 8-D
# search space on the five representative datasets, two jobs at a time.
# Each job uses five PATGEN workers; two concurrent jobs fill the machine.
# Compare against the recorded GP runs in results/gpopt260828 (same protocol).

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PATGEN_BIN="${PATGEN_BIN:-/home/dev/patgen-10x}"
OUTPUT_DIR="${OUTPUT_DIR:-results/hpo_baselines_8d}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
LOG_DIR="${OUTPUT_DIR}/_logs"
STATUS_FILE="${LOG_DIR}/run-status.tsv"
mkdir -p "${LOG_DIR}"

DATASETS=(
  cssk/cshyphen
  th/orchid
  ru/wiktionary
  nl/wiktionary
  de/wortliste
)
METHODS=(random tpe)

printf 'dataset\tmethod\tstarted_utc\tfinished_utc\texit\n' > "${STATUS_FILE}"

run_job() {
  local dataset="$1" method="$2"
  local safe_name="${dataset//\//_}_${method}"
  local selected="${OUTPUT_DIR}/${method}/${dataset}/selected_profile.json"
  local log="${LOG_DIR}/${safe_name}.log"
  local started finished rc

  if [[ -s "${selected}" ]]; then
    started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s\t%s\t%s\t%s\t%s\n' "${dataset}" "${method}" "${started}" "${started}" "skipped-complete" >> "${STATUS_FILE}"
    echo "[${started}] SKIP complete ${dataset} ${method}"
    return 0
  fi

  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[${started}] START ${dataset} ${method}"
  uv run python -u -m scripts.per_level_hpo_baselines \
    --lang "${dataset}" \
    --method "${method}" \
    --patgen "${PATGEN_BIN}" \
    --output-dir "${OUTPUT_DIR}" \
    --iterations 30 \
    --batch-size 5 \
    --final-extra 3 \
    --seed 42 \
    --min-threshold 1 \
    --max-threshold 42 \
    --objective f17_trie \
    --trie-weight 0.0005 \
    > "${log}" 2>&1
  rc=$?
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\t%s\n' "${dataset}" "${method}" "${started}" "${finished}" "${rc}" >> "${STATUS_FILE}"
  echo "[${finished}] END ${dataset} ${method} exit=${rc}"
  return "${rc}"
}

running=0
for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    run_job "${dataset}" "${method}" &
    running=$((running + 1))
    if [[ "${running}" -ge "${MAX_PARALLEL}" ]]; then
      wait -n
      running=$((running - 1))
    fi
  done
done
wait

finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[${finished}] HPO-8D QUEUE COMPLETE"
