#!/usr/bin/env bash
# Run the 17-dataset shared-parameter search sequentially.
# One run uses five PATGEN workers. Complete 153-evaluation histories are skipped.

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PATGEN_BIN="${PATGEN_BIN:-patgen}"
OUTPUT_DIR="${OUTPUT_DIR:-results/shared_parameter_search}"
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

history_rows() {
  local history="$1"
  if [[ ! -f "${history}" ]]; then
    echo 0
    return
  fi
  local lines
  lines="$(wc -l < "${history}")"
  echo $((lines - 1))
}

if [[ ! -f "${STATUS_FILE}" ]]; then
  printf 'dataset\tstarted_utc\tfinished_utc\texit\n' > "${STATUS_FILE}"
fi

for dataset in "${DATASETS[@]}"; do
  safe_name="${dataset//\//_}"
  history="${OUTPUT_DIR}/${dataset}/wider_history.csv"
  patterns="${OUTPUT_DIR}/${dataset}/wider_final.pat"
  log="${LOG_DIR}/${safe_name}.log"

  if [[ "$(history_rows "${history}")" -eq 153 && -s "${patterns}" ]]; then
    echo "SKIP complete ${dataset}"
    continue
  fi

  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[${started}] START ${dataset}"
  uv run python -u -m scripts.optimize_shared_parameters \
    --lang "${dataset}" \
    --patgen "${PATGEN_BIN}" \
    --output-dir "${OUTPUT_DIR}" \
    --iterations 30 \
    --batch-size 5 \
    --seed 42 \
    --ucb-kappa 2.5 \
    --objective f17_trie \
    --trie-weight 0.0005 \
    --export-final-patterns \
    2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\n' \
    "${dataset}" "${started}" "${finished}" "${rc}" >> "${STATUS_FILE}"
  echo "[${finished}] END ${dataset} exit=${rc}"
done
