#!/usr/bin/env bash
# Threshold ablation driver: completes the two incomplete fixed1 GPoptval4
# baselines, then runs the threshold-mode x method ablation queue across all
# 18 datasets with two concurrent lanes (2 x --workers 3 = 6 CPU cores).
#
# Idempotent: a run with a complete history.csv (budget rows) is skipped, an
# interrupted run is resumed from its saved state.
#
# Usage:  bash scripts/run_threshold_ablation.sh [--smoke]

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PATGEN_BIN="${PATGEN_BIN:-patgen}"
ABL_DIR="${ROOT_DIR}/results/threshold_ablation"
BASE_DIR="${ROOT_DIR}/results/gpoptval4"
LOG_DIR="${ABL_DIR}/logs"
WORKERS="${WORKERS:-3}"
LANES="${LANES:-2}"
BUDGET=153  # 30 rounds x 5 + 3 final
mkdir -p "${LOG_DIR}"

# Big datasets first: bottlenecks and failures surface early.
# (Endgame edit: with all big-dataset arms already complete, the TPE-perlayer
# block reuses this list; order it by descending GP-perlayer delta so the most
# informative TPE replications land first.)
LANGS=(
  ms/wiktionary
  uk/wiktionary
  is/hyphenation-is
  cs/wiktionary
  cs/cshyphen_ujc
  ru/wiktionary
  pl/wiktionary
  el/wiktionary
  it/wiktionary
  th/orchid
  tr/wiktionary
  pt/wiktionary
  nl/wiktionary
  de/wortliste
  cssk/cshyphen
  cs/cshyphen_cstenten
  de/wiktionary
  es/wiktionary
)
MODES=(shared perlayer)
METHODS=(gp tpe random)

log() { echo "[$(date '+%F %T')] $*"; }

history_rows() {
  local f="$1"
  [[ -f "${f}" ]] && { local n; n=$(wc -l < "${f}"); echo $((n - 1)); } || echo 0
}

# --- Phase 1: complete the two incomplete fixed1 baselines -------------------
if [[ "${1:-}" != "--smoke" ]]; then
  for LANG in el/wiktionary de/wiktionary; do
    HIST="${BASE_DIR}/${LANG}/gpoptval4_history.csv"
    if [[ "$(history_rows "${HIST}")" -ge ${BUDGET} ]]; then
      log "baseline ${LANG}: already complete, skipping"
      continue
    fi
    LOG="${LOG_DIR}/baseline_$(echo "${LANG}" | tr / _).log"
    log "baseline ${LANG}: resuming fixed1 run (-> ${LOG})"
    uv run python -m scripts.optimize_validation \
      --lang "${LANG}" --patgen "${PATGEN_BIN}" \
      --iterations 30 --batch-size 5 --objective f17_trie \
      --good-weight 3 --max-bad-weight 30 --max-threshold 1 \
      --ucb-kappa 2.5 --trie-weight 0.0005 --resume \
      >> "${LOG}" 2>&1
    rc=$?
    log "baseline ${LANG}: exit=${rc} rows=$(history_rows "${HIST}")"
  done
fi

# --- Phase 2: ablation queue --------------------------------------------------
run_ablation_job() {
  local LANG="$1" MODE="$2" METHOD="$3"
  local NAME
  NAME="$(echo "${LANG}" | tr / _)_${MODE}_${METHOD}"
  local RUN_DIR="${ABL_DIR}/${LANG}/${MODE}_${METHOD}"
  local HIST="${RUN_DIR}/history.csv"
  if [[ "$(history_rows "${HIST}")" -ge ${BUDGET} && -f "${RUN_DIR}/summary.json" ]]; then
    log "skip ${NAME}: complete"
    return 0
  fi
  local LOG="${LOG_DIR}/${NAME}.log"
  log "start ${NAME} (-> ${LOG})"
  uv run python -m scripts.threshold_ablation \
    --lang "${LANG}" --threshold-mode "${MODE}" --method "${METHOD}" \
    --patgen "${PATGEN_BIN}" --workers "${WORKERS}" --resume \
    >> "${LOG}" 2>&1
  local rc=$?
  log "done ${NAME}: exit=${rc} rows=$(history_rows "${HIST}")"
  return ${rc}
}

smoke_one() {  # tiny budgets, sequential
  local MODE="$1" METHOD="$2"
  uv run python -m scripts.threshold_ablation \
    --lang ms/wiktionary --threshold-mode "${MODE}" --method "${METHOD}" \
    --patgen "${PATGEN_BIN}" --workers 2 \
    --iterations 2 --batch-size 3 --final-exploitation 1 \
    --output-dir results/threshold_ablation_smoke
}

if [[ "${1:-}" == "--smoke" ]]; then
  for M in gp tpe random; do
    for MODE in fixed1 shared perlayer; do
      log "smoke ${MODE}/${M}"
      smoke_one "${MODE}" "${M}" || { log "SMOKE FAILED ${MODE}/${M}"; exit 1; }
    done
  done
  log "smoke OK"
  exit 0
fi

# FIFO queue over (method, mode, lang); modes grouped so cross-products
# complete in priority order even if the queue is cut short.
QUEUE=()
for METHOD in "${METHODS[@]}"; do
  for MODE in "${MODES[@]}"; do
    for LANG in "${LANGS[@]}"; do
      QUEUE+=("${LANG}|${MODE}|${METHOD}")
    done
  done
done

log "queue: ${#QUEUE[@]} ablation jobs, ${LANES} lanes x ${WORKERS} workers"
# Up to 3 sweeps: a crashed non-complete run is retried on the next sweep;
# completed runs are skipped, so extra sweeps are nearly free.
for SWEEP in 1 2 3; do
  log "sweep ${SWEEP} starting"
  running=0
  launched=0
  for JOB in "${QUEUE[@]}"; do
    IFS='|' read -r LANG MODE METHOD <<< "${JOB}"
    NAME="$(echo "${LANG}" | tr / _)_${MODE}_${METHOD}"
    HIST="${ABL_DIR}/${LANG}/${MODE}_${METHOD}/history.csv"
    if [[ "$(history_rows "${HIST}")" -ge ${BUDGET} && \
          -f "${ABL_DIR}/${LANG}/${MODE}_${METHOD}/summary.json" ]]; then
      continue
    fi
    run_ablation_job "${LANG}" "${MODE}" "${METHOD}" &
    running=$((running + 1))
    launched=$((launched + 1))
    if [[ ${running} -ge ${LANES} ]]; then
      wait -n
      running=$((running - 1))
    fi
  done
  wait
  log "sweep ${SWEEP} done (${launched} jobs launched)"
  if [[ ${launched} -eq 0 ]]; then
    break
  fi
done
log "ablation queue drained"
