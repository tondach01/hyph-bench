#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -eq 0 ]]; then
  DATASETS=(
    "el/wiktionary"
    "th/orchid"
  )
else
  DATASETS=("$@")
fi

exec uv run python -m scripts.compare_hpo_methods \
  --datasets "${DATASETS[@]}" \
  --methods gp random tpe \
  --objective f17_trie \
  --iterations 100 \
  --batch-size 1 \
  --good-weight 3 \
  --max-bad-weight 30 \
  --max-threshold 1 \
  --ucb-kappa 2.5 \
  --trie-weight 0.0005 \
  --nfold 10 \
  --reuse-existing-gp \
  --output-dir results/hpo_baselines
