#!/usr/bin/env bash
# Re-run HPO comparison (GP vs Random vs TPE). Dataset-proportional trie
# normalization (N = |D|, number of wordlist lines per dataset) is now the
# default in scripts.compare_hpo_methods.
#
# IMPORTANT: No --reuse-existing-gp here. All three methods run fresh
# under the corrected objective for a fair comparison.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -eq 0 ]]; then
  DATASETS=(
    "cssk/cshyphen"
    "cs/cshyphen_cstenten"
    "de/wortliste"
    "th/orchid"
    "el/wiktionary"
    "es/wiktionary"
    "ms/wiktionary"
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
  --output-dir results/hpo_baselines_proportional
