#!/bin/bash
# Legacy in-sample optimization queue (scripts.optimize); kept for history.
# Do not use for camera-ready results; see scripts/run_full_search.sh.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p optimization_logs
mkdir -p results

LANGS=(
    "cssk/cshyphen"
    "cs/cshyphen_cstenten"
    "cs/cshyphen_ujc"
    "cs/wiktionary"
    "de/wiktionary"
    "de/wortliste"
    "el/wiktionary"
    "es/wiktionary"
    "is/hyphenation-is"
    "it/wiktionary"
    "nl/wiktionary"
    "pl/wiktionary"
    "pt/wiktionary"
    "ru/wiktionary"
    "th/orchid"
    "tr/wiktionary"
    "uk/wiktionary"
)

for LANG in "${LANGS[@]}"; do
    LOG_NAME=$(echo $LANG | tr '/' '_')
    echo "Starting optimization for $LANG..."
    uv run python -m scripts.optimize \
        --lang "$LANG" \
        --iterations 100 \
        --batch-size 5 \
        --objective f17_trie \
        --good-weight 3 \
        --max-bad-weight 30 \
        --max-threshold 1 \
        --ucb-kappa 2.5 \
        --trie-weight 0.0005 \
        --resume \
        > "optimization_logs/${LOG_NAME}.log" 2>&1
    echo "Finished optimization for $LANG."
done
