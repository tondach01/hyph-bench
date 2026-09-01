# Reproducing “The Art of Hierarchical Competing Patterns: Gaussian Process Optimization of Hyphenation” (EMNLP 2026)

This document is the paper artifact protocol. For general use of the optimizer, start from the [main README](../README.md).

## Requirements

You need:

- Python 3.10 or newer;
- [uv](https://docs.astral.sh/uv/);
- `patgen` from a recent TeX Live installation.

On Debian and Ubuntu, the `texlive-binaries` package provides `patgen`. Other TeX Live distributions usually install it with the core binaries.

Install the Python environment:

```bash
uv sync
```

The paper runs use `/home/dev/patgen-10x`, a local name for a TeX Live 2024 PATGEN build with the upstream Web2C capacity settings: a 10,000,000-entry pattern trie, a 5,000,000-entry count trie, and 40,800 output operations. A recent TeX Live source build applies these values through `texk/web2c/patgen.ch`. Smaller datasets work with the standard packaged binary. Large datasets require the higher-capacity build.

Pass a non-default binary with `--patgen /path/to/patgen` or set `PATGEN_BIN` where a batch script supports it.

The repository tracks the weighted `cssk/cshyphen` source (`.wlhw`) but not its expanded word list. `scripts/run_full_search.sh` regenerates the expansion automatically. Before an individual `cssk/cshyphen` command, run:

```bash
uv run python -m scripts.expand_weights data/cssk/cshyphen/cssk-all-weighted.wlhw
```

The canonical splitter reads the source priorities, resolves duplicate surface forms before splitting, and expands priorities in training only. Validation and test retain one entry per resolved word type.

## The final per-level search protocol

The final per-level search uses the following protocol for every manuscript dataset:

- deterministic, surface-form-disjoint 8/1/1 train, validation, and held-out test split: normalized word identities are ranked by a seed-42 SHA-256 digest;
- source priorities are expanded in training only; validation and test contain one entry per resolved word type;
- four independently selected weight ratios, one per PATGEN level;
- four independently selected thresholds, one per PATGEN level;
- weight ratios in `{1/5, 1/4, 1/3, 1/2, 1, ..., 30}`;
- thresholds in `[1,42]`;
- pattern ranges `[(1,4), (2,5), (2,6), (2,7)]`;
- 30 GP iterations with batches of 5, followed by three exploitation evaluations;
- seed 42 and UCB $\kappa=2.5$;
- proportional trie normalization by the number of resolved word types $|D|$ and `trie_weight=0.0005`;
- selection by the best observed validation objective;
- one held-out test evaluation after selection.

The dated repository identifier for the final 17-dataset paper run is `gpopt260828`. It is an artifact directory name, not a method name. The default `scripts.run_full_search.sh` command reproduces that matrix with the canonical `scripts.per_level_search` defaults and writes `results/gpopt260828/`.

Run the full matrix or fill missing datasets:

```bash
PATGEN_BIN=/path/to/high-capacity/patgen \
  bash scripts/run_full_search.sh
```

By default, the runner writes to `results/gpopt260828/`. Each complete dataset contains:

- `run_config.json`, recording the exact command and search space;
- `final_history.csv`, with 153 evaluated profiles;
- `selected_profile.json`, with validation selection and held-out metrics;
- `final_patterns.pat`, the selected deployable pattern set;
- deterministic split files and optimizer state.

The result directory is gitignored by default. For a paper release, add only the reviewed lightweight configurations, histories, selected profiles, patterns, and aggregate evidence; do not publish machine-specific optimizer pickle state or the split files.

A one-iteration smoke run (see the main README quickstart) exercises the same code path but does not reproduce a paper score. Use the full protocol above for reported results.

## Audit a published artifact set

The split files are a pure function of the source word list, so they are not published. Regenerate them and re-derive every reported held-out number from a clean clone with:

```bash
uv run python -m scripts.analyze_gpopt260828 \
  --results-dir results/gpopt260828 \
  --output-dir results/gpopt260828_analysis \
  --write-splits
```

Without `--write-splits` the analysis refuses to run against a run directory that has no split files. With it, `scripts.dataset_split.create_clean_split` rewrites the deterministic grouped 8/1/1 partition. The audit verifies exact seeded hash membership, zero cross-partition surface overlap, recorded split counts, and SHA-256 equality with the split hashes stored in an existing `bootstrap_ci.json`. A source word list that differs from the reported run fails the hash comparison before any PATGEN work starts.

The analysis regenerates the selected profile and both hand-tuned baselines with PATGEN, asserts that the regenerated held-out Good/Bad/Missed counts and $F_{1/7}$ reproduce `selected_profile.json` exactly, and writes `bootstrap_ci.json`, `bootstrap_ci_table.tex`, and `summary.json`.

## Historical and auxiliary experiments

Historical GPopt8 artifacts remain under `results/gpopt8/`. Its launcher pins `PATGEN_OPT_WEIGHT_SPACE=legacy`, preserving the original fractional choices `{1/3, 1/2}` and threshold range `[1,5]`. Other archived pre-cutover analyses explicitly use `scripts.legacy_split` to reproduce their original line-index partitions; canonical searches never use that splitter.

Frozen run records (`run_config.json` command strings and `_logs/`) may reference the module by its historical name `scripts.paper2_final_search`; the canonical implementation is now `scripts.per_level_search`.

Related scripts:

| Purpose | Command or script |
|---|---|
| Final per-level GP search | `python -m scripts.per_level_search` |
| Final 17-dataset queue | `scripts/run_full_search.sh` |
| Historical GPopt8 queue | `scripts/run_gpopt8.sh` |
| Random/TPE comparison queue | `scripts/run_hpo_baselines_8d.sh` (per-dataset: `python -m scripts.per_level_hpo_baselines`) |
| Legacy restricted-space comparison | `python -m scripts.compare_hpo_methods` |
| Reported-result audit | `python -m scripts.analyze_gpopt260828 --write-splits` |

The paper's budget-matched HPO comparison (`tab:hpo-baselines`) runs Random Search and TPE in the same 8-D per-level space, grouped split, and 153-evaluation budget as the final search; the GP column is the recorded main-experiment runs. Its published evidence lives under `results/hpo_baselines_grouped/` (run `OUTPUT_DIR=results/hpo_baselines_grouped bash scripts/run_hpo_baselines_8d.sh` to regenerate; seed 42 makes the runs deterministic). The hand-tuned column comes from the regenerated baselines in `bootstrap_ci.json`. An older restricted four-parameter comparison remains under `results/hpo_representative_150/`; it is superseded and must not be mixed with the 8-D results.

The threshold ablation under `results/threshold_ablation/` motivates searching thresholds per level at all: with thresholds in `[1,5]` and the GPoptval4 budget (153 evaluations per arm), per-level thresholds beat the fixed-at-1 baseline on 18 of 18 datasets under GP, and shared thresholds are weaker than per-level under every optimizer (see `SUMMARY.md`; histories, per-variant summaries, `ablation_summary.json`, and run logs are alongside). It is background evidence, not a paper table; the final protocol adopts per-level thresholds and extends the searched range to `[1,42]`.

## Reduced searches

Two reduced held-out workflows share the deterministic 8/1/1 protocol with the final per-level search:

- `scripts.optimize_validation` searches four per-level bad weights with a shared threshold and a fixed good weight:

```bash
uv run python -m scripts.optimize_validation \
  --lang xx/example \
  --patgen "$(command -v patgen)" \
  --iterations 30 \
  --batch-size 5 \
  --objective f17_trie \
  --good-weight 3 \
  --max-bad-weight 30 \
  --max-threshold 1 \
  --ucb-kappa 2.5 \
  --trie-weight 0.0005 \
  --output-dir results/gpoptval4 \
  --export-final-patterns
```

- `scripts.optimize_shared_parameters` searches four level-specific bad weights, one shared threshold, and one shared good weight:

```bash
uv run python -m scripts.optimize_shared_parameters \
  --lang xx/example \
  --wordlist /absolute/path/example.wlh \
  --translate /absolute/path/example.wlh.tra \
  --patgen "$(command -v patgen)" \
  --output-dir results/shared_parameter_search \
  --iterations 30 \
  --batch-size 5 \
  --seed 42 \
  --ucb-kappa 2.5 \
  --objective f17_trie \
  --trie-weight 0.0005 \
  --export-final-patterns
```

The default bounds are:

- each `bad_wt` in $[1,30]$;
- `threshold` in $\{1,2\}$;
- `good_wt` in $\{1,2,3,4,5\}$.

PATGEN evaluates five candidates in parallel during each iteration. A full run therefore uses five worker processes and 153 evaluations: 150 search evaluations plus three final exploitation evaluations.

## Preprocess the bundled datasets

All run inputs are tracked in git; you never need the raw Wiktionary dumps to run optimizations or the audit. The dump archive (`wikt_dump.zip`, 806 MB via Git LFS, ~10 GB extracted) is needed only to regenerate the Wiktionary-derived word lists from scratch. To skip downloading it at clone time:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone <repo-url>
```

and fetch it later, only if you regenerate datasets, with `git lfs pull --include=wikt_dump.zip`.

The Makefile provides bulk preprocessing targets:

```bash
make process_wikt       # Extract and convert JSONL dumps to .wlh word lists
make disambiguate_all   # Remove conflicting annotations
make translate_all      # Generate PATGEN .tra files (uses tracked inputs; no dump needed)
make stats_all_datasets # Report dataset statistics (no dump needed)
```

`make process_wikt` extracts each language's JSONL from the archive individually and deletes it after processing, so peak extra disk stays around the largest member (~3 GB for German) instead of the full ~10 GB. Set `KEEP_JSONL=1` to keep the extracted dumps for reruns.

## Known scope boundaries

- `scripts.per_level_search` is the final, canonical workflow; `scripts.optimize_validation` and `scripts.optimize_shared_parameters` are the reduced held-out workflows.
- `scripts.optimize` performs in-sample optimization and serves older experiments. Do not use it to reproduce held-out camera-ready results.
- Malay (`ms/wiktionary`) was dropped from the benchmark and is not part of the reported 17-dataset collection. The preprocessing and optimization queues no longer list it. Older ablation artifacts under `results/threshold_ablation/` and `results/hpo_representative_150/` still contain recorded Malay rows; those are historical evidence and are left untouched.
- The Wiktionary-derived word lists retain a small residue of non-letter characters, and the reported runs were trained on them as-is. Do not normalize the datasets to "clean them up" without renaming the artifact directory: every character is part of the PATGEN alphabet declared in the `.tra` file, so normalization changes the trie and invalidates comparability with the reported numbers. The residues, and the one real preprocessing defect among them, are catalogued in [future_work.md](future_work.md#data-quality).
