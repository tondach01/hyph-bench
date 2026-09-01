# pat-gen-opt: Pattern Generation Optimization

### The Art of Hierarchical Competing Patterns: Gaussian Process Optimization of Hyphenation

`pat-gen-opt` automates the tuning of PATGEN, Liang's generator of competing hyphenation patterns. Give it a hyphenated word list; it searches PATGEN's per-level weights and thresholds with Gaussian-process optimization under a held-out train/validation/test protocol, and exports a TeX-compatible `.pat` file you can deploy.

> **Reproducing the paper?** If you are here to reproduce
> *The Art of Hierarchical Competing Patterns: Gaussian Process Optimization of Hyphenation*
> (Ondřej Sojka and Petr Sojka, EMNLP 2026), follow **[docs/REPRODUCING.md](docs/REPRODUCING.md)**.
> This README covers general use of the optimizer.

## Who this is for

- Maintainers of TeX hyphenation patterns who today hand-tune PATGEN parameter profiles and would rather have a reproducible, measured search.
- Anyone building patterns for a new language or corpus who has (or can build) a hyphenated word list.
- Researchers comparing pattern-generation setups: the repository bundles 17 curated and Wiktionary-derived datasets with a deterministic evaluation protocol.

## Who this is not for

- End users who just want ready-made hyphenation patterns: use the patterns shipped by [hyph-utf8](https://github.com/hyphenation/tex-hyphen) or your TeX distribution.
- Applications that need a runtime hyphenation library; this repository generates and evaluates patterns, it is not a hyphenation engine for documents.
- Languages without a hyphenated word list. PATGEN learns from examples; without training data there is nothing to optimize.

This is not a PATGEN replacement: the optimizer drives an unmodified TeX Live `patgen` binary. The paper runs use a build whose only change is raised Web2C capacity limits, so that the largest datasets fit; the algorithm is stock PATGEN either way. The exact limits and how to build them are in [docs/REPRODUCING.md](docs/REPRODUCING.md#requirements).

## The measure

The optimization objective is

$$
F_{1/7} - 0.0005\,\frac{\text{trie nodes}}{|D|}.
$$

$F_{1/7}$ is an F-score that weights precision far more strongly than recall, because in typesetting an incorrect hyphen is usually much worse than a missed optional break point. The second term penalizes pattern-trie size, normalized by dataset size $|D|$ so the compactness pressure is comparable across languages. The paper motivates and validates both choices.

Every reported score is computed once on a held-out test split that the search never sees. The default splitter groups entries by normalized, case-folded surface form, ranks the groups by a seed-42 SHA-256 digest, and assigns exact 8/1/1 train, validation, and test partitions. Source priorities such as those in `cssk/cshyphen` are expanded in training only; validation and test contain one entry per resolved word type. Parameters are selected on validation only.

## Requirements

- Python 3.10 or newer and [uv](https://docs.astral.sh/uv/); install the environment with `uv sync`.
- `patgen` from a recent TeX Live (`texlive-binaries` on Debian/Ubuntu). Large datasets need a higher-capacity PATGEN build; see [docs/REPRODUCING.md](docs/REPRODUCING.md#requirements).
- All datasets you need are tracked in git. The 806 MB Wiktionary dump archive (Git LFS) is only for regenerating datasets from scratch; skip it with `GIT_LFS_SKIP_SMUDGE=1 git clone ...`.

## Quickstart

The default and recommended workflow is the per-level search: one weight ratio and one threshold per PATGEN level, eight parameters in total for the standard four-level profile. The script defaults match the paper protocol. This command exercises the full train, validation, selection, held-out test, and pattern-export path on the bundled Thai ORCHID dataset:

```bash
uv run python -m scripts.per_level_search \
  --lang th/orchid \
  --patgen "$(command -v patgen)" \
  --iterations 1 \
  --batch-size 1 \
  --output-dir /tmp/pat-gen-opt-smoke \
  --export-final-patterns
```

It writes, under `/tmp/pat-gen-opt-smoke/th/orchid/`:

- `run_config.json`, recording the exact command and search space;
- `final_history.csv` with every evaluated profile, and resumable optimizer state;
- `selected_profile.json`, with validation selection and held-out test metrics;
- `final_patterns.pat`, the pattern set selected on validation data and evaluated once on the held-out test split;
- deterministic split files under `splits/`.

A one-iteration run is a smoke test, not a result. For a real search, drop `--iterations` and `--batch-size` to use the defaults: 30 GP iterations with batches of 5, followed by three exploitation evaluations.

To reproduce the paper's complete 17-dataset matrix, run:

```bash
PATGEN_BIN=/path/to/high-capacity/patgen bash scripts/run_full_search.sh
```

This is the canonical paper protocol and writes `results/gpopt260828/`.
`gpopt260828` is the dated artifact identifier, not a separate optimizer; the
method is the per-level GP search implemented by `scripts.per_level_search`.
The runner regenerates the expanded CSSK input when needed and refuses to treat
an older result using a different split protocol as complete.

## Optimize patterns for your own word list

A dataset lives under `data/<language>/<name>/` and consists of:

- a `.wlh` word list with one entry per line, hyphens at allowed break positions, for example `hy-phen-a-tion`;
- a matching `.tra` translate file defining PATGEN characters and left/right hyphen minima.

Generate the translate file:

```bash
uv run python -m scripts.make_tr data/xx/example/example.wlh
```

This writes `example.wlh.tra` with left and right minima of 2; override with `--left_hyphen_min` / `--right_hyphen_min` when the orthography requires different values.

Then run the search (either place files under `data/` and use `--lang`, or pass explicit `--wordlist`/`--translate` paths):

```bash
uv run python -m scripts.per_level_search \
  --lang xx/example \
  --patgen "$(command -v patgen)" \
  --output-dir results/final \
  --export-final-patterns
```

Split membership is content-derived and independent of input order. Preserve source order anyway: it is the deterministic final tie-breaker when duplicate surface forms have equally prioritized conflicting annotations.

Reduced searches with fewer parameters — four per-level weights with a shared threshold (`scripts.optimize_validation`), or shared weights across levels (`scripts.optimize_shared_parameters`) — are described in [docs/REPRODUCING.md](docs/REPRODUCING.md).

## Apply generated patterns

```bash
uv run python -m scripts.hyphenate_wordlist \
  --wordlist words.txt \
  --patterns results/final/xx/example/final_patterns.pat \
  --translate data/xx/example/example.wlh.tra \
  --output words.hyphenated.txt
```

The command uses the repository's Liang-pattern implementation; it does not substitute a language-specific dictionary from another library. The exported `.pat` file is directly usable as TeX `\patterns` input.

## Repository layout

- `data/`: hyphenated word lists and translate files for 17 datasets.
- `profiles/`: hand-tuned PATGEN baselines used for comparison.
- `scripts/`: preprocessing, optimization, evaluation, and reporting code. `scripts.per_level_search` is the canonical workflow; much of the rest serves the paper experiments.
- `results/`: paper run histories, selected profiles, ablations, and figures — historical evidence, not needed to use the optimizer.
- `docs/REPRODUCING.md`: the paper reproduction and audit protocol.
- `docs/future_work.md`: known deferred items — data-quality residues, tooling gaps, and planned experiments.

## Datasets and licenses

The repository combines datasets with different licenses. Preserve the attribution and license of each source when redistributing data or generated derivatives.

| Dataset | License | Source note |
|---|---|---|
| `cs/cshyphen_cstenten` | CC BY-NC-SA 3.0 | Czech–Slovak curated data |
| `cs/cshyphen_ujc` | MIT | Czech curated data |
| `cssk/cshyphen` | MIT | Weighted Czech–Slovak data |
| `de/wortliste` | MIT | German curated data |
| `is/hyphenation-is` | CC BY 4.0 | Icelandic curated data |
| `th/orchid` | CC BY-SA 4.0 | Licensed in 2025 from the public-domain ORCHID source |
| Wiktionary-derived datasets | CC BY-SA 4.0 | `cs`, `de`, `el`, `es`, `it`, `nl`, `pl`, `pt`, `ru`, and `tr` |
| `uk/wiktionary` | CC BY-SA 4.0 | Prepared for Sojka, O.: *Transfer Learning of Slavic Syllabification for Hyphenation Patterns*, Bachelor Thesis, Masaryk University, Brno, 2025 |
| `uk/dict_uk` | GPL-3.0 | Derived from [brown-uk/dict_uk](https://github.com/brown-uk/dict_uk) |

The original software in this repository is available under the [MIT License](LICENSE). Dataset files retain the separate licenses and attribution requirements listed above; the software license does not override those terms.

## Citation

If you use this software or the optimized patterns in your work, please cite:

```bibtex
@inproceedings{sojka-sojka-2026-competing-patterns,
  title     = {The Art of Hierarchical Competing Patterns:
               {Gaussian} Process Optimization of Hyphenation},
  author    = {Sojka, Ond\v{r}ej and Sojka, Petr},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods
               in Natural Language Processing (EMNLP)},
  year      = {2026},
  url       = {https://github.com/ondrejsojka/pat-gen-opt},
}
```

## Acknowledgments

This repository began as a fork of [hyph-bench](https://github.com/tondach01/hyph-bench) by Ondřej Metelka, whom we thank for assembling and curating the original datasets.
