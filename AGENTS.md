# pat-gen-opt

Optimization of PATGEN pattern-generation hyperparameters, plus benchmark datasets (github.com/ondrejsojka/pat-gen-opt). A local checkout may still live in a directory named `hyph-bench`, the repository's pre-rename name.

Read `README.md` for setup, dataset layout, optimizer workflows, and dataset licenses. The paper reproduction and audit protocol lives in `docs/REPRODUCING.md`.

Use `uv run ...` for Python commands. The project requires Python >=3.10.
Patgen must be available. Pass a non-default binary with `--patgen`, or `PATGEN_BIN` where a batch script supports it.
Large datasets need the high-capacity build (`/home/dev/patgen-10x` here); the packaged `patgen` aborts on them.

Key commands:

- Use `uv run python -m scripts.per_level_search ...` for the canonical per-level held-out search (per-level weights and thresholds; defaults match the paper protocol).
- Use `uv run python -m scripts.optimize_validation ...` and `uv run python -m scripts.optimize_shared_parameters ...` for the reduced held-out searches.
- Use `uv run python -m scripts.cross_validate ...` for cross-validation.
- `scripts.optimize` is in-sample and legacy; do not use it for camera-ready results.
- Use `make translate_all` to regenerate translate files.

There is no test suite and no linter. Verify changes with a short smoke run on a small dataset such as `th/orchid`, not with invented test commands.

Do not commit changes unless explicitly requested.
Do not launch long optimization sweeps, full benchmark runs, or dataset regeneration unless explicitly requested.
Avoid rewriting generated results, optimizer state, or large dataset files unless the task specifically requires it.
