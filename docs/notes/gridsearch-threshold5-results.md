✦ The "sizeable gridsearch" run was unsuccessful against the baseline base.in parameters.

  Here is the analysis of the results found in hyph-bench/results/:

  1. The Grid Search Parameters
   * Search Space: 4-dimensional exhaustive grid for bad_weights at levels 1-4.
   * Values: Likely integer values from 1 to 9 for each level (total 6561 points).
   * Fixed Parameters:
       * good_weight: 1
       * threshold: 5 (Fixed for all levels)
       * Pattern ranges: Fixed (e.g., 1-4, 2-5, etc.)

  2. Comparison with Baseline


  ┌──────────────┬──────────────────────────┬──────────────────────────┬──────────────────────────────┐
  │ Metric       │ Baseline (base.in)       │ Grid Search Best         │ Difference                   │
  ├──────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────┤
  │ Parameters   │ 1, 2, 3, 4 (Threshold 1) │ 9, 1, 9, 2 (Threshold 5) │                              │
  │ Score (F1/7) │ 0.9993                   │ 0.9949                   │ Grid is worse                │
  │ Good (TP)    │ 311,602                  │ 305,317                  │ Grid found ~6k fewer         │
  │ Bad (FP)     │ 219                      │ 1,474                    │ Grid allowed ~7x more errors │
  │ Missed (FN)  │ 380                      │ 6,665                    │ Grid missed ~17x more        │
  │ Patterns     │ 7,653                    │ 2,343                    │ Grid generated ~3x fewer     │
  └──────────────┴──────────────────────────┴──────────────────────────┴──────────────────────────────┘

  3. Hypothesis on Failure
  The grid search failed primarily due to the fixed `threshold=5`.

   * Threshold Mismatch: The baseline base.in uses a threshold of 1, which makes it very sensitive and allows it to capture many
     subtle patterns (generating ~7.6k patterns). The grid search was forced to use a threshold of 5, which is much stricter.
   * Consequence: The high threshold prevented the algorithm from accepting many useful patterns (only ~2.3k generated). This led to
     a massive increase in "Missed" hyphenations (low recall), which the bad_weight parameter tuning could not compensate for.
   * Optimization Trap: The optimizer tried to maximize the score by finding edge-case weights (like 9, 1, 9, 2) that likely
     exploited the specific "F1/7" scoring metric to balance the high miss rate, but it could never reach the performance of the
     more permissive baseline configuration.

  Recommendation: Rerun the optimization with threshold included in the search space or fixed to 1 to match the baseline's
  sensitivity.
