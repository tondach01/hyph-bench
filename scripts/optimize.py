#!/usr/bin/env python3
"""
Gaussian Process optimization for patgen parameters.

This script optimizes the bad_weight parameters for a 4-level patgen run
using Gaussian Process regression with Upper Confidence Bound acquisition.

Usage:
    python -m scripts.optimize --lang pl --iterations 30
    python -m scripts.optimize --lang uk --objective bounded_bad --bad-threshold 500
    python -m scripts.optimize --lang pl --resume --iterations 30

The optimizer searches over a 5-dimensional space:
    bad_1, bad_2, bad_3, bad_4 in [1, 9]
    threshold in [1, 5]

With fixed values:
    good_weight = 1 (all layers)
    pat_start, pat_finish from profile
"""

import argparse
import os
import sys
import time
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed

from .gp_optimizer import GPOptimizer
from .objectives import get_objective
from .hyperparameters.score import PatgenScorer
from .hyperparameters.sample import Sample
from .cross_validate import run_cross_validation
from .dataset_utls import DEFAULT_PAT_RANGES, find_dataset, parse_profile
from .trie_normalizer import (
    add_trie_normalizer_args,
    resolve_trie_normalizer,
    warn_fixed_trie_normalizer,
)

def run_patgen_multilevel(scorer: PatgenScorer, params: Tuple[int, ...],
                          pat_ranges: List[Tuple[int, int]],
                          good_weight: int = 1) -> dict:
    """
    Run full multi-level patgen with given parameters.
    """
    # Unpack parameters: last one is threshold
    if len(params) == 5:
        bad_weights = params[:4]
        threshold = params[4]
    else:
        bad_weights = params
        threshold = 1

    prev_id = 0
    total_patterns = 0
    final_stats = None

    n_levels = min(len(bad_weights), len(pat_ranges))

    for level in range(1, n_levels + 1):
        bad_wt = bad_weights[level - 1]
        ps, pf = pat_ranges[level - 1]

        sample = Sample({
            'level': level,
            'prev': prev_id,
            'pat_start': ps,
            'pat_finish': pf,
            'good_weight': good_weight,
            'bad_weight': bad_wt,
            'threshold': threshold
        })

        try:
            scorer.score(sample)
        except FileNotFoundError as exc:
            print(f"Warning: patgen failed for params={params} at level={level}: {exc}")
            return {
                'good': 0,
                'bad': 1,
                'missed': 1,
                'n_patterns': total_patterns,
                'trie_nodes': 0
            }
        prev_id = sample.run_id
        total_patterns += sample.stats.get('level_patterns', 0)
        final_stats = sample.stats

    return {
        'good': final_stats['tp'],
        'bad': final_stats['fp'],
        'missed': final_stats['fn'],
        'n_patterns': total_patterns,
        'trie_nodes': final_stats['trie_nodes']
    }

def run_parallel_cross_validation(wl_path: str, tr_path, lang: str, params: Tuple[int], 
                                  pat_ranges: List[Tuple[int, int]], nfold: int,
                                  good_weight: int, verbose: bool ) -> Tuple[Tuple[int, int], dict]:
    results, _ = run_cross_validation(wl_path, tr_path, lang, list(params), pat_ranges, nfold, good_weight, verbose)
    return params, results

def run_parallel_patgen(patgen_path: str, wl_path: str, tr_path: str,
                      params: Tuple[int, ...], pat_ranges: List[Tuple[int, int]],
                      good_weight: int, verbose: bool, worker_id: int) -> Tuple[Tuple[int, ...], dict]:
    """
    Helper for parallel execution of patgen.
    """
    scorer = PatgenScorer(patgen_path, wl_path, tr_path, verbose=verbose, tmp_suffix=f"_worker_{worker_id}")
    try:
        results = run_patgen_multilevel(scorer, params, pat_ranges, good_weight=good_weight)
        return params, results
    finally:
        scorer.clean()


def main():
    parser = argparse.ArgumentParser(
        description='GP optimization for patgen parameters (bad_weights + threshold)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument('--lang', required=True,
                        help='Language code (e.g., pl, uk, cs, de)')

    # Objective configuration
    parser.add_argument('--objective', default='f17',
                        choices=['f17', 'f17_trie', 'bounded_bad', 'weighted', 'pr_curve', 'min_size', 'f17_target', 'f17_cv', 'f17_target_w_trie'],
                        help='Objective function (default: f17)')
    parser.add_argument('--bad-threshold', type=int, default=500,
                        help='Bad threshold for bounded_bad/min_size objectives')
    parser.add_argument('--beta', type=float, default=1/7,
                        help='Beta for F-score (default: 1/7 for F_1/7)')
    parser.add_argument('--trie-weight', type=float, default=0.0005,
                        help='Weight for trie size penalty in f17_trie (default: 0.0005)')
    add_trie_normalizer_args(parser)
    parser.add_argument('--bad-target', type=float, default=500,
                        help='Target of bad hyphenations for f17_target (default: 500)')
    parser.add_argument('--bad-tolerance', type=float, default=500,
                        help='Tolerance defining interval around bad target ' \
                             'where F1/7 gets precedence (default: 50)')
    parser.add_argument('--n-fold', type=int, default=10,
                        help="How many folds of crossvalidation are done")

    # Optimization parameters
    parser.add_argument('--iterations', type=int, default=30,
                        help='Number of optimization iterations (default: 30)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Suggestions per iteration (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--ucb-kappa', type=float, default=2.0,
                        help='UCB exploration weight (default: 2.0)')
    parser.add_argument('--coarse-grid', action='store_true',
                        help='Run coarse grid warmup before GP optimization')
    parser.add_argument('--grid-step', type=int, default=4,
                        help='Grid step size (default: 4)')
    parser.add_argument('--max-bad-weight', type=int, default=30,
                        help='Maximum bad_weight for search (default: 30)')
    parser.add_argument('--max-threshold', type=int, default=5,
                        help='Maximum threshold for search (default: 5)')

    # Patgen parameters
    parser.add_argument('--good-weight', type=int, default=1,
                        help='Patgen good_weight for all levels (default: 1)')
    parser.add_argument('--profile', type=str,
                        help='Profile file for pat_start/pat_finish')

    # Data paths
    parser.add_argument('--wordlist', type=str,
                        help='Override wordlist path')
    parser.add_argument('--translate', type=str,
                        help='Override translate file path')
    parser.add_argument('--patgen', type=str, default='patgen',
                        help='Path to patgen binary (default: patgen)')

    # Output and state
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from saved state')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--export-iteration-results', action='store_true',
                        help='Saves final patterns and badly hyphenated words to the results folder')

    args = parser.parse_args()

    # Find dataset before building objectives so trie normalizer defaults to |D|.
    if args.wordlist and args.translate:
        wl_path, tr_path = args.wordlist, args.translate
    else:
        wl_path, tr_path = find_dataset(args.lang)

    uses_trie_objective = args.objective in ('f17_trie', 'f17_target_w_trie')
    fixed_trie_normalizer = False
    trie_normalizer = None
    if uses_trie_objective:
        trie_normalizer, fixed_trie_normalizer = resolve_trie_normalizer(
            args,
            wl_path,
            "scripts.optimize",
            dataset=args.lang,
        )

    # Setup objective
    if args.objective == 'f17':
        objective = get_objective('f17', beta=args.beta)
    elif args.objective == 'f17_trie':
        objective = get_objective('f17_trie', beta=args.beta,
                                  trie_weight=args.trie_weight,
                                  trie_normalizer=trie_normalizer)
    elif args.objective == 'f17_target':
        objective = get_objective('f17_target', bad_target=args.bad_target,
                                  tol=args.bad_tolerance, beta=args.beta)
    elif args.objective == 'f17_target_w_trie':
        objective = get_objective('f17_target_w_trie', bad_target=args.bad_target,
                                  tol=args.bad_tolerance, beta=args.beta,
                                  trie_weight=args.trie_weight,
                                  trie_normalizer=trie_normalizer)
    elif args.objective == 'bounded_bad':
        objective = get_objective('bounded_bad', bad_threshold=args.bad_threshold)
    elif args.objective == 'min_size':
        objective = get_objective('min_size', bad_threshold=args.bad_threshold)
    elif args.objective == 'f17_cv':
        objective = get_objective('f17_cv', n_fold=args.n_fold)
    else:
        objective = get_objective(args.objective)

    print(f"Objective: {objective.name}")

    print(f"Wordlist: {wl_path}")
    print(f"Translate: {tr_path}")

    # Parse profile for pat_ranges
    if args.profile:
        pat_ranges = parse_profile(args.profile)
    else:
        pat_ranges = DEFAULT_PAT_RANGES

    print(f"Pattern ranges: {pat_ranges}")

    # Setup scorer
    scorer = PatgenScorer(args.patgen, wl_path, tr_path, verbose=args.verbose)

    # Keep nested dataset identifiers inside one output directory.
    dataset_name = args.lang.replace("/", "_")
    os.makedirs(args.output_dir, exist_ok=True)
    state_path = os.path.join(args.output_dir, f"{dataset_name}_gp_state.pkl")
    csv_path = os.path.join(args.output_dir, f"{dataset_name}_history.csv")
    bad_path = os.path.join(args.output_dir, f"{dataset_name}_bad.txt")
    patterns_path = os.path.join(args.output_dir, f"{dataset_name}_final.pat")

    # Define bounds: 4 bad_weights (1-max) + 1 threshold (1-max)
    bounds = [(1, args.max_bad_weight)] * 4 + [(1, args.max_threshold)]

    # Determine min samples for GP based on coarse grid
    if args.coarse_grid:
        temp_opt = GPOptimizer(objective, seed=args.seed, bounds=bounds)
        grid = temp_opt.generate_coarse_grid(args.grid_step)
        min_samples = len(grid)
        print(f"Coarse grid enabled: {min_samples} grid points")
    else:
        min_samples = 5

    # Setup optimizer
    if args.resume and os.path.exists(state_path):
        optimizer = GPOptimizer.load(state_path, objective)
        if not hasattr(optimizer, 'bounds') or len(optimizer.bounds) != 5:
            print("Warning: Loaded state has incompatible bounds. Starting fresh.")
            optimizer = GPOptimizer(objective, seed=args.seed, bounds=bounds, min_samples_for_gp=min_samples)
        else:
            optimizer.min_samples_for_gp = min_samples
            print(f"Resumed from {len(optimizer.X)} observations")
    else:
        optimizer = GPOptimizer(objective, seed=args.seed, bounds=bounds, min_samples_for_gp=min_samples)
        print("Starting fresh optimization")

    # Coarse grid warmup phase
    if args.coarse_grid and len(optimizer.X) < min_samples:
        grid = optimizer.generate_coarse_grid(args.grid_step)
        remaining = [g for g in grid if list(g) not in optimizer.X]
        print(f"\n{'=' * 60}")
        print(f"COARSE GRID WARMUP: {len(remaining)} points to evaluate")
        print(f"{ '=' * 60}")

        for i, params in enumerate(remaining):
            print(f"  Grid [{i+1}/{len(remaining)}]: params={params}")
            scorer.reset()
            
            if args.objective == "f17_cv":
                results, _ = run_cross_validation(wl_path, tr_path, args.lang, list(params), pat_ranges, good_weight=args.good_weight, nfold=args.n_fold)
            else:
                results = run_patgen_multilevel(scorer, params, pat_ranges, good_weight=args.good_weight)
            
            score = optimizer.update(params, **results)
            print(f"    good={results['good']}, bad={results['bad']}, missed={results['missed']}, patterns={results.get('n_patterns', 'N/A')}, nodes={results['trie_nodes']}, score={score:.4f}")
            if (i + 1) % 10 == 0:
                optimizer.save(state_path)
                best = optimizer.best_so_far()
                print(f"\n  [Checkpoint] Best so far: {best['params']} -> {best['score']:.4f}\n")

        optimizer.save(state_path)
        best = optimizer.best_so_far()
        print(f"\n{'=' * 60}")
        print(f"GRID WARMUP COMPLETE")
        print(f"Best from grid: {best['params']} -> {best['score']:.4f}")
        print(f"{ '=' * 60}")

    # Main optimization loop
    import time
    start_time = time.time()
    last_reported_progress = -1

    try:
        for iteration in range(args.iterations):
            progress = (iteration + 1) / args.iterations * 100
            elapsed = time.time() - start_time
            if iteration > 0:
                avg_time_per_iter = elapsed / iteration
                remaining_iters = args.iterations - iteration
                eta_seconds = remaining_iters * avg_time_per_iter
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            else:
                eta_str = "Calculating..."

            if progress >= last_reported_progress + 5 or iteration == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Progress: {progress:.1f}% | ETA: {eta_str}")
                last_reported_progress = progress

            print(f"\n{'=' * 60}")
            print(f"Iteration {iteration + 1}/{args.iterations}")
            suggestions = optimizer.suggest_batch(args.batch_size, ucb_kappa=args.ucb_kappa)

            if args.batch_size > 1:
                with ProcessPoolExecutor(max_workers=args.batch_size) as executor:
                    if args.objective == "f17_cv":
                        futures = {
                            executor.submit(run_parallel_cross_validation, wl_path, tr_path, args.lang, params, 
                                            pat_ranges, args.n_fold, args.good_weight, args.verbose)
                            for params in suggestions
                        }
                    else:
                        futures = {
                            executor.submit(run_parallel_patgen, args.patgen, wl_path, tr_path,
                                            params, pat_ranges, args.good_weight, args.verbose, i): params
                            for i, params in enumerate(suggestions)
                        }

                    for future in as_completed(futures):
                        params, results = future.result()
                        score = optimizer.update(params, **results)
                        print(f"  Tested: params={params}")
                        print(f"    good={results['good']}, bad={results['bad']}, missed={results['missed']}, patterns={results.get('n_patterns', 'N/A')}, nodes={results['trie_nodes']}, score={score:.4f}")
            else:
                for params in suggestions:
                    print(f"  Testing: params={params}")
                    scorer.reset()

                    if args.objective == 'f17_cv':
                        results, _ = run_cross_validation(wl_path, tr_path, args.lang, list(params), pat_ranges, nfold=args.n_fold, good_weight=args.good_weight, verbose=args.verbose)
                    else:
                        results = run_patgen_multilevel(scorer, params, pat_ranges, good_weight=args.good_weight)
                    
                    score = optimizer.update(params, **results)
                    print(f"    good={results['good']}, bad={results['bad']}, missed={results['missed']}, patterns={results.get('n_patterns', 'N/A')}, nodes={results['trie_nodes']}, score={score:.4f}")

            best = optimizer.best_so_far()
            if best:
                print(f"\n  Best so far: {best['params']} -> {best['score']:.4f}")
                print(f"    good={best['good']}, bad={best['bad']}, missed={best['missed']}, nodes={best['trie_nodes']}")
            optimizer.save(state_path)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving state...")
        optimizer.save(state_path)

    # Final exploitation phase
    print(f"\n{'=' * 60}")
    print("Final exploitation phase")
    best_params_to_test = optimizer.exploit_best(n=3)

    if len(best_params_to_test) > 1:
        with ProcessPoolExecutor(max_workers=len(best_params_to_test)) as executor:
            if args.objective == "f17_cv":
                futures = {
                    executor.submit(run_parallel_cross_validation, wl_path, tr_path, args.lang, params, 
                                    pat_ranges, args.n_fold, args.good_weight, args.verbose)
                    for params in best_params_to_test
                }
            else:
                futures = {
                    executor.submit(run_parallel_patgen, args.patgen, wl_path, tr_path,
                                    params, pat_ranges, args.good_weight, args.verbose, i): params
                    for i, params in enumerate(best_params_to_test)
                }

            for future in as_completed(futures):
                params, results = future.result()
                score = optimizer.update(params, **results)
                print(f"  {params}: good={results['good']}, bad={results['bad']}, patterns={results.get('n_patterns', 'N/A')}, nodes={results['trie_nodes']}, score={score:.4f}")
    else:
        for params in best_params_to_test:
            scorer.reset()
            if args.objective == 'f17_cv':
                results, _ = run_cross_validation(wl_path, tr_path, args.lang, params, pat_ranges, good_weight=args.good_weight, nfold=args.n_fold)
            else:
                results = run_patgen_multilevel(scorer, params, pat_ranges, good_weight=args.good_weight)

            score = optimizer.update(params, **results)
            print(f"  {params}: good={results['good']}, bad={results['bad']}, patterns={results.get('n_patterns', 'N/A')}, nodes={results['trie_nodes']}, score={score:.4f}")
        
    # Final report
    best = optimizer.best_so_far()

    # Run patgen on optimal parameters to populate missing 
    # results if the objective does not support all of them
    if args.batch_size > 1 or args.objective == "f17_cv":
        results = run_patgen_multilevel(scorer, best['params'], pat_ranges, good_weight=args.good_weight)
        
        if args.objective == "f17_cv":
            best['n_patterns'] = results['n_patterns']

    print(f"\n{'=' * 60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{ '=' * 60}")
    print(f"Best parameters: {best['params']}")
    if len(best['params']) >= 5:
        print(f"  bad_weights={best['params'][:len(pat_ranges)]}, threshold={best['params'][-1]}")
    print(f"Results:")
    print(f"  good={best['good']}, bad={best['bad']}, missed={best['missed']}")
    if args.objective == 'f17_cv':
        print(f"  good_variance={best['good_variance']:.4f}, bad_variance={best['bad_variance']:.4f}, missed_variance={best['missed_variance']:.4f}")
    print(f"  n_patterns={best['n_patterns']}, trie_nodes={best['trie_nodes']}")
    print(f"  score={best['score']:.4f}")

    optimizer.save(state_path)
    print(f"\nState saved to: {state_path}")

    if args.export_iteration_results:
        scorer.dump_bad(bad_path, len(pat_ranges))
        print(f"Bad words saved to: {bad_path}")
        scorer.export_patterns(patterns_path, len(pat_ranges))
        print(f"Final patterns saved to: {patterns_path}")

    try:
        df = optimizer.get_history_dataframe()
        df.to_csv(csv_path, index=False)
        print(f"History saved to: {csv_path}")
    except ImportError:
        print("(pandas not available, skipping CSV export)")
    scorer.clean()
    if fixed_trie_normalizer:
        warn_fixed_trie_normalizer(
            "scripts.optimize",
            trie_normalizer,
            "END WARNING",
        )

if __name__ == '__main__':
    main()
