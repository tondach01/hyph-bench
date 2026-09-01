#!/usr/bin/env python3
"""Cross-validation for optimized PATGEN parameters.

Folds group normalized surface forms before seeded hash assignment. Source
priorities are expanded in training only; each test fold is type-based.
"""

import argparse
import os
import uuid
from typing import List, Tuple

from .train_test import NFoldCrossValidator, extract_files
from .hyperparameters import combine, score, sample, metaheuristic
from .dataset_utls import parse_profile, DEFAULT_PAT_RANGES, find_dataset


def create_dynamic_profile(params: List[int], pat_ranges: List[Tuple[int, int]], 
                           good_weight: int = 1, output_path: str = "dynamic.in"):
    """
    Create a profile file from optimized parameters.
    """
    bad_weights = params[:-1]
    threshold = params[-1]
        
    with open(output_path, "w") as f:
        for i, (ps, pf) in enumerate(pat_ranges):
            bw = bad_weights[i] if i < len(bad_weights) else bad_weights[-1]
            f.write(f"{ps} {pf} {good_weight} {bw} {threshold}\n")
    return output_path

def run_cross_validation(wl_path: str, tr_path, lang: str, params: List[int], 
                         pat_ranges: List[Tuple[int, int]], nfold: int = 10,
                         good_weight: int = 1, verbose: bool = False, 
                         fixed_test: str = None) -> Tuple[dict, str]:
    tmp_dir = os.path.dirname(wl_path)
    run_id = str(uuid.uuid4().hex)
    safe_lang = lang.replace(os.sep, "_").replace("/", "_")
    profile_path = os.path.join(tmp_dir, f"{safe_lang}_dynamic_{run_id}.in")

    create_dynamic_profile(params, pat_ranges, good_weight, profile_path)
    print(f"Created dynamic profile: {profile_path}")
    
    scorer = score.PatgenScorer("patgen", "", tr_path, verbose=verbose, tmp_suffix=run_id)
    sampler = sample.FileSampler(profile_path)
    meta = metaheuristic.NoMetaheuristic(scorer, sampler)
    combiner = combine.SimpleCombiner(meta, verbose=verbose)
    
    validator = NFoldCrossValidator(combiner, tr_path, nfold, tmp_suffix=run_id)
    print(f"Running {nfold}-fold cross-validation...")
    validator.validate(wl_path, verbose=verbose, fixed_test=fixed_test)

    os.remove(profile_path)
    scorer.clean()
    
    lang_name = os.path.basename(os.path.dirname(wl_path))
    ds_name = os.path.basename(wl_path).replace("_dis.wlh", "").replace(".wlh", "")
    
    return validator.report(lang=lang_name, name=ds_name, profile="dynamic", tabular=True)

def main():
    parser = argparse.ArgumentParser(description='Cross-validation for optimized parameters')
    parser.add_argument('--lang', required=True, help='Language code')
    parser.add_argument('--wordlist', type=str, required=False,
                        help='Set wordlist to compare params on, requires translate also')
    parser.add_argument('--translate', type=str, required=False,
                        help='Translate wordlist for wordlist')
    parser.add_argument('--params', type=int, nargs='+', required=True,
                        help='Best parameters found (4 bad_weights + 1 threshold)')
    parser.add_argument('--profile', type=str, help='Base profile for pat_ranges')
    parser.add_argument('--nfold', type=int, default=10, help='Number of folds')
    parser.add_argument('--good-weight', type=int, default=1, help='Good weight')
    parser.add_argument('--fixed-test', type=str, required=False, help='Fixed test case for each fold')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Find dataset
    if args.wordlist:
        wl_path, tr_path = args.wordlist, args.translate
    else:
        wl_path, tr_path = find_dataset(args.lang)
    print(f"Dataset: {wl_path}")
    
    # Get pattern ranges
    if args.profile:
        pat_ranges = parse_profile(args.profile)
    else:
        pat_ranges = DEFAULT_PAT_RANGES
        
    _, report = run_cross_validation(wl_path, tr_path, args.lang, args.params, pat_ranges,
                                     args.nfold, args.good_weight, args.verbose, args.fixed_test)

    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    print(report)
    print("="*60)

if __name__ == "__main__":
    main()
