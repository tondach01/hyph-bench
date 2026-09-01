import os
import sys
from .hyperparameters.score import PatgenScorer
from .hyperparameters.sample import Sample
from .hyphenator.hyphenator import Hyphenator

def find_errors(lang, bad_weights, threshold):
    # Find dataset
    wl_path = f"data/{lang}/wiktionary/uk-full-wiktionary.wlh_dis.wlh"
    tr_path = f"{wl_path}.tra"
    
    if not os.path.exists(wl_path):
        print(f"Error: {wl_path} not found")
        return

    # Setup scorer
    scorer = PatgenScorer("patgen", wl_path, tr_path, verbose=False, tmp_suffix="_find_errors")
    scorer.reset("_find_errors")

    # Default pattern ranges
    pat_ranges = [(1, 4), (2, 5), (2, 6), (2, 7)]
    
    prev_id = 0
    for level, (bad_wt, (ps, pf)) in enumerate(zip(bad_weights, pat_ranges), 1):
        sample = Sample({
            'level': level,
            'prev': prev_id,
            'pat_start': ps,
            'pat_finish': pf,
            'good_weight': 1,
            'bad_weight': bad_wt,
            'threshold': threshold
        })
        scorer.score(sample)
        prev_id = sample.run_id

    # The final pattern file
    pattern_file = f"{scorer.temp_dir}/{prev_id}.pat"
    
    # Setup hyphenator
    hyphenator = Hyphenator(pattern_file, translate_file=tr_path)
    
    errors = []
    with open(wl_path) as f:
        for line in f:
            correct = line.strip()
            # Remove existing marks to get raw word
            word = correct.replace("-", "")
            hyphenated = hyphenator.hyphenate(word)
            
            if hyphenated != correct:
                errors.append((correct, hyphenated))
    
    print(f"\nFound {len(errors)} errors:")
    for i, (corr, hyph) in enumerate(errors, 1):
        print(f"{i:2}. Correct: {corr:20} | Hyphenated: {hyph}")
        if i >= 100: # Safety break
            break

    # Cleanup
    # scorer.clean()

if __name__ == "__main__":
    # Best params for uk: (7, 1, 9, 3, 1)
    find_errors("uk", (7, 1, 9, 3), 1)
