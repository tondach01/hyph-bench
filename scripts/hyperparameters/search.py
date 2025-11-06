import argparse
import os
import time

import score
import sample
import metaheuristic
import combine
import stats

if __name__ == "__main__":
    t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help="Directory with wordlist and translate file")
    parser.add_argument("--score", choices=["precision", "recall", "f1"], default="precision", help="Function to use for scoring")
    parser.add_argument("--nsamples", required=False, default=10, type=int, help="Number of samples to generate in each iteration")
    parser.add_argument("--accthresh", required=False, default=1.0, type=float, help="Score threshold to pass for termination 0.0 <= accthresh <= 1.0")
    parser.add_argument("--maxit", required=False, default=5, type=int, help="Maximal number of iterations until termination, 1 <= maxit <= 9")
    parser.add_argument("--tol", required=False, default=0.02, type=float, help="Pattern count increase tolerance, 0.0 <= tol")

    args = parser.parse_args()

    datadir = args.datadir.rstrip("/")
    wl_file, tr_file = "", ""
    for file in os.listdir(datadir):
        if file.endswith(".wlh"):
            wl_file = datadir + "/" + file
        elif file.endswith(".tra"):
            tr_file = datadir + "/" + file

    if not wl_file or not tr_file:
        print(f"Wordlist or translate file not present in {datadir} directory")
        exit(1)

    scorer = score.PatgenScorer(
        "patgen", wl_file, tr_file
    )

    par_file = ""
    par_dir = datadir
    for _ in range(3):  # assume the directory structure .../data/<lang>/<dataset>
        if "patgen_params.in" in os.listdir(par_dir):
            par_file = par_dir + "/patgen_params.in"
            break
        par_dir = par_dir + "/.."

    if not par_file:
        print(f"Patgen parameters file <patgen_params.in> not found in {datadir} or 2 level above")
        exit(1)

    sampler = sample.FileSampler(par_file)
    statistic = stats.LearningInfo()

    meta = metaheuristic.NoMetaheuristic(
        scorer,
        sampler,
        statistic=statistic
    )

    comb = combine.SimpleCombiner(meta)

    comb.run()
    print([(pop.f_score(1.0), pop.f_score(100.0)) for pop in meta.population])
    meta.statistic.visualise(metric=["patterns"])

    for s in meta.population:
        print(str(stats.PatternsInfo(f"{datadir}/{s.run_id}.pat", s)))

    print("Ran", round(time.time() - t, 2), "seconds")
