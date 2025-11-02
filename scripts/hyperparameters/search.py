import argparse
import score
import sample
import metaheuristic
import combine
import time

if __name__ == "__main__":
    t = time.time()
    parser = argparse.ArgumentParser()
    #parser.add_argument("heuristic", help="Metaheuristic to use")
    parser.add_argument("--score", choices=["precision", "recall", "f1"], default="precision", help="Function to use for scoring")
    parser.add_argument("--nsamples", required=False, default=10, type=int, help="Number of samples to generate in each iteration")
    parser.add_argument("--accthresh", required=False, default=1.0, type=float, help="Score threshold to pass for termination 0.0 <= accthresh <= 1.0")
    parser.add_argument("--maxit", required=False, default=5, type=int, help="Maximal number of iterations until termination, 1 <= maxit <= 9")
    parser.add_argument("--tol", required=False, default=0.02, type=float, help="Pattern count increase tolerance, 0.0 <= tol")

    args = parser.parse_args()

    scorer = score.PatgenScorer(
        "../../../patgen/cmake-build-debug/patgen_sandbox",
        "../patgen/sandbox/wortliste.wlh",
        "../patgen/sandbox/german.tr"
    )

    sampler = sample.FileSampler("../../../patgen/sandbox/test_parameters.in")

    meta = metaheuristic.NoMetaheuristic(
        scorer,
        sampler,
        #logfile="wortliste.log"
    )

    comb = combine.SimpleCombiner(meta)

    comb.run()
    print([str(pop) for pop in meta.population])
    print("Ran", round(time.time() - t, 2), "seconds")
