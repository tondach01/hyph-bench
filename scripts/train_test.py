import argparse
import os
import sys
import shutil

from .hyperparameters import combine, score, sample, metaheuristic
from .hyphenator.hyphenator import Hyphenator
from .dataset_split import rank_word_entries, resolve_word_entries

class Validator:
    """
    Class for evaluation of patgen runs and their parameters. Abstract class, instantiate one of its subclasses
    """
    def __init__(self, model: combine.Combiner, translate_file: str, tmp_suffix: str = ""):
        """
        Create superclass validator. Should not be called by itself.
        :param model: model to evaluate
        :param translate_file: path to translate file
        """
        self.model = model
        self.hyphenation_mark = "-"
        self.translate_file = translate_file
        self.results = None
        self.tmp_dir_name = f"test{tmp_suffix}"

    def process_results(self, results: list):
        """
        Aggregate results from validation runs by averaging
        :param results: validation run results
        :return: nothing, set .results attribute
        """
        self.results = dict()
        good_total, bad_total, missed_total = 0, 0, 0
        good_total_sq, bad_total_sq, missed_total_sq = 0, 0, 0
        nodes_total = 0
        n = len(results)

        for (good, bad, missed), trie_nodes in results:
            good_total += good
            good_total_sq += good ** 2
            bad_total += bad
            bad_total_sq += bad ** 2
            missed_total += missed
            missed_total_sq += missed ** 2
            nodes_total += trie_nodes

        good_mean = good_total / n
        bad_mean = bad_total / n
        missed_mean = missed_total / n

        self.results["good"] = good_mean
        self.results["bad"] = bad_mean
        self.results["missed"] = missed_mean
        self.results["trie_nodes"] = nodes_total / n

        self.results["good_variance"] = (good_total_sq / n - good_mean ** 2) / n
        self.results["bad_variance"] = (bad_total_sq / n - bad_mean ** 2) / n
        self.results["missed_variance"] = (missed_total_sq / n - missed_mean ** 2) / n

    def precision(self):
        """
        Compute precision of the results
        :return: precision as (good) / (good + bad), or 0 if .results is not set or does not contain required values
        """
        if self.results is None or "good" not in self.results or "bad" not in self.results or self.results["good"] == 0:
            return 0
        return self.results["good"] / (self.results["good"] + self.results["bad"])

    def recall(self):
        """
        Compute recall of the results
        :return: recall as (good) / (good + missed), or 0 if .results is not set or does not contain required values
        """
        if self.results is None or "good" not in self.results or "missed" not in self.results or self.results["good"] == 0:
            return 0
        return self.results["good"] / (self.results["good"] + self.results["missed"])

    def f_score(self, n: float):
        """
        Compute F-n score of the results
        :param n: weight of precision
        :return: (1 + n*n) * precision * recall / ((n*n * precision) + recall)
        """
        p, r = self.precision(), self.recall()
        if p == 0 or r == 0:
            return 0
        return (1 + n * n) * p * r / ((n * n * p) + r)

    def report(self, lang: str = "", name: str = "", profile: str = "", tabular: bool = False):
        """
        Report the validation results
        :param lang: dataset language ID
        :param name: dataset name
        :param profile: parameter profile name
        :param tabular: output in LaTeX tabular format
        :return: a dict with results and statistics in desired format
        """
        results = {
            "f_17": self.f_score(1/7),
            "bad": self.results["bad"],
            "good": self.results["good"],
            "missed": self.results["missed"],
            "trie_nodes": self.results["trie_nodes"],
            "good_variance": self.results["good_variance"],
            "bad_variance": self.results["bad_variance"],
            "missed_variance": self.results["missed_variance"]
        }

        if not tabular:
            return results, f"precision={self.precision():.4f}, recall={self.recall():.4f}"
        
        f_score = round(self.f_score(1/7), 4)
        trie_nodes = round(self.results["trie_nodes"], 1)
        return results, f"{lang} & {name} & {profile} & {f_score:.4f} & {trie_nodes:.1f} \\\\"

    def train_patterns(self, train_file: str, tmp_suffix: str = ""):
        """
        Create patterns from train split
        :param train_file: path to train dataset
        :param tmp_suffix: suffix to temporary directory name
        :return: path to pattern file, the number of nodes in pattern trie
        """

        self.model.meta.scorer.wordlist_path = train_file
        self.model.reset(tmp_suffix)
        pattern_file, trie_nodes = self.model.run(self.model.meta.scorer.temp_dir)
        self.model.meta.scorer.wordlist_path = ""

        return pattern_file, trie_nodes

    def validate_patterns(self, test_file: str, pattern_file: str):
        """
        Evaluate trained patterns against test split
        :param test_file: path to test dataset
        :param pattern_file: path to trained patterns
        :return: computed statistics (TP, FP, FN)
        """
        hyphenator = Hyphenator(pattern_file, hyphenation_mark=self.hyphenation_mark, translate_file=self.translate_file)
        good, bad, missed = 0, 0, 0
        with open(test_file) as test:
            for correct in test:
                correct = correct.strip()
                hyphenated = hyphenator.hyphenate(correct)
                i_corr, i_hyph = 0, 0
                while i_corr < len(correct) and i_hyph < len(hyphenated):
                    if correct[i_corr] == self.hyphenation_mark and hyphenated[i_hyph] == self.hyphenation_mark:
                        good += 1
                        i_hyph += 1
                        i_corr += 1
                    elif hyphenated[i_hyph] == self.hyphenation_mark:
                        bad += 1
                        i_hyph += 1
                    elif correct[i_corr] == self.hyphenation_mark:
                        missed += 1
                        i_corr += 1
                    else:
                        i_hyph += 1
                        i_corr += 1
        return good, bad, missed

    def validate(self, wordlist_file: str, verbose: bool = False):
        """
        Run evaluation against given dataset. Abstract method, implementation differs between subclasses
        :param wordlist_file: path to wordlist
        :param verbose: enable printing out progress status
        :return: computed statistics (TP, FP, FN)
        """
        return NotImplemented

class NFoldCrossValidator(Validator):
    """
    N-fold cross-validation
    """
    def __init__(
        self,
        model: combine.Combiner,
        translate_file: str,
        n: int,
        tmp_suffix: str = "",
        seed: int = 42,
    ):
        """Create a deterministic, surface-form-disjoint cross-validator."""
        super().__init__(model, translate_file, tmp_suffix=tmp_suffix)
        self.n = n
        self.seed = seed
        self._entries_path = ""
        self._entries = []
        self._weighted = False


    def n_fold_split(self, wordlist_file: str, index: int = 0, outfile_train: str = "", outfile_test: str = "", tmp_suffix: str = ""):
        """
        Split dataset into train and test in 1:<n>-1 ratio
        :param wordlist_file: path to wordlist
        :param index: which of the n splits to use for test (when used for cross-validation)
        :param outfile_train: name of output train file (<file>.train by default)
        :param outfile_test: name of output test file (<file>.test by default)
        :param tmp_suffix: suffix to temporary directory name
        :return: (train file name, test file name)
        """
        p = wordlist_file.rsplit("/", maxsplit=1)
        if len(p) == 1:
            wl_dir = "."
        else:
            wl_dir = p[0]

        tmp_dir = os.path.abspath(os.path.join(wl_dir, self.tmp_dir_name))
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

        if not outfile_train:
            outfile_train = os.path.join(tmp_dir, f"data.train{tmp_suffix}")
        train = open(outfile_train, "w")

        if not outfile_test:
            outfile_test = os.path.join(tmp_dir, f"data.test{tmp_suffix}")
        test = open(outfile_test, "w")

        if self._entries_path != wordlist_file:
            self._entries, self._weighted, _ = resolve_word_entries(wordlist_file)
            rank_word_entries(self._entries, self.seed)
            self._entries_path = wordlist_file
        for position, (_, annotation, priority) in enumerate(self._entries):
            if position % self.n == index:
                test.write(annotation + "\n")
            else:
                repetitions = priority if self._weighted else 1
                train.write((annotation + "\n") * repetitions)

        train.close()
        test.close()
        return outfile_train, outfile_test, tmp_dir

    def validate(self, wordlist_file: str, verbose: bool = False, fixed_test: str = None):
        """
        Perform n-fold cross-validation of a model against given dataset
        :param wordlist_file: path to wordlist
        :param verbose: enable printing out progress status
        :return: computed statistics
        """
        results = []
        for i in range(self.n):
            suffix = str(i)
            if verbose:
                print(f"Validation step {i+1}/{self.n}")
                print("Creating train-test split...")
            train, test, tmp_dir = self.n_fold_split(wordlist_file, index=i, tmp_suffix=suffix)
            if verbose:
                print("Generating patterns...")
            patterns, trie_nodes = self.train_patterns(train, tmp_suffix=suffix)
            if verbose:
                print("Validation on test set...")
            results.append((self.validate_patterns(test if fixed_test is None else fixed_test, patterns), trie_nodes))
            os.remove(train)
            os.remove(test)
            os.remove(patterns)
            shutil.rmtree(tmp_dir)
        self.process_results(results)
        return results


def extract_files(data_directory: str):
    """
    Screen given directory for wordlist, translate file and input parameters (these may be in parent directory as well)
    :param data_directory: directory to be searched
    :return: (wordlist name, translate file name, input parameters file name), if they are found '' otherwise
    """
    files = sorted(os.listdir(data_directory))
    preferred_suffixes = ("_dis.wlh", "_expanded.wlh", ".wlh")
    wl_file, tr_file = "", ""
    for suffix in preferred_suffixes:
        candidates = [file for file in files if file.endswith(suffix)]
        if candidates:
            wl_file = os.path.join(data_directory, candidates[0])
            break
    translate_files = [file for file in files if file.endswith(".tra")]
    if translate_files:
        tr_file = os.path.join(data_directory, translate_files[0])
    if not wl_file or not tr_file:
        print(f"Wordlist or translate file not present in {data_directory} directory", file=sys.stderr)

    par_file = ""
    par_dir = data_directory
    for _ in range(3):  # assume the directory structure .../data/<lang>/<dataset>
        if "patgen_params.in" in os.listdir(par_dir):
            par_file = par_dir + "/patgen_params.in"
            break
        par_dir = par_dir + "/.."

    if not par_file:
        print(f"Patgen parameters file <patgen_params.in> not found in {data_directory} or 2 level above", file=sys.stderr)

    return wl_file, tr_file, par_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help="Directory with wordlist and translate file")
    parser.add_argument("-n", "--nfold", type=int, default=10, required=False, help="Number of folds to use in cross-validation")
    parser.add_argument("-p", "--profile", type=str, default="", required=False, help="Parameter profile to use")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose printout")
    parser.add_argument("-t", "--tabular", action="store_true", help="Output in LateX tabular format")
    args = parser.parse_args()

    datadir = args.datadir.rstrip("/")
    wl, tr, par = extract_files(datadir)

    # wordlist is empty so that error is raised when scorer is used prior to setting it
    scorer = score.PatgenScorer("patgen", "", tr, verbose=args.verbose)
    sampler = sample.FileSampler(par if not args.profile else args.profile)
    meta = metaheuristic.NoMetaheuristic(scorer, sampler)
    combiner = combine.SimpleCombiner(meta, verbose=args.verbose)

    validator = NFoldCrossValidator(combiner, tr, args.nfold)
    validator.validate(wl, verbose=args.verbose)

    path = datadir.split("/")
    language = "" if len(path) < 2 else path[-2]
    d_name = "" if len(path) < 1 else path[-1]
    _, report = validator.report(lang=language, name=d_name, profile=args.profile, tabular=args.tabular)
    print(report)

