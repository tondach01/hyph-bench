import argparse
import os
import sys

from hyperparameters import combine, score, sample, metaheuristic
from hyphenator.hyphenator import Hyphenator

class Validator:
    """
    Class for evaluation of patgen runs and their parameters. Abstract class, instantiate one of its subclasses
    """
    def __init__(self, model: combine.Combiner):
        """
        Create superclass validator. Should not be called by itself.
        :param model: model to evaluate
        """
        self.model = model
        self.hyphenation_mark = "-"
        self.results = None

    def process_results(self, results: list):
        """
        Aggregate results from validation runs by averaging
        :param results: validation run results
        :return: nothing, set .results attribute
        """
        self.results = dict()
        good_total, bad_total, missed_total = 0, 0, 0
        for (good, bad, missed) in results:
            good_total += good
            bad_total += bad
            missed_total += missed
        self.results["good"] = good_total / len(results)
        self.results["bad"] = bad_total / len(results)
        self.results["missed"] = missed_total / len(results)

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

    def report(self, lang: str = "", name: str = "", tabular: bool = False):
        """
        Report the validation results
        :param lang: dataset language ID
        :param name: dataset name
        :param tabular: output in LaTeX tabular format
        :return: statistics in desired format
        """
        if not tabular:
            return str(self.precision(), self.recall())
        precision = round(self.precision(), 4)
        recall = round(self.recall(), 4)
        return f"{lang} & {name} & {precision:.4f} & {recall:.4f} \\\\"

    def train_patterns(self, train_file: str):
        """
        Create patterns from train split
        :param train_file: path to train dataset
        :return: path to pattern file
        """

        self.model.meta.scorer.wordlist_path = train_file
        self.model.reset()
        pattern_file = self.model.run(self.model.meta.scorer.temp_dir)
        self.model.meta.scorer.wordlist_path = ""

        return pattern_file

    def validate_patterns(self, test_file: str, pattern_file: str):
        """
        Evaluate trained patterns against test split
        :param test_file: path to test dataset
        :param pattern_file: path to trained patterns
        :return: computed statistics (TP, FP, FN)
        """
        hyphenator = Hyphenator(pattern_file, hyphenation_mark=self.hyphenation_mark)
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
    def __init__(self, model: combine.Combiner, n: int):
        """
        Create validator
        :param model: model to evaluate
        :param n: number of folds
        """
        super().__init__(model)
        self.n = n

    def n_fold_split(self, wordlist_file: str, index: int = 0, outfile_train: str = "", outfile_test: str = ""):
        """
        Split dataset into train and test in 1:<n>-1 ratio
        :param wordlist_file: path to wordlist
        :param index: which of the n splits to use for test (when used for cross-validation)
        :param outfile_train: name of output train file (<file>.train by default)
        :param outfile_test: name of output test file (<file>.test by default)
        :return: (train file name, test file name)
        """
        p = wordlist_file.rsplit("/", maxsplit=1)
        if len(p) == 1:
            wl_dir = "."
        else:
            wl_dir = p[0]

        if "test" not in os.listdir(wl_dir):
            os.mkdir(wl_dir + "/test")

        if not outfile_train:
            outfile_train = wl_dir + "/test/data.train"
        train = open(outfile_train, "w")

        if not outfile_test:
            outfile_test = wl_dir + "/test/data.test"
        test = open(outfile_test, "w")

        with open(wordlist_file) as wordlist:
            for i, line in enumerate(wordlist):
                if i % self.n == index:
                    test.write(line)
                else:
                    train.write(line)

        train.close()
        test.close()
        return outfile_train, outfile_test

    def validate(self, wordlist_file: str, verbose: bool = False):
        """
        Perform n-fold cross-validation of a model against given dataset
        :param wordlist_file: path to wordlist
        :param verbose: enable printing out progress status
        :return: computed statistics
        """
        results = []
        for i in range(self.n):
            if verbose:
                print(f"Validation step {i+1}/{self.n}")
                print("Creating train-test split...")
            train, test = self.n_fold_split(wordlist_file, index=i)
            if verbose:
                print("Generating patterns...")
            patterns = self.train_patterns(train)
            if verbose:
                print("Validation on test set...")
            results.append(self.validate_patterns(test, patterns))
            os.remove(train)
            os.remove(test)
            os.remove(patterns)
        self.process_results(results)
        return results


def extract_files(data_directory: str):
    """
    Screen given directory for wordlist, translate file and input parameters (these may be in parent directory as well)
    :param data_directory: directory to be searched
    :return: (wordlist name, translate file name, input parameters file name), if they are found '' otherwise
    """
    wl_file, tr_file = "", ""
    for file in os.listdir(data_directory):
        if file.endswith("_dis.wlh"):
            wl_file = data_directory + "/" + file
        elif file.endswith(".tra"):
            tr_file = data_directory + "/" + file

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
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose printout")
    parser.add_argument("-t", "--tabular", action="store_true", help="Output in LateX tabular format")
    args = parser.parse_args()

    datadir = args.datadir.rstrip("/")
    wl, tr, par = extract_files(datadir)

    # wordlist is empty so that error is raised when scorer is used prior to setting it
    scorer = score.PatgenScorer("patgen", "", tr, verbose=args.verbose)
    sampler = sample.FileSampler(par)
    meta = metaheuristic.NoMetaheuristic(scorer, sampler)
    combiner = combine.SimpleCombiner(meta, verbose=args.verbose)

    validator = NFoldCrossValidator(combiner, args.nfold)
    validator.validate(wl, verbose=args.verbose)

    path = datadir.split("/")
    language = "" if len(path) < 2 else path[-2]
    d_name = "" if len(path) < 1 else path[-1]
    print(validator.report(lang=language, name=d_name, tabular=args.tabular))

