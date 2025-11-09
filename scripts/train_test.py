import argparse
import os

import hyperparameters as hp

class Validator:
    """
    Class for evaluation of patgen runs and their parameters. Abstract class, instantiate one of its subclasses
    """
    def __init__(self, model: hp.combine.Combiner):
        """
        Create superclass validator. Should not be called by itself.
        :param model: model to evaluate
        """
        self.model = model

    def train_patterns(self, train_file: str, pattern_file: str = ""):
        """
        Create patterns from train split
        :param train_file: path to train dataset
        :param pattern_file: path to output pattern file (out.pat by default)
        :return: path to pattern file
        """
        if not pattern_file:
            pattern_file = "out.pat"

        self.model.meta.scorer.wordlist_path = train_file

        self.model.run()

        self.model.meta.scorer.wordlist_path = ""

        return pattern_file

    def validate_patterns(self, test_file: str, patters_file: str):
        """
        Evaluate trained patterns against test split
        :param test_file: path to test dataset
        :param patters_file: path to trained patterns
        :return: computed statistics
        """

        # TODO

        pass

    def validate(self, wordlist_file: str):
        """
        Run evaluation against given dataset. Abstract method, implementation differs between subclasses
        :param wordlist_file: path to wordlist
        :return: computed statistics
        """
        return NotImplemented

class NFoldCrossValidator(Validator):
    """
    N-fold cross-validation
    """
    def __init__(self, model: hp.combine.Combiner, n: int):
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
        if not outfile_train:
            outfile_train = wordlist_file + ".train"
        train = open(outfile_train, "w")

        if not outfile_test:
            outfile_test = wordlist_file + ".test"
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

    def validate(self, wordlist_file: str):
        """
        Perform n-fold cross-validation of a model against given dataset
        :param wordlist_file: path to wordlist
        :return: computed statistics
        """
        results = []
        for i in range(self.n):
            train, test = self.n_fold_split(wordlist_file, index=i)
            patterns = self.train_patterns(train)
            results.append(self.validate_patterns(test, patterns))
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help="Directory with wordlist and translate file")
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

    # wordlist is empty so that error is raised when scorer is used prior to setting it
    scorer = hp.score.PatgenScorer("patgen", "", tr_file, verbose=True)
    sampler = hp.sample.FileSampler(par_file)
    meta = hp.metaheuristic.NoMetaheuristic(scorer, sampler)
    combiner = hp.combine.SimpleCombiner(meta, verbose=True)

    validator = NFoldCrossValidator(combiner, 10)
    print(validator.validate(wl_file))
