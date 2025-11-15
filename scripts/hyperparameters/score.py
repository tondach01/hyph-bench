import datetime
import os
import re

from . import sample


class PatgenScorer:
    """
    Class for patgen hyperparameter setting evaluation
    """
    def __init__(self, patgen_path: str, wordlist_path: str, translate_path: str, verbose: bool = False):
        self.patgen_path: str = patgen_path
        self.wordlist_path: str = wordlist_path
        self.translate_path: str = translate_path
        self.verbose = verbose

        wl_dir = wordlist_path.split("/")
        if len(wl_dir) > 1:
            tmp_path = "/".join(wl_dir[:-1])
        else:
            tmp_path = "."
        if "tmp" not in os.listdir(tmp_path):
            os.mkdir(tmp_path+"/tmp")

        self.temp_dir: str = tmp_path+"/tmp"

        if "0.pat" not in os.listdir(self.temp_dir):
            os.system(f"touch {self.temp_dir}/0.pat")

        self.max_id: int = 0

        self._cached: dict = dict()

    def score(self, s: sample.Sample):
        """
        Evaluate hyperparameter setting and set corresponding attributes in sample
        :param s: hyperparameter values in Sample object
        :return: setting score (number of patterns, precision, recall)
        """
        run_id = self.max_id + 1
        s.run_id = run_id
        self.max_id += 1

        s_hash = s.__hash__()
        if s_hash in self._cached:
            pat, stats = self._cached[s_hash]
            s.stats = stats
            s.n_patterns = pat
            return pat, s.precision(), s.recall()

        #TODO filter out ambiguous hyphenations

        with open(f"{self.temp_dir}/{run_id}.in", "w") as par:
            par.write("\n".join([f"{s.level} {s.level}",
                                 f"{s.pat_start} {s.pat_finish}",
                                 f"{s.good_weight} {s.bad_weight} {s.threshold}",
                                 "n",
                                 ""]
                                )
                      )

        command = " ".join([f"cat {self.temp_dir}/{run_id}.in | (",
                            self.patgen_path,
                            self.wordlist_path,
                            f"{self.temp_dir}/{s.prev}.pat",
                            f"{self.temp_dir}/{run_id}.pat",
                            self.translate_path, ") >",
                            f"{self.temp_dir}/{run_id}.log"])
        os.system(command)

        n_patterns = self.count_patterns(run_id)

        stats = self.get_statistics(run_id)
        self._cached[s_hash] = (n_patterns, stats)

        s.stats = stats
        s.n_patterns = n_patterns
        s.timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        if self.verbose:
            print(str(s))

        return n_patterns, s.precision(), s.recall()

    def count_patterns(self, run_id: int):
        """
        Count the patterns generated in pattern file (<run_id>.pat)
        :param run_id: ID of the execution
        :return: number of patterns read
        """
        n_patterns = 0
        with open(f"{self.temp_dir}/{run_id}.pat") as outfile:
            for _ in outfile:
                n_patterns += 1
        return n_patterns

    def get_statistics(self, run_id: int):
        """
        Analyze dumped output from patgen run (<run_id>.out) to find information about hyphenation accuracy
        :return: dictionary of ('tp' true positives, 'fp' false positives, 'fn' false negatives)
        """
        tp, fp, fn = 0, 0, 0

        with open(f"{self.temp_dir}/{run_id}.log") as out:
            for line in out:
                stat = re.match(r"(?P<tp>\d+) good, (?P<fp>\d+) bad, (?P<fn>\d+) missed", line)
                if stat is not None:
                    tp = int(stat["tp"])
                    fp = int(stat["fp"])
                    fn = int(stat["fn"])

        return {"tp": tp, "fp": fp, "fn": fn}

    def clean(self):
        """
        Delete al temporary files used during computations.
        """
        os.system("rm -rf "+self.temp_dir)

    def clean_unused(self, ids: set):
        """
        Delete temporary files that are not used anymore
        :param ids: IDs that are still in use
        """
        for file in os.listdir(self.temp_dir):
            match = re.match(r"(?P<id>\d+).pat", file)
            if match is not None and int(match["id"]) not in ids:
                os.remove(f"{self.temp_dir}/{file}")
                if int(match["id"]) == 0:
                    continue
                os.remove(f"{self.temp_dir}/{match['id']}.log")
                os.remove(f"{self.temp_dir}/{match['id']}.in")

    def clear_cache(self):
        """
        Clear cached scores
        """
        self._cached.clear()

    def reset(self):
        """
        Reset the object to initial state
        """
        wl_dir = self.wordlist_path.split("/")
        if len(wl_dir) > 1:
            tmp_path = "/".join(wl_dir[:-1])
        else:
            tmp_path = "."
        if "tmp" not in os.listdir(tmp_path):
            os.mkdir(tmp_path + "/tmp")

        self.temp_dir: str = tmp_path + "/tmp"

        if "0.pat" not in os.listdir(self.temp_dir):
            os.system(f"touch {self.temp_dir}/0.pat")

        self.max_id: int = 0

        self.clear_cache()
