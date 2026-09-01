import datetime
import os
import re

from . import sample


class PatgenScorer:
    """
    Class for patgen hyperparameter setting evaluation
    """
    def __init__(self, patgen_path: str, wordlist_path: str, translate_path: str, verbose: bool = False, tmp_suffix: str = ""):
        self.patgen_path: str = patgen_path
        self.wordlist_path: str = os.path.abspath(wordlist_path)
        self.translate_path: str = os.path.abspath(translate_path)
        self.verbose = verbose
        self.initial_suffix = tmp_suffix

        self.__create_temp_env(tmp_suffix)

        self._cached: dict = dict()

    def score(self, s: sample.Sample):
        """
        Evaluate hyperparameter setting and set corresponding attributes in sample
        :param s: hyperparameter values in Sample object
        """
        run_id = self.max_id + 1
        s.run_id = run_id
        self.max_id += 1

        s_hash = s.__hash__()
        if s_hash in self._cached:
            stats = self._cached[s_hash]
            s.stats = stats.copy()

        cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)

            with open(f"{run_id}.in", "w") as par:
                par.write("\n".join([f"{s.level} {s.level}",
                                     f"{s.pat_start} {s.pat_finish}",
                                     f"{s.good_weight} {s.bad_weight} {s.threshold}",
                                     "y",
                                     ""]
                                    )
                          )

            command = " ".join([f"cat {run_id}.in | (",
                                self.patgen_path,
                                self.wordlist_path,
                                f"{s.prev}.pat",
                                f"{run_id}.pat",
                                self.translate_path, ") >",
                                f"{run_id}.log"])
            os.system(command)

            stats = self.get_statistics(run_id)
            stats["n_patterns"] = self.count_patterns(run_id)
            self._cached[s_hash] = stats
        finally:
            os.chdir(cwd)

        s.stats = stats
        s.timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        if self.verbose:
            print(str(s))

    def count_patterns(self, run_id: int):
        """
        Count the patterns generated in pattern file (<run_id>.pat)
        :param run_id: ID of the execution
        :return: number of patterns read
        """
        n_patterns = 0
        with open(f"{run_id}.pat") as outfile:
            for _ in outfile:
                n_patterns += 1
        return n_patterns

    def get_statistics(self, run_id: int):
        """
        Analyze dumped output from patgen run (<run_id>.log) to find information about hyphenation accuracy
        :return: dictionary of ('tp' true positives, 'fp' false positives, 'fn' false negatives, 'trie_nodes'
        the number of nodes in pattern trie)
        """
        tp, fp, fn = 0, 0, 0
        trie_nodes = 0
        level_patterns = 0

        with open(f"{run_id}.log") as out:
            for line in out:
                stat = re.match(r"(?P<tp>\d+) good, (?P<fp>\d+) bad, (?P<fn>\d+) missed", line)
                if stat is not None:
                    tp = int(stat["tp"])
                    fp = int(stat["fp"])
                    fn = int(stat["fn"])
                stat = re.match(r"pattern trie has (?P<trie_nodes>\d+) nodes, trie_max = \d+, \d+ outputs", line)
                if stat is not None:
                    trie_nodes = int(stat["trie_nodes"])
                stat = re.match(r"total of (?P<level_patterns>\d+) patterns at hyph_level \d+", line)
                if stat is not None:
                    level_patterns = int(stat["level_patterns"])
                stat = re.match(r"(?P<nodes_deleted>\d+) nodes and \d+ outputs deleted", line)
                if stat is not None:
                    trie_nodes -= int(stat["nodes_deleted"])

        return {"tp": tp, "fp": fp, "fn": fn, "trie_nodes" : trie_nodes, "level_patterns": level_patterns}

    def dump_bad(self, output_file: str, levels: int):
        pattmp_path = os.path.join(self.temp_dir, f"pattmp.{levels}")
        os.system(f"grep '\\.' {pattmp_path} > {output_file}")

    def export_patterns(self, output_path: str, levels: int):
        patterns_path = os.path.join(self.temp_dir, f"{levels}.pat")
        os.system(f"mv {patterns_path} {output_path}")

    def clean(self):
        """
        Delete all temporary files used during computations.
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

    def reset(self, tmp_suffix: str = ""):
        """
        Reset the object to initial state
        :param tmp_suffix: suffix to temporary directory name
        """
        if (os.path.exists(self.temp_dir)):
            self.clean()

        self.__create_temp_env(self.initial_suffix + tmp_suffix)
        self.clear_cache()

    def __create_temp_env(self, tmp_suffix: str):
        wl_dir = os.path.dirname(self.wordlist_path)
        self.temp_dir: str = os.path.join(wl_dir, "tmp" + tmp_suffix)

        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)

        if "0.pat" not in os.listdir(self.temp_dir):
            os.system(f"touch {self.temp_dir}/0.pat")

        self.max_id: int = 0
