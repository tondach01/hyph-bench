import os
import re
import matplotlib
import matplotlib.pyplot as plt
import sys

from . import sample

class DatasetInfo:
    def __init__(self, file: str):
        abspath = os.path.abspath(file)
        abspath = re.sub(r"\\+", "/", abspath)
        path = abspath.split("/")
        self.lang = "" if len(path) < 3 else path[-3]
        self.dataset_name = "" if len(path) < 2 else path[-2]
        self.size_lines = 0
        len_total = 0
        hyph_total = 0
        words = set()
        hyphenations = set()
        self.ambiguous = 0
        self.len_min, self.len_max = -1, -1
        with open(file, "r") as f:
            for line in f:
                self.size_lines += 1
                line_len = len(line.strip())
                len_total += line_len
                hyph_total += line.count("-")
                line_nohyph = re.sub("-", "", line)
                if line_nohyph in words and line not in hyphenations:
                    self.ambiguous += 1
                words.add(line_nohyph)
                hyphenations.add(line)
                if self.len_min == -1 or line_len < self.len_min:
                    self.len_min = line_len
                if self.len_max == -1 or line_len > self.len_max:
                    self.len_max = line_len
        self.len_avg = len_total / self.size_lines
        self.hyph_avg = hyph_total / self.size_lines
        self.size_bytes = os.path.getsize(file)

    def report(self, tabular: bool = True):
        """
        Report the statistics of the dataset
        :param tabular: output in LaTeX tabular format
        :return: statistics in desired format
        """
        if not tabular:
            return str(self)
        len_avg = round(self.len_avg, 2)
        hyph_avg = round(self.hyph_avg, 2)
        name = re.sub("_", "\\_", self.dataset_name)
        return f"{self.lang} & {name} & {self.convert_kilobytes()} & {self.size_lines} & {len_avg:.2f} & {hyph_avg:.2f} \\\\"

    def convert_kilobytes(self):
        val = round(self.size_bytes/1024, 1)
        return f"{val:.1f}"

    def __str__(self):
        return (f"Dataset {self.lang}/{self.dataset_name}:\n "
                f"\tsize: {self.convert_kilobytes()} kB, {self.size_lines} lines\n "
                f"\t{round(self.hyph_avg, 2)} avg hyphenators per line, {self.ambiguous} ambiguous hyphenations\n"
                f"\tword lengths: min {self.len_min} max {self.len_max} avg {round(self.len_avg, 2)} (hyphenators incl.)")

class LearningInfo:
    def __init__(self):
        self.level_outputs = list()

    def visualise(self, metric = "precision"):
        metric_funcs = list()
        warn_abs = ("Warning: do not combine absolute accuracy numbers (tp, fp, fn) with "
                    "relative accuracy metrics (precision, recall, fN), number of patterns or "
                    "patgen hyperparameters, the scales are different")
        warn_pat = ("Warning: do not combine number of patterns with accuracy metrics or "
                    "patgen hyperparameters, the scales are different")
        warn_param = ("Warning: do not combine patgen hyperparameters (good_wt, bad_wt, threshold) "
                      "with accuracy metrics or number of patterns, the scales are different")
        if isinstance(metric, str):
            metric = [metric]
        for m in metric:
            f_score = re.match(r"f(?P<n>\d+(\.\d+)?)", m)
            if m == "precision":
                metric_funcs.append(("Precision", sample.Sample.precision))
            elif m == "recall":
                metric_funcs.append(("Recall", sample.Sample.recall))
            elif f_score:
                n = float(f_score["n"])
                metric_funcs.append((f"F{n}", lambda x: x.f_score(n)))
            elif m == "tp":
                if "precision" in metric or "recall" in metric:
                    print(warn_abs, file=sys.stderr)
                metric_funcs.append(("True positives", lambda x: x.stats.get("tp", 0)))
            elif m == "fp":
                if "precision" in metric or "recall" in metric:
                    print(warn_abs, file=sys.stderr)
                metric_funcs.append(("False positives", lambda x: x.stats.get("fp", 0)))
            elif m == "fn":
                if "precision" in metric or "recall" in metric:
                    print(warn_abs, file=sys.stderr)
                metric_funcs.append(("False negatives", lambda x: x.stats.get("fn", 0)))
            elif m == "patterns":
                if len(metric) > 1:
                    print(warn_pat, file=sys.stderr)
                metric_funcs.append(("Patterns", lambda x: x.n_patterns))
            elif m == "good_wt":
                if ("precision" in metric or "recall" in metric or "tp" in metric or "fp" in metric
                        or "fn" in metric or "patterns" in metric):
                    print(warn_param, file=sys.stderr)
                metric_funcs.append(("Good weight", lambda x: x.good_weight))
            elif m == "bad_wt":
                if ("precision" in metric or "recall" in metric or "tp" in metric or "fp" in metric
                        or "fn" in metric or "patterns" in metric):
                    print(warn_param, file=sys.stderr)
                metric_funcs.append(("Bad weight", lambda x: x.bad_weight))
            elif m == "threshold":
                if ("precision" in metric or "recall" in metric or "tp" in metric or "fp" in metric
                        or "fn" in metric or "patterns" in metric):
                    print(warn_param, file=sys.stderr)
                metric_funcs.append(("Threshold", lambda x: x.threshold))
            else:
                print(f"Unknown metric {m} provided", file=sys.stderr)

        matplotlib.use('TkAgg')

        for name, func in metric_funcs:
            data = dict(level=list(), metric=list())
            for population in self.level_outputs:
                for pop in population:
                    data["level"].append(pop.level)
                    data["metric"].append(func(pop))
            plt.plot("level", "metric", data=data.copy(), label=name)

        plt.legend()
        plt.xlabel("Hyphenation level")
        plt.xticks([n+1 for n in range(len(self.level_outputs))])
        plt.show()

    def reset(self):
        self.level_outputs = list()

class PatternsInfo:
    def __init__(self, file: str, s: sample.Sample):
        abspath = os.path.abspath(file)
        abspath = re.sub(r"\\+", "/", abspath)
        path = abspath.split("/")
        self.lang = "" if len(path) < 3 else path[-3]
        self.dataset_name = "" if len(path) < 2 else path[-2]
        self.n_patterns = s.n_patterns
        self.size_bytes = os.path.getsize(file)
        self.levels = 0
        len_total = 0
        with open(file) as f:
            for line in f:
                len_line = len(line.strip())
                len_total += len_line
                levels = re.findall("\d+", line)
                for level in levels:
                    if int(level) > self.levels:
                        self.levels = int(level)
        self.len_avg = len_total / self.n_patterns
        self.s = s

    def __str__(self):
        return (f"Patterns for {self.lang}-{self.dataset_name}:\n"
                f"\tsize: {self.size_bytes} B, {self.n_patterns} patterns\n"
                f"\tavg. length {round(self.len_avg, 2)}, {self.levels} levels\n"
                f"\tprecision {round(self.s.precision(),3)}, recall {round(self.s.recall(),3)}")
