import os
import re
import matplotlib
import matplotlib.pyplot as plt

import sample

class DatasetInfo:
    def __init__(self, file: str):
        abspath = os.path.abspath(file)
        abspath = re.sub(r"\\+", "/", abspath)
        path = abspath.split("/")
        self.lang = "" if len(path) < 3 else path[-3]
        self.dataset_name = "" if len(path) < 2 else path[-2]
        with open(file, "rb") as f:
            self.size_lines = sum(1 for _ in f)
        self.size_bytes = os.path.getsize(file)

    def __str__(self):
        return f"Dataset {self.dataset_name}: lang = {self.lang}, {self.size_bytes} B, {self.size_lines} lines"

class LearningInfo:
    def __init__(self):
        self.level_outputs = list()

    def visualise(self, metric = "precision"):
        metric_funcs = list()
        if isinstance(metric, str):
            metric = [metric]
        if "precision" in metric:
            metric_funcs.append(("precision", sample.Sample.precision))
        if "recall" in metric:
            metric_funcs.append(("recall", sample.Sample.recall))

        matplotlib.use('TkAgg')

        for name, func in metric_funcs:
            data = dict(level=list(), metric=list())
            for population in self.level_outputs:
                for pop in population:
                    data["level"].append(pop.level)
                    data["metric"].append(func(pop))
            plt.plot("level", "metric", data=data.copy(), label=name)

        plt.legend()
        plt.show()

class PatternsInfo:
    def __init__(self, file: str, s: sample.Sample):
        abspath = os.path.abspath(file)
        abspath = re.sub(r"\\+", "/", abspath)
        path = abspath.split("/")
        self.lang = "" if len(path) < 3 else path[-3]
        self.dataset_name = "" if len(path) < 2 else path[-2]
        self.n_patterns = s.n_patterns
        self.size_bytes = os.path.getsize(file)
        self.s = s

    def __str__(self):
        return f"Patterns for {self.lang}/{self.dataset_name}: {self.size_bytes} B, {self.n_patterns} patterns, precision {self.s.precision()}, recall {self.s.recall()}"

if __name__ == "__main__":
    print(str(DatasetInfo("../../data/it/wiktionary/it_wiktionary_251001.wlh")))