import random
import re

MAX_PAT_FINISH = 15

class Sample:
    """
    Single sampled hyperparameter setting
    """
    def __init__(self, params):
        self.param_dict: dict = dict()

        self.level: int = params.get("level", 1)
        self.param_dict["level"] = self.level

        self.prev: int = params.get("prev", 0)
        self.param_dict["prev"] = self.prev

        self.pat_start: int = params.get('pat_start', 2)
        self.param_dict["pat_start"] = self.pat_start

        self.pat_finish: int = params.get('pat_finish', 2)
        self.param_dict["pat_finish"] = self.pat_finish

        self.good_weight: int = params.get('good_weight', 1)
        self.param_dict["good_weight"] = self.good_weight

        self.bad_weight: int = params.get('bad_weight', 1)
        self.param_dict["bad_weight"] = self.bad_weight

        self.threshold: int = params.get('threshold', 1)
        self.param_dict["threshold"] = self.threshold

        self.stats: dict = dict()
        self.run_id: int = -1
        self.n_patterns: int = -1

    def __eq__(self, other):
        """
        Equality between two samples. Not dependent on level in which they are processed (self.level)
        :param other: instance to compare with
        :return: True if the two object represent the same sample
        """
        if not isinstance(other, Sample):
            return False
        s_good_base, s_bad_base = self.base_values()
        o_good_base, o_bad_base = other.base_values()
        return (
            self.prev == other.prev and self.pat_start == other.pat_start and self.pat_finish == other.pat_finish and
            s_good_base == o_good_base and s_bad_base == o_bad_base and self.threshold == other.threshold
        )

    def __hash__(self):
        """
        Hash the sample based on its attributes. Not dependent on level in which they are processed (self.level)
        :return: hash of the sample
        """
        s_good_base, s_bad_base = self.base_values()
        return hash((self.prev, self.pat_start, self.pat_finish, s_good_base, s_bad_base, self.threshold))

    def __str__(self):
        return (f"Sample {self.run_id}: prev={self.prev} pat_start={self.pat_start} pat_finish={self.pat_finish} "
                f"good_weight={self.good_weight} bad_weight={self.bad_weight} threshold={self.threshold} "
                f"n_patterns={self.n_patterns} precision={round(self.precision(), 3)} recall={round(self.recall(), 3)}")

    def copy(self, new_vals=None):
        """
        Create new sample with the same attributes as this one
        :param new_vals: optional dictionary with new values to assign, irrelevant keys are ignored
        :return: new Sample object
        """
        new_dict = self.param_dict.copy()
        if new_vals is not None and isinstance(new_vals, dict):
            for key in new_vals:
                new_dict[key] = new_vals[key]
        return Sample(new_dict)

    def base_values(self):
        """
        Divide .good_weight and .bad_weight attributes by their GCD to get the smallest possible combination.
        Threshold is fixed at 1, thus not included in computation
        :return: the three attributes in aforementioned order divided by their GCD
        """
        gcd = 1
        good_wt, bad_wt = self.good_weight, self.bad_weight
        while gcd < min(good_wt, bad_wt):
            gcd += 1
            while good_wt % gcd == 0 and bad_wt % gcd == 0:
                good_wt //= gcd
                bad_wt //= gcd
        return good_wt, bad_wt

    def precision(self):
        """
        Compute precision of the sample, if previously run through scorer (.stats are set)
        :return: precision TP/(TP+FP) if .stats set, else -1
        """
        if (not isinstance(self.stats, dict) or "tp" not in self.stats or "fp" not in self.stats
                or not isinstance(self.stats["tp"], int) or not isinstance(self.stats["fp"], int)):
            return -1
        return self.stats["tp"]/(self.stats["tp"]+self.stats["fp"])

    def recall(self):
        """
        Compute recall of the sample, if previously run through scorer (.stats are set)
        :return: recall TP/(TP+FN) if .stats set, else -1
        """
        if (not isinstance(self.stats, dict) or "tp" not in self.stats or "fn" not in self.stats
                or not isinstance(self.stats["tp"], int) or not isinstance(self.stats["fn"], int)):
            return -1
        return self.stats["tp"]/(self.stats["tp"]+self.stats["fn"])

    def f_score(self, n: float):
        if n <= 0:
            return -1
        p, r = self.precision(), self.recall()
        if p == -1 or r == -1:
            return -1
        return (1 + n*n) * p * r / ((n*n * p) + r)


class Sampler:
    """
    Abstract class encompassing all samplers. Should not be instantiated itself.
    """
    def __init__(self, ranges: dict):
        self.pat_start_range: tuple = (1, 15) if "pat_start" not in ranges else ranges["pat_start"]
        self.pat_finish_range: tuple = (1, 15) if "pat_finish" not in ranges else ranges["pat_finish"]
        self.good_weight_range: tuple = (1, 15) if "good_weight" not in ranges else ranges["good_weight"]
        self.bad_weight_range: tuple = (1, 15) if "bad_weight" not in ranges else ranges["bad_weight"]
        self.threshold_range: tuple = (1, 15) if "threshold" not in ranges else ranges["threshold"]

    def sample(self):
        """
        Create one sample hyperparameter setting. Exact implementation differs for each sampler
        :return: Sample object with hyperparameter values
        """
        pass

    def sample_n(self, n: int):
        """
        Create multiple samples.
        :param n: number of samples to generate
        :return: list of samples
        """
        samples = list()
        for _ in range(n):
            samples.append(self.sample())
        return samples

    def is_ok_value(self, sample: Sample, param: str, val: int):
        """
        Checks whether the sample parameter can have given value
        :param sample: sample to check
        :param param: hyperparameter name
        :param val: hyperparameter value
        :return: True if parameter exists and value is possible
        """
        if param == "good_weight":
            return self.good_weight_range[0] <= val <= self.good_weight_range[1]
        elif param == "bad_weight":
            return self.bad_weight_range[0] <= val <= self.bad_weight_range[1]
        elif param == "threshold":
            return self.threshold_range[0] <= val <= self.threshold_range[1]
        elif param == "pat_start":
            return (self.pat_start_range[0] <= val <= self.pat_start_range[1]) and val <= sample.pat_finish
        elif param == "pat_finish":
            return (self.pat_finish_range[0] <= val <= self.pat_finish_range[1]) and val >= sample.pat_start
        return False


class RandomSampler(Sampler):
    """
    Sampling by random initialisation
    """
    def __init__(self, ranges: dict, random_state=None):
        super().__init__(ranges)
        random.seed(random_state)

    def sample(self):
        values = dict()

        values["pat_start"] = random.randint(self.pat_start_range[0], self.pat_start_range[1])
        values["pat_finish"] = random.randint(max(self.pat_finish_range[0], values["pat_start"]), self.pat_finish_range[1])
        values["good_weight"] = random.randint(self.good_weight_range[0], self.good_weight_range[1])
        values["bad_weight"] = random.randint(self.bad_weight_range[0], self.bad_weight_range[1])
        values["threshold"] = random.randint(self.threshold_range[0], self.threshold_range[1])

        return Sample(values)


class FileSampler(Sampler):
    """
    Produce samples from file. If end of file is reached and .repeat flag is set to True repeats sampling from
    the beginning, otherwise returns None upon reaching EOF. If line format does not conform to required format,
    returns None.
    """
    def __init__(self, file, repeat: bool = False):
        super().__init__(ranges=dict())
        self.file = file
        self.file_ptr = open(file)
        self.file_open = True
        self.repeat = repeat
        self.pattern = re.compile(r"\s*(?P<pat_start>\d+)\s+(?P<pat_finish>\d+)\s+(?P<good_weight>\d+)\s+(?P<bad_weight>\d+)\s+(?P<threshold>\d+)")

    def sample(self):
        if not self.file_open and self.repeat:
            self.file_ptr = open(self.file)
            self.file_open = True
        elif not self.file_open:
            return None
        line = self.file_ptr.readline()
        while line.startswith("#"):
            line = self.file_ptr.readline()
            continue
        if not line.strip():
            self.file_ptr.close()
            self.file_open = False
            return None
        match = re.match(self.pattern, line)
        if match is None:
            return None
        return Sample({param: int(match[param]) for param in ["pat_start", "pat_finish", "good_weight", "bad_weight", "threshold"]})
