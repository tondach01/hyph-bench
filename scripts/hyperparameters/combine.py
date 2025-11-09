import os
import metaheuristic

class Combiner:
    """
    Combine samples through levels. Abstract class, instantiate one of its subclasses.
    """
    def __init__(self, meta: metaheuristic.Metaheuristic, verbose: bool = False):
        self.meta = meta
        self.level = 0
        self.verbose = verbose

    def run(self, pattern_file: str = ""):
        """
        Run the metaheuristic for all levels. Exact implementation depends on subclasses.
        :param pattern_file: output pattern file name (<timestamp>-<run_id>.pat by default)
        :return: pattern file names
        """
        return NotImplemented

    def final_patterns(self, pattern_file: str = ""):
        """
        Move pattern file to wordlist directory
        :param pattern_file: output pattern file name (<timestamp>-<run_id>.pat by default)
        :return: pattern file names
        """
        pattern_files = []
        for pop in self.meta.population:
            if not pattern_file:
                pattern_file = f"{pop.timestamp}-{pop.run_id}.pat"
            command = (f"mv {self.meta.scorer.temp_dir}/{pop.run_id}.pat "
                       f"{self.meta.scorer.wordlist_path.rsplit('/', maxsplit=1)[0]}/{pattern_file}")
            os.system(command)
            pattern_files.append(pattern_file)
        self.meta.scorer.clean()
        return pattern_files


class SimpleCombiner(Combiner):
    """
    No combinations of samples from previous level with new candidates, just one of each it selected (potentially
    unstable if .meta.population_size > 1)
    """
    def __init__(self, meta: metaheuristic.Metaheuristic, verbose: bool = False):
        super().__init__(meta, verbose)

    def run(self, pattern_file: str = ""):
        s = self.meta.sampler.sample()
        while s is not None:
            self.level += 1
            if self.verbose:
                print("Running metaheuristic on level", self.level)

            prev = self.meta.get_ids().pop()
            candidate = s.copy({"level": self.level, "prev": prev})
            self.meta.scorer.score(candidate)

            self.meta.population = [candidate]

            self.meta.run_level()
            if self.verbose:
                print("Population selected for next level:", [str(pop) for pop in self.meta.population])
            if self.meta.statistic is not None:
                self.meta.statistic.level_outputs.append(self.meta.population.copy())
            s = self.meta.sampler.sample()
        return Combiner.final_patterns(self, pattern_file)


class AllWithAllCombiner(Combiner):
    """
    All samples from previous levels are evaluated with each fresh sample, <.meta.population_size> best are selected
    """
    def __init__(self, meta: metaheuristic.Metaheuristic, n_levels: int, verbose: bool = False):
        super().__init__(meta, verbose)
        self.n_levels = n_levels

    def run(self, pattern_file: str = ""):
        while self.level < self.n_levels:
            self.level += 1
            if self.verbose:
                print("Running metaheuristic on level", self.level)
            fresh = self.meta.sampler.sample_n(self.meta.population_size)
            candidates = []

            for prev in self.meta.get_ids():
                for f in fresh:
                    candidate = f.copy({"level": self.level, "prev": prev})
                    self.meta.scorer.score(candidate)
                    candidates.append(candidate)

            candidates.sort(key=lambda x: (x.n_patterns, x.precision(), x.recall()))
            self.meta.population = candidates[:self.meta.population_size]

            self.meta.run_level()
            if self.verbose:
                print("Population selected for next level:", [str(pop) for pop in self.meta.population])
            if self.meta.statistic is not None:
                self.meta.statistic.level_outputs.append(self.meta.population.copy())
        return Combiner.final_patterns(self, pattern_file)
