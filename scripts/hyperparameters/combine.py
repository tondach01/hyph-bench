import os

from . import metaheuristic


class Combiner:
    """
    Combine samples through levels. Abstract class, instantiate one of its subclasses.
    """
    def __init__(self, meta: metaheuristic.Metaheuristic, verbose: bool = False):
        self.meta = meta
        self.level = 0
        self.verbose = verbose

    def run(self, out_dir: str = ""):
        """
        Run the metaheuristic for all levels. Exact implementation depends on subclasses.
        :param out_dir: output directory name (. by default)
        :return: pattern file name
        """
        return NotImplemented

    def final_patterns(self, out_dir: str = ""):
        """
        Determine the best sample out of the final population and move its generated patterns into specified directory.
        Afterward the temporaries are cleaned.<timestamp>-<run_id>.pat
        :param out_dir: output directory name (. by default)
        :return: pattern file name
        """
        best = self.meta.population[0] #TODO determine the best sample out of the final population
        pattern_file = f"{best.timestamp}-{best.run_id}.pat"
        if not out_dir:
            out_dir = "."
        new_patfile = f"{out_dir}/{pattern_file}"
        command = f"mv {self.meta.scorer.temp_dir}/{best.run_id}.pat {new_patfile}"
        os.system(command)
        return new_patfile

    def reset(self):
        """
        Reset the object to initial state.
        """
        self.meta.reset()
        self.level = 0


class SimpleCombiner(Combiner):
    """
    No combinations of samples from previous level with new candidates, just one of each it selected (potentially
    unstable if .meta.population_size > 1)
    """
    def __init__(self, meta: metaheuristic.Metaheuristic, verbose: bool = False):
        super().__init__(meta, verbose)

    def run(self, out_dir: str = ""):
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
        return Combiner.final_patterns(self, out_dir)


class AllWithAllCombiner(Combiner):
    """
    All samples from previous levels are evaluated with each fresh sample, <.meta.population_size> best are selected
    """
    def __init__(self, meta: metaheuristic.Metaheuristic, n_levels: int, verbose: bool = False):
        super().__init__(meta, verbose)
        self.n_levels = n_levels

    def run(self, out_dir: str = ""):
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
        return Combiner.final_patterns(self, out_dir)
