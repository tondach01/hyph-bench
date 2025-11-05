import os
import sys
import metaheuristic

class Combiner:
    """
    Combine samples through levels. Abstract class, instantiate one of its subclasses.
    """
    def __init__(self, meta: metaheuristic.Metaheuristic):
        self.meta = meta
        self.level = 0

    def run(self):
        """
        Run the metaheuristic for all levels. Exact implementation depends on subclasses
        """
        pass

class SimpleCombiner(Combiner):
    def __init__(self, meta: metaheuristic.Metaheuristic):
        super().__init__(meta)

    def run(self):
        s = self.meta.sampler.sample()
        while s is not None:
            self.level += 1
            print("Running metaheuristic on level", self.level)

            prev = self.meta.get_ids().pop()
            candidate = s.copy({"level": self.level, "prev": prev})
            self.meta.scorer.score(candidate)
            print(str(candidate))

            self.meta.population = [candidate]

            self.meta.run_level()
            print("Population selected for next level:", [str(pop) for pop in self.meta.population])
            if self.meta.statistic is not None:
                self.meta.statistic.level_outputs.append(self.meta.population.copy())
            s = self.meta.sampler.sample()


class AllWithAllCombiner(Combiner):
    def __init__(self, meta: metaheuristic.Metaheuristic, n_levels: int):
        super().__init__(meta)
        self.n_levels = n_levels

    def run(self):
        """
        Run the metaheuristic for all levels
        """
        while self.level < self.n_levels:
            self.level += 1
            print("Running metaheuristic on level", self.level)
            fresh = self.meta.sampler.sample_n(self.meta.population_size)
            candidates = []

            for prev in self.meta.get_ids():
                for f in fresh:
                    candidate = f.copy({"level": self.level, "prev": prev})
                    self.meta.scorer.score(candidate)
                    print(str(candidate))
                    candidates.append(candidate)

            candidates.sort(key=lambda x: (x.n_patterns, x.precision(), x.recall()))
            self.meta.population = candidates[:self.meta.population_size]

            self.meta.run_level()
            print("Population selected for next level:", [str(pop) for pop in self.meta.population])
            if self.meta.statistic is not None:
                self.meta.statistic.level_outputs.append(self.meta.population.copy())

        for run_id in self.meta.get_ids():
            os.system(f"mv {self.meta.scorer.temp_dir}/{run_id}.pat "
                      f"{self.meta.scorer.wordlist_path.rsplit('/', maxsplit=1)[0]}/{run_id}.pat")
