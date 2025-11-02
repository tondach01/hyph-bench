import sys
import metaheuristic

class Combiner:
    """
    Combine samples through levels. Abstract class, instantiate one of its subclasses.
    """
    def __init__(self, meta: metaheuristic.Metaheuristic, logfile: str = ""):
        self.meta = meta
        self.logfile = logfile
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
        if self.logfile:
            log = open(self.logfile, "w")
        else:
            log = sys.stdout

        s = self.meta.sampler.sample()
        while s is not None:
            self.level += 1
            print("Running metaheuristic on level", self.level, file=log)

            prev = self.meta.get_ids().pop()
            candidate = s.copy({"level": self.level, "prev": prev})
            self.meta.scorer.score(candidate)
            print(str(candidate), file=log)

            self.meta.population = [candidate]

            self.meta.run_level(log)
            print("Population selected for next level:", [str(pop) for pop in self.meta.population], file=log)
            s = self.meta.sampler.sample()

        if self.logfile:
            log.close()

class AllWithAllCombiner(Combiner):
    def __init__(self, meta: metaheuristic.Metaheuristic, n_levels: int, logfile: str = ""):
        super().__init__(meta, logfile)
        self.n_levels = n_levels

    def run(self):
        """
        Run the metaheuristic for all levels
        """
        if self.logfile:
            log = open(self.logfile, "w")
        else:
            log = sys.stdout

        while self.level < self.n_levels:
            self.level += 1
            print("Running metaheuristic on level", self.level, file=log)
            fresh = self.meta.sampler.sample_n(self.meta.population_size)
            candidates = []

            for prev in self.meta.get_ids():
                for f in fresh:
                    candidate = f.copy({"level": self.level, "prev": prev})
                    self.meta.scorer.score(candidate)
                    print(str(candidate), file=log)
                    candidates.append(candidate)

            candidates.sort(key=lambda x: (x.n_patterns, x.precision(), x.recall()))
            self.meta.population = candidates[:self.meta.population_size]

            self.meta.run_level(log)
            print("Population selected for next level:", [str(pop) for pop in self.meta.population], file=log)

        if self.logfile:
            log.close()
