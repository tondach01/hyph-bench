import sample
import score

PATGEN_MAX_LEVELS = 9

class Metaheuristic:
    """
    Abstract class encompassing all metaheuristics. Should not be instantiated itself.
    """
    def __init__(self, scorer: score.PatgenScorer, sampler: sample.Sampler, n_samples: int, logfile: str = ""):
        self.sampler: sample.Sampler = sampler
        self.scorer: score.PatgenScorer = scorer
        self.population: list = []
        self.population_size: int = n_samples
        self.logfile = logfile

    def new_population(self, log):
        """
        Create new population and assign it to self.population. Exact implementation differs for each metaheuristic
        :param log: opened stream to print logs
        :return: True if new population is different from the old one
        """
        pass

    def run_level(self, log):
        """
        Compute final population for one level of patgen generation and then remove unused temporary files
        :param log: opened stream to print logs
        """
        while self.new_population(log):
            continue
        self.scorer.clean_unused(self.get_ids())
        self.scorer.clear_cache()

    def get_ids(self):
        """
        Get IDs of all samples currently in use
        :return: set of IDs in use
        """
        if self.population:
            return set([pop.run_id for pop in self.population])
        else:
            return {0}


class HillClimbing(Metaheuristic):
    """
    Hill climbing metaheuristic: always choose the best neighbour
    """
    def __init__(self, scorer: score.PatgenScorer, sampler: sample.Sampler, n_samples: int = 1, logfile: str = ""):
        super().__init__(scorer, sampler, n_samples, logfile)
        self.visited = set()

    def new_population(self, log):
        climbed = False

        for i in range(self.population_size):

            pat_old = self.population[i].n_patterns
            pat_new = pat_old
            acc_old = (self.population[i].precision(), self.population[i].recall())

            for n in self.get_neighbours(i):
                pat, prec, rec = self.scorer.score(n)
                print(str(n), file=log)
                if pat < pat_old and (prec, rec) > acc_old and pat < pat_new:
                    self.population[i] = n
                    pat_new = pat
                    climbed = True

        return climbed

    def get_neighbours(self, ind: int = 0):
        """
        Find all neighbouring (one hyperparameter differs by one) samples
        :param ind: index of the sample to expand in self.population
        :return: list of neighbours
        """
        pop: sample.Sample = self.population[ind]
        neighbours = []

        for param, val in pop.param_dict.items():
            if param in ["level", "prev"]:
                continue

            for new_val in [val+1, val-1]:
                new_pop = pop.copy({ param: new_val })
                if not self.sampler.is_ok_value(new_pop, param, new_val):
                    continue
                p_hash = new_pop.__hash__()
                if p_hash in self.visited:
                    continue
                neighbours.append(new_pop)
                self.visited.add(p_hash)

        return neighbours

    def run_level(self, log):
        self.visited.clear()
        Metaheuristic.run_level(self, log)


class NoMetaheuristic(Metaheuristic):
    """
    No metaheuristic
    """
    def __init__(self, scorer: score.PatgenScorer, sampler: sample.Sampler, logfile: str = ""):
        super().__init__(scorer, sampler, n_samples=1, logfile=logfile)

    def new_population(self, log):
        return False
