from . import sample
from . import score
from . import stats

PATGEN_MAX_LEVELS = 9

class Metaheuristic:
    """
    Abstract class encompassing all metaheuristics. Should not be instantiated itself.
    """
    def __init__(self, scorer: score.PatgenScorer, sampler: sample.Sampler, n_samples: int, statistic: stats.LearningInfo = None):
        self.sampler: sample.Sampler = sampler
        self.scorer: score.PatgenScorer = scorer
        self.population: list = []
        self.population_size: int = n_samples
        self.statistic = statistic

    def new_population(self):
        """
        Create new population and assign it to self.population. Exact implementation differs for each metaheuristic
        :return: True if new population is different from the old one
        """
        pass

    def run_level(self):
        """
        Compute final population for one level of patgen generation and then remove unused temporary files
        """
        while self.new_population():
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

    def reset(self):
        """
        Reset the object to initial state
        """
        self.scorer.reset()
        self.sampler.reset()
        if self.statistic is not None:
            self.statistic.reset()
        self.population = []


class HillClimbing(Metaheuristic):
    """
    Hill climbing metaheuristic: always choose the best neighbour
    """
    def __init__(self, scorer: score.PatgenScorer, sampler: sample.Sampler, n_samples: int = 1, statistic: stats.LearningInfo = None):
        super().__init__(scorer, sampler, n_samples, statistic)
        self.visited = set()

    def reset(self):
        Metaheuristic.reset(self)
        self.visited = set()

    def new_population(self):
        climbed = False

        for i in range(self.population_size):

            nodes_old = self.population[i].stats["trie_nodes"]
            nodes_new = nodes_old
            f_old = self.population[i].f_score(1/7)

            for n in self.get_neighbours(i):
                trie_nodes, f = self.scorer.score(n)
                if trie_nodes < nodes_old and f > f_old and trie_nodes < nodes_new:
                    self.population[i] = n
                    nodes_new = trie_nodes
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

    def run_level(self):
        self.visited.clear()
        Metaheuristic.run_level(self)


class NoMetaheuristic(Metaheuristic):
    """
    No metaheuristic
    """
    def __init__(self, scorer: score.PatgenScorer, sampler: sample.Sampler, statistic: stats.LearningInfo = None):
        super().__init__(scorer, sampler, n_samples=1, statistic=statistic)

    def new_population(self):
        return False
