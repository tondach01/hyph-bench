"""
Gaussian Process optimizer for patgen parameters.

Optimizes a 4-dimensional parameter space: bad_weight for each of 4 levels.
Uses Upper Confidence Bound (UCB) acquisition for exploration-exploitation balance.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from typing import List, Tuple, Optional, Dict, Any
import pickle
import os

from .objectives import ObjectiveFunction


class GPOptimizer:
    """
    Gaussian Process optimizer for patgen parameters.

    Optimizes N integer parameters using Upper Confidence Bound (UCB).
    """

    def __init__(self, objective: ObjectiveFunction, seed: int = 42,
                 bounds: List[Tuple[int, int]] = None,
                 min_samples_for_gp: int = 5):
        """
        Initialize GP Optimizer.

        Args:
            objective: Objective function to minimize/maximize
            seed: Random seed
            bounds: List of (min, max) tuples for each dimension
            min_samples_for_gp: Minimum samples before fitting GP
        """
        self.objective = objective
        self.rng = np.random.RandomState(seed)
        self.bounds = bounds if bounds else [(1, 9)] * 4
        self.n_dims = len(self.bounds)
        self.min_samples_for_gp = min_samples_for_gp

        # N-dimensional Matern kernel
        kernel = Matern(
            nu=2.5,
            length_scale=[1.0] * self.n_dims,
            length_scale_bounds=(1e-3, 1e3)
        ) + WhiteKernel(
            noise_level=0.1,
            noise_level_bounds=(1e-4, 1)
        )

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=seed,
            n_restarts_optimizer=2
        )

        self.X: List[List[int]] = []      # Parameter history
        self.y: List[float] = []          # Score history
        self.results: List[Dict] = []     # Full results history

    def _random_params(self) -> Tuple[int, ...]:
        """Generate random parameter set within bounds."""
        return tuple(self.rng.randint(low, high + 1) for low, high in self.bounds)

    def generate_coarse_grid(self, step: int = 4) -> List[Tuple[int, ...]]:
        """
        Generate coarse grid of parameters.
        
        Args:
            step: Step size (applied to all dimensions)
        """
        from itertools import product
        
        dim_values = []
        for low, high in self.bounds:
            vals = list(range(low, high + 1, step))
            if high not in vals:
                vals.append(high)
            dim_values.append(vals)
            
        grid = list(product(*dim_values))
        self.rng.shuffle(grid)
        return [tuple(g) for g in grid]

    def _predict(self, params: Tuple[int, ...]) -> Tuple[float, float]:
        """
        Get GP prediction and uncertainty for a parameter set.

        Args:
            params: (bad_1, bad_2, bad_3, bad_4) tuple

        Returns:
            (mean, std) prediction
        """
        if len(self.X) < self.min_samples_for_gp:
            return 0.5, 1.0  # Prior before enough data

        x = np.array([list(params)])
        mean, std = self.gp.predict(x, return_std=True)
        return float(mean[0]), float(std[0])

    def update(self, params: Tuple[int, ...], good: int, bad: int, missed: int, 
               n_patterns: int = 0, trie_nodes: int = 0, f_17: float = 0.0,
               good_variance: float = 0.0, bad_variance: float = 0.0, missed_variance: float = 0.0) -> float:
        """
        Update GP with new observation.

        Args:
            params: (bad_1, bad_2, bad_3, bad_4) tuple
            good: True positives
            bad: False positives
            missed: False negatives
            n_patterns: Total pattern count
            trie_nodes: Trie node count

        Returns:
            The computed score
        """
        score = self.objective.score(
            good, bad, missed,
            n_patterns=n_patterns,
            trie_nodes=trie_nodes,
            f17cv=f_17
        )

        self.X.append(list(params))
        self.y.append(score)
        self.results.append({
            'params': params,
            'good': good,
            'bad': bad,
            'missed': missed,
            'n_patterns': n_patterns,
            'trie_nodes': trie_nodes,
            'score': score,
            'good_variance': good_variance,
            'bad_variance': bad_variance,
            'missed_variance': missed_variance
        })

        # Refit GP after enough observations
        if len(self.X) >= self.min_samples_for_gp:
            self.gp.fit(np.array(self.X), np.array(self.y))

        return score

    def suggest_batch(self, n: int = 5, n_candidates: int = 5000,
                      ucb_kappa: float = 2.0) -> List[Tuple[int, ...]]:
        """
        Suggest diverse batch of promising parameters.

        Uses Upper Confidence Bound (UCB) acquisition function:
            UCB = mean + kappa * std

        Args:
            n: Number of suggestions to return
            n_candidates: Number of random candidates to evaluate
            ucb_kappa: Exploration weight (higher = more exploration)

        Returns:
            List of parameter tuples
        """
        if len(self.X) < self.min_samples_for_gp:
            # Random sampling before GP is fitted
            return [self._random_params() for _ in range(n)]

        # Generate candidates
        candidates = [self._random_params() for _ in range(n_candidates)]
        X_cand = np.array([list(c) for c in candidates])

        # UCB acquisition: mean + kappa * std
        mean, std = self.gp.predict(X_cand, return_std=True)
        ucb = mean + ucb_kappa * std

        # Select diverse set from top quartile
        top_indices = np.where(ucb >= np.percentile(ucb, 75))[0]

        selected = []
        selected_set = set()

        # Always include best UCB
        best_idx = np.argmax(ucb)
        selected.append(candidates[best_idx])
        selected_set.add(candidates[best_idx])

        # Add diverse samples from top quartile using max-min distance
        while len(selected) < n and len(top_indices) > 0:
            max_min_dist = -1
            best_idx = -1

            for idx in top_indices:
                if candidates[idx] in selected_set:
                    continue

                # L1 distance to all selected
                dists = [np.sum(np.abs(np.array(candidates[idx]) - np.array(s)))
                         for s in selected]
                min_dist = min(dists)

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = idx

            if best_idx != -1:
                selected.append(candidates[best_idx])
                selected_set.add(candidates[best_idx])
                top_indices = top_indices[top_indices != best_idx]
            else:
                break

        # Fill remaining with random if needed
        while len(selected) < n:
            new_params = self._random_params()
            if new_params not in selected_set:
                selected.append(new_params)
                selected_set.add(new_params)

        return selected

    def exploit_best(self, n: int = 5, n_candidates: int = 10000) -> List[Tuple[int, ...]]:
        """
        Return parameters with highest predicted scores (pure exploitation).

        Args:
            n: Number of suggestions
            n_candidates: Candidates to evaluate

        Returns:
            List of parameter tuples sorted by predicted score (best first)
        """
        if len(self.X) < self.min_samples_for_gp:
            return self.suggest_batch(n)

        candidates = [self._random_params() for _ in range(n_candidates)]
        X_cand = np.array([list(c) for c in candidates])
        mean = self.gp.predict(X_cand)

        # Get unique top predictions
        sorted_indices = np.argsort(mean)[::-1]
        selected = []
        selected_set = set()

        for idx in sorted_indices:
            if candidates[idx] not in selected_set:
                selected.append(candidates[idx])
                selected_set.add(candidates[idx])
                if len(selected) >= n:
                    break

        return selected

    def best_so_far(self) -> Optional[Dict]:
        """Return best result observed so far."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r['score'])

    def get_predictions_for_params(self, params_list: List[Tuple[int, ...]]) -> List[Tuple[float, float]]:
        """
        Get predictions for a list of parameter sets.

        Returns:
            List of (mean, std) tuples
        """
        return [self._predict(p) for p in params_list]

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics of optimization so far."""
        if not self.results:
            return {'n_observations': 0}

        scores = [r['score'] for r in self.results]
        best = self.best_so_far()

        return {
            'n_observations': len(self.results),
            'best_score': best['score'],
            'best_params': best['params'],
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'best_good': best['good'],
            'best_bad': best['bad'],
            'best_missed': best['missed'],
        }

    def save(self, path: str):
        """Save optimizer state to file."""
        state = {
            'gp': self.gp,
            'X': self.X,
            'y': self.y,
            'results': self.results,
            'bounds': self.bounds,
            'objective_name': self.objective.name,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str, objective: ObjectiveFunction) -> 'GPOptimizer':
        """
        Load optimizer state from file.
        """
        opt = cls(objective)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                state = pickle.load(f)
            opt.gp = state['gp']
            opt.X = state['X']
            opt.y = state['y']
            opt.results = state['results']
            if 'bounds' in state:
                opt.bounds = state['bounds']
            elif 'param_range' in state:
                # Backwards compatibility
                opt.bounds = [state['param_range']] * 4
            opt.n_dims = len(opt.bounds)
        return opt

    def get_history_dataframe(self):
        """
        Return optimization history as a pandas DataFrame.
        """
        import pandas as pd

        rows = []
        for i, r in enumerate(self.results):
            row = {
                'iteration': i,
                'good': r['good'],
                'bad': r['bad'],
                'missed': r['missed'],
                'n_patterns': r['n_patterns'],
                'trie_nodes': r['trie_nodes'],
                'score': r['score'],
            }
            # Add params dynamically
            for j, p in enumerate(r['params']):
                row[f'param_{j+1}'] = p
            rows.append(row)

        return pd.DataFrame(rows)
