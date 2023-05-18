import numpy as np
import scipy.interpolate as spi
from scipy.spatial import distance_matrix
import random
from statsmodels.distributions.empirical_distribution import ECDF

def _point_distances(data):
    if len(data.shape) == 1:
        data = data.reshape(-1,1)
    dist = distance_matrix(data.reshape(-1,1), data.reshape(-1,1))
    dist = np.triu(dist).flatten()
    dist = [d for d in dist if d>0]
    return dist

class pairdistance_representation:
    def __init__(self, norm="sup", n_interpolation_points=100, mode="approximate"):
        self.representation = None
        self.xs = None
        self.max_range = -np.Inf
        self.n_interpolation_points = n_interpolation_points
        self.n_fitted = 0
        self.fitted = False
        self.norm = norm
        self.mode = mode
        self.approximate_n_trials = 100
        self.approximate_points = 20000

    def fit(self, samples):
        self.max_range = -np.Inf
        if self.mode == "exact":
            raise NotImplemented('mode=exact not implemented')
            # for sample in samples:
            #     ecc = np.array(compute_ECC_contributions_alpha(sample))
            #     ecc[:, 1] = np.cumsum(ecc[:, 1])
            #     jumps.update(ecc[:, 0])  # FIXME: ecc[:, 0] is stored in eccs anyway
            #     self.max_range = max(self.max_range, ecc[-1, 0])
            #     eccs.append(ecc)
            # self.xs = np.sort(list(jumps))
            # self.representation = self.xs * 0
            # # extend all ecc so that it include the max_range
            # for ecc in eccs:
            #     ecc_extended = np.vstack([ecc, [self.max_range, 1]])
            #     interpolator = spi.interp1d(ecc_extended[:, 0], ecc_extended[:, 1], kind="previous")
            #     y_inter = interpolator(self.xs)
            #     self.representation += y_inter
            # self.representation /= len(samples)
        else:
            # find jump positions based on given number of ecc curves
            approximate_n_trials = np.min([self.approximate_n_trials, len(samples)])
            trial_samples = random.choices(samples, k=approximate_n_trials)
            jumps = set()
            for sample in trial_samples:
                dist = _point_distances(sample)
                self.max_range = max([self.max_range, np.max(dist)])
                jumps.update(dist)
            jumps = np.sort(list(jumps))
            jumps_step = int(len(jumps) / self.approximate_points)
            jumps_step = max(jumps_step, 1)
            self.xs = jumps[::jumps_step]
            self.representation = self.xs * 0
            # interpolate ECC curves on the grid
            for sample in samples:
                dist = _point_distances(sample)
                ecdf = ECDF(dist)
                self.representation += ecdf(self.xs)
            self.representation /= len(samples)
        self.fitted = True

    def transform(self, samples):
        if not self.fitted:
            raise RuntimeError("Run fit() before transform()")

        dist = []
        representations = []
        for sample in samples:
            pair_dist = _point_distances(sample)
            ecdf = ECDF(pair_dist)
            representation = ecdf(self.xs)
            representations.append(representation)
            if self.norm == "l1":
                dist.append(np.trapz(np.abs(representation - self.representation), x=self.xs))
            elif self.norm == "l2":
                dist.append(np.trapz((representation - self.representation) ** 2, x=self.xs))
            else:  # sup
                dist.append(np.max(np.abs(representation - self.representation)))

        return dist, representations
