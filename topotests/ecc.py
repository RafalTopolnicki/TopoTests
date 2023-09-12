import numpy as np
import gudhi as gd
import scipy.interpolate as spi
import random

# computes each simplex contribution to the ECC
# function by Davide
def compute_ECC_contributions_alpha(point_cloud):
    points_n = point_cloud.shape[0]
    points_dim = point_cloud.shape[1]
    alpha_complex = gd.AlphaComplex(points=point_cloud)
    simplex_tree = alpha_complex.create_simplex_tree()

    ecc = {}

    for s, f in simplex_tree.get_filtration():
        dim = len(s) - 1
        ecc[f] = ecc.get(f, 0) + (-1) ** dim

    # remove the contributions that are 0
    to_del = []
    for key in ecc:
        if ecc[key] == 0:
            to_del.append(key)
    for key in to_del:
        del ecc[key]

    ecc = sorted(list(ecc.items()), key=lambda x: x[0])
    factor = points_n**(2.0/points_dim)
    ecc = [(e[0]*factor, e[1]/points_n) for e in ecc]
    return ecc

class ecc_representation:
    def __init__(self, norm="sup", mode="approximate"):
        self.representation = None
        self.xs = None
        self.max_range = -np.Inf
        self.n_fitted = 0
        self.fitted = False
        self.norm = norm
        self.mode = mode
        self.approximate_n_trials = 100
        self.approximate_points = 20000

    def fit(self, samples):
        # TODO: this implementation is memomy-ineffcient - we don't have to store all ecc to compute the mean
        # TODO: this becomes a problem for Rips
        self.max_range = -np.Inf
        eccs = []
        jumps = set()
        if self.mode == "exact":
            raise NotImplemented("Mode exact is not supported anymore")
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
                ecc = np.array(compute_ECC_contributions_alpha(sample))
                self.max_range = max([self.max_range, ecc[-1, 0]])
                jumps.update(ecc[:, 0])
            jumps = np.sort(list(jumps))
            jumps_step = int(len(jumps) / self.approximate_points)
            jumps_step = max(jumps_step, 1)
            self.xs = jumps[::jumps_step]
            self.representation = self.xs * 0
            # interpolate ECC curves on the grid
            for sample in samples:
                ecc = np.array(compute_ECC_contributions_alpha(sample))
                ecc[:, 1] = np.cumsum(ecc[:, 1])
                # cut ecc on self.max_range
                range_ind = ecc[:, 0] < self.max_range
                ecc = ecc[range_ind, :]
                # add ecc max to the end - should we 0 or 1/n? but what if we have different n?
                ecc_extended = np.vstack([ecc, [self.max_range, 0]])
                interpolator = spi.interp1d(ecc_extended[:, 0], ecc_extended[:, 1], kind="previous")
                y_inter = interpolator(self.xs)
                self.representation += y_inter
            self.representation /= len(samples)
        self.fitted = True

    def transform(self, samples):
        if not self.fitted:
            raise RuntimeError("Run fit() before transform()")

        dist = []
        representations = []
        for sample in samples:
            ecc = np.array(compute_ECC_contributions_alpha(sample))
            ecc[:, 1] = np.cumsum(ecc[:, 1])
            range_ind = ecc[:, 0] < self.max_range
            ecc = ecc[range_ind, :]
            # add ecc max to the end - should we 0 or 1/n? but what if we have different n?
            ecc = np.vstack([ecc, [self.max_range, 0]])
            interpolator = spi.interp1d(ecc[:, 0], ecc[:, 1], kind="previous")
            representation = interpolator(self.xs)
            representations.append(representation)
            if self.norm == "l1":
                dist.append(np.trapz(np.abs(representation - self.representation), x=self.xs) * np.sqrt(sample.shape[0]))
            elif self.norm == "l2":
                dist.append(np.trapz((representation - self.representation) ** 2, x=self.xs) * np.sqrt(sample.shape[0]))
            else:  # sup
                dist.append(np.max(np.abs(representation - self.representation)) * np.sqrt(sample.shape[0]))

        return dist, representations
