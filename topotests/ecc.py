import numpy as np
import gudhi as gd
import scipy.interpolate as spi
import random

# computes each simplex contribution to the ECC
# function by Davide
def compute_ECC_contributions_alpha(point_cloud):
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

    return sorted(list(ecc.items()), key=lambda x: x[0])


def prune_contributions(contributions):
    contr_dict = dict()

    for c in contributions:
        contr_dict[c[0]] = contr_dict.get(c[0], 0) + c[1]

    # remove the contributions that are 0
    to_del = []
    for key in contr_dict:
        if contr_dict[key] == 0:
            to_del.append(key)
    for key in to_del:
        del contr_dict[key]

    return sorted(list(contr_dict.items()), key=lambda x: x[0])


class ecc_representation:
    def __init__(self, norm="sup", n_interpolation_points=100, mode="approximate"):
        self.representation = None
        self.representation2 = (
            None  # this can be changed later on to representation2 (no self.)
        )
        self.xs = None
        self.std = None
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
        eccs = []
        jumps = set()
        if self.mode == "exact":
            for sample in samples:
                ecc = np.array(compute_ECC_contributions_alpha(sample))
                ecc[:, 1] = np.cumsum(ecc[:, 1])
                jumps.update(ecc[:, 0])  # FIXME: ecc[:, 0] is stored in eccs anyway
                self.max_range = max(self.max_range, ecc[-1, 0])
                eccs.append(ecc)
            self.xs = np.sort(list(jumps))
            self.representation = self.xs * 0
            self.representation2 = self.xs * 0
            # extend all ecc so that it include the max_range
            for ecc in eccs:
                ecc_extended = np.vstack([ecc, [self.max_range, 1]])
                interpolator = spi.interp1d(
                    ecc_extended[:, 0], ecc_extended[:, 1], kind="previous"
                )
                y_inter = interpolator(self.xs)
                self.representation += y_inter
                self.representation2 += y_inter * y_inter
            self.representation /= len(samples)
            self.representation2 /= len(samples)
            self.std = np.sqrt(
                self.representation2 - self.representation * self.representation
            )
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
            self.xs = jumps[::jumps_step]
            self.representation = self.xs * 0
            self.representation2 = self.xs * 0
            # interpolate ECC curves on the grid
            for sample in samples:
                ecc = np.array(compute_ECC_contributions_alpha(sample))
                ecc[:, 1] = np.cumsum(ecc[:, 1])
                # cut ecc on self.max_range
                range_ind = ecc[:, 0] < self.max_range
                ecc = ecc[range_ind, :]
                # add ecc max to the end
                ecc_extended = np.vstack([ecc, [self.max_range, 1]])
                interpolator = spi.interp1d(
                    ecc_extended[:, 0], ecc_extended[:, 1], kind="previous"
                )
                y_inter = interpolator(self.xs)
                self.representation += y_inter
                self.representation2 += y_inter * y_inter
            self.representation /= len(samples)
            self.representation2 /= len(samples)
            self.std = np.sqrt(
                self.representation2 - self.representation * self.representation
            )
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
            ecc = np.vstack([ecc, [self.max_range, 1]])
            interpolator = spi.interp1d(ecc[:, 0], ecc[:, 1], kind="previous")
            representation = interpolator(self.xs)
            representations.append(representation)
            if self.norm == "l1":
                dist.append(
                    np.trapz(np.abs(representation - self.representation), x=self.xs)
                )
            elif self.norm == "l2":
                dist.append(
                    np.trapz((representation - self.representation) ** 2, x=self.xs)
                )
            else:  # sup
                dist.append(np.max(np.abs(representation - self.representation)))

        # if self.mode == 'exact':
        #     for sample in samples:
        #         ecc = np.array(compute_ECC_contributions_alpha(sample))
        #         ecc[:, 1] = np.cumsum(ecc[:, 1])
        #         ecc = np.vstack([ecc, [self.max_range, 1]]) # FIXME: ecc[:, 0] can be > self.max_range
        #         interpolator = spi.interp1d(ecc[:, 0], ecc[:, 1], kind='previous')
        #         representation = interpolator(self.xs)
        #         representations.append(representation)
        #         if self.norm == 'l1':
        #             dist.append(np.trapz(np.abs(representation-self.representation), x=self.xs))
        #         elif self.norm == 'l2':
        #             dist.append(np.trapz((representation-self.representation)**2, x=self.xs))
        #         else: # sup
        #             dist.append(np.max(np.abs(representation-self.representation)))
        # else: #this is almost the same code as in the first if
        #     for sample in samples:
        #         ecc = np.array(compute_ECC_contributions_alpha(sample))
        #         ecc[:, 1] = np.cumsum(ecc[:, 1])
        #         range_ind = ecc[:, 0] < self.max_range
        #         ecc = ecc[range_ind, :]
        #         ecc = np.vstack([ecc, [self.max_range, 1]])
        #         interpolator = spi.interp1d(ecc[:, 0], ecc[:, 1], kind='previous')
        #         representation = interpolator(self.xs)
        #         representations.append(representation)
        #         if self.norm == 'l1':
        #             dist.append(np.trapz(np.abs(representation-self.representation), x=self.xs))
        #         elif self.norm == 'l2':
        #             dist.append(np.trapz((representation-self.representation)**2, x=self.xs))
        #         else: # sup
        #             dist.append(np.max(np.abs(representation-self.representation)))
        return dist, representations
