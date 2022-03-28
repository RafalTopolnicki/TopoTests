import numpy as np
import gudhi as gd
import scipy.interpolate as spi

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


class ecc_representation():
    def __init__(self, norm='sup', n_interpolation_points=100):
        self.representation = None
        self.max_range = -np.Inf
        self.xs = None
        self.n_interpolation_points = n_interpolation_points
        self.n_fitted = 0
        self.fitted = False
        self.norm = norm

    def fit(self, samples):
        self.max_range = -np.Inf
        eccs = []
        jumps = set()
        for count, sample  in enumerate(samples):
            ecc = np.array(compute_ECC_contributions_alpha(sample))
            ecc[:, 1] = np.cumsum(ecc[:, 1])
            jumps.update(ecc[:, 0])
            self.max_range = max(self.max_range, ecc[-1, 0])
            eccs.append(ecc)
        self.xs = np.sort(list(jumps))
        self.representation = self.xs * 0
        # extend all ecc so that it include the max_range
        # TODO: here we assume that the last value of ecc is always 1
        #       is that true?
        for count, ecc in enumerate(eccs):
            ecc_extended = np.vstack([ecc, [self.max_range, 1]])
            interpolator = spi.interp1d(ecc_extended[:, 0], ecc_extended[:, 1], kind='previous')
            self.representation += interpolator(self.xs)
        self.representation /= len(samples)
        self.fitted = True

    def transform(self, samples):
        if not self.fitted:
            raise RuntimeError('Run fit() before transform()')

        dist = []
        for count, sample in enumerate(samples):
            ecc = np.array(compute_ECC_contributions_alpha(sample))
            ecc[:, 1] = np.cumsum(ecc[:, 1])
            ecc = np.vstack([ecc, [self.max_range, 1]])
            interpolator = spi.interp1d(ecc[:, 0], ecc[:, 1], kind='previous')
            representation = interpolator(self.xs)
            if self.norm == 'l1':
                dist.append(np.trapz(np.abs(representation-self.representation), x=self.xs))
            elif self.norm == 'l2':
                dist.append(np.trapz((representation-self.representation)**2, x=self.xs))
            else:
                dist.append(np.max(np.abs(representation-self.representation)))
        return dist
