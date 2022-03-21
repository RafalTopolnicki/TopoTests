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


# # # function by Davide
# def ECC_from_contributions(local_contributions):
#     euler_characteristic = []
#     old_f, current_characteristic = local_contributions[0]
#
#     for filtration, contribution in local_contributions[1:]:
#         if filtration > old_f:
#             euler_characteristic.append([old_f, current_characteristic])
#             old_f = filtration
#
#         current_characteristic += contribution
#
#     # add last contribution
#     if len(local_contributions) > 1:
#         euler_characteristic.append([filtration, current_characteristic])
#
#     if len(local_contributions) == 1:
#         euler_characteristic.append(local_contributions[0])
#
#     return euler_characteristic


# taken form https://github.com/dgurnari/PURPLE/blob/main/alpha/bifiltrations_alpha.py
# DISTANCES
# scan the contribution list (after the merging) and removes the contributions
# that are 0
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


def get_ecc_distance_from_contributions(contr0, contr1):
    # merge the two contributions list, reversing the signs on the second one
    contributions = prune_contributions(contr0 + [(c[0], -1 * c[1]) for c in contr1])

    diff = 0
    current_ec = contributions[0][1]
    for i in range(1, len(contributions)):
        diff += np.abs(current_ec * (contributions[i][0] - contributions[i - 1][0]))
        current_ec += contributions[i][1]
    return diff

# # calaulate the distance between two ECC
# # in fact this is integrated squared difference between two picewise constant functions
# def get_ecc_distance(ecc0, ecc1):
#     # ecc0 and ecc1 are the picewise constant functions - each element is a (x, y) tuple
#     # and represents a jump of the function. segment is created using the x position of the next jump
#     ecc0 = np.array(ecc0)
#     ecc1 = np.array(ecc1)
#     # find the largest x value - this will determine the range of integration
#     xlim = max(np.max(ecc0[:, 0]), np.max(ecc1[:, 0]))
#     # add one segment to each function so that ecc0 and ecc1 have the same support
#     ecc0 = np.vstack([ecc0, [xlim, ecc0[-1, 1]]])
#     ecc1 = np.vstack([ecc1, [xlim, ecc1[-1, 1]]])
#     # ecc0 and ecc1 have jumps in different locations
#     # the difference ecc0-ecc1 is approximaed by step picewise constant function, hence we
#     # need to find all possible jumps of ecc0 and ecc1
#     bin_domain = np.unique(np.sort(np.concatenate([ecc0[:, 0], ecc1[:, 0]])))
#     ecc_diff = np.zeros_like(bin_domain)
#     # now we need to have y-value of ecc0 and ecc1 in the combined jumps locations
#     for id_b, b in enumerate(bin_domain):
#         id0 = np.argmax(ecc0[:, 0] > b)-1 # FIXME: there must be a faster way
#         id1 = np.argmax(ecc1[:, 0] > b)-1
#         ecc_diff[id_b] = ecc0[id0, 1] - ecc1[id1, 1] # differnece of ecc0 and ecc1
#     ecc_dist = 0
#     id_val = 0
#     for bl, br in zip(bin_domain[:-1], bin_domain[1:]):
#         ecc_dist += np.abs(ecc_diff[id_val])*(br-bl) #L1 distance
#         id_val += 1
#     return ecc_dist


class ECC_representation():

    def __init__(self):
        self.representation = None
        self.n_fitted = 0

    def fit(self, representation):
        self.representation = representation
        self.n_fitted = len(self.representation)

    def transform(self, representation):
        if not self.representation:
            raise RuntimeError('Run fit() before transform()')
        n_data = len(representation)

        dist = np.zeros((self.n_fitted, n_data))
        for id_rep, ecc_rep in enumerate(self.representation):
            for id_data, ecc_data in enumerate(representation):
                dist[id_rep, id_data] = get_ecc_distance_from_contributions(ecc_rep, ecc_data)
        return dist

class ECC_representation_mean():
    def __init__(self):
        self.representation = None
        self.x_max = 3
        self.xs = np.linspace(0, self.x_max, 100)
        self.n_fitted = 0

    def fit(self, samples):
        self.representation = self.xs * 0
        for sample in samples:
            ecc = compute_ECC_contributions_alpha(sample)
            ecc = np.array(ecc)
            ecc[:, 1] = np.cumsum(ecc[:, 1])
            ecc = np.vstack([ecc, [self.x_max, 1]])
            interpolator = spi.interp1d(ecc[:, 0], ecc[:, 1])
            self.representation += interpolator(self.xs)
        self.representation /= len(samples)

    def transform(self, samples):
        # if not self.representation:
        #     raise RuntimeError('Run fit() before transform()')

        representation = []
        for sample in samples:
            ecc = compute_ECC_contributions_alpha(sample)
            ecc = np.array(ecc)
            ecc[:, 1] = np.cumsum(ecc[:, 1])
            ecc = np.vstack([ecc, [self.x_max, 1]])
            interpolator = spi.interp1d(ecc[:, 0], ecc[:, 1])
            representation.append(interpolator(self.xs))

        dist = []
        for rep in representation:
            dist.append(np.mean(np.abs(rep-self.representation)))
        return dist
