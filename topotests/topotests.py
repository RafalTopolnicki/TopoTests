from collections import namedtuple
from gudhi import representations

import gudhi as gd
import numpy as np
from mergegram import mergegram

distance_aggregators = namedtuple('distance_aggregators', 'min mean max quantile')


def sample_standarize(sample):
    return (sample - np.mean(sample, axis=0))/np.std(sample, axis=0)


class TopoTest:
    def __init__(self, n: int, dim: int, significance_level: float = 0.05,
                 method='mergegram', wasserstein_p=1, wasserstein_order=1,
                 standarize=False):
        """

        :param n:
        :param dim:
        :param significance_level:
        :param method:
        :param wasserstein_p:
        :param wasserstein_order:
        """

        if method not in ['mergegram', 'persistence']:
            raise ValueError(f'Incorrect method. Found method={method}. Possible options are '
                             f'"mergegram", "persistence"')
        self.fitted = False
        self.sample_pts_n = n
        self.sample_pt_dim = dim
        self.significance_level = significance_level
        self.method = method
        self.wasserstein_p = wasserstein_p
        self.wasserstein_order = wasserstein_order
        self.standarize = standarize
        self.wasserstein_representation = representations.WassersteinDistance(n_jobs=-1,
                                                                              order=self.wasserstein_order,
                                                                              internal_p=self.wasserstein_p)
        self.wasserstein_distance = None
        self.wasserstein_thresholds = None
        self.wasserstein_distance_predict = None


    def fit(self, rv, n_signature, n_test):
        # generate signature samples and test sample
        samples = [rv.rvs(self.sample_pts_n).reshape(-1, self.sample_pt_dim) for i in range(n_signature)]
        samples_test = [rv.rvs(self.sample_pts_n).reshape(-1, self.sample_pt_dim) for i in range(n_test)]
        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]
            samples_test = [sample_standarize(sample) for sample in samples_test]

        # get signatures representations of both samples
        signature_samples = [self.get_signature(sample) for sample in samples]
        signature_samples_test = [self.get_signature(sample) for sample in samples_test]
        # get WasserstainDistance representation
        self.wasserstein_representation.fit(signature_samples)
        self.wasserstein_distance = self.wasserstein_representation.transform(signature_samples_test)
        dmin, dmean, dmax, dq = self.aggregate_distances(self.wasserstein_distance)
        self.wasserstein_thresholds = distance_aggregators(min=np.quantile(dmin, 1-self.significance_level),
                                                           mean=np.quantile(dmean, 1-self.significance_level),
                                                           max=np.quantile(dmax, 1-self.significance_level),
                                                           quantile=np.quantile(dq, 1-self.significance_level))
        self.fitted = True

    def aggregate_distances(self, distance_matrix):
        dmean = np.mean(distance_matrix, axis=1)
        dmin = np.min(distance_matrix, axis=1)
        dmax = np.max(distance_matrix, axis=1)
        dq = np.quantile(distance_matrix, q=0.9, axis=1)
        return dmin, dmean, dmax, dq

    def predict(self, samples):
        if not self.fitted:
            raise RuntimeError('Cannot run predict(). Run fit() first!')
        if len(samples) == 1:
            samples = [samples]

        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]

        reprs = [self.get_signature(sample) for sample in samples]
        self.wasserstein_distance_predict = self.wasserstein_representation.transform(reprs)
        dmin, dmean, dmax, dq = self.aggregate_distances(self.wasserstein_distance_predict)
        return distance_aggregators(min=dmin < self.wasserstein_thresholds.min,
                                    mean=dmean < self.wasserstein_thresholds.mean,
                                    max=dmax < self.wasserstein_thresholds.max,
                                    quantile=dq < self.wasserstein_thresholds.quantile)

    def get_signature(self, sample):
        if self.method == 'mergegram':
            return mergegram(sample)

        if self.method == 'persistence':
            ac = gd.AlphaComplex(points=sample)
            st = ac.create_simplex_tree()
            st.compute_persistence()
            return st.persistence_intervals_in_dimension(0)

    def save_distance_matrix(self, filename):
        np.save(filename, self.wasserstein_distance)

    def save_predict_distance_matrix(self, filename):
        np.save(filename, self.wasserstein_distance_predict)

    def save_model(self):
        pass

    def load_model(self):
        pass