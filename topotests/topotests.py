from collections import namedtuple
from gudhi import representations

from ecc import *


def sample_standarize(sample):
    return (sample - np.mean(sample, axis=0)) / np.std(sample, axis=0)


class TopoTest_onesample:
    def __init__(
        self,
        n: int,
        dim: int,
        significance_level: float = 0.05,
        norm="sup",
        method="approximate",
        scaling=1,
        standarize=False,
    ):
        """
        TODO: add comment
        """
        self.fitted = False
        self.sample_pts_n = n
        self.sample_pt_dim = dim
        self.significance_level = significance_level
        self.method = method
        self.norm = norm
        self.standarize = standarize
        self.scaling = scaling
        if method not in ["approximate", "exact"]:
            raise ValueError(f"method must be approximate or exact, got {method} instead")
        if method == "approximate":
            self.representation = ecc_representation(self.norm, mode="approximate")
        if method == "exact":
            self.representation = ecc_representation(self.norm, mode="exact")

        self.n_points_to_save = 5000
        self.representation_distance = None
        self.representation_threshold = None
        self.representation_distance_predict = None
        self.representation_signature = None

    def fit(self, rv, n_signature, n_test):
        # generate signature samples and test sample
        samples = [rv.rvs(self.sample_pts_n) * self.scaling for i in range(n_signature)]
        samples_test = [rv.rvs(self.sample_pts_n) * self.scaling for i in range(n_test)]
        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]
            samples_test = [sample_standarize(sample) for sample in samples_test]

        # get signatures representations of both samples
        self.representation.fit(samples)
        (
            self.representation_distance,
            self.representation_signature,
        ) = self.representation.transform(samples_test)
        self.representation_threshold = np.quantile(self.representation_distance, 1 - self.significance_level)
        self.fitted = True

    def predict(self, samples):
        if not self.fitted:
            raise RuntimeError("Cannot run predict(). Run fit() first!")
        if len(samples) == 1:
            samples = [samples]

        samples = [sample * self.scaling for sample in samples]

        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]

        self.representation_distance_predict, _ = self.representation.transform(samples)
        accpect_h0 = [dp < self.representation_threshold for dp in self.representation_distance_predict]
        # calculate pvalues
        pvals = [np.mean(self.representation_distance > dp) for dp in self.representation_distance_predict]
        return accpect_h0, pvals

    def save_distance_matrix(self, filename):
        np.save(filename, self.representation_distance)

    def save_predict_distance_matrix(self, filename):
        np.save(filename, self.representation_distance_predict)

    def save_model(self):
        pass

    def load_model(self):
        pass
