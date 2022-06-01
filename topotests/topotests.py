from collections import namedtuple
from gudhi import representations

from mergegram import mergegram
from ecc import *
import pickle

distance_aggregators = namedtuple("distance_aggregators", "min mean max quantile")


def sample_standarize(sample):
    return (sample - np.mean(sample, axis=0)) / np.std(sample, axis=0)


class TopoTest:
    def __init__(
        self,
        n: int,
        dim: int,
        significance_level: float = 0.05,
        method="mergegram",
        wasserstein_p=1,
        wasserstein_order=1,
        ecc_norm="sup",
        scaling=1,
        standarize=False,
    ):
        """

        :param n:
        :param dim:
        :param significance_level:
        :param method:
        :param wasserstein_p:
        :param wasserstein_order:
        """

        if method not in ["mergegram", "persistence", "ecc", "ecc-exact"]:
            raise ValueError(
                f"Incorrect method. Found method={method}. Possible options are "
                f'"mergegram", "persistence", "ecc", "ecc-exact"'
            )
        self.fitted = False
        self.sample_pts_n = n
        self.sample_pt_dim = dim
        self.significance_level = significance_level
        self.method = method
        self.wasserstein_p = wasserstein_p
        self.wasserstein_order = wasserstein_order
        self.ecc_norm = ecc_norm
        self.standarize = standarize
        self.scaling = scaling
        if method in ["mergegram", "persistence"]:
            self.representation = representations.WassasersteinDistance(
                n_jobs=-1, order=self.wasserstein_order, internal_p=self.wasserstein_p
            )
        if method == "ecc":
            self.representation = ecc_representation(self.ecc_norm)
        if method == "ecc-exact":
            self.representation = ecc_representation(self.ecc_norm, mode="exact")

        self.n_points_to_save = 1000
        self.representation_distance = None
        self.representation_threshold = None
        self.representation_distance_predict = None
        self.representation_signature = None
        self.representation_predict = {}

    def fit(self, rv, n_signature, n_test):
        # generate signature samples and test sample
        samples = [rv.rvs(self.sample_pts_n).reshape(-1, self.sample_pt_dim) * self.scaling for i in range(n_signature)]
        samples_test = [rv.rvs(self.sample_pts_n).reshape(-1, self.sample_pt_dim) * self.scaling for i in range(n_test)]
        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]
            samples_test = [sample_standarize(sample) for sample in samples_test]

        # get signatures representations of both samples

        signature_samples = [self.get_signature(sample) for sample in samples]
        signature_samples_test = [self.get_signature(sample) for sample in samples_test]
        self.representation.fit(signature_samples)
        (
            self.representation_distance,
            self.representation_signature,
        ) = self.representation.transform(signature_samples_test)
        representation_test_skip = int(len(self.representation_signature[0]) / self.n_points_to_save)
        self.representation_signature = self.representation_signature[::representation_test_skip]

        if self.method not in ["ecc", "ecc-exact"]:
            dmin, dmean, dmax, dq = self.aggregate_distances(self.representation_distance)
            self.representation_threshold = {
                "min": np.quantile(dmin, 1 - self.significance_level),
                "mean": np.quantile(dmean, 1 - self.significance_level),
                "max": np.quantile(dmax, 1 - self.significance_level),
                "quantile": np.quantile(dq, 1 - self.significance_level),
            }
        else:
            self.representation_threshold = {
                "mean": np.quantile(self.representation_distance, 1 - self.significance_level)
            }
        self.fitted = True

    def aggregate_distances(self, distance_matrix):
        dmean = np.mean(distance_matrix, axis=1)
        dmin = np.min(distance_matrix, axis=1)
        dmax = np.max(distance_matrix, axis=1)
        dq = np.quantile(distance_matrix, q=0.9, axis=1)
        return dmin, dmean, dmax, dq

    def predict(self, samples, label):
        if not self.fitted:
            raise RuntimeError("Cannot run predict(). Run fit() first!")
        if len(samples) == 1:
            samples = [samples]

        samples = [sample * self.scaling for sample in samples]

        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]

        signatures = [self.get_signature(sample) for sample in samples]
        (
            self.representation_distance_predict,
            representation_test,
        ) = self.representation.transform(signatures)
        representation_test_skip = int(len(representation_test[0]) / self.n_points_to_save)
        self.representation_predict[label] = [rep[::representation_test_skip] for rep in representation_test]

        if self.method not in ["ecc", "ecc-exact"]:
            dmin, dmean, dmax, dq = self.aggregate_distances(self.representation_distance_predict)
            return {
                "min": dmin < self.representation_threshold["min"],
                "mean": dmean < self.representation_threshold["mean"],
                "max": dmax < self.representation_threshold["max"],
                "quantile": dq < self.representation_threshold["quantile"],
            }
        # method=ecc
        res = {"mean": [dp < self.representation_threshold["mean"] for dp in self.representation_distance_predict]}
        return res

    def get_signature(self, sample):
        if self.method == "mergegram":
            return mergegram(sample)

        if self.method == "persistence":
            ac = gd.AlphaComplex(points=sample)
            st = ac.create_simplex_tree()
            st.compute_persistencerep()
            return st.persistence_intervals_in_dimension(0)

        # in all other cases, i.e. for ecc method simply return sample
        return sample

    def save_distance_matrix(self, filename):
        np.save(filename, self.representation_distance)

    def save_predict_distance_matrix(self, filename):
        np.save(filename, self.representation_distance_predict)

    def save_representation(self, filename):
        representation_test_skip = int(len(self.representation.xs) / self.n_points_to_save)
        with open(filename, "wb") as fp:
            pickle.dump(
                [
                    self.representation.xs,
                    self.representation.representation,
                    self.representation.representation2,
                    self.representation.std,
                    self.representation_distance_predict,  # sup values
                    self.representation_signature,  # what is that?
                    self.representation.xs[::representation_test_skip],
                    self.representation_predict,
                ],
                fp,
            )

    def save_model(self):
        pass

    def load_model(self):
        pass
