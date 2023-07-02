import numpy as np
import gudhi.representations as gdr
import gudhi as gd
from mergegram import mergegram

def sample_standarize(sample):
    return (sample - np.mean(sample, axis=0)) / np.std(sample, axis=0)

def get_meregram(samples):
    mergegrams = [mergegram(sample) for sample in samples]
    return mergegrams
class PDMergegram_onesample:
    def __init__(
        self,
        n: int,
        dim: int,
        wasserstein_order: float = 1.0,
        wasserstein_p: float = 1.0,
        significance_level: float = 0.05,
        scaling=1,
        standarize=False,
        jobs = 1,
    ):
        """
        TODO: add comment
        """
        self.fitted = False
        self.sample_pts_n = n
        self.data_dim = int(dim)
        self.significance_level = significance_level
        self.wasserstein_order = wasserstein_order
        self.wasserstein_p = wasserstein_p
        self.standarize = standarize
        self.scaling = scaling
        self.representation = None
        self.wdmatrix = None
        self.wdmatrix_flat = None
        self.representation_threshold = None
        self.representation_distances = None
        self.mergegrams = None
        self.mergegrams_test = None
        self.jobs = jobs

    def fit(self, rv, n_signature, n_test):
        # generate signature samples and test sample
        samples = [rv.rvs(self.sample_pts_n) * self.scaling for i in range(n_signature)]
        samples_test = [rv.rvs(self.sample_pts_n) * self.scaling for i in range(n_test)]
        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]
            samples_test = [sample_standarize(sample) for sample in samples_test]

        # get signatures representations of both samples
        self.mergegrams = get_meregram(samples)
        self.mergegrams_test = get_meregram(samples_test)

        # get representation
        self.representation = gdr.WassersteinDistance(n_jobs=self.jobs,
                                                      order=self.wasserstein_order,
                                                      internal_p=self.wasserstein_p).fit(self.mergegrams)

        self.representation_distances = np.max(self.representation.transform(self.mergegrams_test), axis=1)
        self.representation_threshold = np.quantile(self.representation_distances, 1-self.significance_level)
        self.fitted = True

    def predict(self, samples):
        if not self.fitted:
            raise RuntimeError("Cannot run predict(). Run fit() first!")
        if len(samples) == 1:
            samples = [samples]

        samples = [sample * self.scaling for sample in samples]

        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]

        mergegrams = get_meregram(samples)
        distances = np.max(self.representation.transform(mergegrams), axis=1)

        accpect_h0 = [d < self.representation_threshold for d in distances]
        pvals = [np.mean(d < self.representation_distances) for d in distances]

        return accpect_h0, pvals
