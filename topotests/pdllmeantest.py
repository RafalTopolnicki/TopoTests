import numpy as np
from pdllmean import pdllmean_representation


def sample_standarize(sample):
    return (sample - np.mean(sample, axis=0)) / np.std(sample, axis=0)


class PDLLMeanTest_onesample:
    def __init__(
        self,
        n: int,
        dim: int,
        persistence_dim: int,
        significance_level: float = 0.05,
        scaling=1,
        standarize=False,
    ):
        """
        TODO: add comment
        """
        self.fitted = False
        self.sample_pts_n = n
        self.data_dim = int(dim)
        self.persistence_dim = persistence_dim
        self.significance_level = significance_level
        self.standarize = standarize
        self.scaling = scaling
        self.representation = pdllmean_representation(persistence_dim=self.persistence_dim)
        self.loglikelihoods = None
        self.representation_threshold = None
        if self.persistence_dim >= self.data_dim:
            raise ValueError(f'persistence_dim must be smaller than data_dim')

    def fit(self, rv, n_signature, n_test):
        # generate signature samples and test sample
        samples = [rv.rvs(self.sample_pts_n) * self.scaling for i in range(n_signature)]
        samples_test = [rv.rvs(self.sample_pts_n) * self.scaling for i in range(n_test)]
        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]
            samples_test = [sample_standarize(sample) for sample in samples_test]

        # get signatures representations of both samples
        self.representation.fit(samples)
        self.loglikelihoods = [self.representation.get_ll(sample=x) for x in samples_test]
        self.representation_threshold = np.quantile(self.loglikelihoods, 1 - self.significance_level)
        self.fitted = True

    def predict(self, samples):
        if not self.fitted:
            raise RuntimeError("Cannot run predict(). Run fit() first!")
        if len(samples) == 1:
            samples = [samples]

        samples = [sample * self.scaling for sample in samples]

        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]

        ll_predict = [self.representation.get_ll(sample=sample) for sample in samples]
        accpect_h0 = [ll < self.representation_threshold for ll in ll_predict]
        # calculate pvalues
        pvals = [np.mean(self.loglikelihoods > ll) for ll in ll_predict]
        return accpect_h0, pvals
