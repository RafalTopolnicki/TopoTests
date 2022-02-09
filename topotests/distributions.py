import numpy as np
import scipy.stats as st


class GaussianMixture:
    def __init__(self, locations, scales, probas):
        # locations - vector of location parameters
        # scales - vector of scale parameters
        # probas - vector of mixture coefficients
        if not (len(locations) == len(scales) and len(scales) == len(probas)):
            raise ValueError('Wrong number of components for Gaussian Mixture')
        self.locations = locations
        self.scales = scales
        self.n_gauss = len(locations)
        self.gauss_rv = [st.norm(loc, scale) for loc, scale in zip(locations, scales)]
        probas_sum = np.sum(probas)
        probas = [proba / probas_sum for proba in probas]
        self.probas = probas

    # draw sample from GaussianMixture model
    def rvs(self, N):
        inds = st.rv_discrete(values=(range(self.n_gauss), self.probas)).rvs(size=N)
        X = [self.gauss_rv[ind].rvs(size=1)[0] for ind in inds]
        return X


class AbsoluteDistribution:
    def __init__(self, rv):
        self.rv = rv

    def rvs(self, size):
        return np.abs(self.rv.rvs(size))


class MultivariateDistribution:
    def __init__(self, univariates, label=None):
        self.univariates = univariates
        self.label = label

    def rvs(self, size):
        sample = []
        for univariate in self.univariates:
            sample.append(univariate.rvs(size))
        return np.transpose(sample)