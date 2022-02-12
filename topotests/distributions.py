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

    def cdf(self, x):
        cdf=0
        for p, rv in zip(self.probas, self.gauss_rv):
            cdf += p*rv.cdf(x)
        return cdf

    def pdf(self, x):
        pdf=0
        for p, rv in zip(self.probas, self.gauss_rv):
            pdf += p*rv.pdf(x)
        return pdf


class AbsoluteDistribution:
    def __init__(self, rv):
        self.rv = rv

    def rvs(self, size):
        return np.abs(self.rv.rvs(size))

    def cdf(self, x):
        return 2*self.rv.cdf(x) - 1

    def pdf(self, x):
        return 2*self.rv.pdf(x)



class MultivariateDistribution:
    def __init__(self, univariates, label=None):
        self.univariates = univariates
        self.label = label
        self.dim = len(univariates)

    def rvs(self, size):
        sample = []
        for univariate in self.univariates:
            sample.append(univariate.rvs(size))
        return np.transpose(sample)

    def cdf(self, pts):
        # FIXME: this work only for multivariate distributions with diagonal covariance matrix
        # no correlations between axies are allowed
        if self.dim == 1:
            pts = [pts]
        cdf = 1
        for pt, univariate in zip(pts, self.univariates):
            cdf *= univariate.cdf(pt)
        return cdf

    def pdf(self, pts):
        # FIXME: this work only for multivariate distributions with diagonal covariance matrix
        if self.dim == 1:
            pts = [pts]
        pdf = 1
        for pt, univariate in zip(pts, self.univariates):
            pdf *= univariate.pdf(pt)
        return pdf
