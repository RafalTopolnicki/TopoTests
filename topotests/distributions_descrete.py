import numpy as np
import scipy.stats as st


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
