import numpy as np
import scipy.stats as st


class GaussianMixture:
    def __init__(self, locations, scales, probas):
        # locations - vector of location parameters
        # scales - vector of scale parameters
        # probas - vector of mixture coefficients
        if not (len(locations) == len(scales) and len(scales) == len(probas)):
            raise ValueError("Wrong number of components for Gaussian Mixture")
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
        cdf = 0
        for p, rv in zip(self.probas, self.gauss_rv):
            cdf += p * rv.cdf(x)
        return cdf

    def pdf(self, x):
        pdf = 0
        for p, rv in zip(self.probas, self.gauss_rv):
            pdf += p * rv.pdf(x)
        return pdf

class GaussianMixture_nd:
    def __init__(self, locations, covs, probas, label=None):
        # locations - vector of location parameters
        # covs - vector of covarance matrices
        # probas - vector of mixture coefficients
        if not (len(locations) == len(covs) and len(covs) == len(probas)):
            raise ValueError("Wrong number of components for Gaussian Mixture")
        self.locations = locations
        self.covs = covs
        self.n_gauss = len(locations)
        self.gauss_rv = [st.multivariate_normal(mean=loc, cov=cov) for loc, cov in zip(locations, covs)]
        probas_sum = np.sum(probas)
        probas = [proba / probas_sum for proba in probas]
        self.probas = probas
        self.label = label

    # draw sample from GaussianMixture model
    def rvs(self, N):
        inds = st.rv_discrete(values=(range(self.n_gauss), self.probas)).rvs(size=N)
        X = [self.gauss_rv[ind].rvs(size=1) for ind in inds] # this is slow but good enough for now
        return np.array(X)

    def cdf(self, x):
        cdf = 0
        for p, rv in zip(self.probas, self.gauss_rv):
            cdf += p * rv.cdf(x)
        return cdf

    def pdf(self, x):
        pdf = 0
        for p, rv in zip(self.probas, self.gauss_rv):
            pdf += p * rv.pdf(x)
        return pdf

class AbsoluteDistribution:
    def __init__(self, rv):
        self.rv = rv

    def rvs(self, size):
        return np.abs(self.rv.rvs(size))

    def cdf(self, x):
        return 2 * self.rv.cdf(x) - 1

    def pdf(self, x):
        return 2 * self.rv.pdf(x)


class MultivariateDistribution:
    def __init__(self, univariates, label=None, shift_and_scale=False):
        self.univariates = univariates
        self.label = label
        self.dim = len(univariates)
        self.shift = shift_and_scale
        self.shift_vec = []
        self.scale_vec = []
        for uni in self.univariates:
            if self.shift:
                self.shift_vec.append(uni.stats(moments='m'))
                self.scale_vec.append(np.sqrt(uni.stats(moments='v')))
            else:
                self.shift_vec.append(0)
                self.scale_vec.append(1)
        self.shift_vec = np.array(self.shift_vec)
        self.scale_vec = np.array(self.scale_vec)

    def rvs(self, size):
        sample = []
        if self.shift:
            for univariate, shift_val, scale_val in zip(self.univariates, self.shift_vec, self.scale_vec):
                sample.append((univariate.rvs(size)-shift_val)/scale_val)
        else:
            for univariate in self.univariates:
                sample.append(univariate.rvs(size))
        return np.transpose(sample)

    def cdf(self, pts):
        # FIXME: this work only for multivariate distributions with diagonal covariance matrix
        # no correlations between axies are allowed
        if self.dim == 1:
            pts = pts * self.scale_vec[0]
            pts = pts + self.shift_vec
            pts = [pts]
        else:
            for i in range(self.dim):
                pts[i] = pts[i] * self.scale_vec[i]
            pts = pts + self.shift_vec
        cdf = 1
        for pt, univariate in zip(pts, self.univariates):
            cdf *= univariate.cdf(pt)
        return cdf

    # def pdf(self, pts):
    #     # FIXME: this work only for multivariate distributions with diagonal covariance matrix
    #     if self.dim == 1:
    #         pts = [pts]
    #     pdf = 1
    #     for pt, univariate in zip(pts, self.univariates):
    #         pdf *= univariate.pdf(pt)
    #     return pdf

class MultivariateDistributionJitter:
    def __init__(self, univariates, jitter=0.05, label=None):
        self.univariates = univariates
        self.label = label
        self.dim = len(univariates)
        self.noise = st.norm(loc=0, scale=jitter)

    def rvs(self, size):
        sample = []
        for univariate in self.univariates:
            sample.append(univariate.rvs(size) + self.noise.rvs(size))
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


class MultivariateGaussian:
    def __init__(self, dim, a, label=None):
        self.dim = dim
        self.label = label
        self.cov = np.ones((dim, dim)) * a + np.identity(dim) * (1 - a)
        self.mean = [0] * dim
        self.rv = st.multivariate_normal(mean=self.mean, cov=self.cov)

    def rvs(self, size):
        return self.rv.rvs(size)

    def cdf(self, pts):
        return self.rv.cdf(pts)

    def pdf(self, pts):
        return self.rv.pdf(pts)
