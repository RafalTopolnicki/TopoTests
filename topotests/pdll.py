import numpy as np
import gudhi as gd
from sklearn.mixture import GaussianMixture

def normal_density(x, loc, sigma):
    return np.exp(-(x-loc)**2/(2*sigma))

class pdll_representation:
    def __init__(self, persistence_dim):
        self.pdpoints = None
        self.n_fitted = 0
        self.fitted = False
        self.persistence_dim = persistence_dim
        self.gaussian_mixture = None
        self.gaussian_mixture_n_components = None

    def _compute_persistance_points(self, samples):
        all_pers_points = None
        for sample_id, sample in enumerate(samples):
            if len(sample.shape) == 1:
                sample = sample.reshape(-1, 1)
            ac = gd.AlphaComplex(points=sample).create_simplex_tree()
            ac.compute_persistence()
            pers_points = ac.persistence_intervals_in_dimension(self.persistence_dim)
            # remove infs
            pers_points = [pp for pp in pers_points if np.isfinite(pp[1])]
            if sample_id == 0:
                all_pers_points = pers_points
            else:
                all_pers_points = np.vstack((all_pers_points, pers_points))
        return all_pers_points

    def fit(self, samples):
        self.pdpoints = self._compute_persistance_points(samples)
        # find best number of components based on BIC criterion
        n_componentss = [1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 20, 50, 100]
        bics = [GaussianMixture(n_components=n_components).fit(self.pdpoints).bic(self.pdpoints)
                for n_components in n_componentss]
        self.gaussian_mixture_n_components = n_componentss[np.argmin(bics)]
        self.gaussian_mixture = GaussianMixture(self.gaussian_mixture_n_components).fit(self.pdpoints)
        self.fitted = True

    def get_ll(self, sample):
        pd = self._compute_persistance_points([sample])
        return -np.sum(self.gaussian_mixture.score_samples(pd))

    # def transform(self, samples):
    #     pdpoints = self._compute_persistance_points(samples)
    #     return pdpoints

    # def get_ll(self, sample):
    #     pdpoints = self.transform([sample])
    #     loglikelihood = 0
    #     for pdpoint in pdpoints:
    #         pdpoints_density = normal_density(x=pdpoint, loc=self.pdpoints, sigma=self.sigma)
    #         # since MVN has diagonal covarinace matrix we can total density is just a product of densities in each axis
    #         loglikelihood += -np.log(np.sum(np.prod(pdpoints_density, axis=1)))
    #     return loglikelihood
    #
