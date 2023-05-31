import numpy as np
import gudhi as gd
from sklearn.mixture import GaussianMixture


class pdllmean_representation:
    def __init__(self, persistence_dim):
        self.pdpoints = None
        self.n_fitted = 0
        self.fitted = False
        self.persistence_dim = persistence_dim
        self.gaussian_mixture = None
        self.gaussian_mixture_n_components = None
        self.gaussian_models = []

    def _compute_persistance_points(self, samples):
        all_pers_points = []
        for sample_id, sample in enumerate(samples):
            if len(sample.shape) == 1:
                sample = sample.reshape(-1, 1)
            ac = gd.AlphaComplex(points=sample).create_simplex_tree()
            ac.compute_persistence()
            pers_points = ac.persistence_intervals_in_dimension(self.persistence_dim)
            # remove infs
            pers_points = [pp for pp in pers_points if np.isfinite(pp[1])]
            pers_points = np.array(pers_points)
            all_pers_points.append(pers_points)
        return all_pers_points

    def _gaussian_mixture_mean_score(self, pd_sample):
        scores = [gm.score(pd_sample) for gm in self.gaussian_models]
        return np.mean(scores)

    def fit(self, samples):
        self.pdpoints = self._compute_persistance_points(samples)
        self.gaussian_models = []
        # fit gaussians to each PD
        # TODO: fit also BayesianGaussianMixture
        n_componentss = [1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 20]
        for pdp in self.pdpoints:
            # bics = [GaussianMixture(n_components=n_components).fit(pdp).bic(pdp) if n_components < 0.5*pdp.shape[0] else np.Inf
            #         for n_components in n_componentss]
            bics = [GaussianMixture(n_components=n_components).fit(pdp).bic(pdp) for n_components in n_componentss]
            n_components = n_componentss[np.argmin(bics)]
            self.gaussian_models.append(GaussianMixture(n_components=n_components).fit(pdp))
        self.fitted = True

    def get_ll(self, sample):
        pd = self._compute_persistance_points([sample])[0]
        return -self._gaussian_mixture_mean_score(pd)

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
