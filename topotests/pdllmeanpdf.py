import numpy as np
import gudhi as gd
from sklearn.mixture import GaussianMixture

def normal_density(x, loc, conv, convinv):
    det = np.linalg.det(conv)
    return np.exp((x-loc).dot(convinv).dot(np.transpose(x-loc))[0, :])/np.sqrt(det)


class pdllmeanpdf_representation:
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
        gm = self.gaussian_models[0]
        density = np.zeros(pd_sample.shape[0])
        for mean, cov, cov_inv in zip(gm.means_, gm.covariances_, gm.covariances_inv_):
            density += normal_density(pd_sample, mean,  cov, cov_inv)
        return np.mean(np.log(density))

    def fit(self, samples):
        self.pdpoints = self._compute_persistance_points(samples)
        self.gaussian_models = []
        # fit gaussians to each PD
        # TODO: fit also BayesianGaussianMixture
        n_componentss = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        for pdp in self.pdpoints:
            # bics = [GaussianMixture(n_components=n_components).fit(pdp).bic(pdp) if n_components < 0.5*pdp.shape[0] else np.Inf
            #         for n_components in n_componentss]
            bics = [GaussianMixture(n_components=n_components).fit(pdp).bic(pdp) for n_components in n_componentss]
            n_components = n_componentss[np.argmin(bics)]
            gm = GaussianMixture(n_components=n_components).fit(pdp)
            gm.covariances_inv_ = np.array([np.linalg.inv(cov) for cov in gm.covariances_])
            self.gaussian_models.append(gm)
        self.fitted = True

    def get_ll(self, sample):
        pd = self._compute_persistance_points([sample])[0]
        return -self._gaussian_mixture_mean_score(pd)
