import numpy as np
import gudhi as gd

def normal_density(x, loc, sigma):
    return np.exp(-(x-loc)**2/(2*sigma))

class pdll_representation:
    def __init__(self, persistence_dim, sigma):
        self.pdpoints = None
        self.n_fitted = 0
        self.fitted = False
        self.persistence_dim = persistence_dim
        self.sigma = sigma

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
        self.fitted = True
    def transform(self, samples):
        pdpoints = self._compute_persistance_points(samples)
        return pdpoints

    def get_ll(self, sample):
        pdpoints = self.transform([sample])
        loglikelihood = 0
        for pdpoint in pdpoints:
            pdpoints_density = normal_density(x=pdpoint, loc=self.pdpoints, sigma=self.sigma)
            # since MVN has diagonal covarinace matrix we can total density is just a product of densities in each axis
            loglikelihood += -np.log(np.sum(np.prod(pdpoints_density, axis=1)))
        return loglikelihood

