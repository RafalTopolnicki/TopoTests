import numpy as np
#from pdwd import pdwd_representation
import gudhi.wasserstein as gdw
import gudhi as gd


def sample_standarize(sample):
    return (sample - np.mean(sample, axis=0)) / np.std(sample, axis=0)

def get_pds(samples, persistence_dim):
    pers_diagrams = []
    for sample_id, sample in enumerate(samples):
        if len(sample.shape) == 1:
            sample = sample.reshape(-1, 1)
        ac = gd.AlphaComplex(points=sample).create_simplex_tree()
        ac.compute_persistence()
        pers_diagram = ac.persistence_intervals_in_dimension(persistence_dim)
        pers_diagrams.append(pers_diagram)
    return pers_diagrams

class PDWDTest_onesample:
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
        self.representation = None
        self.wdmatrix = None
        self.representation_threshold = None
        self.representation_distances = None
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
        self.pds = get_pds(samples, persistence_dim=self.persistence_dim)
        self.pds_test = get_pds(samples_test, persistence_dim=self.persistence_dim)

        self.representation_distances = []
        for itest in range(n_test):
            wds = []
            for isig in range(n_signature):
                wds.append(gdw.wasserstein_distance(self.pds[isig], self.pds_test[itest]))
            self.representation_distances.append(np.max(wds))
        self.representation_threshold = np.quantile(self.representation_distances, 1 - self.significance_level)
        self.fitted = True

    def predict(self, samples):
        if not self.fitted:
            raise RuntimeError("Cannot run predict(). Run fit() first!")
        if len(samples) == 1:
            samples = [samples]

        samples = [sample * self.scaling for sample in samples]

        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]

        pds = get_pds(samples, persistence_dim=self.persistence_dim)
        accpect_h0 = []
        pvals = []

        for pd_pred in pds:
            wds = []
            for pd_train in self.pds:
                wds.append(gdw.wasserstein_distance(pd_train, pd_pred))
            wds_sup = np.max(wds)
            accpect_h0.append(wds_sup < self.representation_threshold)
            pvals.append(np.mean([wds_sup < rp for rp in self.representation_distances]))

        return accpect_h0, pvals
