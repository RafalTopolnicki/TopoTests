import numpy as np
import gudhi.wasserstein as gdw
import gudhi.representations as gdr
import gudhi as gd
from scipy.stats import ks_2samp, cramervonmises_2samp, anderson_ksamp

def sample_standarize(sample):
    return (sample - np.mean(sample, axis=0)) / np.std(sample, axis=0)

def get_pds(samples, persistence_dim, log=False):
    log_offest = 1e-2
    pers_diagrams = []
    for sample_id, sample in enumerate(samples):
        if len(sample.shape) == 1:
            sample = sample.reshape(-1, 1)
        ac = gd.AlphaComplex(points=sample).create_simplex_tree()
        ac.compute_persistence()
        pers_diagram = ac.persistence_intervals_in_dimension(persistence_dim)
        if log:
            pers_diagrams.append(np.log(pers_diagram+log_offest))
        else:
            pers_diagrams.append(pers_diagram)

    return pers_diagrams

class PDImageTest_onesample:
    def __init__(
        self,
        n: int,
        dim: int,
        persistence_dim: int,
        order: float = 1.0,
        significance_level: float = 0.05,
        scaling=1,
        standarize=False,
        log=False,
        image_bandwidth=1e-2
    ):
        """
        TODO: add comment
        """
        self.fitted = False
        self.sample_pts_n = n
        self.data_dim = int(dim)
        self.persistence_dim = persistence_dim
        self.significance_level = significance_level
        self.order = order
        self.standarize = standarize
        self.scaling = scaling
        self.representation = None
        self.representation_threshold = None
        self.representation_distances = None
        self.landscape = None
        self.image_resolution = 400
        self.image_bandwidth = image_bandwidth
        self.log = log
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
        self.pds = get_pds(samples, persistence_dim=self.persistence_dim, log=self.log)
        self.pds_test = get_pds(samples_test, persistence_dim=self.persistence_dim, log=self.log)

        # fit landscape
        xmin, xmax = np.mean([np.quantile(pd[:, 0], [0.025, 0.95]) for pd in self.pds], axis=0)
        ymin, ymax = np.mean([np.quantile(pd[:, 1], [0.025, 0.95]) for pd in self.pds], axis=0)

        self.image = gdr.PersistenceImage(resolution=(self.image_resolution, self.image_resolution),
                                          bandwidth=self.image_bandwidth, im_range=[xmin, xmax, ymin, ymax]).fit(self.pds)
        self.representation = self.image.transform(self.pds)
        self.representation = np.mean(self.representation, axis=0)

        representation_test = self.image.transform(self.pds_test)

        self.representation_distances = []
        for representation in representation_test:
            self.representation_distances.append(np.max(np.abs(representation - self.representation))) # L1

        self.representation_threshold = np.quantile(self.representation_distances, 1-self.significance_level)
        self.fitted = True

    def predict(self, samples):
        if not self.fitted:
            raise RuntimeError("Cannot run predict(). Run fit() first!")
        if len(samples) == 1:
            samples = [samples]

        samples = [sample * self.scaling for sample in samples]

        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]

        pds = get_pds(samples, persistence_dim=self.persistence_dim, log=self.log)
        representations = self.image.transform(pds)

        accpect_h0 = []
        pvals = []

        for representation in representations:
            d = np.max(np.abs(representation - self.representation))
            accpect_h0.append(d<self.representation_threshold)
            pvals.append(np.mean(d > self.representation_distances))
        return accpect_h0, pvals
