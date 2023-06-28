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
        image_bandwidth=1e-2,
        norm='sup',
        aggregate='mean',
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
        self.norm = norm
        self.aggregate = aggregate
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
        #xmin, xmax = np.mean([np.quantile(pd[:, 0], [0.025, 0.95]) for pd in self.pds], axis=0)
        #ymin, ymax = np.mean([np.quantile(pd[:, 1], [0.025, 0.95]) for pd in self.pds], axis=0)
        xmin, xmax = np.mean([np.quantile(pd[:, 0], [0.025, 0.95]) for pd in self.pds], axis=0)
        ymin, ymax = np.mean([np.quantile(pd[:, 1], [0.025, 0.95]) for pd in self.pds], axis=0)

        self.image = gdr.PersistenceImage(resolution=(self.image_resolution, self.image_resolution),
                                          bandwidth=self.image_bandwidth, im_range=[xmin, xmax, ymin, ymax],
                                          ).fit(self.pds)
        self.representation = self.image.transform(self.pds)
        # compute mean image
        if self.aggregate == 'mean':
            self.representation = np.mean(self.representation, axis=0)

        representation_test = self.image.transform(self.pds_test)

        self.representation_distances = []
        for representation in representation_test:
            if self.aggregate == 'mean':
                if self.norm == 'sup':
                    self.representation_distances.append(np.max(np.abs(representation - self.representation)))
                elif self.norm == 'l1':
                    self.representation_distances.append(np.sum(np.abs(representation - self.representation)))
                else:
                    self.representation_distances.append(np.sum((representation - self.representation)**2))
            else: # aggregate none
                if self.norm == 'sup':
                    self.representation_distances.append(np.max([np.max(np.abs(representation - rep_train)) for rep_train in self.representation]))
                if self.norm == 'l1':
                    self.representation_distances.append(np.max([np.sum(np.abs(representation - rep_train)) for rep_train in self.representation]))
                if self.norm == 'l2':
                    self.representation_distances.append(np.max([np.sum((representation - rep_train)**2) for rep_train in self.representation]))

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
            if self.aggregate == 'mean':
                if self.norm == 'sup':
                    d = np.max(np.abs(representation - self.representation))
                elif self.norm == 'l1':
                    d = np.sum(np.abs(representation - self.representation))
                else:
                    d = np.sum((representation - self.representation)**2)
            else: # self.aggregate == 'none'
                if self.norm == 'sup':
                    d = np.max([np.max(np.abs(representation - rep_train)) for rep_train in self.representation])
                if self.norm == 'l1':
                    d = np.max([np.sum(np.abs(representation - rep_train)) for rep_train in self.representation])
                if self.norm == 'l2':
                    d = np.max([np.sum((representation - rep_train)**2) for rep_train in self.representation])

            accpect_h0.append(d<self.representation_threshold)
            pvals.append(np.mean(d > self.representation_distances))
        return accpect_h0, pvals
