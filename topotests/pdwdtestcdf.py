import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import gudhi.wasserstein as gdw
import gudhi as gd
from scipy.stats import ks_2samp, cramervonmises_2samp, anderson_ksamp, ks_1samp

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

class cdfinterpolation:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def cdf(self, x):
        return np.interp(x, self.x, self.y)
class PDWDCDFTest_onesample:
    def __init__(
        self,
        n: int,
        dim: int,
        persistence_dim: int,
        order: float = 1.0,
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
        self.order = order
        self.standarize = standarize
        self.scaling = scaling
        self.representation = None
        self.wdmatrix = None
        self.wdmatrix_flat = None
        self.representation_threshold = None
        self.representation_distances = None
        self.pds = None
        self.pds_test = None
        self.pds_threshold = None
        self.ks_stats = None
        if self.persistence_dim >= self.data_dim:
            raise ValueError(f'persistence_dim must be smaller than data_dim')

    def fit(self, rv, n_signature, n_test):
        # generate signature samples and test sample
        samples = [rv.rvs(self.sample_pts_n) * self.scaling for i in range(n_signature)]
        samples_test = [rv.rvs(self.sample_pts_n) * self.scaling for i in range(n_test)]
        samples_threshold = [rv.rvs(self.sample_pts_n) * self.scaling for i in range(n_test)]
        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]
            samples_test = [sample_standarize(sample) for sample in samples_test]
            samples_threshold = [sample_standarize(sample) for sample in samples_threshold]

        # get signatures representations of both samples
        self.pds = get_pds(samples, persistence_dim=self.persistence_dim)
        self.pds_test = get_pds(samples_test, persistence_dim=self.persistence_dim)
        self.pds_threshold = get_pds(samples_threshold, persistence_dim=self.persistence_dim)

        self.representation_distances = []
        self.wdmatrix = np.zeros((n_test, n_signature))
        for isig in range(n_signature):
            for itest in range(n_test):
                self.wdmatrix[itest, isig] = gdw.wasserstein_distance(self.pds[isig], self.pds_test[itest],
                                                                      order=self.order)
        # compute average distance CDF
        self.wd_max = np.max(self.wdmatrix)
        self.wd_cdf_x = np.linspace(0, self.wd_max, 1000)
        self.wd_cdf_y = np.array([0.0]*1000)
        # iterate over rows and interpolate ECDF onto wd_cdf_x
        for row in self.wdmatrix:
            self.wd_cdf_y += ECDF(row)(self.wd_cdf_x)
        self.wd_cdf_y /= n_signature
        self.cdf = cdfinterpolation(self.wd_cdf_x, self.wd_cdf_y)

        self.ks_stats = []
        for pd_thres in self.pds_threshold:
            wds = []
            for pd_train in self.pds:
                wds.append(gdw.wasserstein_distance(pd_train, pd_thres, order=self.order))
            res = ks_1samp(wds, self.cdf.cdf)
            self.ks_stats.append(res.statistic)
        self.representation_threshold = np.quantile(self.ks_stats, 1-self.significance_level)
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
                wds.append(gdw.wasserstein_distance(pd_train, pd_pred, order=self.order))
            stat = ks_1samp(wds, self.cdf.cdf).statistic
            accpect_h0.append(stat <= self.representation_threshold)
            pvals.append(np.mean(stat <= self.ks_stats))
        return accpect_h0, pvals
