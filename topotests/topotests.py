from collections import namedtuple
from gudhi import representations

from ecc import *


def sample_standarize(sample):
    return (sample - np.mean(sample, axis=0)) / np.std(sample, axis=0)


class TopoTest_onesample:
    def __init__(
        self,
        n: int,
        dim: int,
        significance_level: float = 0.05,
        norm="sup",
        method="approximate",
        scaling=1,
        standarize=False,
    ):
        """
        TODO: add comment
        """
        self.fitted = False
        self.sample_pts_n = n
        self.sample_pt_dim = dim
        self.significance_level = significance_level
        self.method = method
        self.norm = norm
        self.standarize = standarize
        self.scaling = scaling
        if method not in ["approximate", "exact"]:
            raise ValueError(f"method must be approximate or exact, got {method} instead")
        if method == "approximate":
            self.representation = ecc_representation(self.norm, mode="approximate")
        if method == "exact":
            self.representation = ecc_representation(self.norm, mode="exact")

        self.n_points_to_save = 5000
        self.representation_distance = None
        self.representation_threshold = None
        self.representation_distance_predict = None
        self.representation_signature = None

    def fit(self, rv, n_signature, n_test):
        # generate signature samples and test sample
        samples = [rv.rvs(self.sample_pts_n) * self.scaling for i in range(n_signature)]
        samples_test = [rv.rvs(self.sample_pts_n) * self.scaling for i in range(n_test)]
        if self.sample_pt_dim == 1:
            samples = [np.expand_dims(sample, 1) for sample in samples]
            samples_test = [np.expand_dims(sample, 1) for sample in samples_test]		       
        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]
            samples_test = [sample_standarize(sample) for sample in samples_test]

        # get signatures representations of both samples
        self.representation.fit(samples)
        (
            self.representation_distance,
            self.representation_signature,
        ) = self.representation.transform(samples_test)
        self.representation_threshold = np.quantile(self.representation_distance, 1 - self.significance_level)
        self.fitted = True

    def scale_threshold_with_samplesize(self, n):
        return self.representation_threshold / np.sqrt(n) * np.sqrt(self.n)
    def predict(self, samples):
        if not self.fitted:
            raise RuntimeError("Cannot run predict(). Run fit() first!")
        if len(samples) == 1:
            samples = [samples]

        samples = [sample * self.scaling for sample in samples]

        if self.standarize:
            samples = [sample_standarize(sample) for sample in samples]
        self.representation_distance_predict, _ = self.representation.transform(samples)
        accepect_h0 = [dp < self.representation_threshold for dp in self.representation_distance_predict]
        # calculate pvalues
        pvals = [np.mean(self.representation_distance > dp) for dp in self.representation_distance_predict]
        return accepect_h0, pvals

    def save_distance_matrix(self, filename):
        np.save(filename, self.representation_distance)

    def save_predict_distance_matrix(self, filename):
        np.save(filename, self.representation_distance_predict)


def TopoTest_twosample(X1, X2, norm="sup", loops=100):
    n_grids = 2000
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    def _get_ecc(X, epsmax=None):
        n = X.shape[0]
        dim = X.shape[1]
        ecc = np.array(compute_ECC_contributions_alpha(X))
        ecc[:, 1] = np.cumsum(ecc[:, 1])
        ecc[:, 1] = ecc[:, 1]/n
        #ecc[:, 0] = ecc[:, 0]*n # THIS WORKS ONLY in dim=2
        ecc[:, 0] = ecc[:, 0]*n**(2/dim)
        if epsmax is not None:
            ecc = np.vstack([ecc, [epsmax, 1]])
        return ecc

    def _dist_ecc(ecc1, ecc2):
        # ecc1 and ecc2 are of equal length and have jumps in the same location
        if norm == "sup":
            return np.max(np.abs(ecc1[:, 1] - ecc2[:, 1]))
        if norm == "l1":
            return np.trapz(np.abs(ecc1[:, 1] - ecc2[:, 1]), x=ecc1[:, 0])
        if norm == "l2":
            return np.trapz((ecc1[:, 1] - ecc2[:, 1]) ** 2, x=ecc1[:, 0])

    def _interpolate(ecc, epsgrid):
        interpolator = spi.interp1d(ecc[:, 0], ecc[:, 1], kind="previous")
        y = interpolator(epsgrid)
        return np.column_stack([epsgrid, y])

    ecc1 = _get_ecc(X1)
    ecc2 = _get_ecc(X2)
    epsmax = max(np.max(ecc1[:, 0]), np.max(ecc2[:, 0]))
    epsgrid = np.linspace(0, epsmax, n_grids)
    ecc1 = _interpolate(_get_ecc(X1, epsmax), epsgrid)
    ecc2 = _interpolate(_get_ecc(X2, epsmax), epsgrid)
    D = _dist_ecc(ecc1, ecc2)

    X12 = np.vstack([X1, X2])
    distances = []
    for _ in range(loops):
        inds = np.random.permutation(n1 + n2)
        x1 = X12[inds[:n1]]
        x2 = X12[inds[n1:]]
        y1 = _interpolate(_get_ecc(x1, epsmax), epsgrid=epsgrid)
        y2 = _interpolate(_get_ecc(x2, epsmax), epsgrid=epsgrid)
        distances.append(_dist_ecc(y1, y2))
    pval = np.mean(distances > D)
    return D, pval, distances
