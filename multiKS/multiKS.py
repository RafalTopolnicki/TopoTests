import numpy as np
from scipy import stats
from itertools import combinations


def IntCdf(fun, p_high, p_low):
    """
    Calculate the integral of probability density function over a hyper-rectangle.
    The CDF (Cumulative distribution function) must be passed.
    The hyper-rectangle is given by two diagonal points p_high and p_low.
    p_high >= p_low componment-wise!

    :param fun: multidimentional CDF, i.e. function that is increasing in each axis
    :param p_high: coordinates of a top diagonal point
    :param p_low: coordinates of a bottom diagonal point
    :return: a integral of probability density function over the hyper-rectangle
    """
    n_dim = len(p_high)
#    for i in range(n_dim):
#        if p_high[i] < p_low[i]:
#            raise ValueError('p_high must be greater than p_low component-wise!')
    if n_dim == 1: # 1-d data must be treated as float (no list od len 1) due to implementation of the cdf
        p_high = p_high[0]
        p_low = p_low[0]
    value = fun(p_high)
    for idx_len in range(1, n_dim):
        factor = (-1) ** idx_len
        change_idxs = list(combinations(range(n_dim), idx_len))
        for change_idx in change_idxs:
            p = p_high.copy()
            for change_id in change_idx:
                p[change_id] = p_low[change_id]
            value += factor * fun(p)
    value += (-1) ** n_dim * fun(p_low)
    return value


def orthant(arr_nd, func, point, axis_low=-np.Inf, axis_high=np.Inf):
    """

    :param arr_nd:
    :param func:
    :param point:
    :param axis_low:
    :param axis_high:
    :return:
    """
    # use binary representation to iterate over orthans
    n_dim = len(point)
    n_data = arr_nd.shape[0]

    # totInt = IntCdf(func, axis_high, axis_low)
    # if totInt < 0.99:
    #    print(totInt)

    theoretical_values = []
    data_values = []

    for i in range(2**n_dim):
        # iterate over all possible orthants
        bin_mask = bin(i)[2:].zfill(n_dim)
        p_low = [None] * n_dim
        p_high = [None] * n_dim
        # parse binary representation into orthants in R^n_dim
        filtration_mask = [True] * n_data
        # print(filtration_mask)

        for id_b, b in enumerate(bin_mask):
            if b == "1":
                p_high[id_b] = axis_high[id_b]
                p_low[id_b] = point[id_b]
                filtration_mask = np.logical_and(filtration_mask, arr_nd[:, id_b] > point[id_b])
            else:
                p_high[id_b] = point[id_b]
                p_low[id_b] = axis_low[id_b]
                # remove datapoints in the orthant
                filtration_mask = np.logical_and(filtration_mask, arr_nd[:, id_b] < point[id_b])
        # get values of theoretical cdf over the orthant
        theoretical_values.append(IntCdf(func, p_high, p_low))
        data_values.append(np.sum(filtration_mask) / n_data)
    d = np.max(np.abs(np.array(theoretical_values) - np.array(data_values)))
    return d


def multiKS(arr_nd, cdf_nd):
    """
    Multidimension one-sample Kolmogorov-Smirnov test.
    The test works in any dimension but quantiles of test statistics must be provided
    (e.g. computed using Monte-Carlo method)

    :param arr_nd: (N, n) array of N data points. n is the data dimension.
    :param cdf_nd: Cumulative distribution function of the test distribution.
                    It must takes n-element list as an argument
    :return: Value of KS statistic.
    """
    # for 1d data the Scipy implementation of KS test is 20-200x faster - use it
    if len(arr_nd.shape) == 1 or arr_nd.shape[1] == 1:
        ks_out = stats.ks_1samp(arr_nd[:, 0], cdf_nd)
        return ks_out[0]
    # for unknown reason Cumulative distribution functions implemented in Scipy
    # does not work properly with np.Inf and -np.Inf
    # this is an ugly bypass
    # numeric_low = -1e4
    # numeric_high = 1e4
    numeric_low = np.min(arr_nd, axis=0) - 20
    numeric_high = np.max(arr_nd, axis=0) + 20
    d_ks = []
    for point in arr_nd:
        d_ks.append(orthant(arr_nd=arr_nd, func=cdf_nd, point=point, axis_low=numeric_low, axis_high=numeric_high))
    return np.max(d_ks)


def orthant_data(arr_nd1, arr_nd2, point):
    """

    :param arr_nd:
    :param func:
    :param point:
    :param axis_low:
    :param axis_high:
    :return:
    """
    # use binary representation to iterate over orthans
    n_dim = len(point)
    n_data = arr_nd1.shape[0]

    data_values1 = []
    data_values2 = []

    for i in range(2**n_dim):
        # iterate over all possible orthants
        bin_mask = bin(i)[2:].zfill(n_dim)
        p_low = [None] * n_dim
        p_high = [None] * n_dim
        # parse binary representation into orthants in R^n_dim
        filtration_mask1 = [True] * n_data
        filtration_mask2 = [True] * n_data

        for id_b, b in enumerate(bin_mask):
            if b == "1":
                filtration_mask1 = np.logical_and(filtration_mask1, arr_nd1[:, id_b] > point[id_b])
                filtration_mask2 = np.logical_and(filtration_mask2, arr_nd2[:, id_b] > point[id_b])
            else:
                filtration_mask1 = np.logical_and(filtration_mask1, arr_nd1[:, id_b] < point[id_b])
                filtration_mask2 = np.logical_and(filtration_mask2, arr_nd2[:, id_b] < point[id_b])
        data_values1.append(np.sum(filtration_mask1) / n_data)
        data_values2.append(np.sum(filtration_mask2) / n_data)

    d = np.max(np.abs(np.array(data_values1) - np.array(data_values2)))
    return d


def multiKS2s(arr_nd1, arr_nd2):
    if arr_nd1.shape[0] != arr_nd2.shape[0]:
        raise NotImplementedError('Samples must be of equal length!')
    d_ks1 = []
    d_ks2 = []
    for point1 in arr_nd1:
        d_ks1.append(orthant_data(arr_nd1=arr_nd1, arr_nd2=arr_nd2, point=point1))
    for point2 in arr_nd2:
        d_ks2.append(orthant_data(arr_nd1=arr_nd1, arr_nd2=arr_nd2, point=point2))
    d_ks1 = np.max(d_ks1)
    d_ks2 = np.max(d_ks2)
    d = (d_ks1 + d_ks2)/2
    return d, np.nan

