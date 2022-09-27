import sys

sys.path.append("../topotests/")
import scipy.stats as st
import numpy as np
from distributions import *


def get_random_variables(dim):
    rvs = []
    # # new 1d distributions for tests
    # if dim == -1:
    #     rvs = [
    #         MultivariateDistribution([st.norm()], label="N_0_1"),
    #         MultivariateDistribution([st.beta(2, 2)], label="beta_2_2", shift_and_scale=True),
    #         MultivariateDistribution([st.beta(5, 5)], label="beta_5_5", shift_and_scale=True),
    #         MultivariateDistribution([st.beta(2, 1)], label="beta_2_1", shift_and_scale=True),
    #         MultivariateDistribution([st.beta(3, 2)], label="beta_3_2", shift_and_scale=True),
    #         MultivariateDistribution([st.beta(6, 2)], label="beta_6_2", shift_and_scale=True),
    #         MultivariateDistribution([st.gamma(4, 5)], label="gamma_4_5", shift_and_scale=True),
    #         MultivariateDistribution([GaussianMixture([0, 0], [1, 0.5], [0.9, 0.1])], label="0GM_1"),
    #         MultivariateDistribution([GaussianMixture([0, 0], [1, 0.5], [0.7, 0.3])], label="0GM_2"),
    #         MultivariateDistribution([GaussianMixture([0, 0], [1, 0.5], [0.5, 0.5])], label="0GM_3"),
    #         MultivariateDistribution([GaussianMixture([0, 0], [1, 0.5], [0.3, 0.7])], label="0GM_4"),
    #         MultivariateDistribution([GaussianMixture([0, 0], [1, 2], [0.1, 0.9])], label="1GM_5"),
    #         MultivariateDistribution([GaussianMixture([0, 0], [1, 2], [0.9, 0.1])], label="1GM_1"),
    #         MultivariateDistribution([GaussianMixture([0, 0], [1, 2], [0.7, 0.3])], label="1GM_2"),
    #         MultivariateDistribution([GaussianMixture([0, 0], [1, 2], [0.5, 0.5])], label="1GM_3"),
    #         MultivariateDistribution([GaussianMixture([0, 0], [1, 2], [0.3, 0.7])], label="1GM_4"),
    #         MultivariateDistribution([GaussianMixture([0, 0], [1, 2], [0.1, 0.9])], label="1GM_5"),
    #         MultivariateDistribution([GaussianMixture([0, 4], [1, 0.5], [0.9, 0.1])], label="2GM_1"),
    #         MultivariateDistribution([GaussianMixture([0, 4], [1, 0.5], [0.7, 0.3])], label="2GM_2"),
    #         MultivariateDistribution([GaussianMixture([0, 4], [1, 0.5], [0.5, 0.5])], label="2GM_3"),
    #         MultivariateDistribution([GaussianMixture([0, 4], [1, 0.5], [0.3, 0.7])], label="2GM_4"),
    #         MultivariateDistribution([GaussianMixture([0, 4], [1, 0.5], [0.1, 0.9])], label="2GM_5"),
    #         MultivariateDistribution([GaussianMixture([0, 1], [1, 3], [0.9, 0.1])], label="3GM_1"),
    #         MultivariateDistribution([GaussianMixture([0, 1], [1, 3], [0.7, 0.3])], label="3GM_2"),
    #         MultivariateDistribution([GaussianMixture([0, 1], [1, 3], [0.5, 0.5])], label="3GM_3"),
    #         MultivariateDistribution([GaussianMixture([0, 1], [1, 3], [0.3, 0.7])], label="3GM_4"),
    #         MultivariateDistribution([GaussianMixture([0, 1], [1, 3], [0.1, 0.9])], label="3GM_5"),
    #     ]
    if dim == '1compact':
        rvs = [
            MultivariateDistribution([st.beta(3, 3)], label="beta_3_3"),
            MultivariateDistribution([st.beta(2, 2)], label="beta_2_2"),
            MultivariateDistribution([st.beta(4, 4)], label="beta_4_4"),
            MultivariateDistribution([st.beta(5, 5)], label="beta_5_5"),
            MultivariateDistribution([st.beta(3, 2)], label="beta_3_2"),
            MultivariateDistribution([st.beta(4, 3)], label="beta_4_3"),
            MultivariateDistribution([st.uniform(0, 1)], label="U_0_1"),
            MultivariateDistribution([st.beta(0.8, 0.8)], label="beta_0.8_0.8"),
            MultivariateDistribution([st.beta(0.5, 0.5)], label="beta_0.5_0.5"),
            MultivariateDistribution([st.beta(1.5, 1.5)], label="beta_1.5_1.5"),
            MultivariateDistribution([st.uniform(0.1, 0.8)], label="U_0.1_0.8"),
            MultivariateDistribution([st.argus(chi=0.1)], label="argus_0.1"),
            MultivariateDistribution([st.argus(chi=1)], label="argus_1"),
            MultivariateDistribution([st.argus(chi=2)], label="argus_2"),
            MultivariateDistribution([st.cosine(loc=0.5, scale=0.5/np.pi)], label="cosine"),
        ]
    if dim == 1:
        rvs = [
            MultivariateDistribution([st.norm()], label="N_0_1"),
            MultivariateDistribution([st.norm(0, 0.5)], label="N_0_0.5"),
            MultivariateDistribution([st.norm(0, 0.75)], label="N_0_0.75"),
            MultivariateDistribution([st.norm(0, 1.25)], label="N_0_1.25"),
            MultivariateDistribution([st.norm(0, 1.5)], label="N_0_1.5"),
            MultivariateDistribution([st.beta(2, 2)], label="beta_2_2_scale", shift_and_scale=True),
            MultivariateDistribution([st.beta(5, 5)], label="beta_5_5_scale", shift_and_scale=True),
            MultivariateDistribution([st.beta(2, 1)], label="beta_2_1_scale", shift_and_scale=True),
            MultivariateDistribution([st.beta(3, 2)], label="beta_3_2_scale", shift_and_scale=True),
            MultivariateDistribution([st.beta(6, 2)], label="beta_6_2_scale", shift_and_scale=True),
            MultivariateDistribution([st.gamma(4, 5)], label="gamma_4_5_scale", shift_and_scale=True),
            MultivariateDistribution([st.laplace()], label="laplace"),
            MultivariateDistribution([st.uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3))], label="Unif"),
            MultivariateDistribution([st.uniform()], label="U01"),
            MultivariateDistribution([st.t(df=3)], label="T_3"),
            MultivariateDistribution([st.t(df=5)], label="T_5"),
            MultivariateDistribution([st.t(df=10)], label="T_10"),
            MultivariateDistribution([st.t(df=25)], label="T_25"),
            MultivariateDistribution([st.cauchy()], label="Cauchy"),
            MultivariateDistribution([st.logistic()], label="Logistic"),
            MultivariateDistribution([GaussianMixture([0, 0], [1, 0.5], [0.9, 0.1])], label="1dGM1"),
            MultivariateDistribution([GaussianMixture([0, 0], [1, 0.5], [0.7, 0.3])], label="1dGM2"),
            MultivariateDistribution([GaussianMixture([0, 0], [1, 0.5], [0.5, 0.5])], label="1dGM3"),
            MultivariateDistribution([GaussianMixture([0, 0], [1, 0.5], [0.3, 0.7])], label="1dGM4"),
            MultivariateDistribution([GaussianMixture([0, 0], [1, 0.5], [0.1, 0.9])], label="1dGM5"),
            MultivariateDistribution([GaussianMixture([0, 0], [1, 2], [0.9, 0.1])], label="1dGM6"),
            MultivariateDistribution([GaussianMixture([0, 0], [1, 2], [0.7, 0.3])], label="1dGM7"),
            MultivariateDistribution([GaussianMixture([0, 0], [1, 2], [0.5, 0.5])], label="1dGM8"),
            MultivariateDistribution([GaussianMixture([0, 0], [1, 2], [0.3, 0.7])], label="1dGM9"),
            MultivariateDistribution([GaussianMixture([0, 0], [1, 2], [0.1, 0.9])], label="1dGM10"),
            MultivariateDistribution([GaussianMixture([0, 1], [1, 2], [0.9, 0.1])], label="1dGM11"),
            MultivariateDistribution([GaussianMixture([0, 1], [1, 2], [0.7, 0.3])], label="1dGM12"),
            MultivariateDistribution([GaussianMixture([0, 1], [1, 2], [0.5, 0.5])], label="1dGM13"),
            MultivariateDistribution([GaussianMixture([0, 1], [1, 2], [0.3, 0.7])], label="1dGM14"),
            MultivariateDistribution([GaussianMixture([0, 1], [1, 2], [0.1, 0.9])], label="1dGM15"),
            MultivariateDistribution([GaussianMixture([0, 1], [1, 1], [0.9, 0.1])], label="1dGM16"),
            MultivariateDistribution([GaussianMixture([0, 1], [1, 1], [0.7, 0.3])], label="1dGM17"),
            MultivariateDistribution([GaussianMixture([0, 1], [1, 1], [0.5, 0.5])], label="1dGM18"),
            MultivariateDistribution([GaussianMixture([0, 1], [1, 1], [0.3, 0.7])], label="1dGM19"),
            MultivariateDistribution([GaussianMixture([0, 1], [1, 1], [0.1, 0.9])], label="1dGM20"),
            MultivariateDistribution([GaussianMixture([0, 1], [2, 2], [0.9, 0.1])], label="1dGM21"),
            MultivariateDistribution([GaussianMixture([0, 1], [2, 2], [0.7, 0.3])], label="1dGM22"),
            MultivariateDistribution([GaussianMixture([0, 1], [2, 2], [0.5, 0.5])], label="1dGM23"),
            MultivariateDistribution([GaussianMixture([0, 1], [2, 2], [0.3, 0.7])], label="1dGM24"),
            MultivariateDistribution([GaussianMixture([0, 1], [2, 2], [0.1, 0.9])], label="1dGM25")
        ]
    # if dim == 2:
    #     rvs = [
    #         MultivariateDistribution([st.norm(), st.norm()], label="N01xN01"),
    #         MultivariateGaussian(dim=2, a=0.05, label="MultiGauss0.05"),
    #         MultivariateGaussian(dim=2, a=0.1, label="MultiGauss0.1"),
    #         MultivariateGaussian(dim=2, a=0.2, label="MultiGauss0.2"),
    #         MultivariateGaussian(dim=2, a=0.3, label="MultiGauss0.3"),
    #         MultivariateGaussian(dim=2, a=0.5, label="MultiGauss0.5"),
    #         MultivariateDistribution([st.t(df=3), st.t(df=3)], label="T3xT3"),
    #         MultivariateDistribution([st.t(df=5), st.t(df=5)], label="T5xT5"),
    #         MultivariateDistribution([st.t(df=10), st.t(df=10)], label="T10xT10"),
    #         MultivariateDistribution([st.t(df=25), st.t(df=25)], label="T25x25"),
    #         MultivariateDistribution([st.logistic(), st.logistic()], label="LogisticxLogistic"),
    #         MultivariateDistribution([st.laplace(), st.laplace()], label="LaplacexLaplace"),
    #         MultivariateDistribution([st.norm(), st.t(df=3)], label="N01xT3"),
    #         MultivariateDistribution([st.norm(), st.t(df=5)], label="N01xT5"),
    #         MultivariateDistribution([st.norm(), st.t(df=10)], label="N01xT10"),
    #         MultivariateDistribution(
    #             [GaussianMixture([0, 1], [1, 0.5], [0.9, 0.1]),
    #              GaussianMixture([0, 1], [1, 0.5], [0.9, 0.1])],
    #             label="GM_1xGM_1"
    #         ),
    #         MultivariateDistribution(
    #             [GaussianMixture([0, 1], [1, 0.5], [0.7, 0.3]),
    #              GaussianMixture([0, 1], [1, 0.5], [0.7, 0.3])],
    #             label="GM_2xGM_2"
    #         ),
    #         MultivariateDistribution(
    #             [GaussianMixture([0, 1], [1, 0.5], [0.5, 0.5]),
    #              GaussianMixture([0, 1], [1, 0.5], [0.5, 0.5])],
    #             label="GM_3xGM_3"
    #         )
    #     ]
    # new 2d distributions for tests
    if dim == 2:
        rvs = [
            MultivariateDistribution([st.norm(), st.norm()], label="N01xN01"),
            MultivariateGaussian(dim=2, a=0.05, label="MultiGauss0.05"),
            MultivariateGaussian(dim=2, a=0.1, label="MultiGauss0.1"),
            MultivariateGaussian(dim=2, a=0.2, label="MultiGauss0.2"),
            MultivariateGaussian(dim=2, a=0.3, label="MultiGauss0.3"),
            MultivariateGaussian(dim=2, a=0.5, label="MultiGauss0.5"),
            MultivariateGaussian(dim=2, a=0.7, label="MultiGauss0.7"),
            MultivariateDistribution([st.uniform(loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
                                      st.uniform(loc=-np.sqrt(3), scale=2 * np.sqrt(3))], label="UnifxUnif"),
            MultivariateDistribution([st.uniform(), st.uniform()], label="U01xU01"),
            MultivariateDistribution([st.t(df=3), st.t(df=3)], label="T3xT3"),
            MultivariateDistribution([st.t(df=5), st.t(df=5)], label="T5xT5"),
            MultivariateDistribution([st.t(df=10), st.t(df=10)], label="T10xT10"),
            MultivariateDistribution([st.t(df=25), st.t(df=25)], label="T25x25"),
            MultivariateDistribution([st.norm(), st.t(df=3)], label="N01xT3"),
            MultivariateDistribution([st.norm(), st.t(df=5)], label="N01xT5"),
            MultivariateDistribution([st.norm(), st.t(df=10)], label="N01xT10"),
            GaussianMixture_nd(locations=[[0, 0], [1, 1]], covs=[[[1, 0], [0, 1]], [[3, 0], [0, 3]]], probas=[0.9, 0.1],
                               label='2dGM1'),
            GaussianMixture_nd(locations=[[0, 0], [1, 1]], covs=[[[1, 0], [0, 1]], [[3, 0], [0, 3]]], probas=[0.7, 0.3],
                               label='2dGM2'),
            GaussianMixture_nd(locations=[[0, 0], [1, 1]], covs=[[[1, 0], [0, 1]], [[3, 0], [0, 3]]], probas=[0.5, 0.5],
                               label='2dGM3'),
            GaussianMixture_nd(locations=[[0, 0], [1, 1]], covs=[[[1, 0], [0, 1]], [[3, 0], [0, 3]]], probas=[0.3, 0.7],
                               label='2dGM4'),
            GaussianMixture_nd(locations=[[0, 0], [1, 1]], covs=[[[1, 0], [0, 1]], [[3, 0], [0, 3]]], probas=[0.1, 0.9],
                               label='2dGM5'),
            GaussianMixture_nd(locations=[[0, 0], [0, 0]], covs=[[[1, 0], [0, 1]], [[3, 0], [0, 3]]], probas=[0.9, 0.1],
                               label='2dGM6'),
            GaussianMixture_nd(locations=[[0, 0], [0, 0]], covs=[[[1, 0], [0, 1]], [[3, 0], [0, 3]]], probas=[0.7, 0.3],
                               label='2dGM7'),
            GaussianMixture_nd(locations=[[0, 0], [0, 0]], covs=[[[1, 0], [0, 1]], [[3, 0], [0, 3]]], probas=[0.5, 0.5],
                               label='2dGM8'),
            GaussianMixture_nd(locations=[[0, 0], [0, 0]], covs=[[[1, 0], [0, 1]], [[3, 0], [0, 3]]], probas=[0.3, 0.7],
                               label='2dGM9'),
            GaussianMixture_nd(locations=[[0, 0], [0, 0]], covs=[[[1, 0], [0, 1]], [[3, 0], [0, 3]]], probas=[0.1, 0.9],
                               label='2dGM10'),
            GaussianMixture_nd(locations=[[0, 0], [1, 1], [-1, -1]],
                               covs=[[[1, 0], [0, 1]], [[3, 0], [0, 3]], [[3, 0], [0, 3]]],
                               probas=[0.9, 0.05, 0.05],
                               label='2dGM11'),
            GaussianMixture_nd(locations=[[0, 0], [1, 1], [-1, -1]],
                               covs=[[[1, 0], [0, 1]], [[3, 0], [0, 3]], [[3, 0], [0, 3]]],
                               probas=[0.7, 0.15, 0.15],
                               label='2dGM12'),
            GaussianMixture_nd(locations=[[0, 0], [1, 1], [-1, -1]],
                               covs=[[[1, 0], [0, 1]], [[3, 0], [0, 3]], [[3, 0], [0, 3]]],
                               probas=[0.5, 0.25, 0.25],
                               label='2dGM13'),
        ]

    if dim == 3:
        rvs = [
            MultivariateDistribution([st.norm(), st.norm(), st.norm()], label="N01xN01xN01"),
            MultivariateGaussian(dim=3, a=0.05, label="MultiGauss0.05"),
            MultivariateGaussian(dim=3, a=0.1, label="MultiGauss0.1"),
            MultivariateGaussian(dim=3, a=0.2, label="MultiGauss0.2"),
            MultivariateGaussian(dim=3, a=0.3, label="MultiGauss0.3"),
            MultivariateGaussian(dim=3, a=0.5, label="MultiGauss0.5"),
            MultivariateDistribution([st.uniform(loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
                                      st.uniform(loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
                                      st.uniform(loc=-np.sqrt(3), scale=2 * np.sqrt(3))], label="UnifxUnifxUnif"),
            MultivariateDistribution([st.uniform(), st.uniform(), st.uniform()], label="UxUxU"),
            MultivariateDistribution([st.t(df=3), st.t(df=3), st.t(df=3)], label="T3xT3xT3"),
            MultivariateDistribution([st.t(df=5), st.t(df=5), st.t(df=5)], label="T5xT5xT5"),
            MultivariateDistribution([st.t(df=10), st.t(df=10), st.t(df=10)], label="T10xT10xT10"),
            MultivariateDistribution([st.logistic(), st.logistic(), st.logistic()], label="LogisticxLogisticxLogistic"),
            MultivariateDistribution([st.laplace(), st.laplace(), st.laplace()], label="LaplacexLaplacexLaplace"),
            MultivariateDistribution([st.norm(), st.t(df=5), st.t(df=5)], label="N01xT5xT5"),
            MultivariateDistribution([st.norm(), st.norm(), st.t(df=5)], label="N01xN01xT5"),
            GaussianMixture_nd(locations=[[0, 0, 0], [1, 1, 1]],
                               covs=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[3, 0, 0], [0, 3, 0], [0, 0, 3]]],
                               probas=[0.9, 0.1],
                               label='3dGM1'),
            GaussianMixture_nd(locations=[[0, 0, 0], [1, 1, 1]],
                               covs=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[3, 0, 0], [0, 3, 0], [0, 0, 3]]],
                               probas=[0.5, 0.5],
                               label='3dGM2'),
            GaussianMixture_nd(locations=[[0, 0, 0], [1, 1, 1]],
                               covs=[[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[3, 0, 0], [0, 3, 0], [0, 0, 3]]],
                               probas=[0.1, 0.9],
                               label='3dGM3'),
        ]
    if dim == 5:
        rvs = [
            MultivariateDistribution(
                [st.norm(), st.norm(), st.norm(), st.norm(), st.norm()], label="N01xN01xN01xN01xN01"
            ),
            MultivariateGaussian(dim=5, a=0.1, label="MultiGauss0.1"),
            MultivariateGaussian(dim=5, a=0.5, label="MultiGauss0.5"),
            MultivariateGaussian(dim=5, a=0.9, label="MultiGauss0.9"),
            MultivariateDistribution(
                [st.t(df=3), st.t(df=3), st.t(df=3), st.t(df=3), st.t(df=3)], label="T3xT3xT3xT3xT3"
            ),
            MultivariateDistribution(
                [st.t(df=5), st.t(df=5), st.t(df=5), st.t(df=5), st.t(df=5)], label="T5xT5xT5xT5xT5"
            ),
            MultivariateDistribution(
                [st.t(df=10), st.t(df=10), st.t(df=10), st.t(df=10), st.t(df=10)], label="T10xT10xT10xT10xT10"
            ),
            MultivariateDistribution(
                [st.norm(), st.t(df=5), st.t(df=5), st.t(df=5), st.t(df=5)], label="N01xT5xT5xT4xT5"
            ),
            MultivariateDistribution(
                [st.norm(), st.norm(), st.t(df=5), st.t(df=5), st.t(df=5)], label="-N01xN01xT5xT5xT5"
            ),
            MultivariateDistribution(
                [st.norm(), st.norm(), st.norm(), st.norm(), st.t(df=5)], label="N01xN01xN01xN01xT5"
            ),
            MultivariateDistribution(
                [st.laplace(), st.laplace(), st.laplace(), st.laplace(), st.laplace()], label="LapxLapxLapxLapxLap"
            ),
            MultivariateDistribution(
                [st.norm(), st.norm(), st.laplace(), st.laplace(), st.laplace()], label="N01xN01xLapxLapxLap"
            ),
        ]
    if dim == 7:
        rvs = [
            MultivariateDistribution(
                [st.norm(), st.norm(), st.norm(), st.norm(), st.norm(), st.norm(), st.norm()],
                label="N01xN01xN01xN01xN01"
            ),
            MultivariateGaussian(dim=7, a=0.1, label="MultiGauss0.1"),
            MultivariateGaussian(dim=7, a=0.5, label="MultiGauss0.5"),
            MultivariateDistribution(
                [st.t(df=3), st.t(df=3), st.t(df=3), st.t(df=3), st.t(df=3), st.t(df=3), st.t(df=3)],
                label="T3xT3xT3xT3xT3xT3xT3"
            ),
            MultivariateDistribution(
                [st.t(df=5), st.t(df=5), st.t(df=5), st.t(df=5), st.t(df=5), st.t(df=5), st.t(df=5)],
                label="T5xT5xT5xT5xT5xT5xT5"
            ),
            MultivariateDistribution(
                [st.norm(), st.norm(), st.norm(), st.t(df=5), st.t(df=5), st.t(df=5), st.t(df=5)],
                label="N01xN01xN01xT5xT5xT5xT5"
            ),
            MultivariateDistribution(
                [st.norm(), st.norm(), st.norm(), st.norm(), st.laplace(), st.laplace(), st.laplace()],
                label="N01xN01xN01xN01xLapxLapxLap"
            ),
        ]
    if len(rvs) == 0:
        raise NotImplementedError(f"Random variables for dim={dim} not found")
    return rvs
