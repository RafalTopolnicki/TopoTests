import sys

sys.path.append("../topotests/")
import scipy.stats as st
from distributions import *


def get_random_variables(dim):
    rvs = []
    if dim == 1:
        rvs = [
            MultivariateDistribution([st.norm()], label="N_0_1"),
            MultivariateDistribution([st.norm(0, 1.5)], label="N_0_2"),
            MultivariateDistribution([st.beta(2, 2)], label="beta_2_2"),
            MultivariateDistribution([st.beta(5, 5)], label="beta_5_5"),
            MultivariateDistribution([st.laplace()], label="laplace"),
            MultivariateDistribution([st.uniform()], label="U_0_1"),
            MultivariateDistribution([st.t(df=3)], label="T_3"),
            MultivariateDistribution([st.t(df=5)], label="T_5"),
            MultivariateDistribution([st.t(df=10)], label="T_10"),
            MultivariateDistribution([st.cauchy()], label="Cauchy"),
            MultivariateDistribution([st.logistic()], label="Logistic"),
            MultivariateDistribution([AbsoluteDistribution(rv=st.norm())], label="HalfNormal"),
            MultivariateDistribution([GaussianMixture([-1, 1], [1, 1], [0.5, 0.5])], label="GM_1"),
            MultivariateDistribution([GaussianMixture([-0.5, -0.5], [1, 1], [0.5, 0.5])], label="GM_2"),
            MultivariateDistribution([GaussianMixture([0, 1], [1, 2], [0.9, 0.1])], label="GM_3"),
        ]
    if dim == 2:
        rvs = [
            MultivariateDistribution([st.norm(), st.norm()], label="N01xN01"),
            MultivariateGaussian(dim=2, a=0.1, label="MultiGauss0.1"),
            MultivariateGaussian(dim=2, a=0.5, label="MultiGauss0.5"),
            MultivariateGaussian(dim=2, a=0.9, label="MultiGauss0.9"),
            MultivariateDistribution([st.t(df=3), st.t(df=3)], label="T3xT3"),
            MultivariateDistribution([st.t(df=5), st.t(df=5)], label="T5xT5"),
            MultivariateDistribution([st.t(df=10), st.t(df=10)], label="T10xT10"),
            MultivariateDistribution([st.logistic(), st.logistic()], label="LogisticxLogistic"),
            MultivariateDistribution([st.laplace(), st.laplace()], label="LaplacexLaplace"),
            MultivariateDistribution([st.norm(), st.t(df=5)], label="N01xT5"),
            MultivariateDistribution(
                [GaussianMixture([-1, 1], [1, 1], [0.5, 0.5]), GaussianMixture([-1, 1], [1, 1], [0.5, 0.5])],
                label="GM_1xGM_1",
            ),
            MultivariateDistribution([st.norm(), GaussianMixture([-1, 1], [1, 1], [0.5, 0.5])], label="N01xGM_1"),
        ]
    if dim == 3:
        rvs = [
            MultivariateDistribution([st.norm(), st.norm(), st.norm()], label="N01xN01xN01"),
            MultivariateGaussian(dim=3, a=0.1, label="MultiGauss0.1"),
            MultivariateGaussian(dim=3, a=0.5, label="MultiGauss0.5"),
            MultivariateGaussian(dim=3, a=0.9, label="MultiGauss0.9"),
            MultivariateDistribution([st.t(df=3), st.t(df=3), st.t(df=3)], label="T3xT3xT3"),
            MultivariateDistribution([st.uniform(), st.uniform(), st.uniform()], label="UxUxU"),
            MultivariateDistribution([st.beta(2, 2), st.beta(2, 2), st.beta(2, 2)], label="B22xB22xB22"),
            MultivariateDistribution([st.t(df=5), st.t(df=5), st.t(df=5)], label="T5xT5xT5"),
            MultivariateDistribution([st.t(df=10), st.t(df=10), st.t(df=10)], label="T10xT10xT10"),
            MultivariateDistribution([st.logistic(), st.logistic(), st.logistic()], label="LogisticxLogisticxLogistic"),
            MultivariateDistribution([st.laplace(), st.laplace(), st.laplace()], label="LaplacexLaplacexLaplace"),
            MultivariateDistribution([st.norm(), st.t(df=5), st.t(df=5)], label="N01xT5xT5"),
            MultivariateDistribution([st.norm(), st.norm(), st.t(df=5)], label="N01xN01xT5"),
            MultivariateDistribution(
                [
                    GaussianMixture([-1, 1, 0], [1, 1, 1], [0.33, 0.33, 0.34]),
                    GaussianMixture([-1, 1, 0], [1, 1, 1], [0.33, 0.33, 0.34]),
                    GaussianMixture([-1, 1, 0], [1, 1, 2], [0.33, 0.33, 0.34]),
                ],
                label="GM1",
            ),
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
                [st.norm(), st.norm(), st.t(df=5), st.t(df=5), st.t(df=5)], label="N01xN01xT5xT5xT5"
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
    if len(rvs) == 0:
        raise NotImplemented(f"Random variables for dim={dim} not found")
    return rvs
