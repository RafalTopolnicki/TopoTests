import sys

sys.path.append("../topotests/")
import scipy.stats as st
from distributions import *


def get_random_variables(dim, jitter=0.05):
    rvs = []
    if dim == 1:
        rvs = [
            MultivariateDistributionJitter([st.poisson(0.5)], label="Poisson(0.5)"),
            MultivariateDistributionJitter([st.poisson(0.75)], label="Poisson(0.75)"),
            MultivariateDistributionJitter([st.poisson(1)], label="Poisson(1)"),
            MultivariateDistributionJitter([st.poisson(1.25)], label="Poisson(1.25)"),
            MultivariateDistributionJitter([st.poisson(1.5)], label="Poisson(1.5)"),
            MultivariateDistributionJitter([st.poisson(2)], label="Poisson(2)"),
            MultivariateDistributionJitter([st.poisson(5)], label="Poisson(5)"),
            MultivariateDistributionJitter([st.binom(100, 0.3)], label="Binomial(100, 0.3)"),
            MultivariateDistributionJitter([st.binom(100, 0.5)], label="Binomial(100, 0.5)"),
            MultivariateDistributionJitter([st.binom(100, 0.7)], label="Binomial(100, 0.7)"),
            MultivariateDistributionJitter([st.binom(1000, 0.3)], label="Binomial(1000, 0.3)"),
            MultivariateDistributionJitter([st.binom(1000, 0.5)], label="Binomial(1000, 0.5)"),
            MultivariateDistributionJitter([st.binom(1000, 0.5)], label="Binomial(1000, 0.6)"),
            MultivariateDistributionJitter([st.binom(1000, 0.7)], label="Binomial(1000, 0.7)"),
        ]
    if len(rvs) == 0:
        raise NotImplementedError(f"Random variables for dim={dim} not found")
    return rvs
