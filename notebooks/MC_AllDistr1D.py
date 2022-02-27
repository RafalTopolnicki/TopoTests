#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import scipy.stats as st
import matplotlib.pyplot as plt
from pathlib import Path
# setting path
sys.path.append('../topotests/')
from topotests import TopoTest
from distributions import MultivariateDistribution, GaussianMixture, AbsoluteDistribution
import pandas as pd
import datetime
import sys, getopt


# In[2]:


def gof_tests(samples, cdf):
    ks = [st.kstest(sample.reshape(-1, ), cdf).pvalue > significance_level for sample in samples]
    cvm = [st.cramervonmises(sample.reshape(-1, ), cdf).pvalue > significance_level for sample in samples]
    ks = np.sum(ks)/len(ks)
    cvm = np.sum(cvm)/len(cvm)
    return ks, cvm


# In[5]:


def run_mc(N, rvs):
    # generate representation for standard normal distribution
    topo_test = TopoTest(n=N, dim=dim, method=method, 
                         wasserstein_p=wasserstein_p, wasserstein_order=wasserstein_order)
    
    results = []
    result_labels = ['true_distrib', 'alter_distrib', 'method', 'sign_level', 'wasserstein_p', 'wasserstein_order',
                 'mc_loops', 'n_signature', 'n_test', 
                 'topo_min', 'topo_mean', 'topo_max', 'topo_quantile',
                 'ks', 'cvm']
    for rv_true in rvs:
        print(f'*** {datetime.datetime.now()} N={N} RV={rv_true.label}')
        topo_test.fit(rv=rv_true, n_signature=n_signature, n_test=n_test)
        # write signature distance matrix
        topo_test.save_distance_matrix(outputfile_basename+f'_N={N}_{rv_true.label}_signature_distance_matrix.npy')
        for rv_alter in rvs:
            # generate samples
            samples = [rv_alter.rvs(N) for i in range(mc_samples)]
            # perform topo tests
            topo_out = topo_test.predict(samples)
            # write representation distance matrix
            topo_test.save_predict_distance_matrix(outputfile_basename+f'_N={N}_{rv_true.label}-{rv_alter.label}_distance_matrix.npy')
            # aggregate results of topo tests
            topo_min = np.mean(topo_out.min)
            topo_mean = np.mean(topo_out.mean)
            topo_max = np.mean(topo_out.max)
            topo_quantile = np.mean(topo_out.quantile)
            # collect results of KS and CvM tests
            ks, cvm = gof_tests(samples, cdf=rv_true.cdf)
            # collect results of topo tests and goodness of fit (gof) tests
            result = [rv_true.label, rv_alter.label, method, significance_level, wasserstein_p, wasserstein_order, 
                      mc_samples, n_signature, n_test, 
                      topo_min, topo_mean, topo_max, topo_quantile,
                      ks, cvm]
            results.append(result)
            # save results to .csv file
            results_df = pd.DataFrame(results, columns=result_labels)
            results_df.to_csv(f'{outputfile_basename}_N={N}.csv')
    return results


# In[4]:


rvs = [MultivariateDistribution([st.norm()], label='N_0_1'),
       MultivariateDistribution([st.norm(0, 1.5)], label='N_0_2'),
       MultivariateDistribution([st.norm(0.5, 1)], label='N_0.5_1'),
       MultivariateDistribution([st.beta(2, 2)], label='beta_2_2'),
       MultivariateDistribution([st.beta(5, 5)], label='beta_5_5'),
       MultivariateDistribution([st.laplace()], label='laplace'),
       MultivariateDistribution([st.uniform()], label='U_0_1'),
       MultivariateDistribution([st.t(df=3)], label='T_3'),
       MultivariateDistribution([st.t(df=5)], label='T_5'),
       MultivariateDistribution([st.t(df=10)], label='T_10'),
       MultivariateDistribution([st.cauchy()], label='Cauchy'),
       MultivariateDistribution([st.logistic()], label='Logistic'),
       MultivariateDistribution([AbsoluteDistribution(rv=st.norm())], label='HalfNormal'),
       MultivariateDistribution([GaussianMixture([-1, 1], [1, 1], [0.5, 0.5])], label='GM_1'),
       MultivariateDistribution([GaussianMixture([-0.5, -0.5], [1, 1], [0.5, 0.5])], label='GM_2'),
       MultivariateDistribution([GaussianMixture([0, 1], [1, 2], [0.9, 0.1])], label='GM_3')
      ]


# In[ ]:


argv = sys.argv[1:]
method = None
try:
    opts, args = getopt.getopt(argv,"m:")
except getopt.GetoptError:
    print('MC_AllDistr1D.py -m <method>')
    sys.exit(2)
for opt, arg in opts:
    if opt in ('-m'):
        method = arg

if method == None:
    raise ValueError('-m parameter missing')


# In[5]:


# set random numbers generator seed to have reproducibale results
np.random.seed(1)

# set simulation parameters
Ns = [10, 25, 50, 75, 100, 200, 300, 500]
mc_samples = 250
n_signature = n_test = 750
#method = 'ecc'

dim = 1
significance_level = 0.05
wasserstein_p=1
wasserstein_order=1

outputfile_basename = f'results.{dim}d/{method}_{wasserstein_p}_{wasserstein_order}'


# In[6]:


for N in Ns:
    results = run_mc(N=N, rvs=rvs)

