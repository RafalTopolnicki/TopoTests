#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import sys
import scipy.stats as st
import matplotlib.pyplot as plt
from pathlib import Path
# setting path
sys.path.append('../topotests/')
sys.path.append('../multiKS/')
from topotests import TopoTest
from distributions import MultivariateDistribution, GaussianMixture, AbsoluteDistribution
from multiKS import multiKS
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import datetime
import sys, getopt


# In[2]:


# split list l into m chunks
# if len(l) is not devisibale by m chunks will have different length
def chunks(l, m):
    chunks = np.array_split(l, m)
    chunks = [chunk for chunk in chunks]
    return chunks


# In[3]:


# run KS it in parallell
def gof_test_process(subsamples):
    ks = []
    for sample in subsamples:
        ks_out = multiKS(sample, cdf_global)
        ks.append(ks_out)
    return ks

def gof_test(samples, Dstar):
    n_cores = 4 
    samples_chunks = chunks(samples, n_cores)
    with ProcessPoolExecutor() as executor:
        results_gen = executor.map(gof_test_process, samples_chunks)
    results = list(results_gen)   
    results_flat = [item for sublist in results for item in sublist]
    ks = [d < Dstar for d in results_flat]
    ks = np.sum(ks)/len(ks)
    return ks, results_flat


# In[4]:


def run_mc(N, rvs, Dstar):
    global cdf_global
    # generate representation for standard normal distribution
    topo_test = TopoTest(n=N, dim=dim, method=method, 
                         wasserstein_p=wasserstein_p, wasserstein_order=wasserstein_order)
    
    results = []
    result_labels = ['true_distrib', 'alter_distrib', 'method', 'sign_level', 'wasserstein_p', 'wasserstein_order',
                 'mc_loops', 'n_signature', 'n_test', 
                 'topo_min', 'topo_mean', 'topo_max', 'topo_quantile',
                 'ks', 'ks_d']
    
    for rv_true in rvs:
        print(f'*** {datetime.datetime.now()} N={N} RV={rv_true.label}')
        topo_test.fit(rv=rv_true, n_signature=n_signature, n_test=n_test)
        # write signature distance matrix
        topo_test.save_distance_matrix(outputfile_basename+f'_N={N}_{rv_true.label}_signature_distance_matrix.npy')
        for rv_alter in rvs:
            print(f'*** {datetime.datetime.now()} N={N} RV={rv_true.label} {rv_alter.label}')
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
            # collect results of KS test
            cdf_global = rv_true.cdf
            ks, dvalues = gof_test(samples, Dstar=Dstar) #Dstar valid for alpha=0.05 N=100, see Justel Table 1 
            #ks, dvalues = [], []
            # collect results of topo tests and goodness of fit (gof) tests
            result = [rv_true.label, rv_alter.label, method, significance_level, wasserstein_p, wasserstein_order, 
                      mc_samples, n_signature, n_test, 
                      topo_min, topo_mean, topo_max, topo_quantile,
                      ks, dvalues]
            results.append(result)
            # save results to .csv file
            results_df = pd.DataFrame(results, columns=result_labels)
            results_df.to_csv(f'{outputfile_basename}_N={N}.csv')
    return results


# In[14]:


rvs = [MultivariateDistribution([st.norm(), st.norm(), st.norm()], label='N01xN01xN01'),
       MultivariateDistribution([st.t(df=3), st.t(df=3), st.t(df=3)], label='T3xT3xT3'),
       MultivariateDistribution([st.t(df=5), st.t(df=5), st.t(df=5)], label='T5xT5xT5'),
       MultivariateDistribution([st.t(df=10), st.t(df=10), st.t(df=10)], label='T10xT10xT10'),
       MultivariateDistribution([st.logistic(), st.logistic(), st.logistic()], label='LogisticxLogisticxLogistic'),
       MultivariateDistribution([st.laplace(), st.laplace(), st.laplace()], label='LaplacexLaplacexLaplace'),
       MultivariateDistribution([st.norm(), st.t(df=5), st.t(df=5)], label='N01xT5xT5'),
       MultivariateDistribution([st.norm(), st.norm(), st.t(df=5)], label='N01xN01xT5'),
       MultivariateDistribution([GaussianMixture([-1, 1, 0], [1, 1, 1], [0.33, 0.33, 0.34]),
                                GaussianMixture([-1, 1, 0], [1, 1, 1], [0.33, 0.33, 0.34]),
                                GaussianMixture([-1, 1, 0], [1, 1, 2], [0.33, 0.33, 0.34])], label='GM1')
                                ]


# In[ ]:


argv = sys.argv[1:]
method = None
try:
    opts, args = getopt.getopt(argv,"m:")
except getopt.GetoptError:
    print('MC_AllDistr3D.py -m <method>')
    sys.exit(2)
for opt, arg in opts:
    if opt in ('-m'):
        method = arg

if method == None:
    raise ValueError('-m parameter missing')


# In[16]:


# set random numbers generator seed to have reproducibale results
np.random.seed(1)

cdf_global = None

# set simulation parameters
Ns = [20, 50, 100, 200]
Dstars = [0.151494, 0.110596, 0.086214, 0.064754] # values obtained from separate KS simulations
mc_samples = 250
n_signature = n_test = 750

# Ns = [20]
# Dstars = [0.151494]
# mc_samples = 10
# n_signature = n_test = 25

dim = 3
significance_level = 0.05
wasserstein_p=1
wasserstein_order=1

outputfile_basename = f'results.{dim}d/{method}_{wasserstein_p}_{wasserstein_order}'


# In[ ]:


for N, Dstar in zip(Ns, Dstars):
    results = run_mc(N=N, rvs=rvs, Dstar=Dstar)

