#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


# split list l into m chunks
# if len(l) is not devisibale by m chunks will have different length
def chunks(l, m):
    chunks = np.array_split(l, m)
    chunks = [chunk for chunk in chunks]
    return chunks


# In[5]:


# run KS it in parallell
def gof_test_process(subsamples):
    ks = []
    for sample in subsamples:
        ks_out = multiKS(sample, cdf_global)
        ks.append(ks_out)
    return ks

def gof_test(samples, Dstar):
    # SET HERE number of cores that will be used to run KS tests in parallel
    n_cores = 4 
    samples_chunks = chunks(samples, n_cores)
    with ProcessPoolExecutor() as executor:
        results_gen = executor.map(gof_test_process, samples_chunks)
    results = list(results_gen)   
    results_flat = [item for sublist in results for item in sublist]
    ks = [d < Dstar for d in results_flat]
    ks = np.sum(ks)/len(ks)
    return ks, results_flat


# In[6]:


def run_ks(N, rvs, Dstar):
    global cdf_global
    results = []
    result_labels = ['true_distrib', 'alter_distrib', 'method', 'sign_level', 'mc_loops', 'Dstar', 'ks', 'ks_d']
    for rv_true in rvs:
        print(f'*KS {datetime.datetime.now()} N={N} RV={rv_true.label}')
        cdf_global = rv_true.cdf
        for rv_alter in rvs:
            print(f'*KS {datetime.datetime.now()} N={N} RV={rv_true.label} {rv_alter.label}')
            samples = [rv_alter.rvs(N) for i in range(mc_samples)]
            ks, dvalues = gof_test(samples, Dstar=Dstar) #Dstar valid for alpha=0.05 N=100, see Justel Table 1
            result = [rv_true.label, rv_alter.label, method, significance_level, mc_samples, Dstar, ks, dvalues]
            results.append(result)
            
            # save results to .csv file
            results_df = pd.DataFrame(results, columns=result_labels)
            results_df.to_csv(f'{outputfile_basename}_N={N}.csv', index=False)
    return results


# In[35]:


def run_mc(N, rvs):
    # generate representation for standard normal distribution
    topo_test = TopoTest(n=N, dim=dim, method=method, 
                         wasserstein_p=wasserstein_p, wasserstein_order=wasserstein_order, ecc_norm=ecc_norm)
    results = []
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
            topo_aggr = {}
            for key in topo_out.keys():
                topo_aggr[f'topo_{key}'] = np.mean(topo_out[key])
            # save thresholds to the csv as well
            topo_thres = {}
            for key in topo_test.representation_threshold.keys():
                topo_thres[f'thres_{key}'] = topo_test.representation_threshold[key]

            # the is really no point in determining labels in each loop as they are constant, but keep it as it is for simplicity
            result_labels = ['true_distrib', 'alter_distrib', 'method', 'sign_level', 'wasserstein_p', 'wasserstein_order', 'ecc_norm',
                 'mc_loops', 'n_signature', 'n_test', *topo_thres.keys(), *topo_aggr.keys()]

            result = [rv_true.label, rv_alter.label, method, significance_level, wasserstein_p, wasserstein_order, ecc_norm,
                      mc_samples, n_signature, n_test, *topo_thres.values(), *topo_aggr.values()]
            results.append(result)
            
            # save results to .csv file
            results_df = pd.DataFrame([result], columns=result_labels)
            #results_df.reset_index(drop=True, inplace=True)
            try:
                df_old = pd.read_csv(f'{outputfile_basename}_n={N}_M={n_signature}_m={n_test}.csv')
                #df_old.reset_index(drop=True, inplace=True)
                results_df = pd.concat([df_old, results_df], axis=0, ignore_index=True)
            except:
                pass
            results_df.to_csv(f'{outputfile_basename}_n={N}_M={n_signature}_m={n_test}.csv', index=False)
    return results


# In[8]:


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
ecc_norm = None
n_signature = None
n_test = None
mc_samples = None
n_sample = None 

try:
    opts, args = getopt.getopt(argv,"t:M:n:m:C:e:")
except getopt.GetoptError:
    print('MC_AllDistr3D.py -t -M -n -m -C -e')
    sys.exit(2)
for opt, arg in opts:
    if opt in ('-t'):
        method = arg
    if opt in ('-e'):
        ecc_norm = arg
    if opt in ('-M'):
        n_signature = int(arg)
    if opt in ('-m'):
        n_test = int(arg)
    if opt in ('-n'):
        n_sample = int(arg)
    if opt in ('-C'):
        mc_samples = int(arg)

if method == None:
    raise ValueError('-t parameter missing')
if n_signature == None:
    raise ValueError('-M parameter missing')
if n_test == None:
    raise ValueError('-m parameter missing')
if mc_samples == None:
    raise ValueError('-C parameter missing')


# In[13]:


# method='ecc'
# ecc_norm='sup'
# n_signature=25
# n_test=25
# mc_samples=100
# n_sample=100


# In[26]:


# set random numbers generator seed to have reproducibale results
np.random.seed(1)

cdf_global = None

dim = 3
significance_level = 0.05
wasserstein_p=1
wasserstein_order=1

dirname = f'results.{dim}d_convergence'

if method not in ['ecc', 'ks']:
    outputfile_basename = f'{dirname}/{method}_{wasserstein_p}_{wasserstein_order}'
elif method == 'ecc':
    outputfile_basename = f'{dirname}/{method}_{ecc_norm}'
elif method == 'ks':
    outputfile_basename = f'{dirname}/{method}'


# In[36]:


if method != 'ks':
    results = run_mc(N=n_sample, rvs=rvs)
else:
    results = run_ks(N=n_sample, rvs=rvs, Dstar=Dstar)

