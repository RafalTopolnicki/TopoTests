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
import pickle


# In[2]:


def run_ecc(N, rvs, mc_sample):
    # generate representation for standard normal distribution
    topo_test = TopoTest(n=N, dim=dim, method='ecc', ecc_norm=ecc_norm, scaling=N**(1.0/3.0))
    results = {}
    for rv_true in rvs:
        print(f'*** {datetime.datetime.now()} N={N} mc_sample={mc_sample} RV={rv_true.label}')
        topo_test.fit(rv=rv_true, n_signature=mc_sample, n_test=mc_sample)
        results[rv_true.label] = {rv_true.label: topo_test.representation_threshold['mean']}
    return results


# In[3]:


rvs = [MultivariateDistribution([st.norm(), st.norm(), st.norm()], label='N01xN01xN01'),
       MultivariateDistribution([st.t(df=3), st.t(df=3), st.t(df=3)], label='T3xT3xT3'),
       #MultivariateDistribution([st.t(df=5), st.t(df=5), st.t(df=5)], label='T5xT5xT5'),
       MultivariateDistribution([st.t(df=10), st.t(df=10), st.t(df=10)], label='T10xT10xT10'),
       MultivariateDistribution([st.logistic(), st.logistic(), st.logistic()], label='LogisticxLogisticxLogistic'),
       MultivariateDistribution([st.laplace(), st.laplace(), st.laplace()], label='LaplacexLaplacexLaplace'),
       MultivariateDistribution([st.norm(), st.t(df=5), st.t(df=5)], label='N01xT5xT5'),
       MultivariateDistribution([st.norm(), st.norm(), st.t(df=5)], label='N01xN01xT5'),
       #MultivariateDistribution([GaussianMixture([-1, 1, 0], [1, 1, 1], [0.33, 0.33, 0.34]),
       #                         GaussianMixture([-1, 1, 0], [1, 1, 1], [0.33, 0.33, 0.34]),
       #                         GaussianMixture([-1, 1, 0], [1, 1, 2], [0.33, 0.33, 0.34])], label='GM1')
                                ]


# In[ ]:


argv = sys.argv[1:]
ecc_norm = None
N = None

try:
    opts, args = getopt.getopt(argv,"n:N:")
except getopt.GetoptError:
    print('MC_ECC.py -m <method>')
    sys.exit(2)
for opt, arg in opts:
    if opt in ('-n'):
        ecc_norm = arg
    if opt in ('-N'):
        N = int(arg)

if ecc_norm == None:
    raise ValueError('-n parameter missing')
if N == None:
    raise ValueError('-N parameter missing')


# In[ ]:


dim=3
mc_samples = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

results = {}
for mc_sample in mc_samples:
    out = run_ecc(N, rvs, mc_sample)
    results[mc_sample] = out
    with open(f'results.3d/xx_ecc_convergence_{ecc_norm}_N={N}_scaling.pickle', 'wb') as f:
        pickle.dump(results, f)

