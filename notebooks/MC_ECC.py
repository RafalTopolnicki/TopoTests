#!/usr/bin/env python
# coding: utf-8

# In[36]:


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


# In[44]:


def run_ecc(N, rvs):
    # generate representation for standard normal distribution
    topo_test = TopoTest(n=N, dim=dim, method='ecc', ecc_norm=ecc_norm)
    results = {}
    for rv_true in rvs:
        print(f'*** {datetime.datetime.now()} N={N} RV={rv_true.label}')
        topo_test.fit(rv=rv_true, n_signature=n_signature, n_test=n_test)
        results[rv_true.label] = {rv_true.label: topo_test.representation_threshold['mean']}
        
        for rv_alter in rvs:
            if rv_alter.label == rv_true.label:
                continue
            samples = [rv_alter.rvs(N) for i in range(mc_samples)]
            topo_out = topo_test.predict(samples)
            results[rv_true.label][rv_alter.label] = np.quantile(topo_test.representation_distance_predict, 1-topo_test.significance_level)
    return results


# In[15]:


rvs = [MultivariateDistribution([st.norm(), st.norm(), st.norm()], label='N01xN01xN01'),
       MultivariateDistribution([st.t(df=3), st.t(df=3), st.t(df=3)], label='T3xT3xT3'),
       # MultivariateDistribution([st.t(df=5), st.t(df=5), st.t(df=5)], label='T5xT5xT5'),
       # MultivariateDistribution([st.t(df=10), st.t(df=10), st.t(df=10)], label='T10xT10xT10'),
       # MultivariateDistribution([st.logistic(), st.logistic(), st.logistic()], label='LogisticxLogisticxLogistic'),
       # MultivariateDistribution([st.laplace(), st.laplace(), st.laplace()], label='LaplacexLaplacexLaplace'),
       # MultivariateDistribution([st.norm(), st.t(df=5), st.t(df=5)], label='N01xT5xT5'),
       # MultivariateDistribution([st.norm(), st.norm(), st.t(df=5)], label='N01xN01xT5'),
       # MultivariateDistribution([GaussianMixture([-1, 1, 0], [1, 1, 1], [0.33, 0.33, 0.34]),
       #                          GaussianMixture([-1, 1, 0], [1, 1, 1], [0.33, 0.33, 0.34]),
       #                          GaussianMixture([-1, 1, 0], [1, 1, 2], [0.33, 0.33, 0.34])], label='GM1')
                                ]


# In[ ]:


argv = sys.argv[1:]
ecc_norm = None

try:
    opts, args = getopt.getopt(argv,"n:")
except getopt.GetoptError:
    print('MC_ECC.py -m <method>')
    sys.exit(2)
for opt, arg in opts:
    if opt in ('-n'):
        ecc_norm = arg

if ecc_norm == None:
    raise ValueError('-n parameter missing')


# In[45]:


Ns = [25, 50, 100, 200, 300, 500, 1000]
dim=3
mc_samples = 500
n_signature = n_test = 250
results = {}
for N in Ns:
    out = run_ecc(N, rvs)
    results[N] = out

with open(f'results.3d/ecc_quantiles_{ecc_norm}.pickle', 'wb') as f:
    pickle.dump(results, f)

