import sys
import numpy as np
sys.path.append('../topotests/')
#from pdlltest import *
from pdwdtestcdf import PDWDCDFTest_onesample
import scipy.stats as ss

#norm = ss.norm()
norm = ss.multivariate_normal()
unif = ss.uniform()

pdwd = PDWDCDFTest_onesample(n=100, dim=2, persistence_dim=1)
pdwd.fit(norm, n_signature=100, n_test=100)

samples = [norm.rvs(100) for _ in range(100)]
pred = pdwd.predict(samples)
print(np.mean(pred[0]))

samples = [unif.rvs(100) for _ in range(100)]
pred = pdwd.predict(samples)
print(np.mean(pred[0]))
