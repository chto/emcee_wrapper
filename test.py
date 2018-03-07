import numpy as np
import emcee
from redm_fitting import emcee_wrapper
import sys
def lnprob(x, mu, icov):
    diff = x-mu
    return -np.dot(diff,np.dot(icov,diff))/2.0

from emcee.utils import MPIPool
pool = MPIPool(debug=False)
if not pool.is_master():
    pool.wait()
    sys.exit(0)
ndim = 10

means = np.arange(ndim)

icov = np.diag(0.5 - 0.1*np.ones(ndim))
#cov = np.triu(cov)
#cov += cov.T - np.diag(cov.diagonal())
#cov = np.dot(cov,cov)
#icov = np.linalg.inv(cov)
nwalkers = 20
startpoint = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))

for nsteps in [4000]:
    nburn = 20
    outfile="test_chain_mpi.dat"
    sampler = emcee_wrapper.Sampler(nsteps, nburn, outfile, nwalkers, ndim, startpoint, lnprob, [means, icov], pool)
    sampler.config()
    sampler.execute()
sampler.done()
