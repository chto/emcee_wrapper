import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import pandas as pd 
def doubleMADsfromMedian(y,thresh=3.5, badmask=None):
    # warning: this function does not check for NAs
    # nor does it address issues when
    # more than 50% of your data have identical values
    m = np.median(y)
    if badmask is not None:
        m=np.median(y[np.logical_not(badmask)])
    else:
        std = np.std(y)
        m=np.median(y[np.abs(y-m)<std])
    abs_dev = np.abs(y - m)
    left_mad = np.median(abs_dev[y <= m])
    right_mad = np.median(abs_dev[y >= m])
    y_mad = left_mad * np.ones(len(y))
    y_mad[y > m] = right_mad
    modified_z_score = 0.6745 * abs_dev / y_mad
    modified_z_score[y == m] = 0
    return modified_z_score > thresh
def plot_progress(chain,nwalker, outdir=None, cut=0):
    newchain = pd.read_csv(chain, delimiter=" ", header=None)
    newchain = newchain[newchain.index>=cut*nwalker].as_matrix()
    #newchain = np.loadtxt(chain, skiprows=cut*nwalker)
    newchain = newchain.reshape(-1,nwalker,newchain.shape[1]).transpose((1,0,2))
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1,newchain.shape[0] )]
    badwalker=[]
    for paramid in range(newchain.shape[2]):
        #fig=plt.figure(figsize=(20,10))
        badmask = doubleMADsfromMedian(np.mean(newchain[:,-nwalker:,paramid], axis=1),thresh=1.3)
        badmask = doubleMADsfromMedian(np.mean(newchain[:,-nwalker:,paramid], axis=1),thresh=2.0, badmask=badmask)
        for workerid in range(newchain.shape[0]):
            #plt.plot(newchain[workerid,:,paramid],"+", color=colors[workerid])
            if badmask[workerid]:
                badwalker.append(workerid)
            #    plt.text(len(newchain[workerid,:,paramid]),newchain[workerid,-1,paramid],str(workerid))
    for paramid in range(newchain.shape[2]):
        fig=plt.figure(figsize=(10,5))
        for workerid in range(newchain.shape[0]):
            plt.plot(newchain[workerid,:,paramid],"+", color=colors[workerid])
            if workerid in badwalker:
                plt.text(len(newchain[workerid,:,paramid]),newchain[workerid,-1,paramid],str(workerid))

        if paramid == newchain.shape[2]-1:
            ax = plt.gca()
            ax.set_title("score")
        if outdir is not None:
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
            fig.savefig(outdir+"mcmc_param_{0}.jpg".format(paramid))
            plt.close()

    return newchain, np.unique(badwalker)
