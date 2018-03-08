import numpy as np
from matplotlib import pyplot as plt
def doubleMADsfromMedian(y,thresh=3.5):
    # warning: this function does not check for NAs
    # nor does it address issues when
    # more than 50% of your data have identical values
    m = np.median(y)
    abs_dev = np.abs(y - m)
    left_mad = np.median(abs_dev[y <= m])
    right_mad = np.median(abs_dev[y >= m])
    y_mad = left_mad * np.ones(len(y))
    y_mad[y > m] = right_mad
    modified_z_score = 0.6745 * abs_dev / y_mad
    modified_z_score[y == m] = 0
    return modified_z_score > thresh
def plot_progress(chain,nwalker):
    newchain = np.loadtxt(chain)
    newchain = newchain.reshape(-1,nwalker,newchain.shape[1]).transpose((1,0,2))
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1,newchain.shape[0] )]
    badwalker=[]
    for paramid in range(newchain.shape[2]):
        fig=plt.figure(figsize=(20,10))
        median = np.median(newchain[:,-1,paramid])
        std = np.std(newchain[:,-1,paramid])
        print(median,std)
        badmask=doubleMADsfromMedian(newchain[:,-1,paramid],thresh=2.0)
        for workerid in range(newchain.shape[0]):
            plt.plot(newchain[workerid,:,paramid],"+", color=colors[workerid])
            if badmask[workerid]:
                badwalker.append(workerid)
                plt.text(len(newchain[workerid,:,paramid]),newchain[workerid,-1,paramid],str(workerid))
    return np.unique(badwalker)
