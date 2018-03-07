import emcee
import os 
import sys
import numpy as np
import cPickle as pickle
class Sampler():
    def __init__(self,nsteps, nburn, outfile, nwalkers, nparam ,startpoint, model, args, pool):
        self.nsteps=nsteps
        self.nburn = nburn
        self.outfile=outfile
        self.nwalkers=nwalkers
        self.startpoint=startpoint
        self.model=model
        self.args = args
        self.pool = pool 
        self.nparam = nparam
        self.sampler = None
        self.pos = None
        self.ind = 0
        self.rstate=None
        self.randomName = outfile+".rand"
        self.lnprob=None
    def is_master(self):
        return self.pool.is_master()
    def config(self):
        if self.pool is not None:
            if not self.is_master():
                print("I am not a master")
                return
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.nparam, self.model, args=self.args,pool=self.pool)
        if os.path.isfile(self.outfile):
            print("reuse file {0}".format(self.outfile))
            sys.stdout.flush()
            chain = np.loadtxt(self.outfile)
            try:
                recoveredchain = chain.reshape(-1,self.nwalkers,self.startpoint.shape[1]+1).transpose((1,0,2))
            except:
                print("number of worker doesn't match, expected {0}".format(self.nwalkers))
            self.pos = recoveredchain[:,-1,:-1]
            self.lnprob = recoveredchain[:,-1,-1]
            self.ind = recoveredchain.shape[1]
            with open(self.randomName) as f:
                self.rstate = pickle.load(f)
            print("number of steps:{0}".format(self.ind))
        else:
            print("Beginning burn-in with ",self.nwalkers, " walkers and ",self.nburn," steps")
            sys.stdout.flush()
            self.pos, lnprob, rstate = self.sampler.run_mcmc(self.startpoint, self.nburn)
            print >> sys.stderr, "First part of MCMC done.  Starting main segment..."
            self.sampler.reset()

    def execute(self):
        for result in self.sampler.sample(self.pos,lnprob0=self.lnprob,iterations=self.nsteps, storechain=False,rstate0=self.rstate):
            if self.ind > self.nsteps:
                print("done")
                break 
            if (self.ind+1) % 100 == 0:
                print("{0:5.1%}".format(float(self.ind) / self.nsteps))
                sys.stdout.flush()
            position = result[0]
            lnprob = result[1]
            rstate = result[2]
            f = open(self.outfile,'a')
            for i in range(self.nwalkers):
                for j in range(self.nparam):
                    f.write(str(position[i][j])+" ")
                f.write(str(lnprob[i])+"\n")
            self.ind +=1
            f.close()
            with open(self.randomName, "w") as f:
                    pickle.dump(rstate, f, -1)
    def done(self):
        if self.pool is not None:
           self.pool.close()
        print("done mcmc")
