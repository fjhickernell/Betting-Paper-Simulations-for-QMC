from scipy.stats import beta
import qmcpy as qp
import numpy as np 
def gen_qmc_samples_iid(distribution = qp.SciPyWrapper(discrete_distrib=qp.DigitalNetB2(1,seed = 7),scipy_distribs=beta(a=10,b=30))
                         , spawn_samples = 2**4,gen_samples = 2**6, comb = False):
    spw = distribution
    spws = spw.spawn(spawn_samples)
    if comb == False:
        samples = np.array([])
        for i in range (len(spws)):
            curr_samples = spws[i].gen_samples(gen_samples).flatten()
            samples = np.append(samples,curr_samples)
        return samples
    samples = spws[0].gen_samples(gen_samples).flatten()
    for i in range (len(spws)- 1):
        curr_samples = spws[i + 1].gen_samples(gen_samples).flatten()
        samples = np.vstack((samples,curr_samples))
    return np.mean(samples, axis = 0)

for i in range (5):
    print(gen_qmc_samples_iid(gen_samples = 4,comb = True))