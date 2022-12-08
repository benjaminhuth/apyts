from apyts.gsf import *

import pprint

import numpy as np
import matplotlib.pyplot as plt


def sample_mixture(components, n_samples):
    a = np.stack([ np.random.multivariate_normal(mu, cov, n_samples) for _, mu, cov in components ])
    weights = [ w for w, _, _ in components]
    idxs = np.random.choice(len(components), size=n_samples, p=weights)

    return np.stack([ a[i,n,:] for i, n in zip(idxs, range(n_samples))])

def test(components):

    # plt.scatter(samples[:,0], samples[:,1], alpha=0.1)
    # plt.show()

    mean, cov = merge_components(components)

    print("mean estimated:", mean)
    print("cov estimated:\n", cov)
    print()
    samples = sample_mixture(components, 100000)
    print("mean samples:",np.mean(samples, axis=0))
    print("cov samples:\n",np.cov(samples, rowvar=False))


components = [
    (0.25, np.array([0,0]), np.eye(2)),
    (0.25, np.array([4,0]), 0.5*np.eye(2)),
    (0.5, np.array([-1,4]), 2*np.eye(2)),
]

# test(components)


components = [
    (0.1, np.array([1,-1]), 2*np.eye(2)),
    (0.1, np.array([1,-1]), 2*np.eye(2)),
]

test(components)

