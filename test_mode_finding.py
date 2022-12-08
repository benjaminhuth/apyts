import numpy as np
import matplotlib.pyplot as plt

from apyts.gsf_utils import gaussian_mixture_mode
from scipy.stats import multivariate_normal

pp = [ 0 + i*2 for i in range(4) ]

mixture = [
    (0.1, [pp[0], 0], [[1.0, 0.0], [0.0, 1.0]]),
    (0.2, [pp[1], 0], [[1.0, 0.0], [0.0, 1.0]]),
    (0.3, [pp[2], 0], [[1.0, 0.0], [0.0, 1.0]]),
    (0.4, [pp[3], 0], [[1.0, 0.0], [0.0, 1.0]]),
]

M = gaussian_mixture_mode(mixture)
print(M)

M = M[0][0]

a = min(pp) - 0.5
b = max(pp) + 0.5
x, y = np.mgrid[a:b:.1, -0.5:0.5:.1]

pos = np.dstack((x, y))

dists = [ (w, multivariate_normal(m, c)) for w, m, c in mixture ]

fig, ax = plt.subplots(1,2)
ax[0].contourf(x, y, sum([ w*d.pdf(pos) for w, d in dists ]))
ax[0].scatter([d.mean[0] for w, d in dists], [d.mean[1] for w, d in dists], c='grey')
ax[0].scatter(M[0], M[1], c='r')

x = np.linspace(a,b,200)
ax[1].plot(x, [sum([ w*d.pdf(np.array([xx,0])) for w, d in dists ]) for xx in x])

mode_x = [d.mean[0] for w, d in dists]
ax[1].scatter(mode_x, [ sum([ w*d.pdf(np.array([xx,0])) for w, d in dists ]) for xx in mode_x ], c='grey')
ax[1].scatter([M[0]], [ sum([ w*d.pdf(np.array([float(M[0]),0])) for w, d in dists ]) ], c='r')

plt.show()
