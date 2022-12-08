from apyts.gsf import *

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import gamma

def bethe_heitler(z, t):
    c = t/np.log(2)
    return (-np.log(z))**(c-1) / gamma(c)

fig, axes = plt.subplots(1,3)
z = np.linspace(0,0.98,200)

for ax, thickness_in_x0 in zip(axes, [0.02, 0.1, 0.2]):
    fz = bethe_heitler(z, thickness_in_x0)

    ax.set_title("x/x0 = {}".format(thickness_in_x0))
    ax.plot(z, fz, c='r', lw=2)

    cmps = approx_bethe_heitler_distribution(thickness_in_x0)

    ax.plot(z, [ sum([ w*stats.norm.pdf(xx, mean, np.sqrt(var)) for w, mean, var in cmps ]) for xx in z], lw=2, color='black')

plt.show()
