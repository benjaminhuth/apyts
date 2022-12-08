import pprint

from apyts.constants import *
from apyts.gsf_utils import *

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def plot_hists(data, fit_gaussian=False, **kwargs):
    fig, axes = plt.subplots(1, 3)
    for ax, idx, name in zip(axes, [eBoundLoc, eBoundPhi, eBoundQoP], ["LOC", "PHI", "QOP"]):
        _, bins, _ = ax.hist(data[:,idx], **kwargs, density=fit_gaussian)
        ax.set_title(name)
        if fit_gaussian:
            mean = np.mean(data[:,idx])
            std = np.std(data[:,idx])
            x = np.linspace(bins[0], bins[-1], 200)
            ax.plot(x, stats.norm.pdf(x, mean, std), label="µ={:.2f}, s={:.2f}".format(mean, std))
            ax.legend()
    return fig, ax

def mixture_fn(x, components, idx, unit=1.0):
        return [
            sum([
                w*stats.norm.pdf(xx, mean[idx]/unit, np.sqrt(cov[idx, idx])/unit)
                for w, mean, cov in components
            ])
            for xx in x
        ]

def component_fn(x, component, idx, unit=1.0):
    w, mean, cov = component
    return [ w*stats.norm.pdf(xx, mean[idx]/unit, np.sqrt(cov[idx, idx])/unit) for xx in x ]


def plot_mixture(components, true_pars):
    fig, axes = plt.subplots(1,4)
    fig.suptitle("Final Mixture")
    for ax, idx, name, xlim in zip(axes[:3], [eBoundLoc, eBoundPhi, eBoundQoP], ["LOC", "PHI", "QOP"], [5, 3, 0.3]):
        unit, unit_name = unit_info(idx)
        mode = gaussian_mixture_mode(components)[0][0][idx]/unit
        mean = gaussian_mixture_moments(components)[0][idx]/unit
        maxw = max(components, key=lambda c: c[0])[1][idx]/unit
        true = true_pars[idx]/unit
        xlim = (mode-xlim, mode+xlim)

        logging.info("{}: true={:.3f}, mode={:.3f}, mean={:.3f}, maxw={:.3f}".format(name, true, mode, mean, maxw))

        x = np.linspace(*xlim,200)
        ax.plot(x, np.array(mixture_fn(x, components, idx, unit=unit)), lw=3, color='black')
        for cmp in components:
            ax.plot(x, np.array(component_fn(x, cmp, idx, unit=unit)), lw=1, color='darkgrey')
        ax.set_title(name)
        ax.set_xlabel("{} [{}]".format(name, unit_name))

        ymax = ax.get_ylim()[1]

        ax.vlines(true, color='tab:blue', ls="-", ymin=0, ymax = ymax, label="true", lw=2)
        ax.vlines(mean, color='tab:orange', ls="--", ymin=0, ymax = ymax, label="mean", lw=1)
        ax.vlines(mode, color='tab:red', ls="--", ymin=0, ymax = ymax, label="mode", lw=1)
        ax.vlines(maxw, color='tab:green', ls="--", ymin=0, ymax = ymax, label="maxw", lw=1)
        ax.legend()
        ax.set_ylim((0, ymax))

    # momentum
    mode = abs(1./gaussian_mixture_mode(components)[0][0][eBoundQoP])/u.GeV
    mean = abs(1./gaussian_mixture_moments(components)[0][eBoundQoP])/u.GeV
    maxw = abs(1./max(components, key=lambda c: c[0])[1][eBoundQoP])/u.GeV
    true = abs(1./true_pars[eBoundQoP])/u.GeV
    xlim = (mode-1, mode+1)

    logging.info("Momentum: true={:.3f}, mode={:.3f}, mean={:.3f}, maxw={:.3f}".format(true, mode, mean, maxw))

    def trafo_to_momentum(mean, cov):
        p = abs(1./mean[eBoundQoP])
        mean[eBoundQoP] = p
        cov[eBoundQoP, eBoundQoP] = p*p*cov[eBoundQoP, eBoundQoP]
        return mean, cov

    components = [ (w, *trafo_to_momentum(p, c)) for w, p, c in components ]

    x = np.linspace(*xlim,200)
    axes[3].plot(x, np.array(mixture_fn(x, components, eBoundQoP, unit=u.GeV)), lw=3, color='black')
    for cmp in components:
        axes[3].plot(x, np.array(component_fn(x, cmp, eBoundQoP, unit=u.GeV)), lw=1, color='darkgrey')
    axes[3].set_title("Momentum")
    axes[3].set_xlabel("{} [{}]".format("p", "GeV"))

    ymax = axes[3].get_ylim()[1]

    axes[3].vlines(true, color='tab:blue', ls="-", ymin=0, ymax = ymax, label="true", lw=2)
    axes[3].vlines(mean, color='tab:orange', ls="--", ymin=0, ymax = ymax, label="mean", lw=1)
    axes[3].vlines(mode, color='tab:red', ls="--", ymin=0, ymax = ymax, label="mode", lw=1)
    axes[3].vlines(maxw, color='tab:green', ls="--", ymin=0, ymax = ymax, label="maxw", lw=1)
    axes[3].legend()
    axes[3].set_ylim((0, ymax))

    return fig, ax
