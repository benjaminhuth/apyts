import pprint

from apyts.constants import *
from apyts.gsf_utils import *

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
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


def plot_mixtures_loc_phi_qop_p(components, true_pars):
    fig, axes = plt.subplots(1,4)
    fig.suptitle("Final Mixture")
    for ax, idx, name in zip(axes[:3], [eBoundLoc, eBoundPhi, eBoundQoP], ["LOC", "PHI", "QOP"]):
        unit, unit_name = unit_info(idx)
        plot_mixture(ax, components, idx, unit)
        ax.set_xlabel("{} [{}]".format(name.lower(), unit_name))
        ax.set_title(name)
        ax.legend()
        ax.vlines(true_pars[idx]/unit, color='tab:blue', ls="-", ymin=0, ymax = ax.get_ylim()[1], label="true", lw=2)

    # Momentum
    def trafo_to_momentum(mean, cov):
        p = abs(1./mean[eBoundQoP])
        mean[eBoundQoP] = p
        cov[eBoundQoP, eBoundQoP] = p*p*cov[eBoundQoP, eBoundQoP]
        return mean, cov

    pcomponents = [ (w, *trafo_to_momentum(p, c)) for w, p, c in components ]
    plot_mixture(axes[3], pcomponents, eBoundQoP, unit=1.0)
    axes[3].set_xlabel("{} [{}]".format("p", "GeV"))
    axes[3].set_title(name)
    axes[3].legend()
    axes[3].vlines(1./true_pars[eBoundQoP], color='tab:blue', ls="-", ymin=0, ymax = axes[3].get_ylim()[1], label="true", lw=2)

    return fig, ax

def plot_mixture(ax, components, idx, unit):
    mode = gaussian_mixture_mode(components)[0][0][idx]/unit
    moments = gaussian_mixture_moments(components)
    mean = moments[0][idx]/unit
    stddev = np.sqrt(moments[1][idx,idx])/unit
    maxw = max(components, key=lambda c: c[0])[1][idx]/unit
    xlim = (mode-3*stddev, mode+3*stddev)


    x = np.linspace(*xlim,200)
    ax.plot(x, np.array(mixture_fn(x, components, idx, unit=unit)), lw=3, color='black')
    for cmp in components:
        ax.plot(x, np.array(component_fn(x, cmp, idx, unit=unit)), lw=1, color='darkgrey')

    ymax = ax.get_ylim()[1]

    ax.vlines(mean, color='tab:orange', ls="--", ymin=0, ymax = ymax, label="mean", lw=1)
    ax.vlines(mode, color='tab:red', ls="--", ymin=0, ymax = ymax, label="mode", lw=1)
    ax.vlines(maxw, color='tab:green', ls="--", ymin=0, ymax = ymax, label="maxw", lw=1)

    logging.debug("plot_mixture | idx {}: mode={:.3f}, mean={:.3f}, maxw={:.3f}".format(idx, mode, mean, maxw))

    ax.set_ylim((0, ymax))
    return ax

class SurfaceSlideShow:
    def make_slideshow(self):
        fig, axes = plt.subplots(self.n_rows,self.n_cols)
        axes = axes.reshape(self.n_rows,self.n_cols)

        def plot_surface(surface):
            surface = surface % self.n_surfaces
            for row in range(self.n_rows):
                for col, ax in enumerate(axes[row,:]):
                    ax.clear()
                    self.plot(ax, surface, row, col)

            fig.suptitle("Surface {}".format(surface))

        plot_surface(0)

        class Callback(object):
            idx = 0

            def next(self, event):
                self.idx += 1
                plot_surface(self.idx)
                plt.draw()

            def prev(self, event):
                self.idx -= 1
                plot_surface(self.idx)
                plt.draw()

        self.callback = Callback()

        fig.subplots_adjust(left = 0.0, top = 0.9, right = 1, bottom = 0.0, hspace = 0.0, wspace = 0.0)
        fig.tight_layout(pad=0, h_pad=0.1, w_pad=0)

        button_x = 0.7
        button_y = 0.97

        button_width = 0.1
        button_height = 0.02

        self.bprev = Button(plt.axes([button_x, button_y, button_width, button_height]), 'Previous')
        self.bprev.on_clicked(self.callback.prev)
        self.bnext = Button(plt.axes([button_x + button_width + 0.01, button_y, button_width, button_height]), 'Next')
        self.bnext.on_clicked(self.callback.next)
