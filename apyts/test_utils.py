import pprint

from .constants import *
from .gsf_utils import *

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import scipy.stats as stats

def plot_hists(df, fit_gaussian=False, stddev_range=3, **kwargs):
    if len(df.columns) == 4:
        shape = (2,2)
    else:
        shape = (1, len(df.columns))
    
    fig, axes = plt.subplots(*shape)
    for ax, name in zip(axes.flatten(), df.columns):
        mean = np.mean(df[name])
        std = np.std(df[name])
        x_range = (mean-stddev_range*std, mean+stddev_range*std)
        
        ax.hist(np.clip(df[name], *x_range),
                range=x_range, density=fit_gaussian, **kwargs)
        ax.set_title(name)
        if fit_gaussian:
            x = np.linspace(*x_range, 200)
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


def plot_mixtures_loc_phi_qop_p(components, true_pars=None, draw_mode=True):
    fig, axes = plt.subplots(1,4)
    for ax, idx, name in zip(axes[:3], [eBoundLoc, eBoundPhi, eBoundQoP], ["LOC", "PHI", "QOP"]):
        unit, unit_name = unit_info(idx)
        plot_mixture(ax, components, idx, unit, draw_mode=draw_mode)
        ax.set_xlabel("{} [{}]".format(name.lower(), unit_name))
        ax.set_title(name)
        ax.legend()
        if not true_pars is None:
            ax.vlines(true_pars[idx]/unit, color='tab:blue', ls="-", ymin=0, ymax = ax.get_ylim()[1], label="true", lw=2)

    # Momentum
    def trafo_to_momentum(mean, cov):
        p = abs(1./mean[eBoundQoP])
        mean[eBoundQoP] = p
        cov[eBoundQoP, eBoundQoP] = p*p*cov[eBoundQoP, eBoundQoP]
        return mean, cov

    pcomponents = [ (w, *trafo_to_momentum(p.copy(), c.copy())) for w, p, c in components ]
    plot_mixture(axes[3], pcomponents, eBoundQoP, unit=1.0, draw_mode=draw_mode)
    axes[3].set_xlabel("{} [{}]".format("p", "GeV"))
    axes[3].set_title("P")
    axes[3].legend()
    if not true_pars is None:
        axes[3].vlines(1./true_pars[eBoundQoP], color='tab:blue', ls="-", ymin=0, ymax = axes[3].get_ylim()[1], label="true", lw=2)

    return fig, ax

def plot_mixture(ax, components, idx, unit, draw_mode=True):
    ranges = sum([ [m[idx]/unit+3*np.sqrt(s[idx,idx])/unit, m[idx]/unit-3*np.sqrt(s[idx,idx])/unit] for _, m, s in components ], [])
    xlim = (min(ranges), max(ranges))

    x = np.linspace(*xlim,200)
    ax.plot(x, np.array(mixture_fn(x, components, idx, unit=unit)), lw=3, color='black')
    for cmp in components:
        ax.plot(x, np.array(component_fn(x, cmp, idx, unit=unit)), lw=1, color='darkgrey')

    ymax = ax.get_ylim()[1]

    moments = gaussian_mixture_moments(components)
    mean = moments[0][idx]/unit
    maxw = max(components, key=lambda c: c[0])[1][idx]/unit

    ax.vlines(mean, color='tab:orange', ls="--", ymin=0, ymax = ymax, label="mean", lw=1)
    ax.vlines(maxw, color='tab:green', ls="--", ymin=0, ymax = ymax, label="maxw", lw=1)
    
    if draw_mode:
        mode = gaussian_mixture_mode(components)[0][0][idx]/unit
        ax.vlines(mode, color='tab:red', ls="--", ymin=0, ymax = ymax, label="mode", lw=1)
    else:
        mode = np.nan

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
