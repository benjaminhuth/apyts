import logging
import pickle
from itertools import cycle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from apyts.constants import *
from apyts.gsf_utils import *
from test_utils import *



def analyse_gsf(true_pars, sim_data, gsf_fit_data, kf_fit_data):
    energy_loss = np.array([
        abs(1./truth_sim[-1][eBoundQoP]) - abs(1./truth_start[eBoundQoP]) for (_, truth_sim, _), truth_start in zip(sim_data, true_pars)
    ])

    for fit_data, name in zip([ gsf_fit_data, kf_fit_data], ["GSF", "KF"]):
        mask = np.array([ False if data is None else True for data in fit_data ])
        fit_data = [ d for d in fit_data if d is not None ]

        assert len(mask) > 0
        logging.info("fit failed for {}/{} ({:.2f}%) of particles".format(sum(np.logical_not(mask)), len(mask), 100.*sum(np.logical_not(mask))/len(mask)))

        # Make residuals
        if name == "GSF":
            #final_pars = np.stack([ gaussian_mixture_mode(cmps) for cmps, _, _, _ in fit_data ])
            #final_pars = np.stack([ max(cmps, key=lambda c: c[0])[1] for cmps, _, _, _ in fit_data ])
            final_pars = np.stack([ gaussian_mixture_moments(cmps)[0] for cmps, _, _, _ in fit_data ])
        else:
            final_pars = np.stack([ data[0][0] for data in fit_data ])

        residuals = final_pars - true_pars[mask]

        residuals[:, eBoundLoc] = np.clip(residuals[:,eBoundLoc], -2*u.mm, 2*u.mm)
        residuals[:, eBoundPhi] = np.clip(residuals[:,eBoundPhi], -10*u.degree, 10*u.degree)
        residuals[:, eBoundPhi] = np.clip(residuals[:,eBoundPhi], -0.5, 0.5)

        res_momentum = abs(1./final_pars[:,eBoundQoP]) - abs(1./true_pars[mask][:,eBoundQoP])

        fig, ax = plot_hists(residuals, bins="rice")
        fig.suptitle("Residuals {}".format(name))

        fig, ax = plt.subplots()
        ax.hist(res_momentum, bins="rice")
        ax.set_title("Momentum res {}".format(name))

        fig, ax = plt.subplots(2)
        ax[0].set_title("{}: res vs loss".format(name))

        ax[0].scatter(res_momentum, energy_loss[mask], alpha=0.3)

        ax[1].set_title("Correlation coefficient")
        keys = ["res p", "energy_loss"]
        im = ax[1].imshow(np.corrcoef(np.vstack([res_momentum, energy_loss[mask]])), origin="lower", aspect=0.3, vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax[1], label='Interactive colorbar')

        ax[1].set_xticks(np.arange(len(keys)))
        ax[1].set_yticks(np.arange(len(keys)))

        ax[1].set_xticklabels(keys)
        ax[1].set_yticklabels(keys)
        fig.tight_layout()

class ResidualSlideShow(SurfaceSlideShow):
    def __init__(self):
        self.n_surfaces = 0
        self.n_rows = 0
        self.residuals = []
        self.row_names = []
        self.n_cols = 4
        self.col_names = ["LOC", "PHI", "QOP", "P"]
        self.col_ranges = [(-0.1,0.1), (-0.05,0.05), (-0.5,0.5), (-1,1)]

    def plot(self, ax, surface, row, col):
            rmin = min(self.col_ranges[col][0], min(self.residuals[surface][row][:,col]))
            rmax = max(self.col_ranges[col][1], max(self.residuals[surface][row][:,col]))
            ax.hist(self.residuals[surface][row][:,col], bins="rice", range=(rmin, rmax), histtype="step")
            ax.set_title("{} ({})".format(self.col_names[col], self.row_names[row]))

    def add_row(self, sim_data, fit_data_surface_fn, rowname=""):
        self.n_rows += 1
        self.row_names.append(rowname)

        n_surfaces = max([ len(truth) for _, truth, _ in sim_data ])

        for surface in range(n_surfaces):
            surface_filtered = fit_data_surface_fn(surface)
            surface_truth = np.stack([ truth[surface] for _, truth, _ in sim_data if len(truth) > surface ])

            surface_filtered = np.append(surface_filtered, abs(1./surface_filtered[:,eBoundQoP]).reshape(-1,1), axis=1)
            surface_truth = np.append(surface_truth, abs(1./surface_truth[:,eBoundQoP]).reshape(-1,1), axis=1)

            try:
                self.residuals[surface].append(surface_filtered - surface_truth)
            except:
                self.residuals.append([surface_filtered - surface_truth])

        self.n_surfaces = max(n_surfaces, self.n_surfaces)



if __name__ == "__main__":
    input_path = Path("output/20221208_150248_no_loss")

    with open(input_path / "sim_data.pickle", "rb") as f:
        true_pars, sim_data, _ ,_ = pickle.load(f)
    with open(input_path / "start_parameters.pickle", "rb") as f:
        start_pars, start_covs = pickle.load(f)
    with open(input_path / "gsf_fit_data.pickle", "rb") as f:
        gsf_fit_data = pickle.load(f)
    with open(input_path / "kf_fit_data.pickle", "rb") as f:
        kf_fit_data = pickle.load(f)

    # analyse_gsf(true_pars, sim_data, gsf_fit_data, kf_fit_data)

    kf_mask = np.array([ False if data is None else True for data in kf_fit_data ])
    kf_fit_data = [ d for d in kf_fit_data if d is not None ]
    kf_sim_data = [ d for d, keep in zip(sim_data, kf_mask) if keep ]

    gsf_mask = np.array([ False if data is None else True for data in gsf_fit_data ])
    gsf_fit_data = [ d for d in gsf_fit_data if d is not None ]
    gsf_sim_data = [ d for d, keep in zip(sim_data, gsf_mask) if keep ]

    kf_surface_filtered_fn = lambda i: np.stack([ filtered[i][0] for _, _, filtered, _ in kf_fit_data if len(filtered) > i ])
    gsf_surface_filtered_maxw_fn = lambda i: np.stack([ max(filtered[i], key=lambda c: c[0])[1] for _, _, filtered, _ in gsf_fit_data if len(filtered) > i ])
    gsf_surface_filtered_mean_fn = lambda i: np.stack([ gaussian_mixture_moments(filtered[i])[0] for _, _, filtered, _ in gsf_fit_data if len(filtered) > i ])

    rs = ResidualSlideShow()
    rs.add_row(kf_sim_data, kf_surface_filtered_fn, "KF")
    rs.add_row(gsf_sim_data, gsf_surface_filtered_maxw_fn, "GSF|maxw")
    rs.add_row(gsf_sim_data, gsf_surface_filtered_mean_fn, "GSF|mean")
    rs.make_slideshow()

    plt.show()

