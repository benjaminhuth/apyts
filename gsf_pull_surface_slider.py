import logging
import pickle
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from apyts.constants import *
from apyts.gsf_utils import *
from test_utils import *
        

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
    input_path = Path("output/20230117_132335-500particles")

    with open(input_path / "sim_result.pickle", "rb") as f:
        data = pickle.load(f)
        true_pars = data["true_pars"]
        sim_results = data["sim_results"]
    with open(input_path / "start_parameters.pickle", "rb") as f:
        data = pickle.load(f)
        start_pars = data["parameters"]
        start_covs = data["covs"]
    with open(input_path / "gsf_fit_result.pickle", "rb") as f:
        gsf_fit_results = pickle.load(f)["fit_result"]
    with open(input_path / "kf_fit_result.pickle", "rb") as f:
        kf_fit_results = pickle.load(f)["fit_result"]
    
    kf_mask = np.array([ False if data is None else True for data in kf_fit_results ])
    kf_fit_results = [ d for d in kf_fit_results if d is not None ]
    kf_sim_results = [ d for d, keep in zip(sim_results, kf_mask) if keep ]
    kf_final_pars = np.stack([ data[0][0] for data in kf_fit_results ])
    
    gsf_mask = np.array([ False if data is None else True for data in gsf_fit_results ])
    gsf_fit_results = [ d for d in gsf_fit_results if d is not None ]
    gsf_sim_results = [ d for d, keep in zip(sim_results, gsf_mask) if keep ]
    gsf_final_pars = np.stack([ max(cmps, key=lambda c: c[0])[1] for cmps, _, _, _ in gsf_fit_results ])
    
    kf_surface_filtered_fn = lambda i: \
        np.stack([ filtered[i][0] for _, _, filtered, _ in kf_fit_results if len(filtered) > i ])
    gsf_surface_filtered_maxw_fn = lambda i: \
        np.stack([ max(filtered[i], key=lambda c: c[0])[1] for _, _, filtered, _ in gsf_fit_results if len(filtered) > i ])
    gsf_surface_filtered_mean_fn = lambda i: \
        np.stack([ gaussian_mixture_moments(filtered[i])[0] for _, _, filtered, _ in gsf_fit_results if len(filtered) > i ])

    rs = ResidualSlideShow()
    rs.add_row(kf_sim_results, kf_surface_filtered_fn, "KF")
    rs.add_row(gsf_sim_results, gsf_surface_filtered_maxw_fn, "GSF|maxw")
    rs.add_row(gsf_sim_results, gsf_surface_filtered_mean_fn, "GSF|mean")
    rs.make_slideshow()

    plt.show()

