import logging
import pickle
import datetime
from typing import NamedTuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from apyts.geometry import *
from apyts.simulation import *
from apyts.kalman_fitter import *
from apyts.gsf import *
from apyts.gsf_utils import *
from apyts.constants import *
import apyts.units as u

from test_utils import *

def test_pulls():
    logging.info("Start")
    geometry = Geometry([0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
                        surface_radius=300 * u.mm,
                        b_field=2 * u.Tesla,
                        no_plot=True,
    )

    n_particles = 100
    write_to_file = False

    if write_to_file:
        output_folder = Path("output/{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        output_folder.mkdir(parents=True)

    ############
    # Simulate #
    ############

    std_simulation = 0.01 * u.mm
    simulate_radiation_loss = False
    simulation = Simulation(geometry, smearing_stddev=std_simulation, simulate_radiation_loss=simulate_radiation_loss)

    charges = np.random.choice([-1, 1], size=n_particles)
    momenta = np.random.uniform(4.0*u.GeV, 4.0*u.GeV, size=n_particles)
    phis = np.random.uniform(60*u.degree, 120*u.degree, size=n_particles)
    locs = np.random.uniform(-10*u.mm, 10*u.mm, size=n_particles)
    qops = charges / momenta

    true_pars = np.stack((locs, phis, qops), axis=1)

    sim_data = [ simulation.simulate(pars, kElectronMass) for pars in true_pars ]

    # Filter energy loss on first surface out since GSF cannot detect it
    if False:
        eloss_on_first_surface_mask = np.array([
            True if abs(1./truth[1][eBoundQoP])/abs(1./truth[0][eBoundQoP]) < 0.99 else False for _, truth, _ in sim_data
        ])
        logging.info("Number of energy loss on first surface: {}".format(sum(eloss_on_first_surface_mask)))

        true_pars = true_pars[np.logical_not(eloss_on_first_surface_mask)]
        sim_data = [ sd for sd, keep in zip(sim_data, np.logical_not(eloss_on_first_surface_mask)) if keep ]
        n_particles -= sum(eloss_on_first_surface_mask)

    energy_loss = np.array([
        abs(1./truth_sim[-1][eBoundQoP]) - abs(1./truth_start[eBoundQoP]) for (_, truth_sim, _), truth_start in zip(sim_data, true_pars)
    ])

    if write_to_file:
        with open(output_folder / "sim_data.pickle", 'wb') as f:
            pickle.dump((true_pars, sim_data, std_simulation, simulate_radiation_loss), f)
            logging.info("Wrote simulation data to '{}'".format(f.name))

    logging.info("Done with simulation")

    ##########################
    # Smear start parameters #
    ##########################

    # Fixed initial cov for loc and phi
    std_loc = 20*u.um
    std_phi = 1*u.degree

    # Momentum dependent cov for qop
    std_p_rel = 0.05
    p = true_pars[:, eBoundQoP]
    std_qop = std_p_rel * p / p**2

    smearing = np.stack([
        std_loc * np.random.normal(0, 1, size=n_particles),
        std_phi * np.random.normal(0, 1, size=n_particles),
        std_qop * np.random.normal(0, 1, size=n_particles),
    ], axis=-1)

    smeared_pars = true_pars + smearing

    var_inflation = 100
    covs = np.zeros((n_particles,3,3))
    covs[:,eBoundLoc, eBoundLoc] = var_inflation * std_loc**2
    covs[:,eBoundPhi, eBoundPhi] = var_inflation * std_phi**2
    covs[:,eBoundQoP, eBoundQoP] = var_inflation * std_qop**2

    if write_to_file:
        with open(output_folder / "start_parameters.pickle", 'wb') as f:
            pickle.dump((smeared_pars, covs), f)
            logging.info("Wrote start data to '{}'".format(f.name))

    ##################
    # Fit & Analysis #
    ##################

    projector = np.array([[1, 0, 0]])

    kf = KalmanFitter(geometry, projector)
    gsf = GSF(geometry, projector, max_components=2, weight_cutoff=1.e-2, full_kl_divergence=False)

    def fit(fitter, name):
        fit_data = [
            fitter.fit(pars, cov, surfaces, measuements, len(surfaces)*[std_simulation**2])
            for pars, cov, (measuements, _, surfaces) in zip(smeared_pars, covs, sim_data)
        ]

        if write_to_file:
            with open(output_folder / "{}_fit_data.pickle".format(name.lower()), 'wb') as f:
                pickle.dump(fit_data, f)
                logging.info("Wrote fit data to '{}'".format(f.name))

        logging.info("Done with {} fitting".format(name))
        mask = np.array([ False if data is None else True for data in fit_data ])
        fit_data = [ d for d in fit_data if d is not None ]

        assert len(mask) > 0
        logging.info("fit failed for {}/{} ({:.2f}%) of particles".format(sum(np.logical_not(mask)), len(mask), 100.*sum(np.logical_not(mask))/len(mask)))

        # Make residuals
        if type(fitter) == GSF:
            # final_pars = np.stack([ max(data[0], key=lambda c: c[0])[1] for data in fit_data ])
            final_pars = np.stack([ gaussian_mixture_moments(data[0])[0] for data in fit_data ])
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



    fit(kf, "KF")
    fit(gsf, "GSF")

    plt.show()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%H:%M:%S')

    plt.set_loglevel("warn")
    np.random.seed(1234)
    test_pulls()


