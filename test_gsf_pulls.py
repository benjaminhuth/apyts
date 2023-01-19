import logging
import pickle
import warnings
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

def test_pulls(n_particles : int, output_path : Path = None):
    logging.info("Start")
    geometry = Geometry([0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
                        surface_radius=300 * u.mm,
                        thickness_in_x0=0.5*u.mm / kSiRadiationLength,
                        b_field=2 * u.Tesla,
                        no_plot=True,
    )

    ############
    # Simulate #
    ############

    stddev_sim = 0.01 * u.mm
    simulate_radiation_loss = True
    logging.info("Simulation: smearing std = {}, radiation_loss = {}".format(stddev_sim, simulate_radiation_loss))
    simulation = Simulation(geometry, smearing_stddev=stddev_sim, simulate_radiation_loss=simulate_radiation_loss)

    if False:
        charges = np.random.choice([-1, 1], size=n_particles)
        momenta = np.random.uniform(4.0*u.GeV, 4.0*u.GeV, size=n_particles)
        phis = np.random.uniform(60*u.degree, 120*u.degree, size=n_particles)
        locs = np.random.uniform(-10*u.mm, 10*u.mm, size=n_particles)
        qops = charges / momenta
        logging.info("Random start parameters")
    else:
        locs = np.full(n_particles, 0*u.mm)
        phis = np.full(n_particles, 90*u.degree)
        qops = -1./np.full(n_particles, 4*u.GeV)
        logging.info("Fixed start parameters ({:.2f},{:.2f},{:.2f})".format(locs[0], phis[0], qops[0]))

    true_pars = np.stack((locs, phis, qops), axis=1)

    sim_results = [ simulation.simulate(pars, kElectronMass) for pars in true_pars ]

    if output_path:
        with open(output_path / "sim_result.pickle", 'wb') as f:
            pickle.dump(dict(true_pars=true_pars, sim_results=sim_results, stddev_sim=stddev_sim, simulate_rad_loss=simulate_radiation_loss), f)
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
    p = abs(1./true_pars[:, eBoundQoP])
    std_qop = std_p_rel * p / p**2
    
    logging.info("Start parameter smearing std: ({:.2f}, {:.2f}, {:.2f}+-{:.2f})".format(std_loc, std_phi, np.mean(std_qop), np.std(std_qop)))

    smearing = np.stack([
        std_loc * np.random.normal(0, 1, size=n_particles),
        std_phi * np.random.normal(0, 1, size=n_particles),
        std_qop * np.random.normal(0, 1, size=n_particles),
    ], axis=-1)

    smeared_pars = true_pars + smearing

    # hack so that all inital particles are the same
    smeared_pars = smeared_pars[0]
    smeared_pars = np.ones((n_particles, 3)) * smeared_pars

    # plot_hists(smeared_pars)
    # plt.show()
    # exit()

    var_inflation = 100
    logging.info("variance inflation: {}".format(var_inflation))
    
    covs = np.zeros((n_particles,3,3))
    covs[:,eBoundLoc, eBoundLoc] = var_inflation * std_loc**2
    covs[:,eBoundPhi, eBoundPhi] = var_inflation * std_phi**2
    covs[:,eBoundQoP, eBoundQoP] = var_inflation * std_qop**2

    if output_path:
        with open(output_path / "start_parameters.pickle", 'wb') as f:
            pickle.dump(dict(parameters=smeared_pars, covs=covs, smeared=True), f)
            logging.info("Wrote start data to '{}'".format(f.name))

    ##################
    # Fit & Analysis #
    ##################

    projector = np.array([[1, 0, 0]])

    kf = KalmanFitter(geometry, projector)
    gsf = GSF(geometry, projector, max_components=12, weight_cutoff=1.e-8, full_kl_divergence=False)

    def fit(fitter, name):
        fit_result = [
            fitter.fit(pars, cov, surfaces, measuements, len(surfaces)*[stddev_sim**2])
            for pars, cov, (measuements, _, surfaces) in zip(smeared_pars, covs, sim_results)
        ]

        if output_path:
            with open(output_path / "{}_fit_result.pickle".format(name.lower()), 'wb') as f:
                pickle.dump(dict(fit_result=fit_result), f)
                logging.info("Wrote fit data to '{}'".format(f.name))

        logging.info("Done with {} fitting".format(name))

    fit(kf, "KF")
    fit(gsf, "GSF")

    plt.show()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%H:%M:%S')

    plt.set_loglevel("warn")
    np.random.seed(12345)
    
    n_particles = 2000
    logging.info("2000 particles")

    output_path = Path("output/{}-{}particles".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), n_particles))
    output_path.mkdir(parents=True)
    test_pulls(n_particles, output_path)


