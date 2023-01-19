import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from apyts.geometry import *
from apyts.simulation import *
from apyts.kalman_fitter import *
from apyts.constants import *
import apyts.units as u

from test_utils import plot_hists

def test_pulls():
    logging.info("Start")
    geometry = Geometry([0, 100, 200, 300, 400, 500],
                        surface_radius=200 * u.mm,
                        b_field=2 * u.Tesla,
                        no_plot=True)

    n_particles = 1000

    ############
    # Simulate #
    ############

    std_simulation = 0.1 * u.mm
    simulation = Simulation(geometry, smearing_stddev=std_simulation, simulate_radiation_loss=False)

    charges = np.random.choice([-1, 1], size=n_particles)
    momenta = np.random.uniform(2.0*u.GeV, 4.0*u.GeV, size=n_particles)
    phis = np.random.uniform(60*u.degree, 120*u.degree, size=n_particles)
    locs = np.random.uniform(-10*u.mm, 10*u.mm, size=n_particles)
    qops = charges / momenta

    true_pars = np.stack((locs, phis, qops), axis=1)

    sim_data = [ simulation.simulate(pars, kElectronMass) for pars in true_pars ]
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

    #######
    # Fit #
    #######

    projector = np.array([[1, 0, 0]])
    kf = KalmanFitter(geometry, projector)

    fit_data = [ kf.fit(pars, cov, surfaces, measuements, len(surfaces)*[std_simulation**2]) for pars, cov, (measuements, _, surfaces) in zip(smeared_pars, covs, sim_data) ]
    logging.info("Done with fitting")
    mask = np.array([ False if data is None else True for data in fit_data ])

    logging.info("fit failed for {}/{} ({:.2f}%) of particles".format(sum(np.logical_not(mask)), len(mask), 100.*sum(np.logical_not(mask))/len(mask)))

    ###########
    # Analyse #
    ###########

    # Make residuals
    final_pars = np.stack([ data[0][0] for data in fit_data if data is not None ])
    
    residuals = pd.DataFrame() 
    residuals[["LOC","PHI","QOP"]] = true_pars[mask] - final_pars

    fig, ax = plot_hists(residuals, bins="rice")
    fig.suptitle("Residuals")

    # Make pulls
    final_covs = np.stack([ data[0][1] for data in fit_data if data is not None ])
    final_variances = np.diagonal(final_covs, axis1=1, axis2=2)

    # print(final_variances)

    pulls = residuals / np.sqrt(final_variances)
    fig, ax = plot_hists(pulls, fit_gaussian=True, bins="rice")
    fig.suptitle("Pulls")

    plt.show()




logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%H:%M:%S')

test_pulls()

