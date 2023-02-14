import logging
import pickle
import warnings
import json
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
from apyts.test_utils import *
import apyts.units as u


def run(args):
    if args["output_path"] is not None:
        output_path = Path(args["output_path"])
        assert output_path.exists()
    else:
        output_path = None

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

    logging.info("Simulation: smearing std = {}, radiation_loss = {}".format(args["simulation_std"], args["radiation_loss"]))
    simulation = Simulation(geometry, smearing_stddev=args["simulation_std"], simulate_radiation_loss=args["radiation_loss"])

    locs = np.full(args["n_particles"], 0*u.mm)
    phis = np.full(args["n_particles"], 90*u.degree)
    qops = -1./np.full(args["n_particles"], 4*u.GeV)
    logging.info("Fixed start parameters ({:.2f},{:.2f},{:.2f})".format(locs[0], phis[0], qops[0]))

    true_pars = np.stack((locs, phis, qops), axis=1)

    sim_results = [ simulation.simulate(pars, kElectronMass) for pars in true_pars ]

    if output_path:
        with open(output_path / "sim_result.pickle", 'wb') as f:
            pickle.dump(dict(
                true_pars=true_pars,
                sim_results=sim_results,
                stddev_sim=args["simulation_std"],
                simulate_rad_loss=args["radiation_loss"]
            ), f)
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
        std_loc * np.random.normal(0, 1, size=args["n_particles"]),
        std_phi * np.random.normal(0, 1, size=args["n_particles"]),
        std_qop * np.random.normal(0, 1, size=args["n_particles"]),
    ], axis=-1)

    smeared_pars = true_pars + smearing
    
    covs = np.zeros((args["n_particles"],3,3))
    covs[:,eBoundLoc, eBoundLoc] = std_loc**2
    covs[:,eBoundPhi, eBoundPhi] = std_phi**2
    covs[:,eBoundQoP, eBoundQoP] = std_qop**2

    if output_path:
        with open(output_path / "start_parameters.pickle", 'wb') as f:
            pickle.dump(dict(parameters=smeared_pars, covs=covs, smeared=True), f)
            logging.info("Wrote start data to '{}'".format(f.name))

    ##################
    # Fit & Analysis #
    ##################
    
    logging.info("variance inflation: {}".format(args["variance_inflation"]))
    
    covs = covs * args["variance_inflation"]

    projector = np.array([[1, 0, 0]])

    kf = KalmanFitter(geometry, projector)
    gsf = GSF(geometry, projector, max_components=12, weight_cutoff=1.e-8, full_kl_divergence=False)

    for fitter, name in zip([kf, gsf], ["KF", "GSF"]):
        fit_result = [
            fitter.fit(pars, cov, surfaces, measuements, len(surfaces)*[args["simulation_std"]**2])
            for pars, cov, (measuements, _, surfaces) in zip(smeared_pars, covs, sim_results)
        ]

        if output_path:
            with open(output_path / "{}_fit_result.pickle".format(name.lower()), 'wb') as f:
                pickle.dump(dict(fit_result=fit_result, var_inflation=args["variance_inflation"]), f)
                logging.info("Wrote fit data to '{}'".format(f.name))

        logging.info("Done with {} fitting".format(name))


if __name__ == "__main__":
    args = {
        "simulation_std": 0.01 * u.mm,
        "n_particles" : 2000,
        "radiation_loss": True,
        "variance_inflation": 100.0,
    }

    args["output_path"] = "output/{}-{}particles".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), args["n_particles"])
    Path(args["output_path"]).mkdir(parents=True, exist_ok=True)
    
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%H:%M:%S')

    pprint.pprint(args, indent=4)

    with open(Path(args["output_path"]) / "args.json", 'w') as f:
        json.dump(args, f, indent=4)

    run(args)


