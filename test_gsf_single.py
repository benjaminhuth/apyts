import pickle

import numpy as np
import matplotlib.pyplot as plt

from apyts.geometry import *
from apyts.gsf import *
from apyts.gsf_utils import *
from apyts.simulation import *
from apyts.constants import *
import apyts.units as u

from test_utils import *


def test_single_particle():
    geometry = Geometry([0, 100, 200, 300, 400, 500, 600, 700],
                        surface_radius=60,
                        b_field=2*u.Tesla)

    smearing_stddev = 0.01
    simulation = Simulation(geometry, smearing_stddev=smearing_stddev, simulate_radiation_loss=True)

    # true_pars = np.array([ 0*u.mm, 90*u.degree, -1 / (4*u.GeV) ])
    # true_pars = np.array([2.13350432, 1.68005464, 0.32533483])
    true_pars = np.array([2.13350432, 1.68005464, 0.25])

    logging.info("True pars: {}".format(true_pars))

    measurements, truth, surfaces = simulation.simulate(true_pars, kElectronMass) #, force_energy_loss=(3,0.4))
    measument_variances = len(measurements) * [ smearing_stddev**2 ]

    std_loc0 = 20*u.um
    std_phi = 1*u.degree
    std_qop = 0.05 / 4

    start_cov = np.zeros((3,3))
    np.fill_diagonal(start_cov, [std_loc0**2, std_phi**2, std_qop**2])
    start_cov *= 100

    start_pars = np.array([
        true_pars[eBoundLoc] + np.random.normal(0, std_loc0),
        true_pars[eBoundPhi] + np.random.normal(0, std_phi),
        true_pars[eBoundQoP] + np.random.normal(0, std_qop)
    ])
    start_pars = np.array([2.82482606, 1.67006317, 0.27])
    logging.info("Start pars: {}".format(start_pars))

    geometry.draw_surfaces()
    geometry.ax.scatter(geometry.surfaces[1:len(measurements)+1], measurements, color='black', marker="x", s=100)

    for pars, surface_id in zip(truth, range(1, len(truth)+1)):
        geometry.draw_circle(pars, surface_id)
        break

    energy_loss = abs(1./truth[-1][eBoundQoP]) - abs(1./true_pars[eBoundQoP])
    logging.info("Energy loss: {:.5f}".format(energy_loss))

    projector = np.array([[1, 0, 0]])
    gsf = GSF(geometry, projector, max_components=6, weight_cutoff=1.e-2, disable_energy_loss=False, full_kl_divergence=False, single_component_approx=False)

    components, predicted, filtered, smoothed = gsf.fit(start_pars, start_cov, surfaces, measurements, len(surfaces)*[ smearing_stddev ])
    #final_pars, final_cov = merge_components(components)
    _, final_pars, final_cov = copy.deepcopy(max(components, key=lambda c: c[0]))
    # final_pars = gaussian_mixture_mode(components)[0][0]

    logging.info("Res momentum: {}".format(abs(1./final_pars[eBoundQoP]) - abs(1./true_pars[eBoundQoP])))

    geometry.draw_local_params(start_pars, 0, label="start pars")
    geometry.draw_local_params(final_pars, 0, label="final pars")

    geometry.ax.scatter([ geometry.surfaces[surface_id] for surface_id in surfaces],
                        [ gaussian_mixture_moments(cmps)[0][eBoundLoc] for cmps in predicted],
                        color='red', marker="^", s=100, label="predicted mean")
    geometry.ax.scatter([ geometry.surfaces[surface_id] for surface_id in surfaces],
                        [ gaussian_mixture_moments(cmps)[0][eBoundLoc] for cmps in filtered],
                        color='blue', marker="v", s=100, label="filtered mean")
    geometry.ax.scatter([ geometry.surfaces[surface_id] for surface_id in surfaces],
                        [ gaussian_mixture_moments(cmps)[0][eBoundLoc] for cmps in smoothed],
                        color='green', marker=">", s=100, label="smoothed mean")

    geometry.ax.legend()

    for i, (w, p, c) in enumerate(components):
        to_print = (i, w, *p, c[0,0], c[1,1], c[2,2])
        logging.info("#{:2}  |  w: {:.2f}  |  mu:  {:.2f}  {:.2f}  {:.2f}  |  var:  {:.2f}  {:.2f}  {:.2f}".format(*to_print))

    fig, ax = plot_mixture(copy.deepcopy(components), true_pars)

    fig, ax = plt.subplots()

    ax.set_title("Momentum")
    def collect(states, surface_id_modifier):
        x, y, alpha = [], [], []
        for surface_id, state in enumerate(states):
            for w, pars, _ in state:
                x.append(geometry.surfaces[surface_id+surface_id_modifier])
                y.append(abs(1./pars[eBoundQoP]))
                alpha.append(max(0.4, w))
        return x, y, alpha

    x_flt, y_flt, alpha_flt = collect(filtered, +1)
    x_smt, y_smt, alpha_smt = collect([components] + list(smoothed), 0)
    ax.scatter(x_flt,y_flt, alpha=alpha_flt, label="fwd/filtered cmps")
    ax.plot(geometry.surfaces,
            [abs(1./start_pars[eBoundQoP])] + [ abs(1./gaussian_mixture_moments(state)[0][eBoundQoP]) for state in filtered ],
            label="fwd/filtered mean")
    ax.scatter(-1*np.array(x_smt), y_smt, alpha=alpha_smt, label="bwd/smoothed cmps")
    ax.plot(-1*np.array(geometry.surfaces),
            [abs(1./final_pars[eBoundQoP])] + [ abs(1./gaussian_mixture_moments(state)[0][eBoundQoP]) for state in smoothed ],
            label="bwd/smoothed mean")
    ax.plot(geometry.surfaces, [abs(1./true_pars[eBoundQoP])] + [ abs(1./t[eBoundQoP]) for t in truth ], ls="--", color="tab:red")
    ax.plot(-1*np.array(geometry.surfaces), [abs(1./true_pars[eBoundQoP])] + [ abs(1./t[eBoundQoP]) for t in truth ], ls="--", color="tab:red")

    ax.set_ylim(0,6)
    ax.vlines(0, color='black', ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1])

    ax.legend()

    plt.show()




if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s  %(levelname)s  %(message)s', datefmt='%H:%M:%S')

    plt.set_loglevel("info")

    # seed = np.random.randint(0, 2**32-1, 1)
    seed = 4235113935
    np.random.seed(seed)

    np.set_printoptions(precision=3, suppress=True)

    test_single_particle()






