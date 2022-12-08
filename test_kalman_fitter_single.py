import numpy as np
import matplotlib.pyplot as plt

from apyts.geometry import *
from apyts.kalman_fitter import *
from apyts.simulation import *
from apyts.constants import *
import apyts.units as u


def test_single_particle():
    geometry = Geometry([0, 100, 200, 300, 400, 500],
                        surface_radius=50,
                        b_field=2*u.Tesla)

    geometry.draw_surfaces()


    # Simulate
    smearing_stddev = 1
    simulation = Simulation(geometry, smearing_stddev=smearing_stddev, simulate_radiation_loss=False)


    true_pars = np.array([
        0, # loc0
        0.5*np.pi, # phi
        -1 / (4*u.GeV) # qop
    ])

    measurements, truth, surfaces = simulation.simulate(true_pars, kElectronMass)

    for pars, surface_id in zip(truth, range(1, len(truth)+1)):
        geometry.draw_circle(pars, surface_id)
        break
        # geometry.draw_local_params(pars, surface_id)

    geometry.ax.scatter(geometry.surfaces[1:len(measurements)+1], measurements, color='black', marker="x", s=100)


    # Fit
    projector = np.array([[1, 0, 0]])
    measument_variances = len(measurements) * [ smearing_stddev**2 ]

    std_loc0 = 20*u.um
    std_phi = 1*u.degree
    std_qop = 0.05 / 4

    start_cov = np.zeros((3,3))
    np.fill_diagonal(start_cov, [std_loc0**2, std_phi**2, std_qop**2])

    start_pars = np.array([
        true_pars[eBoundLoc] + np.random.normal(0, std_loc0),
        true_pars[eBoundPhi] + np.random.normal(0, std_phi),
        true_pars[eBoundLoc] + np.random.normal(0, std_qop)
    ])
    start_cov *= 100

    kf = KalmanFitter(geometry, projector)

    final, predicted, filtered, smoothed = kf.fit(start_pars, start_cov, surfaces, measurements, len(surfaces)*[ smearing_stddev ])
    final_pars, final_cov = final

    geometry.draw_local_params(start_pars, 0, label="start pars")
    geometry.draw_local_params(final_pars, 0, label="final pars")

    geometry.ax.scatter([ geometry.surfaces[surface_id] for surface_id in surfaces],
                        [ pars[eBoundLoc] for pars, cov in predicted], color='red', marker="^", s=100, label="predicted")
    geometry.ax.scatter([ geometry.surfaces[surface_id] for surface_id in surfaces],
                        [ pars[eBoundLoc] for pars, cov in filtered], color='blue', marker="v", s=100, label="filtered")
    geometry.ax.scatter([ geometry.surfaces[surface_id] for surface_id in surfaces],
                        [ pars[eBoundLoc] for pars, cov in smoothed], color='green', marker=">", s=100, label="smoothed")

    geometry.ax.legend()

    fig, axes = plt.subplots(1,3)
    fig.suptitle("Variance evolution")
    for ax, idx, name in zip(axes, [eBoundLoc, eBoundPhi, eBoundQoP], ["LOC", "PHI", "QOP"]):
        fwd = [ cov[idx, idx] for _, cov in filtered]
        bwd = [ cov[idx, idx] for _, cov in smoothed] + [final_cov[idx, idx]]
        ax.plot(fwd + bwd, label="flt/smt")

        fwd = [ cov[idx, idx] for _, cov in predicted]
        ax.plot(fwd, label="prt")

        ax.vlines([len(fwd), len(fwd)+len(bwd)-2], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black')
        ax.set_title(name)
        # ax.set_yscale('log')
        ax.legend()

    plt.show()

    plt.show()



if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%H:%M:%S')

    plt.set_loglevel("info")

    np.random.seed(12345)
    np.set_printoptions(precision=5, suppress=True)

    test_single_particle()





