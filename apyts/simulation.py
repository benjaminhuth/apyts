import numpy as np

from .geometry import *


class Simulation:
    def __init__(self, geometry : Geometry, smearing_stddev, simulate_radiation_loss):
        self.geometry = geometry
        self.smearing_stddev = smearing_stddev
        self.simulate_radiation_loss = simulate_radiation_loss

    def bethe_heitler_loss(self, p, m):
        energy_initial = np.sqrt(p**2 + m**2)
        u = np.random.gamma(self.geometry.thickness_in_x0 / np.log(2), 1.0, 1)
        z = np.exp(-u)
        e_final = energy_initial * z
        return np.sqrt(e_final**2 - m**2)

    def simulate(self, pars, mass):
        measurements = []
        truth = []
        surfaces = []

        for surface_id in range(len(self.geometry.surfaces)-1):
            pars = self.geometry.propagate(pars, from_surface=surface_id, to_surface=surface_id+1)

            if pars is None:
                break

            surfaces.append(surface_id+1)
            truth.append(pars)
            measurements.append(pars[eBoundLoc] + np.random.normal(0, self.smearing_stddev))

            if self.simulate_radiation_loss:
                q = pars[eBoundQoP] / abs(pars[eBoundQoP])
                new_p = self.bethe_heitler_loss(abs(1./pars[eBoundQoP]), mass)
                pars[eBoundQoP] = q/new_p

        return measurements, truth, surfaces



