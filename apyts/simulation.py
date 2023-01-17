import numpy as np

from .geometry import *


class Simulation:
    def __init__(self, geometry : Geometry, smearing_stddev, simulate_radiation_loss):
        self.geometry = geometry
        self.smearing_stddev = smearing_stddev
        self.simulate_radiation_loss = simulate_radiation_loss

    def bethe_heitler_loss(self, thickness_in_x0, p, m):
        energy_initial = np.sqrt(p**2 + m**2)
        u = np.random.gamma(thickness_in_x0 / np.log(2), 1.0, 1)
        z = np.exp(-u)
        e_final = energy_initial * z
        return np.sqrt(e_final**2 - m**2)

    def simulate(self, pars, mass, force_energy_loss=None):
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

            if force_energy_loss is not None:
                loss_surface_id, loss_ratio = force_energy_loss
                if surface_id == loss_surface_id:
                    q = pars[eBoundQoP] / abs(pars[eBoundQoP])
                    new_p = loss_ratio * abs(1./pars[eBoundQoP])
                    pars[eBoundQoP] = q/new_p
            elif self.simulate_radiation_loss:
                corrected_thickness = self.geometry.thickness_in_x0 / np.sin(pars[eBoundPhi])
                q = pars[eBoundQoP] / abs(pars[eBoundQoP])
                new_p = self.bethe_heitler_loss(corrected_thickness, abs(1./pars[eBoundQoP]), mass)
                pars[eBoundQoP] = q/new_p

        return tuple(measurements), tuple(truth), tuple(surfaces)



