from enum import Enum
import copy

import numpy as np
import scipy.stats as stats

from .geometry import *
from .kalman_fitter import kalman_update
from .gsf_utils import *
from .test_utils import *

def full_symmetric_kl_divergence(mu_a, cov_a, mu_b, cov_b):
    def kl_divergence(mu_a, cov_a, mu_b, cov_b):
        return np.trace(np.linalg.inv(cov_b) @ cov_a) \
            - len(mu_a) \
            + float((mu_b - mu_a).reshape(1,-1) @ cov_a @ (mu_b - mu_a).reshape(1,-1).transpose()) \
            + np.log(np.linalg.det(cov_b)/np.linalg.det(cov_a))

    return 0.5 * (kl_divergence(mu_a, cov_a, mu_b, cov_b) + kl_divergence(mu_b, cov_b, mu_a, cov_a))


def normalize_components(components : list):
    sum_w = 0
    for w, _, _ in components:
        sum_w += w

    for i in range(len(components)):
        w, pars, cov = components[i]
        components[i] = (w / sum_w, pars, cov)

    return components



class PropagationDirection(Enum):
    Forward = 1
    Backward = -1


class GSF:
    def __init__(self,
                 geometry : Geometry,
                 meas_projector : np.ndarray,
                 max_components : int,
                 weight_cutoff = 1.e-4,
                 disable_energy_loss=False,
                 full_kl_divergence=False,
                 single_component_approx=False,
    ):
        self.geometry = geometry
        self.projector = meas_projector
        self.max_components = max_components
        self.count = 0
        self.weight_cutoff = weight_cutoff
        self.disable_energy_loss = disable_energy_loss
        self.full_kl_divergence = full_kl_divergence
        self.single_component_approx = single_component_approx

        pass

    def kalman_update(self, predicted_components : list, measurement, meas_cov):
        # weight update
        for i in range(len(predicted_components)):
            w, predicted, predicted_cov = predicted_components[i]

            factor = stats.norm.pdf(float(measurement),
                                    float(self.projector @ predicted),
                                    np.sqrt(float(meas_cov + self.projector @ predicted_cov @ self.projector.T)))

            predicted_components[i] = (w*factor, predicted, predicted_cov)
            logging.debug("GSF | reweight {:.3f} -> {:.3f}   {}".format(w, w*factor, predicted))
            assert np.isfinite(predicted_components[i][0])

        predicted_components = normalize_components(predicted_components)
        predicted_components = [ c for c in predicted_components if c[0] > self.weight_cutoff ]

        # kalman update
        filtered_components = []

        for w, predicted, predicted_cov in predicted_components:
            filtered, filtered_cov = kalman_update(predicted, predicted_cov, measurement, meas_cov, self.projector)

            filtered_components.append((w, filtered, filtered_cov))
            logging.debug("GSF | prt: {}  ->  flt: {}".format(predicted, filtered))

        return filtered_components


    def apply_energy_loss(self, components : list, direction : PropagationDirection):
        new_components = []

        for w, mean, cov in components:
            p_initial = abs(1/mean[eBoundQoP])
            q = mean[eBoundQoP]/abs(mean[eBoundQoP])
            corrected_thickness = self.geometry.thickness_in_x0 / np.sin(mean[eBoundPhi])

            logging.debug("GSF | (w, p, v_inv_p):  {:.3f}, {:.3f}, {:.3f}".format(w, p_initial, cov[eBoundQoP, eBoundQoP]))

            for loss_w, loss_mean, loss_var in approx_bethe_heitler_distribution(corrected_thickness, self.single_component_approx):
                # loss_mean = p_f/p_i
                if direction == PropagationDirection.Forward:
                    new_p = p_initial * loss_mean
                    new_var_inv_p = loss_var / ((p_initial * loss_mean)**2)
                    assert new_p <= p_initial
                else:
                    new_p = p_initial / loss_mean
                    new_var_inv_p = loss_var / (p_initial**2)
                    assert new_p >= p_initial

                new_mean = mean.copy()
                new_mean[eBoundQoP] = q/new_p

                new_cov = cov.copy()
                new_cov[eBoundQoP, eBoundQoP] += new_var_inv_p

                new_components.append((w*loss_w, new_mean, new_cov))
                logging.debug("GSF |   ->  {:.3f}, {:.3f}, {:.3f}".format(w*loss_w, new_p, new_cov[eBoundQoP, eBoundQoP]))

        return new_components


    def reduce_components(self, components : list):
        # from scipy.stats import multivariate_normal
        
        # Just loc and qop
        # mask = np.array([True, False, True])
        
        # loc = [ m[eBoundLoc] for w, m, c in components ]
        # 
        # dists_start = [ (w, multivariate_normal(m[mask], c[:,mask][mask,:])) for w, m, c in components ]
        # fig, ax = plt.subplots()
        # x, y = np.mgrid[min(loc)-.5:max(loc)+.5:.1, -1.:0.5:.1]
        # ax.contourf(x, y, sum([ w*d.pdf(np.dstack((x, y))) for w, d in dists_start ]))
        
        # fig, ax = plot_mixtures_loc_phi_qop_p(components, draw_mode=False)
        # fig.suptitle("BEFORE ({} cmps)".format(len(components)))
        
        while len(components) > self.max_components:
            min_dist = np.inf

            for i in range(len(components)):
                for j in range(i+1, len(components)):
                    if self.full_kl_divergence:
                        _, mu_a, cov_a = components[i]
                        _, mu_b, cov_b = components[j]
                        dist = full_symmetric_kl_divergence(mu_a, cov_a, mu_b, cov_b)
                    else:
                        mu_a, var_a = components[i][1][eBoundQoP], components[i][2][eBoundQoP, eBoundQoP]
                        mu_b, var_b = components[j][1][eBoundQoP], components[j][2][eBoundQoP, eBoundQoP]
                        dist = var_a/var_b + var_b/var_a + (mu_a - mu_b) * (1/var_a + 1/var_b) * (mu_a - mu_b) - 2

                    if dist < min_dist:
                        min_dist = dist
                        min_pair = (i, j)

            i, j = min_pair
            new_mean, new_cov = gaussian_mixture_moments([components[i], components[j]])
            new_weight = components[i][0] + components[j][0]

            logging.debug("GSF | min kl dist: {:.5f}".format(min_dist))
            to_print = (components[i][1][eBoundQoP], components[i][2][eBoundQoP, eBoundQoP],
                        components[j][1][eBoundQoP], components[j][2][eBoundQoP, eBoundQoP],
                        new_mean[eBoundQoP], new_cov[eBoundQoP, eBoundQoP], min_dist)
            logging.debug("GSF | merge (qop, var_qop):  ({:.3f}, {:.3f})  ,  ({:.3f}, {:.3f})  ->  ({:.3f}, {:.3f})     [kl-dist: {:.3f}]".format(*to_print))

            components[i] = (new_weight, new_mean, new_cov)
            components.pop(j)

        assert len(components) <= self.max_components

        # fig, ax = plot_mixtures_loc_phi_qop_p(components, draw_mode=False)
        # fig.suptitle("AFTER ({} cmps)".format(len(components)))
        # plt.show()

        return components


    def propagation_loop(self, components, measurement_data, direction):
        direction_str = "forward" if direction == PropagationDirection.Forward else "backward"
        from_surface_modifier = -1 if direction == PropagationDirection.Forward else +1

        predicted_states = []
        filtered_states = []

        for meas_surface, meas_pars, meas_cov in measurement_data:
            prop_results = [ (w, self.geometry.propagate(p, from_surface=meas_surface+from_surface_modifier, to_surface=meas_surface, cov=c)) for w, p, c in components ]
            components = [ (w, *res) for w, res in prop_results if res is not None ]

            if len(components) == 0:
                logging.debug("GSF | {} propagation failed".format(direction_str))
                return None

            logging.debug("GSF | {} step parameter: {} ({} cmps)".format(direction_str, gaussian_mixture_moments(components)[0], len(components)))
            predicted_states.append(copy.deepcopy(components))

            components = self.kalman_update(components, meas_pars, meas_cov)
            filtered_states.append(copy.deepcopy(components))

            if not self.disable_energy_loss:
                components = self.apply_energy_loss(components, direction)
                components = self.reduce_components(components)

            assert (sum([ w for w, _, _ in components]) - 1.0) < 1.e-8

        if direction == PropagationDirection.Backward:
            assert meas_surface == 1

        return components, predicted_states, filtered_states


    def fit(self, pars : np.ndarray, cov : np.ndarray, surfaces : list, measurements : list, meas_covs : list):
        assert len(surfaces) == len(measurements) == len(meas_covs)

        logging.info("GSF | start fit #{}".format(self.count))
        self.count += 1

        data = [ (s, m, c) for s, m, c in zip(surfaces, measurements, meas_covs) ]
        if len(data) < 3:
            logging.debug("GSF | No fit for less than 3 hits")
            return None

        # Forward
        components = [(1.0, pars, cov)]
        prop_res = self.propagation_loop(components, data, PropagationDirection.Forward)
        if prop_res is None:
            return None

        components, predicted_states, filtered_states = prop_res

        if len(components) == 0:
            logging.warn("GSF | No components, should not happen")
            return None

        # Backward
        prop_res = self.propagation_loop(components, reversed(data[:-1]), PropagationDirection.Backward)
        if prop_res is None:
            return None

        components, _, smoothed_states = prop_res
        smoothed_states = [ filtered_states[-1] ] + smoothed_states

        # Final parameters
        prop_results = [ (w, self.geometry.propagate(p, from_surface=1, to_surface=0, cov=c)) for w, p, c in components ]
        components = [ (w, *res) for w, res in prop_results if res is not None ]

        if len(components) == 0:
            logging.debug("GSF | final propagation failed")
            return None

        logging.debug("GSF | final parameter: {}".format(gaussian_mixture_moments(components)[0]))
        return tuple(components), tuple(predicted_states), tuple(filtered_states), tuple(reversed(smoothed_states))






























