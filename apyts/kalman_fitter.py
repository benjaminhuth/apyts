import numpy as np

from .geometry import *


def kalman_update(predicted : np.ndarray, predicted_cov : np.ndarray, measurement, measurement_cov, H : np.ndarray):
    # Make scalars to vectors/matrixes, so that the algorithms works
    if type(measurement) != np.ndarray:
        measurement = np.array([measurement])
    if type(measurement_cov) != np.ndarray:
        measurement_cov = np.array([[measurement_cov]])

    # Check shapes
    assert predicted.shape[0] == predicted_cov.shape[0] == predicted_cov.shape[1]
    assert measurement.shape[0] == measurement_cov.shape[0] == measurement_cov.shape[1]
    assert H.shape[0] == measurement.shape[0]
    assert H.shape[1] == predicted.shape[0]

    # Do update
    K = predicted_cov @ H.T @ np.linalg.inv(H @ predicted_cov @ H.T + measurement_cov)

    filtered = predicted + K @ (measurement - H @ predicted)
    filtered_cov = (np.eye(len(predicted)) - K @ H) @ predicted_cov

    return filtered, filtered_cov

class KalmanFitter:
    def __init__(self, geometry : Geometry, meas_projector : np.ndarray):
        self.geometry = geometry
        self.meas_projector = meas_projector

    def fit(self, pars : np.ndarray, cov : np.ndarray, surfaces : list, measurements : list, meas_covs : list):
        predicted_states = []
        filtered_states = []
        smoothed_states = []

        assert len(surfaces) == len(measurements) == len(meas_covs)

        data = [ (s, m, c) for s, m, c in zip(surfaces, measurements, meas_covs) ]

        # forward
        for meas_surface, meas_pars, meas_cov in data:
            prop_result = self.geometry.propagate(pars, from_surface=meas_surface-1, to_surface=meas_surface, cov=cov)

            if prop_result is None:
                return None

            predicted, predicted_cov = prop_result
            filtered, filtered_cov = kalman_update(predicted, predicted_cov, meas_pars, meas_cov, self.meas_projector)

            pars = filtered
            cov = filtered_cov

            predicted_states.append((predicted, predicted_cov))
            filtered_states.append((filtered, filtered_cov))

        # backward (cut last measurement, since we are already on that surface)
        reversed_data = reversed(data[:-1])

        # Skip first surfaces
        next(reversed_data)
        smoothed_states.append(filtered_states[-1])

        for meas_surface, meas_pars, meas_cov in reversed(data[:-1]):
            prop_result = self.geometry.propagate(pars, from_surface=meas_surface+1, to_surface=meas_surface, cov=cov)

            if prop_result is None:
                return None

            smoothed, smoothed_cov = kalman_update(prop_result[0], prop_result[1], meas_pars, meas_cov, self.meas_projector)
            pars, cov = smoothed, smoothed_cov

            smoothed_states.append((smoothed, smoothed_cov))

        # final parameters
        assert meas_surface == 1
        final, final_cov = self.geometry.propagate(pars, from_surface=1, to_surface=0, cov=cov)

        return (final, final_cov), predicted_states, filtered_states, reversed(smoothed_states)











