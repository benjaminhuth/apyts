'''
Python GSF
'''
from enum import Enum
import logging

import numpy as np
import matplotlib.pyplot as plt

import autograd.numpy as anp
import autograd

from .constants import *


def min_angle_diff(alpha, beta):
    a = anp.array([anp.cos(alpha), anp.sin(alpha)])
    b = anp.array([anp.cos(beta), anp.sin(beta)])
    return anp.arccos(anp.dot(a,b))

class Geometry:
    def __init__(self, surfaces, surface_radius, b_field, thickness_in_x0=1/kSiRadiationLength, limit_plot_to_detector=True, fig_and_ax=None, no_plot=False):
        self.surfaces = surfaces
        self.surface_radius = surface_radius
        self.thickness_in_x0 = thickness_in_x0
        self.b_field = b_field
        self.no_plot = no_plot

        if not no_plot:
            if fig_and_ax is None:
                self.fig, self.ax = plt.subplots()
            else:
                self.fig, self.ax = fig_and_ax[0], fig_and_ax[1]

            detector_length = surfaces[-1] - surfaces[0]

            if limit_plot_to_detector:
                self.ax.set_xlim(surfaces[0] - 0.1*detector_length, surfaces[-1] + 0.1*detector_length)
                self.ax.set_ylim(-1.1*surface_radius, 1.1*surface_radius)



    def draw_surfaces(self):
        if self.no_plot:
            return

        self.ax.vlines(self.surfaces, ymin=-self.surface_radius, ymax=self.surface_radius, color='black', lw=2, zorder=-10)

    def draw_local_params(self, loc_params, surface_id, label=None):
        if self.no_plot:
            return

        point = np.array([self.surfaces[surface_id], loc_params[eBoundLoc]])
        l = 10
        dir = np.array([l*np.sin(loc_params[eBoundPhi]), l*np.cos(loc_params[eBoundPhi])])
        line_plots = self.ax.plot([point[0], point[0]+dir[0]], [point[1], point[1]+dir[1]])
        self.ax.scatter([point[0]], point[1], color=line_plots[0]._color, label=label)

    def compute_circle(self, loc_params, surface_id):
        q_sign = loc_params[eBoundQoP] / abs(loc_params[eBoundQoP])

        r = abs(1. / (self.b_field * loc_params[eBoundQoP]))

        M = [
            self.surfaces[surface_id] - q_sign * r * anp.cos(loc_params[eBoundPhi]),
            loc_params[eBoundLoc]     + q_sign * r * anp.sin(loc_params[eBoundPhi]),
        ]

        return M, r

    def draw_circle(self, loc_params, surface_id):
        if self.no_plot:
            return

        M, r = self.compute_circle(loc_params, surface_id)
        self.ax.scatter([M[0]], [M[1]], c='grey', s=0.5)
        self.ax.add_patch(plt.Circle(M, r, fill=False, linestyle='--', color='grey'))

    def propagate_params(self, loc_params, from_surface, to_surface):
        assert from_surface >= 0 and from_surface < len(self.surfaces)
        assert to_surface >= 0 and from_surface < len(self.surfaces)

        M, r = self.compute_circle(loc_params, from_surface)

        # Equation:
        # x_new = M_x + r cos phi_new
        # y_new = M_y - r sin phi_new
        # x_new, M and r are fixed -> solve for phi_new and y_new

        new_x = self.surfaces[to_surface]

        if M[0] + r < new_x:
            logging.debug("Will not hit next surface, radius to small")
            return None

        # Two solutions for the angle phi (cos is symmetric)
        new_phi = anp.arccos( (new_x - M[0]) / r )

        # Different charges correspond to different direction of rotation
        # Multiply pathlengths by sign to model this
        q_sign = loc_params[eBoundQoP] / abs(loc_params[eBoundQoP])

        pathlenght_a = q_sign * r * min_angle_diff(loc_params[eBoundPhi], new_phi)
        pathlenght_b = q_sign * r * min_angle_diff(loc_params[eBoundPhi], -new_phi)

        if pathlenght_a > pathlenght_b:
            new_y = M[1] - r * anp.sin(-new_phi)
            new_phi = np.pi - new_phi
            pathlength = pathlenght_b
        else:
            new_y = M[1] - r * anp.sin(new_phi)
            pathlength = pathlenght_a

        return anp.array([new_y, new_phi, loc_params[eBoundQoP]])

    def propagate(self, params, from_surface, to_surface, cov=None):
        def f(x):
            return self.propagate_params(x, from_surface, to_surface)

        new_params = f(params)

        if new_params is None:
            return None

        if new_params[eBoundLoc] > self.surface_radius or new_params[eBoundLoc] < -self.surface_radius:
            logging.debug("propagation state not on surface")
            return None

        if cov is None:
            return new_params
        else:
            jacobian = autograd.jacobian(f)(params)
            # logging.debug("Jacobian = \n{}".format(jacobian))

            new_cov = jacobian @ cov @ jacobian.T
            assert np.all(np.linalg.eigvals(new_cov) > 0)

            return new_params, new_cov



