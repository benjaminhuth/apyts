import numpy as np

from apyts.geometry import *
import apyts.units as u

def test_propagation(pars, from_surface, to_surface, fig_and_ax):
    geometry = Geometry(
        surfaces=[0,50],
        surface_radius=100,
        thickness_in_x0=1/kSiRadiationLength,
        b_field = 2*u.Tesla,
        limit_plot_to_detector=True,
        fig_and_ax=fig_and_ax,
    )

    geometry.draw_surfaces()
    cov = np.eye(3)

    print("old pars:",pars)
    print("old cov:\n", cov)

    geometry.draw_circle(pars, from_surface)
    geometry.draw_local_params(pars, from_surface)

    test_offset = 1

    geometry.ax.text(geometry.surfaces[from_surface]+test_offset, pars[eBoundLoc]+test_offset, "from")

    new_local_pars, new_cov = geometry.propagate(pars, from_surface=from_surface, to_surface=to_surface, cov=cov)
    print("new pars:",new_local_pars)
    print("new cov:\n",new_cov)
    geometry.draw_local_params(new_local_pars, to_surface)
    geometry.ax.text(geometry.surfaces[to_surface]+test_offset, new_local_pars[eBoundLoc]+test_offset, "to")
    print()


# Test forward
fig, axes = plt.subplots(2,2)

directions = []
charges = []
for d  in ["forward", "backward"]:
    for q in [-1., 1.]:
        directions.append(d)
        charges.append(q)

for direction, q, ax in zip(directions, charges, axes.flatten()):
    if direction == "forward":
        from_surface = 0
        to_surface = 1
        loc0 = -80
    else:
        from_surface = 1
        to_surface = 0
        loc0 = 80

    pars = np.array([
        loc0,
        0.15*np.pi, # phi
        q / (0.3*u.GeV), # qop
    ])

    ax.set_title("q = {}, direction={}".format(q, direction))
    print("q = {}, direction = {}".format(q, direction))
    print("==========")
    test_propagation(pars, from_surface, to_surface, (fig, ax))
    print("")

fig.tight_layout()


plt.show()

