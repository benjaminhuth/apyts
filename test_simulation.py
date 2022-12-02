import numpy as np

from geometry import *
from simulation import *

geometry = Geometry(
    surfaces=[0,50,100,150,200,250,300],
    surface_radius=100,
    thickness_in_x0=1/kSiRadiationLength,
    b_field = 2*kTesla
)

simulation = Simulation(geometry, smearing_stddev=1, simulate_radiation_loss=True)

geometry.draw_surfaces()

start_pars = np.array([
    0, # loc0
    0.5*np.pi, # phi
    -1 / (2*kGeV) # qop
])

measurements, truth = simulation.simulate(start_pars, kElectronMass)

print("Measurements", measurements)
for pars, surface_id in zip(truth, range(1, len(truth)+1)):
    geometry.draw_circle(pars, surface_id)
    geometry.draw_local_params(pars, surface_id)

geometry.ax.scatter(geometry.surfaces[1:len(measurements)+1], measurements, color='black', marker="x", s=100)


plt.show()
