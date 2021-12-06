# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
from source.System import System
import time
import numpy as np

dim = 3
geometry = "cantilever_beam_elastoplastic_3D"
sys = System(dim, geometry)

b = 0.1  # width
h = 0.5  # height
E = 3e7  # Young's modulus

sys.initialize_structure(beamtype="Timoshenko",
                         youngs_modulus=E, width=b, height=h, e_3o=np.array([[0., 0., 1.]]).T)
sys.initialize_with_plasticity(hardening_model="linear hardening", gauss_number=(
    1, 2, 2), yield_stress=3e4, plastic_modulus=E/29)

sys.add_dirichlet_bc(0, "fixed")
sys.add_load(10, force=np.array([[0., 0., 1.]]).T)
sys.define_interesting_dof(10, "z")

time_start = time.time()
sys.solve_the_system(solver="load-control",
                     number_of_increments=500, load=-1200, max_iteration_steps=1000)
time_end = time.time()

print("Time cost: " + str(time_end - time_start) + "s")
sys.output_equilibrium_path("cantilever_beam_3d_elastoplastic")
sys.plot_the_structure()
sys.plot_equilibrium_path(horizontal_flip=True, vertical_flip=True)
