# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
from source.System import System
import time
import numpy as np

dim = 3
geometry = "lees_frame_rotated"
sys = System(dim, geometry)

b = 3  # width
h = 2  # height
E = 720  # Young's modulus
nu = 0.3  # Poisson ratio

sys.initialize_structure(beamtype="Timoshenko",
                         youngs_modulus=E, poisson_ratio=nu, width=b, height=h, e_2o=np.array([[-np.sqrt(2)/2, np.sqrt(2)/2, 0.]]).T)
sys.initialize_with_plasticity(hardening_model="linear hardening", gauss_number=(
    1, 2, 2), yield_stress=10.44, plastic_modulus=80)

sys.add_dirichlet_bc(0, "xyz")
sys.add_dirichlet_bc(20, "xyz")
sys.add_load(12, force=np.array([[0., 0., 1.]]).T)
sys.define_interesting_dof(12, "z")

time_start = time.time()
sys.solve_the_system(solver="arc-length-control", number_of_increments=100,
                     direction="negative", max_iteration_steps=10000, arc_length=1.0)
time_end = time.time()

print("Time cost: " + str(time_end - time_start) + "s")
sys.output_equilibrium_path("Lees_frame_3d_elastoplastic_rot")
sys.plot_the_structure()
sys.plot_equilibrium_path(horizontal_flip=True, vertical_flip=True)
