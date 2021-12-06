# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
from source.System import System
import time
import numpy as np

dim = 2
geo = "toggle_frame"
sys = System(dim, geo)

b = 0.753  # width
h = 0.243  # height
E = 1.0e7 # Young's modulus

sys.initialize_structure(beamtype="Bernoulli", youngs_modulus=E, poisson_ratio=0.499, width=b, height=h)
sys.initialize_with_plasticity(hardening_model="linear hardening", gauss_number=(2,2,2), yield_stress=3.0e3, plastic_modulus=1.0e7)

sys.add_dirichlet_bc(0, "xz")
sys.add_dirichlet_bc(8, "xz")
sys.add_load(4, force=np.array([[0., 1., 0.]]).T)
sys.define_interesting_dof(4, "z")

time_start = time.time()
sys.solve_the_system(solver="displacement-control", number_of_increments=100, displacement=-0.8, max_iteration_steps=1000)
time_end = time.time()

print("Time cost: " + str(time_end - time_start) + "s")
sys.output_equilibrium_path("toggle_frame_2d_elastoplas")
sys.plot_the_structure()
sys.plot_equilibrium_path(horizontal_flip=True, vertical_flip=True)