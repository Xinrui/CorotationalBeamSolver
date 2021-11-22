# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
from source.System import System
import time

dim = 3
geo = "toggle_frame"
sys = System(dim, geo)

b = 0.243  # width
h = 0.753  # height
E = 1.0e7 # Young's modulus

# nu = 0.3: secondary path

sys.initialize_structure(beamtype="Timoshenko", youngs_modulus=E, poisson_ratio=0.499, width=b, height=h)
sys.initialize_with_plasticity(hardening_model="linear hardening", gauss_number=(1,2,2), yield_stress=3.0e3, plastic_modulus=1.0e7)

sys.add_dirichlet_bc(0, "xyz")
sys.add_dirichlet_bc(8, "xyz")
sys.add_load_bc(4, "y")

time_start = time.time()
sys.solve_the_system(solver="displacement-control", number_of_increments=100, displacement=-0.8, max_iteration_steps=1000)
time_end = time.time()

print("Time cost: " + str(time_end - time_start) + "s")
sys.output_equilibrium_path("toggle_frame_3d_elastoplas")
sys.plot_the_structure()
sys.plot_equilibrium_path(horizontal_flip=True, vertical_flip=True)