# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
from source.System import System
import time
import numpy as np

dim = 3
geo = "toggle_frame_rotated"
sys = System(dim, geo)

b = 0.753  # width
h = 0.243  # height
E = 1.0e7  # Young's modulus

sys.initialize_structure(beamtype="Timoshenko", youngs_modulus=E, 
                         width=b, height=h, e_2o=np.array([[-np.sqrt(2)/2, np.sqrt(2)/2, 0.]]).T)
sys.initialize_with_plasticity(hardening_model="linear hardening", gauss_number=(
    1, 2, 2), yield_stress=3.0e3, plastic_modulus=1.0e7)

sys.add_dirichlet_bc(0, "xyz")
sys.add_dirichlet_bc(8, "xyz")
sys.add_load(4, force=np.array([[0., 0., 1.]]).T)
sys.define_interesting_dof(4, "z")

time_start = time.time()
sys.solve_the_system(solver="displacement-control", number_of_increments=400,
                     displacement=-0.8, max_iteration_steps=10000)
time_end = time.time()

print("Time cost: " + str(time_end - time_start) + "s")
sys.output_equilibrium_path("toggle_frame_3d_elastoplas")
sys.plot_the_structure()
sys.plot_equilibrium_path(horizontal_flip=True, vertical_flip=True)
