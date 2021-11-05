# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
from source.System import System
import time
"""
--------------------------------
1. Define / Initialize variables
--------------------------------
"""
sys = System()
sys.dimension = 2
sys.geometry_name = "cantilever_beam"

b = 0.1  # cross-sectional area
h = 0.5  # moment of inertia
E = 3.0e7 # Young's modulus
G = 11538461.538461538 # Shear modulus

sys.initialize_structure(youngs_modulus=E, shear_modulus=G, width=b, height=h)
# sys.initialize_with_plasticity(analysis="linear hardening", yield_stress=3.0e4, plastic_modulus=E/29)

sys.add_dirichlet_bc(0, "fixed")
sys.add_load_bc(sys._number_of_nodes - 1, "y")

time_start = time.time()
# sys.solve_the_system(solver="load-control", number_of_increments = 1000, load=-1400, max_iteration_steps=2000)
sys.solve_the_system(solver="displacement-control", number_of_increments=50, displacement=4.4, max_iteration_steps=100)
# sys.solve_the_system(solver="arc-length-control", number_of_increments=20, direction="negative", arc_length=0.5)
time_end = time.time()

print("Time cost: " + str(time_end - time_start) + "s")

sys.plot_equilibrium_path(horizontal_flip=False, vertical_flip=False)
sys.plot_the_structure()
