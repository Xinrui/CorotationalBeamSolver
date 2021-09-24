# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
from source.CorotationalBeamElement import System
import time
"""
--------------------------------
1. Define / Initialize variables
--------------------------------
"""
sys = System()
sys.dimension = 3
sys.geometry_name = "TipLoad2D"

b = 2.0  # cross-sectional area
h = 2.0  # moment of inertia
E = 100. # Young's modulus
G = 1. # Shear modulus

sys.initialize_structure(youngs_modulus=E, shear_modulus=G, width=b, height=h)
# sys.initialize_with_plasticity(analysis="linear hardening", yield_stress=5.0, plastic_modulus=50.0)

sys.add_dirichlet_bc(0, "fixed")
sys.add_load_bc(sys._number_of_nodes - 1, "y")

time_start = time.time()
# sys.solve_the_system(solver="load-control", number_of_increments=20, load=-10, max_iteration_steps=2000)
# sys.solve_the_system(solver="displacement-control", number_of_increments=20, displacement=8.0, max_iteration_steps=100)
sys.solve_the_system(solver="arc-length-control", number_of_increments=20, direction="negative", arc_length=0.5)
time_end = time.time()

print("Time cost: " + str(time_end - time_start) + "s")

sys.plot_equilibrium_path(horizontal_flip=False, vertical_flip=False)
sys.plot_the_structure()
