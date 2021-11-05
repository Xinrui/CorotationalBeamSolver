# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
from source.CorotationalBeamElement import System
import time
import math
"""
--------------------------------
1. Define / Initialize variables
--------------------------------
"""
sys = System()
sys.dimension = 2
sys.geometry_name = "TipLoad2D"

b = 2.0  # cross-sectional area
h = 2.0  # moment of inertia
E = 100. # Young's modulus
G = 11538461.538461538 # Shear modulus

sys.initialize_structure(youngs_modulus=E, shear_modulus=G, width=b, height=h)
# sys.initialize_with_plasticity(analysis="linear hardening", yield_stress=3.0e4, plastic_modulus=E/29)

sys.add_dirichlet_bc(0, "fixed")
sys.add_load_bc(sys._number_of_nodes - 1, "m")

time_start = time.time()
sys.solve_the_system(solver="load-control", number_of_increments=100, load=2*math.pi*E*1.3333/10)
# sys.solve_the_system(solver="displacement-control", number_of_increments=20, displacement=8.0, max_iteration_steps=100)
# sys.solve_the_system(solver="arc-length-control", number_of_increments=20, direction="negative", arc_length=0.5)
time_end = time.time()

print("Time cost: " + str(time_end - time_start) + "s")

sys.plot_equilibrium_path(horizontal_flip=False, vertical_flip=False)
sys.plot_the_structure()
