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
sys.geometry_name = "cyclic_loading"

b = 1.0  # cross-sectional area
h = 1.0  # moment of inertia
E = 29.0e3  # Young's modulus
G = E / 2.999  # Shear modulus

# G = 0.
sys.initialize_structure(youngs_modulus=E, shear_modulus=G, width=b, height=h)
# sys.initialize_structure(youngs_modulus=E, width=b, height=h)
sys.initialize_with_plasticity(
    analysis="linear hardening", yield_stress=36., plastic_modulus=500.)

sys.add_dirichlet_bc(0, "fixed")
sys.add_load_bc(sys._number_of_nodes - 1, "x")

time_start = time.time()

sys.solve_the_system(solver="displacement-control",
                     number_of_increments=20, displacement=0.1, max_iteration_steps=100)
sys.solve_the_system(solver="displacement-control",
                     number_of_increments=20, displacement=-0.2, max_iteration_steps=100)
sys.solve_the_system(solver="displacement-control",
                     number_of_increments=20, displacement=0.3, max_iteration_steps=100)
sys.solve_the_system(solver="displacement-control",
                     number_of_increments=20, displacement=-0.4, max_iteration_steps=100)
sys.solve_the_system(solver="displacement-control",
                     number_of_increments=20, displacement=0.5, max_iteration_steps=100)
sys.solve_the_system(solver="displacement-control",
                     number_of_increments=20, displacement=-0.6, max_iteration_steps=100)

time_end = time.time()

print("Time cost: " + str(time_end - time_start) + "s")

sys.plot_equilibrium_path()
sys.plot_the_structure()