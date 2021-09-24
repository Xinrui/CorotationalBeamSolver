# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
from source.CorotationalBeamElement import System

"""
--------------------------------
1. Define / Initialize variables
--------------------------------
"""
sys = System()
sys.dimension = 2
sys.geometry_name = "cyclic_loading"
sys.analysis = "linear hardening"
sys.solver = "displacement-control"

sys.number_of_load_increments = 20
sys.tolerance = 1e-3
sys.max_iteration_steps = 100

b = 1.0  # cross-sectional area
h = 1.0  # moment of inertia
E = 29.0e3 # Young's modulus
sys.initialize_structure(E, b, h)

if sys.analysis == "perfect plasticity":
    sigma_y = 3.0e4
    sys.initialize_with_plasticity(sigma_y)
elif sys.analysis == "linear hardening":
    sigma_y = 36.
    K = 500.
    sys.initialize_with_plasticity(sigma_y, K)

# sys.initialize_structure(E, A, I, I_z, I_t, G)
sys.add_dirichlet_bc(0, "fixed")
sys.add_load_bc(sys._number_of_nodes - 1, "x")
sys._interesting_dof = 3

if sys.solver == "load-control":
    sys.load_direction = "negative"
    sys.max_load = -10.
elif sys.solver == "displacement-control":
    sys.load_direction = "positive"
    sys.max_displacement = 0.1
elif sys.solver == "arc-length-control":
    sys.load_direction = "negative"
    sys.arc_length = 0.5

sys.solve_the_system()

if sys.solver == "displacement-control":
    sys.load_direction = "positive"
    sys.max_displacement = -0.2

sys.solve_the_system()

if sys.solver == "displacement-control":
    sys.load_direction = "positive"
    sys.max_displacement = 0.3

sys.solve_the_system()

if sys.solver == "displacement-control":
    sys.load_direction = "positive"
    sys.max_displacement = -0.4

sys.solve_the_system()

if sys.solver == "displacement-control":
    sys.load_direction = "positive"
    sys.max_displacement = 0.5

sys.solve_the_system()

if sys.solver == "displacement-control":
    sys.load_direction = "positive"
    sys.max_displacement = -0.6

sys.solve_the_system()

sys.plot_equilibrium_path(horizontal_flip=False, vertical_flip=False)
sys.plot_the_structure()