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
sys.geometry_name = "TipLoad2D"
sys.analysis = "elastic"
sys.solver = "displacement-control"

b = 2.0  # cross-sectional area
h = 2.0  # moment of inertia
E = 100. # Young's modulus
sys.initialize_structure(E, b, h)

if sys.analysis == "perfect plasticity":
    sigma_y = 3.0e4
    sys.initialize_with_plasticity(sigma_y)
elif sys.analysis == "linear hardening":
    sigma_y = 3.0e4
    K = E / 29
    sys.initialize_with_plasticity(sigma_y, K)

# sys.initialize_structure(E, A, I, I_z, I_t, G)
sys.add_dirichlet_bc(0, "fixed")
sys.add_load_bc(sys._number_of_nodes - 1, "y", "+")
sys._interesting_dof = 22

if sys.solver == "load-control":    
    sys.max_load = 10.
elif sys.solver == "displacement-control":    
    sys.max_displacement = 8.0
elif sys.solver == "arc-length-control":    
    sys.arc_length = 0.1

sys.number_of_load_increments = 20
sys.tolerance = 1e-3
sys.max_iteration_steps = 100

sys.solve_the_system()

sys.plot_equilibrium_path()
sys.plot_the_structure()
