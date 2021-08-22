# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
from source.CorotationalBeamElement import System
import source.Utilities as util
"""
--------------------------------
1. Define / Initialize variables
--------------------------------
"""
b = 0.1  # cross-sectional area
h = 0.5  # moment of inertia
E = 3.0e7  # Young's modulus

sigma_y = 3.0e4
K = E/29

# I_z = 1.3333
# I_t = 2.6666
# G = 100000.

sys = System()
sys.dimension = 2
sys.geometry_name = "cantilever_beam"
sys.initialize_structure(E, b, h, sigma_y, K)
# sys.initialize_structure(E, A, I, I_z, I_t, G)
sys.add_dirichlet_bc(0, "fixed")
sys.add_load_bc(sys._number_of_nodes - 1, "y")

# sys.max_load = 10.

sys.solver = "Load-Control"
sys.number_of_load_increments = 5000
sys.tolerance = 1e-3
sys.max_iteration_steps = 100

U, LAM = sys.solve_the_system_dis()

util.plotLoadDisplacementCurve(U, LAM)
sys.plot_the_structure()
