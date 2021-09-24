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
# I_z = 1.3333
# I_t = 2.6666
# G = 100000.

sys = System()
sys.dimension = 2
sys.geometry_name = "cantilever_beam"
sys.analysis = "elastic"
sys.solver = "arc-length-control"

b = 0.1  # cross-sectional area
h = 0.5  # moment of inertia
E = 3.0e7  # Young's modulus
sys.initialize_structure_2d(E, b, h)

if sys.analysis == "perfect plasticity":
    sigma_y = 3.0e4
    sys.initialize_with_plasticity(sigma_y)
elif sys.analysis == "linear hardening":
    sigma_y = 3.0e4
    K = E / 29
    sys.initialize_with_plasticity(sigma_y, K)

# sys.initialize_structure(E, A, I, I_z, I_t, G)
sys.add_dirichlet_bc(0, "fixed")
sys.add_load_bc(sys._number_of_nodes - 1, "y")

# sys.max_load = 10.

if sys.solver == "load-control":    
    sys.max_load = 1.0
elif sys.solver == "displacement-control":    
    sys.max_displacement = 1.0
elif sys.solver == "arc-length-control":    
    sys.arc_length = 0.1

sys.number_of_load_increments = 20
sys.tolerance = 1e-8
sys.max_iteration_steps = 1000

U, LAM = sys.solve_the_system()

sys.plotLoadDisplacementCurve()
sys.plot_the_structure()
