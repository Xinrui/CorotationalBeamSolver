# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 12:43:56 2021

@author: Xinrui Zhou
"""
import numpy as np
from source.CorotationalBeamElement import CorotationalBeamElement2D as beam2d
from source.CorotationalBeamElement import System
import source.Utilities as util

"""
--------------------------------
1. Define / Initialize variables
--------------------------------
"""
A = 4.0  # cross-sectional area
I = 1.3333  # moment of inertia
E = 100.  # Young's modulus

sys = System()
sys.dimension = 2
sys.geometry_name = "TipLoad2D"
sys.initialize_structure(E, A, I)
sys.add_dirichlet_bc(0, "fixed")
sys.add_load_bc(sys._number_of_nodes - 1, "y")

sys.max_load = 10.

sys.solver = "Load-Control"
sys.number_of_load_increments = 20
sys.tolerance = 1e-3
sys.max_iteration_steps = 100

U, LAM = sys.solve_the_system()

util.plotLoadDisplacementCurve(U, LAM)
sys.plotTheStructure()

# L = 10. # Length, undeformed state
# nele = 7 # the number of elements
# nnode = nele + 1 # the number of nodes
# ndof = 3 * nnode # the number of dofs

# # upper bound of the load


# # initialize the Dirichlet / Neumann boundary condition
# DBCdof = [0, 1, 2]
# NBCdof = -2

# # incremental load vector
# q = np.zeros((ndof, 1), dtype = float)
# q[NBCdof] = -1

# # initialize control parameter, system displacement vector and storage vector
# lam = 0.
# u = np.zeros((ndof, 1), dtype = float)

# # initialize the geometry of the beam
# X = np.linspace(0, L, nnode)
# Y = np.zeros(nnode)

# # initialize the beam
# beam = []
# for iele in range(nele):

#     X1 = np.array([[X[iele], Y[iele]]]).T
#     X2 = np.array([[X[iele + 1], Y[iele + 1]]]).T

#     ele = beam2d()
#     ele.initial_coordinate_node_1 = X1
#     ele.initial_coordinate_node_2 = X2
#     ele.youngs_modulus = E
#     ele.area = A
#     ele.moment_of_inertia = I
#     ele.current_coordinate_node_1 = ele.initial_coordinate_node_1
#     ele.current_coordinate_node_2 = ele.initial_coordinate_node_2
#     ele.global_nodal_rotation_node_1 = 0.0
#     ele.global_nodal_rotation_node_2 = 0.0

#     beam.append(ele)

# # initialize the vector of interesting displacement and control parameter lambda
# U = np.array([0.], dtype = float)
# LAM = np.array([0.], dtype = float)

# # set up iteration variables


# """
# ----------------------------------------
# 2. Start loop over equal load increments
# ----------------------------------------
# """
# for n in range(ninc):

#     # set the predictor by equal load increments
#     K = util.mergeElemIntoMasterStiff(beam)
#     K_s = util.modifyMasterStiffForDBC(K, DBCdof)
#     dF = P / ninc * q

#     u_pre = u + np.linalg.solve(K_s, dF)
#     lam_pre = lam + P / ninc

#     # update member data
#     util.updateMemberData(u_pre, beam)

#     # calculate internal force vector
#     F_int = util.getInternalForceVector(beam)

#     # calculate the residual of the system
#     r = F_int - lam_pre * q
#     r = util.modifyTheResidual(r, DBCdof)
#     r_norm = np.linalg.norm(r)

#     # copy them for iteration, "temp" means they are not on equilibrium path.
#     u_temp = u_pre

#     # initialize iteration counter
#     kiter = 0

#     # iterate, until good result or so many iteration steps
#     while(r_norm > tolerance and kiter < maxiter):

#         # load-Control
#         K = util.mergeElemIntoMasterStiff(beam)
#         K_s = util.modifyMasterStiffForDBC(K, DBCdof)
#         deltau = np.linalg.solve(K_s, -r)
#         u_temp += deltau

#         # update member data
#         util.updateMemberData(u_temp, beam)

#         # calculate internal force vector
#         F_int = util.getInternalForceVector(beam)

#         # calculate the residual of the system
#         r = F_int - lam_pre * q
#         r = util.modifyTheResidual(r, DBCdof)
#         r_norm = np.linalg.norm(r)

#         # update iterations counter
#         kiter += 1
#         if(kiter == maxiter):
#             raise RuntimeError('Newton-Raphson iterations did not converge!')

#     """
#     ------------------------------------------------------------------
#     3. Update variables to their final value for the current increment
#     ------------------------------------------------------------------
#     """
#     u = u_temp
#     lam = lam_pre

#     U = np.append(U, -u[NBCdof])
#     LAM = np.append(LAM, lam)

# """
# ------------------------------------------------------------------
# 4. Plot equilibrium path and both configurations
# ------------------------------------------------------------------
# """
# util.plotLoadDisplacementCurve(U, LAM)
# util.plotTheStructure(beam)
