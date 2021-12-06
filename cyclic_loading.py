# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
from source.System import System
import time
import numpy as np

dim = 2
geo = "cyclic_loading"
sys = System(dim, geo)

b = 1.0  # cross-sectional area
h = 1.0  # moment of inertia
E = 29.0e3  # Young's modulus

sys.initialize_structure(beamtype="Bernoulli", youngs_modulus=E, width=b, height=h)
sys.initialize_with_plasticity(
    hardening_model="linear hardening", gauss_number=(2,2), yield_stress=36., plastic_modulus=500.)

sys.add_dirichlet_bc(0, "fixed")
sys.add_load(1, force=np.array([[1., 0., 0.]]).T)
sys.define_interesting_dof(1, "x")

time_start = time.time()

sys.solve_the_system(solver="displacement-control",
                     number_of_increments=20, displacement=0.1, max_iteration_steps=1000)
sys.solve_the_system(solver="displacement-control",
                     number_of_increments=20, displacement=-0.2, max_iteration_steps=1000)
sys.solve_the_system(solver="displacement-control",
                     number_of_increments=20, displacement=0.3, max_iteration_steps=1000)
sys.solve_the_system(solver="displacement-control",
                     number_of_increments=20, displacement=-0.4, max_iteration_steps=1000)
sys.solve_the_system(solver="displacement-control",
                     number_of_increments=20, displacement=0.5, max_iteration_steps=1000)
sys.solve_the_system(solver="displacement-control",
                     number_of_increments=20, displacement=-0.6, max_iteration_steps=1000)

time_end = time.time()

print("Time cost: " + str(time_end - time_start) + "s")

sys.output_equilibrium_path("cyclic_loading")
sys.plot_equilibrium_path()
sys.plot_the_structure()