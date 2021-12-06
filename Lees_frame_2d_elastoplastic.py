# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
from source.System import System
import time
import numpy as np

dim = 2
geometry = "Lees_frame"
sys = System(dim, geometry)

b = 3  # width
h = 2  # height
E = 720  # Young's modulus
nu = 0.3 # Poisson ratio

sys.initialize_structure(beamtype="Bernoulli",
                         youngs_modulus=E, poisson_ratio=nu, width=b, height=h)
sys.initialize_with_plasticity(hardening_model="linear hardening", gauss_number=(
    7, 7), yield_stress=10.44, plastic_modulus=80)

sys.add_dirichlet_bc(0, "xz")
sys.add_dirichlet_bc(20, "xz")
sys.add_load(12, force=np.array([[0., 1., 0.]]).T)
sys.define_interesting_dof(12, "z")

time_start = time.time()
sys.solve_the_system(solver="arc-length-control", number_of_increments=1000,
                     direction="negative", max_iteration_steps=10000, arc_length=0.5)
time_end = time.time()

print("Time cost: " + str(time_end - time_start) + "s")
sys.output_equilibrium_path("Lees_frame_2d_elastoplastic")
sys.plot_the_structure()
sys.plot_equilibrium_path(horizontal_flip=True, vertical_flip=True)
