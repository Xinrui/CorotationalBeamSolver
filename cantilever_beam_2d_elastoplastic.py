# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
from source.System import System
import time
import numpy as np

dim = 2
geometry = "cantilever_beam_elastoplastic_2D"
sys = System(dim, geometry)

b = 0.1  # width
h = 0.5  # height
E = 3e7  # Young's modulus

sys.initialize_structure(beamtype="Bernoulli",
                         youngs_modulus=E, width=b, height=h)
sys.initialize_with_plasticity(hardening_model="linear hardening", gauss_number=(
    2, 2), yield_stress=3e4, plastic_modulus=E/29)

sys.add_dirichlet_bc(0, "fixed")
sys.add_load(10, force=np.array([[0., 1., 0.]]).T)
sys.define_interesting_dof(10, "z")

time_start = time.time()
sys.solve_the_system(solver="load-control",
                     number_of_increments=1000, load=-1200, max_iteration_steps=1000)
time_end = time.time()

print("Time cost: " + str(time_end - time_start) + "s")

sys.output_equilibrium_path("cantilever_beam_2d_elastoplastic")
sys.plot_the_structure()
sys.plot_equilibrium_path(horizontal_flip=True, vertical_flip=True)