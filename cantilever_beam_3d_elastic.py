# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
from source.System import System
import time

dim = 3
geometry = "cantilever_beam_elastic_3d"
sys = System(dim, geometry)

b = 2  # width
h = 2  # height
E = 100  # Young's modulus

sys.initialize_structure(beamtype="Bernoulli",
                         youngs_modulus=E, width=b, height=h)

sys.add_dirichlet_bc(0, "fixed")
sys.add_load_bc(7, "z")

time_start = time.time()
sys.solve_the_system(solver="load-control",
                     number_of_increments=100, load=-5, max_iteration_steps=1000)
time_end = time.time()

print("Time cost: " + str(time_end - time_start) + "s")
sys.output_equilibrium_path("cantilever_beam_3d_elastic")
sys.plot_the_structure()
sys.plot_equilibrium_path(horizontal_flip=True, vertical_flip=True)