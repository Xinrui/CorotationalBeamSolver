# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
from source.System import System
import time
import numpy as np

dim = 3
geometry = "cantilever_beam_elastic_3d"
sys = System(dim, geometry)

b = 2  # width
h = 2  # height
E = 100  # Young's modulus

sys.initialize_structure(beamtype="Bernoulli",
                         youngs_modulus=E, width=b, height=h, e_2o=np.array([[-np.sqrt(2)/2, np.sqrt(2)/2, 0.]]).T)

sys.add_dirichlet_bc(0, "fixed")
sys.add_load(7, force=np.array([[0., 0., 1.]]).T)
sys.define_interesting_dof(7, "z")

time_start = time.time()
sys.solve_the_system(solver="load-control",
                     number_of_increments=100, load=-5, max_iteration_steps=1000)
time_end = time.time()

print("Time cost: " + str(time_end - time_start) + "s")
sys.output_equilibrium_path("cantilever_beam_3d_elastic")
sys.plot_the_structure()
sys.plot_equilibrium_path(horizontal_flip=True, vertical_flip=True)