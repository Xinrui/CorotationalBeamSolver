# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
from source.System import System
import time

dim = 3
geometry = "lees_frame"
sys = System(dim, geometry)

b = 2  # width
h = 3  # height
E = 720  # Young's modulus
nu = 0.3 # Poisson ratio

sys.initialize_structure(beamtype="Timoshenko",
                         youngs_modulus=E, poisson_ratio=nu, width=b, height=h)

sys.add_dirichlet_bc(0, "xyz")
sys.add_dirichlet_bc(20, "xyz")
sys.add_load_bc(12, "y")

time_start = time.time()
sys.solve_the_system(solver="arc-length-control", number_of_increments=600,
                     direction="negative", max_iteration_steps=10000, arc_length=1.0)
time_end = time.time()

print("Time cost: " + str(time_end - time_start) + "s")
# sys.output_equilibrium_path("Lees_frame_3d_elastic")
sys.plot_the_structure()
sys.plot_equilibrium_path(horizontal_flip=True, vertical_flip=True)
