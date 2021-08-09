# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou

"""
import source.Utilities as util


class System():
    def __init__(self):
        self._geometry_name = None
        self._initial_structure = None
        self._deformed_structure = None
        self._number_of_load_increments = None
        self._dirichlet_boundary_condition = None
        self._load_boundary_condition = None
        self._tolerance = None
        self._max_iteration_steps = None
        self._solver = None

    @property
    def geometry_name(self):
        return self._geometry_name

    @geometry_name.setter
    def geometry_name(self, val):
        if isinstance(val, str):
            self._geometry_name = val
        else:
            raise TypeError("The name of mesh file must be a string!")

    @property
    def initial_structure(self):
        return self._initial_structure

    @initial_structure.setter
    def initial_structure(self, mesh_file):
        array_nodes, array_elements = util.load_mesh_file(self.geometry_name)

    @property
    def number_of_elements(self):
        return self._number_of_elements

    @number_of_elements.setter
    def number_of_elements(self, val):
        if isinstance(val, int) and val > 0:
            self._number_of_elements = val
        else:
            raise TypeError("Number of elements must be a positive integer!")
