# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou
"""

from math import atan2

import matplotlib.pyplot as plt
import numpy as np

from source.CorotationalBeamElement2D import CorotationalBeamElement2D
from source.CorotationalBeamElement3D import CorotationalBeamElement3D

from scipy.linalg import expm
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

import source.Utilities as util

class System():
    def __init__(self):
        self.__dimension = None
        self.__geometry_name = None
        self.__analysis = "elastic"

        self._structure = None

        self._number_of_elements = None
        self._number_of_nodes = None
        self._number_of_dofs = None

        self._dirichlet_boundary_condition = []
        self._load_boundary_condition = None
        self._load_increment_vector = None

        self._state_variable = None
        self._control_parameter = None

        self._interesting_dof = None
        self._state_variable_plot = [0.]
        self._control_parameter_plot = [0.]

    @property
    def dimension(self):
        return self.__dimension

    @dimension.setter
    def dimension(self, val):
        """
        Set the dimension of the problem. The problem could be
        2-dimensional or 3-dimensional.

        Raises:
            TypeError: If value is not an integer.
            ValueError: If value is not 2 or 3.

        """
        if not isinstance(val, int):
            raise TypeError("The dimension must be 2 or 3!")
        elif val != 2 and val != 3:
            raise ValueError("The dimension must be 2 or 3!")
        else:
            self.__dimension = val

    @property
    def geometry_name(self):
        return self.__geometry_name

    @geometry_name.setter
    def geometry_name(self, val):
        if isinstance(val, str):
            self.__geometry_name = val
        else:
            raise TypeError("The name of mesh file must be a string!")

    @property
    def analysis(self):
        return self.__analysis

    @analysis.setter
    def analysis(self, val):
        self.__analysis = val

    def initialize_structure(self, youngs_modulus, width, height, shear_modulus=1.0):
        structure = []
        array_nodes, array_elements, self._number_of_nodes, self._number_of_elements = util.load_mesh_file(
            self.geometry_name)
        if self.__dimension == 2:
            self._number_of_dofs = 3 * self._number_of_nodes
        else:
            self._number_of_dofs = 6 * self._number_of_nodes

        for iele, ele in enumerate(array_elements.T):
            if self.__dimension == 2:
                co_ele = CorotationalBeamElement2D()
                co_ele.initial_coordinate_node_1 = array_nodes[0: 2, ele[0] - 1].reshape(
                    2, 1)
                co_ele.initial_coordinate_node_2 = array_nodes[0: 2, ele[1] - 1].reshape(
                    2, 1)
            else:
                co_ele = CorotationalBeamElement3D()
                co_ele.initial_coordinate_node_1 = array_nodes[:, ele[0] - 1].reshape(
                    3, 1)
                co_ele.initial_coordinate_node_2 = array_nodes[:, ele[1] - 1].reshape(
                    3, 1)

            co_ele.youngs_modulus = youngs_modulus
            co_ele.shear_modulus = shear_modulus
            co_ele.width = width
            co_ele.height = height

            co_ele.element_freedom_table = iele

            structure.append(co_ele)

        self._state_variable = np.zeros((self._number_of_dofs, 1), dtype=float)
        self._control_parameter = 0.
        self._structure = structure

    def initialize_with_plasticity(self, analysis, yield_stress, plastic_modulus=None,
                                   quadratic_coefficient=None, saturation_stress=None,
                                   modified_modulus=None, exponent=None):

        if analysis == "perfect plasticity":
            self.analysis = analysis
            for ele in self._structure:
                ele.analysis = analysis
                ele.yield_stress = yield_stress
        elif analysis == "linear hardening":
            self.analysis = analysis
            for ele in self._structure:
                ele.analysis = analysis
                ele.yield_stress = yield_stress
                ele.plastic_modulus = plastic_modulus
        elif analysis == "exponential hardening":
            self.analysis = analysis
            for ele in self._structure:
                ele.analysis = analysis
                ele.yield_stress = yield_stress
                ele.saturation_stress = saturation_stress
                ele.exponent = exponent
        elif analysis == "ramberg-osgood hardening":
            self.analysis = analysis
            for ele in self._structure:
                ele.analysis = analysis
                ele.yield_stress = yield_stress
                ele.modified_modulus = modified_modulus
                ele.exponent = exponent
        else:
            raise ValueError("Please input an available plastic model!")

    def add_dirichlet_bc(self, node, dof):
        if self.dimension == 2:
            if dof == "x":
                self._dirichlet_boundary_condition.append(3 * node)
            elif dof == "y":
                self._dirichlet_boundary_condition.append(3 * node + 1)
            elif dof == "xy":
                self._dirichlet_boundary_condition.append(3 * node)
                self._dirichlet_boundary_condition.append(3 * node + 1)
            elif dof == "fixed":
                self._dirichlet_boundary_condition.append(3 * node)
                self._dirichlet_boundary_condition.append(3 * node + 1)
                self._dirichlet_boundary_condition.append(3 * node + 2)
        else:
            if dof == "x":
                self._dirichlet_boundary_condition.append(6 * node)
            elif dof == "y":
                self._dirichlet_boundary_condition.append(6 * node + 1)
            elif dof == "xy":
                self._dirichlet_boundary_condition.append(6 * node)
                self._dirichlet_boundary_condition.append(6 * node + 1)
            elif dof == "fixed":
                self._dirichlet_boundary_condition.append(6 * node)
                self._dirichlet_boundary_condition.append(6 * node + 1)
                self._dirichlet_boundary_condition.append(6 * node + 2)
                self._dirichlet_boundary_condition.append(6 * node + 3)
                self._dirichlet_boundary_condition.append(6 * node + 4)
                self._dirichlet_boundary_condition.append(6 * node + 5)

    def add_load_bc(self, node, dof):
        if self.dimension == 2:
            if dof == "x":
                self._load_boundary_condition = 3 * node
            elif dof == "y":
                self._load_boundary_condition = 3 * node + 1
            elif dof == "m":
                self._load_boundary_condition = 3 * node + 2
        else:
            if dof == "x":
                self._load_boundary_condition = 6 * node
            elif dof == "y":
                self._load_boundary_condition = 6 * node + 1
            elif dof == "m":
                self._load_boundary_condition = 6 * node + 2

        self._load_increment_vector = np.zeros(
            (self._number_of_dofs, 1), dtype=float)
        self._load_increment_vector[self._load_boundary_condition] = 1
        self._interesting_dof = self._load_boundary_condition

    @property
    def master_stiffness_matrix(self):
        """ Assemble system stiffness matrix K from member stiffness matrices.

            Args:
                beam: the list of beam elements
                nele: the number of elements
                ndof: the number of degree of freedoms
                q_l: the storage vector of local forces, [3 * nele x 1]

            Returns:
                K: the system stiffness matrix, [ndof x ndof]
        """
        # K = np.zeros((self._number_of_dofs, self._number_of_dofs), dtype=float)

        # if self.__dimension == 2:
        #     dof_per_ele = 6
        # else:
        #     dof_per_ele = 12

        # for ele in self._structure:
        #     eft = ele.element_freedom_table
        #     esm = ele.global_stiffness_matrix
        #     for idof in range(dof_per_ele):
        #         for jdof in range(idof, dof_per_ele):
        #             K[eft[idof], eft[jdof]
        #               ] += esm[idof, jdof]
        #             K[eft[jdof], eft[idof]
        #               ] = K[eft[idof], eft[jdof]]
        # return K

        val = np.array([], dtype=float)
        row = np.array([], dtype=int)
        col = np.array([], dtype=int)

        for ele in self._structure:
            el_dofs = ele.element_freedom_table
            row = np.append(row, np.repeat(el_dofs, len(el_dofs)))

            for i in range(len(el_dofs)):
                col = np.append(col, el_dofs)

            val = np.append(val, ele.global_stiffness_matrix)

        return coo_matrix((val, (row, col)), shape=(self._number_of_dofs, self._number_of_dofs))

    @property
    def modified_master_stiffness_matrix(self):
        """ Modify the system stiffness matrix K to K_s according to Drichlet
            Boundary Conditions.

            Args:
                K: the system stiffness matrix, [ndof x ndof]
                DBCdof: a list contains the dofs, such as [0, 1, 2]

            Returns:
                K_s: the modified system stiffness matrix, [ndof x ndof]
        """
        K_s = self.master_stiffness_matrix.copy()
        K_s = csr_matrix(K_s)
        for idof in self._dirichlet_boundary_condition:
            for jentry in range(self._number_of_dofs):
                K_s[idof, jentry] = 0.
                K_s[jentry, idof] = 0.
                K_s[idof, idof] = 1.

        return K_s

    @property
    def internal_force_vector(self):
        """ Assemble global internal force vector F_int extracting the internal
            force vector of every beam element.

            Args:
                beam: the list of beam elements
                nele: the number of elements
                ndof: the number of degree of freedoms
                q_l: the storage vector of local forces, [3 * nele x 1]

            Returns:
                F_int: the global internal force vector, [ndof x 1]
        """
        F_int = np.zeros((self._number_of_dofs, 1), dtype=float)
        for ele in self._structure:
            eft = ele.element_freedom_table
            ef = ele.global_force
            for idof, iEFT in enumerate(eft):
                F_int[iEFT] += ef[idof]
        return F_int

    @property
    def residual(self):
        return self.internal_force_vector - self._control_parameter * self._load_increment_vector

    @property
    def modified_residual(self):
        """ Modify the residual according to Drichlet Boundary Conditions.

            Args:
                r: the residual of the system, [ndof x 1]
                DBCdof: a list contains the dofs, such as [0, 1, 2]

            Returns:
                r: the MODIFIED residual of the system, [ndof x 1]
        """
        modified_residual = np.copy(self.residual)

        for idof in self._dirichlet_boundary_condition:
            modified_residual[idof] = 0.

        return modified_residual

    @property
    def residual_norm(self):
        return np.linalg.norm(self.modified_residual)

    def update_member_data_2d(self, u, lam):
        for iele, ele in enumerate(self._structure):
            ele.incremental_global_displacement = u[3 * iele: 3 * iele + 6]
            if ele.analysis == "perfect plasticity":
                ele.perform_perfect_plasticity()
            elif ele.analysis == "linear hardening":
                ele.perform_linear_hardening()
            elif ele.analysis == "quadratic hardening":
                ele.perform_quadratic_hardening()
            elif ele.analysis == "exponential hardening":
                ele.perform_exponential_hardening()
            elif ele.analysis == "ramberg-osgood hardening":
                ele.perform_ramberg_osgood_hardening()

        self._state_variable = u
        self._control_parameter = lam

    def update_member_data_2d_iteration(self, u, lam, *args):
        for iele, ele in enumerate(self._structure):
            ele.incremental_global_displacement = u[3 * iele: 3 * iele + 6]
            if ele.analysis == "perfect plasticity":
                ele.plastic_strain = args[0][iele]
                ele.internal_hardening_variable = args[1][iele]
                ele.back_stress = args[2][iele]
                ele.perform_perfect_plasticity()
            elif ele.analysis == "linear hardening":
                ele.plastic_strain = args[0][iele]
                ele.internal_hardening_variable = args[1][iele]
                ele.back_stress = args[2][iele]
                ele.perform_linear_hardening()
            elif ele.analysis == "exponential hardening":
                ele.plastic_strain = args[0][iele]
                ele.internal_hardening_variable = args[1][iele]
                ele.back_stress = args[2][iele]
                ele.perform_exponential_hardening()
            elif ele.analysis == "ramberg-osgood hardening":
                ele.plastic_strain = args[0][iele]
                ele.internal_hardening_variable = args[1][iele]
                ele.back_stress = args[2][iele]
                ele.perform_ramberg_osgood_hardening()

        self._state_variable = u
        self._control_parameter = lam

    def update_member_data_3d(self, u, lam, deltau):
        for iele, ele in enumerate(self._structure):
            ele.incremental_global_displacement = u[6 * iele: 6 * iele + 12]
            ele.current_orientation_node_1 = expm(util.getSkewSymmetric(
                deltau[6 * iele + 3: 6 * iele + 6])) @ ele.current_orientation_node_1
            ele.current_orientation_node_2 = expm(util.getSkewSymmetric(
                deltau[6 * iele + 9: 6 * iele + 12])) @ ele.current_orientation_node_2
            if ele.analysis == "perfect plasticity":
                ele.perform_perfect_plasticity()
            elif ele.analysis == "linear hardening":
                ele.perform_linear_hardening()
            elif ele.analysis == "quadratic hardening":
                ele.perform_quadratic_hardening()
            elif ele.analysis == "exponential hardening":
                ele.perform_exponential_hardening()
            elif ele.analysis == "ramberg-osgood hardening":
                ele.perform_ramberg_osgood_hardening()

        self._state_variable = u
        self._control_parameter = lam

    def update_member_data_3d_iteration(self, u, lam, deltau, *args):
        for iele, ele in enumerate(self._structure):
            ele.incremental_global_displacement = u[6 * iele: 6 * iele + 12]
            ele.current_orientation_node_1 = expm(util.getSkewSymmetric(
                deltau[6 * iele + 3: 6 * iele + 6])) @ ele.current_orientation_node_1
            ele.current_orientation_node_2 = expm(util.getSkewSymmetric(
                deltau[6 * iele + 9: 6 * iele + 12])) @ ele.current_orientation_node_2
            if ele.analysis == "perfect plasticity":
                ele.plastic_strain = args[0][iele]
                ele.internal_hardening_variable = args[1][iele]
                ele.back_stress = args[2][iele]
                ele.perform_perfect_plasticity()
            elif ele.analysis == "linear hardening":
                ele.plastic_strain = args[0][iele]
                ele.internal_hardening_variable = args[1][iele]
                ele.back_stress = args[2][iele]
                ele.perform_linear_hardening()
            elif ele.analysis == "exponential hardening":
                ele.plastic_strain = args[0][iele]
                ele.internal_hardening_variable = args[1][iele]
                ele.back_stress = args[2][iele]
                ele.perform_exponential_hardening()
            elif ele.analysis == "ramberg-osgood hardening":
                ele.plastic_strain = args[0][iele]
                ele.internal_hardening_variable = args[1][iele]
                ele.back_stress = args[2][iele]
                ele.perform_ramberg_osgood_hardening()

        self._state_variable = u
        self._control_parameter = lam
    
    def load_control(self, number_of_increments, tolerance, max_iteration_steps, load):
        for n in range(number_of_increments):

            Deltalam = load / number_of_increments
            Deltau = spsolve(self.modified_master_stiffness_matrix, Deltalam * self._load_increment_vector).reshape(
                self._number_of_dofs, 1)

            u_pre = self._state_variable + Deltau
            lam_pre = self._control_parameter + Deltalam

            # update member data
            if self.__dimension == 2:
                self.update_member_data_2d(u_pre, lam_pre)
            else:
                self.update_member_data_3d(u_pre, lam_pre, Deltau)

            u_temp = u_pre
            # initialize iteration counter
            kiter = 0

            if self.__analysis != "elastic":
                plastic_strain = []
                internal_hardening_variable = []
                back_stress = []
                for ele in self._structure:
                    plastic_strain.append(ele.plastic_strain)
                    internal_hardening_variable.append(
                        ele.internal_hardening_variable)
                    back_stress.append(ele.back_stress)

            # iterate, until good result or so many iteration steps
            while(self.residual_norm > tolerance and kiter < max_iteration_steps):
                deltau = -spsolve(
                    self.modified_master_stiffness_matrix, self.modified_residual).reshape(self._number_of_dofs, 1).reshape(self._number_of_dofs, 1)
                u_temp += deltau

                if self.__analysis != "elastic":
                    if self.__dimension == 2:
                        self.update_member_data_2d_iteration(
                            u_temp, lam_pre, plastic_strain, internal_hardening_variable, back_stress)
                    else:
                        self.update_member_data_3d_iteration(
                            u_temp, lam_pre, deltau, plastic_strain, internal_hardening_variable, back_stress)
                else:
                    if self.__dimension == 2:
                        self.update_member_data_2d(u_temp, lam_pre)
                    else:
                        self.update_member_data_3d(u_temp, lam_pre, deltau)

                # update iterations counter
                kiter += 1
                print("Iteration step: " + str(kiter))
                print("residual norm: " + str(self.residual_norm))
                if(kiter == max_iteration_steps):
                    raise RuntimeError(
                        'Newton-Raphson iterations did not converge!')

            self._state_variable_plot.append(
                float(self._state_variable[self._interesting_dof]))
            self._control_parameter_plot.append(self._control_parameter)
            print("Incrementation step: " + str(n + 1))

    def displacement_control(self, number_of_increments, tolerance, max_iteration_steps, displacement):
        for n in range(number_of_increments):
            velocity = spsolve(
                self.modified_master_stiffness_matrix, self._load_increment_vector).reshape(self._number_of_dofs, 1)

            Deltalam = float(
                displacement / (number_of_increments * velocity[self._load_boundary_condition]))
            Deltau = spsolve(self.modified_master_stiffness_matrix, Deltalam * self._load_increment_vector).reshape(
                self._number_of_dofs, 1)

            u_pre = self._state_variable + Deltau
            lam_pre = self._control_parameter + Deltalam

            # update member data
            if self.__dimension == 2:
                self.update_member_data_2d(u_pre, lam_pre)
            else:
                self.update_member_data_3d(u_pre, lam_pre, Deltau)

            u_temp = u_pre
            lam_temp = lam_pre

            # initialize iteration counter
            kiter = 0

            if self.__analysis != "elastic":
                plastic_strain = []
                internal_hardening_variable = []
                back_stress = []
                for ele in self._structure:
                    plastic_strain.append(ele.plastic_strain)
                    internal_hardening_variable.append(
                        ele.internal_hardening_variable)
                    back_stress.append(ele.back_stress)

            # iterate, until good result or so many iteration steps
            while(self.residual_norm > tolerance and kiter < max_iteration_steps):
                # u_invhat = spsolve(
                #     self.modified_master_stiffness_matrix, self.modified_residual).reshape(self._number_of_dofs, 1)
                # u_hat = spsolve(
                #     self.modified_master_stiffness_matrix, self._load_increment_vector).reshape(self._number_of_dofs, 1)

                # deltalam += float(u_invhat[self._load_boundary_condition] /
                #                   u_hat[self._load_boundary_condition])
                # deltau = -spsolve(self.modified_master_stiffness_matrix, self.modified_residual - float(
                #     u_invhat[self._load_boundary_condition] / u_hat[self._load_boundary_condition]) * self._load_increment_vector).reshape(self._number_of_dofs, 1)

                # u_temp += deltau

                
                adjusted_stiffness_matrix = self.modified_master_stiffness_matrix.toarray()
                adjusted_stiffness_matrix[:, self._load_boundary_condition] = -self._load_increment_vector.reshape(self._number_of_dofs)
                
                augmented_solution = -np.linalg.solve(adjusted_stiffness_matrix, self.modified_residual)
                deltalam = float(augmented_solution[self._load_boundary_condition])
                deltau = np.copy(augmented_solution)
                deltau[self._load_boundary_condition, 0] = 0.0
                
                u_temp += deltau
                lam_temp += deltalam
                
                if self.__analysis != "elastic":
                    if self.__dimension == 2:
                        self.update_member_data_2d_iteration(
                            u_temp, lam_temp, plastic_strain, internal_hardening_variable, back_stress)
                    else:
                        self.update_member_data_3d_iteration(
                            u_temp, lam_temp, deltau, plastic_strain, internal_hardening_variable, back_stress)
                else:
                    if self.__dimension == 2:
                        self.update_member_data_2d(
                            u_temp, lam_temp)
                    else:
                        self.update_member_data_3d(
                            u_temp, lam_temp, deltau)

                # update iterations counter
                kiter += 1
                print("Iteration step: " + str(kiter))
                print("residual norm: " + str(self.residual_norm))
                if(kiter == max_iteration_steps):
                    raise RuntimeError(
                        'Newton-Raphson iterations did not converge!')

            self._state_variable_plot.append(
                float(self._state_variable[self._interesting_dof]))
            self._control_parameter_plot.append(self._control_parameter)
            print("Incrementation step: " + str(n + 1))

    def arc_length_control(self, number_of_increments, tolerance, max_iteration_steps, direction, arc_length):
        if direction == "positive":
            Deltau_prev = np.ones((self._number_of_dofs, 1), dtype=float)
        elif direction == "negative":
            Deltau_prev = -np.ones((self._number_of_dofs, 1), dtype=float)
        else:
            raise ValueError("Please input the right direction!")

        for n in range(number_of_increments):

            deltaubar = spsolve(
                self.modified_master_stiffness_matrix, self._load_increment_vector).reshape(self._number_of_dofs, 1)
            Deltalam = np.sign(float(Deltau_prev.T @ deltaubar)) * \
                arc_length / np.sqrt(float(deltaubar.T @ deltaubar))

            Deltau = spsolve(
                self.modified_master_stiffness_matrix, Deltalam * self._load_increment_vector).reshape(self._number_of_dofs, 1)

            u_pre = self._state_variable + Deltau
            lam_pre = self._control_parameter + Deltalam

            # update member data
            if self.__dimension == 2:
                self.update_member_data_2d(u_pre, lam_pre)
            else:
                self.update_member_data_3d(u_pre, lam_pre, Deltau)

            # initialize iteration counter
            kiter = 0

            u_temp = u_pre
            lam_temp = lam_pre
            # deltau = np.zeros((self._number_of_dofs, 1), dtype=float)
            # deltalam = 0.

            if self.__analysis != "elastic":
                plastic_strain = []
                internal_hardening_variable = []
                back_stress = []
                for ele in self._structure:
                    plastic_strain.append(ele.plastic_strain)
                    internal_hardening_variable.append(
                        ele.internal_hardening_variable)
                    back_stress.append(ele.back_stress)

            # iterate, until good result or so many iteration steps
            while(self.residual_norm > tolerance and kiter < max_iteration_steps):

                deltau_star = spsolve(
                    self.modified_master_stiffness_matrix, -self.modified_residual).reshape(self._number_of_dofs, 1)
                deltau_bar = spsolve(
                    self.modified_master_stiffness_matrix, self._load_increment_vector).reshape(self._number_of_dofs, 1)

                if kiter == 0:
                    deltalam = np.sign(float(Deltau.T @ deltaubar)) * \
                        arc_length / np.sqrt(float(deltaubar.T @ deltaubar))
                else:
                    a = float(deltau_bar.T @ deltau_bar)
                    b = float(2 * (Deltau + deltau_star).T @ deltau_bar)
                    c = float((Deltau + deltau_star).T @
                              (Deltau + deltau_star)) - arc_length ** 2

                    [deltalam2_hat, deltalam1_hat] = np.roots([a, b, c])

                    dotprod1 = float(
                        (Deltau + deltau_star + deltalam1_hat * deltau_bar).T @ Deltau)
                    dotprod2 = float(
                        (Deltau + deltau_star + deltalam2_hat * deltau_bar).T @ Deltau)

                    if dotprod1 >= dotprod2:
                        deltalam = deltalam1_hat
                    else:
                        deltalam = deltalam2_hat

                deltau = deltau_star + deltalam * deltau_bar
                u_temp += deltau
                lam_temp += deltalam

                # update member data
                if self.__analysis != "elastic":
                    self.update_member_data_2d_iteration(
                        u_temp, lam_temp, plastic_strain, internal_hardening_variable, back_stress)
                else:
                    if self.__dimension == 2:
                        self.update_member_data_2d(
                            u_temp, lam_temp)
                    else:
                        self.update_member_data_3d(u_temp, lam_temp, deltau)

                # update iterations counter
                kiter += 1
                print("Iteration step: " + str(kiter))
                print("residual norm: " + str(self.residual_norm))
                if(kiter == max_iteration_steps):
                    raise RuntimeError(
                        'Newton-Raphson iterations did not converge!')

            """
            ------------------------------------------------------------------
            3. Update variables to their final value for the current increment
            ------------------------------------------------------------------
            """
            Deltau_prev = Deltau + deltau

            self._state_variable_plot.append(
                float(self._state_variable[self._interesting_dof]))
            self._control_parameter_plot.append(self._control_parameter)
            print("Incrementation step: " + str(n + 1))

    def solve_the_system(self, solver, number_of_increments, tolerance=1e-3, max_iteration_steps=100, load=None, displacement=None, direction=None, arc_length=None):
        if solver == "load-control":
            self.load_control(number_of_increments, tolerance,
                              max_iteration_steps, load)
        elif solver == "displacement-control":
            self.displacement_control(
                number_of_increments, tolerance, max_iteration_steps, displacement)
        elif solver == "arc-length-control":
            self.arc_length_control(
                number_of_increments, tolerance, max_iteration_steps, direction, arc_length)
        else:
            raise ValueError("Please input the correct solver!")

    def plot_equilibrium_path(self, horizontal_flip=False, vertical_flip=False):
        """ Plot the equilibrium path.

            Args:
                U: vector of the state args at the interesting dof, [ninc x 1]
                LAM: vector of the control args, [ninc x 1]
        """

        # Plot both configurations
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots()
        if horizontal_flip == False and vertical_flip == False:
            ax.plot(self._state_variable_plot,
                    self._control_parameter_plot, '.-')
        elif horizontal_flip == True and vertical_flip == False:
            inv_state_variable_plot = [-u for u in self._state_variable_plot]
            ax.plot(inv_state_variable_plot,
                    self._control_parameter_plot, '.-')
        elif horizontal_flip == False and vertical_flip == True:
            inv_control_parameter_plot = [
                -lam for lam in self._control_parameter_plot]
            ax.plot(self._state_variable_plot, -
                    inv_control_parameter_plot, '.-')
        elif horizontal_flip == True and vertical_flip == True:
            inv_state_variable_plot = [-u for u in self._state_variable_plot]
            inv_control_parameter_plot = [
                -lam for lam in self._control_parameter_plot]
            ax.plot(inv_state_variable_plot, inv_control_parameter_plot, '.-')

        ax.set_xlabel('$u$')
        ax.set_ylabel('$\lambda$')
        ax.set_title('Equilibrium Path')
        ax.grid()
        plt.show()

    def plot_the_structure(self):
        """ Plot the UNDEFORMED and the DEFORMED structure.
            Args:
                X, Y: coordinates of the undeformed structure
                self.number_of_nodes: the number of nodes of the system
                self.number_of_elements: the number of elements
                beam: the list of the beam elements
        """
        # generate coordinates of deformed structure

        if self.__dimension == 2:
            X = np.zeros((self._number_of_nodes))
            Z = np.zeros((self._number_of_nodes))
            x = np.zeros((self._number_of_nodes))
            z = np.zeros((self._number_of_nodes))
            for iele in range(self._number_of_elements):
                X[iele] = self._structure[iele].initial_coordinate_node_1[0]
                Z[iele] = self._structure[iele].initial_coordinate_node_1[1]
                x[iele] = self._structure[iele].current_coordinate_node_1[0]
                z[iele] = self._structure[iele].current_coordinate_node_1[1]
                if iele == self._number_of_elements - 1:
                    X[iele + 1] = self._structure[iele].initial_coordinate_node_2[0]
                    Z[iele + 1] = self._structure[iele].initial_coordinate_node_2[1]
                    x[iele + 1] = self._structure[iele].current_coordinate_node_2[0]
                    z[iele + 1] = self._structure[iele].current_coordinate_node_2[1]
            # Plot both configurations
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            fig, ax = plt.subplots()
            ax.plot(X, Z, '.--', label='undeformed configuration')
            # ax.scatter(X, Y)
            ax.plot(x, z, '.-', label='deformed configuration')
            # ax.scatter(x, y)
            ax.legend(loc='lower right')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$z$')
            ax.set_title(
                'Undeflected(dashed) and Deflected(solid) 2D beam structure')
            ax.grid()
            plt.show()
        else:
            X = np.zeros((self._number_of_nodes))
            Y = np.zeros((self._number_of_nodes))
            Z = np.zeros((self._number_of_nodes))
            x = np.zeros((self._number_of_nodes))
            y = np.zeros((self._number_of_nodes))
            z = np.zeros((self._number_of_nodes))
            for iele in range(self._number_of_elements):
                X[iele] = self._structure[iele].initial_coordinate_node_1[0]
                Y[iele] = self._structure[iele].initial_coordinate_node_1[1]
                Z[iele] = self._structure[iele].initial_coordinate_node_1[2]
                x[iele] = self._structure[iele].current_coordinate_node_1[0]
                y[iele] = self._structure[iele].current_coordinate_node_1[1]
                z[iele] = self._structure[iele].current_coordinate_node_1[2]
                if iele == self._number_of_elements - 1:
                    X[iele + 1] = self._structure[iele].initial_coordinate_node_2[0]
                    Y[iele + 1] = self._structure[iele].initial_coordinate_node_2[1]
                    Z[iele + 1] = self._structure[iele].initial_coordinate_node_2[2]
                    x[iele + 1] = self._structure[iele].current_coordinate_node_2[0]
                    y[iele + 1] = self._structure[iele].current_coordinate_node_2[1]
                    z[iele + 1] = self._structure[iele].current_coordinate_node_2[2]
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            ax = plt.axes(projection='3d')
            ax.plot(X, Y, Z, '.--', label='undeformed configuration')
            ax.plot(x, y, z, '.-', label='deformed configuration')
            ax.legend(loc='lower right')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_zlabel('$z$')
            ax.set_title(
                'Undeflected(dashed) and Deflected(solid) 3D beam structure')
            ax.grid()
            plt.show()
