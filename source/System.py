# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou
"""

import matplotlib.pyplot as plt
import numpy as np

from source.CorotationalBeamElement2D import CorotationalBeamElement2D
from source.CorotationalBeamElement3D import CorotationalBeamElement3D

import source.Utilities as util


class System():
    """It is a class constructing a non-linear system with corotational beam elements."""

    def __init__(self, dimension, geometry_name):
        self.dimension = dimension
        self.geometry_name = geometry_name
        self.analysis = "elastic"
        self.dirichlet_boundary_condition = []

        self.state_variable_plot = [0.]
        self.control_parameter_plot = [0.]

    def initialize_structure(self, beamtype, youngs_modulus, width, height, poisson_ratio=0.499, e_2o=None, e_3o=None):
        structure = []
        array_nodes, array_elements, self.number_of_nodes, self.number_of_elements = util.load_mesh_file(
            self.geometry_name)
        if self.dimension == 2:
            self.number_of_dofs = 3 * self.number_of_nodes
        else:
            self.number_of_dofs = 6 * self.number_of_nodes

        for iele, ele in enumerate(array_elements.T):
            if self.dimension == 2:
                initial_coordinate_node_1 = array_nodes[0: 2, ele[0] - 1].reshape(
                    2, 1)
                initial_coordinate_node_2 = array_nodes[0: 2, ele[1] - 1].reshape(
                    2, 1)
                co_ele = CorotationalBeamElement2D(
                    beamtype, youngs_modulus, poisson_ratio, width, height, initial_coordinate_node_1, initial_coordinate_node_2, iele)
            else:
                initial_coordinate_node_1 = array_nodes[:, ele[0] - 1].reshape(
                    3, 1)
                initial_coordinate_node_2 = array_nodes[:, ele[1] - 1].reshape(
                    3, 1)
                co_ele = CorotationalBeamElement3D(beamtype, youngs_modulus, poisson_ratio, width,
                                                   height, initial_coordinate_node_1, initial_coordinate_node_2, iele, e_2o, e_3o)

            structure.append(co_ele)

        self.state_variable = np.zeros((self.number_of_dofs, 1), dtype=float)
        self.control_parameter = 0.
        self.structure = structure

    def initialize_with_plasticity(self, hardening_model, gauss_number, yield_stress, kinematic_hardening_modulus=0.0, plastic_modulus=None, saturation_stress=None,
                                   modified_modulus=None, exponent=None):
        self.analysis = hardening_model
        for ele in self.structure:
            ele.apply_hardening_model(hardening_model, gauss_number, yield_stress, kinematic_hardening_modulus,
                                      plastic_modulus, saturation_stress, modified_modulus, exponent)

    def add_dirichlet_bc(self, node, dof):
        if self.dimension == 2:
            if dof == "x":
                self.dirichlet_boundary_condition.append(3 * node)
            elif dof == "z":
                self.dirichlet_boundary_condition.append(3 * node + 1)
            elif dof == "xz":
                self.dirichlet_boundary_condition.append(3 * node)
                self.dirichlet_boundary_condition.append(3 * node + 1)
            elif dof == "fixed":
                self.dirichlet_boundary_condition.append(3 * node)
                self.dirichlet_boundary_condition.append(3 * node + 1)
                self.dirichlet_boundary_condition.append(3 * node + 2)
        else:
            if dof == "x":
                self.dirichlet_boundary_condition.append(6 * node)
            elif dof == "y":
                self.dirichlet_boundary_condition.append(6 * node + 1)
            elif dof == "z":
                self.dirichlet_boundary_condition.append(6 * node + 2)
            elif dof == "xy":
                self.dirichlet_boundary_condition.append(6 * node)
                self.dirichlet_boundary_condition.append(6 * node + 1)
            elif dof == "xz":
                self.dirichlet_boundary_condition.append(6 * node)
                self.dirichlet_boundary_condition.append(6 * node + 2)
            elif dof == "yz":
                self.dirichlet_boundary_condition.append(6 * node + 1)
                self.dirichlet_boundary_condition.append(6 * node + 2)
            elif dof == "xyz":
                self.dirichlet_boundary_condition.append(6 * node)
                self.dirichlet_boundary_condition.append(6 * node + 1)
                self.dirichlet_boundary_condition.append(6 * node + 2)
            elif dof == "fixed":
                self.dirichlet_boundary_condition.append(6 * node)
                self.dirichlet_boundary_condition.append(6 * node + 1)
                self.dirichlet_boundary_condition.append(6 * node + 2)
                self.dirichlet_boundary_condition.append(6 * node + 3)
                self.dirichlet_boundary_condition.append(6 * node + 4)
                self.dirichlet_boundary_condition.append(6 * node + 5)

    def add_load(self, node, force):
        self.load_increment_vector = np.zeros(
            (self.number_of_dofs, 1), dtype=float)
        if self.dimension == 2:
            self.load_increment_vector[3 * node: 3 * node + 3] = force
        else:
            self.load_increment_vector[6 * node: 6 * node + 3] = force

    def define_interesting_dof(self, node, direction):
        if self.dimension == 2:
            if direction == "x":
                self.interesting_dof = 3 * node
            elif direction == "z":
                self.interesting_dof = 3 * node + 1
            elif direction == "m":
                self.interesting_dof = 3 * node + 2
        else:
            if direction == "x":
                self.interesting_dof = 6 * node
            elif direction == "y":
                self.interesting_dof = 6 * node + 1
            elif direction == "z":
                self.interesting_dof = 6 * node + 2

    def master_stiffness_force(self):
        K = np.zeros((self.number_of_dofs, self.number_of_dofs), dtype=float)
        F = np.zeros((self.number_of_dofs, 1), dtype=float)

        if self.dimension == 2:
            dof_per_ele = 6
        else:
            dof_per_ele = 12

        for ele in self.structure:
            eft = ele.element_freedom_table
            k, f = ele.global_stiffness_force()
            for idof in range(dof_per_ele):
                F[eft[idof]] += f[idof]
                for jdof in range(idof, dof_per_ele):
                    K[eft[idof], eft[jdof]] += k[idof, jdof]
                    K[eft[jdof], eft[idof]] = K[eft[idof], eft[jdof]]

        return K, F

    def modified_stiffness_force(self):
        K, F = self.master_stiffness_force()

        for idof in self.dirichlet_boundary_condition:
            F[idof] = 0.0
            for jentry in range(self.number_of_dofs):
                K[idof, jentry] = 0.0
                K[jentry, idof] = 0.0
                K[idof, idof] = 1.0

        return K, F

    def update_member_data_2d(self, u, lam):
        for iele, ele in enumerate(self.structure):
            ele.global_displacement = u[3 * iele: 3 * iele + 6]
            if ele.analysis == "linear hardening":
                ele.perform_linear_hardening()
            elif ele.analysis == "exponential hardening":
                ele.perform_exponential_hardening()
            elif ele.analysis == "ramberg-osgood hardening":
                ele.perform_ramberg_osgood_hardening()

        self.state_variable = u
        self.control_parameter = lam

    def update_member_data_2d_iteration(self, u, lam, *args):
        for iele, ele in enumerate(self.structure):
            ele.global_displacement = u[3 * iele: 3 * iele + 6]
            if ele.analysis == "linear hardening":
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

        self.state_variable = u
        self.control_parameter = lam

    def update_member_data_3d(self, u, lam, deltau):
        for iele, ele in enumerate(self.structure):
            ele.global_displacement = u[6 * iele: 6 * iele + 12]
            ele.current_orientation_node_1 = util.rodrigues(
                deltau[6 * iele + 3: 6 * iele + 6]) @ ele.current_orientation_node_1
            ele.current_orientation_node_2 = util.rodrigues(
                deltau[6 * iele + 9: 6 * iele + 12]) @ ele.current_orientation_node_2
            if ele.analysis == "linear hardening":
                ele.perform_linear_hardening()
        self.state_variable = u
        self.control_parameter = lam

    def update_member_data_3d_iteration(self, u, lam, deltau, *args):
        for iele, ele in enumerate(self.structure):
            ele.global_displacement = u[6 * iele: 6 * iele + 12]
            ele.current_orientation_node_1 = util.rodrigues(
                deltau[6 * iele + 3: 6 * iele + 6]) @ ele.current_orientation_node_1
            ele.current_orientation_node_2 = util.rodrigues(
                deltau[6 * iele + 9: 6 * iele + 12]) @ ele.current_orientation_node_2
            if ele.analysis == "linear hardening":
                ele.plastic_strain = args[0][iele]
                ele.internal_hardening_variable = args[1][iele]
                ele.perform_linear_hardening()

        self.state_variable = u
        self.control_parameter = lam

    def load_control(self, number_of_increments, tolerance, max_iteration_steps, load):
        for n in range(number_of_increments):
            K_s, F_s = self.modified_stiffness_force()
            Deltalam = load / number_of_increments
            Deltau = np.linalg.solve(
                K_s, Deltalam * self.load_increment_vector)

            u_pre = self.state_variable + Deltau
            lam_pre = self.control_parameter + Deltalam

            # update member data
            if self.dimension == 2:
                self.update_member_data_2d(u_pre, lam_pre)
            else:
                self.update_member_data_3d(u_pre, lam_pre, Deltau)

            u_temp = u_pre

            # initialize iteration counter
            kiter = 0

            K_s, F_s = self.modified_stiffness_force()
            residual = F_s - self.control_parameter * self.load_increment_vector
            residual_norm = np.linalg.norm(residual)

            if self.analysis != "elastic":
                if self.dimension == 2:
                    plastic_strain = []
                    internal_hardening_variable = []
                    back_stress = []
                    for ele in self.structure:
                        plastic_strain.append(ele.plastic_strain)
                        internal_hardening_variable.append(
                            ele.internal_hardening_variable)
                        back_stress.append(ele.back_stress)
                else:
                    plastic_strain = []
                    internal_hardening_variable = []
                    for ele in self.structure:
                        plastic_strain.append(ele.plastic_strain)
                        internal_hardening_variable.append(
                            ele.internal_hardening_variable)

            # iterate, until good result or so many iteration steps
            while(residual_norm > tolerance and kiter < max_iteration_steps):
                K_s, F_s = self.modified_stiffness_force()
                deltau = -np.linalg.solve(K_s, residual)
                u_temp += deltau

                if self.analysis != "elastic":
                    if self.dimension == 2:
                        self.update_member_data_2d_iteration(
                            u_temp, lam_pre, plastic_strain, internal_hardening_variable, back_stress)
                    else:
                        self.update_member_data_3d_iteration(
                            u_temp, lam_pre, deltau, plastic_strain, internal_hardening_variable)
                else:
                    if self.dimension == 2:
                        self.update_member_data_2d(u_temp, lam_pre)
                    else:
                        self.update_member_data_3d(u_temp, lam_pre, deltau)

                K_s, F_s = self.modified_stiffness_force()
                residual = F_s - self.control_parameter * self.load_increment_vector
                residual_norm = np.linalg.norm(residual)

                # update iterations counter
                kiter += 1

                print("Iteration step: " + str(kiter))
                print("residual norm: " + str(residual_norm))
                if(kiter == max_iteration_steps):
                    self.plot_equilibrium_path()
                    self.plot_the_structure()
                    raise RuntimeError(
                        'Newton-Raphson iterations did not converge!')

            self.state_variable_plot.append(
                float(self.state_variable[self.interesting_dof]))
            self.control_parameter_plot.append(self.control_parameter)
            print("Incrementation step: " + str(n + 1))

    def displacement_control(self, number_of_increments, tolerance, max_iteration_steps, displacement):
        for n in range(number_of_increments):
            K_s, F_s = self.modified_stiffness_force()
            velocity = np.linalg.solve(
                K_s, self.load_increment_vector)

            Deltalam = float(
                displacement / (number_of_increments * velocity[self.interesting_dof]))
            Deltau = np.linalg.solve(
                K_s, Deltalam * self.load_increment_vector)

            u_pre = self.state_variable + Deltau
            lam_pre = self.control_parameter + Deltalam

            # update member data
            if self.dimension == 2:
                self.update_member_data_2d(u_pre, lam_pre)
            else:
                self.update_member_data_3d(u_pre, lam_pre, Deltau)

            u_temp = u_pre
            lam_temp = lam_pre

            # initialize iteration counter
            kiter = 0

            K_s, F_s = self.modified_stiffness_force()
            residual = F_s - self.control_parameter * self.load_increment_vector
            residual_norm = np.linalg.norm(residual)

            if self.analysis != "elastic":
                if self.dimension == 2:
                    plastic_strain = []
                    internal_hardening_variable = []
                    back_stress = []
                    for ele in self.structure:
                        plastic_strain.append(ele.plastic_strain)
                        internal_hardening_variable.append(
                            ele.internal_hardening_variable)
                        back_stress.append(ele.back_stress)
                else:
                    plastic_strain = []
                    internal_hardening_variable = []
                    for ele in self.structure:
                        plastic_strain.append(ele.plastic_strain)
                        internal_hardening_variable.append(
                            ele.internal_hardening_variable)

            # iterate, until good result or so many iteration steps
            while(residual_norm > tolerance and kiter < max_iteration_steps):
                K_s, F_s = self.modified_stiffness_force()

                adjusted_stiffness_matrix = K_s.copy()
                adjusted_stiffness_matrix[:, self.interesting_dof] = - \
                    self.load_increment_vector.reshape(self.number_of_dofs)

                augmented_solution = - \
                    np.linalg.solve(adjusted_stiffness_matrix, residual)
                deltalam = float(
                    augmented_solution[self.interesting_dof])
                deltau = np.copy(augmented_solution)
                deltau[self.interesting_dof, 0] = 0.0

                u_temp += deltau
                lam_temp += deltalam

                if self.analysis != "elastic":
                    if self.dimension == 2:
                        self.update_member_data_2d_iteration(
                            u_temp, lam_temp, plastic_strain, internal_hardening_variable, back_stress)
                    else:
                        self.update_member_data_3d_iteration(
                            u_temp, lam_temp, deltau, plastic_strain, internal_hardening_variable)
                else:
                    if self.dimension == 2:
                        self.update_member_data_2d(
                            u_temp, lam_temp)
                    else:
                        self.update_member_data_3d(
                            u_temp, lam_temp, deltau)

                K_s, F_s = self.modified_stiffness_force()

                residual = F_s - self.control_parameter * self.load_increment_vector
                residual_norm = np.linalg.norm(residual)

                # update iterations counter
                kiter += 1

                print("Iteration step: " + str(kiter))
                print("residual norm: " + str(residual_norm))
                if(kiter == max_iteration_steps):
                    self.plot_equilibrium_path()
                    self.plot_the_structure()
                    raise RuntimeError(
                        'Newton-Raphson iterations did not converge!')

            self.state_variable_plot.append(
                float(self.state_variable[self.interesting_dof]))
            self.control_parameter_plot.append(self.control_parameter)
            print("Incrementation step: " + str(n + 1))

    def arc_length_control(self, number_of_increments, tolerance, max_iteration_steps, direction, arc_length):
        if direction == "positive":
            Deltau_prev = np.ones((self.number_of_dofs, 1), dtype=float)
        elif direction == "negative":
            Deltau_prev = -np.ones((self.number_of_dofs, 1), dtype=float)
        else:
            raise ValueError("Please input the right direction!")

        for n in range(number_of_increments):

            K_s, F_s = self.modified_stiffness_force()

            velocity = np.linalg.solve(K_s, self.load_increment_vector)
            Deltalam = float(np.sign(Deltau_prev.T @ velocity) *
                             arc_length / np.sqrt(velocity.T @ velocity))

            Deltau = np.linalg.solve(
                K_s, Deltalam * self.load_increment_vector)

            u_pre = self.state_variable + Deltau
            lam_pre = self.control_parameter + Deltalam

            # update member data
            if self.dimension == 2:
                self.update_member_data_2d(u_pre, lam_pre)
            else:
                self.update_member_data_3d(u_pre, lam_pre, Deltau)

            # initialize iteration counter
            kiter = 0

            K_s, F_s = self.modified_stiffness_force()
            residual = F_s - self.control_parameter * self.load_increment_vector
            residual_norm = np.linalg.norm(residual)

            u_temp = u_pre
            lam_temp = lam_pre

            if self.analysis != "elastic":
                if self.dimension == 2:
                    plastic_strain = []
                    internal_hardening_variable = []
                    back_stress = []
                    for ele in self.structure:
                        plastic_strain.append(ele.plastic_strain)
                        internal_hardening_variable.append(
                            ele.internal_hardening_variable)
                        back_stress.append(ele.back_stress)
                else:
                    plastic_strain = []
                    internal_hardening_variable = []
                    for ele in self.structure:
                        plastic_strain.append(ele.plastic_strain)
                        internal_hardening_variable.append(
                            ele.internal_hardening_variable)

            deltau = np.zeros((self.number_of_dofs, 1), dtype=float)
            deltalam = 0.

            # iterate, until good result or so many iteration steps
            while(residual_norm > tolerance and kiter < max_iteration_steps):
                K_s, F_s = self.modified_stiffness_force()

                deltaustar = -np.linalg.solve(K_s, residual)
                velocity = np.linalg.solve(K_s, self.load_increment_vector)
                Deltau += deltau
                a = float(velocity.T @ velocity)
                b = float(2 * (Deltau + deltaustar).T @ velocity)
                c = float((Deltau + deltaustar).T @
                          (Deltau + deltaustar) - arc_length ** 2)

                deltalam1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
                deltalam2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)

                dotprod1 = float(
                    (Deltau + deltaustar + deltalam1 * velocity).T @ Deltau)
                dotprod2 = float(
                    (Deltau + deltaustar + deltalam2 * velocity).T @ Deltau)

                if dotprod1 >= dotprod2:
                    deltalam = deltalam1
                else:
                    deltalam = deltalam2

                deltau = deltaustar + deltalam * velocity

                u_temp += deltau
                lam_temp += deltalam

                # update member data
                if self.analysis != "elastic":
                    if self.dimension == 2:
                        self.update_member_data_2d_iteration(
                            u_temp, lam_temp, plastic_strain, internal_hardening_variable, back_stress)
                    else:
                        self.update_member_data_3d_iteration(
                            u_temp, lam_temp, deltau, plastic_strain, internal_hardening_variable)
                else:
                    if self.dimension == 2:
                        self.update_member_data_2d(
                            u_temp, lam_temp)
                    else:
                        self.update_member_data_3d(u_temp, lam_temp, deltau)

                K_s, F_s = self.modified_stiffness_force()
                residual = F_s - self.control_parameter * self.load_increment_vector
                residual_norm = np.linalg.norm(residual)

                # update iterations counter
                kiter += 1
                print("Iteration step: " + str(kiter))
                print("residual norm: " + str(residual_norm))
                if(kiter == max_iteration_steps):
                    self.plot_equilibrium_path()
                    self.plot_the_structure()
                    raise RuntimeError(
                        'Newton-Raphson iterations did not converge!')

            """
            ------------------------------------------------------------------
            3. Update variables to their final value for the current increment
            ------------------------------------------------------------------
            """
            Deltau_prev = Deltau + deltau

            self.state_variable_plot.append(
                float(self.state_variable[self.interesting_dof]))
            self.control_parameter_plot.append(self.control_parameter)
            print("Incrementation step: " + str(n + 1))

    # def arc_length_control(self, number_of_increments, tolerance, max_iteration_steps, direction, arc_length):
    #     if direction == "positive":
    #         Deltau_prev = np.ones((self.number_of_dofs, 1), dtype=float)
    #     elif direction == "negative":
    #         Deltau_prev = -np.ones((self.number_of_dofs, 1), dtype=float)
    #     else:
    #         raise ValueError("Please input the right direction!")

    #     for n in range(number_of_increments):

    #         K_s, F_s = self.modified_stiffness_force()

    #         velocity = np.linalg.solve(K_s, self.load_increment_vector)
    #         scaling_factor = float(np.sqrt(1 + velocity.T @ velocity))
    #         Deltalam = np.sign(float(Deltau_prev.T @ velocity)) * \
    #             arc_length / scaling_factor

    #         Deltau = np.linalg.solve(
    #             K_s, Deltalam * self.load_increment_vector)

    #         u_pre = self.state_variable + Deltau
    #         lam_pre = self.control_parameter + Deltalam

    #         # update member data
    #         if self.dimension == 2:
    #             self.update_member_data_2d(u_pre, lam_pre)
    #         else:
    #             self.update_member_data_3d(u_pre, lam_pre, Deltau)

    #         # initialize iteration counter
    #         kiter = 0

    #         K_s, F_s = self.modified_stiffness_force()
    #         residual = F_s - self.control_parameter * self.load_increment_vector
    #         residual_norm = np.linalg.norm(residual)

    #         u_temp = u_pre
    #         lam_temp = lam_pre

    #         if self.analysis != "elastic":
    #             if self.dimension == 2:
    #                 plastic_strain = []
    #                 internal_hardening_variable = []
    #                 back_stress = []
    #                 for ele in self.structure:
    #                     plastic_strain.append(ele.plastic_strain)
    #                     internal_hardening_variable.append(
    #                         ele.internal_hardening_variable)
    #                     back_stress.append(ele.back_stress)
    #             else:
    #                 plastic_strain = []
    #                 internal_hardening_variable = []
    #                 for ele in self.structure:
    #                     plastic_strain.append(ele.plastic_strain)
    #                     internal_hardening_variable.append(
    #                         ele.internal_hardening_variable)

    #         deltau = np.zeros((self.number_of_dofs, 1), dtype=float)
    #         deltalam = 0.

    #         # iterate, until good result or so many iteration steps
    #         while(residual_norm > tolerance and kiter < max_iteration_steps):
    #             K_s, F_s = self.modified_stiffness_force()
    #             velocity = np.linalg.solve(K_s, self.load_increment_vector)

    #             scaling_factor = float(np.sqrt(1 + velocity.T @ velocity))

    #             constraint = 1/scaling_factor * \
    #                 np.abs(velocity.T @ (Deltau + deltau) +
    #                        (Deltalam + deltalam)) - arc_length

    #             augmented_stiffness = np.r_[np.c_[K_s, -self.load_increment_vector],
    #                                         np.c_[velocity.T/scaling_factor, 1/scaling_factor]]

    #             augmented_force = -np.r_[residual, constraint]

    #             augmented_displacement = np.linalg.solve(
    #                 augmented_stiffness, augmented_force)

    #             deltau = augmented_displacement[0: -1]
    #             deltalam = float(augmented_displacement[-1])

    #             u_temp += deltau
    #             lam_temp += deltalam

    #             # update member data
    #             if self.analysis != "elastic":
    #                 if self.dimension == 2:
    #                     self.update_member_data_2d_iteration(
    #                         u_temp, lam_temp, plastic_strain, internal_hardening_variable, back_stress)
    #                 else:
    #                     self.update_member_data_3d_iteration(
    #                         u_temp, lam_temp, deltau, plastic_strain, internal_hardening_variable)
    #             else:
    #                 if self.dimension == 2:
    #                     self.update_member_data_2d(
    #                         u_temp, lam_temp)
    #                 else:
    #                     self.update_member_data_3d(u_temp, lam_temp, deltau)

    #             K_s, F_s = self.modified_stiffness_force()
    #             residual = F_s - self.control_parameter * self.load_increment_vector
    #             residual_norm = np.linalg.norm(residual)

    #             # update iterations counter
    #             kiter += 1
    #             print("Iteration step: " + str(kiter))
    #             print("residual norm: " + str(residual_norm))
    #             if(kiter == max_iteration_steps):
    #                 self.plot_equilibrium_path()
    #                 self.plot_the_structure()
    #                 raise RuntimeError(
    #                     'Newton-Raphson iterations did not converge!')

    #         """
    #         ------------------------------------------------------------------
    #         3. Update variables to their final value for the current increment
    #         ------------------------------------------------------------------
    #         """
    #         Deltau_prev = Deltau + deltau

    #         self.state_variable_plot.append(
    #             float(self.state_variable[self.interesting_dof]))
    #         self.control_parameter_plot.append(self.control_parameter)
    #         print("Incrementation step: " + str(n + 1))

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
            ax.plot(self.state_variable_plot,
                    self.control_parameter_plot, '.-')
        elif horizontal_flip == True and vertical_flip == False:
            invstate_variable_plot = [-u for u in self.state_variable_plot]
            ax.plot(invstate_variable_plot,
                    self.control_parameter_plot, '.-')
        elif horizontal_flip == False and vertical_flip == True:
            invcontrol_parameter_plot = [
                -lam for lam in self.control_parameter_plot]
            ax.plot(self.state_variable_plot, -
                    invcontrol_parameter_plot, '.-')
        elif horizontal_flip == True and vertical_flip == True:
            invstate_variable_plot = [-u for u in self.state_variable_plot]
            invcontrol_parameter_plot = [
                -lam for lam in self.control_parameter_plot]
            ax.plot(invstate_variable_plot, invcontrol_parameter_plot, '.-')

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

        if self.dimension == 2:
            X = np.zeros((self.number_of_nodes))
            Z = np.zeros((self.number_of_nodes))
            x = np.zeros((self.number_of_nodes))
            z = np.zeros((self.number_of_nodes))
            for iele in range(self.number_of_elements):
                X[iele] = self.structure[iele].initial_coordinate_node_1[0]
                Z[iele] = self.structure[iele].initial_coordinate_node_1[1]
                x[iele] = self.structure[iele].current_coordinate_node_1()[0]
                z[iele] = self.structure[iele].current_coordinate_node_1()[1]
                if iele == self.number_of_elements - 1:
                    X[iele + 1] = self.structure[iele].initial_coordinate_node_2[0]
                    Z[iele + 1] = self.structure[iele].initial_coordinate_node_2[1]
                    x[iele + 1] = self.structure[iele].current_coordinate_node_2()[0]
                    z[iele + 1] = self.structure[iele].current_coordinate_node_2()[1]
            # Plot both configurations()
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
            X = np.zeros((self.number_of_nodes))
            Y = np.zeros((self.number_of_nodes))
            Z = np.zeros((self.number_of_nodes))
            x = np.zeros((self.number_of_nodes))
            y = np.zeros((self.number_of_nodes))
            z = np.zeros((self.number_of_nodes))
            for iele in range(self.number_of_elements):
                X[iele] = self.structure[iele].initial_coordinate_node_1[0]
                Y[iele] = self.structure[iele].initial_coordinate_node_1[1]
                Z[iele] = self.structure[iele].initial_coordinate_node_1[2]
                x[iele] = self.structure[iele].current_coordinate_node_1()[0]
                y[iele] = self.structure[iele].current_coordinate_node_1()[1]
                z[iele] = self.structure[iele].current_coordinate_node_1()[2]
                if iele == self.number_of_elements - 1:
                    X[iele + 1] = self.structure[iele].initial_coordinate_node_2[0]
                    Y[iele + 1] = self.structure[iele].initial_coordinate_node_2[1]
                    Z[iele + 1] = self.structure[iele].initial_coordinate_node_2[2]
                    x[iele + 1] = self.structure[iele].current_coordinate_node_2()[0]
                    y[iele + 1] = self.structure[iele].current_coordinate_node_2()[1]
                    z[iele + 1] = self.structure[iele].current_coordinate_node_2()[2]
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

    def output_equilibrium_path(self, file_name):
        u = self.state_variable_plot
        lam = self.control_parameter_plot
        file = open(file_name + ".txt", "w")
        file.write("u" + "\t" + "lambda" + "\n")
        for i in range(len(u)):
            s = str(u[i]) + "\t" + str(lam[i]) + "\n"
            file.write(s)
        file.close()
        print("The equilibrium path has successfully saved.")
