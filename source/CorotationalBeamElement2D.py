# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou
"""

import numpy as np

from numpy import cos, sin
from math import atan2


class CorotationalBeamElement2D():
    """
    It is a class formulating two-dimensional corotational beam elements.

    """

    def __init__(self, beamtype, youngs_modulus, poisson_ratio, width, height, initial_coordinate_node_1, initial_coordinate_node_2, element_id):
        """
        To initialize a 2D coroational beam element, the nodal coordinates in
        undeformed configuration (X1, Y1) and (X2, Y2), and of course, Young's
        modulus, are necessary.

        The solver can deal with rectangular cross section. In elastic
        analysis, it is enough knowing cross-sectional area 'A' and moment of
        inertia 'I'. However, in elasto-plasticity, we need to know the stress
        at some certain points. The stress depends on the shape of the beam,
        so instead of A and I, the user should input the width 'b' and the
        height 'h' of the beam, then A = b * h and I = b * h ^ 3 / 12.

        As for elasto-plasticity, only linear isotropic hardening is
        implemented. In this case yield stress and plastic modulus must be
        considered, because local force and local material stiffness matrix
        cannot be directly calculated(due to unhomogeneous stress distribution),
        Gauss integration is adopted here. The user can input his/her number
        of Gauss locations, or use the default value from this program.


        Returns
        -------
        None.

        """
        self.initial_coordinate_node_1 = initial_coordinate_node_1
        self.initial_coordinate_node_2 = initial_coordinate_node_2
        self.incremental_global_displacement = np.zeros((6, 1), dtype=float)

        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        self.width = width
        self.height = height

        self.initial_length = self.calculate_initial_length()
        self.initial_angle = self.calculate_initial_angle()

        self.area = self.calculate_area()
        self.moment_of_inertia = self.calculate_moment_of_inertia()

        self.analysis = "elastic"
        self.beamtype = beamtype

        self.element_freedom_table = np.linspace(
            3 * element_id, 3 * element_id + 5, num=6, dtype=int)

    def calculate_area(self):
        """
        Set the cross-sectional area of the beam element.

        A = b * h

        """
        return self.width * self.height

    def calculate_moment_of_inertia(self):
        """
        Set the moment of inertia of the beam element.

        I = 1/12 * b * h^3

        """
        return 1/12 * self.width * (self.height) ** 3

    def current_coordinate_node_1(self):
        """
        Get the coodinate of node 1 in current configuration.

        x1 = X1 + u1

        """
        return self.initial_coordinate_node_1 + self.incremental_global_displacement[0: 2]

    def current_coordinate_node_2(self):
        """
        Get the coodinate of node 2 in current configuration.

        x2 = X2 + u2

        """
        return self.initial_coordinate_node_2 + self.incremental_global_displacement[3: 5]

    def apply_hardening_model(self, hardening_model, gauss_number, yield_stress, kinematic_hardening_modulus, plastic_modulus, saturation_stress,
                              modified_modulus, exponent):

        self.num_of_gauss_locations_xi = gauss_number[0]
        self.num_of_gauss_locations_eta = gauss_number[1]

        self.num_of_gauss_locations = self.num_of_gauss_locations_xi * \
            self.num_of_gauss_locations_eta

        [gauss_locations_xi, weights_xi] = np.polynomial.legendre.leggauss(
            self.num_of_gauss_locations_xi)
        [gauss_locations_eta, weights_eta] = np.polynomial.legendre.leggauss(
            self.num_of_gauss_locations_eta)

        global_gauss_locations_mat = np.zeros(
            (2, self.num_of_gauss_locations), dtype=float)

        weights_mat = np.zeros(self.num_of_gauss_locations, dtype=float)

        for ixi in range(self.num_of_gauss_locations_xi):
            for jeta in range(self.num_of_gauss_locations_eta):
                x, z = self.map_local_coordinate_to_global_coordinate(
                    gauss_locations_xi[ixi], gauss_locations_eta[jeta])

                global_gauss_locations_mat[0, ixi *
                                           self.num_of_gauss_locations_eta + jeta] = x
                global_gauss_locations_mat[1, ixi *
                                           self.num_of_gauss_locations_eta + jeta] = z
                weights_mat[ixi * self.num_of_gauss_locations_eta +
                            jeta] = weights_xi[ixi] * weights_eta[jeta]

        self.global_gauss_locations_mat = global_gauss_locations_mat
        self.weights_mat = weights_mat

        if hardening_model == "linear hardening":

            self.analysis = hardening_model
            self.yield_stress = yield_stress
            self.kinematic_hardening_modulus = kinematic_hardening_modulus
            self.plastic_modulus = plastic_modulus

            self.stress = np.zeros(self.num_of_gauss_locations, dtype=float)
            self.plastic_strain = np.zeros(
                self.num_of_gauss_locations, dtype=float)
            self.back_stress = np.zeros(
                self.num_of_gauss_locations, dtype=float)
            self.tangent_modulus = self.youngs_modulus * \
                np.ones(self.num_of_gauss_locations, dtype=float)
            self.internal_hardening_variable = np.zeros(
                self.num_of_gauss_locations, dtype=float)

        elif hardening_model == "exponential hardening":

            self.analysis = hardening_model
            self.yield_stress = yield_stress
            self.kinematic_hardening_modulus = kinematic_hardening_modulus
            self.saturation_stress = saturation_stress
            self.exponent = exponent

            self.stress = np.zeros(self.num_of_gauss_locations, dtype=float)
            self.plastic_strain = np.zeros(
                self.num_of_gauss_locations, dtype=float)
            self.back_stress = np.zeros(
                self.num_of_gauss_locations, dtype=float)
            self.tangent_modulus = self.youngs_modulus * \
                np.ones(self.num_of_gauss_locations, dtype=float)
            self.internal_hardening_variable = np.zeros(
                self.num_of_gauss_locations, dtype=float)

        elif hardening_model == "ramberg-osgood hardening":

            self.analysis = hardening_model
            self.yield_stress = yield_stress
            self.kinematic_hardening_modulus = kinematic_hardening_modulus
            self.modified_modulus = modified_modulus
            self.exponent = exponent

            self.stress = np.zeros(self.num_of_gauss_locations, dtype=float)
            self.plastic_strain = np.zeros(
                self.num_of_gauss_locations, dtype=float)
            self.back_stress = np.zeros(
                self.num_of_gauss_locations, dtype=float)
            self.tangent_modulus = self.youngs_modulus * \
                np.ones(self.num_of_gauss_locations, dtype=float)
            self.internal_hardening_variable = np.zeros(
                self.num_of_gauss_locations, dtype=float)

        else:

            print("Three hardening models are implemented here: 1. linear hardening; 2. exponential hardening; 3. ramberg-osgood hardening.")
            raise ValueError("Wrong hardening model!")

    def local_displacement(self):

        global_rotation_node_1 = float(self.incremental_global_displacement[2])
        global_rotation_node_2 = float(self.incremental_global_displacement[5])

        u_bar = self.current_length() - self.initial_length
        beta = self.current_angle()

        theta1_bar = np.arctan((cos(beta) * sin(global_rotation_node_1 + self.initial_angle)
                            - sin(beta) * cos(global_rotation_node_1 + self.initial_angle))/
                           (cos(beta) * cos(global_rotation_node_1 + self.initial_angle)
                            + sin(beta) * sin(global_rotation_node_1 + self.initial_angle)))

        theta2_bar = np.arctan((cos(beta) * sin(global_rotation_node_2 + self.initial_angle)
                            - sin(beta) * cos(global_rotation_node_2 + self.initial_angle))/
                           (cos(beta) * cos(global_rotation_node_2 + self.initial_angle)
                            + sin(beta) * sin(global_rotation_node_2 + self.initial_angle)))

        return np.array([[u_bar, theta1_bar, theta2_bar]]).T

    def calculate_initial_length(self):
        return np.linalg.norm(self.initial_coordinate_node_2 -
                              self.initial_coordinate_node_1)

    def calculate_initial_angle(self):
        return atan2(self.initial_coordinate_node_2[1] -
                     self.initial_coordinate_node_1[1],
                     self.initial_coordinate_node_2[0] -
                     self.initial_coordinate_node_1[0])

    def current_length(self):
        return np.linalg.norm(self.current_coordinate_node_2() -
                              self.current_coordinate_node_1())

    def current_angle(self):
        x1 = self.current_coordinate_node_1()
        x2 = self.current_coordinate_node_2()
        return atan2(x2[1] - x1[1], x2[0] - x1[0])

    def local_stiffness_force(self):
        if self.analysis == "elastic":
            k_l = self.youngs_modulus / self.initial_length * \
                np.array([[self.area, 0., 0.],
                          [0., 4 * self.moment_of_inertia,
                              2 * self.moment_of_inertia],
                          [0., 2 * self.moment_of_inertia, 4 * self.moment_of_inertia]])
            f_l = k_l @ self.local_displacement()

        else:
            kl_11, kl_22, kl_33, kl_12, kl_13, kl_23, N, M1, M2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            for n in range(self.num_of_gauss_locations):
                x, z = self.global_gauss_locations_mat[0: 2, n]
                fac = self.width * self.height * self.initial_length / 4

                kl_11 += fac * self.weights_mat[n] * \
                    self.tangent_modulus[n] / \
                    (self.initial_length) ** 2
                kl_22 += fac * self.weights_mat[n] * self.tangent_modulus[n] * z ** 2 * (4 / self.initial_length - 6 *
                                                                                         x / (self.initial_length) ** 2) ** 2
                kl_33 += fac * self.weights_mat[n] * self.tangent_modulus[n] * z ** 2 * (2 / self.initial_length - 6 *
                                                                                         x / (self.initial_length) ** 2) ** 2
                kl_12 += fac * self.weights_mat[n] * self.tangent_modulus[n] * z * (
                    4 / (self.initial_length) ** 2 - 6 * x / (self.initial_length) ** 3)
                kl_13 += fac * self.weights_mat[n] * self.tangent_modulus[n] * z * (
                    2 / (self.initial_length) ** 2 - 6 * x / (self.initial_length) ** 3)
                kl_23 += fac * self.weights_mat[n] * self.tangent_modulus[n] * z ** 2 * (4 / self.initial_length - 6 * x / (
                    self.initial_length) ** 2) * (2 / self.initial_length - 6 * x / (self.initial_length) ** 2)
                N += fac * self.weights_mat[n] * \
                    self.stress[n] / self.initial_length

                M1 += fac * self.weights_mat[n] * self.stress[n] * z * (
                    4 / self.initial_length - 6 * x / (self.initial_length) ** 2)

                M2 += fac * self.weights_mat[n] * self.stress[n] * z * (
                    2 / self.initial_length - 6 * x / (self.initial_length) ** 2)

            k_l = np.array([[kl_11, kl_12, kl_13],
                            [kl_12, kl_22, kl_23],
                            [kl_13, kl_23, kl_33]])

            f_l = np.array([[N, M1, M2]]).T

        return k_l, f_l

    def map_local_coordinate_to_global_coordinate(self, xi, eta):
        x = self.initial_length / 2 * (xi + 1)
        z = self.height / 2 * eta

        return x, z

    def strain(self, x, z):
        pl = self.local_displacement()
        return float(pl[0] / self.initial_length + z * ((4 / self.initial_length - 6 *
                                                         x / (self.initial_length) ** 2) * pl[1] +
                                                        (2 / self.initial_length - 6 *
                                                         x / (self.initial_length) ** 2) * pl[2]))

    def perform_linear_hardening(self):

        def yield_stress_function(
            alpha): return self.yield_stress + self.plastic_modulus * alpha

        for n in range(self.num_of_gauss_locations):
            x, z = self.global_gauss_locations_mat[0: 2, n]

            stress_trial = self.youngs_modulus * \
                (self.strain(x, z) - self.plastic_strain[n])
            relative_stress_trial = stress_trial - \
                self.back_stress[n]
            yield_condition_trial = np.abs(
                relative_stress_trial) - yield_stress_function(self.internal_hardening_variable[n])

            if yield_condition_trial <= 0:
                self.stress[n] = stress_trial
                self.tangent_modulus[n] = self.youngs_modulus
            else:
                deltagamma = yield_condition_trial / \
                    (self.youngs_modulus + self.plastic_modulus +
                        self.kinematic_hardening_modulus)
                self.stress[n] = stress_trial - deltagamma * \
                    self.youngs_modulus * np.sign(relative_stress_trial)
                self.plastic_strain[n] += deltagamma * \
                    np.sign(relative_stress_trial)
                self.back_stress[n] += deltagamma * \
                    self.kinematic_hardening_modulus * \
                    np.sign(relative_stress_trial)
                self.internal_hardening_variable[n] += deltagamma
                self.tangent_modulus[n] = self.youngs_modulus * (self.plastic_modulus + self.kinematic_hardening_modulus) / (
                    self.youngs_modulus + self.plastic_modulus + self.kinematic_hardening_modulus)

    def perform_exponential_hardening(self):

        def yield_stress_function(alpha): return self.yield_stress + (
            self.saturation_stress - self.yield_stress) * (1 - np.exp(-self.exponent * alpha))

        hardening_limit_stress = self.saturation_stress - self.yield_stress

        for n in range(self.num_of_gauss_locations):
            x, z = self.global_gauss_locations_mat[0: 2, n]

            stress_trial = self.youngs_modulus * \
                (float(self.strain(x, z)) -
                    self.plastic_strain[n])
            relative_stress_trial = stress_trial - \
                self.back_stress[n]
            yield_condition_trial = np.abs(
                relative_stress_trial) - yield_stress_function(self.internal_hardening_variable[n])

            if yield_condition_trial <= 0:
                self.stress[n] = stress_trial
                self.tangent_modulus[n] = self.youngs_modulus
            else:
                deltagamma = 0
                residual = yield_condition_trial - deltagamma * (self.youngs_modulus + self.kinematic_hardening_modulus) - yield_stress_function(
                    self.internal_hardening_variable[n] + deltagamma) + yield_stress_function(self.internal_hardening_variable[n])

                maxiter = 10
                iteration_counter = 0
                tolerance = 1.0e-5

                while residual > tolerance and iteration_counter < maxiter:
                    dR_ddeltagamma = -(self.youngs_modulus + self.kinematic_hardening_modulus) - hardening_limit_stress * self.exponent * \
                        np.exp(-self.exponent *
                               (self.internal_hardening_variable[n] + deltagamma))
                    d_g = - residual / dR_ddeltagamma
                    deltagamma += d_g
                    residual = yield_condition_trial - deltagamma * (self.youngs_modulus + self.kinematic_hardening_modulus) - yield_stress_function(
                        self.internal_hardening_variable[n] + deltagamma) + yield_stress_function(self.internal_hardening_variable[n])
                    iteration_counter += 1

                self.stress[n] = stress_trial - deltagamma * \
                    self.youngs_modulus * np.sign(relative_stress_trial)
                self.plastic_strain[n] += deltagamma * \
                    np.sign(relative_stress_trial)
                self.back_stress[n] += deltagamma * \
                    self.kinematic_hardening_modulus * \
                    np.sign(relative_stress_trial)
                self.internal_hardening_variable[n] += deltagamma
                self.tangent_modulus[n] = self.youngs_modulus * (self.kinematic_hardening_modulus + hardening_limit_stress * self.exponent * np.exp(-self.exponent * self.internal_hardening_variable[n])) / (
                    self.youngs_modulus + self.kinematic_hardening_modulus + hardening_limit_stress * self.exponent * np.exp(-self.exponent * self.internal_hardening_variable[n]))

    def perform_ramberg_osgood_hardening(self):

        def yield_stress_function(alpha): return self.yield_stress + \
            self.modified_modulus * np.power(alpha, self.exponent)

        for n in range(self.num_of_gauss_locations):
            x, z = self.global_gauss_locations_mat[0: 2, n]
            stress_trial = self.youngs_modulus * \
                (float(self.strain(x, z)) -
                    self.plastic_strain[n])
            relative_stress_trial = stress_trial - \
                self.back_stress[n]
            yield_condition_trial = np.abs(
                relative_stress_trial) - yield_stress_function(self.internal_hardening_variable[n])

            if yield_condition_trial <= 0:
                self.stress[n] = stress_trial
                self.tangent_modulus[n] = self.youngs_modulus
            else:
                deltagamma = 0
                residual = yield_condition_trial - deltagamma * (self.youngs_modulus + self.kinematic_hardening_modulus) - yield_stress_function(
                    self.internal_hardening_variable[n] + deltagamma) + yield_stress_function(self.internal_hardening_variable[n])

                maxiter = 10
                iteration_counter = 0
                tolerance = 1.0e-5

                while residual > tolerance and iteration_counter < maxiter:
                    dR_ddeltagamma = -(self.youngs_modulus + self.kinematic_hardening_modulus) - self.exponent * self.modified_modulus * \
                        np.power(
                            self.internal_hardening_variable[n] + deltagamma, self.exponent - 1)
                    d_g = - residual / dR_ddeltagamma
                    deltagamma += d_g
                    residual = yield_condition_trial - deltagamma * (self.youngs_modulus + self.kinematic_hardening_modulus) - yield_stress_function(
                        self.internal_hardening_variable[n] + deltagamma) + yield_stress_function(self.internal_hardening_variable[n])
                    iteration_counter += 1

                self.stress[n] = stress_trial - deltagamma * \
                    self.youngs_modulus * np.sign(relative_stress_trial)
                self.plastic_strain[n] += deltagamma * \
                    np.sign(relative_stress_trial)
                self.back_stress[n] += deltagamma * \
                    self.kinematic_hardening_modulus * \
                    np.sign(relative_stress_trial)
                self.internal_hardening_variable[n] += deltagamma
                self.tangent_modulus[n] = self.youngs_modulus * (self.kinematic_hardening_modulus + self.exponent * self.modified_modulus * np.power(self.internal_hardening_variable[n], self.exponent - 1)) / (
                    self.youngs_modulus + self.kinematic_hardening_modulus + self.exponent * self.modified_modulus * np.power(self.internal_hardening_variable[n], self.exponent - 1))

    def global_stiffness_force(self):
        
        beta = self.current_angle()
        l = self.current_length()
        
        B = np.array([[-cos(beta), -sin(beta), 0., cos(beta), sin(beta), 0.],
                      [-sin(beta) / l, cos(beta)/l, 1., sin(beta)/l, -cos(beta)/l, 0.],
                      [-sin(beta) / l, cos(beta)/l, 0., sin(beta)/l, -cos(beta)/l, 1.]])
        r = np.array(
            [[-cos(beta), -sin(beta), 0.,  cos(beta), sin(beta), 0.]]).T
        z = np.array(
            [[sin(beta), -cos(beta), 0., -sin(beta), cos(beta), 0.]]).T
        
        K_l, f_l = self.local_stiffness_force()
        
        K_m = f_l[0] / l * z @ z.T
        + (f_l[1] + f_l[2]) / (l ** 2) * (r @ z.T + z @ r.T)
        
        K_g = B.T @ K_l @ B + K_m
        f_g = B.T @ f_l
        
        return K_g, f_g
