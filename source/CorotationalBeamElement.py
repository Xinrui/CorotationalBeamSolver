# -*- coding: utf-8 -*-
"""
@author: Yutingk Wang
"""

from math import atan2

import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin  # , tan
from scipy.linalg import expm, logm

import source.Utilities as util


class CorotationalBeamElement2D():
    """
    It is a class formulating two-dimensional corotational beam elements.

    """

    def __init__(self):
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
        self.__initial_coordinate_node_1 = None
        self.__initial_coordinate_node_2 = None

        self.__width = None
        self.__height = None

        self.__analysis = None
        self.__beamtype = "Bernoulli"

        self.__youngs_modulus = None

        self.__element_freedom_table = None

        self.__incremental_global_displacement = np.zeros((6, 1), dtype=float)
        self.__num_of_gauss_locations_ksi = 3
        self.__num_of_gauss_locations_eta = 3

        # All plastic cases, including Perfect Plasticity / No Hardening
        self.__yield_stress = None

        # Linear Hardening
        self.__plastic_modulus = None

        # Quadratic Hardening
        self.__quadratic_coefficient = None

        # Exponential Hardening
        self.__saturation_stress = None

        # Ramberg-Osgood Hardening
        self.__modified_modulus = None

        # "exponent" appears in both Exponential Hardening and Ramberg-Osgood 
        # Hardening with different physical meaning
        self.__exponent = None

    @property
    def initial_coordinate_node_1(self):
        return self.__initial_coordinate_node_1

    @initial_coordinate_node_1.setter
    def initial_coordinate_node_1(self, val):
        """
        Set the initial coordinate of node 1: [X1, Y1].T

        Raises:
            TypeError: If value is not a numpy array.

        """

        if isinstance(val, np.ndarray):
            self.__initial_coordinate_node_1 = val
        else:
            raise TypeError(
                "The initial coordinate of node 1 must be a 2x1 array!")

    @property
    def initial_coordinate_node_2(self):
        return self.__initial_coordinate_node_2

    @initial_coordinate_node_2.setter
    def initial_coordinate_node_2(self, val):
        """
        Set the initial coordinate of node 2: [X2, Y2].T

        Raises:
            TypeError: If value is not a numpy array.

        """

        if isinstance(val, np.ndarray):
            self.__initial_coordinate_node_2 = val
        else:
            raise TypeError(
                "The initial coordinate of node 2 must be a 2x1 array!")

    @property
    def incremental_global_displacement(self):
        return self.__incremental_global_displacement

    @incremental_global_displacement.setter
    def incremental_global_displacement(self, val):
        """
        Set the global displacement of element.

        Raises:
            TypeError: If value is not a numpy array.

        """
        if isinstance(val, np.ndarray):
            self.__incremental_global_displacement = val
        else:
            raise TypeError(
                "incremental global displacement must be a 6x1 array!")

    @property
    def current_coordinate_node_1(self):
        """
        Get the coodinate of node 1 in current configuration.

        x1 = X1 + u1

        """
        return self.__initial_coordinate_node_1 + self.__incremental_global_displacement[0: 2]

    @property
    def current_coordinate_node_2(self):
        """
        Get the coodinate of node 2 in current configuration.

        x2 = X2 + u2

        """
        return self.__initial_coordinate_node_2 + self.__incremental_global_displacement[3: 5]

    @property
    def global_rotation_node_1(self):
        """
        Get the global nodal rotation θ1.

        """
        return self.__incremental_global_displacement[2]

    @property
    def global_rotation_node_2(self):
        """
        Get the global nodal rotation θ2.

        """
        return self.__incremental_global_displacement[5]

    @property
    def youngs_modulus(self):
        return self.__youngs_modulus

    @youngs_modulus.setter
    def youngs_modulus(self, val):
        """
        Set the Young's modulus of the beam element.

        Raises:
            TypeError: If value is not a float number.
            ValueError: If value is not positive.

        """
        if not isinstance(val, float):
            raise TypeError("Young's modulus must be a float number!")
        elif val <= 0:
            raise ValueError("Young's modulus must be positive!")
        else:
            self.__youngs_modulus = val

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, val):
        """
        Set the width of the beam element.

        Raises:
            TypeError: If value is not a float number.
            ValueError: If value is not positive.

        """
        if not isinstance(val, float):
            raise TypeError("The width of beam must be a float number!")
        elif val <= 0:
            raise ValueError("The width of beam must be positive!")
        else:
            self.__width = val

    @property
    def height(self):
        return self.__height

    @height.setter
    def height(self, val):
        """
        Set the height of the beam element.

        Raises:
            TypeError: If value is not a float number.
            ValueError: If value is not positive.

        """
        if not isinstance(val, float):
            raise TypeError("The height of beam must be a float number!")
        elif val <= 0:
            raise ValueError("The height of beam must be positive!")
        else:
            self.__height = val

    @property
    def analysis(self):
        return self.__analysis

    @analysis.setter
    def analysis(self, val):
        if val == "elastic":
            self.__analysis = val
        elif val == "perfect plasticity":
            self.__analysis = val
            self.__stress = np.zeros(
                (self.__num_of_gauss_locations_ksi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__plastic_strain = np.zeros(
                (self.__num_of_gauss_locations_ksi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__tangent_modulus = self.__youngs_modulus * \
                np.ones((self.__num_of_gauss_locations_ksi,
                        self.__num_of_gauss_locations_eta), dtype=float)
        elif val == "linear hardening":
            self.__analysis = val
            self.__stress = np.zeros(
                (self.__num_of_gauss_locations_ksi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__plastic_strain = np.zeros(
                (self.__num_of_gauss_locations_ksi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__tangent_modulus = self.__youngs_modulus * \
                np.ones((self.__num_of_gauss_locations_ksi,
                        self.__num_of_gauss_locations_eta), dtype=float)
            self.__internal_hardening_variable = np.zeros(
                (self.__num_of_gauss_locations_ksi, self.__num_of_gauss_locations_eta), dtype=float)
        elif val == "quadratic hardening":
            self.__analysis = val
            self.__stress = np.zeros(
                (self.__num_of_gauss_locations_ksi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__plastic_strain = np.zeros(
                (self.__num_of_gauss_locations_ksi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__tangent_modulus = self.__youngs_modulus * \
                np.ones((self.__num_of_gauss_locations_ksi,
                        self.__num_of_gauss_locations_eta), dtype=float)
            self.__internal_hardening_variable = np.zeros(
                (self.__num_of_gauss_locations_ksi, self.__num_of_gauss_locations_eta), dtype=float)
        elif val == "exponential hardening":
            self.__analysis = val
            self.__stress = np.zeros(
                (self.__num_of_gauss_locations_ksi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__plastic_strain = np.zeros(
                (self.__num_of_gauss_locations_ksi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__tangent_modulus = self.__youngs_modulus * \
                np.ones((self.__num_of_gauss_locations_ksi,
                        self.__num_of_gauss_locations_eta), dtype=float)
            self.__internal_hardening_variable = np.zeros(
                (self.__num_of_gauss_locations_ksi, self.__num_of_gauss_locations_eta), dtype=float)
        elif val == "ramberg-osgood hardening":
            self.__analysis = val
            self.__stress = np.zeros(
                (self.__num_of_gauss_locations_ksi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__plastic_strain = np.zeros(
                (self.__num_of_gauss_locations_ksi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__tangent_modulus = self.__youngs_modulus * \
                np.ones((self.__num_of_gauss_locations_ksi,
                        self.__num_of_gauss_locations_eta), dtype=float)
            self.__internal_hardening_variable = np.zeros(
                (self.__num_of_gauss_locations_ksi, self.__num_of_gauss_locations_eta), dtype=float)
        else:
            print("These five models are implemented here: 1. elastic; 2. linear hardening; 3. quadratic hardening; 4. exponential hardening; 5. ramberg-osgood hardening.")
            raise ValueError("Wrong constitutive law!")

    @property
    def area(self):
        """
        Set the cross-sectional area of the beam element.

        A = b * h

        """
        return self.__width * self.__height

    @property
    def moment_of_inertia(self):
        """
        Set the moment of inertia of the beam element.

        I = 1/12 * b * h^3

        """
        return 1/12 * self.__width * (self.__height) ** 3

    @property
    def yield_stress(self):
        return self.__yield_stress

    @yield_stress.setter
    def yield_stress(self, val):
        """
        Set the yield stress of the beam element.

        Raises:
            TypeError: If value is not a float number.
            ValueError: If value is not positive.

        """
        if not isinstance(val, float):
            raise TypeError("Yield stress must be a float number!")
        elif val <= 0:
            raise ValueError("Moment of inertia must be positive!")
        else:
            self.__yield_stress = val

    @property
    def plastic_modulus(self):
        return self.__plastic_modulus

    @plastic_modulus.setter
    def plastic_modulus(self, val):
        """
        Set the plastic modulus of the beam element.

        Raises:
            TypeError: If value is not a float number.
            ValueError: If value is not positive.

        """
        if not isinstance(val, float):
            raise TypeError("Plastic modulus must be a float number!")
        elif val <= 0:
            raise ValueError("Plastic modulus must be positive!")
        else:
            self.__plastic_modulus = val

    @property
    def quadratic_coefficient(self):
        return self.__quadratic_coefficient

    @quadratic_coefficient.setter
    def quadratic_coefficient(self, val):
        """
        Set the quadratic coefficient of the beam element.

        Raises:
            TypeError: If value is not a float number.
            ValueError: If value is not positive.

        """
        if not isinstance(val, float):
            raise TypeError("Quadratic coefficient must be a float number!")
        elif val <= 0:
            raise ValueError("Quadratic coefficient must be positive!")
        else:
            self.__quadratic_coefficient = val
    
    @property
    def saturation_stress(self):
        return self.__saturation_stress

    @saturation_stress.setter
    def saturation_stress(self, val):
        """
        Set the saturation stress of the beam element.

        Raises:
            TypeError: If value is not a float number.
            ValueError: If value is not positive.

        """
        if not isinstance(val, float):
            raise TypeError("Saturation stress must be a float number!")
        elif val <= 0:
            raise ValueError("Saturation stress must be positive!")
        else:
            self.__saturation_stress = val
            
    @property
    def modified_modulus(self):
        return self.__modified_modulus

    @modified_modulus.setter
    def modified_modulus(self, val):
        """
        Set the modified modulus of the beam element.

        Raises:
            TypeError: If value is not a float number.
            ValueError: If value is not positive.

        """
        if not isinstance(val, float):
            raise TypeError("Modified modulus must be a float number!")
        elif val <= 0:
            raise ValueError("Modified modulus must be positive!")
        else:
            self.__modified_modulus = val
            
    @property
    def exponent(self):
        return self.__exponent

    @exponent.setter
    def exponent(self, val):
        """
        Set the exponent of the beam element.

        Raises:
            TypeError: If value is not a float number.
            ValueError: If value is not positive.

        """
        if not isinstance(val, float):
            raise TypeError("Exponent must be a float number!")
        elif val <= 0:
            raise ValueError("Exponent must be positive!")
        else:
            self.__exponent = val

    @property
    def element_freedom_table(self):
        return self.__element_freedom_table

    @element_freedom_table.setter
    def element_freedom_table(self, val):
        """
        Set the element freedom table according to the number of the element in the system.

        Raises:
            TypeError: If value is not a integer.
            ValueError: If value is not positive.

        """
        if not isinstance(val, int):
            raise TypeError("Number of the member must be a integer!")
        elif val < 0:
            raise ValueError("Number of the member must be non-negative!")
        else:
            self.__element_freedom_table = np.linspace(
                3 * val, 3 * val + 5, num=6, dtype=int)

    @property
    def initial_length(self):
        return np.linalg.norm(self.initial_coordinate_node_2 -
                              self.initial_coordinate_node_1)

    @property
    def initial_angle(self):
        return atan2(self.initial_coordinate_node_2[1] -
                     self.initial_coordinate_node_1[1],
                     self.initial_coordinate_node_2[0] -
                     self.initial_coordinate_node_1[0])

    @property
    def current_length(self):
        return np.linalg.norm(self.current_coordinate_node_2 -
                              self.current_coordinate_node_1)

    @property
    def current_angle(self):
        return atan2(self.current_coordinate_node_2[1] -
                     self.current_coordinate_node_1[1],
                     self.current_coordinate_node_2[0] -
                     self.current_coordinate_node_1[0])

    @property
    def local_displacement(self):

        u_bar = self.current_length - self.initial_length

        theta1_bar = atan2((cos(self.current_angle) * sin(self.global_rotation_node_1 + self.initial_angle)
                            - sin(self.current_angle) * cos(self.global_rotation_node_1 + self.initial_angle)),
                           (cos(self.current_angle) * cos(self.global_rotation_node_1 + self.initial_angle)
                            + sin(self.current_angle) * sin(self.global_rotation_node_1 + self.initial_angle)))

        theta2_bar = atan2((cos(self.current_angle) * sin(self.global_rotation_node_2 + self.initial_angle)
                            - sin(self.current_angle) * cos(self.global_rotation_node_2 + self.initial_angle)),
                           (cos(self.current_angle) * cos(self.global_rotation_node_2 + self.initial_angle)
                            + sin(self.current_angle) * sin(self.global_rotation_node_2 + self.initial_angle)))

        return np.array([[u_bar, theta1_bar, theta2_bar]]).T

    @property
    def local_material_stiffness(self):
        if self.__analysis == "elastic":
            return self.__youngs_modulus / self.current_length * \
                np.array([[self.area, 0., 0.],
                          [0., 4 * self.moment_of_inertia,
                              2 * self.moment_of_inertia],
                          [0., 2 * self.moment_of_inertia,
                           4 * self.moment_of_inertia]])
        else:
            kl_11 = 0.0
            kl_22 = 0.0
            kl_33 = 0.0
            kl_12 = 0.0
            kl_13 = 0.0
            kl_23 = 0.0
            [gauss_locations_ksi, weights_ksi] = np.polynomial.legendre.leggauss(
                self.__num_of_gauss_locations_ksi)
            [gauss_locations_eta, weights_eta] = np.polynomial.legendre.leggauss(
                self.__num_of_gauss_locations_eta)

            for iksi in range(self.__num_of_gauss_locations_ksi):
                for jeta in range(self.__num_of_gauss_locations_eta):
                    x, z = self.map_local_coordinate_to_global_coordinate(
                        gauss_locations_ksi[iksi], gauss_locations_eta[jeta])
                    fac = self.__width * self.__height * self.initial_length / 4
                    weight = weights_ksi[iksi] * weights_eta[jeta]

                    kl_11 += fac * \
                        self.__tangent_modulus[iksi, jeta] / \
                        (self.initial_length) ** 2 * weight
                    kl_22 += fac * self.__tangent_modulus[iksi, jeta] * (z * (4 / self.initial_length - 6 *
                                                                              x / (self.initial_length) ** 2)) ** 2 * weight
                    kl_33 += fac * self.__tangent_modulus[iksi, jeta] * (z * (2 / self.initial_length - 6 *
                                                                              x / (self.initial_length) ** 2)) ** 2 * weight
                    kl_12 += fac * self.__tangent_modulus[iksi, jeta] * z * (
                        4 / (self.initial_length) ** 2 - 6 * x / (self.initial_length) ** 3) * weight
                    kl_13 += fac * self.__tangent_modulus[iksi, jeta] * z * (
                        2 / (self.initial_length) ** 2 - 6 * x / (self.initial_length) ** 3) * weight
                    kl_23 += fac * self.__tangent_modulus[iksi, jeta] * z ** 2 * (4 / self.initial_length - 6 * x / (
                        self.initial_length) ** 2) * (2 / self.initial_length - 6 * x / (self.initial_length) ** 2) * weight

            return np.c_[np.r_[kl_11, kl_12, kl_13], np.r_[kl_12, kl_22, kl_23], np.r_[kl_13, kl_23, kl_33]]

    @property
    def local_force(self):
        if self.__analysis == "elastic":
            return self.local_material_stiffness @ self.local_displacement
        else:
            # local_force = np.zeros((3, 1), dtype=float)
            [gauss_locations_ksi, weights_ksi] = np.polynomial.legendre.leggauss(
                self.__num_of_gauss_locations_ksi)
            [gauss_locations_eta, weights_eta] = np.polynomial.legendre.leggauss(
                self.__num_of_gauss_locations_eta)

            N = 0.0
            M1 = 0.0
            M2 = 0.0

            for i in range(self.__num_of_gauss_locations_ksi):
                for j in range(self.__num_of_gauss_locations_eta):
                    x, z = self.map_local_coordinate_to_global_coordinate(
                        gauss_locations_ksi[i], gauss_locations_eta[j])

                    fac = self.__width * self.__height * self.initial_length / 4
                    weight = weights_ksi[i] * weights_eta[j]

                    N += fac * \
                        self.__stress[i, j] / self.initial_length * weight

                    M1 += fac * self.__stress[i, j] * z * (
                        4 / self.initial_length - 6 * x / (self.initial_length) ** 2) * weight

                    M2 += fac * self.__stress[i, j] * z * (
                        2 / self.initial_length - 6 * x / (self.initial_length) ** 2) * weight

            return np.r_[N, M1, M2]

    def map_local_coordinate_to_global_coordinate(self, ksi, eta):
        x = self.initial_length / 2 * (ksi + 1)
        z = self.__height / 2 * eta

        return x, z

    def axial_strain(self, x, z):
        return self.local_displacement[0] / self.initial_length + z * ((4 / self.initial_length - 6 *
                                                                        x / (self.initial_length) ** 2) * self.local_displacement[1] +
                                                                       (2 / self.initial_length - 6 *
                                                                        x / (self.initial_length) ** 2) * self.local_displacement[2])

    def perform_perfect_plasticity(self):
        gauss_locations_ksi = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_ksi)[0]
        gauss_locations_eta = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_eta)[1]

        def yield_stress_function(alpha):
            return self.__yield_stress

        for iksi in range(self.__num_of_gauss_locations_ksi):
            for jeta in range(self.__num_of_gauss_locations_eta):
                x, z = self.map_local_coordinate_to_global_coordinate(
                    gauss_locations_ksi[iksi], gauss_locations_eta[jeta])
                stress_trial = self.__youngs_modulus * \
                    (float(self.axial_strain(x, z)) -
                     self.__plastic_strain[iksi, jeta])
                yield_condition_trial = np.abs(
                    stress_trial) - yield_stress_function(self.__internal_hardening_variable[iksi, jeta])

                if yield_condition_trial <= 0:
                    self.__stress[iksi, jeta] = stress_trial
                    self.__tangent_modulus[iksi, jeta] = self.__youngs_modulus
                else:
                    deltagamma = yield_condition_trial / self.__youngs_modulus
                    self.__stress[iksi, jeta] = (
                        1 - deltagamma * self.__youngs_modulus / np.abs(stress_trial)) * stress_trial
                    self.__plastic_strain[iksi,
                                          jeta] += deltagamma * np.sign(stress_trial)
                    self.__tangent_modulus[iksi, jeta] = 0.0

    def perform_linear_hardening(self):
        gauss_locations_ksi = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_ksi)[0]
        gauss_locations_eta = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_eta)[0]

        def yield_stress_function(alpha):
            return self.__yield_stress + self.__plastic_modulus * alpha

        for iksi in range(self.__num_of_gauss_locations_ksi):
            for jeta in range(self.__num_of_gauss_locations_eta):
                x, z = self.map_local_coordinate_to_global_coordinate(
                    gauss_locations_ksi[iksi], gauss_locations_eta[jeta])
                stress_trial = self.__youngs_modulus * \
                    (float(self.axial_strain(x, z)) -
                     self.__plastic_strain[iksi, jeta])
                yield_condition_trial = np.abs(
                    stress_trial) - yield_stress_function(self.__internal_hardening_variable[iksi, jeta])

                if yield_condition_trial <= 0:
                    self.__stress[iksi, jeta] = stress_trial
                    self.__tangent_modulus[iksi, jeta] = self.__youngs_modulus
                else:
                    deltagamma = yield_condition_trial / \
                        (self.__youngs_modulus + self.__plastic_modulus)
                    self.__stress[iksi, jeta] = (
                        1 - deltagamma * self.__youngs_modulus / np.abs(stress_trial)) * stress_trial
                    self.__plastic_strain[iksi,
                                          jeta] += deltagamma * np.sign(stress_trial)
                    self.__internal_hardening_variable[iksi,
                                                       jeta] += deltagamma
                    self.__tangent_modulus[iksi, jeta] = self.__youngs_modulus * \
                        self.__plastic_modulus / \
                        (self.__youngs_modulus + self.__plastic_modulus)

    def perform_quadratic_hardening(self):
        gauss_locations_ksi = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_ksi)[0]
        gauss_locations_eta = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_eta)[0]

        def yield_stress_function(alpha):
            return self.__yield_stress + self.__youngs_modulus * (alpha - self.__quadratic_coefficient * alpha ** 2)

        for iksi in range(self.__num_of_gauss_locations_ksi):
            for jeta in range(self.__num_of_gauss_locations_eta):
                x, z = self.map_local_coordinate_to_global_coordinate(
                    gauss_locations_ksi[iksi], gauss_locations_eta[jeta])
                stress_trial = self.__youngs_modulus * \
                    (float(self.axial_strain(x, z)) -
                     self.__plastic_strain[iksi, jeta])
                yield_condition_trial = np.abs(
                    stress_trial) - yield_stress_function(self.__internal_hardening_variable[iksi, jeta])

                if yield_condition_trial <= 0:
                    self.__stress[iksi, jeta] = stress_trial
                    self.__tangent_modulus[iksi, jeta] = self.__youngs_modulus
                else:
                    a = self.__quadratic_coefficient * self.__youngs_modulus
                    b = 2 * self.__youngs_modulus * \
                        (self.__internal_hardening_variable[iksi,
                         jeta] * self.__quadratic_coefficient - 1)
                    c = yield_condition_trial
                    deltagamma = np.roots([a, b, c])[-1]
                    self.__stress[iksi, jeta] = (
                        1 - deltagamma * self.__youngs_modulus / np.abs(stress_trial)) * stress_trial
                    self.__plastic_strain[iksi,
                                          jeta] += deltagamma * np.sign(stress_trial)
                    self.__internal_hardening_variable[iksi,
                                                       jeta] += deltagamma
                    self.__tangent_modulus[iksi, jeta] = self.__youngs_modulus * (2 * self.__quadratic_coefficient * self.__internal_hardening_variable[iksi, jeta] - 1) / (
                        2 * self.__quadratic_coefficient * self.__internal_hardening_variable[iksi, jeta] - 2)

    def perform_exponential_hardening(self):
        gauss_locations_ksi = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_ksi)[0]
        gauss_locations_eta = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_eta)[0]

        def yield_stress_function(alpha):
            return self.__yield_stress + (self.__saturation_stress - self.__yield_stress) * (1 - np.exp(-self.__exponent * alpha))

        hardening_limit_stress = self.__saturation_stress - self.__yield_stress

        for iksi in range(self.__num_of_gauss_locations_ksi):
            for jeta in range(self.__num_of_gauss_locations_eta):
                x, z = self.map_local_coordinate_to_global_coordinate(
                    gauss_locations_ksi[iksi], gauss_locations_eta[jeta])
                stress_trial = self.__youngs_modulus * \
                    (float(self.axial_strain(x, z)) -
                     self.__plastic_strain[iksi, jeta])
                yield_condition_trial = np.abs(
                    stress_trial) - yield_stress_function(self.__internal_hardening_variable[iksi, jeta])

                if yield_condition_trial <= 0:
                    self.__stress[iksi, jeta] = stress_trial
                    self.__tangent_modulus[iksi, jeta] = self.__youngs_modulus
                else:
                    deltagamma = 0
                    residual = yield_condition_trial - deltagamma * self.__youngs_modulus - yield_stress_function(
                        self.__internal_hardening_variable[iksi, jeta] + deltagamma) + yield_stress_function(self.__internal_hardening_variable[iksi, jeta])

                    maxiter = 10
                    iteration_counter = 0
                    tolerance = 1.0e-5

                    while residual > tolerance and iteration_counter < maxiter:
                        dR_ddeltagamma = -self.__youngs_modulus - hardening_limit_stress * self.__exponent * \
                            np.exp(-self.__exponent *
                                   (self.__internal_hardening_variable[iksi, jeta] + deltagamma))
                        d_g = - residual / dR_ddeltagamma
                        deltagamma += d_g
                        residual = yield_condition_trial - deltagamma * self.__youngs_modulus - yield_stress_function(
                            self.__internal_hardening_variable[iksi, jeta] + deltagamma) + yield_stress_function(self.__internal_hardening_variable[iksi, jeta])
                        iteration_counter += 1

                    self.__stress[iksi, jeta] = (
                        1 - deltagamma * self.__youngs_modulus / np.abs(stress_trial)) * stress_trial
                    self.__plastic_strain[iksi,
                                          jeta] += deltagamma * np.sign(stress_trial)
                    self.__internal_hardening_variable[iksi,
                                                       jeta] += deltagamma
                    self.__tangent_modulus[iksi, jeta] = self.__youngs_modulus * hardening_limit_stress * self.__exponent * np.exp(-self.__exponent * self.__internal_hardening_variable[iksi, jeta]) / (
                        self.__youngs_modulus + hardening_limit_stress * self.__exponent * np.exp(-self.__exponent * self.__internal_hardening_variable[iksi, jeta]))

    def perform_ramberg_osgood_hardening(self):
        gauss_locations_ksi = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_ksi)[0]
        gauss_locations_eta = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_eta)[0]

        def yield_stress_function(alpha):
            return self.__yield_stress + self.__modified_modulus * np.power(alpha, self.__exponent)

        for iksi in range(self.__num_of_gauss_locations_ksi):
            for jeta in range(self.__num_of_gauss_locations_eta):
                x, z = self.map_local_coordinate_to_global_coordinate(
                    gauss_locations_ksi[iksi], gauss_locations_eta[jeta])
                stress_trial = self.__youngs_modulus * \
                    (float(self.axial_strain(x, z)) -
                     self.__plastic_strain[iksi, jeta])
                yield_condition_trial = np.abs(
                    stress_trial) - yield_stress_function(self.__internal_hardening_variable[iksi, jeta])

                if yield_condition_trial <= 0:
                    self.__stress[iksi, jeta] = stress_trial
                    self.__tangent_modulus[iksi, jeta] = self.__youngs_modulus
                else:
                    deltagamma = 0
                    residual = yield_condition_trial - deltagamma * self.__youngs_modulus - yield_stress_function(
                        self.__internal_hardening_variable[iksi, jeta] + deltagamma) + yield_stress_function(self.__internal_hardening_variable[iksi, jeta])

                    maxiter = 10
                    iteration_counter = 0
                    tolerance = 1.0e-5

                    while residual > tolerance and iteration_counter < maxiter:
                        dR_ddeltagamma = -self.__youngs_modulus - self.__exponent * self.__modified_modulus * \
                            np.power(
                                self.__internal_hardening_variable[iksi, jeta] + deltagamma, self.__exponent - 1)
                        d_g = - residual / dR_ddeltagamma
                        deltagamma += d_g
                        residual = yield_condition_trial - deltagamma * self.__youngs_modulus - yield_stress_function(
                            self.__internal_hardening_variable[iksi, jeta] + deltagamma) + yield_stress_function(self.__internal_hardening_variable[iksi, jeta])
                        iteration_counter += 1

                    self.__stress[iksi, jeta] = (
                        1 - deltagamma * self.__youngs_modulus / np.abs(stress_trial)) * stress_trial
                    self.__plastic_strain[iksi,
                                          jeta] += deltagamma * np.sign(stress_trial)
                    self.__internal_hardening_variable[iksi,
                                                       jeta] += deltagamma
                    self.__tangent_modulus[iksi, jeta] = self.__youngs_modulus * self.__exponent * self.__modified_modulus * np.power(self.__internal_hardening_variable[iksi, jeta], self.__exponent - 1) / (
                        self.__youngs_modulus + self.__exponent * self.__modified_modulus * np.power(self.__internal_hardening_variable[iksi, jeta], self.__exponent - 1))

    @property
    def transformation_matrix(self):
        # Calculate B-matrix for transformed material stiffness matrix.
        # B-matrix is the relation between infinitesimal deformation inducing
        # work and infinitesimal global deformation.
        # deltap_l = B @ deltap

        return np.array([[-cos(self.current_angle), -sin(self.current_angle), 0., cos(self.current_angle), sin(self.current_angle), 0.],
                         [-sin(self.current_angle) / self.current_length, cos(self.current_angle)/self.current_length,
                          1., sin(self.current_angle)/self.current_length, -cos(self.current_angle)/self.current_length, 0.],
                         [-sin(self.current_angle) / self.current_length, cos(self.current_angle)/self.current_length,
                          0., sin(self.current_angle)/self.current_length, -cos(self.current_angle)/self.current_length, 1.]])

    @property
    def material_stiffness_matrix(self):
        # Calculate transformed material stiffness matrix k_t1.
        return self.transformation_matrix.T @ self.local_material_stiffness @ self.transformation_matrix

    @property
    def geometry_stiffness_matrix(self):
        """ Calculate geometric stiffness matrix k_tsigma.

        Arg:
            q_li: the storage vector local forces of the beam, = [N, M1_bar, M2_bar].
            Instead of using the function getLocalForces(), the forces are
            input outside.
        """

        r = np.array([[-cos(self.current_angle), -sin(self.current_angle),
                     0.,  cos(self.current_angle), sin(self.current_angle), 0.]]).T
        z = np.array([[sin(self.current_angle), -cos(self.current_angle),
                     0., -sin(self.current_angle), cos(self.current_angle), 0.]]).T
        k_tsigma = self.local_force[0] / self.current_length * z @ z.T
        + (self.local_force[1] + self.local_force[2]) / \
            (self.current_length ** 2) * (r @ z.T + z @ r.T)

        return k_tsigma

    @property
    def global_stiffness_matrix(self):
        # Calculate tangent stiffness matrix k_t.
        return self.material_stiffness_matrix + self.geometry_stiffness_matrix

    @property
    def global_force(self):
        return self.transformation_matrix.T @ self.local_force


class CorotationalBeamElement3D():
    """A 3D corotational beam element.

    This class helps formulate a 2D corotational beam element, and calculate
    e.g. its local forces and tangential stiffness matrix.

    DoF / Global Displacement vector: u = [u1, v1, theta_1, u2, v2, theta_2]
    Local force: q_l = [N, M1_bar, M2_bar]

    Attributes:
        For undeformed state, letters of variables are always capitalized. If
        the structure is deformed, they will be lowercases.
    """

    def __init__(self):
        """Initialize the beam element with reference configuration.

        Args:
            X1: The position vector of the 1st node in undeformed state
            X2: The position vector of the 2nd node in undeformed state
            __youngs_modulus: Young's modulus
            A: Cross-Sectional areas
            _moment_of_inertia_y: 2nd moments of inertia
        """
        self.__initial_coordinate_node_1 = None
        self.__initial_coordinate_node_2 = None
        self.__incremental_global_displacement = np.zeros((12, 1), dtype=float)

        self.__youngs_modulus = None
        self._shear_modulus = None
        self._area = None

        self._moment_of_inertia_y = None
        self._moment_of_inertia_z = None
        self._polar_moment_of_inertia = None

        self._current_frame_node_1 = np.eye(3)
        self._current_frame_node_2 = np.eye(3)

        self.__element_freedom_table = None

    @property
    def initial_coordinate_node_1(self):
        return self.__initial_coordinate_node_1

    @initial_coordinate_node_1.setter
    def initial_coordinate_node_1(self, val):
        """Set the initial coordinate of node 1: [X1, Y1, Z1]

        Raises:
            TypeError: If value is not a numpy array.

        """

        if isinstance(val, np.ndarray):
            self.__initial_coordinate_node_1 = val
        else:
            raise TypeError(
                "The initial coordinate of node 1 must be a 2x1 array!")

    @property
    def initial_coordinate_node_2(self):
        return self.__initial_coordinate_node_2

    @initial_coordinate_node_2.setter
    def initial_coordinate_node_2(self, val):
        """Set the initial coordinate of node 2: [X2, Y2, Z2]

        Raises:
            TypeError: If value is not a numpy array.

        """

        if isinstance(val, np.ndarray):
            self.__initial_coordinate_node_2 = val
        else:
            raise TypeError(
                "The initial coordinate of node 2 must be a 2x1 array!")

    @property
    def incremental_global_displacement(self):
        return self.__incremental_global_displacement

    @incremental_global_displacement.setter
    def incremental_global_displacement(self, val):
        if isinstance(val, np.ndarray):
            self.__incremental_global_displacement = val
        else:
            raise TypeError(
                "Global displacement must be a 6x1 array!")

    @property
    def current_coordinate_node_1(self):
        return self.__initial_coordinate_node_1 + self.__incremental_global_displacement[0: 3]

    @property
    def current_coordinate_node_2(self):
        return self.__initial_coordinate_node_2 + self.__incremental_global_displacement[6: 9]

    @property
    def current_orientation_node_1(self):
        return self._current_frame_node_1

    @current_orientation_node_1.setter
    def current_orientation_node_1(self, val):
        """Set the global nodal rotation of node 1: θ1

        Raises:
            TypeError: If value is not a float number.

        """

        if isinstance(val, np.ndarray):
            self._current_frame_node_1 = val
        else:
            raise TypeError(
                "Orientation matrix at node 1 must be a 3x3 matrix!")

    @property
    def current_orientation_node_2(self):
        return self._current_frame_node_2

    @current_orientation_node_2.setter
    def current_orientation_node_2(self, val):
        """Set the global nodal rotation of node 2: θ2

        Raises:
            TypeError: If value is not a float number.

        """

        if isinstance(val, np.ndarray):
            self._current_frame_node_2 = val
        else:
            raise TypeError(
                "Orientation matrix at node 2 must be a 3x3 matrix!")

    @property
    def element_freedom_table(self):
        return self.__element_freedom_table

    @element_freedom_table.setter
    def element_freedom_table(self, val):
        if not isinstance(val, int):
            raise TypeError("Number of the member must be a integer!")
        elif val < 0:
            raise ValueError("Number of the member must be non-negative!")
        else:
            self.__element_freedom_table = np.linspace(
                6 * val, 6 * val + 11, num=12, dtype=int)

    @property
    def youngs_modulus(self):
        return self.__youngs_modulus

    @youngs_modulus.setter
    def youngs_modulus(self, val):
        """Set the Young's modulus of the beam element: __youngs_modulus > 0.

        Raises:
            TypeError: If value is not a positive float number.

        """

        if not isinstance(val, float):
            raise TypeError("Young's modulus must be a float number!")
        elif val <= 0:
            raise ValueError("Young's modulus must be positive!")
        else:
            self.__youngs_modulus = val

    @property
    def shear_modulus(self):
        return self._shear_modulus

    @shear_modulus.setter
    def shear_modulus(self, val):
        """Set the Shear modulus of the beam element: _shear_modulus > 0.

        Raises:
            TypeError: If value is not a positive float number.

        """

        if not isinstance(val, float):
            raise TypeError("Shear modulus must be a float number!")
        elif val <= 0:
            raise ValueError("Shear modulus must be positive!")
        else:
            self._shear_modulus = val

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, val):
        """Set the cross-sectional area of the beam element: A > 0.

        Raises:
            TypeError: If value is not a positive float number.

        """
        if not isinstance(val, float):
            raise TypeError("Cross-sectional area must be a float number!")
        elif val <= 0:
            raise ValueError("Cross-sectional area must be positive!")
        else:
            self._area = val

    @property
    def moment_of_inertia_y(self):
        return self._moment_of_inertia_y

    @moment_of_inertia_y.setter
    def moment_of_inertia_y(self, val):
        """Set the moment of inertia of the beam element: I > 0.

        Raises:
            TypeError: If value is not a positive float number.

        """
        if not isinstance(val, float):
            raise TypeError("Moment of inertia must be a float number!")
        elif val <= 0:
            raise ValueError("Moment of inertia must be positive!")
        else:
            self._moment_of_inertia_y = val

    @property
    def moment_of_inertia_z(self):
        return self._moment_of_inertia_z

    @moment_of_inertia_z.setter
    def moment_of_inertia_z(self, val):
        """Set the moment of inertia of the beam element: I > 0.

        Raises:
            TypeError: If value is not a positive float number.

        """
        if not isinstance(val, float):
            raise TypeError("Moment of inertia must be a float number!")
        elif val <= 0:
            raise ValueError("Moment of inertia must be positive!")
        else:
            self._moment_of_inertia_z = val

    @property
    def polar_moment_of_inertia(self):
        return self._polar_moment_of_inertia

    @polar_moment_of_inertia.setter
    def polar_moment_of_inertia(self, val):
        """Set the moment of inertia of the beam element: I > 0.

        Raises:
            TypeError: If value is not a positive float number.

        """
        if not isinstance(val, float):
            raise TypeError("Moment of inertia must be a float number!")
        elif val <= 0:
            raise ValueError("Moment of inertia must be positive!")
        else:
            self._polar_moment_of_inertia = val

    @property
    def initial_local_frame(self):
        e_3o = np.array([[0, 0, 1]], dtype=float).T
        e_1o = (self.initial_coordinate_node_2 -
                self.initial_coordinate_node_1) / self.initial_length
        e_2o = np.cross(e_3o, e_1o, axisa=0, axisb=0, axisc=0)
        return np.c_[e_1o, e_2o, e_3o]

    @property
    def initial_length(self):
        return np.linalg.norm(self.initial_coordinate_node_2 -
                              self.initial_coordinate_node_1)

    @property
    def current_length(self):
        """
        Calculate the deformed length l of the beam element and
        save it as an attribute.
        """

        return np.linalg.norm(self.current_coordinate_node_2 -
                              self.current_coordinate_node_1)

    @property
    def local_material_stiffness(self):

        k_l = np.zeros((7, 7), dtype=float)

        k_l[0, 0] = self.__youngs_modulus * self._area

        k_l[1, 1], k_l[4, 4] = self._shear_modulus * \
            self._polar_moment_of_inertia, self._shear_modulus * self._polar_moment_of_inertia
        k_l[1, 4], k_l[4, 1] = -self._shear_modulus * self._polar_moment_of_inertia, - \
            self._shear_modulus * self._polar_moment_of_inertia

        k_l[2, 2], k_l[5, 5] = 4.0 * self.__youngs_modulus * \
            self._moment_of_inertia_z, 4.0 * self.__youngs_modulus * self._moment_of_inertia_z
        k_l[2, 5], k_l[5, 2] = 2.0 * self.__youngs_modulus * \
            self._moment_of_inertia_z, 2.0 * self.__youngs_modulus * self._moment_of_inertia_z

        k_l[3, 3], k_l[6, 6] = 4.0 * self.__youngs_modulus * \
            self._moment_of_inertia_y, 4.0 * self.__youngs_modulus * self._moment_of_inertia_y
        k_l[3, 6], k_l[6, 3] = 2.0 * self.__youngs_modulus * \
            self._moment_of_inertia_y, 2.0 * self.__youngs_modulus * self._moment_of_inertia_y

        k_l /= self.current_length

        return k_l

    @property
    def auxiliary_vector(self):
        q_1 = self.current_orientation_node_1 @ self.initial_local_frame @ np.array(
            [[0, 1, 0]], dtype=float).T
        q_2 = self.current_orientation_node_2 @ self.initial_local_frame @ np.array(
            [[0, 1, 0]], dtype=float).T
        q = (q_1 + q_2) / 2

        return q_1, q_2, q

    @property
    def current_local_frame(self):
        r_1 = (self.current_coordinate_node_2 -
               self.current_coordinate_node_1) / self.current_length

        _, _, q = self.auxiliary_vector
        r_3 = np.cross(r_1, q, axisa=0, axisb=0, axisc=0)
        r_3 = r_3 / np.linalg.norm(r_3)

        r_2 = np.cross(r_3, r_1, axisa=0, axisb=0, axisc=0)

        return np.c_[r_1, r_2, r_3]

    @property
    def local_displacement(self):
        p_l = np.zeros((7, 1), dtype=float)

        theta_1_tilde = logm(self.current_local_frame.T @
                             self.current_orientation_node_1 @ self.initial_local_frame)
        theta_2_tilde = logm(self.current_local_frame.T @
                             self.current_orientation_node_2 @ self.initial_local_frame)

        p_l[0] = self.current_length - self.initial_length
        p_l[1: 4] = util.decomposeSkewSymmetric(theta_1_tilde)
        p_l[4: 7] = util.decomposeSkewSymmetric(theta_2_tilde)

        return p_l

    @property
    def local_force(self):
        # Assemble axial force and local end moments into a vector q_l.

        return self.local_material_stiffness @ self.local_displacement

    @property
    def Ba_matrix(self):
        p_l = self.local_displacement
        T_s_inv_theta_1bar = util.getTransformation(p_l[1: 4], 'inv')
        T_s_inv_theta_2bar = util.getTransformation(p_l[4: 7], 'inv')

        B_a = np.zeros((7, 7), dtype=float)
        B_a[0, 0] = 1.
        B_a[1: 4, 1: 4] = T_s_inv_theta_1bar
        B_a[4: 7, 4: 7] = T_s_inv_theta_2bar

        return B_a

    @property
    def local_force_a(self):
        return self.Ba_matrix.T @ self.local_force

    def Khi_matrix(self, i):
        if i == 1:
            theta_bar = self.local_displacement[1: 4]
            v = self.local_force[1: 4]
        elif i == 2:
            theta_bar = self.local_displacement[4: 7]
            v = self.local_force[4: 7]
        else:
            raise ValueError("Wrong value of i!")

        alpha = np.linalg.norm(theta_bar)
        eta, mu = util.getEtaAndMu(alpha)
        theta_bar_tilde = util.getSkewSymmetric(theta_bar)
        v_tilde = util.getSkewSymmetric(v)
        T_s_inv = util.getTransformation(theta_bar, 'inv')

        K_hi = (eta * (theta_bar @ v.T - 2 * v @ theta_bar.T + float(theta_bar.T @ v) * np.eye(3))
                + mu * theta_bar_tilde @ theta_bar_tilde @ (v @ theta_bar.T) - 1/2 * v_tilde) @ T_s_inv

        return K_hi

    @property
    def Kh_matrix(self):
        K_h = np.zeros((7, 7), dtype=float)
        K_h1 = self.Khi_matrix(1)
        K_h2 = self.Khi_matrix(2)
        K_h[1: 4, 1: 4] = K_h1
        K_h[4: 7, 4: 7] = K_h2

        return K_h

    @property
    def Ka_matrix(self):
        return self.Ba_matrix.T @ self.local_material_stiffness @ self.Ba_matrix + self.Kh_matrix

    @property
    def G_matrix(self):

        q_1, q_2, q = self.auxiliary_vector

        q1 = float((self.current_local_frame.T @ q)[0])
        q2 = float((self.current_local_frame.T @ q)[1])
        q11 = float((self.current_local_frame.T @ q_1)[0])
        q12 = float((self.current_local_frame.T @ q_1)[1])
        q21 = float((self.current_local_frame.T @ q_2)[0])
        q22 = float((self.current_local_frame.T @ q_2)[1])

        eta = q1 / q2
        eta_11 = q11 / q2
        eta_12 = q12 / q2
        eta_21 = q21 / q2
        eta_22 = q22 / q2

        G = np.array([
            [0, 0, eta/self.current_length, eta_12/2, -eta_11/2, 0, 0,
                0, -eta/self.current_length, eta_22/2, -eta_21/2, 0],
            [0, 0, 1/self.current_length, 0, 0, 0,
             0, 0, -1/self.current_length, 0, 0, 0],
            [0, -1/self.current_length, 0, 0, 0, 0,
             0, 1/self.current_length, 0, 0, 0, 0]], dtype=float).T

        return G

    @property
    def E_matrix(self):

        E = np.zeros((12, 12), dtype=float)
        E[0: 3, 0: 3] = self.current_local_frame
        E[3: 6, 3: 6] = self.current_local_frame
        E[6: 9, 6: 9] = self.current_local_frame
        E[9: 12, 9: 12] = self.current_local_frame

        return E

    @property
    def r_vector(self):

        r_1 = (self.current_coordinate_node_2 -
               self.current_coordinate_node_1) / self.current_length
        r = np.c_[-r_1.T, np.array([[0, 0, 0]], dtype=float),
                  r_1.T, np.array([[0, 0, 0]], dtype=float)]

        return r

    @property
    def P_matrix(self):

        P = np.r_[np.c_[np.zeros((3, 3), dtype=float),
                        np.eye(3),
                        np.zeros((3, 3), dtype=float),
                        np.zeros((3, 3), dtype=float)],
                  np.c_[np.zeros((3, 3), dtype=float),
                        np.zeros((3, 3), dtype=float),
                        np.zeros((3, 3), dtype=float),
                        np.eye(3)]] - np.r_[self.G_matrix.T, self.G_matrix.T]

        return P

    @property
    def Bg_matrix(self):

        return np.r_[self.r_vector, self.P_matrix @ self.E_matrix.T]

    @property
    def global_force(self):
        return self.Bg_matrix.T @ self.local_force_a

    @property
    def D_matrix(self):
        r_1 = (self.current_coordinate_node_2 -
               self.current_coordinate_node_1) / self.current_length
        D_3 = 1/self.current_length * (np.eye(3) - r_1 @ r_1.T)

        D = np.zeros((12, 12), dtype=float)
        D[0: 3, 0: 3] = D_3
        D[0: 3, 6: 9] = -D_3
        D[6: 9, 0: 3] = -D_3
        D[6: 9, 6: 9] = D_3

        return D

    @property
    def a_vector(self):

        _, _, q = self.auxiliary_vector
        q1 = float((self.current_local_frame.T @ q)[0, 0])
        q2 = float((self.current_local_frame.T @ q)[1, 0])
        eta = q1 / q2

        a = np.zeros((3, 1), dtype=float)
        a[1] = eta/self.current_length * (self.local_force_a[1] + self.local_force_a[4]) - \
            1/self.current_length * \
            (self.local_force_a[2] + self.local_force_a[5])
        a[2] = 1/self.current_length * \
            (self.local_force_a[3] + self.local_force_a[6])

        return a

    @property
    def Q_matrix(self):

        n_1 = (self.P_matrix.T @ self.local_force_a[1: 7])[0: 3]
        m_1 = (self.P_matrix.T @ self.local_force_a[1: 7])[3: 6]
        n_2 = (self.P_matrix.T @ self.local_force_a[1: 7])[6: 9]
        m_2 = (self.P_matrix.T @ self.local_force_a[1: 7])[9: 12]

        n_1tilde = util.getSkewSymmetric(n_1)
        m_1tilde = util.getSkewSymmetric(m_1)
        n_2tilde = util.getSkewSymmetric(n_2)
        m_2tilde = util.getSkewSymmetric(m_2)

        Q = np.r_[n_1tilde, m_1tilde, n_2tilde, m_2tilde]

        return Q

    @property
    def Km_matrix(self):

        K_m = self.D_matrix * self.local_force_a[0]
        - self.E_matrix @ self.Q_matrix @ self.G_matrix.T @ self.E_matrix.T
        + self.E_matrix @ self.G_matrix @ self.a_vector @ self.r_vector

        return K_m

    @property
    def global_stiffness_matrix(self):

        return self.Bg_matrix.T @ self.Ka_matrix @ self.Bg_matrix + self.Km_matrix


class System():
    def __init__(self):
        self.__dimension = None
        self._geometry_name = None
        self.__analysis = "elastic"

        self._structure = None

        self._number_of_elements = None
        self._number_of_nodes = None
        self._number_of_dofs = None
        
        self._number_of_increments = None
        
        self._max_load = None
        self._max_displacement = None
        self._arc_length = None

        self._dirichlet_boundary_condition = []
        self._load_boundary_condition = []
        self._load_increment_vector = None

        self._tolerance = None
        self._max_iteration_steps = None
        self._solver = None

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
        return self._geometry_name

    @geometry_name.setter
    def geometry_name(self, val):
        if isinstance(val, str):
            self._geometry_name = val
        else:
            raise TypeError("The name of mesh file must be a string!")

    @property
    def analysis(self):
        return self.__analysis

    @analysis.setter
    def analysis(self, val):
        if not ((val == "elastic") or
                (val == "perfect plasticity") or
                (val == "linear hardening") or
                (val == "quadratic hardening") or
                (val == "exponential hardening") or
                (val == "ramberg-osgood hardening")):
            print("These six models are implemented here: 1. elastic; 2. perfect plasticity; 3. linear hardening; 4. quadratic hardening; 5. exponential hardening; 6. ramberg-osgood hardening.")
            raise ValueError("Wrong constitutive law!")
        else:
            self.__analysis = val

    @property
    def number_of_increments(self):
        return self._number_of_increments

    @number_of_increments.setter
    def number_of_increments(self, val):
        if not isinstance(val, int):
            raise TypeError("Number of load increments must be a integer!")
        elif val <= 0:
            raise ValueError("Number of load increments must be positive!")
        else:
            self._number_of_increments = val

    @property
    def max_load(self):
        return self._max_load

    @max_load.setter
    def max_load(self, val):
        if not isinstance(val, float):
            raise TypeError("The upper bound of load must be a float!")
        elif val <= 0:
            raise ValueError("The upper bound of load must be positive!")
        else:
            self._max_load = val

    @property
    def max_displacement(self):
        return self._max_displacement

    @max_displacement.setter
    def max_displacement(self, val):
        if not isinstance(val, float):
            raise TypeError("The upper bound of displacement must be a float!")
        elif val <= 0:
            raise ValueError(
                "The upper bound of displacement must be positive!")
        else:
            self._max_displacement = val

    @property
    def arc_length(self):
        return self._arc_length

    @arc_length.setter
    def arc_length(self, val):
        if not isinstance(val, float):
            raise TypeError("The arc length must be a float!")
        elif val <= 0:
            raise ValueError(
                "The arc length must be positive!")
        else:
            self._arc_length = val


    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, val):
        if not isinstance(val, float):
            raise TypeError("The tolerance must be a float number!")
        elif val <= 0:
            raise ValueError("The tolerance must be positive!")
        else:
            self._tolerance = val

    @property
    def max_iteration_steps(self):
        return self._max_iteration_steps

    @max_iteration_steps.setter
    def max_iteration_steps(self, val):
        if not isinstance(val, int):
            raise TypeError("The maximum iteration steps must be a integer!")
        elif val <= 0:
            raise ValueError("The maximum iteration steps must be positive!")
        else:
            self._max_iteration_steps = val

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, val):
        if val != "load-control" and val != "displacement-control" and val != "arc-length-control":
            raise ValueError("Invalid solver!")
        else:
            self._solver = val

    def initialize_structure(self, *parameters):
        structure = []
        array_nodes, array_elements, self._number_of_nodes, self._number_of_elements = util.load_mesh_file(
            self.geometry_name)
        if self.__dimension == 2:
            self._number_of_dofs = 3 * self._number_of_nodes
            for iele, ele in enumerate(array_elements.T):
                co_ele = CorotationalBeamElement2D()
                co_ele.initial_coordinate_node_1 = array_nodes[0: 2, ele[0] - 1].reshape(
                    2, 1)
                co_ele.initial_coordinate_node_2 = array_nodes[0: 2, ele[1] - 1].reshape(
                    2, 1)

                co_ele.youngs_modulus = parameters[0]
                co_ele.width = parameters[1]
                co_ele.height = parameters[2]

                co_ele.analysis = self.__analysis
                co_ele.element_freedom_table = iele

                structure.append(co_ele)
        else:
            self._number_of_dofs = 6 * self._number_of_nodes
            for iele, ele in enumerate(array_elements.T):
                co_ele = CorotationalBeamElement3D()
                co_ele.initial_coordinate_node_1 = array_nodes[:, ele[0] - 1].reshape(
                    3, 1)
                co_ele.initial_coordinate_node_2 = array_nodes[:, ele[1] - 1].reshape(
                    3, 1)

                co_ele.youngs_modulus = parameters[0]
                co_ele.area = parameters[1]
                co_ele.moment_of_inertia_y = parameters[2]
                co_ele.moment_of_inertia_z = parameters[3]
                co_ele.polar_moment_of_inertia = parameters[4]
                co_ele.shear_modulus = parameters[5]

                co_ele.analysis = self.__analysis
                co_ele.element_freedom_table = iele

                structure.append(co_ele)
                
        self._structure = structure

    def initialize_with_plasticity(self, *parameters):
        for ele in self._structure:
            if self.__analysis == "perfect plasticity":
                ele.yield_stress = parameters[0]
            elif self.__analysis == "linear hardening":
                ele.yield_stress = parameters[0]
                ele.plastic_modulus = parameters[1]
            elif self.__analysis == "quadratic hardening":
                ele.yield_stress = parameters[0]
                ele.quadratic_coefficient = parameters[1]
            elif self.__analysis == "exponential hardening":
                ele.yield_stress = parameters[0]
                ele.saturation_stress = parameters[1]
                ele.exponent = parameters[2]
            elif self.__analysis == "ramberg-osgood hardening":
                ele.yield_stress = parameters[0]
                ele.modified_modulus = parameters[1]
                ele.exponent = parameters[2]

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

    def add_load_bc(self, node, dof, direction):
        if self.dimension == 2:
            if dof == "x":
                self._load_boundary_condition.append(3 * node)
            elif dof == "y":
                self._load_boundary_condition.append(3 * node + 1)
            elif dof == "m":
                self._load_boundary_condition.append(3 * node + 2)
        else:
            if dof == "x":
                self._load_boundary_condition.append(6 * node)
            elif dof == "y":
                self._load_boundary_condition.append(6 * node + 1)
            elif dof == "m":
                self._load_boundary_condition.append(6 * node + 2)

        self._load_increment_vector = np.zeros(
            (self._number_of_dofs, 1), dtype=float)
        if direction == "+":
            self._load_increment_vector[self._load_boundary_condition] = 1
        elif direction == "-":
            self._load_increment_vector[self._load_boundary_condition] = -1
        else:
            raise ValueError("Please assign the correct load direction!")

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
        K = np.zeros((self._number_of_dofs, self._number_of_dofs), dtype=float)

        if self.__dimension == 2:
            dof_per_ele = 6
        else:
            dof_per_ele = 12

        for ele in self._structure:
            eft = ele.element_freedom_table
            esm = ele.global_stiffness_matrix
            for idof in range(dof_per_ele):
                for jdof in range(idof, dof_per_ele):
                    K[eft[idof], eft[jdof]
                      ] += esm[idof, jdof]
                    K[eft[jdof], eft[idof]
                      ] = K[eft[idof], eft[jdof]]
        return K

    def modified_master_stiffness_matrix(self, K):
        """ Modify the system stiffness matrix K to K_s according to Drichlet
            Boundary Conditions.

            Args:
                K: the system stiffness matrix, [ndof x ndof]
                DBCdof: a list contains the dofs, such as [0, 1, 2]

            Returns:
                K_s: the modified system stiffness matrix, [ndof x ndof]
        """
        K_s = np.copy(K)

        for idof in self._dirichlet_boundary_condition:
            for jentry in range(self._number_of_dofs):
                K_s[idof, jentry] = 0.
                K_s[jentry, idof] = 0.
                K_s[idof, idof] = 1.
        return K_s

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
            for idof, iEFT in enumerate(ele.element_freedom_table):
                F_int[iEFT] += ele.global_force[idof]
        return F_int

    def update_member_data(self, u, deltau):
        """ Update nodal displacements, local forces and storage vector.

            Args:
                u: global displacement vector, [ndof x 1]
                beam: the list of beam elements
                nele: the number of elements
                q_l: the storage vector of local forces, [3 * nele x 1]

            Returns:
                beam: the list of beam elements, DATA UPDATED
                q_l: the UPDATED storage vector of local forces, [3 * nele x 1]
        """
        if self.__dimension == 2:
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
        else:
            for iele, ele in enumerate(self._structure):
                ele.incremental_global_displacement = u[6 *
                                                        iele: 6 * iele + 12]
                ele.current_orientation_node_1 = expm(util.getSkewSymmetric(
                    deltau[6 * iele + 3: 6 * iele + 6])) @ ele.current_orientation_node_1
                ele.current_orientation_node_2 = expm(util.getSkewSymmetric(
                    deltau[6 * iele + 9: 6 * iele + 12])) @ ele.current_orientation_node_2

    def update_member_data_iteration(self, u, deltau, plastic_strain, internal_hardening_variable):
        """ Update nodal displacements, local forces and storage vector.

            Args:
                u: global displacement vector, [ndof x 1]
                beam: the list of beam elements
                nele: the number of elements
                q_l: the storage vector of local forces, [3 * nele x 1]

            Returns:
                beam: the list of beam elements, DATA UPDATED
                q_l: the UPDATED storage vector of local forces, [3 * nele x 1]
        """
        if self.__dimension == 2:
            for iele, ele in enumerate(self._structure):
                ele.incremental_global_displacement = u[3 * iele: 3 * iele + 6]
                # ele.__plastic_strain = plastic_strain[iele]
                # ele.__internal_hardening_variable = internal_hardening_variable[iele]
                # ele.perform_perfect_plasticity()
        else:
            for iele, ele in enumerate(self._structure):
                ele.incremental_global_displacement = u[6 *
                                                        iele: 6 * iele + 12]
                ele.current_orientation_node_1 = expm(util.getSkewSymmetric(
                    deltau[6 * iele + 3: 6 * iele + 6])) @ ele.current_orientation_node_1
                ele.current_orientation_node_2 = expm(util.getSkewSymmetric(
                    deltau[6 * iele + 9: 6 * iele + 12])) @ ele.current_orientation_node_2

    def modified_residual(self, r):
        """ Modify the residual according to Drichlet Boundary Conditions.

            Args:
                r: the residual of the system, [ndof x 1]
                DBCdof: a list contains the dofs, such as [0, 1, 2]

            Returns:
                r: the MODIFIED residual of the system, [ndof x 1]
        """
        for idof in self._dirichlet_boundary_condition:
            r[idof] = 0
        return r

    def load_control(self):
        return
    
    def displacement_control(self):
        return
    
    def arc_length_control(self):
        U = np.array([0.], dtype=float)
        LAM = np.array([0.], dtype=float)
        u = np.zeros((self._number_of_dofs, 1), dtype=float)
        lam = 0.
        Deltau_prev = np.ones((self._number_of_dofs, 1), dtype=float)
        for n in range(self.number_of_load_increments):

            # set the predictor by equal load increments
            K = self.master_stiffness_matrix()
            K_s = self.modified_master_stiffness_matrix(K)

            deltaubar = np.linalg.solve(K_s, self._load_increment_vector)
            Deltalam = np.sign(float(Deltau_prev.T @ deltaubar)) * self._arc_length / np.sqrt(float(deltaubar.T @ deltaubar))
            lam += Deltalam
            Delta_increment = Deltalam * self._load_increment_vector
            Deltau = np.linalg.solve(K_s, Delta_increment)
 
            u += Deltau

            # update member data
            self.update_member_data(u, Deltau)

            # calculate internal force vector
            F_int = self.internal_force_vector()

            # calculate the residual of the system
            r = F_int - lam * self._load_increment_vector
            r = self.modified_residual(r)
            r_norm = np.linalg.norm(r)

            # initialize iteration counter
            kiter = 0
            deltau = np.zeros((self._number_of_dofs, 1), dtype=float)
            deltalam = 0.

            # iterate, until good result or so many iteration steps
            while(r_norm > self.tolerance and kiter < self.max_iteration_steps):

                # load-Control
                K = self.master_stiffness_matrix()
                K_s = self.modified_master_stiffness_matrix(K)
                
                deltau_star = np.linalg.solve(K_s, -r)
                deltau_bar = np.linalg.solve(K_s, self._load_increment_vector)
                Deltau += deltau
                
                a = float(deltau_bar.T @ deltau_bar)
                b = float(2 * (Deltau + deltau_star).T @ deltau_bar)
                c = float((Deltau + deltau_star).T @ (Deltau + deltau_star)) - self._arc_length ** 2
                
                [deltalam2_hat, deltalam1_hat] = np.roots([a, b, c])

                dotprod1 = float((Deltau + deltau_star + deltalam1_hat * deltau_bar).T @ Deltau)
                dotprod2 = float((Deltau + deltau_star + deltalam2_hat * deltau_bar).T @ Deltau)
                
                if dotprod1 >= dotprod2:
                    deltalam += deltalam1_hat
                    deltau += (deltau_star + deltalam1_hat * deltau_bar)
                else:
                    deltalam += deltalam2_hat
                    deltau += (deltau_star + deltalam2_hat * deltau_bar)
                    
                # update member data
                self.update_member_data(u + deltau, deltau)

                # calculate internal force vector
                F_int = self.internal_force_vector()

                # calculate the residual of the system
                r = F_int - (lam + deltalam) * self._load_increment_vector
                r = self.modified_residual(r)
                r_norm = np.linalg.norm(r)

                # update iterations counter
                kiter += 1
                if(kiter == self.max_iteration_steps):
                    raise RuntimeError(
                        'Newton-Raphson iterations did not converge!')

            """
            ------------------------------------------------------------------
            3. Update variables to their final value for the current increment
            ------------------------------------------------------------------
            """
            u += deltau
            Deltau_prev = Deltau + deltau
            lam += deltalam

            U = np.append(U, -u[self._load_boundary_condition])
            LAM = np.append(LAM, lam)

        return U, LAM
    
    def solve_the_system(self):
        if self._solver == "load-control":
            U, LAM = self.load_control()
        elif self._solver == "displacement-control":
            U, LAM = self.displacement_control()
        elif self._solver == "arc-length-control":
            U, LAM = self.arc_length_control()
        return U, LAM
    
    def solve_the_system_(self):
        lam = 0.
        u = np.zeros((self._number_of_dofs, 1), dtype=float)
        U = np.array([0.], dtype=float)
        LAM = np.array([0.], dtype=float)

        for n in range(self.number_of_load_increments):

            # set the predictor by equal load increments
            K = self.master_stiffness_matrix()
            K_s = self.modified_master_stiffness_matrix(K)
            dF = self.max_load / self.number_of_load_increments * self._load_increment_vector

            deltau = np.linalg.solve(K_s, dF)
            u_pre = u + deltau
            lam_pre = lam + self.max_load / self.number_of_load_increments

            # update member data
            self.update_member_data(u_pre, deltau)

            # calculate internal force vector
            F_int = self.internal_force_vector()

            # calculate the residual of the system
            r = F_int - lam_pre * self._load_increment_vector
            r = self.modified_residual(r)
            r_norm = np.linalg.norm(r)

            # copy them for iteration, "temp" means they are not on equilibrium path.
            u_temp = u_pre

            # initialize iteration counter
            kiter = 0

            plastic_strain = []
            internal_hardening_variable = []
            for ele in self._structure:
                plastic_strain.append(ele.__plastic_strain)
                internal_hardening_variable.append(
                    ele.__internal_hardening_variable)

            # iterate, until good result or so many iteration steps
            while(r_norm > self.tolerance and kiter < self.max_iteration_steps):

                # load-Control
                K = self.master_stiffness_matrix()
                K_s = self.modified_master_stiffness_matrix(K)
                deltau = np.linalg.solve(K_s, -r)
                u_temp += deltau

                # update member data
                self.update_member_data_iteration(
                    u_temp, deltau, plastic_strain, internal_hardening_variable)

                # calculate internal force vector
                F_int = self.internal_force_vector()

                # calculate the residual of the system
                r = F_int - lam_pre * self._load_increment_vector
                r = self.modified_residual(r)
                r_norm = np.linalg.norm(r)

                # update iterations counter
                kiter += 1
                if(kiter == self.max_iteration_steps):
                    raise RuntimeError(
                        'Newton-Raphson iterations did not converge!')

            """
            ------------------------------------------------------------------
            3. Update variables to their final value for the current increment
            ------------------------------------------------------------------
            """
            u = u_temp
            lam = lam_pre

            U = np.append(U, -u[self._load_boundary_condition])
            LAM = np.append(LAM, lam)

        return U, LAM

    def solve_the_system_dis(self):
        lam = 0.
        u = np.zeros((self._number_of_dofs, 1), dtype=float)
        U = np.array([0.], dtype=float)
        LAM = np.array([0.], dtype=float)

        for n in range(self.number_of_load_increments):

            # set the predictor by equal load increments
            K = self.master_stiffness_matrix()
            K_s = self.modified_master_stiffness_matrix(K)
            q = self._load_increment_vector

            v = np.linalg.solve(K_s, q)

            f = float(np.sqrt(1 + v.T @ v))
            l = 0.01

            deltalam = np.sign(q.T @ v) * l / np.sqrt(1**2*abs(float(q.T@v)))
            deltau = deltalam * v

            lam_pre = lam + deltalam
            u_pre = u + deltau

            # update member data
            self.update_member_data(u_pre, deltau)

            # calculate internal force vector
            F_int = self.internal_force_vector()

            c = np.array([[0]])

            # calculate the residual of the system
            r = F_int - lam_pre * self._load_increment_vector
            r = self.modified_residual(r)
            r_norm = np.linalg.norm(r)

            # copy them for iteration, "temp" means they are not on equilibrium path.
            u_temp = u_pre
            lam_temp = lam_pre

            # initialize iteration counter
            kiter = 0

            plastic_strain = []
            internal_hardening_variable = []
            # for ele in self._structure:
            #     plastic_strain.append(ele.__plastic_strain)
            # internal_hardening_variable.append(
            #     ele.__internal_hardening_variable)

            # iterate, until good result or so many iteration steps
            while(r_norm > self.tolerance and kiter < self.max_iteration_steps):

                # load-Control
                K = self.master_stiffness_matrix()
                K_s = self.modified_master_stiffness_matrix(K)

                v = np.linalg.solve(K_s, q)
                f = float(np.sqrt(1 + v.T @ v))

                lhs = np.r_[np.c_[K_s, -q], np.c_[(v / f).T, 1 / f]]
                rhs = -np.r_[r, c]
                x = np.linalg.solve(lhs, rhs)
                deltau = x[0: -1]
                deltalam = float(x[-1])

                u_temp += deltau
                lam_temp += deltalam

                # update member data
                self.update_member_data_iteration(
                    u_temp, deltau, plastic_strain, internal_hardening_variable)

                # calculate internal force vector
                F_int = self.internal_force_vector()

                # calculate the residual of the system
                r = F_int - lam_pre * self._load_increment_vector
                r = self.modified_residual(r)
                r_norm = np.linalg.norm(r)

                # update iterations counter
                kiter += 1
                if(kiter == self.max_iteration_steps):
                    raise RuntimeError(
                        'Newton-Raphson iterations did not converge!')

            """
            ------------------------------------------------------------------
            3. Update variables to their final value for the current increment
            ------------------------------------------------------------------
            """
            u = u_temp
            lam = lam_temp

            U = np.append(U, -u[self._load_boundary_condition])
            LAM = np.append(LAM, lam)

        return U, LAM

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
            Y = np.zeros((self._number_of_nodes))
            x = np.zeros((self._number_of_nodes))
            y = np.zeros((self._number_of_nodes))
            for iele in range(self._number_of_elements):
                X[iele] = self._structure[iele].initial_coordinate_node_1[0]
                Y[iele] = self._structure[iele].initial_coordinate_node_1[1]
                x[iele] = self._structure[iele].current_coordinate_node_1[0]
                y[iele] = self._structure[iele].current_coordinate_node_1[1]
                if iele == self._number_of_elements - 1:
                    X[iele + 1] = self._structure[iele].initial_coordinate_node_2[0]
                    Y[iele + 1] = self._structure[iele].initial_coordinate_node_2[1]
                    x[iele + 1] = self._structure[iele].current_coordinate_node_2[0]
                    y[iele + 1] = self._structure[iele].current_coordinate_node_2[1]
            # Plot both configurations
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            fig, ax = plt.subplots(dpi=150)
            ax.plot(X, Y, '.--', label='undeformed configuration')
            # ax.scatter(X, Y)
            ax.plot(x, y, '.-', label='deformed configuration')
            # ax.scatter(x, y)
            ax.legend(loc='lower right')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
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
            fig = plt.figure(dpi=150)
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
