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

        self.__analysis = "elastic"
        self.__beamtype = "Bernoulli"

        self.__youngs_modulus = None
        self.__shear_modulus = None

        self.__element_freedom_table = None

        self.__incremental_global_displacement = np.zeros((6, 1), dtype=float)
        self.__num_of_gauss_locations_xi = 2
        self.__num_of_gauss_locations_eta = 2

        # All plastic cases, including Perfect Plasticity / No Hardening
        self.__yield_stress = None

        # Kinematic Hardening
        self.__kinematic_hardening_modulus = 0.0

        # Linear Hardening
        self.__plastic_modulus = None

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
    def shear_modulus(self):
        return self.__shear_modulus

    @shear_modulus.setter
    def shear_modulus(self, val):
        """Set the Shear modulus of the beam element: __shear_modulus > 0.

        Raises:
            TypeError: If value is not a positive float number.

        """

        if not isinstance(val, float):
            raise TypeError("Shear modulus must be a float number!")
        elif val <= 0:
            raise ValueError("Shear modulus must be positive!")
        else:
            self.__shear_modulus = val

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
    def plastic_strain(self):
        return self.__plastic_strain

    @plastic_strain.setter
    def plastic_strain(self, val):
        if not isinstance(val, np.ndarray):
            raise TypeError(
                "The plastic strain at gauss points of beam must be a numpy array!")
        else:
            self.__plastic_strain = val

    @property
    def internal_hardening_variable(self):
        return self.__internal_hardening_variable

    @internal_hardening_variable.setter
    def internal_hardening_variable(self, val):
        if not isinstance(val, np.ndarray):
            raise TypeError(
                "The internal hardening variable at gauss points of beam must be a numpy array!")
        else:
            self.__internal_hardening_variable = val

    @property
    def back_stress(self):
        return self.__back_stress

    @back_stress.setter
    def back_stress(self, val):
        if not isinstance(val, np.ndarray):
            raise TypeError(
                "The back stress at gauss points of beam must be a numpy array!")
        else:
            self.__back_stress = val

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
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__plastic_strain = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__back_stress = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__tangent_modulus = self.__youngs_modulus * \
                np.ones((self.__num_of_gauss_locations_xi,
                        self.__num_of_gauss_locations_eta), dtype=float)
        elif val == "linear hardening":
            self.__analysis = val
            self.__stress = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__plastic_strain = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__back_stress = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__tangent_modulus = self.__youngs_modulus * \
                np.ones((self.__num_of_gauss_locations_xi,
                        self.__num_of_gauss_locations_eta), dtype=float)
            self.__internal_hardening_variable = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
        elif val == "exponential hardening":
            self.__analysis = val
            self.__stress = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__plastic_strain = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__back_stress = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__tangent_modulus = self.__youngs_modulus * \
                np.ones((self.__num_of_gauss_locations_xi,
                        self.__num_of_gauss_locations_eta), dtype=float)
            self.__internal_hardening_variable = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
        elif val == "ramberg-osgood hardening":
            self.__analysis = val
            self.__stress = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__plastic_strain = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__back_stress = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__tangent_modulus = self.__youngs_modulus * \
                np.ones((self.__num_of_gauss_locations_xi,
                        self.__num_of_gauss_locations_eta), dtype=float)
            self.__internal_hardening_variable = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
        else:
            print("These four models are implemented here: 1. elastic; 2. linear hardening; 3. exponential hardening; 4. ramberg-osgood hardening.")
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
    def kinematic_hardening_modulus(self):
        return self.__kinematic_hardening_modulus

    @kinematic_hardening_modulus.setter
    def kinematic_hardening_modulus(self, val):
        """
        Set the kinematic hardening modulus of the beam element.

        Raises:
            TypeError: If value is not a float number.
            ValueError: If value is not positive.

        """
        if not isinstance(val, float):
            raise TypeError(
                "Kinematic hardening modulus must be a float number!")
        elif val <= 0:
            raise ValueError("Kinematic hardening modulus must be non-negative!")
        else:
            self.__kinematic_hardening_modulus = val

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
    def local_stiffness_matrix(self):
        if self.__analysis == "elastic":
            return self.__youngs_modulus / self.initial_length * \
                np.array([[self.area, 0., 0.],
                          [0., 4 * self.moment_of_inertia, 2 * self.moment_of_inertia],
                          [0., 2 * self.moment_of_inertia, 4 * self.moment_of_inertia]])
        else:
            kl_11 = 0.0
            kl_22 = 0.0
            kl_33 = 0.0
            kl_12 = 0.0
            kl_13 = 0.0
            kl_23 = 0.0
            [gauss_locations_xi, weights_xi] = np.polynomial.legendre.leggauss(
                self.__num_of_gauss_locations_xi)
            [gauss_locations_eta, weights_eta] = np.polynomial.legendre.leggauss(
                self.__num_of_gauss_locations_eta)

            for ixi in range(self.__num_of_gauss_locations_xi):
                for jeta in range(self.__num_of_gauss_locations_eta):
                    x, z = self.map_local_coordinate_to_global_coordinate(
                        gauss_locations_xi[ixi], gauss_locations_eta[jeta])
                    fac = self.__width * self.__height * self.initial_length / 4
                    weight = weights_xi[ixi] * weights_eta[jeta]

                    kl_11 += fac * weight * \
                        self.__tangent_modulus[ixi, jeta] / \
                        (self.initial_length) ** 2
                    kl_22 += fac * weight * self.__tangent_modulus[ixi, jeta] * z ** 2 * (4 / self.initial_length - 6 *
                                                                             x / (self.initial_length) ** 2) ** 2
                    kl_33 += fac * weight * self.__tangent_modulus[ixi, jeta] * z ** 2 * (2 / self.initial_length - 6 *
                                                                             x / (self.initial_length) ** 2) ** 2
                    kl_12 += fac * weight * self.__tangent_modulus[ixi, jeta] * z * (
                        4 / (self.initial_length) ** 2 - 6 * x / (self.initial_length) ** 3)
                    kl_13 += fac * weight * self.__tangent_modulus[ixi, jeta] * z * (
                        2 / (self.initial_length) ** 2 - 6 * x / (self.initial_length) ** 3)
                    kl_23 += fac * weight * self.__tangent_modulus[ixi, jeta] * z ** 2 * (4 / self.initial_length - 6 * x / (
                        self.initial_length) ** 2) * (2 / self.initial_length - 6 * x / (self.initial_length) ** 2)

            return np.c_[np.r_[kl_11, kl_12, kl_13], np.r_[kl_12, kl_22, kl_23], np.r_[kl_13, kl_23, kl_33]]

    @property
    def local_force(self):
        if self.__analysis == "elastic":
            return self.local_stiffness_matrix @ self.local_displacement
        else:
            # local_force = np.zeros((3, 1), dtype=float)
            [gauss_locations_xi, weights_xi] = np.polynomial.legendre.leggauss(
                self.__num_of_gauss_locations_xi)
            [gauss_locations_eta, weights_eta] = np.polynomial.legendre.leggauss(
                self.__num_of_gauss_locations_eta)

            N = 0.0
            M1 = 0.0
            M2 = 0.0

            for i in range(self.__num_of_gauss_locations_xi):
                for j in range(self.__num_of_gauss_locations_eta):
                    x, z = self.map_local_coordinate_to_global_coordinate(
                        gauss_locations_xi[i], gauss_locations_eta[j])

                    fac = self.__width * self.__height * self.initial_length / 4
                    weight = weights_xi[i] * weights_eta[j]

                    N += fac * weight * \
                        self.__stress[i, j] / self.initial_length

                    M1 += fac * weight * self.__stress[i, j] * z * (
                        4 / self.initial_length - 6 * x / (self.initial_length) ** 2)

                    M2 += fac * weight * self.__stress[i, j] * z * (
                        2 / self.initial_length - 6 * x / (self.initial_length) ** 2)

            return np.r_[N, M1, M2]

    def map_local_coordinate_to_global_coordinate(self, xi, eta):
        x = self.initial_length / 2 * (xi + 1)
        z = self.__height / 2 * eta

        return x, z

    def strain(self, x, z):
        return float(self.local_displacement[0] / self.initial_length + z * ((4 / self.initial_length - 6 *
                                                                        x / (self.initial_length) ** 2) * self.local_displacement[1] +
                                                                       (2 / self.initial_length - 6 *
                                                                        x / (self.initial_length) ** 2) * self.local_displacement[2]))

    def perform_perfect_plasticity(self):
        gauss_locations_xi = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_xi)[0]
        gauss_locations_eta = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_eta)[1]

        def yield_stress_function(alpha):
            return self.__yield_stress

        for ixi in range(self.__num_of_gauss_locations_xi):
            for jeta in range(self.__num_of_gauss_locations_eta):
                x, z = self.map_local_coordinate_to_global_coordinate(
                    gauss_locations_xi[ixi], gauss_locations_eta[jeta])
                stress_trial = self.__youngs_modulus * \
                    (float(self.strain(x, z)) -
                     self.__plastic_strain[ixi, jeta])
                yield_condition_trial = np.abs(
                    stress_trial) - yield_stress_function(self.__internal_hardening_variable[ixi, jeta])

                if yield_condition_trial <= 0:
                    self.__stress[ixi, jeta] = stress_trial
                    self.__tangent_modulus[ixi, jeta] = self.__youngs_modulus
                else:
                    deltagamma = yield_condition_trial / self.__youngs_modulus
                    self.__stress[ixi, jeta] = (
                        1 - deltagamma * self.__youngs_modulus / np.abs(stress_trial)) * stress_trial
                    self.__plastic_strain[ixi,
                                          jeta] += deltagamma * np.sign(stress_trial)
                    self.__tangent_modulus[ixi, jeta] = 0.0

    def perform_linear_hardening(self):
        gauss_locations_xi = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_xi)[0]
        gauss_locations_eta = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_eta)[0]

        yield_stress_function = lambda alpha: self.__yield_stress + self.__plastic_modulus * alpha

        for ixi in range(self.__num_of_gauss_locations_xi):
            for jeta in range(self.__num_of_gauss_locations_eta):
                x, z = self.map_local_coordinate_to_global_coordinate(gauss_locations_xi[ixi], gauss_locations_eta[jeta])
                stress_trial = self.__youngs_modulus * (self.strain(x, z) - self.__plastic_strain[ixi, jeta])
                relative_stress_trial = stress_trial - self.__back_stress[ixi, jeta]
                yield_condition_trial = np.abs(relative_stress_trial) - yield_stress_function(self.__internal_hardening_variable[ixi, jeta])

                if yield_condition_trial <= 0:
                    self.__stress[ixi, jeta] = stress_trial
                    self.__tangent_modulus[ixi, jeta] = self.__youngs_modulus
                else:
                    deltagamma = yield_condition_trial / (self.__youngs_modulus + self.__plastic_modulus + self.__kinematic_hardening_modulus)
                    self.__stress[ixi, jeta] = stress_trial - deltagamma * self.__youngs_modulus * np.sign(relative_stress_trial)
                    self.__plastic_strain[ixi, jeta] += deltagamma * np.sign(relative_stress_trial)
                    self.__back_stress += deltagamma * self.__kinematic_hardening_modulus * np.sign(relative_stress_trial)
                    self.__internal_hardening_variable[ixi, jeta] += deltagamma
                    self.__tangent_modulus[ixi, jeta] = self.__youngs_modulus * (self.__plastic_modulus + self.__kinematic_hardening_modulus) / (self.__youngs_modulus + self.__plastic_modulus + self.__kinematic_hardening_modulus)

    def perform_exponential_hardening(self):
        gauss_locations_xi = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_xi)[0]
        gauss_locations_eta = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_eta)[0]

        def yield_stress_function(alpha):
            return self.__yield_stress + (self.__saturation_stress - self.__yield_stress) * (1 - np.exp(-self.__exponent * alpha))

        hardening_limit_stress = self.__saturation_stress - self.__yield_stress

        for ixi in range(self.__num_of_gauss_locations_xi):
            for jeta in range(self.__num_of_gauss_locations_eta):
                x, z = self.map_local_coordinate_to_global_coordinate(
                    gauss_locations_xi[ixi], gauss_locations_eta[jeta])
                stress_trial = self.__youngs_modulus * \
                    (float(self.strain(x, z)) -
                     self.__plastic_strain[ixi, jeta])
                relative_stress_trial = stress_trial - \
                    self.__back_stress[ixi, jeta]
                yield_condition_trial = np.abs(
                    relative_stress_trial) - yield_stress_function(self.__internal_hardening_variable[ixi, jeta])

                if yield_condition_trial <= 0:
                    self.__stress[ixi, jeta] = stress_trial
                    self.__tangent_modulus[ixi, jeta] = self.__youngs_modulus
                else:
                    deltagamma = 0
                    residual = yield_condition_trial - deltagamma * (self.__youngs_modulus + self.__kinematic_hardening_modulus) - yield_stress_function(
                        self.__internal_hardening_variable[ixi, jeta] + deltagamma) + yield_stress_function(self.__internal_hardening_variable[ixi, jeta])

                    maxiter = 10
                    iteration_counter = 0
                    tolerance = 1.0e-5

                    while residual > tolerance and iteration_counter < maxiter:
                        dR_ddeltagamma = -(self.__youngs_modulus + self.__kinematic_hardening_modulus) - hardening_limit_stress * self.__exponent * \
                            np.exp(-self.__exponent *
                                   (self.__internal_hardening_variable[ixi, jeta] + deltagamma))
                        d_g = - residual / dR_ddeltagamma
                        deltagamma += d_g
                        residual = yield_condition_trial - deltagamma * (self.__youngs_modulus + self.__kinematic_hardening_modulus) - yield_stress_function(
                            self.__internal_hardening_variable[ixi, jeta] + deltagamma) + yield_stress_function(self.__internal_hardening_variable[ixi, jeta])
                        iteration_counter += 1

                    self.__stress[ixi, jeta] = stress_trial - deltagamma * \
                        self.__youngs_modulus * np.sign(relative_stress_trial)
                    self.__plastic_strain[ixi,
                                          jeta] += deltagamma * np.sign(relative_stress_trial)
                    self.__back_stress += deltagamma * \
                        self.__kinematic_hardening_modulus * \
                        np.sign(relative_stress_trial)
                    self.__internal_hardening_variable[ixi,
                                                       jeta] += deltagamma
                    self.__tangent_modulus[ixi, jeta] = self.__youngs_modulus * (self.__kinematic_hardening_modulus + hardening_limit_stress * self.__exponent * np.exp(-self.__exponent * self.__internal_hardening_variable[ixi, jeta])) / (
                        self.__youngs_modulus + self.__kinematic_hardening_modulus + hardening_limit_stress * self.__exponent * np.exp(-self.__exponent * self.__internal_hardening_variable[ixi, jeta]))

    def perform_ramberg_osgood_hardening(self):
        gauss_locations_xi = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_xi)[0]
        gauss_locations_eta = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_eta)[0]

        def yield_stress_function(alpha):
            return self.__yield_stress + self.__modified_modulus * np.power(alpha, self.__exponent)

        for ixi in range(self.__num_of_gauss_locations_xi):
            for jeta in range(self.__num_of_gauss_locations_eta):
                x, z = self.map_local_coordinate_to_global_coordinate(
                    gauss_locations_xi[ixi], gauss_locations_eta[jeta])
                stress_trial = self.__youngs_modulus * \
                    (float(self.strain(x, z)) -
                     self.__plastic_strain[ixi, jeta])
                relative_stress_trial = stress_trial - \
                    self.__back_stress[ixi, jeta]
                yield_condition_trial = np.abs(
                    relative_stress_trial) - yield_stress_function(self.__internal_hardening_variable[ixi, jeta])

                if yield_condition_trial <= 0:
                    self.__stress[ixi, jeta] = stress_trial
                    self.__tangent_modulus[ixi, jeta] = self.__youngs_modulus
                else:
                    deltagamma = 0
                    residual = yield_condition_trial - deltagamma * (self.__youngs_modulus + self.__kinematic_hardening_modulus) - yield_stress_function(
                        self.__internal_hardening_variable[ixi, jeta] + deltagamma) + yield_stress_function(self.__internal_hardening_variable[ixi, jeta])

                    maxiter = 10
                    iteration_counter = 0
                    tolerance = 1.0e-5

                    while residual > tolerance and iteration_counter < maxiter:
                        dR_ddeltagamma = -(self.__youngs_modulus + self.__kinematic_hardening_modulus) - self.__exponent * self.__modified_modulus * \
                            np.power(
                                self.__internal_hardening_variable[ixi, jeta] + deltagamma, self.__exponent - 1)
                        d_g = - residual / dR_ddeltagamma
                        deltagamma += d_g
                        residual = yield_condition_trial - deltagamma * (self.__youngs_modulus + self.__kinematic_hardening_modulus) - yield_stress_function(
                            self.__internal_hardening_variable[ixi, jeta] + deltagamma) + yield_stress_function(self.__internal_hardening_variable[ixi, jeta])
                        iteration_counter += 1

                    self.__stress[ixi, jeta] = stress_trial - deltagamma * \
                        self.__youngs_modulus * np.sign(relative_stress_trial)
                    self.__plastic_strain[ixi,
                                          jeta] += deltagamma * np.sign(relative_stress_trial)
                    self.__back_stress += deltagamma * \
                        self.__kinematic_hardening_modulus * \
                        np.sign(relative_stress_trial)
                    self.__internal_hardening_variable[ixi,
                                                       jeta] += deltagamma
                    self.__tangent_modulus[ixi, jeta] = self.__youngs_modulus * (self.__kinematic_hardening_modulus + self.__exponent * self.__modified_modulus * np.power(self.__internal_hardening_variable[ixi, jeta], self.__exponent - 1)) / (
                        self.__youngs_modulus + self.__kinematic_hardening_modulus + self.__exponent * self.__modified_modulus * np.power(self.__internal_hardening_variable[ixi, jeta], self.__exponent - 1))

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
        return self.transformation_matrix.T @ self.local_stiffness_matrix @ self.transformation_matrix

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