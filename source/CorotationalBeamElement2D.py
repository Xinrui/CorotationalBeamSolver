# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou
"""

import numpy as np

from numpy import inf, tan, arctan2, arctan


class CorotationalBeamElement2D():
    """It is a class formulating two-dimensional corotational beam elements."""

    def __init__(self, beamtype, youngs_modulus, poisson_ratio, width, height, initial_coordinate_node_1, initial_coordinate_node_2, element_id):
        """ Initialize a 2D co-rotational beam element.

        Parameters
        ----------
        beamtype: str
            the type of beam element. Here only Bernoulli element implemented.
        youngs_modulus: int/float
            Young's modulus.
        poisson_ratio: float
            Poisson ratio, not interesting for Bernoulli elements.
        width: int/float
            width of the beam, perpendicular to local z-axis.
        height: int/float
            height of the beam, perpendicular to local y-axis.
        initial_coordinate_node_1: numpy.ndarray
            initial coordinate of the 1st node: [X1, Z1].T.
        initial_coordinate_node_2: numpy.ndarray
            initial coordinate of the 2nd node: [X2, Z2].T.
        element_id: int
            The number of the element in the system.

        Returns
        -------
        None.

        """
        self.beamtype = beamtype
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        self.width = width
        self.height = height
        self.initial_coordinate_node_1 = initial_coordinate_node_1
        self.initial_coordinate_node_2 = initial_coordinate_node_2

        self.initial_length = self.calculate_initial_length()
        self.initial_angle = self.calculate_initial_angle()
        self.area = self.calculate_area()
        self.moment_of_inertia = self.calculate_moment_of_inertia()

        self.analysis = "elastic"  # default value
        self.global_displacement = np.zeros((6, 1), dtype=float)
        self.element_freedom_table = np.linspace(
            3 * element_id, 3 * element_id + 5, num=6, dtype=int)

    def calculate_area(self):
        """Calculate the cross-sectional area of the beam element.

        Parameters
        ----------
        None.

        Returns
        -------
        cross-sectional area: A = b * h

        """
        return self.width * self.height

    def calculate_moment_of_inertia(self):
        """Calculate the moment of inertia of the beam element.

        Parameters
        ----------
        None.

        Returns
        -------
        moment of inertia: I = 1/12 * b * h^3

        """
        return 1/12 * self.width * (self.height) ** 3

    def current_coordinate_node_1(self):
        """Calculate the coodinate of node 1 in current configuration.

        Parameters
        ----------
        None.

        Returns
        -------
        current coordinate of the 1st node: x1 = X1 + u1.

        """
        return self.initial_coordinate_node_1 + self.global_displacement[0: 2]

    def current_coordinate_node_2(self):
        """Calculate the coodinate of node 2 in current configuration.

        Parameters
        ----------
        None.

        Returns
        -------
        current coordinate of the 2nd node: x2 = X2 + u2.

        """
        return self.initial_coordinate_node_2 + self.global_displacement[3: 5]

    def apply_hardening_model(self, hardening_model, gauss_number, yield_stress, kinematic_hardening_modulus, plastic_modulus, saturation_stress,
                              modified_modulus, exponent):
        """Apply hardening model on the beam element.

        Parameters
        ----------
        hardening_model: str
            hardening model. linear hardening, exponential hardening and ramberg-osgood hardening are implemented.
        gauss_number: tuple
            Number of Gauss points on xi-axis and eta-axis. Set as a tuple with 2 numbers, such as (2, 2)
        yield_stress: int/float
            Value of yield stress.
        kinematic_hardening_modulus: int/float
            Value of kinematic hardening modulus.
        plastic_modulus: int/float
            Value of plastic modulus.
        saturation_stress: int/float
            Value of saturation stress.
        modified_modulus: int/float
            Value of modified modulus.
        exponent: int/float
            Value of exponent.

        Returns
        -------
        None.

        """
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
        """Calculate local displacement vector at current state.

        Parameters
        ----------
        None.

        Returns
        -------
        p_l: numpy.ndarray
            local displacement p_l = [u_bar, theta1_bar, theta2_bar].T


        """
        global_rotation_node_1 = float(self.global_displacement[2])
        global_rotation_node_2 = float(self.global_displacement[5])

        u_bar = self.current_length() - self.initial_length
        tanbeta = self.current_angle_trigonometric("local")

        if tanbeta == inf:
            theta1_bar = arctan(-1 /
                                tan(global_rotation_node_1 + self.initial_angle))
            theta2_bar = arctan(-1 /
                                tan(global_rotation_node_2 + self.initial_angle))
        else:
            theta1_bar = arctan((tan(global_rotation_node_1 + self.initial_angle) - tanbeta) / (
                1 + tan(global_rotation_node_1 + self.initial_angle) * tanbeta))
            theta2_bar = arctan((tan(global_rotation_node_2 + self.initial_angle) - tanbeta) / (
                1 + tan(global_rotation_node_2 + self.initial_angle) * tanbeta))

        return np.array([[u_bar, theta1_bar, theta2_bar]]).T

    def calculate_initial_length(self):
        """Calculate the initial length of the beam element.

        Parameters
        ----------
        None.

        Returns
        -------
        L: float

        """
        return np.linalg.norm(self.initial_coordinate_node_2 -
                              self.initial_coordinate_node_1)

    def calculate_initial_angle(self):
        """Calculate the initial angle of the beam element.

        Parameters
        ----------
        None.

        Returns
        -------
        beta_0: float

        """
        return float(arctan2(self.initial_coordinate_node_2[1] -
                     self.initial_coordinate_node_1[1],
                     self.initial_coordinate_node_2[0] -
                     self.initial_coordinate_node_1[0]))

    def current_length(self):
        """Calculate the current length of the beam element.

        Parameters
        ----------
        None.

        Returns
        -------
        l: float

        """
        return np.linalg.norm(self.current_coordinate_node_2() -
                              self.current_coordinate_node_1())

    def current_angle_trigonometric(self, str):
        """Calculate the sine and cosine value of current angle of the beam element.

        Parameters
        ----------
        str: str
            can be 'local' or 'global'.

        Returns
        -------
        cos(beta): float
        sin(beta): float
        tan(beta): float

            If calculate local displacement, the function returns only tangent;
            If calculate global matrix or force, return cosine and sine.
        """
        x1 = self.current_coordinate_node_1()
        x2 = self.current_coordinate_node_2()
        l = self.current_length()
        if str == "global":
            sinbeta = float((x2[1] - x1[1]) / l)
            cosbeta = float((x2[0] - x1[0]) / l)
            return cosbeta, sinbeta
        elif str == "local":
            tanbeta = float((x2[1] - x1[1]) / (x2[0] - x1[0]))
            return tanbeta
        else:
            raise ValueError("Wrong input!")

    def local_stiffness_force(self):
        """Calculate the local stiffness matrix kl and local force vector fl of the beam element.

        Parameters
        ----------
        None.

        Returns
        -------
        kl: np.ndarray
            local stiffness matrix.
        fl: np.ndarray
            local force vector

        """
        E, A, I, L = self.youngs_modulus, self.area, self.moment_of_inertia, self.initial_length

        if self.analysis == "elastic":
            k_l = np.array([[A,  0.,    0.],
                           [0., 4 * I, 2 * I],
                           [0., 2 * I, 4 * I]])
            k_l *= E / L
            f_l = k_l @ self.local_displacement()

        else:
            kl_11, kl_22, kl_33, kl_12, kl_13, kl_23, N, M1, M2 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
            for n in range(self.num_of_gauss_locations):
                x, z = self.global_gauss_locations_mat[0: 2, n]
                fac = self.width * self.height * L / 4

                kl_11 += fac * self.weights_mat[n] * \
                    self.tangent_modulus[n] / L ** 2
                kl_22 += fac * self.weights_mat[n] * self.tangent_modulus[n] * z ** 2 * (4 / L - 6 *
                                                                                         x / L ** 2) ** 2
                kl_33 += fac * self.weights_mat[n] * self.tangent_modulus[n] * z ** 2 * (2 / L - 6 *
                                                                                         x / L ** 2) ** 2
                kl_12 += fac * self.weights_mat[n] * self.tangent_modulus[n] * z * (
                    4 / L ** 2 - 6 * x / L ** 3)
                kl_13 += fac * self.weights_mat[n] * self.tangent_modulus[n] * z * (
                    2 / L ** 2 - 6 * x / L ** 3)
                kl_23 += fac * self.weights_mat[n] * self.tangent_modulus[n] * z ** 2 * (4 / L - 6 * x / (
                    L) ** 2) * (2 / L - 6 * x / L ** 2)

                N += fac * self.weights_mat[n] * \
                    self.stress[n] / L
                M1 += fac * self.weights_mat[n] * self.stress[n] * z * (
                    4 / L - 6 * x / L ** 2)
                M2 += fac * self.weights_mat[n] * self.stress[n] * z * (
                    2 / L - 6 * x / L ** 2)

            k_l = np.array([[kl_11, kl_12, kl_13],
                            [kl_12, kl_22, kl_23],
                            [kl_13, kl_23, kl_33]])

            f_l = np.array([[N, M1, M2]]).T

        return k_l, f_l

    def map_local_coordinate_to_global_coordinate(self, xi, eta):
        """Define the linear map between Gaussian space and local space.

        Parameters
        ----------
        xi: float
            the 1st Gaussian coordinate.
        eta: float
            the 2nd Gaussian coordinate.

        Returns
        -------
        x: float
            the 1st local coordinate.
        z: float
            the 2nd local coordinate.

        """
        x = self.initial_length / 2 * (xi + 1)
        z = self.height / 2 * eta

        return x, z

    def strain(self, x, z):
        """Calculate the strain at a point expressed in local coordinate.

        Parameters
        ----------
        x: float
            the 1st local coordinate.
        z: float
            the 2nd local coordinate.

        Returns
        -------
        strain: float/numpy.ndarray
            If it is Timoshenko beam, shear strain will occur so that there are two components
            in the strain.

        """
        pl = self.local_displacement()
        if self.beamtype == "Bernoulli":
            return float(pl[0] / self.initial_length + z * ((4 / self.initial_length - 6 *
                                                            x / (self.initial_length) ** 2) * pl[1] +
                                                            (2 / self.initial_length - 6 *
                                                            x / (self.initial_length) ** 2) * pl[2]))
        else:
            raise ValueError("Wrong element type, please check!")

    def perform_linear_hardening(self):
        """Update state variables using return-mapping algorithm for linear hardening model.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
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
        """Update state variables using return-mapping algorithm for exponential hardening model.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """

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
        """Update state variables using return-mapping algorithm for Ramberg-Osgood hardening model.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
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
        """Calculate the global stiffness matrix kg and global force vector fg of the beam element.

        Parameters
        ----------
        None.

        Returns
        -------
        kg: np.ndarray
            global stiffness matrix.
        fg: np.ndarray
            global force vector

        """
        cosbeta, sinbeta = self.current_angle_trigonometric("global")
        l = self.current_length()

        B = np.array([[-cosbeta, -sinbeta, 0., cosbeta, sinbeta, 0.],
                      [-sinbeta / l, cosbeta/l, 1., sinbeta/l, -cosbeta/l, 0.],
                      [-sinbeta / l, cosbeta/l, 0., sinbeta/l, -cosbeta/l, 1.]])
        r = np.array([[-cosbeta, -sinbeta, 0.,  cosbeta, sinbeta, 0.]]).T
        z = np.array([[sinbeta, -cosbeta, 0., -sinbeta, cosbeta, 0.]]).T

        K_l, f_l = self.local_stiffness_force()

        K_m = f_l[0] / l * z @ z.T
        + (f_l[1] + f_l[2]) / (l ** 2) * (r @ z.T + z @ r.T)

        K_g = B.T @ K_l @ B + K_m
        f_g = B.T @ f_l

        return K_g, f_g
