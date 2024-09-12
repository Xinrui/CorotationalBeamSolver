# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou
"""
from math import sin, cos
import numpy as np
import source.Utilities as util


class CorotationalBeamElement3D():
    """It is a class formulating three-dimensional corotational beam elements."""

    def __init__(self, beamtype, youngs_modulus, poisson_ratio, width, height, initial_coordinate_node_1, initial_coordinate_node_2, element_id, e_2o, e_3o):
        """ Initialize a 3D co-rotational beam element.

        Parameters
        ----------
        beamtype: str
            the type of beam element, which can be Bernoulli or Timoshenko.
        youngs_modulus: int/float
            Young's modulus.
        poisson_ratio: float
            Poisson ratio, not interesting for Bernoulli elements.
        width: int/float
            width of the beam, perpendicular to local z-axis.
        height: int/float
            height of the beam, perpendicular to local y-axis.
        initial_coordinate_node_1: numpy.ndarray
            initial coordinate of the 1st node: [X1, Y1, Z1].T.
        initial_coordinate_node_2: numpy.ndarray
            initial coordinate of the 2nd node: [X2, Y2, Z2].T.
        element_id: int
            The number of the element in the system.

        Returns
        -------
        None.

        """
        self.initial_coordinate_node_1 = initial_coordinate_node_1
        self.initial_coordinate_node_2 = initial_coordinate_node_2

        self.beamtype = beamtype
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        self.width = width
        self.height = height

        self.e_2o = e_2o
        self.e_3o = e_3o

        self.initial_length = self.calculate_initial_length()
        self.initial_local_frame = self.calculate_initial_local_frame()
        self.first_lame_constant = self.calculate_first_lame_constant()
        self.second_lame_constant = self.calculate_second_lame_constant()
        self.bulk_modulus = self.calculate_bulk_modulus()
        self.area = self.calculate_area()
        self.moment_of_inertia_y = self.calculate_moment_of_inertia_y()
        self.moment_of_inertia_z = self.calculate_moment_of_inertia_z()
        self.polar_moment_of_inertia = self.calculate_polar_moment_of_inertia()
        self.fourth_order_polar_moment_of_inertia = self.calculate_fourth_order_polar_moment_of_inertia()

        self.current_orientation_node_1 = np.eye(3)
        self.current_orientation_node_2 = np.eye(3)

        self.analysis = "elastic"
        self.global_displacement = np.zeros((12, 1), dtype=float)
        self.element_freedom_table = np.linspace(
            6 * element_id, 6 * element_id + 11, num=12, dtype=int)

    def calculate_first_lame_constant(self):
        """ Calculate the first lame constant λ.

        Parameters
        ----------
        None.

        Returns
        -------
        the first lame constant: float

        """
        return self.youngs_modulus * self.poisson_ratio / ((1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio))

    def calculate_second_lame_constant(self):
        """ Calculate the second lame constant μ.

        Parameters
        ----------
        None.

        Returns
        -------
        the second lame constant: float

        """
        return self.youngs_modulus / (2 * (1 + self.poisson_ratio))

    def calculate_bulk_modulus(self):
        """ Calculate the bulk modulus κ.

        Parameters
        ----------
        None.

        Returns
        -------
        the bulk modulus: float

        """
        return self.first_lame_constant + 2/3 * self.second_lame_constant

    def calculate_area(self):
        """Calculate the cross-sectional area of the beam element.

        Parameters
        ----------
        None.

        Returns
        -------
        the cross-sectional area: float
            A = b * h

        """
        return self.width * self.height

    def calculate_moment_of_inertia_y(self):
        """Calculate the moment of inertia Iy / I33 of the beam element.

        Parameters
        ----------
        None.

        Returns
        -------
        the moment of inertia Iy: float
            I_y = 1/12 * b * h^3

        """
        return 1/12 * self.width * (self.height) ** 3

    def calculate_moment_of_inertia_z(self):
        """Calculate the moment of inertia Iz / I22 of the beam element.

        Parameters
        ----------
        None.

        Returns
        -------
        the moment of inertia Iz: float
            I_z = 1/12 * h * b^3

        """
        return 1/12 * (self.width) ** 3 * self.height

    def calculate_polar_moment_of_inertia(self):
        """Calculate the polar moment of inertia Io of the beam element.

        Parameters
        ----------
        None.

        Returns
        -------
        the polar moment of inertia Io: float
            I_o = 1/12 * (b^3 * h + b * h^3)

        """
        return 1/12 * (self.width * (self.height) ** 3 + (self.width) ** 3 * self.height)

    def calculate_fourth_order_polar_moment_of_inertia(self):
        """Calculate the 4th polar moment of inertia Irr of the beam element.

        Parameters
        ----------
        None.

        Returns
        -------
        the 4th polar moment of inertia Irr: float
            I_rr = b^5 * h / 80 + b^3 * h^3 / 72 + b * h^5 / 80

        """
        return 1/80 * (self.width * (self.height) ** 5 + (self.width) ** 5 * self.height) + 1/72 * (self.width) ** 3 * (self.height) ** 3

    def calculate_initial_local_frame(self):
        """Calculate the initial local frame Ro with defined e_2o / e_3o.

        Parameters
        ----------
        None.

        Returns
        -------
        the initial local frame Ro.

        """
        e_1o = (self.initial_coordinate_node_2 -
                self.initial_coordinate_node_1) / self.initial_length
        if type(self.e_3o) == np.ndarray:
            e_2o = np.cross(self.e_3o, e_1o, axisa=0, axisb=0, axisc=0)
            return np.c_[e_1o, e_2o, self.e_3o]
        elif type(self.e_2o) == np.ndarray:
            e_3o = np.cross(e_1o, self.e_2o, axisa=0, axisb=0, axisc=0)
            return np.c_[e_1o, self.e_2o, e_3o]
        else:
            raise ValueError("Check the input of e2o and e3o!")

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

    def apply_hardening_model(self, hardening_model, gauss_number, yield_stress, kinematic_hardening_modulus, plastic_modulus, saturation_stress,
                              modified_modulus, exponent):
        """Apply hardening model on the beam element.

        Parameters
        ----------
        hardening_model: str
            hardening model. only linear hardening is implemented.
        gauss_number: tuple
            Number of Gauss points on xi-axis, eta-axis, mu-axis. Set as a tuple with 3 numbers, such as (1, 2, 2)
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
        self.num_of_gauss_locations_xi = gauss_number[0]   # x-axis
        self.num_of_gauss_locations_eta = gauss_number[1]  # y-axis
        self.num_of_gauss_locations_mu = gauss_number[2]   # z-axis

        self.num_of_gauss_locations = self.num_of_gauss_locations_xi * \
            self.num_of_gauss_locations_eta * self.num_of_gauss_locations_mu

        [gauss_locations_xi, weights_xi] = np.polynomial.legendre.leggauss(
            self.num_of_gauss_locations_xi)
        [gauss_locations_eta, weights_eta] = np.polynomial.legendre.leggauss(
            self.num_of_gauss_locations_eta)
        [gauss_locations_mu, weights_mu] = np.polynomial.legendre.leggauss(
            self.num_of_gauss_locations_mu)

        global_gauss_locations_mat = np.zeros(
            (3, self.num_of_gauss_locations), dtype=float)

        weights_mat = np.zeros(self.num_of_gauss_locations, dtype=float)

        for ixi in range(self.num_of_gauss_locations_xi):
            for jeta in range(self.num_of_gauss_locations_eta):
                for kmu in range(self.num_of_gauss_locations_mu):
                    x, y, z = self.map_local_coordinate_to_global_coordinate(
                        gauss_locations_xi[ixi], gauss_locations_eta[jeta], gauss_locations_mu[kmu])
                    global_gauss_locations_mat[0, ixi * self.num_of_gauss_locations_eta *
                                               self.num_of_gauss_locations_mu + jeta * self.num_of_gauss_locations_mu + kmu] = x
                    global_gauss_locations_mat[1, ixi * self.num_of_gauss_locations_eta *
                                               self.num_of_gauss_locations_mu + jeta * self.num_of_gauss_locations_mu + kmu] = y
                    global_gauss_locations_mat[2, ixi * self.num_of_gauss_locations_eta *
                                               self.num_of_gauss_locations_mu + jeta * self.num_of_gauss_locations_mu + kmu] = z
                    weights_mat[ixi * self.num_of_gauss_locations_eta *
                                self.num_of_gauss_locations_mu + jeta *
                                self.num_of_gauss_locations_mu + kmu] = weights_xi[ixi] * weights_eta[jeta] * weights_mu[kmu]

        self.global_gauss_locations_mat = global_gauss_locations_mat
        self.weights_mat = weights_mat

        if hardening_model == "linear hardening":

            self.analysis = hardening_model
            self.yield_stress = yield_stress
            self.plastic_modulus = plastic_modulus

            self.stress = np.zeros(
                (6, 1, self.num_of_gauss_locations), dtype=float)
            self.plastic_strain = np.zeros(
                (6, 1, self.num_of_gauss_locations), dtype=float)
            self.tangent_moduli = np.zeros(
                (6, 6, self.num_of_gauss_locations), dtype=float)
            self.internal_hardening_variable = np.zeros(
                self.num_of_gauss_locations, dtype=float)

            # initialize tangent moduli
            elastic_tangent_moduli = self.first_lame_constant * util.unit_tensor() @ util.unit_tensor().T + \
                2 * self.second_lame_constant * np.eye(6)
            for n in range(self.num_of_gauss_locations):
                self.tangent_moduli[:, :, n] = elastic_tangent_moduli

        else:
            print("In 3D, only linear hardening is implemented.")
            raise ValueError("Wrong hardening model!")

    def local_displacement(self):
        """Calculate local displacement vector at current state.

        Parameters
        ----------
        None.

        Returns
        -------
        p_l: numpy.ndarray
            local displacement.

        """
        p_l = np.zeros((7, 1), dtype=float)
        R_g = self.current_local_frame()

        R1_bar = (
            R_g.T) @ self.current_orientation_node_1 @ self.initial_local_frame
        R2_bar = (
            R_g.T) @ self.current_orientation_node_2 @ self.initial_local_frame

        theta1_bar_skew_symmetric = util.log(R1_bar)
        theta2_bar_skew_symmetric = util.log(R2_bar)

        p_l[0] = self.current_length() - self.initial_length
        p_l[1: 4] = util.get_rotational_vector(theta1_bar_skew_symmetric)
        p_l[4: 7] = util.get_rotational_vector(theta2_bar_skew_symmetric)

        return p_l

    def current_local_frame(self):
        """Calculate local frame Rr at current state.

        Parameters
        ----------
        None.

        Returns
        -------
        Rr: numpy.ndarray
            local frame at current state.

        """
        r_1 = (self.current_coordinate_node_2() -
               self.current_coordinate_node_1()) / self.current_length()

        _, _, q = self.auxiliary_vector()

        r_3 = np.cross(r_1, q, axisa=0, axisb=0, axisc=0)
        r_3 = r_3 / np.linalg.norm(r_3)

        r_2 = np.cross(r_3, r_1, axisa=0, axisb=0, axisc=0)

        return np.c_[r_1, r_2, r_3]

    def current_coordinate_node_1(self):
        """Calculate the coodinate of node 1 in current configuration.

        Parameters
        ----------
        None.

        Returns
        -------
        current coordinate of the 1st node: x1 = X1 + u1.

        """
        return self.initial_coordinate_node_1 + self.global_displacement[0: 3]

    def current_coordinate_node_2(self):
        """Calculate the coodinate of node 2 in current configuration.

        Parameters
        ----------
        None.

        Returns
        -------
        current coordinate of the 2nd node: x2 = X2 + u2.

        """
        return self.initial_coordinate_node_2 + self.global_displacement[6: 9]

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

    def auxiliary_vector(self):
        """Calculate the auxiliary vectors.

        Parameters
        ----------
        None.

        Returns
        -------
        q_1, q_2, q: numpy.ndarray

        """
        q_1 = self.current_orientation_node_1 @ self.initial_local_frame @ np.array([
                                                                                    [0., 1., 0.]]).T
        q_2 = self.current_orientation_node_2 @ self.initial_local_frame @ np.array([
                                                                                    [0., 1., 0.]]).T
        q = (q_1 + q_2) / 2

        return q_1, q_2, q

    def elastic_local_stiffness_force(self):
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
        u, t11, t21, t31, t12, t22, t32 = self.local_displacement().reshape(7)
        E, G, A, L, I22, I33, Irr = self.youngs_modulus, self.second_lame_constant, self.area, self.initial_length, self.moment_of_inertia_z, self.moment_of_inertia_y, self.fourth_order_polar_moment_of_inertia

        if self.beamtype == "Bernoulli":
            kl = np.array([[1.0*A*E/L, 1.0*E*(I22*t11 - I22*t12 + I33*t11 - I33*t12)/L**2, A*E*(0.133333333333333*t21 - 0.0333333333333333*t22), A*E*(0.133333333333333*t31 - 0.0333333333333333*t32), 1.0*E*(-I22*t11 + I22*t12 - I33*t11 + I33*t12)/L**2, A*E*(-0.0333333333333333*t21 + 0.133333333333333*t22), A*E*(-0.0333333333333333*t31 + 0.133333333333333*t32)], [1.0*E*(I22*t11 - I22*t12 + I33*t11 - I33*t12)/L**2, 0.0666666666666667*E*I22*t21**2/L - 0.0333333333333333*E*I22*t21*t22/L + 0.0666666666666667*E*I22*t22**2/L + 0.0666666666666667*E*I22*t31**2/L - 0.0333333333333333*E*I22*t31*t32/L + 0.0666666666666667*E*I22*t32**2/L + 1.0*E*I22*u/L**2 + 0.0666666666666667*E*I33*t21**2/L - 0.0333333333333333*E*I33*t21*t22/L + 0.0666666666666667*E*I33*t22**2/L + 0.0666666666666667*E*I33*t31**2/L - 0.0333333333333333*E*I33*t31*t32/L + 0.0666666666666667*E*I33*t32**2/L + 1.0*E*I33*u/L**2 + 1.5*E*Irr*t11**2/L**3 - 3.0*E*Irr*t11*t12/L**3 + 1.5*E*Irr*t12**2/L**3 + 1.0*G*I22/L + 1.0*G*I33/L, E*(0.133333333333333*I22*t11*t21 - 0.0333333333333333*I22*t11*t22 - 0.133333333333333*I22*t12*t21 + 0.0333333333333333*I22*t12*t22 + 0.133333333333333*I33*t11*t21 - 0.0333333333333333*I33*t11*t22 - 0.133333333333333*I33*t12*t21 + 0.0333333333333333*I33*t12*t22)/L, E*(0.133333333333333*I22*t11*t31 - 0.0333333333333333*I22*t11*t32 - 0.133333333333333*I22*t12*t31 + 0.0333333333333333*I22*t12*t32 + 0.133333333333333*I33*t11*t31 - 0.0333333333333333*I33*t11*t32 - 0.133333333333333*I33*t12*t31 + 0.0333333333333333*I33*t12*t32)/L, -0.0666666666666667*E*I22*t21**2/L + 0.0333333333333333*E*I22*t21*t22/L - 0.0666666666666667*E*I22*t22**2/L - 0.0666666666666667*E*I22*t31**2/L + 0.0333333333333333*E*I22*t31*t32/L - 0.0666666666666667*E*I22*t32**2/L - 1.0*E*I22*u/L**2 - 0.0666666666666667*E*I33*t21**2/L + 0.0333333333333333*E*I33*t21*t22/L - 0.0666666666666667*E*I33*t22**2/L - 0.0666666666666667*E*I33*t31**2/L + 0.0333333333333333*E*I33*t31*t32/L - 0.0666666666666667*E*I33*t32**2/L - 1.0*E*I33*u/L**2 - 1.5*E*Irr*t11**2/L**3 + 3.0*E*Irr*t11*t12/L**3 - 1.5*E*Irr*t12**2/L**3 - 1.0*G*I22/L - 1.0*G*I33/L, E*(-0.0333333333333333*I22*t11*t21 + 0.133333333333333*I22*t11*t22 + 0.0333333333333333*I22*t12*t21 - 0.133333333333333*I22*t12*t22 - 0.0333333333333333*I33*t11*t21 + 0.133333333333333*I33*t11*t22 + 0.0333333333333333*I33*t12*t21 - 0.133333333333333*I33*t12*t22)/L, E*(-0.0333333333333333*I22*t11*t31 + 0.133333333333333*I22*t11*t32 + 0.0333333333333333*I22*t12*t31 - 0.133333333333333*I22*t12*t32 - 0.0333333333333333*I33*t11*t31 + 0.133333333333333*I33*t11*t32 + 0.0333333333333333*I33*t12*t31 - 0.133333333333333*I33*t12*t32)/L], [A*E*(0.133333333333333*t21 - 0.0333333333333333*t22), E*(0.133333333333333*I22*t11*t21 - 0.0333333333333333*I22*t11*t22 - 0.133333333333333*I22*t12*t21 + 0.0333333333333333*I22*t12*t22 + 0.133333333333333*I33*t11*t21 - 0.0333333333333333*I33*t11*t22 - 0.133333333333333*I33*t12*t21 + 0.0333333333333333*I33*t12*t22)/L, E*(0.0266666666666667*A*L**2*t21**2 - 0.0133333333333333*A*L**2*t21*t22 + 0.01*A*L**2*t22**2 + 0.00888888888888889*A*L**2*t31**2 - 0.00444444444444444*A*L**2*t31*t32 + 0.00888888888888889*A*L**2*t32**2 + 0.133333333333333*A*L*u + 0.0666666666666667*I22*t11**2 - 0.133333333333333*I22*t11*t12 + 0.0666666666666667*I22*t12**2 + 0.0666666666666667*I33*t11**2 - 0.133333333333333*I33*t11*t12 + 0.0666666666666667*I33*t12**2 + 4.0*I33)/L, A*E*L*(0.0177777777777778*t21*t31 - 0.00444444444444444*t21*t32 - 0.00444444444444444*t22*t31 + 0.00111111111111111*t22*t32), E*(-0.133333333333333*I22*t11*t21 + 0.0333333333333333*I22*t11*t22 + 0.133333333333333*I22*t12*t21 - 0.0333333333333333*I22*t12*t22 - 0.133333333333333*I33*t11*t21 + 0.0333333333333333*I33*t11*t22 + 0.133333333333333*I33*t12*t21 - 0.0333333333333333*I33*t12*t22)/L, E*(-0.00666666666666667*A*L**2*t21**2 + 0.02*A*L**2*t21*t22 - 0.00666666666666667*A*L**2*t22**2 - 0.00222222222222222*A*L**2*t31**2 + 0.00111111111111111*A*L**2*t31*t32 - 0.00222222222222222*A*L**2*t32**2 - 0.0333333333333333*A*L*u - 0.0166666666666667*I22*t11**2 + 0.0333333333333333*I22*t11*t12 - 0.0166666666666667*I22*t12**2 - 0.0166666666666667*I33*t11**2 + 0.0333333333333333*I33*t11*t12 - 0.0166666666666667*I33*t12**2 + 2.0*I33)/L, A*E*L*(-0.00444444444444444*t21*t31 + 0.0177777777777778*t21*t32 + 0.00111111111111111*t22*t31 - 0.00444444444444444*t22*t32)], [A*E*(0.133333333333333*t31 - 0.0333333333333333*t32), E*(0.133333333333333*I22*t11*t31 - 0.0333333333333333*I22*t11*t32 - 0.133333333333333*I22*t12*t31 + 0.0333333333333333*I22*t12*t32 + 0.133333333333333*I33*t11*t31 - 0.0333333333333333*I33*t11*t32 - 0.133333333333333*I33*t12*t31 + 0.0333333333333333*I33*t12*t32)/L, A*E*L*(0.0177777777777778*t21*t31 - 0.00444444444444444*t21*t32 - 0.00444444444444444*t22*t31 + 0.00111111111111111*t22*t32), E*(0.00888888888888889*A*L**2*t21**2 - 0.00444444444444444*A*L**2*t21*t22 + 0.00888888888888889*A*L**2*t22**2 + 0.0266666666666667*A*L**2*t31**2 - 0.0133333333333333*A*L**2*t31*t32 + 0.01*A*L**2*t32**2 + 0.133333333333333*A*L*u + 0.0666666666666667*I22*t11**2 - 0.133333333333333*I22*t11*t12 + 0.0666666666666667*I22*t12**2 + 4.0*I22 + 0.0666666666666667*I33*t11**2 - 0.133333333333333*I33*t11*t12 + 0.0666666666666667*I33*t12**2)/L, E*(-0.133333333333333*I22*t11*t31 + 0.0333333333333333*I22*t11*t32 + 0.133333333333333*I22*t12*t31 - 0.0333333333333333*I22*t12*t32 - 0.133333333333333*I33*t11*t31 + 0.0333333333333333*I33*t11*t32 + 0.133333333333333*I33*t12*t31 - 0.0333333333333333*I33*t12*t32)/L, A*E*L*(-0.00444444444444444*t21*t31 + 0.00111111111111111*t21*t32 + 0.0177777777777778*t22*t31 - 0.00444444444444444*t22*t32), E*(-0.00222222222222222*A*L**2*t21**2 + 0.00111111111111111*A*L**2*t21*t22 - 0.00222222222222222*A*L**2*t22**2 - 0.00666666666666667*A*L**2*t31**2 + 0.02*A*L**2*t31*t32 - 0.00666666666666667*A*L**2*t32**2 - 0.0333333333333333*A*L*u - 0.0166666666666667*I22*t11**2 +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  0.0333333333333333*I22*t11*t12 - 0.0166666666666667*I22*t12**2 + 2.0*I22 - 0.0166666666666667*I33*t11**2 + 0.0333333333333333*I33*t11*t12 - 0.0166666666666667*I33*t12**2)/L], [1.0*E*(-I22*t11 + I22*t12 - I33*t11 + I33*t12)/L**2, -0.0666666666666667*E*I22*t21**2/L + 0.0333333333333333*E*I22*t21*t22/L - 0.0666666666666667*E*I22*t22**2/L - 0.0666666666666667*E*I22*t31**2/L + 0.0333333333333333*E*I22*t31*t32/L - 0.0666666666666667*E*I22*t32**2/L - 1.0*E*I22*u/L**2 - 0.0666666666666667*E*I33*t21**2/L + 0.0333333333333333*E*I33*t21*t22/L - 0.0666666666666667*E*I33*t22**2/L - 0.0666666666666667*E*I33*t31**2/L + 0.0333333333333333*E*I33*t31*t32/L - 0.0666666666666667*E*I33*t32**2/L - 1.0*E*I33*u/L**2 - 1.5*E*Irr*t11**2/L**3 + 3.0*E*Irr*t11*t12/L**3 - 1.5*E*Irr*t12**2/L**3 - 1.0*G*I22/L - 1.0*G*I33/L, E*(-0.133333333333333*I22*t11*t21 + 0.0333333333333333*I22*t11*t22 + 0.133333333333333*I22*t12*t21 - 0.0333333333333333*I22*t12*t22 - 0.133333333333333*I33*t11*t21 + 0.0333333333333333*I33*t11*t22 + 0.133333333333333*I33*t12*t21 - 0.0333333333333333*I33*t12*t22)/L, E*(-0.133333333333333*I22*t11*t31 + 0.0333333333333333*I22*t11*t32 + 0.133333333333333*I22*t12*t31 - 0.0333333333333333*I22*t12*t32 - 0.133333333333333*I33*t11*t31 + 0.0333333333333333*I33*t11*t32 + 0.133333333333333*I33*t12*t31 - 0.0333333333333333*I33*t12*t32)/L, 0.0666666666666667*E*I22*t21**2/L - 0.0333333333333333*E*I22*t21*t22/L + 0.0666666666666667*E*I22*t22**2/L + 0.0666666666666667*E*I22*t31**2/L - 0.0333333333333333*E*I22*t31*t32/L + 0.0666666666666667*E*I22*t32**2/L + 1.0*E*I22*u/L**2 + 0.0666666666666667*E*I33*t21**2/L - 0.0333333333333333*E*I33*t21*t22/L + 0.0666666666666667*E*I33*t22**2/L + 0.0666666666666667*E*I33*t31**2/L - 0.0333333333333333*E*I33*t31*t32/L + 0.0666666666666667*E*I33*t32**2/L + 1.0*E*I33*u/L**2 + 1.5*E*Irr*t11**2/L**3 - 3.0*E*Irr*t11*t12/L**3 + 1.5*E*Irr*t12**2/L**3 + 1.0*G*I22/L + 1.0*G*I33/L, E*(0.0333333333333333*I22*t11*t21 - 0.133333333333333*I22*t11*t22 - 0.0333333333333333*I22*t12*t21 + 0.133333333333333*I22*t12*t22 + 0.0333333333333333*I33*t11*t21 - 0.133333333333333*I33*t11*t22 - 0.0333333333333333*I33*t12*t21 + 0.133333333333333*I33*t12*t22)/L, E*(0.0333333333333333*I22*t11*t31 - 0.133333333333333*I22*t11*t32 - 0.0333333333333333*I22*t12*t31 + 0.133333333333333*I22*t12*t32 + 0.0333333333333333*I33*t11*t31 - 0.133333333333333*I33*t11*t32 - 0.0333333333333333*I33*t12*t31 + 0.133333333333333*I33*t12*t32)/L], [A*E*(-0.0333333333333333*t21 + 0.133333333333333*t22), E*(-0.0333333333333333*I22*t11*t21 + 0.133333333333333*I22*t11*t22 + 0.0333333333333333*I22*t12*t21 - 0.133333333333333*I22*t12*t22 - 0.0333333333333333*I33*t11*t21 + 0.133333333333333*I33*t11*t22 + 0.0333333333333333*I33*t12*t21 - 0.133333333333333*I33*t12*t22)/L, E*(-0.00666666666666667*A*L**2*t21**2 + 0.02*A*L**2*t21*t22 - 0.00666666666666667*A*L**2*t22**2 - 0.00222222222222222*A*L**2*t31**2 + 0.00111111111111111*A*L**2*t31*t32 - 0.00222222222222222*A*L**2*t32**2 - 0.0333333333333333*A*L*u - 0.0166666666666667*I22*t11**2 + 0.0333333333333333*I22*t11*t12 - 0.0166666666666667*I22*t12**2 - 0.0166666666666667*I33*t11**2 + 0.0333333333333333*I33*t11*t12 - 0.0166666666666667*I33*t12**2 + 2.0*I33)/L, A*E*L*(-0.00444444444444444*t21*t31 + 0.00111111111111111*t21*t32 + 0.0177777777777778*t22*t31 - 0.00444444444444444*t22*t32), E*(0.0333333333333333*I22*t11*t21 - 0.133333333333333*I22*t11*t22 - 0.0333333333333333*I22*t12*t21 + 0.133333333333333*I22*t12*t22 + 0.0333333333333333*I33*t11*t21 - 0.133333333333333*I33*t11*t22 - 0.0333333333333333*I33*t12*t21 + 0.133333333333333*I33*t12*t22)/L, E*(0.01*A*L**2*t21**2 - 0.0133333333333333*A*L**2*t21*t22 + 0.0266666666666667*A*L**2*t22**2 + 0.00888888888888889*A*L**2*t31**2 - 0.00444444444444444*A*L**2*t31*t32 + 0.00888888888888889*A*L**2*t32**2 + 0.133333333333333*A*L*u + 0.0666666666666667*I22*t11**2 - 0.133333333333333*I22*t11*t12 + 0.0666666666666667*I22*t12**2 + 0.0666666666666667*I33*t11**2 - 0.133333333333333*I33*t11*t12 + 0.0666666666666667*I33*t12**2 + 4.0*I33)/L, A*E*L*(0.00111111111111111*t21*t31 - 0.00444444444444444*t21*t32 - 0.00444444444444444*t22*t31 + 0.0177777777777778*t22*t32)], [A*E*(-0.0333333333333333*t31 + 0.133333333333333*t32), E*(-0.0333333333333333*I22*t11*t31 + 0.133333333333333*I22*t11*t32 + 0.0333333333333333*I22*t12*t31 - 0.133333333333333*I22*t12*t32 - 0.0333333333333333*I33*t11*t31 + 0.133333333333333*I33*t11*t32 + 0.0333333333333333*I33*t12*t31 - 0.133333333333333*I33*t12*t32)/L, A*E*L*(-0.00444444444444444*t21*t31 + 0.0177777777777778*t21*t32 + 0.00111111111111111*t22*t31 - 0.00444444444444444*t22*t32), E*(-0.00222222222222222*A*L**2*t21**2 + 0.00111111111111111*A*L**2*t21*t22 - 0.00222222222222222*A*L**2*t22**2 - 0.00666666666666667*A*L**2*t31**2 + 0.02*A*L**2*t31*t32 - 0.00666666666666667*A*L**2*t32**2 - 0.0333333333333333*A*L*u - 0.0166666666666667*I22*t11**2 + 0.0333333333333333*I22*t11*t12 - 0.0166666666666667*I22*t12**2 + 2.0*I22 - 0.0166666666666667*I33*t11**2 + 0.0333333333333333*I33*t11*t12 - 0.0166666666666667*I33*t12**2)/L, E*(0.0333333333333333*I22*t11*t31 - 0.133333333333333*I22*t11*t32 - 0.0333333333333333*I22*t12*t31 + 0.133333333333333*I22*t12*t32 + 0.0333333333333333*I33*t11*t31 - 0.133333333333333*I33*t11*t32 - 0.0333333333333333*I33*t12*t31 + 0.133333333333333*I33*t12*t32)/L, A*E*L*(0.00111111111111111*t21*t31 - 0.00444444444444444*t21*t32 - 0.00444444444444444*t22*t31 + 0.0177777777777778*t22*t32), E*(0.00888888888888889*A*L**2*t21**2 - 0.00444444444444444*A*L**2*t21*t22 + 0.00888888888888889*A*L**2*t22**2 + 0.01*A*L**2*t31**2 - 0.0133333333333333*A*L**2*t31*t32 + 0.0266666666666667*A*L**2*t32**2 + 0.133333333333333*A*L*u + 0.0666666666666667*I22*t11**2 - 0.133333333333333*I22*t11*t12 + 0.0666666666666667*I22*t12**2 + 4.0*I22 + 0.0666666666666667*I33*t11**2 - 0.133333333333333*I33*t11*t12 + 0.0666666666666667*I33*t12**2)/L]])
            fl = np.array([[E*(0.0666666666666667*A*L**2*t21**2 - 0.0333333333333333*A*L**2*t21*t22 + 0.0666666666666667*A*L**2*t22**2 + 0.0666666666666667*A*L**2*t31**2 - 0.0333333333333333*A*L**2*t31*t32 + 0.0666666666666667*A*L**2*t32**2 + 1.0*A*L*u + 0.5*I22*t11**2 - 1.0*I22*t11*t12 + 0.5*I22*t12**2 + 0.5*I33*t11**2 - 1.0*I33*t11*t12 + 0.5*I33*t12**2)/L**2, 0.0666666666666667*E*I22*t11*t21**2/L - 0.0333333333333333*E*I22*t11*t21*t22/L + 0.0666666666666667*E*I22*t11*t22**2/L + 0.0666666666666667*E*I22*t11*t31**2/L - 0.0333333333333333*E*I22*t11*t31*t32/L + 0.0666666666666667*E*I22*t11*t32**2/L - 0.0666666666666667*E*I22*t12*t21**2/L + 0.0333333333333333*E*I22*t12*t21*t22/L - 0.0666666666666667*E*I22*t12*t22**2/L - 0.0666666666666667*E*I22*t12*t31**2/L + 0.0333333333333333*E*I22*t12*t31*t32/L - 0.0666666666666667*E*I22*t12*t32**2/L + 1.0*E*I22*t11*u/L**2 - 1.0*E*I22*t12*u/L**2 + 0.0666666666666667*E*I33*t11*t21**2/L - 0.0333333333333333*E*I33*t11*t21*t22/L + 0.0666666666666667*E*I33*t11*t22**2/L + 0.0666666666666667*E*I33*t11*t31**2/L - 0.0333333333333333*E*I33*t11*t31*t32/L + 0.0666666666666667*E*I33*t11*t32**2/L - 0.0666666666666667*E*I33*t12*t21**2/L + 0.0333333333333333*E*I33*t12*t21*t22/L - 0.0666666666666667*E*I33*t12*t22**2/L - 0.0666666666666667*E*I33*t12*t31**2/L + 0.0333333333333333*E*I33*t12*t31*t32/L - 0.0666666666666667*E*I33*t12*t32**2/L + 1.0*E*I33*t11*u/L**2 - 1.0*E*I33*t12*u/L**2 + 0.5*E*Irr*t11**3/L**3 - 1.5*E*Irr*t11**2*t12/L**3 + 1.5*E*Irr*t11*t12**2/L**3 - 0.5*E*Irr*t12**3/L**3 + 1.0*G*I22*t11/L - 1.0*G*I22*t12/L + 1.0*G*I33*t11/L - 1.0*G*I33*t12/L, 1.0*E*(0.00888888888888889*A*L**2*t21**3 - 0.00666666666666667*A*L**2*t21**2*t22 + 0.01*A*L**2*t21*t22**2 + 0.00888888888888889*A*L**2*t21*t31**2 - 0.00444444444444444*A*L**2*t21*t31*t32 + 0.00888888888888889*A*L**2*t21*t32**2 - 0.00222222222222222*A*L**2*t22**3 - 0.00222222222222222*A*L**2*t22*t31**2 + 0.00111111111111111*A*L**2*t22*t31*t32 - 0.00222222222222222*A*L**2*t22*t32**2 + 0.133333333333333*A*L*t21*u - 0.0333333333333333*A*L*t22*u + 0.0666666666666667*I22*t11**2*t21 - 0.0166666666666667*I22*t11**2*t22 - 0.133333333333333*I22*t11*t12*t21 + 0.0333333333333333*I22*t11*t12*t22 + 0.0666666666666667*I22*t12**2*t21 - 0.0166666666666667*I22*t12**2*t22 + 0.0666666666666667*I33*t11**2*t21 - 0.0166666666666667*I33*t11**2*t22 - 0.133333333333333*I33*t11*t12*t21 + 0.0333333333333333*I33*t11*t12*t22 + 0.0666666666666667*I33*t12**2*t21 - 0.0166666666666667*I33*t12**2*t22 + 4.0*I33*t21 + 2.0*I33*t22)/L, 1.0*E*(0.00888888888888889*A*L**2*t21**2*t31 - 0.00222222222222222*A*L**2*t21**2*t32 - 0.00444444444444444*A*L**2*t21*t22*t31 + 0.00111111111111111*A*L**2*t21*t22*t32 + 0.00888888888888889*A*L**2*t22**2*t31 - 0.00222222222222222*A*L**2*t22**2*t32 + 0.00888888888888889*A*L**2*t31**3 - 0.00666666666666667*A*L**2*t31**2*t32 + 0.01*A*L**2*t31*t32**2 - 0.00222222222222222*A*L**2*t32**3 + 0.133333333333333*A*L*t31*u - 0.0333333333333333*A*L*t32*u + 0.0666666666666667*I22*t11**2*t31 - 0.0166666666666667*I22*t11**2*t32 - 0.133333333333333*I22*t11*t12*t31 + 0.0333333333333333*I22*t11*t12*t32 + 0.0666666666666667*I22*t12**2*t31 - 0.0166666666666667*I22*t12**2*t32 + 4.0*I22*t31 + 2.0*I22*t32 + 0.0666666666666667*I33*t11**2*t31 -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     0.0166666666666667*I33*t11**2*t32 - 0.133333333333333*I33*t11*t12*t31 + 0.0333333333333333*I33*t11*t12*t32 + 0.0666666666666667*I33*t12**2*t31 - 0.0166666666666667*I33*t12**2*t32)/L, -0.0666666666666667*E*I22*t11*t21**2/L + 0.0333333333333333*E*I22*t11*t21*t22/L - 0.0666666666666667*E*I22*t11*t22**2/L - 0.0666666666666667*E*I22*t11*t31**2/L + 0.0333333333333333*E*I22*t11*t31*t32/L - 0.0666666666666667*E*I22*t11*t32**2/L + 0.0666666666666667*E*I22*t12*t21**2/L - 0.0333333333333333*E*I22*t12*t21*t22/L + 0.0666666666666667*E*I22*t12*t22**2/L + 0.0666666666666667*E*I22*t12*t31**2/L - 0.0333333333333333*E*I22*t12*t31*t32/L + 0.0666666666666667*E*I22*t12*t32**2/L - 1.0*E*I22*t11*u/L**2 + 1.0*E*I22*t12*u/L**2 - 0.0666666666666667*E*I33*t11*t21**2/L + 0.0333333333333333*E*I33*t11*t21*t22/L - 0.0666666666666667*E*I33*t11*t22**2/L - 0.0666666666666667*E*I33*t11*t31**2/L + 0.0333333333333333*E*I33*t11*t31*t32/L - 0.0666666666666667*E*I33*t11*t32**2/L + 0.0666666666666667*E*I33*t12*t21**2/L - 0.0333333333333333*E*I33*t12*t21*t22/L + 0.0666666666666667*E*I33*t12*t22**2/L + 0.0666666666666667*E*I33*t12*t31**2/L - 0.0333333333333333*E*I33*t12*t31*t32/L + 0.0666666666666667*E*I33*t12*t32**2/L - 1.0*E*I33*t11*u/L**2 + 1.0*E*I33*t12*u/L**2 - 0.5*E*Irr*t11**3/L**3 + 1.5*E*Irr*t11**2*t12/L**3 - 1.5*E*Irr*t11*t12**2/L**3 + 0.5*E*Irr*t12**3/L**3 - 1.0*G*I22*t11/L + 1.0*G*I22*t12/L - 1.0*G*I33*t11/L + 1.0*G*I33*t12/L, E*(-0.00222222222222222*A*L**2*t21**3 + 0.01*A*L**2*t21**2*t22 - 0.00666666666666667*A*L**2*t21*t22**2 - 0.00222222222222222*A*L**2*t21*t31**2 + 0.00111111111111111*A*L**2*t21*t31*t32 - 0.00222222222222222*A*L**2*t21*t32**2 + 0.00888888888888889*A*L**2*t22**3 + 0.00888888888888889*A*L**2*t22*t31**2 - 0.00444444444444444*A*L**2*t22*t31*t32 + 0.00888888888888889*A*L**2*t22*t32**2 - 0.0333333333333333*A*L*t21*u + 0.133333333333333*A*L*t22*u - 0.0166666666666667*I22*t11**2*t21 + 0.0666666666666667*I22*t11**2*t22 + 0.0333333333333333*I22*t11*t12*t21 - 0.133333333333333*I22*t11*t12*t22 - 0.0166666666666667*I22*t12**2*t21 + 0.0666666666666667*I22*t12**2*t22 - 0.0166666666666667*I33*t11**2*t21 + 0.0666666666666667*I33*t11**2*t22 + 0.0333333333333333*I33*t11*t12*t21 - 0.133333333333333*I33*t11*t12*t22 - 0.0166666666666667*I33*t12**2*t21 + 0.0666666666666667*I33*t12**2*t22 + 2.0*I33*t21 + 4.0*I33*t22)/L, E*(-0.00222222222222222*A*L**2*t21**2*t31 + 0.00888888888888889*A*L**2*t21**2*t32 + 0.00111111111111111*A*L**2*t21*t22*t31 - 0.00444444444444444*A*L**2*t21*t22*t32 - 0.00222222222222222*A*L**2*t22**2*t31 + 0.00888888888888889*A*L**2*t22**2*t32 - 0.00222222222222222*A*L**2*t31**3 + 0.01*A*L**2*t31**2*t32 - 0.00666666666666667*A*L**2*t31*t32**2 + 0.00888888888888889*A*L**2*t32**3 - 0.0333333333333333*A*L*t31*u + 0.133333333333333*A*L*t32*u - 0.0166666666666667*I22*t11**2*t31 + 0.0666666666666667*I22*t11**2*t32 + 0.0333333333333333*I22*t11*t12*t31 - 0.133333333333333*I22*t11*t12*t32 - 0.0166666666666667*I22*t12**2*t31 + 0.0666666666666667*I22*t12**2*t32 + 2.0*I22*t31 + 4.0*I22*t32 - 0.0166666666666667*I33*t11**2*t31 + 0.0666666666666667*I33*t11**2*t32 + 0.0333333333333333*I33*t11*t12*t31 - 0.133333333333333*I33*t11*t12*t32 - 0.0166666666666667*I33*t12**2*t31 + 0.0666666666666667*I33*t12**2*t32)/L]]).T

        elif self.beamtype == "Timoshenko":
            kl = np.array([[1.0*A*E/L, 0.75*E*Irr*(I22 + I33)*(t11 - t12)**5/L**6, 0, 0, -0.75*E*Irr*(I22 + I33)*(t11 - t12)**5/L**6, 0, 0], [0.75*E*Irr*(I22 + I33)*(t11 - t12)**5/L**6, (I22 + I33)*(3.75*E*Irr*u*(t11 - t12)**4 + 1.0*G*L**5)/L**6, 0, 0, -(I22 + I33)*(3.75*E*Irr*u*(t11 - t12)**4 + 1.0*G*L**5)/L**6, 0, 0], [0, 0, 0.25*A*G*L + 1.0*E*I33/L, 0, 0, 0.25*A*G*L - 1.0*E*I33/L, 0], [
                          0, 0, 0, 0.25*A*G*L + 1.0*E*I22/L, 0, 0, 0.25*A*G*L - 1.0*E*I22/L], [-0.75*E*Irr*(I22 + I33)*(t11 - t12)**5/L**6, -(I22 + I33)*(3.75*E*Irr*u*(t11 - t12)**4 + 1.0*G*L**5)/L**6, 0, 0, (I22 + I33)*(3.75*E*Irr*u*(t11 - t12)**4 + 1.0*G*L**5)/L**6, 0, 0], [0, 0, 0.25*A*G*L - 1.0*E*I33/L, 0, 0, 0.25*A*G*L + 1.0*E*I33/L, 0], [0, 0, 0, 0.25*A*G*L - 1.0*E*I22/L, 0, 0, 0.25*A*G*L + 1.0*E*I22/L]])
            fl = np.array([[0.125*E*(8*A*L**5*u + Irr*(I22 + I33)*(t11 - t12)**6)/L**6, (I22 + I33)*(t11 - t12)*(0.75*E*Irr*u*(t11 - t12)**4 + G*L**5)/L**6, (0.25*A*G*L**2*(t21 + t22) + E*I33*(t21 - t22))/L, (0.25*A*G*L**2*(
                t31 + t32) + E*I22*(t31 - t32))/L, -(I22 + I33)*(t11 - t12)*(0.75*E*Irr*u*(t11 - t12)**4 + 1.0*G*L**5)/L**6, (0.25*A*G*L**2*(t21 + t22) - 1.0*E*I33*(t21 - t22))/L, (0.25*A*G*L**2*(t31 + t32) - 1.0*E*I22*(t31 - t32))/L]]).T

        return kl, fl

    def strain(self, x, y, z):
        """Calculate the strain at a point expressed in local coordinate.

        Parameters
        ----------
        x: float
            the 1st local coordinate.
        y: float
            the 2nd local coordinate.
        z: float
            the 3rd local coordinate.

        Returns
        -------
        strain: numpy.ndarray
            the strain vector with six components.

        """
        u, t11, t21, t31, t12, t22, t32 = self.local_displacement().reshape(7)
        L = self.initial_length

        # Shape functions and their derivatives
        # def f1(x): return 1 - 3 * (x/L) ** 2 + 2 * (x/L) ** 3
        # def f2(x): return x * (1 - x/L) ** 2
        # def f3(x): return 1 - f1(x)
        # def f4(x): return (x ** 2) * (x/L - 1) / L
        def f5(x): return 1 - x/L
        def f6(x): return x/L

        # def df1(x): return -6*x/L**2 + 6*x**2/L**3
        def df2(x): return (1 - x/L)**2 - 2*x*(1 - x/L)/L
        # def df3(x): return 6*x/L**2 - 6*x**2/L**3
        def df4(x): return 2*x*(-1 + x/L)/L + x**2/L**2
        def df5(x): return -1/L
        def df6(x): return 1/L

        # def ddf1(x): return -6/L**2 + 12*x/L**3
        def ddf2(x): return -4*(1 - x/L)/L + 2*x/L**2
        # def ddf3(x): return 6/L**2 - 12*x/L**3
        def ddf4(x): return 2*(-1 + x/L)/L + 4*x/L**2
        # def ddf5(x): return 0
        # def ddf6(x): return 0

        if self.beamtype == "Bernoulli":
            # u1 = f6(x) * u
            # u2 = f2(x) * t31 + f4(x) * t32
            # u3 = -f2(x) * t21 - f4(x) * t22

            du1 = df6(x) * u
            du2 = df2(x) * t31 + df4(x) * t32
            du3 = -df2(x) * t21 - df4(x) * t22

            # ddu1 = ddf6(x)
            ddu2 = ddf2(x) * t31 + ddf4(x) * t32
            ddu3 = -ddf2(x) * t21 - ddf4(x) * t22

            # t1 = f5(x) * t11 + f6(x) * t12
            t2 = -du3
            t3 = du2

            dt1 = df5(x) * t11 + df6(x) * t12
            dt2 = -ddu3
            dt3 = ddu2

        elif self.beamtype == "Timoshenko":
            du1 = u / L
            du2 = 0.0
            du3 = 0.0

            ddu2 = 0.0
            ddu3 = 0.0

            t1 = f5(x) * t11 + f6(x) * t12
            t2 = f5(x) * t21 + f6(x) * t22
            t3 = f5(x) * t31 + f6(x) * t32

            dt1 = (t12 - t11) / L
            dt2 = (t22 - t21) / L
            dt3 = (t32 - t31) / L

        eps_11 = du1 + (du2 ** 2 + du3 ** 2) / 2 - y * dt3 + \
            z * dt2 + (y ** 2 + z ** 2) * dt1 ** 2 / 2
        eps_22 = -self.first_lame_constant * eps_11 / \
            (2 * (self.first_lame_constant + self.second_lame_constant))
        eps_33 = eps_22
        gamma_12 = du2 - t3 - z * dt1
        gamma_13 = du3 + t2 + y * dt1

        return np.array([[eps_11, eps_22, eps_33, gamma_12, 0., gamma_13]]).T

    def map_local_coordinate_to_global_coordinate(self, xi, eta, mu):
        """Define the linear map between Gaussian space and local space.

        Parameters
        ----------
        xi: float
            the 1st Gaussian coordinate.
        eta: float
            the 2nd Gaussian coordinate.
        mu: float
            the 3rd Gaussian coordinate.

        Returns
        -------
        x: float
            the 1st local coordinate.
        y: float
            the 2nd local coordinate.
        z: float
            the 3rd local coordinate.

        """
        x = self.initial_length / 2 * (xi + 1)
        y = self.width / 2 * eta
        z = self.height / 2 * mu

        return x, y, z

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

        elastic_tangent_moduli = self.first_lame_constant * util.unit_tensor() @ util.unit_tensor().T + \
            2 * self.second_lame_constant * np.eye(6)

        for n in range(self.num_of_gauss_locations):
            x, y, z = self.global_gauss_locations_mat[0,
                                                      n], self.global_gauss_locations_mat[1, n], self.global_gauss_locations_mat[2, n]

            strain_trial = self.strain(x, y, z) - self.plastic_strain[:, :, n]

            volumetric_strain_trial, deviatoric_strain_trial = util.decompose_strain(
                strain_trial)

            p_trial = self.bulk_modulus * volumetric_strain_trial
            s_trial = 2 * self.second_lame_constant * deviatoric_strain_trial
            q_trial = np.sqrt(3/2) * util.stress_norm(s_trial)

            f_trial = q_trial - \
                yield_stress_function(self.internal_hardening_variable[n])

            if f_trial <= 0:
                self.stress[:, :, n] = s_trial + p_trial * util.unit_tensor()
                self.tangent_moduli[:, :, n] = elastic_tangent_moduli
            else:
                deltagamma = f_trial / \
                    (self.plastic_modulus + 3 * self.second_lame_constant)

                s = (1 - deltagamma * 3 *
                     self.second_lame_constant / q_trial) * s_trial
                self.stress[:, :, n] = s + p_trial * util.unit_tensor()
                self.plastic_strain[:, :, n] += deltagamma * \
                    np.sqrt(3/2) * s / util.stress_norm(s)
                self.internal_hardening_variable[n] += deltagamma

                # flow_vec = np.sqrt(3/2) * s / util.stress_norm(s)
                # self.tangent_moduli[:, :, n] = elastic_tangent_moduli - float(((elastic_tangent_moduli @ flow_vec).T @ (elastic_tangent_moduli @ flow_vec)) / (flow_vec.T @ elastic_tangent_moduli @ flow_vec - self.plastic_modulus))

                unit_flow_vec = s_trial / util.stress_norm(s_trial)
                self.tangent_moduli[:, :, n] = 2 * self.second_lame_constant * (1 - deltagamma * 3 * self.second_lame_constant / q_trial) * util.deviatoric_projection_tensor() + 6 * self.second_lame_constant ** 2 * (
                    deltagamma / q_trial - 1 / (3 * self.second_lame_constant + self.plastic_modulus)) * unit_flow_vec @ unit_flow_vec.T + self.bulk_modulus * util.unit_tensor() @ util.unit_tensor().T

    # def L_matrix(self):
    #     L = np.zeros((6, 6, self.num_of_gauss_locations), dtype=float)
    #     eta = -self.first_lame_constant / \
    #         (2 * (self.first_lame_constant +
    #          self.second_lame_constant))
    #     for n in range(self.num_of_gauss_locations):
    #         y, z = self.global_gauss_locations_mat[1: 3, n]
    #         L[3, 3, n] = (y ** 2 + z ** 2) * self.stress[0, 0, n] + eta * (y ** 2 + z ** 2) * \
    #             self.stress[1, 0, n] + eta * \
    #             (y ** 2 + z ** 2) * self.stress[2, 0, n]

    #     return L

    def A_matrix(self, y, z):
        p_l = self.local_displacement()
        t12, t11 = p_l[4, 0], p_l[1, 0]
        L = self.initial_length

        # def df1(x): return -6*x/L**2 + 6*x**2/L**3
        # def df3(x): return 6*x/L**2 - 6*x**2/L**3

        # def dt1(x): return -1/L * t11 + 1/L * t12
        A = np.array([[1., 0., 0., (y ** 2 + z ** 2) * (t12 - t11)/L, z, -y],
                      [0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0.],
                      [0., -1., 0., -z, 0., 0.],
                      [0., 0., 0., 0., 0., 0.],
                      [0., 0., 1., y, 0., 0.]])
        A[1, :] = -self.first_lame_constant / \
            (2 * (self.first_lame_constant +
             self.second_lame_constant)) * A[0, :]
        A[2, :] = A[1, :]
        return A

    def G_mat(self, x):

        L = self.initial_length

        # def f1(x): return 1 - 3 * (x/L) ** 2 + 2 * (x/L) ** 3
        # def f2(x): return x * (1 - x/L) ** 2
        # def f3(x): return 1 - f1(x)
        # def f4(x): return (x ** 2) * (x/L - 1) / L
        def f5(x): return 1 - x/L
        def f6(x): return x/L

        # def df1(x): return -6*x/L**2 + 6*x**2/L**3
        def df2(x): return (1 - x/L)**2 - 2*x*(1 - x/L)/L
        # def df3(x): return 6*x/L**2 - 6*x**2/L**3
        def df4(x): return 2*x*(-1 + x/L)/L + x**2/L**2
        def df5(x): return -1/L
        def df6(x): return 1/L

        # def ddf1(x): return -6/L**2 + 12*x/L**3
        def ddf2(x): return -4*(1 - x/L)/L + 2*x/L**2
        # def ddf3(x): return 6/L**2 - 12*x/L**3
        def ddf4(x): return 2*(-1 + x/L)/L + 4*x/L**2
        # def ddf5(x): return 0
        # def ddf6(x): return 0

        if self.beamtype == "Bernoulli":
            G = np.zeros((6, 7), dtype=float)
            G[0, 0] = 1/L
            G[1, 3], G[2, 2] = df2(x), df2(x)
            G[1, 6], G[2, 5] = df4(x), df4(x)
            G[3, 1], G[4, 2], G[5, 3] = df5(x), ddf2(x), ddf2(x)
            G[3, 4], G[4, 5], G[5, 6] = df6(x), ddf4(x), ddf4(x)
        elif self.beamtype == "Timoshenko":
            G = np.zeros((6, 7), dtype=float)
            G[0, 0] = 1/L
            G[1, 3], G[2, 2] = f5(x), f5(x)
            G[1, 6], G[2, 5] = f6(x), f6(x)
            G[3, 1], G[4, 2], G[5, 3] = -1/L, -1/L, -1/L
            G[3, 4], G[4, 5], G[5, 6] = 1/L, 1/L, 1/L
        return G

    def elasto_plastic_local_stiffness_force(self):
        k_l = np.zeros((7, 7), dtype=float)
        f_l = np.zeros((7, 1), dtype=float)
        # L = self.L_matrix()

        L = np.zeros((6, 6, self.num_of_gauss_locations), dtype=float)
        eta = -self.first_lame_constant / \
            (2 * (self.first_lame_constant +
             self.second_lame_constant))

        for n in range(self.num_of_gauss_locations):
            x, y, z = self.global_gauss_locations_mat[0: 3, n]
            G = self.G_mat(x)
            A = self.A_matrix(y, z)

            L[3, 3, n] = (y ** 2 + z ** 2) * self.stress[0, 0, n] + eta * (y ** 2 + z ** 2) * \
                self.stress[1, 0, n] + eta * \
                (y ** 2 + z ** 2) * self.stress[2, 0, n]

            fac = self.initial_length * self.width * self.height / 8
            k_l += G.T @ (A.T @ self.tangent_moduli[:, :, n] @ A + L[
                          :, :, n]) @ G * fac * self.weights_mat[n]
            f_l += G.T @ A.T @ self.stress[:, :, n] * fac * self.weights_mat[n]

        return k_l, f_l

    def Ba_matrix(self):
        p_l = self.local_displacement()
        T_s_inv_theta_1bar = util.get_transformation_inv(p_l[1: 4])
        T_s_inv_theta_2bar = util.get_transformation_inv(p_l[4: 7])

        B_a = np.zeros((7, 7), dtype=float)
        B_a[0, 0] = 1.
        B_a[1: 4, 1: 4] = T_s_inv_theta_1bar
        B_a[4: 7, 4: 7] = T_s_inv_theta_2bar

        return B_a

    def a_stiffness_force(self):
        B_a = self.Ba_matrix()

        if self.analysis == "elastic":
            k_l, f_l = self.elastic_local_stiffness_force()

        else:
            k_l, f_l = self.elasto_plastic_local_stiffness_force()

        p_l = self.local_displacement()
        theta_bar_1 = p_l[1: 4]
        theta_bar_2 = p_l[4: 7]

        v_1 = f_l[1: 4]
        v_2 = f_l[4: 7]

        alpha_1, theta_bar_tilde_1 = util.decompose_rotational_vector(
            theta_bar_1)
        eta_1, mu_1 = util.get_eta_and_mu(alpha_1)
        v_tilde_1 = util.get_skew_symmetric(v_1)
        T_s_inv_1 = util.get_transformation_inv(theta_bar_1)

        K_h1 = (eta_1 * (theta_bar_1 @ v_1.T - 2 * v_1 @ theta_bar_1.T + float(theta_bar_1.T @ v_1) * np.eye(3))
                + mu_1 * theta_bar_tilde_1 @ theta_bar_tilde_1 @ (v_1 @ theta_bar_1.T) - 1/2 * v_tilde_1) @ T_s_inv_1

        alpha_2, theta_bar_tilde_2 = util.decompose_rotational_vector(
            theta_bar_2)
        eta_2, mu_2 = util.get_eta_and_mu(alpha_2)
        v_tilde_2 = util.get_skew_symmetric(v_2)
        T_s_inv_2 = util.get_transformation_inv(theta_bar_2)

        K_h2 = (eta_2 * (theta_bar_2 @ v_2.T - 2 * v_2 @ theta_bar_2.T + float(theta_bar_2.T @ v_2) * np.eye(3))
                + mu_2 * theta_bar_tilde_2 @ theta_bar_tilde_2 @ (v_2 @ theta_bar_2.T) - 1/2 * v_tilde_2) @ T_s_inv_2

        K_h = np.zeros((7, 7), dtype=float)
        K_h[1: 4, 1: 4] = K_h1
        K_h[4: 7, 4: 7] = K_h2

        K_a = B_a.T @ k_l @ B_a + K_h
        f_a = B_a.T @ f_l

        return K_a, f_a

    def G_matrix(self):

        q_1, q_2, q = self.auxiliary_vector()
        R_g = self.current_local_frame()

        q1, q2 = (R_g.T @ q)[0: 2, 0]
        q11, q12 = (R_g.T @ q_1)[0: 2, 0]
        q21, q22 = (R_g.T @ q_2)[0: 2, 0]

        eta = q1 / q2
        eta_11 = q11 / q2
        eta_12 = q12 / q2
        eta_21 = q21 / q2
        eta_22 = q22 / q2

        l = self.current_length()

        G = np.array([
            [0, 0, eta/l, eta_12/2, -eta_11/2, 0, 0,
                0, -eta/l, eta_22/2, -eta_21/2, 0],
            [0, 0, 1/l, 0, 0, 0,
             0, 0, -1/l, 0, 0, 0],
            [0, -1/l, 0, 0, 0, 0,
             0, 1/l, 0, 0, 0, 0]], dtype=float).T

        return G

    def r_vector(self):

        r_1 = (self.current_coordinate_node_2() -
               self.current_coordinate_node_1()) / self.current_length()
        r = np.zeros((1, 12), dtype=float)
        r[:, 0: 3], r[:, 6: 9] = -r_1.T, r_1.T
        return r

    def P_matrix(self):

        G = self.G_matrix()

        temp1, temp2 = np.zeros((6, 12), dtype=float), np.zeros(
            (6, 12), dtype=float)
        temp1[0: 3, 3: 6], temp1[3: 6, 9: 12] = np.eye(3), np.eye(3)

        temp2[0: 3, :], temp2[3: 6, :] = G.T, G.T

        return temp1 - temp2

    def E_matrix(self):
        E = np.zeros((12, 12), dtype=float)
        R_g = self.current_local_frame()

        E[0: 3, 0: 3] = R_g
        E[3: 6, 3: 6] = R_g
        E[6: 9, 6: 9] = R_g
        E[9: 12, 9: 12] = R_g

        return E

    def Bg_matrix(self):
        return np.r_[self.r_vector(), self.P_matrix() @ self.E_matrix().T]

    def global_stiffness_force(self):
        B_g = self.Bg_matrix()
        K_a, f_a = self.a_stiffness_force()

        l = self.current_length()
        r_1 = (self.current_coordinate_node_2() -
               self.current_coordinate_node_1()) / l
        G = self.G_matrix()
        r = self.r_vector()
        E = self.E_matrix()

        # -----------------------------------------
        # D_mat
        D_3 = 1/l * (np.eye(3) - r_1 @ r_1.T)
        D = np.zeros((12, 12), dtype=float)

        D[0: 3, 0: 3] = D_3
        D[0: 3, 6: 9] = -D_3
        D[6: 9, 0: 3] = -D_3
        D[6: 9, 6: 9] = D_3
        # -----------------------------------------
        # Q_mat
        P = self.P_matrix()
        temp = P.T @ f_a[1: 7]

        n_1, m_1, n_2, m_2 = temp[0: 3], temp[3: 6], temp[6: 9], temp[9: 12]

        Q = np.zeros((12, 3), dtype=float)
        Q[0: 3, 0: 3] = util.get_skew_symmetric(n_1)
        Q[3: 6, 0: 3] = util.get_skew_symmetric(m_1)
        Q[6: 9, 0: 3] = util.get_skew_symmetric(n_2)
        Q[9: 12, 0: 3] = util.get_skew_symmetric(m_2)

        # -----------------------------------------
        # a_vec
        eta = float(G[2, 0]) * l
        a = np.array([[0.0, float(eta/l * (f_a[1] + f_a[4]) - 1 /
                     l * (f_a[2] + f_a[5])), float(1/l * (f_a[3] + f_a[6]))]]).T
        # -----------------------------------------
        # K_mmat
        K_m = D * f_a[0] - E @ Q @ G.T @ E.T + E @ G @ a @ r
        # -----------------------------------------

        K_g = B_g.T @ K_a @ B_g + K_m
        f_g = B_g.T @ f_a

        return K_g, f_g
