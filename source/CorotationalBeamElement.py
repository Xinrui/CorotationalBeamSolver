# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou
"""

from math import atan2

import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, tan
from scipy.linalg import expm, logm

import source.Utilities as util


class CorotationalBeamElement2D():
    """A 2D corotational beam element.

    This class helps formulate a 2D corotational beam element, and calculate
    e.g. its local forces and tangential stiffness matrix.

    DoF / Global Displacement vector: 
        p = [u1, v1, theta_1, u2, v2, theta_2]

    Local force: 
        q_l = [N, M1_bar, M2_bar]

    """

    def __init__(self):
        # beamtype = 'Bernoulli', analysis = 'elastic'
        self._initial_coordinate_node_1 = None
        self._initial_coordinate_node_2 = None
        self._youngs_modulus = None
        self._area = None
        self._moment_of_inertia = None
        self._current_coordinate_node_1 = None
        self._current_coordinate_node_2 = None
        self._global_nodal_rotation_node_1 = None
        self._global_nodal_rotation_node_2 = None

    @property
    def initial_coordinate_node_1(self):
        return self._initial_coordinate_node_1

    @initial_coordinate_node_1.setter
    def initial_coordinate_node_1(self, val):
        """Set the initial coordinate of node 1: [X1, Y1, Z1]

        Raises:
            TypeError: If value is not a numpy array.

        """

        if isinstance(val, np.ndarray):
            self._initial_coordinate_node_1 = val
        else:
            raise TypeError(
                "The initial coordinate of node 1 must be a 2x1 array!")

    @property
    def initial_coordinate_node_2(self):
        return self._initial_coordinate_node_2

    @initial_coordinate_node_2.setter
    def initial_coordinate_node_2(self, val):
        """Set the initial coordinate of node 2: [X2, Y2, Z2]

        Raises:
            TypeError: If value is not a numpy array.

        """

        if isinstance(val, np.ndarray):
            self._initial_coordinate_node_2 = val
        else:
            raise TypeError(
                "The initial coordinate of node 2 must be a 2x1 array!")

    @property
    def current_coordinate_node_1(self):
        return self._current_coordinate_node_1

    @current_coordinate_node_1.setter
    def current_coordinate_node_1(self, val):
        """Set the current coordinate of node 1: [x1, y1, z1]

        Raises:
            TypeError: If value is not a numpy array.

        """

        if isinstance(val, np.ndarray):
            self._current_coordinate_node_1 = val
        else:
            raise TypeError(
                "The current coordinate of node 1 must be a 2x1 array!")

    @property
    def current_coordinate_node_2(self):
        return self._current_coordinate_node_2

    @current_coordinate_node_2.setter
    def current_coordinate_node_2(self, val):
        """Set the current coordinate of node 2: [x2, y2, z2]

        Raises:
            TypeError: If value is not a numpy array.

        """

        if isinstance(val, np.ndarray):
            self._current_coordinate_node_2 = val
        else:
            raise TypeError(
                "The current coordinate of node 2 must be a 2x1 array!")

    @property
    def global_nodal_rotation_node_1(self):
        return self._global_nodal_rotation_node_1

    @global_nodal_rotation_node_1.setter
    def global_nodal_rotation_node_1(self, val):
        """Set the global nodal rotation of node 1: θ1

        Raises:
            TypeError: If value is not a float number.

        """

        if isinstance(val, float):
            self._global_nodal_rotation_node_1 = val
        else:
            raise TypeError(
                "Global nodal rotation at node 1 must be a float number!")

    @property
    def global_nodal_rotation_node_2(self):
        return self._global_nodal_rotation_node_2

    @global_nodal_rotation_node_2.setter
    def global_nodal_rotation_node_2(self, val):
        """Set the global nodal rotation of node 2: θ2

        Raises:
            TypeError: If value is not a float number.

        """

        if isinstance(val, float):
            self._global_nodal_rotation_node_2 = val
        else:
            raise TypeError(
                "Global nodal rotation at node 2 must be a float number!")

    @property
    def youngs_modulus(self):
        return self._youngs_modulus

    @youngs_modulus.setter
    def youngs_modulus(self, val):
        """Set the Young's modulus of the beam element: E > 0.

        Raises:
            TypeError: If value is not a positive float number.

        """

        if not isinstance(val, float):
            raise TypeError("Young's modulus must be a float number!")
        elif val <= 0:
            raise ValueError("Young's modulus must be positive!")
        else:
            self._youngs_modulus = val

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
    def moment_of_inertia(self):
        return self._moment_of_inertia

    @moment_of_inertia.setter
    def moment_of_inertia(self, val):
        """Set the moment of inertia of the beam element: I > 0.

        Raises:
            TypeError: If value is not a positive float number.

        """
        if not isinstance(val, float):
            raise TypeError("Moment of inertia must be a float number!")
        elif val <= 0:
            raise ValueError("Moment of inertia must be positive!")
        else:
            self._moment_of_inertia = val

    @property
    def undeformed_length(self):
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
    def local_material_stiffness(self):
        k_l = float(self.youngs_modulus / self.current_length) * \
            np.array([[self.area,                         0.,                         0.],
                      [0., 4 * self.moment_of_inertia, 2 * self.moment_of_inertia],
                      [0., 2 * self.moment_of_inertia, 4 * self.moment_of_inertia]])

        return k_l

    @property
    def local_dof(self):
        p_l = np.zeros((3, 1), dtype=float)

        p_l[0] = self.current_length - self.undeformed_length

        p_l[1] = atan2((cos(self.current_angle) * sin(self.global_nodal_rotation_node_1 + self.initial_angle)
                        - sin(self.current_angle) * cos(self.global_nodal_rotation_node_1 + self.initial_angle)),
                       (cos(self.current_angle) * cos(self.global_nodal_rotation_node_1 + self.initial_angle)
                        + sin(self.current_angle) * sin(self.global_nodal_rotation_node_1 + self.initial_angle)))

        p_l[2] = atan2((cos(self.current_angle) * sin(self.global_nodal_rotation_node_2 + self.initial_angle)
                        - sin(self.current_angle) * cos(self.global_nodal_rotation_node_2 + self.initial_angle)),
                       (cos(self.current_angle) * cos(self.global_nodal_rotation_node_2 + self.initial_angle)
                        + sin(self.current_angle) * sin(self.global_nodal_rotation_node_2 + self.initial_angle)))

        return p_l

    @property
    def local_force(self):
        # Assemble axial force and local end moments into a vector q_l.
        p_l = self.local_dof
        k_l = self.local_material_stiffness

        return k_l @ p_l

    @property
    def B_matrix(self):
        # Calculate B-matrix for transformed material stiffness matrix.
        # B-matrix is the relation between infinitesimal deformation inducing
        # work and infinitesimal global deformation.
        # deltap_l = B @ deltap

        B = np.array([[-cos(self.current_angle), -sin(self.current_angle), 0., cos(self.current_angle), sin(self.current_angle), 0.],
                      [-sin(self.current_angle) / self.current_length, cos(self.current_angle)/self.current_length,
                       1., sin(self.current_angle)/self.current_length, -cos(self.current_angle)/self.current_length, 0.],
                      [-sin(self.current_angle) / self.current_length, cos(self.current_angle)/self.current_length,
                       0., sin(self.current_angle)/self.current_length, -cos(self.current_angle)/self.current_length, 1.]])

        return B

    @property
    def material_stiffness_matrix(self):
        # Calculate transformed material stiffness matrix k_t1.
        B = self.B_matrix
        k_l = self.local_material_stiffness
        k_t1 = B.T @ k_l @ B

        return k_t1

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
        k_t1 = self.material_stiffness_matrix
        k_tsigma = self.geometry_stiffness_matrix
        k_t = k_t1 + k_tsigma

        return k_t


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
            E: Young's modulus
            A: Cross-Sectional areas
            I_y: 2nd moments of inertia
        """
        self._initial_coordinate_node_1 = None
        self._initial_coordinate_node_2 = None
        self._current_coordinate_node_1 = None
        self._current_coordinate_node_2 = None

        self._youngs_modulus = None
        self._shear_modulus = None
        self._area = None

        self._moment_of_inertia_y = None
        self._moment_of_inertia_z = None
        self._polar_moment_of_inertia = None

        self._current_frame_node_1 = None
        self._current_frame_node_2 = None

        # theta_1, theta_2: Global nodal rotations of the beam element
        self.R_1g, self.R_2g = np.eye(3), np.eye(3)

    @property
    def initial_coordinate_node_1(self):
        return self._initial_coordinate_node_1

    @initial_coordinate_node_1.setter
    def initial_coordinate_node_1(self, val):
        """Set the initial coordinate of node 1: [X1, Y1, Z1]

        Raises:
            TypeError: If value is not a numpy array.

        """

        if isinstance(val, np.ndarray):
            self._initial_coordinate_node_1 = val
        else:
            raise TypeError(
                "The initial coordinate of node 1 must be a 2x1 array!")

    @property
    def initial_coordinate_node_2(self):
        return self._initial_coordinate_node_2

    @initial_coordinate_node_2.setter
    def initial_coordinate_node_2(self, val):
        """Set the initial coordinate of node 2: [X2, Y2, Z2]

        Raises:
            TypeError: If value is not a numpy array.

        """

        if isinstance(val, np.ndarray):
            self._initial_coordinate_node_2 = val
        else:
            raise TypeError(
                "The initial coordinate of node 2 must be a 2x1 array!")

    @property
    def current_coordinate_node_1(self):
        return self._current_coordinate_node_1

    @current_coordinate_node_1.setter
    def current_coordinate_node_1(self, val):
        """Set the current coordinate of node 1: [x1, y1, z1]

        Raises:
            TypeError: If value is not a numpy array.

        """

        if isinstance(val, np.ndarray):
            self._current_coordinate_node_1 = val
        else:
            raise TypeError(
                "The current coordinate of node 1 must be a 2x1 array!")

    @property
    def current_coordinate_node_2(self):
        return self._current_coordinate_node_2

    @current_coordinate_node_2.setter
    def current_coordinate_node_2(self, val):
        """Set the current coordinate of node 2: [x2, y2, z2]

        Raises:
            TypeError: If value is not a numpy array.

        """

        if isinstance(val, np.ndarray):
            self._current_coordinate_node_2 = val
        else:
            raise TypeError(
                "The current coordinate of node 2 must be a 2x1 array!")

    @property
    def current_frame_node_1(self):
        return self._current_frame_node_1

    @current_frame_node_1.setter
    def current_frame_node_1(self, val):
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
    def current_frame_node_2(self):
        return self._current_frame_node_2

    @current_frame_node_2.setter
    def current_frame_node_2(self, val):
        """Set the global nodal rotation of node 2: θ2

        Raises:
            TypeError: If value is not a float number.

        """

        if isinstance(val, float):
            self._current_frame_node_2 = val
        else:
            raise TypeError(
                "Orientation matrix at node 2 must be a 3x3 matrix!")

    @property
    def youngs_modulus(self):
        return self._youngs_modulus

    @youngs_modulus.setter
    def youngs_modulus(self, val):
        """Set the Young's modulus of the beam element: E > 0.

        Raises:
            TypeError: If value is not a positive float number.

        """

        if not isinstance(val, float):
            raise TypeError("Young's modulus must be a float number!")
        elif val <= 0:
            raise ValueError("Young's modulus must be positive!")
        else:
            self._youngs_modulus = val

    @property
    def shear_modulus(self):
        return self._shear_modulus

    @youngs_modulus.setter
    def shear_modulus(self, val):
        """Set the Shear modulus of the beam element: G > 0.

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
    def initial_frame(self):
        e_3o = np.array([[0, 0, 1]], dtype=float).T
        e_1o = (self.initial_coordinate_node_2 -
                self.initial_coordinate_node_1) / self.undeformed_length
        e_2o = np.cross(e_3o, e_1o, axisa=0, axisb=0, axisc=0)
        return np.c_[e_1o, e_2o, e_3o]

    @property
    def undeformed_length(self):
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

        k_l = np.zeros((7, 7))
        k_l[0, 0] = self._youngs_modulus * self._area

        k_l[1, 1], k_l[4, 4] = self.G * self.I_t, self.G * self.I_t
        k_l[1, 4], k_l[4, 1] = -self.G * self.I_t, -self.G * self.I_t

        k_l[2, 2], k_l[5, 5] = 4.0 * self.E * self.I_z, 4.0 * self.E * self.I_z
        k_l[2, 5], k_l[5, 2] = 2.0 * self.E * self.I_z, 2.0 * self.E * self.I_z

        k_l[3, 3], k_l[6, 6] = 4.0 * self.E * self.I_y, 4.0 * self.E * self.I_y
        k_l[3, 6], k_l[6, 3] = 2.0 * self.E * self.I_y, 2.0 * self.E * self.I_y

        k_l /= self.current_length

        return k_l

    @property
    def auxiliary_vector(self):
        q_1 = self.current_frame_node_1 @ self.initial_frame @ np.array(
            [[0, 1, 0]], dtype=float).T
        q_2 = self.current_frame_node_2 @ self.initial_frame @ np.array(
            [[0, 1, 0]], dtype=float).T
        q = (q_1 + q_2) / 2

        return q_1, q_2, q

    @property
    def orthogonal(self):
        r_1 = (self.current_coordinate_node_2 -
               self.current_coordinate_node_1) / self.current_length

        _, _, q = self.auxiliary_vector
        r_3 = np.cross(r_1, q, axisa=0, axisb=0, axisc=0)
        r_3 = r_3 / np.linalg.norm(r_3)

        r_2 = np.cross(r_3, r_1, axisa=0, axisb=0, axisc=0)

        return np.c_[r_1, r_2, r_3]

    @property
    def local_dof(self):
        p_l = np.zeros((7, 1), dtype=float)

        theta_1_tilde = logm(self.orthogonal.T @
                             self.current_frame_node_1 @ self.initial_frame)
        theta_2_tilde = logm(self.orthogonal.T @
                             self.current_frame_node_2 @ self.initial_frame)

        p_l[0] = self.current_length - self.undeformed_length
        p_l[1: 4] = util.decomposeSkewSymmetric(theta_1_tilde)
        p_l[4: 7] = util.decomposeSkewSymmetric(theta_2_tilde)

        return p_l

    @property
    def local_force(self):
        # Assemble axial force and local end moments into a vector q_l.
        f_l = self.local_material_stiffness @ self.local_dof

        return f_l

    @property
    def Ba_matrix(self):
        p_l = self.local_dof
        T_s_inv_theta_1bar = util.getTransformation(p_l[1: 4], 'inv')
        T_s_inv_theta_2bar = util.getTransformation(p_l[4: 7], 'inv')

        B_a = np.zeros((7, 7), dtype=float)
        B_a[0, 0] = 1.
        B_a[1:4, 1:4] = T_s_inv_theta_1bar
        B_a[4:7, 4:7] = T_s_inv_theta_2bar

        return B_a

    @property
    def local_force_a(self):
        f_a = self.Ba_matrix.T @ self.local_force

        return f_a

    def Khi_matrix(self, i):
        if i == 1:
            theta_bar = self.local_dof[1: 4]
            v = self.local_force[1: 4]
        elif i == 2:
            theta_bar = self.local_dof[4: 7]
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
        K_h[1:4, 1:4] = K_h1
        K_h[4:7, 4:7] = K_h2

        return K_h

    @property
    def Ka_matrix(self):

        K_a = self.Ba_matrix.T @ self.local_material_stiffness @ self.Ba_matrix + self.Kh_matrix

        return K_a

    @property
    def G_matrix(self):

        q_1, q_2, q = self.auxiliary_vector

        q1 = float((self.orthogonal.T @ q)[0])
        q2 = float((self.orthogonal.T @ q)[1])
        q11 = float((self.orthogonal.T @ q_1)[0])
        q12 = float((self.orthogonal.T @ q_1)[1])
        q21 = float((self.orthogonal.T @ q_2)[0])
        q22 = float((self.orthogonal.T @ q_2)[1])

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
        E[0: 3, 0: 3] = self.orthogonal
        E[3: 6, 3: 6] = self.orthogonal
        E[6: 9, 6: 9] = self.orthogonal
        E[9: 12, 9: 12] = self.orthogonal

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

        B_g = np.r_[self.r_vector, self.P_matrix @ self.E_matrix.T]

        return B_g

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
        q1 = float((self.orthogonal.T @ q)[0, 0])
        q2 = float((self.orthogonal.T @ q)[1, 0])
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
        f_a = self.local_force_a

        n_1 = (self.P_matrix.T @ f_a[1: 7])[0: 3]
        m_1 = (self.P_matrix.T @ f_a[1: 7])[3: 6]
        n_2 = (self.P_matrix.T @ f_a[1: 7])[6: 9]
        m_2 = (self.P_matrix.T @ f_a[1: 7])[9: 12]

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

        K_g = self.Bg_matrix.T @ self.Ka_matrix @ self.Bg_matrix + self.Km_matrix

        return K_g


class System():
    def __init__(self):
        self._dimension = None
        self._geometry_name = None

        self._structure = None

        self._number_of_elements = None
        self._number_of_nodes = None
        self._number_of_dofs = None
        self._number_of_load_increments = None
        self._max_load = None

        self._dirichlet_boundary_condition = []
        self._load_boundary_condition = []
        self._load_increment_vector = None

        self._tolerance = None
        self._max_iteration_steps = None
        self._solver = None

    @property
    def dimension(self):
        return self._dimension

    @dimension.setter
    def dimension(self, val):
        if not isinstance(val, int):
            raise TypeError("The dimension must be 2 or 3!")
        elif val != 2 and val != 3:
            raise ValueError("The dimension must be 2 or 3!")
        else:
            self._dimension = val

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
    def number_of_load_increments(self):
        return self._number_of_load_increments

    @number_of_load_increments.setter
    def number_of_load_increments(self, val):
        if not isinstance(val, int):
            raise TypeError("Number of load increments must be a integer!")
        elif val <= 0:
            raise ValueError("Number of load increments must be positive!")
        else:
            self._number_of_load_increments = val

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
        if val != "Load-Control" and val != "Arc-Length-Control":
            raise ValueError("Invalid solver!")
        else:
            self._solver = val

    def initialize_structure(self, *parameters):
        self._structure = []
        array_nodes, array_elements, self._number_of_nodes, self._number_of_elements = util.load_mesh_file(
            self.geometry_name)
        self._number_of_dofs = 3 * self._number_of_nodes
        for ele in array_elements.T:
            co_ele = CorotationalBeamElement2D()
            co_ele.initial_coordinate_node_1 = array_nodes[0: 2, ele[0] - 1].reshape(
                2, 1)
            co_ele.initial_coordinate_node_2 = array_nodes[0: 2, ele[1] - 1].reshape(
                2, 1)

            co_ele.youngs_modulus = parameters[0]
            co_ele.area = parameters[1]
            co_ele.moment_of_inertia = parameters[2]

            # co_ele.current_coordinate_node_1 = array_nodes[:, ele[0] - 1].reshape(3, 1)
            # co_ele.current_coordinate_node_2 = array_nodes[:, ele[1] - 1].reshape(3, 1)

            co_ele.current_coordinate_node_1 = co_ele.initial_coordinate_node_1
            co_ele.current_coordinate_node_2 = co_ele.initial_coordinate_node_2

            co_ele.global_nodal_rotation_node_1 = 0.0
            co_ele.global_nodal_rotation_node_2 = 0.0
            self._structure.append(co_ele)

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

    def add_load_bc(self, node, dof):
        if self.dimension == 2:
            if dof == "x":
                self._load_boundary_condition.append(3 * node)
            elif dof == "y":
                self._load_boundary_condition.append(3 * node + 1)
            elif dof == "m":
                self._load_boundary_condition.append(3 * node + 2)
        self._load_increment_vector = np.zeros(
            (self._number_of_dofs, 1), dtype=float)
        self._load_increment_vector[self._load_boundary_condition] = -1

    def mergeElemIntoMasterStiff(self):
        """ Assemble system stiffness matrix K from member stiffness matrices.

            Args:
                beam: the list of beam elements
                nele: the number of elements
                ndof: the number of degree of freedoms
                q_l: the storage vector of local forces, [3 * nele x 1]

            Returns:
                K: the system stiffness matrix, [ndof x ndof]
        """
        # dof_per_node = 3

        K = np.zeros((self._number_of_dofs, self._number_of_dofs), dtype=float)

        for iele, ele in enumerate(self._structure):
            k_t = ele.global_stiffness_matrix
            EFT = np.linspace(3 * iele, 3 * iele + 5, num=6, dtype=int)

            for idof, iEFT in enumerate(EFT):
                for jdof, jEFT in enumerate(EFT):
                    K[iEFT, jEFT] += k_t[idof, jdof]
        return K

    def getInternalForceVector(self):
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
        for iele, ele in enumerate(self._structure):
            q = ele.B_matrix.T @ ele.local_force
            EFT = np.linspace(3 * iele, 3 * iele + 5, num=6, dtype=int)
            for idof, iEFT in enumerate(EFT):
                F_int[iEFT] += q[idof]
        return F_int

    def updateMemberData(self, u):
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

        for iele, ele in enumerate(self._structure):
            # update nodal displacements
            ele.current_coordinate_node_1 = ele.initial_coordinate_node_1 + \
                u[3 * iele: 3 * iele + 2]
            ele.current_coordinate_node_2 = ele.initial_coordinate_node_2 + \
                u[3 * iele + 3: 3 * iele + 5]

            ele.global_nodal_rotation_node_1 = float(u[3 * iele + 2])
            ele.global_nodal_rotation_node_2 = float(u[3 * iele + 5])

            # self._structure[iele].current_coordinate_node_1 = self._structure[iele].initial_coordinate_node_1 + u[3 * iele: 3 * iele + 2]
            # self._structure[iele].current_coordinate_node_2 = self._structure[iele].initial_coordinate_node_2 + u[3 * iele + 3: 3 * iele + 5]

            # self._structure[iele].global_nodal_rotation_node_1 = float(u[3 * iele + 2])
            # self._structure[iele].global_nodal_rotation_node_2 = float(u[3 * iele + 5])

    def solve_the_system(self):
        lam = 0.
        u = np.zeros((self._number_of_dofs, 1), dtype=float)
        U = np.array([0.], dtype=float)
        LAM = np.array([0.], dtype=float)

        for n in range(self.number_of_load_increments):

            # set the predictor by equal load increments
            K = self.mergeElemIntoMasterStiff()
            K_s = util.modifyMasterStiffForDBC(
                K, self._dirichlet_boundary_condition)
            dF = self.max_load / self.number_of_load_increments * self._load_increment_vector

            u_pre = u + np.linalg.solve(K_s, dF)
            lam_pre = lam + self.max_load / self.number_of_load_increments

            # update member data
            self.updateMemberData(u_pre)

            # calculate internal force vector
            F_int = self.getInternalForceVector()

            # calculate the residual of the system
            r = F_int - lam_pre * self._load_increment_vector
            r = util.modifyTheResidual(r, self._dirichlet_boundary_condition)
            r_norm = np.linalg.norm(r)

            # copy them for iteration, "temp" means they are not on equilibrium path.
            u_temp = u_pre

            # initialize iteration counter
            kiter = 0

            # iterate, until good result or so many iteration steps
            while(r_norm > self.tolerance and kiter < self.max_iteration_steps):

                # load-Control
                K = self.mergeElemIntoMasterStiff()
                K_s = util.modifyMasterStiffForDBC(
                    K, self._dirichlet_boundary_condition)
                deltau = np.linalg.solve(K_s, -r)
                u_temp += deltau

                # update member data
                self.updateMemberData(u_temp)

                # calculate internal force vector
                F_int = self.getInternalForceVector()

                # calculate the residual of the system
                r = F_int - lam_pre * self._load_increment_vector
                r = util.modifyTheResidual(
                    r, self._dirichlet_boundary_condition)
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

    def plotTheStructure(self):
        """ Plot the UNDEFORMED and the DEFORMED structure.
            Args:
                X, Y: coordinates of the undeformed structure
                self.number_of_nodes: the number of nodes of the system
                self.number_of_elements: the number of elements
                beam: the list of the beam elements
        """
        # generate coordinates of deformed structure
        X = np.zeros((self._number_of_nodes, 1))
        Y = np.zeros((self._number_of_nodes, 1))
        x = np.zeros((self._number_of_nodes, 1))
        y = np.zeros((self._number_of_nodes, 1))
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
        fig, ax = plt.subplots()
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

# def mergeElemIntoMasterStiff(beam):
#     """ Assemble system stiffness matrix K from member stiffness matrices.

#         Args:
#             beam: the list of beam elements

#         Returns:
#             K: the system stiffness matrix, [ndof x ndof]
#     """
#     self.number_of_elements = len(beam)
#     ndof = 6 * (self.number_of_elements + 1)
#     K = np.zeros((ndof, ndof), dtype=float)
#     for iele in range(self.number_of_elements):
#         k_t = beam[iele].getGlobalStiffness_K_g()
#         EFT = np.linspace(6 * iele, 6 * iele + 11, num=12, dtype=int)
#         for idof in range(len(EFT)):
#             for jdof in range(len(EFT)):
#                 K[EFT[idof], EFT[jdof]] += k_t[idof, jdof]
#     return K


def modifyMasterStiffForDBC(K, DBCdof):
    """ Modify the system stiffness matrix K to K_s according to Drichlet
        Boundary Conditions.

        Args:
            K: the system stiffness matrix, [ndof x ndof]
            DBCdof: a list contains the dofs, such as [0, 1, 2]

        Returns:
            K_s: the modified system stiffness matrix, [ndof x ndof]
    """
    ndof = np.shape(K)[0]
    nDBCdof = len(DBCdof)
    K_s = np.copy(K)

    for idof in range(nDBCdof):
        for jentry in range(ndof):
            K_s[DBCdof[idof], jentry] = 0
            K_s[jentry, DBCdof[idof]] = 0
            K_s[DBCdof[idof], DBCdof[idof]] = 1
    return K_s

# def getInternalForceVector(beam):
#     """ Assemble global internal force vector F_int extracting the internal
#         force vector of every beam element.

#         Args:
#             beam: the list of beam elements

#         Returns:
#             F_int: the global internal force vector, [ndof x 1]
#     """
#     self.number_of_elements = len(beam)
#     ndof = 6 * (self.number_of_elements + 1)
#     F_int = np.zeros((ndof, 1), dtype=float)
#     for iele in range(self.number_of_elements):
#         q = beam[iele].getVector_f_g()
#         EFT = np.linspace(6 * iele, 6 * iele + 11, num=12, dtype=int)
#         for idof in range(len(EFT)):
#             F_int[EFT[idof]] += q[idof]
#     return F_int


# def updateMemberData(u, beam, deltau):
#     """ Update nodal displacements, local forces and storage vector.

#         Args:
#             u: global displacement vector, [ndof x 1]
#             beam: the list of beam elements

#     """
#     self.number_of_elements = len(beam)
#     for iele in range(self.number_of_elements):
#         # update nodal displacements
#         beam[iele].x1 = beam[iele].X1 + u[6 * iele: 6 * iele + 3]
#         beam[iele].x2 = beam[iele].X2 + u[6 * iele + 6: 6 * iele + 9]

#         # beam[iele].psi_1 += u[6 * iele + 3: 6 * iele + 6]
#         # beam[iele].psi_2 += u[6 * iele + 9: 6 * iele + 12]

#         # beam[iele].R_1g = expm(getSkewSymmetric(beam[iele].psi_1)) #@ beam[iele].R_1g
#         # beam[iele].R_2g = expm(getSkewSymmetric(beam[iele].psi_2)) #@ beam[iele].R_2g

#         beam[iele].R_1g = expm(getSkewSymmetric(deltau[6 * iele + 3: 6 * iele + 6])) @ beam[iele].R_1g
#         beam[iele].R_2g = expm(getSkewSymmetric(deltau[6 * iele + 9: 6 * iele + 12])) @ beam[iele].R_2g

def modifyTheResidual(r, DBCdof):
    """ Modify the residual according to Drichlet Boundary Conditions.

        Args:
            r: the residual of the system, [ndof x 1]
            DBCdof: a list contains the dofs, such as [0, 1, 2]

        Returns:
            r: the MODIFIED residual of the system, [ndof x 1]
    """
    for idof in range(len(DBCdof)):
        r[DBCdof[idof]] = 0
    return r


# def plotTheStructure(beam):
#     """ Plot the UNDEFORMED and the DEFORMED structure, as the name of the
#         function.

#         Args:
#             beam: the list of the beam elements
#     """
#     self.number_of_elements = len(beam)
#     self.number_of_nodes = self.number_of_elements + 1
#     # generate coordinates of deformed structure
#     X = np.zeros(self.number_of_nodes)
#     Y = np.zeros(self.number_of_nodes)
#     Z = np.zeros(self.number_of_nodes)
#     x = np.zeros(self.number_of_nodes)
#     y = np.zeros(self.number_of_nodes)
#     z = np.zeros(self.number_of_nodes)
#     for iele in range(self.number_of_elements):
#         X[iele] = beam[iele].X1[0]
#         Y[iele] = beam[iele].X1[1]
#         Z[iele] = beam[iele].X1[2]
#         x[iele] = beam[iele].x1[0]
#         y[iele] = beam[iele].x1[1]
#         z[iele] = beam[iele].x1[2]
#         if iele == self.number_of_elements - 1:
#             X[iele + 1] = beam[iele].X2[0]
#             Y[iele + 1] = beam[iele].X2[1]
#             Z[iele + 1] = beam[iele].X2[2]
#             x[iele + 1] = beam[iele].x2[0]
#             y[iele + 1] = beam[iele].x2[1]
#             z[iele + 1] = beam[iele].x2[2]

#     # Plot both configurations
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
#     fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')
#     ax = plt.axes(projection='3d')
#     ax.plot(X, Y, Z, linestyle='--', label='undeformed configuration')
#     ax.plot(x, y, z, linestyle='-', label='deformed configuration')
#     ax.legend(loc='lower right')
#     ax.set_xlabel('$x$')
#     ax.set_ylabel('$y$')
#     ax.set_zlabel('$z$')
#     ax.set_title('Undeflected(dashed) and Deflected(solid) 3D beam structure')
#     ax.grid()
#     plt.show()


def plotLoadDisplacementCurve(U, LAM):
    """ Plot the equilibrium path.

        Args:
            U: vector of the state parameters at the interesting dof, [ninc x 1]
            LAM: vector of the control parameters, [ninc x 1]
    """

    # Plot both configurations
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots()
    ax.plot(U, LAM, '.-')
    ax.set_xlabel('$u$')
    ax.set_ylabel('$\lambda$')
    ax.set_title('Equilibrium Path')
    ax.grid()
    plt.show()
