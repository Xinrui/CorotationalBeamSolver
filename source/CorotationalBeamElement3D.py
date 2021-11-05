# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou
"""
import numpy as np
from scipy.linalg import logm
import source.Utilities as util

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
            moment_of_inertia_y: 2nd moments of inertia
        """
        self.__initial_coordinate_node_1 = None
        self.__initial_coordinate_node_2 = None
        self.__incremental_global_displacement = np.zeros((12, 1), dtype=float)

        self.__youngs_modulus = None
        self.__shear_modulus = None

        self.__width = None
        self.__height = None

        self.__analysis = "elastic"
        self.__beamtype = "Bernoulli"
        
        self._current_frame_node_1 = np.eye(3)
        self._current_frame_node_2 = np.eye(3)

        self.__element_freedom_table = None

        self.__num_of_gauss_locations_xi = 2   # x-axis
        self.__num_of_gauss_locations_eta = 2  # y-axis
        self.__num_of_gauss_locations_mu = 2   # z-axis

        # All plastic cases, including Perfect Plasticity / No Hardening
        self.__yield_stress = None

        # Kinematic Hardening
        self.__kinematic_hardening_modulus = 0.0

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
        
        # basic variables
        self.__current_length = self.initial_length
        self.__q1, self.__q2, self.__q = None, None, None
        self.__local_stiffness_matrix = None

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
                "The initial coordinate of node 1 must be a 3x1 array!")

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
                "The initial coordinate of node 2 must be a 3x1 array!")

    @property
    def incremental_global_displacement(self):
        return self.__incremental_global_displacement

    @incremental_global_displacement.setter
    def incremental_global_displacement(self, val):
        if isinstance(val, np.ndarray):
            self.__incremental_global_displacement = val
        else:
            raise TypeError(
                "Global displacement must be a 12x1 array!")

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
            raise ValueError("Kinematic hardening modulus must be positive!")
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
    def poisson_ratio(self):
        return self.__youngs_modulus / (2 * self.__shear_modulus) - 1
    
    @property
    def first_lame_constant(self):
        return self.__youngs_modulus * self.poisson_ratio / (1 + self.poisson_ratio) / (1 - 2 * self.poisson_ratio)
    
    @property
    def second_lame_constant(self):
        return self.__shear_modulus
        
    @property
    def area(self):
        """
        Set the cross-sectional area of the beam element.

        A = b * h

        """
        return self.__width * self.__height

    @property
    def moment_of_inertia_y(self):
        """
        Set the moment of inertia Iy / I33 of the beam element.

        I_y = 1/12 * b * h^3

        """
        return 1/12 * self.__width * (self.__height) ** 3

    @property
    def moment_of_inertia_z(self):
        """
        Set the moment of inertia Iz / I22 of the beam element.

        I_z = 1/12 * b^3 * h

        """
        return 1/12 * (self.__width) ** 3 * self.__height

    @property
    def polar_moment_of_inertia(self):
        """
        Set the polar moment of inertia Io of the beam element.

        I_o = 1/12 * (b^3 * h + b * h^3)

        """
        return 1/12 * (self.__width * (self.__height) ** 3 + (self.__width) ** 3 * self.__height)

    @property
    def fourth_order_polar_moment_of_inertia(self):
        """
        Set the 4th polar moment of inertia Io of the beam element.

        I_rr = b^5 * h / 80 + b^3 * h^3 / 72 + b * h^5 / 80

        """
        return 1/80 * (self.__width * (self.__height) ** 5 + (self.__width) ** 5 * self.__height) + 1/72 * (self.__width) ** 3 * (self.__height) ** 3

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
                (3, 1, self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta, self.__num_of_gauss_locations_mu), dtype=float)
            self.__plastic_strain = np.zeros(
                (3, 1, self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta, self.__num_of_gauss_locations_mu), dtype=float)
            self.__back_stress = np.zeros(
                (3, 1, self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta, self.__num_of_gauss_locations_mu), dtype=float)
            self.__tangent_moduli = np.zeros(
                (3, 3, self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta, self.__num_of_gauss_locations_mu), dtype=float)
            tangent_moduli = np.array([[self.__youngs_modulus, 0., 0.],
                                       [0., self.__shear_modulus, 0.],
                                       [0., 0., self.__shear_modulus]])
            for ixi in range(self.__num_of_gauss_locations_xi):
                for jeta in range(self.__num_of_gauss_locations_eta):
                    for kmu in range(self.__num_of_gauss_locations_mu):
                        self.__tangent_moduli[:, :, ixi,
                                              jeta, kmu] = tangent_moduli
        elif val == "linear hardening":
            self.__analysis = val
            self.__stress = np.zeros(
                (6, 1, self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta, self.__num_of_gauss_locations_mu), dtype=float)
            self.__plastic_strain = np.zeros(
                (6, 1, self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta, self.__num_of_gauss_locations_mu), dtype=float)
            self.__back_stress = np.zeros(
                (6, 1, self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta, self.__num_of_gauss_locations_mu), dtype=float)
            self.__tangent_moduli = np.zeros(
                (6, 6, self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta, self.__num_of_gauss_locations_mu), dtype=float)
            
            tangent_moduli = self.first_lame_constant * util.unit_tensor() @ util.unit_tensor().T + 2 * self.second_lame_constant * np.eye(6)
            for ixi in range(self.__num_of_gauss_locations_xi):
                for jeta in range(self.__num_of_gauss_locations_eta):
                    for kmu in range(self.__num_of_gauss_locations_mu):
                        self.__tangent_moduli[:, :, ixi,
                                              jeta, kmu] = tangent_moduli
            self.__internal_hardening_variable = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta, self.__num_of_gauss_locations_mu), dtype=float)
        elif val == "quadratic hardening":
            self.__analysis = val
            self.__stress = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__plastic_strain = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__back_stress = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
            self.__tangent_moduli = self.__youngs_modulus * \
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
            self.__tangent_moduli = self.__youngs_modulus * \
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
            self.__tangent_moduli = self.__youngs_modulus * \
                np.ones((self.__num_of_gauss_locations_xi,
                        self.__num_of_gauss_locations_eta), dtype=float)
            self.__internal_hardening_variable = np.zeros(
                (self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta), dtype=float)
        else:
            print("These five models are implemented here: 1. elastic; 2. linear hardening; 3. quadratic hardening; 4. exponential hardening; 5. ramberg-osgood hardening.")
            raise ValueError("Wrong constitutive law!")

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


    def update_current_length(self):
        """
        Calculate the deformed length l of the beam element and
        save it as an attribute.
        """

        self.__current_length = np.linalg.norm(self.current_coordinate_node_2 -
                              self.current_coordinate_node_1)
    
    @property
    def current_length(self):
        """
        Calculate the deformed length l of the beam element and
        save it as an attribute.
        """

        return self.__current_length

    def update_local_stiffness_matrix(self):
        u, t11, t21, t31, t12, t22, t32 = self.local_displacement.reshape(7)
        E, G, A, L, I22, I33, Irr = self.__youngs_modulus, self.__shear_modulus, self.area, self.initial_length, self.moment_of_inertia_z, self.moment_of_inertia_y, self.fourth_order_polar_moment_of_inertia
        kl = np.array([[1.0*A*E/L, 1.2*E*(I22*t11 - I22*t12 + I33*t11 - I33*t12)/L**2, A*E*(0.133333333333333*t21 - 0.0333333333333333*t22), A*E*(0.133333333333333*t31 - 0.0333333333333333*t32), 1.2*E*(-I22*t11 + I22*t12 - I33*t11 + I33*t12)/L**2, A*E*(-0.0333333333333333*t21 + 0.133333333333333*t22), A*E*(-0.0333333333333333*t31 + 0.133333333333333*t32)], [1.2*E*(I22*t11 - I22*t12 + I33*t11 - I33*t12)/L**2, 0.08*E*I22*t21**2/L - 0.04*E*I22*t21*t22/L + 0.08*E*I22*t22**2/L + 0.08*E*I22*t31**2/L - 0.04*E*I22*t31*t32/L + 0.08*E*I22*t32**2/L + 1.2*E*I22*u/L**2 + 0.08*E*I33*t21**2/L - 0.04*E*I33*t21*t22/L + 0.08*E*I33*t22**2/L + 0.08*E*I33*t31**2/L - 0.04*E*I33*t31*t32/L + 0.08*E*I33*t32**2/L + 1.2*E*I33*u/L**2 + 3.08571428571469*E*Irr*t11**2/L**3 - 6.17142857142937*E*Irr*t11*t12/L**3 + 3.08571428571469*E*Irr*t12**2/L**3 + 1.2*G*I22/L + 1.2*G*I33/L - 0.925714285714687*E*I22**2*t11**2/(A*L**3) + 1.85142857142937*E*I22**2*t11*t12/(A*L**3) - 0.925714285714687*E*I22**2*t12**2/(A*L**3) - 1.85142857142937*E*I22*I33*t11**2/(A*L**3) + 3.70285714285875*E*I22*I33*t11*t12/(A*L**3) - 1.85142857142937*E*I22*I33*t12**2/(A*L**3) - 0.925714285714687*E*I33**2*t11**2/(A*L**3) + 1.85142857142937*E*I33**2*t11*t12/(A*L**3) - 0.925714285714687*E*I33**2*t12**2/(A*L**3), E*(0.16*I22*t11*t21 - 0.04*I22*t11*t22 - 0.16*I22*t12*t21 + 0.04*I22*t12*t22 + 0.16*I33*t11*t21 - 0.04*I33*t11*t22 - 0.16*I33*t12*t21 + 0.04*I33*t12*t22)/L, E*(0.16*I22*t11*t31 - 0.04*I22*t11*t32 - 0.16*I22*t12*t31 + 0.04*I22*t12*t32 + 0.16*I33*t11*t31 - 0.04*I33*t11*t32 - 0.16*I33*t12*t31 + 0.04*I33*t12*t32)/L, -0.08*E*I22*t21**2/L + 0.04*E*I22*t21*t22/L - 0.08*E*I22*t22**2/L - 0.08*E*I22*t31**2/L + 0.04*E*I22*t31*t32/L - 0.08*E*I22*t32**2/L - 1.2*E*I22*u/L**2 - 0.08*E*I33*t21**2/L + 0.04*E*I33*t21*t22/L - 0.08*E*I33*t22**2/L - 0.08*E*I33*t31**2/L + 0.04*E*I33*t31*t32/L - 0.08*E*I33*t32**2/L - 1.2*E*I33*u/L**2 - 3.08571428571469*E*Irr*t11**2/L**3 + 6.17142857142937*E*Irr*t11*t12/L**3 - 3.08571428571469*E*Irr*t12**2/L**3 - 1.2*G*I22/L - 1.2*G*I33/L + 0.925714285714687*E*I22**2*t11**2/(A*L**3) - 1.85142857142937*E*I22**2*t11*t12/(A*L**3) + 0.925714285714687*E*I22**2*t12**2/(A*L**3) + 1.85142857142937*E*I22*I33*t11**2/(A*L**3) - 3.70285714285875*E*I22*I33*t11*t12/(A*L**3) + 1.85142857142937*E*I22*I33*t12**2/(A*L**3) + 0.925714285714687*E*I33**2*t11**2/(A*L**3) - 1.85142857142937*E*I33**2*t11*t12/(A*L**3) + 0.925714285714687*E*I33**2*t12**2/(A*L**3), E*(-0.04*I22*t11*t21 + 0.16*I22*t11*t22 + 0.04*I22*t12*t21 - 0.16*I22*t12*t22 - 0.04*I33*t11*t21 + 0.16*I33*t11*t22 + 0.04*I33*t12*t21 - 0.16*I33*t12*t22)/L, E*(-0.04*I22*t11*t31 + 0.16*I22*t11*t32 + 0.04*I22*t12*t31 - 0.16*I22*t12*t32 - 0.04*I33*t11*t31 + 0.16*I33*t11*t32 + 0.04*I33*t12*t31 - 0.16*I33*t12*t32)/L], [A*E*(0.133333333333333*t21 - 0.0333333333333333*t22), E*(0.16*I22*t11*t21 - 0.04*I22*t11*t22 - 0.16*I22*t12*t21 + 0.04*I22*t12*t22 + 0.16*I33*t11*t21 - 0.04*I33*t11*t22 - 0.16*I33*t12*t21 + 0.04*I33*t12*t22)/L, E*(0.0266666666666667*A*L**2*t21**2 - 0.0133333333333333*A*L**2*t21*t22 + 0.01*A*L**2*t22**2 + 0.00888888888888889*A*L**2*t31**2 - 0.00444444444444444*A*L**2*t31*t32 + 0.00888888888888889*A*L**2*t32**2 + 0.133333333333333*A*L*u + 0.08*I22*t11**2 - 0.16*I22*t11*t12 + 0.08*I22*t12**2 + 0.08*I33*t11**2 - 0.16*I33*t11*t12 + 0.08*I33*t12**2 + 4.0*I33)/L, A*E*L*(0.0177777777777778*t21*t31 - 0.00444444444444444*t21*t32 - 0.00444444444444444*t22*t31 + 0.00111111111111111*t22*t32), E*(-0.16*I22*t11*t21 + 0.04*I22*t11*t22 + 0.16*I22*t12*t21 - 0.04*I22*t12*t22 - 0.16*I33*t11*t21 + 0.04*I33*t11*t22 + 0.16*I33*t12*t21 - 0.04*I33*t12*t22)/L, E*(-0.00666666666666667*A*L**2*t21**2 + 0.02*A*L**2*t21*t22 - 0.00666666666666667*A*L**2*t22**2 - 0.00222222222222222*A*L**2*t31**2 + 0.00111111111111111*A*L**2*t31*t32 - 0.00222222222222222*A*L**2*t32**2 - 0.0333333333333333*A*L*u - 0.02*I22*t11**2 + 0.04*I22*t11*t12 - 0.02*I22*t12**2 - 0.02*I33*t11**2 + 0.04*I33*t11*t12 - 0.02*I33*t12**2 + 2.0*I33)/L, A*E*L*(-0.00444444444444444*t21*t31 + 0.0177777777777778*t21*t32 + 0.00111111111111111*t22*t31 - 0.00444444444444444*t22*t32)], [A*E*(0.133333333333333*t31 - 0.0333333333333333*t32), E*(0.16*I22*t11*t31 - 0.04*I22*t11*t32 - 0.16*I22*t12*t31 + 0.04*I22*t12*t32 + 0.16*I33*t11*t31 - 0.04*I33*t11*t32 - 0.16*I33*t12*t31 + 0.04*I33*t12*t32)/L, A*E*L*(0.0177777777777778*t21*t31 - 0.00444444444444444*t21*t32 - 0.00444444444444444*t22*t31 + 0.00111111111111111*t22*t32), E*(0.00888888888888889*A*L**2*t21**2 - 0.00444444444444444*A*L**2*t21*t22 + 0.00888888888888889*A*L**2*t22**2 + 0.0266666666666667*A*L**2*t31**2 - 0.0133333333333333*A*L**2*t31*t32 + 0.01*A*L**2*t32**2 + 0.133333333333333*A*L*u + 0.08*I22*t11**2 - 0.16*I22*t11*t12 + 0.08*I22*t12**2 + 4.0*I22 + 0.08*I33*t11**2 - 0.16*I33*t11*t12 + 0.08*I33*t12**2)/L, E*(-0.16*I22*t11*t31 + 0.04*I22*t11*t32 + 0.16*I22*t12*t31 - 0.04*I22*t12*t32 - 0.16*I33*t11*t31 + 0.04*I33*t11*t32 + 0.16*I33*t12*t31 - 0.04*I33*t12*t32)/L, A*E*L*(-0.00444444444444444*t21*t31 + 0.00111111111111111*t21*t32 + 0.0177777777777778*t22*t31 - 0.00444444444444444*t22*t32), E*(-0.00222222222222222*A*L**2*t21**2 + 0.00111111111111111*A*L**2*t21*t22 - 0.00222222222222222*A*L**2*t22**2 - 0.00666666666666667*A*L**2*t31**2 + 0.02*A*L**2*t31*t32 - 0.00666666666666667*A*L**2*t32**2 - 0.0333333333333333*A*L*u - 0.02*I22*t11**2 + 0.04*I22*t11*t12 - 0.02*I22*t12**2 + 2.0*I22 - 0.02*I33*t11**2 + 0.04*I33*t11*t12 - 0.02*I33*t12**2)/L], [1.2*E*(-I22*t11 + I22*t12 - I33*t11 + I33*t12)/L**2, -0.08*E*I22*t21**2/L + 0.04*E*I22*t21*t22/L - 0.08*E*I22*t22**2/L - 0.08*E*I22*t31**2/L + 0.04*E*I22*t31*t32/L - 0.08*E*I22*t32**2/L - 1.2*E*I22*u/L**2 - 0.08*E*I33*t21**2/L + 0.04*E*I33*t21*t22/L - 0.08*E*I33*t22**2/L - 0.08*E*I33*t31**2/L + 0.04*E*I33*t31*t32/L - 0.08*E*I33*t32**2/L - 1.2*E*I33*u/L**2 - 3.08571428571469*E*Irr*t11**2/L**3 + 6.17142857142937*E*Irr*t11*t12/L**3 - 3.08571428571469*E*Irr*t12**2/L**3 - 1.2*G*I22/L - 1.2*G*I33/L + 0.925714285714687*E*I22**2*t11**2/(A*L**3) - 1.85142857142937*E*I22**2*t11*t12/(A*L**3) + 0.925714285714687*E*I22**2*t12**2/(A*L**3) + 1.85142857142937*E*I22*I33*t11**2/(A*L**3) - 3.70285714285875*E*I22*I33*t11*t12/(A*L**3) + 1.85142857142937*E*I22*I33*t12**2/(A*L**3) + 0.925714285714687*E*I33**2*t11**2/(A*L**3) - 1.85142857142937*E*I33**2*t11*t12/(A*L**3) + 0.925714285714687*E*I33**2*t12**2/(A*L**3), E*(-0.16*I22*t11*t21 + 0.04*I22*t11*t22 + 0.16*I22*t12*t21 - 0.04*I22*t12*t22 - 0.16*I33*t11*t21 + 0.04*I33*t11*t22 + 0.16*I33*t12*t21 - 0.04*I33*t12*t22)/L, E*(-0.16*I22*t11*t31 + 0.04*I22*t11*t32 + 0.16*I22*t12*t31 - 0.04*I22*t12*t32 - 0.16*I33*t11*t31 + 0.04*I33*t11*t32 + 0.16*I33*t12*t31 - 0.04*I33*t12*t32)/L, 0.08*E*I22*t21**2/L - 0.04*E*I22*t21*t22/L + 0.08*E*I22*t22**2/L + 0.08*E*I22*t31**2/L - 0.04*E*I22*t31*t32/L + 0.08*E*I22*t32**2/L + 1.2*E*I22*u/L**2 + 0.08*E*I33*t21**2/L - 0.04*E*I33*t21*t22/L + 0.08*E*I33*t22**2/L + 0.08*E*I33*t31**2/L - 0.04*E*I33*t31*t32/L + 0.08*E*I33*t32**2/L + 1.2*E*I33*u/L**2 + 3.08571428571469*E*Irr*t11**2/L**3 - 6.17142857142937*E*Irr*t11*t12/L**3 + 3.08571428571469*E*Irr*t12**2/L**3 + 1.2*G*I22/L + 1.2*G*I33/L - 0.925714285714687*E*I22**2*t11**2/(A*L**3) + 1.85142857142937*E*I22**2*t11*t12/(A*L**3) - 0.925714285714687*E*I22**2*t12**2/(A*L**3) - 1.85142857142937*E*I22*I33*t11**2/(A*L**3) + 3.70285714285875*E*I22*I33*t11*t12/(A*L**3) - 1.85142857142937*E*I22*I33*t12**2/(A*L**3) - 0.925714285714687*E*I33**2*t11**2/(A*L**3) + 1.85142857142937*E*I33**2*t11*t12/(A*L**3) - 0.925714285714687*E*I33**2*t12**2/(A*L**3), E*(0.04*I22*t11*t21 - 0.16*I22*t11*t22 - 0.04*I22*t12*t21 + 0.16*I22*t12*t22 + 0.04*I33*t11*t21 - 0.16*I33*t11*t22 - 0.04*I33*t12*t21 + 0.16*I33*t12*t22)/L, E*(0.04*I22*t11*t31 - 0.16*I22*t11*t32 - 0.04*I22*t12*t31 + 0.16*I22*t12*t32 + 0.04*I33*t11*t31 - 0.16*I33*t11*t32 - 0.04*I33*t12*t31 + 0.16*I33*t12*t32)/L], [A*E*(-0.0333333333333333*t21 + 0.133333333333333*t22), E*(-0.04*I22*t11*t21 + 0.16*I22*t11*t22 + 0.04*I22*t12*t21 - 0.16*I22*t12*t22 - 0.04*I33*t11*t21 + 0.16*I33*t11*t22 + 0.04*I33*t12*t21 - 0.16*I33*t12*t22)/L, E*(-0.00666666666666667*A*L**2*t21**2 + 0.02*A*L**2*t21*t22 - 0.00666666666666667*A*L**2*t22**2 - 0.00222222222222222*A*L**2*t31**2 + 0.00111111111111111*A*L**2*t31*t32 - 0.00222222222222222*A*L**2*t32**2 - 0.0333333333333333*A*L*u - 0.02*I22*t11**2 + 0.04*I22*t11*t12 - 0.02*I22*t12**2 - 0.02*I33*t11**2 + 0.04*I33*t11*t12 - 0.02*I33*t12**2 + 2.0*I33)/L, A*E*L*(-0.00444444444444444*t21*t31 + 0.00111111111111111*t21*t32 + 0.0177777777777778*t22*t31 - 0.00444444444444444*t22*t32), E*(0.04*I22*t11*t21 - 0.16*I22*t11*t22 - 0.04*I22*t12*t21 + 0.16*I22*t12*t22 + 0.04*I33*t11*t21 - 0.16*I33*t11*t22 - 0.04*I33*t12*t21 + 0.16*I33*t12*t22)/L, E*(0.01*A*L**2*t21**2 - 0.0133333333333333*A*L**2*t21*t22 + 0.0266666666666667*A*L**2*t22**2 + 0.00888888888888889*A*L**2*t31**2 - 0.00444444444444444*A*L**2*t31*t32 + 0.00888888888888889*A*L**2*t32**2 + 0.133333333333333*A*L*u + 0.08*I22*t11**2 - 0.16*I22*t11*t12 + 0.08*I22*t12**2 + 0.08*I33*t11**2 - 0.16*I33*t11*t12 + 0.08*I33*t12**2 + 4.0*I33)/L, A*E*L*(0.00111111111111111*t21*t31 - 0.00444444444444444*t21*t32 - 0.00444444444444444*t22*t31 + 0.0177777777777778*t22*t32)], [A*E*(-0.0333333333333333*t31 + 0.133333333333333*t32), E*(-0.04*I22*t11*t31 + 0.16*I22*t11*t32 + 0.04*I22*t12*t31 - 0.16*I22*t12*t32 - 0.04*I33*t11*t31 + 0.16*I33*t11*t32 + 0.04*I33*t12*t31 - 0.16*I33*t12*t32)/L, A*E*L*(-0.00444444444444444*t21*t31 + 0.0177777777777778*t21*t32 + 0.00111111111111111*t22*t31 - 0.00444444444444444*t22*t32), E*(-0.00222222222222222*A*L**2*t21**2 + 0.00111111111111111*A*L**2*t21*t22 - 0.00222222222222222*A*L**2*t22**2 - 0.00666666666666667*A*L**2*t31**2 + 0.02*A*L**2*t31*t32 - 0.00666666666666667*A*L**2*t32**2 - 0.0333333333333333*A*L*u - 0.02*I22*t11**2 + 0.04*I22*t11*t12 - 0.02*I22*t12**2 + 2.0*I22 - 0.02*I33*t11**2 + 0.04*I33*t11*t12 - 0.02*I33*t12**2)/L, E*(0.04*I22*t11*t31 - 0.16*I22*t11*t32 - 0.04*I22*t12*t31 + 0.16*I22*t12*t32 + 0.04*I33*t11*t31 - 0.16*I33*t11*t32 - 0.04*I33*t12*t31 + 0.16*I33*t12*t32)/L, A*E*L*(0.00111111111111111*t21*t31 - 0.00444444444444444*t21*t32 - 0.00444444444444444*t22*t31 + 0.0177777777777778*t22*t32), E*(0.00888888888888889*A*L**2*t21**2 - 0.00444444444444444*A*L**2*t21*t22 + 0.00888888888888889*A*L**2*t22**2 + 0.01*A*L**2*t31**2 - 0.0133333333333333*A*L**2*t31*t32 + 0.0266666666666667*A*L**2*t32**2 + 0.133333333333333*A*L*u + 0.08*I22*t11**2 - 0.16*I22*t11*t12 + 0.08*I22*t12**2 + 4.0*I22 + 0.08*I33*t11**2 - 0.16*I33*t11*t12 + 0.08*I33*t12**2)/L]])
        self.__local_stiffness_matrix = kl
    
    @property
    def local_stiffness_matrix(self):
        return self.__local_stiffness_matrix
    
    def update_auxiliary_vector(self):
        q_1 = self.current_orientation_node_1 @ self.initial_local_frame @ np.array([
                                                                                    [0., 1., 0.]]).T
        q_2 = self.current_orientation_node_2 @ self.initial_local_frame @ np.array([
                                                                                    [0., 1., 0.]]).T
        q = (q_1 + q_2) / 2

        self.__q1, self.__q2, self.__q = q_1, q_2, q

    @property
    def auxiliary_vector(self):
        return self.__q1, self.__q2, self.__q
    
    @property
    def current_local_frame(self):
        r_1 = (self.current_coordinate_node_2 -
               self.current_coordinate_node_1) / self.current_length

        self.update_auxiliary_vector()
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
        u, t11, t21, t31, t12, t22, t32 = self.local_displacement.reshape(7)
        E, G, A, L, I22, I33, Irr = self.__youngs_modulus, self.__shear_modulus, self.area, self.initial_length, self.moment_of_inertia_z, self.moment_of_inertia_y, self.fourth_order_polar_moment_of_inertia
        
        fl = np.array([[E*(0.0666666666666667*A*L**2*t21**2 - 0.0333333333333333*A*L**2*t21*t22 + 0.0666666666666667*A*L**2*t22**2 + 0.0666666666666667*A*L**2*t31**2 - 0.0333333333333333*A*L**2*t31*t32 + 0.0666666666666667*A*L**2*t32**2 + 1.0*A*L*u + 0.6*I22*t11**2 - 1.2*I22*t11*t12 + 0.6*I22*t12**2 + 0.6*I33*t11**2 - 1.2*I33*t11*t12 + 0.6*I33*t12**2)/L**2, 0.08*E*I22*t11*t21**2/L - 0.04*E*I22*t11*t21*t22/L + 0.08*E*I22*t11*t22**2/L + 0.08*E*I22*t11*t31**2/L - 0.04*E*I22*t11*t31*t32/L + 0.08*E*I22*t11*t32**2/L - 0.08*E*I22*t12*t21**2/L + 0.04*E*I22*t12*t21*t22/L - 0.08*E*I22*t12*t22**2/L - 0.08*E*I22*t12*t31**2/L + 0.04*E*I22*t12*t31*t32/L - 0.08*E*I22*t12*t32**2/L + 1.2*E*I22*t11*u/L**2 - 1.2*E*I22*t12*u/L**2 + 0.08*E*I33*t11*t21**2/L - 0.04*E*I33*t11*t21*t22/L + 0.08*E*I33*t11*t22**2/L + 0.08*E*I33*t11*t31**2/L - 0.04*E*I33*t11*t31*t32/L + 0.08*E*I33*t11*t32**2/L - 0.08*E*I33*t12*t21**2/L + 0.04*E*I33*t12*t21*t22/L - 0.08*E*I33*t12*t22**2/L - 0.08*E*I33*t12*t31**2/L + 0.04*E*I33*t12*t31*t32/L - 0.08*E*I33*t12*t32**2/L + 1.2*E*I33*t11*u/L**2 - 1.2*E*I33*t12*u/L**2 + 1.02857142857155*E*Irr*t11**3/L**3 - 3.08571428571469*E*Irr*t11**2*t12/L**3 + 3.08571428571469*E*Irr*t11*t12**2/L**3 - 1.02857142857155*E*Irr*t12**3/L**3 + 1.2*G*I22*t11/L - 1.2*G*I22*t12/L + 1.2*G*I33*t11/L - 1.2*G*I33*t12/L - 0.308571428571553*E*I22**2*t11**3/(A*L**3) + 0.925714285714687*E*I22**2*t11**2*t12/(A*L**3) - 0.925714285714687*E*I22**2*t11*t12**2/(A*L**3) + 0.308571428571553*E*I22**2*t12**3/(A*L**3) - 0.617142857143106*E*I22*I33*t11**3/(A*L**3) + 1.85142857142937*E*I22*I33*t11**2*t12/(A*L**3) - 1.85142857142937*E*I22*I33*t11*t12**2/(A*L**3) + 0.617142857143106*E*I22*I33*t12**3/(A*L**3) - 0.308571428571553*E*I33**2*t11**3/(A*L**3) + 0.925714285714687*E*I33**2*t11**2*t12/(A*L**3) - 0.925714285714687*E*I33**2*t11*t12**2/(A*L**3) + 0.308571428571553*E*I33**2*t12**3/(A*L**3), 1.0*E*(0.00888888888888889*A*L**2*t21**3 - 0.00666666666666667*A*L**2*t21**2*t22 + 0.01*A*L**2*t21*t22**2 + 0.00888888888888889*A*L**2*t21*t31**2 - 0.00444444444444444*A*L**2*t21*t31*t32 + 0.00888888888888889*A*L**2*t21*t32**2 - 0.00222222222222222*A*L**2*t22**3 - 0.00222222222222222*A*L**2*t22*t31**2 + 0.00111111111111111*A*L**2*t22*t31*t32 - 0.00222222222222222*A*L**2*t22*t32**2 + 0.133333333333333*A*L*t21*u - 0.0333333333333333*A*L*t22*u + 0.08*I22*t11**2*t21 - 0.02*I22*t11**2*t22 - 0.16*I22*t11*t12*t21 + 0.04*I22*t11*t12*t22 + 0.08*I22*t12**2*t21 - 0.02*I22*t12**2*t22 + 0.08*I33*t11**2*t21 - 0.02*I33*t11**2*t22 - 0.16*I33*t11*t12*t21 + 0.04*I33*t11*t12*t22 + 0.08*I33*t12**2*t21 - 0.02*I33*t12**2*t22 + 4.0*I33*t21 + 2.0*I33*t22)/L, 1.0*E*(0.00888888888888889*A*L**2*t21**2*t31 - 0.00222222222222222*A*L**2*t21**2*t32 - 0.00444444444444444*A*L**2*t21*t22*t31 + 0.00111111111111111*A*L**2*t21*t22*t32 + 0.00888888888888889*A*L**2*t22**2*t31 - 0.00222222222222222*A*L**2*t22**2*t32 + 0.00888888888888889*A*L**2*t31**3 - 0.00666666666666667*A*L**2*t31**2*t32 + 0.01*A*L**2*t31*t32**2 - 0.00222222222222222*A*L**2*t32**3 + 0.133333333333333*A*L*t31*u - 0.0333333333333333*A*L*t32*u + 0.08*I22*t11**2*t31 - 0.02*I22*t11**2*t32 - 0.16*I22*t11*t12*t31 + 0.04*I22*t11*t12*t32 + 0.08*I22*t12**2*t31 - 0.02*I22*t12**2*t32 + 4.0*I22*t31 + 2.0*I22*t32 + 0.08*I33*t11**2*t31 - 0.02*I33*t11**2*t32 - 0.16*I33*t11*t12*t31 + 0.04*I33*t11*t12*t32 + 0.08*I33*t12**2*t31 - 0.02*I33*t12**2*t32)/L, -0.08*E*I22*t11*t21**2/L + 0.04*E*I22*t11*t21*t22/L - 0.08*E*I22*t11*t22**2/L - 0.08*E*I22*t11*t31**2/L + 0.04*E*I22*t11*t31*t32/L - 0.08*E*I22*t11*t32**2/L + 0.08*E*I22*t12*t21**2/L - 0.04*E*I22*t12*t21*t22/L + 0.08*E*I22*t12*t22**2/L + 0.08*E*I22*t12*t31**2/L - 0.04*E*I22*t12*t31*t32/L + 0.08*E*I22*t12*t32**2/L - 1.2*E*I22*t11*u/L**2 + 1.2*E*I22*t12*u/L**2 - 0.08*E*I33*t11*t21**2/L + 0.04*E*I33*t11*t21*t22/L - 0.08*E*I33*t11*t22**2/L - 0.08*E*I33*t11*t31**2/L + 0.04*E*I33*t11*t31*t32/L - 0.08*E*I33*t11*t32**2/L + 0.08*E*I33*t12*t21**2/L - 0.04*E*I33*t12*t21*t22/L + 0.08*E*I33*t12*t22**2/L + 0.08*E*I33*t12*t31**2/L - 0.04*E*I33*t12*t31*t32/L + 0.08*E*I33*t12*t32**2/L - 1.2*E*I33*t11*u/L**2 + 1.2*E*I33*t12*u/L**2 - 1.02857142857155*E*Irr*t11**3/L**3 + 3.08571428571469*E*Irr*t11**2*t12/L**3 - 3.08571428571469*E*Irr*t11*t12**2/L**3 + 1.02857142857155*E*Irr*t12**3/L**3 - 1.2*G*I22*t11/L + 1.2*G*I22*t12/L - 1.2*G*I33*t11/L + 1.2*G*I33*t12/L + 0.308571428571553*E*I22**2*t11**3/(A*L**3) - 0.925714285714687*E*I22**2*t11**2*t12/(A*L**3) + 0.925714285714687*E*I22**2*t11*t12**2/(A*L**3) - 0.308571428571553*E*I22**2*t12**3/(A*L**3) + 0.617142857143106*E*I22*I33*t11**3/(A*L**3) - 1.85142857142937*E*I22*I33*t11**2*t12/(A*L**3) + 1.85142857142937*E*I22*I33*t11*t12**2/(A*L**3) - 0.617142857143106*E*I22*I33*t12**3/(A*L**3) + 0.308571428571553*E*I33**2*t11**3/(A*L**3) - 0.925714285714687*E*I33**2*t11**2*t12/(A*L**3) + 0.925714285714687*E*I33**2*t11*t12**2/(A*L**3) - 0.308571428571553*E*I33**2*t12**3/(A*L**3), E*(-0.00222222222222222*A*L**2*t21**3 + 0.01*A*L**2*t21**2*t22 - 0.00666666666666667*A*L**2*t21*t22**2 - 0.00222222222222222*A*L**2*t21*t31**2 + 0.00111111111111111*A*L**2*t21*t31*t32 - 0.00222222222222222*A*L**2*t21*t32**2 + 0.00888888888888889*A*L**2*t22**3 + 0.00888888888888889*A*L**2*t22*t31**2 - 0.00444444444444444*A*L**2*t22*t31*t32 + 0.00888888888888889*A*L**2*t22*t32**2 - 0.0333333333333333*A*L*t21*u + 0.133333333333333*A*L*t22*u - 0.02*I22*t11**2*t21 + 0.08*I22*t11**2*t22 + 0.04*I22*t11*t12*t21 - 0.16*I22*t11*t12*t22 - 0.02*I22*t12**2*t21 + 0.08*I22*t12**2*t22 - 0.02*I33*t11**2*t21 + 0.08*I33*t11**2*t22 + 0.04*I33*t11*t12*t21 - 0.16*I33*t11*t12*t22 - 0.02*I33*t12**2*t21 + 0.08*I33*t12**2*t22 + 2.0*I33*t21 + 4.0*I33*t22)/L, E*(-0.00222222222222222*A*L**2*t21**2*t31 + 0.00888888888888889*A*L**2*t21**2*t32 + 0.00111111111111111*A*L**2*t21*t22*t31 - 0.00444444444444444*A*L**2*t21*t22*t32 - 0.00222222222222222*A*L**2*t22**2*t31 + 0.00888888888888889*A*L**2*t22**2*t32 - 0.00222222222222222*A*L**2*t31**3 + 0.01*A*L**2*t31**2*t32 - 0.00666666666666667*A*L**2*t31*t32**2 + 0.00888888888888889*A*L**2*t32**3 - 0.0333333333333333*A*L*t31*u + 0.133333333333333*A*L*t32*u - 0.02*I22*t11**2*t31 + 0.08*I22*t11**2*t32 + 0.04*I22*t11*t12*t31 - 0.16*I22*t11*t12*t32 - 0.02*I22*t12**2*t31 + 0.08*I22*t12**2*t32 + 2.0*I22*t31 + 4.0*I22*t32 - 0.02*I33*t11**2*t31 + 0.08*I33*t11**2*t32 + 0.04*I33*t11*t12*t31 - 0.16*I33*t11*t12*t32 - 0.02*I33*t12**2*t31 + 0.08*I33*t12**2*t32)/L]]).T
        
        return fl

    def strain(self, x, y, z):
        u, t11, t21, t31, t12, t22, t32 = self.local_displacement.reshape(7)
        L = self.initial_length
        
        # Hermitian polynomials and their derivatives
        f1 = lambda x: 1 - 3 * (x/L) ** 2 + 2 * (x/L) ** 3
        f2 = lambda x: x * (1 - x/L) ** 2
        f3 = lambda x: 1 - f1(x)
        f4 = lambda x: (x ** 2) * (x/L - 1) / L
        f5 = lambda x: x/L

        df1 = lambda x: -6*x/L**2 + 6*x**2/L**3
        df2 = lambda x: (1 - x/L)**2 - 2*x*(1 - x/L)/L
        df3 = lambda x: 6*x/L**2 - 6*x**2/L**3
        df4 = lambda x: 2*x*(-1 + x/L)/L + x**2/L**2
        df5 = lambda x: 1/L

        ddf1 = lambda x: -6/L**2 + 12*x/L**3
        ddf2 = lambda x: -4*(1 - x/L)/L + 2*x/L**2
        ddf3 = lambda x: 6/L**2 - 12*x/L**3
        ddf4 = lambda x: 2*(-1 + x/L)/L + 4*x/L**2
        ddf5 = lambda x: 0

        u1 = f5(x) * u
        u2 = f2(x) * t31 + f4(x) * t32
        u3 = -f2(x) * t21 - f4(x) * t22

        du1 = df5(x) * u
        du2 = df2(x) * t31 + df4(x) * t32
        du3 = -df2(x) * t21 - df4(x) * t22

        ddu1 = ddf5(x) * u
        ddu2 = ddf2(x) * t31 + ddf4(x) * t32
        ddu3 = -ddf2(x) * t21 - ddf4(x) * t22

        t1 = f1(x) * t11 + f3(x) * t12
        t2 = -du3 + du2 * t1 / 2 + du1 * du3
        t3 = du2 + du3 * t1 / 2 - du1 * du2

        dt1 = df1(x) * t11 + df3(x) * t12
        dt2 = -ddu3 + (ddu2 * t1 + du2 * dt1) / 2 + ddu1 * du3 + du1 * ddu3
        dt3 = ddu2 + (ddu3 * t1 + du3 * dt1) / 2 - ddu1 * du2 - du1 * ddu2

        eps_11 = du1 + (du2 ** 2 + du3 ** 2) / 2 - y * dt3 + \
            z * dt2 + (y ** 2 + z ** 2) * dt1 ** 2 / 2
        eps_22 = -self.first_lame_constant * eps_11 / (2 * (self.first_lame_constant + self.second_lame_constant))
        eps_33 = eps_22
        gamma_12 = du2 - t3 - z * dt1
        gamma_13 = du3 + t2 + y * dt1

        return np.array([[eps_11, eps_22, eps_33, gamma_12, 0., gamma_13]]).T

    def map_local_coordinate_to_global_coordinate(self, xi, eta, mu):
        x = self.initial_length / 2 * (xi + 1)
        y = self.__width / 2 * eta
        z = self.__height / 2 * mu

        return x, y, z

    def perform_linear_hardening(self):
        gauss_locations_xi = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_xi)[0]
        gauss_locations_eta = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_eta)[0]
        gauss_locations_mu = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_mu)[0]

        yield_stress_function = lambda alpha: self.__yield_stress + self.__plastic_modulus * alpha

        equivalent_stress = lambda stress: np.sqrt(stress[0, 0] ** 2 + 3 * stress[1, 0] ** 2 + 3 * stress[2, 0] ** 2)
        
        stress_norm = lambda stress: np.sqrt(stress[0, 0] ** 2 + 
                                             stress[1, 0] ** 2 + 
                                             stress[2, 0] ** 2 + 
                                             2 * stress[3, 0] ** 2 + 
                                             2 * stress[4, 0] ** 2 +
                                             2 * stress[5, 0] ** 2)
        
        a = lambda stress: np.array([[stress[0, 0], 3 * stress[1, 0], 3 * stress[2, 0]]]).T / equivalent_stress(stress)
        
        da = lambda stress: 3 * np.array([[stress[1, 0] ** 2 + stress[2, 0] ** 2, -stress[0, 0] * stress[1, 0], -stress[0, 0] * stress[2, 0]],
                                      [-stress[0, 0] * stress[1, 0], stress[0, 0] ** 2 + 3 * stress[2, 0] ** 2, -3 * stress[1, 0] * stress[2, 0]],
                                      [-stress[0, 0] * stress[2, 0], -3 * stress[1, 0] * stress[2, 0], stress[0, 0] ** 2 + 3 * stress[1, 0] ** 2]]) / \
                                          equivalent_stress(stress) ** 3
        
        for ixi in range(self.__num_of_gauss_locations_xi):
            for jeta in range(self.__num_of_gauss_locations_eta):
                for kmu in range(self.__num_of_gauss_locations_mu):
                    x, y, z = self.map_local_coordinate_to_global_coordinate(
                        gauss_locations_xi[ixi], gauss_locations_eta[jeta], gauss_locations_mu[kmu])
                    constitutive_matrix = self.first_lame_constant * util.unit_tensor() @ util.unit_tensor().T + 2 * self.second_lame_constant * np.eye(6)

                    stress_trial = constitutive_matrix @ \
                        (self.strain(x, y, z) - \
                        self.__plastic_strain[:, :, ixi, jeta, kmu])

                    stress_trace = stress_trial[0, 0] + stress_trial[1, 0] + stress_trial[2, 0]
                    shifted_stress = stress_trial - 1/3 * stress_trace * util.unit_tensor()
                    
                    # relative_stress_trial = stress_trial - self.__back_stress[ixi, jeta]
                    yield_condition_trial = stress_norm(
                        shifted_stress) - np.sqrt(2/3) * yield_stress_function(self.__internal_hardening_variable[ixi, jeta, kmu])

                    if yield_condition_trial <= 0:
                        self.__stress[:, :, ixi, jeta, kmu] = stress_trial
                        self.__tangent_moduli[:, :, ixi, jeta, kmu] = constitutive_matrix
                    else:
                        deltagamma = yield_condition_trial / (2 * self.second_lame_constant + 2/3 * self.__plastic_modulus)
                        
                        unit_deviatoric_tensor = shifted_stress / stress_norm(shifted_stress)
                        
                        stress = stress_trial - 2 * self.second_lame_constant * deltagamma * unit_deviatoric_tensor
                        self.__stress[:, :, ixi, jeta, kmu] = stress
                            
                        self.__plastic_strain[:, :, ixi, jeta, kmu] += deltagamma * unit_deviatoric_tensor
                        
                        # self.__back_stress += deltagamma * \
                        #     self.__kinematic_hardening_modulus * \
                        #     a(relative_stress_trial)
                            
                        self.__internal_hardening_variable[ixi, jeta, kmu] += np.sqrt(2/3) * deltagamma
                        
                        c1 = 4 * self.second_lame_constant ** 2 / (2 * self.second_lame_constant + 2/3 * self.__plastic_modulus)
                        c2 = 4 * self.second_lame_constant ** 2 * deltagamma / stress_norm(shifted_stress)
                        
                        tangent_moduli = constitutive_matrix - (c1 - c2) * unit_deviatoric_tensor @ unit_deviatoric_tensor.T - c2 * util.I_dev()
                        
                        self.__tangent_moduli[:, :, ixi, jeta, kmu] =  tangent_moduli

    def L_matrix(self):
        gauss_locations_xi = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_xi)[0]
        gauss_locations_eta = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_eta)[0]
        gauss_locations_mu = np.polynomial.legendre.leggauss(
            self.__num_of_gauss_locations_mu)[0]
        L = np.zeros((6, 6, self.__num_of_gauss_locations_xi, self.__num_of_gauss_locations_eta, self.__num_of_gauss_locations_mu), dtype=float)
        for ixi in range(self.__num_of_gauss_locations_xi):
            for jeta in range(self.__num_of_gauss_locations_eta):
                for kmu in range(self.__num_of_gauss_locations_mu):
                    x, y, z = self.map_local_coordinate_to_global_coordinate(
                        gauss_locations_xi[ixi], gauss_locations_eta[jeta], gauss_locations_mu[kmu])
                    L[3, 3, ixi, jeta, kmu] = (y ** 2 + z ** 2) * self.__stress[0, 0, ixi, jeta, kmu]
        
        return L
    
    def A_matrix(self, x, y, z):
        t12, t11 = self.local_displacement[4, 0], self.local_displacement[1, 0]
        L = self.initial_length
        
        df1 = lambda x: -6*x/L**2 + 6*x**2/L**3
        df3 = lambda x: 6*x/L**2 - 6*x**2/L**3
        
        dt1 = lambda x: df1(x) * t11 + df3(x) * t12
        A = np.array([[1., 0., 0., (y ** 2 + z ** 2) * dt1(x), z, -y],
                      [0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0.],
                      [0., -1., 0., -z, 0., 0.],
                      [0., 0., 0., 0., 0., 0.],
                      [0., 0., 1., y, 0., 0.]])
        A[1, :] = -self.first_lame_constant / (2 * (self.first_lame_constant + self.second_lame_constant)) * A[0, :]
        A[2, :] = A[1, :]
        return A
    
    def G_mat(self, x):
        u, t11, t21, t31, t12, t22, t32 = self.local_displacement.reshape(7)
        L = self.initial_length
        
        f1 = lambda x: 1 - 3 * (x/L) ** 2 + 2 * (x/L) ** 3
        f2 = lambda x: x * (1 - x/L) ** 2
        f3 = lambda x: 1 - f1(x)
        f4 = lambda x: (x ** 2) * (x/L - 1) / L
        f5 = lambda x: x/L

        df1 = lambda x: -6*x/L**2 + 6*x**2/L**3
        df2 = lambda x: (1 - x/L)**2 - 2*x*(1 - x/L)/L
        df3 = lambda x: 6*x/L**2 - 6*x**2/L**3
        df4 = lambda x: 2*x*(-1 + x/L)/L + x**2/L**2
        df5 = lambda x: 1/L

        ddf1 = lambda x: -6/L**2 + 12*x/L**3
        ddf2 = lambda x: -4*(1 - x/L)/L + 2*x/L**2
        ddf3 = lambda x: 6/L**2 - 12*x/L**3
        ddf4 = lambda x: 2*(-1 + x/L)/L + 4*x/L**2
        ddf5 = lambda x: 0
        
        pdu1 = lambda x: np.array([[df5(x), 0, 0, 0, 0, 0, 0]])
        pt3 = lambda x: np.array([[-df5(x)*(df2(x)*t31 + df4(x)*t32), f1(x)*(-0.5*df2(x)*t21 - 0.5*df4(x)*t22), -0.5*df2(x)*(f1(x)*t11 + f3(x)*t12), -df2(x)*df5(x)*u + df2(x), f3(x)*(-0.5*df2(x)*t21 - 0.5*df4(x)*t22), -0.5*df4(x)*(f1(x)*t11 + f3(x)*t12), -df4(x)*df5(x)*u + df4(x)]])
        pt2 = lambda x: np.array([[df5(x)*(-df2(x)*t21 - df4(x)*t22), f1(x)*(0.5*df2(x)*t31 + 0.5*df4(x)*t32), -df2(x)*df5(x)*u + df2(x), 0.5*df2(x)*(f1(x)*t11 + f3(x)*t12), f3(x)*(0.5*df2(x)*t31 + 0.5*df4(x)*t32), -df4(x)*df5(x)*u + df4(x), 0.5*df4(x)*(f1(x)*t11 + f3(x)*t12)]])
        pdt1 = lambda x: np.array([[0, df1(x), 0, 0, df3(x), 0, 0]])
        pdt2 = lambda x: np.array([[ddf5(x)*(-df2(x)*t21 - df4(x)*t22) + df5(x)*(-ddf2(x)*t21 - ddf4(x)*t22), f1(x)*(0.5*ddf2(x)*t31 + 0.5*ddf4(x)*t32), -ddf2(x)*df5(x)*u + ddf2(x) - ddf5(x)*df2(x)*u, 0.5*ddf2(x)*(f1(x)*t11 + f3(x)*t12), f3(x)*(0.5*ddf2(x)*t31 + 0.5*ddf4(x)*t32), -ddf4(x)*df5(x)*u + ddf4(x) - ddf5(x)*df4(x)*u, 0.5*ddf4(x)*(f1(x)*t11 + f3(x)*t12)]])
        pdt3 = lambda x: np.array([[-ddf5(x)*(df2(x)*t31 + df4(x)*t32) - df5(x)*(ddf2(x)*t31 + ddf4(x)*t32), f1(x)*(-0.5*ddf2(x)*t21 - 0.5*ddf4(x)*t22), -0.5*ddf2(x)*(f1(x)*t11 + f3(x)*t12), -ddf2(x)*df5(x)*u + ddf2(x) - ddf5(x)*df2(x)*u, f3(x)*(-0.5*ddf2(x)*t21 - 0.5*ddf4(x)*t22), -0.5*ddf4(x)*(f1(x)*t11 + f3(x)*t12), -ddf4(x)*df5(x)*u + ddf4(x) - ddf5(x)*df4(x)*u]])

        G = np.r_[pdu1(x), pt3(x), pt2(x), pdt1(x), pdt2(x), pdt3(x)]
        
        return G
    
    @property    
    def elasto_plastic_local_force(self):
        [gauss_locations_xi, weights_xi] = np.polynomial.legendre.leggauss(
                self.__num_of_gauss_locations_xi)
        [gauss_locations_eta, weights_eta] = np.polynomial.legendre.leggauss(
                self.__num_of_gauss_locations_eta)
        [gauss_locations_mu, weights_mu] = np.polynomial.legendre.leggauss(
                self.__num_of_gauss_locations_mu)
        
        f_l = np.zeros((7, 1), dtype=float)
        
        for ixi in range(self.__num_of_gauss_locations_xi):
            for jeta in range(self.__num_of_gauss_locations_eta):
                for kmu in range(self.__num_of_gauss_locations_mu):
                    x, y, z = self.map_local_coordinate_to_global_coordinate(
                        gauss_locations_xi[ixi], gauss_locations_eta[jeta], gauss_locations_mu[kmu])
                    fac = self.initial_length * self.__width * self.__height / 8
                    wt = weights_xi[ixi] * weights_eta[jeta] * weights_mu[kmu]
                    f_l += self.G_mat(x).T @ self.A_matrix(x, y, z).T @ self.__stress[:, :, ixi, jeta, kmu] * fac * wt
        
        return f_l
     
    @property      
    def elasto_plastic_local_stiffness(self):
        [gauss_locations_xi, weights_xi] = np.polynomial.legendre.leggauss(
                self.__num_of_gauss_locations_xi)
        [gauss_locations_eta, weights_eta] = np.polynomial.legendre.leggauss(
                self.__num_of_gauss_locations_eta)
        [gauss_locations_mu, weights_mu] = np.polynomial.legendre.leggauss(
                self.__num_of_gauss_locations_mu)
        
        k_l = np.zeros((7, 7), dtype=float)
        
        for ixi in range(self.__num_of_gauss_locations_xi):
            for jeta in range(self.__num_of_gauss_locations_eta):
                for kmu in range(self.__num_of_gauss_locations_mu):
                    x, y, z = self.map_local_coordinate_to_global_coordinate(
                        gauss_locations_xi[ixi], gauss_locations_eta[jeta], gauss_locations_mu[kmu])
                    fac = self.initial_length * self.__width * self.__height / 8
                    wt = weights_xi[ixi] * weights_eta[jeta] * weights_mu[kmu]
                    k_l += self.G_mat(x).T @ (self.A_matrix(x, y, z).T @ self.__tangent_moduli[:, :, ixi, jeta, kmu] @ self.A_matrix(x, y, z) + self.L_matrix()[:, :, ixi, jeta, kmu]) @ self.G_mat(x) * fac * wt
        
        return k_l
    
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
        if self.analysis == "elastic":
            return self.Ba_matrix.T @ self.local_force
        else:
            return self.Ba_matrix.T @ self.elasto_plastic_local_force

    def Khi_matrix(self, i):
        if i == 1:
            theta_bar = self.local_displacement[1: 4]
            if self.analysis == "elastic":
                v = self.local_force[1: 4]
            else:
                v = self.elasto_plastic_local_force[1: 4]
        elif i == 2:
            theta_bar = self.local_displacement[4: 7]
            if self.analysis == "elastic":
                v = self.local_force[4: 7]
            else:
                v = self.elasto_plastic_local_force[4: 7]
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
        K_h[1: 4, 1: 4] = self.Khi_matrix(1)
        K_h[4: 7, 4: 7] = self.Khi_matrix(2)

        return K_h

    @property
    def Ka_matrix(self):
        if self.__analysis == "elastic":
            return self.Ba_matrix.T @ self.local_stiffness_matrix @ self.Ba_matrix + self.Kh_matrix
        else:
            return self.Ba_matrix.T @ self.elasto_plastic_local_stiffness @ self.Ba_matrix + self.Kh_matrix

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
        r = np.c_[-r_1.T, np.array([[0., 0., 0.]]),
                  r_1.T, np.array([[0., 0., 0.]])]
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

        f_a = self.local_force_a
        _, _, q = self.auxiliary_vector
        q1 = float((self.current_local_frame.T @ q)[0, 0])
        q2 = float((self.current_local_frame.T @ q)[1, 0])
        eta = q1 / q2

        a = np.zeros((3, 1), dtype=float)
        a[1] = eta/self.current_length * (f_a[1] + f_a[4]) - \
            1/self.current_length * \
            (f_a[2] + f_a[5])
        a[2] = 1/self.current_length * \
            (f_a[3] + f_a[6])

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

        return self.Bg_matrix.T @ self.Ka_matrix @ self.Bg_matrix + self.Km_matrix
