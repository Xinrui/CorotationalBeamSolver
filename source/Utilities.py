# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou
"""

import numpy as np
from numpy import cos, sin, tan


def load_mesh_file(filename):
    """Load GiD .msh file.

    Parameters
    ----------
    filename: str
        the name of the GiD mesh file in meshfile folder without extension.

    Returns
    -------
    array_nodes: 
        an array of coordinates of all nodes. 
    array_elements: 
        an array of numbers of nodes of all elements.
    number_of_nodes: 
        the number of nodes.
    number_of_elements: 
        the number of elements.

    """
    with open("meshfile\\" + filename + ".msh") as file:

        lines = file.readlines()

        for i, line in enumerate(lines):
            if line.strip() == "Coordinates":
                index_node_start = i
            if line.strip() == "End Coordinates":
                index_node_end = i
            if line.strip() == "Elements":
                index_element_start = i
            if line.strip() == "End Elements":
                index_element_end = i

        number_of_elements = index_element_end - index_element_start - 1
        number_of_nodes = index_node_end - index_node_start - 1

        array_nodes = np.empty((3, number_of_nodes))

        for i in range(index_node_start + 1, index_node_end):
            line = lines[i].split()
            for j in range(3):
                array_nodes[j, i - index_node_start - 1] = line[j + 1]

        array_elements = np.empty((2, number_of_elements), dtype=int)

        for i in range(index_element_start + 1, index_element_end):
            line = lines[i].split()
            for j in range(2):
                array_elements[j, i - index_element_start - 1] = line[j + 1]

    return array_nodes, array_elements, number_of_nodes, number_of_elements


def get_eta_and_mu(alpha):
    """Get the value of eta and mu. See (4.46) of the PhD thesis of J.-M. Battini.

    Parameters
    ----------
    alpha: float
        the angle of the rotation.

    Returns
    -------
    The first coefficient eta: float.
    The second coefficient mu: float.

    """
    if alpha == 0.:
        eta = 1 / 12
        mu = 1 / 360
    else:
        eta = (2 * sin(alpha) - alpha * (1 + cos(alpha))) / \
            (2 * alpha ** 2 * sin(alpha))
        mu = (alpha * (alpha + sin(alpha)) - 8 * sin(alpha / 2) ** 2) / \
            (4 * alpha ** 4 * sin(alpha / 2) ** 2)
    return eta, mu


def get_skew_symmetric(rotational_vector):
    """Get the skew symmetric matrix of a rotational vector.

    Parameters
    ----------
    rotational_vector: numpy.ndarray
        the rotational vector.

    Returns
    -------
    skew_symmetric: 
        the skew symmetric matrix of the rotational vector.

    """
    skew_symmetric = np.array([[0.0, -rotational_vector[2], rotational_vector[1]],
                               [rotational_vector[2], 0.0, -rotational_vector[0]],
                               [-rotational_vector[1], rotational_vector[0], 0.0]], dtype=float)
    return skew_symmetric


def get_rotational_vector(skew_symmetric):
    """Get the rotational vector from a skew symmetric matrix.

    Parameters
    ----------
    skew_symmetric: numpy.ndarray
        the skew symmetric matrix.

    Returns
    -------
    rotational_vector: 
        the rotational vector.

    """
    # make sure that the input is skew symmetric
    if np.linalg.norm(skew_symmetric + skew_symmetric.T) > 1e-12:
        raise ValueError("The input is not skew symmetric!")

    rotational_vector = np.zeros((3, 1), dtype=float)
    rotational_vector[0] = skew_symmetric[2, 1]
    rotational_vector[1] = skew_symmetric[0, 2]
    rotational_vector[2] = skew_symmetric[1, 0]

    return rotational_vector


def decompose_rotational_vector(rotational_vector):
    """Get the angle of rotation and the skew symmetric matrix from a rotational vector.

    Parameters
    ----------
    rotational_vector: numpy.ndarray
        the rotational vector.

    Returns
    -------
    angle: 
        the angle of rotation.
    skew_symmetric: 
        the skew symmetric matrix.

    """
    angle = np.linalg.norm(rotational_vector)
    skew_symmetric = get_skew_symmetric(rotational_vector)
    return angle, skew_symmetric


def rodrigues(rotational_vector):
    """Get the exponent of a skew symmetric, which can be calculated from the rotational vector as input. 
    This function performs the famous Rodrigues' formula.

    Parameters
    ----------
    rotational_vector: np.ndarray
        the rotational vector.

    Returns
    -------
    rotation_matrix: 
        the rotation matrix.

    """
    angle, skew_symmetric = decompose_rotational_vector(rotational_vector)
    if angle == 0.0:
        rotation_matrix = np.eye(3)
    else:
        rotation_matrix = np.eye(3) + sin(angle) / angle * skew_symmetric + (1 -
                                                                             cos(angle)) / (angle ** 2) * skew_symmetric @ skew_symmetric
    return rotation_matrix


def log(rotation_matrix):
    """Perform the inverse operation of Rodrigues' formula: get the logarithm,
    which should be a skew symmetric matrix, of the input rotation matrix.

    Parameters
    ----------
    rotation_matrix: np.ndarray
        the rotation matrix.

    Returns
    -------
    skew_symmetric: 
        the skew symmetric, which can be furthur expressed as rotational vector.

    """
    cosalpha = (np.trace(rotation_matrix) - 1) / 2

    # in initial state, cos(alpha) may equal to 1.000000002 due to some numerical problems,
    # which is not acceptable for arccos function. The error is here filtered for this reason.
    if np.abs(cosalpha - 1.0) < 1e-8:
        cosalpha = 1.0
    if np.abs(cosalpha + 1.0) < 1e-8:
        cosalpha = -1.0

    alpha = np.arccos(cosalpha)

    if alpha == 0.0:
        skew_symmetric = 1/2 * (rotation_matrix - rotation_matrix.T)
    else:
        skew_symmetric = alpha / \
            (2 * sin(alpha)) * (rotation_matrix - rotation_matrix.T)

    return skew_symmetric


def get_det_transformation(rotational_vector):
    """Get the determinant of the transformation matrix. See (4.13) 
    of the PhD thesis of J.-M. Battini.

    Parameters
    ----------
    rotational_vector: np.ndarray
        the rotational vector.

    Returns
    -------
    det(T_s): 
        the determinant of the transformation matrix.

    """
    angle, skew_symmetric = decompose_rotational_vector(rotational_vector)
    return 2 * (1 - cos(angle)) / angle ** 2


def get_transformation(rotational_vector):
    """Get the transformation matrix. See (4.12) of the PhD thesis of J.-M. Battini.

    Parameters
    ----------
    rotational_vector: numpy.ndarray
        the rotational vector.

    Returns
    -------
    T_s: 
        the transformation matrix.

    """
    angle, skew_symmetric = decompose_rotational_vector(rotational_vector)
    if angle == 0.:
        T_s = np.eye(3)

    else:
        T_s = np.eye(3) + (1 - cos(angle)) / angle**2 * skew_symmetric + \
            (angle - sin(angle)) / angle**3 * skew_symmetric @ skew_symmetric

    return T_s


def get_transformation_inv(rotational_vector):
    """Get the inverse of the transformation matrix. See (4.15) of the PhD thesis of J.-M. Battini.

    Parameters
    ----------
    rotational_vector: numpy.ndarray
        the rotational vector.

    Returns
    -------
    inv(T_s): 
        the inverse of the transformation matrix.

    """
    angle, skew_symmetric = decompose_rotational_vector(rotational_vector)
    if angle == 0.:
        T_s_inv = np.eye(3)

    else:
        T_s_inv = ((angle / 2) / tan(angle / 2)) * np.eye(3) + (
            1 - (angle / 2) / tan(angle / 2)) * rotational_vector @ rotational_vector.T / angle ** 2 - 1 / 2 * skew_symmetric

    return T_s_inv


def decompose_strain(strain_vector):
    """
    Parameters
    ----------
    strain_vector: numpy.ndarray
        the strain vector.

    Returns
    -------
    volumetric_strain: float
        eps_v = eps_11 + eps_22 + eps_33
    deviatoric_strain: numpy.ndarray
        eps_d = eps - 1/3 * volumetric_strain * [1, 1, 1, 0, 0, 0].T

    """
    volumetric_strain = trace(strain_vector)

    isotropic_strain = 1/3 * volumetric_strain * unit_tensor()

    deviatoric_strain = strain_vector - isotropic_strain

    return volumetric_strain, deviatoric_strain


def stress_norm(stress_vector):
    """There are nine components in stress matrix, however, only 
    six of which are imdependent and can be written in vector form.
    If so, np.linalg.norm() does not work, this function must be adopted
    instead.

    Parameters
    ----------
    stress_vector: numpy.ndarray
        the stress vector.

    Returns
    -------
    norm: 
        the norm of the stress matrix.

    """
    return np.sqrt(stress_vector[0, 0] ** 2 +
                   stress_vector[1, 0] ** 2 +
                   stress_vector[2, 0] ** 2 +
                   2 * stress_vector[3, 0] ** 2 +
                   2 * stress_vector[4, 0] ** 2 +
                   2 * stress_vector[5, 0] ** 2)


def unit_tensor():
    """
    Returns
    -------
    unit tensor: 
        [1., 1., 1., 0., 0., 0.].T

    """
    return np.array([[1., 1., 1., 0., 0., 0.]]).T


def deviatoric_projection_tensor():
    """
    Returns
    -------
    deviatoric projection tensor: numpy.ndarray
        I - 1/3 * unit_tensor ⊗ unit_tensor.T

    """
    return np.eye(6) - 1/3 * unit_tensor() @ unit_tensor().T


def trace(vector):
    """np.trace() only work for square matrices. To get the trace of 
    stress/strain vectors, this function must be applied.

    Parameters
    ----------
    vector: numpy.ndarray
        the stress/strain vector.

    Returns
    -------
    trace: 
        the sum of diagonal components of the stress/strain matrix.

    """
    return float(vector[0, 0] + vector[1, 0] + vector[2, 0])
