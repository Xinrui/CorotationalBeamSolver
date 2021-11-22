# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou
"""

import numpy as np
from numpy import cos, sin, tan
from numpy.linalg import det

def load_mesh_file(filename):

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
    if alpha == 0.:
        eta = 1 / 12
        mu = 1 / 360
    else:
        eta = (2 * sin(alpha) - alpha * (1 + cos(alpha))) / \
            (2 * alpha ** 2 * sin(alpha))
        mu = (alpha * (alpha + sin(alpha)) - 8 * sin(alpha / 2) ** 2) / \
            (4 * alpha ** 4 * sin(alpha / 2) ** 2)
    return eta, mu


def get_skew_symmetric(vec_3d):
    skew_symmetric = np.array([[0.0, -vec_3d[2], vec_3d[1]],
                               [vec_3d[2], 0.0, -vec_3d[0]],
                               [-vec_3d[1], vec_3d[0], 0.0]], dtype=float)
    return skew_symmetric


def get_rotational_vector(skew_symmetric):
    if((abs(skew_symmetric + skew_symmetric.T) < 1e-12 * np.eye(3)).all()):
        raise TypeError("The input is not skew symmetric!")
    vec_3d = np.zeros((3, 1), dtype=float)
    vec_3d[0], vec_3d[1], vec_3d[2] = skew_symmetric[2, 1], skew_symmetric[0, 2], skew_symmetric[1, 0]

    return vec_3d


def decompose_rotational_vector(vec_3d):

    angle = np.linalg.norm(vec_3d)
    skew_symmetric = get_skew_symmetric(vec_3d)
    return angle, skew_symmetric


def rodrigues(vec_3d):
    angle, skew_symmetric = decompose_rotational_vector(vec_3d)
    if angle == 0.0:
        R = np.eye(3)
    else:
        R = np.eye(3) + sin(angle) / angle * skew_symmetric + (1 - cos(angle)) / (angle ** 2) * skew_symmetric @ skew_symmetric
    return R

def log(skew_symmetric):
    cosalpha = (np.trace(skew_symmetric) - 1) / 2
    
    if np.abs(cosalpha - 1.0) < 1e-3:
        cosalpha = 1.0
    if np.abs(cosalpha + 1.0) < 1e-3:
        cosalpha = -1.0 
    
    alpha = np.arccos(cosalpha)
    
    
    if alpha == 0.0:
        log_skew_symmetric = 1/2 * (skew_symmetric - skew_symmetric.T)
    else:
        log_skew_symmetric = alpha / (2 * sin(alpha)) * (skew_symmetric - skew_symmetric.T)
        
    return log_skew_symmetric

def get_det_transformation(vec_3d):
    angle, skew_symmetric = decompose_rotational_vector(vec_3d)
    return 2 * (1 - cos(angle)) / angle ** 2

def get_transformation(vec_3d):
    angle, skew_symmetric = decompose_rotational_vector(vec_3d)
    if angle == 0.:
        T_s = np.eye(3)

    else:
        T_s = np.eye(3) + (1 - cos(angle)) / angle**2 * skew_symmetric + (angle - sin(angle)) / angle**3 * skew_symmetric @ skew_symmetric

    return T_s

def get_transformation_inv(vec_3d):
    angle, skew_symmetric = decompose_rotational_vector(vec_3d)
    if angle == 0.:
        T_s_inv = np.eye(3)

    else:
        T_s_inv = ((angle / 2) / tan(angle / 2)) * np.eye(3) + (
            1 - (angle / 2) / tan(angle / 2)) * vec_3d @ vec_3d.T / angle ** 2 - 1 / 2 * skew_symmetric

    return T_s_inv

def decompose_strain(strain):
    
    volumetric_strain = trace(strain)
    
    isotropic_strain = 1/3 * volumetric_strain * unit_tensor()
    
    deviatoric_strain = strain - isotropic_strain
    
    return volumetric_strain, deviatoric_strain

def stress_norm(stress):
    return np.sqrt(stress[0, 0] ** 2 +
                   stress[1, 0] ** 2 +
                   stress[2, 0] ** 2 +
                   2 * stress[3, 0] ** 2 +
                   2 * stress[4, 0] ** 2 +
                   2 * stress[5, 0] ** 2)


def unit_tensor():
    return np.array([[1., 1., 1., 0., 0., 0.]]).T

def deviatoric_projection_tensor():
    return np.eye(6) - 1/3 * unit_tensor() @ unit_tensor().T

def trace(vec):
    return float(vec[0, 0] + vec[1, 0] + vec[2, 0])


# # def eight_node_serendipity_shape_function(xi, eta):
# #     return np.array([
# #         -1/4 * (1 - xi) * (1 - eta) * (1 + xi + eta),
# #         -1/4 * (1 + xi) * (1 - eta) * (1 - xi + eta),
# #         -1/4 * (1 + xi) * (1 + eta) * (1 - xi - eta),
# #         -1/4 * (1 - xi) * (1 + eta) * (1 + xi - eta),
# #         1/2 * (1 - xi ** 2) * (1 - eta),
# #         1/2 * (1 - eta ** 2) * (1 + xi),
# #         1/2 * (1 - xi ** 2) * (1 + eta),
# #         1/2 * (1 - eta ** 2) * (1 - xi)])


# # def eight_node_serendipity_shape_function_ders(xi, eta):
# #     dN_dxi = np.array([
# #         (1/4 - 1/4 * eta) * (eta + xi + 1) + (1 - eta) * (1/4 * xi - 1/4),
# #         -(1 - eta) * (-1/4 * xi - 1/4) + (1/4 * eta - 1/4) * (eta - xi + 1),
# #         (-1/4 * eta - 1/4) * (-eta - xi + 1) - (eta + 1) * (-1/4 * xi - 1/4),
# #         (1/4 * eta + 1/4) * (-eta + xi + 1) + (eta + 1) * (1/4 * xi - 1/4),
# #         -xi * (1 - eta),
# #         1/2 - 1/2 * eta ** 2,
# #         -xi * (eta + 1),
# #         1/2 * eta ** 2 - 1/2])

# #     dN_deta = np.array([
# #         (1/4 - 1/4 * xi) * (eta + xi + 1) + (1 - eta) * (1/4 * xi - 1/4),
# #         (1 - eta) * (-1/4 * xi - 1/4) + (1/4 * xi + 1/4) * (eta - xi + 1),
# #         -(eta + 1) * (-1/4 * xi - 1/4) + (-1/4 * xi - 1/4) * (-eta - xi + 1),
# #         -(eta + 1) * (1/4 * xi - 1/4) + (1/4 * xi - 1/4) * (-eta + xi + 1),
# #         1/2 * xi ** 2 - 1/2,
# #         -eta * (xi + 1),
# #         1/2 - 1/2 * xi ** 2,
# #         -eta * (1 - xi)])

# #     return dN_dxi, dN_deta


# def nodal_warping_function(width, height):

#     def N(xi, eta): return np.array([
#         -1/4 * (1 - xi) * (1 - eta) * (1 + xi + eta),
#         -1/4 * (1 + xi) * (1 - eta) * (1 - xi + eta),
#         -1/4 * (1 + xi) * (1 + eta) * (1 - xi - eta),
#         -1/4 * (1 - xi) * (1 + eta) * (1 + xi - eta),
#         1/2 * (1 - xi ** 2) * (1 - eta),
#         1/2 * (1 - eta ** 2) * (1 + xi),
#         1/2 * (1 - xi ** 2) * (1 + eta),
#         1/2 * (1 - eta ** 2) * (1 - xi)])

#     def dN_dxi(xi, eta): return np.array([
#         -1/4 * (-1 + eta) * (2 * xi + eta),
#         1/4 * (-1 + eta) * (eta - 2 * xi),
#         1/4 * (1 + xi) * (2 * eta + xi),
#         -1/4 * (1 + xi) * (eta - 2 * xi),
#         xi * (-1 + eta),
#         -1/2 * (1 + eta) * (-1 + eta),
#         -xi * (1 + eta),
#         1/2 * (1 + eta) * (-1 + eta)])

#     def dN_deta(xi, eta): return np.array([
#         -1/4 * (-1 + xi) * (xi + 2 * eta),
#         1/4 * (1 + xi) * (2 * eta - xi),
#         1/4 * (1 + xi) * (xi + 2 * eta),
#         -1/4 * (-1 + xi) * (2 * eta - xi),
#         1/2 * (1 + xi) * (-1 + xi),
#         -eta * (1 + xi),
#         -1/2 * (1 + xi) * (-1 + xi),
#         eta * (-1 + xi)])

#     [gauss_locations_xi, weights_xi] = np.polynomial.legendre.leggauss(4)
#     [gauss_locations_eta, weights_eta] = np.polynomial.legendre.leggauss(4)

#     # y_I = np.array([-width/2, width/2, width/2, -width/2,
#     #                 0., width/2, 0., -width/2])
#     # z_I = np.array([-height/2, -height/2, height/2, height/2,
#     #                 -height/2, 0., height/2, 0.])

#     # y_numerical = lambda xi, eta: np.dot(N(xi, eta), y_I)
#     # z_numerical = lambda xi, eta: np.dot(N(xi, eta), z_I)

#     def y(xi): return width/2 * xi
#     def z(eta): return height/2 * eta

#     # XI = np.linspace(-1, 1, 21)
#     # ETA = np.linspace(-1, 1, 21)
#     # XI_mesh, ETA_mesh = np.meshgrid(XI, ETA)

#     # Y_numerical = np.zeros((21, 21), dtype=float)
#     # Z_numerical = np.zeros((21, 21), dtype=float)
#     # for i, xi in enumerate(XI):
#     #     for j, eta in enumerate(ETA):
#     #         Y_numerical[i, j] = y_numerical(xi, eta)
#     #         Z_numerical[i, j] = z_numerical(xi, eta)

#     # fig = plt.figure()
#     # ax = Axes3D(fig, auto_add_to_figure=False)
#     # fig.add_axes(ax)
#     # ax.plot_surface(XI_mesh, ETA_mesh, Y_numerical, rstride=1, cstride=1, cmap=cm.viridis)
#     # ax.set_xlabel('$y$')
#     # ax.set_ylabel('$z$')

#     # plt.show()


# #     stiffness = np.zeros((8, 8), dtype=float)
# #     for i in range(8):
# #         for j in range(8):
# #             for ixi in range(4):
# #                 for jeta in range(4):
# #                     stiffness[i, j] += (height / width + width / height) * \
# #                     dN_dxi(gauss_locations_xi[ixi], gauss_locations_eta[jeta])[i] * \
# #                     dN_deta(gauss_locations_xi[ixi], gauss_locations_eta[jeta])[j] * \
# #                     weights_xi[ixi] * weights_eta[jeta]

# #     load = np.zeros((8, 1), dtype=float)
# #     for i in range(8):
# #         for ixi in range(4):
# #             for jeta in range(4):
# #                 load[i, 0] += (height/2 * dN_dxi(gauss_locations_xi[ixi], gauss_locations_eta[jeta])[i] * z(gauss_locations_eta[jeta]) - \
# #                             width/2 * dN_dxi(gauss_locations_xi[ixi], gauss_locations_eta[jeta])[i] * y(gauss_locations_xi[ixi])) * \
# #                             weights_xi[ixi] * weights_eta[jeta]

# #     arbitrary_dof = 2
# #     modified_stiffness = stiffness.copy()
# #     for ientry in range(8):
# #         modified_stiffness[arbitrary_dof, ientry] = 0.
# #         modified_stiffness[ientry, arbitrary_dof] = 0.
# #         modified_stiffness[arbitrary_dof, arbitrary_dof] = 1.

# #     modified_load = load.copy()
# #     modified_load[arbitrary_dof] = 0.

# #     nodal_warping_function = np.linalg.solve(modified_stiffness, modified_load)

# #     warping_function = lambda xi, eta: np.dot(N(xi, eta), nodal_warping_function.reshape(8))

# #     average_integral = 0.0
# #     for ixi in range(4):
# #         for jeta in range(4):
# #             average_integral += warping_function(gauss_locations_xi[ixi], gauss_locations_eta[jeta]) * \
# #             width * height / 4 * weights_xi[ixi] * weights_eta[jeta]
# #     average_integral /= width * height

# #     adjusted_nodal_warping_function = nodal_warping_function.copy()
# #     for i in range(8):
# #         adjusted_nodal_warping_function[i] -= average_integral

# #     return adjusted_nodal_warping_function


# # def warping_function(y, z, width, height, nodal_warping_vector):

# #     N = lambda xi, eta: np.array([
# #         -1/4 * (1 - xi) * (1 - eta) * (1 + xi + eta),
# #         -1/4 * (1 + xi) * (1 - eta) * (1 - xi + eta),
# #         -1/4 * (1 + xi) * (1 + eta) * (1 - xi - eta),
# #         -1/4 * (1 - xi) * (1 + eta) * (1 + xi - eta),
# #         1/2 * (1 - xi ** 2) * (1 - eta),
# #         1/2 * (1 - eta ** 2) * (1 + xi),
# #         1/2 * (1 - xi ** 2) * (1 + eta),
# #         1/2 * (1 - eta ** 2) * (1 - xi)])

# #     xi = 2 * y / width
# #     eta = 2 * z / height

# #     return np.dot(N(xi, eta), nodal_warping_vector)


# # d = nodal_warping_function(2., 2.)

# # Y = np.linspace(-1, 1, 21)
# # Z = np.linspace(-1, 1, 21)
# # Y_grid, Z_grid = np.meshgrid(Y, Z)
# # Omega_numerical = np.zeros((21, 21), dtype=float)
# # for i, y in enumerate(Y):
# #     for j, z in enumerate(Z):
# #         Omega_numerical[i, j] = warping_function(y, z, 2.0, 2.0, d)

# # fig = plt.figure()
# # ax = Axes3D(fig, auto_add_to_figure=False)
# # fig.add_axes(ax)
# # ax.plot_surface(Y_grid, Z_grid, Omega_numerical, rstride=1, cstride=1, cmap=cm.viridis)

# # plt.show()

# # integration = integrate.nquad(lambda y, z: warping_function(y, z, 2.0, 2.0, d), [[-1, 1],[-1, 1]])
# # print(integration)
