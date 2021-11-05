# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou
"""

from os import device_encoding
import numpy as np
from numpy import cos, sin, tan

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from scipy import integrate


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


def getEtaAndMu(alpha):
    if alpha == 0.:
        eta = 1 / 12
        mu = 1 / 360
    else:
        eta = (2 * sin(alpha) - alpha * (1 + cos(alpha))) / \
            (2 * alpha ** 2 * sin(alpha))
        mu = (alpha * (alpha + sin(alpha)) - 8 * sin(alpha / 2) ** 2) / \
            (4 * alpha ** 4 * sin(alpha / 2) ** 2)
    return eta, mu


def getSkewSymmetric(PSI):
    PSI_tilde = np.array([[0, -PSI[2], PSI[1]],
                          [PSI[2], 0, -PSI[0]],
                          [-PSI[1], PSI[0], 0]], dtype=float)
    return PSI_tilde


def decomposeSkewSymmetric(PSI_tilde):
    if((abs(PSI_tilde + PSI_tilde.T) < 1e-12 * np.eye(3)).all()):
        raise RuntimeError('The input matrix is not skew symmetric!')
    PSI = np.zeros((3, 1), dtype=float)
    PSI[0], PSI[1], PSI[2] = PSI_tilde[2, 1], PSI_tilde[0, 2], PSI_tilde[1, 0]

    return PSI


def decomposeRotationalVector(PSI, output='all'):
    if output == 'norm':
        psi = np.linalg.norm(PSI)
        return psi

    elif output == 'unit_vector':
        psi = np.linalg.norm(PSI)
        if psi == 0.:
            u = PSI
        else:
            u = PSI / psi
            return u

    elif output == 'all':
        psi = np.linalg.norm(PSI)
        if psi == 0.:
            u = PSI
        else:
            u = PSI / psi
        PSI_tilde = getSkewSymmetric(PSI)
        return u, psi, PSI_tilde


def getTransformation(PSI, output='all'):
    u, psi, PSI_tilde = decomposeRotationalVector(PSI)

    if output == 'all':
        if psi == 0.:
            T_s = np.eye(3) + 1 / 2 * PSI_tilde
            detT_s = 1
            T_s_inv = np.eye(3) - 1 / 2 * PSI_tilde

        else:
            T_s = sin(psi) / psi * np.eye(3) + (1 - sin(psi) / psi) * u @ u.T + 1 / 2 * (
                sin(psi / 2) / (psi / 2)) ** 2 * PSI_tilde
            detT_s = 2 * (1 - np.cos(psi)) / psi ** 2

            T_s_inv = (psi / 2) / tan(psi / 2) * np.eye(3) + \
                      (1 - (psi / 2) / tan(psi / 2)) * \
                u @ u.T - 1 / 2 * PSI_tilde
            T_s_inv = ((psi / 2) / tan(psi / 2)) * np.eye(3) + (
                1 - (psi / 2) / tan(psi / 2)) * u @ u.T - 1 / 2 * PSI_tilde
        return T_s, detT_s, T_s_inv

    elif output == 'inv':
        if psi == 0.:
            T_s_inv = np.eye(3) - 1 / 2 * PSI_tilde

        else:
            T_s_inv = ((psi / 2) / tan(psi / 2)) * np.eye(3) + (
                1 - (psi / 2) / tan(psi / 2)) * u @ u.T - 1 / 2 * PSI_tilde

        return T_s_inv


def deviator(stress):
    if stress.shape != (3, 1):
        raise TypeError("Stress must be a 3x1 numpy array!")
    else:
        stress_tensor = np.array([[stress[0, 0], stress[1, 0], stress[2, 0]],
                                  [stress[1, 0], 0., 0.],
                                  [stress[2, 0], 0., 0.]])
        p = 1/3 * stress_tensor[0, 0]
        deviator = stress_tensor - p * np.eye(3)
        
        return deviator


def unit_tensor():
    return np.array([[1., 1., 1., 0., 0., 0.]]).T

def I_dev():
    # return np.array([[2/3, -1/3, -1/3, 0., 0., 0.],
    #                  [-1/3, 2/3, -1/3, 0., 0., 0.],
    #                  [-1/3, -1/3, 2/3, 0., 0., 0.],
    #                  [0., 0., 0., 1/2, 0., 0.],
    #                  [0., 0., 0., 0, 1/2, 0.],
    #                  [0., 0., 0., 0, 0., 1/2]])
    
    return np.eye(6) - 1/3 * unit_tensor()

# a = np.tensordot(unit_tensor(), unit_tensor(), axes=0)
# print(a)

# eta = deviator(np.array([[1, 2, 3]]).T)
# print(eta)
# norm = np.linalg.norm(eta)

# print(norm * np.sqrt(1.5))
# print(sigma_e)
# print(sigma_e/norm)

# def eight_node_serendipity_shape_function(xi, eta):
#     return np.array([
#         -1/4 * (1 - xi) * (1 - eta) * (1 + xi + eta),
#         -1/4 * (1 + xi) * (1 - eta) * (1 - xi + eta),
#         -1/4 * (1 + xi) * (1 + eta) * (1 - xi - eta),
#         -1/4 * (1 - xi) * (1 + eta) * (1 + xi - eta),
#         1/2 * (1 - xi ** 2) * (1 - eta),
#         1/2 * (1 - eta ** 2) * (1 + xi),
#         1/2 * (1 - xi ** 2) * (1 + eta),
#         1/2 * (1 - eta ** 2) * (1 - xi)])


# def eight_node_serendipity_shape_function_ders(xi, eta):
#     dN_dxi = np.array([
#         (1/4 - 1/4 * eta) * (eta + xi + 1) + (1 - eta) * (1/4 * xi - 1/4),
#         -(1 - eta) * (-1/4 * xi - 1/4) + (1/4 * eta - 1/4) * (eta - xi + 1),
#         (-1/4 * eta - 1/4) * (-eta - xi + 1) - (eta + 1) * (-1/4 * xi - 1/4),
#         (1/4 * eta + 1/4) * (-eta + xi + 1) + (eta + 1) * (1/4 * xi - 1/4),
#         -xi * (1 - eta),
#         1/2 - 1/2 * eta ** 2,
#         -xi * (eta + 1),
#         1/2 * eta ** 2 - 1/2])
    
#     dN_deta = np.array([
#         (1/4 - 1/4 * xi) * (eta + xi + 1) + (1 - eta) * (1/4 * xi - 1/4),
#         (1 - eta) * (-1/4 * xi - 1/4) + (1/4 * xi + 1/4) * (eta - xi + 1),
#         -(eta + 1) * (-1/4 * xi - 1/4) + (-1/4 * xi - 1/4) * (-eta - xi + 1),
#         -(eta + 1) * (1/4 * xi - 1/4) + (1/4 * xi - 1/4) * (-eta + xi + 1),
#         1/2 * xi ** 2 - 1/2,
#         -eta * (xi + 1),
#         1/2 - 1/2 * xi ** 2,
#         -eta * (1 - xi)])
    
#     return dN_dxi, dN_deta


def nodal_warping_function(width, height):
        
    N = lambda xi, eta: np.array([
        -1/4 * (1 - xi) * (1 - eta) * (1 + xi + eta),
        -1/4 * (1 + xi) * (1 - eta) * (1 - xi + eta),
        -1/4 * (1 + xi) * (1 + eta) * (1 - xi - eta),
        -1/4 * (1 - xi) * (1 + eta) * (1 + xi - eta),
        1/2 * (1 - xi ** 2) * (1 - eta),
        1/2 * (1 - eta ** 2) * (1 + xi),
        1/2 * (1 - xi ** 2) * (1 + eta),
        1/2 * (1 - eta ** 2) * (1 - xi)])
    
    dN_dxi = lambda xi, eta: np.array([
        -1/4 * (-1 + eta) * (2 * xi + eta),
        1/4 * (-1 + eta) * (eta - 2 * xi),
        1/4 * (1 + xi) * (2 * eta + xi),
        -1/4 * (1 + xi) * (eta - 2 * xi),
        xi * (-1 + eta),
        -1/2 * (1 + eta) * (-1 + eta),
        -xi * (1 + eta),
        1/2 * (1 + eta) * (-1 + eta)])
    
    dN_deta = lambda xi, eta: np.array([
        -1/4 * (-1 + xi) * (xi + 2 * eta),
        1/4 * (1 + xi) * (2 * eta - xi),
        1/4 * (1 + xi) * (xi + 2 * eta),
        -1/4 * (-1 + xi) * (2 * eta - xi),
        1/2 * (1 + xi) * (-1 + xi),
        -eta * (1 + xi),
        -1/2 * (1 + xi) * (-1 + xi),
        eta * (-1 + xi)])
           
    [gauss_locations_xi, weights_xi] = np.polynomial.legendre.leggauss(4)
    [gauss_locations_eta, weights_eta] = np.polynomial.legendre.leggauss(4)

    # y_I = np.array([-width/2, width/2, width/2, -width/2,
    #                 0., width/2, 0., -width/2])
    # z_I = np.array([-height/2, -height/2, height/2, height/2,
    #                 -height/2, 0., height/2, 0.])
    
    # y_numerical = lambda xi, eta: np.dot(N(xi, eta), y_I)
    # z_numerical = lambda xi, eta: np.dot(N(xi, eta), z_I)
    
    y = lambda xi: width/2 * xi
    z = lambda eta: height/2 * eta
    
    # XI = np.linspace(-1, 1, 21)
    # ETA = np.linspace(-1, 1, 21)
    # XI_mesh, ETA_mesh = np.meshgrid(XI, ETA)

    # Y_numerical = np.zeros((21, 21), dtype=float)
    # Z_numerical = np.zeros((21, 21), dtype=float)
    # for i, xi in enumerate(XI):
    #     for j, eta in enumerate(ETA):
    #         Y_numerical[i, j] = y_numerical(xi, eta)
    #         Z_numerical[i, j] = z_numerical(xi, eta)
    
    # fig = plt.figure()
    # ax = Axes3D(fig, auto_add_to_figure=False)
    # fig.add_axes(ax)
    # ax.plot_surface(XI_mesh, ETA_mesh, Y_numerical, rstride=1, cstride=1, cmap=cm.viridis)
    # ax.set_xlabel('$y$')
    # ax.set_ylabel('$z$')
    
    # plt.show()
    
    
#     stiffness = np.zeros((8, 8), dtype=float)
#     for i in range(8):
#         for j in range(8):
#             for ixi in range(4):
#                 for jeta in range(4):
#                     stiffness[i, j] += (height / width + width / height) * \
#                     dN_dxi(gauss_locations_xi[ixi], gauss_locations_eta[jeta])[i] * \
#                     dN_deta(gauss_locations_xi[ixi], gauss_locations_eta[jeta])[j] * \
#                     weights_xi[ixi] * weights_eta[jeta]
                    
#     load = np.zeros((8, 1), dtype=float)
#     for i in range(8):
#         for ixi in range(4):
#             for jeta in range(4):
#                 load[i, 0] += (height/2 * dN_dxi(gauss_locations_xi[ixi], gauss_locations_eta[jeta])[i] * z(gauss_locations_eta[jeta]) - \
#                             width/2 * dN_dxi(gauss_locations_xi[ixi], gauss_locations_eta[jeta])[i] * y(gauss_locations_xi[ixi])) * \
#                             weights_xi[ixi] * weights_eta[jeta]
    
#     arbitrary_dof = 2
#     modified_stiffness = stiffness.copy()
#     for ientry in range(8):
#         modified_stiffness[arbitrary_dof, ientry] = 0.
#         modified_stiffness[ientry, arbitrary_dof] = 0.
#         modified_stiffness[arbitrary_dof, arbitrary_dof] = 1.
    
#     modified_load = load.copy()
#     modified_load[arbitrary_dof] = 0.
    
#     nodal_warping_function = np.linalg.solve(modified_stiffness, modified_load)
    
#     warping_function = lambda xi, eta: np.dot(N(xi, eta), nodal_warping_function.reshape(8))
    
#     average_integral = 0.0
#     for ixi in range(4):
#         for jeta in range(4):
#             average_integral += warping_function(gauss_locations_xi[ixi], gauss_locations_eta[jeta]) * \
#             width * height / 4 * weights_xi[ixi] * weights_eta[jeta]
#     average_integral /= width * height
    
#     adjusted_nodal_warping_function = nodal_warping_function.copy()
#     for i in range(8):
#         adjusted_nodal_warping_function[i] -= average_integral
    
#     return adjusted_nodal_warping_function


# def warping_function(y, z, width, height, nodal_warping_vector):
    
#     N = lambda xi, eta: np.array([
#         -1/4 * (1 - xi) * (1 - eta) * (1 + xi + eta),
#         -1/4 * (1 + xi) * (1 - eta) * (1 - xi + eta),
#         -1/4 * (1 + xi) * (1 + eta) * (1 - xi - eta),
#         -1/4 * (1 - xi) * (1 + eta) * (1 + xi - eta),
#         1/2 * (1 - xi ** 2) * (1 - eta),
#         1/2 * (1 - eta ** 2) * (1 + xi),
#         1/2 * (1 - xi ** 2) * (1 + eta),
#         1/2 * (1 - eta ** 2) * (1 - xi)])
    
#     xi = 2 * y / width
#     eta = 2 * z / height
    
#     return np.dot(N(xi, eta), nodal_warping_vector)
    

# d = nodal_warping_function(2., 2.)

# Y = np.linspace(-1, 1, 21)
# Z = np.linspace(-1, 1, 21)
# Y_grid, Z_grid = np.meshgrid(Y, Z)
# Omega_numerical = np.zeros((21, 21), dtype=float)
# for i, y in enumerate(Y):
#     for j, z in enumerate(Z):
#         Omega_numerical[i, j] = warping_function(y, z, 2.0, 2.0, d)

# fig = plt.figure()
# ax = Axes3D(fig, auto_add_to_figure=False)
# fig.add_axes(ax)
# ax.plot_surface(Y_grid, Z_grid, Omega_numerical, rstride=1, cstride=1, cmap=cm.viridis)

# plt.show()

# integration = integrate.nquad(lambda y, z: warping_function(y, z, 2.0, 2.0, d), [[-1, 1],[-1, 1]])
# print(integration)