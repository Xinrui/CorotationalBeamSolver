# -*- coding: utf-8 -*-
"""
@author: Xinrui Zhou
"""

import numpy as np
from numpy import cos, sin, tan
import matplotlib.pyplot as plt


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
    if alpha == 0:
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
        if psi == 0:
            u = PSI
        else:
            u = PSI / psi
            return u

    elif output == 'all':
        psi = np.linalg.norm(PSI)
        if psi == 0:
            u = PSI
        else:
            u = PSI / psi
        PSI_tilde = getSkewSymmetric(PSI)
        return u, psi, PSI_tilde


def getTransformation(PSI, output='all'):
    u, psi, PSI_tilde = decomposeRotationalVector(PSI)

    if output == 'all':
        if psi == 0:
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
        if psi == 0:
            T_s_inv = np.eye(3) - 1 / 2 * PSI_tilde

        else:
            T_s_inv = ((psi / 2) / tan(psi / 2)) * np.eye(3) + (
                1 - (psi / 2) / tan(psi / 2)) * u @ u.T - 1 / 2 * PSI_tilde

        return T_s_inv
