#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gaik Tamazian, 2017-2018
# mail (at) gtamazian (dot) com

"""Contains mathematical routines for protein modeling.

The routines include functions for interpolation, the Kabsch transformation and
conversions between Cartesian and internal coordinates. Numpy arrays are used
to implement the module functions. The module is called `geometry` instead of
`math` to avoid collisions with the `math` module from the standard library.
"""


import numpy as np
import numpy.linalg as la
import sys


assert sys.version_info >= (3, 6), 'Python 3.6 or higher required'


def kabsch(x, y):
    """
    Given two matrices, perform the Kabsch transformation.

    The second matrix is superposed to the first one. The function returns a
    tuple containing the superposed second matrix and the rotation matrix.
    """
    assert np.shape(x) == np.shape(y), 'matrix sizes must be equal'

    # shift the matrices
    mu_x, mu_y = np.mean(x, axis=0), np.mean(y, axis=0)
    x_shifted = x - mu_x
    y_shifted = y - mu_y

    u, s, v = la.svd(np.dot(x_shifted.T, y_shifted))
    v = v.T
    q = np.dot(v, u.T)
    if la.det(q) < 0:
        # change the last column of matrix U to prevent reflection
        u[:, -1] = -u[:, -1]
        q = np.dot(v, u.T)

    z = np.dot(y_shifted, q) + mu_x

    return z, q


def rmsd(x, y):
    """
    Calculate the root-mean-square deviation (RMSD) between matrices.
    """
    assert np.shape(x) == np.shape(y), 'matrix sizes must be equal'

    return np.sqrt(np.mean(np.sum(np.square(x - y), axis=1)))


def reduce_angles(x, lower=-np.pi, upper=np.pi):
    """
    Return reduced angular values from the given array.

    The values are reduced by adding or subtracting 2*pi until all of them lie
    within the specified lower and upper bounds.
    """
    y = x.copy()

    while np.any(y < lower) or np.any(y > upper):
        y[y < lower] += 2*np.pi
        y[y > upper] -= 2*np.pi

    return y


def cart2inter(x):
    """
    Convert Cartesian coordinates to internal.
    """
    assert np.ndim(x) == 2 and np.shape(x)[1] == 3, 'matrix n x 3 required'

    delta_x = x[1:, ] - x[:-1, ]
    r = np.sqrt(np.sum(np.square(delta_x), axis=1))
    alpha = np.arccos(np.sum(delta_x[1:, ] * delta_x[:-1, ], axis=1) /
                      (r[1:] * r[:-1]))
    n = np.cross(delta_x[:-1, ], delta_x[1:, ])
    gamma = np.arctan2(r[1:-1] *
                       np.sum(delta_x[:-2, ] * n[1:, ], axis=1),
                       np.sum(n[:-1, ] * n[1:, ], axis=1))

    return r, alpha, gamma


def cross3d(x, y):
    """
    Vector cross product for three-dimensional vectors.
    """
    return np.array([x[1]*y[2] - x[2]*y[1],
                     x[2]*y[0] - x[0]*y[2],
                     x[0]*y[1] - x[1]*y[0]])


def inter2cart(r, alpha, gamma):
    """
    Convert internal coordinates to Cartesian.

    The function implements the Natural Extension Reference Frame (NERF) method
    as described in

    Parsons, J., Holmes, J. B., Rojas, J. M., Tsai, J., & Strauss, C. E.
    (2005). Practical conversion from torsion space to Cartesian space for in
    silico protein synthesis. Journal of Computational Chemistry, 26(10),
    1063-1068.
    """
    assert np.ndim(r) == np.ndim(alpha) == np.ndim(gamma) == 1, \
        'one-dimensional arrays required'
    assert np.size(r) == np.size(alpha) + 1 == np.size(gamma) + 2, \
        'inconsistent array sizes'

    p = np.size(r) + 1  # the number of points

    # the first three points are located in the same plane
    x = np.empty((p, 3))
    x[0, ] = 0
    x[1, ] = np.array([r[0], 0, 0])
    x[2, ] = np.array([r[0] + r[1]*np.cos(alpha[0]), r[1]*np.sin(alpha[0]),
                       0])

    x[3:, 0] = r[2:] * np.cos(alpha[1:])
    x[3:, 1] = r[2:] * np.sin(alpha[1:]) * np.cos(gamma)
    x[3:, 2] = r[2:] * np.sin(alpha[1:]) * np.sin(gamma)

    # process points from #4 to the last one
    for k in range(3, p):
        bc = x[k-1, ] - x[k-2, ]
        bc /= la.norm(bc)
        n = cross3d(x[k-2, ] - x[k-3, ], bc)
        n /= la.norm(n)
        m = np.array((bc, cross3d(n, bc), n)).T
        x[k, ] = np.dot(m, x[k, ]) + x[k-1, ]

    return x


def circ_dist(x, y):
    """
    Calculate circular distances between angles in given arrays.
    """
    return reduce_angles(y - x)


def lin_interp(x, y, num):
    """
    Perform linear interpolation between two arrays.
    """
    assert np.ndim(x) == np.ndim(y) == 1, 'one-dimensional arrays required'

    n = np.size(x)
    mask = np.tile(np.arange(num + 2), (n, 1))/(num + 1)
    return np.transpose(np.tile(x, (num + 2, 1))) + \
        mask * np.transpose(np.tile(y - x, (num + 2, 1)))


def circ_interp(x, y, num, long_arc=False):
    """
    Perform circular interpolation between two arrays.

    The `long_arc` argument specifies circular interpolation direction.
    """
    assert np.ndim(x) == np.ndim(y) == 1, 'one-dimensional arrays required'

    n = np.size(x)
    mask = np.tile(np.arange(num + 2), (n, 1))/(num + 1)
    result = np.transpose(np.tile(x, (num + 2, 1)))
    delta = np.transpose(np.tile(circ_dist(x, y), (num + 2, 1)))
    if not long_arc:
        result += delta * mask
    else:
        result += -np.sign(delta) * (2*np.pi - np.abs(delta)) * mask

    return result
