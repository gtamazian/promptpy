#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gaik Tamazian, 2017-2018
# mail (at) gtamazian (dot) com

"""Implements routines related to protein transformation model.

This module does not contain routines related to reading and writing
transformations from or to files. All file-related routines should be imported
from the corresponding modules, for example, PDB-related routines are available
in the `pdb` module.
"""

import copy
import h5py
import logging
import numpy as np
import numpy.linalg as la
import os
import os.path
import scipy.spatial.distance
import sys
from collections import namedtuple
from operator import itemgetter
from . import geometry as geo

assert sys.version_info >= (3, 6), 'Python 3.6 or higher required'

logging.basicConfig(format='%(asctime)-15s - %(levelname)-8s: '
                    '%(message)s')
logger = logging.getLogger(__name__)


class Transformation(object):
    def __init__(self, r, alpha, gamma, w, start_coords=None, rot_mat=None):
        """
        Create a transformation model given its contents.
        """
        logger.debug('started creating a model')

        logger.info('original weight range: %e - %e', np.min(w),
                    np.max(w))
        self.w = w/np.sum(w)
        logger.info('weight range after normalization: %e - %e, '
                    'normalization denominator: %e',
                    np.min(self.w), np.max(self.w), np.sum(w))

        self._r = None
        self._alpha = None
        self._gamma = None
        self._start_coords = None
        self._rot_mat = None
        self._is_fortran_layout = False
        logger.debug('by default, the Fortran layout is used')

        logger.debug('loading matrix of bond lengths of shape (%d, %d)',
                     *np.shape(r))
        self.r = r

        logger.debug('loading matrix of planar angles of shape (%d, %d)',
                     *np.shape(alpha))
        self.alpha = alpha

        logger.debug('loading matrix of torsion angles of shape (%d, %d)',
                     *np.shape(gamma))
        self.gamma = gamma

        if start_coords is None:
            logger.info('missing start coordinates: starting from origin')
            start_coords = geo.inter2cart(r[0, :], alpha[0, :], gamma[0, :])

        self.start_coords = start_coords

        # the center of the first (starting) configuration is used further,
        # so we  precalculate it
        self.start_center = np.mean(start_coords, axis=0)

        # calculate rotation matrices
        if rot_mat is None:
            m = np.shape(r)[0]
            logger.debug('missing rotation matrices: started calculation of '
                         '%d rotation matrices', m)
            temp_rot_mat = np.empty((m, 3, 3))
            temp_rot_mat[0, :, :] = np.eye(3)
            self.rot_mat = temp_rot_mat
            self.update_rotations()
            logger.debug('calculation of rotation matrices completed')
        else:
            self.rot_mat = rot_mat

        logger.debug('model creation completed')

    @property
    def is_fortran_layout(self):
        return self._is_fortran_layout

    @is_fortran_layout.setter
    def is_fortran_layout(self, value):
        if self._is_fortran_layout != value:
            if value:
                logger.debug('changing the model layout from C to Fortran')
            else:
                logger.debug('changing the model layout from Fortran to C')
            self._is_fortran_layout = value
            # arrays _r, _alpha, _gamma and _start_coords are transposed in the
            # both cases
            self._r = np.transpose(self._r)
            self._alpha = np.transpose(self._alpha)
            self._gamma = np.transpose(self._gamma)
            self._start_coords = np.transpose(self._start_coords)
            self._rot_mat = np.transpose(self._rot_mat)

    @property
    def r(self):
        if self._is_fortran_layout:
            return np.transpose(self._r)
        else:
            return self._r

    @r.setter
    def r(self, value):
        if self._is_fortran_layout:
            self._r = np.transpose(value)
        else:
            self._r = value

    @property
    def alpha(self):
        if self._is_fortran_layout:
            return np.transpose(self._alpha)
        else:
            return self._alpha

    @alpha.setter
    def alpha(self, value):
        if self._is_fortran_layout:
            self._alpha = np.transpose(value)
        else:
            self._alpha = value

    @property
    def gamma(self):
        if self._is_fortran_layout:
            return np.transpose(self._gamma)
        else:
            return self._gamma

    @gamma.setter
    def gamma(self, value):
        if self._is_fortran_layout:
            return np.transpose(self._gamma)
        else:
            self._gamma = value

    @property
    def start_coords(self):
        if self.is_fortran_layout:
            return np.transpose(self._start_coords)
        else:
            return self._start_coords

    @start_coords.setter
    def start_coords(self, value):
        if self.is_fortran_layout:
            self._start_coords = np.transpose(value)
        else:
            self._start_coords = value

    @property
    def rot_mat(self):
        if self.is_fortran_layout:
            temp = np.empty((self.n_conf(), 3, 3))
            for k in range(self.n_conf()):
                temp[k, :, :] = np.transpose(self._rot_mat[:, :, k])
            return temp
        else:
            return self._rot_mat

    @rot_mat.setter
    def rot_mat(self, value):
        if self.is_fortran_layout:
            for k in range(self.n_conf()):
                self._rot_mat[:, :, k] = np.transpose(value[k, :, :])
        else:
            self._rot_mat = value

    def update_rotations(self):
        """
        Update rotation matrices for superposing the transformation
        configurations to each other. The matrices are calculated using the
        Kabsch transformation.
        """
        new_rot_mat = np.empty((self.n_conf(), 3, 3))
        new_rot_mat[0, :, :] = np.eye(3)

        prev_conf = self.start_coords
        for i in range(1, self.n_conf()):
            curr_conf = geo.inter2cart(self.r[i, :], self.alpha[i, :],
                                       self.gamma[i, :])
            prev_conf, curr_rot = geo.kabsch(prev_conf, curr_conf)
            new_rot_mat[i, :, :] = curr_rot

        self.rot_mat = new_rot_mat

    def n_atom(self):
        """
        Return the number of atoms in the transformation.
        """
        return np.shape(self.r)[1] + 1

    def n_conf(self):
        """
        Return the number of configurations in the transformation.
        """
        return np.shape(self.r)[0]

    def get_cart_coords(self):
        """
        Return a tuple of two matrices: Cartesian coordinates of
        the model points and Cartesian coordinates of the points
        before configuration superposition.
        """
        m = self.n_conf()
        n = self.n_atom()

        logger.debug('restoring coordinates for %d atoms and %d '
                     'configurations', n, m)

        coords = np.empty((m, n, 3))
        coords[0, :, :] = self.start_coords

        logger.debug('converting internal coordinates to Cartesian '
                     'ones for each of %d configurations', m)
        pre_coords = np.empty((m, n, 3))
        for i in range(m):
            pre_coords[i, :, :] = geo.inter2cart(self.r[i, :],
                                                 self.alpha[i, :],
                                                 self.gamma[i, :])

        logger.debug('superposing restored configurations')
        for i in range(1, m):
            # apply rotation and translation
            coords[i, :, :] = np.dot(pre_coords[i, :, :],
                                     self.rot_mat[i, :, :])

            coords[i, :, :] -= np.mean(coords[i, :, :], axis=0)
            coords[i, :, :] += self.start_center

        logger.debug('restored coordinate matrices of original and '
                     'superposed transformation of total size 2 x '
                     '%d = %d bytes', coords.nbytes, 2*coords.nbytes)

        return (coords, pre_coords)

    def __eq__(self, y):
        return np.allclose(self.w, y.w) and \
            np.allclose(self.r, y.r) and \
            np.allclose(self.alpha, y.alpha) and \
            np.allclose(self.gamma, y.gamma) and \
            np.allclose(self.start_coords, y.start_coords)

    def __iter__(self):
        yield 'w', self.w.tolist()
        yield 'r', self.r.tolist()
        yield 'alpha', self.alpha.tolist()
        yield 'gamma', self.gamma.tolist()
        yield 'start_coords', self.start_coords.tolist()
        yield 'rot_mat', self.rot_mat.tolist()

    def from_dict(d):
        """
        Create a transformation model from a dictionary with its contents.

        This function was designed to be used with `json.load`.
        """
        return Transformation(np.array(d['r']), np.array(d['alpha']),
                              np.array(d['gamma']), np.array(d['w']),
                              np.array(d['start_coords']),
                              np.array(d['rot_mat']))

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def rmsd(self):
        """
        Return RMSD values between adjacent configurations.
        """
        x = self.get_cart_coords()[0]
        result = []
        for i in range(self.n_conf() - 1):
            result.append(geo.rmsd(x[i, :, :], x[i+1, :, :]))

        return result

    def rmsd_to_conf(self, conf_number):
        """
        Return RMSD values to the specified configuration.
        """
        x = self.get_cart_coords()[0]
        result = []
        for i in range(self.n_conf()):
            result.append(geo.rmsd(x[i, :, :], x[conf_number, :, :]))

        return result

    def min_interatomic_dist(self):
        """
        Return minimal interatomic distances within configurations.
        """
        result = []
        x = self.get_cart_coords()[0]
        for i in range(self.n_conf()):
            result.append(np.min(scipy.spatial.distance.pdist(x[i, :, :])))
        return result

    def cost(self, p_indices=None, t_indices=None):
        """
        Calculate the cost value of the transformation and the gradient vector
        for the specified planar and torsion angles. If no angles are
        specified, then the gradient computation is skipped.
        """
        TransformationCost = namedtuple('TransformationCost', ('cost',
                                                               'gradient'))

        logger.debug('started transformation cost computation')

        x_tilde, x = self.get_cart_coords()
        cost = np.sum(self.w * np.sum(np.square(
            x_tilde[1:, :, :] - x_tilde[:-1, :, :]), axis=(0, 2)))

        logger.debug('transformation cost computed')

        if p_indices is None and t_indices is None:
            # the gradient is not required, just return the transformation cost
            # value
            return TransformationCost(cost, None)

        if p_indices is None:
            p_indices = np.array([])

        if t_indices is None:
            t_indices = np.array([])

        p_num = p_indices.size
        t_num = t_indices.size

        logger.debug('started calculating gradient for %d planar and '
                     '%d torsion angles', p_num, t_num)

        n = self.n_atom()
        m = self.n_conf()

        logger.debug('considered transformation contains %d atoms and '
                     '%d intermediate configurations', n, m)

        s = np.empty((m - 1, n, 3))
        s[1:, :, :] = 2*x_tilde[1:-1, :, :] - x_tilde[:-2, :, :] \
            - x_tilde[2:, :, :]
        logger.debug('allocated %d bytes for array s', s.nbytes)

        r = x[:, 1:, :] - x[:, :-1, :]
        logger.debug('allocated %d bytes for array r', r.nbytes)

        w = np.reshape(np.tile(x, (1, n - 1, 1)) -
                       np.repeat(x[:, 1:, ], n, axis=1), (m, n - 1, n, 3))
        logger.debug('allocated %d bytes for array w', w.nbytes)

        A = self.rot_mat

        t = np.empty((m, p_num, n, 3))
        p_grad = np.zeros((m, p_num))
        logger.debug('allocated %d bytes for array t', t.nbytes)
        logger.debug('allocated %d bytes for array p_grad',
                     p_grad.nbytes)

        q = np.empty((m, t_num, n, 3))
        t_grad = np.zeros((m, t_num))
        logger.debug('allocated %d bytes for array q', q.nbytes)
        logger.debug('allocated %d bytes for array t_grad',
                     t_grad.nbytes)

        # consider partial derivatives with respect to planar angles
        if p_num > 0:
            logger.debug('considering partial derivatives with respect '
                         'to %d planar angles', p_num)
            u = np.empty((m, n - 2, 3))
            u[:, :, 0] = r[:, :-1, 1] * r[:, 1:, 2] - \
                r[:, :-1, 2] * r[:, 1:, 1]
            u[:, :, 1] = r[:, :-1, 2] * r[:, 1:, 0] - \
                r[:, :-1, 0] * r[:, 1:, 2]
            u[:, :, 2] = r[:, :-1, 0] * r[:, 1:, 1] - \
                r[:, :-1, 1] * r[:, 1:, 0]
            u = u / np.tile(np.expand_dims(la.norm(u, axis=2), 2), (1, 1, 3))
            logger.debug('allocated %d bytes for array u', u.nbytes)

            for j in range(1, m - 1):
                for i_idx, i in enumerate(p_indices):
                    mask = np.ones((n, 3), dtype=np.int8)
                    mask[:(i + 2), :] = 0

                    t[j, i_idx, :, :] = np.matmul(
                        np.cross(u[j, i, :],
                                 w[j, i, :, :] * mask -
                                 np.sum(w[j, i, (i + 2):, :], axis=0)/n),
                        A[j, :, :])

                    p_grad[j, i_idx] = np.sum(
                        self.w *
                        np.sum(s[j, :, :] * t[j, i_idx, :, :], axis=1))

            p_grad = p_grad[1:-1, :].flatten()
            p_grad *= 2
        else:
            p_grad = np.array([])

        # consider partial derivatives with respect to torsion angles
        if t_num > 0:
            logger.debug('considering partial derivatives with respect '
                         'to %d torsion angles', t_num)
            v = r[:, 1:, :]
            v = v / np.tile(np.expand_dims(la.norm(v, axis=2), 2), (1, 1, 3))
            logger.debug('allocated %d bytes for array v', v.nbytes)

            for j in range(1, m - 1):
                for i_idx, i in enumerate(t_indices):
                    mask = np.ones((n, 3), dtype=np.int8)
                    mask[:(i + 3), :] = 0

                    q[j, i_idx, :, :] = np.matmul(
                        np.cross(v[j, i, :],
                                 w[j, i, :, :] * mask -
                                 np.sum(w[j, i, (i + 3):, :], axis=0)/n),
                        A[j, :, :])

                    t_grad[j, i_idx] = np.sum(
                        self.w *
                        np.sum(s[j, :, :] * q[j, i_idx, :, :], axis=1))

            t_grad = t_grad[1:-1, :].flatten()
            t_grad *= 2
        else:
            t_grad = np.array([])

        gradient = np.concatenate([p_grad, t_grad])

        return TransformationCost(cost, gradient)

    def get_angles(self, p_indices, t_indices):
        """
        Get values for planar and torsion angles with the specified indices.
        """
        if p_indices.size > 0:
            p_angles = self.alpha[1:-1, p_indices].flatten()
        else:
            p_angles = np.array([])

        if t_indices.size > 0:
            t_angles = self.gamma[1:-1, t_indices].flatten()
        else:
            t_angles = np.array([])

        return np.concatenate((p_angles, t_angles))

    def set_angles(self, p_indices, t_indices, angles):
        """
        Set values for planar and torsion angles with the specified indices.
        """
        n_planar = np.size(p_indices)
        n_torsion = np.size(t_indices)

        m = self.n_conf()

        if n_planar > 0:
            new_alpha = np.copy(self.alpha)
            new_alpha[1:-1, p_indices] = np.reshape(
                angles[:n_planar*(m - 2)], (m - 2, n_planar)
            )
            self.alpha = new_alpha

        if n_torsion > 0:
            new_gamma = np.copy(self.gamma)
            self.gamma[1:-1, t_indices] = np.reshape(
                angles[n_planar*(m - 2):], (m - 2, n_torsion)
            )
            self.gamma = new_gamma

    def get_ordered_angles(self):
        """
        Return a named tuple containing two lists - indices of planar and
        torsion angles sorted by absolute circular difference between their
        values in the given conformation in the descending order.
        """
        OrderedAngles = namedtuple('OrderedAngles', ('planar', 'torsion'))

        planar = sorted(enumerate(np.abs(geo.circ_dist(
            self.alpha[0, :], self.alpha[-1, :]))),
            key=itemgetter(1), reverse=True)

        torsion = sorted(enumerate(np.abs(geo.circ_dist(
            self.gamma[0, :], self.gamma[-1, :]))),
            key=itemgetter(1), reverse=True)

        return OrderedAngles(list(map(itemgetter(0), planar)),
                             list(map(itemgetter(0), torsion)))

    def get_obj_func(self, p_indices, t_indices, gradient=False):
        """
        Return the function that returns the transformation cost function value
        and the gradient vector at the specified point.
        """
        m_copy = copy.deepcopy(self)
        no_iter = 0
        min_cost = np.inf

        def obj_func(x):
            nonlocal no_iter
            nonlocal min_cost

            m_copy.set_angles(p_indices, t_indices, x)
            result = m_copy.cost(p_indices if gradient else None,
                                 t_indices if gradient else None)
            if result.cost < min_cost:
                min_cost = result.cost
                no_iter += 1
                if result.gradient is not None:
                    logger.info('Calc %d\tcost %e\tgr norm %e',
                                no_iter, min_cost, la.norm(result.gradient))
                else:
                    logger.info('Calc %d\tcost %e', no_iter, min_cost)
            return result

        return obj_func


def read_hdf5(fname):
    """
    Read a transformation model from the specified HDF5 file.
    """
    result = None
    with h5py.File(fname) as f:
        result = Transformation.from_dict(
            {'w': f['w'],
             'r': f['r'],
             'alpha': f['alpha'],
             'gamma': f['gamma'],
             'rot_mat': f['rot_mat'],
             'start_coords': f['start_coords']})

    return result


def write_hdf5(model, fname):
    """
    Write a transformation model to the specified file in the HDF5 format.
    """
    if os.path.exists(fname):
        os.unlink(fname)

    with h5py.File(fname, 'w') as f:
        for k, v in model:
            f.create_dataset(k, data=v)


def from_marg_coords(start_coords, end_coords, weights, n_conf):
    """
    Create a transformation model given marginal atom coordinates and
    weights.
    """
    assert np.ndim(start_coords) == np.ndim(end_coords) == 2, \
        'matrices n x 3 required'
    assert np.shape(start_coords) == np.shape(end_coords), \
        'start and end coordinate matrices must be of same size'
    assert np.alen(start_coords) == np.size(weights), \
        'discordant sizes of coordinate matrices and weight array'
    assert np.shape(start_coords)[1] == 3, '3D coordinates required'

    # calculate internal coordinates
    r_s, alpha_s, gamma_s = geo.cart2inter(start_coords)
    r_e, alpha_e, gamma_e = geo.cart2inter(end_coords)
    r = geo.lin_interp(r_s, r_e, n_conf)
    alpha = geo.circ_interp(alpha_s, alpha_e, n_conf)
    gamma = geo.circ_interp(gamma_s, gamma_e, n_conf)

    return Transformation(r.T, alpha.T, gamma.T, weights, start_coords)


def from_conf_coords(coords, weights):
    """
    Create a transformation model given weights and atom coordinates for
    every its configuration.
    """
    assert isinstance(coords, list), 'list of coordinate matrices required'
    assert np.ndim(weights) == 1, '1D array of weights required'

    n = np.alen(coords[0])
    assert np.size(weights) == n, \
        'inconsistent numbers of atoms and weights'

    m = len(coords)
    assert m > 2, 'more than two coordinate matrices required'

    r = np.empty((m, n - 1))
    alpha = np.empty((m, n - 2))
    gamma = np.empty((m, n - 3))

    for i in range(m):
        assert np.alen(coords[i]) == n, 'inconsistent number of atoms'
        assert np.shape(coords[i])[1] == 3, '3D coordinates required'
        r[i, :], alpha[i, :], gamma[i, :] = geo.cart2inter(coords[i])

    return Transformation(r, alpha, gamma, weights, coords[0])
