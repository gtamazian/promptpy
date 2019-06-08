#!/usr/bin/env python3
# -*- coding: utf8 -*-

# Gaik Tamazian, 2018
# mail (at) gtamazian (dot) com

"""Implements launcher for the `promptpy optimize` command.

Optimize a transformation model by local optimization of the transformation
cost objective function.
"""

import argparse
import logging
import numpy as np
import sys
from scipy.optimize import minimize
from .. import model

assert sys.version_info >= (3, 6), 'Python 3.6 or higher required'


def create_parser(parser):
    """
    Create a parser for the `optimize` command.
    """
    assert isinstance(parser, argparse.ArgumentParser), \
        'argparse.ArgumentParser required'

    parser.add_argument('input', help='input transformation model file')
    parser.add_argument('p_num', type=int,
                        help='number of optimized planar angles')
    parser.add_argument('t_num', type=int,
                        help='number of optimized torsion angles')
    parser.add_argument('output', help='output transformation model file')
    parser.add_argument('-i', '--iterations', type=int, default=1000,
                        help='number of optimization iterations')


def optimize(tr_model, p_num, t_num, n_iter):
    """
    Optimize the transformation model using the local optimization algorithm
    with the specified number of iterations and return the optimized model.
    """
    p_indices = np.array(tr_model.get_ordered_angles().planar[:p_num])
    t_indices = np.array(tr_model.get_ordered_angles().torsion[:t_num])

    f = tr_model.get_obj_func(p_indices, t_indices, gradient=True)
    x_initial = tr_model.get_angles(p_indices, t_indices)

    result = minimize(f, x_initial, method='L-BFGS-B', jac=True,
                      options={'maxiter': n_iter})

    tr_model.set_angles(p_indices, t_indices, result.x)
    return tr_model


def launcher(args):
    """
    Given argument parser results, launch the `optimize` command.
    """
    assert isinstance(args, argparse.Namespace), \
            'argparse.Namespace required'

    model.logger.setLevel(logging.INFO)

    m = model.read_hdf5(args.input)
    m = optimize(m, args.p_num, args.t_num, args.iterations)
    model.write_hdf5(m, args.output)
