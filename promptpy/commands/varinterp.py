#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gaik Tamazian, 2017-2018
# mail (at) gtamazian (dot) com

"""Implements launcher for the `promptpy varinterp` command.

Vary circular interpolation directions for torsion angles of a transformation
model.
"""

import argparse
import copy
import numpy as np
import sys
from tqdm import tqdm
from .. import geometry as geo
from .. import model

assert sys.version_info >= (3, 6), 'Python 3.6 or higher required'


def create_parser(parser):
    """
    Create a parser for the `varinterp` command.
    """
    assert isinstance(parser, argparse.ArgumentParser), \
        'argparse.ArgumentParser required'

    parser.add_argument('input_file', help='input JSON transformation file')
    parser.add_argument('output_prefix', help='prefix for output files')
    parser.add_argument('angle_indices', type=int, nargs='+',
                        help='indices of torsion angles to variate '
                        'interpolation directions')
    parser.add_argument('-p', '--progressbar', action='store_true',
                        help='show progress bar')


def variate_interpolation_directions(m, angle_indices, output_prefix,
                                     show_progress_bar=False):
    """
    Given a transformation model and its torsion angle indices, generate models
    from it by variating circular interpolation directions of the specified
    angles and write results to JSON transformation files with the specified
    prefix.
    """
    n_angles = np.size(angle_indices)
    angle_indices -= 1
    for i in tqdm(range(2**n_angles), disable=not show_progress_bar):
        cur_model = copy.deepcopy(m)
        mask = np.array([j == '1' for j in bin(i)[2:].zfill(n_angles)])
        cur_model.gamma[:, angle_indices[mask]] = np.transpose(geo.circ_interp(
            cur_model.gamma[0, angle_indices[mask]],
            cur_model.gamma[-1, angle_indices[mask]],
            cur_model.n_conf() - 2,
            long_arc=True
        ))
        output_filename = '{}_{:04}.hdf5'.format(output_prefix, i+1)
        model.write_hdf5(cur_model, output_filename)


def launcher(args):
    """
    Given argument parser results, launch the `varinterp` command.
    """
    assert isinstance(args, argparse.Namespace), \
        'argparse.Namespace required'

    m = model.read_hdf5(args.input_file)
    variate_interpolation_directions(m, np.array(args.angle_indices),
                                     args.output_prefix, args.progressbar)
