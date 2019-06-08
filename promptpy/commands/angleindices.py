#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gaik Tamazian, 2017-2018
# mail (at) gtamazian (dot) com

"""Implements launcher for the `promptpy angleindices` command.

The command returns a list of planar or torsion angle indices sorted by the
absolute circular difference between their values in the first and last
configurations of a transformation.
"""

import argparse
import math
import numpy as np
import sys
from .. import geometry as geo
from .. import model

assert sys.version_info >= (3, 6), 'Python 3.6 or higher required'


def create_parser(parser):
    """
    Create a parser for the `angleindices` command.
    """
    assert isinstance(parser, argparse.ArgumentParser), \
        'argparse.ArgumentParser required'

    parser.add_argument('input_file', help='input JSON transformation file')
    parser.add_argument('angle_type', choices=('torsion', 'planar'),
                        help='angle type')
    parser.add_argument('n_angles', type=int, help='number of angles')

    parser.add_argument('-s', '--start', type=int,
                        help='start of the region to report angles from')
    parser.add_argument('-e', '--end', type=int,
                        help='end of the region to report angles from')
    parser.add_argument('-t', '--tabular', action='store_true',
                        help='output results in a tabular from with angle '
                        'difference values')
    parser.add_argument('--types', action='store_true',
                        help='show torsion angle types and residue numbers')


def print_angle_indices(angles_1, angles_2, n_angles, start, end,
                        is_tabular=False, show_tors_angle_types=False):
    """
    Given a transformation model, print indices of angles with greatest
    absolute circular distances between its first and last configurations
    according to the specified arguments.
    """
    dist = np.abs(geo.circ_dist(angles_1[start:end],
                                angles_2[start:end]))
    indices = np.argsort(dist)[::-1][:n_angles]

    angle_type = ('phi', 'psi', 'omega')

    if is_tabular:
        print('{:>3}\t{:>5}\t'.format('#', 'Index'), end='')
        if show_tors_angle_types:
            print('{:>10}\t'.format('Angle type'), end='')
        print('{:>6}'.format('Value'))
        for i in range(min(n_angles, np.size(angles_1))):
            index = indices[i]
            print('{:>3}\t{:>5}\t'.format(i + 1, start + index + 1), end='')
            if show_tors_angle_types:
                print('{:>5}-{:<4}\t'.format(
                    angle_type[(index + 1) % 3], math.ceil((index + 1)/3)),
                    end='')
            print('{:>6.2f}'.format(dist[index] / np.pi * 180.0))
    else:
        print(' '.join(map(str, indices.tolist())))


def launcher(args):
    """
    Given argument parser results, launch the `angleindices` command.
    """
    assert isinstance(args, argparse.Namespace), \
        'argparse.Namespace required'

    m = model.read_hdf5(args.input_file)
    if args.angle_type == 'planar':
        angles_1 = m.alpha[0, :]
        angles_2 = m.alpha[-1, :]
    else:
        angles_1 = m.gamma[0, :]
        angles_2 = m.gamma[-1, :]
    start = args.start - 1 if args.start is not None else 0
    end = args.end if args.end is not None else np.alen(angles_1)
    n_angles = np.size(angles_1)
    assert start < end, 'start position must be less than end position'
    assert 0 <= start <= n_angles, 'incorrect start position'
    assert 0 <= end <= n_angles, 'incorrect end position'
    print_angle_indices(angles_1, angles_2, args.n_angles,
                        start, end, args.tabular,
                        args.types and (args.angle_type == 'torsion'))
