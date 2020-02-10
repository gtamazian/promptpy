#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gaik Tamazian, 2017-2018
# mail (at) gtamazian (dot) com

"""Implements launcher for the `promptpy plotangles` command.

Visualize angles from the specified transformation model.
"""

import argparse
import matplotlib.pyplot as plt
import os.path
import sys
from .. import visualization as vis
from .. import model

assert sys.version_info >= (3, 5), 'Python 3.5 or higher required'


def create_parser(parser):
    """
    Create a parser for the `plotangles` command.
    """
    assert isinstance(parser, argparse.ArgumentParser), \
        'argparse.ArgumentParser required'

    parser.add_argument('input_file', help='JSON transformation file')
    parser.add_argument('angle_type', choices=('planar', 'torsion'),
                        help='angle type to be plotted')
    parser.add_argument('--width', type=float, default=10,
                        help='figure width in inches')
    parser.add_argument('--height', type=float, default=4,
                        help='figure height in inches')
    parser.add_argument('--dpi', type=int, default=300,
                        help='figure resolution in dpi')
    parser.add_argument('-o', '--output', help='output file name')
    parser.add_argument('-l', '--linestyle', default='-',
                        help='line style in Matplotlib notation')
    parser.add_argument('-n', '--number', type=int,
                        help='number of angles to plot; if specified, the '
                        'plotted values are sorted in descending order')
    parser.add_argument('-s', '--start', type=int, default=1,
                        help='start position of model region to plot angles '
                        'from')
    parser.add_argument('-e', '--end', type=int,
                        help='end position of model region to plot angles '
                        'from')
    parser.add_argument('--small', action='store_true',
                        help='optimize plot for small number of plotted '
                        'values')
    parser.add_argument('-a', '--angle_threshold', type=float,
                        help='plot angle number threshold line at specified '
                        'level')
    parser.add_argument('-d', '--diff_threshold', type=float,
                        help='plot angle difference threshold line at '
                        'specified level')
    parser.add_argument('-t', '--torsion_type',
                        choices=('all', 'phi', 'psi', 'omega'), default='all',
                        help='torsion angle type to be plotted')


def launcher(args):
    """
    Given argument parser results, launch the `plotangles` command.
    """
    assert isinstance(args, argparse.Namespace), \
        'argparse.Namespace required'

    m = model.read_hdf5(args.input_file)
    vis.plot_circ_dist(m, args.width, args.height, args.dpi,
                       args.angle_type, args.start - 1, args.end,
                       args.linestyle, args.number, args.angle_threshold,
                       args.diff_threshold, args.small, args.torsion_type)
    output_filename = args.output if args.output is not None else \
        os.path.splitext(args.input_file)[0] + '_angles.png'
    plt.savefig(output_filename)
