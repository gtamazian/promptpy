#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gaik Tamazian, 2017-2018
# mail (at) gtamazian (dot) com

"""Implements launcher for the `promptpy plotstats` command.

Visualize transformation model statistics: RMSDs between its configurations
and minimal interatomic distances within the configurations.
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
    Create a parser for the `plotstats` command.
    """
    assert isinstance(parser, argparse.ArgumentParser), \
        'argparse.ArgumentParser required'

    parser.add_argument('input_file', nargs='+',
                        help='JSON transformation file')
    parser.add_argument('--width', type=float, default=10,
                        help='figure width in inches')
    parser.add_argument('--height', type=float, default=4,
                        help='figure height in inches')
    parser.add_argument('--dpi', type=int, default=300,
                        help='figure resolution in dpi')
    parser.add_argument('--no_titles', action='store_true',
                        help='do not show plot titles')
    parser.add_argument('-o', '--output', default='plot',
                        help='prefix of output files')


def produce_plots(filenames, output_prefix, width, height, dpi, no_titles):
    """
    Produce plots for specified JSON transformation files.
    """
    models = []
    for i in filenames:
        m = model.read_hdf5(i)
        models.append(m)

    filenames = [os.path.splitext(i)[0] for i in filenames]

    vis.plot_rmsd_adj_conf(models, filenames, width, height, dpi)
    if not no_titles:
        plt.title('RMSDs between Adjacent Configurations')
    plt.savefig(output_prefix + '_adj_rmsd.png')

    vis.plot_rmsd_fixed_conf(models, 0, filenames, width, height, dpi)
    if not no_titles:
        plt.title('RMSDs to First Configuration')
    plt.savefig(output_prefix + '_fixed_rmsd.png')

    vis.plot_min_interatomic_dist(models, filenames, width, height, dpi)
    if not no_titles:
        plt.title('Minimal Interatomic Distances')
    plt.savefig(output_prefix + '_min_dist.png')


def launcher(args):
    """
    Given argument parser results, launch the `plotstats` command.
    """
    assert isinstance(args, argparse.Namespace), \
        'argparse.Namespace required'

    produce_plots(args.input_file, args.output, args.width, args.height,
                  args.dpi, args.no_titles)
