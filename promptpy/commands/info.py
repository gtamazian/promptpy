#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gaik Tamazian, 2017-2018
# mail (at) gtamazian (dot) com

"""Implements launcher for the `promptpy info` command.

Print statistics about the specified transformation model.
"""

import argparse
import numpy as np
import statistics
import sys
from tqdm import tqdm
from .. import model

assert sys.version_info >= (3, 6), 'Python 3.6 or higher required'


def create_parser(parser):
    """
    Create a parser for the `info` command.
    """
    assert isinstance(parser, argparse.ArgumentParser), \
        'argparse.ArgumentParser required'

    parser.add_argument('input_file', nargs='+',
                        help='JSON transformation file')
    parser.add_argument('--header', action='store_true',
                        help='print header')
    parser.add_argument('-p', '--progressbar', action='store_true',
                        help='show progress bar')


def print_info_header(first_len):
    """
    Print transformation information header.
    """
    fmt_line = '{:>' + str(first_len) + \
        '}\t{:>7}\t{:>6}\t{:>12}\t{:>9}\t{:>13}\t{:>14}'
    print(fmt_line.format(
        'Filename', '# Atoms', '# Conf', 'Cost', 'Mean RMSD', 'Min dist conf',
        'Min dist value'
    ))


def print_transformation_info(first_len, name, model):
    """
    Given a transformation model, print its info line.
    """
    fmt_line = '{:>' + str(first_len) + \
        '}\t{:>7}\t{:>6}\t{:>12.5e}\t{:>9.2e}\t{:>13d}\t{:>14.2f}'
    min_dist = np.array(model.min_interatomic_dist())
    print(fmt_line.format(
        name, model.n_atom(), model.n_conf(), model.cost().cost,
        statistics.mean(model.rmsd()), np.argmin(min_dist), np.min(min_dist)
    ))


def launcher(args):
    """
    Given argument parser results, launch the `info` command.
    """
    assert isinstance(args, argparse.Namespace), \
        'argparse.Namespace required'

    first_len = max(map(len, args.input_file))

    if args.header:
        print_info_header(first_len)

    for filename in tqdm(args.input_file, disable=not args.progressbar):
        m = model.read_hdf5(filename)
        print_transformation_info(first_len, filename, m)
