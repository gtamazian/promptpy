#!/usr/bin/env python3
# -*- coding: utf8 -*-

# Gaik Tamazian, 2018
# mail (at) gtamazian (dot) com

"""Launcher for the `promptpy updaterotations` command.

The command updates rotation matrices that specify how adjacent configurations
of a transformation model are superposed to each other. The Kabsch
transformation is used to calculate the matrices.
"""

import argparse
import logging
import sys
from .. import geometry as geo
from .. import model

assert sys.version_info >= (3, 6), 'Python 3.6 or higher required'

logging.basicConfig(format='%(asctime)-15s - %(levelname)-8s: '
                    '%(message)s')
logger = logging.getLogger(__name__)


def create_parser(parser):
    assert isinstance(parser, argparse.ArgumentParser), \
        'argparse.ArgumentParser required'

    parser.add_argument('input', help='input file name')
    parser.add_argument('output', help='output file name')


def launcher(args):
    """
    Given the argument parser result, launch the `updaterotations` command.
    """
    assert isinstance(args, argparse.Namespace), \
        'argparse.Namespace required'

    m = model.read_hdf5(args.input)
    m.update_rotations()
    model.write_hdf5(m, args.output)
