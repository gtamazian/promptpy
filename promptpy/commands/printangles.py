#!/usr/bin/env python3

"""Implements launcher for the 'promptpy printangles' command."""

import argparse
import sys

import numpy as np

from .. import model


def create_parser(parser):
    """
    Create a parser for the 'promptpy printangles' command.
    """
    assert isinstance(parser, argparse.ArgumentParser), \
        'argparse.ArgumentParser required'

    parser.add_argument('input_file', help='transformation file')
    parser.add_argument('angle_type', choices=('planar', 'torsion'),
                        help='angle type to be plotted')


def launcher(args):
    """
    Given parsed arguments, launch the printangles command.
    """
    assert isinstance(args, argparse.Namespace), \
        'argparse.Namespace required'

    m = model.read_hdf5(args.input_file)
    np.savetxt(sys.stdout,
               np.transpose(
                   m.alpha if args.angle_type == 'planar' else m.gamma) *
               180 / np.pi)
