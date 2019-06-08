#!/usr/bin/env python3
# -*- coding: utf8 -*-

# Gaik Tamazian, 2018
# mail (at) gtamazian (dot) com

"""
Interpolate angles in a transformation following a reference.

For every torsion or planar angle in a tranformation model there are two
directions of circular interpolation between its values in first and
last conformations of the transformation: by the short or long arc. One
may choose the directions using a given transformation as a reference.
This script implements a command that accepts a transformation and
outputs the transformation of the same protein but with interpolated
planar and torsion angles.
"""

import argparse
import numpy as np
import sys

from .. import geometry as geo
from .. import model

assert sys.version_info >= (3, 5), "Python 3.5 or higher required"


def creaser_parser(parser):
    assert isinstance(parser, argparse.ArgumentParser), \
            "argparse.ArgumentParser required"

    parser.add_argument("input_file", help="input transformation")
    parser.add_argument("output_file", help="output transformation")


def interpolate_angles(mdl):
    """
    Interpolate planar and torsion angles of a transformation model,
    choosing the circular interpolation direction closest to the
    original values of the angles.
    """
    n_conf = mdl.n_conf()

    # planar angles

    # sa and la stand for 'short arc' and 'long arc'
    sa = geo.circ_interp(mdl.alpha[0,:], mdl.alpha[-1,:], n_conf - 2)
    la = geo.circ_interp(mdl.alpha[0,:], mdl.alpha[-1,:], n_conf - 2,
                               long_arc=True)

    sa, la = np.transpose(sa), np.transpose(la)

    # sd and ld stand for 'short distance' and 'long distance'
    sd = np.sum(np.abs(geo.circ_dist(mdl.alpha, sa)), axis=0)
    ld = np.sum(np.abs(geo.circ_dist(mdl.alpha, la)), axis=0)

    n_alpha = np.shape(mdl.alpha)[1]
    for k in range(n_alpha):
        if ld[k] < sd[k]:
            mdl.alpha[:,k] = la[:,k]
            print("%s\t%d\t%s" % ("ALPHA", k, "LONG"))
        else:
            mdl.alpha[:,k] = sa[:,k]
            print("%s\t%d\t%s" % ("ALPHA", k, "SHORT"))

    # torsion angles
    sa = geo.circ_interp(mdl.gamma[0,:], mdl.gamma[-1,:], n_conf - 2)
    la = geo.circ_interp(mdl.gamma[0,:], mdl.gamma[-1,:], n_conf - 2,
                               long_arc=True)

    sa, la = np.transpose(sa), np.transpose(la)

    # sd and ld stand for 'short distance' and 'long distance'
    sd = np.sum(np.abs(geo.circ_dist(mdl.gamma, sa)), axis=0)
    ld = np.sum(np.abs(geo.circ_dist(mdl.gamma, la)), axis=0)

    n_gamma = np.shape(mdl.gamma)[1]
    for k in range(n_gamma):
        if ld[k] < sd[k]:
            mdl.gamma[:,k] = la[:,k]
            print("%s\t%d\t%s" % ("GAMMA", k, "LONG"))
        else:
            mdl.gamma[:,k] = sa[:,k]
            print("%s\t%d\t%s" % ("GAMMA", k, "SHORT"))

    return mdl


def launcher(args):
    assert isinstance(args, argparse.Namespace), \
            "argparse.Namespace required"

    mdl = model.read_hdf5(args.input_file)
    mdl = interpolate_angles(mdl)
    model.write_hdf5(mdl, args.output_file)
