#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gaik Tamazian, 2017-2018
# mail (at) gtamazian (dot) com

import argparse
import sys
from .commands import angleindices
from .commands import convert
from .commands import info
from .commands import interpolate
from .commands import optimize
from .commands import plotangles
from .commands import plotstats
from .commands import refinterp
from .commands import strip
from .commands import updaterotations
from .commands import varinterp
from . import __version__

assert sys.version_info >= (3, 5), 'Python 3.5 or higher required'


def cli():
    """
    Command-line interface for the prompt tool.
    """
    parser = argparse.ArgumentParser(
        prog='promptpy',
        description='Protein conformational motion prediction toolbox.'
    )
    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    subparsers = parser.add_subparsers(
        dest='command',
    )
    subparsers.required = True

    # command subparsers are listed below
    parser_angleindices = subparsers.add_parser(
        'angleindices',
        help='print absolute circular distances between transformation angles')
    parser_convert = subparsers.add_parser(
        'convert',
        help='convert a transformation model between various file formats')
    parser_interpolate = subparsers.add_parser('interpolate',
                                               help='create a model from a '
                                               'pair of PDB files by '
                                               'interpolating in internal '
                                               'coordinates')
    parser_info = subparsers.add_parser(
        'info',
        help='print information about JSON transformation files')
    parser_optimize = subparsers.add_parser(
        'optimize',
        help='perform local optimization of a transformation')
    parser_plotangles = subparsers.add_parser(
        'plotangles',
        help='plot circular distances between transformation model angles'
    )
    parser_plotstats = subparsers.add_parser(
        'plotstats',
        help='plot statistics for JSON transformation files')
    parser_refinterp = subparsers.add_parser(
        'refinterp',
        help='interpolate angles using a transformation as a reference')
    parser_strip = subparsers.add_parser('strip', help='preprocess a PDB file')
    parser_updaterotations = subparsers.add_parser(
        'updaterotations', help='update rotations in a transformation model')
    parser_varinterp = subparsers.add_parser('varinterp', help='variate angle '
                                             'interpolation directions')

    angleindices.create_parser(parser_angleindices)
    convert.create_parser(parser_convert)
    info.create_parser(parser_info)
    interpolate.create_parser(parser_interpolate)
    strip.create_parser(parser_strip)
    optimize.create_parser(parser_optimize)
    plotangles.create_parser(parser_plotangles)
    plotstats.create_parser(parser_plotstats)
    updaterotations.create_parser(parser_updaterotations)
    varinterp.create_parser(parser_varinterp)
    refinterp.creaser_parser(parser_refinterp)

    # the dictionary below contains command launchers
    launchers = {
        'angleindices': angleindices.launcher,
        'convert': convert.launcher,
        'info': info.launcher,
        'interpolate': interpolate.launcher,
        'optimize': optimize.launcher,
        'plotangles': plotangles.launcher,
        'plotstats': plotstats.launcher,
        'refinterp': refinterp.launcher,
        'strip': strip.launcher,
        'updaterotations': updaterotations.launcher,
        'varinterp': varinterp.launcher
    }

    # parse command-line arguments, choose the appropriate launcher and run it
    # withe the specified arguments
    args = parser.parse_args()
    launchers[args.command](args)


if __name__ == '__main__':
    cli()
