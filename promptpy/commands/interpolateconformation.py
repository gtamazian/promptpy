import argparse
import sys
from .. import model


def create_parser(parser):
    assert isinstance(parser, argparse.ArgumentParser), \
        'argparse.ArgumentParser required'

    parser.add_argument('input', help='input file name')
    parser.add_argument('conf_num', type=int, help='conformation number')
    parser.add_argument('output', help='output file name')


def launcher(args):
    assert isinstance(args, argparse.Namespace), \
        'argparse.Namespace required'

    m = model.read_hdf5(args.input)
    m.interpolate_conformation(args.conf_num - 1)
    m.update_rotations()
    model.write_hdf5(m, args.output)
