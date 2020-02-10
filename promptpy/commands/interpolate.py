#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gaik Tamazian, 2017-2018
# mail (at) gtamazian (dot) com

"""Implements launcher for the `promptpy interpolate` command.

Create the transformation model from a pair of PDB files by interpolating their
internal coordinates.
"""

import argparse
import sys
from Bio.PDB.PDBParser import PDBParser
from .. import model
from .. import pdb

assert sys.version_info >= (3, 5), 'Python 3.5 or higher required'


def create_parser(parser):
    """
    Create a parser for the `interpolate` command.
    """
    assert isinstance(parser, argparse.ArgumentParser), \
        'argparse.ArgumentParser required'

    parser.add_argument('pdb1', help='first PDB file')
    parser.add_argument('pdb2', help='second PDB file')
    parser.add_argument('nconf', type=int,
                        help='number of configurations')
    parser.add_argument('output_file', help='output JSON file')

    parser.add_argument('-m1', '--model1', type=int,
                        help='model number from first PDB file')
    parser.add_argument('-m2', '--model2', type=int,
                        help='model number from second PDB file')


def interpolate(pdb_struct_1, pdb_struct_2, n_conf, output_file):
    """
    Given two PDB structures, construct a transformation model from them by
    interpolating the structure internal coordinates.
    """
    atom_set_1 = pdb.get_atom_set(pdb_struct_1)
    atom_set_2 = pdb.get_atom_set(pdb_struct_2)
    assert atom_set_1 == atom_set_2, 'atoms sets must be the same'

    assert atom_set_1 == {'CA'} or atom_set_1 == {'N', 'CA', 'C'}, \
        'atom set must be {CA} or {CA, C, N}'

    assert pdb.get_atom_list(pdb_struct_1) == \
        pdb.get_atom_list(pdb_struct_2), 'atoms must be the same'
    assert pdb.get_residue_list(pdb_struct_1) == \
        pdb.get_residue_list(pdb_struct_2), 'residues must be the same'

    x1 = pdb.get_atom_coordinates(pdb_struct_1)[0]
    x2 = pdb.get_atom_coordinates(pdb_struct_2)[0]
    w = pdb.get_atom_masses(pdb_struct_1, True)

    assert n_conf > 2, 'more than two configurations required'
    m = model.from_marg_coords(x1, x2, w, n_conf-2)
    model.write_hdf5(m, output_file)


def launcher(args):
    """
    Given argument parser results, launch the `interpolate` command.
    """
    assert isinstance(args, argparse.Namespace), \
        'argparse.Namespace required'

    p = PDBParser(PERMISSIVE=1)
    s1 = p.get_structure('STRUCT1', args.pdb1)
    s2 = p.get_structure('STRUCT2', args.pdb2)

    if args.model1 is not None:
        s1 = pdb.extract_model(s1, args.model1 - 1)
    if args.model2 is not None:
        s2 = pdb.extract_model(s2, args.model2 - 1)

    interpolate(s1, s2, args.nconf, args.output_file)
