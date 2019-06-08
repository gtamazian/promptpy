#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gaik Tamazian, 2017-2018
# mail (at) gtamazian (dot) com

"""Implements launcher for the `promptpy strip` command.

Reduce contents of a PDB file, removing entries unrelated to creating a
transformation model.
"""

import argparse
import tempfile
import warnings
import sys

assert sys.version_info >= (3, 6), 'Python 3.6 or higher required'


def create_parser(parser):
    """
    Create a parser for the `strip` command.
    """
    assert isinstance(parser, argparse.ArgumentParser), \
        'argparse.ArgumentParser required'

    parser.add_argument('input_pdb', help='input PDB file')
    parser.add_argument('output_pdb', help='output PDB file')
    parser.add_argument('-a', '--atoms',
                        choices=('all', 'alpha-carbons', 'backbone'),
                        default='alpha-carbons',
                        help='atoms to remain in output file')
    parser.add_argument('-m', '--modelno', type=int, default=0,
                        help='only model with the specified number will '
                        'be written to output file')
    parser.add_argument('-p', '--preserve', action='store_true',
                        help='preserve PDB records unrelated to atoms')


def all_atom_filter(line):
    return True


def alpha_carbon_atom_filter(line):
    return line.split()[2] == 'CA'


def backbone_atom_filter(line):
    return line.split()[2] in {'N', 'C', 'CA'}


def strip_pdb_file(input_pdb, output_pdb, filter_func, preserve_nonatom):
    """
    Strip a given PDB file and write the result to the specified output file.
    """
    pdb_tags = {'HEADER', 'TITLE', 'NUMMDL', 'MODEL', 'TER', 'ENDMDL', 'END'}
    for line in input_pdb:
        if line.startswith('ATOM'):
            if filter_func(line):
                output_pdb.write(line)
        elif line.split()[0] in pdb_tags or preserve_nonatom:
            output_pdb.write(line)


def extract_pdb_model(input_pdb, output_pdb, modelno):
    """
    Extract a model to a separate PDB file.
    """
    assert modelno > 0, 'model number must be positive'

    pdb_tags = {'HEADER', 'TITLE', 'TER', 'END'}
    write_atoms = False
    for line in input_pdb:
        if line.startswith('MODEL') and int(line.split()[1]) == modelno:
            write_atoms = True
            output_pdb.write(line)
        elif line.startswith('ATOM') and write_atoms:
            output_pdb.write(line)
        elif line.startswith('ENDMDL') and write_atoms:
            # we have written the required model to the output file, so now we
            # write the 'END' record and exit the function
            output_pdb.write(line)
            output_pdb.write('END   ')
            return
        elif line.startswith('NUMMDL'):
            # the output file will contain a single model, so this record's
            # value must be 1
            output_pdb.write('NUMMDL    1   ')
        elif line.split()[0] in pdb_tags:
            output_pdb.write(line)
    if not write_atoms:
        warnings.warn('the specified model was not found')
        sys.exit(1)


def launcher(args):
    """
    Given argument parser results, launch the `strip` command.
    """
    assert isinstance(args, argparse.Namespace), \
        'argparse.Namespace required'

    filter_func_dict = {
        'all': all_atom_filter,
        'alpha-carbons': alpha_carbon_atom_filter,
        'backbone': backbone_atom_filter
    }
    filter_func = filter_func_dict[args.atoms]

    if args.modelno > 0:
        # first, we filter the whole PDB file to the temporary one
        temp_file = tempfile.TemporaryFile(mode='w+')
        with open(args.input_pdb) as input_file:
            strip_pdb_file(input_file, temp_file, filter_func, args.preserve)
        # next, we extract the specified model from the temporary file
        temp_file.seek(0)
        with open(args.output_pdb, 'w') as output_file:
            extract_pdb_model(temp_file, output_file, args.modelno)
    else:
        with open(args.input_pdb) as input_file:
            with open(args.output_pdb, 'w') as output_file:
                strip_pdb_file(input_file, output_file, filter_func,
                               args.preserve)
