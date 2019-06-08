#!/usr/bin/env python3
# -*- coding: utf8 -*-

# Gaik Tamazian, 2018
# mail (at) gtamazian (dot) com

"""Launcher for the `promptpy convert` command.

The command converts transformation models between the following formats: PDB,
HDF5, and JSON. A reference PDB file is required for conversion to the PDB
format.
"""

import argparse
import h5py
import json
import logging
import os.path
import sys
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBParser import PDBParser
from .. import model
from .. import pdb

assert sys.version_info >= (3, 6), 'Python 3.6 or higher required'

logging.basicConfig(format='%(asctime)-15s - %(levelname)-8s: '
                    '%(message)s')
logger = logging.getLogger(__name__)


def create_parser(parser):
    assert isinstance(parser, argparse.ArgumentParser), \
        'argparse.ArgumentParser required'

    parser.add_argument('input', help='input file name')
    parser.add_argument('--input_fmt',
                        choices={'pdb', 'json', 'hdf5'},
                        help='input file format')
    parser.add_argument('output', help='output file name')
    parser.add_argument('--output_fmt',
                        choices={'pdb', 'json', 'hdf5'},
                        help='output file format')
    parser.add_argument('--pdb', help='reference PDB file')


def load_model(model_fname, model_fmt):
    """
    Load a transformation model from the file of the specified format.

    If something goes wrong, the function returns None, otherwise the loaded
    transformation model is returned.
    """
    result = None

    if model_fmt == 'pdb':
        parser = PDBParser(PERMISSIVE=True)
        struct = parser.get_structure('PROMPTPY', model_fname)
        if not pdb.is_transformation(struct):
            logger.error('specified PDB file is not a transformation')
            return None
        x = pdb.get_atom_coordinates(struct)
        w = pdb.get_atom_masses(pdb.extract_model(struct, 0), True)
        result = model.from_conf_coords(x, w)
    elif model_fmt == 'json':
        with open(model_fname) as f:
            result = model.Transformation.from_dict(json.load(f))
    elif model_fmt == 'hdf5':
        result = model.read_hdf5(model_fname)
    else:
        logger.error('unknown model format %s', model_fmt)

    return result


def save_model(mdl, model_fname, model_fmt, pdb_fname=None):
    """
    Save a transformation model to the file in the specified format. For saving
    the model in the PDB format, a reference PDB file is required. The
    reference PDB file name is specified in the optional argument.

    If something goes wrong, the function returns False, otherwise True is
    returned.
    """
    if model_fmt == 'pdb':
        if pdb_fname is None:
            logger.error('missing the reference PDB file name')
            return False
        parser = PDBParser(PERMISSIVE=True)
        ref_struct = parser.get_structure('PROMPTPY', pdb_fname)
        if len(ref_struct) == 1:
            logger.debug('multiplying models in the reference PDB structure')
            ref_struct = pdb.multiply_model(ref_struct, mdl.n_conf())
        elif len(ref_struct) != mdl.n_conf():
            logger.warning('inconsistent numbers of models in the reference '
                           'PDB structure and the transformation, '
                           'multiplying the first model from the reference '
                           'structure')
            ref_struct = pdb.multiply_model(pdb.extract_model(ref_struct, 0),
                                            mdl.n_conf())

        pdb.update_structure(ref_struct, mdl.get_cart_coords()[0])

        io = PDBIO()
        io.set_structure(ref_struct)
        io.save(model_fname)
    elif model_fmt == 'json':
        with open(model_fname, 'w') as f:
            json.dump(dict(mdl), f, indent=4)
    elif model_fmt == 'hdf5':
        model.write_hdf5(mdl, model_fname)
    else:
        logger.error('unknown model format %s', model_fmt)

    return True


def launcher(args):
    """
    Given the argument parser result, launch the `convert` command.
    """
    assert isinstance(args, argparse.Namespace), \
        'argparse.Namespace required'

    formats = {'pdb', 'hdf5', 'json'}

    # if input or output formats are not specified, try to infer them from
    # extensions of the input and output files
    if args.input_fmt is None:
        input_fmt = os.path.splitext(args.input)[1][1:]
        if input_fmt not in formats:
            logger.error('incorrect inferred input file format %s',
                         input_fmt)
            sys.exit(1)
    else:
        input_fmt = args.input_fmt

    if args.output_fmt is None:
        output_fmt = os.path.splitext(args.output)[1][1:]
        if output_fmt not in formats:
            logger.error('incorrect inferred output file format %s',
                         output_fmt)
            sys.exit(1)
    else:
        output_fmt = args.output_fmt

    m = load_model(args.input, input_fmt)
    if m is None:
        sys.exit(1)

    if not save_model(m, args.output, output_fmt, args.pdb):
        sys.exit(1)
