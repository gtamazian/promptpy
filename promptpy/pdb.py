#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gaik Tamazian, 2017-2018
# mail (at) gtamazian (dot) com

"""Contains routines related to Protein Data Bank (PDB) format.

The routines in this module include functions to read and write protein atom
coodinates. The functions work with atom coordinates as numpy arrays.
"""

import numpy as np
import sys
from Bio.PDB.Structure import Structure


assert sys.version_info >= (3, 5), 'Python 3.5 or higher required'

# average masses of amino acids residues from
# http://www.matrixscience.com/help/aa_help.html
residue_mass = dict(ALA=71.0779,
                    ARG=156.1857,
                    ASN=114.1026,
                    ASP=115.0874,
                    CYS=103.1429,
                    GLU=129.114,
                    GLN=128.1292,
                    GLY=57.0513,
                    HIS=137.1393,
                    ILE=113.1576,
                    LEU=113.1576,
                    LYS=128.1723,
                    MET=131.1961,
                    PHE=147.1739,
                    PRO=97.1152,
                    SER=87.0773,
                    THR=101.1039,
                    SEC=150.0379,
                    TRP=186.2099,
                    TYR=163.1733,
                    VAL=99.1311)


# atomic masses
am = dict(H=1.008, C=12.0107, N=14.0067, O=15.9994)


def get_atom_coordinates(pdb_struct, chain='A'):
    """
    Given a PDB structure, extract coordinates of its atoms.
    """
    return list(np.array(list(
        atom.get_coord() for atom in model[chain].get_atoms()))
        for model in pdb_struct)


def update_structure(pdb_struct, new_coords):
    """
    Given a PDB structure, update its atom coordinates.
    """
    assert len(pdb_struct) == len(new_coords), 'inconsistent number of models'

    # now we assume that each model contains a single chain which atom
    # coordinates are updated; this assumption may be changed further
    assert is_single_chain(pdb_struct), 'single-chain structure required'

    for i in range(len(pdb_struct)):
        atom_num = 0
        cur_chain = pdb_struct[i].child_list[0]
        for j in range(len(cur_chain)):
            cur_residue = cur_chain.child_list[j]
            for k in range(len(cur_residue)):
                # j and k iterate through residues and atoms, respectively
                cur_residue.child_list[k].set_coord(new_coords[i, atom_num, :])
                atom_num += 1


def is_single_chain(pdb_struct):
    """
    Check if every model of a PDB structure contains a single chain.
    """
    for model in pdb_struct:
        if len(model) > 1:
            return False
    return True


def is_transformation(pdb_struct):
    """
    Check if a PDB structure represents a transformation.
    """
    # for now, we consider only single-chain transformations
    if not is_single_chain(pdb_struct):
        return False

    atom_num = []
    for model in pdb_struct:
        atom_num.append(len(model.child_list[0]))

    return (len(atom_num) > 1) and all(x == atom_num[0] for x in atom_num)


def multiply_model(pdb_struct, num_models):
    """
    Given a single-model PDB structure, multiply that model.
    """
    assert len(pdb_struct) == 1, 'single-model PDB file required'

    new_struct = Structure(pdb_struct.id)

    for i in range(num_models):
        new_model = pdb_struct[0].copy()
        new_model.detach_parent()
        new_model.id = i
        new_model.serial_num = i + 1
        new_struct.add(new_model)
        new_model.set_parent(new_struct)

    return new_struct


def extract_model(pdb_struct, k):
    """
    Extract a model from the given PDB structure.
    """
    assert k < len(pdb_struct), 'missing specified model'

    new_struct = Structure(pdb_struct.id)
    new_model = pdb_struct[k].copy()
    new_model.id = 0
    new_model.serial_num = 1
    new_struct.add(new_model)

    return new_struct


def get_atom_set(pdb_struct):
    """
    Given a PDB structure, return a set of names of its atoms.
    """
    return set(atom.get_name() for model in pdb_struct for chain in model
               for residue in chain for atom in residue)


def get_atom_list(pdb_struct):
    """
    Given a PDB structure, return a list of names of its atoms.
    """
    return list(atom.get_name() for model in pdb_struct for chain in model
                for residue in chain for atom in residue)


def get_residue_list(pdb_struct):
    """
    Given a PDB structure, return a list of names of its residues.
    """
    return list(residue.resname for model in pdb_struct for chain in model
                for residue in chain)


def get_atom_masses(pdb_struct, extended=False):
    """
    Given a single-model PDB, return an array of its atom masses.

    If the `extended` flag is specified and the PDB structure contains only
    alpha-carbon or backbone atoms, then masses of other protein atoms,
    including residue atoms, are added to the masses of atoms given in the
    structure.
    """
    assert len(pdb_struct) == 1, 'single-model PDB structure required'

    atom_set = get_atom_set(pdb_struct)
    if extended and (atom_set == {'CA'} or atom_set == {'N', 'CA', 'C'}):
        residue_names = list(atom.get_parent().get_resname()
                             for model in pdb_struct for chain in model
                             for residue in chain for atom in residue)
        if atom_set == {'CA'}:
            result = list(residue_mass[i] for i in residue_names)
        else:
            result = [0] * len(residue_names)
            # we have the full-backbone model
            for i in range(0, len(residue_names), 3):
                # N + H
                result[i] = am['N'] + am['H']
            for i in range(1, len(residue_names), 3):
                # remove NH and CO from the residue mass
                result[i] = residue_mass[residue_names[i]] - \
                    (am['H'] + am['N'] + am['O'] +
                     am['C'])
            for i in range(2, len(residue_names), 3):
                # C + O
                result[i] = am['C'] + am['O']
        result[0] += am['H']  # H at the N-terminus
        result[-1] += am['H'] + am['O']  # OH at the C-terminus
    else:
        result = list(atom.mass for model in pdb_struct for chain in model
                      for residue in chain for atom in residue)

    return np.array(result)
