#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Gaik Tamazian, 2017-2018
# mail (at) gtamazian (dot) com

"""Containts routines related to visualizing transformation models.

The routines include functions to produce various plots, including the plot of
circular differences between conformation angles, the plot of RMSDs between
adjacent configurations, the plot of RMSDs to a fixed configuration and the
plot of minimal interatomic distances.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from . import geometry as geo

assert sys.version_info >= (3, 5), 'Python 3.5 or higher required'


def plot_rmsd_adj_conf(models, labels=None, width=10, height=4, dpi=300):
    """
    Plot RMSDs between adjacent configurations of given models.
    """
    assert isinstance(models, list), 'list of transformation models required'

    plt.figure(figsize=(width, height), dpi=dpi)
    plt.xlabel('Configuration Pair')
    plt.ylabel('RMSD in Angstroms')
    plt.grid()

    conf_number = models[0].n_conf()

    conf_pair_label = []
    for i, j in zip(range(1, conf_number), range(2, conf_number + 1)):
        conf_pair_label.append(str(i) + '-' + str(j))

    plt.xticks(range(conf_number - 1), conf_pair_label)
    plt.xlim(-0.5, conf_number - 1.5)

    if labels is None:
        labels = [str(i+1) for i in range(len(models))]

    plotted_values = np.empty((len(models), conf_number - 1))
    for i in range(len(models)):
        cur_rmsd = models[i].rmsd()
        plt.plot(cur_rmsd, '-p', label=labels[i])
        plotted_values[i, :] = cur_rmsd

    max_rmsd, min_rmsd = np.max(plotted_values), np.min(plotted_values)
    rmsd_range = max_rmsd - min_rmsd
    plt.ylim(min_rmsd - rmsd_range * 0.1, max_rmsd + rmsd_range * 0.3)
    if len(models) > 1:
        plt.legend()
    return plotted_values


def plot_rmsd_fixed_conf(models, conf_num=0, labels=None, width=10, height=4,
                         dpi=300):
    """
    Plot RMSDs to a fixed configuration for given models.
    """
    assert isinstance(models, list), 'list of transformation models required'

    plt.figure(figsize=(width, height), dpi=dpi)
    plt.xlabel('Configuration Number')
    plt.ylabel('RMSD in Angstroms')
    plt.grid()

    conf_number = models[0].n_conf()

    conf_num_label = [str(i+1) for i in range(conf_number)]

    plt.xticks(range(conf_number), conf_num_label)
    plt.xlim(-0.5, conf_number - 0.5)

    if labels is None:
        labels = [str(i+1) for i in range(len(models))]

    plotted_values = np.empty((len(models), conf_number))
    for i in range(len(models)):
        conf_rmsd = models[i].rmsd_to_conf(conf_num)
        plt.plot(conf_rmsd, '-p', label=labels[i])
        plotted_values[i, :] = conf_rmsd

    max_rmsd, min_rmsd = np.max(plotted_values), np.min(plotted_values)
    rmsd_range = max_rmsd - min_rmsd
    plt.ylim(min_rmsd - rmsd_range * 0.1, max_rmsd + rmsd_range * 0.3)
    if len(models) > 1:
        plt.legend()
    return plotted_values


def plot_min_interatomic_dist(models, labels=None, width=10, height=4,
                              dpi=300):
    """
    Plot minimal interatomic distances within each configuration for given
    models.
    """
    assert isinstance(models, list), 'list of transformation models required'

    plt.figure(figsize=(width, height), dpi=dpi)
    plt.xlabel('Configuration Number')
    plt.ylabel('Minimal Interatomic Distance in Angstroms')
    plt.grid()

    conf_number = models[0].n_conf()

    conf_num_label = [str(i+1) for i in range(conf_number)]

    plt.xticks(range(conf_number), conf_num_label)
    plt.xlim(-0.5, conf_number - 0.5)

    if labels is None:
        labels = [str(i+1) for i in range(len(models))]

    plotted_values = np.empty((len(models), conf_number))
    for i in range(len(models)):
        min_dist = models[i].min_interatomic_dist()
        plt.plot(min_dist, '-p', label=labels[i])
        plotted_values[i, :] = min_dist

    max_rmsd, min_rmsd = np.max(plotted_values), np.min(plotted_values)
    rmsd_range = max_rmsd - min_rmsd
    plt.ylim(min_rmsd - rmsd_range * 0.1, max_rmsd + rmsd_range * 0.3)
    if len(models) > 1:
        plt.legend()
    return plotted_values


def plot_circ_dist(model, width=10, height=4, dpi=300, angle_type='torsion',
                   start=None, end=None, linetype='-', angle_num=None,
                   angle_threshold=None, diff_threshold=None,
                   use_offset=False, torsion_type='all'):
    """
    Plot absolute circular distances between torsion or planar angles of the
    first and last configurations of a given transformation model.
    """
    assert angle_type in {'planar', 'torsion'}, 'incorrect angle type'
    assert torsion_type in {'all', 'phi', 'psi', 'omega'}, \
        'incorrect torsion angle type'

    plt.figure(figsize=(width, height), dpi=dpi)
    plt.xlabel(' '.join((angle_type.capitalize(),
                         torsion_type.capitalize()
                         if torsion_type != 'all' else '',
                         'Angle Rank')))
    plt.ylabel('Circular Difference Abs Value in Degrees')
    plt.grid()

    if angle_type == 'torsion':
        if torsion_type == 'all':
            dist_values = geo.circ_dist(model.gamma[0, :], model.gamma[-1, :])
        elif torsion_type == 'phi':
            dist_values = geo.circ_dist(model.gamma[0, 2::3],
                                        model.gamma[-1, 2::3])
        elif torsion_type == 'psi':
            dist_values = geo.circ_dist(model.gamma[0, 0::3],
                                        model.gamma[-1, 0::3])
        elif torsion_type == 'omega':
            dist_values = geo.circ_dist(model.gamma[0, 1::3],
                                        model.gamma[-1, 1::3])
    else:
        dist_values = geo.circ_dist(model.alpha[0, :], model.alpha[-1, :])

    dist_values = 180*np.abs(dist_values)/np.pi
    n = np.size(dist_values)

    if start is None:
        start = 0
    if end is None:
        end = n
    dist_values = dist_values[start:end]

    if angle_num is not None:
        dist_values = np.sort(dist_values)[::-1]
        dist_values = dist_values[:angle_num]
        end = start + angle_num

    if use_offset:
        plt.xlim(start - 0.5, end - 0.5)
        plt.xticks(range(start, end), map(str, range(start + 1, end + 1)))
        plotted_value_range = np.ptp(dist_values)
        plt.ylim(np.min(dist_values) - plotted_value_range * 0.1,
                 np.max(dist_values) + plotted_value_range * 0.1)

    plt.plot(range(start, end), dist_values, linetype)

    if angle_threshold:
        plt.axvline(x=angle_threshold, linewidth=2, color='black',
                    linestyle='dashed')

    if diff_threshold:
        plt.axhline(y=diff_threshold, linewidth=2, color='black',
                    linestyle='dashed')
