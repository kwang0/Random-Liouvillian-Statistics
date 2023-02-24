#!/usr/bin/env python
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
import itertools
import re
import matplotlib
import sys
sys.path.append('..')
import figures_module
import scipy
import numerics
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#fig, axs = figures_module.prepare_standard_figure(ncols=1, nrows=4, sharey=False, sharex=False, aspect_ratio=0.5)
fig, ax = figures_module.prepare_standard_figure(ncols=1, nrows=1, sharey=False, sharex=False)

fn = "/data3/kwang/N50_random_ratios.h5" # 1000 realizations
randomratios = h5py.File(fn,'r')['ratios'][...]

fn = "/data3/kwang/denisov_N50_ratios.h5" # 100 realizations
denisovratios = h5py.File(fn,'r')['ratios'][...]

fn = "/data3/kwang/L7_allsinglelind_ratios.h5" # 100 realizations
singleratios = h5py.File(fn,'r')['ratios'][...]
fn = "/data3/kwang/L7_alldoublelind_ratios.h5" # 91 realizations
doubleratios = h5py.File(fn,'r')['ratios'][...]
fn = "/data3/kwang/L7_alltriplelind_ratios.h5" # 35 realizations
tripleratios = h5py.File(fn,'r')['ratios'][...]

fn = "/data3/kwang/liou_spec_H0_L7_gamma1_alldoublelind_run0.h5"
eigs = h5py.File(fn,'r')['eigenvalues'][...]
fn = "/data3/kwang/L7_alldoublelind_triplelemon_ratios.h5" # 91
doubleratios_lemon3 = h5py.File(fn,'r')['ratios'][...]
lims_lemon3 = h5py.File(fn,'r')['lims'][...]
fn = "/data3/kwang/L7_alldoublelind_overlaplemon_ratios.h5" # 91
doubleratios_overlaplemon = h5py.File(fn,'r')['ratios'][...]
lims_overlap = h5py.File(fn,'r')['lims'][...]


fs = 2.5
bins = 50

ax.hist([  denisovratios, singleratios, doubleratios, tripleratios], bins, label=[ '$N=50$ Fully Random $\mathcal{L}_D$', '$l=7$ One-body','$l=7$ One-body, Two-body','$l=7$ One-body, Two-body, Three-body'], density=True,histtype ='step')

ax.hist([randomratios], bins, label=['$N=50$ Random Ginibre'], density=True, histtype='step', color=['k'])
ax.hist([ doubleratios_lemon3, doubleratios_overlaplemon], bins, label=['red subset','blue subset'], color = ['r','b'], density=True,histtype='step')

ax.legend(loc='upper left')
ax.set_xlim((0,1))
ax.set_xlabel("$r$")
ax.set_ylabel("$p(r)$")


axins = inset_axes(ax, width="35%", height="45%", loc=4)

axins.scatter(eigs[1:].real, eigs[1:].imag, marker='.', color='k', edgecolors='none', s=0.2, rasterized=True)

axins.add_patch(matplotlib.patches.Rectangle((lims_lemon3[0][0], lims_lemon3[1][0]), lims_lemon3[0][1] - lims_lemon3[0][0], lims_lemon3[1][1] - lims_lemon3[1][0], color = 'r', fill=False, linewidth=0.5))
axins.add_patch(matplotlib.patches.Rectangle((lims_overlap[0][0], lims_overlap[1][0]), lims_overlap[0][1] - lims_overlap[0][0], lims_overlap[1][1] - lims_overlap[1][0], color = 'b', fill=False, linewidth=0.5))

axins.get_xaxis().set_visible(False)
axins.get_yaxis().set_visible(False)

#ax.hist(singleratios, bins, alpha=0.5, density=True)
#ax.hist(doubleratios, bins, alpha=0.5, density=True)
#ax.hist(tripleratios, bins, alpha=0.5, density=True)
#plt.rc('font', family='serif', size=3)
#fig.tight_layout(w_pad=2, h_pad=2, pad=0.1)
fig.tight_layout()
fig.savefig("fig2_ratio_comparison.pdf")
