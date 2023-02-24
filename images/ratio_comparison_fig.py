#!/usr/bin/env python
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
import itertools
import re
import matplotlib
import sys
import figures_module
import scipy
import numerics

fig, axs_ = figures_module.prepare_standard_figure(ncols=2, nrows=2, sharey=False, sharex=False, aspect_ratio=2.0)
axs = [axs_[0,0], axs_[0,1], axs_[1,0], axs_[1,1]]

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
axs[0].hist([randomratios, denisovratios], bins, label=['$N=50$ Random Ginibre','$N=50$ Fully Random $\mathcal{L}_D$'], density=True)
axs[0].tick_params(size=1, width=0.2, labelsize=fs, pad=1)
axs[0].legend(fontsize=fs)

axs[1].hist([randomratios, singleratios, doubleratios, tripleratios], bins, label=['$N=50$ Random Ginibre','$l=7$ One-body','$l=7$ One-body, Two-body','$l=7$ One-body, Two-body, Three-body'], density=True)
axs[1].tick_params(size=1, width=0.2, labelsize=fs, pad=1)
axs[1].legend(fontsize=fs)

axs[2].hist([randomratios, doubleratios_lemon3, doubleratios_overlaplemon], bins, label=['$N=50$ Random Ginibre','',''], color = ['c','r','b'], density=True)
axs[2].tick_params(size=1, width=0.2, labelsize=fs, pad=1)
axs[2].legend(fontsize=fs)

axs[3].scatter(eigs[1:].real, eigs[1:].imag, marker='.', color='c', edgecolors='none', s=0.2)
axs[3].tick_params(size=1, width=0.2, labelsize=fs, pad = 1)
axs[3].add_patch(matplotlib.patches.Rectangle((lims_lemon3[0][0], lims_lemon3[1][0]), lims_lemon3[0][1] - lims_lemon3[0][0], lims_lemon3[1][1] - lims_lemon3[1][0], color = 'r', fill=False, linewidth=0.1))
axs[3].add_patch(matplotlib.patches.Rectangle((lims_overlap[0][0], lims_overlap[1][0]), lims_overlap[0][1] - lims_overlap[0][0], lims_overlap[1][1] - lims_overlap[1][0], color = 'b', fill=False, linewidth=0.1))
#ax.hist(singleratios, bins, alpha=0.5, density=True)
#ax.hist(doubleratios, bins, alpha=0.5, density=True)
#ax.hist(tripleratios, bins, alpha=0.5, density=True)
plt.rc('font', family='serif', size=3)
fig.tight_layout(w_pad=2, h_pad=2, pad=0.1)
fig.savefig("ratio_hist.png", dpi=1600)
