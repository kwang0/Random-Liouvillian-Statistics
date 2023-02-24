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

fig, axs = figures_module.prepare_standard_figure(ncols=1, nrows=2, sharey=False, sharex=True, aspect_ratio=1.2)

fn = "/data3/kwang/liou_spec_H0_L8_gamma1_allNNPBCdoublelind_run0.h5"
eigs = h5py.File(fn,'r')['eigenvalues'][...]
fn = "/data3/kwang/liou_spec_H0_L8_gamma1_allNNdoublelind_run0.h5" 
eigs1 = h5py.File(fn,'r')['eigenvalues'][...]

fs = 2.5
#plt.rc('font', family='serif', size=3)
axs[0].scatter(eigs[1:].real, eigs[1:].imag, marker='.', color='r', edgecolors='none', s=0.4,rasterized=True)
#axs[0].tick_params(size=1, width=0.2, labelsize=fs, pad = 1)
axs[0].set_ylabel("Im($\lambda$)", labelpad=2) 

axs[0].text(0.95,0.9, "$\ell=8$, n.n. PBC", transform=axs[0].transAxes,horizontalalignment='right')


axs[1].scatter(eigs1[1:].real, eigs1[1:].imag, marker='.', color='r', edgecolors='none', s=0.4, rasterized=True)
#axs[1].tick_params(size=1, width=0.2, labelsize=fs, pad = 1)
axs[1].set_xlabel("Re($\lambda$)", labelpad=2)
axs[1].set_ylabel("Im($\lambda$)", labelpad=2) 

axs[1].text(0.95,0.9, "$\ell=8$, n.n. OBC", transform=axs[1].transAxes,horizontalalignment='right')

fig.tight_layout(w_pad=0, h_pad=0.4, pad=0.1)


fig.savefig("fig5-nearest-neighbor.pdf", dpi=400)
