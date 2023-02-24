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

fig, axs = figures_module.prepare_standard_figure(ncols=2, nrows=1, sharey=False, sharex=False, aspect_ratio=2.0)

fn = "/data3/kwang/liou_spec_H0_L8_gamma1_allNNPBCdoublelind_run0.h5"
eigs = h5py.File(fn,'r')['eigenvalues'][...]
fn = "/data3/kwang/liou_spec_H0_L8_gamma1_allNNdoublelind_run0.h5" 
eigs1 = h5py.File(fn,'r')['eigenvalues'][...]

fs = 2.5
plt.rc('font', family='serif', size=3)
axs[0].scatter(eigs[1:].real, eigs[1:].imag, marker='.', color='c', edgecolors='none', s=0.2)
axs[0].tick_params(size=1, width=0.2, labelsize=fs, pad = 1)
axs[0].set_xlabel("Re($\lambda$)", size=4, labelpad=2)
axs[0].set_ylabel("Im($\lambda$)", size=4, labelpad=2) 

axs[1].scatter(eigs1[1:].real, eigs1[1:].imag, marker='.', color='c', edgecolors='none', s=0.2)
axs[1].tick_params(size=1, width=0.2, labelsize=fs, pad = 1)
axs[1].set_xlabel("Re($\lambda$)", size=4, labelpad=2)
fig.tight_layout(w_pad=0, h_pad=2, pad=0.1)
fig.savefig("L8_NN.png", dpi=800)
