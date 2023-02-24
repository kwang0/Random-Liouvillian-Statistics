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

fig, axs_ = figures_module.prepare_standard_figure(ncols=4, nrows=2, sharey=False, sharex=False, aspect_ratio=2.0)

axs = [axs_[0,0], axs_[0,1], axs_[0,2], axs_[0,3], axs_[1,0], axs_[1,1], axs_[1,2], axs_[1,3]]

fn = "/data3/kwang/liou_spec_H0_L8_gamma1_alldoublelind_run0.h5"
eigs = h5py.File(fn,'r')['eigenvalues'][...]
fn = "/data3/kwang/L8_alldoublelind_highlighting_run0.h5"
vals = h5py.File(fn,'r')['vals'][...]
vals[:] /= vals.sum(axis=0)
# eigs += 1

labels = ['a)','b)','c)','d)','e)','f)','g)','h)']

fs = 2.5
plt.rc('font', family='serif', size=3)
for i in range(8):
    ax = axs[i]
    im = ax.scatter(eigs[1:].real, eigs[1:].imag, marker='.', c=vals[i][1:], cmap='cool', edgecolors='none', s=0.1)
    ax.tick_params(size=1, width=0.2, labelsize=fs, pad = 1)
    analytic = -2 * (numerics.single(8,i+1) + numerics.double(8,i+1)) / numerics.n_ops(8,2)
    ax.scatter(analytic, 0, marker='.', c='k', edgecolors='none', s=1)
    if i > 2:
        axs[i].set_xlabel("Re($\lambda$)", size=4, labelpad=2)
    if i % 4 == 0:
        axs[i].set_ylabel("Im($\lambda$)", size=4, labelpad=2) 
    ax.text(0.98, 0.88, labels[i], verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, fontsize=4)
fig.colorbar(im, ax=ax)
fig.tight_layout(w_pad=0, h_pad=2, pad=0.1)
fig.savefig("L8_highlighting.png", dpi=1600)
