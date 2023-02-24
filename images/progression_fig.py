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

fig, axs_ = figures_module.prepare_standard_figure(ncols=3, nrows=2, sharey=False, sharex=False, aspect_ratio=2.0)

axs = [axs_[0,0], axs_[0,1], axs_[0,2], axs_[1,0], axs_[1,1], axs_[1,2]]

eigs = np.empty(6, dtype=object)
fn = "/data3/kwang/liou_spec_H0_L6_gamma1_allsinglelind_run0.h5"
eigs[0] = h5py.File(fn,'r')['eigenvalues'][...]
fn = "/data3/kwang/liou_spec_H0_L6_gamma1_alldoublelind_run0.h5"
eigs[1] = h5py.File(fn,'r')['eigenvalues'][...]
fn = "/data3/kwang/liou_spec_H0_L6_gamma1_alltriplelind_run0.h5"
eigs[2] = h5py.File(fn,'r')['eigenvalues'][...]
fn = "/data3/kwang/liou_spec_H0_L6_gamma1_allfourlind_run0.h5"
eigs[3] = h5py.File(fn,'r')['eigenvalues'][...]
fn = "/data3/kwang/liou_spec_H0_L6_gamma1_allfivelind_run0.h5"
eigs[4] = h5py.File(fn,'r')['eigenvalues'][...]
fn = "/data3/kwang/liou_spec_H0_L6_gamma1_allsixlind_run0.h5"
eigs[5] = h5py.File(fn,'r')['eigenvalues'][...]

# eigs += 1

labels = ['a)','b)','c)','d)','e)','f)']

fs = 2.5
plt.rc('font', family='serif', size=3)
for i in range(6):
    ax = axs[i]
    ax.scatter(eigs[i][1:].real, eigs[i][1:].imag, marker='.', color='c', edgecolors='none', s=0.2)
    ax.tick_params(size=1, width=0.2, labelsize=fs, pad = 1)
    if i > 2:
        axs[i].set_xlabel("Re($\lambda$)", size=4, labelpad=2)
    if i % 3 == 0:
        axs[i].set_ylabel("Im($\lambda$)", size=4, labelpad=2) 
    ax.text(0.1, 0.88, labels[i], verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, fontsize=4)
fig.tight_layout(w_pad=0, h_pad=2, pad=0.1)
fig.savefig("L6_progression.png", dpi=800)
