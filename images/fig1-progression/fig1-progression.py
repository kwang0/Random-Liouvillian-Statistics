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

#fig, axs_ = figures_module.prepare_standard_figure(ncols=3, nrows=2, sharey=False, sharex=False, aspect_ratio=2.0, width=5.90)
fig, axs_ = figures_module.prepare_standard_figure(ncols=2, nrows=3, sharey=False, sharex=False, aspect_ratio=0.8)

axs = [axs_[0,0], axs_[1,0], axs_[2,0], axs_[0,1], axs_[1,1], axs_[2,1]]

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
#plt.rc('font', family='serif', size=3)
for i in range(6):
    ax = axs[i]
    ax.scatter(eigs[i][1:].real, eigs[i][1:].imag, marker='.', color='r', edgecolors='none', s=2)
#    ax.tick_params(width=0.2, labelsize=fs, pad = 1)
    ax.text(0.12, 0.9, r"\textbf{"+labels[i]+"}", verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes)
    ax.text(0.95, 0.9, r"{} body".format(i+1), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes)




for ax in [axs_[2,0], axs_[2,1]]:
    ax.set_xlabel("Re($\lambda$)",  labelpad=2)


for ax in [axs_[0,0],axs_[1,0],axs_[2,0]]:
    ax.set_ylabel("Im($\lambda$)" , labelpad=1) 


for ax in [axs_[0,1],axs_[1,1],axs_[2,1]]:
    ax.set_yticks([-0.01,0,0.01])
axs_[0,0].set_yticks([-0.05,0,0.05])


fig.tight_layout(w_pad=0, h_pad=0.5, pad=0.8)
fig.savefig("fig1_progression.pdf")
