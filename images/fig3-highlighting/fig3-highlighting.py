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

fig, axs_ = figures_module.prepare_standard_figure(ncols=4, nrows=2, sharey=True, sharex=True, aspect_ratio=2.0, width=6.75)


axs = [axs_[0,0], axs_[0,1], axs_[0,2], axs_[0,3], axs_[1,0], axs_[1,1], axs_[1,2], axs_[1,3]]


fn = "/data3/kwang/liou_spec_H0_L8_gamma1_alldoublelind_run0.h5"
eigs = h5py.File(fn,'r')['eigenvalues'][...]
fn = "/data3/kwang/L8_alldoublelind_highlighting_run0.h5"
vals = h5py.File(fn,'r')['vals'][...]
vals[:] /= vals.max(axis=0)
#vals[:] /= vals.sum(axis=0)
# eigs += 1

print("SHAPE: " , vals.shape)

labels = ['a)','b)','c)','d)','e)','f)','g)','h)']

fs = 2.5
#plt.rc('font', family='serif', size=3)


for i in range(8):
    ax = axs[i]
    im = ax.scatter(eigs[1:].real, eigs[1:].imag, marker='.', c=vals[i][1:], cmap='viridis', edgecolors='none', s=0.2, rasterized=True)
    #ax.tick_params(size=1, width=0.2, pad = 1)

    analytic = -2 * (numerics.single(8,i+1) + numerics.double(8,i+1)) / numerics.n_ops(8,2)
    ax.scatter(analytic, 0, marker='.', c='r', edgecolors='none', s=10, rasterized=True)

    if i > 3:
        axs[i].set_xlabel("Re($\lambda$)", labelpad=2)
    if i % 4 == 0:
        axs[i].set_ylabel("Im($\lambda$)", labelpad=2) 
    ax.text(0.98, 0.93, labels[i], verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes)

    ax.text(0.98, 0.05, "{} body operator $\hat O$".format(i+1), verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes)


axs[0].text(0.18,0.85,"$\ell=8$, up to 2-body $\hat L_i$",transform=axs[0].transAxes)


ins = inset_axes(axs[-1], width="70%", height="5%",loc='upper center')

cbar = fig.colorbar(im, cax=ins, ax=axs_.ravel().tolist(), shrink=0.9, aspect=50, pad =0.01, orientation='horizontal')
ins.set_xlabel(r"Tr($\rho_i \hat O$) [normalized]")
fig.tight_layout(w_pad=0.5, h_pad=0.5, pad=1)

fig.savefig("fig3-highlighting.pdf", dpi=300)
