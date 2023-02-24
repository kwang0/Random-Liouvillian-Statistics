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

fig, ax = figures_module.prepare_standard_figure(ncols=1, nrows=1, sharey=False, sharex=False, aspect_ratio=1.0)

fn = "/data3/kwang/L5_orderedbasis.h5"
evecs = h5py.File(fn,'r')['eigenvectors'][...]

fs = 2.5
ax.matshow(np.abs(evecs))
plt.rc('font', family='serif', size=3)
fig.tight_layout(w_pad=0, h_pad=2, pad=0.1)
fig.savefig("L5_submatrix_evecs.png", dpi=1600)
