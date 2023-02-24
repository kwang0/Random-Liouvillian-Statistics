#!/usr/bin/env python
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
import itertools
import re
import matplotlib
import sys
sys.path.append("../")
import numerics as num
import figures_module

fig, ax = figures_module.prepare_standard_figure(ncols=1, nrows=2, sharey=False, sharex=False, aspect_ratio=1.0)

n = 50
start = 4
centers = np.full((start + n - 1, n), np.nan)
for L in range(start, start + n):
    for i in range(1, L + 1):
        centers[i-1][L-start] = -2 * (num.single(L,i) + num.double(L,i))/num.n_ops(L,2)

fs = 2.5

nmax=15

cmap2 = matplotlib.cm.get_cmap('viridis_r')
colors___ = [cmap2(n) for n in np.linspace(0, 1, 15)]

for i in range(nmax):
    ax[0].plot(centers[i], 1./np.linspace(start, start + n - 1, n), '-', lw=1, c=colors___[i], label='{}'.format(i+1))
ax[0].legend(loc='upper right', ncol=2)


ax[0].set_ylabel("$1/\ell$", labelpad=2)
#ax[0].errorbar([-0.49500823, -0.41849965, -0.73142983, -0.36205874, -0.64742353, -0.31897079, -0.57968741], 1./np.array([5,6,6,7,7,8,8]), xerr=[0.03039484, 0.02082587, 0.05135065, 0.01472035, 0.03624463, 0.00946248, 0.02398741], fmt='o', markersize=2)

ax[0].errorbar([-0.49500823, -0.41849965,  -0.36205874,  -0.31897079, ], 1./np.array([5,6,7,8]), xerr=[0.03039484, 0.02082587,  0.01472035,  0.00946248], fmt='o', markersize=2, c=colors___[0])

ax[0].errorbar([  -0.73142983,  -0.64742353,  -0.57968741], 1./np.array([6,7,8]), xerr=[  0.05135065,  0.03624463,  0.02398741], fmt='o', markersize=2, c=colors___[1])

n = 50
dists = np.zeros(n)
for L in range(4,4+n):
    dists[L-4] = -2 * (num.single(L,1) - num.single(L,2) + num.double(L,1) - num.double(L,2))/num.n_ops(L,2)
ax[1].scatter(dists, 1./np.linspace(4,4+n-1,n), s=3, c='k',label='lemon 1/2 distance')
    

ax[1].plot([0.03039484, 0.02082587, 0.01472035, 0.00946248], 1./np.array([5,6,7,8]), 'o',  c=colors___[0],markersize=3, label='width lemon 1')
ax[1].plot([0.05135065, 0.03624463, 0.02398741], 1./np.array([6,7,8]), 'o', c=colors___[1], markersize=3,label='width lemon 2')
ax[1].set_ylim([0,0.26])

ax[1].legend(loc='upper center')

ax[0].set_xlabel("lemon centers", labelpad=2) 
ax[1].set_xlabel("lemon distance/width", labelpad=2) 
ax[1].set_ylabel("$1/\ell$", labelpad=2)

fig.tight_layout(w_pad=0, h_pad=0.5, pad=0.1)
fig.savefig("fig4-lemon_centers.pdf")
