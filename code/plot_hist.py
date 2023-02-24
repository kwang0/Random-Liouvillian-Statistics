#!/usr/bin/env python
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob

fns = list(glob.glob("/home/kwang/Documents/Random Liouvillians/data/match_runs/density*"))

alleigs = np.array([])
for fn in fns:
    eigs = h5py.File(fn, 'r')['eigenvalues'][...]
    eigs += 20
    eigs *= 2.0/1.2
    cross = eigs.imag[np.abs(eigs.real) < 0.05]
    alleigs = np.concatenate((alleigs, cross))
plt.hist(alleigs, range = [-1, 1], bins = 100)
plt.savefig("/home/kwang/Documents/Random Liouvillians/data/plots/N20_cross_hist.pdf")
