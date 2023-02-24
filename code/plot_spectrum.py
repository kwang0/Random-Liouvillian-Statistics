#!/usr/bin/env python
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob

fns = list(glob.glob("/home/kwang/Documents/Random Liouvillians//data/match_runs/*N30*.h5"))
plt.xlim([-2,2])
for fn in fns:
    eigs = h5py.File(fn, 'r')['eigenvalues'][...]
    eigs += 30
    eigs *= 2.0/1.2
    plt.scatter(eigs.real, eigs.imag, marker='.',s=1)
#plt.savefig("N30_spectrum.pdf")
