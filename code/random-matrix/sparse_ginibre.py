#!/usr/bin/python
import scipy.sparse as spr
import numpy as np 
import scipy.special as sp
import scipy.linalg as spl
from scipy.linalg import expm
import argparse
import time
import sys
import h5py
import matplotlib.pyplot as plt



N = 200
G = np.random.randn(N,N) 
eG = np.linalg.eigvals(G)
print(eG)

plt.plot(eG.real,eG.imag,'r.')



S = np.zeros((N,N))
rowidxs = np.random.randint(N,size=4*N)
colidxs = np.random.randint(N,size=4*N)

for i,j in zip(rowidxs,colidxs):
    S[i,j]=np.random.randn()

eS = np.linalg.eigvals(S)
print(eS)
plt.plot(eS.real,eS.imag,'k.')



phis = np.linspace(0,2*np.pi,1000)
r = np.sqrt(N)
plt.plot(r*np.cos(phis),r*np.sin(phis),'r-')



plt.gca().set_aspect(aspect='equal')
plt.savefig('ginibre_spec.pdf')
