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

def RandomKossakowski(N, nl):
    # Random diagonal matrix
    D = np.diag(np.random.rand(nl))
#    D = np.diag(1/nl * (0.99 * np.ones(nl) + 0.02 * np.random.random(nl)))
    D /= np.trace(D)
    D *= N
    
    # Q random unitary matrix
    A = np.random.randn(nl, nl) + 1.0j * np.random.randn(nl, nl)
    Q, R = np.linalg.qr(A)
    #K = np.conj(Q.T).dot(D).dot(Q)
    return D, Q

def TransformLindblad(Lindbladops, U, L):
    nl = len(Lindbladops)
    Lindbladops_new = []
    for i in range(nl):
        op = spr.csr_matrix((2**L, 2**L))
        for j in range(nl):
            op += U[j, i] * Lindbladops[j]
        Lindbladops_new.append(op)
    return Lindbladops_new




def random_CUE(N):
    # Q random unitary matrix
    A = np.random.randn(N, N) + 1.0j * np.random.randn(N, N)
    Q, R = np.linalg.qr(A)
    return Q



def gauss(x,mu,sigma):
    return 1./(np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))



if __name__=="__main__":


    N=200
    matels=[]
    for n in range(1000):
        U  = np.array(random_CUE(N))
        matels.append(U.flatten())

    matels = np.array(matels).flatten()
    print(matels)

        
    plt.hist(matels.real,bins=100,histtype='step',density=True)

    sigma = 1/np.sqrt(2*N)
    mu = 0.0
    x= np.linspace(-0.3,0.3,1000)

    print(1/N)
    print(np.mean(np.abs(matels)**2))
    plt.plot(x,gauss(x,mu,sigma),'r-')

    plt.savefig('hist.pdf')

