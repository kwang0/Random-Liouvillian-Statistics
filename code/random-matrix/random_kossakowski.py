#!/usr/bin/env python3
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
    D /= np.trace(D)
    D *= N
    
    # Q random unitary matrix
    A = np.random.randn(nl, nl) + 1.0j * np.random.randn(nl, nl)
    Q, R = np.linalg.qr(A)
#    K = np.conj(Q.T).dot(D).dot(Q)

    K = np.dot( np.dot( np.conj(Q).T , D ) , Q)

    return K, D, Q


def random_CUE(N):
    # Q random unitary matrix
    A = np.random.randn(N, N) + 1.0j * np.random.randn(N, N)
    Q, R = np.linalg.qr(A)
    return Q



def gauss(x,mu,sigma):
    return 1./(np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))



if __name__=="__main__":

    L=10
    N = 2**L
    nl=L*(L-1)

    matels=[]

    ds = []
    boxstd = np.std(np.random.rand(100000))
    print("boxstd: " , boxstd)
    print(1/np.sqrt(12))
    print("sigma_d: ", boxstd*2*N/nl)

    sigma_d = 1./np.sqrt(12)*2*N/nl


    diag_matels = []

    for n in range(1000):
        K,D,Q  = np.array(RandomKossakowski(N,nl))

        for i in range(nl):
            ds.append(D[i,i])
            diag_matels.append(K[i,i])
            for j in range(i):
                if j<i:
                    matels.append(K[i,j])

    matels = np.array(matels).flatten()
    print(matels)

#    sigma_d_num= np.std(ds)
#    print("sigma_d: " ,sigma_d_num, "mean_d: ", np.mean(ds))
#    print(sigma_d)

        
    plt.hist(matels.real,bins=300,histtype='step',density=True)

#    sigma = sigma_d/nl**(1/2)/np.sqrt(2)

    sigma = 1/np.sqrt(6)*N/nl**(3/2)
    mu = 0.0
    x= np.linspace(-5*sigma,5*sigma,10000)

    mean_Kii = np.mean(diag_matels)
    print("mean(Kii):", mean_Kii)
    print("mean(Kii) analytical:", N/nl)

    print(1/N)
    print(np.mean(np.abs(matels)**2))
    plt.plot(x,gauss(x,mu,sigma),'r--')

    plt.axvline(mean_Kii.real)

    plt.savefig('histK.pdf')

