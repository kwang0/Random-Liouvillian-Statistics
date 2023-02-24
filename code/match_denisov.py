#!/usr/bin/env python
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

def RandomKossakowski(N):
    nl = N ** 2 - 1
    # Random diagonal matrix
    D = np.diag(np.random.rand(nl))
#   mask = np.random.choice(N, int(0.2 * N), replace=False)
#   D[mask, mask] = 0.0
    D /= np.trace(D)
    D *= N
    
    # Q random unitary matrix
    A = np.random.randn(nl, nl) + 1.0j * np.random.randn(nl, nl)
    Q, R = np.linalg.qr(A)
    #K = np.conj(Q.T).dot(D).dot(Q)
    return D, Q

def TransformLindblad(Lindbladops, U, N):
    nl = len(Lindbladops)
    Lindbladops_new = []
    for i in range(nl):
        op = spr.csr_matrix((N, N))
        for j in range(nl):
            op += U[j, i] * Lindbladops[j]
        Lindbladops_new.append(op)
    return Lindbladops_new



if __name__=="__main__":
    Lindbladops = []
    N = 30 
    for i in range(N):
        for j in range(i + 1, N):
            S = spr.csr_matrix(([1./np.sqrt(2),1./np.sqrt(2)],([i, j], [j, i])), shape = (N, N))
            J = spr.csr_matrix(([1.j/np.sqrt(2),-1.j/np.sqrt(2)],([i, j], [j, i])), shape = (N, N))
            Lindbladops.append(S)
            Lindbladops.append(J)
        if i != 0:
            D1 = spr.diags([1./np.sqrt(i * (i + 1)) for k in range(i)] + [-i/np.sqrt(i * (i + 1))] + [0 for k in range(N - 1 - i)], 0)
            Lindbladops.append(D1)
    nl = len(Lindbladops)

    for run in range(1):
        D, U = RandomKossakowski(N)
      
        print("Transforming Lindblad operators")
        Lindbladops = TransformLindblad(Lindbladops, U, N)
    
        # Create the Liouvillian
        def Lmatvec(v):
            rho = np.reshape(v,(N, N))
    
            Lrho = np.zeros((N, N))
#            D = np.random.rand(nl)
#            D *= (N**2 - 1)/np.sum(D)
            for n in range(nl):
                X = Lindbladops[n].dot(rho)
                Lrho = Lrho + D[n,n] * ( (np.conj(Lindbladops[n]).dot(X.T)).T - 0.5 * (np.conj(Lindbladops[n].T).dot(X) + (Lindbladops[n].T.dot(np.conj(Lindbladops[n]).dot(rho.T))).T) )
            
            return Lrho.reshape(N ** 2)
    
       
        from scipy.sparse.linalg import LinearOperator
        Liouvillian = LinearOperator((N ** 2, N ** 2), matvec = Lmatvec)
        
        print("Creating Liouvillian")
        identity  = np.eye(N ** 2)
        Liou = Liouvillian.dot(identity) # Dense matrix form of Liou
        print("Calculating eigenvalues")
    
        liou_evals = np.linalg.eigvals( Liou )
        print(liou_evals)
    
#        plt.scatter(liou_evals.real, liou_evals.imag,marker='.',s=1)
#        plt.savefig("liou_eth_H0_L{}_gamma{}.pdf".format(L, gamma))
#        plt.clf()
        with h5py.File("testdensity_N{}_run{}.h5".format(N, run),'w') as F:
            F['eigenvalues'] = liou_evals
