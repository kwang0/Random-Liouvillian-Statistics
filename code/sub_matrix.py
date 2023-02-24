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
from sklearn.cluster import DBSCAN

def SpinOps(L):
    sx = spr.csr_matrix(np.array([[0.,1.],[1.,0.]]))
    sy = spr.csr_matrix(np.array([[0.,-1.j],[1.j,0.]]))
    sz = spr.csr_matrix(np.array([[1.,0.],[0.,-1.]]))      
    Sxs = []
    Sys = []
    Szs = []
    for i in range(L):
        leftdim = 2**i
        rightdim = 2**(L+1-i-2)
        Sx = spr.kron(spr.kron(spr.eye(leftdim),sx),spr.eye(rightdim),'csr')
        Sy = spr.kron(spr.kron(spr.eye(leftdim),sy),spr.eye(rightdim),'csr')
        Sz = spr.kron(spr.kron(spr.eye(leftdim),sz),spr.eye(rightdim),'csr')
        Sxs.append(Sx)
        Sys.append(Sy)
        Szs.append(Sz)

    return Sxs, Sys, Szs

def Hamiltonian(Sxs, Sys, Szs, L, Jx, Jy, Jz, Jx2, Jy2, Jz2): 
    H = spr.csr_matrix((2**L,2**L))  
    Lb = L - 1 #-- open boundary conditions
    for i in range(Lb):   
        H += Jz * Szs[i].dot(Szs[(i+1)%L])
        H += Jx * Sxs[i].dot(Sxs[(i+1)%L])
        H += Jy * Sys[i].dot(Sys[(i+1)%L])

    Lb = L - 2 #-- open boundary conditions
    for i in range(Lb):   
        H += Jz2 * Szs[i].dot(Szs[(i+2)%L])
        H += Jx2 * Sxs[i].dot(Sxs[(i+2)%L])
        H += Jy2 * Sys[i].dot(Sys[(i+2)%L])
    for i in range(L):
        H += (0.5 * np.random.random() - 0.25) * Szs[i]
    return H 

def RandomKossakowski(N, nl):
    # Random diagonal matrix
    D = np.diag(np.random.rand(nl))
#    eps = 0.1
#    D = np.diag(1/nl * ((1 - eps) * np.ones(nl) + 2 * eps * np.random.random(nl)))
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



if __name__=="__main__":
    L = 5

    ops = 2
    
    N = 2**L

    Sxs, Sys, Szs  = SpinOps(L)
    Spins = [Sxs, Sys, Szs]
    
    print("Building Lindblad operator basis")
    Lindbladops = []
    fullorderedbasis = [spr.eye(N)]
    singleops = []
    doubleops = []
    tripleops = []
    fourops = []
    fiveops = []
    for a in range(3):
        for i in range(L):
            if ops >= 1: Lindbladops.append(1./np.sqrt(N) * Spins[a][i])
            singleops.append(1./np.sqrt(N) * Spins[a][i])
            for b in range(3):
                for j in range(i):
                    if ops >= 2: Lindbladops.append(1./np.sqrt(N) * Spins[a][i] * Spins[b][j])
                    doubleops.append(1./np.sqrt(N) * Spins[a][i] * Spins[b][j])
                    for c in range(3):
                        for k in range(j):
                            if ops >= 3: Lindbladops.append(1./np.sqrt(N) * Spins[a][i] * Spins[b][j] * Spins[c][k])
                            tripleops.append(1./np.sqrt(N) * Spins[a][i] * Spins[b][j] * Spins[c][k])
                            for d in range(3):
                                for l in range(k):
                                    if ops >= 4: Lindbladops.append(1./np.sqrt(N) * Spins[a][i] * Spins[b][j] * Spins[c][k] * Spins[d][l])
                                    fourops.append(1./np.sqrt(N) * Spins[a][i] * Spins[b][j] * Spins[c][k] * Spins[d][l])
                                    for e in range(3):
                                        for m in range(l):
                                            if ops >= 5: Lindbladops.append(1./np.sqrt(N) * Spins[a][i] * Spins[b][j] * Spins[c][k] * Spins[d][l] * Spins[e][m])
                                            fiveops.append(1./np.sqrt(N) * Spins[a][i] * Spins[b][j] * Spins[c][k] * Spins[d][l] * Spins[e][m])
                                            for f in range(3):
                                                for n in range(m):
                                                    if ops >= 6: Lindbladops.append(1./np.sqrt(N) * Spins[a][i] * Spins[b][j] * Spins[c][k] * Spins[d][l] * Spins[e][m] * Spins[f][n])
                                                    for g in range(3):
                                                        for o in range(n):
                                                            if ops >= 7: Lindbladops.append(1./np.sqrt(N) * Spins[a][i] * Spins[b][j] * Spins[c][k] * Spins[d][l] * Spins[e][m] * Spins[f][n] * Spins[g][o])
                                            
    fullorderedbasis += singleops + doubleops + tripleops + fourops + fiveops
    basis_mat = np.zeros((N**2, len(fullorderedbasis)), dtype=complex)
    for i in range(len(fullorderedbasis)):
        basis_mat[:,i] = np.reshape(fullorderedbasis[i].todense(), N**2)
    print("Generating random Kossakowski")
    nl = len(Lindbladops)
    D, U = RandomKossakowski(N, nl)
#    K = np.zeros((nl,nl),dtype=complex)
#    for i in range(nl):
#        K[i,i] = 1
#        for j in range(i):
#            K[i,j] = 1 + 1.j
#            K[j,i] = 1 - 1.j
#    D,U = np.linalg.eig(K)
#    D = np.diag(D)

    print("Transforming Lindblad operators to diagonal basis")
    Lindbladops_old = Lindbladops
    Lindbladops = TransformLindblad(Lindbladops, U, L)

    # Create the Liouvillian
    def Lmatvec(v):
        rho = np.reshape(v,(2**L,2**L))

        Lrho = spr.csr_matrix((N,N))

        for n in range(nl):
            X = Lindbladops[n].dot(rho)
            X1 = np.conj(Lindbladops[n].T).dot(rho)
            Lrho = Lrho + D[n,n] * ( ((Lindbladops[n].T).dot(X1.T)).T - 0.0 * (np.conj(Lindbladops[n].T).dot(X) + (Lindbladops[n].T.dot(np.conj(Lindbladops[n]).dot(rho.T))).T) )
        return Lrho.reshape(2**(2*L))
    
    from scipy.sparse.linalg import LinearOperator
    Liouvillian = LinearOperator((2**(2*L),2**(2*L)), matvec = Lmatvec)

    print("Creating Liouvillian")
    Liou = Liouvillian.dot(np.eye(N**2))
    submatrix = Liou.dot(basis_mat)
    submatrix = np.conj(basis_mat.T).dot(submatrix)
    
