#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.sparse as spr

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

def Expectation(op, v):
    rho = np.reshape(v, (2**L,2**L))
    return np.abs(np.trace(op.dot(rho)))

if __name__=="__main__":
    L = 8
    ops_name = "single"
    run = 1
    
    fn = "/data3/kwang/liou_spec_H0_L{}_gamma1_all{}lind_run{}.h5".format(L, ops_name, run)

    eigs = h5py.File(fn, 'r')['eigenvalues'][...]
    evecs = h5py.File(fn, 'r')['eigenvectors'][...]


    Sxs, Sys, Szs  = SpinOps(L)
    Spins = [Sxs, Sys, Szs]
    
    expectation_vals = np.zeros((L, 4**L))
    for n in range(10):
        ops = [spr.csr_matrix((2**L, 2**L)) for i in range(L)]
        
        for a in range(3):
            for i in range(L):
                ops[0] += (np.random.uniform(-1, 1) + 1.j * np.random.uniform(-1, 1)) * Spins[a][i]
                for b in range(3):
                    for j in range(i):
                        ops[1] += (np.random.uniform(-1, 1) + 1.j * np.random.uniform(-1, 1)) * Spins[a][i] * Spins[b][j]
                        for c in range(3):
                            for k in range(j):
                                ops[2] += (np.random.uniform(-1, 1) + 1.j * np.random.uniform(-1, 1)) * Spins[a][i] * Spins[b][j] * Spins[c][k]
                                for d in range(3):
                                    for l in range(k):
                                        ops[3] += (np.random.uniform(-1, 1) + 1.j * np.random.uniform(-1, 1)) * Spins[a][i] * Spins[b][j] * Spins[c][k] * Spins[d][l]
                                        for e in range(3):
                                            for m in range(l):
                                                ops[4] += (np.random.uniform(-1, 1) + 1.j * np.random.uniform(-1, 1)) * Spins[a][i] * Spins[b][j] * Spins[c][k] * Spins[d][l] * Spins[e][m]
                                                for f in range(3):
                                                    for n in range(m):
                                                        ops[5] += (np.random.uniform(-1, 1) + 1.j * np.random.uniform(-1, 1)) * Spins[a][i] * Spins[b][j] * Spins[c][k] * Spins[d][l] * Spins[e][m] * Spins[f][n]
                                                        for g in range(3):
                                                            for o in range(n):
                                                                ops[6] += (np.random.uniform(-1, 1) + 1.j * np.random.uniform(-1, 1)) * Spins[a][i] * Spins[b][j] * Spins[c][k] * Spins[d][l] * Spins[e][m] * Spins[f][n] * Spins[g][o]
#                                                                for h in range(3):
#                                                                    for p in range(o):
#                                                                        ops[7] += (np.random.uniform(-1, 1) + 1.j * np.random.uniform(-1, 1)) * Spins[a][i] * Spins[b][j] * Spins[c][k] * Spins[d][l] * Spins[e][m] * Spins[f][n] * Spins[g][o] * Spins[g][p]
        
        
        for i, op in enumerate(ops):
            for j in range(4**L):
                expectation_vals[i, j] += Expectation(op, evecs[:, j])
    
    with h5py.File("/data3/kwang/L{}_all{}lind_highlighting_run{}.h5".format(L, ops_name, run), 'w') as F:
        F['vals'] = expectation_vals
    
