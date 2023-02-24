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



if __name__=="__main__":
    start = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--length", type=int, help = "number of sites in chain")
    parser.add_argument("-r", "--run", help="run number")
    parser.add_argument("-o", "--ops", type=int, help="maximum combination of spin operators to include")
    args = parser.parse_args()
    run = args.run
    ops = args.ops
    L = args.length

    ops_string = ""
    if ops == 1:
        ops_string = "single"
    elif ops == 2:
        ops_string = "double"
    elif ops == 3:
        ops_string = "triple"
    elif ops == 4:
        ops_string = "four"
    elif ops == 5:
        ops_string = "five"
    elif ops == 6:
        ops_string = "six"
    elif ops == 7:
        ops_string = "seven"
#    L = 3
    Jx, Jy, Jz, Jx2, Jy2, Jz2 = 2 * np.random.rand(6) - 1
    gamma = 1

    N = 2**L

    Sxs, Sys, Szs  = SpinOps(L)
    Spins = [Sxs, Sys, Szs]

    H = 0 * Hamiltonian(Sxs, Sys, Szs, L, Jx, Jy, Jz, Jx2, Jy2, Jz2)
    H_info = "No hamiltonian"
#   H_info = "Open BC. Nearest and next nearest neighbor interactions X,Y,Z with random couplings [-1,1] uniform across system. Random transverse field [-0.5,0.5] on each Z_i"

    print("Building Lindblad operator basis")
    # Lindbladops = [ sxs[i] +1.j*sys[i] for i in range(0,L) ] +  [ sxs[i] -1.j*sys[i] for i in range(0,L) ]
    # Lindbladops = [ Sxs[i] for i in range(L) ]
    Lindbladops = []
    for a in range(3):
        for i in range(L):
            if ops >= 1: Lindbladops.append(1./np.sqrt(N) * Spins[a][i])
            for b in range(3):
                for j in range(i):
                    if ops >= 2: Lindbladops.append(1./np.sqrt(N) * Spins[a][i] * Spins[b][j])
                    for c in range(3):
                        for k in range(j):
                            if ops >= 3: Lindbladops.append(1./np.sqrt(N) * Spins[a][i] * Spins[b][j] * Spins[c][k])
                            for d in range(3):
                                for l in range(k):
                                    if ops >= 4: Lindbladops.append(1./np.sqrt(N) * Spins[a][i] * Spins[b][j] * Spins[c][k] * Spins[d][l])
                                    for e in range(3):
                                        for m in range(l):
                                            if ops >= 5: Lindbladops.append(1./np.sqrt(N) * Spins[a][i] * Spins[b][j] * Spins[c][k] * Spins[d][l] * Spins[e][m])
                                            for f in range(3):
                                                for n in range(m):
                                                    if ops >= 6: Lindbladops.append(1./np.sqrt(N) * Spins[a][i] * Spins[b][j] * Spins[c][k] * Spins[d][l] * Spins[e][m] * Spins[f][n])
                                                    for g in range(3):
                                                        for o in range(n):
                                                            if ops >= 7: Lindbladops.append(1./np.sqrt(N) * Spins[a][i] * Spins[b][j] * Spins[c][k] * Spins[d][l] * Spins[e][m] * Spins[f][n] * Spins[g][o])
                                            

                   
    Lindblad_info = "Up to " + ops_string + " site spin operators. PBC."

    print("Time elapsed: " + str(time.time() - start)) 
    print() 
    print("Generating random Kossakowski")
    nl = len(Lindbladops)
    D, U = RandomKossakowski(N, nl)
    

    print("Time elapsed: " + str(time.time() - start)) 
    print() 
    print("Transforming Lindblad operators to diagonal basis")
    Lindbladops_old = Lindbladops
    Lindbladops = TransformLindblad(Lindbladops, U, L)

    # Create the Liouvillian
    def Liouvillian():
        Liou = spr.csr_matrix((N**2,N**2)) 

        for n in range(nl):
            Lop = Lindbladops[n]
            Ldag = np.conj(Lop.T)
            prod = Ldag.dot(Lop)
            Liou = Liou + gamma * D[n,n] * (spr.kron(Lop,Ldag.T) - 0.5 * (spr.kron(prod,spr.eye(N)) + spr.kron(spr.eye(N),prod.T)))
        return Liou.todense()
    
    def Expectation(op, v):
        rho = np.reshape(v, (2**L,2**L))
#        rho /= np.trace(rho).real
        return np.trace(op.dot(rho))


    print("Time elapsed: " + str(time.time() - start)) 
    print() 
    print("Creating Liouvillian")
    Liou = Liouvillian() # Dense matrix form of Liou
    print("Time elapsed: " + str(time.time() - start)) 
    print()
    print("Calculating eigenvalues")

    liou_evals, liou_evecs = np.linalg.eig( Liou )
 
        
#    print("Calculating expectation values of jump operators")
#    expectation_vals = []
#    for op in Lindbladops_old:
#        op_expecs = []
#        for i in range(len(liou_evals)):
#            print(i)
#            op_expecs.append(Expectation(op, liou_evecs[:, i]))
#        expectation_vals.append(op_expecs)
#    print(liou_evals)

#    plt.scatter(liou_evals.real, liou_evals.imag,marker='.',s=1)
#    plt.savefig("liou_eth_H0_L{}_gamma{}.pdf".format(L, gamma))
#    plt.clf()

    # Save eigenvalues and info
    with h5py.File("liou_spec_H0_L{}_gamma{}_all{}lind_run{}.h5".format(L, gamma, ops_string, run),'w') as F:
        F['eigenvalues'] = liou_evals
        F['eigenvectors'] = liou_evecs
        F.attrs['gamma'] = gamma
        F.attrs['L'] = L
        F.attrs['H_info'] = H_info
        F.attrs['Lindblad_info'] = Lindblad_info
        
    end = time.time()
    print("Time elapsed: " + str(end - start))

