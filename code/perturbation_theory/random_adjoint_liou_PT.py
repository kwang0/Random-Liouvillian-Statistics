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
import itertools

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


def CreatePauliStringBasis(spinops):
    l=len(spinops[0])
    print("Creating Pauli string basis for system of length {}".format(l))
    basis={i:[] for i in range(l+1)}
    sx = spr.csr_matrix(np.array([[0.,1.],[1.,0.]]))
    sy = spr.csr_matrix(np.array([[0.,-1.j],[1.j,0.]]))
    sz = spr.csr_matrix(np.array([[1.,0.],[0.,-1.]]))      
    s0 = spr.csr_matrix(np.array([[1.,0.],[0.,1.]]))      

    def get_pauli(i):
        if i==0: return s0
        if i==1: return sx
        if i==2: return sy
        if i==3: return sz

    for string in itertools.product([0,1,2,3], repeat=l):
        op=get_pauli(string[0])
        for i in range(1,l):
            op = spr.kron(op,get_pauli(string[i]))
        strlen = l-string.count(0)
        basis[strlen].append(op)
    return basis


if __name__=="__main__":
    start = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--length", type=int, help = "number of sites in chain")
    parser.add_argument("-o", "--ops", type=int, help="maximum combination of spin operators to include")
    args = parser.parse_args()
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

    gamma = 1.0

    N = 2**L

    Sxs, Sys, Szs  = SpinOps(L)
    Spins = [Sxs, Sys, Szs]

    H_info = "No hamiltonian"

    print("Building Lindblad operator basis")
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
    def adjLmatvec(v):
        O = np.reshape(v,(2**L,2**L))
        LO = np.zeros((2**L,2**L))  + 1.j* np.zeros((2**L,2**L))

        for n in range(nl):
            LdagL = np.conj(Lindbladops[n].T).dot(Lindbladops[n])

            A = np.conj(Lindbladops[n].T).dot( np.dot(O,Lindbladops[n].toarray()) )
            B = (LdagL.dot(O) + np.dot(O,LdagL.toarray()) )
            LO = LO + gamma * D[n,n] * ( A - 0.5 * B )
        return LO.reshape(2**(2*L))
   

    from scipy.sparse.linalg import LinearOperator
    adjLiouvillian = LinearOperator((2**(2*L),2**(2*L)), matvec = adjLmatvec)

    print("Time elapsed: " + str(time.time() - start)) 
    print() 
    print("Creating Liouvillian")
    identity  = np.eye(N**2)
    aLiou = adjLiouvillian.dot(identity) # Dense matrix form of Liou
    print("Time elapsed: " + str(time.time() - start)) 
    print()
    print("Calculating eigenvalues")


    

    liou_evals = np.linalg.eigvals( aLiou )

    print(liou_evals)
        
    pauli_basis = CreatePauliStringBasis(Spins)


    for sector in [1,2,3,4]:
        sector_offdiag_matels=[]
        dim = len(pauli_basis[sector])
        Lsec = np.zeros((dim,dim)) + 1.j* np.zeros((dim,dim))
        for i,op1 in enumerate(pauli_basis[sector]):
            for j,op2 in enumerate(pauli_basis[sector]):
                Lop1 = adjLiouvillian.dot(op1.toarray().reshape(4**L))
                matel = np.vdot( op2.toarray().reshape(4**L), Lop1 )
                Lsec[i,j] = matel
                if i!=j:
                    sector_offdiag_matels.append(matel)
        print(Lsec)

        sector_offdiag_matels=np.array(sector_offdiag_matels)
        sec_std = np.std(sector_offdiag_matels.real)
        plt.hist(sector_offdiag_matels.real,histtype='step',bins=30,label="sec {}, std={:.05f}".format(sector,sec_std),density=True)
    #    plt.hist(sector_offdiag_matels.imag,histtype='step',bins=40)

        print("STD re:", np.std(sector_offdiag_matels.real), "MEAN:",  np.mean(sector_offdiag_matels.real))
        print("STD im:", np.std(sector_offdiag_matels.imag), "MEAN:",  np.mean(sector_offdiag_matels.imag))
        
    plt.legend()
    #plt.plot(liou_evals.real,liou_evals.imag,'r+')
    #plt.imshow(np.abs(Lsec))
    plt.savefig('test.pdf')
