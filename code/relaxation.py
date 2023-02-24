#!/usr/bin/env python
import scipy.sparse as spr
import numpy as np
import scipy.special as sp
import scipy.linalg as spl
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator

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

def RandomKossakowski(N, nl):
    # Random diagonal matrix
    D = np.diag(np.random.rand(nl))
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

    L = 4
    N = 2**L
    ops = 2

    Sxs, Sys, Szs  = SpinOps(L)
    Spins = [Sxs, Sys, Szs]
    H = spr.csr_matrix((2**L,2**L))
    gamma = 1
    print("Ham:", H.toarray())
    
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

    nl = len(Lindbladops)
    D, U = RandomKossakowski(N, nl)

    print("K:", D)

    Lindbladops_old = Lindbladops
    Lindbladops = TransformLindblad(Lindbladops, U, L)

    # Create the Liouvillian
    def Lmatvec(v):
        rho = np.reshape(v,(2**L,2**L))

#        Lrho = -1.j*(H.dot(rho) - (H.T.dot(rho.T)).T) # FIXME we don't have a Hamiltonian here

        for n in range(nl):
            X = Lindbladops[n].dot(rho)
            Lrho = Lrho + gamma * D[n,n] * ( (np.conj(Lindbladops[n]).dot(X.T)).T - 0.5 * (np.conj(Lindbladops[n].T).dot(X) + (Lindbladops[n].T.dot(np.conj(Lindbladops[n]).dot(rho.T))).T) )
        return Lrho.reshape(2**(2*L))
    
    def Expectation(op, v):
        rho = np.reshape(v, (2**L,2**L))
        return np.abs(np.trace(op.dot(rho)))
    

    identity = np.eye(N**2)
    Liouvillian = LinearOperator((2**(2*L),2**(2*L)), matvec = Lmatvec)
    Liou = Liouvillian.dot(identity)
    Liou = spr.csr_matrix(Liou)
    
    rho_0 = np.zeros(4**L)
    random_index = np.random.randint(2**L)
    rho_0[random_index * 2**L] = 1
    
    op1 = Spins[0][0]
    op2 = Spins[0][0] * Spins[0][1]
    op3 = Spins[0][0] * Spins[0][1] * Spins[0][2]
    
    ts = np.zeros(101)
    expecs = np.zeros((3,101))
    dt = 0.1
    t = 0.0
    for i in range(1000):
        print("t: " + str(t))
        ts[i] = t
        expecs[0][i] = Expectation(op1, rho_0)
        expecs[1][i] = Expectation(op2, rho_0)
        expecs[2][i] = Expectation(op3, rho_0)
        t += dt
        rho_0 = spr.linalg.expm_multiply(Liou * dt, rho_0)
    
    ts[-1] = 2
    expecs[0][-1] = Expectation(op1, rho_0)
    expecs[1][-1] = Expectation(op2, rho_0)
    expecs[2][-1] = Expectation(op3, rho_0)

