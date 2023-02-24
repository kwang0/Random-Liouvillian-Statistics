#!/usr/bin/env python
import scipy.sparse as spr
import numpy as np
import scipy.special as sp
import scipy.linalg as spl
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator
import argparse

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
        Lindbladops_new.append(op.toarray())
    return Lindbladops_new



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--length", type=int, help = "number of sites in chain")
    args = parser.parse_args()
    L = args.length
    N = 2**L
    ops = 2

    Sxs, Sys, Szs  = SpinOps(L)
    Spins = [Sxs, Sys, Szs]
    H = spr.csr_matrix((2**L,2**L))
    gamma = 1
 #   print("Ham:", H.toarray())
    
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

#    print("K:", D)

    Lindbladops_old = Lindbladops
    Lindbladops = TransformLindblad(Lindbladops, U, L)

    print("Lindblads transformed to diag Kossakowski basis")

    # Create the Liouvillian
    def Lmatvec(v):
        rho = np.reshape(v,(2**L,2**L))
        
        Lrho = np.zeros((2**L,2**L))
#        Lrho = -1.j*(H.dot(rho) - (H.T.dot(rho.T)).T) # FIXME we don't have a Hamiltonian here

        for n in range(nl):
            X = Lindbladops[n].dot(rho)
            Lrho = Lrho + gamma * D[n,n] * ( (np.conj(Lindbladops[n]).dot(X.T)).T - 0.5 * (np.conj(Lindbladops[n].T).dot(X) + (Lindbladops[n].T.dot(np.conj(Lindbladops[n]).dot(rho.T))).T) )
        return Lrho.reshape(2**(2*L))
    

    def adjLiouvillian():
        """
        Calculates the adjoint Liouvillian in superoperator form.
        Explicitly:
        $$ \sum_i  ( D_i L_i^\dag O L_i - 1/2 [L_i^\dag L_i O  + O L_i^\dag L_i ] ) $$

        WARNING: THERE IS NO HAMILTONIAN!
        """
        Liou = spr.csr_matrix((N**2,N**2))

        for n in range(nl):
            Lop = Lindbladops[n]
            Ldag = np.conj(Lop.T)
            prod = Ldag.dot(Lop)
            Liou = Liou + gamma * D[n,n] * (spr.kron(Ldag,Lop.T) - 0.5 * (spr.kron(prod,spr.eye(N)) + spr.kron(spr.eye(N),prod.T)))
        return Liou.todense()    

    def Expectation(op, v):
        rho = np.reshape(v, (2**L,2**L))
        return np.abs(np.trace(op.dot(rho)))
    

    adjLiou = adjLiouvillian()

    op1 = Spins[2][0]
    op2 = Spins[2][0]*Spins[2][1]
    op3 = Spins[2][0]*Spins[2][1]*Spins[2][2]
    op4 = Spins[2][0]*Spins[2][1]*Spins[2][2]*Spins[2][3]
    op5 = Spins[2][0]*Spins[2][1]*Spins[2][2]*Spins[2][3]*Spins[2][4]
    op6 = Spins[2][0]*Spins[2][1]*Spins[2][2]*Spins[2][3]*Spins[2][4]*Spins[2][5]
 
    rho_0 = np.zeros(4**L)
    random_index = np.random.randint(2**L)
    rho_0[random_index + random_index* 2**L] = 1 ## "diagonal" product state
    
    from scipy.integrate import ode
    import h5py
    import os

    def LadjO(t,O):
        return adjLiou.dot(O)

    r = ode(LadjO).set_integrator('zvode', method='bdf')


    os.makedirs("data/",exist_ok=True)
    h5F=h5py.File("data/op_decay_L{}_fast.h5".format(L),'w')

    for Oi,op in enumerate([op1,op2,op3,op4,op5,op6]):
        print("starting integration for {} body operators".format(Oi+1))
        O0 = np.array(op.toarray().reshape(4**L),dtype='complex128')

        t0=0.0
        t1=10.0
        dt=0.001
        r.set_initial_value(O0,t0)

        times = []
        opexp = []

        while r.successful() and r.t < t1:
            t=r.t+dt
            Ot = r.integrate(r.t+dt)


            opt = np.reshape(Ot,(2**L,2**L))
            expval  = Expectation(opt, rho_0)

            times.append(t)
            opexp.append(expval)

            print(r.t+dt, expval)

        if 'times' not in h5F:
            h5F['times']=times
        h5F["exp[O{}]".format(Oi)]=opexp
    h5F.close()



