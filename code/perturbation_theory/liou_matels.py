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

def CreatePauliStringBasis(l):
    
    print("Creating Pauli string basis for system of length {}".format(l))
    basis={i:[] for i in range(l+1)}
    basis_strs={i:[] for i in range(l+1)}
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
        basis_strs[strlen].append(string)
    return basis,basis_strs



L = 4
pauli_strings,strings = CreatePauliStringBasis(L)
print(pauli_strings[2])

sec = 2
lindblads = pauli_strings[2] + pauli_strings[1]


nnz=0
for i,sx in enumerate(pauli_strings[sec]):
    for j,sy in enumerate(pauli_strings[sec]):
        terms=[]
        for n,Ln in enumerate(lindblads):
            for m,Lm in enumerate(lindblads):
                tr1 = np.sum(sx.dot(Lm).dot(sy).dot(Ln).diagonal())
                tr2 = -0.5*np.sum(sx.dot(Lm).dot(Ln).dot(sy).diagonal())
                tr3 = -0.5*np.sum(sy.dot(Lm).dot(Ln).dot(sx).diagonal())

                total = (2**(-L))*(tr1+tr2+tr3)
                if np.abs(total.real)>1e-3:
                    terms.append(total.real)

        nterms=len(terms)
        if nterms!=0:
            nnz+=1
        print(i,j, nterms, terms)

print("Found nnz = {}".format(nnz))
print(len(lindblads))
print(len(pauli_strings[sec]))
