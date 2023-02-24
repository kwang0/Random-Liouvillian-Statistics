#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


x1 = -10.0
x2 = -20.0
x3 = -30.0

n1=20
n2=50
n3=70
eps=0.02

diag = np.concatenate([x1*np.ones(n1), x2*np.ones(n2), x3*np.ones(n3)])

D = np.diag(diag)
print(D)


ntot=n1+n2+n3
N = eps* (np.random.randn(ntot,ntot) + 1.j*np.random.randn(ntot,ntot) )

L = D + N

eigs = np.linalg.eigvals(L)

print(eigs)



### PERTURBATION THEORY
L0 = L[:n1,:n1]
eigs0 = np.linalg.eigvals(L0)
plt.plot(eigs0.real,eigs0.imag,'r+')


L1 = L[n1:n1+n2,n1:n1+n2]
eigs1 = np.linalg.eigvals(L1)
plt.plot(eigs1.real,eigs1.imag,'g+')


L2 = L[n1+n2:,n1+n2:]
eigs2 = np.linalg.eigvals(L2)
plt.plot(eigs2.real,eigs2.imag,'k+')




plt.plot(eigs.real,eigs.imag,'o',markersize=2)
plt.plot([x1,x2,x3],[0,0,0],'r.')
plt.savefig("toy-eigvals.pdf")
