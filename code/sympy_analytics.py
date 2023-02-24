from sympy import *
from sympy.physics.paulialgebra import Pauli, evaluate_pauli_product
from sympy.physics.quantum import TensorProduct
import sympy.physics.quantum.tensorproduct as tp
import numpy as np
import pickle

def mul(x1, x2):
    return tp.tensor_product_simp_Mul(x1 * x2)

def kossterm(gamma, L1, L2, O):
    return gamma * (mul(L1, mul(O, L2)) - 0.5 * (mul(L2, mul(L1, O)) + mul(O, mul(L2, L1))))

def lindblad(gammas, L_alphas, O):
    s = 0
    for i, L1 in enumerate(L_alphas):
        for j, L2 in enumerate(L_alphas):
            s += kossterm(gammas[i,j], L1, L2, O)   
    return s

def spinOps(L):
    Sxs = []
    Sys = []
    Szs = []
    for i in range(L):
        Sx = TensorProduct(*[1 if j != i else Pauli(1) for j in range(L)])
        Sy = TensorProduct(*[1 if j != i else Pauli(2) for j in range(L)])
        Sz = TensorProduct(*[1 if j != i else Pauli(3) for j in range(L)])
        Sxs.append(Sx)
        Sys.append(Sy)
        Szs.append(Sz)
    return [Sxs, Sys, Szs]




L = 3
N = 2**L
Spins = spinOps(L)

print("Generating basis")
all_Ls = []
L_alphas = []
for a in range(3):
    for i in range(L):
        all_Ls.append(Spins[a][i])
        for b in range(3):
            for j in range(i):
                all_Ls.append(mul(Spins[a][i], Spins[b][j]))
                L_alphas.append(mul(Spins[a][i], Spins[b][j]))
                for c in range(3):
                    for k in range(j):
                        all_Ls.append(mul(Spins[a][i], mul(Spins[b][j], Spins[c][k])))
#                        for d in range(3):
#                            for l in range(k):
#                                all_Ls.append(Spins[a][i] * Spins[b][j] * Spins[c][k] * Spins[d][l])


nl = len(L_alphas)

print("Building Kossakowski symbolic coefficients")
# Kossakowski symbolic values
gammas = np.empty((nl,nl), dtype = object)
for i in range(nl):
    gammas[i,i] = symbols('g' + str(i) + '\,' + str(i))
    for j in range(i):
        r = symbols('r' + str(i) + '\,' + str(j), real=True)
        m = symbols('m' + str(i) + '\,' + str(j), real=True)
        gammas[i,j] = r + I * m
        gammas[j,i] = r - I * m
#        gammas[i,j] = symbols('g' + str(i) + '\,' + str(j))

# Arbitrary linear combination of single-site Lindblad operators
coeffs = []
single_Ls = []
for i in range(3):
    coeff1 = symbols('c' + str(2*i))
    coeff2 = symbols('c' + str(2*i+1))
    coeffs.append(coeff1)
    coeffs.append(coeff2)
    
    single_Ls.append(TensorProduct(Pauli(i + 1), 1))
    single_Ls.append(TensorProduct(1, Pauli(i + 1)))

s = 0

print("Building expression")
#for i in range(6):
#    s += lindblad(gammas, L_alphas, coeffs[i] * single_Ls[i])
O = Spins[0][0]
s = lindblad(gammas, L_alphas, O)


print("Simplifying")
# Evaluate all squared Pauli terms
for L_op in all_Ls:
    s = s.subs(L_op**2,1)

# Collect coefficients
for L_op in all_Ls:
    s = collect(s, L_op)

s = s.simplify()

#N = 4
#eps = symbols('eps')
# Random Kossakowski
#D = np.diag(np.random.rand(nl))
#D = np.diag(1/nl * ((1 - eps) * np.ones(nl) + 2 * eps * np.random.random(nl)))
#D /= np.trace(D)
#D *= N
#
#
#A = np.random.randn(nl, nl) + 1.0j * np.random.randn(nl, nl)
#Q, R = np.linalg.qr(A)
#K = np.conj(Q.T).dot(D).dot(Q)

#print("Substituting Kossakowski values")
#for i in range(nl):
#    for j in range(nl):
#        s = s.subs(gammas[i,j], K[i,j])

#print("Generating Liouvillian matrix on single-site observables")
#Liou = np.zeros((6,6), dtype = complex)
#for i in range(6):
#    for j in range(6):

#with open('/data3/kwang/L{}_onlydouble_measuresingle.p'.format(L), 'wb') as f:
#    pickle.dump({'expr':s, 'gammas':gammas, 'L_alphas':L_alphas, 'O':O, 'spins':Spins}, f)

