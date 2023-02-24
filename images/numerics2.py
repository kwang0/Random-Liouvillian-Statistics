#!/usr/bin/env python
from scipy.special import comb
import sys


def single(L,n):
    return 2*n

def double(L,n):
    return ((6*n*L) - (4*n**2) - (2*n))

def triple(L,n):
    s = 2 * n * comb(n - 1, 2)
    s += 2 * n * (n - 1) * 3 * (L - n)
    s += 2 * n * comb(L - n, 2)
    s += 2 ** 3 * comb(n,3)
    return s

def n_ops(L, basis_ops):
    s = 0
    for i in range(1, int(basis_ops) + 1):
        s += comb(L, i) * 3**i
    return s

def double_decay_rate(L,n):
    return -2*(single(L,n) + double(L,n))/n_ops(L,2)
