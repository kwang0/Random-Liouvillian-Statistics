#!/usr/bin/env python3
import numpy as np
import h5py 
import glob
import matplotlib.pyplot as plt
import scipy.optimize
import sys
import numerics as num

L=10
with h5py.File('op_decay_L{}_run0.h5'.format(L),'r') as F:
    print(list(F))

    t= F['times'][...]
    o0=F['exp[O0]'][...]
    o1=F['exp[O1]'][...]
    o2=F['exp[O2]'][...]
    o3=F['exp[O3]'][...]
    o4=F['exp[O4]'][...]
    o5=F['exp[O5]'][...]
    

    # fit decay rates
    expfun = lambda x,a,b: b*np.exp(a*x)
    fitmask = t<4
    p0,c = scipy.optimize.curve_fit(expfun,t[fitmask],o0[fitmask])
    p1,c = scipy.optimize.curve_fit(expfun,t[fitmask],o1[fitmask])
    p2,c = scipy.optimize.curve_fit(expfun,t[fitmask],o2[fitmask])
    p3,c = scipy.optimize.curve_fit(expfun,t[fitmask],o3[fitmask])
    p4,c = scipy.optimize.curve_fit(expfun,t[fitmask],o4[fitmask])
    p5,c = scipy.optimize.curve_fit(expfun,t[fitmask],o5[fitmask])


    plt.plot(t,o0,label='one body, $\gamma={:.04f}$'.format(p0[0]))
    plt.plot(t,o1,label='two body, $\gamma={:.04f}$'.format(p1[0]))
    plt.plot(t,o2,label='three body, $\gamma={:.04f}$'.format(p2[0]))
    plt.plot(t,o3,label='four body, $\gamma={:.04f}$'.format(p3[0]))
    plt.plot(t,o4,label='five body, $\gamma={:.04f}$'.format(p4[0]))
    plt.plot(t,o5,label='six body, $\gamma={:.04f}$'.format(p5[0]))
    
    plt.plot(t,np.exp(num.double_decay_rate(L,1)*t),'--')
    plt.plot(t,np.exp(num.double_decay_rate(L,2)*t),'--')
    plt.plot(t,np.exp(num.double_decay_rate(L,3)*t),'--')
    plt.plot(t,np.exp(num.double_decay_rate(L,4)*t),'--')
    plt.plot(t,np.exp(num.double_decay_rate(L,5)*t),'--')
    plt.plot(t,np.exp(num.double_decay_rate(L,6)*t),'--')


plt.title("Operator decay, L={}".format(L))
plt.xlabel("time")
plt.ylabel("$Tr(rho_0 O(t))$")
plt.yscale('log')
plt.legend(loc='best',frameon=False)
plt.savefig('L{}_relaxation.png'.format(L), dpi=800)


