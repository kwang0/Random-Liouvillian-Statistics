#!/usr/bin/env python3
import numpy as np
import h5py 
import glob
import matplotlib.pyplot as plt
import scipy.optimize
import sys
sys.path.append("../")
import numerics2 as num
import figures_module

fig, ax = figures_module.prepare_standard_figure(ncols=1, nrows=1, sharey=False, sharex=False)


L=10
with h5py.File('../../code/data/op_decay_L{}_run0.h5'.format(L),'r') as F:
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


    ax.plot(t,o0,label='one body, $\gamma={:.04f}$'.format(p0[0]))
    ax.plot(t,o1,label='two body, $\gamma={:.04f}$'.format(p1[0]))
    ax.plot(t,o2,label='three body, $\gamma={:.04f}$'.format(p2[0]))
    ax.plot(t,o3,label='four body, $\gamma={:.04f}$'.format(p3[0]))
    ax.plot(t,o4,label='five body, $\gamma={:.04f}$'.format(p4[0]))
    ax.plot(t,o5,label='six body, $\gamma={:.04f}$'.format(p5[0]))
    

    
    ax.plot(t,np.exp(num.double_decay_rate(L,1)*t),'--',c='k',lw=0.7, alpha=0.6)
    ax.plot(t,np.exp(num.double_decay_rate(L,2)*t),'--',c='k',lw=0.7, alpha=0.6)
    ax.plot(t,np.exp(num.double_decay_rate(L,3)*t),'--',c='k',lw=0.7, alpha=0.6)
    ax.plot(t,np.exp(num.double_decay_rate(L,4)*t),'--',c='k',lw=0.7, alpha=0.6)
    ax.plot(t,np.exp(num.double_decay_rate(L,5)*t),'--',c='k',lw=0.7, alpha=0.6)
    ax.plot(t,np.exp(num.double_decay_rate(L,6)*t),'--',c='k',lw=0.7, alpha=0.6)



#plt.title("Operator decay, L={}".format(L))
ax.set_xlabel("$t$")
ax.set_ylabel(r"Tr$(\rho_0 O(t))$")
ax.set_yscale('log')
ax.legend(loc='best',frameon=False)
fig.tight_layout()
fig.savefig('fig6-operator_decay.pdf')


