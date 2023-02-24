#!/usr/bin/env python
import numpy as np
import h5py
import matplotlib.pyplot as plt
import glob
import sklearn.neighbors

fns = list(glob.glob("/data3/kwang/*L7*alldouble*run*"))

alldists1 = []
allangles = []
allratios = np.array([])
slice_spacings = np.array([])
slice_ratios = np.array([])
real_spacings = []
real_ratios = []
imag_spacings = []
imag_ratios = []
neigh = sklearn.neighbors.NearestNeighbors(n_neighbors = 2, radius=1)
for fn in fns:
    print("Plotting " + fn)
    eigs = h5py.File(fn, 'r')['eigenvalues'][...]
#   real_slice = eigs[np.abs(eigs.imag) < 1e-5].real # Slice of eigs on real line
#   real_slice.sort()
#   diff = real_slice[1:-1] - real_slice[:-2] # Exclude steady state eig
#    print(diff)
#   slice_spacings = np.concatenate((slice_spacings, diff))
#   ratio = diff[1:] / diff[:-1]
#   slice_ratios = np.concatenate((slice_ratios, np.min([ratio, 1/ratio], axis = 0)))
    
    eigs = eigs[eigs.imag > 1e-5] # Eigs outside real line
    eigs = eigs[eigs.imag < 0.013]
    eigs = eigs[eigs.real < -0.94]
    eigs = eigs[eigs.real > -1.02]
    coords = np.array([[eigs.real[i], eigs.imag[i]] for i in range(len(eigs))])

    neigh.fit(coords)
    graph = neigh.kneighbors_graph(mode='distance').todense()
    graph.sort()
    graph = np.array(graph)
#   for i in range(len(coords)):
#       coord = coords[i]
#       diff = coords[graph[i] == 1][0] - coord
#       alldists1.append(np.sqrt(np.sum(diff ** 2)))
#       allangles.append(np.angle(diff[0] + 1.j * diff[1]))

#       diff2 = np.abs(coords[graph[i] == 1][1] - coord)
#       real_spacings.append(min(diff1[0], diff2[0]))
#       real_ratio = diff1[0] / diff2[0]
#       real_ratios.append(min(real_ratio, 1/real_ratio))
#       imag_spacings.append(min(diff1[1], diff2[1]))
#       imag_ratio = diff1[1] / diff2[1]
#       imag_ratios.append(min(imag_ratio, 1/imag_ratio))

    dists1 = graph[:,-2]
    dists2 = graph[:,-1]
    alldists1 = np.concatenate((alldists1, dists1))
    allratios = np.concatenate((allratios, dists1/dists2))
#plt.figure(1)
#plt.hist(slice_spacings, bins=50, range=[0,0.1])
#plt.savefig("/home/kwang/Documents/Random Liouvillians/data/plots/N20_slice_spacings_hist.pdf")
#
#plt.figure(2)
#plt.hist(slice_ratios, bins=50, range=[0,1])
#plt.savefig("/home/kwang/Documents/Random Liouvillians/data/plots/N20_slice_ratios_hist.pdf")
#
#plt.figure(3)
#plt.subplot(211)
#plt.hist(real_spacings, bins=200, range=[0,0.1])
#plt.title('Real')
#plt.subplot(212)
#plt.hist(imag_spacings, bins=200, range=[0,0.1])
#plt.title('Imag')
#plt.savefig("/home/kwang/Documents/Random Liouvillians/data/plots/N20_nonslice_spacings_hist.pdf")
#
#plt.figure(4)
#plt.subplot(211)
#plt.hist(real_ratios, bins=100, range=[0,1])
#plt.title('Real')
#plt.subplot(212)
#plt.hist(imag_ratios, bins=100, range=[0,1])
#plt.title('Imag')
#plt.savefig("/home/kwang/Documents/Random Liouvillians/data/plots/N20_nonslice_ratios_hist.pdf")
with h5py.File("L7_alldoublelind_overlap_ratios.h5", 'w') as f:
    f['ratios'] = allratios
#   f['dists'] = np.array(alldists1)
#   f['angles'] = np.array(allangles)
