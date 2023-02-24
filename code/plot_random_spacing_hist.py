#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors
import h5py

N = 30
alldists = np.array([])
allangles = np.array([])
allratios = np.array([])
neigh = sklearn.neighbors.NearestNeighbors(n_neighbors = 2, radius = 3)
for i in range(1000):
    print("Iteration: " + str(i))
    eigs = np.linalg.eigvals(np.random.randn(N**2, N**2))
    eigs = eigs[eigs.imag > 1e-5]
    coords = np.array([[eigs.real[i], eigs.imag[i]] for i in range(len(eigs))])
    neigh.fit(coords)
    graph = neigh.kneighbors_graph().todense()
    graph = np.array(graph)
#   for i in range(len(coords)):
#       diff = coords[graph[i] == 1][0] - coords[i]
#       alldists.append(np.abs(diff))
#       allangles.append(np.angle(diff[0] + 1.j * diff[1]))
#    graph.sort()
     dists1 = graph[:,-2][:,0]
     dists2 = graph[:,-1][:,0]
     alldists = np.concatenate((alldists, dists1))
     allratios = np.concatenate((allratios, dists1/dists2))
with h5py.File("N30_random_spacing_polar.h5", 'w') as f:
    f['ratios'] = allratios
#   f['dists'] = np.array(alldists)
#   f['angle'] = np.array(allangles)
#plt.hist(alldists, bins = 200, range=[0,2.2])
#plt.savefig("/home/kwang/Documents/Random Liouvillians/data/plots/N20_randommatrix_spacings_hist.pdf")
#plt.figure()
#plt.hist(allratios, bins = 100, range=[0,1])
#plt.savefig("/home/kwang/Documents/Random Liouvillians/data/plots/N20_randommatrix_ratios_hist.pdf")
