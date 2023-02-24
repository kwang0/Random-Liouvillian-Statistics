import numpy as np
import h5py
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
filename = "liou_spec_H0_L5_gamma1_singlelind.h5"
f = h5py.File(filename, 'r')
eigs = f['eigenvalues'][...]
evecs = f['eigenvectors'][...]

norm_squareds = (np.abs(evecs)**2).T

def dist(x,y):
    d = 0.0
    N = len(x)
    for i in range(N):
        d -= 0.5 * x[i] * np.log(y[i] / x[i])
        d -= 0.5 * y[i] * np.log(x[i] / y[i])
    return d

e = 0.5

print("Building neighbor graph")
neigh = NearestNeighbors(radius = 1, metric = dist)
neigh.fit(norm_squareds)
graph = neigh.radius_neighbors_graph(norm_squareds, mode='distance')

print("Clustering")
clustering = DBSCAN(eps = e, metric='precomputed')
clustering.fit(graph)
with h5py.File("KLcluster_eps{}_".format(e) + filename,'w') as F:
    F['eigenvalues'] = eigs
    F['eigenvectors'] = evecs
    F['clusters'] = clustering.labels_

