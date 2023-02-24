#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

def plot_xi(purities, xi_scales):
	for l in range(len(xi_scales)):
		plt.plot(ts,purities[l],label=r"$\xi_{rms} = $" + str(xi_scales[l]))
	plt.legend()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=int, help = "name")
    args = parser.parse_args()

    h5F=h5py.File("data/{}.h5".format(name),'r')

    for key in list(h5F.keys()):
    	if key != 'purities' and key != 'omegas':
    		print(key + ' = ' + h5F['key'])
	purities = h5F['purities']
	omegas = h5F['omegas']
	xi_scales = h5F['xi_scales']

	h5F.close()

	plt.figure(1)
