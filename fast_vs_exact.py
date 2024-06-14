# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 21:16:54 2024

@author: Linne
"""

from FuncsAPE import *
import numpy as np
import matplotlib.pyplot as plt
import os 
import xarray as xr

#setting data directory
datadir = datapath + 'Data'
#getting list of all datafiles in directory + how many
allfiles =  os.listdir(datadir)
nfiles = len(allfiles)

#fraction of number of files to take
fraction = 0.01
#generating a random number between 0 and 1 for each file
rands = np.random.random(nfiles)
#creating array of bools. 1 means use the file
filebool = rands < fraction
indexes = np.where(filebool == 1)[0]

file = datadir + allfiles[indexes[0]]

filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
shape = data.salinity.squeeze().shape
#converting lat, z into same shape as data
lat1D = data.lat.to_numpy()
lat = np.zeros(shape)
for i in range(len(lat1D)):
    lat[:, i, :] = lat1D[i]

z1D = data.depth.to_numpy()
z = np.zeros(shape)
for i in range(len(z1D)):
    z[i, :, :] = z1D[i]

depth_bnds = data.depth_bnds.to_numpy()
A_ij = calc_Aij(data)
V_ijk = np.zeros((shape))
for i in range(len(depth_bnds)):
    V_ijk[i, :, :] = (depth_bnds[i, 1] - depth_bnds[i, 0])*A_ij
    
p = pr(z) 

APE_density = calc_APE(datadir, filename, V_ijk, p, z, routine = 'fast',
                       nonegs = False)[2].flatten()
APE_density = APE_density[~np.isnan(APE_density)]

datafast = np.zeros((len(indexes), len(APE_density)))*np.nan
dataexact = datafast.copy()

for i in range(len(indexes)):
    filename = allfiles[indexes[i]]
    fast = calc_APE(datadir, filename, V_ijk, p, z, routine = 'fast',
                    nonegs = False)[2].flatten()
    datafast[i] = fast[~np.isnan(fast)]
    exact = calc_APE(datadir, filename, V_ijk, p,z, routine = 'exact',
                     nonegs = False)[2].flatten()
    dataexact[i] = exact[~np.isnan(exact)]
#%%
dataexact = dataexact.flatten()
datafast = datafast.flatten()

#%%
#only negative numbers
neg_exact = dataexact[dataexact<0]
neg_fast= datafast[datafast<0]

plt.hist(np.abs(neg_exact), label = f'exact: n = {len(neg_exact)}')
plt.hist(np.abs(neg_fast), zorder = 5, label = f'fast: n = {len(neg_fast)}')
plt.title(f'Distribution of Magnitude of Negative APE Values in {len(indexes)} files')
plt.xlabel('APE Magnitude, J/kg')
plt.ylabel('Number of Negative APE Values')
plt.yscale('log')
plt.xscale('log')
plt.legend()

#%%
diff = dataexact - datafast
print(f'''exact is greater than fast for {np.sum(diff>0)/len(diff)} of 
datapoints, indicating that there may be a systematic bias''')


#%%

plt.hist(dataexact[dataexact>0], label = f'exact', color = 'black', alpha = 0.5)
plt.hist(datafast[datafast>0], zorder = -1, label = f'fast', alpha = 0.5)
plt.title(f'Distribution of Magnitude of Positive APE Values in {len(indexes)} files')
plt.xlabel('APE Magnitude, J/kg')
plt.ylabel('Number of Negative APE Values')
plt.yscale('log')
plt.xscale('log')
plt.legend()

#exact seems to have a minimum (both positive and negative) that is 10^-7 
(not true, true min is ^-12, but anyway it seems like a lot of values can't reach this?)

