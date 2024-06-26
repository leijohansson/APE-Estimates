# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:07:23 2024

@author: Linne
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import pickle
from FuncsAPE import calc_APE, datapath, calc_Aij, pr
import seaborn as sns
sns.set_context('paper')

year = 2020
month = '04'
datadir = datapath + 'Data' 
#data file has all monthly files inside (no subfolders)
#nothing else in the data file

filename = f'EN.4.2.2.f.analysis.g10.{year}{month}.nc'
data = xr.open_dataset(f'{datadir}/{filename}')

shape = data.salinity.squeeze().shape
#converting lat, z into same shape as data
lat1D = data.lat.to_numpy()
lat = np.zeros(shape)
for i in range(len(lat1D)):
    lat[:, i, :] = lat1D[i]
lon1D = data.lon.to_numpy()
lon = np.zeros(shape)
for i in range(len(lon1D)):
    lon[:, :, i] = lon1D[i]


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

fast = calc_APE(datadir, filename, V_ijk, p, z, routine = 'fast', nonegs = False)[2].flatten()
exact = calc_APE(datadir, filename, V_ijk, p, z, routine = 'exact', nonegs = False)[2].flatten()
diff = fast-exact

salinity = data.salinity.to_numpy().flatten()
temperature = data.temperature.to_numpy().flatten()
#%%
plt.figure()
plt.title(f'EN4 Data, {year}-{month}')
plt.scatter(salinity, temperature, color = 'blue', alpha = 0.4,
            label = 'All Points', edgecolors=None)
plt.xlabel('Salinity')
plt.ylabel('Temperature')


neg_idx = np.where(exact < 0)
neg_sal = salinity[neg_idx]
neg_temp = temperature[neg_idx]
plt.scatter(neg_sal, neg_temp, color = 'red', marker = 'd',
            label = 'Points with Negative APE (exact)')
plt.legend()
plt.savefig('EN4 Plots/Temp_salinity_distribution.png', bbox_inches = 'tight')
#%%
plt.figure()
plt.title(f' EN4 Data, {year}-{month}, Points with Negative Exact APE')
df = pd.DataFrame(exact[neg_idx], columns = ['exact'])
df['fast'] = fast[neg_idx]
df.sort_values('exact', ascending = False, inplace=True)


plt.scatter(np.arange(len(df)), df.exact, label = 'exact APE')
plt.scatter(np.arange(len(df)), df.fast, label = 'fast APE')
plt.ylabel('APE, J/kg')
plt.xlabel('Points, sorted by magnitude of exact APE')
plt.legend()



plt.savefig('EN4 Plots/Fast_vs_Exact_negatives.png', bbox_inches = 'tight')

#%%
p99 = np.nanquantile(diff, 0.99)
large_idx = np.where(np.abs(diff)>p99)
large_sal = salinity[large_idx]
large_temp = temperature[large_idx]
plt.figure()
plt.title(f'EN4 Data, {year}-{month}')
plt.scatter(salinity, temperature, color = 'blue', alpha = 0.4,
            label = 'All Points', edgecolors=None)

plt.scatter(large_sal, large_temp, color = 'red', marker = 'd',
            label = 'Difference (fast-exact) > 99%')
plt.xlabel('Salinity')
plt.ylabel('Temperature')
plt.legend()
plt.savefig('EN4 Plots/Temp_salinity_distribution_99pdiff_1.png', bbox_inches = 'tight')


