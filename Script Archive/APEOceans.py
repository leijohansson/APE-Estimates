# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:33:19 2023

@author: Linne
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')

ocean = 'PO' #PO or IO

max_depth = np.inf
# max_depth = 700
APE = np.zeros(len(os.listdir('APEarrays')))
datadir = 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
depth_bnds = data.depth_bnds.to_numpy()
depths = data.depth.to_numpy()
ndepths = len(depths[depths<max_depth])
filt = np.load(f'{ocean}_filter.npy', allow_pickle = True)[:ndepths, :, :]


idx = 0
for year in range(1950, 2024):
    for month in range(1, 13):
        month = str(month)
        if len(month) == 1:
            month = '0'+month
        APEarr = np.load(f'APEarrays/APE_{year}-{month}.npy')[:ndepths, :, :]
        APE[idx] = np.sum(APEarr*filt)
        idx+=1
        if idx==len(APE):
            break

plt.figure()
plt.title(f'APE ({ocean}, <{max_depth} m)')
plt.plot(APE)
plt.xticks(np.arange(0, len(APE), 120), labels = np.arange(1950, 2024, 10))
plt.xticks(ticks=np.arange(0, len(APE), 12), labels=None, minor=True)
plt.xlabel('Time')
plt.ylabel(r'$APE$')
plt.savefig(f'APE_monthly_{ocean}_{max_depth}.pdf', bbox_inches='tight')