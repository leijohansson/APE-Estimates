# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:09:49 2023

@author: Linne

Old APE using coordinates for IO and PO provided by Remi
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr
import seaborn as sns
sns.set_context('paper')

# max_depth = 700
max_depth = np.inf

APE = np.zeros(len(os.listdir('APEarrays')))
datadir = 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
depth_bnds = data.depth_bnds.to_numpy()
depths = data.depth.to_numpy()
ndepths = len(depths[depths<max_depth])
BGE = np.zeros(len(os.listdir('BGEarrays')))
V_ijk = np.load('dVolume.npy', allow_pickle=True)[:ndepths, :, :]

idx = 0
for year in range(1950, 2024):
    for month in range(1, 13):
        month = str(month)
        if len(month) == 1:
            month = '0'+month
        APEarr = np.load(f'APEarrays/APE_{year}-{month}.npy')[:ndepths, :, :]
        BGEarr = np.load(f'BGEarrays/BGE_{year}-{month}.npy')[:ndepths, :, :]
        volbool = (BGEarr>0).astype(int)
        BGE[idx] = np.sum(BGEarr)/np.sum(V_ijk*volbool)
        # pot_temp = np.sum(BGE)#

        idx+=1
        if idx==len(APE):
            break
plt.figure(figsize = (6.5, 4))
plt.title(f'APE (depth < {max_depth}m)')
plt.plot(APE)
plt.xticks(np.arange(0, len(APE), 120), labels = np.arange(1950, 2024, 10))
plt.xticks(ticks=np.arange(0, len(APE), 12), labels=None, minor=True)
plt.xlabel('Time')
plt.ylabel(r'$APE$')
plt.savefig(f'APE_monthly_{max_depth}.pdf', bbox_inches='tight')


plt.figure(figsize = (6.5, 4))
plt.title(f'BGE (depth < {max_depth}m)')
plt.plot(BGE)
plt.xticks(np.arange(0, len(APE), 120), labels = np.arange(1950, 2024, 10))
plt.xticks(ticks=np.arange(0, len(APE), 12), labels=None, minor=True)
plt.xlabel('Time')
plt.ylabel(r'BGE density (per $m^3$)')
plt.savefig(f'BGEmonthly_{max_depth}.pdf', bbox_inches='tight')