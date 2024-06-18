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
from FuncsAPE import calc_APE, datapath
import seaborn as sns
sns.set_context('paper')

year = 2020
month = '10'
datadir = datapath + 'Data' 
#data file has all monthly files inside (no subfolders)
#nothing else in the data file

filename = f'EN.4.2.2.f.analysis.g10.{year}{month}.nc'
data = xr.open_dataset(f'{datadir}/{filename}')

salinity = data.salinity.to_numpy().flatten()
temperature = data.temperature.to_numpy().flatten()

plt.title(f'EN4 Data, {year}-{month}')
plt.scatter(salinity, temperature, color = 'blue', alpha = 0.4,
            label = 'All Points', edgecolors=None)
plt.xlabel('Salinity')
plt.ylabel('Temperature')


file = f'\APEarrays\APE_{year}-{month}.npy'
APE_all = np.load(datapath + file).flatten()
neg_idx = np.where(APE_all == 1)
neg_sal = salinity[neg_idx]
neg_temp = temperature[neg_idx]
plt.scatter(neg_sal, neg_temp, color = 'red', marker = 'd',
            label = 'Points with Negative APE (exact)')
plt.legend()
plt.savefig('EN4 Plots/Temp_salinity_distribution.png', bbox_inches = 'tight')

