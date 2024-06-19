# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:34:18 2024

@author: Linne

Plot time series of volume integrated APE for different ocean basins
using EN4 data.
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import pickle 
import matplotlib.dates as mdates
from FuncsAPE import find_depthfracs, datapath
import seaborn as sns
palette = sns.color_palette("tab10", 10)[::2]
sns.set_theme(context='paper', style='white', palette=palette,
              rc={'xtick.bottom': True,'ytick.left': True,})
#%%
# from polygons_EN4 import oceans
#set max depth to take
max_depth = np.inf

#loading filters
with open('RegionFilters/ocean_filters-EN4.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)
dictkey = list(ocean_filters.keys())[0]
ocean_filters['World'] = np.ones(ocean_filters[dictkey].shape)

#set time range
startyear = 1960
endyear = 2020    

n_months = (endyear + 1 - startyear)*12
x_time =  pd.date_range(f'{startyear}-01-01', periods=n_months, freq='m')

#making array to account for maximum depth
datadir = datapath
filename = 'Data/EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
depth_bnds = data.depth_bnds.to_numpy()
dz = depth_bnds[:, 1] - depth_bnds[:, 0]

depth_fracs = find_depthfracs(dz, data.salinity.squeeze().shape, max_depth)


#creating arrays to put time series data into
TS_oceans = {}
for OB in ocean_filters.keys():
    TS_oceans[OB] = np.zeros((n_months))


#Calculating volume integrated APE for each ocean basin at each time step
time_id = -1
for year in range(startyear, endyear+1):
    for month in range(1, 13):
        time_id += 1
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)
            
        file = f'\APEarrays\APE_{year}-{month}.npy'
        APE_all = np.load(datadir + file)
        #calculating APE up to depth 700
        APE_700 = np.sum(APE_all*depth_fracs, axis = 0)
        
        for OB in ocean_filters:
            APE_ocean = np.nansum(APE_700 * ocean_filters[OB])
            TS_oceans[OB][time_id] = APE_ocean
#%%
#number of months to take mean over
rolling_n = 12
#creating dataframe
df = pd.DataFrame()
for OB in ocean_filters.keys():
    df[OB] = TS_oceans[OB]
rolling = df.rolling(rolling_n, center = True)

#%%
#Plots
fig, axs = plt.subplots(4, 3, figsize=(12, 15))
f_i = 0
for OB in TS_oceans.keys():
    #plotting data
    axs[f_i//3, f_i%3].plot(x_time, TS_oceans[OB], alpha = 0.6, label = 'Data')
    #plotting rolling mean
    axs[f_i//3, f_i%3].plot(x_time, rolling.mean()[OB], color = 'black', label = f'{rolling_n} month rolling mean')
    axs[f_i//3, f_i%3].set_title(OB)
    axs[f_i//3, f_i%3].set_ylabel('Volume Integrated APE, $J$')
    f_i += 1
    

axs[3, 0].set_xlabel('Time')
axs[3, 1].set_xlabel('Time')
fig.suptitle(f'Volume Integrated APE, depths < {max_depth}')
fig.tight_layout()

axs[-1, -1].axis('off')
axs[-1, -1].plot([], alpha = 0.6, label = 'Data')
axs[-1, -1].plot([], color = 'black', label = f'{rolling_n} month rolling mean')
axs[-1, -1].legend(fontsize = 12, loc = 'upper left')

# fig.savefig('EN4 Plots/Ocean_APE_TS.pdf', bbox_inches = 'tight')

#%%
def plot_one(OB, ax):
    ax.set_title(OB)
    ax.plot(x_time, TS_oceans[OB], alpha = 0.6, label = 'Data')
    #plotting rolling mean
    ax.plot(x_time, rolling.mean()[OB], color = 'black', label = f'{rolling_n} month rolling mean')
    ax.set_xlabel('Time')
    ax.set_ylabel('Volume Integrated APE, $J$')

    
#%%
#pacific
fig, axs = plt.subplots(2,  1, sharex = True)
plot_one('North Pacific Ocean', axs[0])
plot_one('South Pacific Ocean', axs[1])
fig.tight_layout()
axs[0].legend(loc = 'upper right')
plt.savefig('EN4 Plots/Pacific_TS.pdf')

#%%
fig, axs = plt.subplots(2,  1, sharex = True)
plot_one('North Atlantic Ocean', axs[0])
plot_one('South Atlantic Ocean', axs[1])
fig.tight_layout()
axs[0].legend(loc = 'upper right')
plt.savefig('EN4 Plots/Atlantic_TS.pdf')

#%%
fig, axs = plt.subplots(2,  1, sharex = True)
plot_one('Arctic Ocean', axs[0])
plot_one('Southern Ocean', axs[1])
fig.tight_layout()
axs[0].legend(loc = 'lower left')
plt.savefig('EN4 Plots/Poles_TS.pdf')

print('Peak at:', x_time[np.where(TS_oceans['Arctic Ocean'] == TS_oceans['Arctic Ocean'].max())][0])
x_warm = pd.date_range('2016-01-30', periods=1, freq='m')

print('Arctic warming ', x_warm)
axs[0].vlines(x_warm, 2.7e20, 3e20)
#coincides with arctic warming. 
#%%
fig, ax = plt.subplots(1,  1, sharex = True, figsize = (6.4, 3))
plot_one('Indian Ocean', ax)
fig.tight_layout()
ax.legend(loc = 'upper left')
plt.savefig('EN4 Plots/Indian_TS.pdf')
    
        
