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
from scipy import stats
import seaborn as sns
palette = sns.color_palette("tab10", 10)[::2]
sns.set_theme(context='paper', style='white', palette=palette,
              rc={'xtick.bottom': True,'ytick.left': True,})
#%%
# from polygons_EN4 import oceans
#set max depth to take
max_depth = np.inf

#loading filters
with open('RegionFilters/ocean_filters-EN4_-45.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)
dictkey = list(ocean_filters.keys())[0]
ocean_filters['World'] = np.ones(ocean_filters[dictkey].shape)

#loading filters
with open(f'RegionFilters/ocean_filters-EN4_-30.pkl', 'rb') as f:
    ocean_filters30 = pickle.load(f)
ocean_filters['IO30'] = ocean_filters30['Indian Ocean']

with open(f'RegionFilters/ocean_filters-EN4_-45.pkl', 'rb') as f:
    IO45 = np.nan_to_num(pickle.load(f)['Indian Ocean'])
with open(f'RegionFilters/ocean_filters-EN4_-30.pkl', 'rb') as f:
    IO30 = np.nan_to_num(pickle.load(f)['Indian Ocean'])
    
ocean_filters['IOBand'] = IO45-IO30


#set time range
startyear = 1960
endyear = 2023

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
    axs[f_i//3, f_i%3].plot(x_time, TS_oceans[OB], alpha = 0.6, label = 'Data', color = 'royalblue')
    #plotting rolling mean
    axs[f_i//3, f_i%3].plot(x_time, rolling.mean()[OB], color = 'black', label = f'{rolling_n} month rolling mean')
    axs[f_i//3, f_i%3].set_title(OB)
    axs[f_i//3, f_i%3].set_ylabel('Volume Integrated APE, $J$')
    f_i += 1
    axs[f_i//3, f_i%3].spines[['right', 'top']].set_visible(False)

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
    # ax.set_title(OB)
    ax.plot(x_time, TS_oceans[OB], alpha = 0.6, label = 'Data', color = 'royalblue')
    #plotting rolling mean
    ax.plot(x_time, rolling.mean()[OB], color = 'black', label = f'{rolling_n} month rolling mean')
    ax.set_xlabel('Time')
    ax.set_ylabel('Volume Integrated APE, $J$')

def lin_regress_subsetyears(OB, startyear = 0, endyear = np.inf, ax = None, alternative = 'two-sided'):
    bool1 = (x_time.year<=endyear).astype(int)
    bool2 = (x_time.year>=startyear).astype(int)
    time_sel = np.where((bool1+bool2) == 2)
    
    slope, intercept, r, p, std_err = stats.linregress((np.arange(len(x_time))/12)[time_sel], TS_oceans[OB][time_sel], alternative = alternative
                                                       )
    print(f'OB {startyear}-{endyear}, Trend : {slope/1e17} +/- {std_err/1e17} x 10^17 J/year,')
    print(f'R2 value: {r}, p_value: {p}')
    if ax: 
        ax.plot(x_time[time_sel], slope*(np.arange(len(x_time))/12)[time_sel] + intercept, lw = '2', color = 'red', label = 'Trend')

#%%
fig, ax = plt.subplots(1,  1, figsize = (3.5, 3))#, sharex = True)
plot_one('North Pacific Ocean', ax)    
ax.spines[['right', 'top']].set_visible(False)

ax.legend(loc = 'upper right')
fig.tight_layout()
plt.savefig('EN4 Plots/TS Plots/NPO_TS.pdf')

fig, ax = plt.subplots(1,  1, figsize = (3.5, 3))#, sharex = True)
plot_one('South Pacific Ocean', ax)
lin_regress_subsetyears('South Pacific Ocean', 1990, 2015, ax)
lin_regress_subsetyears('South Pacific Ocean')

ax.spines[['right', 'top']].set_visible(False)

ax.legend(loc = 'upper left')
fig.tight_layout()
plt.savefig('EN4 Plots/TS Plots/SPO_TS.pdf')



#%%
fig, ax = plt.subplots(1,  1, figsize = (3.5, 3))#, sharex = True)
plot_one('North Atlantic Ocean', ax)
ax.spines[['right', 'top']].set_visible(False)

# ax.legend(loc = 'upper right')
fig.tight_layout()
plt.savefig('EN4 Plots/TS Plots/NAO_TS.pdf')
lin_regress_subsetyears('North Atlantic Ocean')

fig, ax = plt.subplots(1,  1, figsize = (3.5, 3))#, sharex = True)
plot_one('South Atlantic Ocean', ax)
ax.spines[['right', 'top']].set_visible(False)

ax.legend(loc = 'upper right')
fig.tight_layout()
plt.savefig('EN4 Plots/TS Plots/SAO_TS.pdf')
lin_regress_subsetyears('South Atlantic Ocean')


#%%
fig, ax = plt.subplots(1,  1, figsize = (6, 3))#, sharex = True)
plot_one('Southern Ocean', ax)
ax.spines[['right', 'top']].set_visible(False)
print('Peak at:', x_time[np.where(rolling.mean()['Southern Ocean'] == rolling.mean()['Southern Ocean'].min())][0])
x_dip = pd.date_range('2006-06-01', periods=1, freq='m')
ax.axvline(x_dip, label = '2006-06', linestyle = '--', color = 'red', zorder = -1)
ax.legend(loc = 'lower left')
fig.tight_layout()
plt.savefig('EN4 Plots/TS Plots/Southern_TS.pdf')

print('SO')
#%%
fig, ax = plt.subplots(1,  1, figsize = (6, 3))#, sharex = True)
plot_one('Arctic Ocean', ax)
ax.spines[['right', 'top']].set_visible(False)
# lin_regress_subsetyears('Arctic Ocean', ax = ax)


print('Peak at:', x_time[np.where(TS_oceans['Arctic Ocean'] == TS_oceans['Arctic Ocean'].max())][0])
x_warm = pd.date_range('2016-09-01', periods=1, freq='m')
ax.axvline(x_warm, label = '2016-09', linestyle = '--', color = 'red', zorder = -1)

print('Dip at:', x_time[np.where(TS_oceans['Arctic Ocean'][:int(0.7*len(x_time))] == TS_oceans['Arctic Ocean'][:int(0.7*len(x_time))].min())][0])
x_dip = pd.date_range('1990-10-01', periods=1, freq='m')
ax.axvline(x_dip, label = '1990-10', linestyle = '-.', color = 'blue', zorder = -1)

ax.legend(loc = 'lower left')


fig.tight_layout()
plt.savefig('EN4 Plots/TS Plots/Arctic_TS.pdf')


#coincides with arctic warming. 
#%%
fig, ax = plt.subplots(1,  1, figsize = (6, 3))#, sharex = True)
plot_one('Indian Ocean', ax)
ax.legend(loc = 'upper left')
ax.spines[['right', 'top']].set_visible(False)

fig.tight_layout()
plt.savefig('EN4 Plots/TS Plots/Indian_TS.pdf')
    
#%%
fig, ax = plt.subplots(1,  1, figsize = (7, 3))#, sharex = True)
plot_one('World', ax)
fig.tight_layout()
lin_regress_subsetyears('World', ax = ax)
ax.legend(loc = 'lower left')
ax.spines[['right', 'top']].set_visible(False)

plt.savefig('EN4 Plots/TS Plots/World_TS.pdf')

#%%
fig, ax = plt.subplots(1,  1, figsize = (3.5, 3))#, sharex = True)
plot_one('IO30', ax)    
ax.spines[['right', 'top']].set_visible(False)

# ax.legend(loc = 'upper right')
fig.tight_layout()
plt.savefig('EN4 Plots/TS Plots/IO30_TS.pdf')

fig, ax = plt.subplots(1,  1, figsize = (3.5, 3))#, sharex = True)
plot_one('IOBand', ax)
ax.spines[['right', 'top']].set_visible(False)

ax.legend(loc = 'upper left')
fig.tight_layout()
plt.savefig('EN4 Plots/TS Plots/IOBand_TS.pdf')


#%%
from scipy.signal import welch
fig, axs = plt.subplots(4, 3, figsize=(12, 15), sharex = True)
f_i = 0
fs = 12
for OB in TS_oceans.keys():
    #plotting data
    frequencies, psd_values = welch(TS_oceans[OB], fs, nperseg=100)

    # Plotting the estimated PSD
    axs[f_i//3, f_i%3].semilogy(frequencies, psd_values)
    # axs[f_i//3, f_i%3].plot(x_time, TS_oceans[OB], alpha = 0.6, label = 'Data')
    #plotting rolling mean
    # axs[f_i//3, f_i%3].plot(x_time, rolling.mean()[OB], color = 'black', label = f'{rolling_n} month rolling mean')
    axs[f_i//3, f_i%3].set_title(OB)
    axs[f_i//3, f_i%3].set_ylabel('Power Spectral Density')
    axs[f_i//3, f_i%3].set_xlabel('Frequency, $year^{-1}$')
    # axs[f_i//3, f_i%3].set_xscale('log')
    f_i += 1
# plt.savefig('EN4 Plots/Oceans_spectral.pdf')

    
