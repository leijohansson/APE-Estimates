# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:34:18 2024

@author: Linne

Plot time series of volume integrated BGE for different ocean basins
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
with open('RegionFilters/ocean_filters-EN4.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)
dictkey = list(ocean_filters.keys())[0]
ocean_filters['World'] = np.ones(ocean_filters[dictkey].shape)

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
TS_APE, TS_BGE= {}, {}
for OB in ocean_filters.keys():
    TS_APE[OB] = np.zeros((n_months))
    TS_BGE[OB] = np.zeros((n_months))



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
        APE = np.sum(APE_all*depth_fracs, axis = 0)
        
        for OB in ocean_filters:
            APE_ocean = np.nansum(APE * ocean_filters[OB])
            TS_APE[OB][time_id] = APE_ocean
        
        file = f'\BGEarrays\BGE_{year}-{month}.npy'
        BGE_all = np.load(datadir + file)
        #calculating APE up to depth 700
        BGE = np.sum(BGE_all*depth_fracs, axis = 0)
        
        for OB in ocean_filters:
            BGE_ocean = np.nansum(BGE * ocean_filters[OB])
            TS_BGE[OB][time_id] = BGE_ocean
#%%
#number of months to take mean over
rolling_n = 12
#creating dataframe
df_APE, df_BGE = pd.DataFrame(), pd.DataFrame()
for OB in ocean_filters.keys():
    df_APE[OB] = TS_APE[OB]
    df_BGE[OB] = TS_BGE[OB]

rolling_APE = df_APE.rolling(rolling_n, center = True)
rolling_BGE= df_BGE.rolling(rolling_n, center = True)

#%%
#Plots
fig_dict = {}
for OB in TS_APE.keys():
    fig, axs = plt.subplots(2,  1, sharex = True)
    fig.suptitle(OB)
    
    axs[0].set_title('Available Potential Energy')
    axs[0].set_ylabel('Volume Integrated APE, $J$')
    axs[0].plot(x_time, TS_APE[OB], alpha = 0.6, label = 'Data')
    axs[0].plot(x_time, rolling_APE.mean()[OB], color = 'black', label = f'{rolling_n} month rolling mean')
    

    axs[1].set_title('Background Potential Energy')
    axs[1].set_ylabel('Volume Integrated BGE, $J$')
    axs[1].plot(x_time, TS_BGE[OB], alpha = 0.6, label = 'Data')
    axs[1].plot(x_time, rolling_BGE.mean()[OB], color = 'black', label = f'{rolling_n} month rolling mean')
    axs[1].set_xlabel('Time')
    axs[1].legend()
    if OB == 'South Pacific Ocean':
        lin_regress_subsetyears(OB, 1990, 2015, axs[0])
    elif OB == 'Indian Ocean':
        lin_regress_subsetyears(OB, 2003, np.inf, axs[0])
        lin_regress_subsetyears(OB, 1960, 2000, axs[0])
    else:
        lin_regress_subsetyears(OB, ax = axs[0])

    plt.savefig(f'EN4 Plots/TS Plots/BGE_APE_TS_{OB}.png', bbox_inches = 'tight')

#%%
def plot_one(OB, ax):
    ax.set_title(OB)
    ax.plot(x_time, TS_APE[OB], alpha = 0.6, label = 'Data')
    #plotting rolling mean
    ax.plot(x_time, rolling_APE.mean()[OB], color = 'black', label = f'{rolling_n} month rolling mean')
    ax.set_xlabel('Time')
    ax.set_ylabel('Volume Integrated APE, $J$')

def lin_regress_subsetyears(OB, startyear = 0, endyear = np.inf, ax = None, alternative = 'two-sided'):
    bool1 = (x_time.year<=endyear).astype(int)
    bool2 = (x_time.year>=startyear).astype(int)
    time_sel = np.where((bool1+bool2) == 2)
    
    slope, intercept, r, p, std_err = stats.linregress((np.arange(len(x_time))/12)[time_sel], TS_APE[OB][time_sel], alternative = alternative
                                                       )
    print(f'OB {startyear}-{endyear}, Trend : {slope/1e17} +/- {std_err/1e17} x 10^17 J/year,')
    print(f'R2 value: {r}, p_value: {p}')
    if ax: 
        ax.plot(x_time[time_sel], slope*(np.arange(len(x_time))/12)[time_sel] + intercept, lw = '2', color = 'red', label = 'Trend Line')

def lin_regress_BGEAPE(OB, startyear = 0, endyear = np.inf, alternative = 'two-sided'):
    bool1 = (x_time.year<=endyear).astype(int)
    bool2 = (x_time.year>=startyear).astype(int)
    time_sel = np.where((bool1+bool2) == 2)
    
    slope, intercept, r, p, std_err = stats.linregress(TS_APE[OB][time_sel], TS_BGE[OB][time_sel], alternative = alternative)
    print(f'OB {startyear}-{endyear}, Trend : {slope} +/- {std_err},')
    print(f'R2 value: {r}, p_value: {p}')
    return slope, r
#%%
OB = 'South Pacific Ocean'
fig, axs = fig_dict[OB]
lin_regress_subsetyears(OB, 1990, 2015, axs[0])


OB = 'North Pacific Ocean'
fig, axs = fig_dict[OB]
lin_regress_subsetyears(OB, ax = axs[0])


OB = 'North Atlantic Ocean'
fig, axs = fig_dict[OB]
lin_regress_subsetyears(OB, ax = axs[0])

OB = 'South Atlantic Ocean'
fig, axs = fig_dict[OB]
lin_regress_subsetyears(OB', ax = axs[0])



#%%
fig, axs = plt.subplots(2,  1, sharex = True)
plot_one('North Atlantic Ocean', axs[0])
plot_one('South Atlantic Ocean', axs[1])
fig.tight_layout()



axs[0].legend(loc = 'upper right')
plt.savefig('EN4 Plots/BGE_Atlantic_TS.pdf')

#%%
fig, axs = plt.subplots(2,  1, sharex = True)
fig.tight_layout()

# lin_regress_subsetyears('Arctic Ocean', ax = axs[0])
# lin_regress_subsetyears('Southern Ocean', ax = axs[1])
axs[0].legend(loc = 'lower left')

plt.savefig('EN4 Plots/BGE_Poles_TS.pdf')


# print('Peak at:', x_time[np.where(TS_oceans['Arctic Ocean'] == TS_oceans['Arctic Ocean'].max())][0])
x_warm = pd.date_range('2016-01-30', periods=1, freq='m')

print('Arctic warming ', x_warm)
axs[0].vlines(x_warm, 2.7e20, 3e20)

# print('SO')
# print('Peak at:', x_time[np.where(rolling.mean()['Southern Ocean'] == rolling.mean()['Southern Ocean'].min())][0])

#coincides with arctic warming. 
#%%
fig, ax = plt.subplots(1,  1, sharex = True, figsize = (6.4, 3))
plot_one('Indian Ocean', ax)
ax.legend(loc = 'upper left')

fig.tight_layout()
plt.savefig('EN4 Plots/BGE_Indian_TS.pdf')
    
#%%
fig, ax = plt.subplots(1,  1, sharex = True, figsize = (6.4, 3))
plot_one('World', ax)
fig.tight_layout()
# lin_regress_subsetyears('World', ax = ax)
ax.legend(loc = 'lower left')
plt.savefig('EN4 Plots/World_TS.pdf')

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

    
