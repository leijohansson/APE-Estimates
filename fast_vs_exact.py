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
import seaborn as sns
palette = sns.color_palette("gnuplot", 10)[::5]
sns.set_theme(context='paper', style='white', palette=palette,
              rc={'xtick.bottom': True,'ytick.left': True,})
#%%
def find_plot_magnitude(data, label = None, marker = 'o', ax = None):
    minpower = np.floor(np.log10(data.min()))
    maxpower = np.floor(np.log10(data.max())) + 1

#trying to log bin (simple version, everything with the same power in a bin)
    ns = np.zeros(int(maxpower-minpower))
    powers = ns.copy()
    power = minpower
    i = 0
    while power < maxpower:
        ns[i] = len(data[np.where(data <= 10**(power+1))]) - np.sum(ns)
        powers[i] = 10 ** power
        power += 1
        i += 1
    
    if ax:
        ax.scatter(powers, ns, label = label, marker = marker)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Number of Points')
        ax.set_xlabel('Magnitude')
        return ax, ns, powers
    else:
        plt.scatter(powers, ns, label = label, marker = marker)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('Number of Points')
        plt.xlabel('Magnitude')
        return ns, powers
#%%

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

BGE_V, APE_V, APE_density = calc_APE(datadir, filename, V_ijk, p, z, routine = 'fast',
                        nonegs = False)
APE_density = APE_density.flatten()
APE_density = APE_density[~np.isnan(APE_density)]
latall = np.zeros((len(indexes), len(APE_density)))*np.nan
lonall = np.zeros((len(indexes), len(APE_density)))*np.nan
zall = np.zeros((len(indexes), len(APE_density)))*np.nan

datafast = np.zeros((len(indexes), len(APE_density)))*np.nan
dataexact = datafast.copy()

for i in range(len(indexes)):
    filename = allfiles[indexes[i]]
    BGE_f, APE_Vf, fast = calc_APE(datadir, filename, V_ijk, p, z, routine = 'fast',
                    nonegs = False)
    fast = fast.flatten()
    datafast[i] = fast[~np.isnan(fast)]
    BGE_e, APE_Ve, exact = calc_APE(datadir, filename, V_ijk, p,z, routine = 'exact',
                      nonegs = False)
    exact = exact.flatten()
    print(filename)
    print('% Difference in volume integrated APE:', np.nansum(APE_Vf-APE_Ve)/np.nansum(APE_Ve)*100)
    dataexact[i] = exact[~np.isnan(exact)]
    latall[i] = lat.flatten()[~np.isnan(exact)]
    lonall[i] = lon.flatten()[~np.isnan(exact)]
    zall[i] = z.flatten()[~np.isnan(exact)]

#%%
dataexact = dataexact.flatten()
datafast = datafast.flatten()

#%%
#only negative numbers
# neg_exact = dataexact[dataexact<0]
# neg_fast= datafast[datafast<0]

# plt.hist(np.abs(neg_exact), label = f'exact: n = {len(neg_exact)}')
# plt.hist(np.abs(neg_fast), zorder = 5, label = f'fast: n = {len(neg_fast)}')
# plt.title(f'Distribution of Magnitude of Negative APE Values in {len(indexes)} files')
# plt.xlabel('APE Magnitude, J/kg')
# plt.ylabel('Number of Negative APE Values')
# plt.yscale('log')
# plt.xscale('log')
# plt.legend()

#%%
diff = dataexact.flatten() - datafast.flatten()
difflat = latall.flatten()
difflon = lonall.flatten()
diffz = zall.flatten()
print(f'''exact is greater than fast for {np.sum(diff>0)/len(diff)} of 
datapoints, indicating that there may be a systematic bias''')
print(np.sum(diff>0))
import pandas as pd
e_gt_f = pd.DataFrame(diff[np.where(diff>0)], columns = ['diff'])
e_gt_f['lat'] = difflat[np.where(diff>0)]
e_gt_f['lon'] = difflon[np.where(diff>0)]
e_gt_f['z'] = diffz[np.where(diff>0)]
e_gt_f['exact'] = dataexact.flatten()[np.where(diff>0)]
e_gt_f['fast'] = datafast.flatten()[np.where(diff>0)]


print(f'mean difference excluding  outliers: {np.mean(e_gt_f[e_gt_f["diff"] < 0.01]["diff"])}')
print(f'mean difference for fast>exact: {np.mean(diff[np.where(diff <0)])}')

#%%
diff_neg = np.abs(diff[np.where(diff <0)])


find_plot_magnitude(diff_neg)
diff_pos = np.abs(diff[np.where(diff >0)])
find_plot_magnitude(diff_pos)
plt.title('Distribution of Magnitudes of (exact - fast)')
plt.legend(['Negative', 'Positive'])

plt.savefig('Pos_neg_diff_magnitude_comparison.png', bbox_inches = 'tight')



#%%
fig, axs = plt.subplots(2, height_ratios = [0.7, 0.3], sharex = True)
axs[0], ne, powere = find_plot_magnitude(dataexact[dataexact>0], label = 'exact', marker = 'o', ax = axs[0])
axs[0], nf, powerf = find_plot_magnitude(datafast[datafast>0], label = 'fast', marker = 'x', ax = axs[0])


axs[0].set_title(f'Distribution of Magnitude of Positive APE Values in {len(indexes)} files')
axs[0].legend()

if len(ne) >= len(nf):
    powers = powere
else:
    powers = powerf

ne_new = np.zeros(len(powers))
nf_new = np.zeros(len(powers))
for i in range(len(powers)):
    try:
        ne_new[i] = ne[np.where(powere == powers[i])]
    except:
        pass
    try:
        nf_new[i] = nf[np.where(powerf == powers[i])]
    except:
        pass
axs[0].set_xlabel(None)
axs[1].scatter(powers, nf_new - ne_new)
axs[1].set_xscale('log')
# axs[1].set_yscale('log')
axs[1].set_ylabel('$n_{fast} - n_{exact}$')
axs[1].set_xlabel('Magnitude')

plt.savefig('positive_magnitude_comparison.png', bbox_inches = 'tight')


    
    
    
#exact seems to have a minimum (both positive and negative) that is 10^-7 
#(not true, true min is ^-12, but anyway it seems like a lot of values can't reach this?)

