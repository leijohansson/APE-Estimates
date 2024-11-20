# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:38:20 2024

@author: Linne

Creating ocean filters using polygons for WOCE data
"""


import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import pickle 
from FuncsAPE import datapath

#import files with polygons for oceans/seas
worldseas = gpd.read_file(datapath + 'goas\goas_v01.shp')

#reading in data
datadir = datapath+'WOCE_Data/Data/'
method = 'BAR'
month = '01'
filename = f'WAGHC_{method}_{month}_UHAM-ICDC_v1_0_1.nc'
data = xr.open_dataset(datadir+filename)

#getting lon and lat 
lat = data.latitude.to_numpy()
lon = data.longitude.to_numpy()

#converting lon and lat into the same range
oor_lon = np.where(lon > 180)[0]
lon[oor_lon] = lon[oor_lon] - 360

shape_2d = data.temperature.to_numpy()[0, 0, :, :].shape

#meshgrid of lon lat
LON, LAT = np.meshgrid(lon, lat)
lon_I, lat_J = np.meshgrid(np.arange(len(lon)), np.arange(len(lat)))

#creating dataframe of grid points
values = {'lon': LON.flatten(), 'lat': LAT.flatten(), 'lon_i': lon_I.flatten(),
          'lat_j': lat_J.flatten()}
df = pd.DataFrame(values)
df['coords'] = list(zip(df['lon'],df['lat']))
df['coords'] = df['coords'].apply(Point)
points = gpd.GeoDataFrame(df, geometry='coords', crs = 'epsg:4326')

SO_cutoff_lat = -45

oceans = ['North Atlantic Ocean', 'South Atlantic Ocean',
          'North Pacific Ocean', 'South Pacific Ocean',
          'Indian Ocean', 'Southern Ocean', 'Arctic Ocean',
          'Mediterranean Region', 'Baltic Sea',
          'South China and Easter Archipelagic Seas']


acros = oceans.copy()
#creating ocean dataframe
# oceans_df = worldseas[worldseas['NAME'].isin(oceans)]
oceans_df = worldseas[worldseas['name'].isin(oceans)]

#finding points in ocean polygons
pointInPolys= gpd.tools.sjoin(points, oceans_df, predicate="within", how='left')

#creating filters for different ocean basins
#colors for plotting
colors = ['Reds_r', 'Oranges_r', 'plasma_r', 'Greens_r', 'Blues_r', 'Purples_r', 'Greys_r', 'summer', 'spring', 'autumn'] 
c = 0 #color index for plotting
ocean_dict = {}
plotting = np.zeros(shape_2d)
plt.figure()
for name in oceans:
    LonLatBool = np.zeros(shape_2d)
    # ocean_points = pointInPolys[pointInPolys.NAME == name]
    ocean_points = pointInPolys[pointInPolys.name == name]
    for pt in ocean_points.index:
        i = ocean_points.lon_i
        j = ocean_points.lat_j
        LonLatBool[j, i] = 1
    if name == 'Southern Ocean':
        LonLatBool[np.where(LAT<SO_cutoff_lat)] = 1
    else:
        LonLatBool[np.where(LAT<SO_cutoff_lat)] = 0
    if name == 'Indian Ocean':
        print(colors[c])
        sum_bool = (LAT >= 13).astype(int) + (LON < 44).astype(int)
        LonLatBool[np.where(sum_bool >1)] = 0
   
    plotting += LonLatBool*(c+1)
    LonLatBool[LonLatBool == 0] = np.nan
    for j in range(len(lat)):
        if (LonLatBool[j, 179+1] == 1) and (LonLatBool[j, 179-1]==1):
            LonLatBool[j, 179] = 1
        if (LonLatBool[j, 0] == 1) and (LonLatBool[j, -2]==1):
            LonLatBool[j, -1] = 1
    ocean_dict[name] = LonLatBool.copy()
    
    # plotting to check that its correct
    plt.imshow(np.flip(LonLatBool,  axis = 0), cmap = colors[c])#,
                # extent = extent)
    
    c += 1 #updating color index for plotting
pacific = np.nan_to_num(ocean_dict['South Pacific Ocean']) +\
    np.nan_to_num(ocean_dict['North Pacific Ocean'])
eq_pacific = np.zeros(pacific.shape)
eq_pacific[np.where(np.abs(LAT) <= 20)] = 1

eq_pacific = pacific * eq_pacific
eq_pacific[eq_pacific == 0] = np.nan
plt.imshow(np.flip(eq_pacific,  axis = 0), cmap = 'cividis')#,
            # extent = extent)
ocean_dict['Equatorial Pacific'] = eq_pacific
#saving filters as a dictionary
with open(f'RegionFilters/ocean_filters-WOCE_{SO_cutoff_lat}.pkl', 'wb') as f:
    pickle.dump(ocean_dict, f)
# plt.contour(lon, lat, plotting)
# 