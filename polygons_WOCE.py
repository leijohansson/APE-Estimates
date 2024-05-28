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


#import files with polygons for oceans/seas
worldseas = gpd.read_file('World_Seas_IHO_v3\World_Seas_IHO_v3.shp')

#reading in data
datadir = 'WOCE_Data/Data/'
method = 'BAR'
month = '01'
filename = f'WAGHC_{method}_{month}_UHAM-ICDC_v1_0_1.nc'
data = xr.open_dataset(datadir+filename)

#getting lon and lat 
lat = data.latitude.to_numpy()
lon = data.longitude.to_numpy()

#converting lon and lat into the same range
oor_lon = np.where(lon > 180)[0] #out of range lon
lon[oor_lon] = lon[oor_lon] - 360

#saving 2D shape of the data
shape_2d = data.temperature.to_numpy()[0, 0, :, :].shape

#creating arrays with lon and lat at each grid points
LON, LAT = np.meshgrid(lon, lat)
#creating arrays with lon and lat indexes at each grid points
lon_I, lat_J = np.meshgrid(np.arange(len(lon)), np.arange(len(lat)))

#creating a dictionary with all the data
values = {'lon': LON.flatten(), 'lat': LAT.flatten(), 'lon_i': lon_I.flatten(),
          'lat_j': lat_J.flatten()}

#creating a dataframe
df = pd.DataFrame(values)

#making a dataframe of each grid point as a Point
df['coords'] = list(zip(df['lon'],df['lat']))
df['coords'] = df['coords'].apply(Point)
points = gpd.GeoDataFrame(df, geometry='coords', crs = 'epsg:4326')

#creating a list of all the ocean basin names
oceans = ['North Atlantic Ocean', 'South Atlantic Ocean',
          'North Pacific Ocean', 'South Pacific Ocean',
          'Indian Ocean', 'Southern Ocean', 'Arctic Ocean']

#acronym
# acros = oceans.copy()
# def acronym(string):
#     words = string.split()
#     j = ''
#     for w in words:
#         j += w[0]
#     return j

#creating a dataframe with all the oceans
oceans_df = worldseas[worldseas['NAME'].isin(oceans)]
#finding the ocean basin that each point is in
pointInPolys= gpd.tools.sjoin(points, oceans_df, predicate="within", how='left')

colors = ['Reds_r', 'Oranges_r', 'plasma_r', 'Greens_r', 'Blues_r', 'Purples_r', 'Greys_r']
c = 0
#creating ocean filters for each ocean
ocean_dict = {}
for name in oceans:
    LonLatBool = np.zeros(shape_2d)
    ocean_points = pointInPolys[pointInPolys.NAME == name]
    for pt in ocean_points.index:
        i = ocean_points.lon_i
        j = ocean_points.lat_j
        LonLatBool[j, i] = 1
        LonLatBool[LonLatBool == 0] = np.nan
    for j in range(len(lat)):
        if (LonLatBool[j, 179+1] == 1) and (LonLatBool[j, 179-1]==1):
            LonLatBool[j, 179] = 1
    #saving ocean filters in dictionary
    ocean_dict[name] = LonLatBool.copy()
    plt.imshow(LonLatBool, cmap = colors[c])
    c += 1

#saving dictionary as file
with open('RegionFilters/ocean_filters-WOCE.pkl', 'wb') as f:
    pickle.dump(ocean_dict, f)

#repeating for seas of interest
seas = ['Mediterranean Sea - Eastern Basin', 'Mediterranean Sea - Western Basin', 'Red Sea']
seas_df = worldseas[worldseas['NAME'].isin(seas)]
pointInPolys_sea= gpd.tools.sjoin(points, seas_df, predicate="within", how='left')

seas_dict = {}
for name in seas:
    LonLatBool = np.zeros(shape_2d)
    sea_points = pointInPolys[pointInPolys_sea.NAME == name]
    for pt in sea_points.index:
        i = sea_points.lon_i
        j = sea_points.lat_j
        LonLatBool[j, i] = 1
        LonLatBool[LonLatBool == 0] = np.nan
    seas_dict[name] = LonLatBool.copy()

with open('RegionFilters/sea_filters-WOCE.pkl', 'wb') as f:
    pickle.dump(seas_dict, f)