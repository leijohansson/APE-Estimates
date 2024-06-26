# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:41:21 2024

@author: Linne

Functions for calculation of APE
"""

import numpy as np 
import xarray as xr
import gsw
from gsw_gammat_analytic_CT_exact import *
from gsw_gammat_analytic_CT_fast import *
from gsw_gammat_analytic_CT import *

datapath = r'C:\\Users\\Linne\\Documents\\Github\\APE Data\\'

a = 4.56016575
b = -1.24898501
c = 0.00439778209
d = 1030.99373
e = 8.32218903 

# Define gravity 
grav = 9.81

# Define analytical expressions for reference density and pressure
# ----------------------------------------------------------------
rhor = lambda z: a/(b+1)*(z+e)**(b+1) + c*z + d   ## This analytical density profile fitted on the WOCE dataset.
pr = lambda z: grav * (a/((b+1)*(b+2))*((z+e)**(b+2)) + c/2.*z**2 + d*z 
                       - a/((b+1)*(b+2))*e**(b+2))/1e4

def calc_APE(datadir, filename, V_ijk, p, z, routine = 'exact', nonegs = True):
    '''
    

    Parameters
    ----------
    datadir : str
        Directory that the data is in.
    filename : str
        filename for data.
    V_ijk : 3D array of same shape as the data
        Volume covered by each grid point.
    p : 3D array of same shape as the data
        Pressure at each grid point.
    z : 3D array of same shape as the data
        Depth of each grid point.
    routine: string
        type of routine to use to calculate the reference values
        'fast' or 'exact'
        The default is 'exact'

    Returns
    -------
    BGE : 3D array of same shape as the data
        Background energy at each grid point.
    APE_dV : 3D array of same shape as the data
        Available potential energy at each grid point
    Pi2: 3D array of the same shape as the data
        APE density (J/kg) at each grid point

    '''
    # start = time.time()
    data = xr.open_dataset(f'{datadir}/{filename}')
    SP = data.salinity.to_numpy().squeeze()
    PT = data.temperature.to_numpy().squeeze()
    if 'WAGHC' not in filename:
        #convert potential temperature from K to C
        PT -= 273.15
    #calculating reference salinity from practical salinity
    SR = gsw.conversions.SR_from_SP(SP)
    #calculating conservative temperature from SR and 
    #potential temperature
    # CT = gsw.conversions.CT_from_pt(SR, PT)
    CT = gsw.conversions.CT_from_pt(SP, PT)
    
    #calculating zref, pref
    if routine == 'exact':
        gammat, zref, pref, sigref = gsw_gammat_analytic_CT_exact(SR, CT)
    elif routine == 'fast':
        gammat, zref, pref, sigref = gsw_gammat_analytic_CT_fast(SR, CT)

    #calculating enthalpies
    href = gsw.energy.enthalpy(SR, CT, pref)
    h = gsw.energy.enthalpy(SR, CT, p)
    
    rho = gsw.density.rho(SR, CT, p)
    
    #background energy
    BGE = (href-grav*zref)*rho*V_ijk
    BGE= np.nan_to_num(BGE)
    
    #total energy
    TE = (h-grav*z)*rho*V_ijk
    TE = np.nan_to_num(TE)
    
    # APE density (from Tailleux 2018) = APE_dV/(rho*V_ijk) 
    Pi2 = h - href - grav*(z-zref) 
    
    #APE
    APE_dV = TE-BGE
    APE_dV= np.nan_to_num(APE_dV)
    if nonegs:
        APE_dV[APE_dV < 0] = 1
    # print(time.time()-start)
    
    return BGE, APE_dV, Pi2

def calc_Aij(data):
    '''
    Calculate surface area covered by each grid point (X by Y) for a data 
    array where the grid spacing is the same in the x and y directions

    Parameters
    ----------
    data : xarray data array
        data to calculate area for 

    Returns
    -------
    A_ij : 2D array of (nlat x nlon)
        Area represented by each grid point

    '''
    #Radius of the earth
    R = 6.371e6
    #finding shape of data array
    try:
        shape = data.salinity.to_numpy().squeeze().shape
    except:
        shape = data.s_an.to_numpy().squeeze().shape
    
    #two different datasets have different names for latitude
    try:
        lat1D = data.latitude.to_numpy()
    except:
        lat1D = data.lat.to_numpy()

    
    #distance between grid points
    dgrid = lat1D[1] - lat1D[0]
    #calculating the area at each latitude
    A_j = dgrid* R**2 * np.pi/180*(np.sin((lat1D+dgrid/2)*np.pi/180) - 
                            np.sin((lat1D-dgrid/2)*np.pi/180))
    #creating a 2D array (lat and lon)
    A_ij = np.repeat(A_j.reshape(len(lat1D), 1), shape[2], axis = 1)
    return A_ij

def find_depthfracs(dz, shape, depth_co = 700):
    '''
    Creating an array that accounts for a maximum depth in calculations, 
    such that the depth_fracs array multiplied by a 3D array of dz returns 
    the maximum depth at each lonlat point when summed vertically.
    
    Parameters
    ----------
    dz : 1D array
        Array of depths in the data.
    shape : tuple
        shape of the data array for which the depth_frac array is being created.
    depth_co : float, optional
        Depth cut off. Maximum depth to calculate for. The default is 700.

    Returns
    -------
    depth_fracs : 3D array of the shape inputted
        1s in all depths above the maximum, some fraction between 0 and 1 in
        the depth that contains the maximum, and 0s for all depths below the 
        maximum.

    '''
    
    #finding the index for the maximum depth.
    if depth_co > np.sum(dz):
        depth_fracs = np.ones(shape)
    else:
        depth_sum = 0
        i = 0
        while depth_sum < depth_co:
            depth_sum += dz[i]
            i += 1
        i-=1
        
        #finding the fraction of the last depth taken 
        dz_left = depth_co - depth_sum + dz[i]
        frac_last = dz_left/dz[i]
        
        #creating the depth_frac array
        depth_fracs = np.zeros(shape)
        depth_fracs[:i, :, :] = 1
        depth_fracs[i] = frac_last
    
    return depth_fracs

def crop_oceanbasin(data, lon, lat):
    '''
    Crops data to just the ocean basin. Beginning with full data where 
    data not from that ocean is cropped out and set to nan or 0. Only written 
    for singular ocean basins. 

    Parameters
    ----------
    data : 2D array
        Filtered data, where valid values (non nan, non zero) only exist within
        the ocean basin.
    lon : list/1d array
        list of longitudes.
    lat : list/1d array
        list of latitudes.

    Returns
    -------
    cdata : 2D array
        cropped data.
    lon : 1d array
        cropped longitude array.
    lat : 1d array
        cropped latitude array.

    '''
    #sum each row to find top and bottom
    data = data.filled(np.nan)
    rows = np.nansum(data, axis = 1)
    rows_i = np.where(rows!=0)[0]
    mini, maxi = np.min(rows_i), np.max(rows_i)
    
    #cropping latitudes
    cdata = data[mini:maxi+1, :]
    lat = lat[mini:maxi+1]
    
    #sum each column to find left and right
    cols = np.nansum(cdata, axis = 0)
    cols_i = np.where(cols!=0)[0]
    
    
    #cases where ocean covers all longitudes (SO, AO)
    if len(cols_i) == cdata.shape[1]:
        pass
    else:
        #cases where ocean basin overlaps with where latitude starts and ends
        if cols[0]!=0 and cols[-1]!=0 and 0 in cols:
            zero_i = np.where(cols == 0)[0]
            
            #if not all indices with 0 are consescutive
            if True in ((zero_i[1:]-zero_i[:-1])>1):
                print('Not all 0 between sections')
            right = cdata[:, zero_i[0]:]
            left = cdata[:, :zero_i[0]]
            cdata = np.concatenate((right, left), axis = 1)
            
            #update new column sums and indices
            cols = np.nansum(cdata, axis = 0)
            cols_i = np.where(cols!=0)[0]
        
        minj, maxj = np.min(cols_i), np.max(cols_i)
        cdata = cdata[:, minj:maxj+1]
        lon = lon[minj:maxj+1]

    return cdata, lon, lat

def rearrange_OB(data, lon, lat):
    print(data.shape)
    data = data.filled(np.nan)
    #find all columns with data
    cols = np.nansum(data, axis = 0)
    #indexes of columns with data
    cols_i = np.where(cols!=0)[0]

    if cols[0]!=0 and cols[-1]!=0 and 0 in cols:
        zero_i = np.where(cols == 0)[0]
        
        #if not all indices with 0 are consescutive
        if True in ((zero_i[1:]-zero_i[:-1])>1):
            print('Not all 0 between sections')
        right = data[:, zero_i[0]:]
        left = data[:, :zero_i[0]]
        data = np.concatenate((right, left), axis = 1)
        lon = np.concatenate((lon[zero_i[0]:], lon[:zero_i[0]]))
        # lat = np.concatenate((lon[:, zero_i[0]:], lon[:, :zero_i[0]]), axis = 1)

        print(data.shape)
    return data, lon, lat



def calc_APE_WOA(datadir, V_ijk, p, z):
    '''

    Parameters
    ----------
    datadir : str
        Directory that the data is in.
    V_ijk : 3D array of same shape as the data
        Volume covered by each grid point.
    p : 3D array of same shape as the data
        Pressure at each grid point.
    z : 3D array of same shape as the data
        Depth of each grid point.

    Returns
    -------
    BGE : 3D array of same shape as the data
        Background energy at each grid point.
    APE_dV : 3D array of same shape as the data
        Available potential energy at each grid point

    '''
    datat = xr.open_dataset(f'{datadir}/temperature_annual_1deg.nc', decode_times = False)
    datas= xr.open_dataset(f'{datadir}/salinity_annual_1deg.nc', decode_times = False)

    SP = datas.s_an.to_numpy().squeeze()
    PT = datat.t_an.to_numpy().squeeze()
    
    #calculating reference salinity from practical salinity
    SR = gsw.conversions.SR_from_SP(SP)
    #calculating conservative temperature from SR and 
    #potential temperature
    CT = gsw.conversions.CT_from_pt(SR, PT)
    
    #calculating zref, pref
    gammat, zref, pref, sigref = gsw_gammat_analytic_CT_exact(SR, CT)
    #calculating enthalpies
    href = gsw.energy.enthalpy(SR, CT, pref)
    h = gsw.energy.enthalpy(SR, CT, p)
    
    rho = gsw.density.rho(SR, CT, p)
    
    #background energy
    BGE = (href-grav*zref)*rho*V_ijk
    BGE= np.nan_to_num(BGE)
    
    #total energy
    TE = (h-grav*z)*rho*V_ijk
    TE = np.nan_to_num(TE)
    
    # APE density (from Tailleux 2018) = APE_dV/(rho*V_ijk) 
    Pi2 = h - href - grav*(z-zref) 
    
    #APE
    APE_dV = TE-BGE
    APE_dV= np.nan_to_num(APE_dV)
    APE_dV[APE_dV < 0] = 1

    
    return BGE, APE_dV, Pi2

    