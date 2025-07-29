"""
Name: interpolationroutine
Interpolation function

Requirement:
    numpy, xarray, RandomForestRegressor, StandardScaler, matplotlib

Inputs:
    Global climate model data
    Local climate model data

Output:
    interpolated global variable on the local (finer) grid

"""
#%% ##### Import modules ######

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#%% Read global climate data:

def westernAustraliaGlobal(ds, var_global, T, depth, latmin, latmax, lonmin, lonmax):
     
    """
    Inputs:
        global climate model data: ds
        global variable of interest: var_global
        day of the month: T
        latmin, latmax, lonmin, lonmax: latitude, longitude of the region of interest
    
    Output:
        Global climate model data for a given region and a desired variable
    """
    
    Lat = ds.nav_lat.to_numpy()
    Lon = ds.nav_lon.to_numpy()

    Lat_1 = Lat[:,0]
    Lon_1 = Lon[0,:]

    # find the region of interest based on given lat, lon
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    LatGlobTemp1, Lat_loc1 = find_nearest(Lat_1, latmin) #-34.3265
    LatGlobTemp2, Lat_loc2 = find_nearest(Lat_1, latmax) #-22.5763

    LonGlobTemp1, Lon_loc1 = find_nearest(Lon_1, lonmin) #108.511
    LonGlobTemp2, Lon_loc2 = find_nearest(Lon_1, lonmax) #116.284

    Lat_glob = Lat_1[Lat_loc1:Lat_loc2]
    Lon_glob = Lon_1[Lon_loc1:Lon_loc2]

    LatGlobX, LonGlobY = np.meshgrid(Lat_glob, Lon_glob)

    ds_QoI = ds[var_global].isel(time_counter=T, deptht=depth, x=slice(Lon_loc1, Lon_loc2) , y=slice(Lat_loc1, Lat_loc2))
    # ds_QoI.plot(x="x", y="y", figsize=(15, 6), clim=(25, 35))
    
    return ds_QoI, Lat_glob, Lon_glob
    
#%% Padding to avoid those nasty zeros:

def padding(ds_QoI):
    
    """
    The problem is the data beyond coast line have zero values
    These zeros values are not good for interpolation
    Simple trick is to pad them
    
    Inputs:
        Global climate model data for a given region and a desired variable
    Output: 
        padded GCM data
    """
    
    ds_QoI_np = ds_QoI.to_numpy()
    
    # Find the maximum value and its indices
    maxval = np.max(ds_QoI_np)
    max_indices = np.unravel_index(np.argmax(ds_QoI_np), ds_QoI_np.shape)
    
    # Find the indices of the right, top, and top diagonal entries
    right_indices = [(max_indices[0], (max_indices[1] + i) % ds_QoI_np.shape[1]) for i in range(1, 3)]
    top_indices = [((max_indices[0] - i) % ds_QoI_np.shape[0], max_indices[1]) for i in range(1, 3)]
    top_diagonal_indices = [((max_indices[0] - i) % ds_QoI_np.shape[0], (max_indices[1] + i) % ds_QoI_np.shape[1]) for i in range(1, 3)]

    # Replace the right, top, and top diagonal entries with maxval
    for index in set(right_indices + top_indices + top_diagonal_indices):
        ds_QoI_np[index] = maxval


    for i in range(ds_QoI_np.shape[0]):
        # Find indices where the value is zero
        zero_indices = np.where(ds_QoI_np[i] == 0)[0]

        # Copy the previous value at those indices
        # ds_QoI_np[i, zero_indices] = ds_QoI_np[i, zero_indices - 8]
        for idx in zero_indices:
            ds_QoI_np[i, idx:idx + 1] = np.mean(ds_QoI_np[i, max(0, idx - 3):idx])
            # ds_QoI_np[i, idx:idx + 1] = np.max(ds_QoI_np)
            
    ds_QoI_np = np.nan_to_num(ds_QoI_np)
    return  ds_QoI_np

#%% Read local climate model data

def westernAustraliaLocal(ds_local, var_local, T, depth):
    
    """
    Inputs:
        Local climate model data: ds_local
        global variable of interest: var_local
        day of the month: T
    
    Output:
        local climate model data for a given region and a desired variable
    """
    
    Lat_np = ds_local.lat_rho.to_numpy()
    Lon_np = ds_local.lon_rho.to_numpy()

    Latlocal_1 = Lat_np[:,0]
    Lonlocal_1 = Lon_np[0,:]

    ds_sstloc = ds_local[var_local].isel(s_rho=24)
    ds_sstloc_np = ds_sstloc.to_numpy()
    ds_sstloc_mean_np = ds_sstloc_np[0]
    
    return Latlocal_1, Lonlocal_1, Lat_np, Lon_np, ds_sstloc_mean_np

#%% Interpolation!:

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.interpolate import griddata
import numpy as np
import time
from scipy.interpolate import RBFInterpolator

def interpolator(ds, ds_local, var_global, var_local, T, depth, latmin, latmax, lonmin, lonmax, method='random_forest'):
    ds_QoI, Lat_glob, Lon_glob = westernAustraliaGlobal(ds, var_global, T, depth, latmin, latmax, lonmin, lonmax)
    ds_QoI_np = padding(ds_QoI)
    
    idx = np.argwhere(np.all(ds_QoI_np[..., :] == 0, axis=0))
    ds_QoI_np = np.delete(ds_QoI_np, idx, axis=1)
    Lon_glob = np.delete(Lon_glob, idx)
    
    LatGlobX, LonGlobY = np.meshgrid(Lat_glob, Lon_glob)
    LatGlobX = LatGlobX.T
    LonGlobY = LonGlobY.T

    X = np.concatenate((LatGlobX.ravel().reshape(-1,1), LonGlobY.ravel().reshape(-1,1)), axis=1)
    y = ds_QoI_np.ravel()
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X)
    
    
    start_time = time.time()
    # Choose model based on the method parameter
    if method == 'random_forest':
        model = RandomForestRegressor(n_estimators=500)
        model.fit(X_train, y)
        interpolation_function = model.predict
    elif method == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=500)
        model.fit(X_train, y)
        interpolation_function = model.predict
    elif method == 'gaussian_process':
        model = GaussianProcessRegressor()
        model.fit(X_train, y)
        interpolation_function = model.predict
    elif method == 'rbf':
        interpolator = RBFInterpolator(X_train, y, kernel='linear')
        interpolation_function = interpolator
    else:
        raise ValueError("Invalid interpolation method")
    
    Latlocal_1, Lonlocal_1, Lat_np, Lon_np, ds_sstloc_mean_np = westernAustraliaLocal(ds_local, var_local, T, depth)
    X_test = np.concatenate((Lat_np.ravel().reshape(-1,1), Lon_np.ravel().reshape(-1,1)), axis=1)
    X_test_std = sc.transform(X_test)

    interpolated = interpolation_function(X_test_std).reshape(640, 480)
    noise = np.random.normal(0, 50, interpolated.shape) / 10000
    interpolated = interpolated + noise
    
    end_time = time.time()
    print(f"Time taken for {method} model fitting: {end_time - start_time:.2f} seconds")
    
    idx0 = np.argwhere(np.isnan(ds_sstloc_mean_np))
    interpolated[idx0[:,0], idx0[:,1]] = 0
    interpolated[interpolated == 0] = 'nan'
    
    interpolated = np.nan_to_num(interpolated)
    
    return interpolated


#%% Some postprocessing