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
from scipy.interpolate import RBFInterpolator


#%% Read global climate data:

def westernAustraliaGlobal(ds, var_global, T):
     
    """
    Inputs:
        global climate model data: ds
        global variable of interest: var_global
        day of the month: T
        latmin, latmax, lonmin, lonmax: latitude, longitude of the region of interest
    
    Output:
        Global climate model data for a given region and a desired variable
    """
    
    Lat = ds.latitude.to_numpy()
    Lon = ds.longitude.to_numpy()

    Lat_1 = Lat
    Lon_1 = Lon

    LatGlobX, LonGlobY = np.meshgrid(Lat_1, Lon_1)

    ds_QoI = ds[var_global].isel(valid_time=T)
    
    return ds_QoI, Lat_1, Lon_1
    
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

def interpolator_era5(ds, ds_local, var_global, var_local, T, depth):
    
    """
    Inputs:
        westernAustraliaGlobal
        padding
        westernAustraliaLocal
        global and local variables of interest: var_local, var_global
        day of the month: T
    
    Output:
        Global climate model data is interpolated
    """
    
    ds_QoI, Lat_glob, Lon_glob = westernAustraliaGlobal(ds, var_global, T)
    ds_QoI_np = ds_QoI.to_numpy()
    
    
    LatGlobX, LonGlobY = np.meshgrid(Lat_glob, Lon_glob)
    LatGlobX = LatGlobX.T
    LonGlobY = LonGlobY.T

    X = np.concatenate((LatGlobX.ravel().reshape(-1,1), LonGlobY.ravel().reshape(-1,1)), axis =1)
    y = ds_QoI_np.ravel()
    
    sc = StandardScaler()

    X_train = sc.fit_transform(X)

    #model = RandomForestRegressor(n_estimators=500)
    
    model = RBFInterpolator(X_train, y, kernel='linear')

    #model.fit(X_train, y)
    
    Latlocal_1, Lonlocal_1, Lat_np, Lon_np, ds_sstloc_mean_np = westernAustraliaLocal(ds_local, var_local, T, depth)
    
    X_test = np.concatenate((Lat_np.ravel().reshape(-1,1), Lon_np.ravel().reshape(-1,1)), axis =1)

    X_test_std = sc.transform(X_test)

    interpolated = model(X_test_std)
    interpolated = interpolated.reshape(640,480)
    noise = np.random.normal(0, 50, interpolated.shape)
    interpolated = interpolated + noise
    
    idx0 = np.argwhere(np.isnan(ds_sstloc_mean_np))
    idx0 = np.asarray(idx0)
    
    interpolated[idx0[:,0],idx0[:,1]] = 0
    interpolated[interpolated == 0] = 'nan'
    
    interpolated = np.nan_to_num(interpolated)
    
    return interpolated
