#%% Statistical downscaling model that predicts SST of the regional model (ROMS) by relating global and local variables

"""
Name: runscript

Requirement:
    numpy, xarray, interpolationroutine, os, time, readfiles
    nnmodel, postprocessing

Inputs:
    year - The year for which to run the interpolation

Output:
    A pickle file containing the interpolated data
"""

#%% Import necessary libraries
import numpy as np
import xarray as xr
from interpolation.access_interpolator import interpolator
from interpolation.mld1_interpolator import interpolator_mld1
from interpolation.era5_interpolator import interpolator_era5
import calendar
import pickle
import os


def interpolationroutine(year):
    # Define region of interest and model parameters for Western Australia
    depth = 0
    latmin = -34.3265 
    latmax = -22.5763
    lonmin = 108.511
    lonmax = 116.284

    monthstart = 1
    monthend = 13

    daysinm = []

    for k in range(monthstart, monthend):
        dmv = calendar.monthrange(year, k)[1]
        daysinm.append(dmv)
        
    days = sum(daysinm)

    st_days = 0

    var_local = 'temp'
    var_global_sst = 'sst'
    var_global_salt = 'salt'
    var_global_slhf = 'slhf'
    var_global_snsr = 'ssr'
    var_global_sntr = 'str'
    var_global_sshf = 'sshf'
    var_global_mld1 = 'mld1'

    # Interpolation lists
    interpolatedlist_SST = []
    interpolatedlist_Salt = []
    interpolatedlist_slhf = []
    interpolatedlist_snsr = []
    interpolatedlist_sntr = []
    interpolatedlist_sshf = []
    interpolatedlist_mld1 = []

    # SST interpolation cwa_20210101_12__avg
    ds = xr.open_dataset(f'data/access/daily/sst/do_sst_{year}.nc')
    ds_local = xr.open_dataset('data/roms/2021/cwa_20210101_12__avg.nc')

    for T in range(st_days, days):
        interpolationresults_SST = interpolator(ds, ds_local, var_global_sst, var_local, 
                                                T, depth, latmin, latmax, lonmin, lonmax)
        interpolatedlist_SST.append(interpolationresults_SST.ravel())
    
    
    # Salt interpolation
    ds = xr.open_dataset(f'data/access/daily/salt/do_salt_{year}.nc')
    ds_local = xr.open_dataset('data/roms/2021/cwa_20210101_12__avg.nc')

    for T in range(st_days, days):
        interpolationresults_salt = interpolator(ds, ds_local, var_global_salt, var_local, 
                                                T, depth, latmin, latmax, lonmin, lonmax)
        interpolatedlist_Salt.append(interpolationresults_salt.ravel())

    # slhf interpolation
    ds = xr.open_dataset(f'data/era5/daily/slhf/era5_slhf_daily_{year}.nc')
    ds_local = xr.open_dataset('data/roms/2021/cwa_20210101_12__avg.nc')

    for T in range(st_days, days):
        interpolationresults_slhf = interpolator_era5(ds, ds_local, var_global_slhf, var_local, 
                                                      T, depth)
        interpolatedlist_slhf.append(interpolationresults_slhf.ravel())

    # snsr interpolation
    ds = xr.open_dataset(f'data/era5/daily/snsr/era5_ssr_daily_{year}.nc')
    ds_local = xr.open_dataset('data/roms/2021/cwa_20210101_12__avg.nc')

    for T in range(st_days, days):
        interpolationresults_snsr = interpolator_era5(ds, ds_local, var_global_snsr, var_local, 
                                                      T, depth)
        interpolatedlist_snsr.append(interpolationresults_snsr.ravel())

    # sntr interpolation
    ds = xr.open_dataset(f'data/era5/daily/sntr/era5_str_daily_{year}.nc')
    ds_local = xr.open_dataset('data/roms/2021/cwa_20210101_12__avg.nc')

    for T in range(st_days, days):
        interpolationresults_sntr = interpolator_era5(ds, ds_local, var_global_sntr, var_local, 
                                                      T, depth)
        interpolatedlist_sntr.append(interpolationresults_sntr.ravel())

    # sshf interpolation
    ds = xr.open_dataset(f'data/era5/daily/sshf/era5_sshf_daily_{year}.nc')
    ds_local = xr.open_dataset('data/roms/2021/cwa_20210101_12__avg.nc')

    for T in range(st_days, days):
        interpolationresults_sshf = interpolator_era5(ds, ds_local, var_global_sshf, var_local, 
                                                      T, depth)
        interpolatedlist_sshf.append(interpolationresults_sshf.ravel())

    # Mld1
    ds = xr.open_dataset(f'data/access/daily/mld1/do_mld1_{year}.nc')
    ds_local = xr.open_dataset('data/roms/2021/cwa_20210101_12__avg.nc') 
    
    for T in range(st_days, days):
        interpolationresults_mld1 = interpolator_mld1(ds, ds_local, var_global_mld1, var_local,
                                        T, depth, latmin, latmax, lonmin, lonmax)

        interpolatedlist_mld1.append(interpolationresults_mld1.ravel())


    # Save all interpolated variables to a pickle file for later use
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed", f"Data{year}_gcm.p"))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "wb") as f:
        pickle.dump([
            interpolatedlist_SST, interpolatedlist_Salt, 
            interpolatedlist_slhf, interpolatedlist_snsr, 
            interpolatedlist_sntr, interpolatedlist_sshf, interpolatedlist_mld1
        ], f)
    
    print(f"Data for year {year} has been processed and saved to {output_file}")


# Example usage
interpolationroutine(2021)
# run_downscaling(2018)
