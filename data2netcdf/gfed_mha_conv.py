import xarray as xr
import os
import matplotlib.pyplot as plt
import warnings
import numpy as np
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
BA = xr.open_dataset('/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/gfed_burn_area.nc')
BA_sorted = BA.sortby('time')
ds = BA.sortby('time')
for var_name in ds.data_vars:
    var = ds[var_name]
           
               #var_sqha = var * conversion_factor_sqm_to_sqha
    var_Mha = var * 1e-4
    ds[var_name + '_Mha'] = var_Mha
    ds[var_name + '_Mha'].attrs['units'] = 'Mha'
    ds = ds.drop_vars(var_name)  # Drop the original variable

                               # Save the modified Dataset to a new NetCDF file
    output_filename = "/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/GFED_Mha.nc"
    ds.to_netcdf(output_filename, format='netcdf4')
