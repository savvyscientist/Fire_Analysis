from glob import glob
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime

gfed_cover_labels = {
        0: "Ocean",
        1: "BONA",
        2: "TENA",
        3: "CEAM",
        4: "NHSA",
        5: "SHSA",
        6: "EURO",
        7: "MIDE",
        8: "NHAF",
        9: "SHAF",
        10: "BOAS",
        11: "CEAS",
        12: "SEAS",
        13: "EQAS",
        14: "AUST",   15: "Total"
    }


land_cover_labels = {
        0: "Water",
        1: "Boreal forest",
        2: "Tropical forest",
        3: "Temperate forest",
        4: "Temperate mosaic",
        5: "Tropical shrublands",
        6: "Temperate shrublands",
        7: "Temperate grasslands",
        8: "Woody savanna",
        9: "Open savanna",
        10: "Tropical grasslands",
        11: "Wetlands",
        12: "",
        13: "Urban",
        14: "",
        15: "Snow and Ice",
        16: "Barren",
        17: "Sparse boreal forest",
        18: "Tundra",
        19: ""
    }

regrid_mask = xr.open_dataset('/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/GFED4_mask_regions_90X144_nearest_fin.nc')
regrid_mask = regrid_mask.to_array().values

######################################################
#                  NUDGED                            #
######################################################

import xarray as xr
import os

# Set the directory where your netCDF files are located

os.chdir('/discover/nobackup/kmezuman/E6TpyrEPDnu') #path to model op>
# List of months and years to consider
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
years = range(1997, 2020)  # Update this range with the years you want
#variables_to_extract = ['fireCount', 'BA_tree', 'BA_shrub', 'BA_grass', 'FLAMM', 'FLAMM_prec', 'f_ignCG', 'f_ignHUMAN']
variables_to_extract = ['FLAMM_prec']
variables_list = ['FLAMM_prec']

# Open each file and load them into separate Datasets
datasets = []

for year in years:
    for month in months:
        file_pattern = f'{month}{year}.aijE6TpyrEPDnu.nc'
        file_paths = [f for f in os.listdir('.') if f.startswith(file_pattern)]
        
        for file_path in file_paths:
            dataset = xr.open_dataset(file_path)
            extracted_dataset = dataset.drop_vars([var for var in dataset.variables if var not in variables_to_extract])
            time_stamp = f'{month}{year}'  # Create a time stamp like 'JAN2013'
            extracted_dataset = extracted_dataset.expand_dims(time=[time_stamp])  # Add time as a new dimension
            datasets.append(extracted_dataset)

# Access and work with individual Datasets
for i, dataset in enumerate(datasets):
    print(f"Dataset {i+1}:")
    print(dataset)
##########################################
#              APPLY MASK                #
##########################################
mask_val = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
time_values = []
total_nudged_precip = np.zeros((len(datasets),len(mask_val)))
#conversion_factor = 86400/1000000
#conversion_factor = 1/864000*1000000
#conversion_factor = 1
for t,data in enumerate(datasets):
        
    print(data.time)
    time_values.append(data.coords['time'].values[0])

    for var_idx,i in enumerate(variables_list):
        
        total_model_arr = data[i]
        
        for m in mask_val:
      
            masked_data_array = np.ma.masked_array(total_model_arr, mask=np.where(regrid_mask[m] == 0,True,False))
               
            print("nonnan count")
           
            print(np.count_nonzero(~np.isnan(masked_data_array)))
            print("nan count")
            print(np.count_nonzero(np.isnan(masked_data_array)))
            
            region_total = masked_data_array
            total_nudged_precip[t,m] = np.nansum(region_total)
            
#################################################################
#                          MODEL                                #
#################################################################

import xarray as xr
import os

# Set the directory where your netCDF files are located
os.chdir('/discover/nobackup/kmezuman/E6TpyrEPD') #path to model op>

# List of months and years to consider
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
years = range(1997, 2020)  # Update this range with the years you want
#variables_to_extract = ['fireCount', 'BA_tree', 'BA_shrub', 'BA_grass', 'FLAMM', 'FLAMM_prec', 'f_ignCG', 'f_ignHUMAN']
variables_to_extract = ['FLAMM_prec']

# Open each file and load them into separate Datasets
datasets_m = []

for year in years:
    for month in months:
        file_pattern = f'{month}{year}.aijE6TpyrEPD.nc'
        file_paths = [f for f in os.listdir('.') if f.startswith(file_pattern)]
        
        for file_path in file_paths:
            dataset = xr.open_dataset(file_path)
            extracted_dataset = dataset.drop_vars([var for var in dataset.variables if var not in variables_to_extract])
            time_stamp = f'{month}{year}'  # Create a time stamp like 'JAN2013'
            extracted_dataset = extracted_dataset.expand_dims(time=[time_stamp])  # Add time as a new dimension
            datasets_m.append(extracted_dataset)

# Access and work with individual Datasets
for i, dataset in enumerate(datasets_m):
    print(f"Dataset {i+1}:")
    print(dataset)

mask_val = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
time_values = []
total_model_precip = np.zeros((len(datasets_m),len(mask_val)))
#conversion_factor = 86400/1000000
#conversion_factor = 1
#conversion_factor = 1/86400*1000000z



for t,data in enumerate(datasets_m):
        
    print(data.time)
    time_values.append(data.coords['time'].values[0])

    for var_idx,i in enumerate(variables_list):
        
        total_model_arr = data[i] 
        
        for m in mask_val:
      
            masked_data_array = np.ma.masked_array(total_model_arr, mask=np.where(regrid_mask[m] == 0,True,False))
               
            print("nonnan count")
           
            print(np.count_nonzero(~np.isnan(masked_data_array)))
            print("nan count")
            print(np.count_nonzero(np.isnan(masked_data_array)))
            
            region_total = masked_data_array
            total_model_precip[t,m] = np.nansum(region_total)
            
import xarray as xr

# Assuming you have 'total_model_precip', 'total_nudged_precip', 'time_values', and 'mask_val' defined

# Create xarray DataArrays for model and nudged results
import xarray as xr

# Assuming you have 'total_model_precip', 'total_nudged_precip', 'time_values', and 'mask_val' defined

# Create xarray DataArrays for model and nudged results
model_data_array = xr.DataArray(
    total_model_precip,
    dims=["time", "mask"],
    coords={
        "time": time_values,
        "mask": mask_val,
    },
    attrs={'units': 'mm/day'}
)

nudged_data_array = xr.DataArray(
    total_nudged_precip,
    dims=["time", "mask"],
    coords={
        "time": time_values,
        "mask": mask_val,
    },
    attrs={'units': 'mm/day'}
)

# Create xarray Datasets for model and nudged results
model_dataset = xr.Dataset({"FLAMM_prec": model_data_array})
nudged_dataset = xr.Dataset({"FLAMM_prec": nudged_data_array})

# Define output paths for separate netCDF files
model_output_file = "/discover/nobackup/projects/giss_ana/users/kmezuman/Precip/model_precipitation.nc"
nudged_output_file = "/discover/nobackup/projects/giss_ana/users/kmezuman/Precip/nudged_precipitation.nc"



# Save model and nudged datasets to separate netCDF files
model_dataset.to_netcdf(model_output_file)
nudged_dataset.to_netcdf(nudged_output_file)
