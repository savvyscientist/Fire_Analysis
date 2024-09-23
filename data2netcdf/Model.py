#Q the output netcdf does not have units, also, it does not extend to 2020
from glob import glob
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import matplotlib.dates as mdates



variables_list = [
 'BA_tree',
 'BA_shrub',
 'BA_grass',

 ]
regrid_mask = xr.open_dataset('/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/GFED4_mask_regions_90X144_nearest_fin.nc')
regrid_mask = regrid_mask.to_array().values



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


import xarray as xr
import os



############################################################
#                        MODEL BA                          #
############################################################

os.chdir('/discover/nobackup/kmezuman/E6TpyrEPD')

# List of months and years to consider
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
years = range(1997, 2020)  # Update this range with the years you want

variables_to_extract = ['BA_tree', 'BA_shrub', 'BA_grass']

# Open each file and load them into separate Datasets
datasets = []

for year in years:
    for month in months:
        file_pattern = f'{month}{year}.aijE6TpyrEPD.nc'
        file_paths = [f for f in os.listdir('.') if f.startswith(file_pattern)]
        
        for file_path in file_paths:
            dataset = xr.open_dataset(file_path)
            extracted_dataset = dataset.drop_vars([var for var in dataset.variables if var not in variables_to_extract])
            time_stamp = f'{month}{year}'  # Create a time stamp like 'JAN2013'
            extracted_dataset = extracted_dataset.expand_dims(time=[time_stamp])  # Add time as a new dimension
            datasets.append(extracted_dataset)



#MASKING

total_model_sum = np.empty((len(datasets),len(variables_list),16))
mask_val = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
time_values = []


for t,data in enumerate(datasets):
        

    time_values.append(data.coords['time'].values[0])
    model_array = data['BA_tree'] + data['BA_shrub']+data['BA_grass']

    for var_idx,i in enumerate(variables_list):
        total_model_arr = data[i]
        
        for m in mask_val:
          
            masked_data_array = np.ma.masked_array(total_model_arr, mask=np.where(regrid_mask[m] == 0,True,False))
           
      
            total_model_sum[t,var_idx,m] = np.nansum(masked_data_array)





import xarray as xr
import numpy as np


data_vars = {}

for var_idx, var in enumerate(variables_list):
    print(f"Processing variable {var}...")
    # Create an xarray DataArray for the current variable
    print("var_idx",var_idx)
    data_vars[var] = xr.DataArray(
        total_model_sum[:,var_idx,:],
        dims=['time', 'mask_val'],
        coords={
            'time': time_values,  # Assign time_values to the 'time' coordinate
            'mask_val': mask_val
        }
    )

# Create the xarray dataset from the dictionary of DataArrays
total_model_sum_dataset = xr.Dataset(data_vars)



total_model_sum_dataset.to_netcdf('/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/MODELtotal_not_nudged.nc', format='NETCDF4')


# Load your existing xarray Dataset
input_filename = '/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/MODELtotal_not_nudged.nc'
ds = xr.open_dataset(input_filename)

# Conversion factor from square meters to megahectares
conversion_factor_sqm_to_mha = 1e-10  # Conversion from square meters to megahectares

# Create a new Dataset to store the converted variables
ds_converted = ds.copy(deep=True)

# Loop through all variables in the Dataset
for var_name in ds.data_vars:
    converted_var_name = f"{var_name}_Mha"
    converted_var = ds[var_name] * conversion_factor_sqm_to_mha
    ds_converted[converted_var_name] = converted_var
    ds_converted[converted_var_name].attrs['units'] = 'Mha'
print("ds")
print(ds)
# Save the modified Dataset to a new NetCDF file
output_filename = '/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/converted_model_not_nudged.nc'
ds_converted['BA_Mha']= ds_converted['BA_shrub_Mha']+ds_converted['BA_tree_Mha'] + ds_converted['BA_grass_Mha']
ds_converted.to_netcdf(output_filename, format='netcdf4')








##########################################################
#                     NUDGED BA                          #
##########################################################



#Qdo you really need to keep on importing withing the same file? also why not at the top of the file?
import xarray as xr
import os

# Set the directory where your netCDF files are located
os.chdir('/discover/nobackup/kmezuman/E6TpyrEPDnu')

# List of months and years to consider
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
years = range(1997, 2020)  # Update this range with the years you want

variables_to_extract = ['BA_tree', 'BA_shrub', 'BA_grass']

# Open each file and load them into separate Datasets
datasets_n = []

for year in years: 
    for month in months:
        file_pattern = f'{month}{year}.aijE6TpyrEPDnu.nc'
        file_paths = [f for f in os.listdir('.') if f.startswith(file_pattern)]
        
        for file_path in file_paths:
            dataset = xr.open_dataset(file_path)
            extracted_dataset = dataset.drop_vars([var for var in dataset.variables if var not in variables_to_extract])
            time_stamp = f'{month}{year}'  # Create a time stamp like 'JAN2013'
            extracted_dataset = extracted_dataset.expand_dims(time=[time_stamp])  # Add time as a new dimension
            datasets_n.append(extracted_dataset)




total_model_sum = np.empty((276,len(variables_list),16))
mask_val = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
time_values = []

for t,data in enumerate(datasets_n):
        
    
    time_values.append(data.coords['time'].values[0])
   
    for var_idx,i in enumerate(variables_list):
        total_model_arr = data[i]
        
        for m in mask_val:
              
            masked_data_array = np.ma.masked_array(total_model_arr, mask=np.where(regrid_mask[m] == 0,True,False))
               
         
            total_model_sum[t,var_idx,m] = np.nansum(masked_data_array)

            





import xarray as xr
import numpy as np


# Create an empty dictionary to store DataArrays for each variable
data_vars = {}

# Loop through each variable
for var_idx, var in enumerate(variables_list):
    print(f"Processing variable {var}...")
    # Create an xarray DataArray for the current variable
    print("var_idx",var_idx)
    data_vars[var] = xr.DataArray(
        total_model_sum[:,var_idx,:],
        dims=['time', 'mask_val'],
        coords={
            'time': time_values,  # Assign time_values to the 'time' coordinate
            'mask_val': mask_val
        }
    )

# Create the xarray dataset from the dictionary of DataArrays
total_model_sum_dataset = xr.Dataset(data_vars)


# Save the xarray dataset to a NetCDF file
total_model_sum_dataset.to_netcdf('/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/MODELtotal_nudged.nc', format='NETCDF4')


import xarray as xr

# Load your existing xarray Dataset
input_filename = '/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/MODELtotal_nudged.nc'
ds = xr.open_dataset(input_filename)

# Conversion factor from square meters to megahectares
conversion_factor_sqm_to_mha = 1e-10  # Conversion from square meters to megahectares

# Create a new Dataset to store the converted variables
ds_converted = ds.copy(deep=True)

# Loop through all variables in the Dataset
for var_name in ds.data_vars:
    converted_var_name = f"{var_name}_Mha"
    converted_var = ds[var_name] * conversion_factor_sqm_to_mha
    ds_converted[converted_var_name] = converted_var
    ds_converted[converted_var_name].attrs['units'] = 'Mha'

# Save the modified Dataset to a new NetCDF file
output_filename = '/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/converted_model_nudged.nc'
ds_converted['BA_Mha'] = ds_converted['BA_shrub_Mha']+ds_converted['BA_grass_Mha']+ds_converted['BA_tree_Mha']
ds_converted.to_netcdf(output_filename, format='netcdf4')

