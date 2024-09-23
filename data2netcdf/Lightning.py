#Author : Sanjana Senthilkumar (sanjana.sk08@gmail.com)
# LIGHTINING


from glob import glob
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime

# Create Dictionary for GFED REGION LABELS

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


# GFED REGRID 720 X 1440 TO 720 X 360

#GFED region mask(720 X 1440) needs TO be regridded to match the resoultion of WGLC 30M dataset (720 X 360)



    
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import os

# Set the working directory where you want to save the new netCDF file
os.chdir('/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC')

# Load the GFED4.1s_2004.hdf5 file to extract the GFED region mask
file = '/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/GFED4.1s_2004.hdf5'
data = Dataset(file, mode='r')
mask = np.flip(data['ancill']['basis_regions'][:],axis = 0)

df = pd.DataFrame(mask)

# # Print the shape and values of the mask
print("Mask shape:", mask.shape)
print("Mask values:\n", mask)
# Define latitude and longitude arrays
lat = np.linspace(-90, 90, 720)
lon = np.linspace(-180, 180, 1440)

# Create an xarray DataArray from the numpy array with latitude and longitude as coordinates
mask_da = xr.DataArray(mask, coords=[('lat', lat), ('lon', lon)])

# Create an xarray Dataset with the DataArray and add a variable name ('mask' in this case)
ds = xr.Dataset({'mask': mask_da})

# Save the xarray Dataset to a NetCDF file
ds.to_netcdf('GFED5_area_0.25_flipped.nc')

data = xr.open_dataset('GFED5_area_0.25_flipped.nc') 

# Regrid the GFED maks to 360X720

# Load the mask data

mask = data['mask'][:]
# Define latitude and longitude arrays for the new grid (360X720)
lat_720x360 = np.linspace(-90, 90, 360)
lon_720x360 = np.linspace(-180, 180, 720)

# Create a new xarray dataset for the output file
output_ds = xr.Dataset()

# Add latitude and longitude dimensions for the new grid (90x144)
output_ds['latitude'] = lat_720x360
output_ds['longitude'] = lon_720x360

# Add a variable for each mask region
for region in range(0, 15):
    region_name = f'region_{region}'
    region_data = np.where(mask == region, 1, 0)  # Set to 1 where the mask equals the region, 0 elsewhere

    # Create a DataArray with the original lat/lon coordinates
    region_da = xr.DataArray(region_data, coords=[np.linspace(-90, 90, 720), np.linspace(-180, 180, 1440)], dims=['latitude', 'longitude'])
    region_data_floor = np.floor(region_da)

    # Regrid the region_data to the new lat/lon grid using reindex
    region_data_regridded = region_data_floor.interp(latitude=lat_720x360, longitude=lon_720x360, method='nearest')

    output_ds[region_name] = region_data_regridded

#Create a region 15 which is a sum of all gfed regions
   

output_ds['region_15'] = output_ds['region_1']+ output_ds['region_2']+ output_ds['region_3']+output_ds['region_4']+output_ds['region_5']+output_ds['region_6']+output_ds['region_7']+output_ds['region_8']+output_ds['region_9']+output_ds['region_10']+output_ds['region_11']+output_ds['region_12']+output_ds['region_13']+output_ds['region_14']
output_ds.to_netcdf('/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/GFed_region_mask_lightning.nc', format='NETCDF4')

#CALCULATE AREA OF GRIDCELL

import math

# Earth's radius in kilometers
EARTH_RADIUS = 6371.0

# Number of rows and columns in the grid
num_rows = 360
num_cols = 720

# Calculate the dimensions of each grid cell
grid_cell_width = 360.0 / num_cols 
grid_cell_height = 180.0 / num_rows

# Create a 2D array to store grid cell areas
grid_cell_areas = np.zeros((num_rows,num_cols))

# Function to calculate the area of a grid cell using the Eckert IV projection
def calculate_grid_cell_area(row, col):
    lat1 = -90.0 + row * grid_cell_height
    lat2 = -90.0 + (row + 1) * grid_cell_height
    lon1 = col * grid_cell_width
    lon2 = (col + 1) * grid_cell_width
    
    lat_mid = (lat1 + lat2) / 2
    width_correction = math.cos(math.radians(lat_mid))
    
    area = (EARTH_RADIUS**2) * math.radians(lon2 - lon1) * math.radians(lat2 - lat1) * width_correction
    
    return area

# Calculate and store the area of each grid cell
for row in range(num_rows):
    for col in range(num_cols):
        grid_cell_area = calculate_grid_cell_area(row, col)
        grid_cell_areas[row,col] = grid_cell_area

# Calculate and print the total Earth's surface area
total_earth_surface_area = 4 * math.pi * (EARTH_RADIUS**2)
print(f"Total Earth's surface area: {total_earth_surface_area:.2f} km^2")

# Calculate and print the total grid cell area
total_grid_cell_area = np.sum(grid_cell_areas)
print(f"Total grid cell area: {total_grid_cell_area:.2f} km^2")

##########################################################################################
##########################################################################################

# WLGC


lightning_30m = xr.open_dataset('/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/wglc_timeseries_30m.nc')



def lightning_TS():
    # This function takes the density variable from the input dataset. It applies the regridded gfed mask to the eglc dataset. We calcluate cell wise lightning density for each gfed region.
    
    print("Lightning WGLC Masking")
    density = lightning_30m['density']
    
    mask = xr.open_dataset('/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/GFed_region_mask_lightning.nc')
    regrid_mask = mask.to_array().values
    mask_val = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    time = lightning_30m['time'].values
    
    total_area = np.sum(grid_cell_areas)
    total_density_lightning = np.zeros((len(time),len(mask_val)))
    print("Applying gfed mask to wglc data.....")
    for t, val in enumerate(time): 
        
        
        for m in mask_val:
            #apply mask
            masked_data_array = np.ma.masked_array(density[t,:,:], mask=np.where(regrid_mask[m] == 0,True,False))
            region_total = (masked_data_array * grid_cell_areas)/total_area
            
            total_density_lightning[t,m] = np.nansum(region_total)
        
    total_density_da = xr.DataArray(total_density_lightning, dims=('time', 'mask'), coords={'time': time, 'mask': mask_val},attrs = {'units':'strokes/km²/day'})
    #total_density_da.attrs['units'] = 'strokes/km²/day'
    total_density_da.attrs['date_generated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_density_da.attrs['data_source'] = "wglc_timeseries_30m.nc"
    total_density_da.attrs['contact'] = "km2961@columbia.edu, sanjana.sk08@gmail.com"
    
    # Create a Dataset to store the DataArray
    total_density_ds = xr.Dataset({'Regional_Total': total_density_da})
    
    # Save the Dataset to a NetCDF file
    print("Storing output as netcdf...")
    total_density_ds.to_netcdf('/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/WLGC_GFED_region_data.nc')
                                        

lightning_TS()



####################################################################################
####################################################################################

# MODEL

# GFED REGRID 720 X 1440 TO 90 X 144
#The gfed mask has already been regridded to model resolution
regrid_mask = xr.open_dataset('/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/GFED4_mask_regions_90X144_nearest_fin.nc')
regrid_mask = regrid_mask.to_array().values


variables_list = [

 'f_ignCG',

 ]

import xarray as xr
import os

# Set the directory where your netCDF files are located
os.chdir('/discover/nobackup/kmezuman/E6TpyrEPD') #path to model op>

# List of months and years to consider
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
years = range(1997, 2020)  # Update this range with the years you want
#variables_to_extract = ['fireCount', 'BA_tree', 'BA_shrub', 'BA_grass', 'FLAMM', 'FLAMM_prec', 'f_ignCG', 'f_ignHUMAN']
variables_to_extract = ['axyp','f_ignCG','f_ignHUMAN']

#Setting a time stamp as the model output has files named with time but no variable in the data that has that information

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







mask_val = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
time_values = []
total_density_lightning = np.zeros((len(datasets_m),len(mask_val)))



for t,data in enumerate(datasets_m):
        
        time_values.append(data.coords['time'].values[0])

        for var_idx,i in enumerate(variables_list):
            axyp = data['axyp'] * 1000000
            total_area = np.sum(axyp) #m^2 to km^2 conversion
            total_model_arr = data[i] * 1000000
        
            for m in mask_val:
                #applying the gfed regridded amsk to model output
                masked_data_array = np.ma.masked_array(total_model_arr, mask=np.where(regrid_mask[m] == 0,True,False))
               
            
                #multiplying with gridcell area
                region_total = (masked_data_array * axyp)/total_area
                #wglc data is in km^2/day. Model output is in m^2/s. We convert the model output to match wglc
                region_total = region_total * 86400 #/s to /day conversion
                total_density_lightning[t,m] = np.nansum(region_total)
            


import xarray as xr
import numpy as np


data_vars = {}

# Loop through each variable
for var_idx, var in enumerate(variables_list):
    print(f"Processing variable {var}...")
    # Create an xarray DataArray for the current variable
    print("var_idx",var_idx)
    data_vars[var] = xr.DataArray(
        total_density_lightning[:,:],
        dims=['time', 'mask_val'],
        coords={
            'time': time_values,  # Assign time_values to the 'time' coordinate
            'mask_val': mask_val
        },
        attrs = {'units':'strokes/km²/day'}
    )

# Create the xarray dataset from the dictionary of DataArrays
total_model_sum_dataset = xr.Dataset(data_vars)


# Save the xarray dataset to a NetCDF file
total_model_sum_dataset.to_netcdf('/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/gfed_model_lightning.nc', format='NETCDF4')


# ### NUDGED


import xarray as xr
import os

# Set the directory where your netCDF files are located
os.chdir('/discover/nobackup/kmezuman/E6TpyrEPDnu')#path to input files

# List of months and years to consider
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
years = range(1997, 2020)  # Update this range with the years you want
#variables_to_extract = ['fireCount', 'BA_tree', 'BA_shrub', 'BA_grass', 'FLAMM', 'FLAMM_prec', 'f_ignCG', 'f_ignHUMAN']
variables_to_extract = ['axyp','f_ignCG','f_ignHUMAN']

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



mask_val = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
time_values = []
total_density_lightning = np.zeros((len(datasets),len(mask_val)))
#conversion_factor = 86400/1000000
#conversion_factor = 1/864000*1000000
#conversion_factor = 1
for t,data in enumerate(datasets):
        
    print(data.time)
    time_values.append(data.coords['time'].values[0])

    for var_idx,i in enumerate(variables_list):
        axyp = data['axyp'] * 1000000
        total_area = np.sum(axyp) #m^2 to km^2 conversion
        total_model_arr = data[i] * 1000000
        
        for m in mask_val:
      
            masked_data_array = np.ma.masked_array(total_model_arr, mask=np.where(regrid_mask[m] == 0,True,False))
               
            print("nonnan count")
           
            print(np.count_nonzero(~np.isnan(masked_data_array)))
            print("nan count")
            print(np.count_nonzero(np.isnan(masked_data_array)))
            
            region_total = (masked_data_array * axyp)/total_area
            
            region_total = region_total * 86400 #/s to /day conversion
            total_density_lightning[t,m] = np.nansum(region_total)
            

import xarray as xr
import numpy as np


data_vars = {}

# Loop through each variable
for var_idx, var in enumerate(variables_list):
    print(f"Processing variable {var}...")
    # Create an xarray DataArray for the current variable
    print("var_idx",var_idx)
    data_vars[var] = xr.DataArray(
        total_density_lightning[:,:],
        dims=['time', 'mask_val'],
        coords={
            'time': time_values,  # Assign time_values to the 'time' coordinate
            'mask_val': mask_val
        },
        attrs = {'units':'strokes/km²/day'}
    )

# Create the xarray dataset from the dictionary of DataArrays
total_model_sum_dataset = xr.Dataset(data_vars)


# Save the xarray dataset to a NetCDF file
total_model_sum_dataset.to_netcdf('/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/gfed_model_nudged_lightning.nc', format='NETCDF4')




# ### human VS IGcn

# #### Model (datasets_m)

variables_list = ['f_ignCG','f_ignHUMAN']

mask_val = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
time_values = []
total_density_lightning = np.zeros((len(datasets),len(variables_list),len(mask_val)))
#conversion_factor = 86400/1000000
#conversion_factor = 1/864000*1000000
conversion_factor = 1
for t,data in enumerate(datasets_m):
        
    print(data.time)
    time_values.append(data.coords['time'].values[0])

    for var_idx,i in enumerate(variables_list):
        axyp = data['axyp']
        total_area = np.sum(axyp) #m^2 to km^2 conversion
        total_model_arr = data[i] 
        
        for m in mask_val:
      
            masked_data_array = np.ma.masked_array(total_model_arr, mask=np.where(regrid_mask[m] == 0,True,False))
               
            print("nonnan count")
           
            print(np.count_nonzero(~np.isnan(masked_data_array)))
            print("nan count")
            print(np.count_nonzero(np.isnan(masked_data_array)))
            
            region_total = (masked_data_array * axyp)/total_area
        
            region_total = region_total * conversion_factor #/s to /day conversion
            total_density_lightning[t,var_idx,m] = np.nansum(region_total)
            


# #### NUDGED

mask_val = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
time_values = []

total_density_lightning_n = np.zeros((len(datasets),len(variables_list),len(mask_val)))
#conversion_factor = 86400/1000000
#conversion_factor = 1/864000*1000000
conversion_factor = 1
for t,data in enumerate(datasets):
        
    print(data.time)
    time_values.append(data.coords['time'].values[0])

    for var_idx,i in enumerate(variables_list):
        axyp = data['axyp']
        total_area = np.sum(axyp) #m^2 to km^2 conversion
        total_model_arr = data[i] 
        
        for m in mask_val:
      
            masked_data_array = np.ma.masked_array(total_model_arr, mask=np.where(regrid_mask[m] == 0,True,False))
               
            print("nonnan count")
           
            print(np.count_nonzero(~np.isnan(masked_data_array)))
            print("nan count")
            print(np.count_nonzero(np.isnan(masked_data_array)))
            
            region_total = (masked_data_array * axyp)/total_area

            region_total = region_total * conversion_factor #/s to /day conversion
            total_density_lightning_n[t,var_idx,m] = np.nansum(region_total)

import xarray as xr

# Create xarray DataArrays for model and nudged results
model_data_array = xr.DataArray(
    total_density_lightning,
    dims=["time", "variable", "mask"],
    coords={
        "time": time_values,
        "variable": variables_list,
        "mask": mask_val,
    },
)

nudged_data_array = xr.DataArray(
    total_density_lightning_n,
    dims=["time", "variable", "mask"],
    coords={
        "time": time_values,
        "variable": variables_list,
        "mask": mask_val,
    },attrs = {'units':'strokes/m²/s'}
)

# Create xarray Datasets for model and nudged results
model_dataset = xr.Dataset({"total_density_lightning": model_data_array})
nudged_dataset = xr.Dataset({"total_density_lightning_n": nudged_data_array})

# Combine both datasets into a single dataset
combined_dataset = xr.Dataset({"model":  model_data_array, "nudged": nudged_data_array})

# Save the combined dataset to a netCDF file
output_file =  "/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/lightning_model_result.nc"
combined_dataset.to_netcdf(output_file)

combined_dataset = xr.open_dataset('/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/lightning_model_result.nc')


# Load the combined netCDF file

#combined_dataset = xr.open_dataset(combined_file_path,engine='netcdf4')

# Extract data for model and nudged results
model_data = combined_dataset["model"]
nudged_data = combined_dataset["nudged"]

# Extract time and mask values from the original dataset
time_values = model_data.coords["time"].values
mask_values = model_data.coords["mask"].values

# Extract individual variables from model and nudged datasets
f_ignCG_model = model_data[:, 0, :].values
f_ignHUMAN_model = model_data[:, 1, :].values

f_ignCG_nudged = nudged_data[:, 0, :].values
f_ignHUMAN_nudged = nudged_data[:, 1, :].values

# Create xarray DataArrays for each variable
f_ignCG_model_da = xr.DataArray(
            f_ignCG_model,
                dims=("time", "mask"),
                    coords={"time": time_values, "mask": mask_values},
                    )

f_ignHUMAN_model_da = xr.DataArray(
            f_ignHUMAN_model,
                dims=("time", "mask"),
                    coords={"time": time_values, "mask": mask_values},
                    )

f_ignCG_nudged_da = xr.DataArray(
            f_ignCG_nudged,
                dims=("time", "mask"),
                    coords={"time": time_values, "mask": mask_values},
                    )

f_ignHUMAN_nudged_da = xr.DataArray(
            f_ignHUMAN_nudged,
                dims=("time", "mask"),
                    coords={"time": time_values, "mask": mask_values},
                    )

# Create xarray Datasets for model and nudged results
model_dataset = xr.Dataset({"f_ignCG": f_ignCG_model_da, "f_ignHUMAN": f_ignHUMAN_model_da})
nudged_dataset = xr.Dataset({"f_ignCG": f_ignCG_nudged_da, "f_ignHUMAN": f_ignHUMAN_nudged_da})

# Save the model dataset to a netCDF file
model_output_file = "/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/lightning_model_result_model.nc"
model_dataset.to_netcdf(model_output_file)

# Save the nudged dataset to a netCDF file
nudged_output_file = "/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/lightning_model_result_nudged.nc"
nudged_dataset.to_netcdf(nudged_output_file)

# Close the opened datasets
combined_dataset.close()
model_dataset.close()
nudged_dataset.close()









