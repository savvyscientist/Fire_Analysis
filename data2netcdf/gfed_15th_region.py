# ### 15th region

# In[ ]:


import numpy as np
import xarray as xr
BA_sorted = xr.open_dataset('/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/gfed_burn_area.nc')

mask_lis = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]
time = BA_sorted['time'].values
total_regional_burn = np.zeros((len(time),20)) 
total_burn_list = np.zeros ((len(time)))
total_burn_crop_list = np.zeros((len(time)))
total_burn_peat_list = np.zeros((len(time)))
total_burn_defo_list = np.zeros((len(time)))
total=BA_sorted['Total']
crop = BA_sorted['Crop']
peat = BA_sorted['Peat']
defo = BA_sorted['Defo']
regions = BA_sorted['Regional']
land_cover_types = regions.ilct.values
print(land_cover_types)
unique_land_cover_types = list(set(land_cover_types))


for i in mask_lis:
    print(i)
    total_regional_burn+=regions[:,i,:]
    total_burn_list+=total[:,i]
    total_burn_crop_list+=crop[:,i]
    total_burn_defo_list+=peat[:,i]
    total_burn_peat_list+=defo[:,i]
    

# Create a Dataset
ds = xr.Dataset()

    # Add total_burn_area_lists as a data variable
ds['Regional'] = xr.DataArray(total_regional_burn, dims=('time', 'ilct'), coords={'time': time,  'ilct': unique_land_cover_types})
ds['Regional'].attrs['units'] = 'km^2'

ds['Total'] = xr.DataArray(total_burn_list, dims=('time'), coords={'time': time})
ds['Total'].attrs['units'] = 'km^2'

    # Add total_burn_crop_list as a data variable
ds['Crop'] = xr.DataArray(total_burn_crop_list, dims=('time'), coords={'time': time})
ds['Crop'].attrs['units'] = 'km^2'

    # Add total_burn_defo_list as a data variable
ds['Defo'] = xr.DataArray(total_burn_defo_list, dims=('time'), coords={'time': time})
ds['Defo'].attrs['units'] = 'km^2'

    # Add total_burn_peat_list as a data variable
ds['Peat'] = xr.DataArray(total_burn_peat_list, dims=('time'), coords={'time': time})
ds['Peat'].attrs['units'] = 'km^2'




 # Conversion from square hectares to megahectares

# Iterate over data variables

# multiply by 10^-4  km^2 to mha
for var_name in ds.data_vars:
    var = ds[var_name]
   
    #var_sqha = var * conversion_factor_sqm_to_sqha
    var_Mha = var * 1e-4
    ds[var_name + '_Mha'] = var_Mha
    ds[var_name + '_Mha'].attrs['units'] = 'Mha'
    ds = ds.drop_vars(var_name)  # Drop the original variable

# Save the modified Dataset to a new NetCDF file
output_filename = '/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/15th_region.nc'
ds.to_netcdf(output_filename, format='netcdf4')


    # Save the Dataset as a NetCDF file
#output_filename = 'C:/Users/ssenthil/Desktop/Datasets/netcdf_files_gen/15th_region.nc'
#ds.to_netcdf(output_filename, format='netcdf4')
