#Q what would it take to run it for GFED4 as well? no ilcts and data structure is slightly different but the resolution is the same
# import os
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import warnings
import numpy as np
#Q why do you need the line below:
warnings.filterwarnings("ignore")

from netCDF4 import Dataset
import pandas as pd
os.chdir('/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/BA/')
file = '/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/GFED4.1s_2004.hdf5'
data = Dataset(file, mode='r')
#file contains GFED REGIONS. It is a 720rowsx1440cols (0.25x0.25) 
#the basis_regions variable has values 0 (Ocean) to 14 (AUST) based on the region.
#the data is upside down and needs flipping
mask = np.flip(data['ancill']['basis_regions'][:],axis = 0)
print(data['ancill']['basis_regions'])
df = pd.DataFrame(mask)
# Print the shape and values of the mask
print(df.shape)
print("Mask shape:", mask.shape)
print("Mask values:\n", mask)
# Specify the pattern to match the file names (adjust as needed)
file_pattern = 'BA*.nc'

# Get the list of file paths that match the pattern
#Q explain the use of f in the line below
file_paths = [f for f in os.listdir('.') if f.startswith('BA') and f.endswith('.nc')]

# Open each file and load them into separate Datasets
datasets = [xr.open_dataset(file_path) for file_path in file_paths]
print("lenght")
print(len(datasets))
#Q there was a mask on top, what's going on here?
regrid_mask = xr.open_dataset('/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/GFED4_mask_regions_180X360_nearest_fin.nc')
#Qand here
regrid_mask_180X360 = regrid_mask.to_array().values
gfed_cover_labels = {
        0: "Ocean",
        1: "BONA (Boreal North America)",
        2: "TENA (Temperate North America)",
        3: "CEAM (Central America)",
        4: "NHSA (Northern Hemisphere South America)",
        5: "SHSA (Southern Hemisphere South America)",
        6: "EURO (Europe)",
        7: "MIDE (Middle East)",
        8: "NHAF (Northern Hemisphere Africa)",
        9: "SHAF (Southern Hemisphere Africa)",
        10: "BOAS (Boreal Asia)",
        11: "CEAS (Central Asia)",
        12: "SEAS (Southeast Asia)",
        13: "EQAS (Equatorial Asia)",
        14: "AUST (Australia and New Zealand)",
        15:"Total"
    }
#Q are these the ILCTs? where did you get these from
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



def Data_GFEDRegions(datasets):
    sz = len(datasets)
    print(sz)
    mask_lis = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        # Create a list of lists to store the total burned area for each mask value and ilct type
        #total_burn_area_lists = np.array( [[] for _ in unique_land_cover_types])
    #total_burn_area_lists = np.zeros((24,15,20)) 
#Q what are the below, please comment
    total_regional_burn = np.zeros((sz,15,20)) 
    total_burn_area_lists = np.zeros((sz,15,20)) #timestep,regions,ilct,total
    total_burn_list = np.zeros ((sz,15))
    total_burn_crop_list = np.zeros((sz,15))
    total_burn_peat_list = np.zeros((sz,15))
    total_burn_defo_list = np.zeros((sz,15))
    time =[]
    #print("list")

    #print(total_burn_area_lists.shape)
    #print(total_burn_area_lists[1].shape)
    #unique_land_cover_types=[]

    print(datasets)
    for t,data in enumerate(datasets):
        #print("data")
        #print(data)

        time.append(data.coords['time'].values[0].astype('datetime64[ns]'))
        print(time)
        #print("t val")
        #print(t)
        
        
    
    #burn_area_norm = target_data['Norm']
        total_var_burn = data['Total']
        #print("var")
        #print(total_var_burn)
    #crop_burn = target_data['Crop']   
    
        #mask_val = [11,12,13]
  
       # t = data['time']

        for i in mask_lis:
            print("mask val")
            print(i)


                # Iterate through each time file


                    # Create a list to store the total burned area for each ilct type


            if data.time.dt.year < 2001:
                    
                    #reshaped_mask_cr = np.expand_dims(regrid_mask_180X360 , axis=(0, 1))
                    #total_var_burns = total_var_burn[t,:,:][:]
                    total_var_burns = total_var_burn
                    masked_data_array_cr = np.ma.masked_array(total_var_burns, mask=np.where(regrid_mask_180X360[i] !=1,True,False))
                    total_burn_list[t,i] = np.nansum(masked_data_array_cr)
                    





            else:
                mask_cond = mask!=i

                total_burn_area_ilct = []
                #crop_burn = data['Crop'] 
                burn_area_norm = data['Norm']
                land_cover_types = burn_area_norm.iLCT.values
                print(land_cover_types)
                unique_land_cover_types = list(set(land_cover_types))

                


                 #total
                total_var_burns = total_var_burn
                masked_data_array_t = np.ma.masked_array(total_var_burns, mask=np.where(mask_cond,True,False))
                total_burn_list[t,i] = np.nansum(masked_data_array_t)



                #crop
                crop_area_burns = data['Crop']
                total_crop_burns = crop_area_burns
                masked_data_array_c = np.ma.masked_array(total_crop_burns, mask=np.where(mask_cond,True,False))
                total_burn_crop_list[t,i] = np.nansum(masked_data_array_c)

                #peat
                peat_area_burns = data['Peat']
                total_peat_burns = peat_area_burns
                masked_data_array_p = np.ma.masked_array(total_peat_burns, mask=np.where(mask_cond,True,False))
                total_burn_peat_list[t,i] = np.nansum(masked_data_array_p)

                #defo
                defo_area_burns = data['Defo']
                total_defo_burns = defo_area_burns
                masked_data_array_d = np.ma.masked_array(total_defo_burns, mask=np.where(mask_cond,True,False))
                total_burn_defo_list[t,i] = np.nansum(masked_data_array_d)

                # Iterate through each land cover type
                for ilct_idx, ilct in enumerate(unique_land_cover_types):
                   # print("ilct")
                   # print(ilct)

                        # Capture the burn area for ilct in the particular mask region
                    burn_area_vals = burn_area_norm[0, ilct, :, :][:]
                        #reshaped_mask = np.expand_dims(mask, axis=(0, 1))
                    masked_data_array = np.ma.masked_array(burn_area_vals, mask=np.where(mask_cond,True,False))
                        #masked_data_array = np.ma.masked_array(burn_area_vals, mask=reshaped_mask != i)
                    total_burn_area = np.sum(masked_data_array)
                    print(total_burn_area)
                    total_regional_burn[t,i,ilct_idx] = total_burn_area

            #total_var_burn = target_data['Total']
    
# Plotting the data
#Q do you need to import xarray twice?
    import xarray as xr

# Create a Dataset
    ds = xr.Dataset()

    # Add total_burn_area_lists as a data variable
#Q what is regional? also in the in the netcdf output there's Regional, Total, Crop, Defo, Peat
    ds['Regional'] = xr.DataArray(total_regional_burn, dims=('time', 'mask', 'ilct'), coords={'time': time, 'mask': mask_lis, 'ilct': unique_land_cover_types})
    ds['Regional'].attrs['units'] = 'km^2'

    ds['Total'] = xr.DataArray(total_burn_list, dims=('time', 'mask'), coords={'time': time, 'mask': mask_lis})
    ds['Total'].attrs['units'] = 'km^2'

    # Add total_burn_crop_list as a data variable
    ds['Crop'] = xr.DataArray(total_burn_crop_list, dims=('time', 'mask'), coords={'time': time, 'mask': mask_lis})
    ds['Crop'].attrs['units'] = 'km^2'

    # Add total_burn_defo_list as a data variable
    ds['Defo'] = xr.DataArray(total_burn_defo_list, dims=('time', 'mask'), coords={'time': time, 'mask': mask_lis})
    ds['Defo'].attrs['units'] = 'km^2'

    # Add total_burn_peat_list as a data variable
    ds['Peat'] = xr.DataArray(total_burn_peat_list, dims=('time', 'mask'), coords={'time': time, 'mask': mask_lis})
    ds['Peat'].attrs['units'] = 'km^2'

    # Save the Dataset as a NetCDF file
    output_filename = '/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/gfed_burn_area.nc'
    ds_sorted = ds.sortby('time')
    ds_sorted.to_netcdf(output_filename, format='netcdf4')
Data_GFEDRegions(datasets)

