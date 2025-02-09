#!/usr/bin/env python
# coding: utf-8

# ### Read the data

# In[ ]:


# read in data
import xarray as xr
import os
import matplotlib.pyplot as plt
import warnings
import numpy as np
from netCDF4 import Dataset
import pandas as pd

warnings.filterwarnings("ignore")


# Specify the path to store the PNG files
from glob import glob


fnms = []
for year in range(2001, 2021):
    pattern = f"BA{year}??.nc"
    file_paths = glob(
        os.path.join(
            "/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/BA", pattern
        )
    )
    fnms.extend(file_paths)

# Open the files using xr.open_mfdataset()
target_data = xr.open_mfdataset(fnms)


# In[ ]:


time = target_data["time"]
total = target_data["Total"]
crop = target_data["Crop"]
peat = target_data["Peat"]
defo = target_data["Defo"]


# ### Initialising lat and lon

# In[ ]:


import numpy as np
import xarray as xr


def Data_grid_stat(target_data):
    latitude_coords = np.linspace(-89.88, 89.88, num=720)

    # Create coordinates for the second dimension (1400 points) from -179.9 to 179.9
    longitude_coords = np.linspace(-179.9, 179.9, num=1440)

    # Use numpy.meshgrid to create 2D coordinate grids for latitude and longitude
    longitude_grid, latitude_grid = np.meshgrid(longitude_coords, latitude_coords)

    # Define the months of interest
    months_of_interest = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    months_name = [["DJF"], ["MAM"], ["JJA"], ["SON"]]

    # Create an empty xarray dataset to store the monthly means
    monthly_means_data = xr.Dataset()

    # Loop through each variable
    for variable in ["Total", "Crop", "Defo", "Peat"]:
        # Loop through each month set of interest
        for i, months in enumerate(months_of_interest):
            # Select the burned area values for the current month set and variable
            burned_area = target_data[variable].sel(
                time=target_data["time.month"].isin(months)
            )
            print(burned_area)
            # Calculate the mean burned area for each grid cell
            mean_burned_area = burned_area.mean(dim="time")
            # <CROP/PEAT/DEFO/NAT/TOTAL>_meanBA_DJF,
            # Add the mean burned area as a new variable to the monthly_means_data dataset
            variable_name = f"{variable}_meanBA_{months_name[i][0]}"
            monthly_means_data[variable_name] = mean_burned_area
        annual_name = f"{variable}_meanBA_ANN"
        monthly_means_data[annual_name] = target_data[variable].mean(dim="time")

    # Add the latitude and longitude coordinates to the monthly_means_data dataset
    monthly_means_data["lat"] = target_data["lat"]
    monthly_means_data["lon"] = target_data["lon"]

    # Save the xarray dataset to a new NetCDF file
    monthly_means_data.to_netcdf(
        "/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/GFED5_meanBA_2001_2020.nc"
    )

    #####################################################3

    print("MAX BURN AREA FOR EACH MONTH")

    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    months_name_list = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    monthly_means_data = xr.Dataset()

    lat_coords = np.linspace(-89.88, 89.88, num=720)  # Example latitude coordinates
    lon_coords = np.linspace(-179.9, 179.9, num=1440)  # Example longitude coordinates
    list_label = ["mean", "month"]
    # Assuming you have the 'data' array containing the values for 'total'
    total = xr.DataArray(coords=[lat_coords, lon_coords], dims=["lat", "lon"])
    for variable in ["Total", "Crop", "Peat", "Defo"]:
        array_shape = (720, 1440, 12)
        my_array = np.zeros(array_shape)
        month_mean_array_shape = (720, 1440, 2)
        mm_array = np.zeros(month_mean_array_shape)
        data_array = xr.DataArray(
            my_array,
            coords=[
                ("latitude", latitude_coords),
                ("longitude", longitude_coords),
                ("months", months),
            ],
        )
        month_mean = xr.DataArray(
            mm_array,
            coords=[
                ("latitude", latitude_coords),
                ("longitude", longitude_coords),
                ("month_mean", list_label),
            ],
        )
        variable_data = target_data[variable]
        for i, month in enumerate(months):
            print(month)

            # get data for months across the years
            burned_area = variable_data.sel(time=target_data["time.month"].isin(month))

            mean_burned_area = burned_area.mean(dim="time")

            # store the mean value for each month in each grid cell in the list variable corresponding to the month
            data_array[:, :, i] = mean_burned_area
            # print("total")

        print(data_array[393, 719])
        # print( np.argmax(data_array[:,:,:], axis=2, keepdims=True))

        # get indices of maximum mean val for each grid cell
        # change index to +1
        # argmax_indices = argmax_indices.reshape(argmax_indices.shape + (1,))
        argmax_indices = np.expand_dims(np.argmax(data_array[:, :, :], axis=2), 2) + 1
        max_values = np.expand_dims(np.max(data_array[:, :, :], axis=2), 2)
        # max_values = max_values.reshape(max_values.shape + (1,))

        # Assign the results to month_mean

        # separate the list into 2 variables
        month_mean[:, :, 1] = (
            argmax_indices.squeeze()
        )  # Remove the singleton dimension to match month_mean shape
        month_mean[:, :, 0] = max_values.squeeze()
        # <CROP/PEAT/DEFO/NAT/TOTAL>_monthmaxBA,

        monthly_means_data[f"{variable}_monthmaxBA"] = month_mean

    monthly_means_data.to_netcdf(
        "/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/GFED5_monthmaxBA_2001_2020.nc"
    )


# In[ ]:


Data_grid_stat(target_data)


# ### MODEL NUDGED

# In[6]:


# read in data
import xarray as xr
import os
import matplotlib.pyplot as plt
import warnings
import numpy as np

warnings.filterwarnings("ignore")

os.chdir("/discover/nobackup/kmezuman/E6TpyrEPDnu")

# Specify the path to store the PNG files
from glob import glob

months = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
]
fnms = []

# # Open the files using xr.open_mfdataset()
# target_data = xr.open_mfdataset(fnms)
variables_to_extract = ["BA_tree", "BA_shrub", "BA_grass"]
years = range(1997, 2020)
# Open each file and load them into separate Datasets
datasets = []

for year in years:
    for month in months:
        file_pattern = f"{month}{year}.aijE6TpyrEPDnu.nc"
        file_paths = [f for f in os.listdir(".") if f.startswith(file_pattern)]

        for file_path in file_paths:
            dataset = xr.open_dataset(file_path)
            extracted_dataset = dataset.drop_vars(
                [var for var in dataset.variables if var not in variables_to_extract]
            )
            time_stamp = f"{month}{year}"  # Create a time stamp like 'JAN2013'
            extracted_dataset = extracted_dataset.expand_dims(
                time=[time_stamp]
            )  # Add time as a new dimension
            datasets.append(extracted_dataset)


# In[40]:


# Combine all extracted datasets into a single dataset along the time dimension
combined_dataset = xr.concat(datasets, dim="time")

# Save the combined dataset to a NetCDF file
output_path = (
    "/discover/nobackup/projects/giss_ana/users/kmezuamn/GFED5/combined_monthly_data.nc"
)
combined_dataset.to_netcdf(output_path)

# Close the datasets
for dataset in datasets:
    dataset.close()


# In[1]:


import pandas as pd
import xarray as xr

target_data = xr.open_dataset(
    "/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/combined_monthly_data.nc",
    engine="netcdf4",
)
target_data["BA"] = (
    target_data["BA_shrub"] + target_data["BA_tree"] + target_data["BA_grass"]
)
time_coords_str = target_data["time"].values
time_coords = pd.to_datetime(
    time_coords_str, format="%b%Y"
)  # Assuming 'JAN1997' format

# Update 'time' coordinate in the target_data dataset
target_data = target_data.assign_coords(time=time_coords)
target_data.close()

# In[2]:


target_data["time"]


# In[19]:


import numpy as np
import xarray as xr
import os


def Data_grid_stat_nudged(target_data):
    latitude_coords = np.linspace(-89.88, 89.88, num=90)

    # Create coordinates for the second dimension (1400 points) from -179.9 to 179.9
    longitude_coords = np.linspace(-179.9, 179.9, num=144)

    # Use numpy.meshgrid to create 2D coordinate grids for latitude and longitude
    # longitude_grid, latitude_grid = np.meshgrid(longitude_coords, latitude_coords)

    # Define the months of interest
    months_of_interest = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    months_name = [["DJF"], ["MAM"], ["JJA"], ["SON"]]

    # Create an empty xarray dataset to store the monthly means
    monthly_means_data = xr.Dataset()

    # Loop through each variable
    for variable in ["BA", "BA_shrub", "BA_grass", "BA_tree"]:
        # Loop through each month set of interest
        for i, months in enumerate(months_of_interest):
            # Select the burned area values for the current month set and variable
            burned_area = target_data[variable].sel(
                time=target_data["time"].dt.month.isin(months)
            )
            print(burned_area)
            # Calculate the mean burned area for each grid cell
            mean_burned_area = burned_area.mean(dim="time")
            # <CROP/PEAT/DEFO/NAT/TOTAL>_meanBA_DJF,
            # Add the mean burned area as a new variable to the monthly_means_data dataset
            variable_name = f"{variable}_meanBA_{months_name[i][0]}"
            monthly_means_data[variable_name] = mean_burned_area
        annual_name = f"{variable}_meanBA_ANN"
        monthly_means_data[annual_name] = target_data[variable].mean(dim="time")

    # Add the latitude and longitude coordinates to the monthly_means_data dataset
    monthly_means_data["lat"] = latitude_coords
    monthly_means_data["lon"] = longitude_coords

    # Save the xarray dataset to a new NetCDF file
    monthly_means_data.to_netcdf(
        "/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/Model_meanBA_1997_2019_nudged.nc"
    )

    #####################################################3

    print("MAX BURN AREA FOR EACH MONTH")

    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    months_name_list = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    target_data = target_data.sel(time=slice("2001-01-01", None))
    monthly_means_data = xr.Dataset()

    lat_coords = np.linspace(-89.88, 89.88, num=90)  # Example latitude coordinates
    lon_coords = np.linspace(-179.9, 179.9, num=144)  # Example longitude coordinates
    list_label = ["mean", "month"]
    # Assuming you have the 'data' array containing the values for 'total'
    total = xr.DataArray(coords=[lat_coords, lon_coords], dims=["lat", "lon"])
    for variable in ["BA", "BA_shrub", "BA_grass", "BA_tree"]:
        array_shape = (90, 144, 12)
        my_array = np.zeros(array_shape)
        month_mean_array_shape = (90, 144, 2)
        mm_array = np.zeros(month_mean_array_shape)
        data_array = xr.DataArray(
            my_array,
            coords=[
                ("latitude", latitude_coords),
                ("longitude", longitude_coords),
                ("months", months),
            ],
        )
        month_mean = xr.DataArray(
            mm_array,
            coords=[
                ("latitude", latitude_coords),
                ("longitude", longitude_coords),
                ("month_mean", list_label),
            ],
        )
        variable_data = target_data[variable]
        for i, month in enumerate(months):
            print(month)

            # get data for months across the years
            burned_area = variable_data.sel(
                time=target_data["time"].dt.month.isin(month)
            )

            mean_burned_area = burned_area.mean(dim="time")

            # print(np.nonzero(mean_burned_area))
            # print(months_name_list[i] )
            # print(mean_burned_area[393,719].values)
            # print(mean_burned_area[140,454].values)

            # store the mean value for each month in each grid cell in the list variable corresponding to the month
            data_array[:, :, i] = mean_burned_area
            # print("total")

        # print(data_array[393,719])
        # print( np.argmax(data_array[:,:,:], axis=2, keepdims=True))

        # get indices of maximum mean val for each grid cell
        # change index to +1
        argmax_indices = np.expand_dims(np.argmax(data_array[:, :, :], axis=2), 2) + 1
        max_values = np.expand_dims(np.max(data_array[:, :, :], axis=2), 2)

        # Assign the results to month_mean

        # separate the list into 2 variables
        month_mean[:, :, 1] = (
            argmax_indices.squeeze()
        )  # Remove the singleton dimension to match month_mean shape
        month_mean[:, :, 0] = max_values.squeeze()
        # <CROP/PEAT/DEFO/NAT/TOTAL>_monthmaxBA,

        monthly_means_data[f"{variable}_monthmaxBA"] = month_mean

    monthly_means_data.to_netcdf(
        "/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/Model_monthmaxBA_2001_2019_nudged.nc"
    )
    # data_array.to_netcdf("new_monthly.nc")
    # month_mean.to_netcdf('month_mean.nc')


# In[20]:


Data_grid_stat_nudged(target_data)

datasets = []
os.chdir("/discover/nobackup/kmezuman/E6TpyrEPD")

for year in years:
    for month in months:
        file_pattern = f"{month}{year}.aijE6TpyrEPD.nc"
        file_paths = [f for f in os.listdir(".") if f.startswith(file_pattern)]

        for file_path in file_paths:
            dataset = xr.open_dataset(file_path)
            extracted_dataset = dataset.drop_vars(
                [var for var in dataset.variables if var not in variables_to_extract]
            )
            time_stamp = f"{month}{year}"  # Create a time stamp like 'JAN2013'
            extracted_dataset = extracted_dataset.expand_dims(
                time=[time_stamp]
            )  # Add time as a new dimension
            datasets.append(extracted_dataset)


# In[40]:


# Combine all extracted datasets into a single dataset along the time dimension
combined_dataset = xr.concat(datasets, dim="time")

# Save the combined dataset to a NetCDF file
output_path = (
    "/discover/nobackup/projects/giss_ana/users/kmezuamn/GFED5/combined_monthly_data.nc"
)
combined_dataset.to_netcdf(output_path)

# Close the datasets
for dataset in datasets:
    dataset.close()


# In[1]:


import pandas as pd
import xarray as xr

target_data = xr.open_dataset(
    "/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/combined_monthly_data.nc",
    engine="netcdf4",
)
target_data["BA"] = (
    target_data["BA_shrub"] + target_data["BA_tree"] + target_data["BA_grass"]
)
time_coords_str = target_data["time"].values
time_coords = pd.to_datetime(
    time_coords_str, format="%b%Y"
)  # Assuming 'JAN1997' format

# Update 'time' coordinate in the target_data dataset
target_data = target_data.assign_coords(time=time_coords)


def Data_grid_stat_model(target_data):
    latitude_coords = np.linspace(-89.88, 89.88, num=90)

    # Create coordinates for the second dimension (1400 points) from -179.9 to 179.9
    longitude_coords = np.linspace(-179.9, 179.9, num=144)

    # Use numpy.meshgrid to create 2D coordinate grids for latitude and longitude
    # longitude_grid, latitude_grid = np.meshgrid(longitude_coords, latitude_coords)

    # Define the months of interest
    months_of_interest = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    months_name = [["DJF"], ["MAM"], ["JJA"], ["SON"]]

    # Create an empty xarray dataset to store the monthly means
    monthly_means_data = xr.Dataset()

    # Loop through each variable
    for variable in ["BA", "BA_shrub", "BA_grass", "BA_tree"]:
        # Loop through each month set of interest
        for i, months in enumerate(months_of_interest):
            # Select the burned area values for the current month set and variable
            burned_area = target_data[variable].sel(
                time=target_data["time"].dt.month.isin(months)
            )
            print(burned_area)
            # Calculate the mean burned area for each grid cell
            mean_burned_area = burned_area.mean(dim="time")
            # <CROP/PEAT/DEFO/NAT/TOTAL>_meanBA_DJF,
            # Add the mean burned area as a new variable to the monthly_means_data dataset
            variable_name = f"{variable}_meanBA_{months_name[i][0]}"
            monthly_means_data[variable_name] = mean_burned_area
        annual_name = f"{variable}_meanBA_ANN"
        monthly_means_data[annual_name] = target_data[variable].mean(dim="time")

    # Add the latitude and longitude coordinates to the monthly_means_data dataset
    monthly_means_data["lat"] = latitude_coords
    monthly_means_data["lon"] = longitude_coords

    # Save the xarray dataset to a new NetCDF file
    monthly_means_data.to_netcdf(
        "/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/Model_meanBA_1997_2019.nc"
    )

    #####################################################3

    print("MAX BURN AREA FOR EACH MONTH")

    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    months_name_list = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    target_data = target_data.sel(time=slice("2001-01-01", None))
    monthly_means_data = xr.Dataset()

    lat_coords = np.linspace(-89.88, 89.88, num=90)  # Example latitude coordinates
    lon_coords = np.linspace(-179.9, 179.9, num=144)  # Example longitude coordinates
    list_label = ["mean", "month"]
    # Assuming you have the 'data' array containing the values for 'total'
    total = xr.DataArray(coords=[lat_coords, lon_coords], dims=["lat", "lon"])
    for variable in ["BA", "BA_shrub", "BA_grass", "BA_tree"]:
        array_shape = (90, 144, 12)
        my_array = np.zeros(array_shape)
        month_mean_array_shape = (90, 144, 2)
        mm_array = np.zeros(month_mean_array_shape)
        data_array = xr.DataArray(
            my_array,
            coords=[
                ("latitude", latitude_coords),
                ("longitude", longitude_coords),
                ("months", months),
            ],
        )
        month_mean = xr.DataArray(
            mm_array,
            coords=[
                ("latitude", latitude_coords),
                ("longitude", longitude_coords),
                ("month_mean", list_label),
            ],
        )
        variable_data = target_data[variable]
        for i, month in enumerate(months):
            print(month)

            # get data for months across the years
            burned_area = variable_data.sel(
                time=target_data["time"].dt.month.isin(month)
            )

            mean_burned_area = burned_area.mean(dim="time")

            # print(np.nonzero(mean_burned_area))
            # print(months_name_list[i] )
            # print(mean_burned_area[393,719].values)
            # print(mean_burned_area[140,454].values)

            # store the mean value for each month in each grid cell in the list variable corresponding to the month
            data_array[:, :, i] = mean_burned_area
            # print("total")

        # print(data_array[393,719])
        # print( np.argmax(data_array[:,:,:], axis=2, keepdims=True))

        # get indices of maximum mean val for each grid cell
        # change index to +1
        argmax_indices = np.expand_dims(np.argmax(data_array[:, :, :], axis=2), 2) + 1
        max_values = np.expand_dims(np.max(data_array[:, :, :], axis=2), 2)

        # Assign the results to month_mean

        # separate the list into 2 variables
        month_mean[:, :, 1] = (
            argmax_indices.squeeze()
        )  # Remove the singleton dimension to match month_mean shape
        month_mean[:, :, 0] = max_values.squeeze()
        # <CROP/PEAT/DEFO/NAT/TOTAL>_monthmaxBA,

        monthly_means_data[f"{variable}_monthmaxBA"] = month_mean

    monthly_means_data.to_netcdf(
        "/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/Model_monthmaxBA_2001_2019.nc"
    )
    # data_array.to_netcdf("new_monthly.nc")
    # month_mean.to_netcdf('month_mean.nc')


# In[20]:


Data_grid_stat_model(target_data)


# In[ ]:
