import traceback
import rasterio
from os import listdir, makedirs, remove, mkdir
from os.path import isfile, join, basename, exists, dirname
import os
import re
from rasterio.transform import from_origin
import rioxarray as riox
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import json
import matplotlib.dates as mdates
import xarray as xr
from glob import glob
import h5py
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
import re

from utilityGlobal import (
    M2TOMHA,
    SCRIPTS_ENV_VARIABLES,
    MONTHLIST,
    DISTINCT_COLORS,
    MASK_LIST,
    MONTHS_NUM,
    GFED_COVER_LABELS,
    LAND_COVER_LABELS,
    NUM_MONTHS,
    MARKER,
    SECONDS_IN_A_YEAR,
    KILOGRAMS_TO_GRAMS,
    COLOR_MAP,
    SQM_TO_SQHA,
    KM_NEG_2TOM_NEG_2,
    KM_SQUARED_TO_M_SQUARED,
    DAYS_TO_SECONDS,
    EARTH_RADIUS_KM,
    EARTH_RADIUS_METERS,
    MONTHLISTDICT,
)


def getEnvironmentVariables():
    return json.load(open(SCRIPTS_ENV_VARIABLES, "r"))


def extract_scaling_factor(units):
    """
    Extracts the scaling factor from the units string and converts it to a number.
    For example, '10^-3 kg/m^2/s' will return 0.001 and 'kg/m^2/s'.
    """
    try:
        match = re.match(r"^(10\^(-?\d+)|[-+]?\d*\.?\d+([eE][-+]?\d+)?)\s*(.*)$", units)
        if match:
            if match.group(1).startswith("10^"):
                scaling_factor = float(10) ** float(match.group(2))
            else:
                scaling_factor = float(match.group(1))
        unit = match.group(4)
        return scaling_factor, unit
    except:
        return 1.0, units  # Default scaling factor is 1 if not specified

# Function to convert units
def handle_units(data_array, units, monthly=False, file_path=None, year=None):
    """
    Apply appropriate scaling based on variable units.
   
    Parameters:
        data_array: xarray DataArray
        units: str, unit string from ModelE
        monthly: bool, whether the data is monthly or annual
        file_path: str, path to the file (needed for monthly data to extract month)
        year: int, the year (needed for monthly data to calculate days)

    Returns:
        scaled_data: scaled DataArray
        new_units: str, updated units after scaling
    """
    grid_cell_area = None
    scaling_factor = 1.0
    new_units = units

    # Dictionary of unit handlers
    unit_handlers = {
        'kg CO2n m-2 s-1': {
            'needs_area': True,
            'new_units': 'kg/s'  # After integrating over area
        },
        'kg m-2 s-1': {
            'needs_area': True,
            'new_units': 'kg/s'  # After integrating over area
        },
        'kg/m2/s': {
            'needs_area': True,
            'new_units': 'kg/s'
        },
        'm-2 s-1': {
            'scaling': 1.,  
            'new_units': 'm-2 s-1'
            #'scaling': KM_SQUARED_TO_M_SQUARED,  
            #'new_units': 'km-2 yr-1'
        },
        '/m2': {
            'scaling': 1E6 * 1E-10,  # For flash counts
            'new_units': 'flashes/km2/yr'
        },
        'm-2': {
            'scaling': 1E6 * 1E-10,  # For flash counts
            'new_units': 'flashes/km2/yr'
        }
    }

    # Check exact match first
    if units in unit_handlers:
        handler = unit_handlers[units]
    else:
        # Generic handling based on unit patterns
        if any(pattern in units for pattern in ['kg', 'm-2', 's-1']):
            # Mass flux handling
            handler = {
                'needs_area': True,
                'new_units': 'kg/s'
            }
        else:
            # Default handling
            handler = {
                'scaling': 1.0,
                'new_units': units
            }

    if handler.get('needs_area', False):
        if grid_cell_area is None:
            # Get spatial dimensions only (-2 and -1)
            spatial_shape = data_array.shape[-2:]
            grid_cell_area = calculate_grid_area(
                grid_area_shape=spatial_shape,
                units='m^2'
            )
            # Expand dimensions if needed for broadcasting
            if len(data_array.shape) > 2:
                # Add any leading dimensions (e.g., time)
                for _ in range(len(data_array.shape) - 2):
                    grid_cell_area = grid_cell_area[np.newaxis, ...]
        data_array = data_array * grid_cell_area

    if handler.get('scaling'):
        data_array = data_array * handler['scaling']
    new_units = handler['new_units']

    # Apply time scaling
    if 's-1' in units or '/s' in units:
        if monthly and file_path and year:
            # Get month from filename (e.g. JAN, FEB, etc.)
            month = file_path.split(".")[0][-7:-4]
            # Convert month name to number (1-12)
            month_num = MONTHLIST.index(month) + 1
            # Calculate seconds in this month
            days_in_month = days_to_months(str(month_num).zfill(2), year)
            seconds_in_month = days_in_month * DAYS_TO_SECONDS
            # Apply monthly scaling
            data_array = data_array * seconds_in_month
            new_units = new_units.replace('/s', '/month').replace('s-1', 'month-1')
        else:
            # For annual data, multiply by seconds in year
            data_array = data_array * SECONDS_IN_A_YEAR
            new_units = new_units.replace('/s', '/yr').replace('s-1', 'yr-1')

    return data_array, new_units



def calculate_grid_area(grid_area_shape, units="km"):
    """
    Calculate the area of each grid cell based on grid dimensions.

    Parameters:
    grid_area_shape (tuple): Shape of the grid (nlat,nlon)
    units (str): Units for the calculation ("km or m")

    Returns:
    np.ndarray: Grid area matrix with diensions matching grid_area_shape
    """

    # Grid resolution
    nlat = grid_area_shape[0]  # Number of latitude bands
    nlon = grid_area_shape[1]  # Number of longitude bands

    # Latitude and longitude step size (degrees)
    lat_step = 180 / nlat
    lon_step = 360 / nlon

    # Convert step size to radians
    lat_step_rad = np.deg2rad(lat_step)
    lon_step_rad = np.deg2rad(lon_step)

    # Initialize grid cell area matrix
    grid_area = np.zeros((nlat, nlon))

    # Loop over each latitude band
    for i in range(nlat):
        # Latitude at the center of the grid cell
        lat = -90 + (i + 0.25) * lat_step

        # Convert latitude to radians
        lat_rad = np.deg2rad(lat)

        earth_radius = EARTH_RADIUS_KM if units == "km" else EARTH_RADIUS_METERS
        # Calculate the surface area of the grid cell at this latitude
        area = (
            (earth_radius**2)
            * lon_step_rad
            * (np.sin(lat_rad + lat_step_rad / 2) - np.sin(lat_rad - lat_step_rad / 2))
        )

        # Assign the area to all longitude cells for this latitude band
        grid_area[i, :] = area

    # Display the grid area matrix
    return grid_area


def obtain_netcdf_files(dir_path) -> list:
    """
    loops through files in the current director and returns a list of files that are netcdf files

    :param dir_path: the file path
    :return: all files in the "dir_path" that are netcdf files
    """
    return [
        join(dir_path, file)
        for file in listdir(dir_path)
        if isfile(join(dir_path, file))
        and (file.split(".")[-1] == "hdf5" or file.split(".")[-1] == "nc")
    ]


def read_gfed5(files, upscaled=False, variable_name="Total"):
    """
    Reads multiple HDF5 files using h5py, preserves monthly burned area data,
    and returns the data as xarray.DataArray.
    """
    monthly_data_list = []
    time_values = []
    attribute_dict = {}
    
    # Sort files to ensure chronological order
    files.sort()
    
    for file in files:
        with Dataset(file) as netcdf_dataset:
            # obtain the variables in the netcdf_dataset
            # dimensions (1, 720, 1440)
            var_total_data_array = netcdf_dataset.variables[variable_name][:]
            var_crop_data_array = netcdf_dataset.variables["Crop"][:]
            var_defo_data_array = netcdf_dataset.variables["Defo"][:]
            var_peat_data_array = netcdf_dataset.variables["Peat"][:]

            # obtain the numpy array for each netcdf variable
            # transform the arrays dimensions to (720, 1440) and convert the metric to km^2 -> m^2
            var_data_array = (
                var_total_data_array
                - var_peat_data_array
                - var_crop_data_array
                - var_defo_data_array
            )

            var_data_array = (
                var_data_array if upscaled else var_data_array * KM_SQUARED_TO_M_SQUARED
            )

            # this depends if the shape includes a time dimension
            monthly_burned_area = (
                var_data_array[0] if len(var_data_array.shape) > 2 else var_data_array
            )

            # Copy attributes of the burned area fraction
            if not attribute_dict:
                for attr_name in netcdf_dataset.variables["Total"].ncattrs():
                    attribute_dict[attr_name] = getattr(
                        netcdf_dataset.variables["Total"], attr_name
                    )
                
                # update the units to match the upscaling process
                attribute_dict["units"] = "m^2"

            # obtain the height and width from the upscale shape
            # create an evenly spaced array representing the longitude and the latitude
            height, width = monthly_burned_area.shape
            latitudes = np.linspace(-90, 90, height)
            longitudes = np.linspace(-180, 180, width)

            # Extract year and month from the filename
            # Determine which path separator is used
            if '\\' in file:
                filename = file.split('\\')[-1]
            else:
                filename = file.split('/')[-1]

            # Print file name for debugging
            print(f"Processing file: {filename}")

            # Specific extraction for BA[YYYYMM]_90144.nc pattern
            year_month_match = re.search(r'BA(\d{6})(?:_|\\.)', filename)

            if year_month_match:
                year_month = year_month_match.group(1)
                year = int(year_month[:4])
                month = int(year_month[4:6])
                print(f"Extracted year: {year}, month: {month}")
                
                # Create a decimal year value for this month (e.g., 2009.0 for Jan, 2009.083 for Feb)
                decimal_year = year + (month - 1) / 12.0
                time_values.append(decimal_year)
            else:
                # Fallback to 4-digit pattern
                year_match = re.search(r'(\d{4})', filename)
                if year_match:
                    year = int(year_match.group(1))
                    print(f"Extracted year (fallback): {year}")
                    time_values.append(float(year))
                else:
                    raise ValueError(f"Could not extract year from filename: {filename}")

            # Store the monthly burned area directly
            monthly_data_list.append(monthly_burned_area)

    # Check if we have any data
    if not monthly_data_list:
        raise ValueError("No valid data found in the files")

    # Create xarray DataArray with time dimension preserved
    total_data_array = xr.DataArray(
        monthly_data_list,
        coords={
            "time": time_values,
            "latitude": latitudes,
            "longitude": longitudes,
        },
        dims=["time", "latitude", "longitude"],
        attrs=attribute_dict,
    )

    return total_data_array, longitudes, latitudes


def read_gfed4s_emis(files, upscaled=False):
    """
    Read GFED4s emissions data from multiple netCDF files using xarray. 
    Parameters
    ---------- 
    files : list List of paths to GFED4s emissions netCDF files 
    Returns 
    ------- 
    tuple
       - 'all_data': xarray data array (time, 90, 144)
       - 'lon': longitude array (144,) 
       - 'lat': latitude array (90,) 
    """

    #Initialize lists to store data from each file
    all_data = []
    all_years = []

    #Read first file to get dimensions
    ds = xr.open_dataset(files[0], decode_times=False)
    lon = ds.lon.values
    lat = ds.lat.values

    #Find the emissions variable (first non-coordinated variable)
    data_vars = list(ds.data_vars)
    emis_var = data_vars[0]
    print(ds[emis_var].shape)

    # Get units and other attributed from the first file
    attrs = ds[emis_var].attrs
    units = attrs.get('units','')
    ds.close()

    # Create grid cell area if dealing with per-are units
    if any(pattern in units for pattern in ['kg m-2 s-1', 'kg/m2/s']):
        grid_cell_area = calculate_grid_area(
                grid_area_shape=(lat.size, lon.size),
                units='m^2'
                )
    else:
        grid_cell_area = None

    # Sort files to ensure chronological order
    files.sort()

    # Loop over all files
    for filename in files:
        #extract year from filename
        year = int(os.path.basename(filename).split('.')[0])

        ds = xr.open_dataset(filename, decode_times=False)
        data = ds[emis_var].values

        # Process each month
        for month in range(12):
            month_data = data[month, :, :]
            
            # Convert units if needed
            #1. If per area units, multiply by grid cell area
            if grid_cell_area is not None:
                month_data = month_data * grid_cell_area # Convert from flux per area to total flux

            #2. Convert from per second to per month by multiplying with second in month
            days_in_month = days_to_months(str(month+1).zfill(2), year)
            seconds_in_month = days_in_month * DAYS_TO_SECONDS
            month_data = month_data * seconds_in_month

            # Append the entire data array
            all_data.append(month_data)
            # Add year for each month
            all_years.append(year)

        ds.close()

    # Convert lists to arrays and reshape to proper dimensions
    all_data = np.array(all_data) # (time,lat,lon)
    all_years = np.array(all_years)

    # Update units in attributes
    if grid_cell_area is not None:
        attrs['units'] = 'kg/yr' 

    # Create xarray DataArray
    total_value = xr.DataArray(
            all_data,
            dims=['time', 'lat', 'lon'],
            coords={
                'time': all_years,
                'lat': lat,
                'lon': lon
            },
            attrs=attrs
        )

    return total_value, lon, lat
                  
def read_gfed4s(files, upscaled=False):
    """
    Reads multiple HDF5 files using h5py, calculates the annual burned area,
    and returns the data as xarray.DataArray.
    """
    burned_fraction_list = []
    time_array = []

    for file_path in files:
        attribute_dict = {}
        # Open the HDF5 file using h5py
        with h5py.File(file_path, "r") as h5file:
            # Load lat and lon for constructing the xarray dataset
            lat = h5file["lat"][:]
            lon = h5file["lon"][:]
            burned_area_variable_shape = (
                h5file[f"burned_area/01/burned_fraction"].shape
                if not upscaled
                else h5file[f"burned_areas_01"].shape
            )

            # Sum burned fraction over all months
            annual_burned_fraction = np.zeros(shape=burned_area_variable_shape)
            for month in range(1, 13):
                burned_area_variable = (
                    h5file[f"burned_area/{month:02d}/burned_fraction"]
                    if not upscaled
                    else h5file[f"burned_areas_{month:02d}"]
                )

                month_burned_fraction = (
                    burned_area_variable[:] if not upscaled else burned_area_variable[:]
                )
                annual_burned_fraction += month_burned_fraction
            if not upscaled:
                # Access grid_cell_area using the method suggested
                grid_cell_area = h5file["ancill"]["grid_cell_area"][:]
                # Calculate total burned area
                total_burned_area = annual_burned_fraction * grid_cell_area
            else:
                total_burned_area = annual_burned_fraction
            burned_fraction_list.append(total_burned_area)
        # print(file_path.split("_"), file_path.split("_")[1].split(".")[0])
        # file paths and folders with underscores may effect this year extraction
        year = (
            file_path.split("_")[1].split(".")[0]
            if not upscaled
            else file_path.split("_")[1]
        )
        print(year)
        time_array.append(year)

    attribute_dict["units"] = "Global Burned Area m^2"
    attribute_dict["long_name"] = (
        'GFED4s burned fraction. Note that this INCLUDES an experimental "small fire" estimate and is thus different from the Giglio et al. (2013) paper'
    )
    # Convert the list to xarray.DataArray for further processing
    total_burned_area_all_years = (
        xr.DataArray(
            burned_fraction_list,
            dims=["time", "phony_dim_0", "phony_dim_1"],
            coords={
                "time": time_array,
                "lat": (["phony_dim_0", "phony_dim_1"], lat),
                "lon": (["phony_dim_0", "phony_dim_1"], lon),
            },
            attrs=attribute_dict,
        )
        if not upscaled
        else xr.DataArray(
            burned_fraction_list,
            dims=["time", "lat", "lon"],
            coords={"time": time_array, "lat": lat, "lon": lon},
            attrs=attribute_dict,
        )
    )

    return total_burned_area_all_years, lon, lat


def leap_year_check(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return True


def days_to_months(month, year):
    if (str(month) == "02") and leap_year_check(int(year)):
        return 29
    else:
        return MONTHLISTDICT[str(month)]


def read_ModelE(files, variables=["BA_tree", "BA_shrub", "BA_grass"], monthly=False):
    """
    Reads ModelE data for given variables

    Parameters:
    files (list): List of file paths to ModelE data files
    variables (list): List of variable names to read and sum
    monthly (bool): If True, process as monthly data; otherwise, as annual

    Returns:
    tuple: (data_array, longitude, latitude)
    - data_array: xarray DataArray with Dimensions [time, lat, lon]
    - longitude: array of longitude values
    - latitude: array of latitude values
    """

    # Initialized a list to store each year's dataset
    all_data = []
    all_years = []

    # Loop over each file and process it
    for file_path in files:
        ds = xr.open_dataset(file_path)
        attribute_dict = {}
        # Read dimension sizes dynamically from the dataset
        lat_size = len(ds['lat'])
        lon_size = len(ds['lon'])
        modelE_var_data = np.zeros(shape=(lat_size, lon_size))
        # Sum up all requested variables
        for variable in variables:
            # where function replaces values that do not meet the parameters condition
            # (replaces all values that are not greater than 0)
            var_data = ds[variable].where(ds[variable] > 0.0, 0.0)
            modelE_var_data = modelE_var_data + var_data

            # Get attributes from the first variable
            if not attribute_dict:
               for attr_name in ds[variable].attrs:
                   attribute_dict[attr_name] = getattr(ds[variable], attr_name)
               # Extract scaling factor from units
               units = attribute_dict.get('units', '')
               scaling_factor, units = extract_scaling_factor(units)
               attribute_dict['units'] = units
        # Apply scaling factor
        modelE_var_data *= scaling_factor

        year = int(file_path.split(".")[0][-4:]) if monthly else int(file_path.split("ANN")[1][:4])

        # Append data and time information
        all_data.append(modelE_var_data)
        all_years.append(year)

    # Convert lists to arrays
    all_data = np.array(all_data)
    all_years = np.array(all_years)

    # Create xarray DataArray
    modelE_all_year = xr.DataArray(
            all_data,
            dims=['time', 'lat', 'lon'],
            coords={
                'time': all_years,
                'lat': ds['lat'].values,
                'lon': ds['lon'].values
                },
            attrs=attribute_dict
            )

    # Handle units conversion and time scaling
    modelE_all_year, new_units = handle_units(
        modelE_all_year, 
        attribute_dict.get('units',''), 
        monthly=monthly,
        file_path=files[0] if monthly else None,  # Pass file_path for monthly data
        year=all_years[0] if monthly else None    # Pass year for monthly data
    )

    modelE_all_year.attrs["units"] = new_units
    return modelE_all_year, ds["lon"], ds["lat"]


def read_lightning_data(files, yearly=True, upscaled=False):
    """
    Reads multiple lightning files using h5py, calculates the annual burned area,
    and returns the data as xarray.DataArray.
    """
    start_date = "2010-01-01"
    for file in files:
        with nc.Dataset(file) as netcdf_dataset:
            # dataset containing all xarray data array (used to create the final netcdf file)
            dataset_dict = {}
            attribute_dict = {}
            yearly_var_data = {}
            monthly_data = []

            # update the units to match the upscaling process
            density_variable = netcdf_dataset.variables["density"]
            density_variable_data = netcdf_dataset.variables["density"][:]
            # density_variable_data = density_variable_data.where(
            #     density_variable_data > 0.0, 0.0
            # )
            time_data_array = netcdf_dataset.variables["time"][:]
            # print(netcdf_dataset.variables["time"])

            # Copy attributes of the burned area fraction
            for attr_name in density_variable.ncattrs():
                attribute_dict[attr_name] = getattr(density_variable, attr_name)

            date_range = pd.date_range(
                start_date, freq="MS", periods=len(density_variable_data)
            )
            print(len(density_variable_data))
            for month in range(len(density_variable_data)):
                current_year = int(str(date_range[month]).split("-")[0])
                curr_month = str(date_range[month]).split("-")[1]

                # change to upscaled checks if the data is already upscaled
                if upscaled:
                    grid_cell_area = calculate_grid_area(
                        grid_area_shape=density_variable_data[month].shape
                    )
                    var_data_array = density_variable_data[month]
                # if the data is not upscaled preform further calculations
                else:
                    # var_data_array = density_variable[:][month]
                    var_data_array = density_variable_data[month]

                var_data_array = var_data_array * days_to_months(
                    curr_month, current_year
                )
                if int(current_year) in yearly_var_data:
                    yearly_var_data[int(current_year)] += var_data_array
                else:
                    yearly_var_data[int(current_year)] = var_data_array

                monthly_data.append(var_data_array)
                # print(f"Current Month {month}: ", var_data_array.sum())
            yearly_var_data = dict(sorted(yearly_var_data.items()))
            attribute_dict["units"] = "lightning strokes km-2 d-1"
            latitudes = np.linspace(-90, 90, density_variable.shape[-2])
            longitudes = np.linspace(-180, 180, density_variable.shape[-1])
            # creates the data array and saves it to a file
            var_data_array_xarray = xr.DataArray(
                monthly_data,
                coords={
                    "time": time_data_array,
                    "latitude": latitudes,
                    "longitude": longitudes,
                },
                dims=["time", "latitude", "longitude"],
                attrs=attribute_dict,
            )

            attribute_dict["units"] = "lightning strokes km-2 y-1"
            # yearly_var_data_dict_value = [
            #     data_array * (364 if leap_year_check(int(year)) else 365)
            #     for year, data_array in list(yearly_var_data.items())
            # ]

            yearly_var_data_array_xarray = xr.DataArray(
                list(yearly_var_data.values()),
                coords={
                    "time": list(yearly_var_data.keys()),
                    "latitude": latitudes,
                    "longitude": longitudes,
                },
                dims=["time", "latitude", "longitude"],
                attrs=attribute_dict,
            )
            yearly_var_data_array_xarray = yearly_var_data_array_xarray.where(
                yearly_var_data_array_xarray > 0.0, 0.0
            )

            return (
                (yearly_var_data_array_xarray if yearly else var_data_array_xarray),
                longitudes,
                latitudes,
            )


def define_subplot(
    fig,
    ax,
    decade_data,
    lons,
    lats,
    cborientation,
    fraction,
    pad,
    labelpad,
    fontsize,
    title,
    clabel,
    masx=None,
    is_diff=False,
    glob=None,
    logMap=False,
):
    """Define the properties of a subplot with optional difference normalization.""" 
    # labelpad sets the distance of the colorbar from the map
    # Only use default scaling if masx is None 
    if masx is None: 
        masx = decade_data.max().item() 
        print(f"Auto-setting masx to {masx}")

    # Mask values that are less than or equal to zero
    masked_data = np.ma.masked_where(decade_data <= 0, decade_data)

    ax.coastlines(color="black")
    ax.add_feature(cfeature.LAND, edgecolor="gray")
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=1)

    # Creare a two-line title if global total is provided 
    if glob is not None:
        two_line_title = f"{title}\nGlobal Total: {glob}"
        ax.set_title(two_line_title, fontsize=10, pad=15)
    else:
        ax.set_title(title, fontsize=10, pad=5)

    # Handling difference normalization (if is_diff is true)
    if is_diff:
        data_min, data_max = decade_data.min(), decade_data.max()
        if data_min == data_max:
            norm = mcolors.Normalize(vmin=data_min - 1, vmax=data_max + 1)
        else:
            abs_max = max(abs(0.25 * data_min), abs(0.25 * data_max))
            norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
        p = ax.pcolormesh(
            lons,
            lats,
            decade_data,
            transform=ccrs.PlateCarree(),
            cmap="bwr",
            norm=norm,
            vmin=0 if not is_diff else None,
            vmax=masx if not is_diff else None,
        )
    else:
        # For non-difference plots, either use log scale or linear scale
        # Get the standard jet colormap
        cmap = plt.cm.jet.copy()
        # Set masked values )zerps amd megatives) to white)
        cmap.set_bad('white')

        if logMap and masked_data.min() > 0:
            # For log plots, ensure vmin is positive
            try:
                # try to get the minimum non-masked value
                if np.ma.count(masked_data) > 0:
                    vmin = max(masked_data.min() * 0.9, 1.e-10)
                else:
                    vmin = 1.e-10

                logNorm = mcolors.LogNorm(vmin=vmin, vmax=masx)
                p = ax.pcolormesh(
                        lons,
                        lats,
                        masked_data,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        norm=logNorm,
                        )
            except ValueError as e:
                print(f"Warning: Log scale error - {e}. Switching to linear scale")
                # For linear plots, use the full range
                p = ax.pcolormesh(
                        lons,
                        lats,
                        masked_data,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        vmin=0.0,
                        vmax=masx,
                        )
        else:
            p = ax.pcolormesh(
                        lons,
                        lats,
                        masked_data,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        vmin=0.0,
                        vmax=masx,
                        )

    cbar = fig.colorbar(p, ax=ax, orientation=cborientation, fraction=fraction, pad=pad)
    cbar.set_label(f"{clabel}", labelpad=labelpad, fontsize=fontsize)
    return ax


def map_plot(
    figure,
    axis,
    axis_length,
    axis_index,
    decade_data,
    longitude,
    latitude,
    subplot_title,
    units,
    cbarmax,
    is_diff=False,
    logMap=False,
    variables=None,
):
    """
    Plots spatial data with optimized color scaling to show data variability.

    Parameters:
    decade_data (xarray.DataArray): The spatial data to plot
    """
    print(axis_index, axis_length)
    print(f"Data range: min={decade_data.min().item():.5e}, max={decade_data.max().item():.5e}")

    # Convert to numpy for percentile calculations
    data_np = decade_data.values
    
    # Get positive data only for better scaling
    positive_data = data_np[data_np > 0]
    if len(positive_data) > 0:
        # Calculate percentiles of the positive data
        p95 = np.percentile(positive_data, 95)
        p99 = np.percentile(positive_data, 99)
        median = np.median(positive_data)
        print(f"Data percentiles: 50%={median: .5e}, 95%={p95: .5e}, 99%={p99: .5e}") 
    else:
        p95 = p99 = 1.0
        print("No positive data found")

    # Calculate global total based on units
    global_total = None
    if "m-2" in units.lower() or "m^-2" in units.lower() or "/m2" in units.lower():
        # Area-weighted for per-area units
        grid_cell_area = calculate_grid_area(
                grid_area_shape=decade_data.shape,
                units="m^2"
                )
        weighted_total = (decade_data * grid_cell_area).sum()
        # Extract base unit by removing area component
        base_unit = units.replace("m-2", "").replace("m^-2","").replace("/m2","").strip()
        if not base_unit:
            base_unit = "units"
        global_total = f"{weighted_total:.3e} {base_unit}"
    else:
        # Default case for other units
        global_total = f"{decade_data.sum():.3e} {units}"

    # Set a more reasonable colorbar max for better visualization
    if cbarmax is None or (not is_diff and cbarmax > p95):
        # Use 95th percentile for better visualization of variability,
        # but never go lower than 20% of the actual max to ensure outliers are visible
        min_allowed_cbarmax = 0.2 * decade_data.max().item()
        if p95 < min_allowed_cbarmax:
            cbarmax = min_allowed_cbarmax
        else:
            cbarmax = p95

    # Ensure a reasonable cbarmax value
    if cbarmax <= 0 or (not is_diff and cbarmax < 1e-10):
        cbarmax = 1.0

    print(f"Using cbarmax value: {cbarmax}")

    axis_value = axis if axis_length <= 1 else axis[axis_index]

    define_subplot(
        figure,
        axis_value,
        decade_data,
        longitude,
        latitude,
        cborientation="horizontal",
        fraction=0.05,
        pad=0.005,
        labelpad=0.5,
        fontsize=10,
        title=subplot_title,
        clabel=units,
        masx=cbarmax,
        is_diff=is_diff,
        glob=global_total,
        logMap=logMap,
    )


def time_series_plot(
    axis,
    data,
    marker,
    line_style,
    color,
    label,
    grid_visible=True,
):
    """
    Plots time series data with appropriate formatting for annual or monthly data.

    Parameters:
    axis (matplotlib.axes): The axis to plot on
    data (np.ndarray): A 2D array with columns (time, value)
    marker (str): Marker style for the plot
    line_style (str): Line style for the plot
    color (str): Color for the plot
    label (str): Label for the legend
    grid_visible (bool): Whether to show grid lines
    """
    # Extract time values and data values
    time_values = data[:, 0]
    data_values = data[:, 1]
    
    # Plot the data
    axis.plot(
        time_values,
        data_values,
        marker=marker,
        linestyle=line_style,
        color=color,
        label=label,
    )

    # Check if we have monthly data by looking for fractional parts
    has_monthly_data = np.any(np.mod(time_values, 1) > 0)
    
    if has_monthly_data:
        # For monthly data, create custom tick positions and labels
        # Find the unique years
        years = np.unique(np.floor(time_values).astype(int))
        
        # If we have a single year of data
        if len(years) == 1:
            # Set custom ticks at each month position
            tick_positions = []
            tick_labels = []
            
            # Month abbreviations
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            year = years[0]
            for month in range(12):
                decimal_time = year + month/12.0
                # Check if this month exists in our data
                if any(np.isclose(time_values, decimal_time, atol=0.01)):
                    tick_positions.append(decimal_time)
                    tick_labels.append(f"{month_names[month]}")
            
            # Set custom ticks
            axis.set_xticks(tick_positions)
            axis.set_xticklabels(tick_labels, rotation=45, ha='right')
            
            # Set the title to include the year
            axis.set_title(f"{axis.get_title()} ({year})")
        else:
            # For multiple years, show tick at start of each year and some months
            tick_positions = []
            tick_labels = []
            
            for year in years:
                tick_positions.append(year)
                tick_labels.append(f"{year}")
                
                # Add mid-year tick if we have fewer than 5 years
                if len(years) < 5:
                    tick_positions.append(year + 0.5)
                    tick_labels.append(f"Jul {year}")
            
            axis.set_xticks(tick_positions)
            axis.set_xticklabels(tick_labels, rotation=45 if len(years) < 5 else 0)
        
        # Add more space at the bottom for rotated labels
        plt.subplots_adjust(bottom=0.15)
    else:
        # For annual data, use integer ticks
        if len(time_values) <= 10:
            # For a small number of years, show all years
            axis.set_xticks(time_values)
            axis.set_xticklabels([str(int(year)) for year in time_values])
        else:
            # For many years, let matplotlib handle the ticks
            axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Add legend and grid
    axis.legend()
    axis.grid(grid_visible)
    
    return axis

def handle_time_extraction_type(file_paths, variables, NetCDF_Type):
    match (NetCDF_Type):
        case "BA_GFED4":
            total_value, longitude, latitude = read_gfed4s(
                files=file_paths, upscaled=False
            )
        case "BA_GFED4_upscale":
            total_value, longitude, latitude = read_gfed4s(
                files=file_paths, upscaled=True
            )
        case "BA_GFED5":
            total_value, longitude, latitude = read_gfed5(
                files=file_paths, upscaled=False
            )
        case "BA_GFED5_upscale":
            total_value, longitude, latitude = read_gfed5(
                files=file_paths, upscaled=True
            )
        case "ModelE":
            total_value, longitude, latitude = read_ModelE(
                files=file_paths, variables=variables
            )
        case "ModelE_Monthly":
            total_value, longitude, latitude = read_ModelE(
                files=file_paths, variables=variables, monthly=True
            )
        case "lightning":
            total_value, longitude, latitude = read_lightning_data(
                files=file_paths, upscaled=False
            )
        case "lightning_upscale":
            total_value, longitude, latitude = read_lightning_data(
                files=file_paths, upscaled=True
            )
        case "GFED4s_Yearly":
            total_value, longitude, latitude = read_gfed4s_emis(
                files=file_paths, upscaled=True
            )
        case _:
            print("[-] No Parsing Script Found For", NetCDF_Type)
    return (total_value, longitude, latitude)


def obtain_time_series_xarray(
    variables,
    NetCDF_folder_Path,
    NetCDF_Type,
    annual=True
):
    """
    Calculates the mean and the interannual variability
    """

    file_paths = obtain_netcdf_files(NetCDF_folder_Path)
    print(f"\nProcessing {NetCDF_Type}...")
    total_value, longitude, latitude = handle_time_extraction_type(
        file_paths=file_paths, variables=variables, NetCDF_Type=NetCDF_Type
    )

    time_dimension = total_value.dims[0]
    sum_dimensions = (total_value.dims[-2], total_value.dims[-1])
    # Debug prints
    print(f"Time dimension: {time_dimension}")
    print(f"Time values: {total_value.coords['time'].values}")

    time_mean_data = total_value.mean(dim="time")
    print(f"Shape of time_mean_data: {time_mean_data.shape}")

    units = total_value.attrs["units"]
    print(f"Units: {units}")

    # Calculate spatial sums based on units 
    if " m-2".lower() in units or " m^-2".lower() in units:
        grid_cell_dimension_shape = (total_value.shape[-2], total_value.shape[-1])
        grid_cell_area = calculate_grid_area(
            grid_area_shape=grid_cell_dimension_shape, units="m^2"
        )
        total_data_array = (total_value * grid_cell_area).sum(dim=sum_dimensions).values
        print("Data Array multiplied by grid_cell_area")
    else:
        total_data_array = total_value.sum(dim=sum_dimensions).values
        print("Regular Sum Implemented")

    # Get time values
    time_values = total_value.coords["time"].values
    years = np.unique(time_values)
    start_year = years[0]
    end_year = years[-1]

    # Check if we have monthly data
    is_monthly = len(total_data_array) > len(years)

    if is_monthly:
        print(f"Found monthly data ({len(total_data_array)} months for {len(years)} years)")
        if annual:
            print("Aggregating to annual totals")
            # Check if the data has 12 months per year
            if len(total_data_array) == len(years) * 12:
                # Reshape to (nyears, 12) and sum over months
                monthly_totals = total_data_array.reshape(len(years), 12)
                total_data_array = monthly_totals.sum(axis=1)
                time_values = years
            else:
                print(f"Warning: Number of months ({len(total_data_array)}) is not 12 times the number of years ({len(years)})")
                print("Proceeding with original data without reshaping")
                if annual:
                    # Create yearly values by averaging months
                    yearly_data = {}
                    for i, t in enumerate(time_values):
                        year = int(t)
                        if year in yearly_data:
                            yearly_data[year].append(total_data_array[i])
                        else:
                            yearly_data[year] = [total_data_array[i]]
                    
                    # Average the months for each year
                    time_values = np.array(list(yearly_data.keys()))
                    total_data_array = np.array([np.mean(vals) for vals in yearly_data.values()])
        else:
            print("Keeping monthly resolution")
            # For monthly data create decimal years (e.g. 2009.0, 2009.083, ...)
            if len(time_values) == len(total_data_array):
                # If time_values already has proper decimal years, use them
                print("Using provided time values")
            else:
                # Otherwise, create decimal years
                print("Creating decimal year values for months")
                # Create months array from 0 to 11
                months = np.arange(12)
                # Create decimal years by adding month fractions to each year
                time_values = np.array([year + month/12 for year in years for month in months])
                
                # Make sure the length matches total_data_array
                if len(time_values) > len(total_data_array):
                    time_values = time_values[:len(total_data_array)]
                
                print(f"Created {len(time_values)} time points")

    print(f"Time values shape: {time_values.shape}")
    print(f"Total data array shape: {total_data_array.shape}")
    print(f"First few time values: {time_values[:5]}")

    # Make sure the lengths match before creating column stack
    if len(time_values) != len(total_data_array):
        print(f"Warning: Length mismatch between time_values ({len(time_values)}) and total_data_array ({len(total_data_array)})")
        # Fix by truncating to the shorter length
        min_length = min(len(time_values), len(total_data_array))
        time_values = time_values[:min_length]
        total_data_array = total_data_array[:min_length]
        print(f"Truncated both arrays to length {min_length}")

    data_per_year_stack = np.column_stack((time_values, total_data_array))
    print(f"Final stacked shape: {data_per_year_stack.shape}")

    return (
        time_mean_data,
        data_per_year_stack,
        longitude,
        latitude,
        units,
        int(start_year),
        int(end_year),
    )


def run_time_series_analysis(folder_data_list, time_analysis_figure_data, annual=True):
    """
    Run time series analysis for multiple datasets
    Parameters
    _________
    folder_data_list : list
       List of folder data dictionaries
    time_analysis_figure_data : dict
       Figure metadata and settings
    annual : bool, optional
       If True, show annual totals for data
       If False, show monthly resolution data points 
       Default is True
    """
    # Plot side by side maps for GFED and ModelE
    _, time_analysis_axis = plt.subplots(figsize=(10, 6))

    global_year_max = 0
    global_year_min = 9999
    if not exists("figures"):
        mkdir("figures")
    # Example usage with test parameters
    for index, folder_data in enumerate(folder_data_list):
        map_figure, map_axis = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(18, 10),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )

        folder_path, figure_data, file_type, variables = (
            folder_data["folder_path"],
            folder_data["figure_data"],
            folder_data["file_type"],
            folder_data["variables"],
        )

        (
            time_mean_data,
            data_per_year_stack,
            longitude,
            latitude,
            units,
            start_year,
            end_year,
        ) = obtain_time_series_xarray(
            NetCDF_folder_Path=folder_path,
            NetCDF_Type=file_type,
            variables=variables,
            annual=False
        )

        figure_label = f"{figure_data['label']} ({start_year}-{end_year})"
        # Plot the preiod mean variable 
        map_plot(
            figure=map_figure,
            axis=map_axis,
            axis_length=1,
            axis_index=index,
            decade_data=time_mean_data,
            longitude=longitude,
            latitude=latitude,
            subplot_title=figure_label,
            units=units,
            cbarmax=figure_data["cbarmax"],
            logMap=True,
        )

        # Plot the time series 
        time_series_plot(
            axis=time_analysis_axis,
            data=data_per_year_stack,
            marker=figure_data["marker"],
            line_style=figure_data["line_style"],
            color=figure_data["color"],
            label=figure_label,
        )
        global_year_max = (
            int(end_year)
            if int(global_year_max) < int(end_year)
            else int(global_year_max)
        )
        global_year_min = (
            int(start_year)
            if int(global_year_min) > int(start_year)
            else int(global_year_min)
        )
        map_figure.savefig(f"figures/map_figure_{index}")

    if len(folder_data_list) > 1:
        loop_flag = True
        while loop_flag:
            print("Select the two datasets you would like to subtract")
            print("Enter q to quit the subtraction loop")
            print("NOTE: Please ensure the exact years match")
            valid_selections = range(0, len(folder_data_list))
            for index, folder_data in enumerate(folder_data_list):
                print(str(index) + ".) ", folder_data["folder_path"])
            first_selection = input(
                f"Please enter the number for the first selected data {valid_selections}: "
            )

            if first_selection == "q" or not int(first_selection) in valid_selections:
                loop_flag = False
                break
            else:
                first_selection = int(first_selection)

            second_selection = input(
                f"Please enter the number for the second selected data {valid_selections}: "
            )

            if second_selection == "q" or not int(second_selection) in valid_selections:
                loop_flag = False
                break
            else:
                second_selection = int(second_selection)

            if loop_flag == True:
                map_figure, map_axis = plt.subplots(
                    nrows=1,
                    ncols=1,
                    figsize=(18, 10),
                    subplot_kw={"projection": ccrs.PlateCarree()},
                )
                (
                    (time_mean_data_diff),
                    (data_per_year_stack_diff),
                    longitude_diff,
                    latitude_diff,
                    units_diff,
                    start_year_diff,
                    end_year_diff,
                    figure_label_diff,
                ) = run_time_series_diff_analysis(
                    folder_data_list[first_selection],
                    folder_data_list[second_selection],
                )
                map_plot(
                    figure=map_figure,
                    axis=map_axis,
                    axis_length=1,
                    axis_index=0,
                    decade_data=time_mean_data_diff,
                    longitude=longitude_diff,
                    latitude=latitude_diff,
                    subplot_title=figure_label_diff,
                    units=units_diff,
                    cbarmax=None,
                    is_diff=True,
                )
                #time_series_plot(
                #    axis=time_analysis_axis,
                #    data=data_per_year_stack_diff,
                #    marker="*",
                #    line_style="-",
                #    color="g",
                #    label=figure_label_diff,
                #)
                map_figure.savefig(
                    f"figures/figure{first_selection}_and_figure{second_selection}_diff_map"
                )

    time_analysis_axis.set_title(time_analysis_figure_data["title"])
    time_analysis_axis.set_xlabel(
        f"{time_analysis_figure_data['xlabel']} ({global_year_min}-{global_year_max})"
    )
    time_analysis_axis.set_ylabel(time_analysis_figure_data["ylabel"])
    _.savefig(f"figures/time_analysis_figure")
    plt.show()


def run_time_series_diff_analysis(folder_data_one, folder_data_two):
    """
    Calculate the difference between two datasets for time series analysis.

    Parameters:
    ----------
    folder_data_one : dict
        Dictionary with information about the first dataset
    folder_data_two : dict
        Dictionary with information about the second dataset
        
    Returns:
    -------
    tuple containing:
        - time_mean_data_diff : xarray.DataArray, spatial difference in temporal mean
        - data_per_year_stack_diff : ndarray, difference in time series values
        - longitude : array, longitude values
        - latitude : array, latitude values
        - units : str, units of the difference data
        - start_year : int, starting year
        - end_year : int, ending year
        - figure_label : str, label for the figure
    """
    folder_path_one, figure_data_one, file_type_one, variables_one = (
        folder_data_one["folder_path"],
        folder_data_one["figure_data"],
        folder_data_one["file_type"],
        folder_data_one["variables"],
    )

    folder_path_two, figure_data_two, file_type_two, variables_two = (
        folder_data_two["folder_path"],
        folder_data_two["figure_data"],
        folder_data_two["file_type"],
        folder_data_two["variables"],
    )

    # Obtain data for both datasets
    (
        time_mean_data_one,
        data_per_year_stack_one,
        longitude_one,
        latitude_one,
        units_one,
        start_year_one,
        end_year_one,
    ) = obtain_time_series_xarray(
        NetCDF_folder_Path=folder_path_one,
        NetCDF_Type=file_type_one,
        variables=variables_one,
    )

    (
        time_mean_data_two,
        data_per_year_stack_two,
        longitude_two,
        latitude_two,
        units_two,
        start_year_two,
        end_year_two,
    ) = obtain_time_series_xarray(
        NetCDF_folder_Path=folder_path_two,
        NetCDF_Type=file_type_two,
        variables=variables_two,
    )

    # Check if units are compatible for subtraction
    if units_one != units_two:
        print(f"WARNING: Units do not match! {units_one} vs {units_two}")
        print("Cannot perform subtraction with different units.")
        print("Continuing with calculation, but results may not be meaningful.")

    # Calculate the difference in spatial means
    time_mean_data_diff = time_mean_data_one.copy()
    time_mean_data_diff.values = time_mean_data_one.values - time_mean_data_two.values

    # Calculate the difference in time series data
    print(f"Shape of data_per_year_stack_two: {data_per_year_stack_two.shape}")
    print(f"Shape of data_per_year_stack_one: {data_per_year_stack_one.shape}")

    # Make sure the arrays have compatible shapes for subtraction
    min_length = min(len(data_per_year_stack_one), len(data_per_year_stack_two))
    data_per_year_stack_diff = data_per_year_stack_one[:min_length] - data_per_year_stack_two[:min_length]

    # Adjust the years for the difference time series
    min_year = min(
        data_per_year_stack_one[:, 0].min(), data_per_year_stack_two[:, 0].min()
    )

    for index in range(len(data_per_year_stack_diff)):
        data_per_year_stack_diff[index][0] = min_year + index

    # Create a descriptive figure label
    figure_label = f"{figure_data_one['label']} - {figure_data_two['label']}"

    return (
        time_mean_data_diff,
        data_per_year_stack_diff,
        longitude_one,
        latitude_one,
        units_one, # Use the units from the first dataset
        start_year_one,
        end_year_one,
        figure_label,
    )
