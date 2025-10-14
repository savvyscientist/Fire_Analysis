# Standard library imports
import os
import re
import json
import traceback
from os import listdir, makedirs, remove, mkdir
from os.path import isfile, join, basename, exists, dirname
from glob import glob

# Third-party imports
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import h5py
import rasterio
import rioxarray as riox
from rasterio.transform import from_origin
from netCDF4 import Dataset

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
    MONTHLISTDICT
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
        return 1.0, units  # No match found
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



# Global cache for grid areas
_grid_area_cache = {}
def calculate_grid_area(grid_area_shape, units="km"):
    """
    Calculate the area of each grid cell based on grid dimensions.

    Parameters:
    grid_area_shape (tuple): Shape of the grid (nlat,nlon)
    units (str): Units for the calculation ("km or m")

    Returns:
    np.ndarray: Grid area matrix with diensions matching grid_area_shape
    """
    cache_key = (grid_area_shape, units)
    if cache_key in _grid_area_cache:
        return _grid_area_cache[cache_key]

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

    _grid_area_cache[cache_key] = grid_area
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

def read_gfed5(files, variables=None, upscaled=False):
    """
    Reads multiple netCDF files using netCDF4.Dataset, preserves monthly burned area data,
    and returns the data as xarray.DataArray.

    Parameters:
    -----------
    files : list
        List of paths to GFED5 netCDF files
    variables : list or None, optional
        List of variable names to read and sum. If None, defaults to ["Total"] (default: None)
    upscaled : bool, optional
        If True, data is already in m^2; if False, convert from km^2 to m^2 (default: False)

    Returns:
    --------
    tuple : (total_data_array, longitudes, latitudes)
        - total_data_array: xarray DataArray with dimensions [time, latitude, longitude]
        - longitudes: longitude coordinate array
        - latitudes: latitude coordinate array
    """
    monthly_data_list = []
    time_values = []
    attribute_dict = {}

    # Set default variables if none provided
    if variables is None:
        variables = ["Total"]
    elif isinstance(variables, str):
        variables = [variables]  # Convert single string to list

    if not variables:
        raise ValueError("Variables list cannot be empty")

    # Sort files to ensure chronological order
    files.sort()

    if not files:
        raise ValueError("No files provided to read_gfed5")

    # Initialize coordinate variables
    latitudes = None
    longitudes = None

    for file_idx, file in enumerate(files):
        if not os.path.exists(file):
            print(f"Warning: File not found: {file}")
            continue

        try:
            with Dataset(file) as netcdf_dataset:
                # Initialize data array for this file
                monthly_burned_area = None

                # Process each variable and sum them
                for var_idx, variable_name in enumerate(variables):
                    if variable_name not in netcdf_dataset.variables:
                        print(f"Warning: Variable '{variable_name}' not found in {file}")
                        print(f"Available variables: {list(netcdf_dataset.variables.keys())}")
                        continue

                    # Read the variable from the netCDF dataset
                    var_data_array = netcdf_dataset.variables[variable_name][:]

                    # Convert units from km^2 to m^2 if not already upscaled
                    var_data_array = (
                        var_data_array if upscaled else var_data_array * KM_SQUARED_TO_M_SQUARED
                    )

                    # Handle different array dimensions
                    var_monthly_data = (
                        var_data_array[0] if len(var_data_array.shape) > 2 else var_data_array
                    )

                    # Sum variables (first iteration initializes, subsequent ones add)
                    if monthly_burned_area is None:
                        monthly_burned_area = var_monthly_data.copy()
                    else:
                        monthly_burned_area += var_monthly_data

                    # Get coordinate information from the first variable of the first file
                    if not attribute_dict and file_idx == 0 and var_idx == 0:
                        for attr_name in netcdf_dataset.variables[variable_name].ncattrs():
                            attribute_dict[attr_name] = getattr(
                                netcdf_dataset.variables[variable_name], attr_name
                            )
                        # Update the units to match the upscaling process
                        attribute_dict["units"] = "m^2"

                # Skip this file if no valid variables were found
                if monthly_burned_area is None:
                    print(f"No valid variables found in {file}")
                    continue

                # Get coordinate information from the first file
                if file_idx == 0:
                    # Get spatial dimensions and create coordinate arrays
                    height, width = monthly_burned_area.shape
                    latitudes = np.linspace(-90, 90, height)
                    longitudes = np.linspace(-180, 180, width)

                # Extract year and month from the filename
                filename = os.path.basename(file)
                print(f"Processing file: {filename}")

                # Pattern for GFED5 files: BA[YYYYMM]_*.nc or similar
                year_month_match = re.search(r'BA(\d{6})(?:_|\\.)', filename)

                if year_month_match:
                    year_month = year_month_match.group(1)
                    year = int(year_month[:4])
                    month = int(year_month[4:6])
                    print(f"Extracted year: {year}, month: {month}")

                    # Create a decimal year value for this month
                    decimal_year = year + (month - 1) / 12.0
                    time_values.append(decimal_year)
                else:
                    # Fallback: try to extract just a 4-digit year
                    year_match = re.search(r'(\d{4})', filename)
                    if year_match:
                        year = int(year_match.group(1))
                        print(f"Extracted year (fallback): {year}")
                        time_values.append(float(year))
                    else:
                        print(f"Warning: Could not extract date from filename: {filename}")
                        time_values.append(2001.0 + file_idx)

                # Store the monthly burned area data
                monthly_data_list.append(monthly_burned_area)

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Check if we have any valid data
    if not monthly_data_list:
        raise ValueError("No valid data found in any of the files")

    if latitudes is None or longitudes is None:
        raise ValueError("Could not determine coordinate information from files")

    # Convert lists to arrays and sort by time
    monthly_data_array = np.array(monthly_data_list)
    time_values = np.array(time_values)

    if len(time_values) > 1:
        sort_idx = np.argsort(time_values)
        monthly_data_array = monthly_data_array[sort_idx]
        time_values = time_values[sort_idx]

    # Create xarray DataArray
    total_data_array = xr.DataArray(
        monthly_data_array,
        coords={
            "time": time_values,
            "latitude": latitudes,
            "longitude": longitudes,
        },
        dims=["time", "latitude", "longitude"],
        attrs=attribute_dict,
    )

    print(f"Successfully processed {len(monthly_data_list)} files")
    print(f"Variables processed: {variables}")
    print(f"Time range: {time_values.min():.3f} to {time_values.max():.3f}")
    print(f"Data shape: {total_data_array.shape}")

    return total_data_array, longitudes, latitudes

def read_modelEinput_emis(files, variables, upscaled=False, monthly=False):
    """
    Read modelE input emissions, whether that is GFED4s or FINN data
    processed to be used as modelE input, ans thus has the same format.
    Reading is done from multiple netCDF files using xarray. 
    **the below function is currently written for multiple files but 
    actually uses only one as it does one year monthly calcualtions**
    Parameters
    ---------- 
    files : list List of paths to emissions netCDF files 
    monthly (bool): If True, process as monthly data; otherwise, as annual
    Returns 
    ------- 
    tuple
       - 'all_data': xarray data array (time, 90, 144)
       - 'lon': longitude array (144,) 
       - 'lat': latitude array (90,) 
    """

    #Initialize lists to store data from each file
    all_data = []
    time_values = []
    all_months = []

    #Read first file to get dimensions
    ds = xr.open_dataset(files[0], decode_times=False)
    lon = ds.lon.values
    lat = ds.lat.values

    #Find the emissions variable (first non-coordinated variable)
    emis_var = variables[0]

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

        test =0. 
        # Process each month
        for month in range(12):
            month_data = data[month, :, :]
            
            # Convert units if needed
            #1. If per area units, multiply by grid cell area
            if grid_cell_area is not None:
                month_data = month_data * grid_cell_area # Convert from flux per area to total flux
            #print('grid_cell_area.sum',grid_cell_area.sum())

            #2. Convert from per second to per month by multiplying with second in month
            days_in_month = days_to_months(str(month+1).zfill(2), year)
            seconds_in_month = days_in_month * DAYS_TO_SECONDS
            month_data = month_data * seconds_in_month 
            test = test + month_data
            # Append the entire data array
            all_data.append(month_data)
            # Add month
            all_months.append(str(month+1).zfill(2))

            decimal_year = year + month/12.0
            time_values.append(decimal_year)

        ds.close()

    # Convert lists to arrays and reshape to proper dimensions
    all_data = np.array(all_data) # (time,lat,lon)
    all_months = np.array(all_months)
    time_values = np.array(time_values)

    # Update units in attributes
    if grid_cell_area is not None:
        attrs['units'] = 'kg/month' 

    # Create xarray DataArray
    total_value = xr.DataArray(
            all_data,
            dims=['time', 'lat', 'lon'],
            coords={
                'time': time_values,
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

def read_ModelE(files, variables, monthly=False, scale='none'):
    """         
    Reads ModelE data for given variables                                                                                                                   
                
    Parameters: 
    files (list): List of file paths to ModelE data files
    variables (list): List of variable names to read and sum
    monthly (bool): If True, process as monthly data; otherwise, as annual
    scale (str): Scaling method ('fearth' or 'none')
                
    Returns:    
    tuple: (data_array, longitude, latitude)
    - data_array: xarray DataArray with Dimensions [time, lat, lon]
    - longitude: array of longitude values
    - latitude: array of latitude values
    """         
                
    if not files:                 
        raise ValueError("No files provided to read_ModelE")
    
    if not variables:
        raise ValueError("Variables list cannot be empty. Please specify which variables to read.")
    
    if not isinstance(variables, list):
        raise TypeError("Variables must be provided as a list")
    
    # Check if files exist
    for file_path in files:      
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
               
    # Sort files to ensure chronological order
    files.sort()
    print(f"Processing {len(files)} ModelE files")
    print(f"Reading variables: {variables}")
               
    # Initialize lists to store each year's dataset
    all_data = []
    time_values = []
    
    # Initialize variables to store coordinate information from first file
    longitude = None
    latitude = None
    attribute_dict = {}
               
    # Loop over each file and process it
    for file_idx, file_path in enumerate(files):
        print(f"Processing file: {file_path}")
        
        try:
            ds = xr.open_dataset(file_path)
            
            # Store coordinates from first file
            if file_idx == 0:
                longitude = ds['lon']
                latitude = ds['lat']
                
            # Read dimension sizes dynamically from the dataset
            lat_size = len(ds['lat'])
            lon_size = len(ds['lon'])
            modelE_var_data = np.zeros(shape=(lat_size, lon_size))
                   
            # Account for scaling of gridcells (as defined in denom of the DEFACC)
            scale_data = None
            if scale == 'fearth':
                if 'axyp' in ds.variables and 'soilfr' in ds.variables:
                    scale_data = ds['axyp'].where(ds['axyp'] > 0.0, 0.0)
                    scale_data = scale_data * ds['soilfr'].where(ds['soilfr'] > 0.0, 0.0)
                else:
                    print(f"Warning: Required scaling variables 'axyp' or 'soilfr' not found in {file_path}")
                   
            # Initialize scaling factor for this file
            scaling_factor = 1.0
            
            # Sum up all requested variables
            variables_found = []
            for variable in variables:
                if variable not in ds.variables:
                    print(f"Warning: Variable '{variable}' not found in {file_path}")
                    continue
                    
                variables_found.append(variable)

                # where function replaces values that do not meet the parameters condition
                # (replaces all values that are not greater than 0)
                var_data = ds[variable].where(ds[variable] > 0.0, 0.0)
                
                if scale == 'fearth' and scale_data is not None:
                    var_data = var_data * scale_data
                    
                modelE_var_data = modelE_var_data + var_data
                    
                # Get attributes from the first variable of the first file
                if not attribute_dict and file_idx == 0:
                    for attr_name in ds[variable].attrs:
                        attribute_dict[attr_name] = getattr(ds[variable], attr_name)
                    
                    # Extract scaling factor from units
                    units = attribute_dict.get('units', '')
                    scaling_factor, units = extract_scaling_factor(units)
                    attribute_dict['units'] = units
            
            # Check if any variables were found in this file
            if not variables_found:
                print(f"Error: None of the requested variables {variables} were found in {file_path}")
                continue
                    
            # Apply scaling factor
            modelE_var_data = modelE_var_data * scaling_factor
                    
            # Handle time extraction
            if monthly:
                # Extract year from filename
                try:
                    year = int(file_path.split(".")[0][-4:])
                    
                    # Extract month from filename (e.g. JAN, FEB, etc.)
                    month = file_path.split(".")[0][-7:-4]
                    # Convert month name to number (0-11)
                    month_num = MONTHLIST.index(month)
                    
                    # Create decimal year value (e.g., 2009.0 for Jan, 2009.083 for Feb)
                    decimal_year = year + month_num/12.0
                    
                    print(f"Extracted year: {year}, month: {month} ({month_num+1}), decimal: {decimal_year:.3f}")
                except (ValueError, IndexError, AttributeError) as e:
                    # Fallback to just using the order in the file list if filename parsing fails
                    print(f"WARNING: Failed to extract month/year from filename: {e}. Using file order instead.")
                    year = 2009  # Default year
                    month_num = len(time_values) % 12
                    decimal_year = year + month_num/12.0
                    
                time_values.append(decimal_year)
            else:   
                # For annual data, just use the year
                try:
                    year = int(file_path.split("ANN")[1][:4])
                except (ValueError, IndexError):
                    print(f"WARNING: Could not extract year from annual filename: {file_path}")
                    year = 2009 + file_idx  # Fallback
                time_values.append(year)
                 
            # Append data
            print(f"Extracted year: {year}, month: {month} ({month_num+1}), decimal: {decimal_year:.3f}")
            all_data.append(modelE_var_data)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
        finally:
            # Always close the dataset
            if 'ds' in locals():
                ds.close()
                 
    if not all_data:
        raise ValueError("No valid data was processed from any files")
        
    # Convert lists to arrays
    all_data = np.array(all_data)
    time_values = np.array(time_values)
             
    # Sort by time value if needed
    if len(time_values) > 1:
        sort_idx = np.argsort(time_values)
        all_data = all_data[sort_idx]
        time_values = time_values[sort_idx]
             
    # Create xarray DataArray
    modelE_all_year = xr.DataArray(
        all_data,
        dims=['time', 'lat', 'lon'],
        coords={
            'time': time_values,
            'lat': latitude.values,
            'lon': longitude.values
        },
        attrs=attribute_dict
    )
    
    # Before handle_units
    print(f"Before handle_units - Data shape: {modelE_all_year.shape}")
    print(f"Before handle_units - Sample values: {modelE_all_year.values.flat[:5]}")
    print(f"Before handle_units - Data sum: {modelE_all_year.sum().item()}")


    # Handle units conversion and time scaling
    modelE_all_year, new_units = handle_units(
        modelE_all_year, 
        attribute_dict.get('units',''), 
        monthly=monthly,
        file_path=files[0] if monthly else None,  # Pass file_path for monthly data
        year=int(time_values[0]) if monthly else None    # Pass year for monthly data
    )        

    # After handle_units
    print(f"After handle_units - Data shape: {modelE_all_year.shape}")
    print(f"After handle_units - Sample values: {modelE_all_year.values.flat[:5]}")
    print(f"After handle_units - Data sum: {modelE_all_year.sum().item()}")
    print(f"Units changed from '{attribute_dict.get('units', '')}' to '{new_units}'")

             
    modelE_all_year.attrs["units"] = new_units
    print(f"ModelE data time values: {time_values}")
    
    return modelE_all_year, longitude, latitude

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
    
    # Check for None data
    if decade_data is None:
        print("Error: decade_data is None")
        return ax

    # Convert to numpy if needed
    data_np = decade_data.values if hasattr(decade_data, 'values') else decade_data
    
    # Only use default scaling if masx is None 
    if masx is None: 
        masx = np.nanmax(data_np)
        print(f"Auto-setting masx to {masx}")

    # Create mask for visualization
    # For burned area data, we want to show very small positive values
    if "BA" in title.upper() or "BURN" in title.upper():
        # For burned area, mask only truly zero values and negatives
        masked_data = np.ma.masked_where((data_np <= 0) | np.isnan(data_np), data_np)
        print(f"Burned area masking: masking values <= 0")
    else:
        # For other data, use the original threshold
        masked_data = np.ma.masked_where((data_np <= 0) | np.isnan(data_np), data_np)

    # Check if we have any unmasked data
    if np.ma.count(masked_data) == 0:
        print("Warning: All data is masked! Adjusting masking strategy...")
        # Try less restrictive masking
        masked_data = np.ma.masked_where(np.isnan(data_np), data_np)
        if np.ma.count(masked_data) == 0:
            print("Error: Still no valid data after relaxed masking")
            return ax
    
    # Print mask information for debugging
    mask_percentage = np.ma.count_masked(masked_data) / masked_data.size * 100
    print(f"Masked values: {np.ma.count_masked(masked_data)} of {masked_data.size} ({mask_percentage:.2f}%)")

        # === NEW DEBUGGING SECTION ===
    print(f"=== PLOTTING DEBUG for {title} ===")
    print(f"lons shape: {lons.shape if hasattr(lons, 'shape') else type(lons)}")
    print(f"lats shape: {lats.shape if hasattr(lats, 'shape') else type(lats)}")
    print(f"masked_data shape: {masked_data.shape}")
    print(f"masked_data type: {type(masked_data)}")
    
    # Check coordinate ranges
    if hasattr(lons, 'values'):
        lon_vals = lons.values
        lat_vals = lats.values
    else:
        lon_vals = lons
        lat_vals = lats
        
    print(f"Longitude range: {np.min(lon_vals):.2f} to {np.max(lon_vals):.2f}")
    print(f"Latitude range: {np.min(lat_vals):.2f} to {np.max(lat_vals):.2f}")
    
    # Check some actual data values
    unmasked_data = masked_data[~masked_data.mask] if hasattr(masked_data, 'mask') else masked_data[masked_data > 0]
    if len(unmasked_data) > 0:
        print(f"Sample unmasked values: {unmasked_data.flat[:5]}")
        print(f"Unmasked range: {np.min(unmasked_data):.2e} to {np.max(unmasked_data):.2e}")
    else:
        print("No unmasked data found!")

    # Set up the map
    ax.coastlines(color="black")
    ax.add_feature(cfeature.LAND, edgecolor="gray", alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=1)

    # Creare a two-line title if global total is provided 
    if glob is not None:
        two_line_title = f"{title}\nGlobal Total: {glob}"
        ax.set_title(two_line_title, fontsize=10, pad=15)
    else:
        ax.set_title(title, fontsize=10, pad=5)

    # Handling difference normalization (if is_diff is true)
    if is_diff:
        data_min, data_max = np.nanmin(data_np), np.nanmax(data_np)
        if data_min == data_max:
            norm = mcolors.Normalize(vmin=data_min - 1, vmax=data_max + 1)
        else:
            abs_max = max(abs(data_min), abs(data_max))
            norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
        p = ax.pcolormesh(
            lons,
            lats,
            data_np,
            transform=ccrs.PlateCarree(),
            cmap="bwr",
            norm=norm,
        )
    else:
        # For non-difference plots, either use log scale or linear scale
        # Get the standard jet colormap
        cmap = plt.cm.jet.copy()
        # For Cartopy, bad values must be fully transparent
        cmap.set_bad(alpha=0.0)

        # Check if data has any positive values for log scale
        log_compatible = np.ma.count(masked_data) > 0 and np.ma.min(masked_data) > 0
        
        if logMap and log_compatible: 
            # For log plots, ensure vmin is positive 
            try: 
                # For GFED data, use a more reasonable vmin to avoid extreme compression 
                if "GFED" in title.upper() or "BA" in title.upper(): 
                    # Use a vmin that's not too far from the bulk of the data 
                    positive_vals = masked_data[masked_data > 0] 
                    if len(positive_vals) > 0: 
                        p1 = np.percentile(positive_vals, 1)  # 1st percentile 
                        p10 = np.percentile(positive_vals, 10)  # 10th percentile 
                        # Use 1st percentile, but not smaller than 1e-6 * vmax 
                        vmin = max(p1, masx * 1e-6) 
                        print(f"GFED log scale: using 1st percentile vmin: {vmin:.2e} (p1={p1:.2e})") 
                    else: 
                        vmin = masx * 1e-6 
                else: 
                    # For other data, use the original approach but with limits 
                    vmin = max(np.ma.min(masked_data) * 0.1, masx * 1e-8)

                print(f"Log scale vmin: {vmin:.2e}, vmax: {masx:.2e}")

                if vmin >= masx:
                    vmin = masx * 0.01
                    print(f"Adjusted vmin to {vmin:.2e} (1% of vmax)")

                logNorm = mcolors.LogNorm(vmin=vmin, vmax=masx)

                                # === COORDINATE TEST FOR GFED ===
                if "GFED" in title.upper():
                    print("Testing with forced coordinates for GFED")
                    # Create standard lat/lon arrays
                    test_lons = np.linspace(-180, 180, masked_data.shape[1])
                    test_lats = np.linspace(-90, 90, masked_data.shape[0])
                    lons_mesh, lats_mesh = np.meshgrid(test_lons, test_lats)

                    p = ax.pcolormesh(
                        lons,
                        lats,
                        masked_data,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        norm=logNorm,
                    ) 
                else:
                    # Use original coordinates for ModelE
                    p = ax.pcolormesh(
                        lons,
                        lats,
                        masked_data,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        norm=logNorm,
                    )

            except (ValueError, RuntimeError) as e:
                print(f"Warning: Log scale error - {e}. Switching to linear scale")
                # For linear plots, use a very small vmin to show small values
                vmin = np.ma.min(masked_data) if np.ma.count(masked_data) > 0 else 0

                                # === COORDINATE TEST FOR GFED (Linear fallback) ===
                if "GFED" in title.upper():
                    print("Testing with forced coordinates for GFED (linear fallback)")
                    # Create standard lat/lon arrays
                    test_lons = np.linspace(-180, 180, masked_data.shape[1])
                    test_lats = np.linspace(-90, 90, masked_data.shape[0])
                    lons_mesh, lats_mesh = np.meshgrid(test_lons, test_lats)

                    p = ax.pcolormesh(
                        lons_mesh,
                        lats_mesh,
                        masked_data,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=masx,
                    )
                else:
                    # Use original coordinates for ModelE
                    p = ax.pcolormesh(
                        lons,
                        lats,
                        masked_data,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=masx,
                    )
        else:
            # Linear scale - use very small vmin for burned area data
            if "BA" in title.upper() or "BURN" in title.upper():
                vmin = np.ma.min(masked_data) if np.ma.count(masked_data) > 0 else 0
                if vmin == 0 and np.ma.count(masked_data) > 0:
                    # Find smallest positive value
                    positive_vals = masked_data[masked_data > 0]
                    if len(positive_vals) > 0:
                        vmin = np.min(positive_vals) * 0.1
            else:
                vmin = 0.0
                
            print(f"Linear scale vmin: {vmin:.2e}, vmax: {masx:.2e}")

                        # === COORDINATE TEST FOR GFED (Linear scale) ===
            if "GFED" in title.upper():
                print("Testing with forced coordinates for GFED (linear)")
                # Create standard lat/lon arrays
                test_lons = np.linspace(-180, 180, masked_data.shape[1])
                test_lats = np.linspace(-90, 90, masked_data.shape[0])
                lons_mesh, lats_mesh = np.meshgrid(test_lons, test_lats)
                
                p = ax.pcolormesh(
                    lons_mesh,
                    lats_mesh,
                    masked_data,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=masx,
                )
            else:
                # Use original coordinates for ModelE
                p = ax.pcolormesh(
                    lons,
                    lats,
                    masked_data,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=masx,
                )

    print(f"Final plotting values - vmin: {vmin if 'vmin' in locals() else 'N/A':.2e}, vmax: {masx:.2e}")
    print(f"=== END PLOTTING DEBUG ===")
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

    # Check if decade_data is None
    if decade_data is None:
        print("Error: decade_data is None, cannot create map")
        return

    # Try to convert to numpy safely
    try:
        data_np = decade_data.values if hasattr(decade_data, 'values') else decade_data
        print(f"Data array shape: {data_np.shape}")
        print(f"Data range: min={data_np.min():.5e}, max={data_np.max():.5e}")

        # Get statistics for all data (including zeros)
        non_nan_data = data_np[~np.isnan(data_np)]
        print(f"Non-NaN values: {len(non_nan_data)} out of {data_np.size}")

        # Get positive data only for better scaling
        positive_data = non_nan_data[non_nan_data > 0]
        print(f"Positive values: {len(positive_data)} out of {data_np.size} ({len(positive_data)/data_np.size*100:.2f}%)")

        if len(positive_data) > 0: 
            # Calculate percentiles of the positive data 
            p50 = np.percentile(positive_data, 50) 
            p95 = np.percentile(positive_data, 95) 
            p99 = np.percentile(positive_data, 99) 
            p999 = np.percentile(positive_data, 99.9) 
            data_min = np.min(positive_data) 
            data_max = np.max(positive_data) 

            print(f"Positive data percentiles: 50%={p50:.5e}, 95%={p95:.5e}, 99%={p99:.5e}, 99.9%={p999:.5e}") 
            print(f"Positive data range: min={data_min:.5e}, max={data_max:.5e}") 
            print(f"Dynamic range: {data_max/data_min:.2e} orders of magnitude") 

            # Check for extreme outliers 
            if data_max / p99 > 100: 
                print(f"WARNING: Extreme outliers detected! Max is {data_max/p99:.1f}x larger than 99th percentile") 
                # Find and report outlier locations 
                outlier_threshold = p999 
                outliers = data_np > outlier_threshold 
                if np.any(outliers): 
                    outlier_count = np.sum(outliers) 
                    print(f"Found {outlier_count} extreme outlier(s) above {outlier_threshold:.5e}") 
                    # Get indices of outliers 
                    outlier_indices = np.where(outliers) 
                    if len(outlier_indices[0]) <= 5:  # Report up to 5 outlier locations 
                        for i in range(min(5, len(outlier_indices[0]))): 
                            lat_idx, lon_idx = outlier_indices[0][i], outlier_indices[1][i] 
                            outlier_value = data_np[lat_idx, lon_idx] 
                            print(f"  Outlier at grid [{lat_idx}, {lon_idx}]: {outlier_value:.5e}") 

            # Sample some positive values 
            sample_size = min(10, len(positive_data)) 
            sample_indices = np.linspace(0, len(positive_data)-1, sample_size, dtype=int) 
            print(f"Sample positive values: {positive_data[sample_indices]}") 
        else: 
            p50 = p95 = p99 = p999 = 0.0
            print("No positive data found")

    except Exception as e:
        print(f"Error analyzing data: {e}")
        return

    # Calculate global total based on units
    global_total = None
    try:
        if hasattr(decade_data, 'sum'):
            if any(unit in units.lower() for unit in ["m-2", "m^-2", "/m2"]):
                # Area-weighted for per-area units
                grid_cell_area = calculate_grid_area(
                    grid_area_shape=decade_data.shape,
                    units="m^2"
                )
                weighted_total = (decade_data * grid_cell_area).sum()
                # Extract base unit by removing area component
                base_unit = units.replace("m-2", "").replace("m^-2", "").replace("/m2", "").strip()
                if not base_unit:
                    base_unit = "units"
                global_total = f"{weighted_total.item():.3e} {base_unit}"
            else:
                # Default case for other units
                total_sum = decade_data.sum()
                if hasattr(total_sum, 'item'):
                    global_total = f"{total_sum.item():.3e} {units}"
                else:
                    global_total = f"{float(total_sum):.3e} {units}"

        print(f"Global total: {global_total}")
    except Exception as e:
        print(f"Error calculating global total: {e}")
        global_total = "N/A"

    # Set colorbar max based on data characteristics
    if cbarmax is None:
        if len(positive_data) > 0:
            # For GFED data, which often has very skewed distributions
            if "GFED" in subplot_title.upper() or "BA" in subplot_title.upper():
               # Check for extreme outliers
               data_max = np.max(positive_data) 
               ratio_99_to_max = p99 / data_max if data_max > 0 else 1 
               print(f"Data max: {data_max:.5e}, 99th percentile: {p99:.5e}, ratio: {ratio_99_to_max:.5e}")

               if ratio_99_to_max < 1e-3:  # If 99th percentile is much smaller than max
                   print("Extreme outlier detected! Using 99.9th percentile to avoid outlier distortion")
                   cbarmax = p999 if p999 > 0 else p99
               else: 
                   cbarmax = p99 if p99 > 0 else p95 
                   print(f"Using percentile-based scaling for GFED/BA data: {cbarmax:.5e}")
            else:
                # For other data, use 95th percentile
                cbarmax = p95 if p95 > 0 else p50
                print(f"Using 95th percentile for other data: {cbarmax:.5e}")

            # Ensure minimum threshold
            if cbarmax <= 0:
                cbarmax = data_np.max() if data_np.max() > 0 else 1.0
        else:
            cbarmax = 1.0
            print("No positive data, setting default cbarmax=1.0")
    else:
        print(f"Using provided cbarmax: {cbarmax}")

    # Ensure a reasonable cbarmax value
    if cbarmax <= 0:
        cbarmax = 1.0
        print(f"Corrected non-positive cbarmax to: {cbarmax}")

    # For very small values, consider different scaling
    if cbarmax < 1e-10:
        print(f"Very small cbarmax ({cbarmax:.2e}), consider checking data units or scaling")

    axis_value = axis if axis_length <= 1 else axis[axis_index]

    # Simple test - print exactly what's being passed to define_subplot
    print(f"Calling define_subplot with:")
    print(f"  masx (cbarmax): {cbarmax}")
    print(f"  logMap: {logMap}")
    print(f"  is_diff: {is_diff}")
    print(f"  Data min/max being plotted: {np.nanmin(decade_data.values):.2e} / {np.nanmax(decade_data.values):.2e}")

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
        #plt.subplots_adjust(bottom=0.15, right=0.85)
    else:
        # For annual data, use integer ticks
        if len(time_values) <= 10:
            # For a small number of years, show all years
            axis.set_xticks(time_values)
            axis.set_xticklabels([str(int(year)) for year in time_values])
        else:
            # For many years, let matplotlib handle the ticks
            axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) 

        axis.grid(grid_visible)
    return axis

def handle_time_extraction_type(file_paths, variables, NetCDF_Type, scale='none'):
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
                files=file_paths, upscaled=False, variables=variables
            )
        case "BA_GFED5_upscale":
            total_value, longitude, latitude = read_gfed5(
                files=file_paths, upscaled=True, variables=variables
            )
        case "ModelE":
            total_value, longitude, latitude = read_ModelE(
                files=file_paths, variables=variables, scale=scale
            )
        case "ModelE_Monthly":
            total_value, longitude, latitude = read_ModelE(
                files=file_paths, variables=variables, monthly=True, scale=scale
            )
        case "lightning":
            total_value, longitude, latitude = read_lightning_data(
                files=file_paths, upscaled=False
            )
        case "lightning_upscale":
            total_value, longitude, latitude = read_lightning_data(
                files=file_paths, upscaled=True
            )
        case "GFED4s_Monthly":
            total_value, longitude, latitude = read_modelEinput_emis(
                files=file_paths, variables=variables, upscaled=True, monthly=True
            )
        case "FINN2.5_Monthly":
            total_value, longitude, latitude = read_modelEinput_emis(
                files=file_paths, variables=variables, upscaled=True, monthly=True
            )
        case _:
            print("[-] No Parsing Script Found For", NetCDF_Type)
    return (total_value, longitude, latitude)


def obtain_time_series_xarray(
    variables,
    NetCDF_folder_Path,
    NetCDF_Type,
    annual=True,
    save_netcdf=False,
    output_dir=None
):
    """
    Calculates the mean and the interannual variability
    """

    try:
        file_paths = obtain_netcdf_files(NetCDF_folder_Path)

        if not file_paths:
            print(f"Warning: No netCDF files found in {NetCDF_folder_Path}")
            return None, None, None, None, None, None, None

        print(f"\nProcessing {NetCDF_Type}...")
        total_value, longitude, latitude = handle_time_extraction_type(
            file_paths=file_paths, variables=variables, NetCDF_Type=NetCDF_Type,scale='none'
        )

        if total_value is None:
            print(f"Error: Failed to extract data for {NetCDF_Type}")
            return None, None, None, None, None, None, None

        time_dimension = total_value.dims[0]
        sum_dimensions = (total_value.dims[-2], total_value.dims[-1])

        # Debug prints
        # Debug prints
        print(f"Time dimension: {time_dimension}")
        print(f"Time values: {total_value.coords['time'].values}")
        print(f"Raw data shape: {total_value.shape}")
        print(f"Raw data range: min={total_value.min().item():.5e}, max={total_value.max().item():.5e}")

        # **ADD THIS DEBUGGING BLOCK HERE**
        # Check for problematic values in raw data
        raw_data = total_value.values
        print(f"Raw data statistics:")
        print(f"  - Non-zero values: {np.count_nonzero(raw_data)} out of {raw_data.size}")
        print(f"  - Positive values: {np.sum(raw_data > 0)} out of {raw_data.size}")
        print(f"  - NaN values: {np.sum(np.isnan(raw_data))}")
        print(f"  - Infinite values: {np.sum(np.isinf(raw_data))}")
        
        # Check each time slice
        for t in range(min(3, total_value.shape[0])):  # Check first 3 time slices
            time_slice = total_value.isel(time=t).values
            print(f"  - Time slice {t}: min={np.nanmin(time_slice):.5e}, max={np.nanmax(time_slice):.5e}, positive_count={np.sum(time_slice > 0)}")

        time_mean_data = total_value.mean(dim="time")
        print(f"Shape of time_mean_data: {time_mean_data.shape}")

        print(f"Time mean data statistics:")
        mean_data = time_mean_data.values
        print(f"  - Range: min={np.nanmin(mean_data):.5e}, max={np.nanmax(mean_data):.5e}")
        print(f"  - Non-zero values: {np.count_nonzero(mean_data)} out of {mean_data.size}")
        print(f"  - Positive values: {np.sum(mean_data > 0)} out of {mean_data.size}")
        
        # Check for extreme values
        if np.nanmax(mean_data) / np.nanmin(mean_data[mean_data > 0]) > 1e10:
            print(f"WARNING: Extreme dynamic range detected in time_mean_data!")
            # Find the extreme values
            flat_data = mean_data.flatten()
            flat_data = flat_data[~np.isnan(flat_data)]
            if len(flat_data) > 0:
                sorted_data = np.sort(flat_data)
                print(f"  - Smallest positive: {sorted_data[sorted_data > 0][0]:.5e}")
                print(f"  - Largest value: {sorted_data[-1]:.5e}")
                print(f"  - 99th percentile: {np.percentile(sorted_data[sorted_data > 0], 99):.5e}")
        
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

        if save_netcdf: 
            # Use the provided output_dir or fall back to parent directory of input 
            if output_dir: 
                save_path = output_dir 
            else: 
                save_path = os.path.dirname(NetCDF_folder_Path)

            # Create output filename 
            time_type = "annual" if annual else "monthly" 
            vars_str = "_".join(variables) if variables else "data" 
            output_file = f"{NetCDF_Type}_{vars_str}_{time_type}_{int(start_year)}-{int(end_year)}_timeseries.nc" 
            output_path = os.path.join(save_path, output_file) 

            # Create and save dataset 
            ds = xr.Dataset({ 
                             'total_data': (['time'], data_per_year_stack[:, 1], {'units': units}) 
                             }, coords={ 
                                        'time': (['time'], data_per_year_stack[:, 0], {'units': 'years'}) 
                                        }) 
            ds.to_netcdf(output_path) 
            print(f"Time series saved to: {output_path}")

        return (
            time_mean_data,
            data_per_year_stack,
            longitude,
            latitude,
            units,
            int(start_year),
            int(end_year),
        )

    except Exception as e:
        print(f"Error in obtain_time_series_xarray: {str(e)}")
        print(f"Error details: {traceback.format_exc()}")
        return None, None, None, None, None, None, None

def save_combined_netcdf(all_datasets, output_dir, annual, global_year_min, global_year_max):
    """
    Save all datasets in a single NetCDF file.
    
    Parameters:
    -----------
    all_datasets : dict
        Dictionary with dataset information
    output_dir : str
        Directory to save the NetCDF file
    annual : bool
        Whether data is annual or monthly
    global_year_min : int
        Minimum year across all datasets
    global_year_max : int
        Maximum year across all datasets
    """
    try:
        # Create filename
        time_type = "annual" if annual else "monthly"
        output_file = f"combined_timeseries_{time_type}_{global_year_min}-{global_year_max}.nc"
        output_path = os.path.join(output_dir, output_file)
        
        # Find the most complete time axis (dataset with most time points)
        max_time_points = 0
        reference_time = None
        for dataset_key, dataset_info in all_datasets.items():
            if len(dataset_info['time']) > max_time_points:
                max_time_points = len(dataset_info['time'])
                reference_time = dataset_info['time']
        
        # Create xarray Dataset
        data_vars = {}
        coords = {'time': (['time'], reference_time, {
            'units': 'years' if annual else 'decimal_years',
            'long_name': 'Time coordinate',
            'description': f'Time coordinate for {"annual" if annual else "monthly"} data'
        })}
        
        # Add each dataset as a variable
        for dataset_key, dataset_info in all_datasets.items():
            # Handle case where datasets might have different time lengths
            if len(dataset_info['time']) == len(reference_time):
                # Same length, use directly
                data_values = dataset_info['data']
                time_values = dataset_info['time']
            else:
                # Different lengths, interpolate or pad
                print(f"Warning: Dataset {dataset_key} has different time length. Interpolating...")
                data_values = np.interp(reference_time, dataset_info['time'], dataset_info['data'])
                time_values = reference_time
            
            # Create variable name (clean it for NetCDF compatibility)
            var_name = dataset_key.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            
            data_vars[var_name] = (['time'], data_values, {
                'units': dataset_info['units'],
                'long_name': f'Total data for {dataset_key}',
                'file_type': dataset_info['file_type'],
                'source_variables': ', '.join(dataset_info['variables']) if dataset_info['variables'] else 'unknown',
                'source_folder': dataset_info['folder_path'],
                'time_period': f"{dataset_info['start_year']}-{dataset_info['end_year']}"
            })
        
        # Create the dataset
        ds = xr.Dataset(data_vars, coords=coords)
        
        # Add global attributes
        ds.attrs.update({
            'title': 'Combined time series data from multiple sources',
            'description': 'Spatially integrated total data from various datasets',
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'Generated by time series analysis script',
            'time_type': time_type,
            'time_range': f"{global_year_min}-{global_year_max}",
            'number_of_datasets': len(all_datasets)
        })
        
        # Save to NetCDF
        ds.to_netcdf(output_path)
        print(f"\n=== Combined NetCDF saved ===")
        print(f"File: {output_path}")
        print(f"Datasets included: {list(all_datasets.keys())}")
        print(f"Time points: {len(reference_time)}")
        print(f"Variables saved: {list(data_vars.keys())}")
        
        return True
        
    except Exception as e:
        print(f"Error saving combined NetCDF file: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_time_series_analysis(folder_data_list, time_analysis_figure_data, annual=False, save_netcdf=False,seasonality=False):
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
       Default is False 
    save_netcdf : bool, optional
        Whether to save NetCDF files
    seasonality : bool, optional
        If True, calculate and plot seasonal statistics instead of time series
        Default is False
    """

    # Create appropriate subplot based on analysis type
    if seasonality:
        fig, time_analysis_axis = plt.subplots(figsize=(12, 6))
        time_analysis_axis.set_xlabel("Month")
        time_analysis_axis.set_ylabel(time_analysis_figure_data["ylabel"])
        time_analysis_axis.set_title(f"Seasonal Analysis - {time_analysis_figure_data['title']}")
        time_analysis_axis.set_xticks(range(1, 13))
        time_analysis_axis.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    else:
        fig, time_analysis_axis = plt.subplots(figsize=(14, 6))  # Wider for time series + legend


    global_year_max = 0
    global_year_min = 9999

    # Make sure the output directory exists before saving files
    output_dir = time_analysis_figure_data['figs_folder']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensuring output directory exists: {output_dir}")

    # Get logmapscale setting from configuration (default to True if not specified)
    logmapscale = time_analysis_figure_data.get('logmapscale', True)
    print(f"Using logmapscale setting: {logmapscale}")

    # Initialize dictionary to store all datasets for combined NetCDF
    all_datasets = {}

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

        print(f"\nProcessing dataset {index+1}/{len(folder_data_list)}: {file_type}")
        print(f"Variables: {variables}")
        print(f"Folder path: {folder_path}")
        print(f"Annual aggregation: {annual}") 
        print(f"Seasonality analysis: {seasonality}")

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
            annual=annual,
            save_netcdf=save_netcdf
        )

        figure_label = f"{figure_data['label']} ({start_year}-{end_year})" 

        print(f"=== DEBUGGING MAP DATA ===")
        print(f"time_mean_data shape: {time_mean_data.shape if hasattr(time_mean_data, 'shape') else 'No shape attr'}")
        print(f"longitude shape: {longitude.shape if hasattr(longitude, 'shape') else 'No shape attr'}")
        print(f"latitude shape: {latitude.shape if hasattr(latitude, 'shape') else 'No shape attr'}")
        print(f"units: {units}")
        print(f"figure_label: {figure_label}")
        print(f"=== END DEBUGGING MAP DATA ===\n")

        #Map plotting functionality
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
            logMap=logmapscale,
        )

        #Plot time series
        if seasonality:
            seasonal_data = calculate_seasonal_statistics(data_per_year_stack)
            seasonal_time_series_plot(
                axis=time_analysis_axis,
                seasonal_data=seasonal_data,
                marker=figure_data["marker"],
                line_style=figure_data["line_style"],
                color=figure_data["color"],
                label=figure_label,
            )
        else:
            #Long time series plotting
            time_series_plot(
                axis=time_analysis_axis,
                data=data_per_year_stack,
                marker=figure_data["marker"],
                line_style=figure_data["line_style"],
                color=figure_data["color"],
                label=figure_label,
            )
        #Update global year range (only for non-seasonal analysis)
        if not seasonality:
            if np.issubdtype(type(data_per_year_stack[0, 0]), np.floating):
                year_max = int(np.ceil(data_per_year_stack[:, 0].max()))
                year_min = int(np.floor(data_per_year_stack[:, 0].min()))
            else:
                year_max = int(end_year)
                year_min = int(start_year)

            global_year_max = max(global_year_max, year_max)
            global_year_min = min(global_year_min, year_min)

        print(f"{time_analysis_figure_data['figs_folder']}/map_figure_{index}")

        #Save individual map figures
        map_figure.savefig(f"{time_analysis_figure_data['figs_folder']}/map_figure_{index}")

        #Store dataset info for NetCDF
        all_datasets[f"{file_type}_{variables}"] = {
            'data': np.sum(data_per_year_stack[:, 1]) if len(data_per_year_stack.shape) > 1 else np.sum(data_per_year_stack),
            'time': list(range(int(start_year), int(end_year) + 1)),
            'units': units,
            'start_year': start_year,
            'end_year': end_year,
            'variables': variables,
            'file_type': file_type,
            'folder_path': folder_path
        }

        #Save combined NetCDF file with all datasets 
        if save_netcdf and all_datasets:
            save_combined_netcdf(all_datasets, output_dir, annual, global_year_min, global_year_max)

        #Set labels for time series (non-seasonal) plots 
        if not seasonality: 
            if annual: 
                xlabel = f"Yearly Data ({global_year_min}-{global_year_max})" 
            else: 
                xlabel = f"Monthly Data ({global_year_min})" 
                time_analysis_axis.set_title(time_analysis_figure_data["title"]) 
                time_analysis_axis.set_xlabel(xlabel) 
                time_analysis_axis.set_ylabel(time_analysis_figure_data["ylabel"])
    
        #Manual legend creation and positioning 
        handles, labels = time_analysis_axis.get_legend_handles_labels() 
        legend = time_analysis_axis.legend( 
                                           handles, labels, 
                                           loc="center left", 
                                           fontsize='medium', 
                                           bbox_to_anchor=(1.02, 0.5),  # Right side, vertically centered 
                                           frameon=True, 
                                           fancybox=True, 
                                           shadow=True) 
        #Layout adjustments 
        plt.tight_layout() 
        if not seasonality: 
            plt.subplots_adjust(right=0.8)  # More space needed for time series legend
        else: 
            plt.subplots_adjust(right=0.85)  # Less space needed for seasonal plots 

        #Save with appropriate filename and bbox_inches='tight' 
        filename_suffix = "_seasonality" if seasonality else "" 
        plt.savefig(f"{time_analysis_figure_data['figs_folder']}/time_analysis_figure{filename_suffix}",
                    bbox_inches='tight', dpi=300, facecolor='white') 

        plt.show()

        # Create difference maps if more than one dataset is provided 
        if len(folder_data_list) > 1: 
            print("\n=== Creating difference maps between datasets ===")

            # Get the first two datasets for comparison
            first_selection = 0
            second_selection = 1

            folder_data_one = folder_data_list[first_selection]
            folder_data_two = folder_data_list[second_selection]

            # Get data for both datasets
            (
                time_mean_data_one,
                data_per_year_stack_one,
                longitude_one,
                latitude_one,
                units_one,
                start_year_one,
                end_year_one,
            ) = obtain_time_series_xarray(
                NetCDF_folder_Path=folder_data_one["folder_path"],
                NetCDF_Type=folder_data_one["file_type"],
                variables=folder_data_one["variables"],
                annual=annual,
                save_netcdf=False
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
                NetCDF_folder_Path=folder_data_two["folder_path"],
                NetCDF_Type=folder_data_two["file_type"],
                variables=folder_data_two["variables"],
                annual=annual,
                save_netcdf=False
            )

            # Check if datasets are compatible for difference calculation
            if (time_mean_data_one.shape == time_mean_data_two.shape and
                np.array_equal(longitude_one, longitude_two) and
                np.array_equal(latitude_one, latitude_two)):

                # Calculate difference
                time_mean_data_diff = time_mean_data_one - time_mean_data_two
                longitude_diff = longitude_one
                latitude_diff = latitude_one

                # Create difference map
                map_figure, map_axis = plt.subplots(
                    nrows=1,
                    ncols=1,
                    figsize=(18, 10),
                    subplot_kw={"projection": ccrs.PlateCarree()},
                )

                figure_label_diff = f"Difference: {folder_data_one['figure_data']['label']} - {folder_data_two['figure_data']['label']}"
                units_diff = units_one  # Assuming same units

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


                map_figure.savefig( 
                    f"{time_analysis_figure_data['figs_folder']}/figure{first_selection}_and_figure{second_selection}_diff_map"
                                   )

                print(f"Difference map saved as: figure{first_selection}_and_figure{second_selection}_diff_map") 
            else: 
                print("Warning: Datasets have incompatible dimensions for difference calculation")
                print(f"Dataset 1 shape: {time_mean_data_one.shape}") 
                print(f"Dataset 2 shape: {time_mean_data_two.shape}")

def calculate_seasonal_statistics(data_per_year_stack):
    """
    Calculate seasonal statistics (mean of all Januarys, Februarys, etc.)
    
    Parameters:
    -----------
    data_per_year_stack : numpy.ndarray
        Array with columns [year, month, values] or [decimal_year, values]
    
    Returns:
    --------
    seasonal_data : numpy.ndarray
        Array with columns [month (1-12), mean_value, std_value]
    """
    import pandas as pd
    
    # Convert to DataFrame for easier manipulation
    if data_per_year_stack.shape[1] == 3:  # [year, month, values]
        df = pd.DataFrame(data_per_year_stack, columns=['year', 'month', 'value'])
    else:  # [decimal_year, values] - need to extract month
        df = pd.DataFrame(data_per_year_stack, columns=['decimal_year', 'value'])
        df['month'] = ((df['decimal_year'] % 1) * 12 + 1).round().astype(int)
        df['month'] = df['month'].clip(1, 12)  # Ensure months are 1-12
    
    # Calculate seasonal statistics
    seasonal_stats = df.groupby('month')['value'].agg(['mean', 'std']).reset_index()
    
    # Return as numpy array [month, mean, std]
    seasonal_data = seasonal_stats.values
    return seasonal_data


def seasonal_time_series_plot(axis, seasonal_data, marker, line_style, color, label):
    """
    Plot seasonal statistics
    
    Parameters:
    -----------
    axis : matplotlib axis
        The axis to plot on
    seasonal_data : numpy.ndarray
        Array with columns [month, mean_value, std_value]
    marker, line_style, color, label : str
        Plot formatting parameters
    """
    months = seasonal_data[:, 0]
    means = seasonal_data[:, 1]
    stds = seasonal_data[:, 2] if seasonal_data.shape[1] > 2 else None
    
    # Plot the seasonal means
    axis.plot(months, means, marker=marker, linestyle=line_style, color=color, label=label)
    
    # Optionally add error bars for standard deviation
    if stds is not None and not np.isnan(stds).all():
        axis.fill_between(months, means - stds, means + stds, alpha=0.2, color=color)
    
    axis.grid(True)

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
    try:
        # Extract data from folder dictionaries
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
        print(f"Getting data for dataset 1: {file_type_one}")
        result_one = obtain_time_series_xarray(
            NetCDF_folder_Path=folder_path_one,
            NetCDF_Type=file_type_one,
            variables=variables_one,
            annual=annual
        )

        if result_one[0] is None:
            print(f"Error: Failed to load dataset 1: {file_type_one}")
            return None, None, None, None, None, None, None, None

        (time_mean_data_one, data_per_year_stack_one, longitude_one,
         latitude_one, units_one, start_year_one, end_year_one) = result_one

        print(f"Getting data for dataset 2: {file_type_two}")
        result_two = obtain_time_series_xarray(
            NetCDF_folder_Path=folder_path_two,
            NetCDF_Type=file_type_two,
            variables=variables_two,
            annual=annual
        )

        if result_two[0] is None:
            print(f"Error: Failed to load dataset 2: {file_type_two}")
            return None, None, None, None, None, None, None, None

        (time_mean_data_two, data_per_year_stack_two, longitude_two,
         latitude_two, units_two, start_year_two, end_year_two) = result_two

        # Check if units are compatible for subtraction
        if units_one != units_two:
            print(f"WARNING: Units do not match! {units_one} vs {units_two}")
            print("Cannot perform subtraction with different units.")
            print("Continuing with calculation, but results may not be meaningful.")

        # Check spatial grid compatibility
        if (time_mean_data_one.shape != time_mean_data_two.shape):
            print(f"WARNING: Spatial grids do not match!")
            print(f"Dataset 1 shape: {time_mean_data_one.shape}")
            print(f"Dataset 2 shape: {time_mean_data_two.shape}")
            print("Cannot perform spatial subtraction with different grids.")
            return None, None, None, None, None, None, None, None

        # Calculate the difference in spatial means
        time_mean_data_diff = time_mean_data_one.copy()
        time_mean_data_diff.values = time_mean_data_one.values - time_mean_data_two.values
    
        # Calculate the difference in time series data
        print(f"Shape of data_per_year_stack_two: {data_per_year_stack_two.shape}")
        print(f"Shape of data_per_year_stack_one: {data_per_year_stack_one.shape}")

                # Extract time arrays
        time_one = data_per_year_stack_one[:, 0]
        time_two = data_per_year_stack_two[:, 0]

        # Find common time points with tolerance for floating point comparison
        tolerance = 1e-6
        common_times = []

        for t1 in time_one:
            # Check if this time exists in time_two within tolerance
            matches = np.abs(time_two - t1) < tolerance
            if np.any(matches):
                common_times.append(t1)

        common_times = np.array(common_times)

        if len(common_times) == 0:
            print("ERROR: No common time points between datasets! Cannot calculate difference.")
            return None, None, None, None, None, None, None, None

        print(f"Found {len(common_times)} common time points")

        # Create difference array for common time points
        data_per_year_stack_diff = np.zeros((len(common_times), 2))
        data_per_year_stack_diff[:, 0] = common_times

        for i, t in enumerate(common_times):
            # Find indices for this time point in both datasets
            idx1 = np.argmin(np.abs(time_one - t))
            idx2 = np.argmin(np.abs(time_two - t))

            # Calculate difference
            data_per_year_stack_diff[i, 1] = (data_per_year_stack_one[idx1, 1] -
                                            data_per_year_stack_two[idx2, 1])

        # Determine year range from common times
        start_year_diff = int(np.floor(common_times.min()))
        end_year_diff = int(np.ceil(common_times.max()))

        # Create a descriptive figure label
        figure_label = f"{figure_data_one['label']} - {figure_data_two['label']} ({start_year_diff}-{end_year_diff})"

        print(f"Difference calculation completed successfully")
        print(f"Time range: {start_year_diff} - {end_year_diff}")
        print(f"Units: {units_one}")

        return (
            time_mean_data_diff,
            data_per_year_stack_diff,
            longitude_one,  # Use coordinates from first dataset
            latitude_one,
            units_one,      # Use units from first dataset
            start_year_diff,
            end_year_diff,
            figure_label,
        )

    except Exception as e:
        print(f"Error in run_time_series_diff_analysis: {str(e)}")
        print(f"Error details: {traceback.format_exc()}")
        return None, None, None, None, None, None, None, None
