import traceback
import rasterio
from os import listdir, makedirs, remove
from os.path import isfile, join, basename, exists, dirname
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


def calculate_grid_area(grid_area_shape, units="km"):
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


def read_gfed5(files, upscaled=False, shape=(720, 1440), variable_name="Total"):
    """
    Reads multiple HDF5 files using h5py, calculates the annual burned area,
    and returns the data as xarray.DataArray.
    """
    time_array = [2002]
    yearly_data = []
    attribute_dict = {}
    annual_burned_fraction = np.zeros(shape=(shape))
    for file in files:
        with Dataset(file) as netcdf_dataset:
            # dataset containing all xarray data array (used to create the final netcdf file)
            match variable_name:
                # calculates the Nat array
                case "Nat":
                    # transform the arrays dimensions to (720, 1440) and convert (km^2 -> m^2)
                    # obtain all needed data array
                    var_total_data_array = netcdf_dataset.variables["Total"][:][0]
                    var_crop_data_array = netcdf_dataset.variables["Crop"][:][0]
                    var_defo_data_array = netcdf_dataset.variables["Defo"][:][0]
                    var_peat_data_array = netcdf_dataset.variables["Peat"][:][0]
                    # calculate the Nat numpy array
                    # equation: Total - (Crop + Defo + Peat)
                    var_data_array = var_total_data_array - (
                        var_crop_data_array + var_defo_data_array + var_peat_data_array
                    )
                # base case
                case _:
                    # obtain the variables in the netcdf_dataset
                    # dimensions (1, 720, 1440)
                    var_total_data_array = netcdf_dataset.variables["Total"][:]
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
            annual_burned_fraction += var_data_array

            # Copy attributes of the burned area fraction
            for attr_name in netcdf_dataset.variables["Total"].ncattrs():
                attribute_dict[attr_name] = getattr(
                    netcdf_dataset.variables["Total"], attr_name
                )

            # update the units to match the upscaling process
            attribute_dict["units"] = "m^2"

            # obtain the height and width from the upscale shape
            # create an evenly spaced array representing the longitude and the latitude
            height, width = var_data_array.shape
            latitudes = np.linspace(-90, 90, height)
            longitudes = np.linspace(-180, 180, width)
            year = int(file.split("\\")[-1][2:6])

            if len(time_array) and year != time_array[-1]:
                yearly_data.append(annual_burned_fraction)
                time_array.append(year)
                annual_burned_fraction = np.zeros(shape=shape)

            # flip the data matrix (upside down due to the GFED dataset's orientation)
            # burned_fraction_upscaled = np.flip(burned_fraction_upscaled, 0)

            # create the xarray data array for the upscaled burned area and add it to the dictionary

    # height, width = yearly_data[0].shape
    #     time_array.append(year)
    if len(yearly_data) < len(time_array):
        yearly_data.append(annual_burned_fraction)

    print(len(yearly_data))
    print(time_array)
    latitudes = np.linspace(-90, 90, shape[-2])
    longitudes = np.linspace(-180, 180, shape[-1])
    total_data_array = (
        xr.DataArray(
            yearly_data,
            coords={
                "time": np.array(time_array),
                "latitude": latitudes,
                "longitude": longitudes,
            },
            dims=["time", "latitude", "longitude"],
            attrs=attribute_dict,
        )
        if upscaled
        else xr.DataArray(
            yearly_data,
            coords={
                "time": np.array(time_array),
                "latitude": latitudes,
                "longitude": longitudes,
            },
            dims=["time", "latitude", "longitude"],
            attrs=attribute_dict,
        )
    )

    return total_data_array, longitudes, latitudes


def read_gfed4s(files, upscaled=False, shape=(720, 1440)):
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

            # Sum burned fraction over all months
            annual_burned_fraction = np.zeros(shape=shape)
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
            else file_path.split("_")[2]
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


def read_ModelE(files, variables=["BA_tree", "BA_shrub", "BA_grass"], monthly=False):
    """
    Reads ModelE BA data (BA_tree, BA_shrub, BA_grass) for the given year range, sums them to calculate
    modelE_BA, and returns the annual sum for each year.

    Parameters:
    startyear (int): The starting year.
    endyear (int): The ending year.
    simname (str): The simulation name to match the file format (default 'nk_CCycle_E6obioF40').
    ModelE_path (str): The directory containing ModelE output.

    Returns:
    np.ndarray: A 2D array (year, modelE_BA), where modelE_BA is the sum of BA_tree, BA_shrub, and BA_grass.
    """

    # Initialized a litst to store each year's dataset
    datasets = []
    zero_mat = np.zeros((90, 144), dtype=float)

    # Goal: calculate annual total instead of ANN files
    # for loop over years available within file_path (e.g. there are monthly files JAN2000-DEC2012)
    # so years are 2000-2012
    # for loop over JAN-DEC
    year_dictionary = {}
    # Loop over each file and process it
    for file_path in files:
        # print(file_path)
        ds = xr.open_dataset(file_path)
        attribute_dict = {}
        # Add a time coordinate based on the year from the file name
        # year = int(file_path.split("ANN")[1][:4])
        # TO DO: instead of hard codeing the 90,144 in the shape of modelE_var_data
        # can you read it from the model file?
        # this is important because the next version of the model will be 180x360 so this script
        # will fail with it.
        modelE_var_data = np.zeros(shape=(90, 144))
        for variable in variables:
            # where function replaces values that do not meet the parameters condition
            # (replaces all values that are not greater than 0)
            var_data = ds[variable].where(ds[variable] > 0.0, 0.0)

            modelE_var_data = modelE_var_data + var_data

            for attr_name in ds[variable].attrs:
                attribute_dict[attr_name] = getattr(ds[variable], attr_name)

            modelE_var_data *= SECONDS_IN_A_YEAR

        year = (
            int(file_path.split(".")[1][-4:])
            if monthly
            else int(file_path.split("ANN")[1][:4])
        )
        if year in year_dictionary:
            year_dictionary[year] += modelE_var_data
        else:
            year_dictionary[year] = modelE_var_data

        modelE_var_data = modelE_var_data.expand_dims(
            time=[year]
        )  # Add time dimension for each year

        # Append the processes dataset to the list
        datasets.append(modelE_var_data)
    year_dictionary = dict(sorted(year_dictionary.items()))
    # Concatenate all datasets along the 'time' dimension
    for year in year_dictionary.keys():
        year_dictionary[year] = year_dictionary[year].expand_dims(time=[year])
    modelE_all_year = xr.concat(year_dictionary.values(), dim="time")
    attribute_dict["units"] = "1.e+10 flashes/m2/yr"
    modelE_all_year.attrs = attribute_dict
    modelE_lons = ds["lon"]
    modelE_lats = ds["lat"]
    return modelE_all_year, modelE_lons, modelE_lats


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
            yearly_var_data_array = []
            year_time_data = []
            updated_var_data_array = []

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
            variable_data = np.zeros(shape=(density_variable_data[0].shape))
            year = int(start_date.split("-")[0])
            for month in range(len(density_variable_data)):
                current_year = int(str(date_range[month]).split("-")[0])
                # change to upscaled checks if the data is already upscaled
                if upscaled:
                    var_data_array = density_variable_data[month]
                # if the data is not upscaled preform further calculations
                else:
                    # var_data_array = density_variable[:][month]
                    var_data_array = density_variable_data[month] / DAYS_TO_SECONDS
                variable_data = variable_data + var_data_array
                if (year) < (current_year):
                    year = current_year
                    year_time_data.append(str(year))
                    yearly_var_data_array.append(variable_data)
                    variable_data = np.zeros(shape=(density_variable_data[0].shape))
                # print(f"Current Month {month}: ", var_data_array.sum())
                updated_var_data_array.append(var_data_array)

            attribute_dict["units"] = "lightning strikes/m-2/year"
            latitudes = np.linspace(-90, 90, density_variable.shape[-2])
            longitudes = np.linspace(-180, 180, density_variable.shape[-1])
            # creates the data array and saves it to a file
            var_data_array_xarray = xr.DataArray(
                (updated_var_data_array),
                coords={
                    "time": time_data_array,
                    "latitude": latitudes,
                    "longitude": longitudes,
                },
                dims=["time", "latitude", "longitude"],
                attrs=attribute_dict,
            )

            yearly_var_data_array_xarray = xr.DataArray(
                (yearly_var_data_array),
                coords={
                    "time": year_time_data,
                    "latitude": latitudes,
                    "longitude": longitudes,
                },
                dims=["time", "latitude", "longitude"],
                attrs=attribute_dict,
            )
            var_data_array_xarray = var_data_array_xarray.where(
                var_data_array_xarray > 0.0, 0.0
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
):
    masx = 0.7 * decade_data.max() if masx == None else masx
    # labelpad sets the distance of the colorbar from the map
    """Define the properties of a subplot with optional difference normalization."""
    ax.coastlines(color="black")
    ax.add_feature(cfeature.LAND, edgecolor="gray")
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=1)

    ax.set_title(title, fontsize=10, pad=1)
    props = dict(boxstyle="round", facecolor="lightgray", alpha=0.5)
    (
        (
            ax.text(
                0.5,
                1.07,
                f"Global Total: {glob}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                bbox=props,
                fontsize=10,
            )
        )
        if glob
        else None
    )

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
        norm = None
        # Mask values less than or equal to zero for the custom colormap (set to white)
        # masked_data = np.ma.masked_less_equal(data, 0)  # Mask values <= 0
        # # Create a colormap with white for values <= 0
        # cmap = plt.get_cmap(cmap).copy()
        # cmap.set_bad(color="white")  # Set masked values to white
        logNorm = mcolors.LogNorm(
            vmin=1 if not is_diff else None, vmax=masx if not is_diff else None
        )
        p = ax.pcolormesh(
            lons,
            lats,
            decade_data,
            transform=ccrs.PlateCarree(),
            cmap="jet",
            norm=logNorm,
            # vmin=0 if not is_diff else None,
            # vmax=masx if not is_diff else None,
        )
        #         print(0 if not is_diff else None, masx if not is_diff else None)
        # logNorm = colors.LogNorm(
        #     vmin=0 if not is_diff else None, vmax=masx if not is_diff else None
        # )
        # p = ax.pcolormesh(
        #     lons,
        #     lats,
        #     decade_data,
        #     transform=ccrs.PlateCarree(),
        #     cmap=cmap,
        #     norm=logNorm,
        #     vmin=float(0) if not is_diff else None,
        #     vmax=float(masx) if not is_diff else None,
        # )

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
    cbarmac,
    is_diff=False,
):
    """
    Plots the decadal mean burned area of both GFED and ModelE side by side.

    Parameters:
    decade_mean_gfed4sba (xarray.DataArray): The decadal mean burned area (lat, lon array).
    decade_mean_modelEba (xarray.DataArray): The decadal mean burned area from ModelE(lat, lon array).
    """
    print(axis_index, axis_length)

    axis_value = axis if axis_length <= 1 else axis[axis_index]
    # GFED4s decadal mean map
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
        masx=cbarmac,
        is_diff=is_diff,
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
    Plots the total burned area as a function of year for both GFED and ModelE data.

    Parameters:
    gfed4sba_per_year (np.ndarray): A 2D array with columns (year, totalBA), where totalBA is the sum of burned area for that year.
    modelE_BA_per_year (np.ndarray): A 2D array with columns (year, modelE_BA), where modelE_BA is the sum of burned area for that year.
    """

    # try:
    # Extract years and total burned area for both GFED and ModelE
    years_data = data[:, 0]
    total_data = data[:, 1]

    # Plot the time series of total burned area for both GFED and ModelE
    axis.plot(
        years_data,
        total_data,
        marker=marker,
        linestyle=line_style,
        color=color,
        label=label,
    )
    axis.legend()
    axis.grid(grid_visible)
    # except:
    #     print("title, xlabel...etc already set")


def handle_time_extraction_type(file_paths, variables, NetCDF_Type):
    match (NetCDF_Type):
        case "BA_GFED4":
            total_value, longitude, latitude = read_gfed4s(
                files=file_paths, upscaled=False
            )
        case "BA_GFED4_upscale":
            total_value, longitude, latitude = read_gfed4s(
                files=file_paths, upscaled=True, shape=(90, 144)
            )
        case "BA_GFED5":
            total_value, longitude, latitude = read_gfed5(
                files=file_paths, shape=(720, 1440), upscaled=False
            )
        case "BA_GFED5_upscale":
            total_value, longitude, latitude = read_gfed5(
                files=file_paths, shape=(90, 144), upscaled=True
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
        case _:
            print("[-] No Parsing Script Found For", NetCDF_Type)
    return (total_value, longitude, latitude)


def obtain_time_series_xarray(
    variables,
    NetCDF_folder_Path,
    NetCDF_Type,
):
    """
    Calculates the decade mean burned area (BA) and the interannual variability of BA
    from 2002 to 2012 using read_gfed4s and read_ModelEBA.

    Parameters:
    startyear (int): The start year of the period (default 2002).
    endyear (int): The end year of the period (default 2012).
    GFED_path (str): The directory containing the GFED4s files.
    ModelE_path (str): The directory containing ModelE output.
    simname (str): The simulation name for ModelE data.

    Returns:
    tuple: A tuple containing:
      - decade_mean_gfed4sba (xarray.DataArray): The mean burned area over the decade (lat, lon array).
      - gfed4sba_per_year (np.ndarray): A 2D array with columns (year, totalBA), where totalBA is the sum of burned area for that year.
      - modelE_BA_per_year (np.ndarray): A 2D array with columns (year, modelE_BA), where modelE_BA is the total burned area from ModelE.
    """

    # Call read_gfed4s to load GFED4s data
    file_paths = obtain_netcdf_files(NetCDF_folder_Path)
    total_value, longitude, latitude = handle_time_extraction_type(
        file_paths=file_paths, variables=variables, NetCDF_Type=NetCDF_Type
    )

    time_dimension = total_value.dims[0]
    sum_dimensions = (total_value.dims[-2], total_value.dims[-1])
    # Calculate the mean burned area over the decade
    time_mean_data = total_value.mean(dim=time_dimension)

    # Calculate total burned area for each year from GFED4s data
    # this is fine for BA which is in units of m^2 or km^2, but is not OK
    # for density variables like flash density or pollutant concentration
    # which is reported in #/m^2 or #/km^2
    # To fix for all variables that have an area unit dependancy in the units:
    # (total_value*area_matrix).sum(dim=sum_dimension).values
    # but make the units need to be revised accordingly and are not area dependant any more
    # if data_units string includes /m^2 or m^-2
    # somethign like this: match = re.match("m^-2"or"/m^2 ", units) then:
    # (total_value*area_matrix).sum(dim=sum_dimension).values
    # else
    # total_data_array = total_value.sum(dim=sum_dimension).values

    units = total_value.attrs["units"]
    # For model E data display on the figure weather the scaling factor has been multiplied
    # Calculate climatological total over the time dimension
    time_total_data = total_value.sum(dim=time_dimension)
    if "m2" in units.lower() or "m^2".lower() in units:
        # Calculate the lat-lon total over the climatological period
        total_data_array = total_value.sum(dim=sum_dimensions).values
        # Convert m^2 to mega hectors
        total_data_array = total_data_array
        # print(f"Multiplied the M2TOMHA for {NetCDF_Type}")
    elif "m-2".lower() or "m^-2".lower() in units:
        # Calculate the lat-lon total over the climatological period
        grid_cell_dimension_shape = (total_value.shape[-2], total_value.shape[-1])
        grid_cell_area = calculate_grid_area(
            grid_area_shape=grid_cell_dimension_shape, units="m^2"
        )
        time_total_data *= grid_cell_area
        total_data_array = (total_value * grid_cell_area).sum(dim=sum_dimensions).values
        print("Data Array multiplied by grid_cell_area")
    # Review cases that work for fractions
    else:
        total_data_array = total_value.sum(dim=sum_dimensions).values

    print(total_value.coords["time"])
    start_year = int(total_value.coords["time"].values[0])
    end_year = int(total_value.coords["time"].values[-1])
    print(start_year, end_year)
    years = np.arange(start_year, end_year + 1)
    data_per_year_stack = np.column_stack((years, total_data_array))

    return (
        time_total_data,
        data_per_year_stack,
        longitude,
        latitude,
        units,
        start_year,
        end_year,
    )


def run_time_series_analysis(folder_data_list, time_analysis_figure_data):
    # Plot side by side maps for GFED and ModelE
    _, time_analysis_axis = plt.subplots(figsize=(10, 6))

    global_year_max = 0
    global_year_min = 9999
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

        # Call intann_BA_xarray to calculate decadal mean BA and interannual variability
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
        )

        figure_label = f"{figure_data['label']} ({start_year}-{end_year})"
        # Plot the decadal mean burned area
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
            cbarmac=figure_data["cbarmac"],
        )

        # Plot the time series of burned area for GFED and ModelE
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

    # map_figure, map_axis = plt.subplots(
    #     nrows=1,
    #     ncols=1,
    #     figsize=(18, 10),
    #     subplot_kw={"projection": ccrs.PlateCarree()},
    # )
    # (
    #     (time_mean_data_diff),
    #     (data_per_year_stack_diff),
    #     longitude_diff,
    #     latitude_diff,
    #     units_diff,
    #     start_year_diff,
    #     end_year_diff,
    #     figure_label_diff,
    # ) = run_time_series_diff_analysis(folder_data_list[0], folder_data_list[1])

    # map_plot(
    #     figure=map_figure,
    #     axis=map_axis,
    #     axis_length=1,
    #     axis_index=0,
    #     decade_data=time_mean_data_diff,
    #     longitude=longitude_diff,
    #     latitude=latitude_diff,
    #     subplot_title=figure_label_diff,
    #     units=units_diff,
    #     cbarmac=None,
    #     is_diff=True,
    # )

    # time_series_plot(
    #     axis=time_analysis_axis,
    #     data=data_per_year_stack_diff,
    #     marker="o",
    #     line_style="-",
    #     color="r",
    #     label=figure_label_diff,
    # )

    time_analysis_axis.set_title(time_analysis_figure_data["title"])
    time_analysis_axis.set_xlabel(
        f"{time_analysis_figure_data['xlabel']} ({global_year_min}-{global_year_max})"
    )
    time_analysis_axis.set_ylabel(time_analysis_figure_data["ylabel"])
    plt.show()


def run_time_series_diff_analysis(folder_data_one, folder_data_two):
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

    # Call intann_BA_xarray to calculate decadal mean BA and interannual variability
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
    # print(time_mean_data_one.values)
    # print(time_mean_data_two.values)
    # print(data_per_year_stack_two - data_per_year_stack_one)
    time_mean_data_one.values = time_mean_data_one.values - time_mean_data_two.values
    data_per_year_stack_diff = data_per_year_stack_one - data_per_year_stack_two[:9]
    min_year = min(
        data_per_year_stack_one[:, 0].min(), data_per_year_stack_two[:, 0].min()
    )

    for index in range(0, len(data_per_year_stack_diff)):
        data_per_year_stack_diff[index][0] += min_year + index

    return (
        (time_mean_data_one),
        (data_per_year_stack_diff),
        longitude_one,
        latitude_one,
        "Burned Area m^2",
        start_year_one,
        end_year_one,
        f"{figure_data_one['label']} - {figure_data_two['label']}",
    )
    pass
