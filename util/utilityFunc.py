from os import listdir
import os
from os.path import join, isfile
import re
import warnings
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

from utilityGlobal import (
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
    KM_TO_METERS,
    DAYS_TO_SECONDS,
    EARTH_RADIUS,
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


######################################################
#                    TIME ANALYSIS                   #
######################################################
def plotTimeAnalysis(
    data_set,
    directory_path,
    year_start,
    year_end,
    species,
    legend_array,
    color_array,
    figure_size,
):
    # iterate over species in the species list
    for species_element in species:
        # create the matplotlib figure
        plt.figure(figsize=figure_size)
        # plot values
        for legend_index, legend in enumerate(legend_array):
            plt.plot(
                MONTHLIST,
                data_set[legend_index, :],
                label=legend,
                marker=MARKER,
                color=color_array[legend_index],
            )
        # include various metadata for the created plot
        plt.title(f"{species_element} Emissions by Sector ({year_start} - {year_end})")
        plt.xlabel("Month")
        plt.ylabel(f"{species_element} Emissions [Pg]")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"{directory_path}/plots/fire_repository/Develpment/{species_element}_emissions_by_sector.eps"
        )


def executeTimeAnalysis(
    file_path,
    species,
    sectors,
    simulations,
    directory_path,
    area_variable_name,
):
    # obtain the dataset earth surface area data
    time_netcdf_dataset = nc.Dataset(file_path, "r")
    dataset_area = time_netcdf_dataset.variables[area_variable_name]
    dataset_earth_surface_area = np.sum(dataset_area)
    # This script plots a time series of one year of data (seasonality)
    # of specified emissions sources from two simulations and also calculates
    # the difference between the two
    dest_data = np.zeros((len(sectors) * len(simulations), NUM_MONTHS))
    for species_element in species:
        for month_index, month in enumerate(MONTHLIST):
            row_index = 0
            for simulation_element in simulations:
                # Construct file name
                filename = f"{directory_path}/{species_element}/{month}_1996.taij{simulation_element}.nc"
                try:
                    parseDataset = nc.Dataset(filename, "r")
                    for sector in enumerate(sectors):
                        if (
                            sector != "pyrE_src_hemis"
                            and simulation_element != "E6TomaF40intpyrEtest2"
                        ):
                            var_name = f"{species_element}_{sector}"
                            hemisphere_value = parseDataset.variables[var_name]
                            global_val = hemisphere_value[2,]
                            dest_data[row_index, month_index] = (
                                global_val
                                * dataset_earth_surface_area
                                * SECONDS_IN_A_YEAR
                                * KILOGRAMS_TO_GRAMS
                            )
                            row_index += 1
                    parseDataset.close()
                except FileNotFoundError:
                    print(f"File {filename} not found.")
                except Exception as e:
                    print(f"Error reading from {filename}: {str(e)}")
            dest_data[-1, month_index] = (
                dest_data[0, month_index] + dest_data[1, month_index]
            )
    return dest_data


def timeAnalysisRunner(
    file_path,
    species,
    sectors,
    simulations,
    directory_path,
    area_variable_name,
    year_start,
    year_end,
    legend_array,
    color_array,
    figure_size,
):
    data_set = executeTimeAnalysis(
        file_path,
        species,
        sectors,
        simulations,
        directory_path,
        area_variable_name,
    )
    plotTimeAnalysis(
        data_set,
        directory_path,
        year_start,
        year_end,
        species,
        legend_array,
        color_array,
        figure_size,
    )


def time_series_plot(
    axis,
    data,
    marker,
    line_style,
    color,
    label,
    title,
    x_label,
    y_label,
    grid_visible=True,
):
    """
    Plots the total burned area as a function of year for both GFED and ModelE data.

    Parameters:
    gfed4sba_per_year (np.ndarray): A 2D array with columns (year, totalBA), where totalBA is the sum of burned area for that year.
    modelE_BA_per_year (np.ndarray): A 2D array with columns (year, modelE_BA), where modelE_BA is the sum of burned area for that year.
    """

    try:
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
        axis.title(title)
        axis.xlabel(x_label)
        axis.ylabel(y_label)
        axis.legend()
        axis.grid(grid_visible)
    except:
        print("title, xlabel...etc already set")


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


def calculate_grid_area(grid_area_shape):
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

        # Calculate the surface area of the grid cell at this latitude
        area = (
            (EARTH_RADIUS**2)
            * lon_step_rad
            * (np.sin(lat_rad + lat_step_rad / 2) - np.sin(lat_rad - lat_step_rad / 2))
        )

        # Assign the area to all longitude cells for this latitude band
        grid_area[i, :] = area

    # Display the grid area matrix
    return grid_area


def read_gfed4s(files, upscale=False, shape=(720, 1440)):
    """
    Reads multiple HDF5 files using h5py, calculates the annual burned area,
    and returns the data as xarray.DataArray.
    """
    burned_fraction_list = []

    for file_path in files:
        # Open the HDF5 file using h5py
        with h5py.File(file_path, "r") as h5file:
            # Load lat and lon for constructing the xarray dataset
            lat = h5file["lat"][:]
            lon = h5file["lon"][:]

            # Sum burned fraction over all months
            annual_burned_fraction = np.zeros(shape=shape)
            for month in range(1, 13):
                month_burned_fraction = (
                    h5file[f"burned_area/{month:02d}/burned_fraction"][:]
                    if not upscale
                    else h5file[f"burned_areas_{month:02d}"][:]
                )
                annual_burned_fraction += month_burned_fraction
            if not upscale:
                # Access grid_cell_area using the method suggested
                grid_cell_area = h5file["ancill"]["grid_cell_area"][:]
                # Calculate total burned area
                total_burned_area = annual_burned_fraction * grid_cell_area
            else:
                total_burned_area = annual_burned_fraction
            burned_fraction_list.append(total_burned_area)

    # Convert the list to xarray.DataArray for further processing
    total_burned_area_all_years = (
        xr.DataArray(
            burned_fraction_list,
            dims=["time", "phony_dim_0", "phony_dim_1"],
            coords={
                "lat": (["phony_dim_0", "phony_dim_1"], lat),
                "lon": (["phony_dim_0", "phony_dim_1"], lon),
            },
        )
        if not upscale
        else xr.DataArray(
            burned_fraction_list,
            dims=["time", "lat", "lon"],
            coords={"lat": lat, "lon": lon},
        )
    )

    return total_burned_area_all_years, lon, lat


def read_ModelE(files, variables=["BA_tree", "BA_shrub", "BA_grass"], lightning=False):
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

    # Loop over each file and process it
    for file_path in files:
        print(file_path)
        ds = xr.open_dataset(file_path)
        attribute_dict = {}

        modelE_var_data = np.zeros(shape=(90, 144))
        for variable in variables:
            var_data = ds[variable].where(ds[variable] < 0.0, 0.0)

            modelE_var_data = modelE_var_data + var_data

            for attr_name in ds[variable].attrs:
                attribute_dict[attr_name] = getattr(ds[variable], attr_name)

        if lightning:
            # !! extract units using extract_scaling_factor (had issues extracting scale factor)
            try:
                units_string = attribute_dict["units"]
                print(units_string)
                modelE_var_data = modelE_var_data * (10**-10)
            except:
                pass

        # Add a time coordinate based on the year from the file name
        year = int(file_path.split("ANN")[1][:4])
        modelE_var_data = modelE_var_data.expand_dims(
            time=[year]
        )  # Add time dimension for each year

        # Append the processes dataset to the list
        datasets.append(modelE_var_data)
    # Concatenate all datasets along the 'time' dimension
    modelE_all_year = xr.concat(datasets, dim="time")
    modelE_all_year.attrs = attribute_dict
    print(modelE_all_year.attrs)
    modelE_lons = ds["lon"]
    modelE_lats = ds["lat"]

    return modelE_all_year, modelE_lons, modelE_lats


def read_lightning_data(files, upscale=False):
    """
    Reads multiple lightning files using h5py, calculates the annual burned area,
    and returns the data as xarray.DataArray.
    """
    for file in files:
        with nc.Dataset(file) as netcdf_dataset:
            # dataset containing all xarray data array (used to create the final netcdf file)
            dataset_dict = {}
            attribute_dict = {}
            updated_var_data_array = []

            # update the units to match the upscaling process
            density_variable = netcdf_dataset.variables["density"]
            time_data_array = netcdf_dataset.variables["time"][:]
            # Copy attributes of the burned area fraction
            for attr_name in density_variable.ncattrs():
                attribute_dict[attr_name] = getattr(density_variable, attr_name)

            for month in range(len(density_variable[:])):
                # change to upscaled checks if the data is already upscaled
                if upscale:
                    var_data_array = density_variable[:][month]
                # if the data is not upscaled preform further calculations
                else:
                    # var_data_array = density_variable[:][month]
                    var_data_array = (
                        (density_variable[:][month])
                        * KM_NEG_2TOM_NEG_2
                        / DAYS_TO_SECONDS
                    )

                print(f"density_month_{(month + 1)}")
                updated_var_data_array.append(var_data_array)

            latitudes = np.linspace(-90, 90, density_variable.shape[-2])
            longitudes = np.linspace(-180, 180, density_variable.shape[-1])

            print(attribute_dict)
            attribute_dict["units"] = "lightning strikes/m^2/s"
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

            return var_data_array_xarray, longitudes, latitudes


def define_subplot(
    fig,
    ax,
    data,
    lons,
    lats,
    cmap,
    cborientation,
    fraction,
    pad,
    labelpad,
    fontsize,
    title,
    clabel,
    masx,
    is_diff=False,
    glob=None,
):
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
        data_min, data_max = data.min(), data.max()
        if data_min == data_max:
            norm = mcolors.Normalize(vmin=data_min - 1, vmax=data_max + 1)
        else:
            abs_max = max(abs(0.25 * data_min), abs(0.25 * data_max))
            norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
        p = ax.pcolormesh(
            lons,
            lats,
            data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            vmin=0 if not is_diff else None,
            vmax=masx if not is_diff else None,
        )
    else:
        norm = None
        # Mask values less than or equal to zero for the custom colormap (set to white)
        masked_data = np.ma.masked_less_equal(data, 0)  # Mask values <= 0
        # Create a colormap with white for values <= 0
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad(color="white")  # Set masked values to white
        p = ax.pcolormesh(
            lons,
            lats,
            masked_data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            vmin=0 if not is_diff else None,
            vmax=masx if not is_diff else None,
        )

    cbar = fig.colorbar(p, ax=ax, orientation=cborientation, fraction=fraction, pad=pad)
    cbar.set_label(f"{clabel}", labelpad=labelpad, fontsize=fontsize)

    return ax


def map_plot(
    figure,
    axis,
    axis_length,
    axis_index,
    decade_mean_data,
    longitude,
    latitude,
    subplot_title,
    units,
):
    """
    Plots the decadal mean burned area of both GFED and ModelE side by side.

    Parameters:
    decade_mean_gfed4sba (xarray.DataArray): The decadal mean burned area (lat, lon array).
    decade_mean_modelEba (xarray.DataArray): The decadal mean burned area from ModelE(lat, lon array).
    """

    axis_value = axis if axis_length <= 1 else axis[axis_index]
    # GFED4s decadal mean map
    define_subplot(
        figure,
        axis_value,
        decade_mean_data,
        longitude,
        latitude,
        cmap="jet",
        cborientation="horizontal",
        fraction=0.05,
        pad=0.005,
        labelpad=0.5,
        fontsize=10,
        title=subplot_title,
        clabel=units,
        masx=0.7 * decade_mean_data.max(),
        is_diff=True,
    )


def handle_time_extraction_type(file_paths, variables, NetCDF_Type):
    match (NetCDF_Type):
        case "BA":
            total_value, longitude, latitude = read_gfed4s(
                files=file_paths, upscale=False
            )
        case "BA_upscale":
            total_value, longitude, latitude = read_gfed4s(
                files=file_paths, upscale=True, shape=(90, 144)
            )
        case "ModelE":
            total_value, longitude, latitude = read_ModelE(
                files=file_paths, variables=variables
            )
        case "lightning":
            total_value, longitude, latitude = read_lightning_data(files=file_paths)
        case "lightning_upscale":
            total_value, longitude, latitude = read_lightning_data(
                files=file_paths, upscale=True
            )
        case _:
            print("[-] No Parsing Script Found For", NetCDF_Type)
    return (total_value, longitude, latitude)


#?????? decide whether the climatological total should become a mean
#and if so, do we want to weigh the monthly data according to the number
#of days in a month
def obtain_time_series_xarray(
    start_time,
    end_time,
    variables,
    time_dimension_name,
    sum_dimension,
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

    # Call read_XX to load read data from netcdf file
    file_paths = obtain_netcdf_files(NetCDF_folder_Path)
    total_value, longitude, latitude = handle_time_extraction_type(
        file_paths=file_paths, variables=variables, NetCDF_Type=NetCDF_Type
    )


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
    # Calculate climatological total over the time dimension
    time_total_data = total_value.sum(dim=time_dimension_name)
    if (
        "m2" in units.lower()
        or "m^2".lower() in units
    ):
        # Calculate the lat-lon total over the climatological period
        total_data_array = total_value.sum(dim=sum_dimension).values
        # Convert m^2 to Mha
        total_data_array *= m2toMha
    elif (
        "m-2".lower() 
        or "m^-2".lower() in units
    ):
        # Calculate the lat-lon total over the climatological period
        grid_cell_area = calculate_grid_area(
            grid_area_shape=(total_value.shape[-2], total_value.shape[-1]))
        total_data_array = (total_value * grid_cell_area).sum(dim=sum_dimension).values
        print("Data Array multiplied by grid_cell_area")

    years = np.arange(start_time, end_time + 1)
    if len(years) != len(total_data_array):
        years = np.arange(0, len(total_data_array))
    data_per_year_stack = np.column_stack((years, total_data_array))

    return (
        time_total_data,
        data_per_year_stack,
        longitude,
        latitude,
        units,
    )


def run_time_series_analysis(folder_data_list):
    # Plot side by side maps for GFED and ModelE
    # Creates the figures for the map and time analysis line plot
    axis_length = len(folder_data_list)
    map_figure, map_axis = plt.subplots(
        nrows=axis_length,
        ncols=1,
        figsize=(18, 10),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    _, time_analysis_axis = plt.subplots(figsize=(10, 6))

    # Example usage with test parameters
    for index, folder_data in enumerate(folder_data_list):
        folder_path, parse_data, plot_data, file_type, variables = (
            folder_data["folder_path"],
            folder_data["parse_data"],
            folder_data["plot_data"],
            folder_data["file_type"],
            folder_data["variables"],
        )

        # Call intann_BA_xarray to calculate decadal mean BA and interannual variability
        time_mean_data, data_per_year_stack, longitude, latitude, units = (
            obtain_time_series_xarray(
                NetCDF_folder_Path=folder_path,
                NetCDF_Type=file_type,
                variables=variables,
                start_time=parse_data["start_time"],
                end_time=parse_data["end_time"],
                time_dimension_name=parse_data["time_dim_name"],
                sum_dimension=parse_data["sum_dimensions"],
            )
        )

        # Plot the decadal mean burned area
        map_plot(
            figure=map_figure,
            axis=map_axis,
            axis_length=axis_length,
            axis_index=index,
            decade_mean_data=time_mean_data,
            longitude=longitude,
            latitude=latitude,
            subplot_title=plot_data["title"],
            units=units,
        )

        print(data_per_year_stack)

        # Plot the time series of burned area for GFED and ModelE
        time_series_plot(
            axis=time_analysis_axis,
            data=data_per_year_stack,
            marker=plot_data["marker"],
            line_style=plot_data["line_style"],
            color=plot_data["color"],
            label=plot_data["label"],
            title=plot_data["title"],
            x_label=plot_data["x_label"],
            y_label=plot_data["y_label"],
        )
    plt.show()
