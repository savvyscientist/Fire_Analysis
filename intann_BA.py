import h5py
import xarray as xr
import numpy as np
import matplotlib.colors as mcolors 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def read_gfed4s(file_paths):
    """
    Reads multiple HDF5 files using h5py, calculates the annual burned area,
    and returns the data as xarray.DataArray.
    """
    burned_fraction_list = []

    for file_path in file_paths:
        # Open the HDF5 file using h5py
        print(file_path)
        with h5py.File(file_path, 'r') as h5file:
            # Access grid_cell_area using the method suggested
            grid_cell_area = h5file['ancill']['grid_cell_area'][:]
            
            # Load lat and lon for constructing the xarray dataset
            lat = h5file['lat'][:]
            lon = h5file['lon'][:]
            
            # Sum burned fraction over all months
            annual_burned_fraction = np.zeros_like(grid_cell_area)
            for month in range(1, 13):
                month_burned_fraction = h5file[f'burned_area/{month:02d}/burned_fraction'][:]
                annual_burned_fraction += month_burned_fraction

            # Calculate total burned area
            total_burned_area = annual_burned_fraction * grid_cell_area
            burned_fraction_list.append(total_burned_area)

    # Convert the list to xarray.DataArray for further processing
    total_burned_area_all_years = xr.DataArray(
        burned_fraction_list, 
        dims=['year', 'phony_dim_0', 'phony_dim_1'], 
        coords={'lat': (['phony_dim_0', 'phony_dim_1'], lat), 
                'lon': (['phony_dim_0', 'phony_dim_1'], lon)}
    )

    return total_burned_area_all_years, lon, lat

def read_ModelEBA(startyear, endyear, simname, ModelE_path):
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

    # Generate a list of file paths for the given year range
    file_paths = [f'{ModelE_path}ANN{year}.aij{simname}.nc' for year in range(startyear, endyear + 1)]

    # Initialized a litst to store each year's dataset
    datasets = []
    zero_mat = np.zeros((90, 144), dtype=float)

    # Loop over each file and process it
    for file_path in file_paths:
        print(file_path)
        ds = xr.open_dataset(file_path)
    
        # Read BA_tree, BA_shrub, BA_grass and set negative values to zero
        BA_tree = ds['BA_tree'].where(ds['BA_tree'] >0., 0.)
        BA_shrub = ds['BA_shrub'].where(ds['BA_shrub'] >0., 0.)
        BA_grass = ds['BA_grass'].where(ds['BA_grass'] >0., 0.)

        # Sum BA_tree, BA_shrub, and BA_grass to get modelE_BA
        modelE_BA = BA_tree + BA_shrub + BA_grass

        # Add a time coordinate based on the year from the file name
        year = int(file_path.split('ANN')[1][:4])
        modelE_BA = modelE_BA.expand_dims(time=[year]) # Add time dimension for each year

        # Append the processes dataset to the list
        datasets.append(modelE_BA)
    # Concatenate all datasets along the 'time' dimension
    modelE_BA_all_year = xr.concat(datasets, dim='time')
    modelE_lons = ds['lon']
    modelE_lats = ds['lat']
    
    return modelE_BA_all_year, modelE_lons, modelE_lats

def intann_BA_xarray(startyear, endyear, GFED_path, ModelE_path, simname):
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
    file_paths = [f'{GFED_path}GFED4.1s_{year}.hdf5' for year in range(startyear, endyear + 1)]
    total_burned_area_all_years, GFED_lons, GFED_lats = read_gfed4s(file_paths)

    # Calculate the mean burned area over the decade
    decade_mean_gfed4sba = total_burned_area_all_years.mean(dim='year')

    # Calculate total burned area for each year from GFED4s data
    total_ba_per_year = total_burned_area_all_years.sum(dim=['phony_dim_0', 'phony_dim_1']).values
    years = np.arange(startyear, endyear + 1)
    gfed4sba_per_year = np.column_stack((years, total_ba_per_year))

    # Call read_ModelEBA to load and process ModelE data
    modelE_BA_all_years, modelE_lons, modelE_lats = read_ModelEBA(startyear, endyear, simname, ModelE_path)

    # Calculate the mean burned area over the decade (ModelE)
    decade_mean_modelEba = modelE_BA_all_years.mean(dim='time')

    # Calculate total burned area for each year from ModelE data
    total_modelE_BA_per_year = modelE_BA_all_years.sum(dim=['lat', 'lon']).values
    modelE_BA_per_year = np.column_stack((years, total_modelE_BA_per_year))

    return decade_mean_gfed4sba, GFED_lons, GFED_lats, decade_mean_modelEba, modelE_lons, modelE_lats, gfed4sba_per_year, modelE_BA_per_year 

def define_subplot(fig, ax, data, lons, lats, cmap, cborientation, fraction, pad, labelpad, fontsize, title, clabel, masx, is_diff=False,glob=None):
    #labelpad sets the distance of the colorbar from the map
    """Define the properties of a subplot with optional difference normalization."""
    ax.coastlines(color='black')
    ax.add_feature(cfeature.LAND, edgecolor='gray')
    ax.add_feature(cfeature.OCEAN, facecolor='white', edgecolor='none', zorder=1)

    ax.set_title(title, fontsize=10, pad=1)
    props = dict(boxstyle="round", facecolor='lightgray', alpha=0.5)
    (ax.text(0.5, 1.07, f"Global Total: {glob}", ha="center", va="center", transform=ax.transAxes, bbox=props, fontsize=10)) if glob else None

    # Handling difference normalization (if is_diff is true)
    if is_diff:
        data_min, data_max = data.min(), data.max()
        if data_min == data_max:
            norm = mcolors.Normalize(vmin=data_min - 1, vmax=data_max + 1)
        else:
            abs_max = max(abs(0.25 * data_min), abs(0.25 * data_max))
            norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
        p = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, vmin=0 if not is_diff else None, vmax=masx if not is_diff else None)
    else:
        norm = None
        # Mask values less than or equal to zero for the custom colormap (set to white)
        masked_data = np.ma.masked_less_equal(data, 0)  # Mask values <= 0
        # Create a colormap with white for values <= 0
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad(color='white')  # Set masked values to white
        p = ax.pcolormesh(lons, lats, masked_data, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, vmin=0 if not is_diff else None, vmax=masx if not is_diff else None)

    cbar = fig.colorbar(p, ax=ax, orientation=cborientation, fraction=fraction, pad=pad)
    cbar.set_label(f'{clabel}', labelpad=labelpad, fontsize=fontsize)

    return ax

def map_plot(decade_mean_gfed4sba, GFED_lons, GFED_lats, decade_mean_modelEba, modelE_lons, modelE_lats):
    """
    Plots the decadal mean burned area of both GFED and ModelE side by side.
    
    Parameters:
    decade_mean_gfed4sba (xarray.DataArray): The decadal mean burned area (lat, lon array).
    decade_mean_modelEba (xarray.DataArray): The decadal mean burned area from ModelE(lat, lon array).
    """
   
    # Plot side by side maps for GFED and ModelE
    fig, ax = plt.subplots(3, 1, figsize=(18, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # GFED4s decadal mean map
    define_subplot(fig, ax[0], decade_mean_gfed4sba, GFED_lons, GFED_lats, cmap='jet', cborientation='horizontal', fraction=0.05,
                   pad=0.005, labelpad=0.5, fontsize=10, title=f'GFED4s Decadal Mean Burned Area (2002-2012)',
                   clabel='BA [$m^2$]',masx=0.7*decade_mean_gfed4sba.max())

    # Regridded GFED4s decadal mean map
    define_subplot(fig, ax[0], decade_mean_gfed4supba, GFED_lons, GFED_lats, cmap='jet', cborientation='horizontal', fraction=0.05,
                   pad=0.005, labelpad=0.5, fontsize=10, title=f'Regridded GFED4s Decadal Mean Burned Area (2002-2012)',
                   clabel='BA [$m^2$]',masx=0.7*decade_mean_gfed4sba.max())

    # ModelE decadal mean map 
    define_subplot(fig, ax[1], decade_mean_modelEba, modelE_lons, modelE_lats, cmap='jet', cborientation='horizontal', fraction=0.05,
                   pad=0.005, labelpad=0.5, fontsize=10, title=f'pyrE Decadal Mean Burned Area (2002-2012)',
                   clabel='BA [$m^2$]',masx=0.7*decade_mean_modelEba.max())

    plt.show()

def time_series_plot(gfed4sba_per_year, gfed4supba_per_year, modelE_BA_per_year):
    """
    Plots the total burned area as a function of year for both GFED and ModelE data.
    
    Parameters:
    gfed4sba_per_year (np.ndarray): A 2D array with columns (year, totalBA), where totalBA is the sum of burned area for that year.
    modelE_BA_per_year (np.ndarray): A 2D array with columns (year, modelE_BA), where modelE_BA is the sum of burned area for that year.
    """
    
    # Extract years and total burned area for both GFED and ModelE
    years_gfed = gfed4sba_per_year[:, 0]
    total_ba_gfed = gfed4sba_per_year[:, 1]
    
    years_modelE = modelE_BA_per_year[:, 0]
    total_ba_modelE = modelE_BA_per_year[:, 1]
    
    # Plot the time series of total burned area for both GFED and ModelE
    plt.figure(figsize=(10, 6))
    plt.plot(years_gfed, total_ba_gfed, marker='o', linestyle='-', color='b', label='GFED BA')
    plt.plot(years_modelE, total_ba_modelE, marker='x', linestyle='-', color='r', label='ModelE BA')
    plt.title('Total Burned Area Over Time (2002-2012)')
    plt.xlabel('Year')
    plt.ylabel('Total Burned Area (BA)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Example usage with test parameters
    startyear = 2002
    endyear = 2012
    GFED_path = '/discover/nobackup/projects/giss_ana/users/kmezuman/gfed4s/updates/'  
    ModelE_path = '/discover/nobackup/kmezuman/nk_CCycle_E6obioF40/'
    simname = 'nk_CCycle_E6obioF40'

    # Call intann_BA_xarray to calculate decadal mean BA and interannual variability
    decade_mean_gfed4sba, GFED_lons, GFED_lats, decade_mean_gfed4supba, GFED4sup_lons, GFED4sup_lats, decade_mean_modelEba, modelE_lons, modelE_lats, gfed4sba_per_year, gfed4supba_per_year, modelE_BA_per_year = intann_BA_xarray(startyear, endyear, GFED_path, GFED_path+'upscaled/' ModelE_path, simname)

    # Plot the decadal mean burned area
    map_plot(decade_mean_gfed4sba, GFED_lons, GFED_lats, decade_mean_gfed4supba, GFED4sup_lons, GFED4sup_lats, decade_mean_modelEba, modelE_lons, modelE_lats)

    # Plot the time series of burned area for GFED and ModelE
    time_series_plot(gfed4sba_per_year, gfed4supba_per_year,  modelE_BA_per_year)

if __name__ == '__main__':
    main()
