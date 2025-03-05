import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from utilityFunc import map_plot

# Get variables from plot_peat_cover.py
path = '/discover/nobackup/projects/giss/prod_input_files'
var = 'peat_cover'
filename = 'S144x90_SoilGrids250m2.0_ETOPO1_CongoPeat2_ext.nc'

# Construct the full file path
file_path = os.path.join(path, filename)

# Load the NetCDF file using xarray
try:
    # Open the dataset
    ds = xr.open_dataset(file_path)
    
    # Get the peat cover variable
    peat_data = ds[var]
    
    # Get longitude and latitude
    if 'lon' in ds.dims:
        longitude = ds.lon.values
    elif 'longitude' in ds.dims:
        longitude = ds.longitude.values
    else:
        # Default values if not found
        longitude = np.linspace(-180, 180, peat_data.shape[-1])
    
    if 'lat' in ds.dims:
        latitude = ds.lat.values
    elif 'latitude' in ds.dims:
        latitude = ds.latitude.values
    else:
        # Default values if not found
        latitude = np.linspace(-90, 90, peat_data.shape[-2])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Get units from the data if available, otherwise use a default
    units = getattr(peat_data, 'units', 'fraction')
    
    # Plot the data
    map_plot(
        figure=fig,
        axis=ax,
        axis_length=1,
        axis_index=0,
        decade_data=peat_data,
        longitude=longitude,
        latitude=latitude,
        subplot_title=f"Peat Cover from {filename}",
        units=units,
        cbarmax=None,  # Let the function determine the appropriate maximum
        logMap=True,   # Use logarithmic scale for better visualization
    )
    
    # Save the figure
    output_dir = "figures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, "peat_cover_map.png"), dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {output_dir}/peat_cover_map.png")
    
    # Show the plot
    plt.show()
    
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    print("Please check the path, filename and ensure you have access to the file.")
except KeyError:
    print(f"Error: Variable '{var}' not found in the NetCDF file.")
    print(f"Available variables: {list(ds.data_vars)}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
