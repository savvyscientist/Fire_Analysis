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
    
    # Fix for logarithmic scale: ensure all values are positive
    use_log_scale = False  # Set to match your logMap parameter
    
    if use_log_scale:
        # Check if data contains zeros or negative values
        min_value = float(peat_data.min())
        if min_value <= 0:
            print(f"Warning: Data contains values ≤ 0 (min: {min_value}), which are incompatible with log scale.")
            print("Applying small positive offset to enable logarithmic visualization.")
            
            # Find smallest positive value to use as minimum threshold
            # or use a small fraction of the maximum as fallback
            positive_min = float(peat_data[peat_data > 0].min()) if np.any(peat_data > 0) else float(peat_data.max() * 0.001)
            safe_min = positive_min * 0.5  # Use half of smallest positive value
            
            # Create a copy with adjusted values for visualization
            # This preserves the original data in the dataset
            peat_data_viz = peat_data.copy(deep=True)
            peat_data_viz = peat_data_viz.where(peat_data_viz > 0, safe_min)
            
            # Optional: print data range for debugging
            print(f"Data range after adjustment: {float(peat_data_viz.min())} to {float(peat_data_viz.max())}")
        else:
            peat_data_viz = peat_data
    else:
        peat_data_viz = peat_data
    
    # Calculate appropriate vmin and vmax to prevent colorbar issues
    vmin = float(peat_data_viz.min())
    vmax = float(peat_data_viz.max())
    
    # Ensure vmin < vmax with a small buffer
    if vmin >= vmax:
        print(f"Warning: vmin ({vmin}) ≥ vmax ({vmax}), adding buffer")
        if abs(vmin) < 1e-10:  # Effectively zero
            vmin = 1e-10
            vmax = max(vmax, 1e-9)
        elif vmin == vmax:  # Equal non-zero values
            buffer = abs(vmin) * 0.01  # 1% buffer
            vmin -= buffer
            vmax += buffer
    
    # Now plot with custom vmin/vmax
    try:
        map_plot(
            figure=fig,
            axis=ax,
            axis_length=1,
            axis_index=0,
            decade_data=peat_data_viz,  # Use the adjusted data
            longitude=longitude,
            latitude=latitude,
            subplot_title=f"Peat Cover from {filename}",
            units=units,
            cbarmax=vmax,     # Set the maximum explicitly
            logMap=use_log_scale,
        )
    except ValueError as e:
        if "Invalid vmin or vmax" in str(e):
            # Fallback to linear scale if logarithmic still fails
            print("Error with logarithmic scale. Falling back to linear scale.")
            map_plot(
                figure=fig,
                axis=ax,
                axis_length=1,
                axis_index=0,
                decade_data=peat_data,  # Use original data
                longitude=longitude,
                latitude=latitude,
                subplot_title=f"Peat Cover from {filename} (Linear Scale)",
                units=units,
                cbarmax=None,
                logMap=False,  # Switch to linear scale
            )
        else:
            # Re-raise if it's a different error
            raise
    
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
