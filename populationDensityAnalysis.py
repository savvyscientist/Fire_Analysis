import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import pandas as pd

# Function to create output directory if it doesn't exist
def create_output_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# Path to your NetCDF file - you'll need to modify this
input_file = "/discover/nobackup/projects/giss/prod_input_files/gsin/fire/converted_Pop_Dens_hist_1750_2000_SSP1_2010_2100.nc"

# Create output directories
output_dir = "/discover/nobackup/kmezuman/plots/fire_repository/PopDen/CERESMip" 
maps_dir = create_output_dir(os.path.join(output_dir, "maps"))

# Load the NetCDF file with xarray, with decode_times=False to avoid the error
ds = xr.open_dataset(input_file, decode_times=False)

# Based on the error message, the time values are "years since 1750"
# We need to manually convert these to actual years for better labeling
base_year = 1750
if 'time' in ds.dims:
    time_values = ds['time'].values
    actual_years = base_year + time_values
    
    # Create a new time coordinate with the actual years
    # This doesn't convert to datetime objects but keeps as numeric years
    ds = ds.assign_coords(time=actual_years)

# Extract the population density variable
pop_density = ds['populationDensity']

# Create an area matrix (in km²) based on latitude/longitude grid
# This is a simplified approach - for more accuracy you might want to use proper Earth geometry
lat_values = ds['lat'].values  # Assuming latitude values in degrees
lon_values = ds['lon'].values  # Assuming longitude values in degrees

# Earth's radius in km
earth_radius = 6371.0

# Create empty area matrix
area_matrix = np.zeros((len(lat_values), len(lon_values)))

# Calculate the area of each grid cell
for i, lat in enumerate(lat_values):
    # Convert degrees to radians
    lat_rad = np.deg2rad(lat)
    
    # Calculate latitude bounds
    if i == 0:
        lat_lower = np.deg2rad(-90) if lat < 0 else lat_rad - np.deg2rad(np.abs(lat_values[1] - lat_values[0])/2)
    else:
        lat_lower = np.deg2rad(lat - np.abs(lat_values[i] - lat_values[i-1])/2)
        
    if i == len(lat_values) - 1:
        lat_upper = np.deg2rad(90) if lat > 0 else lat_rad + np.deg2rad(np.abs(lat_values[i] - lat_values[i-1])/2)
    else:
        lat_upper = np.deg2rad(lat + np.abs(lat_values[i+1] - lat_values[i])/2)
    
    # Width in longitude (radians)
    lon_width = np.deg2rad(np.abs(lon_values[1] - lon_values[0]))
    
    # Area calculation for this latitude band
    for j in range(len(lon_values)):
        # Area formula: R² * (lon2 - lon1) * (sin(lat2) - sin(lat1))
        area_matrix[i, j] = earth_radius**2 * lon_width * (np.sin(lat_upper) - np.sin(lat_lower))

# Initialize array to store global totals
global_totals = np.zeros(len(ds['time']))

# Create a custom colormap with gray for NaN values and a gradient for positive values
# Start with white for zero, going to dark red for high values
cmap = LinearSegmentedColormap.from_list(
    'custom_cmap', 
    [(0, 'white'), (0.2, 'yellow'), (0.5, 'orange'), (1, 'darkred')]
)
cmap.set_bad('gray')  # Set NaN values to gray

# Loop through each time step
for t, time_value in enumerate(ds['time'].values):
    # Extract the population density for this time step
    pop_time = pop_density.isel(time=t)
    
    # Replace negative values with 0
    pop_time_clean = pop_time.where(pop_time > 0, 0)
    
    # Calculate the global total by multiplying by area and summing
    global_total = (pop_time_clean * area_matrix).sum().item()
    global_totals[t] = global_total
    
    # Format the time value for the title and filename
    # Since our time values are now just years, use simple string conversion
    time_str = f"{int(time_value)}"
    
    # Create the map visualization
    plt.figure(figsize=(12, 8))
    
    # Plot the data with our custom colormap
    im = plt.pcolormesh(lon_values, lat_values, pop_time_clean, 
                         cmap=cmap, 
                         norm=colors.LogNorm(vmin=0.1, vmax=pop_time_clean.max()),
                         shading='auto')
    
    plt.colorbar(im, label='Population Density')
    plt.title(f'Population Density - {time_str}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add continents/coastlines if cartopy is available
    # If not, the map will still be useful without them
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        ax = plt.subplot(111, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
    except ImportError:
        print("Cartopy not available - plotting without coastlines")
    
    # Save the figure
    plt.savefig(os.path.join(maps_dir, f'pop_density_map_{time_str}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Processed time step {t+1}/{len(ds['time'])}: {time_str}, Global Total: {global_total:,.2f}")

# Create the time series plot of global totals
plt.figure(figsize=(10, 6))
time_labels = [str(int(year)) for year in ds['time'].values]

plt.plot(ds['time'].values, global_totals / 1e9, marker='o', linestyle='-', color='blue')
plt.title('Global Population Total Over Time')
plt.xlabel('Year')
plt.ylabel('Global Population (billions)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the time series plot
plt.savefig(os.path.join(output_dir, 'global_population_timeseries.png'), 
            dpi=300, bbox_inches='tight')

# Display summary statistics
print("\nSummary Statistics:")
print(f"Initial Global Population: {global_totals[0]/1e9:.3f} billion")
print(f"Final Global Population: {global_totals[-1]/1e9:.3f} billion")
print(f"Change: {(global_totals[-1] - global_totals[0])/1e9:.3f} billion")
print(f"Percent Change: {((global_totals[-1] - global_totals[0])/global_totals[0])*100:.2f}%")

# Close the dataset
ds.close()

print("\nProcessing complete. Output files saved to:", output_dir)

