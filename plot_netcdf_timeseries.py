import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

def plot_combined_netcdf(netcdf_file_path, save_plot=True, output_dir=None):
    """
    Read and plot the combined NetCDF time series file.
    
    Parameters:
    -----------
    netcdf_file_path : str
        Path to the combined NetCDF file
    save_plot : bool
        Whether to save the plot as an image
    output_dir : str, optional
        Directory to save the plot (default: same as NetCDF file)
    """
    
    try:
        # Read the NetCDF file
        ds = xr.open_dataset(netcdf_file_path)
        print(f"Successfully loaded NetCDF file: {netcdf_file_path}")
        print(f"Global attributes: {dict(ds.attrs)}")
        print(f"Variables in file: {list(ds.data_vars.keys())}")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define colors and markers (similar to your utilityFunc style)
        colors = ['r', 'b', 'k', 'g', 'm', 'c', 'orange', 'purple']
        markers = ['o', 'x', '*', 's', '^', 'v', 'd', '+']
        line_styles = ['-', '--', '-.', ':']
        
        # Get time values
        time_values = ds['time'].values
        
        # Check if we have monthly data by looking for fractional parts
        is_monthly = np.any(np.mod(time_values, 1) > 0)
        
        # Plot each variable
        for i, var_name in enumerate(ds.data_vars.keys()):
            var_data = ds[var_name]
            
            # Get metadata
            units = var_data.attrs.get('units', 'unknown units')
            long_name = var_data.attrs.get('long_name', var_name)
            
            # Create label from long_name or variable name
            label = long_name.replace('Total data for ', '') if 'Total data for' in long_name else var_name
            
            # Plot with cycling colors and markers
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            line_style = line_styles[i % len(line_styles)]
            
            ax.plot(
                time_values,
                var_data.values,
                marker=marker,
                linestyle=line_style,
                color=color,
                label=label,
                markersize=6,
                linewidth=1.5
            )
        
        # Format the plot similar to utilityFunc
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set title and labels
        time_type = ds.attrs.get('time_type', 'time series')
        time_range = ds.attrs.get('time_range', '')
        ax.set_title(f"Combined Time Series Data ({time_type.title()}) - {time_range}")
        
        # Get units from first variable for y-label (assuming all have same units)
        first_var = list(ds.data_vars.keys())[0]
        y_units = ds[first_var].attrs.get('units', 'units')
        ax.set_ylabel(f"Total Data ({y_units})")
        
        # Format x-axis based on data type
        if is_monthly:
            if len(np.unique(np.floor(time_values))) == 1:
                # Single year monthly data
                year = int(np.floor(time_values[0]))
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # Create custom ticks for months
                tick_positions = []
                tick_labels = []
                for month in range(12):
                    decimal_time = year + month/12.0
                    if any(np.isclose(time_values, decimal_time, atol=0.01)):
                        tick_positions.append(decimal_time)
                        tick_labels.append(month_names[month])
                
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=45, ha='right')
                ax.set_xlabel(f"Month ({year})")
            else:
                # Multi-year monthly data
                ax.set_xlabel("Year")
        else:
            # Annual data
            ax.set_xlabel("Year")
            if len(time_values) <= 10:
                ax.set_xticks(time_values)
                ax.set_xticklabels([str(int(year)) for year in time_values])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            if output_dir is None:
                output_dir = os.path.dirname(netcdf_file_path)
            
            # Create output filename
            base_name = os.path.splitext(os.path.basename(netcdf_file_path))[0]
            plot_filename = f"{base_name}_plot.png"
            plot_path = os.path.join(output_dir, plot_filename)
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")
        
        # Show plot
        plt.show()
        
        # Print summary statistics
        print("\n=== Data Summary ===")
        for var_name in ds.data_vars.keys():
            var_data = ds[var_name].values
            print(f"{var_name}:")
            print(f"  Range: {np.min(var_data):.3e} to {np.max(var_data):.3e}")
            print(f"  Mean: {np.mean(var_data):.3e}")
            print(f"  Units: {ds[var_name].attrs.get('units', 'unknown')}")
        
        # Close dataset
        ds.close()
        
    except Exception as e:
        print(f"Error reading or plotting NetCDF file: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function - modify the file path to match your NetCDF file location
    """
    # UPDATE THIS PATH to match your NetCDF file location
    netcdf_path = "/discover/nobackup/kmezuman/plots/CCycle/Fire_analysis/fuel_CO2/combined_timeseries_monthly_2010-2011.nc"
    
    # Check if file exists
    if not os.path.exists(netcdf_path):
        print(f"NetCDF file not found: {netcdf_path}")
        print("Please update the 'netcdf_path' variable with the correct file location.")
        return
    
    # Plot the data
    plot_combined_netcdf(netcdf_path, save_plot=True)

if __name__ == "__main__":
    main()
