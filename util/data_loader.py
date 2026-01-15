"""
COMPLETE Data Loading Module - All Functionality Integrated
OPTIMIZED for speed with vectorized operations and efficient memory usage
"""
import os
import re
import numpy as np
import xarray as xr
import h5py
from typing import List, Tuple, Optional
from dataclasses import dataclass
from netCDF4 import Dataset
import netCDF4 as nc
import pandas as pd
from pathlib import Path

from constants import MONTHLIST, DAYS_TO_SECONDS, KM_SQUARED_TO_M_SQUARED, MONTHLISTDICT, SECONDS_IN_A_YEAR

# Performance optimizations
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Use faster operations
np.seterr(divide='ignore', invalid='ignore')


@dataclass
class TimeSeriesData:
    """Container for time series data."""
    time_mean: xr.DataArray
    time_series: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    units: str
    start_year: int
    end_year: int


def obtain_netcdf_files(dir_path: str) -> List[str]:
    """Get list of netCDF files in directory."""
    if not os.path.exists(dir_path):
        return []
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f)) and f.split(".")[-1] in ["hdf5", "nc"]]


def days_to_months(month: str, year: int) -> int:
    """Get number of days in a month."""
    days = MONTHLISTDICT.get(month, 30)
    if month in ['02', 'FEB'] and ((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)):
        days = 29
    return days


def extract_scaling_factor(units: str) -> Tuple[float, str]:
    """Extract scaling factor from units string."""
    try:
        pattern = r"^(10\^(-?\d+)|[-+]?\d*\.?\d+([eE][-+]?\d+)?)\s*(.*)$"
        match = re.match(pattern, units)
        if match:
            if match.group(1).startswith("10^"):
                return float(10) ** float(match.group(2)), match.group(4)
            else:
                return float(match.group(1)), match.group(4)
        return 1.0, units
    except:
        return 1.0, units


class DataLoader:
    """Load and process data from various file formats - COMPLETE IMPLEMENTATION."""
    
    def __init__(self, grid_area_calculator=None):
        self.grid_area_calculator = grid_area_calculator
        self._loaders = {
            'ModelE': lambda p,v,a,n,s,t,c=None: self._load_format(p,v,a,n,s,t, self._read_ModelE, False),
            'ModelE_Monthly': lambda p,v,a,n,s,t,c=None: self._load_format(p,v,a,n,s,t, self._read_ModelE, True),
            'Combined_ModelE': lambda p,v,a,n,s,t,c: self._load_combined_ModelE(c, v, a, n, s, t),
            'BA_GFED4': lambda p,v,a,n,s,t,c=None: self._load_gfed4sba_format(p,v,a,n,s,t, False),
            'BA_GFED4_upscale': lambda p,v,a,n,s,t,c=None: self._load_gfed4sba_format(p,v,a,n,s,t, True),
            'BA_GFED5': lambda p,v,a,n,s,t,c=None: self._load_gfed5ba_format(p,v,a,n,s,t, False),
            'BA_GFED5_upscale': lambda p,v,a,n,s,t,c=None: self._load_gfed5ba_format(p,v,a,n,s,t, True),
            'GFED4s_Monthly': lambda p,v,a,n,s,t,c=None: self._load_emis_format(p,v,n,s,t, True),
            'GFED5_Monthly': lambda p,v,a,n,s,t,c=None: self._load_emis_format(p,v,n,s,t, True),
            'FINN2.5_Monthly': lambda p,v,a,n,s,t,c=None: self._load_emis_format(p,v,n,s,t, True),
            'WGLC_lightning': lambda p,v,a,n,s,t,c=None: self._load_format(p,v,n,s,t, self._read_WGLC_lightning, True),
        }
    
    def load_time_series(self, folder_path: str, file_type: str, variables: List[str],
                         name: str = None, annual: bool = False,
                         components: Optional[List[str]] = None,
                         spatial_aggregation: str = 'total',
                         temporal_aggregation_map: str = 'annual_mean') -> Optional[TimeSeriesData]:
        """
        Main entry point for loading data.
        
        Args:
            spatial_aggregation: For time series - 'total' (sum) or 'mean' over SPACE
            temporal_aggregation_map: For spatial maps - how to aggregate over TIME:
                'annual_mean': Mean of annual totals (emissions - sum 12 months, then average years)
                'period_mean': Mean over entire period (flammability, temperature)
                'period_sum': Sum over entire period (rare - cumulative)
        """
        if file_type not in self._loaders:
            print(f"Unknown file type: {file_type}")
            return None
        try:
            return self._loaders[file_type](folder_path, variables, annual, name, spatial_aggregation, temporal_aggregation_map, components)
        except Exception as e:
            print(f"Error loading {file_type}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_combined_ModelE(self, component_paths, variables, annual, name, spatial_aggregation, temporal_aggregation_map):
        """Custom loader for summing multiple ModelE formats."""
        print("\n=== Reading COMBINED ModelE data ===")

        combined_total_value = None
        lon, lat = None, None

        for path in component_paths:
            print(f"  --> Loading component: {path}")
            # The structure of ModelE_Monthly loader is a good reference
            files = obtain_netcdf_files(path)
            if not files:
                print(f"    WARNING: No files found in {path}. Skipping.")
                continue
            files.sort()

            # Use the existing ModelE reader
            total_value, current_lon, current_lat = self._read_ModelE(files, variables, monthly=True, name=None)

            if total_value is not None:
                if combined_total_value is None:
                    combined_total_value = total_value
                    lon, lat = current_lon, current_lat
                else:
                    # Ensure the dimensions match before summing
                    if combined_total_value.shape == total_value.shape:
                        combined_total_value.values += total_value.values
                    else:
                        print(f"    ERROR: Component grid shapes do not match. Skipping {path}.")

        if combined_total_value is None:
            return None

        # Final processing
        return self._process_data(combined_total_value, lon, lat, annual, spatial_aggregation, temporal_aggregation_map)
    
    def _load_format(self, folder_path, variables, annual, name, spatial_aggregation, temporal_aggregation_map, reader_func, monthly):
        """
        Generic loader for various data formats.
   
        This method handles data readers that return either:
        - 3-tuple: (data, lon, lat) - for standard readers (ModelE, GFED, etc.)
        - 5-tuple: (data, lon, lat, time_values, units) - for special readers (WGLC, etc.)
   
        Args:
            folder_path: Path to data folder
            variables: List of variable names to read
            annual: Whether data is annual (vs monthly)
            name: Dataset name for logging
            spatial_aggregation: 'total' or 'mean' for spatial aggregation
            temporal_aggregation_map: How to aggregate time for maps
            reader_func: Function that reads the data files
            monthly: Whether data has monthly time resolution
       
        Returns:
            TimeSeriesData object or None if loading fails
        """
        # Get list of files
        files = obtain_netcdf_files(folder_path)
        if not files:
            return None
        files.sort()
   
        # Call the reader function
        result = reader_func(files, variables, monthly, name)
   
        # Check if reader returned None
        if result is None:
            return None
   
        # Handle different return formats
        if len(result) == 3:
            # Standard format: (data, lon, lat)
            total_value, lon, lat = result
       
        elif len(result) == 5:
            # Extended format: (data, lon, lat, time_values, units)
            # Used by WGLC and other special loaders that provide their own time/units
            total_value, lon, lat, time_values, units = result
       
        # Note: time_values and units from reader could be used here
        # if _process_data is updated to accept them as optional arguments
        # For now, _process_data will extract them from the xarray DataArray
       
    else:
        # Unexpected format
        print(f"  ✗ Error: Reader returned unexpected format (length {len(result)})")
        return None
   
    # Check if data is valid
    if total_value is None:
        return None
   
    # Process the data into TimeSeriesData
    return self._process_data(
        total_value,
        lon,
        lat,
        annual,
        spatial_aggregation,
        temporal_aggregation_map
    ) 
    def _load_gfed4sba_format(self, folder_path, variables, annual, name, spatial_aggregation, temporal_aggregation_map, upscaled):
        """Loader for GFED4s formats."""
        files = obtain_netcdf_files(folder_path)
        if not files:
            return None
        files.sort()
        total_value, lon, lat = self._read_gfed4sba(files, upscaled)
        return self._process_data(total_value, lon, lat, annual, spatial_aggregation, temporal_aggregation_map)

    def _load_gfed5ba_format(self, folder_path, variables, annual, name, spatial_aggregation, temporal_aggregation_map, upscaled):
        """Loader for GFED5 formats."""
        files = obtain_netcdf_files(folder_path)
        if not files:
            return None
        files.sort()
        total_value, lon, lat = self._read_gfed5ba(files, variables, upscaled)
        return self._process_data(total_value, lon, lat, annual, spatial_aggregation, temporal_aggregation_map)
    
    def _load_emis_format(self, folder_path, variables, annual, name, spatial_aggregation, temporal_aggregation_map):
        """Loader for emissions formats."""
        files = obtain_netcdf_files(folder_path)
        if not files:
            return None
        files.sort()
        total_value, lon, lat = self._read_modelEinput_emis(files, variables)
        return self._process_data(total_value, lon, lat, annual, spatial_aggregation, temporal_aggregation_map)
    
    def _days_in_month(self, month: int, year: int) -> int:
    """
    Get number of days in a month, accounting for leap years.
   
    Args:
        month: Month (1-12)
        year: Year (for leap year calculation)
       
    Returns:
        Number of days in the month
    """
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
   
    # Check for leap year in February
    if month == 2:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 29
   
    return days[month - 1]
    
    def _read_WGLC_lightning(self, files: List[str], variables: List[str], monthly: bool, name: str) -> Tuple:
    """
    Read WGLC (World Grid Lightning Climatology) lightning data.
    
    Based on read_lightning_data function but adapted for the data_loader framework.
    
    Args:
        files: List of file paths (should be single file for WGLC)
        variables: List of variables to read (typically ['density'])
        monthly: Whether to return monthly data (True) or annual (False)
        name: Dataset name for logging
        
    Returns:
        Tuple of (data_array, lon, lat) where data_array has shape (time, lat, lon)
    """
    if not files:
        print(f"  ✗ No files found for {name}")
        return None, None, None
    # WGLC typically has one file with all time steps
    file_path = files[0]
    print(f"\n=== Reading WGLC Lightning data ===")
    print(f"File: {file_path}")
    print(f"Variables: {variables}")
    print(f"Monthly: {monthly}")
    try:
        with nc.Dataset(file_path) as netcdf_dataset:
            # Get the density variable (lightning flash density)
            var_name = variables[0] if variables else 'density'
            if var_name not in netcdf_dataset.variables:
                print(f"  ✗ Variable '{var_name}' not found in file")
                print(f"  Available variables: {list(netcdf_dataset.variables.keys())}")
                return None, None, None

            density_variable = netcdf_dataset.variables[var_name]
            density_data = density_variable[:]# Shape: (time, lat, lon)

            # Get units
            units = density_variable.units if hasattr(density_variable, 'units') else 'unknown'
            print(f"Original units: {units}")
            # Get time data
            time_data = netcdf_dataset.variables['time'][:]
            # Get lat/lon
            if 'latitude' in netcdf_dataset.variables:
                latitudes = netcdf_dataset.variables['latitude'][:]
            else:
                latitudes = np.linspace(-90, 90, density_data.shape[-2])
            if 'longitude' in netcdf_dataset.variables:
                longitudes = netcdf_dataset.variables['longitude'][:]
            else:
                longitudes = np.linspace(-180, 180, density_data.shape[-1])
            print(f"Data shape: {density_data.shape}")
            print(f"Time steps: {len(time_data)}")
            print(f"Lat range: {latitudes.min():.2f} to {latitudes.max():.2f}")
            print(f"Lon range: {longitudes.min():.2f} to {longitudes.max():.2f}")

            # Handle fill values / missing data
            if hasattr(density_variable, '_FillValue'):
                fill_value = density_variable._FillValue
                density_data = np.where(density_data == fill_value, 0.0, density_data)

            # Set negative values to zero
            density_data = np.maximum(density_data, 0.0)
            
            if monthly:
                # Return monthly data
                print(f"Returning monthly data: {density_data.shape}")
                # Units are typically "lightning strokes km-2 d-1"
                # Convert to monthly by multiplying by days in month
                start_date = "2010-01-01"# Adjust based on actual data
                date_range = pd.date_range(start_date, freq="MS", periods=len(density_data))
                monthly_data = []
                time_values = []

                for month_idx in range(len(density_data)):
                    current_year = date_range[month_idx].year
                    current_month = date_range[month_idx].month

                    # Get days in month
                    days_in_month = self._days_in_month(current_month, current_year)

                    # Convert from per-day to per-month
                    monthly_value = density_data[month_idx] * days_in_month
                    monthly_data.append(monthly_value)

                   # Calculate decimal year
                   time_values.append(current_year + (month_idx % 12) / 12.0)

                final_data = np.array(monthly_data)
                time_array = np.array(time_values)
                final_units = "lightning strokes km-2 month-1"
                
            else:
                # Return annual data
                print(f"Aggregating to annual data...")
                
                # Group by year and sum
                start_date = "2010-01-01"  # Adjust based on actual data
                date_range = pd.date_range(start_date, freq="MS", periods=len(density_data))
                
                yearly_data = {}
                for month_idx in range(len(density_data)):
                    year = date_range[month_idx].year
                    month = date_range[month_idx].month
                    
                    # Get days in month
                    days_in_month = self._days_in_month(month, year)
                    
                    # Convert to monthly then accumulate
                    monthly_value = density_data[month_idx] * days_in_month
                    
                    if year in yearly_data:
                        yearly_data[year] += monthly_value
                    else:
                        yearly_data[year] = monthly_value.copy()
                
                # Convert to arrays
                years = sorted(yearly_data.keys())
                final_data = np.array([yearly_data[year] for year in years])
                time_array = np.array(years, dtype=float)
                final_units = "lightning strokes km-2 yr-1"
                
                print(f"Annual data shape: {final_data.shape}")
                print(f"Years: {years}")
            
            print(f"Final units: {final_units}")
            print(f"Data range: {final_data.min():.6e} to {final_data.max():.6e}")
            
            return final_data, longitudes, latitudes, time_array, final_units
            
    except Exception as e:
        print(f"  ✗ Error reading WGLC data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

    def _read_ModelE(self, files, variables, monthly=False, name=None):
        """Read ModelE - Logic: Ensure fractions are integrated by setting units to m^2/m^2."""
        first_ds = xr.open_dataset(files[0])
        lat, lon = first_ds['lat'].values, first_ds['lon'].values
        
        # Identify unit scaling from the first variable
        sample_var = variables[0]
        original_units = first_ds[sample_var].attrs.get('units', '1')
        scaling_factor, clean_units = extract_scaling_factor(original_units)
        
        # CRITICAL LOGIC: If this is a BA fraction (unitless), 
        # set units to 'm^2/m^2' so the workflow knows to multiply by Grid Area.
        is_ba = any(x in sample_var.lower() for x in ['ba', 'burn'])
        if is_ba and clean_units in ['1', 'fraction', 'none']:
            clean_units = "m^2/m^2"
            
        first_ds.close()
        
        all_data = []
        time_values = []

        for fpath in files:
            ds = xr.open_dataset(fpath, decode_times=False)
            # Combine requested variables (e.g., BA_tree + BA_shrub)
            combined = sum(ds[v].values for v in variables if v in ds.variables)
            
            # Clean up fill values/NaNs
            combined = np.nan_to_num(combined, nan=0.0)
            combined[combined < 0] = 0.0
            
            # Apply base scaling (e.g., from 10^-6 to 1)
            combined *= scaling_factor
            
            # Handle Time scaling (per second to per month/year)
            fname = os.path.basename(fpath)
            if monthly:
                month_str = fname.split('.')[0][-7:-4].upper() # e.g., 'JAN'
                year = int(fname.split('.')[0][-4:])
                month_idx = MONTHLIST.index(month_str) + 1
                if 's-1' in original_units or '/s' in original_units:
                    days = days_to_months(str(month_idx).zfill(2), year)
                    combined *= (days * 24 * 3600)
                time_values.append(year + (month_idx - 1) / 12.0)
            else:
                if 's-1' in original_units or '/s' in original_units:
                    combined *= SECONDS_IN_A_YEAR
                time_values.append(int(re.search(r'\d{4}', fname).group()))

            all_data.append(combined)
            ds.close()

        return xr.DataArray(
            np.array(all_data),
            dims=['time', 'lat', 'lon'],
            coords={'time': time_values, 'lat': lat, 'lon': lon},
            attrs={'units': clean_units}
        ), lon, lat

    def _read_gfed5ba(self, files, variables, upscaled=False):
        """Read GFED5 NetCDF files - Logic: Normalize all to m^2 based on file units."""
        variables = variables if variables else ["Total"]
        n_files = len(files)
        
        # Open first file to establish grid and metadata
        with Dataset(files[0]) as ds:
            var_name = variables[0] if variables[0] in ds.variables else list(ds.variables.keys())[0]
            sample_var = ds.variables[var_name]
            
            # 1. Logic: Check units attribute
            file_units = getattr(sample_var, 'units', 'km2') # Default to km2 for GFED5
            # If units are km2, we need to convert to m2 to match model integration
            internal_scale = KM_SQUARED_TO_M_SQUARED if 'km' in file_units.lower() else 1.0
            
            # Setup grid
            arr = sample_var[:]
            arr = arr[0] if len(arr.shape) > 2 else arr
            h, w = arr.shape
            lats = np.linspace(-90, 90, h)
            lons = np.linspace(-180, 180, w)
            
            attrs = {a: getattr(sample_var, a) for a in sample_var.ncattrs()}
            attrs["units"] = "m^2" # We normalize the output to m^2

        data_list = np.zeros((n_files, h, w), dtype=np.float32)
        time_vals = np.zeros(n_files, dtype=np.float64)

        for idx, fpath in enumerate(files):
            with Dataset(fpath) as ds:
                month_sum = np.zeros((h, w), dtype=np.float32)
                for var in variables:
                    if var in ds.variables:
                        data = ds.variables[var][:]
                        if len(data.shape) > 2: data = data[0]
                        month_sum += data.astype(np.float32)
                
                # Apply the unit-based scaling
                data_list[idx] = month_sum * internal_scale
                
                # Robust time extraction
                match = re.search(r'BA(\d{4})(\d{2})', os.path.basename(fpath))
                if match:
                    year, month = int(match.group(1)), int(match.group(2))
                    time_vals[idx] = year + (month - 1) / 12.0
                else:
                    time_vals[idx] = 2000.0 + idx

        return xr.DataArray(
            data_list, 
            dims=['time', 'latitude', 'longitude'],
            coords={'time': time_vals, 'latitude': lats, 'longitude': lons},
            attrs=attrs
        ), lons, lats
    
    def _read_gfed4sba(self, files, upscaled=False):
        """Read GFED4s HDF5 files."""
        data_list, time_arr = [], []
        
        for fpath in files:
            with h5py.File(fpath, "r") as h5:
                lat, lon = h5["lat"][:], h5["lon"][:]
                annual_ba = np.zeros(h5["burned_areas_01" if upscaled else 
                                       "burned_area/01/burned_fraction"].shape)
                
                for m in range(1, 13):
                    key = f"burned_areas_{m:02d}" if upscaled else f"burned_area/{m:02d}/burned_fraction"
                    annual_ba += h5[key][:]
                
                if not upscaled:
                    annual_ba *= h5["ancill"]["grid_cell_area"][:]
                
                data_list.append(annual_ba)
                time_arr.append(fpath.split("_")[1].split(".")[0] if not upscaled else fpath.split("_")[1])
        
        attrs = {"units": "m^2", "long_name": "GFED4s burned area"}
        return xr.DataArray(np.array(data_list), dims=['time','lat','lon'],
                           coords={'time':time_arr, 'lat':lat, 'lon':lon},
                           attrs=attrs), lon, lat
    
    def _read_modelEinput_emis(self, files, variables):
        """Read ModelE input emissions."""
        all_data, time_vals = [], []
        
        ds = xr.open_dataset(files[0], decode_times=False)
        lon, lat = ds.lon.values, ds.lat.values
        emis_var = variables[0]
        attrs = dict(ds[emis_var].attrs)
        ds.close()
        
        grid_area = None
        if 'kg m-2 s-1' in attrs.get('units','') or 'kg/m2/s' in attrs.get('units',''):
            if self.grid_area_calculator:
                grid_area = self.grid_area_calculator((lat.size, lon.size), 'm^2')
        
        for fname in sorted(files):
            year = int(os.path.basename(fname).split('.')[0])
            ds = xr.open_dataset(fname, decode_times=False)
            data = ds[emis_var].values
            
            for m in range(12):
                month_data = data[m, :, :]
                if grid_area is not None:
                    month_data *= grid_area
                month_data *= days_to_months(str(m+1).zfill(2), year) * DAYS_TO_SECONDS
                all_data.append(month_data)
                time_vals.append(year + m/12.0)
            ds.close()
        
        if grid_area is not None:
            attrs['units'] = 'kg/month'
        
        return xr.DataArray(np.array(all_data), dims=['time','lat','lon'],
                           coords={'time':time_vals, 'lat':lat, 'lon':lon},
                           attrs=attrs), lon, lat
    
    def _process_data(self, total_value, lon, lat, annual, spatial_aggregation='total', temporal_aggregation_map='annual_mean'):
        """
        Process loaded data into TimeSeriesData with smart temporal aggregation.
        
        Args:
            spatial_aggregation: For time series - 'total' or 'mean' over SPACE
            temporal_aggregation_map: For spatial maps - how to aggregate over TIME:
                'annual_mean': Mean of annual totals (for emissions)
                'period_mean': Mean over entire period (for flammability, temperature)
                'period_sum': Sum over entire period (rare - cumulative)
        """
        if total_value is None:
            return None
        
        try:
            units = total_value.attrs.get("units", "unknown")
            original_units = units  # Save original units for spatial data
            
            print(f"Processing data with units: {units}")
            
            # SPATIAL MAPS: Smart temporal aggregation
            if temporal_aggregation_map == 'annual_mean':
                # For emissions: mean of annual totals (sum 12 months per year, then average)
                print(f"Spatial maps: ANNUAL_MEAN (mean of yearly totals)")
                
                # Get time dimension
                times = total_value['time'].values
                n_times = len(times)
                
                # Determine if monthly data (assume 12 months/year)
                if n_times >= 12:
                    # Reshape into years (assumes complete years)
                    n_years = n_times // 12
                    n_complete_months = n_years * 12
                    
                    # Trim to complete years
                    complete_data = total_value.isel(time=slice(0, n_complete_months))
                    
                    # Reshape: (n_years, 12, lat, lon)
                    reshaped = complete_data.values.reshape(n_years, 12, *complete_data.shape[1:])
                    
                    # Sum over months (axis=1) to get annual totals, then mean over years (axis=0)
                    annual_totals = reshaped.sum(axis=1)  # (n_years, lat, lon)
                    time_mean_values = annual_totals.mean(axis=0)  # (lat, lon)
                    
                    print(f"  Calculated mean of {n_years} annual totals ({n_complete_months} months)")
                else:
                    # Less than 12 months - just use simple mean
                    time_mean_values = total_value.mean(dim="time", skipna=True).values
                    print(f"  Less than 12 months - using simple mean")
                
                # Create DataArray with proper coordinates
                time_mean = type(total_value)(
                    time_mean_values,
                    dims=total_value.dims[1:],  # Remove time dimension
                    coords={k: v for k, v in total_value.coords.items() if k != 'time'},
                    attrs=total_value.attrs
                )
                
            elif temporal_aggregation_map == 'period_sum':
                # Sum over entire period (rare - for cumulative quantities)
                time_mean = total_value.sum(dim="time", skipna=True)
                print(f"Spatial maps: PERIOD_SUM (cumulative total)")
                
            else:  # 'period_mean' (default)
                # Mean over entire period (for intensive quantities)
                time_mean = total_value.mean(dim="time", skipna=True)
                print(f"Spatial maps: PERIOD_MEAN (time-averaged)")
            
            # Preserve original per-area units in time_mean
            time_mean.attrs['units'] = original_units
            print(f"Spatial data (time_mean) units: {original_units}")
            
            
            # Vectorized spatial sum
            sum_dims = (total_value.dims[-2], total_value.dims[-1])
            
            # Check if we need to integrate over area
            # NOTE: After time conversion, units might be 'kg m-2 month-1' or 'kg m-2 yr-1'
            print(f"\nDEBUG - Checking for area integration:")
            print(f"  Units string: '{units}'")
            print(f"  Has ' m-2': {' m-2' in units}")
            print(f"  Has ' m^-2': {' m^-2' in units}")
            print(f"  Has '/m2': {'/m2' in units}")
            print(f"  Has '/m^2': {'/m^2' in units}")
            
            if " m-2" in units or " m^-2" in units or "/m2" in units or "/m^2" in units:
                if self.grid_area_calculator:
                    print("Applying grid area integration...")
                    grid_area = self.grid_area_calculator((total_value.shape[-2], total_value.shape[-1]), "m^2")
                    print(f"  Grid area shape: {grid_area.shape}")
                    print(f"  Grid area min: {grid_area.min():.6e}, max: {grid_area.max():.6e}")
                    print(f"  Total value shape: {total_value.values.shape}")
                    print(f"  Total value min: {total_value.values.min():.6e}, max: {total_value.values.max():.6e}")
                    
                    # Vectorized multiplication and sum
                    totals = np.einsum('tij,ij->t', total_value.values, grid_area)
                    print(f"  After einsum - Totals min: {totals.min():.6e}, max: {totals.max():.6e}")
                    
                    # Update units - remove the m-2 part
                    units = units.replace(' m-2', '').replace(' m^-2', '').replace('/m2', '').replace('/m^2', '')
                    print(f"  Updated units (for time_series): '{units}'")
                    print(f"  Spatial units (time_mean) remain: '{original_units}'")
                else:
                    print("  WARNING: No grid_area_calculator - summing without area weights")
                    totals = total_value.sum(dim=sum_dims).values
            else:
                print("Direct spatial sum (no area integration needed)")
                totals = total_value.sum(dim=sum_dims).values
            
            time_vals = total_value.coords["time"].values
            years = np.unique(np.floor(time_vals).astype(int))
            
            print(f"Time series has {len(totals)} points for {len(years)} years")
            print(f"Sample values: {totals[:3] if len(totals) > 0 else 'none'}")
            print(f"Max value: {np.max(totals) if len(totals) > 0 else 'none'}")
            
            # Fast annual aggregation using numpy
            if len(totals) > len(years) and annual:
                print(f"Aggregating from monthly to annual...")
                if len(totals) == len(years) * 12:
                    # Reshape and sum - fastest way
                    totals = totals.reshape(len(years), 12).sum(axis=1)
                    time_vals = years.astype(np.float64)
                else:
                    # Use bincount for efficient grouping
                    year_indices = np.floor(time_vals).astype(int) - years[0]
                    totals = np.bincount(year_indices, weights=totals)
                    time_vals = years.astype(np.float64)
                print(f"After aggregation: {len(totals)} annual values")
            
            return TimeSeriesData(
                time_mean=time_mean,
                time_series=np.column_stack([time_vals, totals]),
                longitude=lon,
                latitude=lat,
                units=units,
                start_year=int(years[0]),
                end_year=int(years[-1])
            )
        except Exception as e:
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            return None


class DatasetCombiner:
    """Combine multiple datasets for comparison."""
    
    @staticmethod
    def calculate_difference(data1: TimeSeriesData, data2: TimeSeriesData) -> Optional[TimeSeriesData]:
        """Calculate difference between two datasets."""
        if data1.units != data2.units:
            print(f"WARNING: Units don't match: {data1.units} vs {data2.units}")
        
        if data1.time_mean.shape != data2.time_mean.shape:
            print("ERROR: Spatial grids don't match")
            return None
        
        diff_spatial = data1.time_mean.copy()
        diff_spatial.values = data1.time_mean.values - data2.time_mean.values
        
        # Calculate time series difference at common points
        times1, times2 = data1.time_series[:, 0], data2.time_series[:, 0]
        common_times = []
        for t in times1:
            if np.any(np.abs(times2 - t) < 1e-6):
                common_times.append(t)
        
        if not common_times:
            print("ERROR: No common time points")
            return None
        
        diff_ts = np.zeros((len(common_times), 2))
        diff_ts[:, 0] = common_times
        for i, t in enumerate(common_times):
            idx1, idx2 = np.argmin(np.abs(times1 - t)), np.argmin(np.abs(times2 - t))
            diff_ts[i, 1] = data1.time_series[idx1, 1] - data2.time_series[idx2, 1]
        
        return TimeSeriesData(
            time_mean=diff_spatial,
            time_series=diff_ts,
            longitude=data1.longitude,
            latitude=data1.latitude,
            units=data1.units,
            start_year=max(data1.start_year, data2.start_year),
            end_year=min(data1.end_year, data2.end_year)
        )

def save_time_series_data_to_netcdf(
    data: TimeSeriesData,
    output_filepath: str,
    variable_name: str,
    time_units: str = 'decimal_years'
):
    """
    Saves the processed TimeSeriesData object (time series and spatial mean)
    to a new NetCDF file using xarray.

    Args:
        data: The TimeSeriesData object containing processed data.
        output_filepath: The full path where the NetCDF file will be saved.
        variable_name: The name to use for the time series variable in the file.
        time_units: Description of the time coordinate (e.g., 'decimal_years').
    """
    if not isinstance(data, TimeSeriesData):
        print("Error: Input is not a valid TimeSeriesData object. Cannot save.")
        return

    try:
        # 1. Prepare time series data (time, value)
        times = data.time_series[:, 0]
        values = data.time_series[:, 1]

        # 2. Create DataArray for the Time Series
        ts_da = xr.DataArray(
            values,
            coords={'time': times},
            dims=['time'],
            name=variable_name,
            attrs={'units': data.units, 'long_name': f'Spatially Aggregated {variable_name}', 'time_units': time_units}
        )

        # 3. Create Dataset containing both spatial mean and time series
        # data.time_mean is already an xr.DataArray with lat/lon coordinates
        ds = xr.Dataset(
            {
                variable_name: ts_da,
                'spatial_time_mean': data.time_mean
            },
            attrs={
                'title': f'Processed Data for {variable_name}',
                'time_range': f'{data.start_year} - {data.end_year}',
                'created_by': 'TimeSeriesWorkflow'
            }
        )

        # 4. Save to NetCDF
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output_filepath)

        print(f"\n✓ Successfully saved processed NetCDF file to: {output_filepath}")

    except Exception as e:
        print(f"\n❌ ERROR saving NetCDF file to {output_filepath}: {e}")
