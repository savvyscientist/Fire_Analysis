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
            'ModelE': lambda p,v,a,n,s,c: self._load_format(p,v,a,n,s, self._read_ModelE, False),
            'ModelE_Monthly': lambda p,v,a,n,s,c: self._load_format(p,v,a,n,s, self._read_ModelE, True),
            'Combined_ModelE': lambda p,v,a,n,s,c: self._load_combined_ModelE(c, v, a, n, s),
            'BA_GFED4': lambda p,v,a,n,s,c: self._load_gfed4sba_format(p,v,a,n,s, False),
            'BA_GFED4_upscale': lambda p,v,a,n,s,c: self._load_gfed4sba_format(p,v,a,n,s, True),
            'BA_GFED5': lambda p,v,a,n,s,c: self._load_gfed5ba_format(p,v,a,n,s, False),
            'BA_GFED5_upscale': lambda p,v,a,n,s,c: self._load_gfed5ba_format(p,v,a,n,s, True),
            'GFED4s_Monthly': lambda p,v,a,n,s,c: self._load_emis_format(p,v,a,n,s, True),
            'GFED5_Monthly': lambda p,v,a,n,s,c: self._load_emis_format(p,v,a,n,s, True),
            'FINN2.5_Monthly': lambda p,v,a,n,s,c: self._load_emis_format(p,v,a,n,s, True),
        }
    
    def load_time_series(self, folder_path: str, file_type: str, variables: List[str],
                         name: str = None, spatial_aggregation: str = None, annual: bool = False,
                         components: Optional[List[str]] = None):
        """Main entry point for loading data."""
        if file_type not in self._loaders:
            print(f"Unknown file type: {file_type}")
            return None
        try:
            return self._loaders[file_type](folder_path, variables, annual, name, spatial_aggregation, components)
        except Exception as e:
            print(f"Error loading {file_type}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_combined_ModelE(self, component_paths, variables, annual, name, spatial_aggregation):
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
        return self._process_data(combined_total_value, lon, lat, annual, spatial_aggregation)
    
    def _load_format(self, folder_path, variables, annual, name, spatial_aggregation, reader_func, monthly):
        """Generic loader for ModelE formats."""
        files = obtain_netcdf_files(folder_path)
        if not files:
            return None
        files.sort()
        total_value, lon, lat = reader_func(files, variables, monthly, name)
        return self._process_data(total_value, lon, lat, annual, spatial_aggregation)
    
    def _load_gfed4sba_format(self, folder_path, variables, annual, name, spatial_aggregation, upscaled):
        """Loader for GFED4s formats."""
        files = obtain_netcdf_files(folder_path)
        if not files:
            return None
        files.sort()
        total_value, lon, lat = self._read_gfed4sba(files, upscaled)
        return self._process_data(total_value, lon, lat, annual, spatial_aggregation)

    def _load_gfed5ba_format(self, folder_path, variables, annual, name, spatial_aggregation, upscaled):
        """Loader for GFED5 formats."""
        files = obtain_netcdf_files(folder_path)
        if not files:
            return None
        files.sort()
        total_value, lon, lat = self._read_gfed5ba(files, variables, upscaled)
        return self._process_data(total_value, lon, lat, annual, spatial_aggregation)
    
    def _load_emis_format(self, folder_path, variables, annual, name, spatial_aggregation):
        """Loader for emissions formats."""
        files = obtain_netcdf_files(folder_path)
        if not files:
            return None
        files.sort()
        total_value, lon, lat = self._read_modelEinput_emis(files, variables)
        return self._process_data(total_value, lon, lat, annual, spatial_aggregation)
    
    def _read_ModelE(self, files, variables, monthly=False, name=None):
        """Read ModelE NetCDF files - WITH PROPER UNIT CONVERSION."""
        print(f"\n=== Reading ModelE data ===")
        print(f"Number of files: {len(files)}")
        print(f"Monthly: {monthly}")
        print(f"Variables: {variables}")
        
        # Pre-allocate arrays for speed
        first_ds = xr.open_dataset(files[0])
        longitude, latitude = first_ds['lon'].values, first_ds['lat'].values
        lat_size, lon_size = len(latitude), len(longitude)
        
        n_files = len(files)
        time_values = np.zeros(n_files, dtype=np.float64)
        
        # Get attributes and units from first file
        scaling_factor = 1.0
        attribute_dict = {}
        original_units = ''
        for var in variables:
            if var in first_ds.variables:
                attribute_dict = dict(first_ds[var].attrs)
                original_units = attribute_dict.get('units', '')
                scaling_factor, clean_units = extract_scaling_factor(original_units)
                attribute_dict['units'] = clean_units
                print(f"Original units: {original_units}")
                print(f"Scaling factor: {scaling_factor}")
                print(f"Clean units: {clean_units}")
                break
        first_ds.close()
        
        # Will store converted data (with time scaling applied)
        converted_data_list = []
        
        # Process files - apply ALL conversions INCLUDING time scaling
        for idx, fpath in enumerate(files):
            try:
                ds = xr.open_dataset(fpath, decode_times=False)
                
                # Sum variables
                data = np.zeros((lat_size, lon_size), dtype=np.float32)
                for var in variables:
                    if var in ds.variables:
                        var_data = ds[var].values
                        np.maximum(var_data, 0.0, out=var_data)
                        data += var_data
                
                # Apply scaling factor
                data = data * scaling_factor
                
                # Extract year and month FIRST
                if monthly:
                    year = int(fpath.split(".")[0][-4:])
                    month = fpath.split(".")[0][-7:-4]
                    month_num = MONTHLIST.index(month) + 1
                    time_values[idx] = year + (month_num - 1) / 12.0
                else:
                    year = int(fpath.split("ANN")[1][:4]) if "ANN" in fpath else 2009 + idx
                    time_values[idx] = year
                
                # CRITICAL: Apply time scaling BEFORE spatial sum (like original code)
                if 's-1' in original_units or '/s' in original_units:
                    if monthly:
                        # Calculate seconds in this specific month
                        days_in_month = days_to_months(str(month_num).zfill(2), year)
                        seconds_in_month = days_in_month * DAYS_TO_SECONDS
                        
                        # Convert from /s to /month
                        data = data * seconds_in_month
                        
                        if idx == 0:
                            print(f"\nTime conversion applied:")
                            print(f"  Month: {month} ({month_num})")
                            print(f"  Days in month: {days_in_month}")
                            print(f"  Seconds in month: {seconds_in_month:.2e}")
                            print(f"  Sample value before: {data[45, 72]:.6e}")
                    else:
                        # Convert from /s to /year
                        data = data * SECONDS_IN_A_YEAR
                        if idx == 0:
                            print(f"\nAnnual time conversion: multiplying by {SECONDS_IN_A_YEAR:.2e}")
                
                if idx == 0:
                    print(f"  Sample value after time conversion: {data[45, 72]:.6e}")
                    print(f"  Data min: {np.min(data):.6e}, max: {np.max(data):.6e}")
                
                converted_data_list.append(data)
                ds.close()
                
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Stack all converted data
        all_data = np.array(converted_data_list)
        
        print(f"\nFinal data shape: {all_data.shape}")
        print(f"Time values: {time_values[:3]}... (showing first 3)")
        
        # Update units to reflect time conversion
        if 's-1' in original_units or '/s' in original_units:
            if monthly:
                attribute_dict['units'] = attribute_dict['units'].replace('/s', '/month').replace('s-1', 'month-1')
            else:
                attribute_dict['units'] = attribute_dict['units'].replace('/s', '/yr').replace('s-1', 'yr-1')
        
        print(f"Final units: {attribute_dict['units']}")
        
        return xr.DataArray(
            all_data, 
            dims=['time','lat','lon'],
            coords={'time':time_values, 'lat':latitude, 'lon':longitude},
            attrs=attribute_dict
        ), longitude, latitude
    
    def _read_gfed5ba(self, files, variables, upscaled=False):
        """Read GFED5 NetCDF files - OPTIMIZED."""
        variables = variables if variables else ["Total"]
        
        # Pre-allocate for speed
        n_files = len(files)
        
        # Read first file to get dimensions
        with Dataset(files[0]) as ds:
            var = variables[0] if variables[0] in ds.variables else list(ds.variables.keys())[0]
            arr = ds.variables[var][:]
            arr = arr[0] if len(arr.shape) > 2 else arr
            h, w = arr.shape
            
            # Pre-allocate output array
            data_list = np.zeros((n_files, h, w), dtype=np.float32)
            time_vals = np.zeros(n_files, dtype=np.float64)
            
            lats = np.linspace(-90, 90, h)
            lons = np.linspace(-180, 180, w)
            
            # Get attributes once
            attrs = {a: getattr(ds.variables[var], a) for a in ds.variables[var].ncattrs()}
            attrs["units"] = "m^2"
        
        # Process files efficiently
        scale_factor = 1.0 if upscaled else KM_SQUARED_TO_M_SQUARED
        
        for idx, fpath in enumerate(files):
            try:
                with Dataset(fpath) as ds:
                    month_data = np.zeros((h, w), dtype=np.float32)
                    
                    for var in variables:
                        if var in ds.variables:
                            arr = ds.variables[var][:]
                            arr = arr[0] if len(arr.shape) > 2 else arr
                            month_data += arr.astype(np.float32)
                    
                    month_data *= scale_factor
                    data_list[idx] = month_data
                    
                    # Extract time
                    match = re.search(r'BA(\d{6})', os.path.basename(fpath))
                    if match:
                        ym = match.group(1)
                        time_vals[idx] = int(ym[:4]) + (int(ym[4:6])-1)/12.0
                    else:
                        time_vals[idx] = 2001.0 + idx
                        
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        return xr.DataArray(
            data_list, 
            dims=['time','latitude','longitude'],
            coords={'time':time_vals, 'latitude':lats, 'longitude':lons},
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
    
    def _process_data(self, total_value, lon, lat, annual, spatial_aggregation='mean'):
        """Process loaded data into TimeSeriesData - FIXED UNIT HANDLING."""
        if total_value is None:
            return None
        
        try:
            # Calculate spatial aggregation (mean or total)
            if spatial_aggregation == 'total':
                time_mean = total_value.sum(dim="time", skipna=True)
                print(f"Using TOTAL over time for spatial maps")
            else:  # default to mean
                time_mean = total_value.mean(dim="time", skipna=True)
                print(f"Using MEAN over time for spatial maps")
            
            units = total_value.attrs.get("units", "unknown")
            
            print(f"Processing data with units: {units}")
            
            # Vectorized spatial sum
            sum_dims = (total_value.dims[-2], total_value.dims[-1])
            
            # Check if we need to integrate over area
            # NOTE: After time conversion, units might be 'kg m-2 month-1' or 'kg m-2 yr-1'
            if " m-2" in units or " m^-2" in units or "/m2" in units or "/m^2" in units:
                if self.grid_area_calculator:
                    print("Applying grid area integration...")
                    grid_area = self.grid_area_calculator((total_value.shape[-2], total_value.shape[-1]), "m^2")
                    # Vectorized multiplication and sum
                    totals = np.einsum('tij,ij->t', total_value.values, grid_area)
                    # Update units - remove the m-2 part
                    units = units.replace(' m-2', '').replace(' m^-2', '').replace('/m2', '').replace('/m^2', '')
                else:
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
