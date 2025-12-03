def read_gfed5(files, upscaled=False, variable_name="Total"):
    """
    Reads multiple netCDF files using netCDF4.Dataset, preserves monthly burned area data,
    and returns the data as xarray.DataArray.
    
    Parameters:
    -----------
    files : list
        List of paths to GFED5 netCDF files
    upscaled : bool, optional
        If True, data is already in m^2; if False, convert from km^2 to m^2 (default: False)
    variable_name : str, optional
        Name of the variable to read (default: "Total")
        
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
                # Check if the requested variable exists
                if variable_name not in netcdf_dataset.variables:
                    print(f"Warning: Variable '{variable_name}' not found in {file}")
                    print(f"Available variables: {list(netcdf_dataset.variables.keys())}")
                    continue
                
                # Read the specified variable from the netCDF dataset
                # dimensions expected: (time, latitude, longitude) or (latitude, longitude)
                var_data_array = netcdf_dataset.variables[variable_name][:]
                
                # Convert units from km^2 to m^2 if not already upscaled
                var_data_array = (
                    var_data_array if upscaled else var_data_array * KM_SQUARED_TO_M_SQUARED
                )

                # Handle different array dimensions
                # If the array has a time dimension, extract the first time slice
                monthly_burned_area = (
                    var_data_array[0] if len(var_data_array.shape) > 2 else var_data_array
                )

                # Get coordinate information from the first file
                if file_idx == 0:
                    # Copy attributes from the specified variable
                    if variable_name in netcdf_dataset.variables:
                        for attr_name in netcdf_dataset.variables[variable_name].ncattrs():
                            attribute_dict[attr_name] = getattr(
                                netcdf_dataset.variables[variable_name], attr_name
                            )
                    else:
                        # Fallback to "Total" variable if specified variable doesn't exist
                        if "Total" in netcdf_dataset.variables:
                            for attr_name in netcdf_dataset.variables["Total"].ncattrs():
                                attribute_dict[attr_name] = getattr(
                                    netcdf_dataset.variables["Total"], attr_name
                                )
                    
                    # Update the units to match the upscaling process
                    attribute_dict["units"] = "m^2"

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
                    
                    # Create a decimal year value for this month (e.g., 2009.0 for Jan, 2009.083 for Feb)
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
                        # Use file index as fallback time value
                        time_values.append(2001.0 + file_idx)

                # Store the monthly burned area data
                monthly_data_list.append(monthly_burned_area)
                
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    # Check if we have any valid data
    if not monthly_data_list:
        raise ValueError("No valid data found in any of the files")

    if latitudes is None or longitudes is None:
        raise ValueError("Could not determine coordinate information from files")

    # Convert lists to arrays
    monthly_data_array = np.array(monthly_data_list)
    time_values = np.array(time_values)
    
    # Sort by time if needed
    if len(time_values) > 1:
        sort_idx = np.argsort(time_values)
        monthly_data_array = monthly_data_array[sort_idx]
        time_values = time_values[sort_idx]

    # Create xarray DataArray with time dimension preserved
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
    print(f"Time range: {time_values.min():.3f} to {time_values.max():.3f}")
    print(f"Data shape: {total_data_array.shape}")

    return total_data_array, longitudes, latitudes
