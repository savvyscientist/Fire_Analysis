This repository handles various geospatial data processing tasks, primarily focusing on:
1. Reading and processing Earth system data from different formats (NetCDF, HDF5)
2. Unit conversion and standardization
3. Spatial calculations (grid cell areas, global totals)
4. Time series analysis
5. Visualization using matplotlib and cartopy

## Key Components

1. **Data Reading Functions**:
   - `read_gfed4s`, `read_gfed5`: Read Global Fire Emissions Database data
   - `read_ModelE`: Read model output data
   - `read_lightning_data`: Read lightning strike data
   - `read_gfed4s_emis`: Read emissions data

2. **Unit Handling**:
   - `extract_scaling_factor`: Parse unit strings with scaling factors
   - `handle_units`: Apply appropriate scaling based on data units
   - `calculate_grid_area`: Calculate grid cell areas for spatial weighting

3. **Visualization**:
   - `define_subplot`: Set up map visualization with proper styling
   - `map_plot`: Generate geographical maps with appropriate color scales
   - `time_series_plot`: Plot time series data

4. **Analysis Workflows**:
   - `obtain_time_series_xarray`: Process data into time series format
   - `run_time_series_analysis`: Coordinate analysis of multiple datasets
   - `run_time_series_diff_analysis`: Compare two datasets
