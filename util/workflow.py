"""
Enhanced workflow with proper handling of per-area vs integrated units.

Key concept:
- Time series: Integrated totals (Tg CO2/month) - sum over all grid cells
- Spatial maps: Flux density (kg CO2/m²/month) - per-area values

This properly handles the physical meaning of the data.
"""
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from config import ConfigManager, TimeAnalysisConfig, FolderConfig
from data_loader import DataLoader, TimeSeriesData, save_time_series_data_to_netcdf
from analysis import TimeSeriesAnalyzer
from visualization import TimeSeriesPlotter, SpatialPlotter, PlotStyle
from grid_utils import GridAreaCalculator
from unit_converter import UnitConverter, convert_to_target_units, parse_units


class TimeSeriesWorkflow:
    """
    Orchestrates complete time series analysis workflow.
    
    Properly handles:
    - Time series: Spatially integrated totals (e.g., Tg CO2/month)
    - Spatial maps: Per-area flux (e.g., kg CO2/m²/month)
    """
    
    def __init__(self, config: TimeAnalysisConfig):
        """Initialize workflow with configuration."""
        self.config = config
        
        # Initialize components
        self.grid_calculator = GridAreaCalculator()
        self.unit_converter = UnitConverter(
            grid_area_calculator=self.grid_calculator.calculate
        )
        self.data_loader = DataLoader(grid_area_calculator=self.grid_calculator.calculate)
        self.analyzer = TimeSeriesAnalyzer()
        self.ts_plotter = TimeSeriesPlotter()
        self.spatial_plotter = SpatialPlotter()
        
        # Create output directory
        self.output_dir = Path(config.figs_folder)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self):
        """Execute complete workflow."""
        print("=" * 60)
        print("Starting Time Series Analysis Workflow")
        print("=" * 60)
        
        # Load all datasets in NATIVE units
        # Time series will have INTEGRATED values (area already applied)
        # Spatial maps (time_mean) will still be per-area if original data was per-area
        datasets = self._load_all_datasets()
        
        if not datasets:
            print("ERROR: No datasets loaded successfully")
            return
        
        print(f"\nSuccessfully loaded {len(datasets)} datasets")
        
        # Create visualizations
        self._create_time_series_plots(datasets)
        self._create_seasonal_plots(datasets)
        
        if not self.config.annual:
            self._create_annual_total_plots(datasets)
        
        self._create_spatial_maps(datasets)
        
        print("\n" + "=" * 60)
        print("Workflow completed successfully")
        print("=" * 60)
    
    def _load_all_datasets(self) -> List[tuple]:
        """
        Load all configured datasets.
        
        IMPORTANT: After loading:
        - time_series: Already spatially integrated (NO m-2 in units)
        - time_mean: Still PER-AREA if original data was per-area (HAS m-2 in units)
        
        Returns:
            List of (TimeSeriesData, FolderConfig) tuples
        """
        datasets = []
        
        for folder_config in self.config.folders:
            print(f"\n{'='*60}")
            print(f"Loading: {folder_config.figure_data.label}")
            print(f"{'='*60}")
            print(f"  Path: {folder_config.folder_path}")
            print(f"  Type: {folder_config.file_type}")
            print(f"  Variables: {folder_config.variables}")
            print(f"  Spatial Aggregation: {folder_config.spatial_aggregation}")
            
            data = self.data_loader.load_time_series(
                folder_path=folder_config.folder_path,
                file_type=folder_config.file_type,
                variables=folder_config.variables,
                annual=self.config.annual,
                spatial_aggregation=folder_config.spatial_aggregation,
                components=folder_config.components
            )
            
            if data is not None:
                print(f"  ✓ Data loaded successfully")
                print(f"  Time series units: {data.units}")
                print(f"  Time series range: min={data.time_series[:, 1].min():.3e}, max={data.time_series[:, 1].max():.3e}")
                
                # Determine spatial units from time_mean attributes
                spatial_units = data.time_mean.attrs.get('units', data.units)
                print(f"  Spatial data units: {spatial_units}")
                
                # Save both in NATIVE units
                label = folder_config.figure_data.label.replace(' ', '_').replace('/', '_')
                variable = folder_config.variables[0] if folder_config.variables else 'data' 
                output_filename = f"{label}_{variable}_native.nc"
                output_filepath = self.output_dir / output_filename
                
                print(f"  Saving data to NetCDF: {output_filename}")
                save_time_series_data_to_netcdf(
                    data=data,
                    output_filepath=str(output_filepath),
                    variable_name=variable
                )
                
                datasets.append((data, folder_config))
                print(f"  ✓ Processing complete")
            else:
                print(f"  ✗ Failed to load")
        
        return datasets
    
    def _convert_for_time_series(
        self, 
        data: TimeSeriesData, 
        config: FolderConfig
    ) -> Tuple[np.ndarray, str]:
        """
        Convert time series data to target units for plotting.
        
        Time series should be INTEGRATED TOTALS (e.g., Tg CO2/month).
        
        Args:
            data: TimeSeriesData object (time_series already integrated)
            config: FolderConfig with target_units
        
        Returns:
            Tuple of (converted_time_series, units_string)
        """
        target_units = config.figure_data.target_units
        
        # If no target specified, use native units
        if not target_units:
            return data.time_series, data.units
        
        # If target same as current, no conversion needed
        if target_units == data.units:
            return data.time_series, data.units
        
        try:
            print(f"    Converting '{config.figure_data.label}' time series:")
            print(f"      From: {data.units}")
            print(f"      To:   {target_units}")
            
            # Convert values (should NOT have m-2 component)
            if 'm-2' in data.units or 'm^-2' in data.units:
                print(f"      ⚠️  WARNING: Time series still has per-area units!")
                print(f"          This suggests area integration didn't happen in data_loader")
            
            converted_values, new_units = convert_to_target_units(
                data.time_series[:, 1],
                data.units,
                target_units
            )
            
            # Create new time series array with converted values
            converted_ts = data.time_series.copy()
            converted_ts[:, 1] = converted_values
            
            print(f"      Result: {new_units}")
            print(f"      Range: min={converted_values.min():.3e}, max={converted_values.max():.3e}")
            
            return converted_ts, new_units
            
        except Exception as e:
            print(f"    ❌ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"    Using native units instead")
            return data.time_series, data.units
    
    def _prepare_spatial_for_maps(
        self, 
        data: TimeSeriesData, 
        config: FolderConfig
    ) -> Tuple[np.ndarray, str]:
        """
        Prepare spatial data for mapping with proper per-area units.
        
        Maps should show FLUX DENSITY (e.g., kg CO2/m²/month).
        
        Strategy:
        1. If data.time_mean has original per-area units → just convert mass unit
        2. If data.time_mean was integrated → divide by grid area to get per-area
        
        Args:
            data: TimeSeriesData object
            config: FolderConfig
        
        Returns:
            Tuple of (spatial_data_per_area, units_string)
        """
        # Get the spatial data units from the xarray attrs
        spatial_data = data.time_mean.values
        
        # Check if time_mean still has original per-area units
        # (This would be the case if data_loader didn't integrate spatial for maps)
        time_mean_units = data.time_mean.attrs.get('units', data.units)
        
        print(f"    Preparing spatial data for '{config.figure_data.label}':")
        print(f"      Spatial data units: {time_mean_units}")
        print(f"      Time series units: {data.units}")
        
        # Determine if spatial data is per-area or integrated
        is_per_area = ('m-2' in time_mean_units or 'm^-2' in time_mean_units or 
                      '/m2' in time_mean_units or '/m^2' in time_mean_units)
        
        if is_per_area:
            # Data is already per-area - just convert mass prefix if needed
            print(f"      Data is already per-area ✓")
            
            # Target for maps: "kg CO2/m²/month" or similar
            # Convert just the mass component (e.g., kg → kg, or g → kg)
            target_map_units = "kg CO2/m²/month"  # Standard for emissions maps
            
            try:
                converted_spatial, new_units = convert_to_target_units(
                    spatial_data,
                    time_mean_units,
                    target_map_units
                )
                print(f"      Converted to: {new_units}")
                return converted_spatial, new_units
            except Exception as e:
                print(f"      ⚠️  Using original units: {time_mean_units}")
                return spatial_data, time_mean_units
        
        else:
            # Data was integrated - need to divide by grid area
            print(f"      Data was integrated - converting back to per-area...")
            
            # Get grid area
            grid_area = self.grid_calculator.calculate(
                grid_area_shape=spatial_data.shape,
                units='m^2'
            )
            
            # Divide by area to get per-area flux
            spatial_per_area = spatial_data / grid_area
            
            # Update units to include m-2
            per_area_units = time_mean_units + "/m²"
            print(f"      Per-area units: {per_area_units}")
            
            # Now convert mass prefix if needed
            target_map_units = "kg CO2/m²/month"
            
            try:
                converted_spatial, new_units = convert_to_target_units(
                    spatial_per_area,
                    per_area_units,
                    target_map_units
                )
                print(f"      Final units: {new_units}")
                return converted_spatial, new_units
            except Exception as e:
                print(f"      ⚠️  Using per-area without mass conversion: {per_area_units}")
                return spatial_per_area, per_area_units
    
    def _create_time_series_plots(self, datasets: List[tuple]):
        """Create time series plots with integrated totals."""
        print("\n" + "-" * 60)
        print("Creating time series plots (integrated totals)...")
        
        plot_data = []
        target_ylabel = None
        
        for data, config in datasets:
            # Convert time series to target units (e.g., Tg CO2/month)
            converted_ts, converted_units = self._convert_for_time_series(data, config)
            
            # Track the first target units for ylabel
            if target_ylabel is None and config.figure_data.target_units:
                target_ylabel = config.figure_data.target_units
            
            style = PlotStyle(
                color=config.figure_data.color,
                marker=config.figure_data.marker,
                linestyle=config.figure_data.line_style,
                label=config.figure_data.label
            )
            
            # Create a temporary data object with converted values
            temp_data = TimeSeriesData(
                time_mean=data.time_mean,
                time_series=converted_ts,
                longitude=data.longitude,
                latitude=data.latitude,
                units=converted_units,
                start_year=data.start_year,
                end_year=data.end_year
            )
            
            plot_data.append((temp_data, style))
        
        save_path = self.output_dir / f"{self.config.title}_timeseries.png"
        
        # Use target ylabel if any dataset specified it, otherwise use config
        ylabel = target_ylabel if target_ylabel else self.config.ylabel
        
        self.ts_plotter.plot_time_series(
            datasets=plot_data,
            title=f"{self.config.title} Time Series",
            ylabel=ylabel,
            xlabel='Year' if self.config.annual else 'Time',
            save_path=str(save_path)
        )
        
        print(f"  ✓ Time series plot saved to: {save_path}")
    
    def _create_seasonal_plots(self, datasets: List[tuple]):
        """Create seasonal cycle plots."""
        if self.config.annual:
            print("\nSkipping seasonal plots (annual data)")
            return
        
        print("\n" + "-" * 60)
        print("Creating seasonal cycle plots...")
        
        seasonal_data = []
        target_ylabel = None
        
        for data, config in datasets:
            # Convert time series for seasonal analysis
            converted_ts, converted_units = self._convert_for_time_series(data, config)
            
            if target_ylabel is None and config.figure_data.target_units:
                target_ylabel = config.figure_data.target_units
            
            # Calculate seasonal stats on converted data
            stats = self.analyzer.calculate_seasonal_cycle(
                converted_ts,
                annual=self.config.annual
            )
            
            style = PlotStyle(
                color=config.figure_data.color,
                marker=config.figure_data.marker,
                linestyle=config.figure_data.line_style,
                label=config.figure_data.label
            )
            seasonal_data.append((stats, style))
        
        save_path = self.output_dir / f"{self.config.title}_seasonal.png"
        ylabel = target_ylabel if target_ylabel else self.config.ylabel
        
        self.ts_plotter.plot_seasonal_cycle(
            seasonal_data_list=seasonal_data,
            title=f"{self.config.title} Seasonal Cycle",
            ylabel=ylabel,
            save_path=str(save_path)
        )
        
        print(f"  ✓ Seasonal plot saved to: {save_path}")
    
    def _create_annual_total_plots(self, datasets: List[tuple]):
        """Create annual total plots."""
        print("\n" + "-" * 60)
        print("Creating annual total plots...")
        
        annual_data = []
        target_ylabel = None
        
        for data, config in datasets:
            # Convert time series for annual analysis
            converted_ts, converted_units = self._convert_for_time_series(data, config)
            
            if target_ylabel is None and config.figure_data.target_units:
                target_ylabel = config.figure_data.target_units
            
            # Calculate annual totals on converted data
            totals = self.analyzer.calculate_annual_totals(converted_ts)
            
            if totals is not None:
                style = PlotStyle(
                    color=config.figure_data.color,
                    marker=config.figure_data.marker,
                    linestyle=config.figure_data.line_style,
                    label=config.figure_data.label
                )
                annual_data.append((totals, style))
        
        if annual_data:
            save_path = self.output_dir / f"{self.config.title}_annual.png"
            
            ylabel = target_ylabel if target_ylabel else self.config.ylabel
            if ylabel and '/month' in ylabel:
                ylabel = ylabel.replace('/month', '/yr')
            
            self.ts_plotter.plot_annual_totals(
                annual_data_list=annual_data,
                title=f"{self.config.title} Annual Totals",
                ylabel=ylabel,
                save_path=str(save_path)
            )
            
            print(f"  ✓ Annual totals plot saved to: {save_path}")
        else:
            print("  No annual data to plot")
    
    def _create_spatial_maps(self, datasets: List[tuple]):
        """Create spatial distribution maps with per-area flux."""
        print("\n" + "-" * 60)
        print("Creating spatial maps (per-area flux)...")
        
        # Use first dataset's coordinates
        lon = datasets[0][0].longitude
        lat = datasets[0][0].latitude
        
        # Create individual maps
        for i, (data, config) in enumerate(datasets):
            # Prepare spatial data as per-area flux (e.g., kg CO2/m²/month)
            spatial_per_area, spatial_units = self._prepare_spatial_for_maps(data, config)
            
            save_path = self.output_dir / f"{self.config.title}_map_{i+1}.png"
            
            print(f"    Plotting '{config.figure_data.label}'")
            print(f"      Units: {spatial_units}")
            print(f"      Range: min={spatial_per_area.min():.3e}, max={spatial_per_area.max():.3e}")
            
            self.spatial_plotter.plot_spatial_map(
                data=spatial_per_area,
                lon=lon,
                lat=lat,
                title=f"{config.figure_data.label} - Mean {self.config.title}",
                cbar_label=spatial_units,
                vmax=config.figure_data.cbarmax,  # Use config cbarmax if specified
                save_path=str(save_path)
            )
        
        # Create multi-panel comparison
        if len(datasets) > 1:
            converted_datasets = []
            common_units = None
            units_match = True
            
            for data, config in datasets:
                spatial_per_area, spatial_units = self._prepare_spatial_for_maps(data, config)
                converted_datasets.append((spatial_per_area, config.figure_data.label))
                
                if common_units is None:
                    common_units = spatial_units
                elif common_units != spatial_units:
                    units_match = False
                    print(f"  ⚠️  Warning: Units don't match for comparison map")
            
            if units_match:
                save_path = self.output_dir / f"{self.config.title}_comparison.png"
                
                self.spatial_plotter.create_multi_panel_map(
                    datasets=converted_datasets,
                    lon=lon,
                    lat=lat,
                    overall_title=f"{self.config.title} Comparison",
                    cbar_label=common_units,
                    save_path=str(save_path)
                )
                
                print(f"  ✓ Comparison map saved to: {save_path}")


class WorkflowRunner:
    """Main runner for all analysis workflows."""
    
    def __init__(self, config_file: str = "utilityEnvVar.json"):
        self.config_manager = ConfigManager(config_file)
    
    def run(self):
        """Execute all selected workflows."""
        print("\n" + "=" * 60)
        print("WORKFLOW RUNNER - Proper Per-Area vs Integrated Units")
        print("=" * 60)
        
        self.config_manager.load()
        scripts = self.config_manager.get_selected_scripts()
        
        print(f"\nSelected scripts: {scripts}")
        
        for script_name in scripts:
            self._run_script(script_name)
    
    def _run_script(self, script_name: str):
        """Run a specific script."""
        print("\n" + "=" * 60)
        print(f"Running script: {script_name}")
        print("=" * 60)
        
        if script_name == "time_analysis_version_two":
            config = self.config_manager.get_time_analysis_config()
            if config:
                workflow = TimeSeriesWorkflow(config)
                workflow.run()
            else:
                print(f"ERROR: Configuration not found for {script_name}")
        else:
            print(f"ERROR: Unknown script: {script_name}")


def main():
    """Main entry point."""
    runner = WorkflowRunner()
    runner.run()


if __name__ == "__main__":
    main()
