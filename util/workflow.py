"""
Main workflow orchestrator for time series analysis.
UPDATED: Supports target unit conversion and spatial aggregation settings
"""
from pathlib import Path
from typing import List, Optional
import numpy as np

from config import ConfigManager, TimeAnalysisConfig, FolderConfig
from data_loader import DataLoader, TimeSeriesData, save_time_series_data_to_netcdf
from analysis import TimeSeriesAnalyzer
from visualization import TimeSeriesPlotter, SpatialPlotter, PlotStyle
from grid_utils import GridAreaCalculator
from unit_converter import UnitConverter, convert_to_target_units


class TimeSeriesWorkflow:
    """Orchestrates complete time series analysis workflow."""
    
    def __init__(self, config: TimeAnalysisConfig):
        """
        Initialize workflow with configuration.
        
        Args:
            config: TimeAnalysisConfig object
        """
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
        
        # Track final units for ylabel
        self.final_units = None
    
    def run(self):
        """Execute complete workflow."""
        print("=" * 60)
        print("Starting Time Series Analysis Workflow")
        print("=" * 60)
        
        # Load all datasets
        datasets = self._load_all_datasets()
        
        if not datasets:
            print("ERROR: No datasets loaded successfully")
            return
        
        print(f"\nSuccessfully loaded {len(datasets)} datasets")
        
        # Determine final units from first dataset with target_units
        self._determine_final_units(datasets)
        
        # Create visualizations
        self._create_time_series_plots(datasets)
        self._create_seasonal_plots(datasets)
        
        if not self.config.annual:
            self._create_annual_total_plots(datasets)
        
        self._create_spatial_maps(datasets)
        
        print("\n" + "=" * 60)
        print("Workflow completed successfully")
        print("=" * 60)
    
    def _determine_final_units(self, datasets: List[tuple]):
        """Determine the final units to use in ylabel."""
        # Look for first dataset with target_units specified
        for data, config in datasets:
            if config.figure_data.target_units:
                self.final_units = config.figure_data.target_units
                print(f"\nUsing target units for ylabel: {self.final_units}")
                break
        
        # If no target units, use the units from first dataset
        if not self.final_units:
            self.final_units = datasets[0][0].units
            print(f"\nUsing data units for ylabel: {self.final_units}")
    
    def _load_all_datasets(self) -> List[tuple]:
        """
        Load all configured datasets with unit conversion.
        
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
            
            if folder_config.figure_data.target_units:
                print(f"  Target Units: {folder_config.figure_data.target_units}")
            
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
                print(f"  Initial units: {data.units}")
                
                # Apply target unit conversion if specified
                if folder_config.figure_data.target_units:
                    print(f"  Converting to: {folder_config.figure_data.target_units}")
                    
                    # Convert time series values
                    converted_values, new_units = convert_to_target_units(
                        data.time_series[:, 1],
                        data.units,
                        folder_config.figure_data.target_units
                    )
                    data.time_series[:, 1] = converted_values
                    
                    # Convert spatial data
                    converted_spatial, _ = convert_to_target_units(
                        data.time_mean.values,
                        data.units,
                        folder_config.figure_data.target_units
                    )
                    data.time_mean.values = converted_spatial
                    
                    # Update units
                    data.units = new_units
                    print(f"  Final units: {data.units}")
                # Prepare filename from configuration
                label = folder_config.figure_data.label.replace(' ', '_').replace('/', '_')
                # Use the first variable name for the output file
                variable = folder_config.variables[0] if folder_config.variables else 'data' 
            
                output_filename = f"{label}_{variable}_processed.nc"
                output_filepath = self.output_dir / output_filename
            
                print(f"  Saving processed data to NetCDF: {output_filename}")
             
                # Call the new save function (must be defined in data_loader.py)
                save_time_series_data_to_netcdf(
                    data=data,
                    output_filepath=str(output_filepath),
                    variable_name=variable) 
                datasets.append((data, folder_config))
                print(f"  ✓ Processing complete")
            else:
                print(f"  ✗ Failed to load")
        
        return datasets
    
    def _create_time_series_plots(self, datasets: List[tuple]):
        """Create time series plots."""
        print("\n" + "-" * 60)
        print("Creating time series plots...")
        
        plot_data = []
        for data, config in datasets:
            style = PlotStyle(
                color=config.figure_data.color,
                marker=config.figure_data.marker,
                linestyle=config.figure_data.line_style,
                label=config.figure_data.label
            )
            plot_data.append((data, style))
        
        save_path = self.output_dir / f"{self.config.title}_timeseries.png"
        
        # Use final_units if determined, otherwise use config ylabel
        ylabel = self.final_units if self.final_units else self.config.ylabel
        
        self.ts_plotter.plot_time_series(
            datasets=plot_data,
            title=f"{self.config.title} Time Series",
            ylabel=ylabel,
            xlabel='Year' if self.config.annual else 'Time',
            save_path=str(save_path)
        )
        
        print(f"Time series plot saved to: {save_path}")
    
    def _create_seasonal_plots(self, datasets: List[tuple]):
        """Create seasonal cycle plots."""
        if self.config.annual:
            print("\nSkipping seasonal plots (annual data)")
            return
        
        print("\n" + "-" * 60)
        print("Creating seasonal cycle plots...")
        
        seasonal_data = []
        for data, config in datasets:
            stats = self.analyzer.calculate_seasonal_cycle(
                data.time_series,
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
        
        # Use final_units if determined
        ylabel = self.final_units if self.final_units else self.config.ylabel
        
        self.ts_plotter.plot_seasonal_cycle(
            seasonal_data_list=seasonal_data,
            title=f"{self.config.title} Seasonal Cycle",
            ylabel=ylabel,
            save_path=str(save_path)
        )
        
        print(f"Seasonal plot saved to: {save_path}")
    
    def _create_annual_total_plots(self, datasets: List[tuple]):
        """Create annual total plots."""
        print("\n" + "-" * 60)
        print("Creating annual total plots...")
        
        annual_data = []
        for data, config in datasets:
            totals = self.analyzer.calculate_annual_totals(data.time_series)
            
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
            
            # Use final_units if determined, but adjust for annual
            ylabel = self.final_units if self.final_units else self.config.ylabel
            if ylabel and '/month' in ylabel:
                ylabel = ylabel.replace('/month', '/yr')
            
            self.ts_plotter.plot_annual_totals(
                annual_data_list=annual_data,
                title=f"{self.config.title} Annual Totals",
                ylabel=ylabel,
                save_path=str(save_path)
            )
            
            print(f"Annual totals plot saved to: {save_path}")
        else:
            print("No annual data to plot")
    
    def _create_spatial_maps(self, datasets: List[tuple]):
        """Create spatial distribution maps."""
        print("\n" + "-" * 60)
        print("Creating spatial maps...")
        
        map_datasets = []
        for data, config in datasets:
            map_datasets.append((
                data.time_mean.values,
                config.figure_data.label
            ))
        
        # Use first dataset's coordinates
        lon = datasets[0][0].longitude
        lat = datasets[0][0].latitude
        
        # Use final_units for colorbar label
        cbar_label = self.final_units if self.final_units else datasets[0][0].units
        
        # Create individual maps
        for i, (data, config) in enumerate(datasets):
            save_path = self.output_dir / f"{self.config.title}_map_{i+1}.png"
            
            self.spatial_plotter.plot_spatial_map(
                data=data.time_mean.values,
                lon=lon,
                lat=lat,
                title=f"{config.figure_data.label} - Mean {self.config.title}",
                cbar_label=cbar_label,
                save_path=str(save_path)
            )
        
        # Create multi-panel comparison
        if len(map_datasets) > 1:
            save_path = self.output_dir / f"{self.config.title}_comparison.png"
            
            self.spatial_plotter.create_multi_panel_map(
                datasets=map_datasets,
                lon=lon,
                lat=lat,
                overall_title=f"{self.config.title} Comparison",
                cbar_label=cbar_label,
                save_path=str(save_path)
            )
            
            print(f"Comparison map saved to: {save_path}")


class WorkflowRunner:
    """Main runner for all analysis workflows."""
    
    def __init__(self, config_file: str = "utilityEnvVar.json"):
        """
        Initialize workflow runner.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_manager = ConfigManager(config_file)
    
    def run(self):
        """Execute all selected workflows."""
        print("\n" + "=" * 60)
        print("WORKFLOW RUNNER")
        print("=" * 60)
        
        # Load configuration
        self.config_manager.load()
        scripts = self.config_manager.get_selected_scripts()
        
        print(f"\nSelected scripts: {scripts}")
        
        # Run each script
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
