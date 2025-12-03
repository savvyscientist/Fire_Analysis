#!/usr/bin/env python3
"""
Example usage script demonstrating the restructured code.

This shows various ways to use the new modular architecture.
"""

import numpy as np
from pathlib import Path


def example_1_basic_workflow():
    """Example 1: Run complete workflow from config file."""
    print("=" * 60)
    print("Example 1: Basic Workflow")
    print("=" * 60)
    
    from workflow import WorkflowRunner
    
    # Simple one-liner
    runner = WorkflowRunner(config_file="utilityEnvVar.json")
    runner.run()


def example_2_programmatic_config():
    """Example 2: Create configuration programmatically."""
    print("\n" + "=" * 60)
    print("Example 2: Programmatic Configuration")
    print("=" * 60)
    
    from config import TimeAnalysisConfig, FolderConfig, FigureConfig
    from workflow import TimeSeriesWorkflow
    
    # Build configuration in code
    folder = FolderConfig(
        folder_path="/path/to/data",
        file_type="ModelE_Monthly",
        variables=["BA_tree", "BA_shrub"],
        figure_data=FigureConfig(
            color="red",
            marker="o",
            line_style="-",
            label="My Dataset"
        )
    )
    
    config = TimeAnalysisConfig(
        annual=False,
        title="Burned Area",
        ylabel="Area (Mha)",
        figs_folder="./output",
        folders=[folder]
    )
    
    # Run workflow
    workflow = TimeSeriesWorkflow(config)
    workflow.run()


def example_3_load_and_analyze():
    """Example 3: Load data and perform custom analysis."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Analysis")
    print("=" * 60)
    
    from data_loader import DataLoader
    from analysis import TimeSeriesAnalyzer
    
    # Load data
    loader = DataLoader()
    data = loader.load_time_series(
        folder_path="/path/to/data",
        file_type="ModelE_Monthly",
        variables=["BA_tree"],
        annual=False
    )
    
    if data:
        # Perform analysis
        analyzer = TimeSeriesAnalyzer()
        
        # Calculate seasonal statistics
        seasonal = analyzer.calculate_seasonal_cycle(data.time_series)
        print(f"Seasonal means: {seasonal.means}")
        
        # Calculate annual totals
        annual = analyzer.calculate_annual_totals(data.time_series)
        if annual is not None:
            print(f"Years: {annual[:, 0]}")
            print(f"Annual totals: {annual[:, 1]}")
        
        # Calculate trend
        slope, intercept = analyzer.calculate_trend(data.time_series)
        print(f"Trend: {slope:.3f} per year")


def example_4_custom_plots():
    """Example 4: Create custom visualizations."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Plots")
    print("=" * 60)
    
    from visualization import TimeSeriesPlotter, SpatialPlotter, PlotStyle
    import numpy as np
    
    # Create sample data
    times = np.linspace(2000, 2020, 100)
    values = np.sin(times * 0.5) * 10 + 50
    time_series = np.column_stack([times, values])
    
    # Create mock data object
    class MockData:
        def __init__(self):
            self.time_series = time_series
            self.longitude = np.linspace(-180, 180, 144)
            self.latitude = np.linspace(-90, 90, 90)
            self.time_mean = type('obj', (object,), {
                'values': np.random.rand(90, 144)
            })()
            self.units = "kg/s"
    
    data = MockData()
    
    # Create custom plot style
    style = PlotStyle(
        color='darkblue',
        marker='o',
        linestyle='-',
        label='Custom Data',
        linewidth=2.5,
        markersize=4
    )
    
    # Create time series plot
    plotter = TimeSeriesPlotter(figsize=(14, 6))
    plotter.plot_time_series(
        datasets=[(data, style)],
        title="Custom Time Series Analysis",
        ylabel="Value (units)",
        save_path="output/custom_timeseries.png"
    )
    
    # Create spatial plot
    spatial_plotter = SpatialPlotter(figsize=(12, 8))
    spatial_plotter.plot_spatial_map(
        data=data.time_mean.values,
        lon=data.longitude,
        lat=data.latitude,
        title="Spatial Distribution",
        cbar_label="Value (units)",
        cmap='RdYlBu_r',
        save_path="output/custom_spatial.png"
    )
    
    print("Plots saved to output/")


def example_5_grid_calculations():
    """Example 5: Work with grid calculations."""
    print("\n" + "=" * 60)
    print("Example 5: Grid Calculations")
    print("=" * 60)
    
    from grid_utils import GridAreaCalculator
    
    calculator = GridAreaCalculator()
    
    # Calculate grid areas for different resolutions
    for shape in [(90, 144), (180, 360), (360, 720)]:
        areas = calculator.calculate(
            grid_area_shape=shape,
            units="km"
        )
        
        total_area = areas.sum()
        print(f"\nGrid {shape[0]}x{shape[1]}:")
        print(f"  Total area: {total_area:.2e} km²")
        print(f"  Earth surface: ~5.10e8 km²")
        print(f"  Coverage: {(total_area / 5.1e8 * 100):.1f}%")
    
    # Clear cache
    calculator.clear_cache()
    print("\nCache cleared")


def example_6_unit_conversion():
    """Example 6: Unit conversion examples."""
    print("\n" + "=" * 60)
    print("Example 6: Unit Conversion")
    print("=" * 60)
    
    from unit_converter import UnitConverter, extract_scaling_factor
    from grid_utils import GridAreaCalculator
    
    # Test scaling factor extraction
    test_units = [
        "10^-3 kg/m^2/s",
        "kg m-2 s-1",
        "1e-6 kg/m2/s"
    ]
    
    print("Scaling factor extraction:")
    for unit in test_units:
        factor, clean = extract_scaling_factor(unit)
        print(f"  {unit:20s} → factor={factor:.2e}, clean={clean}")
    
    # Setup converter
    grid_calc = GridAreaCalculator()
    converter = UnitConverter(
        grid_area_calculator=grid_calc.calculate
    )
    
    # Example conversion
    sample_data = np.ones((90, 144)) * 0.001  # Sample data
    
    converted, new_units = converter.convert(
        data_array=sample_data,
        units="10^-3 kg/m2/s",
        monthly=False
    )
    
    print(f"\nConversion example:")
    print(f"  Original: {sample_data.mean():.2e} (10^-3 kg/m2/s)")
    print(f"  Converted: {converted.mean():.2e} ({new_units})")


def example_7_compare_datasets():
    """Example 7: Compare multiple datasets."""
    print("\n" + "=" * 60)
    print("Example 7: Dataset Comparison")
    print("=" * 60)
    
    from data_loader import DatasetCombiner
    
    # This example shows the concept
    # In practice, you'd load real data
    
    print("Dataset comparison capabilities:")
    print("  - Calculate spatial differences")
    print("  - Find common time points")
    print("  - Compare time series")
    print("  - Validate unit compatibility")
    print("\nSee DatasetCombiner.calculate_difference() for implementation")


def example_8_extend_functionality():
    """Example 8: Extend with custom components."""
    print("\n" + "=" * 60)
    print("Example 8: Extending Functionality")
    print("=" * 60)
    
    from data_loader import DataLoader
    from analysis import TimeSeriesAnalyzer
    
    # Extend DataLoader with custom file type
    class MyDataLoader(DataLoader):
        def _register_loaders(self):
            loaders = super()._register_loaders()
            loaders['MyCustomFormat'] = self._load_custom
            return loaders
        
        def _load_custom(self, folder_path, variables, annual):
            print(f"Loading custom format from {folder_path}")
            # Custom implementation here
            return None
    
    # Extend analyzer with custom analysis
    class MyAnalyzer(TimeSeriesAnalyzer):
        @staticmethod
        def custom_metric(time_series):
            """Calculate custom metric."""
            values = time_series[:, 1]
            return {
                'mean': np.mean(values),
                'max': np.max(values),
                'variability': np.std(values) / np.mean(values)
            }
    
    # Use extended components
    loader = MyDataLoader()
    analyzer = MyAnalyzer()
    
    print("Custom components created:")
    print("  - MyDataLoader with custom format support")
    print("  - MyAnalyzer with additional metrics")


def main():
    """Run all examples."""
    examples = [
        ("Basic Workflow", example_1_basic_workflow),
        ("Programmatic Config", example_2_programmatic_config),
        ("Load and Analyze", example_3_load_and_analyze),
        ("Custom Plots", example_4_custom_plots),
        ("Grid Calculations", example_5_grid_calculations),
        ("Unit Conversion", example_6_unit_conversion),
        ("Compare Datasets", example_7_compare_datasets),
        ("Extend Functionality", example_8_extend_functionality),
    ]
    
    print("\n" + "=" * 60)
    print("RESTRUCTURED CODE EXAMPLES")
    print("=" * 60)
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n" + "=" * 60)
    print("Running Example 5 (Grid Calculations)...")
    print("=" * 60)
    
    # Run one example that doesn't need data files
    example_5_grid_calculations()
    
    print("\n" + "=" * 60)
    print("Running Example 6 (Unit Conversion)...")
    print("=" * 60)
    
    example_6_unit_conversion()
    
    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)
    print("\nTo run specific examples, call the functions directly:")
    print("  python -c 'from example_usage import example_5_grid_calculations; example_5_grid_calculations()'")


if __name__ == "__main__":
    main()
