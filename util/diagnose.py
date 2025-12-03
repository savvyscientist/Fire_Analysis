#!/usr/bin/env python3
"""
Diagnostic script to check data values
Run this instead of main.py to see what's happening
"""

from config import ConfigManager
from data_loader import DataLoader
from grid_utils import GridAreaCalculator

def diagnose():
    print("="*60)
    print("DIAGNOSTIC MODE")
    print("="*60)
    
    # Load config
    config_mgr = ConfigManager("utilityEnvVar.json")
    config_mgr.load()
    time_config = config_mgr.get_time_analysis_config()
    
    # Setup
    grid_calc = GridAreaCalculator()
    loader = DataLoader(grid_area_calculator=grid_calc.calculate)
    
    # Load each dataset
    for idx, folder_config in enumerate(time_config.folders):
        print(f"\n{'='*60}")
        print(f"Dataset {idx+1}: {folder_config.figure_data.label}")
        print(f"{'='*60}")
        
        data = loader.load_time_series(
            folder_path=folder_config.folder_path,
            file_type=folder_config.file_type,
            variables=folder_config.variables,
            annual=time_config.annual
        )
        
        if data:
            print(f"\n✓ Data loaded successfully!")
            print(f"  Units: {data.units}")
            print(f"  Year range: {data.start_year} - {data.end_year}")
            print(f"  Time series shape: {data.time_series.shape}")
            print(f"\n  Time series data:")
            print(f"    Times (first 5): {data.time_series[:5, 0]}")
            print(f"    Values (first 5): {data.time_series[:5, 1]}")
            print(f"    Min value: {data.time_series[:, 1].min():.6e}")
            print(f"    Max value: {data.time_series[:, 1].max():.6e}")
            print(f"    Mean value: {data.time_series[:, 1].mean():.6e}")
            
            # Check if values are zero
            if data.time_series[:, 1].max() == 0:
                print(f"\n  ❌ WARNING: All values are ZERO!")
            elif data.time_series[:, 1].max() < 1e-10:
                print(f"\n  ⚠️  WARNING: Values are very small (< 1e-10)")
            else:
                print(f"\n  ✓ Values look reasonable")
                
            # Check spatial data
            print(f"\n  Spatial mean:")
            print(f"    Shape: {data.time_mean.shape}")
            print(f"    Min: {data.time_mean.values.min():.6e}")
            print(f"    Max: {data.time_mean.values.max():.6e}")
            print(f"    Mean: {data.time_mean.values.mean():.6e}")
        else:
            print(f"\n✗ Failed to load data!")
    
    print(f"\n{'='*60}")
    print("DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == "__main__":
    diagnose()
