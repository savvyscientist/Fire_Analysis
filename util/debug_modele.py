#!/usr/bin/env python3
"""
Diagnostic script to debug ModelE zero values issue.
"""
import numpy as np
from data_loader import DataLoader
from grid_utils import GridAreaCalculator

def debug_modele_loading():
    print("="*60)
    print("DEBUGGING ModelE LOADING")
    print("="*60)
    
    # Setup
    grid_calc = GridAreaCalculator()
    loader = DataLoader(grid_area_calculator=grid_calc.calculate)
    
    # ModelE configuration from your config
    components = [
        "/discover/nobackup/kmezuman/nkkm_pENINTraf_km_foliage/AIJ",
        "/discover/nobackup/kmezuman/nkkm_pENINTraf_km_cwd/AIJ",
        "/discover/nobackup/kmezuman/nkkm_pENINTraf_km_litter/AIJ"
    ]
    
    variables = ["CO2n_emis"]
    
    print("\n1. Loading ModelE Combined data...")
    print(f"   Components: {len(components)}")
    print(f"   Variables: {variables}")
    
    data = loader.load_time_series(
        folder_path="NOT_USED",  # Not used for Combined_ModelE
        file_type="Combined_ModelE",
        variables=variables,
        annual=False,
        spatial_aggregation='total',
        components=components
    )
    
    if data is None:
        print("\n❌ FAILED: Data is None!")
        return
    
    print("\n2. DATA LOADED - Checking values...")
    print(f"   Units: {data.units}")
    print(f"   Time series shape: {data.time_series.shape}")
    print(f"   Time series values (first 5):")
    for i in range(min(5, len(data.time_series))):
        print(f"      Time {data.time_series[i, 0]:.2f}: {data.time_series[i, 1]:.6e}")
    
    print(f"\n3. STATISTICS:")
    print(f"   Min value: {data.time_series[:, 1].min():.6e}")
    print(f"   Max value: {data.time_series[:, 1].max():.6e}")
    print(f"   Mean value: {data.time_series[:, 1].mean():.6e}")
    print(f"   Std value: {data.time_series[:, 1].std():.6e}")
    
    # Check for zeros
    zero_count = np.sum(data.time_series[:, 1] == 0)
    print(f"   Zero values: {zero_count}/{len(data.time_series)}")
    
    if data.time_series[:, 1].max() == 0:
        print("\n❌ PROBLEM: All values are ZERO!")
        print("\n   Checking spatial data...")
        print(f"   Spatial mean shape: {data.time_mean.shape}")
        print(f"   Spatial mean min: {data.time_mean.values.min():.6e}")
        print(f"   Spatial mean max: {data.time_mean.values.max():.6e}")
        print(f"   Spatial mean attrs: {data.time_mean.attrs}")
        
        if data.time_mean.values.max() > 0:
            print("\n   ⚠️  Spatial data has values but time series is zero!")
            print("   This suggests area integration is zeroing out the data")
        else:
            print("\n   ⚠️  Even spatial data is zero - problem is earlier in loading")
    
    elif data.time_series[:, 1].max() < 1e-10:
        print(f"\n⚠️  WARNING: Values are very small (< 1e-10)")
        print("   This might indicate unit conversion issue")
    
    else:
        print("\n✓ Values look reasonable")
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == "__main__":
    debug_modele_loading()
