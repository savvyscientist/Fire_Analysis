#!/usr/bin/env python3
"""
Check what variables are in ModelE BA files
"""
import xarray as xr
import glob

folder = "/discover/nobackup/kmezuman/nkkm_pENINTraf_km_foliage/AIJ"
files = sorted(glob.glob(f"{folder}/*.nc"))

if not files:
    print(f"No files found in {folder}")
else:
    print(f"Found {len(files)} files")
    print(f"Checking first file: {files[0]}")
    
    ds = xr.open_dataset(files[0])
    
    print("\n" + "="*70)
    print("ALL VARIABLES IN FILE:")
    print("="*70)
    for var in sorted(ds.variables):
        if var not in ['lon', 'lat', 'time']:
            print(f"  {var}")
    
    print("\n" + "="*70)
    print("LOOKING FOR BA VARIABLES:")
    print("="*70)
    
    ba_vars = [v for v in ds.variables if 'BA' in v or 'ba' in v.lower() or 'burn' in v.lower()]
    
    if ba_vars:
        print(f"Found {len(ba_vars)} BA-related variables:")
        for var in ba_vars:
            dims = ds[var].dims
            shape = ds[var].shape
            units = ds[var].attrs.get('units', 'NO UNITS')
            long_name = ds[var].attrs.get('long_name', 'NO NAME')
            print(f"\n  Variable: {var}")
            print(f"    Shape: {shape}")
            print(f"    Dims: {dims}")
            print(f"    Units: {units}")
            print(f"    Long name: {long_name}")
            print(f"    Min: {ds[var].values.min():.6e}")
            print(f"    Max: {ds[var].values.max():.6e}")
    else:
        print("No BA variables found!")
        print("\nAll variables (first 20):")
        for i, var in enumerate(sorted(ds.variables)):
            if i >= 20:
                break
            if var not in ['lon', 'lat', 'time']:
                print(f"  {var}")
    
    ds.close()
