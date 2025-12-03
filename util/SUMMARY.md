# Summary of Changes to Scientific Data Analysis Workflow

## Overview
Updated the workflow to support flexible unit conversion and spatial aggregation control.

## Key Changes

### 1. Target Unit Conversion System

**Problem**: Units were hardcoded in the JSON `ylabel` field, making it difficult to:
- Compare datasets with different native units
- Change output units without modifying code
- Ensure ylabel matches actual data units

**Solution**: Added `target_units` field to configuration:

```json
"figure_data": {
  "label": "My Dataset",
  "target_units": "Tg CO2/yr"  // ← NEW: Specify desired units
}
```

**Benefits**:
- Automatic conversion from native units (e.g., "kg m-2 s-1") to target units
- Dynamic ylabel based on actual converted units
- Consistent units across multiple datasets for comparison
- Supports mass prefixes: Pg, Tg, Gg, Mg, kg, g
- Supports time units: /s, /month, /yr

### 2. Spatial Aggregation Control

**Problem**: Code always summed values over space, which is wrong for intensive quantities like:
- Flammability (should be averaged)
- Combustion completeness (should be averaged)
- Temperature (should be averaged)

**Solution**: Added `spatial_aggregation` parameter:

```json
{
  "folder_path": "/path/to/emissions",
  "spatial_aggregation": "total",  // ← Sum emissions (extensive)
  "variables": ["CO2n_emis"]
}

{
  "folder_path": "/path/to/flammability",
  "spatial_aggregation": "mean",  // ← Average flammability (intensive)
  "variables": ["flammability"]
}
```

**Benefits**:
- Correct handling of extensive vs. intensive quantities
- Can be set globally or per-dataset
- Defaults to "total" (maintains backwards compatibility)

## Modified Files

### 1. config.py
- Added `target_units: Optional[str]` to `FigureConfig`
- Added `spatial_aggregation: str` to both `FolderConfig` and `TimeAnalysisConfig`
- Default `spatial_aggregation` is "total"

### 2. unit_converter.py
- Added `MASS_PREFIXES` dictionary with conversion factors
- New `parse_units()` function to parse unit strings
- New `convert_to_target_units()` function for target conversions
- Enhanced `convert()` method to accept `target_units` parameter

### 3. workflow.py
- Added `_determine_final_units()` method to extract target units
- Updated `_load_all_datasets()` to:
  - Apply target unit conversions
  - Pass spatial_aggregation to data loader
- Modified all plotting methods to use `final_units` for ylabel/colorbar

### 4. data_loader.py (requires update)
- Should accept `spatial_aggregation` parameter in `load_time_series()`
- Should pass parameter through to `_process_data()`
- Already has spatial_aggregation logic in `_process_data()` (line 371-383)

## Configuration Examples

### Example 1: CO2 Emissions (Total)
```json
{
  "folder_path": "/data/emissions",
  "file_type": "ModelE_Monthly",
  "variables": ["CO2n_emis"],
  "spatial_aggregation": "total",  // Sum over space
  "figure_data": {
    "color": "blue",
    "label": "Model Emissions",
    "target_units": "Tg CO2/yr"  // Convert to Tg CO2/yr
  }
}
```

### Example 2: Flammability Index (Mean)
```json
{
  "folder_path": "/data/flammability",
  "file_type": "ModelE_Monthly",
  "variables": ["flammability"],
  "spatial_aggregation": "mean",  // Average over space
  "figure_data": {
    "color": "red",
    "label": "Flammability",
    "target_units": null  // No conversion needed
  }
}
```

## Unit Conversion Flow

```
Raw Data: "10^-3 kg CO2n m-2 s-1"
    ↓
Extract scaling: × 0.001
    ↓
Area integration: × grid_area_m2 → "kg/s"
    ↓
Time scaling: × 31,536,000 → "kg/yr"
    ↓
Target conversion: ÷ 10^9 → "Tg CO2/yr"
    ↓
Used in plots: ylabel = "Tg CO2/yr"
```

## Guidelines for Spatial Aggregation

### Use "total" for:
- Burned area (extensive)
- Emissions (CO2, CH4, etc.)
- Fire counts
- Production/consumption rates
- Any quantity that depends on system size

### Use "mean" for:
- Flammability (intensive)
- Combustion completeness (fraction)
- Temperature
- Precipitation rate
- Indices and fractions
- Any quantity independent of system size

## Backwards Compatibility

✓ Old configurations work without modification
✓ If `target_units` not specified, uses native units
✓ If `spatial_aggregation` not specified, defaults to "total"
✓ Existing behavior preserved when new features not used

## Testing Recommendations

1. Test with different unit combinations:
   - kg → Tg conversion
   - /s → /yr conversion
   - Combined conversions

2. Verify spatial aggregation:
   - Compare "total" vs "mean" for same dataset
   - Check that extensive quantities sum
   - Check that intensive quantities average

3. Validate plot labels:
   - Ylabel matches converted units
   - Colorbar labels correct
   - Multiple datasets use consistent units

## Files Provided

1. **config.py** - Updated configuration classes
2. **unit_converter.py** - Enhanced unit conversion system
3. **workflow.py** - Updated workflow with new features
4. **utilityEnvVar_updated.json** - Example configuration
5. **UNIT_CONVERSION_GUIDE.md** - Comprehensive documentation
6. **SUMMARY.md** - This file

## Next Steps

To use these improvements:

1. Copy updated files to your working directory
2. Update JSON configuration with:
   - `target_units` for datasets needing conversion
   - `spatial_aggregation` based on quantity type
3. Update `data_loader.py` to accept `spatial_aggregation` parameter
4. Test with your datasets
5. Adjust based on results

## Questions or Issues?

Common issues:
- **Units not converting?** Check mass prefix spelling (Tg, Gg, etc.)
- **Wrong totals?** Verify spatial_aggregation setting
- **Ylabel not updating?** Ensure target_units specified in config
