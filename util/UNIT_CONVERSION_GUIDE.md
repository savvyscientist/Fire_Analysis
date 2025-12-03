# Unit Conversion and Spatial Aggregation Updates

## Overview

This update introduces two major improvements to the scientific data analysis workflow:

1. **Target Unit Conversion**: Automatic conversion to specified target units with proper ylabel handling
2. **Spatial Aggregation Control**: Ability to specify whether to calculate totals or means

---

## 1. Target Unit Conversion

### What Changed

Previously, units were hardcoded in the `ylabel` field of the JSON configuration. Now:

- **Target units** are specified in each dataset's `figure_data` section
- **Automatic conversion** from native data units to target units
- **Dynamic ylabel** based on converted units

### Configuration

Add `target_units` to the `figure_data` section:

```json
{
  "figure_data": {
    "color": "blue",
    "marker": "o",
    "line_style": "-",
    "label": "My Dataset",
    "target_units": "Tg CO2/yr"
  }
}
```

### Supported Unit Conversions

#### Mass Prefixes
- **Pg** (Petagram) = 10^15 g = 10^12 kg
- **Tg** (Teragram) = 10^12 g = 10^9 kg
- **Gg** (Gigagram) = 10^9 g = 10^6 kg
- **Mg** (Megagram) = 10^6 g = 10^3 kg = 1 tonne
- **kg** (kilogram) = 10^3 g
- **g** (gram) - base unit

#### Time Units
- **/s** (per second)
- **/month** (per month)
- **/yr** or **/year** (per year)

#### Example Conversions

| From | To | Conversion Factor |
|------|-----|------------------|
| kg/yr | Tg/yr | ÷ 10^9 |
| kg/month | Tg/yr | ÷ 10^9 × 12 |
| g/s | Tg/yr | ÷ 10^12 × 31,536,000 |

### How It Works

1. **Data Loading**: Native units are extracted from data files (e.g., "kg m-2 s-1")
2. **Initial Conversion**: Area integration and time scaling applied (e.g., → "kg/month")
3. **Target Conversion**: If `target_units` specified, converts to target (e.g., → "Tg CO2/yr")
4. **Plot Labels**: Final units automatically used in ylabel and colorbar labels

### Example Usage

```json
{
  "time_analysis_figure_data": {
    "annual": false,
    "title": "CO2 Emissions",
    "ylabel": "CO2 Emissions",  // Will be overridden
    "figs_folder": "./output"
  },
  "folders": [
    {
      "folder_path": "/path/to/data",
      "file_type": "ModelE_Monthly",
      "variables": ["CO2n_emis"],
      "figure_data": {
        "color": "blue",
        "label": "Model Run 1",
        "target_units": "Tg CO2/yr"  // Converts to Tg CO2/yr
      }
    }
  ]
}
```

**Result**: All plots will use "Tg CO2/yr" as the ylabel, regardless of native data units.

---

## 2. Spatial Aggregation Control

### What Changed

Previously, spatial aggregation was hardcoded. Now you can specify whether to:
- Calculate **totals** (sum over space) - for extensive quantities
- Calculate **means** (average over space) - for intensive quantities

### When to Use Each

#### Use `"spatial_aggregation": "total"` for:
- ✓ Burned area
- ✓ Emissions (CO2, CH4, etc.)
- ✓ Fire counts
- ✓ Production/consumption
- ✓ Any **extensive** quantity (depends on system size)

#### Use `"spatial_aggregation": "mean"` for:
- ✓ Flammability
- ✓ Combustion completeness
- ✓ Temperature
- ✓ Precipitation
- ✓ Fractions/percentages
- ✓ Any **intensive** quantity (independent of system size)

### Configuration Levels

You can specify `spatial_aggregation` at two levels:

#### 1. Global Level (applies to all datasets)

```json
{
  "time_analysis_figure_data": {
    "spatial_aggregation": "total"  // Default for all
  }
}
```

#### 2. Per-Dataset Level (overrides global)

```json
{
  "folders": [
    {
      "folder_path": "/path/to/emissions",
      "spatial_aggregation": "total",  // Sum emissions
      "variables": ["CO2n_emis"]
    },
    {
      "folder_path": "/path/to/flammability",
      "spatial_aggregation": "mean",  // Average flammability
      "variables": ["flammability"]
    }
  ]
}
```

### Default Behavior

If not specified:
- **Global default**: `"total"`
- **Dataset default**: Uses global setting

This means emissions and burned area will sum by default (which is typically correct).

---

## 3. Updated Configuration Structure

### Complete Example

```json
{
  "selected_script": ["time_analysis_version_two"],
  "time_analysis_version_two": {
    "save_netcdf": true,
    "time_analysis_figure_data": {
      "annual": false,
      "title": "Fire Emissions Comparison",
      "ylabel": "Emissions",  // Will be replaced by target_units
      "figs_folder": "./output/emissions",
      "spatial_aggregation": "total"  // Global default
    },
    "folders": [
      {
        "folder_path": "/data/model_run1",
        "file_type": "ModelE_Monthly",
        "variables": ["CO2n_emis"],
        "spatial_aggregation": "total",  // Sum emissions
        "figure_data": {
          "color": "blue",
          "marker": "o",
          "line_style": "-",
          "label": "Model v1",
          "target_units": "Tg CO2/yr"  // Convert to Tg CO2/yr
        }
      },
      {
        "folder_path": "/data/observations",
        "file_type": "GFED4s_Monthly",
        "variables": ["CO2n"],
        "spatial_aggregation": "total",  // Sum emissions
        "figure_data": {
          "color": "red",
          "marker": "s",
          "line_style": "--",
          "label": "GFED4s",
          "target_units": "Tg CO2/yr"  // Convert to same units
        }
      },
      {
        "folder_path": "/data/flammability",
        "file_type": "ModelE_Monthly",
        "variables": ["flammability_index"],
        "spatial_aggregation": "mean",  // Average flammability
        "figure_data": {
          "color": "green",
          "marker": "^",
          "line_style": "-",
          "label": "Flammability",
          "target_units": null  // No conversion needed
        }
      }
    ]
  }
}
```

---

## 4. Implementation Details

### Modified Files

1. **config.py**
   - Added `target_units` field to `FigureConfig`
   - Added `spatial_aggregation` to `FolderConfig` and `TimeAnalysisConfig`

2. **unit_converter.py**
   - Added `MASS_PREFIXES` dictionary for unit conversions
   - New `parse_units()` function to parse unit strings
   - New `convert_to_target_units()` function for target conversions
   - Updated `convert()` method to accept `target_units` parameter

3. **workflow.py**
   - Added `_determine_final_units()` method
   - Updated `_load_all_datasets()` to apply target conversions
   - Modified all plotting methods to use `final_units` for labels

4. **data_loader.py** (needs update)
   - `load_time_series()` should accept `spatial_aggregation` parameter
   - `_process_data()` should use the `spatial_aggregation` parameter

### Unit Conversion Flow

```
Raw Data (e.g., "10^-3 kg CO2n m-2 s-1")
    ↓
Extract scaling factor (0.001)
    ↓
Apply area integration (× grid_area_m2)
    → "kg/s"
    ↓
Apply time scaling (× seconds_in_year)
    → "kg/yr"
    ↓
Convert to target (÷ 10^9)
    → "Tg CO2/yr"
```

---

## 5. Migration Guide

### Updating Existing Configurations

#### Before (Old Format):
```json
{
  "time_analysis_figure_data": {
    "ylabel": "Tg CO2"  // Hardcoded, might not match data
  },
  "folders": [
    {
      "figure_data": {
        "label": "My Data"
      }
    }
  ]
}
```

#### After (New Format):
```json
{
  "time_analysis_figure_data": {
    "ylabel": "CO2 Emissions",  // Generic, will be replaced
    "spatial_aggregation": "total"  // Specify default behavior
  },
  "folders": [
    {
      "spatial_aggregation": "total",  // Per-dataset override
      "figure_data": {
        "label": "My Data",
        "target_units": "Tg CO2/yr"  // Automatic conversion
      }
    }
  ]
}
```

### Backwards Compatibility

- If `target_units` not specified: Uses native data units (old behavior)
- If `spatial_aggregation` not specified: Defaults to "total"
- Old configurations will still work but won't benefit from new features

---

## 6. Common Use Cases

### Case 1: CO2 Emissions Comparison

Multiple datasets with different native units, all converted to Tg CO2/yr:

```json
"target_units": "Tg CO2/yr",
"spatial_aggregation": "total"
```

### Case 2: Burned Area Analysis

Sum burned area across space:

```json
"target_units": "Mha/yr",  // Million hectares per year
"spatial_aggregation": "total"
```

### Case 3: Flammability Index

Average flammability (intensive property):

```json
"target_units": null,  // No conversion needed
"spatial_aggregation": "mean"
```

### Case 4: Mixed Analysis

Different aggregation for different datasets:

```json
{
  "folders": [
    {
      "variables": ["burned_area"],
      "spatial_aggregation": "total"
    },
    {
      "variables": ["fire_weather_index"],
      "spatial_aggregation": "mean"
    }
  ]
}
```

---

## 7. Validation and Debugging

### Checking Conversions

The workflow prints detailed information:

```
Loading: My Dataset
  Initial units: kg CO2n m-2 s-1
  Converting to: Tg CO2/yr
  Final units: Tg CO2/yr
```

### Troubleshooting

1. **Units don't convert properly**
   - Check if mass prefix is recognized (Tg, Gg, Mg, kg, g, Pg)
   - Verify time units are standard (/s, /month, /yr)

2. **Wrong aggregation results**
   - For extensive quantities (emissions, area): use "total"
   - For intensive quantities (temperature, fraction): use "mean"

3. **Ylabel doesn't update**
   - Ensure `target_units` is specified in at least one dataset
   - Check for typos in unit strings

---

## 8. Best Practices

1. **Always specify target_units** for emission datasets
2. **Use consistent units** across datasets for comparison
3. **Set spatial_aggregation explicitly** to avoid confusion
4. **Document units** in dataset labels for clarity
5. **Validate conversions** by checking output values make sense

---

## 9. Future Enhancements

Potential additions:
- Support for more compound units (e.g., "g C/m2/yr")
- Custom conversion factors
- Unit validation warnings
- Automatic unit detection from variable names
- Support for area units (ha, km², m²)

---

## Summary

These updates provide:
- ✓ Flexible unit conversion to any target units
- ✓ Automatic ylabel generation from converted units
- ✓ Control over spatial aggregation (total vs. mean)
- ✓ Better handling of extensive vs. intensive quantities
- ✓ Backwards compatible with existing configurations
- ✓ Clear, documented configuration structure

The workflow is now more robust and easier to use for comparing datasets with different native units.
