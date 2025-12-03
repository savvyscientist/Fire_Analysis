# Quick Reference: Unit Conversion & Spatial Aggregation

## Configuration Cheat Sheet

### Basic Structure

```json
{
  "time_analysis_version_two": {
    "time_analysis_figure_data": {
      "spatial_aggregation": "total"  // Global default
    },
    "folders": [
      {
        "spatial_aggregation": "total",  // Dataset override
        "figure_data": {
          "target_units": "Tg CO2/yr"  // Desired units
        }
      }
    ]
  }
}
```

## Mass Prefixes Quick Reference

| Prefix | Full Name  | Value (grams) | Value (kg)    | Example Use         |
|--------|-----------|---------------|---------------|---------------------|
| Pg     | Petagram  | 10^15 g       | 10^12 kg      | Global carbon stock |
| Tg     | Teragram  | 10^12 g       | 10^9 kg       | Regional emissions  |
| Gg     | Gigagram  | 10^9 g        | 10^6 kg       | Country emissions   |
| Mg     | Megagram  | 10^6 g        | 10^3 kg       | Small region        |
| kg     | Kilogram  | 10^3 g        | 1 kg          | Grid cell           |
| g      | Gram      | 1 g           | 10^-3 kg      | Molecular scale     |

## Spatial Aggregation Decision Tree

```
Is your variable an extensive quantity?
(Does it depend on the size of the system?)
│
├─ YES → Use "total"
│  Examples:
│  • Burned area (more area = more burning)
│  • Emissions (larger region = more emissions)
│  • Fire counts (more area = more fires)
│  • Production/consumption
│
└─ NO → Use "mean"
   Examples:
   • Flammability index (average property)
   • Combustion completeness (fraction)
   • Temperature (intensive)
   • Precipitation rate
   • Efficiency ratios
```

## Common Unit Conversions

### Emissions

| From              | To            | Example Use                  |
|-------------------|---------------|------------------------------|
| kg m-2 s-1        | Tg CO2/yr     | Model output → Report units  |
| g C m-2 month-1   | Tg C/yr       | Carbon fluxes                |
| kg/s              | Tg/yr         | Direct emissions             |

### Burned Area

| From              | To            | Example Use                  |
|-------------------|---------------|------------------------------|
| m²                | Mha           | Grid cells → Megahectares    |
| fraction          | Mha           | Fraction × area              |
| km²               | Mha           | Kilometers to Megahectares   |

### Fluxes

| From              | To            | Example Use                  |
|-------------------|---------------|------------------------------|
| kg m-2 s-1        | kg/yr         | Flux → Annual total          |
| g m-2 day-1       | kg m-2 yr-1   | Daily → Annual rate          |

## Configuration Examples by Use Case

### Use Case 1: Compare Model vs. Observations (Different Units)

```json
{
  "folders": [
    {
      "folder_path": "/model/emissions",
      "variables": ["CO2n_emis"],
      "spatial_aggregation": "total",
      "figure_data": {
        "label": "Model",
        "target_units": "Tg CO2/yr"  // Convert from kg m-2 s-1
      }
    },
    {
      "folder_path": "/obs/GFED4s",
      "variables": ["CO2"],
      "spatial_aggregation": "total",
      "figure_data": {
        "label": "GFED4s",
        "target_units": "Tg CO2/yr"  // Convert from g C m-2
      }
    }
  ]
}
```

### Use Case 2: Multiple Variables (Different Aggregations)

```json
{
  "folders": [
    {
      "variables": ["burned_area"],
      "spatial_aggregation": "total",  // Sum area
      "figure_data": {
        "target_units": "Mha/yr"
      }
    },
    {
      "variables": ["fire_weather_index"],
      "spatial_aggregation": "mean",  // Average index
      "figure_data": {
        "target_units": null  // Keep original
      }
    }
  ]
}
```

### Use Case 3: Time Series at Different Scales

```json
{
  "folders": [
    {
      "variables": ["global_emissions"],
      "spatial_aggregation": "total",
      "figure_data": {
        "target_units": "Pg C/yr"  // Global scale
      }
    },
    {
      "variables": ["regional_emissions"],
      "spatial_aggregation": "total",
      "figure_data": {
        "target_units": "Tg C/yr"  // Regional scale
      }
    }
  ]
}
```

## Debugging Checklist

### If conversion doesn't work:

- [ ] Check spelling of mass prefix (case-sensitive: Tg not TG)
- [ ] Verify time unit format (/yr not /year, /s not /sec)
- [ ] Ensure units are recognized format
- [ ] Check workflow output for conversion messages

### If aggregation seems wrong:

- [ ] Verify quantity type (extensive vs. intensive)
- [ ] Check if "total" or "mean" is specified
- [ ] Compare output magnitude with expected values
- [ ] Review global vs. dataset-level settings

### If ylabel doesn't update:

- [ ] Ensure at least one dataset has target_units
- [ ] Check for typos in target_units string
- [ ] Verify workflow prints "Using target units for ylabel"

## Unit String Formats

### Supported Formats

✓ Correct:
- `"Tg CO2/yr"`
- `"kg/m2/s"`
- `"Gg C/month"`
- `"g m-2 s-1"`

✗ Incorrect:
- `"TG CO2/yr"` (wrong case)
- `"kg/m^2/sec"` (use /s not /sec)
- `"tonnes/year"` (use Mg or kg)

## File Update Checklist

To implement these changes:

1. **config.py**
   - [x] Added target_units to FigureConfig
   - [x] Added spatial_aggregation to configs

2. **unit_converter.py**
   - [x] Added MASS_PREFIXES dictionary
   - [x] Added parse_units() function
   - [x] Added convert_to_target_units() function

3. **workflow.py**
   - [x] Added _determine_final_units() method
   - [x] Updated _load_all_datasets() for conversions
   - [x] Modified plotting to use final_units

4. **data_loader.py** (needs update)
   - [ ] Add spatial_aggregation parameter to load_time_series()
   - [ ] Pass through to _process_data()

5. **Configuration JSON**
   - [ ] Add target_units to figure_data
   - [ ] Add spatial_aggregation per dataset
   - [ ] Update global spatial_aggregation default

## Common Patterns

### Pattern 1: Standard Emissions Analysis
```json
"spatial_aggregation": "total",
"target_units": "Tg CO2/yr"
```

### Pattern 2: Intensive Property Analysis
```json
"spatial_aggregation": "mean",
"target_units": null
```

### Pattern 3: Mixed Analysis
```json
// In time_analysis_figure_data
"spatial_aggregation": "total",  // Default

// Override for specific datasets
"folders": [
  {"spatial_aggregation": "mean", ...}  // Override
]
```

## Further Reading

- **UNIT_CONVERSION_GUIDE.md** - Comprehensive documentation
- **SUMMARY.md** - Overview of changes
- **utilityEnvVar_updated.json** - Full example configuration
