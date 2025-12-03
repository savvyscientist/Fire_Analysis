# Scientific Data Analysis Workflow - Unit Conversion & Spatial Aggregation Updates

## 📋 Overview

This package contains updates to your scientific data analysis workflow that add:

1. **Flexible Unit Conversion** - Convert data to any target units (e.g., kg → Tg, /s → /yr)
2. **Spatial Aggregation Control** - Choose between totals (sum) or means (average)

These features solve two key problems:
- **Problem 1**: Units were hardcoded in JSON, making comparisons difficult
- **Problem 2**: Code always summed spatially, which is wrong for intensive quantities like flammability

## 📦 Package Contents

### Core Files (Copy these to your working directory)

| File | Purpose | Lines |
|------|---------|-------|
| **config.py** | Updated configuration classes with new fields | 135 |
| **unit_converter.py** | Enhanced unit conversion system | 313 |
| **workflow.py** | Updated workflow orchestrator | 415 |
| **utilityEnvVar_updated.json** | Example configuration | 135 |

### Documentation Files

| File | Purpose |
|------|---------|
| **SUMMARY.md** | Quick overview of changes (5 min read) |
| **QUICK_REFERENCE.md** | Cheat sheet for common tasks (2 min read) |
| **UNIT_CONVERSION_GUIDE.md** | Comprehensive guide (15 min read) |

### Reference Files

| File | Purpose |
|------|---------|
| **workflow_original.py** | Backup of original workflow |

## 🚀 Quick Start

### 1. Install Updated Files

```bash
# Copy new files to your project directory
cp config.py unit_converter.py workflow.py /path/to/your/project/
```

### 2. Update Your Configuration

Add two new fields to your JSON config:

```json
{
  "folders": [
    {
      "spatial_aggregation": "total",  // ← NEW: "total" or "mean"
      "figure_data": {
        "target_units": "Tg CO2/yr"    // ← NEW: Desired units
      }
    }
  ]
}
```

### 3. Run Your Analysis

```bash
python main.py  # Uses updated workflow automatically
```

## ✨ Key Features

### Feature 1: Target Unit Conversion

**Before:**
```json
"ylabel": "Tg CO2"  // Hardcoded, might not match data
```

**After:**
```json
"target_units": "Tg CO2/yr"  // Automatic conversion + ylabel
```

**Result:** All datasets convert to Tg CO2/yr, ylabel updates automatically

### Feature 2: Spatial Aggregation

**Before:**
```python
# Always summed (wrong for some quantities)
total = data.sum()
```

**After:**
```json
"spatial_aggregation": "total"  // or "mean"
```

**Result:** Correct handling of extensive (sum) vs. intensive (average) quantities

## 📊 Common Use Cases

### Use Case 1: Emissions Comparison

Compare model vs. observations with different units:

```json
{
  "folders": [
    {
      "folder_path": "/model/CO2",
      "spatial_aggregation": "total",
      "figure_data": {
        "label": "Model",
        "target_units": "Tg CO2/yr"
      }
    },
    {
      "folder_path": "/obs/GFED4s",
      "spatial_aggregation": "total",
      "figure_data": {
        "label": "GFED4s",
        "target_units": "Tg CO2/yr"  // Same units!
      }
    }
  ]
}
```

### Use Case 2: Mixed Quantities

Different aggregation for different variables:

```json
{
  "folders": [
    {
      "variables": ["burned_area"],
      "spatial_aggregation": "total"  // Sum area
    },
    {
      "variables": ["flammability"],
      "spatial_aggregation": "mean"   // Average index
    }
  ]
}
```

## 🎯 Decision Guide

### When to use "total" (sum):
✓ Burned area  
✓ Emissions (CO2, CH4, etc.)  
✓ Fire counts  
✓ Production/consumption  
→ Any **extensive** quantity (depends on system size)

### When to use "mean" (average):
✓ Flammability index  
✓ Combustion completeness  
✓ Temperature  
✓ Precipitation rate  
→ Any **intensive** quantity (independent of system size)

## 📖 Documentation Guide

Read these in order based on your needs:

1. **QUICK_REFERENCE.md** (2 min) - Start here for syntax and examples
2. **SUMMARY.md** (5 min) - Understand what changed and why
3. **UNIT_CONVERSION_GUIDE.md** (15 min) - Deep dive into all features

## 🔧 Implementation Details

### Supported Mass Prefixes

| Prefix | Multiplier | Example Use |
|--------|-----------|-------------|
| Pg | × 10^12 kg | Global carbon |
| Tg | × 10^9 kg | Regional emissions |
| Gg | × 10^6 kg | Country emissions |
| Mg | × 10^3 kg | Local emissions |
| kg | × 1 kg | Grid cell |
| g | × 10^-3 kg | Molecular |

### Supported Time Units

- `/s` - per second
- `/month` - per month
- `/yr` - per year

### Conversion Flow

```
Raw: "kg m-2 s-1"
  ↓ Extract scaling
  ↓ Area integration
"kg/s"
  ↓ Time scaling
"kg/yr"
  ↓ Target conversion
"Tg CO2/yr"  ← Used in plots
```

## ⚠️ Important Notes

### Backwards Compatibility
✓ Old configurations work without changes  
✓ New features are opt-in  
✓ Defaults maintain previous behavior

### Known Limitations
- Currently supports standard mass/time units only
- Requires consistent unit formats across datasets
- Some complex compound units may need custom handling

### Data Loader Update Required

The `data_loader.py` needs one small update:

```python
# In load_time_series() method, change signature:
def load_time_series(
    self, 
    folder_path: str, 
    file_type: str, 
    variables: List[str], 
    annual: bool = False,
    spatial_aggregation: str = 'total'  # ← ADD THIS
):
```

Then pass `spatial_aggregation` to `_process_data()`.

## 🧪 Testing

Verify the changes work:

```bash
# 1. Check conversion messages in output
python main.py | grep "Converting to"

# 2. Verify ylabel matches target_units
# Look at generated plots - ylabel should show target units

# 3. Compare total vs mean aggregation
# Run same data with both settings, verify values differ appropriately
```

## 📞 Support

If you encounter issues:

1. Check **QUICK_REFERENCE.md** for syntax
2. Review **SUMMARY.md** troubleshooting section
3. Verify your JSON matches the example format
4. Check workflow output for conversion messages

## 📝 Example Configuration

Here's a complete working example:

```json
{
  "selected_script": ["time_analysis_version_two"],
  "time_analysis_version_two": {
    "time_analysis_figure_data": {
      "annual": false,
      "title": "CO2 Emissions",
      "ylabel": "Emissions",
      "figs_folder": "./output",
      "spatial_aggregation": "total"
    },
    "folders": [
      {
        "folder_path": "/data/model",
        "file_type": "ModelE_Monthly",
        "variables": ["CO2n_emis"],
        "spatial_aggregation": "total",
        "figure_data": {
          "color": "blue",
          "marker": "o",
          "line_style": "-",
          "label": "Model",
          "target_units": "Tg CO2/yr"
        }
      },
      {
        "folder_path": "/data/obs",
        "file_type": "GFED4s_Monthly",
        "variables": ["CO2n"],
        "spatial_aggregation": "total",
        "figure_data": {
          "color": "red",
          "marker": "s",
          "line_style": "--",
          "label": "GFED4s",
          "target_units": "Tg CO2/yr"
        }
      }
    ]
  }
}
```

## 🎉 Benefits

After implementing these updates:

✓ **Flexible**: Convert to any target units  
✓ **Accurate**: Correct aggregation for all quantity types  
✓ **Clear**: Automatic ylabel from converted units  
✓ **Consistent**: Compare datasets with different native units  
✓ **Simple**: Two configuration fields handle everything  
✓ **Compatible**: Works with existing configurations  

## 📦 Version Info

- Package date: November 2025
- Total files: 8 (4 code + 4 docs)
- Total lines: ~2,100
- Python version: 3.7+
- Dependencies: numpy, pandas, xarray (unchanged)

---

**Ready to implement?** Start with **QUICK_REFERENCE.md** for syntax and examples!
