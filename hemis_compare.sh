#!/bin/bash

# List of files to process
files=(
#    "/discover/nobackup/kmezuman/modelE/CC_pyrE/tracersE3/pyrE_bugs/modelE/decks/E6TomaF40pyrEbugsfix/2005/JAN2005.aijE6TomaF40pyrEbugsfix.nc"
#    "/discover/nobackup/kmezuman/modelE/CC_pyrE/tracersE3/pyrE_bugs/modelE/decks/E6TomaF40pyrEbugsfixstash/2005/JAN2005.aijE6TomaF40pyrEbugsfixstash.nc"
     "/discover/nobackup/kmezuman/nkkm_pENINTraf_km_fuel/2010nofearth/JAN2010.aijnkkm_pENINTraf_km_fuel.nc"
     "/discover/nobackup/kmezuman/nkkm_pENINTraf_km_betad/2010nofearth/JAN2010.aijnkkm_pENINTraf_km_betad.nc"
)

# List of variables to extract
variables=(
    "BA_tree"
    "BA_shrub" 
    "BA_grass"
#    "FLAMM"
#    "fireCount"
#     "CO2n_emis"

)

echo "Processing files with variables..."
echo "=================================="

# Loop through each file
for file in "${files[@]}"; do
    echo "File: $(basename "$file")"
    
    # Check if file exists
    if [ ! -f "$file" ]; then
        echo "Warning: File $file does not exist, skipping..."
        continue
    fi
    
    # Extract just the values for each variable
    for var in "${variables[@]}"; do
        var_hemis="${var}_hemis"
        echo -n "$var_hemis: "
        ncdump -v "$var_hemis" "$file" 2>/dev/null | sed -n "/data:/,\$p" | grep "$var_hemis =" | sed 's/.*= //' | tr -d ' ;'
    done
    
    echo "----------------------------------------"
done

echo ""
echo "Extracting units information from first available file..."
echo "========================================================"

# Extract units from the first available file for each variable
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "Using file: $(basename "$file")"
        echo ""
        
        # Extract units for each variable using the improved grep pattern
        for var in "${variables[@]}"; do
            echo "Units for variable '$var':"
            ncdump -h "$file" | grep -i "$var" | grep -i units
            echo ""
        done
        
        # Only process the first available file for units
        break
    fi
done

echo "Script completed!"
