#!/bin/bash

# Set year variable
START_YEAR=1990
END_YEAR=1991

# List of folders to process
folders=(
#     "/discover/nobackup/kmezuman/nkkm_pENINTraf_km_stem0_01"
    "/discover/nobackup/kmezuman/E6TomaF40intfuelpyrEStem"
    "/discover/nobackup/kmezuman/E6TomaF40intfuelpyrEFoliage"
    "/discover/nobackup/kmezuman/E6TomaF40intfuelpyrELitter"
    "/discover/nobackup/kmezuman/E6TomaF40intfuelpyrECWD"
#    "/discover/nobackup/kmezuman/nkkm_pENINTraf_km_cwdtest1"
#    "/discover/nobackup/kmezuman/nkkm_pENINTraf_km_foliagetest1"
#    "/discover/nobackup/kmezuman/nkkm_pENINTraf_km_littertest1"
#    "/discover/nobackup/kmezuman/nkkm_pENINTraf_GFEDbetad"
#    "/discover/nobackup/kmezuman/nkkm_pENINTraf_GFEDbetadcwd"
#    "/discover/nobackup/kmezuman/nkkm_pENINTraf_GFEDbetadfoliage"
#    "/discover/nobackup/kmezuman/nkkm_pENINTraf_GFEDbetadlitter"
#    "/discover/nobackup/kmezuman/nkkm_pENINTraf_GFEDbetadstem"
#    "/discover/nobackup/kmezuman/nkkm_pENINTraf_km_stem"
#    "/discover/nobackup/kmezuman/nkkm_pENINTraf_km_cwd"
#    "/discover/nobackup/kmezuman/nkkm_pENINTraf_km_foliage"
#    "/discover/nobackup/kmezuman/nkkm_pENINTraf_km_litter"
#    "/discover/nobackup/kmezuman/nkkm_pENINTraf_km_fuel"
)

# Loop through each folder
for folder in "${folders[@]}"; do
    echo "Processing folder: $folder"
    
    # Check if folder exists
    if [ ! -d "$folder" ]; then
        echo "Warning: Folder $folder does not exist, skipping..."
        continue
    fi
    
    # Change to the folder
    cd "$folder" || {
        echo "Error: Could not change to directory $folder"
        continue
    }
    
    # Create the TAIJ subfolder
    mkdir -p TAIJ 
    
    # Loop through each year in the range
    for ((year=START_YEAR; year<=END_YEAR; year++)); do
        echo "Processing year: $year" 

        # Find files matching the pattern and process them 
        for f in $(find . -maxdepth 1 -name "*${year}*acc*.nc"); do 
            echo "Processing file: $f" 
            scaleacc_himem "$f" taij 
        done 

        # Move resulting files to the subfolder 
        if ls *${year}.taij*.nc 1> /dev/null 2>&1; then 
            echo "Moving ${year}.taij*.nc files to TAIJ/" 
            mv *${year}.taij*.nc TAIJ/ 
        else 
            echo "No *${year}.taij*.nc files found to move" 
        fi
    done
    
    echo "Completed processing $folder"
done

echo "All folders processed"
