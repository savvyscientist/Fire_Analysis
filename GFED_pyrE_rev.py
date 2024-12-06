#!/usr/bin/env python3
"""
Fire Emissions Analysis Tool

This script analyzes fire emissions data from various sources including GFED4s and ModelE.
It calculates emissions based on burned area and biomass data, comparing different 
experimental scenarios for combustion.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import numpy as np
import xarray as xr
import sys
import netCDF4 as nc
from netCDF4 import Dataset
import re
import calendar
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def load_config() -> Dict:
    """Load configuration settings for file paths, year range, and other parameters."""
    config = {
        'dir_sim': '/discover/nobackup/kmezuman/nk_CCycle_E6obioF40',
        'dir_obs_emis': '/discover/nobackup/projects/giss/prod_input_files/emis/BBURN_ALT/20240517/BBURN_GFED_4s/monthly/NAT',
        'dir_obs_bio': '/discover/nobackup/nkiang/DATA/Vegcov/V2HX2_EntGVSD_v1.1.2_16G_Spawn2020Sa_biomass_agb_2010_ann_pure.nc',
        'dir_obs_ba': '/discover/nobackup/projects/giss_ana/users/kmezuman/GFED5/BA/upscale',
        'dir_obs_wglc': '/discover/nobackup/projects/giss_ana/users/kmezuman/WGLC/regridding/upscale',
        'nlat': 90,
        'nlon': 144,
        'cmap': 'jet',
        'iyear': 2010,
        'fyear': 2010,
        'experiment': 'all',
        # 'all' burns all aboveground vegetation 
        # 'lfine' burns only fine litter 
        # 'lfine&CWD' burns fine litter and coarse woody debris litter
        # 'lfine&CWD&fol' burns fine litter, coarse woody debris litter and foliage
        'srang': ["009", "010"],
        'trang': ["002", "004", "006", "007", "008", "009", "010"],
        'grang': ["011", "012", "013", "014"]
    }
    return config

def extract_scaling_factor(units: str) -> Tuple[float, str]:
    """
    Extract scaling factor from units string.
    
    Args:
        units: String containing units with optional scaling factor
        
    Returns:
        Tuple[float, str]: (scaling_factor, units_without_scaling)
    """
    match = re.match(r"^(10\^(-?\d+)|[-+]?\d*\.?\d+([eE][-+]?\d+)?)\s*(.*)$", units)
    if match:
        if match.group(1).startswith("10^"):
            scaling_factor = float(10) ** float(match.group(2))
        else:
            scaling_factor = float(match.group(1))
        unit = match.group(4)
        return scaling_factor, unit
    #to do: should it return scaling_factor?
    return 1.0, units

def calculate_emis(vrang: List[str], BA: np.ndarray, f: Dataset, 
                  missing_val: Optional[float], nan_mat: Optional[np.ndarray], 
                  lats: np.ndarray, lons: np.ndarray, 
                  zero_mat: np.ndarray, experiment: str) -> np.ndarray:
    """Calculate emissions based on vegetation type and other factors."""
    fuelC = np.zeros((len(lats), len(lons)), dtype=float)
    vf_arr = np.zeros((len(lats), len(lons)), dtype=float)
    
    for t in vrang:
        vf = f.variables[f"ra001{t}"][:]  # vegetation cover fraction
        vf = np.where(vf < 0., zero_mat, vf)
        vf_arr += vf

        try:
            if experiment == 'all':
                # Calculate total above-ground carbon components
                carbon_vars = [
                    (f"ra018{t}", "labile"),
                    (f"ra019{t}", "foliage"),
                    (f"ra020{t}", "sapwood"),
                    (f"ra021{t}", "heartwood")
                ]
                for var_name, _ in carbon_vars:
                    carbon = f.variables[var_name][:]
                    carbon = np.where(carbon < 0., zero_mat, carbon)
                    fuelC += (vf * carbon)

            elif experiment == 'lfine':
                # Fine litter components
                for suffix in ['032', '033', '037']:  # metabolic, structural, microbial
                    carbon = f.variables[f"ra{suffix}{t}"][:]
                    carbon = np.where(carbon < 0., zero_mat, carbon)
                    fuelC += (vf * carbon)

            elif experiment == 'lfine&CWD':
                # Fine litter and CWD components
                for suffix in ['032', '033', '037', '036']:
                    carbon = f.variables[f"ra{suffix}{t}"][:]
                    carbon = np.where(carbon < 0., zero_mat, carbon)
                    fuelC += (vf * carbon)

            elif experiment == 'lfine&CWD&fol':
                # Fine litter, CWD, and foliage components
                for suffix in ['032', '033', '037', '036', '019']:
                    carbon = f.variables[f"ra{suffix}{t}"][:]
                    carbon = np.where(carbon < 0., zero_mat, carbon)
                    fuelC += (vf * carbon)
            else:
                raise ValueError(f'Unsupported experiment type: {experiment}')

        except Exception as e:
            logging.error(f"Error processing carbon variables for type {t}: {str(e)}")
            raise

    # Safe division with zero handling
    emis_type = np.divide(BA * fuelC, vf_arr, 
                         where=vf_arr != 0, 
                         out=np.full_like(vf_arr, 0.))
    
    return emis_type

def calculate_spawnemis(vrang: List[str], BA: np.ndarray, file_path: str, 
                       f: Dataset, lats: np.ndarray, lons: np.ndarray, 
                       zero_mat: np.ndarray) -> np.ndarray:
    """Calculate spawn emissions based on vegetation type and other factors."""
    try:
        with Dataset(file_path, 'r') as d:
            vf_arr = np.zeros((len(lats), len(lons)), dtype=float)
            mult_arr = np.zeros((len(lats), len(lons)), dtype=float)

            for t in vrang:
                vf = f.variables[f"ra001{t}"][:]
                vf = np.where(vf < 0., zero_mat, vf)
                vf_arr += vf
                
                SpawnCM_ABV = d.variables['biomass_agb'][t,:,:]  # [kg m-2]
                mult = vf * SpawnCM_ABV
                mult_arr += mult

            # Safe division with zero handling
            emis_type = np.divide((BA * mult_arr), vf_arr,
                                where=vf_arr != 0,
                                out=np.full_like(vf_arr, 0.))
            emis_type *= 44./12.  # kg to kgC conversion

            return emis_type
    except Exception as e:
        logging.error(f"Error processing spawn emissions: {str(e)}")
        raise

def modelE_emis(year: int, config: Dict, months: List[str], 
                s_in_day: float, kgtog: float, axyp: np.ndarray) -> Tuple[np.ndarray, str]:
    """Calculate ModelE emissions for a given year."""
    ann_sum_pyrE = 0
    
    for month in months:
        tracer_filename = f"{month}{year}.taijnk_CCycle_E6obioF40.nc"
        filepath = os.path.join(config['dir_sim'], tracer_filename)
        
        if os.path.exists(filepath):
            try:
                with nc.Dataset(filepath) as f:
                    emis_pyrE = f.variables['CO2n_pyrE_src'][:]
                    units = f.variables['CO2n_pyrE_src'].units
                    scaling_factor, _ = extract_scaling_factor(units)
                    
                    # Convert units
                    emis_pyrE *= float(scaling_factor)  # kgCO2 m-2 s-1
                    ndays = calendar.monthrange(year, months.index(month)+1)[1]
                    emis_pyrE *= (ndays * s_in_day)  # kgCO2 m-2 yr-1
                    emis_pyrE *= kgtog  # gCO2 m-2 M-1
                    ann_sum_pyrE += emis_pyrE
                    
                tot_emis_pyrE = np.nansum(ann_sum_pyrE * axyp)  # gCO2 yr-1
                tot_emis_pyrE = format(tot_emis_pyrE, '.3e')
            except Exception as e:
                logging.error(f"Error processing ModelE emissions: {str(e)}")
                raise
        else:
            logging.warning(f"File {filepath} not found. Skipping.")
            
    return ann_sum_pyrE, tot_emis_pyrE
################################################################
def modelE_diag(diag, year, config, lons, lats):
    """
    Calculate annual sum of ModelE diagnostics from monthly files.
    For BA, sums BA_tree/shrub/grass components for each month.
    For non-BA diagnostics, applies area weighting using axyp.
    
    Args:
        diag (str): Diagnostic to process ('BA', 'fireCount', 'CtoG', 'flammability')
        year (int): Year to process
        config (dict): Configuration settings
        lons (np.array): Longitude array
        lats (np.array): Latitude array
    
    Returns:
        tuple: (annual_sum_array, total_formatted_string)
    """
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
              'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    m2toMha = 1E-6  # Convert m² to Mha
    
    # Initialize arrays
    nlat = len(lats)
    nlon = len(lons)
    ann_sum = np.zeros((nlat, nlon), dtype=float)
    zero_mat = np.zeros((nlat, nlon), dtype=float)

    for month in months:
        try:
            monthly_filename = f"{month}{year}.aijnk_CCycle_E6obioF40.nc"
            filepath = os.path.join(config['dir_sim'],'monthly', monthly_filename)
            
            if not os.path.exists(filepath):
                logging.warning(f"File {filepath} not found. Skipping month.")
                continue
                
            with nc.Dataset(filepath) as f:
                if diag == 'BA':
                    # Sum the three BA components for each month
                    ba_types = ['BA_tree', 'BA_shrub', 'BA_grass']
                    month_sum = np.zeros((nlat, nlon), dtype=float)
                    for ba_type in ba_types:
                        if ba_type in f.variables:
                            ba_data = f.variables[ba_type][:]
                            ba_data *= m2toMha  # Convert to Mha
                            month_sum += ba_data
                        else:
                            logging.warning(f"Variable {ba_type} not found in file for {month}")
                    ann_sum += month_sum
                else:
                    # Handle other diagnostics
                    if diag in f.variables:
                        axyp = f.variables['axyp'][:]
                        diag_data = f.variables[diag][:]
                        units = f.variables[diag].units
                        scaling_factor, unit = extract_scaling_factor(units)
                        print(diag,scaling_factor,unit)
                        diag_data *= float(scaling_factor)
                        # Add monthly value to annual sum
                        ann_sum += diag_data
                    else:
                        logging.warning(f"Variable {diag} not found in file for {month}")
        
        except Exception as e:
            logging.error(f"Error processing {month} {year}: {str(e)}")
            continue
    
    ann_sum = np.where(ann_sum < 0., zero_mat, ann_sum)
    # Calculate total - different handling for BA vs other diagnostics
    if diag == 'BA':
        total = np.nansum(ann_sum)  # Simple sum for BA
    else:
        # Apply area weighting for non-BA diagnostics
        total = np.nansum(ann_sum * axyp)  # Area-weighted sum
        
    total_formatted = format(total, '.3e')
    
    return ann_sum, total_formatted

###############################################################
def GFED5_BA(year, config, lons, lats):
    """
    Read and process monthly GFED5 burned area data.
    Calculates natural BA as: Total - Deforestation - Cropland - Peatland

    Args:
        year (int): Year to process
        config (dict): Configuration settings
        lons (np.array): Longitude array
        lats (np.array): Latitude array

    Returns:
        tuple: (annual_sum_array, total_formatted_string)
    """
    # Initialize arrays
    nlat = len(lats)
    nlon = len(lons)
    ann_sum = np.zeros((nlat, nlon), dtype=float)
    m2toMha = 1E-6  # Convert m² to Mha

    # Process each month
    for month in range(1, 13):
        try:
            # Construct filename (BAYYYYMM.nc)
            filename = f"BA{year}{month:02d}_90144.nc"
            filepath = os.path.join(config['dir_obs_ba'], filename)

            if not os.path.exists(filepath):
                logging.warning(f"File {filepath} not found. Skipping month.")
                continue

            with nc.Dataset(filepath) as f:
                # Read each component
                components = {
                    'Total': None,
                    'Defo': None,
                    'Crop': None,
                    'Peat': None
                }

                for comp in components.keys():
                    if comp in f.variables:
                        data = f.variables[comp][:]
                        # Handle missing/fill values if present
                        if hasattr(f.variables[comp], '_FillValue'):
                            fill_value = f.variables[comp]._FillValue
                            data = np.where(data == fill_value, 0, data)
                        # Handle negative values
                        data = np.where(data < 0, 0, data)
                        components[comp] = data
                    else:
                        logging.warning(f"Variable {comp} not found in file for {month}")
                        components[comp] = np.zeros((nlat, nlon))

                # Calculate natural BA
                month_ba = (components['Total'] -
                          components['Defo'] -
                          components['Crop'] -
                          components['Peat'])

                # Ensure no negative values
                month_ba = np.where(month_ba < 0, 0, month_ba)

                # Convert units and add to annual sum
                month_ba *= m2toMha
                ann_sum += month_ba

        except Exception as e:
            logging.error(f"Error processing month {month} of {year}: {str(e)}")
            continue

    # Calculate total
    total = np.nansum(ann_sum)
    total_formatted = format(total, '.3e')

    return ann_sum, total_formatted

# Example usage:
# ba_sum, ba_total = GFED5_BA(2020, config, lons, lats)

###############################################################
def process_lightning_density(year, config):
    """
    Process lightning stroke density from monthly data.
    Maps year to correct indices in time dimension (1-144, where 1 is Jan 2013).
    Converts from strokes/km²/day to strokes/km²/year.

    Args:
        year (int): Year to process (2013-2020)
        filepath (str): Path to netCDF file
        var_name (str): Name of the density variable in the file

    Returns:
        tuple: (annual_sum_array, total_formatted_string)
        Returns (None, None) if year is out of range
    """
    filename = "wglc_timeseries_30m_90144.nc"
    filepath = os.path.join(config['dir_obs_wglc'], filename)

    if not os.path.exists(filepath):
        logging.warning(f"File {filepath} not found.")
        return None, None

    # Check valid year range
    if year < 2013 or year > 2020:
        logging.error(f"Year {year} out of valid range (2013-2020)")
        return None, None

    # Calculate time indices for the requested year
    # Index 1 corresponds to Jan 2013
    start_idx = (year - 2013) * 12  # Starting index for requested year
    time_indices = range(start_idx, start_idx + 12)  # 12 months of data

    try:
        with nc.Dataset(filepath) as f:
            # Get dimensions
            lats = f.variables['latitude'][:]
            lons = f.variables['longitude'][:]

            # Initialize array for annual sum
            annual_density = np.zeros((len(lats), len(lons)), dtype=float)

            # Process each month
            for month, time_idx in enumerate(time_indices):
                # Read density data for the month
                density = f.variables['density'][time_idx,:,:]

                # Get number of days in this month
                ndays = calendar.monthrange(year, month+1)[1]

                # Convert from per day to total for month
                monthly_total = density * ndays

                # Add to annual sum
                annual_density += monthly_total

            # Calculate total (optional)
            total = np.nansum(annual_density)
            total_formatted = format(total, '.3e')

            return annual_density, total_formatted

    except Exception as e:
        logging.error(f"Error processing lightning density data: {str(e)}")
        raise

# Example usage:
# density_sum, total = process_lightning_density(2015,
#                                              '/path/to/lightning.nc',
#                                              'stroke_density')

###############################################################

def GFED4s_emis(year: int, config: Dict, zero_mat: np.ndarray, 
                s_in_day: float, kgtog: float, axyp: np.ndarray) -> Tuple[np.ndarray, str]:
    """Calculate GFED4s emissions for a given year."""
    obs_filepath = os.path.join(config['dir_obs_emis'], f"{year}.nc")
    
    try:
        ann_sum = np.zeros((config['nlat'], config['nlon']), dtype=float)
        with nc.Dataset(obs_filepath) as f_obs:
            for k in range(12):
                GFED_data = f_obs.variables['CO2n'][k, :, :]  # [kg m-2 s-1]
                GFED_CO2 = GFED_data.reshape(config['nlat'], config['nlon'])
                GFED_CO2 = np.where(GFED_CO2 <= 0., zero_mat, GFED_CO2)
                ndays = calendar.monthrange(year, k+1)[1]
                ann_sum += (GFED_CO2 * ndays * s_in_day)  # kgCO2 m-2 M-1
                
            ann_sum *= kgtog  # gCO2 m-2 yr-1
            tot_GFED = np.nansum(ann_sum * axyp)  # gCO2 yr-1
            tot_GFED = format(tot_GFED, '.3e')
            
        return ann_sum, tot_GFED
    except FileNotFoundError:
        logging.warning(f"File {obs_filepath} not found. Skipping.")
        return None, None
    except Exception as e:
        logging.error(f"Error processing GFED4s emissions: {str(e)}")
        raise

def define_subplot(fig, ax, data, lons, lats, cmap, cborientation, fraction, pad,
                  labelpad, fontsize, title, clabel, masx, is_diff=False, glob=None, use_log=False):
    """Define the properties of a subplot with optional difference normalization and log scale."""
    ax.coastlines(color='black')
    ax.add_feature(cfeature.LAND, edgecolor='gray')
    ax.add_feature(cfeature.OCEAN, facecolor='white', edgecolor='none', zorder=1)

    ax.set_title(title, fontsize=10, pad=10)
    
    if glob is not None:
        props = dict(boxstyle="round", facecolor='lightgray', alpha=0.5)
        ax.text(0.5, 1.07, f"Global Total: {glob}", ha="center", va="center",
               transform=ax.transAxes, bbox=props, fontsize=10)

    if is_diff:
        data_min, data_max = data.min(), data.max()
        if data_min == data_max:
            norm = mcolors.Normalize(vmin=data_min - 1, vmax=data_max + 1)
        else:
            abs_max = max(abs(0.25 * data_min), abs(0.25 * data_max))
            norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
        cmap = 'bwr'# Use bwr for difference plots
    elif use_log:
        norm = mcolors.LogNorm(vmin=max(1e-10, data[data > 0].min()),
                               vmax=data.max())
    else:
        norm = None
        cmap = 'jet' if cmap == 'jet' else cmap# Use jet for non-difference plots unless specified

    p = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(),
                      cmap=cmap, norm=norm,
                      vmin=0 if not is_diff and not use_log else None,
                      vmax=masx if not is_diff else None)
                      
    cbar = fig.colorbar(p, ax=ax, orientation=cborientation,
                       fraction=fraction, pad=pad)
    cbar.set_label(clabel, labelpad=labelpad, fontsize=fontsize)

    return ax

def process_data():
    """Main data processing function that integrates various components."""
    config = load_config()
    zero_mat = np.zeros((config['nlat'], config['nlon']), dtype=float)
    spawn_filepath = config['dir_obs_bio']
    kgCtogCO2 = 44./12.*1000.
    kgtog = 1000.
    s_in_day = 60.*60.*24.
    s_in_yr = s_in_day*365.
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
              'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    cbar_max = 900.
    exp = config['experiment']

    for year in range(config['iyear'], config['fyear'] + 1):
        try:
            veg_filename = f"ANN{year}.aijnk_CCycle_E6obioF40.nc"
            filepath = os.path.join(config['dir_sim'], 'ANN_aij', veg_filename)
            logging.info(f"Processing file: {filepath}")

            if not os.path.exists(filepath):
                logging.warning(f"File {filepath} not found. Skipping.")
                continue

            with nc.Dataset(filepath) as f:
                lats = f.variables['lat'][:]
                lons = f.variables['lon'][:]
                axyp = f.variables['axyp'][:]

                # Calculate emissions for each vegetation type
                BA_types = {
                    'grass': ('BA_grass', config['grang']),
                    'shrub': ('BA_shrub', config['srang']),
                    'tree': ('BA_tree', config['trang'])
                }

                emis_tot = np.zeros_like(zero_mat)
                emis_sp_tot = np.zeros_like(zero_mat)

                for ba_name, (ba_var, vrange) in BA_types.items():
                    BA = f.variables[ba_var][:]
                    
                    # Calculate Ent biomass emissions
                    emis = calculate_emis(
                        vrang=vrange,
                        BA=BA,
                        f=f,
                        missing_val=None,
                        nan_mat=None,
                        lats=lats,
                        lons=lons,
                        zero_mat=zero_mat,
                        experiment=exp
                    )
                    emis_tot += emis
                    
                    # Calculate Spawn emissions
                    emis_sp = calculate_spawnemis(
                        vrang=vrange,
                        BA=BA,
                        file_path=spawn_filepath,
                        f=f,
                        lats=lats,
                        lons=lons,
                        zero_mat=zero_mat
                    )
                    emis_sp_tot += emis_sp

                # Process total emissions
                emis_tot *= kgCtogCO2  # Convert to gCO2
                tot_emis = np.nansum(emis_tot)  # Calculate total
                tot_emis = format(tot_emis, '.3e')
                emis_tot /= axyp  # Convert to per area

                # Process spawn emissions
                emis_sp_tot *= kgCtogCO2
                tot_sp_emis = np.nansum(emis_sp_tot)
                tot_sp_emis = format(tot_sp_emis, '.3e')
                emis_sp_tot /= axyp

            # Calculate ModelE emissions
            ann_sum_pyrE, tot_emis_pyrE = modelE_emis(
                year=year,
                config=config,
                months=months,
                s_in_day=s_in_day,
                kgtog=kgtog,
                axyp=axyp
            )

            # Calculate GFED4s emissions
            ann_sum, tot_GFED = GFED4s_emis(
                year=year,
                config=config,
                zero_mat=zero_mat,
                s_in_day=s_in_day,
                kgtog=kgtog,
                axyp=axyp
            )

            # Create visualization
            fig, ax = plt.subplots(2, 2, figsize=(18, 10), 
                                 subplot_kw={'projection': ccrs.PlateCarree()})

            # Plot emissions from pyrE BA and Ent Biomass 
            define_subplot(
                fig=fig,
                ax=ax[0, 0],
                data=emis_tot,
                lons=lons,
                lats=lats,
                cmap=config['cmap'],
                cborientation='horizontal',
                fraction=0.05,
                pad=0.05,
                labelpad=5,
                fontsize=10,
                title=f'Offline Emissions (pyrE BA and Ent Biomass)\nTotal: {tot_emis} [gCO2]',
                clabel='CO2 Emissions [gCO2 m-2 yr-1]',
                masx=cbar_max
            )

            # Plot emissions from Spawn dataset
            define_subplot(
                fig=fig,
                ax=ax[0, 1],
                data=emis_sp_tot,
                lons=lons,
                lats=lats,
                cmap=config['cmap'],
                cborientation='horizontal',
                fraction=0.05,
                pad=0.05,
                labelpad=5,
                fontsize=10,
                title=f'Offline Emissions (pyrE BA and Spawn Biomass)\nTotal: {tot_sp_emis} [gCO2]',
                clabel='CO2 Emissions [gCO2 m-2 yr-1]',
                masx=cbar_max
            )

            # Plot emissions from pyrE
            define_subplot(
                fig=fig,
                ax=ax[1, 0],
                data=ann_sum_pyrE,
                lons=lons,
                lats=lats,
                cmap=config['cmap'],
                cborientation='horizontal',
                fraction=0.05,
                pad=0.05,
                labelpad=5,
                fontsize=10,
                title=f'pyrE fire based emissions\nTotal: {tot_emis_pyrE} [gCO2]',
                clabel='CO2 Emissions [gCO2 m-2 yr-1]',
                masx=cbar_max
            )

            # Plot emissions from GFED4s
            define_subplot(
                fig=fig,
                ax=ax[1, 1],
                data=ann_sum,
                lons=lons,
                lats=lats,
                cmap=config['cmap'],
                cborientation='horizontal',
                fraction=0.05,
                pad=0.05,
                labelpad=5,
                fontsize=10,
                title=f'GFED4s emissions\nTotal: {tot_GFED} [gCO2]',
                clabel='CO2 Emissions [gCO2 m-2 yr-1]',
                masx=cbar_max
            )

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logging.error(f"Error processing year {year}: {str(e)}")
            continue

def tiered_experiments():
    """Run tiered experiments comparing different emission calculation methods."""
    config = load_config()
    zero_mat = np.zeros((config['nlat'], config['nlon']), dtype=float)
    kgCtogCO2 = 44./12.*1000.
    kgtog = 1000.
    s_in_day = 60.*60.*24.
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
              'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    cbar_max = 900.
    experiments = ['all', 'lfine&CWD&fol', 'lfine&CWD', 'lfine']

    for year in range(config['iyear'], config['fyear'] + 1):
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                                   subplot_kw={'projection': ccrs.PlateCarree()})
            axes = axes.flatten()

            veg_filename = f"ANN{year}.aijnk_CCycle_E6obioF40.nc"
            filepath = os.path.join(config['dir_sim'], 'ANN_aij', veg_filename)
            
            with nc.Dataset(filepath) as f:
                lats = f.variables['lat'][:]
                lons = f.variables['lon'][:]
                axyp = f.variables['axyp'][:]

                # Process each experiment
                for i, experiment in enumerate(experiments):
                    emis_total = np.zeros_like(zero_mat)
                    
                    # Calculate emissions for each vegetation type
                    for ba_var, vrange in [
                        ('BA_grass', config['grang']),
                        ('BA_shrub', config['srang']),
                        ('BA_tree', config['trang'])
                    ]:
                        BA = f.variables[ba_var][:]
                        emis = calculate_emis(
                            vrang=vrange,
                            BA=BA,
                            f=f,
                            missing_val=None,
                            nan_mat=None,
                            lats=lats,
                            lons=lons,
                            zero_mat=zero_mat,
                            experiment=experiment
                        )
                        emis_total += emis

                    # Convert units
                    emis_total *= kgCtogCO2
                    tot_emis = format(np.nansum(emis_total), '.3e')
                    emis_total /= axyp

                    # Create subplot for this experiment
                    define_subplot(
                        fig=fig,
                        ax=axes[i],
                        data=emis_total,
                        lons=lons,
                        lats=lats,
                        cmap=config['cmap'],
                        cborientation='horizontal',
                        fraction=0.05,
                        pad=0.05,
                        labelpad=5,
                        fontsize=10,
                        title=f'{experiment} Emissions\nTotal: {tot_emis} [gCO2]',
                        clabel='CO2 Emissions [gCO2 m-2 yr-1]',
                        masx=cbar_max
                    )

            # Add ModelE and GFED4s emissions
            ann_sum_pyrE, tot_emis_pyrE = modelE_emis(
                year, config, months, s_in_day, kgtog, axyp
            )
            ann_sum, tot_GFED = GFED4s_emis(
                year, config, zero_mat, s_in_day, kgtog, axyp
            )

            # Plot ModelE and GFED4s results
            for idx, (data, total, title) in enumerate([
                (ann_sum_pyrE, tot_emis_pyrE, 'ModelE Emissions'),
                (ann_sum, tot_GFED, 'GFED4s Emissions')
            ], start=4):
                if idx < len(axes):
                    define_subplot(
                        fig=fig,
                        ax=axes[idx],
                        data=data,
                        lons=lons,
                        lats=lats,
                        cmap=config['cmap'],
                        cborientation='horizontal',
                        fraction=0.05,
                        pad=0.05,
                        labelpad=5,
                        fontsize=10,
                        title=f'{title}\nTotal: {total} [gCO2]',
                        clabel='CO2 Emissions [gCO2 m-2 yr-1]',
                        masx=cbar_max
                    )

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logging.error(f"Error in tiered experiments for year {year}: {str(e)}")
            continue
###########################################################################
def input_eval():
    """
    Evaluate ModelE diagnostics (BA and CtoG) against observational data (GFED5 and lightning).
    Creates 6 plots: Lightning comparisons (top) and BA comparisons (bottom).
    """
    # Load configuration and initialize
    config = load_config()
    year = config['iyear']
    lyear = 2013 if year < 2013 else year
    BAcbarmax = 100000
    
    try:
        # Load basic grid information from model file
        veg_filename = f"ANN{year}.aijnk_CCycle_E6obioF40.nc"
        filepath = os.path.join(config['dir_sim'], 'ANN_aij', veg_filename)
        
        with nc.Dataset(filepath) as f:
            lats = f.variables['lat'][:]
            lons = f.variables['lon'][:]
            axyp = f.variables['axyp'][:]
            
        # Process data
        modelE_ctog_sum, modelE_ctog_tot = modelE_diag('CtoG', year, config, lons, lats)
        light_sum, light_tot = process_lightning_density(lyear, config)
        modelE_ba_sum, modelE_ba_tot = modelE_diag('BA', year, config, lons, lats)
        gfed_ba_sum, gfed_ba_tot = GFED5_BA(year, config, lons, lats)

        if any(x is None for x in [modelE_ba_sum, gfed_ba_sum, modelE_ctog_sum, light_sum]):
            logging.error("One or more datasets returned None")
            return

        # Calculate differences
        ctog_diff = modelE_ctog_sum - light_sum
        ba_diff = modelE_ba_sum - gfed_ba_sum

        # Create figure with 6 subplots (2x3)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12),
                                 subplot_kw={'projection': ccrs.PlateCarree()})

        # First row: Lightning plots
        define_subplot(fig, axes[0,0], modelE_ctog_sum, lons, lats,
                       cmap='jet',
                       cborientation='horizontal',
                       fraction=0.05,
                       pad=0.05,
                       labelpad=5,
                       fontsize=10,
                       title=f'ModelE Lightning {year}\nTotal: {modelE_ctog_tot} [fl km⁻² yr⁻¹]',
                       clabel='Lightning Density [fl km⁻² yr⁻¹]',
                       masx=None,
                       use_log=True)

        define_subplot(fig, axes[0,1], light_sum, lons, lats,
                       cmap='jet',
                       cborientation='horizontal',
                       fraction=0.05,
                       pad=0.05,
                       labelpad=5,
                       fontsize=10,
                       title=f'Observed Lightning {lyear}\nTotal: {light_tot} [fl km⁻² yr⁻¹]',
                       clabel='Lightning Density [fl km⁻² yr⁻¹]',
                       masx=None,
                       use_log=True)

        define_subplot(fig, axes[0,2], ctog_diff, lons, lats,
                       cmap='bwr',
                       cborientation='horizontal',
                       fraction=0.05,
                       pad=0.05,
                       labelpad=5,
                       fontsize=10,
                       title=f'Lightning Difference (ModelE - Obs)',
                       clabel='Lightning Density Difference [fl km⁻² yr⁻¹]',
                       masx=None,
                       is_diff=True)

        # Second row: Burned Area plots
        define_subplot(fig, axes[1,0], modelE_ba_sum, lons, lats,
                       cmap='jet',
                       cborientation='horizontal',
                       fraction=0.05,
                       pad=0.05,
                       labelpad=5,
                       fontsize=10,
                       title=f'ModelE Burned Area {year}\nTotal: {modelE_ba_tot} [Mha]',
                       clabel='Burned Area [Mha]',
                       #masx=BAcbarmax,
                       masx=None,
                       use_log=True)

        define_subplot(fig, axes[1,1], gfed_ba_sum, lons, lats,
                       cmap='jet',
                       cborientation='horizontal',
                       fraction=0.05,
                       pad=0.05,
                       labelpad=5,
                       fontsize=10,
                       title=f'GFED5 Burned Area {year}\nTotal: {gfed_ba_tot} [Mha]',
                       clabel='Burned Area [Mha]',
                       #masx=BAcbarmax,
                       masx=None,
                       use_log=True)

        define_subplot(fig, axes[1,2], ba_diff, lons, lats,
                       cmap='bwr',
                       cborientation='horizontal',
                       fraction=0.05,
                       pad=0.05,
                       labelpad=5,
                       fontsize=10,
                       title=f'BA Difference (ModelE - GFED5)',
                       clabel='Burned Area Difference [Mha]',
                       masx=None,
                       is_diff=True)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logging.error(f"Error in input evaluation: {str(e)}")
        raise
if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == '--tiered':
            logging.info("Running tiered experiments")
            tiered_experiments()
        elif len(sys.argv) > 1 and sys.argv[1] == '--eval':
            logging.info("Running driver evaluation")
            input_eval()
        else:
            logging.info("Running standard process")
            process_data()
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise
