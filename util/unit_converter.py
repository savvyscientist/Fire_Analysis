"""
Unit conversion and handling module.
UPDATED: Added target unit conversion and mass/prefix handling
"""
import re
import numpy as np
from typing import Tuple, Optional, List, Dict
from constants import (
    DAYS_TO_SECONDS,
    SECONDS_IN_A_YEAR,
    EARTH_RADIUS_KM,
    EARTH_RADIUS_METERS,
    MONTHLIST,
    MONTHLISTDICT
)


# Mass prefixes and their conversion factors
MASS_PREFIXES: Dict[str, float] = {
    'Tg': 1e12,    # Teragram = 10^12 g = 10^9 kg
    'Gg': 1e9,     # Gigagram = 10^9 g = 10^6 kg
    'Mg': 1e6,     # Megagram = 10^6 g = 10^3 kg = 1 tonne
    'kg': 1e3,     # kilogram = 10^3 g
    'g': 1.0,      # gram (base)
    'Pg': 1e15,    # Petagram = 10^15 g = 10^12 kg
}


def extract_scaling_factor(units: str) -> Tuple[float, str]:
    """
    Extract scaling factor from units string.
    
    Args:
        units: Unit string (e.g., '10^-3 kg/m^2/s')
    
    Returns:
        Tuple of (scaling_factor, cleaned_units)
    
    Example:
        >>> extract_scaling_factor('10^-3 kg/m^2/s')
        (0.001, 'kg/m^2/s')
    """
    try:
        pattern = r"^(10\^(-?\d+)|[-+]?\d*\.?\d+([eE][-+]?\d+)?)\s*(.*)$"
        match = re.match(pattern, units)
        
        if match:
            if match.group(1).startswith("10^"):
                scaling_factor = float(10) ** float(match.group(2))
            else:
                scaling_factor = float(match.group(1))
            unit = match.group(4)
            return scaling_factor, unit
        
        return 1.0, units
    except Exception:
        return 1.0, units


def days_to_months(month: str, year: int) -> int:
    """
    Get number of days in a specific month.
    
    Args:
        month: Month as string ('01'-'12' or 'JAN'-'DEC')
        year: Year (for leap year calculation)
    
    Returns:
        Number of days in the month
    """
    days = MONTHLISTDICT.get(month, 30)
    
    # Handle February in leap years
    if month in ['02', 'FEB']:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            days = 29
    
    return days


def parse_units(unit_str: str) -> Dict[str, any]:
    """
    Parse unit string to extract components.
    
    Args:
        unit_str: Unit string like "Tg CO2/yr" or "kg/m2/s"
    
    Returns:
        Dictionary with parsed components
    """
    result = {
        'mass_prefix': None,
        'mass_value': 1.0,
        'compound': None,
        'per_area': False,
        'per_time': None,
        'original': unit_str
    }
    
    # Check for mass prefix
    for prefix, value in MASS_PREFIXES.items():
        if unit_str.startswith(prefix):
            result['mass_prefix'] = prefix
            result['mass_value'] = value
            # Check for compound (CO2, CH4, etc.)
            compound_match = re.match(r'(\w+)\s+(\w+)', unit_str)
            if compound_match:
                result['compound'] = compound_match.group(2)
            break
    
    # Check for per area
    if '/m2' in unit_str or '/m^2' in unit_str or ' m-2' in unit_str or ' m^-2' in unit_str:
        result['per_area'] = True
    
    # Check for time component
    if '/yr' in unit_str or 'yr-1' in unit_str:
        result['per_time'] = 'year'
    elif '/month' in unit_str or 'month-1' in unit_str:
        result['per_time'] = 'month'
    elif '/s' in unit_str or 's-1' in unit_str:
        result['per_time'] = 'second'
    
    return result


def convert_to_target_units(
    data: np.ndarray,
    current_units: str,
    target_units: str
) -> Tuple[np.ndarray, str]:
    """
    Convert data from current units to target units.
    
    Args:
        data: Data array to convert
        current_units: Current units (e.g., "kg/yr")
        target_units: Target units (e.g., "Tg CO2/yr")
    
    Returns:
        Tuple of (converted_data, final_units_string)
    
    Example:
        >>> convert_to_target_units(data, "kg/yr", "Tg CO2/yr")
        (data / 1e9, "Tg CO2/yr")
    """
    current = parse_units(current_units)
    target = parse_units(target_units)
    
    converted_data = data.copy()
    
    # Mass conversion
    if target['mass_prefix'] and current['mass_prefix']:
        # Convert from current mass to target mass
        conversion_factor = MASS_PREFIXES[current['mass_prefix']] / MASS_PREFIXES[target['mass_prefix']]
        converted_data *= conversion_factor
    elif target['mass_prefix']:
        # Assume current is in grams if no prefix
        conversion_factor = 1.0 / target['mass_value']
        converted_data *= conversion_factor
    
    # Time conversion (if needed)
    if current['per_time'] != target['per_time']:
        if current['per_time'] == 'second' and target['per_time'] == 'year':
            converted_data *= SECONDS_IN_A_YEAR
        elif current['per_time'] == 'month' and target['per_time'] == 'year':
            converted_data *= 12
        elif current['per_time'] == 'year' and target['per_time'] == 'month':
            converted_data /= 12
    
    return converted_data, target_units


class UnitConverter:
    """Handles unit conversions for scientific data."""
    
    # Unit conversion configurations
    UNIT_HANDLERS = {
        'kg CO2n m-2 s-1': {
            'needs_area': True,
            'new_units': 'kg/s'
        },
        'kg m-2 s-1': {
            'needs_area': True,
            'new_units': 'kg/s'
        },
        'kg/m2/s': {
            'needs_area': True,
            'new_units': 'kg/s'
        },
        'm-2 s-1': {
            'scaling': 1.0,
            'new_units': 'm-2 s-1'
        },
        '/m2': {
            'scaling': 1E6 * 1E-10,
            'new_units': 'flashes/km2/yr'
        },
        'm-2': {
            'scaling': 1E6 * 1E-10,
            'new_units': 'flashes/km2/yr'
        }
    }
    
    def __init__(self, grid_area_calculator=None):
        """
        Initialize unit converter.
        
        Args:
            grid_area_calculator: Function to calculate grid cell areas
        """
        self.grid_area_calculator = grid_area_calculator
    
    def convert(
        self,
        data_array,
        units: str,
        monthly: bool = False,
        file_path: Optional[str] = None,
        year: Optional[int] = None,
        variable_names: Optional[List[str]] = None,
        target_units: Optional[str] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Apply unit conversions to data array.
        
        Args:
            data_array: Input data array
            units: Original units string
            monthly: Whether data is monthly
            file_path: Path to file (for extracting month)
            year: Year of data (for leap year handling)
            variable_names: List of variable names
            target_units: Target units to convert to (optional)
        
        Returns:
            Tuple of (converted_data, new_units)
        """
        # Extract scaling factor
        scaling_factor, clean_units = extract_scaling_factor(units)
        
        # Get conversion handler
        handler = self._get_handler(clean_units)
        
        # Apply initial scaling
        scaled_data = data_array * scaling_factor
        
        # Handle area integration
        if handler.get('needs_area', False):
            scaled_data = self._apply_area_scaling(scaled_data)
            new_units = handler['new_units']
        else:
            if 'scaling' in handler:
                scaled_data = scaled_data * handler['scaling']
            new_units = handler['new_units']
        
        # Handle time scaling
        if self._needs_time_scaling(units):
            scaled_data, new_units = self._apply_time_scaling(
                scaled_data, new_units, monthly, file_path, year
            )
        
        # Convert to target units if specified
        if target_units:
            scaled_data, new_units = convert_to_target_units(
                scaled_data, new_units, target_units
            )
        
        return scaled_data, new_units
    
    def _get_handler(self, clean_units: str) -> dict:
        """Get appropriate handler for unit conversion."""
        # Check for exact match
        if clean_units in self.UNIT_HANDLERS:
            return self.UNIT_HANDLERS[clean_units]
        
        # Generic handling for mass flux
        if any(p in clean_units for p in ['kg', 'm-2', 's-1']):
            return {
                'needs_area': True,
                'new_units': 'kg/s'
            }
        
        # Default
        return {
            'scaling': 1.0,
            'new_units': clean_units
        }
    
    def _apply_area_scaling(self, data):
        """Apply grid cell area scaling."""
        if self.grid_area_calculator is None:
            raise ValueError("Grid area calculator not provided")
        
        spatial_shape = data.shape[-2:]
        grid_cell_area = self.grid_area_calculator(
            grid_area_shape=spatial_shape,
            units='m^2'
        )
        return data * grid_cell_area
    
    def _needs_time_scaling(self, units: str) -> bool:
        """Check if time scaling is needed."""
        return 's-1' in units or '/s' in units
    
    def _apply_time_scaling(
        self,
        data,
        units: str,
        monthly: bool,
        file_path: Optional[str],
        year: Optional[int]
    ) -> Tuple[np.ndarray, str]:
        """Apply time-based scaling."""
        if monthly and file_path and year:
            # Extract month from filename
            month = file_path.split(".")[0][-7:-4]
            month_num = MONTHLIST.index(month) + 1
            
            # Calculate seconds in month
            days_in_month = days_to_months(str(month_num).zfill(2), year)
            seconds_in_month = days_in_month * DAYS_TO_SECONDS
            
            scaled_data = data * seconds_in_month
            new_units = units.replace('/s', '/month').replace('s-1', 'month-1')
        else:
            # Annual scaling
            scaled_data = data * SECONDS_IN_A_YEAR
            new_units = units.replace('/s', '/yr').replace('s-1', 'yr-1')
        
        return scaled_data, new_units
