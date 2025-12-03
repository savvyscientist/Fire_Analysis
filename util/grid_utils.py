"""
Grid utilities for spatial calculations.
"""
import numpy as np
from typing import Tuple, Literal
from constants import EARTH_RADIUS_KM, EARTH_RADIUS_METERS


class GridAreaCalculator:
    """Calculate grid cell areas for different projections - OPTIMIZED."""
    
    def __init__(self):
        """Initialize grid area calculator with cache."""
        self._cache = {}
    
    def calculate(
        self,
        grid_area_shape: Tuple[int, int],
        units: Literal["km", "m", "m^2"] = "km"
    ) -> np.ndarray:
        """
        Calculate area of each grid cell - CACHED & VECTORIZED.
        
        Args:
            grid_area_shape: Shape of grid (nlat, nlon)
            units: Units for calculation ("km", "m", or "m^2")
        
        Returns:
            Grid area matrix matching input shape
        """
        cache_key = (grid_area_shape, units)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = self._calculate_grid_area_fast(grid_area_shape, units)
        self._cache[cache_key] = result
        return result
    
    def _calculate_grid_area_fast(
        self,
        grid_area_shape: Tuple[int, int],
        units: str
    ) -> np.ndarray:
        """Core grid area calculation - VECTORIZED for speed."""
        nlat, nlon = grid_area_shape
        
        # Calculate step sizes
        lat_step = 180 / nlat
        lon_step = 360 / nlon
        
        # Convert to radians
        lat_step_rad = np.deg2rad(lat_step)
        lon_step_rad = np.deg2rad(lon_step)
        
        # Determine Earth radius
        earth_radius = self._get_earth_radius(units)
        
        # VECTORIZED: Calculate all latitudes at once
        lat_centers = -90 + (np.arange(nlat) + 0.25) * lat_step
        lat_rad = np.deg2rad(lat_centers)
        
        # VECTORIZED: Calculate areas for all latitude bands at once
        areas = (
            (earth_radius ** 2) *
            lon_step_rad *
            np.abs(np.sin(lat_rad + lat_step_rad / 2) -
                   np.sin(lat_rad - lat_step_rad / 2))
        )
        
        # Broadcast to full grid (all longitudes have same area at given latitude)
        grid_area = np.tile(areas[:, np.newaxis], (1, nlon))
        
        return grid_area
    
    def _get_earth_radius(self, units: str) -> float:
        """Get Earth radius in appropriate units."""
        if units in ["km", "km^2"]:
            return EARTH_RADIUS_KM
        elif units in ["m", "m^2"]:
            return EARTH_RADIUS_METERS
        else:
            raise ValueError(f"Unknown units: {units}")
    
    def clear_cache(self):
        """Clear the area calculation cache."""
        self._cache.clear()


class SpatialOperations:
    """Spatial operations on gridded data."""
    
    @staticmethod
    def regrid_conservative(
        source_data: np.ndarray,
        source_shape: Tuple[int, int],
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Conservative regridding between different grid resolutions.
        
        Args:
            source_data: Source data array
            source_shape: Shape of source grid
            target_shape: Shape of target grid
        
        Returns:
            Regridded data array
        """
        # Placeholder for conservative regridding
        # Would implement proper area-weighted interpolation
        from scipy.ndimage import zoom
        
        zoom_factors = (
            target_shape[0] / source_shape[0],
            target_shape[1] / source_shape[1]
        )
        return zoom(source_data, zoom_factors, order=1)
    
    @staticmethod
    def calculate_spatial_mean(
        data: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """
        Calculate area-weighted spatial mean.
        
        Args:
            data: Data array
            weights: Area weights
        
        Returns:
            Weighted mean value
        """
        return np.sum(data * weights) / np.sum(weights)
    
    @staticmethod
    def mask_ocean(data: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
        """
        Mask ocean grid cells.
        
        Args:
            data: Data array
            land_mask: Boolean land mask (True for land)
        
        Returns:
            Masked data array
        """
        masked_data = data.copy()
        masked_data[~land_mask] = np.nan
        return masked_data
