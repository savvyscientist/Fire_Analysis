"""
Visualization module for creating plots and figures.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from analysis import SeasonalStatistics
from data_loader import TimeSeriesData


@dataclass
class PlotStyle:
    """Style configuration for plots."""
    color: str
    marker: str = 'o'
    linestyle: str = '-'
    label: str = ''
    linewidth: float = 2.0
    markersize: float = 6.0
    alpha: float = 1.0


class TimeSeriesPlotter:
    """Create time series visualizations."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize plotter.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
    
    def plot_time_series(
        self,
        datasets: List[Tuple[TimeSeriesData, PlotStyle]],
        title: str = '',
        ylabel: str = '',
        xlabel: str = 'Time',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot time series data.
        
        Args:
            datasets: List of (TimeSeriesData, PlotStyle) tuples
            title: Plot title
            ylabel: Y-axis label
            xlabel: X-axis label
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for data, style in datasets:
            times = data.time_series[:, 0]
            values = data.time_series[:, 1]
            
            # Sort by time to ensure proper line plotting
            sort_idx = np.argsort(times)
            times_sorted = times[sort_idx]
            values_sorted = values[sort_idx]
            
            ax.plot(
                times_sorted, values_sorted,
                color=style.color,
                marker=style.marker,
                linestyle=style.linestyle,
                label=style.label,
                linewidth=style.linewidth,
                markersize=style.markersize,
                alpha=style.alpha
            )
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Set X-axis to show only integer years
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Set y-axis formatting for scientific notation (large values)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        ax.yaxis.major.formatter._useMathText = True
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_seasonal_cycle(
        self,
        seasonal_data_list: List[Tuple[SeasonalStatistics, PlotStyle]],
        title: str = 'Seasonal Cycle',
        ylabel: str = '',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot seasonal cycle statistics.
        
        Args:
            seasonal_data_list: List of (SeasonalStatistics, PlotStyle) tuples
            title: Plot title
            ylabel: Y-axis label
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for stats, style in seasonal_data_list:
            ax.plot(
                stats.months, stats.means,
                color=style.color,
                marker=style.marker,
                linestyle=style.linestyle,
                label=style.label,
                linewidth=style.linewidth,
                markersize=style.markersize
            )
            
            # Add error band if std is available
            if stats.stds is not None and not np.isnan(stats.stds).all():
                ax.fill_between(
                    stats.months,
                    stats.means - stats.stds,
                    stats.means + stats.stds,
                    alpha=0.2,
                    color=style.color
                )
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 
                           'J', 'A', 'S', 'O', 'N', 'D'])
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis formatting for scientific notation (large values)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        ax.yaxis.major.formatter._useMathText = True
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_annual_totals(
        self,
        annual_data_list: List[Tuple[np.ndarray, PlotStyle]],
        title: str = 'Annual Totals',
        ylabel: str = '',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot annual totals.
        
        Args:
            annual_data_list: List of (annual_data, PlotStyle) tuples
            title: Plot title
            ylabel: Y-axis label
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for annual_data, style in annual_data_list:
            years = annual_data[:, 0]
            values = annual_data[:, 1]
            
            ax.plot(
                years, values,
                color=style.color,
                marker=style.marker,
                linestyle=style.linestyle,
                label=style.label,
                linewidth=style.linewidth,
                markersize=style.markersize
            )
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Set X-axis to show only integer years
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Set y-axis formatting for scientific notation (large values)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        ax.yaxis.major.formatter._useMathText = True
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _save_figure(self, fig: plt.Figure, save_path: str, dpi: int = 300):
        """Save figure to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")


class SpatialPlotter:
    """Create spatial map visualizations."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize spatial plotter.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
    
    def plot_spatial_map(
        self,
        data: np.ndarray,
        lon: np.ndarray,
        lat: np.ndarray,
        title: str = '',
        cbar_label: str = '',
        cmap: str = 'RdYlBu_r',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create spatial map visualization.
        
        Args:
            data: 2D data array
            lon: Longitude array
            lat: Latitude array
            title: Plot title
            cbar_label: Colorbar label
            cmap: Colormap name
            vmin: Minimum value for colorbar
            vmax: Maximum value for colorbar
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=self.figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add map features
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        
        # Create meshgrid if needed
        if lon.ndim == 1 and lat.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lon, lat)
        else:
            lon_grid, lat_grid = lon, lat
        
        # Plot data
        im = ax.pcolormesh(
            lon_grid, lat_grid, data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading='auto'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                           pad=0.05, shrink=0.8)
        cbar.set_label(cbar_label, fontsize=12)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig
    
    def create_multi_panel_map(
        self,
        datasets: List[Tuple[np.ndarray, str]],
        lon: np.ndarray,
        lat: np.ndarray,
        overall_title: str = '',
        cbar_label: str = '',
        cmap: str = 'RdYlBu_r',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create multi-panel spatial map.
        
        Args:
            datasets: List of (data, title) tuples
            lon: Longitude array
            lat: Latitude array
            overall_title: Overall figure title
            cbar_label: Colorbar label
            cmap: Colormap name
            save_path: Path to save figure
        
        Returns:
            Matplotlib figure object
        """
        n_panels = len(datasets)
        n_cols = min(3, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
        
        for idx, (data, title) in enumerate(datasets):
            ax = fig.add_subplot(
                n_rows, n_cols, idx + 1,
                projection=ccrs.PlateCarree()
            )
            
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            
            im = ax.pcolormesh(
                lon, lat, data,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                shading='auto'
            )
            
            ax.set_title(title, fontsize=12)
            
            plt.colorbar(im, ax=ax, orientation='horizontal',
                        pad=0.05, shrink=0.8)
        
        fig.suptitle(overall_title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig
