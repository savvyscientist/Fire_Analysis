"""
Configuration module for managing environment variables and application settings.
UPDATED: Added target_units and spatial_aggregation options
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class FigureConfig:
    """Configuration for figure appearance."""
    color: str
    marker: str
    line_style: str
    label: str
    target_units: Optional[str] = None  # Target units for conversion (e.g., "Tg CO2/yr")
    cbarmax: Optional[float] = None


@dataclass
class FolderConfig:
    """Configuration for data source folder."""
    folder_path: str
    file_type: str
    variables: List[str]
    figure_data: FigureConfig
    spatial_aggregation: str = 'total'  # 'total' or 'mean' for spatial aggregation
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FolderConfig':
        """Create FolderConfig from dictionary."""
        figure_data = FigureConfig(**data['figure_data'])
        return cls(
            folder_path=data['folder_path'],
            file_type=data['file_type'],
            variables=data['variables'],
            figure_data=figure_data,
            spatial_aggregation=data.get('spatial_aggregation', 'total')  # Default to 'total'
        )


@dataclass
class TimeAnalysisConfig:
    """Configuration for time series analysis."""
    annual: bool
    title: str
    ylabel: str  # Will be overridden by converted units if target_units specified
    figs_folder: str
    folders: List[FolderConfig]
    save_netcdf: bool = False
    spatial_aggregation: str = 'total'  # Global default: 'total' or 'mean'
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeAnalysisConfig':
        """Create TimeAnalysisConfig from dictionary."""
        figure_data = data.get('time_analysis_figure_data', {})
        folders = [FolderConfig.from_dict(f) for f in data['folders']]
        
        return cls(
            annual=figure_data.get('annual', False),
            title=figure_data.get('title', ''),
            ylabel=figure_data.get('ylabel', ''),
            figs_folder=figure_data.get('figs_folder', ''),
            folders=folders,
            save_netcdf=data.get('save_netcdf', False),
            spatial_aggregation=figure_data.get('spatial_aggregation', 'total')
        )


class ConfigManager:
    """Manages application configuration from JSON file."""
    
    def __init__(self, config_file: str = "utilityEnvVar.json"):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to JSON configuration file
        """
        self.config_file = Path(config_file)
        self._config: Optional[Dict[str, Any]] = None
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            self._config = json.load(f)
        
        return self._config
    
    def get_selected_scripts(self) -> List[str]:
        """Get list of selected scripts to run."""
        if self._config is None:
            self.load()
        return self._config.get('selected_script', [])
    
    def get_time_analysis_config(self) -> Optional[TimeAnalysisConfig]:
        """Get time series analysis configuration."""
        if self._config is None:
            self.load()
        
        if 'time_analysis_version_two' in self._config:
            return TimeAnalysisConfig.from_dict(
                self._config['time_analysis_version_two']
            )
        return None
    
    def get_script_config(self, script_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific script."""
        if self._config is None:
            self.load()
        return self._config.get(script_name)
