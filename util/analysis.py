"""
Time series analysis module.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SeasonalStatistics:
    """Container for seasonal statistics."""
    months: np.ndarray
    means: np.ndarray
    stds: np.ndarray


class TimeSeriesAnalyzer:
    """Analyze time series data."""
    
    @staticmethod
    def calculate_seasonal_cycle(
        time_series: np.ndarray,
        annual: bool = False
    ) -> SeasonalStatistics:
        """
        Calculate seasonal statistics from time series.
        
        Args:
            time_series: Array of shape (n, 2) with [time, value]
            annual: Whether data is annual (if True, returns yearly stats)
        
        Returns:
            SeasonalStatistics object
        """
        if time_series is None or len(time_series) == 0:
            return SeasonalStatistics(
                months=np.arange(1, 13),
                means=np.zeros(12),
                stds=np.zeros(12)
            )
        
        # Create DataFrame for easier grouping
        if annual:
            df = pd.DataFrame(time_series, columns=['year', 'value'])
            df['month'] = 1  # Dummy month for annual data
        else:
            df = pd.DataFrame(time_series, columns=['decimal_year', 'value'])
            # Extract month from decimal year
            df['month'] = ((df['decimal_year'] % 1) * 12 + 1).round().astype(int)
            df['month'] = df['month'].clip(1, 12)
        
        # Calculate statistics by month
        stats = df.groupby('month')['value'].agg(['mean', 'std']).reset_index()
        
        return SeasonalStatistics(
            months=stats['month'].values,
            means=stats['mean'].values,
            stds=stats['std'].values
        )
    
    @staticmethod
    def calculate_annual_totals(time_series: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate annual totals from monthly time series.
        
        Args:
            time_series: Array of shape (n, 2) with [decimal_year, value]
        
        Returns:
            Array of shape (n_years, 2) with [year, annual_total]
        """
        if time_series is None or len(time_series) == 0:
            return None
        
        times = time_series[:, 0]
        values = time_series[:, 1]
        
        # Get unique years
        years = np.floor(times).astype(int)
        unique_years = np.unique(years)
        
        # Sum values by year
        annual_data = []
        for year in unique_years:
            year_mask = (years == year)
            year_total = np.sum(values[year_mask])
            annual_data.append([year, year_total])
        
        return np.array(annual_data) if annual_data else None
    
    @staticmethod
    def calculate_trend(
        time_series: np.ndarray,
        method: str = 'linear'
    ) -> Tuple[float, float]:
        """
        Calculate trend in time series.
        
        Args:
            time_series: Array of shape (n, 2) with [time, value]
            method: Trend calculation method ('linear', 'theil-sen')
        
        Returns:
            Tuple of (slope, intercept)
        """
        if time_series is None or len(time_series) < 2:
            return 0.0, 0.0
        
        times = time_series[:, 0]
        values = time_series[:, 1]
        
        if method == 'linear':
            # Simple linear regression
            coeffs = np.polyfit(times, values, 1)
            return coeffs[0], coeffs[1]
        else:
            # Could implement Theil-Sen estimator
            return 0.0, 0.0
    
    @staticmethod
    def calculate_anomalies(
        time_series: np.ndarray,
        baseline_period: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Calculate anomalies relative to baseline period.
        
        Args:
            time_series: Array of shape (n, 2) with [time, value]
            baseline_period: Tuple of (start_time, end_time) for baseline
        
        Returns:
            Array of shape (n, 2) with [time, anomaly]
        """
        if time_series is None or len(time_series) == 0:
            return np.array([])
        
        times = time_series[:, 0]
        values = time_series[:, 1]
        
        # Determine baseline
        if baseline_period is not None:
            start, end = baseline_period
            mask = (times >= start) & (times <= end)
            baseline_mean = np.mean(values[mask])
        else:
            baseline_mean = np.mean(values)
        
        # Calculate anomalies
        anomalies = values - baseline_mean
        
        return np.column_stack([times, anomalies])
    
    @staticmethod
    def smooth_time_series(
        time_series: np.ndarray,
        window: int = 12
    ) -> np.ndarray:
        """
        Smooth time series using moving average.
        
        Args:
            time_series: Array of shape (n, 2) with [time, value]
            window: Window size for moving average
        
        Returns:
            Smoothed time series array
        """
        if time_series is None or len(time_series) < window:
            return time_series
        
        times = time_series[:, 0]
        values = time_series[:, 1]
        
        # Apply moving average
        smoothed = pd.Series(values).rolling(
            window=window,
            center=True,
            min_periods=1
        ).mean().values
        
        return np.column_stack([times, smoothed])


class StatisticalTests:
    """Statistical tests for time series."""
    
    @staticmethod
    def mann_kendall_trend(
        time_series: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[bool, float]:
        """
        Perform Mann-Kendall trend test.
        
        Args:
            time_series: Array of shape (n, 2) with [time, value]
            alpha: Significance level
        
        Returns:
            Tuple of (has_trend, p_value)
        """
        # Placeholder for Mann-Kendall test
        # Would implement proper statistical test
        return False, 1.0
    
    @staticmethod
    def calculate_correlation(
        ts1: np.ndarray,
        ts2: np.ndarray,
        method: str = 'pearson'
    ) -> Tuple[float, float]:
        """
        Calculate correlation between two time series.
        
        Args:
            ts1: First time series
            ts2: Second time series
            method: Correlation method ('pearson', 'spearman')
        
        Returns:
            Tuple of (correlation, p_value)
        """
        from scipy import stats
        
        # Find common time points
        times1 = ts1[:, 0]
        times2 = ts2[:, 0]
        values1 = ts1[:, 1]
        values2 = ts2[:, 1]
        
        # Simple implementation - would need proper alignment
        if len(values1) != len(values2):
            return 0.0, 1.0
        
        if method == 'pearson':
            corr, pval = stats.pearsonr(values1, values2)
        else:
            corr, pval = stats.spearmanr(values1, values2)
        
        return corr, pval
