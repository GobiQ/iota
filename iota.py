#!/usr/bin/env python3
"""
Iota Calculator with Core and Rolling Window Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
from typing import List, Tuple, Dict, Any, Optional
import warnings
import re
import io
import contextlib
import urllib.parse

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Iota Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import dependencies with error handling
@st.cache_data
def check_dependencies():
    """Check if all dependencies are available."""
    missing = []
    
    try:
        from scipy import stats
    except ImportError:
        missing.append("scipy")
    
    try:
        from sim import fetch_backtest, calculate_portfolio_returns
    except ImportError:
        missing.append("sim.py")
    
    # Check for quantstats
    try:
        import quantstats as qs
        # Test basic functionality with proper DatetimeIndex
        test_returns = pd.Series([0.01, -0.01, 0.02, -0.005], 
                                index=pd.date_range('2020-01-01', periods=4, freq='D'))
        qs.stats.sharpe(test_returns)
    except ImportError:
        missing.append("quantstats")
    except Exception as e:
        # quantstats is installed but may have issues
        st.warning(f"quantstats installed but may have issues: {str(e)[:50]}")
        # Try to provide more specific error information
        if "DatetimeIndex" in str(e):
            st.info("quantstats requires proper DatetimeIndex - the app will handle this automatically")
        elif "PeriodIndex" in str(e):
            st.info("quantstats requires DatetimeIndex, not PeriodIndex - the app will convert automatically")
    
    return missing

# Check dependencies at startup
missing_deps = check_dependencies()
if missing_deps:
    st.error(f"Missing dependencies: {', '.join(missing_deps)}")
    st.markdown("""
    **Required files:**
    - `sim.py` - Your portfolio calculation module
    - Install scipy: `pip install scipy`
    - Install quantstats: `pip install quantstats`
    
    Make sure `sim.py` is in the same directory as this Streamlit app.
    """)
    st.stop()

# Now import everything
from scipy import stats
from sim import fetch_backtest, calculate_portfolio_returns

# Import quantstats with fallback
try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
    # Configure quantstats
    qs.extend_pandas()
except ImportError:
    QUANTSTATS_AVAILABLE = False
    st.warning("quantstats not available - using internal calculations")

def test_quantstats_compatibility():
    """Test if quantstats is working properly with our data format."""
    if not QUANTSTATS_AVAILABLE:
        return False, "quantstats not installed"
    
    try:
        # Test with various index types that might be encountered
        test_cases = [
            # Case 1: DatetimeIndex
            pd.Series([0.01, -0.01, 0.02, -0.005], 
                     index=pd.date_range('2020-01-01', periods=4, freq='D')),
            # Case 2: Range index (will be converted)
            pd.Series([0.01, -0.01, 0.02, -0.005], index=range(4)),
            # Case 3: String dates
            pd.Series([0.01, -0.01, 0.02, -0.005], 
                     index=['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']),
        ]
        
        for i, test_series in enumerate(test_cases):
            try:
                # Test the ensure_datetime_index function
                converted_series = ensure_datetime_index(test_series.copy())
                
                # Test quantstats functions
                sharpe = qs.stats.sharpe(converted_series)
                sortino = qs.stats.sortino(converted_series)
                
                if not (np.isfinite(sharpe) and np.isfinite(sortino)):
                    return False, f"quantstats returned non-finite values for test case {i+1}"
                    
            except Exception as e:
                return False, f"quantstats failed for test case {i+1}: {str(e)}"
        
        return True, "quantstats working properly"
        
    except Exception as e:
        return False, f"quantstats compatibility test failed: {str(e)}"

def ensure_datetime_index(returns: pd.Series) -> pd.Series:
    """Ensure the series has a proper DatetimeIndex for quantstats compatibility."""
    if returns.empty:
        return returns
    
    # Debug: Log the index type and first few values
    try:
        st.debug(f"Index type: {type(returns.index)}")
        if len(returns.index) > 0:
            st.debug(f"First index value: {returns.index[0]} (type: {type(returns.index[0])})")
    except:
        pass
    
    # If index is already DatetimeIndex, return as is
    if isinstance(returns.index, pd.DatetimeIndex):
        return returns
    
    # If index is PeriodIndex, convert to DatetimeIndex
    if isinstance(returns.index, pd.PeriodIndex):
        returns.index = returns.index.to_timestamp()
        return returns
    
    # If index is date objects, convert to DatetimeIndex
    if len(returns.index) > 0 and hasattr(returns.index[0], 'year'):  # Check if it's date-like
        returns.index = pd.to_datetime(returns.index)
        return returns
    
    # If index is strings, try to parse as dates
    try:
        returns.index = pd.to_datetime(returns.index)
        return returns
    except Exception as e:
        st.debug(f"Failed to parse index as dates: {str(e)}")
    
    # If all else fails, create a dummy DatetimeIndex
    # This is a fallback for when the index can't be converted
    try:
        # Create a reasonable date range based on the number of periods
        start_date = pd.Timestamp('2020-01-01')
        dummy_index = pd.date_range(start=start_date, periods=len(returns), freq='D')
        returns.index = dummy_index
        st.debug(f"Created dummy DatetimeIndex with {len(dummy_index)} periods")
    except Exception as e:
        st.debug(f"Failed to create dummy index: {str(e)}")
        # Ultimate fallback - create a simple range index and convert
        try:
            returns.index = range(len(returns))
            returns.index = pd.date_range(start='2020-01-01', periods=len(returns), freq='D')
        except:
            # If even this fails, return the original series
            st.warning("Could not create valid DatetimeIndex for quantstats")
            return returns
    
    return returns

# Add requests import for API calls
try:
    import requests
except ImportError:
    st.error("Missing requests library. Install with: pip install requests")
    st.stop()

# ===== QUANTSTATS WRAPPER FUNCTIONS =====

def qs_sharpe_ratio(returns: pd.Series, use_quantstats: bool = True) -> float:
    """Calculate Sharpe ratio using quantstats if available and enabled, fallback to internal calculation."""
    if QUANTSTATS_AVAILABLE and use_quantstats:
        try:
            # Convert percentage returns to decimal if needed
            if returns.max() > 1.0:  # Likely percentage format
                returns_decimal = returns / 100.0
            else:
                returns_decimal = returns
            
            # Ensure proper DatetimeIndex for quantstats
            returns_decimal = ensure_datetime_index(returns_decimal)
            
            # Double-check that we have a valid DatetimeIndex
            if not isinstance(returns_decimal.index, pd.DatetimeIndex):
                st.debug("Failed to create valid DatetimeIndex, using internal calculation")
                return sharpe_ratio_internal(returns)
            
            # Use quantstats Sharpe calculation
            sharpe = qs.stats.sharpe(returns_decimal)
            return float(sharpe) if np.isfinite(sharpe) else 0.0
        except Exception as e:
            error_msg = str(e)
            if "DatetimeIndex" in error_msg or "PeriodIndex" in error_msg:
                st.debug(f"quantstats index issue: {error_msg[:100]}")
            else:
                st.warning(f"quantstats Sharpe calculation failed: {error_msg[:50]}, using internal calculation")
            return sharpe_ratio_internal(returns)
    else:
        return sharpe_ratio_internal(returns)

def qs_sortino_ratio(returns: pd.Series, use_quantstats: bool = True) -> float:
    """Calculate Sortino ratio using quantstats if available and enabled, fallback to internal calculation."""
    if QUANTSTATS_AVAILABLE and use_quantstats:
        try:
            # Convert percentage returns to decimal if needed
            if returns.max() > 1.0:  # Likely percentage format
                returns_decimal = returns / 100.0
            else:
                returns_decimal = returns
            
            # Ensure proper DatetimeIndex for quantstats
            returns_decimal = ensure_datetime_index(returns_decimal)
            
            # Use quantstats Sortino calculation
            sortino = qs.stats.sortino(returns_decimal)
            return float(sortino) if np.isfinite(sortino) else 0.0
        except Exception as e:
            st.warning(f"quantstats Sortino calculation failed: {str(e)[:50]}, using internal calculation")
            return sortino_ratio_internal(returns)
    else:
        return sortino_ratio_internal(returns)

def qs_cumulative_return(returns: pd.Series, use_quantstats: bool = True) -> float:
    """Calculate cumulative return using quantstats if available and enabled, fallback to internal calculation."""
    if QUANTSTATS_AVAILABLE and use_quantstats:
        try:
            # Convert percentage returns to decimal if needed
            if returns.max() > 1.0:  # Likely percentage format
                returns_decimal = returns / 100.0
            else:
                returns_decimal = returns
            
            # Ensure proper DatetimeIndex for quantstats
            returns_decimal = ensure_datetime_index(returns_decimal)
            
            # Use quantstats cumulative return calculation
            cum_ret = qs.stats.comp(returns_decimal)
            return float(cum_ret) if np.isfinite(cum_ret) else 0.0
        except Exception as e:
            st.warning(f"quantstats cumulative return calculation failed: {str(e)[:50]}, using internal calculation")
            return cumulative_return_internal(returns)
    else:
        return cumulative_return_internal(returns)

def qs_annualized_return(returns: pd.Series, use_quantstats: bool = True) -> float:
    """Calculate annualized return using quantstats if available and enabled, fallback to internal calculation."""
    if QUANTSTATS_AVAILABLE and use_quantstats:
        try:
            # Convert percentage returns to decimal if needed
            if returns.max() > 1.0:  # Likely percentage format
                returns_decimal = returns / 100.0
            else:
                returns_decimal = returns
            
            # Ensure proper DatetimeIndex for quantstats
            returns_decimal = ensure_datetime_index(returns_decimal)
            
            # Use quantstats annualized return calculation
            ann_ret = qs.stats.cagr(returns_decimal)
            return float(ann_ret) if np.isfinite(ann_ret) else 0.0
        except Exception as e:
            st.warning(f"quantstats annualized return calculation failed: {str(e)[:50]}, using internal calculation")
            return window_cagr_internal(returns)
    else:
        return window_cagr_internal(returns)

def qs_volatility(returns: pd.Series, use_quantstats: bool = True) -> float:
    """Calculate volatility using quantstats if available and enabled, fallback to internal calculation."""
    if QUANTSTATS_AVAILABLE and use_quantstats:
        try:
            # Convert percentage returns to decimal if needed
            if returns.max() > 1.0:  # Likely percentage format
                returns_decimal = returns / 100.0
            else:
                returns_decimal = returns
            
            # Ensure proper DatetimeIndex for quantstats
            returns_decimal = ensure_datetime_index(returns_decimal)
            
            # Use quantstats volatility calculation
            vol = qs.stats.volatility(returns_decimal)
            return float(vol) if np.isfinite(vol) else 0.0
        except Exception as e:
            st.warning(f"quantstats volatility calculation failed: {str(e)[:50]}, using internal calculation")
            return returns.std(ddof=1) * np.sqrt(252)
    else:
        return returns.std(ddof=1) * np.sqrt(252)

def qs_max_drawdown(returns: pd.Series, use_quantstats: bool = True) -> float:
    """Calculate maximum drawdown using quantstats if available and enabled, fallback to internal calculation."""
    if QUANTSTATS_AVAILABLE and use_quantstats:
        try:
            # Convert percentage returns to decimal if needed
            if returns.max() > 1.0:  # Likely percentage format
                returns_decimal = returns / 100.0
            else:
                returns_decimal = returns
            
            # Ensure proper DatetimeIndex for quantstats
            returns_decimal = ensure_datetime_index(returns_decimal)
            
            # Use quantstats max drawdown calculation
            mdd = qs.stats.max_drawdown(returns_decimal)
            return float(mdd) if np.isfinite(mdd) else 0.0
        except Exception as e:
            st.warning(f"quantstats max drawdown calculation failed: {str(e)[:50]}, using internal calculation")
            return calculate_max_drawdown_internal(returns)
    else:
        return calculate_max_drawdown_internal(returns)

def qs_calmar_ratio(returns: pd.Series, use_quantstats: bool = True) -> float:
    """Calculate Calmar ratio using quantstats if available and enabled, fallback to internal calculation."""
    if QUANTSTATS_AVAILABLE and use_quantstats:
        try:
            # Convert percentage returns to decimal if needed
            if returns.max() > 1.0:  # Likely percentage format
                returns_decimal = returns / 100.0
            else:
                returns_decimal = returns
            
            # Ensure proper DatetimeIndex for quantstats
            returns_decimal = ensure_datetime_index(returns_decimal)
            
            # Use quantstats Calmar ratio calculation
            calmar = qs.stats.calmar(returns_decimal)
            return float(calmar) if np.isfinite(calmar) else 0.0
        except Exception as e:
            st.warning(f"quantstats Calmar ratio calculation failed: {str(e)[:50]}, using internal calculation")
            return calculate_calmar_ratio_internal(returns)
    else:
        return calculate_calmar_ratio_internal(returns)

# ===== INTERNAL CALCULATION FUNCTIONS (FALLBACK) =====

def cumulative_return_internal(daily_pct: pd.Series) -> float:
    """Total compounded return over the period (decimal)."""
    daily_dec = daily_pct.dropna() / 100.0
    return float(np.prod(1 + daily_dec) - 1) if not daily_dec.empty else 0.0

def window_cagr_internal(daily_pct: pd.Series) -> float:
    """Compounded annual growth rate over window."""
    daily_dec = daily_pct.dropna() / 100.0
    if daily_dec.empty:
        return 0.0
    total_return = np.prod(1 + daily_dec) - 1
    days = len(daily_dec)
    if days < 2:
        return 0.0
    try:
        cagr = (1 + total_return) ** (252 / days) - 1
        return cagr
    except (FloatingPointError, ValueError):
        return 0.0

def sharpe_ratio_internal(daily_pct: pd.Series) -> float:
    daily_dec = daily_pct.dropna() / 100.0
    if daily_dec.std(ddof=0) == 0:
        return 0.0
    return (daily_dec.mean() / daily_dec.std(ddof=0)) * np.sqrt(252)

def sortino_ratio_internal(daily_pct: pd.Series) -> float:
    """Enhanced Sortino ratio with proper zero-downside handling."""
    daily_dec = daily_pct.dropna() / 100.0
    if daily_dec.empty:
        return 0.0
    
    downside = daily_dec[daily_dec < 0]
    mean_return = daily_dec.mean()
    
    if len(downside) == 0:
        if mean_return > 0:
            return np.inf
        else:
            return 0.0
    
    downside_std = downside.std(ddof=0)
    if downside_std == 0:
        return 0.0
    
    return (mean_return / downside_std) * np.sqrt(252)

def calculate_max_drawdown_internal(returns: pd.Series) -> float:
    """Calculate maximum drawdown using internal calculation."""
    if returns.empty:
        return 0.0
    
    # Convert to decimal if needed
    if returns.max() > 1.0:  # Likely percentage format
        returns_decimal = returns / 100.0
    else:
        returns_decimal = returns
    
    # Calculate cumulative returns
    cum_returns = (1 + returns_decimal).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.expanding().max()
    
    # Calculate drawdown
    drawdown = (cum_returns - running_max) / running_max
    
    # Return maximum drawdown
    return float(drawdown.min()) if not drawdown.empty else 0.0

def calculate_calmar_ratio_internal(returns: pd.Series) -> float:
    """Calculate Calmar ratio using internal calculation."""
    if returns.empty:
        return 0.0
    
    # Get annualized return
    ann_return = qs_annualized_return(returns)
    
    # Get max drawdown
    max_dd = qs_max_drawdown(returns)
    
    # Calculate Calmar ratio
    if abs(max_dd) < 1e-6:
        return 0.0
    
    return ann_return / abs(max_dd)

# ===== CORE FUNCTIONS (Enhanced from Iota.py) =====

def parse_exclusion_input(user_str: str) -> List[Tuple[date, date]]:
    """Return list of date ranges from user string."""
    if not user_str.strip():
        return []
    out: List[Tuple[date, date]] = []
    for token in user_str.split(","):
        token = token.strip()
        if not token:
            continue
        found = re.findall(r"\d{4}-\d{2}-\d{2}", token)
        if len(found) != 2:
            st.warning(f"Skipping unparsable exclusion token: '{token}'.")
            continue
        a, b = [datetime.strptime(d, "%Y-%m-%d").date() for d in found]
        out.append((min(a, b), max(a, b)))
    return out

def cumulative_return(daily_pct: pd.Series, use_quantstats: bool = True) -> float:
    """Total compounded return over the period (decimal) - uses quantstats if available."""
    return qs_cumulative_return(daily_pct, use_quantstats)

def window_cagr(daily_pct: pd.Series, use_quantstats: bool = True) -> float:
    """Compounded annual growth rate over window - uses quantstats if available."""
    return qs_annualized_return(daily_pct, use_quantstats)

def sharpe_ratio(daily_pct: pd.Series, use_quantstats: bool = True) -> float:
    """Sharpe ratio calculation - uses quantstats if available."""
    return qs_sharpe_ratio(daily_pct, use_quantstats)

def sortino_ratio(daily_pct: pd.Series, use_quantstats: bool = True) -> float:
    """Sortino ratio calculation - uses quantstats if available."""
    return qs_sortino_ratio(daily_pct, use_quantstats)

def assess_sample_reliability(n_is: int, n_oos: int) -> str:
    """Assess statistical reliability based on sample sizes."""
    min_size = min(n_is, n_oos)
    
    if min_size >= 378:
        return "HIGH_CONFIDENCE"
    elif min_size >= 189:
        return "MODERATE_CONFIDENCE"  
    elif min_size >= 90:
        return "LOW_CONFIDENCE"
    else:
        return "INSUFFICIENT_DATA"

def build_slices(is_ret: pd.Series, slice_len: int, n_slices: int, overlap: bool) -> List[pd.Series]:
    """Return list of IS slices each of length slice_len."""
    total_is = len(is_ret)
    max_start = total_is - slice_len

    if max_start < 0:
        return []

    if not overlap:
        slices: List[pd.Series] = []
        end_idx = total_is
        while len(slices) < n_slices and end_idx >= slice_len:
            seg = is_ret.iloc[end_idx - slice_len : end_idx]
            if len(seg) == slice_len:
                slices.append(seg)
            end_idx -= slice_len
        return slices

    if n_slices == 1:
        starts = [max_start]
    else:
        starts = np.linspace(0, max_start, n_slices, dtype=int).tolist()
    starts = sorted(dict.fromkeys(starts))

    return [is_ret.iloc[s : s + slice_len] for s in starts]

def detect_distribution_characteristics(is_values: np.ndarray) -> Dict[str, Any]:
    """Detect distribution shape and characteristics for robust iota calculation."""
    if len(is_values) < 10:
        return {
            'is_normal': False,
            'is_skewed': False,
            'has_fat_tails': False,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'recommended_method': 'standard',
            'confidence': 'low'
        }
    
    try:
        from scipy import stats
        
        # Test for normality using D'Agostino K^2 test
        _, p_value = stats.normaltest(is_values)
        is_normal = p_value > 0.05
        
        # Calculate skewness and kurtosis
        skewness = stats.skew(is_values)
        kurtosis = stats.kurtosis(is_values)
        
        # Determine if distribution is problematic (accurate thresholds)
        is_skewed = abs(skewness) > 1.0  # Standard threshold for significant skewness
        has_fat_tails = kurtosis > 3.0   # Standard threshold for excess kurtosis
        
        # Determine recommended calculation method (accuracy-focused approach)
        if is_normal and not is_skewed and not has_fat_tails:
            recommended_method = 'standard'
            confidence = 'high'
        elif is_skewed or has_fat_tails:
            recommended_method = 'robust'
            confidence = 'medium'
        else:
            # For unclear cases, use robust method as it's more general
            recommended_method = 'robust'
            confidence = 'medium'
        
        return {
            'is_normal': is_normal,
            'is_skewed': is_skewed,
            'has_fat_tails': has_fat_tails,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'normality_p_value': p_value,
            'recommended_method': recommended_method,
            'confidence': confidence
        }
    except ImportError:
        # Fallback if scipy is not available
        return {
            'is_normal': True,
            'is_skewed': False,
            'has_fat_tails': False,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'recommended_method': 'standard',
            'confidence': 'low'
        }

def compute_robust_iota(is_values: np.ndarray, oos_value: float, n_oos: int, lower_is_better: bool = False) -> float:
    """Compute iota using robust statistics (median and IQR)."""
    if len(is_values) < 10:
        return 0.0
    
    finite_is = is_values[np.isfinite(is_values)]
    if len(finite_is) < 2:
        return 0.0
    
    median_is = np.median(finite_is)
    q25, q75 = np.percentile(finite_is, [25, 75])
    iqr = q75 - q25
    
    # Use IQR-based "standard deviation" (IQR/1.35 for normal distributions)
    # This is more robust to outliers than standard deviation
    robust_std = iqr / 1.35
    
    if robust_std < 1e-6:
        return 0.0
    
    standardized_diff = (oos_value - median_is) / robust_std
    
    if lower_is_better:
        standardized_diff = -standardized_diff
    
    w = min(1.0, np.sqrt(n_oos / 252))
    return w * standardized_diff

def compute_percentile_iota(is_values: np.ndarray, oos_value: float, n_oos: int, lower_is_better: bool = False) -> float:
    """Compute iota using percentile-based approach."""
    if len(is_values) < 10:
        return 0.0
    
    finite_is = is_values[np.isfinite(is_values)]
    if len(finite_is) < 2:
        return 0.0
    
    # Find what percentile the OOS value falls in
    percentile = stats.percentileofscore(finite_is, oos_value)
    
    # Convert percentile to z-score using inverse normal CDF
    # This provides a proper conversion from percentile to standard normal z-score
    try:
        from scipy.stats import norm
        # Convert percentile to decimal (0-1)
        percentile_decimal = percentile / 100.0
        
        # Handle edge cases
        if percentile_decimal <= 0:
            z_score = -10.0  # Very low percentile
        elif percentile_decimal >= 1:
            z_score = 10.0   # Very high percentile
        else:
            # Use inverse normal CDF to get z-score
            z_score = norm.ppf(percentile_decimal)
            
            # Don't cap the values - let them be as extreme as the data suggests
            # This allows for proper representation of truly extreme performance
    except ImportError:
        # Fallback method if scipy.norm is not available
        if percentile == 50:
            z_score = 0.0
        elif percentile > 50:
            # Above median - use upper tail
            z_score = (percentile - 50) / 34  # 34% is roughly 1 std dev in normal dist
        else:
            # Below median - use lower tail
            z_score = (percentile - 50) / 34
    
    if lower_is_better:
        z_score = -z_score
    
    w = min(1.0, np.sqrt(n_oos / 252))
    return w * z_score

def compute_iota(is_metric: float, oos_metric: float, n_oos: int, n_ref: int = 252, eps: float = 1e-6, 
                 lower_is_better: bool = False, is_values: np.ndarray = None, use_distribution_aware: bool = False) -> float:
    """Iota calculation with optional distribution-aware methods."""
    if np.isinf(oos_metric):
        return 2.0 if not lower_is_better else -2.0
    
    if is_values is None:
        return 0.0
    
    finite_is = is_values[np.isfinite(is_values)]
    if len(finite_is) < 2:
        return 0.0
    
    # Use distribution-aware methods only if explicitly requested
    if use_distribution_aware:
        # Detect distribution characteristics
        dist_info = detect_distribution_characteristics(finite_is)
        
        # Choose calculation method based on distribution
        if dist_info['recommended_method'] == 'robust':
            return compute_robust_iota(finite_is, oos_metric, n_oos, lower_is_better)
        elif dist_info['recommended_method'] == 'percentile':
            return compute_percentile_iota(finite_is, oos_metric, n_oos, lower_is_better)
    
    # Default to standard method (original behavior)
    is_median = np.median(finite_is)
    is_std = np.std(finite_is, ddof=1)
    
    if is_std < eps:
        return 0.0
    
    standardized_diff = (oos_metric - is_median) / is_std
    
    if lower_is_better:
        standardized_diff = -standardized_diff
    
    w = min(1.0, np.sqrt(n_oos / n_ref))
    
    return w * standardized_diff

def iota_to_persistence_rating(iota_val: float, max_rating: int = 500) -> int:
    """Convert iota to persistence rating."""
    if not np.isfinite(iota_val):
        return 100
    
    k = 0.5
    rating = 100 * np.exp(k * iota_val)
    return max(1, min(max_rating, int(round(rating))))

def interpret_iota_directly(iota_val: float) -> str:
    """Direct interpretation of standardized iota values."""
    if not np.isfinite(iota_val):
        return "UNDEFINED"
    
    if iota_val >= 2.0:
        return "🔥 EXCEPTIONAL: OOS significantly above expectations"
    elif iota_val >= 1.0:
        return "✅ EXCELLENT: OOS well above expectations"
    elif iota_val >= 0.5:
        return "👍 GOOD: OOS above expectations"
    elif iota_val >= 0.25:
        return "📈 SLIGHT_IMPROVEMENT: OOS mildly above expectations"
    elif iota_val >= -0.25:
        return "🎯 OOS closely matches backtest"
    elif iota_val >= -0.5:
        return "📉 OOS slightly below expectations"
    elif iota_val >= -1.0:
        return "🚨 WARNING: OOS below expectations"
    elif iota_val >= -2.0:
        return "🔴 ALERT: OOS significantly below expectations"
    else:
        return "💀 CRITICAL: OOS severely below expectations"

def calculate_autocorrelation_adjustment(values: np.ndarray, overlap: bool) -> float:
    """Calculate autocorrelation adjustment factor for overlapping slice p-values."""
    if not overlap or len(values) < 3:
        return 1.0
    
    try:
        autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
        
        if not np.isfinite(autocorr):
            autocorr = 0.0
        autocorr = np.clip(autocorr, -0.99, 0.99)
        
        n = len(values)
        if autocorr > 0:
            effective_n = n * (1 - autocorr) / (1 + autocorr)
            adjustment_factor = np.sqrt(effective_n / n)
        else:
            adjustment_factor = 1.0
        
        adjustment_factor = np.clip(adjustment_factor, 0.1, 1.0)
        return adjustment_factor
        
    except Exception:
        return 0.7

def parametric_iota_confidence(is_values: np.ndarray, oos_value: float, n_oos: int,
                              confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Parametric confidence interval for iota using delta method approximation.
    Assumes normal distribution of the iota statistic.
    """
    if len(is_values) < 10:
        return np.nan, np.nan
    
    try:
        from scipy import stats
        
        # Estimate parameters
        is_median = np.median(is_values)
        is_std = np.std(is_values, ddof=1)
        
        if is_std < 1e-6:
            return np.nan, np.nan
        
        # Weight factor
        w = min(1.0, np.sqrt(n_oos / 252))
        
        # Point estimate
        iota_val = w * (oos_value - is_median) / is_std
        
        # Standard errors using asymptotic theory
        n_is = len(is_values)
        se_median = 1.25 * is_std / np.sqrt(n_is)  # Asymptotic SE of median
        se_std = is_std / np.sqrt(2 * (n_is - 1))  # SE of std deviation
        
        # Combined standard error using delta method
        se_iota = w * np.sqrt((se_median/is_std)**2 + ((oos_value - is_median) * se_std / is_std**2)**2)
        
        # Normal approximation for CI
        z_crit = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        ci_lower = iota_val - z_crit * se_iota
        ci_upper = iota_val + z_crit * se_iota
        
        return ci_lower, ci_upper
        
    except Exception:
        return np.nan, np.nan

def robust_iota_confidence(is_values: np.ndarray, oos_value: float, n_oos: int,
                          confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Robust confidence interval for iota using IQR-based methods.
    Suitable for skewed or fat-tailed distributions.
    """
    if len(is_values) < 10:
        return np.nan, np.nan
    
    try:
        from scipy import stats
        
        # Use IQR-based robust statistics
        is_median = np.median(is_values)
        q25, q75 = np.percentile(is_values, [25, 75])
        iqr = q75 - q25
        
        if iqr < 1e-6:
            return np.nan, np.nan
        
        # Robust "standard deviation" (IQR/1.35 for normal distributions)
        robust_std = iqr / 1.35
        
        # Weight factor
        w = min(1.0, np.sqrt(n_oos / 252))
        
        # Point estimate
        iota_val = w * (oos_value - is_median) / robust_std
        
        # Bootstrap for robust CI (since analytical formulas are complex)
        bootstrap_iotas = []
        n_bootstrap = 1000
        
        for _ in range(n_bootstrap):
            # Bootstrap the IS values
            boot_sample = np.random.choice(is_values, size=len(is_values), replace=True)
            boot_median = np.median(boot_sample)
            boot_q25, boot_q75 = np.percentile(boot_sample, [25, 75])
            boot_iqr = boot_q75 - boot_q25
            
            if boot_iqr > 1e-6:
                boot_robust_std = boot_iqr / 1.35
                boot_iota = w * (oos_value - boot_median) / boot_robust_std
                if np.isfinite(boot_iota):
                    bootstrap_iotas.append(boot_iota)
        
        if len(bootstrap_iotas) < 50:
            return np.nan, np.nan
        
        # Use percentile method for robust CI
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_iotas, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_iotas, 100 * (1 - alpha/2))
        
        return ci_lower, ci_upper
        
    except Exception:
        return np.nan, np.nan

def percentile_iota_confidence(is_values: np.ndarray, oos_value: float, n_oos: int,
                              confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Confidence interval for percentile-based iota using bootstrap.
    Suitable for complex, non-normal distributions.
    """
    if len(is_values) < 10:
        return np.nan, np.nan
    
    try:
        from scipy import stats
        
        # Weight factor
        w = min(1.0, np.sqrt(n_oos / 252))
        
        # Point estimate
        percentile = stats.percentileofscore(is_values, oos_value)
        percentile_decimal = percentile / 100.0
        
        if percentile_decimal <= 0:
            iota_val = -10.0
        elif percentile_decimal >= 1:
            iota_val = 10.0
        else:
            iota_val = w * stats.norm.ppf(percentile_decimal)
        
        # Bootstrap for percentile-based CI
        bootstrap_iotas = []
        n_bootstrap = 1000
        
        for _ in range(n_bootstrap):
            # Bootstrap the IS values
            boot_sample = np.random.choice(is_values, size=len(is_values), replace=True)
            boot_percentile = stats.percentileofscore(boot_sample, oos_value)
            boot_percentile_decimal = boot_percentile / 100.0
            
            if boot_percentile_decimal <= 0:
                boot_iota = -10.0
            elif boot_percentile_decimal >= 1:
                boot_iota = 10.0
            else:
                boot_iota = w * stats.norm.ppf(boot_percentile_decimal)
            
            if np.isfinite(boot_iota):
                bootstrap_iotas.append(boot_iota)
        
        if len(bootstrap_iotas) < 50:
            return np.nan, np.nan
        
        # Use percentile method for CI
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_iotas, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_iotas, 100 * (1 - alpha/2))
        
        return ci_lower, ci_upper
        
    except Exception:
        return np.nan, np.nan

def distribution_aware_iota_confidence(is_values: np.ndarray, oos_value: float, n_oos: int,
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Distribution-aware confidence interval that chooses the appropriate method
    based on the detected distribution characteristics.
    """
    if len(is_values) < 10:
        return np.nan, np.nan
    
    # Detect distribution characteristics
    dist_info = detect_distribution_characteristics(is_values)
    
    # Choose appropriate CI method based on distribution
    if dist_info['recommended_method'] == 'robust':
        return robust_iota_confidence(is_values, oos_value, n_oos, confidence_level)
    elif dist_info['recommended_method'] == 'percentile':
        return percentile_iota_confidence(is_values, oos_value, n_oos, confidence_level)
    else:  # standard method
        return parametric_iota_confidence(is_values, oos_value, n_oos, confidence_level)

def bootstrap_iota_confidence(is_values: np.ndarray, oos_value: float, n_oos: int, 
                             n_bootstrap: int = 1000, confidence_level: float = 0.95,
                             lower_is_better: bool = False, overlap: bool = True) -> Tuple[float, float]:
    """
    Main confidence interval function - now uses distribution-aware methods.
    Kept for backward compatibility but delegates to distribution-aware method.
    """
    if len(is_values) < 3:
        return np.nan, np.nan
    
    return distribution_aware_iota_confidence(is_values, oos_value, n_oos, confidence_level)

def parametric_p_value(is_values: np.ndarray, oos_value: float, n_oos: int,
                      lower_is_better: bool = False, overlap: bool = True,
                      null_hypothesis: str = "zero_iota") -> Tuple[float, bool, str]:
    """
    Parametric p-value using t-test or z-test depending on sample size.
    """
    try:
        from scipy import stats
        
        n_is = len(is_values)
        is_median = np.median(is_values)
        is_std = np.std(is_values, ddof=1)
        
        if is_std < 1e-6:
            return np.nan, False, "zero_variance"
        
        # Weight factor
        w = min(1.0, np.sqrt(n_oos / 252))
        
        if null_hypothesis == "zero_iota":
            # H0: iota = 0 (performance matches expectations)
            # H1: iota ≠ 0 (performance differs from expectations)
            
            # Calculate iota
            iota_observed = w * (oos_value - is_median) / is_std
            
            # Standard error of iota using delta method
            se_median = 1.25 * is_std / np.sqrt(n_is)
            se_std = is_std / np.sqrt(2 * (n_is - 1))
            se_iota = w * np.sqrt((se_median/is_std)**2 + 
                                 ((oos_value - is_median) * se_std / is_std**2)**2)
            
            # Test statistic
            if se_iota < 1e-6:
                return np.nan, False, "zero_se"
            
            t_stat = iota_observed / se_iota
            
            # Use t-distribution for small samples, normal for large samples
            if n_is < 30:
                p_value_raw = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_is-1))
            else:
                p_value_raw = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        elif null_hypothesis == "median_performance":
            # H0: OOS_value = IS_median
            # H1: OOS_value ≠ IS_median
            
            # One-sample t-test equivalent
            # Standard error of the difference
            se_diff = is_std / np.sqrt(n_is) * 1.25  # 1.25 factor for median vs mean
            
            t_stat = (oos_value - is_median) / se_diff
            
            if n_is < 30:
                p_value_raw = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_is-1))
            else:
                p_value_raw = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        else:
            return np.nan, False, "invalid_hypothesis"
        
        # Adjust for autocorrelation if overlapping
        if overlap:
            autocorr_adjustment = calculate_autocorrelation_adjustment(is_values, overlap)
            p_value_adjusted = min(1.0, p_value_raw / autocorr_adjustment)
        else:
            p_value_adjusted = p_value_raw
        
        significant = p_value_adjusted < 0.05
        
        return p_value_adjusted, significant, "parametric"
        
    except Exception as e:
        return np.nan, False, f"error_{str(e)[:20]}"

def robust_p_value(is_values: np.ndarray, oos_value: float, n_oos: int,
                  lower_is_better: bool = False, overlap: bool = True,
                  null_hypothesis: str = "zero_iota") -> Tuple[float, bool, str]:
    """
    Robust p-value using bootstrap or sign-based tests.
    """
    try:
        from scipy import stats
        
        # For robust method, use bootstrap-based p-value
        n_bootstrap = 2000
        
        # Calculate observed test statistic
        is_median = np.median(is_values)
        q25, q75 = np.percentile(is_values, [25, 75])
        iqr = q75 - q25
        
        if iqr < 1e-6:
            return np.nan, False, "zero_iqr"
        
        robust_std = iqr / 1.35
        w = min(1.0, np.sqrt(n_oos / 252))
        
        if null_hypothesis == "zero_iota":
            # Observed iota
            iota_observed = w * (oos_value - is_median) / robust_std
            
            # Bootstrap under null hypothesis (iota = 0)
            # This means OOS_value should equal IS_median
            bootstrap_stats = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                boot_sample = np.random.choice(is_values, size=len(is_values), replace=True)
                boot_median = np.median(boot_sample)
                boot_q25, boot_q75 = np.percentile(boot_sample, [25, 75])
                boot_iqr = boot_q75 - boot_q25
                
                if boot_iqr > 1e-6:
                    boot_robust_std = boot_iqr / 1.35
                    # Under null: test against bootstrap median (represents null distribution)
                    boot_iota = w * (boot_median - boot_median) / boot_robust_std  # This is 0
                    bootstrap_stats.append(0)  # Under null, iota should be 0
            
            # P-value: proportion of bootstrap statistics as extreme as observed
            bootstrap_stats = np.array(bootstrap_stats)
            p_value_raw = np.mean(np.abs(bootstrap_stats) >= np.abs(iota_observed))
            
        elif null_hypothesis == "median_performance":
            # Use sign test or Wilcoxon signed-rank test
            # H0: OOS performance equals typical IS performance
            
            # Compare OOS value to each IS value
            differences = is_values - oos_value
            
            # Wilcoxon signed-rank test (robust alternative to t-test)
            if len(differences) >= 6:
                _, p_value_raw = stats.wilcoxon(differences, alternative='two-sided')
            else:
                # Sign test for very small samples
                n_positive = np.sum(differences > 0)
                n_total = len(differences)
                p_value_raw = 2 * stats.binom.cdf(min(n_positive, n_total - n_positive), 
                                                 n_total, 0.5)
        
        else:
            return np.nan, False, "invalid_hypothesis"
        
        # Adjust for autocorrelation if overlapping
        if overlap:
            autocorr_adjustment = calculate_autocorrelation_adjustment(is_values, overlap)
            p_value_adjusted = min(1.0, p_value_raw / autocorr_adjustment)
        else:
            p_value_adjusted = p_value_raw
        
        significant = p_value_adjusted < 0.05
        
        return p_value_adjusted, significant, "robust"
        
    except Exception as e:
        return np.nan, False, f"error_{str(e)[:20]}"

def percentile_p_value(is_values: np.ndarray, oos_value: float, n_oos: int,
                      lower_is_better: bool = False, overlap: bool = True,
                      null_hypothesis: str = "zero_iota") -> Tuple[float, bool, str]:
    """
    Percentile-based p-value using rank-based methods.
    """
    try:
        from scipy import stats
        
        if null_hypothesis == "zero_iota":
            # For percentile method, iota=0 means OOS is at 50th percentile
            percentile = stats.percentileofscore(is_values, oos_value)
            
            # Two-tailed test: how far is this percentile from 50?
            deviation_from_median = abs(percentile - 50)
            
            # Convert to p-value: probability of being this far or farther from median
            # Under null, percentiles should be uniformly distributed
            p_value_raw = 2 * min(percentile / 100, (100 - percentile) / 100)
            
        elif null_hypothesis == "median_performance":
            # Direct rank-based test
            # H0: OOS value comes from same distribution as IS values
            
            # Use Mann-Whitney U test (comparing OOS value to IS distribution)
            # Treat OOS as a single observation
            oos_array = np.array([oos_value])
            
            try:
                _, p_value_raw = stats.mannwhitneyu(is_values, oos_array, 
                                                  alternative='two-sided')
            except ValueError:
                # Fallback to simple percentile test
                percentile = stats.percentileofscore(is_values, oos_value)
                p_value_raw = 2 * min(percentile / 100, (100 - percentile) / 100)
        
        else:
            return np.nan, False, "invalid_hypothesis"
        
        # Adjust for autocorrelation if overlapping
        if overlap:
            autocorr_adjustment = calculate_autocorrelation_adjustment(is_values, overlap)
            p_value_adjusted = min(1.0, p_value_raw / autocorr_adjustment)
        else:
            p_value_adjusted = p_value_raw
        
        significant = p_value_adjusted < 0.05
        
        return p_value_adjusted, significant, "percentile"
        
    except Exception as e:
        return np.nan, False, f"error_{str(e)[:20]}"

def distribution_aware_p_value(is_values: np.ndarray, oos_value: float, n_oos: int,
                             lower_is_better: bool = False, overlap: bool = True,
                             null_hypothesis: str = "zero_iota") -> Tuple[float, bool, str]:
    """
    Distribution-aware p-value calculation that matches the iota calculation method.
    
    Parameters:
    - null_hypothesis: "zero_iota" (iota=0) or "median_performance" (OOS=IS_median)
    """
    if len(is_values) < 6:
        return np.nan, False, "insufficient_data"
    
    # Detect distribution characteristics
    dist_info = detect_distribution_characteristics(is_values)
    method = dist_info['recommended_method']
    
    # Calculate p-value using appropriate method
    if method == 'robust':
        return robust_p_value(is_values, oos_value, n_oos, lower_is_better, 
                             overlap, null_hypothesis)
    elif method == 'percentile':
        return percentile_p_value(is_values, oos_value, n_oos, lower_is_better, 
                                 overlap, null_hypothesis)
    else:  # standard method
        return parametric_p_value(is_values, oos_value, n_oos, lower_is_better, 
                                 overlap, null_hypothesis)

def compute_iota_with_stats(is_values: np.ndarray, oos_value: float, n_oos: int, 
                           metric_name: str = "metric", lower_is_better: bool = False,
                           overlap: bool = True) -> Dict[str, Any]:
    """Enhanced iota computation with statistical tests."""
    if len(is_values) == 0:
        return {
            'iota': np.nan,
            'persistence_rating': 100,
            'confidence_interval': (np.nan, np.nan),
            'median_is': np.nan,
            'iqr_is': (np.nan, np.nan)
        }
    
    median_is = np.median(is_values)
    q25_is, q75_is = np.percentile(is_values, [25, 75])
    
    iota = compute_iota(median_is, oos_value, n_oos, lower_is_better=lower_is_better, is_values=is_values, use_distribution_aware=True)
    persistence_rating = iota_to_persistence_rating(iota)
    
    # Get distribution characteristics first
    dist_info = detect_distribution_characteristics(is_values)
    
    # Use distribution-aware confidence interval
    ci_lower, ci_upper = distribution_aware_iota_confidence(is_values, oos_value, n_oos)
    

    
    return {
        'iota': iota,
        'persistence_rating': persistence_rating,
        'confidence_interval': (ci_lower, ci_upper),
        'median_is': median_is,
        'iqr_is': (q25_is, q75_is),
        'distribution_method': dist_info['recommended_method'],
        'distribution_confidence': dist_info['confidence'],
        'is_skewed': dist_info['is_skewed'],
        'has_fat_tails': dist_info['has_fat_tails'],
        'skewness': dist_info['skewness'],
        'kurtosis': dist_info['kurtosis'],
        'ci_method': dist_info['recommended_method']  # Track which CI method was used
    }

def format_sortino_output(sortino_val: float) -> str:
    """Special formatting for Sortino ratio including infinite values."""
    if np.isinf(sortino_val):
        return "∞ (no downside)"
    elif np.isnan(sortino_val):
        return "NaN"
    else:
        return f"{sortino_val:.3f}"

def qs_beta(returns: pd.Series, benchmark_returns: pd.Series = None) -> float:
    """Calculate beta using quantstats if available, fallback to internal calculation."""
    if QUANTSTATS_AVAILABLE and benchmark_returns is not None:
        try:
            # Convert percentage returns to decimal if needed
            if returns.max() > 1.0:  # Likely percentage format
                returns_decimal = returns / 100.0
            else:
                returns_decimal = returns
            
            if benchmark_returns.max() > 1.0:  # Likely percentage format
                benchmark_decimal = benchmark_returns / 100.0
            else:
                benchmark_decimal = benchmark_returns
            
            # Ensure proper DatetimeIndex for quantstats
            returns_decimal = ensure_datetime_index(returns_decimal)
            benchmark_decimal = ensure_datetime_index(benchmark_decimal)
            
            # Use quantstats beta calculation
            beta = qs.stats.beta(returns_decimal, benchmark_decimal)
            return float(beta) if np.isfinite(beta) else 0.0
        except Exception as e:
            st.warning(f"quantstats beta calculation failed: {str(e)[:50]}, using internal calculation")
            return calculate_beta_internal(returns, benchmark_returns)
    else:
        return calculate_beta_internal(returns, benchmark_returns)

def qs_alpha(returns: pd.Series, benchmark_returns: pd.Series = None) -> float:
    """Calculate alpha using quantstats if available, fallback to internal calculation."""
    if QUANTSTATS_AVAILABLE and benchmark_returns is not None:
        try:
            # Convert percentage returns to decimal if needed
            if returns.max() > 1.0:  # Likely percentage format
                returns_decimal = returns / 100.0
            else:
                returns_decimal = returns
            
            if benchmark_returns.max() > 1.0:  # Likely percentage format
                benchmark_decimal = benchmark_returns / 100.0
            else:
                benchmark_decimal = benchmark_returns
            
            # Ensure proper DatetimeIndex for quantstats
            returns_decimal = ensure_datetime_index(returns_decimal)
            benchmark_decimal = ensure_datetime_index(benchmark_decimal)
            
            # Use quantstats alpha calculation
            alpha = qs.stats.alpha(returns_decimal, benchmark_decimal)
            return float(alpha) if np.isfinite(alpha) else 0.0
        except Exception as e:
            st.warning(f"quantstats alpha calculation failed: {str(e)[:50]}, using internal calculation")
            return calculate_alpha_internal(returns, benchmark_returns)
    else:
        return calculate_alpha_internal(returns, benchmark_returns)

def qs_information_ratio(returns: pd.Series, benchmark_returns: pd.Series = None) -> float:
    """Calculate information ratio using quantstats if available, fallback to internal calculation."""
    if QUANTSTATS_AVAILABLE and benchmark_returns is not None:
        try:
            # Convert percentage returns to decimal if needed
            if returns.max() > 1.0:  # Likely percentage format
                returns_decimal = returns / 100.0
            else:
                returns_decimal = returns
            
            if benchmark_returns.max() > 1.0:  # Likely percentage format
                benchmark_decimal = benchmark_returns / 100.0
            else:
                benchmark_decimal = benchmark_returns
            
            # Ensure proper DatetimeIndex for quantstats
            returns_decimal = ensure_datetime_index(returns_decimal)
            benchmark_decimal = ensure_datetime_index(benchmark_decimal)
            
            # Use quantstats information ratio calculation
            ir = qs.stats.information_ratio(returns_decimal, benchmark_decimal)
            return float(ir) if np.isfinite(ir) else 0.0
        except Exception as e:
            st.warning(f"quantstats information ratio calculation failed: {str(e)[:50]}, using internal calculation")
            return calculate_information_ratio_internal(returns, benchmark_returns)
    else:
        return calculate_information_ratio_internal(returns, benchmark_returns)

def calculate_beta_internal(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate beta using internal calculation."""
    if returns.empty or benchmark_returns.empty:
        return 0.0
    
    # Align the series
    aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned_data) < 2:
        return 0.0
    
    strategy_returns = aligned_data.iloc[:, 0]
    benchmark_returns = aligned_data.iloc[:, 1]
    
    # Convert to decimal if needed
    if strategy_returns.max() > 1.0:
        strategy_returns = strategy_returns / 100.0
    if benchmark_returns.max() > 1.0:
        benchmark_returns = benchmark_returns / 100.0
    
    # Calculate covariance and variance
    covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    
    if benchmark_variance == 0:
        return 0.0
    
    return covariance / benchmark_variance

def calculate_alpha_internal(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate alpha using internal calculation."""
    if returns.empty or benchmark_returns.empty:
        return 0.0
    
    # Get beta
    beta = calculate_beta_internal(returns, benchmark_returns)
    
    # Get average returns
    strategy_avg = returns.mean() / 100.0 if returns.max() > 1.0 else returns.mean()
    benchmark_avg = benchmark_returns.mean() / 100.0 if benchmark_returns.max() > 1.0 else benchmark_returns.mean()
    
    # Calculate alpha (annualized)
    alpha = (strategy_avg - beta * benchmark_avg) * 252
    return alpha

def calculate_information_ratio_internal(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate information ratio using internal calculation."""
    if returns.empty or benchmark_returns.empty:
        return 0.0
    
    # Align the series
    aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned_data) < 2:
        return 0.0
    
    strategy_returns = aligned_data.iloc[:, 0]
    benchmark_returns = aligned_data.iloc[:, 1]
    
    # Convert to decimal if needed
    if strategy_returns.max() > 1.0:
        strategy_returns = strategy_returns / 100.0
    if benchmark_returns.max() > 1.0:
        benchmark_returns = benchmark_returns / 100.0
    
    # Calculate excess returns
    excess_returns = strategy_returns - benchmark_returns
    
    # Calculate information ratio
    if excess_returns.std() == 0:
        return 0.0
    
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

def get_comprehensive_stats(returns: pd.Series) -> Dict[str, float]:
    """Get comprehensive statistics using quantstats when available."""
    stats = {}
    
    if QUANTSTATS_AVAILABLE:
        try:
            # Convert percentage returns to decimal if needed
            if returns.max() > 1.0:  # Likely percentage format
                returns_decimal = returns / 100.0
            else:
                returns_decimal = returns
            
            # Ensure proper DatetimeIndex for quantstats
            returns_decimal = ensure_datetime_index(returns_decimal)
            
            # Get comprehensive stats from quantstats
            stats = {
                'sharpe': float(qs.stats.sharpe(returns_decimal)) if np.isfinite(qs.stats.sharpe(returns_decimal)) else 0.0,
                'sortino': float(qs.stats.sortino(returns_decimal)) if np.isfinite(qs.stats.sortino(returns_decimal)) else 0.0,
                'calmar': float(qs.stats.calmar(returns_decimal)) if np.isfinite(qs.stats.calmar(returns_decimal)) else 0.0,
                'volatility': float(qs.stats.volatility(returns_decimal)) if np.isfinite(qs.stats.volatility(returns_decimal)) else 0.0,
                'max_drawdown': float(qs.stats.max_drawdown(returns_decimal)) if np.isfinite(qs.stats.max_drawdown(returns_decimal)) else 0.0,
                'cagr': float(qs.stats.cagr(returns_decimal)) if np.isfinite(qs.stats.cagr(returns_decimal)) else 0.0,
                'comp': float(qs.stats.comp(returns_decimal)) if np.isfinite(qs.stats.comp(returns_decimal)) else 0.0,
                'win_rate': float(qs.stats.win_rate(returns_decimal)) if np.isfinite(qs.stats.win_rate(returns_decimal)) else 0.0,
                'avg_win': float(qs.stats.avg_win(returns_decimal)) if np.isfinite(qs.stats.avg_win(returns_decimal)) else 0.0,
                'avg_loss': float(qs.stats.avg_loss(returns_decimal)) if np.isfinite(qs.stats.avg_loss(returns_decimal)) else 0.0,
                'best': float(qs.stats.best(returns_decimal)) if np.isfinite(qs.stats.best(returns_decimal)) else 0.0,
                'worst': float(qs.stats.worst(returns_decimal)) if np.isfinite(qs.stats.worst(returns_decimal)) else 0.0,
            }
        except Exception as e:
            st.warning(f"quantstats comprehensive stats failed: {str(e)[:50]}, using basic stats")
            stats = get_basic_stats(returns)
    else:
        stats = get_basic_stats(returns)
    
    return stats

def get_basic_stats(returns: pd.Series) -> Dict[str, float]:
    """Get basic statistics using internal calculations."""
    if returns.empty:
        return {}
    
    # Convert to decimal if needed
    if returns.max() > 1.0:  # Likely percentage format
        returns_decimal = returns / 100.0
    else:
        returns_decimal = returns
    
    stats = {
        'sharpe': sharpe_ratio_internal(returns),
        'sortino': sortino_ratio_internal(returns),
        'volatility': returns_decimal.std() * np.sqrt(252),
        'cagr': window_cagr_internal(returns),
        'comp': cumulative_return_internal(returns),
        'max_drawdown': calculate_max_drawdown_internal(returns),
        'calmar': calculate_calmar_ratio_internal(returns),
    }
    
    return stats

# ===== ROLLING WINDOW ANALYSIS FUNCTIONS =====

def smooth_iotas(iotas, window=3):
    """Apply rolling average smoothing to iota series."""
    if len(iotas) < window:
        return iotas
    
    smoothed = []
    for i in range(len(iotas)):
        start_idx = max(0, i - window + 1)
        end_idx = i + 1
        smoothed.append(np.mean(iotas[start_idx:end_idx]))
    return smoothed

def fetch_oos_date_from_symphony(symph_id: str) -> Optional[date]:
    """Fetch OOS date from Composer symphony using multiple API methods."""
    if not symph_id or len(symph_id.strip()) == 0:
        return None
        
    # Clean the symphony ID (remove URL parts if user pasted full URL)
    symph_id = symph_id.strip()
    if symph_id.startswith("https://app.composer.trade/symphony/"):
        symph_id = symph_id.split("/")[-1]
    
    # Method 1: Try Composer Web API
    try:
        composer_url = f"https://app.composer.trade/api/symphony/{symph_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://app.composer.trade/'
        }
        
        response = requests.get(composer_url, headers=headers, timeout=10)
        if response.status_code == 200:
            try:
                data = response.json()
                
                # Look for OOS date in various possible fields
                possible_fields = [
                    'oos_start_date', 'live_start_date', 'trading_start_date',
                    'last_semantic_update_at', 'created_at', 'updated_at',
                    'start_date', 'live_date', 'oos_date'
                ]
                
                for field in possible_fields:
                    if field in data:
                        date_str = data[field]
                        if isinstance(date_str, str):
                            # Handle different date formats
                            if 'T' in date_str:  # ISO format
                                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                                return dt.date()
                            else:  # Try other formats
                                try:
                                    dt = datetime.strptime(date_str, '%Y-%m-%d')
                                    return dt.date()
                                except:
                                    continue
                
                # Debug: Show available fields
                available_fields = list(data.keys())
                st.info(f"Composer API fields: {available_fields}")
                
            except ValueError:
                # Empty response or invalid JSON - this is expected for some symphonies
                pass
            
    except Exception as e:
        # Only show error if it's not an empty response
        if "Expecting value" not in str(e):
            st.info(f"Composer API method failed: {str(e)[:50]}")
    
    # Method 2: Try Firestore API (original method)
    try:
        firestore_url = f"https://firestore.googleapis.com/v1/projects/leverheads-278521/databases/(default)/documents/symphony/{symph_id}"
        
        response = requests.get(firestore_url, timeout=10)
        response.raise_for_status()
        
        if not response.text.strip():
            # Empty response - this is expected for some symphonies
            pass
        else:
            data = response.json()
            
            if 'fields' not in data:
                # Symphony not found - this is expected for some symphonies
                pass
            else:
                fields = data['fields']
                
                # Try multiple Firestore fields
                firestore_fields = [
                    'last_semantic_update_at', 'created_at', 'updated_at',
                    'last_updated', 'oos_start_date', 'live_start_date',
                    'trading_start_date', 'start_date', 'live_date'
                ]
                
                for field in firestore_fields:
                    if field in fields:
                        try:
                            timestamp_str = fields[field]['timestampValue']
                            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            return dt.date()
                        except:
                            continue
                
                # Debug: Show available Firestore fields
                available_fields = list(fields.keys())
                st.info(f"Firestore fields: {available_fields}")
        
    except Exception as e:
        # Only show error if it's not an expected failure
        if "404" not in str(e) and "not found" not in str(e).lower():
            st.info(f"Firestore API method failed: {str(e)[:50]}")
    
    # Method 3: Extract OOS Start Date from Composer page (SPA-aware)
    try:
        # Get the symphony page HTML
        symphony_url = f"https://app.composer.trade/symphony/{symph_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(symphony_url, headers=headers, timeout=15)
        if response.status_code == 200:
            html_content = response.text
            
            # Look for "OOS Start Date" text specifically
            import re
            
            # Pattern 1: Look for "OOS Start Date: [date]" (exact format from image)
            oos_pattern = r'OOS Start Date:\s*([A-Za-z]{3}\s+\d{1,2},\s+\d{4})'
            matches = re.findall(oos_pattern, html_content)
            if matches:
                try:
                    date_str = matches[0]
                    dt = datetime.strptime(date_str, '%b %d, %Y')
                    return dt.date()
                except:
                    pass
            
            # Pattern 2: Look for "OOS Start Date: [date]" with different format
            oos_pattern2 = r'OOS Start Date:\s*(\d{1,2}/\d{1,2}/\d{4})'
            matches = re.findall(oos_pattern2, html_content)
            if matches:
                try:
                    date_str = matches[0]
                    dt = datetime.strptime(date_str, '%m/%d/%Y')
                    return dt.date()
                except:
                    pass
            
            # Pattern 3: Look for "OOS Start Date: [date]" with ISO format
            oos_pattern3 = r'OOS Start Date:\s*(\d{4}-\d{2}-\d{2})'
            matches = re.findall(oos_pattern3, html_content)
            if matches:
                try:
                    date_str = matches[0]
                    dt = datetime.strptime(date_str, '%Y-%m-%d')
                    return dt.date()
                except:
                    pass
            
            # Pattern 4: Look for JSON data in script tags (SPA data)
            script_patterns = [
                r'"oos_start_date":"([^"]+)"',
                r'"live_start_date":"([^"]+)"',
                r'"oosDate":"([^"]+)"',
                r'"liveDate":"([^"]+)"',
                r'"startDate":"([^"]+)"',
                r'"oos_start":"([^"]+)"',
                r'"live_start":"([^"]+)"',
                r'"oosStartDate":"([^"]+)"',
                r'"liveStartDate":"([^"]+)"'
            ]
            
            for pattern in script_patterns:
                matches = re.findall(pattern, html_content)
                if matches:
                    try:
                        date_str = matches[0]
                        if 'T' in date_str:  # ISO format
                            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            return dt.date()
                        else:  # Try other formats
                            try:
                                dt = datetime.strptime(date_str, '%Y-%m-%d')
                                return dt.date()
                            except:
                                try:
                                    dt = datetime.strptime(date_str, '%m/%d/%Y')
                                    return dt.date()
                                except:
                                    continue
                    except:
                        continue
            
            # Pattern 5: Look for React/JavaScript state data
            state_patterns = [
                r'oosStartDate["\']?\s*:\s*["\']([^"\']+)["\']',
                r'liveStartDate["\']?\s*:\s*["\']([^"\']+)["\']',
                r'oos_start_date["\']?\s*:\s*["\']([^"\']+)["\']',
                r'live_start_date["\']?\s*:\s*["\']([^"\']+)["\']'
            ]
            
            for pattern in state_patterns:
                matches = re.findall(pattern, html_content)
                if matches:
                    try:
                        date_str = matches[0]
                        if 'T' in date_str:  # ISO format
                            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            return dt.date()
                        else:  # Try other formats
                            try:
                                dt = datetime.strptime(date_str, '%Y-%m-%d')
                                return dt.date()
                            except:
                                try:
                                    dt = datetime.strptime(date_str, '%m/%d/%Y')
                                    return dt.date()
                                except:
                                    continue
                    except:
                        continue
            
            # Debug: Show what we found in the HTML
            if 'OOS Start Date' in html_content:
                # Find the line containing OOS Start Date
                lines = html_content.split('\n')
                for line in lines:
                    if 'OOS Start Date' in line:
                        st.info(f"Found OOS Start Date line: {line.strip()[:200]}")
                        break
            else:
                st.info("OOS Start Date text not found in HTML - this is expected for SPAs")
                
                # Show what we did find
                if 'symphony' in html_content.lower():
                    st.info("Symphony page loaded but data may be loaded dynamically")
                else:
                    st.info("Page may not be loading correctly")
                        
    except Exception as e:
        st.info(f"Web scraping method failed: {str(e)[:50]}")
    
    # Method 4: Try to wait for dynamic content to load
    try:
        # Try with a longer timeout and different approach
        symphony_url = f"https://app.composer.trade/symphony/{symph_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        response = requests.get(symphony_url, headers=headers, timeout=20)
        if response.status_code == 200:
            html_content = response.text
            
            # Look for any mention of OOS Start Date
            if 'OOS Start Date' in html_content:
                # Try to find the actual date near this text
                lines = html_content.split('\n')
                for i, line in enumerate(lines):
                    if 'OOS Start Date' in line:
                        # Look at this line and the next few lines for a date
                        search_lines = lines[i:i+5]
                        for search_line in search_lines:
                            # Look for date patterns
                            date_patterns = [
                                r'([A-Za-z]{3}\s+\d{1,2},\s+\d{4})',  # Jul 20, 2022
                                r'(\d{1,2}/\d{1,2}/\d{4})',  # 07/20/2022
                                r'(\d{4}-\d{2}-\d{2})',  # 2022-07-20
                            ]
                            
                            for pattern in date_patterns:
                                matches = re.findall(pattern, search_line)
                                if matches:
                                    try:
                                        date_str = matches[0]
                                        if ' ' in date_str:  # Jul 20, 2022 format
                                            dt = datetime.strptime(date_str, '%b %d, %Y')
                                            return dt.date()
                                        elif '/' in date_str:  # 07/20/2022 format
                                            dt = datetime.strptime(date_str, '%m/%d/%Y')
                                            return dt.date()
                                        elif '-' in date_str:  # 2022-07-20 format
                                            dt = datetime.strptime(date_str, '%Y-%m-%d')
                                            return dt.date()
                                    except:
                                        continue
                        
                        st.info(f"Found 'OOS Start Date' but couldn't parse the date from: {line.strip()[:200]}")
                        break
            else:
                st.info("'OOS Start Date' text not found in the page HTML")
                        
    except Exception as e:
        st.info(f"Enhanced web scraping method failed: {str(e)[:50]}")
    
    # Method 4: Try Composer's GraphQL API (if available)
    try:
        graphql_url = "https://app.composer.trade/graphql"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # GraphQL query to get symphony details
        query = {
            "query": """
            query GetSymphony($id: ID!) {
                symphony(id: $id) {
                    id
                    oosStartDate
                    liveStartDate
                    createdAt
                    updatedAt
                }
            }
            """,
            "variables": {"id": symph_id}
        }
        
        response = requests.post(graphql_url, headers=headers, json=query, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'symphony' in data['data']:
                symphony_data = data['data']['symphony']
                
                # Try different field names
                date_fields = ['oosStartDate', 'liveStartDate', 'createdAt', 'updatedAt']
                for field in date_fields:
                    if field in symphony_data and symphony_data[field]:
                        try:
                            date_str = symphony_data[field]
                            if 'T' in date_str:  # ISO format
                                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                                return dt.date()
                            else:
                                dt = datetime.strptime(date_str, '%Y-%m-%d')
                                return dt.date()
                        except:
                            continue
                            
    except Exception as e:
        st.info(f"GraphQL API method failed: {str(e)[:50]}")
    
    st.warning(f"All methods failed for symphony {symph_id}. Please enter OOS date manually.")
    return None

def rolling_oos_analysis(daily_ret: pd.Series, oos_start_dt: date, 
                        is_ret: pd.Series, n_slices: int = 100, overlap: bool = True,
                        window_size: int = None, step_size: int = None, 
                        min_windows: int = 6, use_quantstats: bool = False) -> Dict[str, Any]:
    """Simplified rolling analysis for overfitting detection."""
    oos_data = daily_ret[daily_ret.index >= oos_start_dt]
    total_oos_days = len(oos_data)
    
    if total_oos_days > 1500:
        oos_data = oos_data.iloc[-1000:]
        total_oos_days = len(oos_data)
    
    if total_oos_days < 90:
        return {
            'sufficient_data': False,
            'n_windows': 0,
            'overfitting_risk': 'INSUFFICIENT_DATA',
            'iota_trend_slope': np.nan,
            'degradation_score': np.nan
        }
    
    # Adaptive window sizing
    if window_size is None:
        if total_oos_days >= 504:
            window_size = 126
        elif total_oos_days >= 252:
            window_size = 84
        elif total_oos_days >= 189:
            window_size = 63
        else:
            window_size = max(21, total_oos_days // 4)
    
    if step_size is None:
        step_size = max(5, window_size // 8)
    
    if total_oos_days < window_size + step_size:
        return {
            'sufficient_data': False,
            'n_windows': 0,
            'overfitting_risk': 'INSUFFICIENT_DATA',
            'iota_trend_slope': np.nan,
            'degradation_score': np.nan
        }
    
    max_possible_windows = (total_oos_days - window_size) // step_size + 1
    max_windows = min(60, max_possible_windows)
    
    # Create IS slices
    is_slices = build_slices(is_ret, window_size, n_slices, overlap)
    if not is_slices:
        return {
            'sufficient_data': False,
            'n_windows': 0,
            'overfitting_risk': 'INSUFFICIENT_IS_DATA',
            'iota_trend_slope': np.nan,
            'degradation_score': np.nan
        }
    
    # Pre-compute IS metrics
    is_metrics = {
        'sh': [sharpe_ratio(s, use_quantstats) for s in is_slices], 
        'cr': [cumulative_return(s, use_quantstats) for s in is_slices],
        'so': [sortino_ratio(s, use_quantstats) for s in is_slices]
    }
    
    # Create rolling windows
    windows = []
    start_idx = 0
    window_count = 0
    
    while start_idx + window_size <= len(oos_data) and window_count < max_windows:
        window_data = oos_data.iloc[start_idx:start_idx + window_size]
        if len(window_data) == window_size:
            window_num = len(windows) + 1
            
            window_sh = sharpe_ratio(window_data, use_quantstats)
            window_cr = window_cagr(window_data, use_quantstats)  # Use annualized return for consistency
            window_so = sortino_ratio(window_data, use_quantstats)
            
            window_iotas = {}
            for metric in ['sh', 'cr', 'so']:
                is_values = np.array(is_metrics[metric])
                oos_value = {'sh': window_sh, 'cr': window_cr, 'so': window_so}[metric]
                lower_is_better = False
                
                if len(is_values) > 0 and np.isfinite(oos_value):
                    iota = compute_iota(np.median(is_values), oos_value, window_size, 
                                      lower_is_better=lower_is_better, is_values=is_values, use_distribution_aware=False)
                    window_iotas[metric] = iota
                else:
                    window_iotas[metric] = np.nan
            
            windows.append({
                'start_date': window_data.index[0],
                'end_date': window_data.index[-1],
                'window_num': window_num,
                'returns': window_data,
                'metrics': {
                    'sh': window_sh, 
                    'cr': window_cr,
                    'so': window_so
                },
                'iotas': window_iotas
            })
            window_count += 1
        start_idx += step_size
    
    if len(windows) < min_windows:
        return {
            'sufficient_data': False,
            'n_windows': len(windows),
            'overfitting_risk': 'INSUFFICIENT_WINDOWS',
            'iota_trend_slope': np.nan,
            'degradation_score': np.nan
        }
    
    # Extract iota series for analysis
    window_nums = np.array([w['window_num'] for w in windows])
    
    metric_iotas = {}
    for metric in ['sh', 'cr', 'so']:
        metric_iotas[metric] = np.array([w['iotas'][metric] for w in windows if np.isfinite(w['iotas'][metric])])
    
    # Calculate trend slopes
    metric_slopes = {}
    for metric in ['sh', 'cr', 'so']:
        if len(metric_iotas[metric]) >= 3:
            try:
                slope, _, _, _, _ = stats.linregress(window_nums[:len(metric_iotas[metric])], metric_iotas[metric])
                metric_slopes[f'{metric}_slope'] = slope
            except:
                metric_slopes[f'{metric}_slope'] = np.nan
        else:
            metric_slopes[f'{metric}_slope'] = np.nan

    valid_slopes = [slope for slope in metric_slopes.values() if np.isfinite(slope)]
    avg_trend_slope = np.mean(valid_slopes) if valid_slopes else np.nan
    
    # Calculate degradation score
    degradation_score = 0

    all_iotas = []
    for metric in ['sh', 'cr', 'so']:
        if len(metric_iotas[metric]) > 0:
            all_iotas.extend(metric_iotas[metric])

    if len(all_iotas) > 0:
        all_iotas = np.array(all_iotas)
        
        avg_iota = np.mean(all_iotas)
        if avg_iota < -1.5:
            degradation_score += 4
        elif avg_iota < -1.0:
            degradation_score += 3
        elif avg_iota < -0.5:
            degradation_score += 2
        elif avg_iota < -0.2:
            degradation_score += 1
        
        negative_proportion = np.mean(all_iotas < 0)
        if negative_proportion > 0.9:
            degradation_score += 3
        elif negative_proportion > 0.75:
            degradation_score += 2
        elif negative_proportion > 0.6:
            degradation_score += 1
        
        severely_negative = np.mean(all_iotas < -1.0)
        if severely_negative > 0.5:
            degradation_score += 3
        elif severely_negative > 0.3:
            degradation_score += 2
        elif severely_negative > 0.1:
            degradation_score += 1

    # Calculate proportion of time spent under -0.5 iota for each metric
    for metric in ['sh', 'cr', 'so']:
        if len(metric_iotas[metric]) > 0:
            under_threshold_proportion = np.mean(np.array(metric_iotas[metric]) < -0.5)
            if under_threshold_proportion > 0.8:
                degradation_score += 4
            elif under_threshold_proportion > 0.6:
                degradation_score += 3
            elif under_threshold_proportion > 0.4:
                degradation_score += 2
            elif under_threshold_proportion > 0.2:
                degradation_score += 1
    
    # Simple risk assessment based on degradation score
    if degradation_score >= 12:
        risk_level = "CRITICAL"
    elif degradation_score >= 8:
        risk_level = "HIGH"
    elif degradation_score >= 5:
        risk_level = "MODERATE"
    elif degradation_score >= 2:
        risk_level = "LOW"
    else:
        risk_level = "MINIMAL"
        
    return {
        'sufficient_data': True,
        'n_windows': len(windows),
        'windows': windows,
        'iota_trend_slope': avg_trend_slope,
        'metric_slopes': metric_slopes,
        'degradation_score': degradation_score,
        'overfitting_risk': risk_level,
        'window_size_days': window_size,
        'step_size_days': step_size,
        'total_oos_days': total_oos_days,
        'is_slices_used': len(is_slices),
        'metric_iotas': metric_iotas
    }



def interpret_overfitting_risk(rolling_results: Dict[str, Any]) -> str:
    """Generate interpretation of rolling analysis results."""
    if not rolling_results.get('sufficient_data', False):
        return "Insufficient data for rolling analysis (need longer OOS period)"
    
    n_windows = rolling_results['n_windows']
    
    interpretation = f"**Rolling Analysis Summary** ({n_windows} windows)\n\n"
    interpretation += "Rolling analysis shows how strategy performance varies over time.\n"
    interpretation += "Check the charts above for visual patterns and trends.\n\n"
    
    return interpretation

def create_distribution_histograms(ar_is_values, sh_is_values, cr_is_values, so_is_values, ar_oos, sh_oos, cr_oos, so_oos, symphony_name: str, n_oos_days: int = None):
    """Create histogram plots showing in-sample distributions with OOS values marked."""
    
    # Import make_subplots
    from plotly.subplots import make_subplots
    
    # Define metrics and their data with proper scaling
    metrics_data = [
        ("Annualized Return", ar_is_values, ar_oos, lambda x: f"{x*100:+.2f}%", "Annualized Return (%)"),
        ("Sharpe Ratio", sh_is_values, sh_oos, lambda x: f"{x:+.3f}", "Sharpe Ratio"),
        ("Cumulative Return", cr_is_values, cr_oos, lambda x: f"{x*100:+.2f}%", "Cumulative Return (%)"),
        ("Sortino Ratio", so_is_values, so_oos, format_sortino_output, "Sortino Ratio")
    ]
    
    # Colors for different metrics
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Create subplots - 2x2 grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[metric[0] for metric in metrics_data],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Add histograms for each metric
    for i, (metric_name, is_values, oos_val, formatter, axis_title) in enumerate(metrics_data):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        # Scale values for percentage metrics (Annualized Return and Cumulative Return)
        if metric_name in ["Annualized Return", "Cumulative Return"]:
            scaled_is_values = is_values * 100
            scaled_oos_val = oos_val * 100
        else:
            scaled_is_values = is_values
            scaled_oos_val = oos_val
        
        # Create histogram for in-sample values
        fig.add_trace(
            go.Histogram(
                x=scaled_is_values,
                name=metric_name,
                nbinsx=40,
                opacity=0.7,
                marker_color=colors[i],
                showlegend=False,
                hovertemplate=f"<b>{metric_name}</b><br>" +
                             f"IS Value: %{{x}}{'%' if metric_name in ['Annualized Return', 'Cumulative Return'] else ''}<br>" +
                             "Count: %{y}<extra></extra>"
            ),
            row=row, col=col
        )
        
        # Calculate median for in-sample values
        median_val = np.median(is_values)
        if metric_name in ["Annualized Return", "Cumulative Return"]:
            scaled_median_val = median_val * 100
        else:
            scaled_median_val = median_val
        
        # Add vertical line for median value (blue dashed line)
        fig.add_vline(
            x=scaled_median_val,
            line_dash="dash",
            line_color="blue",
            line_width=2,
            annotation_text=f"Median: {formatter(median_val)}",
            annotation_position="top left",
            annotation=dict(
                font=dict(size=10, color="blue"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="blue",
                borderwidth=1
            ),
            row=row, col=col
        )
        
        # Add vertical line for OOS value (red dashed line)
        fig.add_vline(
            x=scaled_oos_val,
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text=f"OOS: {formatter(oos_val)}",
            annotation_position="top right",
            annotation=dict(
                font=dict(size=10, color="red"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            ),
            row=row, col=col
        )
    
    # Create title with period information
    period_info = ""
    if n_oos_days:
        period_info = f" (OOS Period: {n_oos_days} days)"
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{symphony_name} - In-Sample Distributions with OOS Values{period_info}",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        height=900,
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Value", row=1, col=2)
    fig.update_xaxes(title_text="Value", row=2, col=1)
    fig.update_xaxes(title_text="Value", row=2, col=2)
    
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    return fig

def create_rolling_analysis_plot(rolling_results: Dict[str, Any], symphony_name: str) -> go.Figure:
    """Create interactive Plotly plot for rolling analysis."""
    if not rolling_results.get('sufficient_data', False):
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for rolling analysis",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    windows = rolling_results.get('windows', [])
    if len(windows) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 windows for meaningful plot",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Extract data
    dates = [w['start_date'] for w in windows]
    
    # Create figure
    fig = go.Figure()
    
    # Add metric lines
    colors = {'sh': '#9467bd', 'cr': '#1f77b4', 'so': '#ff7f0e'}
    names = {'sh': 'Sharpe Ratio', 'cr': 'Annualized Return', 'so': 'Sortino Ratio'}
    
    for metric in ['sh', 'cr', 'so']:
        metric_data = []
        metric_dates = []
        for i, (date, window) in enumerate(zip(dates, windows)):
            iota_val = window['iotas'][metric]
            if np.isfinite(iota_val):
                metric_data.append(iota_val)
                metric_dates.append(date)
        
        if len(metric_data) >= 3:
            metric_data_smooth = smooth_iotas(metric_data, window=3)
            fig.add_trace(go.Scatter(
                x=metric_dates,
                y=metric_data_smooth,
                mode='lines+markers',
                name=f'{names[metric]} Iota (smoothed)',
                line=dict(color=colors[metric], width=2),
                marker=dict(size=4)
            ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                  annotation_text="Neutral Performance", annotation_position="bottom right")
    fig.add_hline(y=0.5, line_dash="dot", line_color="lightgreen", 
                  annotation_text="Overperformance (+0.5)", annotation_position="top right")
    fig.add_hline(y=-0.5, line_dash="dot", line_color="lightcoral", 
                  annotation_text="Underperformance (-0.5)", annotation_position="bottom right")
    
    # Update layout
    n_windows = rolling_results.get('n_windows', 0)
    window_size = rolling_results.get('window_size_days', 0)
    
    title_text = f'{symphony_name} - Rolling Iota Analysis'
    subtitle_text = f'{n_windows} windows ({window_size}d each) | Smoothed trends'
    
    fig.update_layout(
        title=dict(
            text=f"{title_text}<br><sub>{subtitle_text}</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis_title="Time Period (OOS)",
        yaxis_title="Iota (ι)",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=500
    )
    
    return fig

def create_full_backtest_rolling_plot(daily_ret: pd.Series, oos_start_dt: date, 
                                     ar_is_values: np.ndarray, sh_is_values: np.ndarray, 
                                     cr_is_values: np.ndarray, so_is_values: np.ndarray,
                                     ar_oos: float, sh_oos: float, cr_oos: float, so_oos: float,
                                     symphony_name: str, window_size: int = 252) -> go.Figure:
    """Create interactive Plotly plot for rolling iota analysis across entire backtest period."""
    
    # Create figure
    fig = go.Figure()
    
    # Get all dates from daily returns
    all_dates = daily_ret.index.tolist()
    
    # Calculate rolling iota for each metric across the entire period
    # For rolling analysis, we'll use annualized returns instead of cumulative returns
    # to ensure scale consistency across different time periods
    metrics_data = {
        'sh': {'is_values': sh_is_values, 'oos_value': sh_oos, 'name': 'Sharpe Ratio', 'color': '#9467bd'},
        'cr': {'is_values': ar_is_values, 'oos_value': ar_oos, 'name': 'Annualized Return', 'color': '#1f77b4'},  # Use annualized return instead
        'so': {'is_values': so_is_values, 'oos_value': so_oos, 'name': 'Sortino Ratio', 'color': '#ff7f0e'}
    }
    
    # Calculate rolling iota for each metric
    for metric_key, metric_info in metrics_data.items():
        rolling_iotas = []
        rolling_dates = []
        
        # Use window_size to calculate rolling iota
        for i in range(window_size, len(all_dates)):
            # Get the window of returns
            window_returns = daily_ret.iloc[i-window_size:i]
            
            # Calculate metric for this window
            if metric_key == 'sh':
                window_metric = sharpe_ratio(window_returns)
            elif metric_key == 'cr':
                # Use annualized return for consistency with the IS annualized return values
                window_metric = window_cagr(window_returns)
            elif metric_key == 'so':
                window_metric = sortino_ratio(window_returns)
            
            # Calculate iota using the IS distribution and this window's value
            if np.isfinite(window_metric):
                # For rolling iota, each window is treated as an "OOS" period
                # So n_oos should be the window size (252 days)
                # Use standard method for rolling analysis (not distribution-aware)
                iota_val = compute_iota(0.0, window_metric, window_size, is_values=metric_info['is_values'], use_distribution_aware=False)
                
                if np.isfinite(iota_val):
                    rolling_iotas.append(iota_val)
                    rolling_dates.append(all_dates[i-1])  # Use end date of window
        
        # Add line for this metric
        if len(rolling_iotas) >= 3:
            rolling_iotas_smooth = smooth_iotas(rolling_iotas, window=3)
            fig.add_trace(go.Scatter(
                x=rolling_dates,
                y=rolling_iotas_smooth,
                mode='lines+markers',
                name=f'{metric_info["name"]} Iota',
                line=dict(color=metric_info['color'], width=2),
                marker=dict(size=3),
                showlegend=True
            ))
        elif len(rolling_iotas) > 0:
            # Add unsmoothed line if we have some data but not enough for smoothing
            fig.add_trace(go.Scatter(
                x=rolling_dates,
                y=rolling_iotas,
                mode='lines+markers',
                name=f'{metric_info["name"]} Iota',
                line=dict(color=metric_info['color'], width=2),
                marker=dict(size=3),
                showlegend=True
            ))
        else:
            # Add empty trace to ensure legend shows all metrics
            fig.add_trace(go.Scatter(
                x=[],
                y=[],
                mode='lines',
                name=f'{metric_info["name"]} Iota (no data)',
                line=dict(color=metric_info['color'], width=2),
                showlegend=True
            ))
    
    # Add OOS start date vertical line using add_shape instead of add_vline
    # Convert date to datetime for Plotly compatibility
    oos_start_datetime = pd.Timestamp(oos_start_dt)
    fig.add_shape(
        type="line",
        x0=oos_start_datetime,
        x1=oos_start_datetime,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Add annotation for the OOS start line
    fig.add_annotation(
        x=oos_start_datetime,
        y=1.02,
        yref="paper",
        text="OOS Start",
        showarrow=False,
        font=dict(color="red", size=12),
        xanchor="left"
    )
    
    # Add reference lines (without annotations to avoid text overlay)
    fig.add_hline(y=0, line_dash="solid", line_color="gray")
    
    # Update layout
    title_text = f'{symphony_name} - Full Backtest Rolling Iota Proxy Analysis'
    subtitle_text = f'{window_size}d rolling windows | Smoothed trends'
    
    # Calculate y-axis range based on all iota values
    all_iotas = []
    for metric_key, metric_info in metrics_data.items():
        rolling_iotas = []
        for i in range(window_size, len(all_dates)):
            window_returns = daily_ret.iloc[i-window_size:i]
            
            if metric_key == 'sh':
                window_metric = sharpe_ratio(window_returns)
            elif metric_key == 'cr':
                window_metric = window_cagr(window_returns)
            elif metric_key == 'so':
                window_metric = sortino_ratio(window_returns)
            
            if np.isfinite(window_metric):
                iota_val = compute_iota(0.0, window_metric, window_size, is_values=metric_info['is_values'])
                if np.isfinite(iota_val):
                    all_iotas.append(iota_val)
    
    # Set reasonable y-axis range
    if all_iotas:
        y_min = min(all_iotas)
        y_max = max(all_iotas)
        y_range = y_max - y_min
        
        # Ensure minimum range and reasonable bounds
        if y_range < 0.1:  # If range is too small, expand it
            y_center = (y_max + y_min) / 2
            y_min = y_center - 0.5
            y_max = y_center + 0.5
        else:
            # Add some padding
            y_min = y_min - 0.1 * y_range
            y_max = y_max + 0.1 * y_range
        
        # Don't cap the bounds - let the data determine the range
        # This allows for extreme iota values to be visible
    else:
        y_min, y_max = -1.0, 1.0
    
    fig.update_layout(
        title=dict(
            text=f"{title_text}<br><sub>{subtitle_text}</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis_title="Date",
        yaxis_title="Iota (ι)",
        yaxis=dict(
            range=[y_min, y_max],
            tickmode='auto',
            nticks=10
        ),
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=500
    )
    
    return fig

# ===== STREAMLIT APP =====

def main():
    
    # Initialize query parameters for sharing
    query_params = st.query_params
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            color: #333;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin: 0.5rem 0;
        }
        .success-card {
            background-color: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #28a745;
        }
        .warning-card {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ffc107;
        }
        .critical-card {
            background-color: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #dc3545;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">📊 Iota Calculator</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; font-size: 1.5rem; color: #666; margin-bottom: 2rem;">Is your strategy\'s performance matching the backtest?</h2>', unsafe_allow_html=True)
    
    # Show quantstats status (only if not available)
    if not QUANTSTATS_AVAILABLE:
        st.warning("⚠️ quantstats not available - using internal calculations")
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔧 Configuration", "🔢 Results", "📊 Distributions", "📈 Rolling Analysis", "📚 Help"])
    
    # Configuration Tab
    with tab1:
        st.header("Analysis Configuration")
        
        # Add reset button
        if st.button("🔄 Reset All Data (Clear Cache)"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("✅ Cache cleared! Please re-run your analysis.")
            st.rerun()
        
                    # Show quantstats status in configuration (only if not available)
            if not QUANTSTATS_AVAILABLE:
                st.warning("⚠️ quantstats not available - install with: pip install quantstats")
            else:
                # Add test button for debugging (only show if user wants to test)
                if st.button("🧪 Test quantstats compatibility"):
                    with st.spinner("Testing quantstats compatibility..."):
                        is_working, message = test_quantstats_compatibility()
                        if is_working:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
                            st.info("The app will fall back to internal calculations when quantstats fails")
        
        st.markdown("---")  # Add divider
        
        # Main configuration form
        with st.form("analysis_form"):
            st.subheader("📝 Required Information")
            
            # Symphony URL
            default_url = query_params.get("url", "")
            url = st.text_input(
                "Composer Symphony URL *",
                value=default_url,
                help="Enter the full URL of your Composer symphony",
                placeholder="https://app.composer.trade/symphony/..."
            )
            
            # Date configuration
            col1, col2 = st.columns(2)
            with col1:
                default_early_date = query_params.get("early_date", "2000-01-01")
                early_date = st.date_input(
                    "Data Start Date:",
                    value=date.fromisoformat(default_early_date),
                    help="How far back to fetch data. Will default to oldest possible date if left untouched."
                )
            
            with col2:
                default_today_date = query_params.get("today_date", date.today().isoformat())
                today_date = st.date_input(
                    "Data End Date:",
                    value=date.fromisoformat(default_today_date),
                    help="End date for data fetching. Will default to most recent possible date if left untouched."
                )
            
            # OOS start date with auto-fetch functionality
            default_oos_start = query_params.get("oos_start", (date.today() - timedelta(days=730)).isoformat())
            
            # Auto-fetch checkbox
            auto_fetch_oos = st.checkbox(
                "🔗 Auto-fetch OOS date from symphony",
                value=True,
                help="Automatically fetch the OOS date from the symphony's last_semantic_update_at field. Uncheck to manually enter the date."
            )
            
            # Extract symphony ID from URL for auto-fetch
            symph_id = ""
            if url and "symphony/" in url:
                # Extract the symphony ID more carefully
                symph_part = url.split("symphony/")[-1]
                # Remove any query parameters, fragments, or additional paths
                symph_id = symph_part.split("?")[0].split("#")[0].split("/")[0]
                st.info(f"Extracted Symphony ID: {symph_id}")
            
            # Auto-fetch OOS date if enabled and symphony ID is available
            fetched_oos_date = None
            if auto_fetch_oos and symph_id:
                try:
                    with st.spinner("Fetching OOS date from symphony..."):
                        fetched_oos_date = fetch_oos_date_from_symphony(symph_id)
                        if fetched_oos_date:
                            st.success(f"✅ Auto-fetched OOS date: {fetched_oos_date}")
                        else:
                            st.warning("⚠️ Could not auto-fetch OOS date. Please enter manually.")
                except Exception as e:
                    st.warning(f"⚠️ Auto-fetch failed: {str(e)}. Please enter OOS date manually.")
                    fetched_oos_date = None
            
            # Smart default: Use fetched date if available, otherwise use a reasonable default
            if fetched_oos_date:
                oos_start_value = fetched_oos_date
            else:
                # Use a smart default based on current date (typically 2 years ago)
                smart_default = date.today() - timedelta(days=730)  # 2 years ago
                oos_start_value = date.fromisoformat(default_oos_start)
                

            
            oos_start = st.date_input(
                "Out-of-Sample Start Date *",
                value=oos_start_value,
                help="⚠️ CRITICAL: Date when your 'live trading' or out-of-sample period begins. Everything before this is historical backtest data, everything after is 'real world' performance."
            )
            
            st.markdown("---")
            st.subheader("⚙️ Analysis Parameters")
            
            # Analysis parameters in columns
            col1, col2 = st.columns(2)
            with col1:
                default_n_slices = int(query_params.get("n_slices", "100"))
                n_slices = st.number_input(
                    "Number of IS Slices:",
                    min_value=10,
                    max_value=500,
                    value=default_n_slices,
                    help="How many historical periods to compare against"
                )
            
            with col2:
                default_overlap = query_params.get("overlap", "true").lower() == "true"
                overlap = st.checkbox(
                    "Allow Overlapping Slices",
                    value=default_overlap,
                    help="Whether historical comparison periods can overlap (recommended: True for more data)"
                )
            
            # Rolling analysis parameters
            st.subheader("🔄 Rolling Analysis Parameters")
            col1, col2 = st.columns(2)
            with col1:
                default_enable_rolling = query_params.get("enable_rolling", "true").lower() == "true"
                enable_rolling = st.checkbox(
                    "Enable Rolling Window Analysis",
                    value=default_enable_rolling,
                    help="Perform overfitting detection using rolling windows"
                )
            
            with col2:
                if enable_rolling:
                    default_auto_window = query_params.get("auto_window", "true").lower() == "true"
                    auto_window = st.checkbox(
                        "Auto Window Size",
                        value=default_auto_window,
                        help="Automatically determine optimal window size based on OOS period length. If unchecked, you'll need to hit 'Run Iota Analysis' and then fill out your preferred window parameters"
                    )
                else:
                    auto_window = True
            
            # Quantstats parameters
            st.subheader("📊 Calculation Engine Parameters")
            col1, col2 = st.columns(2)
            with col1:
                default_use_quantstats = query_params.get("use_quantstats", "false").lower() == "true"
                use_quantstats = st.checkbox(
                    "Use quantstats for calculations",
                    value=default_use_quantstats,
                    help="Use quantstats library for enhanced financial calculations. When disabled, uses internal calculations."
                )
            
            with col2:
                if use_quantstats and not QUANTSTATS_AVAILABLE:
                    st.warning("⚠️ quantstats not available - will use internal calculations")
                elif use_quantstats and QUANTSTATS_AVAILABLE:
                    st.success("✅ quantstats available")
                else:
                    st.info("ℹ️ Using internal calculations")
            
            # Show manual window settings when auto is disabled
            if enable_rolling and not auto_window:
                col1, col2 = st.columns(2)
                with col1:
                    default_window_size = int(query_params.get("window_size", "126"))
                    window_size = st.number_input(
                        "Window Size (days):",
                        min_value=21,
                        max_value=252,
                        value=default_window_size,
                        help="Size of each rolling window in days"
                    )
                with col2:
                    default_step_size = int(query_params.get("step_size", "21"))
                    step_size = st.number_input(
                        "Step Size (days):",
                        min_value=1,
                        max_value=63,
                        value=default_step_size,
                        help="Days between window starts"
                    )
            else:
                window_size = None
                step_size = None
            
            # Note about form behavior
            if enable_rolling and not auto_window:
                st.info("💡 **Note**: Manual window settings will be applied when you submit the form.")
            
            # Optional exclusion windows
            st.subheader("🚫 Exclusion Windows (Optional)")
            default_exclusions_str = query_params.get("exclusions_str", "")
            exclusions_str = st.text_area(
                "Exclude specific date ranges:",
                value=default_exclusions_str,
                help="Exclude market crashes, unusual periods, etc. Format: YYYY-MM-DD to YYYY-MM-DD, separated by commas",
                placeholder="2020-03-01 to 2020-05-01, 2022-01-01 to 2022-02-01"
            )
            
            # Submit button
            submitted = st.form_submit_button("🚀 Run Iota Analysis", type="primary")
            
            if submitted:
                # Validate inputs
                if not url.strip():
                    st.error("❌ Please enter a Composer Symphony URL")
                elif oos_start <= early_date:
                    st.error("❌ Out-of-Sample start date must be after the data start date")
                elif oos_start >= today_date:
                    st.error("❌ Out-of-Sample start date must be before the data end date")
                else:
                    # Store in session state
                    st.session_state.analysis_config = {
                        'url': url,
                        'early_date': early_date,
                        'today_date': today_date,
                        'oos_start': oos_start,
                        'n_slices': n_slices,
                        'overlap': overlap,
                        'exclusions_str': exclusions_str,
                        'enable_rolling': enable_rolling,
                        'auto_window': auto_window,
                        'window_size': window_size,
                        'step_size': step_size,
                        'use_quantstats': use_quantstats
                    }
                    st.session_state.run_analysis = True
                    st.session_state.auto_switch_to_results = True
                    
                    # Update URL with current parameters for sharing
                    
                    params = {
                        "url": url,
                        "early_date": early_date.isoformat(),
                        "today_date": today_date.isoformat(),
                        "oos_start": oos_start.isoformat(),
                        "n_slices": str(n_slices),
                        "overlap": str(overlap).lower(),
                        "exclusions_str": exclusions_str,
                        "enable_rolling": str(enable_rolling).lower(),
                        "auto_window": str(auto_window).lower(),
                        "use_quantstats": str(use_quantstats).lower(),
                    }
                    if window_size is not None:
                        params["window_size"] = str(window_size)
                    if step_size is not None:
                        params["step_size"] = str(step_size)
                    
                    # Set query parameters in the URL
                    for k, v in params.items():
                        if v:  # Only include non-empty values
                            st.query_params[k] = str(v)
                    
                    # Create shareable URL for display
                    encoded_params = []
                    for k, v in params.items():
                        if v:  # Only include non-empty values
                            encoded_key = urllib.parse.quote(k)
                            encoded_value = urllib.parse.quote(str(v))
                            encoded_params.append(f"{encoded_key}={encoded_value}")
                    
                    query_string = "&".join(encoded_params)
                    base_url = "https://iotametrics.streamlit.app/"
                    shareable_url = f"{base_url}?{query_string}"
                    
                    # Store the shareable URL in session state for display outside the form
                    st.session_state.shareable_url = shareable_url
                    
                    # Success message with clear navigation instructions
                    st.success("✅ Configuration saved! Click the '📊 Results' tab above to view your analysis.")

        # Display shareable URL outside the form (if available)
        if hasattr(st.session_state, 'shareable_url') and st.session_state.shareable_url:
            st.markdown("---")
            st.markdown("### 🔗 Share Your Analysis")
            st.info(f"**Shareable URL**: Copy this link to share your analysis settings with others:")
            
            # Create a text area for easy copying
            st.text_area(
                "Shareable URL (select all and copy):",
                value=st.session_state.shareable_url,
                height=100,
                help="Select all text (Ctrl+A) then copy (Ctrl+C)",
                key="shareable_url_textarea"
            )
            






    # Results Tab
    with tab2:
        st.header("🔢 Core Iota Analysis Results")
        st.markdown("")  # Add spacing after header
        
        if hasattr(st.session_state, 'run_analysis') and st.session_state.run_analysis:
            config = st.session_state.analysis_config
            
            try:
                # Data preparation section
                st.markdown("### 🔄 Data Preparation")
                
                # Parse exclusions
                exclusions = parse_exclusion_input(config['exclusions_str'])
                if exclusions:
                    st.info(f"📋 {len(exclusions)} exclusion window(s) will be applied")
                
                # Run the analysis with progress tracking
                with st.spinner("🔄 Fetching backtest data from Composer..."):
                    alloc, sym_name, tickers = fetch_backtest(
                        config['url'], 
                        config['early_date'].strftime("%Y-%m-%d"), 
                        config['today_date'].strftime("%Y-%m-%d")
                    )
                
                st.success(f"✅ Successfully fetched data for strategy: **{sym_name}**")
                st.markdown("")  # Add spacing
                
                with st.spinner("🧮 Calculating portfolio returns..."):
                    # Capture stdout during portfolio calculation
                    captured_output = io.StringIO()
                    with contextlib.redirect_stdout(captured_output):
                        daily_ret, _ = calculate_portfolio_returns(alloc, tickers)
                
                # Convert index to date
                daily_ret.index = pd.to_datetime(daily_ret.index).date
                
                st.info(f"📈 Loaded {len(daily_ret)} days of return data")
                st.markdown("")  # Add spacing
                
                # Apply exclusions
                if exclusions:
                    mask = pd.Series(True, index=daily_ret.index)
                    for s, e in exclusions:
                        mask &= ~((daily_ret.index >= s) & (daily_ret.index <= e))
                    removed = (~mask).sum()
                    daily_ret = daily_ret[mask]
                    st.warning(f"🚫 Excluded {removed} days across {len(exclusions)} window(s)")
                    st.markdown("")  # Add spacing
                
                # Data split section
                st.markdown("### 📊 Data Split Summary")
                
                # Split data
                oos_start_dt = config['oos_start']
                is_ret = daily_ret[daily_ret.index < oos_start_dt]
                oos_ret = daily_ret[daily_ret.index >= oos_start_dt]
                
                if len(is_ret) < 30 or len(oos_ret) < 30:
                    st.error("❌ Insufficient data: Need at least 30 days in both IS and OOS periods")
                    return
                
                n_oos = len(oos_ret)
                n_is = len(is_ret)
                
                # Show data split summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("In-Sample Days", f"{n_is}")
                    st.caption(f"{is_ret.index[0]} to {is_ret.index[-1]}")
                with col2:
                    st.metric("Out-of-Sample Days", f"{n_oos}")
                    st.caption(f"{oos_ret.index[0]} to {oos_ret.index[-1]}")
                with col3:
                    reliability = assess_sample_reliability(n_is, n_oos)
                    st.metric("Reliability", reliability.replace("_", " "), 
                             help="Statistical reliability based on sample size. HIGH_CONFIDENCE: ≥378 days (~1.5 years) - Strong statistical power. MODERATE_CONFIDENCE: 189-377 days (~9 months) - Reasonable statistical power. LOW_CONFIDENCE: 90-188 days (~4.5 months) - Limited but usable. INSUFFICIENT_DATA: <90 days - Insufficient for reliable analysis.")
                
                st.markdown("---")  # Add divider before analysis
                st.markdown("### 🧮 Core Analysis")
                
                with st.spinner("📊 Running core Iota analysis..."):
                    # Get quantstats preference from config
                    use_quantstats = config.get('use_quantstats', False)
                    
                    # Calculate OOS metrics
                    ar_oos = window_cagr(oos_ret, use_quantstats)
                    sh_oos = sharpe_ratio(oos_ret, use_quantstats)
                    cr_oos = cumulative_return(oos_ret, use_quantstats)
                    so_oos = sortino_ratio(oos_ret, use_quantstats)
                    
                    # Build IS slices
                    slice_len = n_oos
                    slices = build_slices(is_ret, slice_len, config['n_slices'], config['overlap'])
                    
                    if not slices:
                        st.error("❌ Could not create IS slices of required length")
                        return
                    
                    # Calculate IS metrics for each slice
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    rows = []
                    for i, s in enumerate(slices[::-1], 1):
                        progress_bar.progress(i / len(slices))
                        status_text.text(f"Processing slice {i}/{len(slices)}")
                        
                        rows.append({
                            "slice": i,
                            "start": s.index[0],
                            "end": s.index[-1],
                            "ar_is": window_cagr(s, use_quantstats),
                            "sh_is": sharpe_ratio(s, use_quantstats),
                            "cr_is": cumulative_return(s, use_quantstats),
                            "so_is": sortino_ratio(s, use_quantstats)
                        })
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    df = pd.DataFrame(rows)
                    
                    # Compute iota statistics
                    ar_stats = compute_iota_with_stats(df["ar_is"].values, ar_oos, n_oos, "Annualized Return", overlap=config['overlap'])
                    sh_stats = compute_iota_with_stats(df["sh_is"].values, sh_oos, n_oos, "Sharpe Ratio", overlap=config['overlap'])
                    cr_stats = compute_iota_with_stats(df["cr_is"].values, cr_oos, n_oos, "Cumulative Return", overlap=config['overlap'])
                    so_stats = compute_iota_with_stats(df["so_is"].values, so_oos, n_oos, "Sortino Ratio", overlap=config['overlap'])
                
                # Store core results in session state for all tabs
                # Convert pandas Series to dict for storage compatibility
                daily_ret_dict = {
                    'dates': daily_ret.index.tolist(),
                    'values': daily_ret.values.tolist()
                }
                is_ret_dict = {
                    'dates': is_ret.index.tolist(),
                    'values': is_ret.values.tolist()
                }
                oos_ret_dict = {
                    'dates': oos_ret.index.tolist(),
                    'values': oos_ret.values.tolist()
                }
                
                st.session_state.core_results = {
                    'sym_name': sym_name,
                    'daily_ret': daily_ret_dict,
                    'is_ret': is_ret_dict,
                    'oos_ret': oos_ret_dict,
                    'ar_stats': ar_stats,
                    'sh_stats': sh_stats,
                    'cr_stats': cr_stats,
                    'so_stats': so_stats,
                    'ar_oos': ar_oos,
                    'sh_oos': sh_oos,
                    'cr_oos': cr_oos,
                    'so_oos': so_oos,
                    'reliability': reliability,
                    'config': config,
                    'ar_is_values': df["ar_is"].values,
                    'sh_is_values': df["sh_is"].values,
                    'cr_is_values': df["cr_is"].values,
                    'so_is_values': df["so_is"].values,
                    'n_slices': len(slices)
                }
                
                # Display core results
                display_core_results(sym_name, ar_stats, sh_stats, cr_stats, so_stats, 
                                   ar_oos, sh_oos, cr_oos, so_oos, reliability, config)
                
                # Show calculation method used
                if config.get('use_quantstats', False) and QUANTSTATS_AVAILABLE:
                    st.success("📊 Financial metrics calculated using quantstats for enhanced accuracy")
                elif config.get('use_quantstats', False) and not QUANTSTATS_AVAILABLE:
                    st.warning("📊 quantstats requested but not available - using internal methods")
                else:
                    st.info("📊 Financial metrics calculated using internal methods")
                
                st.markdown("---")  # Add divider before rolling analysis
                
                # Run rolling analysis if enabled
                if config['enable_rolling']:
                    st.markdown("### 🔄 Rolling Window Analysis")
                    with st.spinner("🔄 Running rolling window analysis..."):
                        rolling_results = rolling_oos_analysis(
                            daily_ret, oos_start_dt, is_ret, 
                            config['n_slices'], config['overlap'],
                            config['window_size'], config['step_size'],
                            use_quantstats=use_quantstats
                        )
                        st.session_state.rolling_results = rolling_results
                    
                    st.success("✅ Rolling analysis complete! Check the 'Rolling Analysis' tab for time specific insights.")
                else:
                    st.info("ℹ️ Rolling analysis disabled. Enable it in the configuration to detect time specific patterns.")
                
                # Reset the flag
                st.session_state.run_analysis = False
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)
        else:
            st.info("👈 Please configure and run your analysis in the 'Configuration' tab first.")

    # Distributions Tab
    with tab3:
        st.header("📊 In-Sample Distributions")
        st.markdown("")  # Add spacing after header
        
        if hasattr(st.session_state, 'core_results') and st.session_state.core_results:
            core_results = st.session_state.core_results
            
            st.success("✅ Distribution analysis ready!")
            st.markdown("")  # Add spacing
            
            # Display time slice information
            st.markdown("### 📅 Time Slice Size")
            if 'config' in core_results and core_results['config']:
                config = core_results['config']
                oos_start = config.get('oos_start')
                today_date = config.get('today_date')
                if oos_start and today_date:
                    oos_days = (today_date - oos_start).days
                    st.info(f"**Time Slice Size**: {oos_days} days (matching OOS period length)")
                else:
                    st.info("**Time Slice Size**: Unable to determine (missing date information)")
            else:
                st.info("**Time Slice Size**: Unable to determine (missing configuration)")
            
            # Display distribution histograms
            st.markdown("### 📈 Metric Distributions")
            st.markdown("Histograms show the distribution of in-sample values for each metric, with red dashed lines indicating where your out-of-sample values fall.")
            
            # Create and display histogram
            oos_days = None
            if 'config' in core_results and core_results['config']:
                config = core_results['config']
                oos_start = config.get('oos_start')
                today_date = config.get('today_date')
                if oos_start and today_date:
                    oos_days = (today_date - oos_start).days
            
            fig = create_distribution_histograms(
                core_results['ar_is_values'], 
                core_results['sh_is_values'], 
                core_results['cr_is_values'], 
                core_results['so_is_values'],
                core_results['ar_oos'], 
                core_results['sh_oos'], 
                core_results['cr_oos'], 
                core_results['so_oos'],
                core_results['sym_name'],
                oos_days
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation
            st.markdown("### 🎯 Interpretation")
            st.markdown("""
            - **Red dashed lines**: Show where your OOS performance falls relative to the IS distribution
            - **Histogram bars**: Show the frequency of different performance levels during the backtest period
            - **Left of distribution**: OOS underperforming relative to backtest expectations
            - **Right of distribution**: OOS outperforming relative to backtest expectations
            - **Center of distribution**: OOS performance matches backtest expectations
            """)
        else:
            st.info("📊 No analysis data available. Please run the analysis first in the 'Results' tab.")

    # Rolling Analysis Tab
    with tab4:
        st.header("📈 Rolling Window Analysis")
        st.markdown("")  # Add spacing after header
        
        if hasattr(st.session_state, 'rolling_results') and st.session_state.rolling_results:
            rolling_results = st.session_state.rolling_results
            
            if rolling_results.get('sufficient_data', False):
                st.success("✅ Rolling analysis completed successfully!")
                st.markdown("")  # Add spacing
                
                # Display results
                st.markdown("### 📊 Analysis Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Windows", rolling_results['n_windows'])
                with col2:
                    st.metric("Window Size", f"{rolling_results['window_size_days']}d")
                
                st.markdown("")  # Add spacing
                
                st.markdown("")  # Add spacing
                
                # Create and display plot
                if hasattr(st.session_state, 'core_results'):
                    sym_name = st.session_state.core_results['sym_name']
                    
                    # Center the chart section header
                    st.markdown('<h3 style="text-align: center;">📈 OOS Rolling Performance Chart</h3>', unsafe_allow_html=True)
                    fig = create_rolling_analysis_plot(rolling_results, sym_name)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add full backtest rolling chart
                    st.markdown("")  # Add spacing
                    st.markdown('<h3 style="text-align: center;">📊 Full Backtest Rolling Iota Chart</h3>', unsafe_allow_html=True)
                    
                    # Get required data from core_results
                    if hasattr(st.session_state, 'core_results') and st.session_state.core_results:
                        core_results = st.session_state.core_results
                        config = core_results.get('config', {})
                        oos_start_dt = config.get('oos_start')
                        
                        if oos_start_dt and 'daily_ret' in core_results:
                            # Reconstruct pandas Series from stored dict
                            daily_ret_data = core_results['daily_ret']
                            daily_ret = pd.Series(
                                data=daily_ret_data['values'],
                                index=pd.to_datetime(daily_ret_data['dates'])
                            )
                            

                            
                            full_fig = create_full_backtest_rolling_plot(
                                daily_ret,
                                oos_start_dt,
                                core_results['ar_is_values'],
                                core_results['sh_is_values'],
                                core_results['cr_is_values'],
                                core_results['so_is_values'],
                                core_results['ar_oos'],
                                core_results['sh_oos'],
                                core_results['cr_oos'],
                                core_results['so_oos'],
                                sym_name
                            )
                            st.plotly_chart(full_fig, use_container_width=True)
                        else:
                            st.warning("⚠️ Missing data for full backtest chart")
                    else:
                        st.warning("⚠️ Core results not available for full backtest chart")
                    
                    # Show last window info (centered)
                    if rolling_results.get('windows'):
                        last_window = rolling_results['windows'][-1]
                        st.markdown("")  # Add spacing
                        st.markdown(f'<p style="text-align: center;"><strong>📅 Last Window</strong>: {last_window["start_date"]} to {last_window["end_date"]}</p>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Insufficient data for rolling analysis")
                st.write("**Recommendation**: Extend OOS period to at least 6 months for meaningful rolling analysis")
        else:
            st.info("📊 No rolling analysis data available. Please run the analysis first in the 'Results' tab with rolling analysis enabled.")

    # Help Tab
    with tab5:
        show_comprehensive_help()

def display_core_results(sym_name, ar_stats, sh_stats, cr_stats, so_stats, 
                        ar_oos, sh_oos, cr_oos, so_oos, reliability, config):
    """Display the core analysis results."""
    
    # Header with success message
    st.success(f"🎉 Core Analysis Complete for **{sym_name}**")
    st.markdown("---")  # Add divider
    
    # Overall summary section
    st.subheader("📊 Overall Summary")
    st.markdown("")  # Add spacing
    
    # Metrics in columns
    col1, col2 = st.columns(2)
    
    # Calculate average iota
    iotas = [ar_stats['iota'], sh_stats['iota'], cr_stats['iota'], so_stats['iota']]
    finite_iotas = [i for i in iotas if np.isfinite(i)]
    avg_iota = np.mean(finite_iotas) if finite_iotas else 0
    avg_rating = iota_to_persistence_rating(avg_iota)
    
    with col1:
        st.metric("Composite Iota", f"{avg_iota:+.3f}", 
                 help="Average of all metric iota values. Measures how many standard deviations your overall OOS performance differs from backtest expectations. Positive = outperforming, negative = underperforming.")
    with col2:
        st.metric("Composite Persistence Rating", f"{avg_rating}", 
                 help="0-100+ scale rating of strategy persistence. 100 = strategy exactly matches backtest, <100 = strategy underperforms relative to backtest, >100 = strategy outperforms backtest.")
    
    st.markdown("")  # Add spacing after metrics
    
    # Overall interpretation with better spacing
    interpretation = interpret_iota_directly(avg_iota)
    st.markdown("### Overall Assessment")
    if avg_iota >= 0.5:
        st.markdown(f'<div class="success-card" style="font-size: 1.2rem;"><strong>{interpretation}</strong></div>', 
                   unsafe_allow_html=True)
    elif avg_iota >= 0.1:
        st.markdown(f'<div class="metric-card" style="font-size: 1.2rem;"><strong>{interpretation}</strong></div>', 
                   unsafe_allow_html=True)
    elif avg_iota >= -0.1:
        st.markdown(f'<div class="info-card" style="font-size: 1.2rem;"><strong>⚠️ {interpretation}</strong></div>', unsafe_allow_html=True)
    elif avg_iota >= -0.5:
        st.markdown(f'<div class="info-card" style="font-size: 1.2rem;"><strong>📉 {interpretation}</strong></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="critical-card" style="font-size: 1.2rem;"><strong>⚠️ {interpretation}</strong></div>', unsafe_allow_html=True)
    
    # Add significant spacing after Overall Assessment
    st.markdown("")  # Add spacing
    st.markdown("")  # Add more spacing
    st.markdown("")  # Add even more spacing
    
    # Detailed metrics section
    st.subheader("📈 Detailed Metric Analysis")
    st.markdown("")  # Add spacing
    
    metrics_data = [
        ("Annualized Return", ar_stats, ar_oos, lambda x: f"{x*100:.2f}%"),
        ("Sharpe Ratio", sh_stats, sh_oos, lambda x: f"{x:.3f}"),
        ("Cumulative Return", cr_stats, cr_oos, lambda x: f"{x*100:.2f}%"),
        ("Sortino Ratio", so_stats, so_oos, format_sortino_output)
    ]
    
    for i, (metric_name, stats_dict, oos_val, formatter) in enumerate(metrics_data):
        with st.expander(f"📊 {metric_name}", expanded=True):
            display_metric_detail(metric_name, stats_dict, oos_val, formatter)
        
        # Add spacing between expanders (except for the last one)
        if i < len(metrics_data) - 1:
            st.markdown("")  # Add spacing between metric sections
    
    # Add distribution-aware analysis section at the bottom
    st.markdown("---")  # Add divider before distribution analysis
    
    # Check for distribution issues and show distribution-aware analysis
    distribution_issues = []
    distribution_info = []
    
    metrics_with_stats = [
        ("Annualized Return", ar_stats),
        ("Sharpe Ratio", sh_stats),
        ("Cumulative Return", cr_stats),
        ("Sortino Ratio", so_stats)
    ]
    
    for metric_name, stats in metrics_with_stats:
        if 'distribution_method' in stats:
            method = stats['distribution_method']
            confidence = stats['distribution_confidence']
            is_skewed = stats.get('is_skewed', False)
            has_fat_tails = stats.get('has_fat_tails', False)
            
            distribution_info.append(f"**{metric_name}**: {method.title()} method ({confidence} confidence)")
            
            if is_skewed or has_fat_tails:
                issues = []
                if is_skewed:
                    issues.append("skewed")
                if has_fat_tails:
                    issues.append("fat-tailed")
                distribution_issues.append(f"{metric_name} ({', '.join(issues)})")
    
    # Show distribution information
    if distribution_info:
        st.info(f"""
        📊 **Distribution-Aware Analysis**
        
        The iota calculation automatically adapts to your data's distribution shape:
        
        {chr(10).join(distribution_info)}
        
        **Standard**: Used for normal distributions
        **Robust**: Used for skewed or fat-tailed distributions (IQR-based)
        **Percentile**: Used for complex distributions (percentile-based)
        """)
    
    # Show info for distribution characteristics
    if distribution_issues:
        st.info(f"""
        📊 **Distribution Characteristics Detected**
        
        The following metrics have non-normal distributions:
        - {', '.join(distribution_issues)}
        
        **Impact**: The iota calculation has been automatically adjusted to use robust methods that are less sensitive to outliers and distribution shape.
        
        **Recommendation**: The adjusted iota values should be more reliable than standard calculations for these metrics.
        """)

def display_metric_detail(metric_name, stats_dict, oos_val, formatter):
    """Display detailed analysis for a single metric."""
    
    # Key metrics section
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("IS Median", formatter(stats_dict['median_is']))
    with col2:
        st.metric("OOS Actual", formatter(oos_val))
    with col3:
        iota = stats_dict['iota']
        st.metric("Iota (ι)", f"{iota:+.3f}")
    with col4:
        st.metric("Persistence Rating", f"{stats_dict['persistence_rating']}")
    
    st.markdown("")  # Add spacing
    
    # Interpretation section
    st.markdown("#### Interpretation")
    interpretation = interpret_iota_directly(stats_dict['iota'])
    if stats_dict['iota'] >= 0.5:
        st.success(f"**{interpretation}**")
    elif stats_dict['iota'] >= -0.5:
        st.info(f"**{interpretation}**")
    else:
        st.warning(f"**{interpretation}**")
    
    st.markdown("")  # Add spacing
    
    # Statistical details section
    st.markdown("#### 📈 Statistical Details")
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence interval
        ci_lower, ci_upper = stats_dict['confidence_interval']
        if np.isfinite(ci_lower) and np.isfinite(ci_upper):
            st.write(f"**95% CI for Iota:** [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        # IQR
        q25, q75 = stats_dict['iqr_is']
        st.write(f"**IS Range (25th-75th):** {formatter(q25)} - {formatter(q75)}")
    
    with col2:
        # Distribution information
        if 'distribution_method' in stats_dict:
            method = stats_dict['distribution_method']
            st.write(f"**Distribution Method:** {method.title()}")
            
            if stats_dict.get('is_skewed', False):
                skewness = stats_dict.get('skewness', 0)
                st.write(f"**Skewness:** {skewness:.3f} (skewed)")
            
            if stats_dict.get('has_fat_tails', False):
                kurtosis = stats_dict.get('kurtosis', 0)
                st.write(f"**Kurtosis:** {kurtosis:.3f} (fat-tailed)")
    
    # Distribution explanation
    if 'distribution_method' in stats_dict and stats_dict['distribution_method'] != 'standard':
        method = stats_dict['distribution_method']
        if method == 'robust':
            st.info("**Robust Method**: Using IQR-based calculation to handle skewed or fat-tailed distributions.")
        elif method == 'percentile':
            st.info("**Percentile Method**: Using percentile-based calculation for complex distributions.")

def show_comprehensive_help():
    """Show help and documentation for the Iota Calculator."""
    
    st.header("📚 Iota Calculator Guide")
    
    # Create help sub-tabs
    help_tab1, help_tab2, help_tab3, help_tab4, help_tab5, help_tab6 = st.tabs([
        "🎯 Quick Start", "🧮 Methodology", "📊 Interpretation", "📈 Distributions", "🔄 Rolling Analysis", "❓ FAQ"
    ])
    
    with help_tab1:
        st.subheader("🎯 Quick Start Guide")
        
        st.markdown("""
        ## What is the Iota Calculator?
        
        The **Iota Calculator** helps you understand whether your trading strategy is performing as expected 
        based on historical patterns. It answers the key question: *"Is my strategy's performance consistent with its backtest?"*
        
        ### Key Features:
        - 📊 **Core Iota Analysis**: Compare OOS performance to historical expectations
        - 🧠 **Distribution-Aware Analysis**: Automatically adapts to your data's distribution shape
        - 📊 **Distribution Analysis**: Visualize in-sample distributions with OOS values
        - 🔄 **Rolling Window Analysis**: Time-specific performance analysis with rolling windows
        - 📈 **Interactive Visualizations**: Track performance trends with Plotly charts
        - 🎯 **Statistical Analysis**: Autocorrelation-adjusted analysis and confidence intervals
        - 📈 **Enhanced Metrics**: Uses quantstats for professional-grade financial calculations (when available)
        
        ## Step-by-Step Guide
        
        ### 1. 🔗 Get Your Composer Symphony URL
        - Log into Composer
        - Open your symphony
        - Copy the full URL from your browser
        - Paste it into the "Composer Symphony URL" field
        
        ### 2. 📅 Set Your Out-of-Sample Date
        **This is the most important setting!**
        
        - Choose the date when you started "live trading" or when your backtest ended
        - Everything **before** this date = historical backtest data
        - Everything **after** this date = "real world" performance
        - Example: If you started trading the strategy on Jan 1, 2022, set OOS start to 2022-01-01
        
        ### 3. ⚙️ Configure Analysis Parameters
        - **Number of IS Slices**: How many historical periods to compare (100 is good default)
        - **Overlapping Slices**: Keep this True for better statistics
        - **Rolling Analysis**: Enable for time-specific performance analysis
        - **Exclusion Windows**: Optional - exclude market crashes or unusual periods
        
        ### 4. 🚀 Run the Analysis
        - Click "Run Iota Analysis"
        - Wait for the analysis to complete (may take 2-3 minutes)
        - View core results in the "Results" tab
        - Explore distributions in the "Distributions" tab
        - Check rolling analysis in the "Rolling Analysis" tab
        
        ### 5. 🔗 Share Your Analysis
        - After running the analysis, a shareable URL will be generated
        - Copy the URL to share your exact configuration and results with others
        - Anyone with the link can view the same analysis settings and results
        - Perfect for team collaboration, peer review, or documentation
        
        ## Understanding Your Results
        
        ### 🎯 Iota (ι) Score
        **The main number that tells you how your strategy is doing:**
        
        **Standard Method** (for normal distributions):
        - **ι = +1.0**: You're doing 1 standard deviation BETTER than expected ✅
        - **ι = 0.0**: You're performing exactly as expected ➡️
        - **ι = -1.0**: You're doing 1 standard deviation WORSE than expected ⚠️
        
        **Robust Method** (for skewed/fat-tailed distributions):
        - **ι = +1.0**: You're doing 1 IQR unit BETTER than expected ✅
        - **ι = 0.0**: You're performing exactly as expected ➡️
        - **ι = -1.0**: You're doing 1 IQR unit WORSE than expected ⚠️
        
        **Percentile Method** (for complex distributions):
        - **ι = +1.0**: You're performing at ~84th percentile vs. historical expectations ✅
        - **ι = 0.0**: You're performing at ~50th percentile (median) ➡️
        - **ι = -1.0**: You're performing at ~16th percentile vs. historical expectations ⚠️
        
        **Note**: The system automatically chooses the most appropriate method based on your data's distribution characteristics.
        
        ### 📊 Persistence Rating
        **Easy-to-understand 0-500 scale:**
        
        - **100**: Neutral performance (matches expectations)
        - **>100**: Outperforming expectations
        - **<100**: Underperforming expectations

        ### 🔄 Rolling Analysis
        **Time-specific analysis shows how strategy performance varies over time:**
        
        - **Stable Performance**: Iota values that stay relatively consistent ✅
        - **Performance Trends**: Patterns that show improving or declining performance over time
        - **Volatile Performance**: High variance in rolling iota values
        - **OOS vs Proxy**: Compare actual OOS performance with pre-OOS proxy iota
        """)
    
    with help_tab2:
        st.subheader("🧮 Detailed Methodology")
        
        st.markdown("""
        ## WHAT IS IOTA (ι)?
        
        Iota is a standardized metric that measures how your out-of-sample performance compares to historical expectations. The calculation method automatically adapts to your data's distribution characteristics.
        
        **Standard Method** (for normal distributions):
        ```
        ι = weight × (OOS_metric - IS_median) / IS_std_dev
        ```
        
        **Robust Method** (for skewed or fat-tailed distributions):
        ```
        ι = weight × (OOS_metric - IS_median) / (IS_IQR / 1.35)
        ```
        
        **Percentile Method** (for complex distributions):
        ```
        ι = weight × z_score_from_percentile(OOS_percentile_rank)
        ```
        
        Where:
        - `weight = min(1.0, √(OOS_days / 252))` accounts for sample size reliability
        - `OOS_metric` = your strategy's out-of-sample performance value
        - `IS_median` = median of all in-sample slice performances  
        - `IS_std_dev` = standard deviation of in-sample slice performances
        - `IS_IQR` = interquartile range of in-sample performances
        
        **INTERPRETATION:**
        - `ι = +1.0`: OOS performed 1 standard deviation better than historical median
        - `ι = -1.0`: OOS performed 1 standard deviation worse than historical median
        - `ι = 0`: OOS performance matches historical expectations exactly
        - `|ι| ≥ 1.0`: Major difference
        - `|ι| < 0.1`: Minimal difference (within noise)
        
        ## Step-by-Step Analysis Process
        
        ### Step 1: Data Preparation and Slice Construction
        **What happens:**
        1. Historical data is split into In-Sample (IS) and Out-of-Sample (OOS) periods at your specified date
        2. The IS period is divided into multiple overlapping or non-overlapping "slices"
        3. Each slice has the same length as your OOS period for fair comparison
        
        **Rationale:**
        - **Temporal consistency**: Each IS slice represents what your strategy would have done during a period of identical length to your actual OOS period
        - **Distribution building**: Multiple slices create a distribution of historical performance under similar conditions
        - **Avoiding look-ahead bias**: Only data prior to OOS start date is used for IS analysis
        
        ### Step 2: Metric Calculation
        **What happens:**
        1. Four core metrics calculated for OOS period: Annualized Return, Sharpe Ratio, Cumulative Return, Sortino Ratio
        2. Same metrics calculated for each IS slice
        3. Statistical distribution properties computed for IS metrics (median, standard deviation, quartiles)
        4. **Note**: Rolling analysis uses Annualized Return instead of Cumulative Return for better scale consistency
        
        **Rationale:**
        - **Multiple metrics**: These metrics capture different aspects of strategy performance
        - **Risk-adjustment**: Sharpe and Sortino ratios account for volatility and downside risk
        - **Comparability**: Same metrics across all time periods enable direct comparison
        
        ### Step 3: Distribution-Aware Iota Calculation
        **What happens:**
        The system automatically detects the distribution characteristics of your data and chooses the most appropriate calculation method:
        
        **Standard Method** (for normal distributions):
        ```
        ι = weight × (OOS_metric - IS_median) / IS_std_dev
        ```
        
        **Robust Method** (for skewed or fat-tailed distributions):
        ```
        ι = weight × (OOS_metric - IS_median) / (IS_IQR / 1.35)
        ```
        
        **Percentile Method** (for complex distributions):
        ```
        ι = weight × z_score_from_percentile(OOS_percentile_rank)
        ```
        
        **Distribution Detection:**
        - **Normality test**: D'Agostino K² test for normal distribution
        - **Skewness analysis**: Detects asymmetric distributions (|skewness| > 1.0)
        - **Kurtosis analysis**: Detects fat-tailed distributions (kurtosis > 3.0)
        - **Method selection**: Uses standard method for normal distributions, robust method for skewed or fat-tailed distributions
        
        **Rationale:**
        - **Adaptive approach**: Different distribution shapes require different statistical methods
        - **Robust statistics**: IQR-based methods are less sensitive to outliers than standard deviation
        - **Percentile ranking**: Direct ranking approach for complex, non-normal distributions
        - **Automatic detection**: No manual intervention required - system adapts to your data
        - **Intuitive scale**: All methods produce comparable iota values for interpretation
        
        ### Step 4: Statistical Testing with Autocorrelation Adjustment
        **What happens:**
        1. **For overlapping slices**: Calculate first-order autocorrelation of IS values
        2. **Effective sample size**: Adjust for temporal correlation using Newey-West correction
        3. **P-value adjustment**: Scale p-values upward to account for reduced independence
        4. **Bootstrap confidence intervals**: Use block bootstrap for overlapping data, standard bootstrap for non-overlapping
        
        **Mathematical detail:**
        ```
        effective_n = n × (1 - autocorr) / (1 + autocorr)
        adjustment_factor = √(effective_n / n)
        p_value_adjusted = min(1.0, p_value_raw / adjustment_factor)
        ```
        
        ### Step 5: Distribution Analysis
        **What happens:**
        1. **Histogram creation**: In-sample distributions plotted for each metric
        2. **OOS value marking**: Red dashed line shows where OOS performance falls
        3. **Median reference**: Blue dashed line shows in-sample median
        4. **Visual comparison**: Easy to see OOS performance relative to historical distribution
        
        **Rationale:**
        - **Visual clarity**: Histograms make distribution shape and OOS position clear
        - **Intuitive interpretation**: Left of distribution = underperforming, right = outperforming
        - **Median reference**: Shows expected performance level
        - **Multi-metric view**: All four metrics displayed simultaneously
        
        ### Step 6: Rolling Window Analysis
        **What happens:**
        1. **OOS Rolling Analysis**: OOS period divided into overlapping windows (e.g., 6-month windows with 1-month steps)
        2. **Historical comparison**: Each window compared against IS slice distribution
        3. **Time-specific analysis**: Performance patterns and trends over time
        4. **Full Backtest Rolling Analysis**: 252-day rolling windows across entire backtest period
        5. **Metric consistency**: Uses Annualized Return instead of Cumulative Return for scale consistency
        
        **Rationale:**
        - **Performance tracking**: Shows how strategy performance evolves over time
        - **Temporal granularity**: Rolling windows reveal when and how performance changes
        - **Pattern identification**: Helps identify consistent vs. volatile performance
        - **Proxy iota analysis**: Full backtest analysis provides context for pre-OOS performance patterns
        - **Scale consistency**: Annualized returns provide better comparison across different time periods
        
        ## CORE METRICS ANALYZED
        
        1. **ANNUALIZED RETURN**: Yearly return percentage (CAGR) - **Used in rolling analysis for scale consistency**
        2. **SHARPE RATIO**: Risk-adjusted return measure (return per unit of total volatility)
        3. **CUMULATIVE RETURN**: Total return over the entire period - **Used in core analysis and distributions**
        4. **SORTINO RATIO**: Downside risk-adjusted return (return per unit of downside volatility)
        
        **Note**: All financial metrics are calculated using quantstats when available, providing professional-grade calculations with industry-standard methodologies. When quantstats is not available, the app falls back to internal calculations.
        
        ## AUTOCORRELATION ADJUSTMENT
        
        When using overlapping slices, the temporal correlation between adjacent slices reduces the effective sample size and can lead to overly optimistic p-values.
        
        **This calculator automatically:**
        1. Detects overlapping slice configurations
        2. Calculates the first-order autocorrelation of IS slice metrics
        3. Adjusts the effective sample size using Newey-West type correction
        4. Provides more conservative, statistically valid p-values
        
        **The adjustment factor** is reported for transparency, typically ranging from 0.3-1.0:
        - `1.0` = No adjustment (non-overlapping slices)
        - `0.7` = Moderate positive autocorrelation typical of overlapping financial data
        - `0.3` = High positive autocorrelation requiring significant adjustment
        """)
    
    with help_tab3:
        st.subheader("📊 Interpretation Guide")
        
        st.markdown("""
        ## Understanding Iota Values
        
        | Iota Range | Rating Range | Interpretation |
        |------------|--------------|----------------|
        | **ι ≥ +2.0** | ~270+ | 🔥 **EXCEPTIONAL**: Significantly above expectations |
        | **ι ≥ +1.0** | ~165+ | ✅ **STRONG**: Well above expectations |
        | **ι ≥ +0.5** | ~128+ | 👍 **GOOD**: Above expectations |
        | **ι ≥ +0.25** | ~113+ | 📈 **SLIGHT_IMPROVEMENT** |
        | **-0.25 ≤ ι ≤ +0.25** | 88-113 | ➡️ **OOS closely matches backtest** |
        | **ι ≤ -0.25** | ~88- | ⚠️ **CAUTION**: Below expectations |
        | **ι ≤ -0.5** | ~78- | 🚨 **WARNING**: Below expectations |
        | **ι ≤ -1.0** | ~60- | 🔴 **ALERT**: Significantly below expectations |
        | **ι ≤ -2.0** | ~36- | 💀 **CRITICAL**: Severely below expectations |
        
        ## Persistence Ratings Explained
        
        The **Persistence Rating** converts iota (ι) into a more intuitive 0–500 point scale using:
        
        **Formula:** `Rating = 100 × exp(0.5 × ι)`
        
        ### Key Insights:
        - **100** = Neutral baseline (matches historical expectations)
        - **>130** = Meaningful outperformance
        - **<80** = Concerning underperformance
        - **>200** = Exceptional performance (rare)
        - **<50** = Critical underperformance
        
        ### Why Use Ratings?
        - **Non-technical summary** of statistical analysis
        - **Cross-strategy comparisons** easier (Rating 170 vs. Rating 90)
        - **Intuitive interpretation** without understanding standard deviations
        

        
        ## Sample Reliability Assessment
        
        | Min Sample Size | Reliability | Interpretation |
        |-----------------|-------------|----------------|
        | **≥378 days** | HIGH_CONFIDENCE | ~1.5 years - strong statistical power |
        | **≥189 days** | MODERATE_CONFIDENCE | ~9 months - reasonable statistical power |
        | **≥90 days** | LOW_CONFIDENCE | ~4.5 months - limited but usable |
        | **<90 days** | INSUFFICIENT_DATA | <90 days - insufficient for reliable statistics |
        
        ## Example Interpretation
        
        **Scenario**: Your strategy backtest showed 25% annual returns with Sharpe ratio of 1.8. After 6 months of live trading, you're getting -5% returns with Sharpe ratio of -0.3.
        
        **What Iota Analysis Shows**:
        1. Looks at 100 historical 6-month periods from your backtest
        2. Finds your strategy typically got 8% to 18% returns with Sharpe ratios of 0.8 to 2.2
        3. Calculates that 5% returns and -0.3 Sharpe are far below historical expectations (Iota ≈ -2.1)
        4. **Conclusion**: "Your strategy is underperforming relative to backtest expectations - likely overfit"
        
        **VS. if you got 12% returns with Sharpe 1.5**:
        1. Same historical analysis
        2. 12% returns and 1.5 Sharpe fall within your historical range (Iota ≈ +0.2)
        3. **Conclusion**: "Your strategy is performing as expected based on backtest"
        
        **Another Example - Extreme Overfitting**:
        
        **Scenario**: Your strategy backtest showed 890% annual returns with Sharpe ratio of 4.5. After 3 months of live trading, you're getting 80% annual returns with Sharpe ratio of 1.5.
        
        **What Iota Analysis Shows**:
        1. Looks at 100 historical 3-month periods from your backtest
        2. Finds your strategy typically got 200% to 400% annual returns with Sharpe ratios of 3.0 to 5.0
        3. Calculates that 80% returns and 1.5 Sharpe are far below historical expectations (Iota ≈ -1.8)
        4. **Conclusion**: "Your strategy is significantly underperforming relative to backtest - likely overfit to specific market conditions". In this scenario, the difference is so large between backtest and out of sample performance that the strategy may carry a larger degree of inherent risk.
        
        ## ⚠️ Limitations and Assumptions
        
        ### Distribution-Aware Analysis
        The iota calculation automatically adapts to your data's distribution shape:
        
        - **Standard Method**: Used for normal distributions (traditional mean/std approach)
        - **Robust Method**: Used for skewed or fat-tailed distributions (IQR-based, less sensitive to outliers)
        - **Percentile Method**: Used for complex distributions (percentile-based ranking)
        
        The system automatically detects distribution characteristics and chooses the most appropriate calculation method.
        
        ### Other Limitations
        
        **Market Regime Changes**: The analysis assumes market conditions during your backtest are representative of future conditions. Significant regime changes (e.g., from low to high volatility) can invalidate historical comparisons.
        
        **Strategy Evolution**: If you've modified your strategy between backtest and live trading, the comparison becomes less meaningful.
                
        **Time Period Sensitivity**: Results can vary significantly based on your chosen in-sample and out-of-sample periods. Consider testing multiple time splits.
        
        ### When to Be Cautious
        
        - **Extreme iota values** (|ι| > 3.0) may indicate data issues or regime changes rather than overfitting
        - **Inconsistent results** across different metrics suggest the analysis may not be applicable
        - **Small out-of-sample periods** (< 6 months) provide limited statistical power
        - **Highly leveraged strategies** may have non-normal risk distributions
        
        ### Best Practices
        
        1. **Trust the distribution detection**: The system automatically chooses the best calculation method for your data
        2. **Use multiple time periods**: Test different in-sample/out-of-sample splits
        3. **Consider market context**: Account for changing market conditions
        4. **Combine with other tools**: Don't rely solely on iota analysis for strategy evaluation
        5. **Monitor over time**: Track iota values as your out-of-sample period grows
        """)
    
    with help_tab4:
        st.subheader("📈 Distribution Analysis Guide")
        
        st.markdown("""
        ## What is Distribution Analysis?
        
        Distribution analysis visualizes the historical in-sample performance distributions for each metric, with your out-of-sample values marked for comparison.
        
        ### Purpose
        - **Visual Comparison**: See where your OOS performance falls relative to historical patterns
        - **Distribution Shape**: Understand the range and variability of historical performance
        - **Intuitive Interpretation**: Left of distribution = underperforming, right = outperforming
        - **Multi-Metric View**: Compare all four metrics simultaneously
        
        ## Understanding the Distribution Charts
        
        ### 📊 Chart Elements
        
        **Histogram Bars**: Show the frequency of different performance levels during the backtest period
        - **Height**: How often that performance level occurred
        - **Width**: Performance range for that bin
        - **Color**: Each metric has its own color for easy identification
        
        **Red Dashed Line**: Your out-of-sample performance value
        - **Position**: Shows exactly where your OOS performance falls
        - **Relative to distribution**: Left = underperforming, right = outperforming
        
        **Blue Dashed Line**: In-sample median (expected performance)
        - **Reference point**: Shows the "typical" performance level
        - **Comparison**: How far your OOS value is from the median
        
        ### 📈 Metric-Specific Interpretations
        
        **Annualized Return**:
        - **Left of distribution**: OOS returns below historical expectations
        - **Right of distribution**: OOS returns above historical expectations
        - **Units**: Percentage points (e.g., 15.2% = 15.2 percentage points)
        
        **Sharpe Ratio**:
        - **Left of distribution**: OOS risk-adjusted returns below historical expectations
        - **Right of distribution**: OOS risk-adjusted returns above historical expectations
        - **Units**: Risk-adjusted return ratio (dimensionless)
        
        **Cumulative Return**:
        - **Left of distribution**: OOS total returns below historical expectations
        - **Right of distribution**: OOS total returns above historical expectations
        - **Units**: Percentage points (e.g., 25.8% = 25.8 percentage points)
        
        **Sortino Ratio**:
        - **Left of distribution**: OOS downside risk management below historical expectations
        - **Right of distribution**: OOS downside risk management above historical expectations
        - **Units**: Downside risk-adjusted return ratio (dimensionless)
        
        ## Interpreting Distribution Results
        
        ### ✅ Healthy Patterns
        - **OOS values near median**: Performance matches historical expectations
        - **OOS values within distribution**: Performance is within normal historical range
        - **Consistent across metrics**: All metrics show similar relative performance
        
        ### ⚠️ Concerning Patterns
        - **OOS values far left of distribution**: Significant underperformance
        - **OOS values outside distribution**: Performance outside historical range
        - **Inconsistent across metrics**: Some metrics performing well, others poorly
        
        ### 🚨 Critical Patterns
        - **OOS values at extreme left**: Severe underperformance
        - **Multiple metrics showing poor performance**: Systematic issues
        - **OOS values well outside distribution**: Unusual performance requiring investigation
        
        ## Using Distribution Analysis
        
        ### 🎯 Quick Assessment
        1. **Look at red lines**: Where do they fall relative to the histograms?
        2. **Compare to blue lines**: How far from the median?
        3. **Check consistency**: Are all metrics telling the same story?
        4. **Consider magnitude**: How far outside the distribution?
        
        ### 📊 Detailed Analysis
        - **Distribution shape**: Wide distributions suggest high variability
        - **Skewness**: Asymmetric distributions indicate bias
        - **Outliers**: Extreme values in historical data
        - **OOS position**: Percentile rank within historical distribution
        
        ## Best Practices
        
        ### ✅ What to Look For
        - **Consistent positioning**: All metrics showing similar relative performance
        - **Reasonable distance**: OOS values not too far from median
        - **Distribution coverage**: OOS values within historical range
        
        ### ⚠️ Warning Signs
        - **Extreme positioning**: OOS values at distribution edges
        - **Inconsistent patterns**: Different metrics showing opposite results
        - **Outside distribution**: OOS values beyond historical range
        
        ### 🔍 Follow-up Actions
        - **If concerning**: Run rolling analysis for temporal patterns
        - **If critical**: Review strategy parameters and market conditions
        - **If unusual**: Investigate specific time periods or market events
        """)
    
    with help_tab5:
        st.subheader("🔄 Rolling Window Analysis Guide")
        
        st.markdown("""
        ## What is Rolling Window Analysis?
        
        Rolling window analysis divides your out-of-sample period into multiple overlapping time windows to detect overfitting patterns and performance persistence/degradation over time.
        
        ## How It Works
        
        ### Step 1: Window Creation
        - Your OOS period is divided into overlapping windows (e.g., 6-month windows)
        - Windows advance by smaller steps (e.g., 1-month steps)
        - Creates multiple "mini out-of-sample" periods for analysis
        
        ### Step 2: Historical Comparison
        - Each window is compared against the same IS slice distribution
        - Iota calculated for each window vs. historical expectations
        - Tracks how each metric performs over time
        
        ### Step 3: Performance Tracking
        - Iota values calculated for each time window
        - Tracks how performance compares to backtest expectations over time
        - Multiple metrics analyzed independently
        

        
        ## Interpreting Rolling Analysis Results
        
        ### 📈 Understanding the Rolling Plot
        
        **Key Elements:**
        - **Gray line at ι = 0**: Neutral performance (matches historical median)
        - **Green dotted line at ι = +0.5**: Good performance threshold
        - **Red dotted line at ι = -0.5**: Poor performance threshold
        - **Colored lines**: Individual metrics (Sharpe, Annualized Return, Sortino)
        - **Smoothing**: 3-period moving average reduces noise
        
        **Performance Patterns:**
        - **Stable Performance**: Iotas fluctuate around zero with minimal variance
        - **Trending Performance**: Consistent upward or downward movement in iota values
        - **Volatile Performance**: High variance in rolling iota values
        - **Metric Divergence**: Different metrics showing different patterns
        
        ### 🔍 Metric-Specific Performance
        
        **Individual metric patterns indicate:**
        - **Sharpe Ratio**: Risk-adjusted performance trends over time
        - **Annualized Return**: Yearly return performance trends over time
        - **Sortino Ratio**: Downside risk management consistency
        
        ## Rolling Analysis Insights
        
        ### 📊 Time-Based Analysis
        - **Performance consistency**: How stable iota values are over time
        - **Trend identification**: Whether performance is improving or declining
        - **Volatility assessment**: How much performance varies between periods
        
        ### 📈 Full Backtest Rolling Iota Proxy Analysis
        **Understanding the "Proxy" Nature:**
        
        **What is Iota?** Iota (ι) is a metric that compares out-of-sample (OOS) performance against in-sample (IS) expectations. By definition, iota only applies to true OOS data.
        
        **Why "Proxy" Iota?** For periods before your OOS start date, we calculate a "proxy iota" by:
        - Treating each 252-day rolling window as a temporary "OOS" period
        - Comparing that window's performance against the full IS distribution
        - This provides insights into how your strategy performed relative to expectations throughout the backtest
        
        **Key Benefits:**
        - **Performance Evolution**: See how strategy performance changed over time
        - **Consistency Assessment**: Identify periods of stable vs. volatile performance
        - **OOS Context**: Compare actual OOS performance with pre-OOS patterns
        - **Early Warning**: Detect performance changes before your OOS period
        
        **Interpretation:**
        - **Pre-OOS periods**: Show "proxy iota" - how each window performed vs. full IS distribution
        - **OOS period**: Shows true iota - actual OOS performance vs. IS expectations
        - **Patterns**: Look for consistency, trends, or sudden changes in performance
        
        ## Technical Parameters
        
        ### 🔧 Window Sizing (Adaptive)
        - **2+ years OOS**: 126-day windows (6 months)
        - **1-2 years OOS**: 84-day windows (4 months)
        - **9+ months OOS**: 63-day windows (3 months)
        - **3-9 months OOS**: Adaptive sizing (minimum 21 days)
        
        ### 📏 Step Sizing
        - **Default**: Window size ÷ 8 (e.g., 126÷8 = ~16 days)
        - **Minimum**: 5 days
        - **Purpose**: Balance between granularity and computational efficiency
        
        ### 🎯 Minimum Requirements
        - **Minimum windows**: 6 for meaningful trend analysis
        - **Maximum windows**: 60 (performance limitation)
        - **Minimum OOS period**: 90 days for any rolling analysis
        """)
    
    with help_tab6:
        st.subheader("❓ Frequently Asked Questions")
        
        st.markdown("""
        ## General Questions
        
        ### Q: What makes this different from just looking at returns?
        **A:** This tool provides **statistical context**. Getting 30% returns is great, but if your strategy had a backtest suggesting 300%, that's not good. However, if the backtest consistently got 20-40%, then 30% means your strategy doesn't look overfit. That's good performance.
        
        ### Q: Can this predict future performance?
        **A:** **No.** This is a **retrospective analysis tool**. It tells you how unusual your recent performance has been relative to history, but cannot predict what will happen next.
        
        ### Q: Why do I need both core analysis AND rolling analysis?
        **A:** 
        - **Core analysis**: Overall assessment of your entire OOS period
        - **Distribution analysis**: Visualize in-sample distributions with OOS values marked
        - **Rolling analysis**: Detects **when** and **how** performance changes over time
        - **Together**: Complete picture of strategy health and performance patterns
        
        ## Interpretation Questions
        
        ### Q: What's a "good" iota score?
        **A:** 
        - **ι > +0.5**: Strong outperformance
        - **ι ≈ 0**: Performing as expected (this is actually good!)
        - **ι < -0.5**: Concerning underperformance
        - **Remember**: ι = 0 means your strategy is working exactly as the backtest suggested
        
        ### Q: What does "autocorrelation adjusted" mean?
        **A:** When using overlapping slices, adjacent periods share most of their data, violating statistical independence assumptions. The adjustment makes p-values more conservative (harder to achieve significance) to account for this correlation.
        
        ### Q: What is "proxy iota" in the full backtest rolling analysis?
        **A:** Since iota technically only applies to true out-of-sample data, we use "proxy iota" for pre-OOS periods. Each 252-day rolling window is treated as a temporary "OOS" period and compared against the full in-sample distribution. This shows how your strategy performed relative to expectations throughout the backtest, providing context for your actual OOS performance.
        
        ### Q: Why does rolling analysis use Annualized Return instead of Cumulative Return?
        **A:** Rolling analysis uses Annualized Return for scale consistency. Cumulative returns can vary dramatically based on the time period length, making comparisons difficult. Annualized returns normalize performance to a yearly basis, providing more consistent and interpretable comparisons across different rolling windows and time periods.
        
        ### Q: What is "Distribution-Aware Analysis" and how does it work?
        **A:** The system automatically detects the shape of your strategy's performance distribution and chooses the most appropriate statistical method:
        
        - **Standard Method**: Used for normal distributions (traditional mean/std approach)
        - **Robust Method**: Used for skewed or fat-tailed distributions (IQR-based, less sensitive to outliers)
        - **Percentile Method**: Used for complex distributions (percentile-based ranking)
        
        The system runs normality tests, analyzes skewness and kurtosis, and automatically selects the best calculation method for your data. This ensures more accurate iota values regardless of your strategy's distribution characteristics.
        
        **Note**: Core analysis uses distribution-aware methods, while rolling analysis uses the standard method for consistency across time periods.
        
        ## Technical Questions
        
        ### Q: How many IS slices should I use?
        **A:**
        - **50-100**: Good balance of statistics and speed
        - **100-200**: Better statistics, slower computation
        - **More isn't always better**: Diminishing returns beyond ~200 slices
        
        ### Q: Should I use overlapping or non-overlapping slices?
        **A:**
        - **Overlapping (recommended)**: More data, better statistics, uses autocorrelation adjustment
        - **Non-overlapping**: Simpler statistics, less data, faster computation
        
        ### Q: How long should my OOS period be?
        **A:**
        - **Minimum**: 3 months (90 days)
        - **Recommended**: 6-12 months for meaningful analysis
        - **Ideal**: 1-2 years for high confidence
        
        ## Common Issues
        
        ### Q: "Insufficient data" error - what do I do?
        **A:** 
        - Extend your data start date further back
        - Reduce the number of IS slices
        - Ensure your OOS period is at least 90 days
        - Check that your symphony has enough historical data
        
        ### Q: Analysis is very slow - how to speed up?
        **A:**
        - Reduce number of IS slices (try 50 instead of 100)
        - Use non-overlapping slices
        - Disable rolling analysis for faster core analysis only
        - Use shorter date ranges for initial testing
        
        ### Q: All my iotas are near zero - is this bad?
        **A:** **No!** Iotas near zero mean your strategy is performing **exactly as expected** based on historical patterns. This is actually **good** - it means your backtest was realistic and your strategy is working as designed.
        
        ### Q: Rolling analysis shows concerning patterns - what now?
        **A:**
        - Don't throw money at obscenely overfit strategies.
        - If you're not sure, get a second opinion.
        - Maybe consider a different strategy. A strategy with a more realistic but less impressive backtest that actually performs as intended is infinitely more valuable than one which promises 1000% annualized returns but can't even achieve 1/10th of that with any degree of certainty.
        
        ## Best Practices
        
        
        ### Q: What exclusion periods should I use?
        **A:** Consider excluding:
        - **Market crashes** (2020 COVID crash, 2008 financial crisis) - strategies are often overfit to these periods.
        - **Extreme volatility periods** that aren't representative
        - **Data quality issues** (corporate actions, splits, etc.)
        - **Be conservative**: Only exclude truly exceptional periods
        
        ### Q: My strategy looks good in core analysis but bad in rolling analysis - which to trust?
        **A:** 
        - **Both are important** - they tell different stories
        - **Core analysis**: Overall performance vs. expectations
        - **Rolling analysis**: Performance consistency over time
        
        ## Troubleshooting
        
        ### Q: What are confidence intervals and how do I interpret them?
        **A:** 
        
        **95% Confidence Intervals** show the range where the true iota value likely falls:
        
        - **Narrow intervals** (e.g., [-0.2, +0.3]): High confidence in the iota estimate
        - **Wide intervals** (e.g., [-1.5, +2.0]): High uncertainty in the estimate
        - **Interval includes 0**: No strong evidence that performance differs from expectations
        - **Interval excludes 0**: Evidence that performance differs from expectations
        
        **Example**: If your iota is +0.8 with 95% CI [-0.1, +1.7]:
        - The interval includes 0, so we can't be confident that performance differs from expectations
        - The wide range suggests high uncertainty in the estimate
        
        **Wide intervals usually mean**:
        - **Need more data** - longer OOS period or more IS slices
        - **High volatility** in your strategy's historical performance
        - **Results less reliable** - interpret with caution
        - **Consider extending** analysis period
        

        ### Q: My symphony stopped working mid-analysis - what happened?
        **A:**
        - **Composer API limits**: Try again in a few minutes
        - **Symphony privacy settings**: Ensure symphony is public or accessible
        - **Network issues**: Check internet connection
        - **Data corruption**: Try with a different date range
        
        ## Contact & Support
        
        ### Q: I found a bug or have a feature request - where do I report it?
        **A:** Reach out to @gobi on discord and I'd be happy to chat.
        
        ### Q: Can I use this for non-Composer strategies?
        **A:** The tool is designed for Composer symphonies, but the statistical methodology can be adapted for any return series with appropriate modifications to the data input module.
        
        ### Q: What is quantstats and why does the app use it?
        **A:** quantstats is a professional Python library for quantitative analysis of financial portfolios. The Iota Calculator uses quantstats when available to provide:
        
        - **Industry-standard calculations**: Uses widely-accepted methodologies for financial metrics
        - **Enhanced accuracy**: Professional-grade implementations of Sharpe, Sortino, and other ratios
        - **Additional metrics**: Access to beta, alpha, information ratio, and other advanced metrics
        - **Robust error handling**: Better handling of edge cases and data quality issues
        
        The app automatically detects if quantstats is installed and uses it for calculations. If not available, it falls back to internal calculations that provide the same core functionality.
        
        **Installation**: `pip install quantstats`
        """)
    
    # Add footer with version info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 2rem;'>
        <p><strong>Iota Calculator - Core and Rolling Analysis</strong></p>
        <p>Edge persistence and performance analysis tool</p>
        <p>Questions? Reach out to @gobi on Discord</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
