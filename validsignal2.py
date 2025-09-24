import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# -----------------------------
# Expanded Preconditions Utils
# -----------------------------
def sma(close: pd.Series, length: int) -> pd.Series:
    return close.rolling(int(length), min_periods=max(3, int(length*0.6))).mean()

def ema(close: pd.Series, length: int) -> pd.Series:
    return close.ewm(span=int(length), adjust=False, min_periods=max(3, int(length*0.6))).mean()

def _cmp_series(a: pd.Series, b: pd.Series, op: str) -> pd.Series:
    op = (op or "greater_than").strip().lower()
    if op == "less_than":      return a <  b
    if op == "greater_than":   return a >  b
    return a > b  # default

def _tz_naive(s: pd.Series) -> pd.Series:
    if hasattr(s.index, "tz") and s.index.tz is not None:
        s = s.copy()
        s.index = s.index.tz_convert(None)
    return s

def _price_series(ticker: str, start_date=None, end_date=None, exclusions=None) -> pd.Series:
    """Wrapper over get_stock_data -> returns 'close' series w/ tz-naive index."""
    s = get_stock_data(ticker, start_date, end_date, exclusions)
    if s is None or s.empty:
        return pd.Series(dtype=float)
    s = s.rename("close")
    s = _tz_naive(s)
    return s

def build_precondition_mask(
    base_index: pd.DatetimeIndex,
    preconditions: List[Dict],
    start_date=None,
    end_date=None,
    rsi_len: int = 14,
    exclusions: Optional[List[Dict]] = None,
) -> pd.Series:
    """
    Returns boolean Series indexed by base_index that is True only when ALL preconditions pass.

    Supported precondition dicts (examples):
      # Static RSI vs threshold
      {"mode":"static", "signal_ticker":"QQQ", "rsi_len":10, "comparison":"greater_than", "threshold":80.0}

      # Pair RSI vs RSI
      {"mode":"pair", "lhs_ticker":"KMLM", "lhs_len":10, "op":"less_than",
                        "rhs_ticker":"XLK",  "rhs_len":10}

      # Price/MA/EMA variants
      {"mode":"price_vs_ma",  "lhs_ticker":"SPY", "lhs_len":1,  "op":"greater_equal",
                               "rhs_ticker":"SPY", "rhs_len":200}
      {"mode":"price_vs_ema", "lhs_ticker":"QQQ", "lhs_len":1,  "op":"less_than",
                               "rhs_ticker":"QQQ", "rhs_len":50}
      {"mode":"ma_vs_ma",     "lhs_ticker":"SPY", "lhs_len":20, "op":"greater_than",
                               "rhs_ticker":"SPY", "rhs_len":50}
      {"mode":"ema_vs_ema",   "lhs_ticker":"IWM", "lhs_len":12, "op":"less_equal",
                               "rhs_ticker":"IWM", "rhs_len":26}
    """
    if not preconditions:
        return pd.Series(True, index=base_index, dtype=bool)

    mask = pd.Series(True, index=base_index, dtype=bool)

    for p in preconditions:
        mode = (p.get("mode") or "static").strip().lower()

        if mode == "pair":
            lhs_tkr = p.get("lhs_ticker","").strip().upper()
            rhs_tkr = p.get("rhs_ticker","").strip().upper()
            lhs_len = int(p.get("lhs_len", rsi_len))
            rhs_len = int(p.get("rhs_len", rsi_len))
            op      = p.get("op","less_than")

            lhs = _price_series(lhs_tkr, start_date, end_date, exclusions)
            rhs = _price_series(rhs_tkr, start_date, end_date, exclusions)
            if lhs.empty or rhs.empty:
                cond = pd.Series(False, index=base_index)
            else:
                lhs_rsi = calculate_rsi(lhs, window=lhs_len, method="wilders")
                rhs_rsi = calculate_rsi(rhs, window=rhs_len, method="wilders")
                cond = _cmp_series(lhs_rsi, rhs_rsi, op).reindex(base_index).fillna(False)

            mask &= cond.astype(bool)

        elif mode in ("price_vs_ma", "price_vs_ema", "ma_vs_ma", "ema_vs_ema"):
            lhs_tkr = p.get("lhs_ticker","").strip().upper()
            rhs_tkr = p.get("rhs_ticker","").strip().upper()
            lhs_len = int(p.get("lhs_len", 1))
            rhs_len = int(p.get("rhs_len", 50))
            op      = p.get("op","greater_than")

            lhs_close = _price_series(lhs_tkr, start_date, end_date, exclusions)
            rhs_close = _price_series(rhs_tkr, start_date, end_date, exclusions)
            if lhs_close.empty or rhs_close.empty:
                cond = pd.Series(False, index=base_index)
            else:
                if mode in ("price_vs_ma","price_vs_ema"):
                    lhs_series = lhs_close  # price
                else:
                    lhs_series = sma(lhs_close, lhs_len) if mode == "ma_vs_ma" else ema(lhs_close, lhs_len)

                if   mode == "price_vs_ma":  rhs_series = sma(rhs_close, rhs_len)
                elif mode == "price_vs_ema": rhs_series = ema(rhs_close, rhs_len)
                elif mode == "ma_vs_ma":     rhs_series = sma(rhs_close, rhs_len)
                else:                        rhs_series = ema(rhs_close, rhs_len)

                cond = _cmp_series(lhs_series, rhs_series, op).reindex(base_index).fillna(False)

            mask &= cond.astype(bool)

        else:
            # static RSI vs absolute threshold
            tkr = p.get("signal_ticker", "").strip().upper()
            cmp_key = p.get("comparison","greater_than")
            thr = float(p.get("threshold", 50.0))
            this_len = int(p.get("rsi_len", rsi_len))

            s = _price_series(tkr, start_date, end_date, exclusions)
            if s.empty:
                cond = pd.Series(False, index=base_index)
            else:
                rsi = calculate_rsi(s, window=this_len, method="wilders")
                cond = _cmp_series(rsi, pd.Series(thr, index=rsi.index), cmp_key).reindex(base_index).fillna(False)

            mask &= cond.astype(bool)

    # Ensure correct shape/dtype
    mask = mask.reindex(base_index).fillna(False).astype(bool)
    return mask

# Try to import QuantStats with error handling
try:
    import quantstats as qs
    # Configure QuantStats
    qs.extend_pandas()
    QUANTSTATS_AVAILABLE = True
    st.success("✅ QuantStats loaded successfully!")
except ImportError as e:
    st.warning(f"⚠️ QuantStats import error: {str(e)}. Install with: pip install quantstats>=0.0.62")
    QUANTSTATS_AVAILABLE = False
except Exception as e:
    st.warning(f"⚠️ QuantStats import failed: {str(e)}. Using fallback calculations.")
    QUANTSTATS_AVAILABLE = False

st.set_page_config(page_title="Signal Check", layout="wide")

st.title("Signal Check")
st.write("RSI Threshold Statistics")

def calculate_rsi(prices: pd.Series, window: int = 14, method: str = "wilders") -> pd.Series:
    """Calculate RSI using specified method (Wilder's smoothing or simple moving average)"""
    if len(prices) < window + 1:
        return pd.Series(index=prices.index, dtype=float)
    
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    if method == "wilders":
        # Wilder's smoothing: use exponential moving average with alpha = 1/window
        alpha = 1.0 / window
        
        # Calculate smoothed average gains and losses
        avg_gains = gains.ewm(alpha=alpha, adjust=False).mean()
        avg_losses = losses.ewm(alpha=alpha, adjust=False).mean()
    else:
        # Simple moving average method
        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()
    
    # Calculate relative strength
    rs = avg_gains / avg_losses
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02, use_quantstats: bool = True) -> float:
    """Calculate Sortino ratio using QuantStats or fallback"""
    if len(returns) == 0:
        return 0
    
    # Convert to pandas Series for QuantStats
    returns_series = pd.Series(returns)
    
    if QUANTSTATS_AVAILABLE and use_quantstats:
        try:
            # Use QuantStats sortino ratio calculation
            sortino_ratio = qs.stats.sortino(returns_series, rf=risk_free_rate)
            return sortino_ratio if not np.isnan(sortino_ratio) else 0
        except Exception:
            pass  # Fall through to original calculation
    
    # Fallback to original calculation
    rf_per_trade = risk_free_rate / 252
    excess_returns = returns - rf_per_trade
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if excess_returns.mean() > 0 else 0
    
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    
    if downside_deviation == 0:
        return np.inf if excess_returns.mean() > 0 else 0
    
    return excess_returns.mean() / downside_deviation

@st.cache_data(show_spinner=False)
def get_stock_data_cached(ticker: str, start_date=None, end_date=None) -> pd.Series:
    """Cached version of stock data fetching (adjusted for splits/dividends)."""
    tkr = yf.Ticker(ticker.upper().strip())
    # auto_adjust=True returns an adjusted 'Close' (already adjusted for splits/dividends)
    if start_date and end_date:
        data = tkr.history(start=start_date, end=end_date, auto_adjust=True)
    else:
        data = tkr.history(period="max", auto_adjust=True)
    s = data['Close'].copy()
    s.name = 'Adj Close'  # standardize the name for downstream use
    return s

def get_stock_data(ticker: str, start_date=None, end_date=None, exclusions=None) -> pd.Series:
    """Fetch adjusted price series with optional exclusions."""
    try:
        data = get_stock_data_cached(ticker, start_date, end_date)
        if data is None or data.empty:
            st.error(f"No data found for ticker: {ticker}")
            return None

        # Keep the same column name as returned ('Adj Close') to avoid KeyErrors
        data_df = pd.DataFrame({'Adj Close': data})

        if exclusions:
            for exclusion in exclusions:
                exclusion_start = pd.Timestamp(exclusion['start'])
                exclusion_end = pd.Timestamp(exclusion['end'])
                data_df = data_df[~((data_df.index >= exclusion_start) & (data_df.index <= exclusion_end))]

        return data_df['Adj Close']
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def analyze_rsi_signals(signal_prices: pd.Series, target_prices: pd.Series, rsi_threshold: float, comparison: str = "less_than", rsi_period: int = 14, rsi_method: str = "wilders", use_quantstats: bool = True, preconditions: List[Dict] = None) -> Dict:
    """Analyze RSI signals for a specific threshold with optional preconditions"""
    # Calculate RSI for the SIGNAL ticker using specified period and method
    signal_rsi = calculate_rsi(signal_prices, window=rsi_period, method=rsi_method)
    
    # Generate buy signals based on SIGNAL RSI threshold and comparison
    if comparison == "less_than":
        # "≤" configuration: Buy TARGET when SIGNAL RSI ≤ threshold, sell when SIGNAL RSI < threshold
        base_signals = (signal_rsi <= rsi_threshold).astype(int)
    else:  # greater_than
        # "≥" configuration: Buy TARGET when SIGNAL RSI ≥ threshold, sell when SIGNAL RSI < threshold
        base_signals = (signal_rsi >= rsi_threshold).astype(int)
    
    # Apply preconditions (expanded set) if provided
    if preconditions:
        pc_mask = build_precondition_mask(
            base_index=signal_prices.index,
            preconditions=preconditions,
            start_date=signal_prices.index.min(),
            end_date=signal_prices.index.max(),
            rsi_len=rsi_period,
        )
        signals = (base_signals.astype(bool) & pc_mask.astype(bool)).fillna(False).astype(int)
    else:
        signals = base_signals.fillna(0).astype(int)
    
    # Calculate equity curve day by day - buy/sell TARGET based on SIGNAL RSI
    equity_curve = pd.Series(1.0, index=target_prices.index)
    current_equity = 1.0
    in_position = False
    entry_equity = 1.0
    entry_date = None
    entry_price = None
    trades = []
    
    for i, date in enumerate(target_prices.index):
        current_signal = signals[date] if date in signals.index else 0
        current_price = target_prices[date]  # TARGET price
        
        if current_signal == 1 and not in_position:
            # Enter position - buy TARGET at close when SIGNAL RSI meets condition
            in_position = True
            entry_equity = current_equity
            entry_date = date
            entry_price = current_price
            
        elif current_signal == 0 and in_position:
            # Exit position - sell TARGET at close when SIGNAL RSI no longer meets condition
            trade_return = (current_price - entry_price) / entry_price
            current_equity = entry_equity * (1 + trade_return)
            
            hold_days = (date - entry_date).days
            trades.append({
                'entry_date': entry_date,
                'exit_date': date,
                'entry_price': entry_price,
                'exit_price': current_price,
                'return': trade_return,
                'hold_days': hold_days
            })
            
            in_position = False
        
        # Update equity curve
        if in_position:
            # Mark-to-market the TARGET position
            current_equity = entry_equity * (current_price / entry_price)
        
        equity_curve[date] = current_equity
    
    # Handle case where we're still in position at the end
    if in_position:
        final_price = target_prices.iloc[-1]
        final_date = target_prices.index[-1]
        trade_return = (final_price - entry_price) / entry_price
        current_equity = entry_equity * (1 + trade_return)
        
        hold_days = (final_date - entry_date).days
        trades.append({
            'entry_date': entry_date,
            'exit_date': final_date,
            'entry_price': entry_price,
            'exit_price': final_price,
            'return': trade_return,
            'hold_days': hold_days
        })
        equity_curve.iloc[-1] = current_equity
    
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'median_return': 0,
            'returns': [],
            'avg_hold_days': 0,
            'sortino_ratio': 0,
            'equity_curve': equity_curve,
            'trades': [],
            'annualized_return': 0
        }
    
    returns = np.array([trade['return'] for trade in trades])
    win_rate = (returns > 0).mean()
    avg_return = returns.mean()
    median_return = np.median(returns)
    avg_hold_days = np.mean([trade['hold_days'] for trade in trades])
    sortino_ratio = calculate_sortino_ratio(returns, use_quantstats=use_quantstats)
    
    # Calculate annualized return
    total_days = (target_prices.index[-1] - target_prices.index[0]).days
    total_return = equity_curve.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (365 / total_days) - 1 if total_days > 0 else 0
    
    return {
        'total_trades': len(returns),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'median_return': median_return,
        'returns': returns,
        'avg_hold_days': avg_hold_days,
        'sortino_ratio': sortino_ratio,
        'equity_curve': equity_curve,
        'trades': trades,
        'annualized_return': annualized_return
    }

def calculate_statistical_significance(strategy_equity_curve: pd.Series, benchmark_equity_curve: pd.Series, 
                                    strategy_annualized: float, benchmark_annualized: float) -> Dict:
    """Calculate statistical significance by comparing strategy vs SPY equity curves under same conditions"""
    
    if len(strategy_equity_curve) == 0 or len(benchmark_equity_curve) == 0:
        return {
            't_statistic': 0,
            'p_value': 1.0,
            'confidence_level': 0,
            'significant': False,
            'effect_size': 0,
            'power': 0,
            'insufficient_data': True
        }
    
    # Align the equity curves on the same dates
    common_dates = strategy_equity_curve.index.intersection(benchmark_equity_curve.index)
    if len(common_dates) < 20:  # Increased minimum data points for more reliable tests
        return {
            't_statistic': 0,
            'p_value': 1.0,
            'confidence_level': 0,
            'significant': False,
            'effect_size': 0,
            'power': 0,
            'insufficient_data': True
        }
    
    strategy_aligned = strategy_equity_curve[common_dates]
    benchmark_aligned = benchmark_equity_curve[common_dates]
    
    # Calculate daily returns for both strategies
    strategy_returns = strategy_aligned.pct_change().dropna()
    benchmark_returns = benchmark_aligned.pct_change().dropna()
    
    # Ensure we have enough data points
    if len(strategy_returns) < 20 or len(benchmark_returns) < 20:
        return {
            't_statistic': 0,
            'p_value': 1.0,
            'confidence_level': 0,
            'significant': False,
            'effect_size': 0,
            'power': 0,
            'insufficient_data': True
        }
    
    # Align returns on common index
    idx = strategy_returns.index.intersection(benchmark_returns.index)
    strat_r = strategy_returns.loc[idx]
    bench_r = benchmark_returns.loc[idx]
    n = min(len(strat_r), len(bench_r))
    
    if n < 20:
        return {
            't_statistic': 0,
            'p_value': 1.0,
            'confidence_level': 0,
            'significant': False,
            'effect_size': 0,
            'power': 0,
            'direction': 'unknown',
            'insufficient_data': True
        }
    
    # Perform two-tailed t-test
    t_stat, p_two = stats.ttest_ind(strat_r.values, bench_r.values, equal_var=False)
    diff = strat_r.mean() - bench_r.mean()
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((np.var(strat_r, ddof=1) + np.var(bench_r, ddof=1)) / 2))
    effect_size = float(diff / pooled_std) if pooled_std > 0 else 0.0
    
    # Use same one-tailed p for both directions
    one_tail_p = p_two / 2.0
    direction = "outperform" if diff > 0 else "underperform"
    significant = one_tail_p < 0.05
    confidence_level = (1 - one_tail_p) * 100
    
    # Calculate statistical power
    power = 0.8 if (n > 30 and abs(effect_size) > 0.5) else 0.5
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(one_tail_p),
        'confidence_level': float(confidence_level),
        'significant': bool(significant),
        'effect_size': float(effect_size),
        'power': float(power),
        'direction': direction,
        'insufficient_data': False
    }

def calculate_max_drawdown(equity_curve: pd.Series, use_quantstats: bool = True) -> float:
    """Calculate maximum drawdown using QuantStats or fallback"""
    if equity_curve.empty:
        return 0.0
    
    if QUANTSTATS_AVAILABLE and use_quantstats:
        try:
            # Use QuantStats max drawdown calculation
            max_dd = qs.stats.max_drawdown(equity_curve)
            return abs(max_dd) if not np.isnan(max_dd) else 0.0
        except Exception:
            pass  # Fall through to original calculation
    
    # Fallback to original calculation
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    return abs(drawdown.min())

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02, use_quantstats: bool = True) -> float:
    """Calculate Sharpe ratio using QuantStats or fallback"""
    if len(returns) == 0:
        return 0.0
    
    # Convert to pandas Series for QuantStats
    returns_series = pd.Series(returns)
    
    if QUANTSTATS_AVAILABLE and use_quantstats:
        try:
            # Use QuantStats sharpe ratio calculation
            sharpe_ratio = qs.stats.sharpe(returns_series, rf=risk_free_rate)
            return sharpe_ratio if not np.isnan(sharpe_ratio) else 0.0
        except Exception:
            pass  # Fall through to original calculation
    
    # Fallback to original calculation
    rf_per_trade = risk_free_rate / 252
    excess_returns = returns - rf_per_trade
    if np.std(excess_returns) == 0:
        return 0.0 if np.mean(excess_returns) == 0 else np.inf
    return np.mean(excess_returns) / np.std(excess_returns)

def calculate_additional_metrics(returns: np.ndarray, equity_curve: pd.Series, annual_return: float, use_quantstats: bool = True, benchmark_equity: Optional[pd.Series] = None) -> Dict:
    """Add more comprehensive risk metrics using QuantStats or fallback"""
    if len(returns) == 0 or equity_curve.empty:
        return {
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'var_95': 0.0,
            'sharpe_ratio': 0.0,
            'volatility': 0.0,
            'beta': 0.0,
            'alpha': 0.0,
            'information_ratio': 0.0
        }
    
    # Convert to pandas Series for QuantStats
    returns_series = pd.Series(returns)
    
    # Use QuantStats if available
    if QUANTSTATS_AVAILABLE and use_quantstats:
        try:
            # Use QuantStats for various metrics
            eq_ret = equity_curve.pct_change().dropna()
            max_dd = qs.stats.max_drawdown(eq_ret)
            sharpe = calculate_sharpe_ratio(returns, use_quantstats=use_quantstats)
            
            # Calculate Calmar ratio using QuantStats
            calmar_ratio = qs.stats.calmar(returns_series) if len(returns) > 0 else 0.0
            
            # Calculate Value at Risk using QuantStats
            var_95 = qs.stats.var(returns_series, 0.05) if len(returns) > 0 else 0.0
            
            # Calculate volatility using QuantStats
            volatility = qs.stats.volatility(returns_series) if len(returns) > 0 else 0.0
            
            # Calculate alpha/beta/IR vs benchmark if available
            beta = alpha = information_ratio = 0.0
            if benchmark_equity is not None:
                br = benchmark_equity.pct_change().dropna().reindex(eq_ret.index).dropna()
                er = eq_ret.reindex(br.index)
                beta = qs.stats.beta(er, br)
                alpha = qs.stats.alpha(er, br)
                information_ratio = qs.stats.information_ratio(er, br)
            
            return {
                'max_drawdown': abs(max_dd) if not np.isnan(max_dd) else 0.0,
                'calmar_ratio': calmar_ratio if not np.isnan(calmar_ratio) else (annual_return / abs(max_dd) if max_dd else 0.0),
                'var_95': var_95 if not np.isnan(var_95) else (np.percentile(returns, 5) if len(returns) > 0 else 0.0),
                'sharpe_ratio': sharpe,
                'volatility': volatility if not np.isnan(volatility) else (np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0),
                'beta': 0.0 if np.isnan(beta) else beta,
                'alpha': 0.0 if np.isnan(alpha) else alpha,
                'information_ratio': 0.0 if np.isnan(information_ratio) else information_ratio
            }
        except Exception:
            pass  # Fall through to fallback calculations
    
    # Fallback to original calculations
    max_dd = calculate_max_drawdown(equity_curve, use_quantstats)
    sharpe = calculate_sharpe_ratio(returns, use_quantstats=use_quantstats)
    
    return {
        'max_drawdown': max_dd,
        'calmar_ratio': annual_return / max_dd if max_dd > 0 else 0.0,
        'var_95': np.percentile(returns, 5) if len(returns) > 0 else 0.0,
        'sharpe_ratio': sharpe,
        'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0,
        'beta': 0.0,
        'alpha': 0.0,
        'information_ratio': 0.0
    }

def validate_data_quality(data: pd.Series, ticker: str) -> Tuple[bool, List[str]]:
    """Add data quality checks and return messages to display later"""
    messages = []
    
    if data is None or data.empty:
        st.error(f"❌ No data available for {ticker}")
        return False, messages
    
    # Check for missing data
    missing_pct = data.isnull().sum() / len(data) * 100
    if missing_pct > 5:  # More than 5% missing
        st.warning(f"⚠️ {missing_pct:.1f}% missing data detected for {ticker}")
    
    # Check for stock splits/dividends (extreme price movements)
    daily_returns = data.pct_change().dropna()
    extreme_moves = abs(daily_returns) > 0.15  # 15% daily moves
    if extreme_moves.sum() > 0:
        messages.append(f"🔍 Detected {extreme_moves.sum()} extreme price movements (>15%) for {ticker}")
    
    # Check for sufficient data
    if len(data) < 252:  # Less than 1 year
        st.warning(f"⚠️ Limited data for {ticker}: {len(data)} days (recommend at least 252 days)")
    
    return True, messages

# QuantStats report generation removed to avoid import issues
# Basic QuantStats metrics are still available in the main analysis functions

def run_rsi_analysis(signal_ticker: str, target_ticker: str, rsi_threshold: float, comparison: str, 
                    start_date=None, end_date=None, rsi_period: int = 14, rsi_method: str = "wilders", benchmark_ticker: str = "SPY", use_quantstats: bool = True, preconditions: List[Dict] = None, exclusions: List[Dict] = None, rsi_min: Optional[float] = None, rsi_max: Optional[float] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Run comprehensive RSI analysis across the specified range with optional preconditions and exclusions"""
    
    # Fetch data with quality validation
    all_messages = []
    
    with st.spinner(f"Fetching data for {signal_ticker}..."):
        signal_data = get_stock_data(signal_ticker, start_date, end_date, exclusions)
        is_valid, messages = validate_data_quality(signal_data, signal_ticker)
        all_messages.extend(messages)
        if not is_valid:
            return None, None, all_messages
    
    with st.spinner(f"Fetching data for {target_ticker}..."):
        target_data = get_stock_data(target_ticker, start_date, end_date, exclusions)
        is_valid, messages = validate_data_quality(target_data, target_ticker)
        all_messages.extend(messages)
        if not is_valid:
            return None, None, all_messages
    
    # Fetch benchmark data for comparison - use user-selected benchmark
    with st.spinner(f"Fetching benchmark data ({benchmark_ticker})..."):
        benchmark_data = get_stock_data(benchmark_ticker, start_date, end_date, exclusions)
        is_valid, messages = validate_data_quality(benchmark_data, benchmark_ticker)
        all_messages.extend(messages)
        if not is_valid:
            return None, None, all_messages
    
    # Note: Precondition data is now fetched directly by build_precondition_mask function
    
    if signal_data is None or target_data is None or benchmark_data is None:
        return None, None, all_messages
    
    # Align data on common dates
    common_dates = signal_data.index.intersection(target_data.index).intersection(benchmark_data.index)
    signal_data = signal_data[common_dates]
    target_data = target_data[common_dates]
    benchmark_data = benchmark_data[common_dates]
    
    # Note: Precondition data alignment is now handled by build_precondition_mask function
    
    # Create buy-and-hold benchmark
    benchmark = benchmark_data / benchmark_data.iloc[0]  # Normalize to start at 1.0
    
    # Calculate benchmark returns for statistical testing
    benchmark_returns = benchmark_data.pct_change().dropna()
    
    # Generate RSI thresholds based on the specified range
    if rsi_min is not None and rsi_max is not None:
        rsi_thresholds = np.arange(rsi_min, rsi_max + 0.5, 0.5)
    else:
        rsi_thresholds = np.array([rsi_threshold], dtype=float)
    
    results = []
    
    progress_bar = st.progress(0)
    total_thresholds = max(1, len(rsi_thresholds))
    
    for i, threshold in enumerate(rsi_thresholds):
        analysis = analyze_rsi_signals(signal_data, target_data, threshold, comparison, rsi_period, rsi_method, use_quantstats, preconditions)
        
        # Calculate statistical significance
        strategy_equity_curve = analysis['equity_curve']
        if len(strategy_equity_curve) > 0:
            # Create benchmark equity curve that follows the same RSI conditions
            # This ensures we're comparing strategy vs benchmark under the same conditions
            signal_rsi = calculate_rsi(signal_data, window=rsi_period, method=rsi_method)
            
            # Generate buy signals for benchmark (same as strategy)
            if comparison == "less_than":
                benchmark_base_signals = (signal_rsi <= threshold).astype(int)
            else:  # greater_than
                benchmark_base_signals = (signal_rsi >= threshold).astype(int)
            
            # Apply expanded preconditions to benchmark signals (same mask engine)
            if preconditions:
                benchmark_pc_mask = build_precondition_mask(
                    base_index=signal_data.index,
                    preconditions=preconditions,
                    start_date=signal_data.index.min(),
                    end_date=signal_data.index.max(),
                    rsi_len=rsi_period,
                ).reindex(signal_data.index).fillna(False)
                benchmark_signals = (benchmark_base_signals.astype(bool) & benchmark_pc_mask).astype(int)
            else:
                benchmark_signals = benchmark_base_signals
            
            # Calculate benchmark equity curve using benchmark prices (same logic as strategy)
            benchmark_equity_curve = pd.Series(1.0, index=benchmark_data.index)
            current_equity = 1.0
            in_position = False
            entry_equity = 1.0
            entry_price = None
            benchmark_trades = []
            
            for date in benchmark_data.index:
                current_signal = benchmark_signals[date] if date in benchmark_signals.index else 0
                current_price = benchmark_data[date]
                
                if current_signal == 1 and not in_position:
                    # Enter position
                    in_position = True
                    entry_equity = current_equity
                    entry_price = current_price
                    
                elif current_signal == 0 and in_position:
                    # Exit position
                    trade_return = (current_price - entry_price) / entry_price
                    current_equity = entry_equity * (1 + trade_return)
                    benchmark_trades.append(trade_return)
                    in_position = False
                
                # Update equity curve
                if in_position:
                    current_equity = entry_equity * (current_price / entry_price)
                
                benchmark_equity_curve[date] = current_equity
            
            # Handle case where we're still in position at the end
            if in_position:
                final_price = benchmark_data.iloc[-1]
                trade_return = (final_price - entry_price) / entry_price
                current_equity = entry_equity * (1 + trade_return)
                benchmark_trades.append(trade_return)
                benchmark_equity_curve.iloc[-1] = current_equity
            
            # Calculate benchmark average and median returns
            benchmark_avg_return = np.mean(benchmark_trades) if benchmark_trades else 0
            benchmark_median_return = np.median(benchmark_trades) if benchmark_trades else 0
            days = (benchmark.index[-1] - benchmark.index[0]).days
            benchmark_annualized = (benchmark.iloc[-1]) ** (365 / days) - 1 if days > 0 else 0
            stats_result = calculate_statistical_significance(
                strategy_equity_curve, 
                benchmark_equity_curve,
                analysis['annualized_return'],
                benchmark_annualized
            )
            
            # Calculate additional risk metrics
            risk_metrics = calculate_additional_metrics(analysis['returns'], analysis['equity_curve'], analysis['annualized_return'], use_quantstats, benchmark_equity_curve)
        else:
            # Calculate benchmark average and median returns even when strategy has no trades
            signal_rsi = calculate_rsi(signal_data, window=rsi_period, method=rsi_method)
            
            # Generate buy signals for benchmark (same as strategy)
            if comparison == "less_than":
                benchmark_base_signals = (signal_rsi <= threshold).astype(int)
            else:  # greater_than
                benchmark_base_signals = (signal_rsi >= threshold).astype(int)
            
            # Apply expanded preconditions to benchmark signals (same mask engine)
            if preconditions:
                benchmark_pc_mask = build_precondition_mask(
                    base_index=signal_data.index,
                    preconditions=preconditions,
                    start_date=signal_data.index.min(),
                    end_date=signal_data.index.max(),
                    rsi_len=rsi_period,
                ).reindex(signal_data.index).fillna(False)
                benchmark_signals = (benchmark_base_signals.astype(bool) & benchmark_pc_mask).astype(int)
            else:
                benchmark_signals = benchmark_base_signals
            
            # Calculate benchmark equity curve using benchmark prices (same logic as strategy)
            benchmark_equity_curve = pd.Series(1.0, index=benchmark_data.index)
            current_equity = 1.0
            in_position = False
            entry_equity = 1.0
            entry_price = None
            benchmark_trades = []
            
            for date in benchmark_data.index:
                current_signal = benchmark_signals[date] if date in benchmark_signals.index else 0
                current_price = benchmark_data[date]
                
                if current_signal == 1 and not in_position:
                    # Enter position
                    in_position = True
                    entry_equity = current_equity
                    entry_price = current_price
                    
                elif current_signal == 0 and in_position:
                    # Exit position
                    trade_return = (current_price - entry_price) / entry_price
                    current_equity = entry_equity * (1 + trade_return)
                    benchmark_trades.append(trade_return)
                    in_position = False
                
                # Update equity curve
                if in_position:
                    current_equity = entry_equity * (current_price / entry_price)
                
                benchmark_equity_curve[date] = current_equity
            
            # Handle case where we're still in position at the end
            if in_position:
                final_price = benchmark_data.iloc[-1]
                trade_return = (final_price - entry_price) / entry_price
                current_equity = entry_equity * (1 + trade_return)
                benchmark_trades.append(trade_return)
                benchmark_equity_curve.iloc[-1] = current_equity
            
            # Calculate benchmark average and median returns
            benchmark_avg_return = np.mean(benchmark_trades) if benchmark_trades else 0
            benchmark_median_return = np.median(benchmark_trades) if benchmark_trades else 0
            
            stats_result = {
                't_statistic': 0,
                'p_value': 1.0,
                'confidence_level': 0,
                'significant': False,
                'effect_size': 0,
                'power': 0,
                'direction': 'unknown',
                'insufficient_data': True
            }
            
            # Calculate additional risk metrics (even when no trades)
            risk_metrics = calculate_additional_metrics(analysis['returns'], analysis['equity_curve'], analysis['annualized_return'], use_quantstats, benchmark_equity_curve)
        
        results.append({
            'RSI_Threshold': threshold,
            'Total_Trades': analysis['total_trades'],
            'Win_Rate': analysis['win_rate'],
            'Avg_Return': analysis['avg_return'],
            'Median_Return': analysis.get('median_return', 0),  # Use get() with default value
            'Benchmark_Avg_Return': benchmark_avg_return,
            'Benchmark_Median_Return': benchmark_median_return,
            'Avg_Hold_Days': analysis['avg_hold_days'],
            'Sortino_Ratio': analysis['sortino_ratio'],
            'Return_Std': np.std(analysis['returns']) if len(analysis['returns']) > 0 else 0,
            'Best_Return': np.max(analysis['returns']) if len(analysis['returns']) > 0 else 0,
            'Worst_Return': np.min(analysis['returns']) if len(analysis['returns']) > 0 else 0,
            'Final_Equity': analysis['equity_curve'].iloc[-1] if analysis['equity_curve'] is not None else 1.0,
            'Total_Return': (analysis['equity_curve'].iloc[-1] - 1) if analysis['equity_curve'] is not None else 0,
            'annualized_return': analysis['annualized_return'],
            'equity_curve': analysis['equity_curve'],
            'trades': analysis['trades'],
            'returns': analysis['returns'],
            't_statistic': stats_result['t_statistic'],
            'p_value': stats_result['p_value'],
            'confidence_level': stats_result['confidence_level'],
            'significant': stats_result['significant'],
            'effect_size': stats_result['effect_size'],
            'power': stats_result['power'],
            'direction': stats_result.get('direction', 'unknown'),
            'insufficient_data': stats_result.get('insufficient_data', False),
            'max_drawdown': risk_metrics['max_drawdown'],
            'calmar_ratio': risk_metrics['calmar_ratio'],
            'var_95': risk_metrics['var_95'],
            'sharpe_ratio': risk_metrics['sharpe_ratio'],
            'volatility': risk_metrics['volatility'],
            'beta': risk_metrics.get('beta', 0.0),
            'alpha': risk_metrics.get('alpha', 0.0),
            'information_ratio': risk_metrics.get('information_ratio', 0.0)
        })
        
        progress_bar.progress((i + 1) / total_thresholds)
    
    return pd.DataFrame(results), benchmark, all_messages



# Streamlit Interface
st.sidebar.header("⚙️ Configuration")

# QuantStats Configuration
use_quantstats = st.sidebar.checkbox("Enable QuantStats Integration", value=True, help="Enable use of QuantStats library. When disabled, the app will use fallback calculations.")

# Preconditions System
st.sidebar.subheader("Preconditions", help="All preconditions must be TRUE to allow the main signal. Supports RSI vs threshold, RSI vs RSI, and Price/MA/EMA comparisons.")
if 'preconditions' not in st.session_state:
    st.session_state.preconditions = []

# Show existing
if st.session_state.preconditions:
    st.sidebar.write("**Current Preconditions (ALL must hold):**")
    for i, p in enumerate(st.session_state.preconditions):
        mode = p.get("mode","static")
        if mode in ("price_vs_ma","price_vs_ema","ma_vs_ma","ema_vs_ema"):
            sym = {"less_than":"<","greater_than":">"}[p.get("op","less_than")]
            st.sidebar.write(f"• {mode}: {p['lhs_ticker']}({p.get('lhs_len', 1)}) {sym} {p['rhs_ticker']}({p.get('rhs_len', 1)})")
        elif mode == "pair":
            sym = {"less_than":"<","greater_than":">"}[p.get("op","less_than")]
            st.sidebar.write(f"• {p['lhs_ticker']} RSI({p.get('lhs_len', 10)}) {sym} {p['rhs_ticker']} RSI({p.get('rhs_len', 10)})")
        else:
            sym = {"less_than":"<","greater_than":">"}[p.get("comparison","greater_than")]
            st.sidebar.write(f"• {p['signal_ticker']} RSI({p.get('rsi_len', 10)}) {sym} {p.get('threshold', 50)}")
        if st.sidebar.button("🗑️", key=f"remove_pre_{i}"):
            st.session_state.preconditions.pop(i)
            st.rerun()
else:
    st.sidebar.caption("No preconditions yet.")

with st.sidebar.expander("➕ Add Precondition", expanded=False):
    tabs = st.tabs(["Static RSI", "Pair RSI", "Price / MA / EMA"])
    # Static RSI vs threshold
    with tabs[0]:
        pc_tkr = st.text_input("Ticker", value="QQQ", key="pc_static_tkr").strip().upper()
        pc_len = st.number_input("RSI length", min_value=2, max_value=200, value=10, step=1, key="pc_static_len")
        pc_cmp = st.selectbox("Comparison", ["less_than","greater_than"], index=1, key="pc_static_cmp",
                              format_func=lambda x: "RSI ≤ thr" if x=="less_than" else "RSI ≥ thr")
        pc_thr = st.number_input("RSI threshold", min_value=0.0, max_value=100.0, value=80.0, step=0.5, key="pc_static_thr")
        if st.button("Add static RSI precondition", key="add_pc_static"):
            st.session_state.preconditions.append({
                "mode":"static",
                "signal_ticker": pc_tkr,
                "rsi_len": int(pc_len),
                "comparison": pc_cmp,
                "threshold": float(pc_thr),
            })
            st.rerun()

    # Pair RSI vs RSI
    with tabs[1]:
        cols = st.columns(2)
        with cols[0]:
            lhs_t = st.text_input("Left ticker (LHS)", value="KMLM", key="pc_pair_lhs").strip().upper()
            lhs_len = st.number_input("LHS RSI length", min_value=2, max_value=200, value=10, step=1, key="pc_pair_lhs_len")
        with cols[1]:
            rhs_t = st.text_input("Right ticker (RHS)", value="XLK", key="pc_pair_rhs").strip().upper()
            rhs_len = st.number_input("RHS RSI length", min_value=2, max_value=200, value=10, step=1, key="pc_pair_rhs_len")
        op = st.selectbox("Comparison", ["less_than","greater_than"], index=0, key="pc_pair_op",
                          format_func=lambda x: {"less_than":"<","greater_than":">"}[x])
        if st.button("Add pair RSI precondition", key="add_pc_pair"):
            st.session_state.preconditions.append({
                "mode":"pair",
                "lhs_ticker": lhs_t, "lhs_len": int(lhs_len),
                "op": op,
                "rhs_ticker": rhs_t, "rhs_len": int(rhs_len),
            })
            st.rerun()

    # Price/MA/EMA family
    with tabs[2]:
        mode_choice = st.selectbox(
            "Condition type",
            ["price_vs_ma","price_vs_ema","ma_vs_ma","ema_vs_ema"],
            index=0,
            key="pc_ma_mode",
            format_func=lambda m: {
                "price_vs_ma":"Price vs SMA",
                "price_vs_ema":"Price vs EMA",
                "ma_vs_ma":"SMA vs SMA",
                "ema_vs_ema":"EMA vs EMA",
            }[m]
        )
        ccols = st.columns(2)
        with ccols[0]:
            lhs_t = st.text_input("Left ticker (LHS)", value="SPY", key="pc_ma_lhs").strip().upper()
            lhs_len = st.number_input("LHS window", min_value=1, max_value=500, value=1, step=1, key="pc_ma_lhs_len",
                                      help="Use 1 for raw price with Price vs MA/EMA.")
        with ccols[1]:
            rhs_t = st.text_input("Right ticker (RHS)", value="SPY", key="pc_ma_rhs").strip().upper()
            default_rhs_len = 200 if "price_vs_" in mode_choice else 50
            rhs_len = st.number_input("RHS window", min_value=1, max_value=500, value=default_rhs_len, step=1, key="pc_ma_rhs_len")
        op = st.selectbox("Comparison", ["less_than","greater_than"], index=1, key="pc_ma_op",
                          format_func=lambda x: {"less_than":"<","greater_than":">"}[x])
        if st.button("Add MA/EMA precondition", key="add_pc_ma"):
            st.session_state.preconditions.append({
                "mode": mode_choice,
                "lhs_ticker": lhs_t, "lhs_len": int(lhs_len),
                "op": op,
                "rhs_ticker": rhs_t, "rhs_len": int(rhs_len),
            })
            st.rerun()

# Clear all button
if st.session_state.preconditions:
    if st.sidebar.button("🗑️ Clear All Preconditions", type="secondary"):
        st.session_state.preconditions = []
        st.rerun()

# Input fields with help tooltips
signal_ticker = st.sidebar.text_input("Signal Ticker", value="QQQ", help="The ticker that generates RSI signals. This is the stock/ETF whose RSI we'll use to decide when to buy/sell the target ticker.")

# RSI Period selection
rsi_period = st.sidebar.number_input("RSI Period (Days)", min_value=1, max_value=50, value=10, 
                                    help="How many days to look back when calculating RSI. 10 is more sensitive to recent changes than the standard 14. Lower numbers make RSI more responsive to recent market movements.")

# RSI Calculation Method - Fixed to Wilder's method
rsi_method = "wilders"

# Conditional target ticker default based on RSI condition
comparison = st.sidebar.selectbox("RSI Condition", 
                               ["greater_than", "less_than"], 
                               format_func=lambda x: "RSI ≥ threshold" if x == "greater_than" else "RSI ≤ threshold",
                               help="Choose when to buy: 'RSI ≥ threshold' means buy when RSI is high (overbought), 'RSI ≤ threshold' means buy when RSI is low (oversold).")

# Set default target ticker based on RSI condition
if comparison == "less_than":
    default_target = "TQQQ"
else:
    default_target = "VIXY"

target_ticker = st.sidebar.text_input("Target Ticker", value=default_target, help="The ticker to buy/sell based on the signal ticker's RSI. This is what you'll actually be trading.")

# Benchmark selection
benchmark_options = ["SPY", "BIL", "TQQQ"]
benchmark_ticker = st.sidebar.selectbox("Benchmark", 
                                       benchmark_options, 
                                       format_func=lambda x: {
                                           "SPY": "SPY (S&P 500)",
                                           "BIL": "BIL (Cash Equivalent)", 
                                           "TQQQ": "TQQQ (3x Nasdaq-100)"
                                       }.get(x, x),
                                       help="Choose your benchmark for comparison. This is what your signal will be compared against under the RSI conditions you specify.")

# Allow custom benchmark input
use_custom_benchmark = st.sidebar.checkbox("Use custom benchmark ticker", help="Check this to specify a custom ticker symbol instead of using the selected benchmark above.")

if use_custom_benchmark:
    custom_benchmark = st.sidebar.text_input("Custom Benchmark Ticker", 
                                            placeholder="e.g., QQQ, VTI, etc.",
                                            help="Enter a custom ticker symbol to use as benchmark.")
else:
    custom_benchmark = ""

# Set default RSI threshold based on condition
if comparison == "less_than":
    default_threshold = 40.0
    st.sidebar.write("Buy signals: Signal RSI ≤ threshold")
else:
    default_threshold = 60.0
    st.sidebar.write("Buy signals: Signal RSI ≥ threshold")

rsi_threshold = st.sidebar.number_input("RSI Threshold", min_value=0.0, max_value=100.0, value=float(default_threshold), step=0.5, help="The RSI threshold to test. For 'RSI ≤ threshold', try 20-40. For 'RSI ≥ threshold', try 60-80.")

# RSI Range options
st.sidebar.subheader("📊 RSI Range Options")
use_custom_range = st.sidebar.checkbox("Use custom RSI range", help="Check this to specify a custom range of RSI values to test instead of the default range.")

# Set default ranges based on condition
if comparison == "less_than":
    default_min, default_max = 0.0, 50.0  # Test from 0 to 50 for less than conditions
else:
    default_min, default_max = 50.0, 100.0  # Test from 50 to 100 for greater than conditions

if use_custom_range:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        rsi_min = st.sidebar.number_input("RSI Range Min", min_value=0.0, max_value=100.0, value=float(default_min), step=0.5, help="The lowest RSI threshold to test.")
    with col2:
        rsi_max = st.sidebar.number_input("RSI Range Max", min_value=0.0, max_value=100.0, value=float(default_max), step=0.5, help="The highest RSI threshold to test.")
    
    if rsi_min >= rsi_max:
        st.sidebar.error("RSI Min must be less than RSI Max")
else:
    rsi_min, rsi_max = default_min, default_max

# Date range selection
st.sidebar.subheader("📅 Date Range")
use_date_range = st.sidebar.checkbox("Use custom date range", help="Check this to specify your own start and end dates. If unchecked, the app will use all available data.")

if use_date_range:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2020, 1, 1), help="The first date to include in your analysis. Earlier dates give more data but may not reflect current market conditions.")
    with col2:
        end_date = st.date_input("End Date", value=datetime.now(), help="The last date to include in your analysis. More recent dates may be more relevant to current market conditions.")
    
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date")
        start_date, end_date = None, None
else:
    start_date, end_date = None, None
    st.sidebar.info("Using maximum available data")

# Exclude date windows
st.sidebar.subheader("🚫 Exclude Date Windows")
use_exclusions = st.sidebar.checkbox("Exclude specific date windows", help="Check this to exclude specific periods like the COVID crash from your analysis.")

if use_exclusions:
    # Initialize exclusions in session state if not exists
    if 'date_exclusions' not in st.session_state:
        st.session_state.date_exclusions = []
    
    # Display existing exclusions
    if st.session_state.date_exclusions:
        st.sidebar.write("**Current Exclusions:**")
        for i, exclusion in enumerate(st.session_state.date_exclusions):
            with st.sidebar.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {exclusion['start']} to {exclusion['end']}")
                with col2:
                    if st.button(f"🗑️", key=f"remove_exclusion_{i}"):
                        st.session_state.date_exclusions.pop(i)
                        st.rerun()
    
    # Add new exclusion
    with st.sidebar.expander("➕ Add Exclusion Window", expanded=False):
        st.write("**Add a new date exclusion:**")
        
        col1, col2 = st.columns(2)
        with col1:
            exclusion_start = st.date_input("Exclusion Start Date", 
                                          value=datetime(2020, 2, 20), 
                                          key="exclusion_start",
                                          help="Start date of the period to exclude.")
        with col2:
            exclusion_end = st.date_input("Exclusion End Date", 
                                        value=datetime(2020, 4, 7), 
                                        key="exclusion_end",
                                        help="End date of the period to exclude.")
        
        # Add exclusion button
        if st.button("➕ Add Exclusion", key="add_exclusion"):
            if exclusion_start < exclusion_end:
                new_exclusion = {
                    'start': exclusion_start,
                    'end': exclusion_end
                }
                st.session_state.date_exclusions.append(new_exclusion)
                st.rerun()
            else:
                st.error("Start date must be before end date")
    
    # Clear all exclusions button
    if st.session_state.date_exclusions:
        if st.sidebar.button("🗑️ Clear All Exclusions", type="secondary"):
            st.session_state.date_exclusions = []
            st.rerun()
else:
    # Clear exclusions if feature is disabled
    if 'date_exclusions' in st.session_state:
        st.session_state.date_exclusions = []

# Add the Run Analysis button to the sidebar
st.sidebar.markdown("---")

# Determine which benchmark to use
final_benchmark_ticker = custom_benchmark.strip() if custom_benchmark.strip() else benchmark_ticker

if st.sidebar.button("🚀 Run RSI Analysis", type="primary", use_container_width=True):
    if (not use_date_range or (start_date and end_date and start_date < end_date)) and (not use_custom_range or (rsi_min and rsi_max and rsi_min < rsi_max)):
        try:
            exclusions = st.session_state.get('date_exclusions', []) if use_exclusions else None
            results_df, benchmark, data_messages = run_rsi_analysis(signal_ticker, target_ticker, rsi_threshold, comparison, start_date, end_date, rsi_period, rsi_method, final_benchmark_ticker, use_quantstats, st.session_state.get('preconditions', []), exclusions, rsi_min=rsi_min, rsi_max=rsi_max)
            
            if results_df is not None and benchmark is not None and not results_df.empty:
                # Store analysis results in session state
                st.session_state['results_df'] = results_df
                st.session_state['benchmark'] = benchmark
                st.session_state['signal_data'] = get_stock_data(signal_ticker, start_date, end_date, exclusions)
                st.session_state['benchmark_data'] = get_stock_data(final_benchmark_ticker, start_date, end_date, exclusions)
                st.session_state['rsi_period'] = rsi_period
                st.session_state['comparison'] = comparison
                st.session_state['benchmark_ticker'] = final_benchmark_ticker
                st.session_state['analysis_completed'] = True
                st.session_state['data_messages'] = data_messages
                
                st.sidebar.success("✅ Analysis completed successfully!")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error during analysis: {str(e)}")
    else:
        if use_date_range and (not start_date or not end_date or start_date >= end_date):
            st.sidebar.error("Please ensure start date is before end date")
        if use_custom_range and (not rsi_min or not rsi_max or rsi_min >= rsi_max):
            st.sidebar.error("Please ensure RSI Min is less than RSI Max")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("⚙️ Analysis Configuration")
    
    # Display preconditions first
    if st.session_state.get('preconditions'):
        st.write("**Preconditions:**")
        for p in st.session_state.preconditions:
            mode = p.get("mode","static")
            if mode in ("price_vs_ma","price_vs_ema","ma_vs_ma","ema_vs_ema"):
                sym = {"less_than":"<","greater_than":">"}[p.get("op","less_than")]
                st.write(f"  • {mode}: {p['lhs_ticker']}({p.get('lhs_len',1)}) {sym} {p['rhs_ticker']}({p.get('rhs_len',1)})")
            elif mode == "pair":
                sym = {"less_than":"<","greater_than":">"}[p.get("op","less_than")]
                st.write(f"  • {p['lhs_ticker']} RSI({p.get('lhs_len',10)}) {sym} {p['rhs_ticker']} RSI({p.get('rhs_len',10)})")
            else:
                sym = {"less_than":"<","greater_than":">"}[p.get("comparison","greater_than")]
                st.write(f"  • {p['signal_ticker']} RSI({p.get('rsi_len',10)}) {sym} {p.get('threshold', 50)}")
    
    st.write(f"**Signal Ticker:** {signal_ticker} (generates RSI signals)")
    st.write(f"**Target Ticker:** {target_ticker} (buy/sell based on signals)")
    
    # Display benchmark information
    if custom_benchmark.strip():
        benchmark_display = custom_benchmark.strip()
        benchmark_description = "Custom Benchmark"
    else:
        benchmark_display = benchmark_ticker
        benchmark_description = {
            "SPY": "S&P 500",
            "BIL": "Cash Equivalent", 
            "TQQQ": "3x Nasdaq-100"
        }.get(benchmark_ticker, benchmark_ticker)
    
    st.write(f"**Benchmark:** {benchmark_display} ({benchmark_description})")
    st.write(f"**RSI Period:** {rsi_period}-day RSI")
    if use_custom_range:
        st.write(f"**RSI Condition:** {signal_ticker} RSI {'≤' if comparison == 'less_than' else '≥'} {rsi_threshold} (testing {rsi_min} to {rsi_max})")
    else:
        st.write(f"**RSI Condition:** {signal_ticker} RSI {'≤' if comparison == 'less_than' else '≥'} {rsi_threshold} (testing {rsi_min} to {rsi_max})")
    
    if use_date_range and start_date and end_date:
        st.write(f"**Date Range:** {start_date} to {end_date}")
    else:
        st.write(f"**Date Range:** Maximum available data")
    
    # Display exclusions
    if use_exclusions and st.session_state.get('date_exclusions'):
        st.write("**Excluded Periods:**")
        for exclusion in st.session_state.date_exclusions:
            st.write(f"  • {exclusion['start']} to {exclusion['end']}")

with col2:
    st.subheader("📋 Signal Logic")
    
    # Build the signal logic description
    if st.session_state.get('preconditions'):
        st.write("**Preconditions (ALL true):**")
        for p in st.session_state.preconditions:
            mode = p.get("mode","static")
            if mode in ("price_vs_ma","price_vs_ema","ma_vs_ma","ema_vs_ema"):
                sym = {"less_than":"<","greater_than":">"}[p.get("op","less_than")]
                st.write(f"  • {mode}: {p['lhs_ticker']}({p.get('lhs_len',1)}) {sym} {p['rhs_ticker']}({p.get('rhs_len',1)})")
            elif mode == "pair":
                sym = {"less_than":"<","greater_than":">"}[p.get("op","less_than")]
                st.write(f"  • {p['lhs_ticker']} RSI({p.get('lhs_len',10)}) {sym} {p['rhs_ticker']} RSI({p.get('rhs_len',10)})")
            else:
                sym = {"less_than":"<","greater_than":">"}[p.get("comparison","greater_than")]
                st.write(f"  • {p['signal_ticker']} RSI({p.get('rsi_len',10)}) {sym} {p.get('threshold', 50)}")
        st.write("**Main Signal:**")
    
    if comparison == "less_than":
        st.info(f"🔵 BUY {target_ticker} when {signal_ticker} {rsi_period}-day RSI ≤ threshold\n\n📈 SELL {target_ticker} when {signal_ticker} {rsi_period}-day RSI > threshold")
    else:
        st.info(f"🔵 BUY {target_ticker} when {signal_ticker} {rsi_period}-day RSI ≥ threshold\n\n📈 SELL {target_ticker} when {signal_ticker} {rsi_period}-day RSI < threshold")

# Check if we have stored analysis results
if 'analysis_completed' in st.session_state and st.session_state['analysis_completed']:
    # Display stored results
    results_df = st.session_state['results_df']
    benchmark = st.session_state['benchmark']
    
    st.success("✅ Analysis completed successfully!")
    
    # Check for data quality issues
    insufficient_data_count = results_df.get('insufficient_data', pd.Series([False] * len(results_df))).sum()
    low_trade_count = (results_df['Total_Trades'] < 5).sum()
    extreme_rsi_count = 0
    
    # Count extreme RSI values (very high or very low depending on comparison)
    if st.session_state.get('comparison') == 'greater_than':
        extreme_rsi_count = (results_df['RSI_Threshold'] >= 85).sum()
    else:
        extreme_rsi_count = (results_df['RSI_Threshold'] <= 15).sum()
    
    if insufficient_data_count > 0 or low_trade_count > 0 or extreme_rsi_count > 0:
        st.warning("⚠️ **Data Quality Warnings:**")
        if insufficient_data_count > 0:
            st.write(f"• {insufficient_data_count} RSI thresholds had insufficient data for reliable statistical testing")
        if low_trade_count > 0:
            st.write(f"• {low_trade_count} RSI thresholds generated fewer than 5 trades")
        if extreme_rsi_count > 0:
            st.write(f"• {extreme_rsi_count} RSI thresholds are at extreme values (may have limited historical occurrences)")
        st.write("**Recommendation:** Focus on RSI thresholds with more trades and higher confidence levels for more reliable results.")
    
    # Display results table
    st.subheader("📊 RSI Analysis Results")
    st.info("💡 **What this shows:** This table displays all the RSI thresholds tested and their performance metrics. Each row represents a different RSI level and shows how well that strategy performed.")
    
    # Format the dataframe for display
    display_df = results_df.copy()
    
    # Check if required columns exist before formatting
    required_columns = ['Win_Rate', 'Avg_Return', 'Median_Return', 'Benchmark_Avg_Return', 'Benchmark_Median_Return', 'Total_Return', 'annualized_return', 
                      'Sortino_Ratio', 'Avg_Hold_Days', 'Return_Std', 'Best_Return', 
                      'Worst_Return', 'Final_Equity', 'confidence_level', 'significant', 'effect_size']
    
    missing_columns = [col for col in required_columns if col not in results_df.columns]
    if missing_columns:
        st.error(f"Missing columns in results: {missing_columns}")
        st.stop()
    
    # Format the columns for display
    display_df['Win_Rate'] = display_df['Win_Rate'].apply(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else x)
    display_df['Avg_Return'] = display_df['Avg_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Median_Return'] = display_df['Median_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Benchmark_Avg_Return'] = display_df['Benchmark_Avg_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Benchmark_Median_Return'] = display_df['Benchmark_Median_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Total_Return'] = display_df['Total_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Annualized_Return'] = display_df['annualized_return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Sortino_Ratio'] = display_df['Sortino_Ratio'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not np.isinf(x) else "∞" if isinstance(x, (int, float)) and np.isinf(x) else x)
    display_df['Sharpe_Ratio'] = display_df['sharpe_ratio'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not np.isinf(x) else "∞" if isinstance(x, (int, float)) and np.isinf(x) else x)
    display_df['Calmar_Ratio'] = display_df['calmar_ratio'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not np.isinf(x) else "∞" if isinstance(x, (int, float)) and np.isinf(x) else x)
    display_df['Max_Drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['VaR_95'] = display_df['var_95'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Avg_Hold_Days'] = display_df['Avg_Hold_Days'].apply(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x)
    display_df['Return_Std'] = display_df['Return_Std'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Best_Return'] = display_df['Best_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Worst_Return'] = display_df['Worst_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Final_Equity'] = display_df['Final_Equity'].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    display_df['Confidence_Level'] = display_df['confidence_level'].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    display_df['Significant'] = display_df['significant'].apply(lambda x: "✓" if x else "✗")
    display_df['Direction'] = display_df['direction'].map({'outperform':'↑', 'underperform':'↓'}).fillna('')
    display_df['Effect_Size'] = display_df['effect_size'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    
    # Add p-value to display columns
    display_df['P_Value'] = display_df['p_value'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    
    # Drop the equity_curve and trades columns for display
    display_cols = ['RSI_Threshold', 'Total_Trades', 'Win_Rate', 'Avg_Return', 'Median_Return', 'Benchmark_Avg_Return', 'Benchmark_Median_Return',
                   'Total_Return', 'Annualized_Return', 'Sortino_Ratio', 'Sharpe_Ratio', 'Calmar_Ratio', 'Final_Equity', 'Avg_Hold_Days', 
                   'Return_Std', 'Best_Return', 'Worst_Return', 'Max_Drawdown', 'VaR_95', 'Confidence_Level', 'Significant', 'Direction', 'Effect_Size', 'P_Value']
    
    # Check if all display columns exist
    missing_display_cols = [col for col in display_cols if col not in display_df.columns]
    if missing_display_cols:
        st.error(f"Missing display columns: {missing_display_cols}")
        st.stop()
    
    # Add filter options for the results table
    with st.expander("📊 Table of Results", expanded=False):
        st.subheader("🔍 Filter Results")
    
        # Create filter columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rsi_min_filter = st.number_input(
                "Min RSI Threshold:",
                min_value=float(display_df['RSI_Threshold'].min()),
                max_value=float(display_df['RSI_Threshold'].max()),
                value=float(display_df['RSI_Threshold'].min()),
                step=0.5,
                help="Minimum RSI threshold to include in results."
            )
            rsi_max_filter = st.number_input(
                "Max RSI Threshold:",
                min_value=float(display_df['RSI_Threshold'].min()),
                max_value=float(display_df['RSI_Threshold'].max()),
                value=float(display_df['RSI_Threshold'].max()),
                step=0.5,
                help="Maximum RSI threshold to include in results."
            )
        with col2:
            confidence_min_filter = st.number_input(
                "Min Confidence Level (%):",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                help="Minimum confidence level to include in results."
            )
            confidence_max_filter = st.number_input(
                "Max Confidence Level (%):",
                min_value=0.0,
                max_value=100.0,
                value=100.0,
                step=1.0,
                help="Maximum confidence level to include in results."
            )
        with col3:
            min_trades_filter = st.number_input(
                "Min Total Trades:",
                min_value=0,
                value=0,
                help="Minimum number of trades to include in results."
            )
            min_win_rate_filter = st.number_input(
                "Min Win Rate (%):",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                help="Minimum win rate percentage to include in results."
            )
        with col4:
            min_avg_return_filter = st.number_input(
                "Min Avg Return (%):",
                min_value=-100.0,
                max_value=100.0,
                value=-100.0,
                step=0.1,
                help="Minimum average return percentage to include in results."
            )
            min_total_return_filter = st.number_input(
                "Min Total Return (%):",
                min_value=-100.0,
                max_value=100.0,
                value=-100.0,
                step=0.1,
                help="Minimum total return percentage to include in results."
            )
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            min_annualized_return_filter = st.number_input(
                "Min Annualized Return (%):",
                min_value=-100.0,
                max_value=100.0,
                value=-100.0,
                step=0.1,
                help="Minimum annualized return percentage to include in results."
            )
        with col6:
            min_sortino_filter = st.number_input(
                "Min Sortino Ratio:",
                min_value=-10.0,
                max_value=10.0,
                value=-10.0,
                step=0.1,
                help="Minimum Sortino ratio to include in results."
            )
        with col7:
            significance_filter = st.selectbox(
                "Significance:",
                ["All", "Significant Only", "Non-Significant Only"],
                help="Filter by statistical significance."
            )
            max_p_value_filter = st.number_input(
                "Max P-Value:",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.001,
                help="Maximum p-value to include in results (lower = more significant)."
            )
        with col8:
            if st.button("Clear All Filters", type="secondary"):
                st.rerun()
        
        # Apply filters to the display dataframe (outside of columns)
        filtered_df = display_df.copy()
        filtered_df = filtered_df[
            (filtered_df['RSI_Threshold'] >= rsi_min_filter) & 
            (filtered_df['RSI_Threshold'] <= rsi_max_filter)
        ]
        filtered_df = filtered_df[
            (filtered_df['Confidence_Level'].str.replace('%', '').astype(float) >= confidence_min_filter) & 
            (filtered_df['Confidence_Level'].str.replace('%', '').astype(float) <= confidence_max_filter)
        ]
        filtered_df = filtered_df[filtered_df['Total_Trades'] >= min_trades_filter]
        filtered_df = filtered_df[
            filtered_df['Win_Rate'].str.replace('%', '').astype(float) >= min_win_rate_filter
        ]
        filtered_df = filtered_df[
            filtered_df['Avg_Return'].str.replace('%', '').astype(float) >= min_avg_return_filter
        ]
        filtered_df = filtered_df[
            filtered_df['Total_Return'].str.replace('%', '').astype(float) >= min_total_return_filter
        ]
        filtered_df = filtered_df[
            filtered_df['Annualized_Return'].str.replace('%', '').astype(float) >= min_annualized_return_filter
        ]
        filtered_df = filtered_df[
            filtered_df['Sortino_Ratio'].apply(lambda x: float(x) if x != "∞" else 999) >= min_sortino_filter
        ]
        if significance_filter == "Significant Only":
            filtered_df = filtered_df[filtered_df['Significant'] == "✓"]
        elif significance_filter == "Non-Significant Only":
            filtered_df = filtered_df[filtered_df['Significant'] == "✗"]
        filtered_df = filtered_df[
            filtered_df['P_Value'].astype(float) <= max_p_value_filter
        ]
        
        # Display the filtered results table (outside of columns)
        st.subheader(f"📊 RSI Analysis Results ({len(filtered_df)} signals)")
        st.dataframe(filtered_df[display_cols], use_container_width=True)

    # Find best strategies (needed for subsequent sections) - use numeric data
    mask = st.session_state['results_df']['RSI_Threshold'].isin(filtered_df['RSI_Threshold'])
    numeric_slice = st.session_state['results_df'][mask]
    
    best_sortino_idx = numeric_slice['Sortino_Ratio'].idxmax()
    best_annualized_idx = numeric_slice['annualized_return'].idxmax()
    best_winrate_idx = numeric_slice['Win_Rate'].idxmax()
    best_total_return_idx = numeric_slice['Total_Return'].idxmax()
    
    # Statistical Significance Analysis
    with st.expander("📊 Statistical Significance Analysis", expanded=True):
        st.subheader("📊 Statistical Significance Analysis")
        stored_benchmark_ticker = st.session_state.get('benchmark_ticker', 'SPY')
        benchmark_description = {
            "SPY": "S&P 500",
            "BIL": "Cash Equivalent", 
            "TQQQ": "3x Nasdaq-100"
        }.get(stored_benchmark_ticker, "Custom Benchmark")
        benchmark_name = f"{stored_benchmark_ticker} ({benchmark_description})"
        
        # Use all signals for the chart (including those with 0 trades)
        valid_signals = filtered_df.copy()
        
        # Add summary of statistical analysis
        if not valid_signals.empty:
            signals_with_trades = valid_signals[valid_signals['Total_Trades'] > 0]
            significant_signals = valid_signals[valid_signals['significant'] == True]
            significant_outperform = len(significant_signals[significant_signals['Direction'] == '↑'])
            significant_underperform = len(significant_signals[significant_signals['Direction'] == '↓'])
            total_signals = len(valid_signals)
            signals_with_trades_count = len(signals_with_trades)
            st.success(f"📊 **Analysis Summary:** Found {len(significant_signals)} statistically significant signals out of {total_signals} total signals ({signals_with_trades_count} with trades).")
            if len(significant_signals) > 0:
                st.info(f"📈 **Direction Breakdown:** {significant_outperform} significant ↑ (outperformance), {significant_underperform} significant ↓ (underperformance)")
        else:
            st.warning("⚠️ **No signals found.** This means none of the RSI thresholds generated any results during the analysis period.")
        
        if not valid_signals.empty:
            # Create significance summary
            significant_signals = valid_signals[valid_signals['significant'] == True]
            
            # Confidence Level vs RSI Threshold Analysis
            st.subheader("📊 Confidence Level vs RSI Threshold Analysis")
            st.info(f"💡 **What This Section Shows:** This section determines whether your signal's performance is statistically significant - meaning the results are likely not due to chance. It compares your signal against {benchmark_name} under the same conditions to see if your target ticker choice is actually better.")
            
            # Create scatter plot for confidence vs RSI threshold
            fig_confidence_rsi = go.Figure()
            
            # Add points for significant signals (green)
            significant_data = valid_signals[valid_signals['significant'] == True]
            if not significant_data.empty:
                fig_confidence_rsi.add_trace(go.Scatter(
                    x=significant_data['RSI_Threshold'],
                    y=significant_data['confidence_level'],
                    mode='markers',
                    name='Significant Signals (≥95%)',
                    marker=dict(
                        color='green',
                        size=abs(significant_data['effect_size']) * 20 + 5,  # Scale effect size for visibility
                        sizemin=5,
                        sizemode='area',
                        opacity=0.7
                    ),
                    hovertemplate='<b>RSI %{x}</b><br>' +
                                'Confidence: %{y:.1f}%<br>' +
                                'Effect Size: %{marker.size:.1f}<br>' +
                                'Significant: ✓<extra></extra>'
                ))
            
            # Add points for borderline significant signals (yellow)
            borderline_data = valid_signals[(valid_signals['confidence_level'] >= 85) & (valid_signals['confidence_level'] < 95)]
            if not borderline_data.empty:
                fig_confidence_rsi.add_trace(go.Scatter(
                    x=borderline_data['RSI_Threshold'],
                    y=borderline_data['confidence_level'],
                    mode='markers',
                    name='Borderline Signals (85-95%)',
                    marker=dict(
                        color='yellow',
                        size=abs(borderline_data['effect_size']) * 20 + 5,  # Scale effect size for visibility
                        sizemin=5,
                        sizemode='area',
                        opacity=0.7
                    ),
                    hovertemplate='<b>RSI %{x}</b><br>' +
                                'Confidence: %{y:.1f}%<br>' +
                                'Effect Size: %{marker.size:.1f}<br>' +
                                'Borderline: ⚠<extra></extra>'
                ))
            
            # Add points for non-significant signals (red)
            non_significant_data = valid_signals[valid_signals['confidence_level'] < 85]
            if not non_significant_data.empty:
                fig_confidence_rsi.add_trace(go.Scatter(
                    x=non_significant_data['RSI_Threshold'],
                    y=non_significant_data['confidence_level'],
                    mode='markers',
                    name='Non-Significant Signals (<85%)',
                    marker=dict(
                        color='red',
                        size=abs(non_significant_data['effect_size']) * 20 + 5,  # Scale effect size for visibility
                        sizemin=5,
                        sizemode='area',
                        opacity=0.7
                    ),
                    hovertemplate='<b>RSI %{x}</b><br>' +
                                'Confidence: %{y:.1f}%<br>' +
                                'Effect Size: %{marker.size:.1f}<br>' +
                                'Significant: ✗<extra></extra>'
                ))
            
            # Add reference lines
            fig_confidence_rsi.add_hline(y=95, line_dash="dash", line_color="red", 
                                       annotation_text="95% Confidence")
            fig_confidence_rsi.add_hline(y=85, line_dash="dash", line_color="yellow", 
                                       annotation_text="85% Confidence")
            
            fig_confidence_rsi.update_layout(
                title="Confidence Level vs RSI Threshold (Point Size = Effect Size)",
                xaxis_title="RSI Threshold",
                yaxis_title="Confidence Level (%)",
                hovermode='closest',
                showlegend=True,
                xaxis=dict(range=[0, 100]),  # Set x-axis range to show full RSI scale
                yaxis=dict(range=[0, 100])  # Set y-axis range to show full confidence scale
            )
            
            st.plotly_chart(fig_confidence_rsi, use_container_width=True, key="confidence_rsi_chart")
            
                    # Add explanation for the new chart
        with st.expander("📚 Understanding Confidence vs RSI Threshold"):
            st.write(f"""

            **📊 Improved Statistical Analysis:**
            The confidence levels now show more realistic variation across RSI thresholds. The analysis properly calculates statistical significance for both outperformance and underperformance, avoiding artificial binary outcomes.
            
            **⚠️ Note on Extreme RSI Values:**
            At the extreme ends of RSI thresholds (very low or very high values), there are often not enough historical events to generate statistically confident results. This is why confidence levels may drop off at these extremes - the sample size becomes too small for reliable statistical analysis.
            
            **What This Chart Tells You:**
            
            **📊 X-Axis (RSI Threshold):**
            - Shows different RSI levels tested
            - Helps identify which RSI ranges are most effective
            
            **📈 Y-Axis (Confidence Level):**
            - Higher values = stronger statistical evidence
            - Above 95% = highly significant (strong evidence)
            - 80-95% = borderline significant (moderate evidence)
            - 60-80% = weak evidence
            - Below 60% = very weak or no evidence
            
            **🎯 Interpretation:**
            - **High confidence (95%+)**: Very strong evidence the signal works
            - **Moderate confidence (80-95%)**: Good evidence, worth considering
            - **Low confidence (<80%)**: Weak evidence, results may be due to chance
            - **Extreme RSI values**: Often show low confidence due to insufficient historical data

            """)
        
        # Total Return vs Confidence Level Analysis
        with st.expander("📊 Total Return vs Confidence Level Analysis", expanded=False):
            st.subheader("📊 Total Return vs Confidence Level Analysis")
            st.info(f"💡 **What this shows:** This scatter plot shows the relationship between total return performance and statistical confidence. Each point represents a signal - the position shows how much money the signal made (total return) and how confident we are in the results (confidence level).")
            
            # Create scatter plot for total return vs confidence level
            fig_total_return = go.Figure()
            
            # Add points for significant signals (green)
            significant_data = valid_signals[valid_signals['significant'] == True]
            if not significant_data.empty:
                fig_total_return.add_trace(go.Scatter(
                    x=significant_data['confidence_level'],
                    y=significant_data['Total_Return'],
                    mode='markers',
                    name='Significant Signals',
                    marker=dict(color='green', size=8),
                    hovertemplate='<b>RSI %{text}</b><br>' +
                                'Total Return: %{y:.3%}<br>' +
                                'Confidence: %{x:.1f}%<br>' +
                                'Significant: ✓<extra></extra>',
                    text=[f"{row['RSI_Threshold']}" for _, row in significant_data.iterrows()]
                ))
            
            # Add points for non-significant signals (red)
            non_significant_data = valid_signals[valid_signals['significant'] == False]
            if not non_significant_data.empty:
                fig_total_return.add_trace(go.Scatter(
                    x=non_significant_data['confidence_level'],
                    y=non_significant_data['Total_Return'],
                    mode='markers',
                    name='Non-Significant Signals',
                    marker=dict(color='red', size=8),
                    hovertemplate='<b>RSI %{text}</b><br>' +
                                'Total Return: %{y:.3%}<br>' +
                                'Confidence: %{x:.1f}%<br>' +
                                'Significant: ✗<extra></extra>',
                    text=[f"{row['RSI_Threshold']}" for _, row in non_significant_data.iterrows()]
                ))
            
            # Add reference lines
            fig_total_return.add_hline(y=0, line_dash="dash", line_color="gray", 
                                     annotation_text="No Return")
            fig_total_return.add_vline(x=95, line_dash="dash", line_color="red", 
                                     annotation_text="95% Confidence")
            
            fig_total_return.update_layout(
                title="Total Return vs Confidence Level",
                xaxis_title="Confidence Level (%)",
                yaxis_title="Total Return (%)",
                hovermode='closest'
            )
            
            st.plotly_chart(fig_total_return, use_container_width=True, key="total_return_chart")
        
        # Sortino Ratio vs Confidence Level Analysis
        with st.expander("📊 Sortino Ratio vs Confidence Level Analysis", expanded=False):
            st.subheader("📊 Sortino Ratio vs Confidence Level Analysis")
            st.info(f"💡 **What this shows:** This scatter plot shows the relationship between risk-adjusted returns (Sortino ratio) and statistical confidence. Each point represents a signal - the position shows how good the risk-adjusted returns are (Sortino ratio) and how confident we are in the results (confidence level).")
            
            # Create scatter plot for sortino ratio vs confidence level
            fig_sortino = go.Figure()
            
            # Add points for significant signals (green)
            significant_data = valid_signals[valid_signals['significant'] == True]
            if not significant_data.empty:
                fig_sortino.add_trace(go.Scatter(
                    x=significant_data['confidence_level'],
                    y=significant_data['Sortino_Ratio'],
                    mode='markers',
                    name='Significant Signals',
                    marker=dict(color='green', size=8),
                    hovertemplate='<b>RSI %{text}</b><br>' +
                                'Sortino Ratio: %{y:.2f}<br>' +
                                'Confidence: %{x:.1f}%<br>' +
                                'Significant: ✓<extra></extra>',
                    text=[f"{row['RSI_Threshold']}" for _, row in significant_data.iterrows()]
                ))
            
            # Add points for non-significant signals (red)
            non_significant_data = valid_signals[valid_signals['significant'] == False]
            if not non_significant_data.empty:
                fig_sortino.add_trace(go.Scatter(
                    x=non_significant_data['confidence_level'],
                    y=non_significant_data['Sortino_Ratio'],
                    mode='markers',
                    name='Non-Significant Signals',
                    marker=dict(color='red', size=8),
                    hovertemplate='<b>RSI %{text}</b><br>' +
                                'Sortino Ratio: %{y:.2f}<br>' +
                                'Confidence: %{x:.1f}%<br>' +
                                'Significant: ✗<extra></extra>',
                    text=[f"{row['RSI_Threshold']}" for _, row in non_significant_data.iterrows()]
                ))
            
            # Add reference lines
            fig_sortino.add_hline(y=0, line_dash="dash", line_color="gray", 
                                    annotation_text="No Risk-Adjusted Return")
            fig_sortino.add_vline(x=95, line_dash="dash", line_color="red", 
                                    annotation_text="95% Confidence")
            
            fig_sortino.update_layout(
                title="Sortino Ratio vs Confidence Level",
                xaxis_title="Confidence Level (%)",
                yaxis_title="Sortino Ratio",
                hovermode='closest'
            )
            
            st.plotly_chart(fig_sortino, use_container_width=True, key="sortino_chart")
        
        # Download results
        st.subheader("📥 Download Results")
        st.info("💡 **What this does:** Download your analysis results as a CSV file that you can open in Excel or other spreadsheet programs. This includes all the performance metrics for every RSI threshold tested.")
        # Use the original column names from results_df for CSV download
        download_cols = ['RSI_Threshold', 'Total_Trades', 'Win_Rate', 'Avg_Return', 'Median_Return', 'Benchmark_Avg_Return', 'Benchmark_Median_Return',
                       'Total_Return', 'annualized_return', 'Sortino_Ratio', 'sharpe_ratio', 'calmar_ratio', 'Final_Equity', 'Avg_Hold_Days', 
                       'Return_Std', 'Best_Return', 'Worst_Return', 'max_drawdown', 'var_95', 'beta', 'alpha', 'information_ratio', 'confidence_level', 'significant', 'effect_size']
        csv = st.session_state['results_df'][download_cols].to_csv(index=False)
        filename_suffix = f"_{start_date}_{end_date}" if use_date_range and start_date and end_date else "_max_range"
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"rsi_analysis_{signal_ticker}_{target_ticker}{filename_suffix}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Calculate delta for all signals
        original_filtered_data = st.session_state['results_df'][st.session_state['results_df']['RSI_Threshold'].isin(filtered_df['RSI_Threshold'])]
        
        # Calculate delta (target return - benchmark return)
        original_filtered_data['Return_Delta'] = original_filtered_data['Avg_Return'] - original_filtered_data['Benchmark_Avg_Return']
        
        # Return Delta (Average) vs RSI Threshold Chart
        st.subheader("📊 Return Delta (Average) vs RSI Threshold")
        st.info("💡 **What this shows:** This chart compares the **average** returns of holding the target ticker versus the benchmark ticker under the same RSI conditions. It displays the **difference** between target ticker **average** return and benchmark **average** return across different RSI thresholds. Positive values mean the target outperforms the benchmark, negative values mean the benchmark outperforms the target.")
        
        fig_delta_rsi = go.Figure()
        
        # Add points for significant signals (green)
        significant_data = original_filtered_data[original_filtered_data['significant'] == True]
        if not significant_data.empty:
            fig_delta_rsi.add_trace(go.Scatter(
                x=significant_data['RSI_Threshold'],
                y=significant_data['Return_Delta'],
                mode='markers',
                name='Significant Signals',
                marker=dict(color='green', size=8),
                hovertemplate='<b>RSI %{x}</b><br>' +
                            'Return Delta (Avg): %{y:.3%}<br>' +
                            'Target Avg: %{customdata[0]:.3%}<br>' +
                            'Benchmark Avg: %{customdata[1]:.3%}<br>' +
                            'Significant: ✓<extra></extra>',
                customdata=significant_data[['Avg_Return', 'Benchmark_Avg_Return']].values
            ))
        
        # Add points for non-significant signals (red)
        non_significant_data = original_filtered_data[original_filtered_data['significant'] == False]
        if not non_significant_data.empty:
            fig_delta_rsi.add_trace(go.Scatter(
                x=non_significant_data['RSI_Threshold'],
                y=non_significant_data['Return_Delta'],
                mode='markers',
                name='Non-Significant Signals',
                marker=dict(color='red', size=8),
                hovertemplate='<b>RSI %{x}</b><br>' +
                            'Return Delta (Avg): %{y:.3%}<br>' +
                            'Target Avg: %{customdata[0]:.3%}<br>' +
                            'Benchmark Avg: %{customdata[1]:.3%}<br>' +
                            'Significant: ✗<extra></extra>',
                customdata=non_significant_data[['Avg_Return', 'Benchmark_Avg_Return']].values
            ))
        
        # Add reference line at y=0
        fig_delta_rsi.add_hline(y=0, line_dash="dash", line_color="gray", 
                               annotation_text="No Difference")
        
        fig_delta_rsi.update_layout(
            title="Return Delta (Average: Target - Benchmark) vs RSI Threshold",
            xaxis_title="RSI Threshold",
            yaxis_title="Return Delta (Average) (%)",
            hovermode='closest',
            xaxis=dict(range=[0, 100]),
            yaxis=dict(tickformat='.1%'),
            showlegend=True
        )
        
        st.plotly_chart(fig_delta_rsi, use_container_width=True, key="delta_rsi_chart")
        
        # Return Delta (Median) vs RSI Threshold Chart
        st.subheader("📊 Return Delta (Median) vs RSI Threshold")
        st.info("💡 **What this shows:** This chart compares the **median** returns of holding the target ticker versus the benchmark ticker under the same RSI conditions. It displays the **difference** between target ticker **median** return and benchmark **median** return across different RSI thresholds. Median returns are less affected by extreme outliers than averages, providing a more robust measure of typical performance.")
        
        # Calculate median delta
        original_filtered_data['Median_Return_Delta'] = original_filtered_data['Median_Return'] - original_filtered_data['Benchmark_Median_Return']
        
        fig_median_delta_rsi = go.Figure()
        
        # Add points for significant signals (green)
        significant_data = original_filtered_data[original_filtered_data['significant'] == True]
        if not significant_data.empty:
            fig_median_delta_rsi.add_trace(go.Scatter(
                x=significant_data['RSI_Threshold'],
                y=significant_data['Median_Return_Delta'],
                mode='markers',
                name='Significant Signals',
                marker=dict(color='green', size=8),
                hovertemplate='<b>RSI %{x}</b><br>' +
                            'Return Delta (Median): %{y:.3%}<br>' +
                            'Target Median: %{customdata[0]:.3%}<br>' +
                            'Benchmark Median: %{customdata[1]:.3%}<br>' +
                            'Significant: ✓<extra></extra>',
                customdata=significant_data[['Median_Return', 'Benchmark_Median_Return']].values
            ))
        
        # Add points for non-significant signals (red)
        non_significant_data = original_filtered_data[original_filtered_data['significant'] == False]
        if not non_significant_data.empty:
            fig_median_delta_rsi.add_trace(go.Scatter(
                x=non_significant_data['RSI_Threshold'],
                y=non_significant_data['Median_Return_Delta'],
                mode='markers',
                name='Non-Significant Signals',
                marker=dict(color='red', size=8),
                hovertemplate='<b>RSI %{x}</b><br>' +
                            'Return Delta (Median): %{y:.3%}<br>' +
                            'Target Median: %{customdata[0]:.3%}<br>' +
                            'Benchmark Median: %{customdata[1]:.3%}<br>' +
                            'Significant: ✗<extra></extra>',
                customdata=non_significant_data[['Median_Return', 'Benchmark_Median_Return']].values
            ))
        
        # Add reference line at y=0
        fig_median_delta_rsi.add_hline(y=0, line_dash="dash", line_color="gray", 
                                      annotation_text="No Difference")
        
        fig_median_delta_rsi.update_layout(
            title="Return Delta (Median: Target - Benchmark) vs RSI Threshold",
            xaxis_title="RSI Threshold",
            yaxis_title="Return Delta (Median) (%)",
            hovermode='closest',
            xaxis=dict(range=[0, 100]),
            yaxis=dict(tickformat='.1%'),
            showlegend=True
        )
        
        st.plotly_chart(fig_median_delta_rsi, use_container_width=True, key="median_delta_rsi_chart")
        

        
        # RSI vs Sortino Ratio Chart
        st.subheader("📊 RSI Threshold vs Sortino Ratio")
        st.info("💡 **What this shows:** This chart displays how the Sortino ratio (risk-adjusted return) varies across different RSI thresholds. Higher Sortino ratios indicate better risk-adjusted performance. Look for peaks in the chart to identify optimal RSI thresholds.")
        
        fig_sortino_rsi = go.Figure()
        
        # Add points for significant signals (green)
        significant_data = valid_signals[valid_signals['significant'] == True]
        if not significant_data.empty:
            fig_sortino_rsi.add_trace(go.Scatter(
                x=significant_data['RSI_Threshold'],
                y=significant_data['Sortino_Ratio'],
                mode='markers',
                name='Significant Signals',
                marker=dict(color='green', size=8),
                line=dict(width=0),  # Explicitly disable lines
                hovertemplate='<b>RSI %{x}</b><br>' +
                            'Sortino Ratio: %{y:.2f}<br>' +
                            'Significant: ✓<extra></extra>'
            ))
        
        # Add points for non-significant signals (red)
        non_significant_data = valid_signals[valid_signals['significant'] == False]
        if not non_significant_data.empty:
            fig_sortino_rsi.add_trace(go.Scatter(
                x=non_significant_data['RSI_Threshold'],
                y=non_significant_data['Sortino_Ratio'],
                mode='markers',
                name='Non-Significant Signals',
                marker=dict(color='red', size=8),
                line=dict(width=0),  # Explicitly disable lines
                hovertemplate='<b>RSI %{x}</b><br>' +
                            'Sortino Ratio: %{y:.2f}<br>' +
                            'Significant: ✗<extra></extra>'
            ))
        
        # Add reference line at y=0
        fig_sortino_rsi.add_hline(y=0, line_dash="dash", line_color="gray", 
                                 annotation_text="No Risk-Adjusted Return")
        
        fig_sortino_rsi.update_layout(
            title="Sortino Ratio vs RSI Threshold",
            xaxis_title="RSI Threshold",
            yaxis_title="Sortino Ratio",
            hovermode='closest',
            xaxis=dict(range=[0, 100]),
            showlegend=True
        )
        
        st.plotly_chart(fig_sortino_rsi, use_container_width=True, key="sortino_rsi_chart")
        
        # RSI vs Cumulative Return Chart
        st.subheader("📊 RSI Threshold vs Cumulative Return")
        st.info("💡 **What this shows:** This chart displays how the total cumulative return varies across different RSI thresholds. Higher cumulative returns **may** indicate better overall performance. Look for peaks in the chart to identify potential RSI thresholds of interest.")
        
        # Use original numerical data for consistency
        original_filtered_data = st.session_state['results_df'][st.session_state['results_df']['RSI_Threshold'].isin(filtered_df['RSI_Threshold'])]
        original_significant_data = original_filtered_data[original_filtered_data['significant'] == True]
        original_non_significant_data = original_filtered_data[original_filtered_data['significant'] == False]
        
        fig_return_rsi = go.Figure()
        
        # Add points for significant signals (green)
        if not original_significant_data.empty:
            fig_return_rsi.add_trace(go.Scatter(
                x=original_significant_data['RSI_Threshold'],
                y=original_significant_data['Total_Return'],
                mode='markers',
                name='Significant Signals',
                marker=dict(color='green', size=8),
                line=dict(width=0),  # Explicitly disable lines
                hovertemplate='<b>RSI %{x}</b><br>' +
                            'Cumulative Return: %{y:.3%}<br>' +
                            'Significant: ✓<extra></extra>'
            ))
        
        # Add points for non-significant signals (red)
        if not original_non_significant_data.empty:
            fig_return_rsi.add_trace(go.Scatter(
                x=original_non_significant_data['RSI_Threshold'],
                y=original_non_significant_data['Total_Return'],
                mode='markers',
                name='Non-Significant Signals',
                marker=dict(color='red', size=8),
                line=dict(width=0),  # Explicitly disable lines
                hovertemplate='<b>RSI %{x}</b><br>' +
                            'Cumulative Return: %{y:.3%}<br>' +
                            'Significant: ✗<extra></extra>'
            ))
        
        # Add reference line at y=0
        fig_return_rsi.add_hline(y=0, line_dash="dash", line_color="gray", 
                                annotation_text="No Return")
        
        fig_return_rsi.update_layout(
            title="Cumulative Return vs RSI Threshold",
            xaxis_title="RSI Threshold",
            yaxis_title="Cumulative Return (%)",
            hovermode='closest',
            xaxis=dict(range=[0, 100]),
            yaxis=dict(tickformat='.1%'),
            showlegend=True
        )
        
        st.plotly_chart(fig_return_rsi, use_container_width=True, key="return_rsi_chart")
        
        # RSI vs Max Drawdown Chart
        st.subheader("📊 RSI Threshold vs Max Drawdown")
        st.info("💡 **What this shows:** This chart displays how the maximum drawdown (equity curve drawdown) varies across different RSI thresholds. Lower drawdown values indicate better risk management. Look for valleys in the chart to identify RSI thresholds with lower risk.")
        
        fig_drawdown_rsi = go.Figure()
        
        # Add points for significant signals (green)
        if not original_significant_data.empty:
            fig_drawdown_rsi.add_trace(go.Scatter(
                x=original_significant_data['RSI_Threshold'],
                y=original_significant_data['max_drawdown'],
                mode='markers',
                name='Significant Signals',
                marker=dict(color='green', size=8),
                line=dict(width=0),  # Explicitly disable lines
                hovertemplate='<b>RSI %{x}</b><br>' +
                            'Max Drawdown: %{y:.1f}<br>' +
                            'Significant: ✓<extra></extra>'
            ))
        
        # Add points for non-significant signals (red)
        if not original_non_significant_data.empty:
            fig_drawdown_rsi.add_trace(go.Scatter(
                x=original_non_significant_data['RSI_Threshold'],
                y=original_non_significant_data['max_drawdown'],
                mode='markers',
                name='Non-Significant Signals',
                marker=dict(color='red', size=8),
                line=dict(width=0),  # Explicitly disable lines
                hovertemplate='<b>RSI %{x}</b><br>' +
                            'Max Drawdown: %{y:.1f}<br>' +
                            'Significant: ✗<extra></extra>'
            ))
        
        # Add reference line at y=0
        fig_drawdown_rsi.add_hline(y=0, line_dash="dash", line_color="gray", 
                                  annotation_text="No Loss")
        
        fig_drawdown_rsi.update_layout(
            title="Max Drawdown vs RSI Threshold",
            xaxis_title="RSI Threshold",
            yaxis_title="Max Drawdown",
            hovermode='closest',
            xaxis=dict(range=[0, 100]),
            showlegend=True
        )
        
        st.plotly_chart(fig_drawdown_rsi, use_container_width=True, key="drawdown_rsi_chart")
        
        # Top significant signals
        if len(significant_signals) > 0:
            st.subheader("🏆 Top Statistically Significant Signals")
            
            # Sort by total return (highest cumulative return) instead of confidence level
            # Use the original results_df for sorting since it has numerical values
            original_significant_signals = st.session_state['results_df'][st.session_state['results_df']['significant'] == True].copy()
            top_significant = original_significant_signals.nlargest(5, 'Total_Return')
            

            
            # Highest Sortino Significant Signals Comparison
            st.subheader("📊 Highest Sortino Significant Signals Comparison")
            st.info(f"💡 **What this shows:** This chart compares the top 5 signals with the highest Sortino ratios (best risk-adjusted returns) among statistically significant signals against {benchmark_name} buy-and-hold. Each line represents a different RSI threshold that showed significant outperformance with excellent risk-adjusted performance. The signals are ranked by Sortino ratio, showing the best risk-adjusted returns first.")
            
            # Sort by Sortino ratio (best risk-adjusted returns) instead of annualized return
            # Use the original results_df for sorting since it has numerical values
            original_significant_signals = st.session_state['results_df'][st.session_state['results_df']['significant'] == True].copy()
            top_sortino_significant = original_significant_signals.nlargest(5, 'Sortino_Ratio')
            
            # Create comparison chart with highest Sortino signals
            fig_sortino_comparison = go.Figure()
            
            # Add benchmark buy-and-hold
            fig_sortino_comparison.add_trace(go.Scatter(
                x=benchmark.index,
                y=benchmark.values,
                mode='lines',
                name=f"{benchmark_name} Buy & Hold",
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Add significant signals with highest Sortino ratios and their corresponding benchmark curves
            colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
            for i, (idx, row) in enumerate(top_sortino_significant.iterrows()):
                # Debug: Check if equity curve exists
                if 'equity_curve' in row and row['equity_curve'] is not None:
                    color = colors[i % len(colors)]
                    
                    # Add strategy equity curve
                    fig_sortino_comparison.add_trace(go.Scatter(
                        x=row['equity_curve'].index,
                        y=row['equity_curve'].values,
                        mode='lines',
                        name=f"RSI {row['RSI_Threshold']} Strategy (Cumulative: {row['Total_Return']:.3%}, Annualized: {row['annualized_return']:.3%}, Sortino: {row['Sortino_Ratio']:.2f})",
                        line=dict(color=color, width=2)
                    ))
                    
                    # Add corresponding benchmark equity curve under same conditions
                    # We need to calculate the benchmark equity curve for this specific RSI threshold
                    signal_data = st.session_state.get('signal_data')
                    benchmark_data = st.session_state.get('benchmark_data')
                    rsi_period = st.session_state.get('rsi_period', 14)
                    comparison = st.session_state.get('comparison', 'less_than')
                    
                    if signal_data is not None and benchmark_data is not None:
                        # Calculate RSI for the signal
                        signal_rsi = calculate_rsi(signal_data, window=rsi_period, method="wilders")
                        
                        # Generate buy signals for benchmark (same as strategy)
                        if comparison == "less_than":
                            benchmark_signals = (signal_rsi <= row['RSI_Threshold']).astype(int)
                        else:  # greater_than
                            benchmark_signals = (signal_rsi >= row['RSI_Threshold']).astype(int)
                        
                        # Calculate benchmark equity curve using benchmark prices (same logic as strategy)
                        benchmark_equity_curve = pd.Series(1.0, index=benchmark_data.index)
                        current_equity = 1.0
                        in_position = False
                        entry_equity = 1.0
                        entry_price = None
                        
                        for date in benchmark_data.index:
                            current_signal = benchmark_signals[date] if date in benchmark_signals.index else 0
                            current_price = benchmark_data[date]
                            
                            if current_signal == 1 and not in_position:
                                # Enter position
                                in_position = True
                                entry_equity = current_equity
                                entry_price = current_price
                                
                            elif current_signal == 0 and in_position:
                                # Exit position
                                trade_return = (current_price - entry_price) / entry_price
                                current_equity = entry_equity * (1 + trade_return)
                                in_position = False
                            
                            # Update equity curve
                            if in_position:
                                current_equity = entry_equity * (current_price / entry_price)
                            
                            benchmark_equity_curve[date] = current_equity
                        
                        # Handle case where we're still in position at the end
                        if in_position:
                            final_price = benchmark_data.iloc[-1]
                            trade_return = (final_price - entry_price) / entry_price
                            current_equity = entry_equity * (1 + trade_return)
                            benchmark_equity_curve.iloc[-1] = current_equity
                        
                        # Add benchmark equity curve under same conditions
                        fig_sortino_comparison.add_trace(go.Scatter(
                            x=benchmark_equity_curve.index,
                            y=benchmark_equity_curve.values,
                            mode='lines',
                            name=f"RSI {row['RSI_Threshold']} Benchmark (same conditions)",
                            line=dict(color=color, width=1, dash='dot'),
                            visible='legendonly'  # Hidden by default
                        ))
                else:
                    st.warning(f"No equity curve found for RSI {row['RSI_Threshold']}")
            
            # Find the shortest time period among visible curves for default scaling
            shortest_period = None
            shortest_duration = float('inf')
            
            # Check strategy curves (these are always visible)
            for i, (idx, row) in enumerate(top_sortino_significant.iterrows()):
                if 'equity_curve' in row and row['equity_curve'] is not None:
                    curve_duration = (row['equity_curve'].index[-1] - row['equity_curve'].index[0]).days
                    if curve_duration < shortest_duration:
                        shortest_duration = curve_duration
                        shortest_period = row['equity_curve']
            
            # If no strategy curves found, use benchmark
            if shortest_period is None:
                shortest_period = benchmark
            
            fig_sortino_comparison.update_layout(
                title=f"Highest Sortino Significant Signals Comparison vs {benchmark_name}",
                xaxis_title="Date",
                yaxis_title="Equity Value",
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                xaxis=dict(range=[shortest_period.index[0], shortest_period.index[-1]])  # Scale to shortest period
            )
            st.plotly_chart(fig_sortino_comparison, use_container_width=True, key="highest_sortino_comparison")
            
        else:
            st.warning("No signals reached statistical significance (p < 0.05)")
        

        
        # Note: QuantStats detailed reports removed to avoid import issues
        # Basic QuantStats metrics are still available in the main results table

    # Statistical interpretation guide
    with st.expander("📚 Statistical Significance Guide"):
        st.write("""
        **Understanding Statistical Significance:**
        
        - **Confidence Level**: Percentage confidence that the signal outperforms the benchmark **under the same RSI conditions**
        - **P-value**: Probability of getting these results by chance (lower is better)
        - **Effect Size**: Magnitude of the difference (Cohen's d)
        - **Significant**: P-value < 0.05 (95% confidence level)
        
        **What This Measures:**
        The confidence level compares your signal (buying/selling the target ticker based on signal RSI) 
        vs. buying/selling the benchmark based on the **same signal RSI conditions**. This ensures a fair comparison 
        of whether your target ticker choice is better than the benchmark when the same RSI signals are applied.
        
        **Interpretation:**
        - ✓ **Significant**: Strong evidence your target ticker beats the benchmark under these RSI conditions
        - ✗ **Not Significant**: Results could be due to chance
        - **Effect Size**: 
          - Small: 0.2-0.5
          - Medium: 0.5-0.8  
          - Large: > 0.8
        
        **Key Metrics Explained:**
        
        **📊 Performance Metrics:**
        - **Total Return**: How much money you would have made (or lost) over the entire period
        - **Annualized Return**: The yearly return rate, useful for comparing signals over different time periods
        - **Win Rate**: Percentage of trades that were profitable
        - **Total Trades**: Number of buy/sell transactions the signal made
        - **Sortino Ratio**: Risk-adjusted return measure (higher is better, focuses on downside risk)
        - **Avg Hold Days**: Average number of days the signal held each position
        
        **📈 Statistical Metrics:**
        - **Confidence Level**: How certain we are that the signal beats the benchmark (higher % = more certain)
        - **P-value**: Probability the results happened by chance (lower = more significant)
        - **Effect Size**: How much better/worse the signal is compared to the benchmark
        - **T-statistic**: Statistical measure of the difference between signal and benchmark
        - **Power**: How likely the test is to detect a real difference if one exists
        
        **🎯 What to Look For:**
        - **High Confidence (>95%)**: Very strong evidence the signal works
        - **Low P-value (<0.05)**: Results are statistically significant
        - **Positive Effect Size**: Signal outperforms the benchmark
        - **High Win Rate**: Signal wins more often than it loses
        - **Good Sortino Ratio**: Signal has good risk-adjusted returns
        """)

st.write("---")
st.write("💡 **Tip:** Try different ticker combinations and RSI conditions to find optimal signal thresholds")



# Display data quality messages at the bottom
if 'data_messages' in st.session_state and st.session_state['data_messages']:
    st.write("---")
    st.subheader("📊 Data Quality Information")
    for message in st.session_state['data_messages']:
        st.info(message)

# Footer
st.write("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
    <strong>RSI Threshold Validation Tool</strong><br>
    Questions? Reach out to @Gobi on Discord
</div>
""", unsafe_allow_html=True)
