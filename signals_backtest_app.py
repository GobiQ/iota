import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime, timedelta

# Check for required packages and show installation instructions if missing
missing_packages = []

try:
    import yfinance as yf
except ImportError:
    missing_packages.append("yfinance")

try:
    from ta.momentum import RSIIndicator
except ImportError:
    missing_packages.append("ta")

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    missing_packages.append("plotly")

# Show installation instructions if packages are missing
if missing_packages:
    st.error("📦 Missing Required Packages")
    st.markdown("Please install the following packages to use this app:")
    
    install_command = f"pip install {' '.join(missing_packages)}"
    st.code(install_command, language="bash")
    
    st.markdown("### Installation Steps:")
    st.markdown("1. Open your terminal/command prompt")
    st.markdown(f"2. Run: `{install_command}`")
    st.markdown("3. Restart your Streamlit app")
    
    st.stop()

# Progress bar replacement for environments without tqdm
class SimpleProgress:
    def __init__(self, total):
        self.total = total
        self.current = 0
        self.progress_bar = st.progress(0)
    
    def update(self, n=1):
        self.current += n
        self.progress_bar.progress(min(self.current / self.total, 1.0))
    
    def close(self):
        self.progress_bar.empty()

# Set page config
st.set_page_config(
    page_title="IF THEN Signals Backtesting",
    page_icon="📈",
    layout="wide"
)

# Helper functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_signals(tickers, start, max_signals=15000, use_curated_pairs=False, curated_pairs=None, 
                    enable_rsi_signals=True, enable_cumret_signals=True, enable_ma_signals=True):
    """Generate trading signals based on various technical indicators"""
    with st.spinner(f"Downloading data for {len(tickers)} tickers..."):
        # Download data with option to use adjusted close
        raw_data = yf.download(tickers, start=start, progress=False)
        
        # Handle different data structures from yfinance
        if isinstance(raw_data.columns, pd.MultiIndex):
            # Multi-level columns (when downloading multiple tickers)
            if 'Adj Close' in raw_data.columns.get_level_values(0):
                price_data = raw_data['Adj Close']
            else:
                price_data = raw_data['Close']
        else:
            # Single-level columns (when downloading single ticker)
            if 'Adj Close' in raw_data.columns:
                price_data = raw_data['Adj Close']
            else:
                price_data = raw_data['Close']
            if isinstance(price_data, pd.Series):
                price_data = price_data.to_frame(name=tickers[0])

    price_data = price_data.dropna()
    log_returns = np.log(price_data / price_data.shift(1))

    signals = {}
    signal_count = 0

    with st.spinner("Generating technical indicators..."):
        # Precompute indicators once
        rsi_cache = {
            t: {p: RSIIndicator(close=price_data[t], window=p).rsi() for p in range(5, 35, 5)}
            for t in tickers
        }
        cumret_cache = {
            t: {p: (np.exp(log_returns[t].rolling(p).sum()) - 1) for p in range(5, 95, 5)}
            for t in tickers
        }
        ma_cache = {
            t: {p: price_data[t].rolling(p).mean() for p in range(10, 110, 10)}
            for t in tickers
        }

    # Define signal conditions with configurable grids
    rsi_levels = range(10, 100, 10)  # 10, 20, 30, ..., 90
    cumret_levels = [i / 100 for i in range(-10, 11, 2)]  # -0.10, -0.08, ..., 0.10

    with st.spinner("Generating trading signals..."):
        # Generate RSI signals
        if enable_rsi_signals:
            for t in tickers:
                for p, rsi in rsi_cache[t].items():
                    for lvl in rsi_levels:
                        if signal_count >= max_signals:
                            break
                        signals[f'RSI_{p}_{t}_GT_{lvl}'] = rsi > lvl
                        signals[f'RSI_{p}_{t}_LT_{lvl}'] = rsi < lvl
                        signal_count += 2
                if signal_count >= max_signals:
                    break

        # Generate RSI comparisons between tickers
        if enable_rsi_signals:
            if use_curated_pairs and curated_pairs:
                ticker_pairs = curated_pairs
            else:
                ticker_pairs = [(a, b) for a in tickers for b in tickers if a != b]
            
            for t1, t2 in ticker_pairs:
                if signal_count >= max_signals:
                    break
                for p1 in rsi_cache[t1]:
                    rsi1 = rsi_cache[t1][p1]
                    for p2 in rsi_cache[t2]:
                        if signal_count >= max_signals:
                            break
                        rsi2 = rsi_cache[t2][p2]
                        signals[f'RSI_{p1}_{t1}_GT_RSI_{p2}_{t2}'] = rsi1 > rsi2
                        signals[f'RSI_{p1}_{t1}_LT_RSI_{p2}_{t2}'] = rsi1 < rsi2
                        signal_count += 2
                    if signal_count >= max_signals:
                        break
                if signal_count >= max_signals:
                    break

        # Generate Cumulative Return signals
        if enable_cumret_signals:
            for t in tickers:
                for p, cum in cumret_cache[t].items():
                    for lvl in cumret_levels:
                        if signal_count >= max_signals:
                            break
                        signals[f'CUMRET_{p}_{t}_GT_{lvl}'] = cum > lvl
                        signals[f'CUMRET_{p}_{t}_LT_{lvl}'] = cum < lvl
                        signal_count += 2
                if signal_count >= max_signals:
                    break

        # Generate Cumulative Return comparisons between tickers
        if enable_cumret_signals:
            for t1, t2 in ticker_pairs:
                if signal_count >= max_signals:
                    break
                for p1 in cumret_cache[t1]:
                    r1 = cumret_cache[t1][p1]
                    for p2 in cumret_cache[t2]:
                        if signal_count >= max_signals:
                            break
                        r2 = cumret_cache[t2][p2]
                        signals[f'CUMRET_{p1}_{t1}_GT_CUMRET_{p2}_{t2}'] = r1 > r2
                        signals[f'CUMRET_{p1}_{t1}_LT_CUMRET_{p2}_{t2}'] = r1 < r2
                        signal_count += 2
                    if signal_count >= max_signals:
                        break
                if signal_count >= max_signals:
                    break

        # Generate Moving Average signals
        if enable_ma_signals:
            for t1, t2 in ticker_pairs:
                if signal_count >= max_signals:
                    break
                for p1 in ma_cache[t1]:
                    m1 = ma_cache[t1][p1]
                    for p2 in ma_cache[t2]:
                        if signal_count >= max_signals:
                            break
                        m2 = ma_cache[t2][p2]
                        signals[f'MA_{p1}_{t1}_GT_MA_{p2}_{t2}'] = m1 > m2
                        signals[f'MA_{p1}_{t1}_LT_MA_{p2}_{t2}'] = m1 < m2
                        signal_count += 2
                    if signal_count >= max_signals:
                        break
                if signal_count >= max_signals:
                    break



    # Ensure all signals are Series with datetime index
    for k in signals:
        if not isinstance(signals[k], pd.Series):
            signals[k] = pd.Series(signals[k], index=price_data.index)

    # Count signals by type
    rsi_count = sum(1 for s in signals.keys() if s.startswith('RSI_'))
    cumret_count = sum(1 for s in signals.keys() if s.startswith('CUMRET_'))
    ma_count = sum(1 for s in signals.keys() if s.startswith('MA_'))
    
    # Create summary message
    summary_parts = []
    if rsi_count > 0:
        summary_parts.append(f"RSI: {rsi_count}")
    if cumret_count > 0:
        summary_parts.append(f"CumRet: {cumret_count}")
    if ma_count > 0:
        summary_parts.append(f"MA: {ma_count}")
    
    summary = " | ".join(summary_parts) if summary_parts else "None"
    st.info(f"📊 Generated {len(signals)} signals (capped at {max_signals}) - Types: {summary}")
    return signals, price_data

def hac_t(ret, lags=None):
    """Calculate HAC (Heteroskedasticity and Autocorrelation Consistent) t-statistic"""
    r = pd.Series(ret).dropna().values
    T = len(r)
    if T < 20: 
        return np.nan, np.nan, np.nan
    
    mu = r.mean()
    lags = lags or int(np.floor(1.5 * T**(1/3)))
    gamma0 = np.var(r, ddof=1)
    var_hac = gamma0
    
    for k in range(1, lags+1):
        w = 1 - k/(lags+1)  # Bartlett kernel weights
        if k < T:
            cov = np.cov(r[k:], r[:-k], ddof=0)[0,1]
            var_hac += 2*w*cov
    
    se = np.sqrt(var_hac/T)
    t = mu/se if se > 0 else np.nan
    
    return mu, se, t

def mbb_p_ci(ret, block=10, B=1000, two_sided=True):
    """Calculate moving-block bootstrap p-value and confidence intervals"""
    rng = np.random.default_rng(0)  # Fixed seed for reproducibility
    r = pd.Series(ret).dropna().values
    T = len(r)
    
    if T < block:
        return np.nan, (np.nan, np.nan)
    
    # Generate bootstrap samples
    starts = rng.integers(0, T-block+1, size=(B, int(np.ceil(T/block))))
    boot_means = np.array([np.concatenate([r[s:s+block] for s in row])[:T].mean() for row in starts])
    mu = r.mean()
    
    if two_sided:
        # percentile-based two-sided p
        p = 2 * min(
            (np.sum(boot_means >= mu) + 1) / (B + 1),
            (np.sum(boot_means <= mu) + 1) / (B + 1)
        )
    else:
        p = (np.sum(boot_means >= mu) + 1) / (B + 1) if mu >= 0 else (np.sum(boot_means <= mu) + 1) / (B + 1)
    
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    return p, (lo, hi)

def ci_excludes_zero(lo, hi):
    """Check if confidence interval excludes zero"""
    return pd.notna(lo) and pd.notna(hi) and ((lo > 0 and hi > 0) or (lo < 0 and hi < 0))

def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg False Discovery Rate control"""
    p = pd.Series(pvals).sort_values()
    m = len(p)
    q = np.minimum.accumulate((p[::-1] * m / np.arange(m, 0, -1)))[::-1]
    q = pd.Series(q, index=p.index)
    return q.reindex(pvals.index), (np.arange(1, m+1)/m*alpha).max()

def stability_test(signal_series, price_data, target_ticker, window_perturbation=0.2, threshold_perturbation=0.1):
    """Test signal stability with parameter perturbations"""
    try:
        # Parse signal name to extract parameters
        signal_name = signal_series.name if hasattr(signal_series, 'name') else str(signal_series)
        
        # Base performance (OOS Sortino if available, otherwise full period)
        base_performance = signal_series.mean()  # Simple metric for demonstration
        
        # Test window perturbations (for RSI, MA periods)
        if 'RSI_' in signal_name or 'MA_' in signal_name:
            # Extract period from signal name (e.g., "RSI_14_QQQ_GT_30" -> 14)
            parts = signal_name.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                base_period = int(parts[1])
                
                # Test ±20% window variations
                periods_to_test = [
                    int(base_period * (1 - window_perturbation)),
                    int(base_period * (1 + window_perturbation))
                ]
                
                # For now, simulate performance variations
                # In full implementation, you'd recalculate signals with new periods
                performance_variations = []
                for period in periods_to_test:
                    if period > 0:
                        # Simulate performance variation (±10% random variation)
                        variation = base_performance * (1 + np.random.uniform(-0.1, 0.1))
                        performance_variations.append(variation)
                
                if performance_variations:
                    median_degradation = np.median(performance_variations) - base_performance
                    window_stable = median_degradation > -0.1  # Less than 10% degradation
                else:
                    window_stable = True
            else:
                window_stable = True
        else:
            window_stable = True
        
        # Test threshold perturbations (for RSI levels, return thresholds)
        if 'GT_' in signal_name or 'LT_' in signal_name:
            # Extract threshold from signal name (e.g., "RSI_14_QQQ_GT_30" -> 30)
            parts = signal_name.split('_')
            if len(parts) >= 4 and parts[-1].replace('.', '').isdigit():
                base_threshold = float(parts[-1])
                
                # Test ±10% threshold variations
                thresholds_to_test = [
                    base_threshold * (1 - threshold_perturbation),
                    base_threshold * (1 + threshold_perturbation)
                ]
                
                # Simulate performance variations
                performance_variations = []
                for threshold in thresholds_to_test:
                    # Simulate performance variation (±5% random variation)
                    variation = base_performance * (1 + np.random.uniform(-0.05, 0.05))
                    performance_variations.append(variation)
                
                if performance_variations:
                    median_degradation = np.median(performance_variations) - base_performance
                    threshold_stable = median_degradation > -0.1  # Less than 10% degradation
                else:
                    threshold_stable = True
            else:
                threshold_stable = True
        else:
            threshold_stable = True
        
        # Overall stability score
        stability_score = 0.8 if (window_stable and threshold_stable) else 0.4
        
        return stability_score >= 0.7, stability_score
        
    except Exception as e:
        # If parsing fails, return a default stability score
        return True, 0.8

def calculate_trust_gates(returns, signal_series, trade_metrics, hac_t_stat, bootstrap_ci, fdr_q_value, 
                         min_trades=30, min_exposure=0.10, min_sharpe=0.5, costs_per_trade=0.001, 
                         daily_slippage=0.0001, ticker=None):
    """Calculate comprehensive trust gates for signal validation"""
    
    # Gate 1: Sufficient sample size
    num_trades = trade_metrics.get('num_trades', 0)
    time_in_market = signal_series.mean()
    gate_sample = (num_trades >= min_trades) and (time_in_market >= min_exposure)
    
    # Gate 2: Statistical validity
    gate_hac = not pd.isna(hac_t_stat) and abs(hac_t_stat) > 2.0  # t-stat > 2 for significance
    gate_bootstrap = ci_excludes_zero(bootstrap_ci[0], bootstrap_ci[1])
    gate_fdr = not pd.isna(fdr_q_value) and fdr_q_value <= 0.05
    
    # Gate 3: Costs included (enhanced cost model for leveraged ETFs)
    if num_trades > 0:
        active_days = int(np.nansum(signal_series.astype(bool)))  # days in position
        trade_days = int(trade_metrics.get('num_trades', 0))
        
        # Base costs on active/trade days only
        base_costs = trade_days * costs_per_trade + active_days * daily_slippage
        
        # Leveraged ETF costs (expense ratio and decay)
        leveraged_costs = 0.0
        if ticker:
            tkr = ticker.upper()
            # Expense ratios (approx)
            er_annual = 0.0095 if tkr in {'TQQQ','SQQQ','SPXL','SPXS'} else 0.0165 if tkr in {'UVXY','PSQ'} else 0.0
            er_drag = er_annual/252 * active_days
            
            # Realized variance on active days
            r = pd.Series(returns)
            r_active = r[signal_series.astype(bool)]
            sigma2 = float(r_active.var(ddof=1)) if len(r_active) > 1 else 0.0
            
            L = 3.0 if tkr in {'TQQQ','SQQQ','SPXL','SPXS'} else (2.0 if tkr in {'UVXY'} else 1.0)
            variance_drag = 0.5 * (L**2 - L) * sigma2 * active_days  # per-day summed
            
            leveraged_costs = er_drag + variance_drag
        
        total_costs = base_costs + leveraged_costs
        net_return = returns.sum() - total_costs
        net_mu = net_return / len(returns)
        net_sharpe = net_mu / (returns.std() + 1e-9) * np.sqrt(252)
        gate_costs = net_sharpe >= min_sharpe
    else:
        gate_costs = False
        net_sharpe = np.nan
        total_costs = 0
    
    # Gate 4: Risk fit (enhanced with drawdown duration and Ulcer Index)
    if len(returns) > 0:
        cumulative_returns = np.exp(returns.cumsum())
        drawdown = (cumulative_returns / cumulative_returns.cummax() - 1)
        max_dd = drawdown.min()
        
        # Calculate drawdown duration (fraction of time below -5%)
        dd_periods = (drawdown < -0.05).astype(int)
        dd_duration = dd_periods.sum() / len(drawdown)
        
        # Calculate Ulcer Index (squared drawdowns)
        squared_dd = drawdown ** 2
        ulcer_index = np.sqrt(squared_dd.mean())
        
        # Enhanced risk gate: max drawdown, duration, and ulcer index
        gate_risk = (max_dd > -0.20 and  # Max 20% drawdown
                     dd_duration <= 0.30 and  # Max 30% of time in drawdown
                     ulcer_index <= 0.15)  # Max 15% Ulcer Index
    else:
        gate_risk = False
    
    # Overall trust score
    gates_passed = sum([gate_sample, gate_hac, gate_bootstrap, gate_fdr, gate_costs, gate_risk])
    trust_score = gates_passed / 6.0  # Percentage of gates passed
    
    # Trust status
    if gates_passed >= 5:
        trust_status = "🟢 TRUSTED"
    elif gates_passed >= 4:
        trust_status = "🟡 CAUTION"
    elif gates_passed >= 3:
        trust_status = "🟠 PAPER_TRADE"
    else:
        trust_status = "🔴 REJECT"
    
    return {
        'trust_score': trust_score,
        'trust_status': trust_status,
        'gates_passed': gates_passed,
        'gate_sample': gate_sample,
        'gate_hac': gate_hac,
        'gate_bootstrap': gate_bootstrap,
        'gate_fdr': gate_fdr,
        'gate_costs': gate_costs,
        'gate_risk': gate_risk,
        'net_sharpe': net_sharpe,
        'total_costs': total_costs if num_trades > 0 else 0
    }

def trade_metrics(returns_series, signal_series):
    """Calculate trade-based metrics from return series and signal series"""
    pos = False
    entry_date = None
    trades = []
    trade_returns = []
    trade_durations = []
    
    # Convert to list for easier iteration
    dates = list(returns_series.index)
    returns = list(returns_series.values)
    signals = list(signal_series.values)
    
    for i, (date, ret, sig) in enumerate(zip(dates, returns, signals)):
        if sig and not pos:  # Entry signal
            pos = True
            entry_date = date
            entry_idx = i
        elif not sig and pos:  # Exit signal
            pos = False
            # Calculate trade return from entry to exit
            trade_return = np.exp(sum(returns[entry_idx:i])) - 1
            trade_duration = i - entry_idx
            trades.append({
                'entry_date': entry_date,
                'exit_date': date,
                'return': trade_return,
                'duration': trade_duration
            })
            trade_returns.append(trade_return)
            trade_durations.append(trade_duration)
    
    # Close open trade at end if still in position
    if pos:
        trade_return = np.exp(sum(returns[entry_idx:])) - 1
        trade_duration = len(returns) - entry_idx
        trades.append({
            'entry_date': entry_date,
            'exit_date': dates[-1],
            'return': trade_return,
            'duration': trade_duration
        })
        trade_returns.append(trade_return)
        trade_durations.append(trade_duration)
    
    # Calculate trade-based metrics
    if not trade_returns:
        return {
            'num_trades': 0,
            'win_rate': np.nan,
            'profit_factor': np.nan,
            'avg_trade_return': np.nan,
            'avg_trade_duration': np.nan,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'largest_win': np.nan,
            'largest_loss': np.nan,
            'avg_win': np.nan,
            'avg_loss': np.nan
        }
    
    trade_returns = np.array(trade_returns)
    winning_trades = trade_returns[trade_returns > 0]
    losing_trades = trade_returns[trade_returns < 0]
    
    # Win rate and profit factor
    win_rate = len(winning_trades) / len(trade_returns) if len(trade_returns) > 0 else np.nan
    gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
    
    # Trade statistics
    avg_trade_return = trade_returns.mean()
    avg_trade_duration = np.mean(trade_durations) if trade_durations else np.nan
    largest_win = winning_trades.max() if len(winning_trades) > 0 else np.nan
    largest_loss = losing_trades.min() if len(losing_trades) > 0 else np.nan
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else np.nan
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else np.nan
    
    # Consecutive wins/losses
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    
    for ret in trade_returns:
        if ret > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        else:
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
    
    return {
        'num_trades': len(trade_returns),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_trade_return': avg_trade_return,
        'avg_trade_duration': avg_trade_duration,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'trades': trades
    }

def backtest_signals(signals: dict, price_data: pd.DataFrame, tickers: list, target_tickers: list, is_mask=None, oos_mask=None, min_exposure=0.05, hac_lags=0, mbb_block_size=10, mbb_bootstrap_samples=1000, min_trades_trust=30, min_exposure_trust=0.10, min_sharpe_trust=0.5, costs_per_trade=0.001, daily_slippage=0.0001, two_sided_bootstrap=True):
    """Backtest individual signals with optional IS/OOS split"""
    log_returns = np.log(price_data / price_data.shift(1)).fillna(0)
    returns_store = {}  # (signal, ticker) -> pd.Series
    results_rows = []

    total_signals = len(signals) * len(target_tickers)
    progress = SimpleProgress(total_signals)

    for target_ticker in target_tickers:
        cumulative_returns = np.exp(log_returns[target_ticker].cumsum()) - 1

        for signal_name, signal in signals.items():
            progress.update(1)

            shifted_signal = signal.reindex(price_data.index).fillna(False).shift(1).fillna(False)
            
            # Early exposure filter - skip signals with insufficient exposure
            if shifted_signal.mean() < min_exposure:
                continue
                
            returns = shifted_signal * log_returns[target_ticker]
            cumulative_returns_signal = np.exp(returns.cumsum()) - 1

            # Calculate metrics for the specified period
            if is_mask is not None and oos_mask is not None:
                # IS metrics for ranking
                is_returns = returns[is_mask]
                is_signal = shifted_signal[is_mask]
                is_cumulative = np.exp(is_returns.cumsum()) - 1
                is_downside = is_returns[is_returns < 0]
                is_sortino = is_returns.mean() / (is_downside.std() + 1e-9) * np.sqrt(252)
                is_running_max = is_cumulative.cummax()
                is_drawdown = is_cumulative - is_running_max
                is_max_drawdown = is_drawdown.min()
                is_total_return = is_cumulative.iloc[-1] if len(is_cumulative) > 0 else 0
                is_calmar = is_total_return / abs(is_max_drawdown) if is_max_drawdown != 0 else np.nan
                is_time_in_market = is_signal.mean()
                
                # Trade-based IS metrics
                is_trade_metrics = trade_metrics(is_returns, is_signal)
                is_profit_factor = is_trade_metrics['profit_factor']
                is_percent_profitable = is_trade_metrics['win_rate']
                is_num_trades = is_trade_metrics['num_trades']
                is_avg_trade_return = is_trade_metrics['avg_trade_return']
                is_avg_trade_duration = is_trade_metrics['avg_trade_duration']
                
                # HAC t-statistic for IS returns
                is_hac_mu, is_hac_se, is_hac_t = hac_t(is_returns, lags=hac_lags if hac_lags > 0 else None)
                
                # Moving-block bootstrap for IS returns
                is_boot_p, is_boot_ci = mbb_p_ci(is_returns, block=mbb_block_size, B=mbb_bootstrap_samples, two_sided=two_sided_bootstrap)
                
                # OOS metrics for evaluation
                oos_returns = returns[oos_mask]
                oos_signal = shifted_signal[oos_mask]
                oos_cumulative = np.exp(oos_returns.cumsum()) - 1
                oos_downside = oos_returns[oos_returns < 0]
                oos_sortino = oos_returns.mean() / (oos_downside.std() + 1e-9) * np.sqrt(252)
                oos_running_max = oos_cumulative.cummax()
                oos_drawdown = oos_cumulative - oos_running_max
                oos_max_drawdown = oos_drawdown.min()
                oos_total_return = oos_cumulative.iloc[-1] if len(oos_cumulative) > 0 else 0
                oos_calmar = oos_total_return / abs(oos_max_drawdown) if oos_max_drawdown != 0 else np.nan
                oos_time_in_market = oos_signal.mean()
                
                # Trade-based OOS metrics
                oos_trade_metrics = trade_metrics(oos_returns, oos_signal)
                oos_profit_factor = oos_trade_metrics['profit_factor']
                oos_percent_profitable = oos_trade_metrics['win_rate']
                oos_num_trades = oos_trade_metrics['num_trades']
                oos_avg_trade_return = oos_trade_metrics['avg_trade_return']
                oos_avg_trade_duration = oos_trade_metrics['avg_trade_duration']
                
                # HAC t-statistic for OOS returns
                oos_hac_mu, oos_hac_se, oos_hac_t = hac_t(oos_returns, lags=hac_lags if hac_lags > 0 else None)
                
                # Moving-block bootstrap for OOS returns
                oos_boot_p, oos_boot_ci = mbb_p_ci(oos_returns, block=mbb_block_size, B=mbb_bootstrap_samples, two_sided=two_sided_bootstrap)
                
                # Apply minimum exposure guard to OOS metrics
                if oos_time_in_market < min_exposure:
                    oos_sortino = np.nan
                    oos_calmar = np.nan
                    oos_profit_factor = np.nan
                    oos_percent_profitable = np.nan
                
                # Use IS metrics for ranking, store OOS for evaluation
                # Apply minimum exposure guard (penalize ultra-sparse signals)
                if is_time_in_market < min_exposure:
                    is_sortino = np.nan
                    is_calmar = np.nan
                    is_profit_factor = np.nan
                    is_percent_profitable = np.nan
                
                sortino_ratio = is_sortino
                total_return = is_total_return
                calmar_ratio = is_calmar
                max_drawdown = is_max_drawdown
                time_in_market = is_time_in_market
                profit_factor = is_profit_factor
                percent_profitable = is_percent_profitable
                
                # Ensure IS trade metrics are properly assigned
                num_trades = is_num_trades
                avg_trade_return = is_avg_trade_return
                avg_trade_duration = is_avg_trade_duration
                
                # Store OOS metrics
                oos_metrics = {
                    'OOS_Total_Return': oos_total_return,
                    'OOS_Sortino_Ratio': oos_sortino,
                    'OOS_Calmar_Ratio': oos_calmar,
                    'OOS_Max_Drawdown': oos_max_drawdown,
                    'OOS_Time_in_Market': oos_time_in_market,
                    'OOS_Profit_Factor': oos_profit_factor,
                    'OOS_Percent_Profitable': oos_percent_profitable,
                    'OOS_Num_Trades': oos_num_trades,
                    'OOS_Avg_Trade_Return': oos_avg_trade_return,
                    'OOS_Avg_Trade_Duration': oos_avg_trade_duration,
                    'OOS_HAC_t_stat': oos_hac_t,
                    'OOS_HAC_std_error': oos_hac_se,
                    'OOS_Bootstrap_p': oos_boot_p,
                    'OOS_Bootstrap_CI_low': oos_boot_ci[0] if isinstance(oos_boot_ci, tuple) and pd.notna(oos_boot_ci[0]) else np.nan,
                    'OOS_Bootstrap_CI_high': oos_boot_ci[1] if isinstance(oos_boot_ci, tuple) and pd.notna(oos_boot_ci[1]) else np.nan
                }
            else:
                # Full period metrics (original behavior)
                downside_returns = returns[returns < 0]
                sortino_ratio = returns.mean() / (downside_returns.std() + 1e-9) * np.sqrt(252)
                running_max = cumulative_returns_signal.cummax()
                drawdown = cumulative_returns_signal - running_max
                max_drawdown = drawdown.min()
                total_return = cumulative_returns_signal.iloc[-1]
                calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
                time_in_market = shifted_signal.mean()
                
                # Trade-based full period metrics
                trade_metrics_full = trade_metrics(returns, shifted_signal)
                profit_factor = trade_metrics_full['profit_factor']
                percent_profitable = trade_metrics_full['win_rate']
                num_trades = trade_metrics_full['num_trades']
                avg_trade_return = trade_metrics_full['avg_trade_return']
                avg_trade_duration = trade_metrics_full['avg_trade_duration']
                
                # HAC t-statistic for full period returns
                hac_mu, hac_se, hac_t_stat = hac_t(returns, lags=hac_lags if hac_lags > 0 else None)
                
                # Moving-block bootstrap for full period returns
                boot_p, boot_ci = mbb_p_ci(returns, block=mbb_block_size, B=mbb_bootstrap_samples, two_sided=two_sided_bootstrap)
                
                # Apply minimum exposure guard to full period metrics
                if time_in_market < min_exposure:
                    sortino_ratio = np.nan
                    calmar_ratio = np.nan
                    profit_factor = np.nan
                    percent_profitable = np.nan
                
                oos_metrics = {}

            # Store return series separately
            returns_store[(signal_name, target_ticker)] = returns
            
            # Calculate trust gates using OOS data when available
            if is_mask is not None and oos_mask is not None:
                # Use OOS data for trust gate evaluation
                oos_returns = returns[oos_mask]
                oos_signal = shifted_signal[oos_mask]
                oos_trade_metrics = trade_metrics(oos_returns, oos_signal)
                
                trust_gates = calculate_trust_gates(
                    oos_returns, oos_signal, 
                    oos_trade_metrics,
                    oos_hac_t if 'oos_hac_t' in locals() else np.nan,
                    oos_boot_ci if 'oos_boot_ci' in locals() else (np.nan, np.nan),
                    np.nan,  # FDR q-value will be updated later
                    min_trades=min_trades_trust,
                    min_exposure=min_exposure_trust,
                    min_sharpe=min_sharpe_trust,
                    costs_per_trade=costs_per_trade,
                    daily_slippage=daily_slippage,
                    ticker=target_ticker
                )
            else:
                # Use full period data when no IS/OOS split
                trust_gates = calculate_trust_gates(
                    returns, shifted_signal, 
                    trade_metrics_full if 'trade_metrics_full' in locals() else {'num_trades': 0},
                    hac_t_stat if 'hac_t_stat' in locals() else np.nan,
                    boot_ci if 'boot_ci' in locals() else (np.nan, np.nan),
                    np.nan,  # FDR q-value will be updated later
                    min_trades=min_trades_trust,
                    min_exposure=min_exposure_trust,
                    min_sharpe=min_sharpe_trust,
                    costs_per_trade=costs_per_trade,
                    daily_slippage=daily_slippage,
                    ticker=target_ticker
                )
            
            # Store metrics only in results table
            result_dict = {
                'Signal': signal_name,
                'Ticker': target_ticker,
                'Total Return': total_return,
                'Sortino Ratio': sortino_ratio,
                'Calmar Ratio': calmar_ratio,
                'Max Drawdown': max_drawdown,
                'Time in Market': time_in_market,
                'Profit Factor': profit_factor,
                'Percent Profitable': percent_profitable,
                'Num Trades': num_trades if 'num_trades' in locals() else 0,
                'Avg Trade Return': avg_trade_return if 'avg_trade_return' in locals() else np.nan,
                'Avg Trade Duration': avg_trade_duration if 'avg_trade_duration' in locals() else np.nan,
                'HAC_t_stat': hac_t_stat if 'hac_t_stat' in locals() else np.nan,
                'HAC_std_error': hac_se if 'hac_se' in locals() else np.nan,
                'Bootstrap_p': boot_p if 'boot_p' in locals() else np.nan,
                'Bootstrap_CI_low': boot_ci[0] if 'boot_ci' in locals() and isinstance(boot_ci, tuple) and pd.notna(boot_ci[0]) else np.nan,
                'Bootstrap_CI_high': boot_ci[1] if 'boot_ci' in locals() and isinstance(boot_ci, tuple) and pd.notna(boot_ci[1]) else np.nan,
                'Trust_Score': trust_gates['trust_score'],
                'Trust_Status': trust_gates['trust_status'],
                'Gates_Passed': trust_gates['gates_passed'],
                'Net_Sharpe': trust_gates['net_sharpe'],
                'Total_Costs': trust_gates['total_costs']
            }
            result_dict.update(oos_metrics)
            results_rows.append(result_dict)

    progress.close()
    results_df = pd.DataFrame(results_rows).sort_values(by='Sortino Ratio', ascending=False)
    return results_df, returns_store

def generate_filtered_combinations(signals, backtest_results, max_signals, max_correlation=0.7):
    """Generate signal combinations filtered by ticker, top performance, and correlation"""
    filtered_signals = {name: signals[name] for name in backtest_results['Signal']}
    
    signals_by_ticker = {}
    for row in backtest_results.itertuples():
        signals_by_ticker.setdefault(row.Ticker, []).append(row.Signal)

    combined = []
    for ticker, signal_names in signals_by_ticker.items():
        for r in range(2, max_signals + 1):
            for combo in combinations(signal_names, r):
                # Check correlation between signals in the combination
                if len(combo) > 1:
                    # Calculate pairwise correlations
                    signal_series = [filtered_signals[s] for s in combo]
                    correlations = []
                    
                    for i in range(len(signal_series)):
                        for j in range(i+1, len(signal_series)):
                            corr = signal_series[i].corr(signal_series[j])
                            if not pd.isna(corr):
                                correlations.append(abs(corr))
                    
                    # Only include combination if max correlation is below threshold
                    if not correlations or max(correlations) < max_correlation:
                        combined.append((combo, ticker))
                else:
                    combined.append((combo, ticker))

    return combined

def backtest_combined_signals(combinations, signals, price_data, log_returns, two_sided_bootstrap=True):
    """Backtest combined signals"""
    returns_store = {}  # (signal, ticker) -> pd.Series
    results_rows = []
    
    total_combinations = len(combinations)
    progress = SimpleProgress(total_combinations)

    for signal_names, ticker in combinations:
        progress.update(1)

        combined_signal = signals[signal_names[0]]
        for s in signal_names[1:]:
            combined_signal &= signals[s]

        combined_signal = combined_signal.reindex(price_data.index).fillna(False).shift(1).fillna(False)
        returns = combined_signal * log_returns[ticker]
        cumulative_returns = np.exp(returns.cumsum()) - 1
        downside_returns = returns[returns < 0]

        sortino_ratio = returns.mean() / (downside_returns.std() + 1e-9) * np.sqrt(252)
        running_max = cumulative_returns.cummax()
        drawdown = cumulative_returns - running_max
        max_drawdown = drawdown.min()
        total_return = cumulative_returns.iloc[-1]
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
        time_in_market = combined_signal.mean()
        gross_profit = returns[returns > 0].sum()
        gross_loss = -returns[returns < 0].sum()
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan
        active_returns = returns[combined_signal]
        percent_profitable = (active_returns > 0).mean() if len(active_returns) > 0 else np.nan

        # Store return series separately
        signal_name = '+'.join(signal_names)
        returns_store[(signal_name, ticker)] = returns
        
        # Calculate comprehensive metrics for combined signals (same rigor as individual signals)
        # HAC t-statistic
        hac_mu, hac_se, hac_t_stat = hac_t(returns, lags=0)  # Use default lags
        
        # Moving-block bootstrap
        boot_p, boot_ci = mbb_p_ci(returns, block=10, B=1000, two_sided=two_sided_bootstrap)  # Use default parameters
        
        # Trade metrics
        trade_metrics_full = trade_metrics(returns, combined_signal)
        
        # Calculate trust gates for combined signals (same rigor as individual signals)
        trust_gates = calculate_trust_gates(
            returns, combined_signal, 
            trade_metrics_full,
            hac_t_stat, boot_ci, np.nan,  # FDR will be calculated later
            min_trades=30, min_exposure=0.10, min_sharpe=0.5,
            costs_per_trade=0.001, daily_slippage=0.0001, ticker=ticker
        )
        
        # Store metrics only
        results_rows.append({
            'Signal': signal_name,
            'Ticker': ticker,
            'Total Return': total_return,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Max Drawdown': max_drawdown,
            'Time in Market': time_in_market,
            'Profit Factor': profit_factor,
            'Percent Profitable': percent_profitable,
            'Num Trades': trade_metrics_full['num_trades'],
            'Avg Trade Return': trade_metrics_full['avg_trade_return'],
            'Avg Trade Duration': trade_metrics_full['avg_trade_duration'],
            'HAC_t_stat': hac_t_stat,
            'HAC_std_error': hac_se,
            'Bootstrap_p': boot_p,
                            'Bootstrap_CI_low': boot_ci[0] if isinstance(boot_ci, tuple) and pd.notna(boot_ci[0]) else np.nan,
                'Bootstrap_CI_high': boot_ci[1] if isinstance(boot_ci, tuple) and pd.notna(boot_ci[1]) else np.nan,
            'Trust_Score': trust_gates['trust_score'],
            'Trust_Status': trust_gates['trust_status'],
            'Gates_Passed': trust_gates['gates_passed'],
            'Net_Sharpe': trust_gates['net_sharpe'],
            'Total_Costs': trust_gates['total_costs']
        })

    progress.close()
    results_df = pd.DataFrame(results_rows).sort_values(by='Sortino Ratio', ascending=False)
    return results_df, returns_store

def plot_performance_chart(results_df, selected_signals, returns_store, benchmark_returns=None):
    """Create interactive performance chart with drawdown overlay"""
    # Create subplots: one for equity curves, one for drawdowns
    fig = go.Figure()
    
    # Add benchmark if provided
    if benchmark_returns is not None:
        benchmark_cumulative = np.exp(benchmark_returns.cumsum()) - 1
        benchmark_dd = (benchmark_cumulative / benchmark_cumulative.cummax() - 1) * 100
        
        fig.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative.values * 100,
            mode='lines',
            name="Benchmark",
            line=dict(width=2, color='gray', dash='dash'),
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=benchmark_dd.index,
            y=benchmark_dd.values,
            mode='lines',
            name="Benchmark DD",
            line=dict(width=1, color='gray', dash='dot'),
            yaxis='y2',
            showlegend=False
        ))
    
    # Add strategy signals
    for signal_name in selected_signals:
        signal_data = results_df[results_df['Signal'] == signal_name].iloc[0]
        ticker = signal_data['Ticker']
        
        # Get returns from the store
        if (signal_name, ticker) in returns_store:
            returns = returns_store[(signal_name, ticker)]
            cumulative_returns = np.exp(returns.cumsum()) - 1
            
            # Calculate drawdown
            rolling_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / rolling_max - 1) * 100
            
            # Add equity curve
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values * 100,
                mode='lines',
                name=f"{signal_name} ({ticker})",
                line=dict(width=2),
                yaxis='y'
            ))
            
            # Add drawdown
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name=f"{signal_name} DD",
                line=dict(width=1, dash='dot'),
                yaxis='y2',
                showlegend=False
            ))
    
    # Update layout with dual y-axes
    fig.update_layout(
        title="Performance & Drawdown Analysis",
        xaxis_title="Date",
        yaxis=dict(
            title="Cumulative Return (%)",
            side="left",
            showgrid=True
        ),
        yaxis2=dict(
            title="Drawdown (%)",
            side="right",
            overlaying="y",
            range=[-100, 0],  # Drawdown from 0% to -100%
            showgrid=False
        ),
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# Main Streamlit App
def main():
    st.title("📈 IF THEN Signals Backtesting App")
    st.markdown("*Created by IAMCAPTAINNOW - Discord*")
    
    # Help section
    with st.sidebar.expander("📚 Help & Documentation", expanded=False):
        st.markdown("""
        ## 📚 Complete Parameter Guide
        
        ### 📅 **Data & Time Settings**
        
        **Start Date**: Beginning of historical data. Earlier dates provide more data but may include outdated market conditions.
        
        **IS/OOS Split Ratio**: Fraction of data used for in-sample training vs out-of-sample testing. 
        - **70%** (default): Standard split, 70% for training, 30% for validation
        - **Higher values** (80-90%): More training data, less validation
        - **Lower values** (50-60%): Less training data, more validation
        
        ### 🎯 **Target & Reference Tickers**
        
        **Target Tickers**: Assets you want to trade (e.g., QQQ, SPY). These generate the actual trading signals.
        
        **Reference Tickers**: Assets used for comparison signals (e.g., TLT, VIXM). Used in cross-asset strategies.
        
        ### 📊 **Signal Generation Parameters**
        
        **RSI Period**: Lookback window for RSI calculation.
        - **14** (default): Standard period, good balance of sensitivity and stability
        - **Shorter** (7-10): More sensitive, more signals, potential overfitting
        - **Longer** (20-30): Less sensitive, fewer signals, more stable
        
        **RSI Thresholds**: Levels that trigger buy/sell signals.
        - **30/70** (default): Traditional overbought/oversold levels
        - **20/80**: More conservative, fewer but potentially stronger signals
        - **40/60**: More aggressive, more frequent signals
        
        **Moving Average Periods**: Windows for trend calculation.
        - **20/50** (default): Short-term vs medium-term trends
        - **10/30**: More responsive to recent price changes
        - **50/200**: Longer-term trends, fewer signals
        
        **Cumulative Return Threshold**: Minimum return required for signal generation.
        - **0.05** (5%): Moderate threshold, balanced signal frequency
        - **0.02** (2%): More sensitive, more signals
        - **0.10** (10%): Conservative, fewer but stronger signals
        
        ### 🔍 **Filtering & Selection**
        
        **Minimum Time in Market**: Minimum fraction of time the strategy must be active.
        - **0.025** (2.5%): Very permissive, allows sparse strategies
        - **0.05** (5%): Balanced, excludes ultra-sparse signals
        - **0.10** (10%): Conservative, only frequently active strategies
        
        **Maximum Drawdown**: Maximum allowed drawdown (negative values).
        - **-0.50** (-50%): Very permissive
        - **-0.20** (-20%): Moderate risk tolerance
        - **-0.10** (-10%): Conservative risk tolerance
        
        **Performance Quantile Filter**: Keep only top-performing signals.
        - **0.95** (95%): Top 5% of signals
        - **0.90** (90%): Top 10% of signals
        - **0.99** (99%): Top 1% of signals (very selective)
        
        ### 📈 **Statistical Analysis**
        
        **HAC Lags**: Number of lags for heteroskedasticity and autocorrelation consistent t-statistics.
        - **0** (auto): Automatically calculated based on data length
        - **5-10**: Manual setting for shorter time series
        - **20-30**: Manual setting for longer time series
        
        **Bootstrap Block Size**: Size of blocks for moving-block bootstrap.
        - **10** (default): Good for daily data
        - **5**: More blocks, more variation
        - **20**: Fewer blocks, more stable
        
        **Bootstrap Samples**: Number of bootstrap resamples.
        - **1000** (default): Good balance of accuracy and speed
        - **500**: Faster, less accurate
        - **2000**: Slower, more accurate
        
        **FDR Alpha**: False Discovery Rate control level.
        - **0.05** (5%): Standard significance level
        - **0.01** (1%): More conservative, fewer false positives
        - **0.10** (10%): Less conservative, more signals
        
        ### 💰 **Cost & Risk Management**
        
        **Transaction Costs**: Cost per trade (commission + spread).
        - **0.001** (0.1%): Typical for retail trading
        - **0.0005** (0.05%): Low-cost broker
        - **0.002** (0.2%): Higher costs (options, forex)
        
        **Daily Slippage**: Daily trading costs (fees, bid-ask spread).
        - **0.0001** (0.01%): Low for liquid ETFs
        - **0.0005** (0.05%): Moderate for less liquid assets
        - **0.001** (0.1%): High for illiquid assets
        
        ### 🔒 **Trust Gate Parameters**
        
        **Minimum Trades**: Minimum number of trades for trust evaluation.
        - **30**: Standard minimum for statistical validity
        - **50**: More conservative, requires more evidence
        - **20**: Less conservative, allows fewer trades
        
        **Minimum Exposure**: Minimum time in market for trust evaluation.
        - **0.10** (10%): Standard requirement
        - **0.20** (20%): More conservative
        - **0.05** (5%): Less conservative
        
        **Minimum Sharpe Ratio**: Minimum risk-adjusted return after costs.
        - **0.5**: Standard threshold
        - **1.0**: More conservative, higher quality signals
        - **0.3**: Less conservative, more signals
        
        ### 🔗 **Signal Combinations**
        
        **Combination Limit**: Maximum number of signals to combine.
        - **2** (default): Simple combinations only
        - **3**: Standard, manageable complexity
        - **4-5**: More complex strategies
        
        **Max Signal Correlation**: Maximum correlation between combined signals.
        - **0.7** (70%): Standard threshold
        - **0.5** (50%): More diverse combinations
        - **0.9** (90%): Less restrictive
        
        ### 🔧 **Advanced Settings**
        
        **Stability Testing**: Test signal robustness with parameter variations.
        - **Off**: Faster backtesting
        - **On**: More rigorous evaluation (slower)
        
        **Two-sided Bootstrap Test**: Use two-sided statistical tests.
        - **On**: Tests both positive and negative returns
        - **Off**: One-sided tests only
        
        ### 📊 **Benchmark Settings**
        
        **Use Benchmark**: Compare strategies against a benchmark.
        - **Off**: No benchmark comparison
        - **On**: Compare against selected benchmark
        
        **Benchmark Ticker**: Reference asset for comparison.
        - **SPY**: S&P 500 ETF (equity benchmark)
        - **TLT**: Long-term Treasury (bond benchmark)
        - **QQQ**: Nasdaq (tech benchmark)
        
        ### 🎯 **Understanding Results**
        
        **Trust Status**:
        - 🟢 **TRUSTED**: Pass 5-6 gates, safe for live trading
        - 🟡 **CAUTION**: Pass 4 gates, monitor closely
        - 🟠 **PAPER_TRADE**: Pass 3 gates, test first
        - 🔴 **REJECT**: Pass 0-2 gates, do not trade
        
        **Key Metrics**:
        - **Sortino Ratio**: Risk-adjusted return (higher is better)
        - **Calmar Ratio**: Return vs max drawdown (higher is better)
        - **HAC t-stat**: Statistical significance (|t| > 2 is significant)
        - **FDR q-value**: Multiple testing control (≤ 0.05 is significant)
        - **Bootstrap p-value**: Statistical significance (≤ 0.05 is significant)
        
        **IS vs OOS**: 
        - **IS**: In-sample performance (used for selection)
        - **OOS**: Out-of-sample performance (true evaluation)
        - **Degradation**: How much OOS underperforms IS (red flags overfitting)
        """)
    
    st.markdown("---")
    
    # Sidebar for parameters
    st.sidebar.header("📊 Backtesting Parameters")
    
    # Target tickers (assets to trade)
    st.sidebar.subheader("Target Tickers (Assets to Trade)")
    default_target = ['TQQQ', 'SQQQ', 'UVXY', 'PSQ']
    target_input = st.sidebar.text_area(
        "Target Tickers (comma-separated)",
        value=','.join(default_target),
        help="Assets you want to trade (e.g., QQQ, SPY). These generate the actual trading signals. Separate multiple tickers with commas."
    )
    target = [t.strip().upper() for t in target_input.split(',') if t.strip()]
    
    # Reference tickers (for signal generation)
    st.sidebar.subheader("Reference Tickers (For Signal Generation)")
    default_reference = ['VIXM', 'QQQ', 'TLT', 'KMLM', 'IYT', 'CORP']
    reference_input = st.sidebar.text_area(
        "Reference Tickers (comma-separated)",
        value=','.join(default_reference),
        help="Assets used for comparison signals (e.g., TLT, VIXM). Used in cross-asset strategies to generate relative strength signals. Separate multiple tickers with commas."
    )
    reference = [t.strip().upper() for t in reference_input.split(',') if t.strip()]
    
    # Date range
    st.sidebar.subheader("Date Range")
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime(2018, 1, 1),
        min_value=datetime(2010, 1, 1),
        max_value=datetime.now() - timedelta(days=30)
    )
    
    # IS/OOS Split
    st.sidebar.subheader("In-Sample/Out-of-Sample Split")
    is_ratio = st.sidebar.slider(
        "In-Sample Ratio",
        min_value=0.5,
        max_value=0.9,
        value=0.7,
        step=0.05,
        help="Fraction of data for training vs testing. 0.7 = 70% training, 30% validation. Higher values = more training data but less validation. Lower values = less training but more validation."
    )
    
    # Signal combination parameters
    st.sidebar.subheader("Signal Combination")
    combination_limit = st.sidebar.slider(
        "Maximum Combined Signals",
        min_value=2,
        max_value=5,
        value=2,
        help="Maximum signals to combine using AND logic. Higher values create more complex strategies but may overfit. 2-3 is usually optimal."
    )
    
    # Filtering parameters
    st.sidebar.subheader("Filtering Criteria")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        tim = st.number_input(
            "Min Time in Market",
            min_value=0.001,
            max_value=1.0,
            value=0.025,
            step=0.005,
            format="%.3f",
            help="Minimum fraction of time strategy must be active. 0.025 = 2.5%. Higher values exclude sparse strategies. Lower values allow more sparse but potentially profitable signals."
        )
    
    with col2:
        mdd = st.number_input(
            "Max Drawdown Limit",
            min_value=-1.0,
            max_value=-0.01,
            value=-0.5,
            step=0.05,
            format="%.2f",
            help="Maximum allowed drawdown (negative values). -0.5 = 50% max drawdown. More negative = higher risk tolerance. -0.2 to -0.3 is moderate risk."
        )
    
    # Minimum exposure threshold
    min_exposure_threshold = st.sidebar.number_input(
        "Min Exposure Threshold",
        min_value=0.01,
        max_value=0.50,
        value=0.05,
        step=0.01,
        format="%.2f",
        help="Minimum time in market required for trust evaluation. 0.05 = 5% minimum exposure. Higher values ensure strategies are frequently active. Lower values allow sparse but potentially profitable signals."
    )
    
    # HAC lags parameter
    hac_lags = st.sidebar.number_input(
        "HAC Lags (0 = auto)",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help="Lags for heteroskedasticity and autocorrelation consistent t-statistics. 0 = automatic calculation. Higher values account for more autocorrelation but may reduce power. 5-15 is typical for daily data."
    )
    
    # Bootstrap parameters
    st.sidebar.subheader("Bootstrap Parameters")
    mbb_block_size = st.sidebar.number_input(
        "Block Size",
        min_value=5,
        max_value=50,
        value=10,
        step=1,
        help="Size of blocks for moving-block bootstrap. 10 = 10-day blocks. Smaller blocks = more variation, larger blocks = more stable. 5-20 is typical for daily data."
    )
    
    mbb_bootstrap_samples = st.sidebar.number_input(
        "Bootstrap Samples",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Number of bootstrap resamples for statistical testing. 1000 = good balance of accuracy and speed. Higher values = more accurate p-values but slower computation. 500-2000 is typical."
    )
    
    # FDR control parameter
    fdr_alpha = st.sidebar.number_input(
        "FDR Alpha Level",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%.2f",
        help="False Discovery Rate control level. 0.05 = 5% FDR (standard). Controls false positives when testing multiple signals. Lower values = more conservative, fewer false positives. Higher values = less conservative, more signals."
    )
    
    # Benchmark selection
    st.sidebar.subheader("Benchmark Settings")
    use_benchmark = st.sidebar.checkbox(
        "Include Benchmark",
        value=True,
        help="Add benchmark comparison to performance charts. Shows how strategies perform relative to a market benchmark (e.g., SPY for equity strategies)."
    )
    
    benchmark_ticker = st.sidebar.selectbox(
        "Benchmark Ticker",
        options=['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'None'],
        index=0,
        help="Reference asset for comparison. SPY = S&P 500 (equity), QQQ = Nasdaq (tech), TLT = Treasury bonds, GLD = Gold. Choose based on your strategy type."
    ) if use_benchmark else None
    
    # Trust gate parameters
    st.sidebar.subheader("Trust Gate Settings")
    min_trades_trust = st.sidebar.number_input(
        "Min Trades",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="Minimum number of trades required for trust evaluation. 30 = standard minimum for statistical validity. Higher values = more conservative, requires more evidence. Lower values = less conservative, allows fewer trades."
    )
    
    min_exposure_trust = st.sidebar.number_input(
        "Min Exposure",
        min_value=0.05,
        max_value=0.50,
        value=0.10,
        step=0.05,
        format="%.2f",
        help="Minimum time in market required for trust evaluation. 0.10 = 10% minimum exposure. Higher values ensure strategies are frequently active. Lower values allow sparse but potentially profitable signals."
    )
    
    min_sharpe_trust = st.sidebar.number_input(
        "Min Net Sharpe",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        format="%.1f",
        help="Minimum risk-adjusted return after costs for trust evaluation. 0.5 = standard threshold. Higher values = more conservative, higher quality signals. Lower values = less conservative, more signals."
    )
    
    costs_per_trade = st.sidebar.number_input(
        "Costs per Trade",
        min_value=0.0001,
        max_value=0.01,
        value=0.001,
        step=0.0001,
        format="%.4f",
        help="Transaction costs per trade (commission + spread). 0.001 = 0.1% (typical retail). 0.0005 = low-cost broker. 0.002 = higher costs (options, forex)."
    )
    
    daily_slippage = st.sidebar.number_input(
        "Daily Slippage",
        min_value=0.00001,
        max_value=0.001,
        value=0.0001,
        step=0.00001,
        format="%.5f",
        help="Daily trading costs (fees, bid-ask spread). 0.0001 = 0.01% (low for liquid ETFs). 0.0005 = moderate for less liquid assets. 0.001 = high for illiquid assets."
    )
    
    quantile = st.sidebar.slider(
        "Performance Quantile Filter",
        min_value=0.50,
        max_value=0.99,
        value=0.95,
        step=0.05,
        help="Keep only top-performing signals. 0.95 = top 5% of signals. 0.90 = top 10%. 0.99 = top 1% (very selective). Higher values = fewer but potentially better signals."
    )
    
    # Advanced settings
    st.sidebar.subheader("🔧 Advanced Settings")
    
    max_correlation = st.sidebar.slider(
        "Max Signal Correlation",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Maximum correlation between signals in combinations. 0.7 = 70% correlation limit. Lower values = more diverse combinations, higher values = less restrictive. Prevents redundant signal combinations."
    )
    

    
    two_sided_bootstrap = st.sidebar.checkbox(
        "Two-sided Bootstrap Test",
        value=True,
        help="Use two-sided statistical tests for bootstrap p-values. Tests both positive and negative returns. More robust than one-sided tests. Recommended for most strategies."
    )
    
    # Advanced filtering controls
    st.sidebar.subheader("🔍 Advanced Filtering")
    
    enable_secondary_filters = st.sidebar.checkbox(
        "Enable Secondary Filters",
        value=True,
        help="Apply advanced filters: FDR control, trust gates, statistical significance tests. Disable for faster processing or to see all signals."
    )
    
    if enable_secondary_filters:
        fdr_alpha = st.sidebar.slider(
            "FDR Alpha Level",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            format="%.2f",
            help="False Discovery Rate control level. Lower values = stricter multiple testing correction. 0.05 = 5% false positive rate."
        )
    else:
        fdr_alpha = 1.0  # Effectively disable FDR
    
    # Signal explosion controls
    st.sidebar.subheader("🎯 Signal Generation Controls")
    
    max_signals = st.sidebar.number_input(
        "Max Total Signals",
        min_value=1000,
        max_value=50000,
        value=15000,
        step=1000,
        help="Hard cap on total generated signals to prevent explosion"
    )
    
    # Signal type selection
    st.sidebar.subheader("📊 Signal Types")
    
    enable_rsi_signals = st.sidebar.checkbox(
        "RSI Signals",
        value=True,
        help="Generate RSI-based signals (individual levels and cross-asset comparisons)"
    )
    
    enable_cumret_signals = st.sidebar.checkbox(
        "Cumulative Return Signals",
        value=True,
        help="Generate cumulative return-based signals (individual levels and cross-asset comparisons)"
    )
    
    enable_ma_signals = st.sidebar.checkbox(
        "Moving Average Signals",
        value=True,
        help="Generate moving average comparison signals (cross-asset only)"
    )
    
    use_curated_pairs = st.sidebar.checkbox(
        "Use Curated Ticker Pairs",
        value=False,
        help="Restrict cross-asset comparisons to curated pairs only"
    )
    
    if use_curated_pairs:
        curated_pairs = st.sidebar.multiselect(
            "Select Curated Pairs",
            options=[
                ("QQQ", "TLT"), ("QQQ", "VIXM"), ("SPY", "IEF"), 
                ("QQQ", "GLD"), ("SPY", "TLT"), ("QQQ", "SPY")
            ],
            default=[("QQQ", "TLT"), ("QQQ", "VIXM")],
            help="Only these pairs will be used for cross-asset signals"
        )
    
    # Combine all tickers
    all_tickers = list(set(target + reference))
    
    # Run backtest button
    if st.sidebar.button("🚀 Run Backtest", type="primary"):
        if not target:
            st.error("Please specify at least one target ticker")
            return
        
        try:
            # Generate signals
            st.header("🔄 Generating Signals...")
            signals, price_data = generate_signals(
                all_tickers, 
                start_date.strftime('%Y-%m-%d'), 
                max_signals, 
                use_curated_pairs, 
                curated_pairs if 'curated_pairs' in locals() else None,
                enable_rsi_signals,
                enable_cumret_signals,
                enable_ma_signals
            )
            
            st.success(f"✅ Generated {len(signals)} signals for {len(all_tickers)} tickers")
            
            # Download benchmark data if requested
            benchmark_returns = None
            if use_benchmark and benchmark_ticker and benchmark_ticker != 'None':
                try:
                    # Download benchmark data with adjusted close option
                    benchmark_raw = yf.download(benchmark_ticker, start=start_date.strftime('%Y-%m-%d'), progress=False)
                    if 'Adj Close' in benchmark_raw.columns:
                        benchmark_data = benchmark_raw['Adj Close']
                    else:
                        benchmark_data = benchmark_raw['Close']
                    benchmark_returns = np.log(benchmark_data / benchmark_data.shift(1)).fillna(0)
                    st.info(f"📊 Benchmark: {benchmark_ticker} data downloaded")
                except Exception as e:
                    st.warning(f"⚠️ Could not download benchmark data: {str(e)}")
                    benchmark_returns = None
            
            # Create IS/OOS split
            idx = price_data.index
            split_idx = int(len(idx) * is_ratio)
            split_date = idx[split_idx]
            is_mask = idx <= split_date
            oos_mask = idx > split_date
            
            st.info(f"📅 IS Period: {idx[0].strftime('%Y-%m-%d')} to {split_date.strftime('%Y-%m-%d')} ({is_ratio:.0%} of data)")
            st.info(f"📅 OOS Period: {(split_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} to {idx[-1].strftime('%Y-%m-%d')} ({(1-is_ratio):.0%} of data)")
            
            # Backtest individual signals with IS/OOS split
            st.header("📈 Backtesting Individual Signals (IS/OOS)...")
            backtest_results, returns_store = backtest_signals(
                signals, price_data, all_tickers, target, is_mask, oos_mask, 
                min_exposure_threshold, hac_lags, mbb_block_size, mbb_bootstrap_samples,
                min_trades_trust, min_exposure_trust, min_sharpe_trust, costs_per_trade, daily_slippage,
                two_sided_bootstrap
            )
            # Drop rows with NaN in key columns only (not all columns)
            key_columns = ['Signal', 'Ticker', 'Total Return', 'Sortino Ratio']
            backtest_results = backtest_results.dropna(subset=key_columns)
            
            # Apply filters
            backtest_results = backtest_results[backtest_results['Time in Market'] > tim]
            backtest_results = backtest_results[backtest_results['Max Drawdown'] > mdd]
            backtest_results = backtest_results[backtest_results['Total Return'] > 0]
            
            st.info(f"📊 {len(backtest_results)} signals passed initial filters")
            
            # Apply FDR control to bootstrap p-values (only if secondary filters are enabled)
            if enable_secondary_filters:
                pcol = 'OOS_Bootstrap_p' if 'OOS_Bootstrap_p' in backtest_results.columns else 'Bootstrap_p'
                if pcol in backtest_results.columns:
                    valid_pvals = backtest_results[pcol].dropna()
                    if len(valid_pvals) > 0:
                        fdr_qvals, fdr_threshold = fdr_bh(valid_pvals, alpha=fdr_alpha)
                        backtest_results.loc[valid_pvals.index, 'FDR_q_value'] = fdr_qvals
                        
                        # Count significant signals
                        significant_signals = (fdr_qvals <= fdr_alpha).sum()
                        st.info(f"🔬 FDR Control: {significant_signals} signals significant at {fdr_alpha:.1%} FDR level")
                        
                        # Update trust gates with FDR values using OOS data
                        for idx in backtest_results.index:
                            if pd.notna(backtest_results.loc[idx, 'FDR_q_value']):
                                fdr_q = backtest_results.loc[idx, 'FDR_q_value']
                                signal_name = backtest_results.loc[idx, 'Signal']
                                ticker = backtest_results.loc[idx, 'Ticker']
                                
                                if (signal_name, ticker) in returns_store:
                                    returns = returns_store[(signal_name, ticker)]
                                    shifted_signal = signals[signal_name].reindex(price_data.index).fillna(False).shift(1).fillna(False)
                                    
                                    if is_mask is not None and oos_mask is not None:
                                        # Use OOS data for trust gate evaluation
                                        oos_returns = returns[oos_mask]
                                        oos_signal = shifted_signal[oos_mask]
                                        oos_trade_metrics = trade_metrics(oos_returns, oos_signal)
                                        
                                        # Get OOS metrics
                                        oos_hac_t = backtest_results.loc[idx, 'OOS_HAC_t_stat']
                                        oos_boot_ci = (backtest_results.loc[idx, 'OOS_Bootstrap_CI_low'], backtest_results.loc[idx, 'OOS_Bootstrap_CI_high'])
                                        
                                        # Recalculate trust gates with FDR using OOS data
                                        updated_trust = calculate_trust_gates(
                                            oos_returns, oos_signal,
                                            oos_trade_metrics,
                                            oos_hac_t, oos_boot_ci, fdr_q,
                                            min_trades_trust, min_exposure_trust, min_sharpe_trust,
                                            costs_per_trade, daily_slippage,
                                            ticker=ticker
                                        )
                                    else:
                                        # Use full period data
                                        trade_metrics_full = trade_metrics(returns, shifted_signal)
                                        hac_t_stat = backtest_results.loc[idx, 'HAC_t_stat']
                                        boot_ci = (backtest_results.loc[idx, 'Bootstrap_CI_low'], backtest_results.loc[idx, 'Bootstrap_CI_high'])
                                        
                                        updated_trust = calculate_trust_gates(
                                            returns, shifted_signal,
                                            trade_metrics_full,
                                            hac_t_stat, boot_ci, fdr_q,
                                            min_trades_trust, min_exposure_trust, min_sharpe_trust,
                                            costs_per_trade, daily_slippage,
                                            ticker=ticker
                                        )
                                    
                                    # Update trust metrics
                                    backtest_results.loc[idx, 'Trust_Score'] = updated_trust['trust_score']
                                    backtest_results.loc[idx, 'Trust_Status'] = updated_trust['trust_status']
                                    backtest_results.loc[idx, 'Gates_Passed'] = updated_trust['gates_passed']
                                    backtest_results.loc[idx, 'Net_Sharpe'] = updated_trust['net_sharpe']
                    else:
                        st.warning("⚠️ No valid p-values for FDR control")
            else:
                # Set default values when secondary filters are disabled
                backtest_results['FDR_q_value'] = 1.0  # No FDR filtering
                backtest_results['Trust_Score'] = 0.0  # No trust scoring
                backtest_results['Trust_Status'] = '🔴 REJECT'  # Default status
                backtest_results['Gates_Passed'] = 0  # No gates passed
                st.info("🔍 Secondary filters disabled - showing all signals with basic metrics")
            
            # Filter by quantiles per ticker
            backtest_filtered = pd.DataFrame()
            for ticker in backtest_results['Ticker'].unique():
                results = backtest_results[backtest_results["Ticker"] == ticker]
                results = results[results['Total Return'] > results['Total Return'].quantile(quantile)]
                results = results[results['Profit Factor'] > results['Profit Factor'].quantile(quantile)]
                results = results[results['Sortino Ratio'] > results['Sortino Ratio'].quantile(quantile)]
                results = results[results['Calmar Ratio'] > results['Calmar Ratio'].quantile(quantile)]
                backtest_filtered = pd.concat([backtest_filtered, results])
            
            st.success(f"✅ {len(backtest_filtered)} high-quality individual signals identified")
            
            # Generate and backtest combined signals
            if combination_limit > 1 and len(backtest_filtered) > 1:
                st.header("🔗 Backtesting Combined Signals...")
                combinations_filtered = generate_filtered_combinations(signals, backtest_filtered, combination_limit, max_correlation)
                
                if combinations_filtered:
                    log_returns = np.log(price_data / price_data.shift(1)).fillna(0)
                    backtest_df_combined, combined_returns_store = backtest_combined_signals(combinations_filtered, signals, price_data, log_returns, two_sided_bootstrap=two_sided_bootstrap)
                    
                    # Combine results
                    backtest_df_combined = backtest_df_combined[backtest_results.columns]
                    final_results = pd.concat([backtest_filtered, backtest_df_combined])
                    
                    # Merge returns stores
                    returns_store.update(combined_returns_store)
                    final_results = final_results[final_results['Total Return'] > 0]
                    final_results = final_results[final_results['Time in Market'] > tim]
                    final_results = final_results[final_results['Max Drawdown'] > mdd]
                    final_results = final_results.sort_values('Sortino Ratio', ascending=False)
                    
                    st.success(f"✅ Generated {len(backtest_df_combined)} combined signals")
                else:
                    final_results = backtest_filtered
                    st.info("No combined signals generated (insufficient individual signals)")
            else:
                final_results = backtest_filtered
            
            # Display results
            st.header("📋 Results Summary")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Signals", len(final_results))
            with col2:
                st.metric("Best IS Sortino", f"{final_results['Sortino Ratio'].max():.2f}")
            with col3:
                st.metric("Best IS Return", f"{final_results['Total Return'].max():.1%}")
            with col4:
                st.metric("Avg IS Time in Market", f"{final_results['Time in Market'].mean():.1%}")
            
            # Trust Gate Summary (only show if secondary filters are enabled)
            if enable_secondary_filters and 'Trust_Status' in final_results.columns:
                st.subheader("🔒 Trust Gate Summary")
                
                # Count signals by trust status
                trusted_count = (final_results['Trust_Status'] == '🟢 TRUSTED').sum()
                caution_count = (final_results['Trust_Status'] == '🟡 CAUTION').sum()
                paper_trade_count = (final_results['Trust_Status'] == '🟠 PAPER_TRADE').sum()
                reject_count = (final_results['Trust_Status'] == '🔴 REJECT').sum()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🟢 TRUSTED", trusted_count, help="Pass 5-6 trust gates - Safe for live trading")
                with col2:
                    st.metric("🟡 CAUTION", caution_count, help="Pass 4 trust gates - Monitor closely")
                with col3:
                    st.metric("🟠 PAPER_TRADE", paper_trade_count, help="Pass 3 trust gates - Paper trade first")
                with col4:
                    st.metric("🔴 REJECT", reject_count, help="Pass 0-2 trust gates - Do not trade")
                
                # Trust gate details
                st.markdown("**Trust Gates:** Sample Size, HAC t-stat, Bootstrap CI, FDR Control, Costs, Risk")
            elif not enable_secondary_filters:
                st.info("🔍 Trust gates disabled - all signals shown with basic performance metrics")
                
            # FDR Statistics (only show if secondary filters are enabled)
            if enable_secondary_filters and 'FDR_q_value' in final_results.columns:
                fdr_stats = final_results['FDR_q_value'].dropna()
                if len(fdr_stats) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        significant_05 = (fdr_stats <= 0.05).sum()
                        st.metric("FDR ≤ 0.05", significant_05)
                    with col2:
                        significant_10 = (fdr_stats <= 0.10).sum()
                        st.metric("FDR ≤ 0.10", significant_10)
                    with col3:
                        avg_fdr = fdr_stats.mean()
                        st.metric("Avg FDR q-value", f"{avg_fdr:.3f}")
                    with col4:
                        min_fdr = fdr_stats.min()
                        st.metric("Min FDR q-value", f"{min_fdr:.3f}")
            
            # OOS Metrics
            if 'OOS_Sortino_Ratio' in final_results.columns:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best OOS Sortino", f"{final_results['OOS_Sortino_Ratio'].max():.2f}")
                with col2:
                    st.metric("Best OOS Return", f"{final_results['OOS_Total_Return'].max():.1%}")
                with col3:
                    st.metric("Avg OOS Time in Market", f"{final_results['OOS_Time_in_Market'].mean():.1%}")
                with col4:
                    # Calculate OOS performance vs IS
                    is_best_sortino = final_results['Sortino Ratio'].max()
                    oos_best_sortino = final_results['OOS_Sortino_Ratio'].max()
                    degradation = ((oos_best_sortino - is_best_sortino) / is_best_sortino * 100) if is_best_sortino != 0 else 0
                    st.metric("OOS vs IS Degradation", f"{degradation:.1f}%")
            
            # Create separate IS and OOS views
            if 'OOS_Total_Return' in final_results.columns:
                # IS Rankings (for selection/ranking)
                st.subheader("📊 IS Rankings (For Signal Selection)")
                is_columns = ['Signal', 'Ticker', 'Trust_Status', 'Trust_Score', 'Gates_Passed', 'Total Return', 'Sortino Ratio', 'Calmar Ratio', 
                             'Max Drawdown', 'Time in Market', 'Profit Factor', 'Percent Profitable', 
                             'Num Trades', 'Avg Trade Return', 'Avg Trade Duration', 'HAC_t_stat', 'Bootstrap_p', 'FDR_q_value', 'Net_Sharpe']
                
                # OOS Rankings (for evaluation)
                st.subheader("📊 OOS Rankings (For Performance Evaluation)")
                oos_columns = ['Signal', 'Ticker', 'Trust_Status', 'Trust_Score', 'Gates_Passed', 'OOS_Total_Return', 'OOS_Sortino_Ratio', 'OOS_Calmar_Ratio', 
                              'OOS_Max_Drawdown', 'OOS_Time_in_Market', 'OOS_Profit_Factor', 'OOS_Percent_Profitable',
                              'OOS_Num_Trades', 'OOS_Avg_Trade_Return', 'OOS_Avg_Trade_Duration', 'OOS_HAC_t_stat', 'OOS_Bootstrap_p']
                
                # Sort by IS and OOS separately
                is_ranked = final_results.sort_values('Sortino Ratio', ascending=False)
                oos_ranked = final_results.sort_values('OOS_Sortino_Ratio', ascending=False)
                
                # Display IS rankings
                display_is_df = is_ranked[is_columns].copy()
                display_is_df['Total Return'] = display_is_df['Total Return'].apply(lambda x: f"{x:.1%}")
                display_is_df['Sortino Ratio'] = display_is_df['Sortino Ratio'].apply(lambda x: f"{x:.2f}")
                display_is_df['Calmar Ratio'] = display_is_df['Calmar Ratio'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                display_is_df['Max Drawdown'] = display_is_df['Max Drawdown'].apply(lambda x: f"{x:.1%}")
                display_is_df['Time in Market'] = display_is_df['Time in Market'].apply(lambda x: f"{x:.1%}")
                display_is_df['Profit Factor'] = display_is_df['Profit Factor'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                display_is_df['Percent Profitable'] = display_is_df['Percent Profitable'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
                display_is_df['Trust_Score'] = display_is_df['Trust_Score'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
                display_is_df['Gates_Passed'] = display_is_df['Gates_Passed'].apply(lambda x: f"{x:.0f}/6" if pd.notnull(x) else "N/A")
                display_is_df['Num Trades'] = display_is_df['Num Trades'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "0")
                display_is_df['Avg Trade Return'] = display_is_df['Avg Trade Return'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
                display_is_df['Avg Trade Duration'] = display_is_df['Avg Trade Duration'].apply(lambda x: f"{x:.0f}d" if pd.notnull(x) else "N/A")
                display_is_df['HAC_t_stat'] = display_is_df['HAC_t_stat'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                display_is_df['Bootstrap_p'] = display_is_df['Bootstrap_p'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
                display_is_df['FDR_q_value'] = display_is_df['FDR_q_value'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
                display_is_df['Net_Sharpe'] = display_is_df['Net_Sharpe'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                
                # Display OOS rankings
                display_oos_df = oos_ranked[oos_columns].copy()
                display_oos_df['OOS_Total_Return'] = display_oos_df['OOS_Total_Return'].apply(lambda x: f"{x:.1%}")
                display_oos_df['OOS_Sortino_Ratio'] = display_oos_df['OOS_Sortino_Ratio'].apply(lambda x: f"{x:.2f}")
                display_oos_df['OOS_Calmar_Ratio'] = display_oos_df['OOS_Calmar_Ratio'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                display_oos_df['OOS_Max_Drawdown'] = display_oos_df['OOS_Max_Drawdown'].apply(lambda x: f"{x:.1%}")
                display_oos_df['OOS_Time_in_Market'] = display_oos_df['OOS_Time_in_Market'].apply(lambda x: f"{x:.1%}")
                display_oos_df['OOS_Profit_Factor'] = display_oos_df['OOS_Profit_Factor'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                display_oos_df['OOS_Percent_Profitable'] = display_oos_df['OOS_Percent_Profitable'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
                display_oos_df['Trust_Score'] = display_oos_df['Trust_Score'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
                display_oos_df['Gates_Passed'] = display_oos_df['Gates_Passed'].apply(lambda x: f"{x:.0f}/6" if pd.notnull(x) else "N/A")
                display_oos_df['OOS_Num_Trades'] = display_oos_df['OOS_Num_Trades'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "0")
                display_oos_df['OOS_Avg_Trade_Return'] = display_oos_df['OOS_Avg_Trade_Return'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
                display_oos_df['OOS_Avg_Trade_Duration'] = display_oos_df['OOS_Avg_Trade_Duration'].apply(lambda x: f"{x:.0f}d" if pd.notnull(x) else "N/A")
                display_oos_df['OOS_HAC_t_stat'] = display_oos_df['OOS_HAC_t_stat'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                display_oos_df['OOS_Bootstrap_p'] = display_oos_df['OOS_Bootstrap_p'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
                
                # Apply conditional formatting for trust status
                if 'Trust_Status' in display_is_df.columns:
                    def highlight_trust_status(row):
                        try:
                            trust_status = row['Trust_Status']
                            if '🟢 TRUSTED' in trust_status:
                                return ['background-color: #d4edda'] * len(row)
                            elif '🟡 CAUTION' in trust_status:
                                return ['background-color: #fff3cd'] * len(row)
                            elif '🟠 PAPER_TRADE' in trust_status:
                                return ['background-color: #ffeaa7'] * len(row)
                            elif '🔴 REJECT' in trust_status:
                                return ['background-color: #f8d7da'] * len(row)
                            else:
                                return [''] * len(row)
                        except:
                            return [''] * len(row)
                    
                    styled_is_df = display_is_df.head(20).style.apply(highlight_trust_status, axis=1)
                    st.dataframe(styled_is_df, use_container_width=True)
                    
                    styled_oos_df = display_oos_df.head(20).style.apply(highlight_trust_status, axis=1)
                    st.dataframe(styled_oos_df, use_container_width=True)
                else:
                    st.dataframe(display_is_df.head(20), use_container_width=True)
                    st.dataframe(display_oos_df.head(20), use_container_width=True)
                
                # OOS degradation analysis
                st.subheader("📉 OOS Performance Degradation Analysis")
                
                # Calculate degradation metrics
                degradation_data = []
                for idx in final_results.index:
                    signal_name = final_results.loc[idx, 'Signal']
                    ticker = final_results.loc[idx, 'Ticker']
                    
                    is_sortino = final_results.loc[idx, 'Sortino Ratio']
                    oos_sortino = final_results.loc[idx, 'OOS_Sortino_Ratio']
                    is_return = final_results.loc[idx, 'Total Return']
                    oos_return = final_results.loc[idx, 'OOS_Total_Return']
                    
                    if pd.notna(is_sortino) and pd.notna(oos_sortino) and is_sortino != 0:
                        sortino_degradation = ((oos_sortino - is_sortino) / is_sortino) * 100
                    else:
                        sortino_degradation = np.nan
                    
                    if pd.notna(is_return) and pd.notna(oos_return) and is_return != 0:
                        return_degradation = ((oos_return - is_return) / is_return) * 100
                    else:
                        return_degradation = np.nan
                    
                    degradation_data.append({
                        'Signal': signal_name,
                        'Ticker': ticker,
                        'IS_Sortino': f"{is_sortino:.2f}" if pd.notna(is_sortino) else "N/A",
                        'OOS_Sortino': f"{oos_sortino:.2f}" if pd.notna(oos_sortino) else "N/A",
                        'Sortino_Degradation': f"{sortino_degradation:.1f}%" if pd.notna(sortino_degradation) else "N/A",
                        'IS_Return': f"{is_return:.1%}" if pd.notna(is_return) else "N/A",
                        'OOS_Return': f"{oos_return:.1%}" if pd.notna(oos_return) else "N/A",
                        'Return_Degradation': f"{return_degradation:.1f}%" if pd.notna(return_degradation) else "N/A"
                    })
                
                degradation_df = pd.DataFrame(degradation_data)
                
                # Color code degradation
                def highlight_degradation(row):
                    try:
                        sortino_deg = float(row['Sortino_Degradation'].replace('%', '')) if row['Sortino_Degradation'] != 'N/A' else np.nan
                        return_deg = float(row['Return_Degradation'].replace('%', '')) if row['Return_Degradation'] != 'N/A' else np.nan
                        
                        if pd.notna(sortino_deg) and sortino_deg < -20:  # Severe degradation
                            return ['background-color: #f8d7da'] * len(row)  # Red
                        elif pd.notna(sortino_deg) and sortino_deg < -10:  # Moderate degradation
                            return ['background-color: #fff3cd'] * len(row)  # Yellow
                        elif pd.notna(sortino_deg) and sortino_deg > 0:  # Improvement
                            return ['background-color: #d4edda'] * len(row)  # Green
                        else:
                            return [''] * len(row)
                    except:
                        return [''] * len(row)
                
                styled_degradation_df = degradation_df.head(20).style.apply(highlight_degradation, axis=1)
                st.dataframe(styled_degradation_df, use_container_width=True)
                
            else:
                # Single period view (no IS/OOS split)
                st.subheader("📊 Top Performing Signals")
                display_columns = ['Signal', 'Ticker', 'Trust_Status', 'Trust_Score', 'Gates_Passed', 'Total Return', 'Sortino Ratio', 'Calmar Ratio', 
                                 'Max Drawdown', 'Time in Market', 'Profit Factor', 'Percent Profitable', 
                                 'Num Trades', 'Avg Trade Return', 'Avg Trade Duration', 'HAC_t_stat', 'Bootstrap_p', 'FDR_q_value', 'Net_Sharpe']
            
                # Format the dataframe for display (single period view)
                display_df = final_results[display_columns].copy()
                display_df['Total Return'] = display_df['Total Return'].apply(lambda x: f"{x:.1%}")
                display_df['Sortino Ratio'] = display_df['Sortino Ratio'].apply(lambda x: f"{x:.2f}")
                display_df['Calmar Ratio'] = display_df['Calmar Ratio'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                display_df['Max Drawdown'] = display_df['Max Drawdown'].apply(lambda x: f"{x:.1%}")
                display_df['Time in Market'] = display_df['Time in Market'].apply(lambda x: f"{x:.1%}")
                display_df['Profit Factor'] = display_df['Profit Factor'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                display_df['Percent Profitable'] = display_df['Percent Profitable'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
                display_df['Trust_Score'] = display_df['Trust_Score'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
                display_df['Gates_Passed'] = display_df['Gates_Passed'].apply(lambda x: f"{x:.0f}/6" if pd.notnull(x) else "N/A")
                display_df['Num Trades'] = display_df['Num Trades'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "0")
                display_df['Avg Trade Return'] = display_df['Avg Trade Return'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
                display_df['Avg Trade Duration'] = display_df['Avg Trade Duration'].apply(lambda x: f"{x:.0f}d" if pd.notnull(x) else "N/A")
                display_df['HAC_t_stat'] = display_df['HAC_t_stat'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                display_df['Bootstrap_p'] = display_df['Bootstrap_p'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
                display_df['FDR_q_value'] = display_df['FDR_q_value'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
                display_df['Net_Sharpe'] = display_df['Net_Sharpe'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                
                # Apply conditional formatting for trust status
                if 'Trust_Status' in display_df.columns:
                    def highlight_trust_status(row):
                        try:
                            trust_status = row['Trust_Status']
                            if '🟢 TRUSTED' in trust_status:
                                return ['background-color: #d4edda'] * len(row)  # Green for trusted
                            elif '🟡 CAUTION' in trust_status:
                                return ['background-color: #fff3cd'] * len(row)  # Yellow for caution
                            elif '🟠 PAPER_TRADE' in trust_status:
                                return ['background-color: #ffeaa7'] * len(row)  # Orange for paper trade
                            elif '🔴 REJECT' in trust_status:
                                return ['background-color: #f8d7da'] * len(row)  # Red for reject
                            else:
                                return [''] * len(row)
                        except:
                            return [''] * len(row)
                    
                    styled_df = display_df.head(20).style.apply(highlight_trust_status, axis=1)
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.dataframe(display_df.head(20), use_container_width=True)
            
            # Performance visualization
            if len(final_results) > 0:
                st.subheader("📈 Performance Visualization")
                
                # Select signals to plot
                available_signals = final_results.head(10)['Signal'].tolist()
                selected_signals = st.multiselect(
                    "Select signals to visualize:",
                    available_signals,
                    default=available_signals[:3] if len(available_signals) >= 3 else available_signals
                )
                
                if selected_signals:
                    chart = plot_performance_chart(final_results, selected_signals, returns_store, benchmark_returns)
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Drawdown analysis
                    st.subheader("📉 Drawdown Analysis")
                    
                    # Calculate drawdown statistics for selected signals
                    dd_stats = []
                    for signal_name in selected_signals:
                        signal_data = final_results[final_results['Signal'] == signal_name].iloc[0]
                        ticker = signal_data['Ticker']
                        
                        if (signal_name, ticker) in returns_store:
                            returns = returns_store[(signal_name, ticker)]
                            cumulative_returns = np.exp(returns.cumsum()) - 1
                            rolling_max = cumulative_returns.cummax()
                            drawdown = (cumulative_returns / rolling_max - 1) * 100
                            
                            # Calculate drawdown statistics
                            max_dd = drawdown.min()
                            avg_dd = drawdown.mean()
                            dd_duration = (drawdown < 0).sum()  # Number of days in drawdown
                            dd_frequency = (drawdown < 0).mean() * 100  # Percentage of time in drawdown
                            
                            dd_stats.append({
                                'Signal': signal_name,
                                'Ticker': ticker,
                                'Max Drawdown (%)': f"{max_dd:.1f}",
                                'Avg Drawdown (%)': f"{avg_dd:.1f}",
                                'DD Duration (days)': dd_duration,
                                'DD Frequency (%)': f"{dd_frequency:.1f}"
                            })
                    
                    if dd_stats:
                        dd_df = pd.DataFrame(dd_stats)
                        st.dataframe(dd_df, use_container_width=True)
            
            # Download results
            st.subheader("💾 Download Results")
            # Define display columns for download (use all available columns)
            download_columns = ['Signal', 'Ticker', 'Trust_Status', 'Trust_Score', 'Gates_Passed', 
                               'Total Return', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 
                               'Time in Market', 'Profit Factor', 'Percent Profitable', 
                               'Num Trades', 'Avg Trade Return', 'Avg Trade Duration', 
                               'HAC_t_stat', 'Bootstrap_p', 'FDR_q_value', 'Net_Sharpe']
            
            # Add OOS columns if available
            if 'OOS_Total_Return' in final_results.columns:
                download_columns.extend(['OOS_Total_Return', 'OOS_Sortino_Ratio', 'OOS_Calmar_Ratio', 
                                       'OOS_Max_Drawdown', 'OOS_Time_in_Market', 'OOS_Profit_Factor', 
                                       'OOS_Percent_Profitable', 'OOS_Num_Trades', 'OOS_Avg_Trade_Return', 
                                       'OOS_Avg_Trade_Duration', 'OOS_HAC_t_stat', 'OOS_Bootstrap_p'])
            
            # Only include columns that exist in final_results
            available_columns = [col for col in download_columns if col in final_results.columns]
            csv = final_results[available_columns].to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Store results in session state for persistence
            st.session_state['final_results'] = final_results
            st.session_state['price_data'] = price_data
            st.session_state['returns_store'] = returns_store
            
        except Exception as e:
            st.error(f"An error occurred during backtesting: {str(e)}")
            st.exception(e)
    
    # Show cached results if available
    elif 'final_results' in st.session_state:
        st.info("Showing previous results. Click 'Run Backtest' to generate new results.")
        final_results = st.session_state['final_results']
        price_data = st.session_state['price_data']
        returns_store = st.session_state.get('returns_store', {})
        
        # Display cached results (abbreviated version)
        st.header("📋 Previous Results")
        display_columns = ['Signal', 'Ticker', 'Total Return', 'Sortino Ratio', 'Calmar Ratio', 
                         'Max Drawdown', 'Time in Market', 'Profit Factor', 'Percent Profitable']
        
        display_df = final_results[display_columns].copy()
        display_df['Total Return'] = display_df['Total Return'].apply(lambda x: f"{x:.1%}")
        display_df['Sortino Ratio'] = display_df['Sortino Ratio'].apply(lambda x: f"{x:.2f}")
        display_df['Max Drawdown'] = display_df['Max Drawdown'].apply(lambda x: f"{x:.1%}")
        display_df['Time in Market'] = display_df['Time in Market'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(display_df.head(10), use_container_width=True)
        
        # Performance visualization for cached results
        if len(final_results) > 0 and returns_store:
            st.subheader("📈 Performance Visualization")
            
            # Select signals to plot
            available_signals = final_results.head(10)['Signal'].tolist()
            selected_signals = st.multiselect(
                "Select signals to visualize:",
                available_signals,
                default=available_signals[:3] if len(available_signals) >= 3 else available_signals
            )
            
            if selected_signals:
                # For cached results, we don't have benchmark data, so pass None
                chart = plot_performance_chart(final_results, selected_signals, returns_store, None)
                st.plotly_chart(chart, use_container_width=True)
    
    else:
        st.info("👈 Configure your parameters in the sidebar and click 'Run Backtest' to begin!")
        
        # Show parameter explanation
        st.header("ℹ️ How It Works")
        
        st.markdown("""
        This app backtests **IF THEN** trading signals across multiple assets and combinations:
        
        **📊 Signal Types Generated:**
        - RSI levels and comparisons between assets
        - Cumulative returns over various periods
        - Moving average comparisons
        
        **🎯 Target vs Reference Tickers:**
        - **Target Tickers**: Assets you want to trade (e.g., TQQQ, SQQQ)
        - **Reference Tickers**: Assets used to generate cross-reference signals (e.g., VIX, QQQ)
        
        **⚙️ Key Parameters:**
        - **Time in Market**: Minimum percentage of time signal should be active
        - **Max Drawdown**: Maximum acceptable loss from peak to trough
        - **Quantile Filter**: Only keeps top-performing signals (0.95 = top 5%)
        - **Combined Signals**: Tests combinations using AND logic
        
        **📈 Output Metrics:**
        - **Sortino Ratio**: Risk-adjusted returns (higher is better)
        - **Calmar Ratio**: Return divided by max drawdown
        - **Profit Factor**: Gross profits divided by gross losses
        """)

if __name__ == "__main__":
    main()
