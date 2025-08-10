import streamlit as st
import pandas as pd
import numpy as np
import itertools
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
def generate_signals(tickers, start):
    """Generate trading signals based on various technical indicators"""
    with st.spinner(f"Downloading data for {len(tickers)} tickers..."):
        price_data = yf.download(tickers, start=start, progress=False)['Close']
        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame(name=tickers[0])

    price_data = price_data.dropna()
    daily_returns = price_data.pct_change()
    log_returns = np.log(price_data / price_data.shift(1))

    signals = {}

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
            t: {p: daily_returns[t].rolling(p).mean() for p in range(10, 110, 10)}
            for t in tickers
        }
        std_cache = {
            t: {p: daily_returns[t].rolling(p).std() for p in range(10, 60, 10)}
            for t in tickers
        }

    # Define signal conditions
    rsi_levels = range(10, 100, 10)
    cumret_levels = [i / 100 for i in range(-10, 11, 2)]

    with st.spinner("Generating trading signals..."):
        # Generate RSI signals
        for t in tickers:
            for p, rsi in rsi_cache[t].items():
                for lvl in rsi_levels:
                    signals[f'RSI_{p}_{t}_GT_{lvl}'] = rsi > lvl
                    signals[f'RSI_{p}_{t}_LT_{lvl}'] = rsi < lvl

        # Generate RSI comparisons between tickers
        ticker_pairs = [(a, b) for a in tickers for b in tickers if a != b]
        # Optional: restrict to specific pairs (uncomment and modify as needed)
        # allowed_pairs = {("QQQ","TLT"), ("QQQ","VIXM"), ("SPY","IEF")}
        # ticker_pairs = [p for p in ticker_pairs if p in allowed_pairs]
        for t1, t2 in ticker_pairs:
            for p1 in rsi_cache[t1]:
                rsi1 = rsi_cache[t1][p1]
                for p2 in rsi_cache[t2]:
                    rsi2 = rsi_cache[t2][p2]
                    signals[f'RSI_{p1}_{t1}_GT_RSI_{p2}_{t2}'] = rsi1 > rsi2
                    signals[f'RSI_{p1}_{t1}_LT_RSI_{p2}_{t2}'] = rsi1 < rsi2

        # Generate Cumulative Return signals
        for t in tickers:
            for p, cum in cumret_cache[t].items():
                for lvl in cumret_levels:
                    signals[f'CUMRET_{p}_{t}_GT_{lvl}'] = cum > lvl
                    signals[f'CUMRET_{p}_{t}_LT_{lvl}'] = cum < lvl

        # Generate Cumulative Return comparisons between tickers
        for t1, t2 in ticker_pairs:
            for p1 in cumret_cache[t1]:
                r1 = cumret_cache[t1][p1]
                for p2 in cumret_cache[t2]:
                    r2 = cumret_cache[t2][p2]
                    signals[f'CUMRET_{p1}_{t1}_GT_CUMRET_{p2}_{t2}'] = r1 > r2
                    signals[f'CUMRET_{p1}_{t1}_LT_CUMRET_{p2}_{t2}'] = r1 < r2

        # Generate Moving Average signals
        for t1, t2 in ticker_pairs:
            for p1 in ma_cache[t1]:
                m1 = ma_cache[t1][p1]
                for p2 in ma_cache[t2]:
                    m2 = ma_cache[t2][p2]
                    signals[f'MA_{p1}_{t1}_GT_MA_{p2}_{t2}'] = m1 > m2
                    signals[f'MA_{p1}_{t1}_LT_MA_{p2}_{t2}'] = m1 < m2

        # Generate Standard Deviation signals
        for t1, t2 in ticker_pairs:
            for p1 in std_cache[t1]:
                s1 = std_cache[t1][p1]
                for p2 in std_cache[t2]:
                    s2 = std_cache[t2][p2]
                    signals[f'STD_{p1}_{t1}_GT_STD_{p2}_{t2}'] = s1 > s2
                    signals[f'STD_{p1}_{t1}_LT_STD_{p2}_{t2}'] = s1 < s2

    # Ensure all signals are Series with datetime index
    for k in signals:
        if not isinstance(signals[k], pd.Series):
            signals[k] = pd.Series(signals[k], index=price_data.index)

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

def mbb_p_ci(ret, block=10, B=1000):
    """Calculate moving-block bootstrap p-value and confidence intervals"""
    rng = np.random.default_rng(0)  # Fixed seed for reproducibility
    r = pd.Series(ret).dropna().values
    T = len(r)
    
    if T < block:
        return np.nan, (np.nan, np.nan)
    
    # Generate bootstrap samples
    starts = rng.integers(0, T-block+1, size=(B, int(np.ceil(T/block))))
    boot = [np.concatenate([r[s:s+block] for s in row])[:T].mean() for row in starts]
    
    # Calculate p-value and confidence intervals
    mu = r.mean()
    p = (np.sum(np.array(boot) >= mu) + 1) / (B + 1)  # One-sided p-value
    lo, hi = np.quantile(boot, [0.025, 0.975])  # 95% confidence interval
    
    return p, (lo, hi)

def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg False Discovery Rate control"""
    p = pd.Series(pvals).sort_values()
    m = len(p)
    q = np.minimum.accumulate((p[::-1] * m / np.arange(m, 0, -1)))[::-1]
    q = pd.Series(q, index=p.index)
    return q.reindex(pvals.index), (np.arange(1, m+1)/m*alpha).max()

def calculate_trust_gates(returns, signal_series, trade_metrics, hac_t_stat, bootstrap_ci, fdr_q_value, 
                         min_trades=30, min_exposure=0.10, min_sharpe=0.5, costs_per_trade=0.001, 
                         daily_slippage=0.0001):
    """Calculate comprehensive trust gates for signal validation"""
    
    # Gate 1: Sufficient sample size
    num_trades = trade_metrics.get('num_trades', 0)
    time_in_market = signal_series.mean()
    gate_sample = (num_trades >= min_trades) and (time_in_market >= min_exposure)
    
    # Gate 2: Statistical validity
    gate_hac = not pd.isna(hac_t_stat) and abs(hac_t_stat) > 2.0  # t-stat > 2 for significance
    gate_bootstrap = (not pd.isna(bootstrap_ci[0]) and not pd.isna(bootstrap_ci[1]) and 
                     bootstrap_ci[0] > 0)  # CI excludes 0
    gate_fdr = not pd.isna(fdr_q_value) and fdr_q_value <= 0.05
    
    # Gate 3: Costs included (simplified cost model)
    if num_trades > 0:
        # Estimate costs: per-trade commission + daily slippage
        total_costs = num_trades * costs_per_trade + len(returns) * daily_slippage
        net_return = returns.sum() - total_costs
        net_sharpe = (net_return / len(returns)) / (returns.std() + 1e-9) * np.sqrt(252)
        gate_costs = net_sharpe >= min_sharpe
    else:
        gate_costs = False
        net_sharpe = np.nan
    
    # Gate 4: Risk fit (basic checks)
    if len(returns) > 0:
        max_dd = (np.exp(returns.cumsum()) / np.exp(returns.cumsum()).cummax() - 1).min()
        gate_risk = max_dd > -0.5  # Max DD < 50%
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

def backtest_signals(signals: dict, price_data: pd.DataFrame, tickers: list, target_tickers: list, is_mask=None, oos_mask=None, min_exposure=0.05):
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
                is_boot_p, is_boot_ci = mbb_p_ci(is_returns, block=mbb_block_size, B=mbb_bootstrap_samples)
                
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
                oos_boot_p, oos_boot_ci = mbb_p_ci(oos_returns, block=mbb_block_size, B=mbb_bootstrap_samples)
                
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
                    'OOS_Bootstrap_CI_low': oos_boot_ci[0] if oos_boot_ci[0] is not np.nan else np.nan,
                    'OOS_Bootstrap_CI_high': oos_boot_ci[1] if oos_boot_ci[1] is not np.nan else np.nan
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
                boot_p, boot_ci = mbb_p_ci(returns, block=mbb_block_size, B=mbb_bootstrap_samples)
                
                # Apply minimum exposure guard to full period metrics
                if time_in_market < min_exposure:
                    sortino_ratio = np.nan
                    calmar_ratio = np.nan
                    profit_factor = np.nan
                    percent_profitable = np.nan
                
                oos_metrics = {}

            # Store return series separately
            returns_store[(signal_name, target_ticker)] = returns
            
            # Calculate trust gates
            trust_gates = calculate_trust_gates(
                returns, shifted_signal, 
                trade_metrics_full if 'trade_metrics_full' in locals() else {'num_trades': 0},
                hac_t_stat if 'hac_t_stat' in locals() else np.nan,
                boot_ci if 'boot_ci' in locals() else (np.nan, np.nan),
                fdr_q_value if 'fdr_q_value' in locals() else np.nan,
                min_trades=min_trades_trust,
                min_exposure=min_exposure_trust,
                min_sharpe=min_sharpe_trust,
                costs_per_trade=costs_per_trade,
                daily_slippage=daily_slippage
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
                'Bootstrap_CI_low': boot_ci[0] if 'boot_ci' in locals() and boot_ci[0] is not np.nan else np.nan,
                'Bootstrap_CI_high': boot_ci[1] if 'boot_ci' in locals() and boot_ci[1] is not np.nan else np.nan,
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

def generate_filtered_combinations(signals, backtest_results, max_signals):
    """Generate signal combinations filtered by ticker and top performance"""
    filtered_signals = {name: signals[name] for name in backtest_results['Signal']}
    
    signals_by_ticker = {}
    for row in backtest_results.itertuples():
        signals_by_ticker.setdefault(row.Ticker, []).append(row.Signal)

    combined = []
    for ticker, signal_names in signals_by_ticker.items():
        for r in range(2, max_signals + 1):
            for combo in combinations(signal_names, r):
                combined.append((combo, ticker))

    return combined

def backtest_combined_signals(combinations, signals, price_data, log_returns):
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
            'Percent Profitable': percent_profitable
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
    
    st.markdown("---")
    
    # Sidebar for parameters
    st.sidebar.header("📊 Backtesting Parameters")
    
    # Target tickers (assets to trade)
    st.sidebar.subheader("Target Tickers (Assets to Trade)")
    default_target = ['TQQQ', 'SQQQ', 'UVXY', 'PSQ']
    target_input = st.sidebar.text_area(
        "Target Tickers (comma-separated)",
        value=','.join(default_target),
        help="These are the assets you want to trade"
    )
    target = [t.strip().upper() for t in target_input.split(',') if t.strip()]
    
    # Reference tickers (for signal generation)
    st.sidebar.subheader("Reference Tickers (For Signal Generation)")
    default_reference = ['VIXM', 'QQQ', 'TLT', 'KMLM', 'IYT', 'CORP']
    reference_input = st.sidebar.text_area(
        "Reference Tickers (comma-separated)",
        value=','.join(default_reference),
        help="These assets will be used to generate cross-reference signals"
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
        help="Percentage of data used for in-sample training (0.7 = 70% IS, 30% OOS)"
    )
    
    # Signal combination parameters
    st.sidebar.subheader("Signal Combination")
    combination_limit = st.sidebar.slider(
        "Maximum Combined Signals",
        min_value=2,
        max_value=5,
        value=2,
        help="Maximum number of signals to combine using AND logic"
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
            help="Minimum percentage of time the signal should be active (0.025 = 2.5%)"
        )
    
    with col2:
        mdd = st.number_input(
            "Max Drawdown Limit",
            min_value=-1.0,
            max_value=-0.01,
            value=-0.5,
            step=0.05,
            format="%.2f",
            help="Maximum acceptable drawdown (-0.5 = 50% maximum drawdown)"
        )
    
    # Minimum exposure threshold
    min_exposure_threshold = st.sidebar.number_input(
        "Min Exposure Threshold",
        min_value=0.01,
        max_value=0.50,
        value=0.05,
        step=0.01,
        format="%.2f",
        help="Minimum time in market required (0.05 = 5% minimum exposure)"
    )
    
    # HAC lags parameter
    hac_lags = st.sidebar.number_input(
        "HAC Lags (0 = auto)",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help="Number of lags for HAC t-statistic (0 = automatic calculation)"
    )
    
    # Bootstrap parameters
    st.sidebar.subheader("Bootstrap Parameters")
    mbb_block_size = st.sidebar.number_input(
        "Block Size",
        min_value=5,
        max_value=50,
        value=10,
        step=1,
        help="Block size for moving-block bootstrap"
    )
    
    mbb_bootstrap_samples = st.sidebar.number_input(
        "Bootstrap Samples",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Number of bootstrap samples (higher = more accurate but slower)"
    )
    
    # FDR control parameter
    fdr_alpha = st.sidebar.number_input(
        "FDR Alpha Level",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%.2f",
        help="False Discovery Rate control level (0.05 = 5% FDR)"
    )
    
    # Benchmark selection
    st.sidebar.subheader("Benchmark Settings")
    use_benchmark = st.sidebar.checkbox(
        "Include Benchmark",
        value=True,
        help="Add benchmark comparison to performance charts"
    )
    
    benchmark_ticker = st.sidebar.selectbox(
        "Benchmark Ticker",
        options=['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'None'],
        index=0,
        help="Select benchmark for comparison"
    ) if use_benchmark else None
    
    # Trust gate parameters
    st.sidebar.subheader("Trust Gate Settings")
    min_trades_trust = st.sidebar.number_input(
        "Min Trades",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="Minimum number of trades for trust gate"
    )
    
    min_exposure_trust = st.sidebar.number_input(
        "Min Exposure",
        min_value=0.05,
        max_value=0.50,
        value=0.10,
        step=0.05,
        format="%.2f",
        help="Minimum time in market for trust gate"
    )
    
    min_sharpe_trust = st.sidebar.number_input(
        "Min Net Sharpe",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        format="%.1f",
        help="Minimum net Sharpe ratio after costs"
    )
    
    costs_per_trade = st.sidebar.number_input(
        "Costs per Trade",
        min_value=0.0001,
        max_value=0.01,
        value=0.001,
        step=0.0001,
        format="%.4f",
        help="Transaction costs per trade (0.001 = 0.1%)"
    )
    
    daily_slippage = st.sidebar.number_input(
        "Daily Slippage",
        min_value=0.00001,
        max_value=0.001,
        value=0.0001,
        step=0.00001,
        format="%.5f",
        help="Daily slippage/fees (0.0001 = 0.01%)"
    )
    
    quantile = st.sidebar.slider(
        "Performance Quantile Filter",
        min_value=0.50,
        max_value=0.99,
        value=0.95,
        step=0.05,
        help="Keep only signals above this performance quantile (0.95 = top 5%)"
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
            signals, price_data = generate_signals(all_tickers, start_date.strftime('%Y-%m-%d'))
            
            st.success(f"✅ Generated {len(signals)} signals for {len(all_tickers)} tickers")
            
            # Download benchmark data if requested
            benchmark_returns = None
            if use_benchmark and benchmark_ticker and benchmark_ticker != 'None':
                try:
                    benchmark_data = yf.download(benchmark_ticker, start=start_date.strftime('%Y-%m-%d'), progress=False)['Close']
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
            backtest_results, returns_store = backtest_signals(signals, price_data, all_tickers, target, is_mask, oos_mask, min_exposure_threshold)
            backtest_results = backtest_results.dropna()
            
            # Apply filters
            backtest_results = backtest_results[backtest_results['Time in Market'] > tim]
            backtest_results = backtest_results[backtest_results['Max Drawdown'] > mdd]
            backtest_results = backtest_results[backtest_results['Total Return'] > 0]
            
            st.info(f"📊 {len(backtest_results)} signals passed initial filters")
            
            # Apply FDR control to bootstrap p-values
            if 'Bootstrap_p' in backtest_results.columns:
                valid_pvals = backtest_results['Bootstrap_p'].dropna()
                if len(valid_pvals) > 0:
                    fdr_qvals, fdr_threshold = fdr_bh(valid_pvals, alpha=fdr_alpha)
                    backtest_results.loc[valid_pvals.index, 'FDR_q_value'] = fdr_qvals
                    
                    # Count significant signals
                    significant_signals = (fdr_qvals <= fdr_alpha).sum()
                    st.info(f"🔬 FDR Control: {significant_signals} signals significant at {fdr_alpha:.1%} FDR level")
                else:
                    st.warning("⚠️ No valid p-values for FDR control")
            
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
                combinations_filtered = generate_filtered_combinations(signals, backtest_filtered, combination_limit)
                
                if combinations_filtered:
                    log_returns = np.log(price_data / price_data.shift(1)).fillna(0)
                    backtest_df_combined, combined_returns_store = backtest_combined_signals(combinations_filtered, signals, price_data, log_returns)
                    
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
            
            # Trust Gate Summary
            if 'Trust_Status' in final_results.columns:
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
                
            # FDR Statistics
            if 'FDR_q_value' in final_results.columns:
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
            
            # Results table
            st.subheader("📊 Top Performing Signals (IS Rankings)")
            display_columns = ['Signal', 'Ticker', 'Trust_Status', 'Trust_Score', 'Gates_Passed', 'Total Return', 'Sortino Ratio', 'Calmar Ratio', 
                             'Max Drawdown', 'Time in Market', 'Profit Factor', 'Percent Profitable', 
                             'Num Trades', 'Avg Trade Return', 'Avg Trade Duration', 'HAC_t_stat', 'Bootstrap_p', 'FDR_q_value', 'Net_Sharpe']
            
            # Add OOS columns if available
            if 'OOS_Total_Return' in final_results.columns:
                display_columns.extend(['OOS_Total_Return', 'OOS_Sortino_Ratio', 'OOS_Calmar_Ratio', 
                                      'OOS_Max_Drawdown', 'OOS_Time_in_Market', 'OOS_Profit_Factor', 'OOS_Percent_Profitable',
                                      'OOS_Num_Trades', 'OOS_Avg_Trade_Return', 'OOS_Avg_Trade_Duration', 'OOS_HAC_t_stat', 'OOS_Bootstrap_p'])
            
            # Format the dataframe for display
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
            
            # Format OOS columns if they exist
            if 'OOS_Total_Return' in display_df.columns:
                display_df['OOS_Total_Return'] = display_df['OOS_Total_Return'].apply(lambda x: f"{x:.1%}")
                display_df['OOS_Sortino_Ratio'] = display_df['OOS_Sortino_Ratio'].apply(lambda x: f"{x:.2f}")
                display_df['OOS_Calmar_Ratio'] = display_df['OOS_Calmar_Ratio'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                display_df['OOS_Max_Drawdown'] = display_df['OOS_Max_Drawdown'].apply(lambda x: f"{x:.1%}")
                display_df['OOS_Time_in_Market'] = display_df['OOS_Time_in_Market'].apply(lambda x: f"{x:.1%}")
                display_df['OOS_Profit_Factor'] = display_df['OOS_Profit_Factor'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                display_df['OOS_Percent_Profitable'] = display_df['OOS_Percent_Profitable'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
                display_df['OOS_Num_Trades'] = display_df['OOS_Num_Trades'].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "0")
                display_df['OOS_Avg_Trade_Return'] = display_df['OOS_Avg_Trade_Return'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
                display_df['OOS_Avg_Trade_Duration'] = display_df['OOS_Avg_Trade_Duration'].apply(lambda x: f"{x:.0f}d" if pd.notnull(x) else "N/A")
                display_df['OOS_HAC_t_stat'] = display_df['OOS_HAC_t_stat'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
                display_df['OOS_Bootstrap_p'] = display_df['OOS_Bootstrap_p'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
            
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
            csv = final_results[display_columns].to_csv(index=False)
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
        - Standard deviation (volatility) comparisons
        
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
