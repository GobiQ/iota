import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

st.set_page_config(page_title="RSI Threshold Optimizer", layout="wide")

st.title("🚀 RSI Threshold Optimizer")
st.write("Optimize RSI thresholds using real market data from Yahoo Finance")

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI with standard 14-period window"""
    if len(prices) < window + 1:
        return pd.Series(index=prices.index, dtype=float)
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio (risk-adjusted return focused on downside risk)"""
    if len(returns) == 0:
        return 0
    
    # Convert annual risk-free rate to per-trade rate (approximate)
    rf_per_trade = risk_free_rate / 252  # Assume 252 trading days per year
    
    excess_returns = returns - rf_per_trade
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if excess_returns.mean() > 0 else 0
    
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    
    if downside_deviation == 0:
        return np.inf if excess_returns.mean() > 0 else 0
    
    return excess_returns.mean() / downside_deviation

def get_stock_data(ticker: str, start_date=None, end_date=None) -> pd.Series:
    """Fetch stock data using yfinance with optional date range"""
    try:
        stock = yf.Ticker(ticker)
        
        if start_date and end_date:
            data = stock.history(start=start_date, end=end_date)
        else:
            # Default to maximum available period
            data = stock.history(period="max")
        
        if data.empty:
            st.error(f"No data found for ticker: {ticker}")
            return None
        return data['Close']
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def analyze_rsi_signals(signal_prices: pd.Series, target_prices: pd.Series, rsi_threshold: float, comparison: str = "less_than", rsi_period: int = 14) -> Dict:
    """Analyze RSI signals for a specific threshold"""
    # Calculate RSI for the SIGNAL ticker using specified period
    signal_rsi = calculate_rsi(signal_prices, window=rsi_period)
    
    # Generate buy signals based on SIGNAL RSI threshold and comparison
    if comparison == "less_than":
        # "≤" configuration: Buy TARGET when SIGNAL RSI ≤ threshold, sell when SIGNAL RSI > threshold
        signals = (signal_rsi <= rsi_threshold).astype(int)
    else:  # greater_than
        # "≥" configuration: Buy TARGET when SIGNAL RSI ≥ threshold, sell when SIGNAL RSI < threshold
        signals = (signal_rsi >= rsi_threshold).astype(int)
    
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
    avg_hold_days = np.mean([trade['hold_days'] for trade in trades])
    sortino_ratio = calculate_sortino_ratio(returns)
    
    # Calculate annualized return
    total_days = (target_prices.index[-1] - target_prices.index[0]).days
    total_return = equity_curve.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (365 / total_days) - 1 if total_days > 0 else 0
    
    return {
        'total_trades': len(returns),
        'win_rate': win_rate,
        'avg_return': avg_return,
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
            'power': 0
        }
    
    # Align the equity curves on the same dates
    common_dates = strategy_equity_curve.index.intersection(benchmark_equity_curve.index)
    if len(common_dates) < 10:  # Need at least 10 data points for meaningful test
        return {
            't_statistic': 0,
            'p_value': 1.0,
            'confidence_level': 0,
            'significant': False,
            'effect_size': 0,
            'power': 0
        }
    
    strategy_aligned = strategy_equity_curve[common_dates]
    benchmark_aligned = benchmark_equity_curve[common_dates]
    
    # Calculate daily returns for both strategies
    strategy_returns = strategy_aligned.pct_change().dropna()
    benchmark_returns = benchmark_aligned.pct_change().dropna()
    
    # Ensure we have enough data points
    if len(strategy_returns) < 10 or len(benchmark_returns) < 10:
        return {
            't_statistic': 0,
            'p_value': 1.0,
            'confidence_level': 0,
            'significant': False,
            'effect_size': 0,
            'power': 0
        }
    
    # Perform one-tailed t-test to test if strategy BEATS benchmark
    t_stat, p_value_two_tail = stats.ttest_ind(strategy_returns.values, benchmark_returns.values)
    
    # Convert to one-tailed test
    if np.mean(strategy_returns.values) > np.mean(benchmark_returns.values):
        p_value_one_tail = p_value_two_tail / 2
        confidence_level = (1 - p_value_one_tail) * 100
        significant = p_value_one_tail < 0.05
    else:
        # Strategy underperforms benchmark
        p_value_one_tail = 1.0
        confidence_level = 0
        significant = False
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(strategy_returns) - 1) * np.var(strategy_returns.values, ddof=1) + 
                          (len(benchmark_returns) - 1) * np.var(benchmark_returns.values, ddof=1)) / 
                         (len(strategy_returns) + len(benchmark_returns) - 2))
    
    effect_size = (np.mean(strategy_returns.values) - np.mean(benchmark_returns.values)) / pooled_std if pooled_std > 0 else 0
    
    # Calculate statistical power (simplified)
    power = 0.8 if len(strategy_returns) > 30 and abs(effect_size) > 0.5 else 0.5
    
    return {
        't_statistic': t_stat,
        'p_value': p_value_one_tail,
        'confidence_level': confidence_level,
        'significant': significant,
        'effect_size': effect_size,
        'power': power
    }

def run_rsi_analysis(signal_ticker: str, target_ticker: str, rsi_min: float, rsi_max: float, comparison: str, 
                    start_date=None, end_date=None, rsi_period: int = 14, benchmark_ticker: str = "SPY") -> Tuple[pd.DataFrame, pd.Series]:
    """Run comprehensive RSI analysis across the specified range"""
    
    # Fetch data
    with st.spinner(f"Fetching data for {signal_ticker}..."):
        signal_data = get_stock_data(signal_ticker, start_date, end_date)
    
    with st.spinner(f"Fetching data for {target_ticker}..."):
        target_data = get_stock_data(target_ticker, start_date, end_date)
    
    # Fetch benchmark data for comparison - use user-selected benchmark
    with st.spinner(f"Fetching benchmark data ({benchmark_ticker})..."):
        benchmark_data = get_stock_data(benchmark_ticker, start_date, end_date)
    
    if signal_data is None or target_data is None or benchmark_data is None:
        return None, None
    
    # Align data on common dates
    common_dates = signal_data.index.intersection(target_data.index).intersection(benchmark_data.index)
    signal_data = signal_data[common_dates]
    target_data = target_data[common_dates]
    benchmark_data = benchmark_data[common_dates]
    
    # Create buy-and-hold benchmark
    benchmark = benchmark_data / benchmark_data.iloc[0]  # Normalize to start at 1.0
    
    # Calculate benchmark returns for statistical testing
    benchmark_returns = benchmark_data.pct_change().dropna()
    
    # Generate RSI thresholds (every 0.5)
    rsi_thresholds = np.arange(rsi_min, rsi_max + 0.5, 0.5)
    
    results = []
    
    progress_bar = st.progress(0)
    total_thresholds = len(rsi_thresholds)
    
    for i, threshold in enumerate(rsi_thresholds):
        analysis = analyze_rsi_signals(signal_data, target_data, threshold, comparison, rsi_period)
        
        # Calculate statistical significance
        strategy_equity_curve = analysis['equity_curve']
        if len(strategy_equity_curve) > 0:
            # Create benchmark equity curve that follows the same RSI conditions
            # This ensures we're comparing strategy vs benchmark under the same conditions
            signal_rsi = calculate_rsi(signal_data, window=rsi_period)
            
            # Generate buy signals for benchmark (same as strategy)
            if comparison == "less_than":
                benchmark_signals = (signal_rsi <= threshold).astype(int)
            else:  # greater_than
                benchmark_signals = (signal_rsi >= threshold).astype(int)
            
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
            
            benchmark_annualized = (benchmark.iloc[-1] - 1) * (365 / (benchmark.index[-1] - benchmark.index[0]).days)
            stats_result = calculate_statistical_significance(
                strategy_equity_curve, 
                benchmark_equity_curve,
                analysis['annualized_return'],
                benchmark_annualized
            )
        else:
            stats_result = {
                't_statistic': 0,
                'p_value': 1.0,
                'confidence_level': 0,
                'significant': False,
                'effect_size': 0,
                'power': 0
            }
        
        results.append({
            'RSI_Threshold': threshold,
            'Total_Trades': analysis['total_trades'],
            'Win_Rate': analysis['win_rate'],
            'Avg_Return': analysis['avg_return'],
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
            'power': stats_result['power']
        })
        
        progress_bar.progress((i + 1) / total_thresholds)
    
    return pd.DataFrame(results), benchmark

# Streamlit Interface
st.sidebar.header("📊 Configuration")

# Input fields with help tooltips
signal_ticker = st.sidebar.text_input("Signal Ticker", value="QQQ", help="The ticker that generates RSI signals. This is the stock/ETF whose RSI we'll use to decide when to buy/sell the target ticker.")

# RSI Period selection
rsi_period = st.sidebar.number_input("RSI Period (Days)", min_value=1, max_value=50, value=10, 
                                    help="How many days to look back when calculating RSI. 10 is more sensitive to recent changes than the standard 14. Lower numbers make RSI more responsive to recent market movements.")

# Conditional target ticker default based on RSI condition
comparison = st.sidebar.selectbox("RSI Condition", 
                               ["less_than", "greater_than"], 
                               format_func=lambda x: "RSI ≤ threshold" if x == "less_than" else "RSI ≥ threshold",
                               help="Choose when to buy: 'RSI ≤ threshold' means buy when RSI is low (oversold), 'RSI ≥ threshold' means buy when RSI is high (overbought).")

# Set default target ticker based on RSI condition
if comparison == "less_than":
    default_target = "TQQQ"
else:
    default_target = "VIXY"

target_ticker = st.sidebar.text_input("Target Ticker", value=default_target, help="The ticker to buy/sell based on the signal ticker's RSI. This is what you'll actually be trading.")

# Benchmark selection
benchmark_ticker = st.sidebar.selectbox("Benchmark", 
                                       ["SPY", "BIL"], 
                                       format_func=lambda x: "SPY (S&P 500)" if x == "SPY" else "BIL (Cash Equivalent)",
                                       help="Choose your benchmark for comparison: SPY represents the S&P 500 index, BIL represents cash (money market). This is what your signal will be compared against.")

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

if comparison == "less_than":
    default_min, default_max = 0, 40
    st.sidebar.write("Buy signals: Signal RSI ≤ threshold")
else:
    default_min, default_max = 70, 100
    st.sidebar.write("Buy signals: Signal RSI ≥ threshold")

rsi_min = st.sidebar.number_input("RSI Range Min", min_value=0.0, max_value=100.0, value=float(default_min), step=0.5, help="The lowest RSI threshold to test. For 'RSI ≤ threshold', try 20-40. For 'RSI ≥ threshold', try 60-80.")
rsi_max = st.sidebar.number_input("RSI Range Max", min_value=0.0, max_value=100.0, value=float(default_max), step=0.5, help="The highest RSI threshold to test. The app will test every 0.5 between min and max.")

if rsi_min >= rsi_max:
    st.sidebar.error("RSI Min must be less than RSI Max")

# Add the Run Analysis button to the sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🚀 Run RSI Analysis", type="primary", use_container_width=True):
    if rsi_min < rsi_max and (not use_date_range or (start_date and end_date and start_date < end_date)):
        try:
            results_df, benchmark = run_rsi_analysis(signal_ticker, target_ticker, rsi_min, rsi_max, comparison, start_date, end_date, rsi_period, benchmark_ticker)
            
            if results_df is not None and benchmark is not None and not results_df.empty:
                # Store analysis results in session state
                st.session_state['results_df'] = results_df
                st.session_state['benchmark'] = benchmark
                st.session_state['signal_data'] = get_stock_data(signal_ticker, start_date, end_date)
                st.session_state['benchmark_data'] = get_stock_data(benchmark_ticker, start_date, end_date)
                st.session_state['rsi_period'] = rsi_period
                st.session_state['comparison'] = comparison
                st.session_state['benchmark_ticker'] = benchmark_ticker
                st.session_state['analysis_completed'] = True
                
                st.sidebar.success("✅ Analysis completed successfully!")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error during analysis: {str(e)}")
    else:
        if rsi_min >= rsi_max:
            st.sidebar.error("Please ensure RSI Min is less than RSI Max")
        if use_date_range and (not start_date or not end_date or start_date >= end_date):
            st.sidebar.error("Please ensure start date is before end date")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎯 Analysis Configuration")
    st.write(f"**Signal Ticker:** {signal_ticker} (generates RSI signals)")
    st.write(f"**Target Ticker:** {target_ticker} (buy/sell based on signals)")
    st.write(f"**Benchmark:** {benchmark_ticker} ({'S&P 500' if benchmark_ticker == 'SPY' else 'Cash Equivalent'})")
    st.write(f"**RSI Period:** {rsi_period}-day RSI")
    st.write(f"**RSI Condition:** {signal_ticker} RSI {'≤' if comparison == 'less_than' else '≥'} threshold")
    st.write(f"**RSI Range:** {rsi_min} - {rsi_max}")
    if use_date_range and start_date and end_date:
        st.write(f"**Date Range:** {start_date} to {end_date}")
    else:
        st.write(f"**Date Range:** Maximum available data")

with col2:
    st.subheader("📋 Signal Logic")
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
    
    # Display results table
    st.subheader("📊 RSI Analysis Results")
    st.info("💡 **What this shows:** This table displays all the RSI thresholds tested and their performance metrics. Each row represents a different RSI level and shows how well that strategy performed.")
    
    # Format the dataframe for display
    display_df = results_df.copy()
    
    # Check if required columns exist before formatting
    required_columns = ['Win_Rate', 'Avg_Return', 'Total_Return', 'annualized_return', 
                      'Sortino_Ratio', 'Avg_Hold_Days', 'Return_Std', 'Best_Return', 
                      'Worst_Return', 'Final_Equity', 'confidence_level', 'significant', 'effect_size']
    
    missing_columns = [col for col in required_columns if col not in results_df.columns]
    if missing_columns:
        st.error(f"Missing columns in results: {missing_columns}")
        st.stop()
    
    # Format the columns for display
    display_df['Win_Rate'] = display_df['Win_Rate'].apply(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else x)
    display_df['Avg_Return'] = display_df['Avg_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Total_Return'] = display_df['Total_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Annualized_Return'] = display_df['annualized_return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Sortino_Ratio'] = display_df['Sortino_Ratio'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not np.isinf(x) else "∞" if isinstance(x, (int, float)) and np.isinf(x) else x)
    display_df['Avg_Hold_Days'] = display_df['Avg_Hold_Days'].apply(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x)
    display_df['Return_Std'] = display_df['Return_Std'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Best_Return'] = display_df['Best_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Worst_Return'] = display_df['Worst_Return'].apply(lambda x: f"{x:.3%}" if isinstance(x, (int, float)) else x)
    display_df['Final_Equity'] = display_df['Final_Equity'].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    display_df['Confidence_Level'] = display_df['confidence_level'].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    display_df['Significant'] = display_df['significant'].apply(lambda x: "✓" if x else "✗")
    display_df['Effect_Size'] = display_df['effect_size'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    
    # Drop the equity_curve and trades columns for display
    display_cols = ['RSI_Threshold', 'Total_Trades', 'Win_Rate', 'Avg_Return', 
                   'Total_Return', 'Annualized_Return', 'Sortino_Ratio', 'Final_Equity', 'Avg_Hold_Days', 
                   'Return_Std', 'Best_Return', 'Worst_Return', 'Confidence_Level', 'Significant', 'Effect_Size']
    
    # Check if all display columns exist
    missing_display_cols = [col for col in display_cols if col not in display_df.columns]
    if missing_display_cols:
        st.error(f"Missing display columns: {missing_display_cols}")
        st.stop()
    
    st.dataframe(display_df[display_cols], use_container_width=True)
    
    # Find best strategies (needed for subsequent sections)
    best_sortino_idx = display_df['Sortino_Ratio'].idxmax()
    best_annualized_idx = display_df['annualized_return'].idxmax()
    best_winrate_idx = display_df['Win_Rate'].idxmax()
    best_total_return_idx = display_df['Total_Return'].idxmax()
    
    # Statistical Significance Analysis
    st.subheader("📊 Statistical Significance Analysis")
    stored_benchmark_ticker = st.session_state.get('benchmark_ticker', 'SPY')
    benchmark_name = f"{stored_benchmark_ticker} ({'S&P 500' if stored_benchmark_ticker == 'SPY' else 'Cash Equivalent'})"
    st.info(f"💡 **What this shows:** This section determines whether your signal's performance is statistically significant - meaning the results are likely not due to chance. It compares your signal against {benchmark_name} under the same conditions to see if your target ticker choice is actually better.")
    
    # Filter signals with trades
    valid_signals = display_df[display_df['Total_Trades'] > 0].copy()
    
    if not valid_signals.empty:
        # Create significance summary
        significant_signals = valid_signals[valid_signals['significant'] == True]
        
        # Effect size vs confidence level
        st.subheader("📊 Effect size vs Confidence Level Analysis")
        st.info(f"💡 **What this shows:** This scatter plot helps you understand the relationship between statistical significance and practical importance. Each point represents a signal - the position shows how confident we are (confidence level) and how much better/worse the signal is compared to {benchmark_name} (effect size).")
        
        # Create scatter plot with hover information
        fig_effect = go.Figure()
        
        # Add points for significant signals (green)
        significant_data = valid_signals[valid_signals['significant'] == True]
        if not significant_data.empty:
            fig_effect.add_trace(go.Scatter(
                x=significant_data['effect_size'],
                y=significant_data['confidence_level'],
                mode='markers',
                name='Significant Signals',
                marker=dict(color='green', size=8),
                hovertemplate='<b>RSI %{text}</b><br>' +
                            'Effect Size: %{x:.3f}<br>' +
                            'Confidence: %{y:.1f}%<br>' +
                            'Significant: ✓<extra></extra>',
                text=[f"{row['RSI_Threshold']}" for _, row in significant_data.iterrows()]
            ))
        
        # Add points for non-significant signals (red)
        non_significant_data = valid_signals[valid_signals['significant'] == False]
        if not non_significant_data.empty:
            fig_effect.add_trace(go.Scatter(
                x=non_significant_data['effect_size'],
                y=non_significant_data['confidence_level'],
                mode='markers',
                name='Non-Significant Signals',
                marker=dict(color='red', size=8),
                hovertemplate='<b>RSI %{text}</b><br>' +
                            'Effect Size: %{x:.3f}<br>' +
                            'Confidence: %{y:.1f}%<br>' +
                            'Significant: ✗<extra></extra>',
                text=[f"{row['RSI_Threshold']}" for _, row in non_significant_data.iterrows()]
            ))
        
        # Add reference lines
        fig_effect.add_hline(y=95, line_dash="dash", line_color="red", 
                                           annotation_text="95% Confidence")
        fig_effect.add_vline(x=0, line_dash="dash", line_color="gray", 
                                           annotation_text="No Effect")
        
        fig_effect.update_layout(
            title="Effect Size vs Confidence Level",
            xaxis_title="Effect Size (Cohen's d)",
            yaxis_title="Confidence Level (%)",
            hovermode='closest'
        )
        
        st.plotly_chart(fig_effect, use_container_width=True)
        
        # Detailed explanation
        with st.expander("📚 Understanding Effect Size vs Confidence Level"):
            st.write("""
            **What This Chart Tells You:**
            
            **🎯 Quadrant Analysis:**
            - **Top Right (Green)**: High confidence + Large positive effect = Best signals
            - **Top Left (Green)**: High confidence + Large negative effect = Poor signals  
            - **Bottom Right (Red)**: Low confidence + Large positive effect = Promising but uncertain
            - **Bottom Left (Red)**: Low confidence + Small effect = Weak signals
            
            **📊 Effect Size Interpretation:**
            - **0.0**: No difference from the benchmark
            - **0.2-0.5**: Small effect (signal slightly better/worse)
            - **0.5-0.8**: Medium effect (meaningful difference)
            - **>0.8**: Large effect (substantial outperformance/underperformance)
            
            **📈 Confidence Level Meaning:**
            - **>95%**: Very strong evidence the signal differs from the benchmark
            - **90-95%**: Strong evidence of difference
            - **80-90%**: Moderate evidence
            - **<80%**: Weak evidence, results could be due to chance
            
            **🎯 What to Look For:**
            - **Green dots in top-right**: Your best signals (high confidence + large positive effect)
            - **Green dots in top-left**: Signals to avoid (high confidence + large negative effect)
            - **Red dots**: Signals with uncertain results (low confidence)
            - **Dots near the center line**: Signals with minimal effect on performance
            
            **💡 Practical Guidance:**
            - Focus on signals in the top-right quadrant
            - Be cautious of signals with high confidence but negative effect size
            - Consider sample size - more data points generally lead to higher confidence
            - Remember that past performance doesn't guarantee future results
            """)
        
        # Total Return vs Confidence Level Analysis
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
        
        st.plotly_chart(fig_total_return, use_container_width=True)
        
        # Sortino Ratio vs Confidence Level Analysis
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
        
        st.plotly_chart(fig_sortino, use_container_width=True)
        
        # Confidence Level vs RSI Threshold Analysis
        st.subheader("📊 Confidence Level vs RSI Threshold Analysis")
        st.info("💡 **What this shows:** This scatter plot shows how confidence levels vary across different RSI thresholds. The size of each point indicates the effect size - larger points mean stronger effects. This helps you identify which RSI levels are most reliable and impactful.")
        
        # Create scatter plot for confidence vs RSI threshold
        fig_confidence_rsi = go.Figure()
        
        # Add points for significant signals (green)
        significant_data = valid_signals[valid_signals['significant'] == True]
        if not significant_data.empty:
            fig_confidence_rsi.add_trace(go.Scatter(
                x=significant_data['RSI_Threshold'],
                y=significant_data['confidence_level'],
                mode='markers',
                name='Significant Signals',
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
        
        # Add points for non-significant signals (red)
        non_significant_data = valid_signals[valid_signals['significant'] == False]
        if not non_significant_data.empty:
            fig_confidence_rsi.add_trace(go.Scatter(
                x=non_significant_data['RSI_Threshold'],
                y=non_significant_data['confidence_level'],
                mode='markers',
                name='Non-Significant Signals',
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
        fig_confidence_rsi.add_hline(y=80, line_dash="dash", line_color="orange", 
                                   annotation_text="80% Confidence")
        
        fig_confidence_rsi.update_layout(
            title="Confidence Level vs RSI Threshold (Point Size = Effect Size)",
            xaxis_title="RSI Threshold",
            yaxis_title="Confidence Level (%)",
            hovermode='closest',
            showlegend=True
        )
        
        st.plotly_chart(fig_confidence_rsi, use_container_width=True)
        
        # Add explanation for the new chart
        with st.expander("📚 Understanding Confidence vs RSI Threshold"):
            st.write("""
            **What This Chart Tells You:**
            
            **📊 X-Axis (RSI Threshold):**
            - Shows different RSI levels tested
            - Helps identify which RSI ranges are most effective
            
            **📈 Y-Axis (Confidence Level):**
            - Higher values = stronger statistical evidence
            - Above 95% = highly significant
            - 80-95% = borderline significant
            - Below 80% = weak evidence
            
            **🔴 Point Size (Effect Size):**
            - Larger points = stronger effects (bigger differences from the benchmark)
            - Smaller points = weaker effects
            - Size is proportional to the absolute effect size
            
            **🎯 Color Coding:**
            - **Green points**: Statistically significant signals
            - **Red points**: Non-significant signals
            
            **💡 What to Look For:**
            - **Large green points high on the chart**: Best signals (high confidence + large effect)
            - **Clusters of large points**: RSI ranges with consistent strong performance
            - **Small red points low on the chart**: Weak signals to avoid
            - **Patterns**: Look for RSI ranges where confidence and effect size are consistently high
            
            **🔍 Practical Insights:**
            - Identify optimal RSI ranges for your signal
            - Spot RSI levels that consistently produce significant results
            - Avoid RSI ranges with low confidence or small effects
            - Understand the relationship between RSI levels and statistical reliability
            """)
        
        # Download results
        st.subheader("📥 Download Results")
        st.info("💡 **What this does:** Download your analysis results as a CSV file that you can open in Excel or other spreadsheet programs. This includes all the performance metrics for every RSI threshold tested.")
        # Use the original column names from results_df for CSV download
        download_cols = ['RSI_Threshold', 'Total_Trades', 'Win_Rate', 'Avg_Return', 
                       'Total_Return', 'annualized_return', 'Sortino_Ratio', 'Final_Equity', 'Avg_Hold_Days', 
                       'Return_Std', 'Best_Return', 'Worst_Return', 'confidence_level', 'significant', 'effect_size']
        csv = st.session_state['results_df'][download_cols].to_csv(index=False)
        filename_suffix = f"_{start_date}_{end_date}" if use_date_range and start_date and end_date else "_max_range"
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"rsi_analysis_{signal_ticker}_{target_ticker}{filename_suffix}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Top significant signals
        if len(significant_signals) > 0:
            st.subheader("🏆 Top Statistically Significant Signals")
            
            # Sort by annualized return (most profitable) instead of confidence level
            # Use the original results_df for sorting since it has numerical values
            original_significant_signals = st.session_state['results_df'][st.session_state['results_df']['significant'] == True]
            top_significant = original_significant_signals.nlargest(5, 'annualized_return')
            
            # Multiple Signal Comparison for Significant Signals
            st.subheader("📊 Most Profitable Significant Signals Comparison")
            st.info(f"💡 **What this shows:** This chart compares the top 5 most profitable statistically significant signals against {benchmark_name} buy-and-hold. Each line represents a different RSI threshold that showed significant outperformance. The signals are ranked by annualized return, showing the most profitable signals first.")
            
            # Create comparison chart with all significant signals
            fig_comparison = go.Figure()
            
            # Add benchmark
            fig_comparison.add_trace(go.Scatter(
                x=benchmark.index,
                y=benchmark.values,
                mode='lines',
                name=f"{benchmark_name} Buy & Hold",
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Add significant signals
            colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
            for i, (idx, row) in enumerate(top_significant.iterrows()):
                if row['equity_curve'] is not None:
                    color = colors[i % len(colors)]
                    fig_comparison.add_trace(go.Scatter(
                        x=row['equity_curve'].index,
                        y=row['equity_curve'].values,
                        mode='lines',
                        name=f"RSI {row['RSI_Threshold']} ({row['annualized_return']:.3%} annualized)",
                        line=dict(color=color, width=2)
                    ))
            
            fig_comparison.update_layout(
                title=f"Most Profitable Significant Signals Comparison vs {benchmark_name}",
                xaxis_title="Date",
                yaxis_title="Equity Value",
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Highest Sortino Significant Signals Comparison
            st.subheader("📊 Highest Sortino Significant Signals Comparison")
            st.info(f"💡 **What this shows:** This chart compares the top 5 signals with the highest Sortino ratios (best risk-adjusted returns) among statistically significant signals against {benchmark_name} buy-and-hold. Each line represents a different RSI threshold that showed significant outperformance with excellent risk-adjusted performance. The signals are ranked by Sortino ratio, showing the best risk-adjusted returns first.")
            
            # Sort by Sortino ratio (best risk-adjusted returns) instead of annualized return
            # Use the original results_df for sorting since it has numerical values
            original_significant_signals = st.session_state['results_df'][st.session_state['results_df']['significant'] == True]
            top_sortino_significant = original_significant_signals.nlargest(5, 'Sortino_Ratio')
            
            # Create comparison chart with highest Sortino signals
            fig_sortino_comparison = go.Figure()
            
            # Add benchmark
            fig_sortino_comparison.add_trace(go.Scatter(
                x=benchmark.index,
                y=benchmark.values,
                mode='lines',
                name=f"{benchmark_name} Buy & Hold",
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Add significant signals with highest Sortino ratios
            colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
            for i, (idx, row) in enumerate(top_sortino_significant.iterrows()):
                if row['equity_curve'] is not None:
                    color = colors[i % len(colors)]
                    fig_sortino_comparison.add_trace(go.Scatter(
                        x=row['equity_curve'].index,
                        y=row['equity_curve'].values,
                        mode='lines',
                        name=f"RSI {row['RSI_Threshold']} (Sortino: {row['Sortino_Ratio']:.2f}, {row['annualized_return']:.3%} annualized)",
                        line=dict(color=color, width=2)
                    ))
            
            fig_sortino_comparison.update_layout(
                title=f"Highest Sortino Significant Signals Comparison vs {benchmark_name}",
                xaxis_title="Date",
                yaxis_title="Equity Value",
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_sortino_comparison, use_container_width=True)
            
            # Individual signal details
            st.subheader("📈 Individual Signal Details (95%+ Confidence)")
            st.info("💡 **What this shows:** Each expandable section shows detailed information about a specific signal with 95% or above confidence level, including performance metrics, statistical significance, and an individual equity curve comparing that signal to the benchmark.")
            
            # Sort options for significant signals
            col1, col2 = st.columns([1, 3])
            with col1:
                significant_sort_by = st.selectbox(
                    "Sort by:",
                    ["RSI_Threshold", "confidence_level", "Sortino_Ratio", "Win_Rate", "annualized_return", "Total_Return"],
                    format_func=lambda x: {
                        "RSI_Threshold": "RSI Threshold",
                        "confidence_level": "Confidence Level", 
                        "Sortino_Ratio": "Sortino Ratio",
                        "Win_Rate": "Win Rate",
                        "annualized_return": "Annualized Return",
                        "Total_Return": "Total Return"
                    }[x],
                    key="significant_sort"
                )
            with col2:
                significant_sort_order = st.radio(
                    "Order:",
                    ["ascending", "descending"],
                    horizontal=True,
                    key="significant_order"
                )
            
            # Sort significant signals
            if significant_sort_order == "ascending":
                sorted_significant = significant_signals.sort_values(significant_sort_by)
            else:
                sorted_significant = significant_signals.sort_values(significant_sort_by, ascending=False)

            for idx, row in sorted_significant.iterrows():
                with st.expander(f"RSI {row['RSI_Threshold']} - {row['confidence_level']:.1f}% Confidence"):
                    # Performance metrics - comprehensive display
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Trades", row['Total_Trades'], 
                                 help="The number of buy/sell transactions the signal made. More trades can mean more opportunities but also more transaction costs.")
                        # Calculate trades won from original win rate data
                        original_row = results_df.loc[idx] if 'analysis_completed' in st.session_state else row
                        win_rate_decimal = original_row['Win_Rate']  # This is already a decimal (0.0 to 1.0)
                        trades_won = int(row['Total_Trades'] * win_rate_decimal)
                        st.metric("Trades Won", trades_won, 
                                 help="The number of profitable trades out of all trades made. Shows how many times the signal was successful.")
                    
                    with col2:
                        st.metric("P-value", f"{row['p_value']:.4f}", 
                                 help="Probability that the results happened by chance. Lower values are better - under 0.05 means statistically significant.")
                        st.metric("Confidence Level", f"{row['confidence_level']:.1f}%", 
                                 help="How certain we are that the signal outperforms the benchmark. Higher percentage means stronger evidence the signal works.")
                    
                    with col3:
                        st.metric("Total Return", f"{row['Total_Return']:.3%}", 
                                 help="The total percentage gain or loss over the entire period. Shows how much money you would have made (or lost) if you invested $100.")
                        st.metric("Annualized Return", f"{row['annualized_return']:.3%}", 
                                 help="The yearly return rate, useful for comparing signals over different time periods. Shows how much you'd earn per year on average.")
                    
                    with col4:
                        st.metric("Win Rate", f"{row['Win_Rate']:.1%}", 
                                 help="Percentage of trades that were profitable. A higher win rate means the signal wins more often than it loses.")
                        st.metric("Sortino Ratio", f"{row['Sortino_Ratio']:.2f}" if not np.isinf(row['Sortino_Ratio']) else "∞", 
                                 help="Risk-adjusted return measure that focuses on downside risk. Higher is better - it shows good returns with less risk of big losses.")
                    
                    # Additional metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Avg Hold Days", f"{row['Avg_Hold_Days']:.1f}", 
                                 help="Average number of days the signal held each position. Shorter holds mean more frequent trading, longer holds mean more patience.")
                        st.metric("Effect Size", f"{row['effect_size']:.2f}", 
                                 help="How much better/worse the signal is compared to the benchmark. Positive means it beats the benchmark, negative means it underperforms.")
                    
                    with col2:
                        st.metric("Best Return", f"{row['Best_Return']:.3%}", 
                                 help="The best single trade return achieved by this signal. Shows the signal's potential upside.")
                        st.metric("Worst Return", f"{row['Worst_Return']:.3%}", 
                                 help="The worst single trade return achieved by this signal. Shows the signal's potential downside risk.")
                    
                    with col3:
                        st.metric("Return Std Dev", f"{row['Return_Std']:.3%}", 
                                 help="Standard deviation of trade returns. Lower values mean more consistent performance, higher values mean more volatile results.")
                        st.metric("Final Equity", f"{row['Final_Equity']:.3f}", 
                                 help="The final value of a $1 investment. Shows how much your money would have grown (or shrunk) over the period.")
                    
                    with col4:
                        st.metric("T-statistic", f"{row['t_statistic']:.3f}", 
                                 help="Statistical measure of how different the signal is from the benchmark. Examples: 2.0 = moderate difference, 3.0 = strong difference, 4.0+ = very strong difference. Higher absolute values mean stronger evidence of a real difference.")
                        st.metric("Power", f"{row['power']:.2f}", 
                                 help="How likely the test is to detect a real difference if one exists. Higher power means more reliable statistical results.")
                    
                    # Final row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Significant", "✓" if row['significant'] else "✗", 
                                 help="Whether the signal reached statistical significance (p < 0.05). ✓ means strong evidence, ✗ means results could be due to chance.")
                    
                    # Show equity curve for this signal
                    if row['equity_curve'] is not None:
                        st.subheader("📈 Equity Curve Comparison")
                        
                        # Create benchmark equity curve that follows the same RSI conditions
                        signal_rsi = calculate_rsi(st.session_state['signal_data'], window=st.session_state['rsi_period'])
                        
                        # Generate buy signals for benchmark (same as signal)
                        if st.session_state['comparison'] == "less_than":
                            benchmark_signals = (signal_rsi <= row['RSI_Threshold']).astype(int)
                        else:  # greater_than
                            benchmark_signals = (signal_rsi >= row['RSI_Threshold']).astype(int)
                        
                        # Calculate benchmark equity curve using benchmark prices (same logic as signal)
                        benchmark_equity_curve = pd.Series(1.0, index=st.session_state['benchmark_data'].index)
                        current_equity = 1.0
                        in_position = False
                        entry_equity = 1.0
                        entry_price = None
                        
                        for date in st.session_state['benchmark_data'].index:
                            current_signal = benchmark_signals[date] if date in benchmark_signals.index else 0
                            current_price = st.session_state['benchmark_data'][date]
                            
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
                            final_price = st.session_state['benchmark_data'].iloc[-1]
                            trade_return = (final_price - entry_price) / entry_price
                            current_equity = entry_equity * (1 + trade_return)
                            benchmark_equity_curve.iloc[-1] = current_equity
                        
                        fig_sig = go.Figure()
                        
                        fig_sig.add_trace(go.Scatter(
                            x=row['equity_curve'].index,
                            y=row['equity_curve'].values,
                            mode='lines',
                            name=f'Signal (RSI {row["RSI_Threshold"]})',
                            line=dict(color='green', width=2)
                        ))
                        
                        fig_sig.add_trace(go.Scatter(
                            x=benchmark_equity_curve.index,
                            y=benchmark_equity_curve.values,
                            mode='lines',
                            name=f'{benchmark_name} (Same RSI {row["RSI_Threshold"]} Conditions)',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig_sig.add_trace(go.Scatter(
                            x=benchmark.index,
                            y=benchmark.values,
                            mode='lines',
                            name=f'{benchmark_name} Buy & Hold',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        fig_sig.update_layout(
                            title=f"Equity Curve - RSI {row['RSI_Threshold']} ({row['confidence_level']:.1f}% Confidence)",
                            xaxis_title="Date",
                            yaxis_title="Equity Value",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_sig, use_container_width=True)
                        
                        # Add explanation
                        st.info(f"💡 **What this shows:** The green line shows your signal performance. The blue line shows {benchmark_name} performance under the same RSI conditions. The red dashed line shows {benchmark_name} buy-and-hold. This helps you see if your target ticker choice beats {benchmark_name} when the same RSI signals are applied.")
        else:
            st.warning("No signals reached statistical significance (p < 0.05)")
        
        # Signals with 80-95% confidence (borderline significant)
        borderline_signals = valid_signals[(valid_signals['confidence_level'] >= 80) & 
                                           (valid_signals['confidence_level'] < 95)].copy()
        
        if not borderline_signals.empty:
            st.subheader("📊 Borderline Significant Signals (80-95% Confidence)")
            st.info("💡 **What this shows:** These signals show promising results but don't quite reach the 95% confidence threshold. They may still be worth considering, especially if they have good performance metrics.")
            
            # Sort options for borderline signals
            col1, col2 = st.columns([1, 3])
            with col1:
                borderline_sort_by = st.selectbox(
                    "Sort by:",
                    ["RSI_Threshold", "confidence_level", "Sortino_Ratio", "Win_Rate", "annualized_return", "Total_Return"],
                    format_func=lambda x: {
                        "RSI_Threshold": "RSI Threshold",
                        "confidence_level": "Confidence Level", 
                        "Sortino_Ratio": "Sortino Ratio",
                        "Win_Rate": "Win Rate",
                        "annualized_return": "Annualized Return",
                        "Total_Return": "Total Return"
                    }[x],
                    key="borderline_sort"
                )
            with col2:
                borderline_sort_order = st.radio(
                    "Order:",
                    ["ascending", "descending"],
                    horizontal=True,
                    key="borderline_order"
                )
            
            # Sort borderline signals
            if borderline_sort_order == "ascending":
                sorted_borderline = borderline_signals.sort_values(borderline_sort_by)
            else:
                sorted_borderline = borderline_signals.sort_values(borderline_sort_by, ascending=False)

            for idx, row in sorted_borderline.iterrows():
                with st.expander(f"RSI {row['RSI_Threshold']} - {row['confidence_level']:.1f}% Confidence"):
                    # Performance metrics with hover tooltips
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Trades", row['Total_Trades'], 
                                 help="The number of buy/sell transactions the signal made. More trades can mean more opportunities but also more transaction costs.")
                        # Calculate trades won from original win rate data
                        original_row = results_df.loc[idx] if 'analysis_completed' in st.session_state else row
                        win_rate_decimal = original_row['Win_Rate']  # This is already a decimal (0.0 to 1.0)
                        trades_won = int(row['Total_Trades'] * win_rate_decimal)
                        st.metric("Trades Won", trades_won, 
                                 help="The number of profitable trades out of all trades made. Shows how many times the signal was successful.")
                    
                    with col2:
                        st.metric("P-value", f"{row['p_value']:.4f}", 
                                 help="Probability that the results happened by chance. Lower values are better - under 0.05 means statistically significant.")
                        st.metric("Confidence Level", f"{row['confidence_level']:.1f}%", 
                                 help="How certain we are that the signal outperforms the benchmark. Higher percentage means stronger evidence the signal works.")
                    
                    with col3:
                        st.metric("Total Return", f"{row['Total_Return']:.3%}", 
                                 help="The total percentage gain or loss over the entire period. Shows how much money you would have made (or lost) if you invested $100.")
                        st.metric("Annualized Return", f"{row['annualized_return']:.3%}", 
                                 help="The yearly return rate, useful for comparing signals over different time periods. Shows how much you'd earn per year on average.")
                    
                    with col4:
                        st.metric("Win Rate", f"{row['Win_Rate']:.1%}", 
                                 help="Percentage of trades that were profitable. A higher win rate means the signal wins more often than it loses.")
                        st.metric("Sortino Ratio", f"{row['Sortino_Ratio']:.2f}" if not np.isinf(row['Sortino_Ratio']) else "∞", 
                                 help="Risk-adjusted return measure that focuses on downside risk. Higher is better - it shows good returns with less risk of big losses.")
                    
                    # Additional metrics with hover tooltips
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Avg Hold Days", f"{row['Avg_Hold_Days']:.1f}", 
                                 help="Average number of days the signal held each position. Shorter holds mean more frequent trading, longer holds mean more patience.")
                        st.metric("Effect Size", f"{row['effect_size']:.2f}", 
                                 help="How much better/worse the signal is compared to the benchmark. Positive means it beats the benchmark, negative means it underperforms.")
                    
                    with col2:
                        st.metric("Best Return", f"{row['Best_Return']:.3%}", 
                                 help="The best single trade return achieved by this signal. Shows the signal's potential upside.")
                        st.metric("Worst Return", f"{row['Worst_Return']:.3%}", 
                                 help="The worst single trade return achieved by this signal. Shows the signal's potential downside risk.")
                    
                    with col3:
                        st.metric("Return Std Dev", f"{row['Return_Std']:.3%}", 
                                 help="Standard deviation of trade returns. Lower values mean more consistent performance, higher values mean more volatile results.")
                        st.metric("Final Equity", f"{row['Final_Equity']:.3f}", 
                                 help="The final value of a $1 investment. Shows how much your money would have grown (or shrunk) over the period.")
                    
                    with col4:
                        st.metric("T-statistic", f"{row['t_statistic']:.3f}", 
                                 help="Statistical measure of how different the signal is from the benchmark. Examples: 2.0 = moderate difference, 3.0 = strong difference, 4.0+ = very strong difference. Higher absolute values mean stronger evidence of a real difference.")
                        st.metric("Power", f"{row['power']:.2f}", 
                                 help="How likely the test is to detect a real difference if one exists. Higher power means more reliable statistical results.")
                    
                    # Final row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Significant", "✓" if row['significant'] else "✗", 
                                 help="Whether the signal reached statistical significance (p < 0.05). ✓ means strong evidence, ✗ means results could be due to chance.")
                    
                    # Show equity curve for this borderline signal
                    if row['equity_curve'] is not None:
                        st.subheader("📈 Equity Curve Comparison")
                        
                        # Create benchmark equity curve that follows the same RSI conditions
                        signal_rsi = calculate_rsi(st.session_state['signal_data'], window=st.session_state['rsi_period'])
                        
                        # Generate buy signals for benchmark (same as signal)
                        if st.session_state['comparison'] == "less_than":
                            benchmark_signals = (signal_rsi <= row['RSI_Threshold']).astype(int)
                        else:  # greater_than
                            benchmark_signals = (signal_rsi >= row['RSI_Threshold']).astype(int)
                        
                        # Calculate benchmark equity curve using benchmark prices (same logic as signal)
                        benchmark_equity_curve = pd.Series(1.0, index=st.session_state['benchmark_data'].index)
                        current_equity = 1.0
                        in_position = False
                        entry_equity = 1.0
                        entry_price = None
                        
                        for date in st.session_state['benchmark_data'].index:
                            current_signal = benchmark_signals[date] if date in benchmark_signals.index else 0
                            current_price = st.session_state['benchmark_data'][date]
                            
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
                            final_price = st.session_state['benchmark_data'].iloc[-1]
                            trade_return = (final_price - entry_price) / entry_price
                            current_equity = entry_equity * (1 + trade_return)
                            benchmark_equity_curve.iloc[-1] = current_equity
                        
                        fig_sig = go.Figure()
                        
                        fig_sig.add_trace(go.Scatter(
                            x=row['equity_curve'].index,
                            y=row['equity_curve'].values,
                            mode='lines',
                            name=f'Signal (RSI {row["RSI_Threshold"]})',
                            line=dict(color='green', width=2)
                        ))
                        
                        fig_sig.add_trace(go.Scatter(
                            x=benchmark_equity_curve.index,
                            y=benchmark_equity_curve.values,
                            mode='lines',
                            name=f'{benchmark_name} (Same RSI {row["RSI_Threshold"]} Conditions)',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig_sig.add_trace(go.Scatter(
                            x=benchmark.index,
                            y=benchmark.values,
                            mode='lines',
                            name=f'{benchmark_name} Buy & Hold',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        fig_sig.update_layout(
                            title=f"Equity Curve - RSI {row['RSI_Threshold']} ({row['confidence_level']:.1f}% Confidence)",
                            xaxis_title="Date",
                            yaxis_title="Equity Value",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_sig, use_container_width=True)
                        
                        # Add explanation
                        st.info(f"💡 **What this shows:** The green line shows your signal performance. The blue line shows {benchmark_name} performance under the same RSI conditions. The red dashed line shows {benchmark_name} buy-and-hold. This helps you see if your target ticker choice beats {benchmark_name} when the same RSI signals are applied.")
        else:
            st.info("No signals found with 80-95% confidence level.")

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
st.write("📈 **Data Source:** Real market data")
