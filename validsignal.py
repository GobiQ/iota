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

def calculate_statistical_significance(strategy_returns: np.ndarray, benchmark_returns: np.ndarray, 
                                    strategy_annualized: float, benchmark_annualized: float) -> Dict:
    """Calculate statistical significance of strategy vs benchmark"""
    
    if len(strategy_returns) == 0:
        return {
            't_statistic': 0,
            'p_value': 1.0,
            'confidence_level': 0,
            'significant': False,
            'effect_size': 0,
            'power': 0
        }
    
    # Perform t-test on returns
    t_stat, p_value = stats.ttest_ind(strategy_returns, benchmark_returns)
    
    # Calculate confidence level (1 - p_value)
    confidence_level = (1 - p_value) * 100
    
    # Determine if significant (p < 0.05)
    significant = p_value < 0.05
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(strategy_returns) - 1) * np.var(strategy_returns, ddof=1) + 
                          (len(benchmark_returns) - 1) * np.var(benchmark_returns, ddof=1)) / 
                         (len(strategy_returns) + len(benchmark_returns) - 2))
    
    effect_size = (np.mean(strategy_returns) - np.mean(benchmark_returns)) / pooled_std if pooled_std > 0 else 0
    
    # Calculate statistical power (simplified)
    # For small sample sizes, power calculation is complex, so we'll use a simplified approach
    power = 0.8 if len(strategy_returns) > 30 and abs(effect_size) > 0.5 else 0.5
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'confidence_level': confidence_level,
        'significant': significant,
        'effect_size': effect_size,
        'power': power
    }

def run_rsi_analysis(signal_ticker: str, target_ticker: str, rsi_min: float, rsi_max: float, comparison: str, 
                    start_date=None, end_date=None, rsi_period: int = 14) -> Tuple[pd.DataFrame, pd.Series]:
    """Run comprehensive RSI analysis across the specified range"""
    
    # Fetch data
    with st.spinner(f"Fetching data for {signal_ticker}..."):
        signal_data = get_stock_data(signal_ticker, start_date, end_date)
    
    with st.spinner(f"Fetching data for {target_ticker}..."):
        target_data = get_stock_data(target_ticker, start_date, end_date)
    
    # Fetch benchmark data for comparison
    with st.spinner("Fetching benchmark data..."):
        benchmark_data = get_stock_data("SPY", start_date, end_date)
    
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
        strategy_returns = analysis['returns']
        if len(strategy_returns) > 0:
            # For statistical testing, we need to compare strategy returns vs benchmark returns
            # We'll use the strategy's trade returns vs random benchmark returns of same frequency
            benchmark_annualized = (benchmark.iloc[-1] - 1) * (365 / (benchmark.index[-1] - benchmark.index[0]).days)
            stats_result = calculate_statistical_significance(
                strategy_returns, 
                benchmark_returns.sample(n=min(len(strategy_returns), len(benchmark_returns))).values,
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

# Input fields
signal_ticker = st.sidebar.text_input("Signal Ticker", value="SPY", help="Ticker to generate RSI signals from")
target_ticker = st.sidebar.text_input("Target Ticker", value="QQQ", help="Ticker to buy/sell based on signal RSI")

# Date range selection
st.sidebar.subheader("📅 Date Range")
use_date_range = st.sidebar.checkbox("Use custom date range", help="Check to specify start and end dates")

if use_date_range:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date")
        start_date, end_date = None, None
else:
    start_date, end_date = None, None
    st.sidebar.info("Using maximum available data range")

# RSI Configuration
st.sidebar.subheader("📈 RSI Configuration")

# RSI Period selection
rsi_period = st.sidebar.number_input("RSI Period (Days)", min_value=1, max_value=50, value=14, 
                                    help="Number of days to calculate RSI (e.g., 14 for 14-day RSI)")

comparison = st.sidebar.selectbox("RSI Condition", 
                               ["less_than", "greater_than"], 
                               format_func=lambda x: "RSI ≤ threshold" if x == "less_than" else "RSI ≥ threshold",
                               help="Buy when RSI is less than or greater than threshold")

if comparison == "less_than":
    default_min, default_max = 20, 40
    st.sidebar.write("Buy signals: Signal RSI ≤ threshold")
else:
    default_min, default_max = 60, 80
    st.sidebar.write("Buy signals: Signal RSI ≥ threshold")

rsi_min = st.sidebar.number_input("RSI Range Min", min_value=0.0, max_value=100.0, value=float(default_min), step=0.5)
rsi_max = st.sidebar.number_input("RSI Range Max", min_value=0.0, max_value=100.0, value=float(default_max), step=0.5)

if rsi_min >= rsi_max:
    st.sidebar.error("RSI Min must be less than RSI Max")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎯 Analysis Configuration")
    st.write(f"**Signal Ticker:** {signal_ticker} (generates RSI signals)")
    st.write(f"**Target Ticker:** {target_ticker} (buy/sell based on signals)")
    st.write(f"**RSI Period:** {rsi_period}-day RSI")
    st.write(f"**RSI Condition:** {signal_ticker} RSI {'≤' if comparison == 'less_than' else '≥'} threshold")
    st.write(f"**RSI Range:** {rsi_min} - {rsi_max}")
    if use_date_range and start_date and end_date:
        st.write(f"**Date Range:** {start_date} to {end_date}")
    else:
        st.write(f"**Date Range:** Maximum available data")

with col2:
    st.subheader("📋 Strategy Logic")
    if comparison == "less_than":
        st.info(f"🔵 BUY {target_ticker} when {signal_ticker} {rsi_period}-day RSI ≤ threshold\n\n📈 SELL {target_ticker} when {signal_ticker} {rsi_period}-day RSI > threshold")
    else:
        st.info(f"🔵 BUY {target_ticker} when {signal_ticker} {rsi_period}-day RSI ≥ threshold\n\n📈 SELL {target_ticker} when {signal_ticker} {rsi_period}-day RSI < threshold")

if st.button("🚀 Run RSI Analysis", type="primary"):
    if rsi_min < rsi_max and (not use_date_range or (start_date and end_date and start_date < end_date)):
        try:
            results_df, benchmark = run_rsi_analysis(signal_ticker, target_ticker, rsi_min, rsi_max, comparison, start_date, end_date, rsi_period)
            
            if results_df is not None and benchmark is not None and not results_df.empty:
                st.success("✅ Analysis completed successfully!")
                
                # Display results table
                st.subheader("📊 RSI Analysis Results")
                
                # Format the dataframe for display
                display_df = results_df.copy()
                display_df['Win_Rate'] = display_df['Win_Rate'].apply(lambda x: f"{x:.1%}")
                display_df['Avg_Return'] = display_df['Avg_Return'].apply(lambda x: f"{x:.2%}")
                display_df['Total_Return'] = display_df['Total_Return'].apply(lambda x: f"{x:.2%}")
                display_df['Annualized_Return'] = display_df['annualized_return'].apply(lambda x: f"{x:.2%}")
                display_df['Sortino_Ratio'] = display_df['Sortino_Ratio'].apply(lambda x: f"{x:.2f}" if not np.isinf(x) else "∞")
                display_df['Avg_Hold_Days'] = display_df['Avg_Hold_Days'].apply(lambda x: f"{x:.1f}")
                display_df['Return_Std'] = display_df['Return_Std'].apply(lambda x: f"{x:.2%}")
                display_df['Best_Return'] = display_df['Best_Return'].apply(lambda x: f"{x:.2%}")
                display_df['Worst_Return'] = display_df['Worst_Return'].apply(lambda x: f"{x:.2%}")
                display_df['Final_Equity'] = display_df['Final_Equity'].apply(lambda x: f"{x:.3f}")
                display_df['Confidence_Level'] = display_df['confidence_level'].apply(lambda x: f"{x:.1f}%")
                display_df['Significant'] = display_df['significant'].apply(lambda x: "✓" if x else "✗")
                display_df['Effect_Size'] = display_df['effect_size'].apply(lambda x: f"{x:.2f}")
                
                # Drop the equity_curve and trades columns for display
                display_cols = ['RSI_Threshold', 'Total_Trades', 'Win_Rate', 'Avg_Return', 
                               'Total_Return', 'Annualized_Return', 'Sortino_Ratio', 'Final_Equity', 'Avg_Hold_Days', 
                               'Return_Std', 'Best_Return', 'Worst_Return', 'Confidence_Level', 'Significant', 'Effect_Size']
                st.dataframe(display_df[display_cols], use_container_width=True)
                
                # Summary Statistics
                st.subheader("📊 Performance Summary")
                
                # Find best strategies
                best_sortino_idx = results_df['Sortino_Ratio'].idxmax()
                best_annualized_idx = results_df['annualized_return'].idxmax()
                best_winrate_idx = results_df['Win_Rate'].idxmax()
                best_total_return_idx = results_df['Total_Return'].idxmax()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Best Annualized Return", 
                             f"{results_df.loc[best_annualized_idx, 'annualized_return']:.2%}",
                             f"RSI {results_df.loc[best_annualized_idx, 'RSI_Threshold']}")
                
                with col2:
                    st.metric("Best Sortino Ratio", 
                             f"{results_df.loc[best_sortino_idx, 'Sortino_Ratio']:.2f}",
                             f"RSI {results_df.loc[best_sortino_idx, 'RSI_Threshold']}")
                
                with col3:
                    st.metric("Best Win Rate", 
                             f"{results_df.loc[best_winrate_idx, 'Win_Rate']:.1%}",
                             f"RSI {results_df.loc[best_winrate_idx, 'RSI_Threshold']}")
                
                with col4:
                    # Fix SPY annualized return calculation
                    total_days = (benchmark.index[-1] - benchmark.index[0]).days
                    spy_total_return = benchmark.iloc[-1] - 1
                    spy_annualized = (1 + spy_total_return) ** (365 / total_days) - 1 if total_days > 0 else 0
                    st.metric("SPY Annualized Return", f"{spy_annualized:.2%}")
                
                # Return Distribution Analysis
                st.subheader("📊 Return Distribution Analysis")
                
                # Use the best annualized return strategy for distribution analysis
                best_strategy_returns = results_df.loc[best_annualized_idx, 'returns']
                
                if len(best_strategy_returns) > 0:
                    fig = px.histogram(
                        x=best_strategy_returns,
                        nbins=20,
                        title=f"Distribution of Trade Returns (Best Annualized Strategy: RSI {results_df.loc[best_annualized_idx, 'RSI_Threshold']})",
                        labels={'x': 'Trade Return', 'y': 'Frequency'}
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Return statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Trades", len(best_strategy_returns))
                        st.metric("Win Rate", f"{(best_strategy_returns > 0).mean():.1%}")
                    
                    with col2:
                        st.metric("Average Return", f"{np.mean(best_strategy_returns):.2%}")
                        st.metric("Best Return", f"{np.max(best_strategy_returns):.2%}")
                    
                    with col3:
                        st.metric("Worst Return", f"{np.min(best_strategy_returns):.2%}")
                        st.metric("Return Std Dev", f"{np.std(best_strategy_returns):.2%}")
                    
                    with col4:
                        st.metric("Skewness", f"{stats.skew(best_strategy_returns):.2f}")
                        st.metric("Kurtosis", f"{stats.kurtosis(best_strategy_returns):.2f}")
                
                # Interactive RSI Analysis Results
                st.subheader("📊 Interactive RSI Analysis Results")
                st.write("Select any strategy from the table below to see detailed analysis:")
                
                # Display the results table
                st.dataframe(display_df[display_cols], use_container_width=True)
                
                # Create a selectbox for individual threshold analysis using existing data
                available_thresholds = results_df[results_df['Total_Trades'] > 0]['RSI_Threshold'].tolist()
                
                if available_thresholds:
                    selected_threshold = st.selectbox(
                        "Select RSI Threshold to analyze:",
                        options=available_thresholds,
                        index=0,
                        format_func=lambda x: f"RSI {x} ({signal_ticker} RSI {'≤' if comparison == 'less_than' else '≥'} {x}) - {results_df[results_df['RSI_Threshold'] == x]['Total_Return'].iloc[0]:.2%} return"
                    )
                    
                    # Get data for selected threshold from existing results
                    selected_idx = results_df[results_df['RSI_Threshold'] == selected_threshold].index[0]
                    selected_data = results_df.loc[selected_idx]
                    
                    # Display detailed analysis for selected threshold
                    st.subheader(f"🎯 Detailed Analysis: RSI {selected_threshold}")
                    
                    # Display metrics for selected threshold
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Return", f"{selected_data['Total_Return']:.2%}")
                        st.metric("Annualized Return", f"{selected_data['annualized_return']:.2%}")
                    with col2:
                        st.metric("Win Rate", f"{selected_data['Win_Rate']:.1%}")
                        st.metric("Total Trades", selected_data['Total_Trades'])
                    with col3:
                        st.metric("Sortino Ratio", f"{selected_data['Sortino_Ratio']:.2f}")
                        st.metric("Avg Hold Days", f"{selected_data['Avg_Hold_Days']:.1f}")
                    with col4:
                        st.metric("Confidence Level", f"{selected_data['confidence_level']:.1f}%")
                        st.metric("Significant", "✓" if selected_data['significant'] else "✗")
                    
                    # Return distribution for selected strategy
                    selected_returns = selected_data['returns']
                    if len(selected_returns) > 0:
                        st.subheader(f"📊 Return Distribution: RSI {selected_threshold}")
                        
                        fig_dist = px.histogram(
                            x=selected_returns,
                            nbins=20,
                            title=f"Distribution of Trade Returns (RSI {selected_threshold})",
                            labels={'x': 'Trade Return', 'y': 'Frequency'}
                        )
                        fig_dist.update_layout(showlegend=False)
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                        # Return statistics for selected strategy
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Trades", len(selected_returns))
                            st.metric("Win Rate", f"{(selected_returns > 0).mean():.1%}")
                        
                        with col2:
                            st.metric("Average Return", f"{np.mean(selected_returns):.2%}")
                            st.metric("Best Return", f"{np.max(selected_returns):.2%}")
                        
                        with col3:
                            st.metric("Worst Return", f"{np.min(selected_returns):.2%}")
                            st.metric("Return Std Dev", f"{np.std(selected_returns):.2%}")
                        
                        with col4:
                            st.metric("Skewness", f"{stats.skew(selected_returns):.2f}")
                            st.metric("Kurtosis", f"{stats.kurtosis(selected_returns):.2f}")
                    
                    # Equity curve for selected threshold using existing data
                    if selected_data['equity_curve'] is not None:
                        st.subheader(f"📈 Equity Curve: RSI {selected_threshold}")
                        
                        fig_selected = go.Figure()
                        
                        fig_selected.add_trace(go.Scatter(
                            x=selected_data['equity_curve'].index,
                            y=selected_data['equity_curve'].values,
                            mode='lines',
                            name=f"RSI {selected_threshold} Strategy",
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig_selected.add_trace(go.Scatter(
                            x=benchmark.index,
                            y=benchmark.values,
                            mode='lines',
                            name="SPY Buy & Hold",
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        # Add trade markers using existing trade data
                        if selected_data['trades']:
                            entry_dates = [trade['entry_date'] for trade in selected_data['trades']]
                            exit_dates = [trade['exit_date'] for trade in selected_data['trades']]
                            entry_equity = [selected_data['equity_curve'].loc[date] for date in entry_dates if date in selected_data['equity_curve'].index]
                            exit_equity = [selected_data['equity_curve'].loc[date] for date in exit_dates if date in selected_data['equity_curve'].index]
                            
                            fig_selected.add_trace(go.Scatter(
                                x=entry_dates,
                                y=entry_equity,
                                mode='markers',
                                name='Buy Signals',
                                marker=dict(color='green', size=8, symbol='triangle-up')
                            ))
                            
                            fig_selected.add_trace(go.Scatter(
                                x=exit_dates,
                                y=exit_equity,
                                mode='markers',
                                name='Sell Signals',
                                marker=dict(color='red', size=8, symbol='triangle-down')
                            ))
                        
                        fig_selected.update_layout(
                            title=f"Equity Curve: RSI {selected_threshold} Strategy vs SPY",
                            xaxis_title="Date",
                            yaxis_title="Equity",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_selected, use_container_width=True)
                        
                        # Trade details for selected threshold using existing data
                        with st.expander(f"📋 Trade Details - RSI {selected_threshold}"):
                            if selected_data['trades']:
                                trade_df = pd.DataFrame(selected_data['trades'])
                                trade_df['return'] = trade_df['return'].apply(lambda x: f"{x:.2%}")
                                trade_df['hold_days'] = trade_df['hold_days'].apply(lambda x: f"{x:.0f}")
                                st.dataframe(trade_df, use_container_width=True)
                            else:
                                st.write("No trades executed.")
                else:
                    st.warning("No strategies with trades available for individual analysis")
                
                # Best Strategy Analysis
                st.subheader("🏆 Best Strategy Analysis")
                
                # Create tabs for different best strategies
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Best Annualized Return", "📊 Best Sortino Ratio", "🎯 Best Win Rate", "💰 Best Total Return"])
                
                with tab1:
                    best_annualized_strategy = results_df.loc[best_annualized_idx]
                    st.write(f"**Strategy:** RSI {best_annualized_strategy['RSI_Threshold']} ({signal_ticker} RSI {'≤' if comparison == 'less_than' else '≥'} {best_annualized_strategy['RSI_Threshold']})")
                    
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Annualized Return", f"{best_annualized_strategy['annualized_return']:.2%}")
                        st.metric("Total Return", f"{best_annualized_strategy['Total_Return']:.2%}")
                    with col2:
                        st.metric("Win Rate", f"{best_annualized_strategy['Win_Rate']:.1%}")
                        st.metric("Total Trades", best_annualized_strategy['Total_Trades'])
                    with col3:
                        st.metric("Sortino Ratio", f"{best_annualized_strategy['Sortino_Ratio']:.2f}")
                        st.metric("Avg Hold Days", f"{best_annualized_strategy['Avg_Hold_Days']:.1f}")
                    with col4:
                        st.metric("Confidence Level", f"{best_annualized_strategy['confidence_level']:.1f}%")
                        st.metric("Significant", "✓" if best_annualized_strategy['significant'] else "✗")
                
                with tab2:
                    best_sortino_strategy = results_df.loc[best_sortino_idx]
                    st.write(f"**Strategy:** RSI {best_sortino_strategy['RSI_Threshold']} ({signal_ticker} RSI {'≤' if comparison == 'less_than' else '≥'} {best_sortino_strategy['RSI_Threshold']})")
                    
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Sortino Ratio", f"{best_sortino_strategy['Sortino_Ratio']:.2f}")
                        st.metric("Annualized Return", f"{best_sortino_strategy['annualized_return']:.2%}")
                    with col2:
                        st.metric("Win Rate", f"{best_sortino_strategy['Win_Rate']:.1%}")
                        st.metric("Total Trades", best_sortino_strategy['Total_Trades'])
                    with col3:
                        st.metric("Total Return", f"{best_sortino_strategy['Total_Return']:.2%}")
                        st.metric("Avg Hold Days", f"{best_sortino_strategy['Avg_Hold_Days']:.1f}")
                    with col4:
                        st.metric("Confidence Level", f"{best_sortino_strategy['confidence_level']:.1f}%")
                        st.metric("Significant", "✓" if best_sortino_strategy['significant'] else "✗")
                
                with tab3:
                    best_winrate_strategy = results_df.loc[best_winrate_idx]
                    st.write(f"**Strategy:** RSI {best_winrate_strategy['RSI_Threshold']} ({signal_ticker} RSI {'≤' if comparison == 'less_than' else '≥'} {best_winrate_strategy['RSI_Threshold']})")
                    
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Win Rate", f"{best_winrate_strategy['Win_Rate']:.1%}")
                        st.metric("Annualized Return", f"{best_winrate_strategy['annualized_return']:.2%}")
                    with col2:
                        st.metric("Total Trades", best_winrate_strategy['Total_Trades'])
                        st.metric("Sortino Ratio", f"{best_winrate_strategy['Sortino_Ratio']:.2f}")
                    with col3:
                        st.metric("Total Return", f"{best_winrate_strategy['Total_Return']:.2%}")
                        st.metric("Avg Hold Days", f"{best_winrate_strategy['Avg_Hold_Days']:.1f}")
                    with col4:
                        st.metric("Confidence Level", f"{best_winrate_strategy['confidence_level']:.1f}%")
                        st.metric("Significant", "✓" if best_winrate_strategy['significant'] else "✗")
                
                with tab4:
                    best_total_return_strategy = results_df.loc[best_total_return_idx]
                    st.write(f"**Strategy:** RSI {best_total_return_strategy['RSI_Threshold']} ({signal_ticker} RSI {'≤' if comparison == 'less_than' else '≥'} {best_total_return_strategy['RSI_Threshold']})")
                    
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Return", f"{best_total_return_strategy['Total_Return']:.2%}")
                        st.metric("Annualized Return", f"{best_total_return_strategy['annualized_return']:.2%}")
                    with col2:
                        st.metric("Win Rate", f"{best_total_return_strategy['Win_Rate']:.1%}")
                        st.metric("Total Trades", best_total_return_strategy['Total_Trades'])
                    with col3:
                        st.metric("Sortino Ratio", f"{best_total_return_strategy['Sortino_Ratio']:.2f}")
                        st.metric("Avg Hold Days", f"{best_total_return_strategy['Avg_Hold_Days']:.1f}")
                    with col4:
                        st.metric("Confidence Level", f"{best_total_return_strategy['confidence_level']:.1f}%")
                        st.metric("Significant", "✓" if best_total_return_strategy['significant'] else "✗")
                
                # Multiple Strategy Comparison
                st.subheader("📊 Multiple Strategy Comparison")
                
                # Create comparison chart with all best strategies
                fig_comparison = go.Figure()
                
                # Add benchmark
                fig_comparison.add_trace(go.Scatter(
                    x=benchmark.index,
                    y=benchmark.values,
                    mode='lines',
                    name="SPY Buy & Hold",
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Add best strategies
                colors = ['blue', 'green', 'purple', 'orange']
                strategies = [
                    (best_annualized_idx, "Best Annualized Return"),
                    (best_sortino_idx, "Best Sortino Ratio"),
                    (best_winrate_idx, "Best Win Rate"),
                    (best_total_return_idx, "Best Total Return")
                ]
                
                for i, (idx, name) in enumerate(strategies):
                    strategy = results_df.loc[idx]
                    if strategy['equity_curve'] is not None:
                        fig_comparison.add_trace(go.Scatter(
                            x=strategy['equity_curve'].index,
                            y=strategy['equity_curve'].values,
                            mode='lines',
                            name=f"{name} (RSI {strategy['RSI_Threshold']})",
                            line=dict(color=colors[i], width=2)
                        ))
                
                fig_comparison.update_layout(
                    title="Multiple Strategy Comparison vs SPY",
                    xaxis_title="Date",
                    yaxis_title="Equity",
                    hovermode='x unified',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig_comparison, use_container_width=True)

                # Download results
                csv = results_df[display_cols].to_csv(index=False)
                filename_suffix = f"_{start_date}_{end_date}" if use_date_range and start_date and end_date else "_max_range"
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"rsi_analysis_{signal_ticker}_{target_ticker}{filename_suffix}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

                # Statistical Significance Analysis
                st.subheader("📊 Statistical Significance Analysis")
                
                # Filter strategies with trades
                valid_strategies = results_df[results_df['Total_Trades'] > 0].copy()
                
                if not valid_strategies.empty:
                    # Create significance summary
                    significant_strategies = valid_strategies[valid_strategies['significant'] == True]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Strategies", len(valid_strategies))
                    
                    with col2:
                        st.metric("Statistically Significant", len(significant_strategies))
                    
                    with col3:
                        significance_rate = len(significant_strategies) / len(valid_strategies) * 100
                        st.metric("Significance Rate", f"{significance_rate:.1f}%")
                    
                    with col4:
                        avg_confidence = valid_strategies['confidence_level'].mean()
                        st.metric("Avg Confidence Level", f"{avg_confidence:.1f}%")
                    
                    # Confidence level distribution
                    fig_confidence = px.histogram(
                        valid_strategies, 
                        x='confidence_level',
                        title="Distribution of Confidence Levels",
                        labels={'confidence_level': 'Confidence Level (%)', 'y': 'Number of Strategies'},
                        nbins=10
                    )
                    fig_confidence.add_vline(x=95, line_dash="dash", line_color="red", 
                                               annotation_text="95% Confidence")
                    st.plotly_chart(fig_confidence, use_container_width=True)
                    
                    # Effect size vs confidence level
                    fig_effect = px.scatter(
                        valid_strategies,
                        x='effect_size',
                        y='confidence_level',
                        color='significant',
                        title="Effect Size vs Confidence Level",
                        labels={'effect_size': 'Effect Size (Cohen\'s d)', 'confidence_level': 'Confidence Level (%)'},
                        color_discrete_map={True: 'green', False: 'red'}
                    )
                    fig_effect.add_hline(y=95, line_dash="dash", line_color="red", 
                                               annotation_text="95% Confidence")
                    st.plotly_chart(fig_effect, use_container_width=True)
                    
                    # Top significant strategies
                    if len(significant_strategies) > 0:
                        st.subheader("🏆 Top Statistically Significant Strategies")
                        
                        # Sort by confidence level
                        top_significant = significant_strategies.nlargest(5, 'confidence_level')
                        
                        for idx, row in top_significant.iterrows():
                            with st.expander(f"RSI {row['RSI_Threshold']} - {row['confidence_level']:.1f}% Confidence"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Total Return", f"{row['Total_Return']:.2%}")
                                    st.metric("Annualized Return", f"{row['annualized_return']:.2%}")
                                
                                with col2:
                                    st.metric("Win Rate", f"{row['Win_Rate']:.1%}")
                                    st.metric("Sortino Ratio", f"{row['Sortino_Ratio']:.2f}" if not np.isinf(row['Sortino_Ratio']) else "∞")
                                
                                with col3:
                                    st.metric("Confidence Level", f"{row['confidence_level']:.1f}%")
                                    st.metric("Effect Size", f"{row['effect_size']:.2f}")
                                
                                st.write(f"**Statistical Details:**")
                                st.write(f"- T-statistic: {row['t_statistic']:.3f}")
                                st.write(f"- P-value: {row['p_value']:.4f}")
                                st.write(f"- Power: {row['power']:.2f}")
                                
                                # Show equity curve for this strategy
                                if row['equity_curve'] is not None:
                                    fig_sig = go.Figure()
                                    
                                    fig_sig.add_trace(go.Scatter(
                                        x=row['equity_curve'].index,
                                        y=row['equity_curve'].values,
                                        mode='lines',
                                        name=f'RSI {row["RSI_Threshold"]} Strategy',
                                        line=dict(color='green', width=2)
                                    ))
                                    
                                    fig_sig.add_trace(go.Scatter(
                                        x=benchmark.index,
                                        y=benchmark.values,
                                        mode='lines',
                                        name='SPY Buy & Hold',
                                        line=dict(color='red', width=2, dash='dash')
                                    ))
                                    
                                    fig_sig.update_layout(
                                        title=f"Equity Curve - RSI {row['RSI_Threshold']} ({row['confidence_level']:.1f}% Confidence)",
                                        xaxis_title="Date",
                                        yaxis_title="Equity Value",
                                        hovermode='x unified'
                                    )
                                    
                                    st.plotly_chart(fig_sig, use_container_width=True)
                    else:
                        st.warning("No strategies reached statistical significance (p < 0.05)")
                
                # Statistical interpretation guide
                with st.expander("📚 Statistical Significance Guide"):
                    st.write("""
                    **Understanding Statistical Significance:**
                    
                    - **Confidence Level**: Percentage confidence that the strategy outperforms SPY
                    - **P-value**: Probability of getting these results by chance (lower is better)
                    - **Effect Size**: Magnitude of the difference (Cohen's d)
                    - **Significant**: P-value < 0.05 (95% confidence level)
                    
                    **Interpretation:**
                    - ✓ **Significant**: Strong evidence the strategy beats SPY
                    - ✗ **Not Significant**: Results could be due to chance
                    - **Effect Size**: 
                      - Small: 0.2-0.5
                      - Medium: 0.5-0.8  
                      - Large: > 0.8
                    """)
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
    else:
        if rsi_min >= rsi_max:
            st.error("Please ensure RSI Min is less than RSI Max")
        if use_date_range and (not start_date or not end_date or start_date >= end_date):
            st.error("Please ensure start date is before end date")

st.write("---")
st.write("💡 **Tip:** Try different ticker combinations and RSI conditions to find optimal signal thresholds")
st.write("📈 **Data Source:** Real market data from Yahoo Finance via yfinance")
