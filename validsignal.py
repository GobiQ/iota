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
                
                # Summary statistics
                st.subheader("📈 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_trades = results_df['Total_Trades'].sum()
                    st.metric("Total Signals Generated", total_trades)
                
                with col2:
                    best_total_return = results_df['Total_Return'].max()
                    st.metric("Best Total Return", f"{best_total_return:.2%}")
                
                with col3:
                    # Best by Sortino ratio (filter out infinite values)
                    valid_sortino = results_df[results_df['Sortino_Ratio'] != np.inf]
                    if not valid_sortino.empty:
                        best_sortino = valid_sortino['Sortino_Ratio'].max()
                        best_sortino_threshold = valid_sortino.loc[valid_sortino['Sortino_Ratio'].idxmax(), 'RSI_Threshold']
                        st.metric("Best Sortino Ratio", f"{best_sortino:.2f}")
                    else:
                        st.metric("Best Sortino Ratio", "N/A")
                
                with col4:
                    benchmark_return = (benchmark.iloc[-1] - 1)
                    st.metric("SPY Buy & Hold Return", f"{benchmark_return:.2%}")
                
                # Additional metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    best_annualized = results_df['annualized_return'].max()
                    st.metric("Best Annualized Return", f"{best_annualized:.2%}")
                
                with col2:
                    best_win_rate = results_df['Win_Rate'].max()
                    st.metric("Best Win Rate", f"{best_win_rate:.1%}")
                
                with col3:
                    avg_win_rate = results_df['Win_Rate'].mean()
                    st.metric("Average Win Rate", f"{avg_win_rate:.1%}")
                
                with col4:
                    total_days = (benchmark.index[-1] - benchmark.index[0]).days
                    benchmark_annualized = (1 + benchmark_return) ** (365 / total_days) - 1 if total_days > 0 else 0
                    st.metric("SPY Annualized Return", f"{benchmark_annualized:.2%}")

                # Visualization
                st.subheader("📊 Performance by RSI Threshold")
                
                # Filter out thresholds with no trades for cleaner charts
                chart_data = results_df[results_df['Total_Trades'] > 0]
                
                if not chart_data.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = px.line(chart_data, x='RSI_Threshold', y='Total_Return', 
                                      title='Total Return by RSI Threshold',
                                      labels={'Total_Return': 'Total Return', 'RSI_Threshold': 'RSI Threshold'})
                        fig1.update_traces(line=dict(color='green'))
                        # Add benchmark line
                        benchmark_return = (benchmark.iloc[-1] - 1)
                        fig1.add_hline(y=benchmark_return, line_dash="dash", line_color="red", 
                                      annotation_text=f"SPY Buy & Hold: {benchmark_return:.2%}")
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # Filter out infinite Sortino ratios for visualization
                        sortino_data = chart_data[chart_data['Sortino_Ratio'] != np.inf]
                        if not sortino_data.empty:
                            fig2 = px.line(sortino_data, x='RSI_Threshold', y='Sortino_Ratio',
                                          title='Sortino Ratio by RSI Threshold',
                                          labels={'Sortino_Ratio': 'Sortino Ratio', 'RSI_Threshold': 'RSI Threshold'})
                            fig2.update_traces(line=dict(color='purple'))
                            st.plotly_chart(fig2, use_container_width=True)
                        else:
                            st.info("No finite Sortino ratios to display")
                    
                    # Trade frequency chart
                    fig3 = px.bar(chart_data, x='RSI_Threshold', y='Total_Trades',
                                 title='Number of Trades by RSI Threshold',
                                 labels={'Total_Trades': 'Number of Trades', 'RSI_Threshold': 'RSI Threshold'})
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Best strategy analysis
                    st.subheader("🏆 Best Strategy Analysis")
                    
                    # Get best strategies by different metrics
                    best_return_idx = results_df['Total_Return'].idxmax()
                    best_return_threshold = results_df.loc[best_return_idx, 'RSI_Threshold']
                    best_return_curve = results_df.loc[best_return_idx, 'equity_curve']
                    
                    # Best Sortino (excluding infinite values)
                    valid_sortino_df = results_df[results_df['Sortino_Ratio'] != np.inf]
                    if not valid_sortino_df.empty:
                        best_sortino_idx = valid_sortino_df['Sortino_Ratio'].idxmax()
                        best_sortino_threshold = valid_sortino_df.loc[best_sortino_idx, 'RSI_Threshold']
                        best_sortino_curve = valid_sortino_df.loc[best_sortino_idx, 'equity_curve']
                        
                        # Create comparison chart
                        fig_comparison = go.Figure()
                        
                        # Add benchmark
                        fig_comparison.add_trace(go.Scatter(
                            x=benchmark.index,
                            y=benchmark.values,
                            mode='lines',
                            name='SPY Buy & Hold',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        # Add best return strategy
                        if best_return_curve is not None:
                            fig_comparison.add_trace(go.Scatter(
                                x=best_return_curve.index,
                                y=best_return_curve.values,
                                mode='lines',
                                name=f'Best Return (RSI {best_return_threshold})',
                                line=dict(color='green', width=2)
                            ))
                        
                        # Add best Sortino strategy (if different)
                        if best_sortino_threshold != best_return_threshold and best_sortino_curve is not None:
                            fig_comparison.add_trace(go.Scatter(
                                x=best_sortino_curve.index,
                                y=best_sortino_curve.values,
                                mode='lines',
                                name=f'Best Sortino (RSI {best_sortino_threshold})',
                                line=dict(color='purple', width=2)
                            ))
                        
                        fig_comparison.update_layout(
                            title="Strategy Comparison vs SPY Buy & Hold",
                            xaxis_title="Date",
                            yaxis_title="Equity Value",
                            hovermode='x unified',
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                        )
                        
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # Performance comparison table
                        st.subheader("📋 Performance Summary")
                        
                        comparison_data = []
                        
                        # Benchmark data
                        benchmark_return = (benchmark.iloc[-1] - 1)
                        total_days = (benchmark.index[-1] - benchmark.index[0]).days
                        benchmark_annualized = (1 + benchmark_return) ** (365 / total_days) - 1 if total_days > 0 else 0
                        comparison_data.append({
                            'Strategy': 'SPY Buy & Hold',
                            'Total Return': f"{benchmark_return:.2%}",
                            'Annualized Return': f"{benchmark_annualized:.2%}",
                            'Win Rate': 'N/A',
                            'Final Value': f"{benchmark.iloc[-1]:.3f}",
                            'Sortino Ratio': 'N/A',
                            'Trades': 0
                        })
                        
                        # Best return strategy
                        best_return_data = results_df.loc[best_return_idx]
                        comparison_data.append({
                            'Strategy': f'Best Return (RSI {best_return_threshold})',
                            'Total Return': f"{best_return_data['Total_Return']:.2%}",
                            'Annualized Return': f"{best_return_data['annualized_return']:.2%}",
                            'Win Rate': f"{best_return_data['Win_Rate']:.1%}",
                            'Final Value': f"{best_return_data['Final_Equity']:.3f}",
                            'Sortino Ratio': f"{best_return_data['Sortino_Ratio']:.2f}" if not np.isinf(best_return_data['Sortino_Ratio']) else "∞",
                            'Trades': best_return_data['Total_Trades']
                        })
                        
                        # Best Sortino strategy (if different)
                        if best_sortino_threshold != best_return_threshold:
                            best_sortino_data = valid_sortino_df.loc[best_sortino_idx]
                            comparison_data.append({
                                'Strategy': f'Best Sortino (RSI {best_sortino_threshold})',
                                'Total Return': f"{best_sortino_data['Total_Return']:.2%}",
                                'Annualized Return': f"{best_sortino_data['annualized_return']:.2%}",
                                'Win Rate': f"{best_sortino_data['Win_Rate']:.1%}",
                                'Final Value': f"{best_sortino_data['Final_Equity']:.3f}",
                                'Sortino Ratio': f"{best_sortino_data['Sortino_Ratio']:.2f}",
                                'Trades': best_sortino_data['Total_Trades']
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Return Distribution Analysis
                        st.subheader("📊 Return Distribution Analysis")
                        
                        # Get the best strategy for detailed analysis
                        best_strategy_idx = results_df['Total_Return'].idxmax()
                        best_strategy_returns = results_df.loc[best_strategy_idx, 'returns']
                        
                        if len(best_strategy_returns) > 0:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Return distribution histogram
                                fig_dist = px.histogram(
                                    x=best_strategy_returns,
                                    title=f"Return Distribution - Best Strategy (RSI {results_df.loc[best_strategy_idx, 'RSI_Threshold']})",
                                    labels={'x': 'Return', 'y': 'Frequency'},
                                    nbins=20
                                )
                                fig_dist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
                                st.plotly_chart(fig_dist, use_container_width=True)
                            
                            with col2:
                                # Return statistics
                                st.subheader("📈 Return Statistics")
                                st.write(f"**Total Trades:** {len(best_strategy_returns)}")
                                st.write(f"**Win Rate:** {results_df.loc[best_strategy_idx, 'Win_Rate']:.1%}")
                                st.write(f"**Average Return:** {results_df.loc[best_strategy_idx, 'Avg_Return']:.2%}")
                                st.write(f"**Best Return:** {np.max(best_strategy_returns):.2%}")
                                st.write(f"**Worst Return:** {np.min(best_strategy_returns):.2%}")
                                st.write(f"**Return Std Dev:** {np.std(best_strategy_returns):.2%}")
                                st.write(f"**Skewness:** {pd.Series(best_strategy_returns).skew():.2f}")
                                st.write(f"**Kurtosis:** {pd.Series(best_strategy_returns).kurtosis():.2f}")
                        
                        # Individual RSI Threshold Equity Curves
                        st.subheader("📈 Individual RSI Threshold Analysis")
                        
                        # Create dropdown for selecting specific RSI threshold
                        available_thresholds = results_df[results_df['Total_Trades'] > 0]['RSI_Threshold'].tolist()
                        
                        if available_thresholds:
                            selected_threshold = st.selectbox(
                                "Select RSI Threshold to analyze:",
                                options=available_thresholds,
                                index=0,
                                format_func=lambda x: f"RSI {comparison.replace('_', ' ').title()} {x}"
                            )
                            
                            # Get data for selected threshold
                            selected_idx = results_df[results_df['RSI_Threshold'] == selected_threshold].index[0]
                            selected_data = results_df.loc[selected_idx]
                            selected_equity_curve = selected_data['equity_curve']
                            selected_trades = selected_data['trades']
                            
                            # Show metrics for selected threshold
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Return", f"{selected_data['Total_Return']:.2%}")
                            with col2:
                                st.metric("Annualized Return", f"{selected_data['annualized_return']:.2%}")
                            with col3:
                                st.metric("Win Rate", f"{selected_data['Win_Rate']:.1%}")
                            with col4:
                                st.metric("Total Trades", selected_data['Total_Trades'])
                            
                            # Create equity curve chart for selected threshold
                            if selected_equity_curve is not None:
                                fig_equity = go.Figure()
                                
                                # Add strategy equity curve
                                fig_equity.add_trace(go.Scatter(
                                    x=selected_equity_curve.index,
                                    y=selected_equity_curve.values,
                                    mode='lines',
                                    name=f'RSI {selected_threshold} Strategy',
                                    line=dict(color='blue', width=2)
                                ))
                                
                                # Add benchmark
                                fig_equity.add_trace(go.Scatter(
                                    x=benchmark.index,
                                    y=benchmark.values,
                                    mode='lines',
                                    name='SPY Buy & Hold',
                                    line=dict(color='red', width=2, dash='dash')
                                ))
                                
                                # Add trade markers
                                if selected_trades:
                                    entry_dates = [trade['entry_date'] for trade in selected_trades]
                                    entry_values = [selected_equity_curve[trade['entry_date']] for trade in selected_trades]
                                    exit_dates = [trade['exit_date'] for trade in selected_trades]
                                    exit_values = [selected_equity_curve[trade['exit_date']] for trade in selected_trades]
                                    
                                    fig_equity.add_trace(go.Scatter(
                                        x=entry_dates,
                                        y=entry_values,
                                        mode='markers',
                                        name='Buy',
                                        marker=dict(color='green', size=8, symbol='triangle-up')
                                    ))
                                    
                                    fig_equity.add_trace(go.Scatter(
                                        x=exit_dates,
                                        y=exit_values,
                                        mode='markers',
                                        name='Sell',
                                        marker=dict(color='red', size=8, symbol='triangle-down')
                                    ))
                                
                                fig_equity.update_layout(
                                    title=f"Equity Curve - RSI {comparison.replace('_', ' ').title()} {selected_threshold}",
                                    xaxis_title="Date",
                                    yaxis_title="Equity Value",
                                    hovermode='x unified',
                                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                                )
                                
                                st.plotly_chart(fig_equity, use_container_width=True)
                                
                                # Trade details for selected threshold
                                if selected_trades:
                                    with st.expander(f"Trade Details - RSI {selected_threshold}"):
                                        trades_df = pd.DataFrame(selected_trades)
                                        trades_df['return'] = trades_df['return'].apply(lambda x: f"{x:.2%}")
                                        trades_df['entry_price'] = trades_df['entry_price'].apply(lambda x: f"${x:.2f}")
                                        trades_df['exit_price'] = trades_df['exit_price'].apply(lambda x: f"${x:.2f}")
                                        st.dataframe(trades_df, use_container_width=True)
                        else:
                            st.warning("No strategies with trades available for individual analysis")
                
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
